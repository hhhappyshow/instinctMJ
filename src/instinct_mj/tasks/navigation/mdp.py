"""Navigation observations and exclusive terminal outcomes. No obstacle poses in actor."""
import torch
from instinct_mj.perception.forward_scan import HEIGHT, WIDTH, forward_scan

_APPROACH_ROWS = slice(9, 14)
_APPROACH_COLS = slice(8, 24)
_APPROACH_DISTANCE = 0.8


def controller(env):
    return env.action_manager.get_term('navigation')


def goal(env):
    return controller(env).relative_goal()


def velocity_command(env):
    return controller(env).velocity_command()


def _body_velocity(env):
    robot = env.scene['robot'].data
    return torch.cat((robot.root_link_lin_vel_b[:, :2], robot.root_link_ang_vel_b[:, 2:3]), -1)


def collision(env):
    hits = []
    for i in range(3):
        data = env.scene[f'obstacle_contact_{i}'].data
        current = data.found.reshape(env.num_envs, -1).any(-1)
        recent = (data.force_history.norm(dim=-1) > 1e-6).reshape(env.num_envs, -1).any(-1)
        hits.append(current | recent)
    return torch.stack(hits).any(0)


def fall(env):
    d = env.scene['robot'].data
    return (d.projected_gravity_b[:, 2] > -.5) | (d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < .45)


def outcome(env, kind):
    hit = collision(env)
    fallen = fall(env) & ~hit
    success = (controller(env).distance() < .3) & ~hit & ~fallen
    timeout = (env.episode_length_buf >= env.max_episode_length) & ~hit & ~fallen & ~success
    return dict(collision=hit, fall=fallen, success=success, timeout=timeout)[kind]


def reward(env, kind):
    term = controller(env)
    if kind == 'progress':
        # Reward only a new closest approach; backing away cannot earn it again.
        if term.reward_step != env.common_step_counter:
            distance = term.distance()
            term.progress.copy_((term.best_distance-distance).clamp_min(0))
            term.best_distance.copy_(torch.minimum(term.best_distance, distance))
            term.reward_step = env.common_step_counter
        value = term.progress / env.step_dt
    elif kind in ('success', 'collision', 'fall'):
        value = outcome(env, kind).float() / env.step_dt
    elif kind == 'time':
        value = torch.ones(env.num_envs, device=env.device)
    elif kind == 'smoothness':
        current = env.action_manager.action.clamp(-1, 1)
        previous = env.action_manager.prev_action.clamp(-1, 1)
        value = (current-previous).square().sum(-1)
    elif kind == 'track':
        error = (_body_velocity(env) - controller(env).velocity_command()).square().sum(-1)
        value = torch.exp(-error / 0.25)
    elif kind == 'approach':
        # Dense training signal only. It does not change the commanded velocity.
        distances = forward_scan(env.scene['forward_scanner'].data)['ray_distances']
        nearest = distances.view(env.num_envs, HEIGHT, WIDTH)[:, _APPROACH_ROWS, _APPROACH_COLS].amin(dim=(1, 2))
        closing = ((_APPROACH_DISTANCE - nearest) / _APPROACH_DISTANCE).clamp(0, 1)
        forward = controller(env).velocity[:, 0] > 0.05
        seen = nearest > 0.05
        value = (forward & seen).float() * closing
    else:
        raise ValueError(kind)
    env.extras.setdefault('step', {})[f'Navigation/raw_{kind}'] = value.mean()
    env.extras['step'][f'Navigation/weighted_{kind}'] = value.mean() * env.cfg.rewards[kind].weight * env.step_dt
    return value
