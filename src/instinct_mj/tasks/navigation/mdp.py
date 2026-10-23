"""Navigation observations and exclusive terminal outcomes. No obstacle poses in actor."""
import torch


def controller(env):
    return env.action_manager.get_term('navigation')


def goal(env):
    return controller(env).relative_goal()


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
    else:
        raise ValueError(kind)
    env.extras.setdefault('step', {})[f'Navigation/raw_{kind}'] = value.mean()
    env.extras['step'][f'Navigation/weighted_{kind}'] = value.mean() * env.cfg.rewards[kind].weight * env.step_dt
    return value
