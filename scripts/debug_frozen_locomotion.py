"""Fixed-command gate before navigation training; emits measured JSON results."""
import argparse
import json
from pathlib import Path
import torch
import instinct_mj.tasks
from instinct_mj.control.frozen_locomotion import FrozenLocomotion
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--output', type=Path, default=Path('outputs/p4/locomotion_gate.json'))
    a = p.parse_args()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    policy = FrozenLocomotion(a.checkpoint, device)
    task = 'Instinct-Locomotion-Flat-G1-v0'
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.observations['policy'].enable_corruption = False
    cfg.events = {k: v for k, v in cfg.events.items() if k in ('reset_base', 'reset_robot_joints')}
    cfg.events['reset_base'].params['pose_range'] = {k: (0., 0.) for k in ('x', 'y', 'yaw')}
    cfg.events['reset_base'].params['velocity_range'] = {k: (0., 0.) for k in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')}
    cfg.events['reset_robot_joints'].params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    cfg.commands['base_velocity'].resampling_time_range = (1e6, 1e6)
    env = load_env_cls(task)(cfg, device)
    vec = InstinctRlVecEnvWrapper(env)
    assert abs(env.step_dt - policy.control_dt) < 1e-9
    commands = torch.tensor([[.4, 0., 0.], [0., 0., .4], [.3, 0., .3], [.3, .1, -.2]], device=device)
    assert torch.equal(policy.command(commands, torch.zeros_like(commands)), commands)
    snapshot = {k: v.clone() for k, v in policy.state_dict().items()}
    alive = torch.ones(4, dtype=torch.bool, device=device)
    lengths = torch.zeros(4, device=device)
    errors = torch.zeros(4, 3, device=device)
    start = env.scene['robot'].data.root_link_pos_w.clone()
    displacement = torch.zeros(4, device=device)
    for step in range(a.steps):
        cmd = env.command_manager.get_term('base_velocity')
        for field in ('is_heading_env', 'is_standing_env', 'is_world_env', 'is_forward_env'):
            getattr(cmd, field)[:] = False
        residual = torch.zeros_like(commands)
        if step >= a.steps // 2:
            residual[3] = torch.tensor([.1, -.1, .2], device=device)
        cmd.vel_command_b[:] = policy.command(commands, residual)
        obs, _ = vec.get_observations()
        action = policy(obs)
        robot = env.scene['robot']
        measured = torch.cat((robot.data.root_link_lin_vel_b[:, :2], robot.data.root_link_ang_vel_b[:, 2:3]), -1)
        errors += (measured - cmd.vel_command_b).abs() * alive[:, None]
        displacement[alive] = (robot.data.root_link_pos_w - start)[alive, :2].norm(dim=-1)
        _, _, done, _ = vec.step(action)
        lengths += alive
        alive &= ~done.bool()
    frozen = all(torch.equal(v, snapshot[k]) for k, v in policy.state_dict().items())
    result = dict(checkpoint=str(Path(a.checkpoint).resolve()), control_hz=1/policy.control_dt,
                  joint_order=list(robot.joint_names), action_joint_order=list(env.action_manager.get_term('joint_pos')._target_names), actuator_order=list(robot.actuator_names),
                  commands=commands.tolist(), survived=alive.tolist(), steps=lengths.tolist(),
                  displacement=displacement.tolist(), mean_abs_velocity_error=(errors/lengths[:, None].clamp_min(1)).tolist(),
                  parameters_and_normalizer_unchanged=frozen,
                  no_gradients=all(p.grad is None and not p.requires_grad for p in policy.parameters()))
    error = errors / lengths[:, None].clamp_min(1)
    result['gate_thresholds'] = {'linear_mae_mps': .15, 'angular_mae_radps': .15}
    result['approved_for_navigation'] = bool(alive.all() and frozen and (error < .15).all())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == '__main__':
    main()
