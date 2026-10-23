"""Stage-0 nominal-control acceptance for the official ONNX low-level.

Upper residual stays zero. This is not trained PPO and does not advance the
navigation stage. Baseline is the gait and goal gate; the ray task is a short
construction check that the torso scanner still steps.
"""
import argparse
import json
from pathlib import Path
import torch
import instinct_mj.tasks  # noqa: F401
from instinct_mj.envs import InstinctRlEnv
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.navigation import mdp
from instinct_mj.tasks.registry import load_env_cfg


def run(task, steps, device):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.auto_reset = False
    cfg.actions['navigation'].stage = 0
    env = InstinctRlEnv(cfg, device)
    vec = InstinctRlVecEnvWrapper(env)
    control = mdp.controller(env)
    robot = env.scene['robot']
    defaults = control.policy.default_joint_pos.to(device)
    if not torch.allclose(robot.data.default_joint_pos[0], defaults, atol=1e-3):
        raise AssertionError('Robot default pose does not match deploy.yaml')
    fingerprint = dict(control.policy.fingerprint)
    saved_limits = control.policy.limits.clone()
    outcomes = dict.fromkeys(('success', 'collision', 'fall', 'timeout'), 0)
    steps_taken = torch.zeros(4, 2, device=device)
    max_swing = torch.zeros(4, 2, device=device)
    min_height = torch.full((4,), float('inf'), device=device)
    episodes = torch.zeros(4, device=device)
    for step in range(steps):
        obs, _ = vec.get_observations()
        assert torch.isfinite(obs).all()
        old_updates = control.low_updates
        _, rewards, done, _ = vec.step(torch.zeros(4, 3, device=device))
        assert control.low_updates - old_updates == 5
        assert torch.isfinite(rewards).all()
        assert control.phase == cfg.decimation
        assert torch.all(control.velocity >= control.policy.limits[:, 0])
        assert torch.all(control.velocity <= control.policy.limits[:, 1])
        min_height = torch.minimum(min_height, robot.data.root_link_pos_w[:, 2])
        for name in outcomes:
            outcomes[name] += int(env.termination_manager.get_term(name).sum())
        if done.any():
            ids = done.nonzero(as_tuple=False).flatten()
            steps_taken[ids] += control.foot_steps[ids]
            max_swing[ids] = torch.maximum(max_swing[ids], control.max_single_swing[ids])
            episodes[ids] += 1
            env.reset(env_ids=ids)
    steps_taken += control.foot_steps
    max_swing = torch.maximum(max_swing, control.max_single_swing)
    total = steps_taken.sum(dim=-1)
    imbalance = (steps_taken[:, 0] - steps_taken[:, 1]).abs() / total.clamp_min(1)
    frozen = torch.equal(control.policy.limits, saved_limits) and control.policy.fingerprint == fingerprint
    swing_s = (max_swing * control.policy.control_dt).tolist()
    result = dict(
        task=task, steps=steps, device=device, frozen=frozen,
        observation_dim=vec.num_obs, action_dim=vec.num_actions,
        high_hz=1 / env.step_dt, low_hz=1 / control.policy.control_dt,
        outcomes=outcomes, episodes_completed=episodes.tolist(),
        foot_steps=steps_taken.tolist(), step_imbalance=imbalance.tolist(),
        max_single_swing_s=swing_s, min_pelvis_height=min_height.tolist(),
        limits=control.policy.limits.tolist(), fingerprint=fingerprint,
    )
    enough = bool((total >= 20).all())
    result['gate'] = dict(
        no_falls=outcomes['fall'] == 0,
        no_collisions=outcomes['collision'] == 0,
        reached=outcomes['success'] >= 4,
        alternating=bool(enough and (imbalance <= 0.10).all()),
        no_prolonged_single_swing=bool((max_swing * control.policy.control_dt <= 0.8).all()),
        frozen=frozen,
        counts_sufficient=enough,
    )
    result['approved'] = all(result['gate'].values())
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--ray-steps', type=int, default=5)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    baseline = run('Instinct-G1-Baseline-Obstacle-Avoidance-Unitree-v0', args.steps, device)
    ray_cfg = load_env_cfg('Instinct-G1-RayCaster-Obstacle-Avoidance-Unitree-v0', play=True)
    ray_cfg.scene.num_envs = 1
    ray_cfg.seed = 42
    ray_cfg.actions['navigation'].stage = 0
    ray_env = InstinctRlEnv(ray_cfg, device)
    ray_vec = InstinctRlVecEnvWrapper(ray_env)
    for _ in range(args.ray_steps):
        obs, _ = ray_vec.get_observations()
        assert obs.shape[-1] == 524 and torch.isfinite(obs).all()
        ray_vec.step(torch.zeros(1, 3, device=device))
    scan = ray_env.scene['forward_scanner'].data.distances
    ray_env.close()
    baseline['ray_task_steps'] = args.ray_steps
    baseline['ray_task_observation_dim'] = 524
    baseline['ray_distances_finite'] = bool(torch.isfinite(scan).any())
    path = Path('outputs/p4/navigation_unitree_stage0.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2))
    print(json.dumps(baseline, indent=2))


if __name__ == '__main__':
    main()
