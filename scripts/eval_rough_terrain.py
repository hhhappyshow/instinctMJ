"""Evaluate a rough-terrain checkpoint, with an optional height-scan ablation.

Short smoke checkpoints are only used to check that this script runs. They are
not evidence that perception beats the baseline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from instinct_rl.runners import OnPolicyRunner

import instinct_mj.tasks  # noqa: F401
from instinct_mj.perception.height_scan import HEIGHT_SCAN_MASK_VALUE
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.locomotion.config.g1.rough_env_cfg import TERRAIN_COLUMN_NAMES
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg

TASKS = {
    "baseline": "Instinct-Locomotion-Rough-G1-Play-v0",
    "raycaster": "Instinct-Locomotion-Rough-RayCaster-G1-Play-v0",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(TASKS), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--terrain", choices=TERRAIN_COLUMN_NAMES, default="flat")
    parser.add_argument("--terrain-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--ablation", choices=("normal", "masked"), default="normal")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _height_slice(vec_env: InstinctRlVecEnvWrapper) -> slice | None:
    segments = vec_env.get_obs_segments("policy")
    cursor = 0
    for name, shape in segments.items():
        width = 1
        for dim in shape:
            width *= int(dim)
        if name == "height_scan":
            return slice(cursor, cursor + width)
        cursor += width
    return None


def _lock_command(env, vx: float, vy: float, wz: float) -> None:
    command = env.command_manager.get_term("base_velocity")
    command.is_heading_env[:] = False
    command.is_standing_env[:] = False
    command.is_world_env[:] = False
    command.is_forward_env[:] = False
    command.vel_command_b[:] = torch.tensor([vx, vy, wz], device=env.device)


def _place_terrain(env, terrain: str, level: int) -> None:
    terrain_entity = env.scene.terrain
    column = TERRAIN_COLUMN_NAMES.index(terrain)
    if not 0 <= level < terrain_entity.terrain_origins.shape[0]:
        raise ValueError(f"terrain level {level} is outside {terrain_entity.terrain_origins.shape[0]} rows")
    terrain_entity.terrain_types[:] = column
    terrain_entity.terrain_levels[:] = level
    terrain_entity.env_origins[:] = terrain_entity.terrain_origins[level, column]


def _apply_ablation(obs: torch.Tensor, height_slice: slice | None, ablation: str) -> torch.Tensor:
    if ablation == "normal" or height_slice is None:
        return obs
    masked = obs.clone()
    masked[:, height_slice] = HEIGHT_SCAN_MASK_VALUE
    return masked


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_id = TASKS[args.variant]
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = args.episodes
    env_cfg.commands["base_velocity"].resampling_time_range = (1.0e6, 1.0e6)
    env_cfg.seed = args.seed
    env = load_env_cls(task_id)(cfg=env_cfg, device=device)
    _place_terrain(env, args.terrain, args.terrain_level)
    agent_cfg = load_instinct_rl_cfg(task_id)
    vec_env = InstinctRlVecEnvWrapper(
        env,
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )
    _lock_command(env, args.vx, args.vy, args.wz)
    height_slice = _height_slice(vec_env)
    if args.variant == "raycaster" and height_slice is None:
        raise SystemExit("raycaster actor has no height_scan segment")
    if args.variant == "baseline" and height_slice is not None:
        raise SystemExit("baseline actor unexpectedly contains height_scan")
    if args.ablation == "masked" and height_slice is None:
        raise SystemExit("masked ablation requires a height_scan actor")

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(args.checkpoint.expanduser().resolve()))
    policy = runner.get_inference_policy(device=device)

    obs, _ = vec_env.get_observations()
    _lock_command(env, args.vx, args.vy, args.wz)
    obs, _ = vec_env.get_observations()
    alive = torch.ones(vec_env.num_envs, dtype=torch.bool, device=vec_env.device)
    fell = torch.zeros_like(alive)
    lengths = torch.zeros(vec_env.num_envs, device=vec_env.device)
    lin_error_sum = torch.zeros(vec_env.num_envs, device=vec_env.device)
    ang_error_sum = torch.zeros(vec_env.num_envs, device=vec_env.device)
    valid_ratio_sum = 0.0
    valid_ratio_count = 0
    infer_seconds = []

    for _ in range(args.max_steps):
        if not bool(alive.any()):
            break
        _lock_command(env, args.vx, args.vy, args.wz)
        obs, _ = vec_env.get_observations()
        model_obs = _apply_ablation(obs, height_slice, args.ablation)
        started = time.perf_counter()
        with torch.no_grad():
            action = policy(model_obs)
        infer_seconds.append(time.perf_counter() - started)
        _, _, dones, extras = vec_env.step(action)
        command = env.command_manager.get_term("base_velocity")
        robot = env.scene["robot"]
        lin_error = torch.norm(command.vel_command_b[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=-1)
        ang_error = torch.abs(command.vel_command_b[:, 2] - robot.data.root_link_ang_vel_b[:, 2])
        lin_error_sum = lin_error_sum + lin_error * alive.float()
        ang_error_sum = ang_error_sum + ang_error * alive.float()
        lengths = lengths + alive.float()
        sensor = env.scene["height_scanner"]
        valid_ratio_sum += float((sensor.data.distances >= 0).float().mean())
        valid_ratio_count += 1
        timeout = extras.get("time_outs", torch.zeros_like(dones))
        newly_fallen = alive & dones.bool() & ~timeout.bool()
        fell = fell | newly_fallen
        alive = alive & ~dones.bool()

    survived = ~fell
    report = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "terrain": args.terrain,
        "terrain_level": args.terrain_level,
        "seed": args.seed,
        "command": [args.vx, args.vy, args.wz],
        "ablation": args.ablation,
        "episodes": args.episodes,
        "success_rate": float(survived.float().mean()),
        "fall_rate": float(fell.float().mean()),
        "lin_vel_error": float((lin_error_sum / lengths.clamp(min=1)).mean()),
        "ang_vel_error": float((ang_error_sum / lengths.clamp(min=1)).mean()),
        "mean_episode_length": float(lengths.mean()),
        "valid_ray_ratio": valid_ratio_sum / max(valid_ratio_count, 1),
        "inference_s": float(sum(infer_seconds) / max(len(infer_seconds), 1)),
    }
    print(json.dumps(report, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    vec_env.close()


if __name__ == "__main__":
    main()
