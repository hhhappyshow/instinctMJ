"""Print the RayCaster actor observation and confirm the scan enters the MLP."""

from __future__ import annotations

import torch
from instinct_rl.runners import OnPolicyRunner

import instinct_mj.tasks  # noqa: F401
from instinct_mj.rl import InstinctRlVecEnvWrapper
from instinct_mj.tasks.registry import load_env_cfg, load_env_cls, load_instinct_rl_cfg

TASK_ID = "Instinct-Perceptive-Shadowing-G1-RayCaster-Play-v0"
PROPRIO = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action")
MOTION = ("joint_pos_ref", "joint_vel_ref", "position_ref", "rotation_ref")
CRITIC_ONLY = ("link_pos", "link_rot", "base_lin_vel")


def _flat_dim(term: torch.Tensor) -> int:
    return int(term[0].numel())


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env = load_env_cls(TASK_ID)(cfg=env_cfg, device=device)
    agent_cfg = load_instinct_rl_cfg(TASK_ID)
    vec_env = InstinctRlVecEnvWrapper(
        env,
        policy_group=agent_cfg.policy_observation_group,
        critic_group=agent_cfg.critic_observation_group,
    )
    policy_obs = env.observation_manager.compute()["policy"]
    critic_obs = env.observation_manager.compute()["critic"]

    print("Actor observation:")
    proprio_dim = 0
    motion_dim = 0
    for name in PROPRIO:
        proprio_dim += _flat_dim(policy_obs[name])
        print(f"    proprioception.{name}: shape={tuple(policy_obs[name].shape)} dtype={policy_obs[name].dtype}")
    for name in MOTION:
        motion_dim += _flat_dim(policy_obs[name])
        print(f"    motion_reference.{name}: shape={tuple(policy_obs[name].shape)} dtype={policy_obs[name].dtype}")
    height = policy_obs["height_scan"]
    print(f"    height_scan: shape={tuple(height.shape)} dtype={height.dtype} min={height.min().item():.4f} max={height.max().item():.4f}")
    print(f"    proprioception dim: {proprio_dim}")
    print(f"    motion_reference dim: {motion_dim}")
    print(f"    height_scan dim: {_flat_dim(height)}")

    packed_policy, extras = vec_env.get_observations()
    packed_critic = extras["observations"]["critic"]
    print(f"Final Actor observation shape: {tuple(packed_policy.shape)}")
    print(f"Critic observation shape: {tuple(packed_critic.shape)}")

    actor_names = set(policy_obs)
    if "depth_image" in actor_names:
        raise SystemExit("depth_image is still in the actor")
    if "height_scan" not in actor_names:
        raise SystemExit("height_scan is missing from the actor")
    leaked = actor_names & set(CRITIC_ONLY)
    if leaked:
        raise SystemExit(f"critic terms leaked into the actor: {sorted(leaked)}")
    if not torch.isfinite(packed_policy).all() or not torch.isfinite(packed_critic).all():
        raise SystemExit("actor or critic observation is not finite")

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=device)
    policy = runner.get_inference_policy(device=device)
    with torch.no_grad():
        action = policy(packed_policy)
    print(f"Policy action shape: {tuple(action.shape)}")
    if not torch.isfinite(action).all():
        raise SystemExit("policy action is not finite")

    module = runner.alg.actor_critic
    first_linear = next(item for item in module.modules() if isinstance(item, torch.nn.Linear))
    print(f"First Linear in_features: {first_linear.in_features}")
    if first_linear.in_features != packed_policy.shape[-1]:
        raise SystemExit("actor input width does not match the packed observation")
    optimizer_params = {id(param) for group in runner.alg.optimizer.param_groups for param in group["params"]}
    trainable = [param for param in module.parameters() if param.requires_grad]
    if not trainable or any(id(param) not in optimizer_params for param in trainable):
        raise SystemExit("actor parameters are not in the PPO optimizer")
    print(f"PPO optimizer covers {len(trainable)} trainable tensors")

    zero = torch.zeros_like(action)
    next_obs, _, _, next_extras = vec_env.step(zero)
    next_height = env.observation_manager.compute()["policy"]["height_scan"]
    if not torch.isfinite(next_obs).all() or not torch.isfinite(next_height).all():
        raise SystemExit("height scan became non-finite after a step")
    print(f"Height scan after one zero step: min={next_height.min().item():.4f} max={next_height.max().item():.4f}")
    vec_env.close()


if __name__ == "__main__":
    main()
