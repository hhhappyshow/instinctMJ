"""Navigation interface smoke test; zero residual is nominal control, not trained PPO."""
import argparse
import json
from pathlib import Path
import torch
from instinct_mj.tasks.navigation.config import env_config
from instinct_mj.tasks.navigation import mdp
from instinct_mj.envs import InstinctRlEnv
from instinct_mj.rl import InstinctRlVecEnvWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--policy-checkpoint', help='Optional trained upper PPO checkpoint for evaluation')
    parser.add_argument('--stage', type=int, choices=range(4), default=0)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()
    cfg = env_config(play=True, rays=not args.baseline)
    cfg.scene.num_envs = 4
    cfg.seed = 42
    cfg.auto_reset = False
    cfg.actions['navigation'].checkpoint = args.checkpoint
    cfg.actions['navigation'].stage = args.stage
    env = InstinctRlEnv(cfg, 'cuda:0')
    vec = InstinctRlVecEnvWrapper(env)
    control = mdp.controller(env)
    saved = {k: v.clone() for k, v in control.policy.state_dict().items()}
    upper = None
    if args.policy_checkpoint:
        from instinct_rl.runners import OnPolicyRunner
        from instinct_mj.tasks.navigation.config import runner_config
        runner = OnPolicyRunner(vec, runner_config(not args.baseline).to_dict(), log_dir=None, device=env.device)
        runner.load(args.policy_checkpoint)
        upper = runner.get_inference_policy(device=env.device)
    outcomes = dict.fromkeys(('success','collision','fall','timeout'), 0)
    # Explicit reset preserves terminal classifications for evaluation.
    for step in range(args.steps):
        obs, _ = vec.get_observations()
        assert torch.isfinite(obs).all()
        old_updates = control.low_updates
        with torch.no_grad():
            action = upper(obs) if upper else torch.zeros(4, 3, device=env.device)
        _, rewards, done, extra = vec.step(action)
        assert control.low_updates - old_updates == 5
        assert torch.isfinite(rewards).all()
        assert control.phase == cfg.decimation
        assert torch.all(control.velocity >= control.policy.limits[:, 0])
        assert torch.all(control.velocity <= control.policy.limits[:, 1])
        for name in outcomes:
            outcomes[name] += int(env.termination_manager.get_term(name).sum())
        if done.any():
            env.reset(env_ids=done.nonzero(as_tuple=False).flatten())
    frozen = all(torch.equal(v, saved[k]) for k, v in control.policy.state_dict().items())
    assert frozen
    result = dict(stage=args.stage, variant='baseline' if args.baseline else 'raycaster',
                  steps=args.steps, observation_dim=vec.num_obs, action_dim=vec.num_actions,
                  high_hz=1/env.step_dt, low_hz=1/control.policy.control_dt, frozen=frozen,
                  outcomes=outcomes, trained_policy=upper is not None,
                  unfinished_episodes=4, checkpoint=args.policy_checkpoint)
    suffix = '_trained' if upper else ''
    path = Path(f'outputs/p4/navigation_{result["variant"]}_stage{args.stage}{suffix}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == '__main__':
    main()
