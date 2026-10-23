"""Strict frozen loader for the native 96-observation G1 flat actor.

Requires the checkpoint's saved env/agent YAML alongside it; never initializes
an untrained fallback. Input order and action scaling belong to the flat task.
"""
from pathlib import Path

import torch
from torch import nn
import yaml
from instinct_rl.modules.normalizer import EmpiricalNormalization


class FrozenLocomotion(nn.Module):
    def __init__(self, checkpoint: str, device='cpu'):
        super().__init__()
        path = Path(checkpoint).expanduser().resolve(strict=True)
        if path.name == 'model_30990.pt':
            raise ValueError('model_30990.pt is not an approved locomotion model')
        with (path.parent / 'params/env.yaml').open() as f:
            env = yaml.safe_load(f)
        with (path.parent / 'params/agent.yaml').open() as f:
            agent = yaml.safe_load(f)
        if env['scene']['entities']['robot']['spec_fn'] != 'instinct_mj.assets.unitree_g1:get_g1_spec':
            raise ValueError('Checkpoint is not for the audited torso-base G1 asset')
        if agent['normalizers']['policy'] != {'class_name': 'EmpiricalNormalization'}:
            raise ValueError('Unsupported policy normalizer configuration')
        self.saved_action_config = env['actions']['joint_pos']
        terms = env['observations']['policy']['terms']
        expected = ['base_ang_vel', 'projected_gravity', 'velocity_commands', 'joint_pos', 'joint_vel', 'actions']
        if list(terms) != expected or any(t.get('history_length', 0) for t in terms.values()):
            raise ValueError('Unsupported locomotion observation layout/history')
        group = env['observations']['policy']
        if group.get('history_length'):
            raise ValueError('Historical policies require a different adapter')
        if agent['policy']['activation'] != 'elu':
            raise ValueError('Only the audited ELU actor is supported')
        ranges = env['commands']['base_velocity']['ranges']
        self.register_buffer('limits', torch.tensor([ranges[k] for k in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')], dtype=torch.float32))
        self.control_dt = env['decimation'] * env['sim']['mujoco']['timestep']
        checkpoint_data = torch.load(path, map_location='cpu', weights_only=True)
        dims = [96, *agent['policy']['actor_hidden_dims'], 29]
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.actor = nn.Sequential(*layers)
        self.actor.load_state_dict({k.removeprefix('actor.'): v for k, v in checkpoint_data['model_state_dict'].items() if k.startswith('actor.')}, strict=True)
        self.normalizer = EmpiricalNormalization(96)
        self.normalizer.load_state_dict(checkpoint_data['policy_normalizer_state_dict'], strict=True)
        self.requires_grad_(False)
        self.eval()
        self.to(device)

    def train(self, mode=True):
        # Even a parent's train() must not update normalization statistics.
        return super().train(False)

    def command(self, nominal, residual):
        return (nominal + residual).clamp(self.limits[:, 0], self.limits[:, 1])

    @torch.no_grad()
    def forward(self, observation):
        if observation.shape[-1] != 96 or not torch.isfinite(observation).all():
            raise ValueError('Expected finite native-order 96-dimensional flat observations')
        return self.actor(self.normalizer(observation))
