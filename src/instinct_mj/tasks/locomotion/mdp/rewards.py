# Copyright (c) 2022-2025, The Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`mjlab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse as quat_rotate_inverse
from mjlab.utils.lab_api.math import yaw_quat

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def feet_air_time_positive_biped(
    env,
    command_name: str,
    threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    # compute the reward
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_link_ang_vel_w[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    dof_error = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)

    return (
        dof_error
        * (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1)
        * (torch.abs(env.command_manager.get_command(command_name)[:, 2]) < 0.1)
    )


def joint_deviation_l1(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute deviation from default joint pose."""
    asset: Entity = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_error), dim=1)


# Sagittal joints share the same axis direction on the left and right legs, so a
# symmetric gait has the same long-run average angle on both sides.
_SAGITTAL_JOINT_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    ("left_knee_joint", "right_knee_joint"),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
)


def update_leg_symmetry_ema(
    ema_left: torch.Tensor,
    ema_right: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    decay: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smooth both legs and return the mean absolute difference."""
    ema_left = decay * ema_left + (1.0 - decay) * left
    ema_right = decay * ema_right + (1.0 - decay) * right
    error = torch.mean(torch.abs(ema_left - ema_right), dim=-1)
    return ema_left, ema_right, error


class LegGaitSymmetry:
    """Penalize one leg keeping a larger step than the other.

    A normal step makes the left and right sagittal angles differ at one
    instant. Their exponential moving averages still match when both legs use
    the same stride. A right-leg lunge with the left leg only trailing keeps
    the two averages apart.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.decay = float(cfg.params.get("ema_decay", 0.99))
        robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.robot = robot
        name_to_index = {name: index for index, name in enumerate(robot.joint_names)}
        left_ids = [name_to_index[left] for left, _right in _SAGITTAL_JOINT_PAIRS]
        right_ids = [name_to_index[right] for _left, right in _SAGITTAL_JOINT_PAIRS]
        self.left_ids = torch.tensor(left_ids, dtype=torch.long, device=env.device)
        self.right_ids = torch.tensor(right_ids, dtype=torch.long, device=env.device)
        num_joints = len(left_ids)
        self.ema_left = torch.zeros(env.num_envs, num_joints, device=env.device)
        self.ema_right = torch.zeros(env.num_envs, num_joints, device=env.device)
        self.ready = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        left, right = self._joint_pos()
        self.ema_left[env_ids] = left[env_ids]
        self.ema_right[env_ids] = right[env_ids]
        self.ready[env_ids] = True

    def _joint_pos(self) -> tuple[torch.Tensor, torch.Tensor]:
        joint_pos = self.robot.data.joint_pos
        return joint_pos[:, self.left_ids], joint_pos[:, self.right_ids]

    def __call__(self, env: ManagerBasedRlEnv, **_params) -> torch.Tensor:
        del env
        left, right = self._joint_pos()
        fresh = ~self.ready
        if bool(fresh.any()):
            self.ema_left[fresh] = left[fresh]
            self.ema_right[fresh] = right[fresh]
            self.ready[fresh] = True
        updated_left, updated_right, error = update_leg_symmetry_ema(
            self.ema_left, self.ema_right, left, right, self.decay
        )
        self.ema_left = updated_left
        self.ema_right = updated_right
        return error


def lin_vel_z_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base vertical linear velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])
