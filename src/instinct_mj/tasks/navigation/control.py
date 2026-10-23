"""Two-rate residual navigation action. The low-level controller runs at 50 Hz."""
from dataclasses import dataclass
import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from instinct_mj.control.frozen_locomotion import FrozenLocomotion
from instinct_mj.control.unitree_onnx_locomotion import OFFICIAL_JOINT_ORDER, UnitreeOnnxLocomotion
from instinct_mj.tasks.locomotion.config.g1.flat_env_cfg import instinct_g1_locomotion_flat_env_cfg


@dataclass(kw_only=True)
class NavigationActionCfg(ActionTermCfg):
    checkpoint: str = ''
    stage: int = 0
    backend: str = 'frozen'

    def build(self, env):
        return NavigationAction(self, env)


class NavigationAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        low_cfg = instinct_g1_locomotion_flat_env_cfg().actions['joint_pos']
        if cfg.backend == 'frozen':
            if not cfg.checkpoint:
                raise ValueError('Set INSTINCT_LOCOMOTION_CHECKPOINT to an explicitly validated flat model; no random fallback')
            self.policy = FrozenLocomotion(cfg.checkpoint, env.device)
            saved = self.policy.saved_action_config
            if saved['scale'] != low_cfg.scale or saved['use_default_offset'] != low_cfg.use_default_offset:
                raise ValueError('Checkpoint action scaling does not match current G1 configuration')
        elif cfg.backend == 'unitree_onnx':
            self.policy = UnitreeOnnxLocomotion(cfg.checkpoint, env.device)
            if tuple(self._entity.joint_names) != OFFICIAL_JOINT_ORDER:
                raise ValueError('Official ONNX requires the pelvis-base G1 joint order')
        else:
            raise ValueError(f'Unknown navigation low-level backend: {cfg.backend}')
        self.low = low_cfg.build(env)
        if self.low.action_dim != 29:
            raise ValueError('Expected 29 native G1 joint actions')
        if cfg.backend == 'unitree_onnx' and tuple(self.low._target_names) != OFFICIAL_JOINT_ORDER:
            raise ValueError('Low-level joint targets are not in official ONNX order')
        ratio = self.policy.control_dt / env.physics_dt
        if abs(ratio - round(ratio)) > 1e-6 or env.cfg.decimation % round(ratio):
            raise ValueError('High-level step must be an integer multiple of the low-level control period')
        self.period = round(ratio)
        self.phase = 0
        self.low_updates = 0
        self._raw = torch.zeros(env.num_envs, 3, device=env.device)
        self.velocity = torch.zeros_like(self._raw)
        self.goal = torch.zeros_like(self._raw)
        self.previous_low = torch.zeros(env.num_envs, 29, device=env.device)
        self.low_time = torch.zeros(env.num_envs, device=env.device)
        self.foot_steps = torch.zeros(env.num_envs, 2, device=env.device)
        self.max_single_swing = torch.zeros(env.num_envs, 2, device=env.device)
        self._swing_run = torch.zeros(env.num_envs, 2, device=env.device)
        self._foot_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        try:
            self._feet = env.scene['feet_contact_forces']
        except KeyError:
            self._feet = None
        self.best_distance = torch.zeros(env.num_envs, device=env.device)
        self.progress = torch.zeros_like(self.best_distance)
        self.reward_step = -1

    @property
    def raw_action(self):
        return self._raw

    @property
    def action_dim(self):
        return 3

    def relative_goal(self):
        robot = self._entity.data
        delta = self.goal - robot.root_link_pos_w
        delta[:, 2] = 0
        local = quat_apply_inverse(robot.root_link_quat_w, delta)
        return torch.cat((local[:, :2], torch.atan2(local[:, 1], local[:, 0])[:, None]), -1)

    def distance(self):
        return (self.goal[:, :2] - self._entity.data.root_link_pos_w[:, :2]).norm(dim=-1)

    def process_actions(self, actions):
        self._raw.copy_(actions)
        goal = self.relative_goal()
        nominal = torch.stack((.5*goal[:, 0], .5*goal[:, 1], goal[:, 2]), -1)
        nominal = self.policy.command(nominal, torch.zeros_like(nominal))
        # Symmetric residual capacity is derived from the saved command intervals.
        scale = (self.policy.limits[:, 1] - self.policy.limits[:, 0]) / 2
        self.velocity.copy_(self.policy.command(nominal, actions.clamp(-1, 1)*scale))
        self.phase = 0

    def apply_actions(self):
        if self.phase % self.period == 0:
            self.low_updates += 1
            d = self._entity.data
            if self.cfg.backend == 'unitree_onnx':
                obs = self.policy.observe(d, self.velocity, self.previous_low, self.low_time)
                self.low_time += self.policy.control_dt
            else:
                obs = torch.cat((d.root_link_ang_vel_b, d.projected_gravity_b, self.velocity,
                                 d.joint_pos-d.default_joint_pos, d.joint_vel, self.previous_low), -1)
            self.previous_low.copy_(self.policy(obs))
            self.low.process_actions(self.previous_low)
            self._update_gait()
        self.low.apply_actions()
        self.phase += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.velocity[ids] = 0
        self.previous_low[ids] = 0
        self.low_time[ids] = 0
        self.foot_steps[ids] = 0
        self.max_single_swing[ids] = 0
        self._swing_run[ids] = 0
        self._foot_contact[ids] = False
        self.low.reset(ids)
        count = self._raw[ids].shape[0]
        origin = self._env.scene.env_origins[ids]
        angle = torch.rand(count, device=self.device)*1.2-.6
        distance = torch.rand(count, device=self.device)*2+3
        if self.cfg.stage == 3:
            distance = torch.rand(count, device=self.device)*.5+4.5
        self.goal[ids, :2] = origin[:, :2] + torch.stack((distance*angle.cos(), distance*angle.sin()), -1)
        self.goal[ids, 2] = 0
        self.best_distance[ids] = distance
        self.progress[ids] = 0
        choice = torch.randint(0, 3, (count,), device=self.device) if self.cfg.stage >= 2 else torch.ones(count, device=self.device, dtype=torch.long)
        heights = (.3, .45, .6)
        for index in range(3):
            pose = torch.zeros(count, 7, device=self.device)
            pose[:, 3] = 1
            along = torch.rand(count, device=self.device)*.8 + 1.2
            if self.cfg.stage == 3:
                along = torch.full_like(along, 1.0 + index)
            lateral = torch.rand(count, device=self.device)*1.0-.5
            pose[:, 0] = origin[:, 0] + along*angle.cos() - lateral*angle.sin()
            pose[:, 1] = origin[:, 1] + along*angle.sin() + lateral*angle.cos()
            pose[:, 2] = heights[index]/2
            active = torch.ones_like(choice, dtype=torch.bool) if self.cfg.stage == 3 else choice == index
            if self.cfg.stage == 0:
                active[:] = False
            pose[~active, 2] = -10
            self._env.scene[f'obstacle_{index}'].write_mocap_pose_to_sim(pose, env_ids)
        self.reward_step = -1

    def _update_gait(self):
        if self._feet is None:
            return
        contact = self._feet.data.current_contact_time > 0
        rising = contact & ~self._foot_contact
        self.foot_steps += rising.float()
        other = contact.flip(-1)
        swinging = (~contact) & other
        self._swing_run = torch.where(swinging, self._swing_run + 1, torch.zeros_like(self._swing_run))
        self.max_single_swing = torch.maximum(self.max_single_swing, self._swing_run)
        self._foot_contact = contact
