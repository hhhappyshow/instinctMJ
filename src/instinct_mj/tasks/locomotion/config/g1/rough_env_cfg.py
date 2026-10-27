"""G1 velocity tracking on a small rough-terrain set.

Baseline and RayCaster share the terrain, commands, rewards, terminations, and
events. The RayCaster actor appends the height scan. The baseline actor does not.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mjlab.envs.mdp as mdp
import torch
from mjlab.managers import CurriculumTermCfg as CurrTerm
from mjlab.managers import RewardTermCfg as RewTerm
from mjlab.managers import EventTermCfg as Event
from mjlab.managers import ObservationGroupCfg as ObsGroup
from mjlab.managers import ObservationTermCfg as ObsTerm
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import flat, hf_pyramid_slope, pyramid_stairs, random_rough
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer.viewer_config import ViewerConfig

import instinct_mj.envs.mdp as instinct_mdp
import instinct_mj.tasks.locomotion.mdp as locomotion_mdp
from instinct_mj.assets.unitree_g1 import (
    G1_29DOF_TORSOBASE_POPSICLE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuator_cfgs,
)
from instinct_mj.envs.manager_based_rl_env_cfg import InstinctRlEnvCfg
from instinct_mj.perception.height_scan import (
    HEIGHT_SCAN_CLIP,
    height_scan_observation,
    make_height_scanner_cfg,
)
from instinct_mj.terrains import TerrainEntityCfg

from .flat_env_cfg import _actions_cfg, _commands_cfg, _events_cfg, _rewards_cfg, _terminations_cfg


def _rough_rewards_cfg() -> dict:
    """Flat rewards plus one penalty for a one-sided walking stride."""
    rewards = _rewards_cfg()
    rewards["leg_gait_symmetry"] = RewTerm(
        func=locomotion_mdp.LegGaitSymmetry,
        weight=-0.5,
        params={"ema_decay": 0.99},
    )
    return rewards

G1_CFG = G1_29DOF_TORSOBASE_POPSICLE_CFG

TERRAIN_COLUMN_NAMES = ("flat", "slope", "stairs", "rough")
TERRAIN_STAGE_NAMES = (
    "stage1_0cm",
    "stage2_2cm",
    "stage3_4cm",
    "stage4_6cm",
    "stage5_8cm",
    "stage6_10cm",
)
TERRAIN_SEED = 0
# Row t in [0, 1] maps to difficulty 0.05 + 0.95*t.
# Six rows keep each stair step near 2 cm. The last row is 10 cm.
_PATCH_SIZE = (8.0, 8.0)
_STAIRS_BORDER_WIDTH = 0.5
# Third-highest row. Rows increase with step height, so this is stage 4 of 6.
_PLAY_STAIRS_ROW = len(TERRAIN_STAGE_NAMES) - 3
_PLAY_STAIRS_FRONT_CLEARANCE = 0.4


def make_rough_terrain_cfg() -> TerrainGeneratorCfg:
    """Three-row curriculum shared by the baseline and RayCaster tasks."""
    return TerrainGeneratorCfg(
        seed=TERRAIN_SEED,
        curriculum=True,
        size=_PATCH_SIZE,
        border_width=1.0,
        border_height=0.5,
        num_rows=len(TERRAIN_STAGE_NAMES),
        num_cols=len(TERRAIN_COLUMN_NAMES),
        difficulty_range=(0.05, 1.0),
        sub_terrains={
            "flat": flat(proportion=0.25, size=_PATCH_SIZE),
            "slope": hf_pyramid_slope(
                proportion=0.25,
                size=_PATCH_SIZE,
                slope_range=(0.0, 0.12),
                platform_width=1.0,
                border_width=0.25,
            ),
            "stairs": pyramid_stairs(
                proportion=0.25,
                size=_PATCH_SIZE,
                step_height_range=(0.0, 0.10),
                step_width=0.35,
                platform_width=1.0,
                border_width=_STAIRS_BORDER_WIDTH,
            ),
            "rough": random_rough(
                proportion=0.25,
                size=_PATCH_SIZE,
                noise_range=(0.01, 0.04),
                noise_step=0.005,
                border_width=0.25,
                scale_with_difficulty=True,
            ),
        },
        add_lights=False,
    )


def _robot_cfg():
    robot_cfg = copy.deepcopy(G1_CFG)
    robot_cfg.articulation.actuators = copy.deepcopy(beyondmimic_g1_29dof_actuator_cfgs)
    return robot_cfg


def _contact_sensors() -> tuple[ContactSensorCfg, ContactSensorCfg]:
    feet_contact_forces = ContactSensorCfg(
        name="feet_contact_forces",
        primary=ContactMatch(
            mode="body",
            pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
            entity="robot",
        ),
        secondary=None,
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
        history_length=3,
    )
    base_contact_forces = ContactSensorCfg(
        name="base_contact_forces",
        primary=ContactMatch(
            mode="body",
            pattern=(
                "torso_link",
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_.*",
                ".*_hip_.*",
                ".*_knee_.*",
            ),
            entity="robot",
        ),
        secondary=None,
        fields=("force",),
        reduce="netforce",
        track_air_time=False,
        history_length=3,
    )
    return feet_contact_forces, base_contact_forces


@dataclass(kw_only=True)
class G1RoughSceneCfg(SceneCfg):
    include_height_scanner: bool = True
    show_height_rays: bool = False

    def __post_init__(self) -> None:
        feet_contact_forces, base_contact_forces = _contact_sensors()
        sensors = [feet_contact_forces, base_contact_forces]
        if self.include_height_scanner:
            sensors.append(make_height_scanner_cfg(debug_vis=self.show_height_rays, show_rays=self.show_height_rays))
        self.terrain = TerrainEntityCfg(
            terrain_type="generator",
            terrain_generator=make_rough_terrain_cfg(),
            max_init_terrain_level=0,
        )
        self.entities = {"robot": _robot_cfg()}
        self.sensors = tuple(sensors)


def _observations_cfg(*, include_height_scan: bool) -> dict[str, ObsGroup]:
    policy_terms = {
        "base_ang_vel": ObsTerm(func=instinct_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)),
        "projected_gravity": ObsTerm(
            func=instinct_mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "velocity_commands": ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
        "joint_vel": ObsTerm(func=locomotion_mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5)),
        "actions": ObsTerm(func=instinct_mdp.last_action),
    }
    critic_terms = {
        "base_lin_vel": ObsTerm(func=mdp.base_lin_vel),
        "base_ang_vel": ObsTerm(func=instinct_mdp.base_ang_vel),
        "projected_gravity": ObsTerm(func=instinct_mdp.projected_gravity),
        "velocity_commands": ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel),
        "joint_vel": ObsTerm(func=locomotion_mdp.joint_vel),
        "actions": ObsTerm(func=instinct_mdp.last_action),
    }
    if include_height_scan:
        height_term = ObsTerm(
            func=height_scan_observation,
            params={"sensor_name": "height_scanner"},
            clip=list(HEIGHT_SCAN_CLIP),
        )
        policy_terms["height_scan"] = height_term
        critic_terms["height_scan"] = ObsTerm(
            func=height_scan_observation,
            params={"sensor_name": "height_scanner"},
            clip=list(HEIGHT_SCAN_CLIP),
        )
    return {
        "policy": ObsGroup(terms=policy_terms, enable_corruption=True, concatenate_terms=False),
        "critic": ObsGroup(terms=critic_terms, enable_corruption=False, concatenate_terms=False),
    }


@dataclass(kw_only=True)
class G1LocomotionRoughEnvCfg(InstinctRlEnvCfg):
    include_height_scan: bool = False
    scene: G1RoughSceneCfg = field(default_factory=G1RoughSceneCfg)
    actions: dict = field(default_factory=_actions_cfg)
    commands: dict = field(default_factory=_commands_cfg)
    observations: dict = field(default_factory=lambda: _observations_cfg(include_height_scan=False))
    rewards: dict = field(default_factory=_rough_rewards_cfg)
    terminations: dict = field(default_factory=_terminations_cfg)
    events: dict = field(default_factory=_events_cfg)
    curriculum: dict = field(
        default_factory=lambda: {"terrain_levels": CurrTerm(func=locomotion_mdp.terrain_levels_vel)}
    )
    monitors: dict = field(default_factory=dict)
    viewer: ViewerConfig = field(
        default_factory=lambda: ViewerConfig(
            lookat=(0.0, 0.0, 0.0),
            distance=4.0,
            elevation=-15.0,
            azimuth=45.0,
            origin_type=ViewerConfig.OriginType.ASSET_ROOT,
            entity_name="robot",
        )
    )
    sim: SimulationCfg = field(
        default_factory=lambda: SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                solver="newton",
                iterations=10,
                ls_iterations=20,
                ccd_iterations=500,
            ),
        )
    )
    decimation: int = 4
    episode_length_s: float = 20.0

    def __post_init__(self) -> None:
        self.sim.njmax = 300
        self.sim.nconmax = 70
        self.scene.num_envs = 64
        self.scene.include_height_scanner = True
        self.observations = _observations_cfg(include_height_scan=self.include_height_scan)
        joint_pos_action = self.actions["joint_pos"]
        joint_pos_action.scale = copy.deepcopy(beyondmimic_action_scale)
        self.run_name = "G1RoughRayCaster" if self.include_height_scan else "G1RoughBaseline"


def _play_spawn_in_front_of_stairs(env, env_ids) -> None:
    """Stand on the flat border in front of the third-highest stair pyramid."""
    del env_ids
    terrain = env.scene.terrain
    column = TERRAIN_COLUMN_NAMES.index("stairs")
    origin = terrain.terrain_origins[_PLAY_STAIRS_ROW, column]
    face_distance = _PATCH_SIZE[1] / 2.0 - _STAIRS_BORDER_WIDTH
    count = terrain.env_origins.shape[0]
    spacing = 1.0
    x_offsets = (torch.arange(count, device=origin.device, dtype=origin.dtype) - (count - 1) / 2.0) * spacing
    terrain.terrain_levels[:] = _PLAY_STAIRS_ROW
    terrain.terrain_types[:] = column
    terrain.env_origins[:, 0] = origin[0] + x_offsets
    terrain.env_origins[:, 1] = origin[1] - face_distance - _PLAY_STAIRS_FRONT_CLEARANCE
    terrain.env_origins[:, 2] = 0.0


def instinct_g1_locomotion_rough_env_cfg(*, play: bool = False, include_height_scan: bool = False):
    cfg = G1LocomotionRoughEnvCfg(include_height_scan=include_height_scan)
    if play:
        cfg.scene.num_envs = 1
        cfg.scene.show_height_rays = include_height_scan
        for sensor in cfg.scene.sensors:
            if sensor.name == "height_scanner":
                sensor.debug_vis = include_height_scan
                sensor.viz.show_rays = include_height_scan
        cfg.sim.mujoco.ccd_iterations = 50
        base_velocity_cmd: UniformVelocityCommandCfg = cfg.commands["base_velocity"]
        base_velocity_cmd.resampling_time_range = (2.0, 2.0)
        cfg.events["base_external_force_torque"] = None
        cfg.events["push_robot"] = None
        cfg.events = {
            "place_in_front_of_stairs": Event(
                func=_play_spawn_in_front_of_stairs,
                mode="startup",
            ),
            **cfg.events,
        }
        cfg.events["reset_base"].params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (math.pi / 2, math.pi / 2),
        }
        cfg.events["reset_base"].params["velocity_range"] = {
            axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        cfg.terminations["time_out"] = None
        cfg.curriculum = {}
        cfg.observations["policy"].enable_corruption = False
    return cfg
