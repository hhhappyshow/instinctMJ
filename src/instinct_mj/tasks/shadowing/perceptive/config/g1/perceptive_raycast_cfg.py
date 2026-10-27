"""Perceptive shadowing actor that reads the height scan instead of depth.

The smoke motion folder is reused so the task can run without the unpublished
dataset. The original perceptive tasks are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.managers import ObservationTermCfg as ObsTermCfg

from instinct_mj.perception.height_scan import HEIGHT_SCAN_CLIP, height_scan_observation

from .perceptive_smoke_cfg import (
    G1PerceptiveShadowingSmokeEnvCfg,
    G1PerceptiveShadowingSmokeEnvCfg_PLAY,
)

_ACTOR_PROPRIO = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "last_action")
_ACTOR_MOTION = ("joint_pos_ref", "joint_vel_ref", "position_ref", "rotation_ref")
_CRITIC_ONLY = ("link_pos", "link_rot", "base_lin_vel")


def _use_actor_height_scan(cfg) -> None:
    policy = cfg.observations["policy"]
    policy.terms.pop("depth_image", None)
    policy.terms["height_scan"] = ObsTermCfg(
        func=height_scan_observation,
        params={"sensor_name": "height_scanner"},
        clip=list(HEIGHT_SCAN_CLIP),
    )
    actor_names = set(policy.terms)
    expected = set(_ACTOR_PROPRIO) | set(_ACTOR_MOTION) | {"height_scan"}
    if actor_names != expected:
        raise RuntimeError(f"RayCaster actor terms changed unexpectedly: {sorted(actor_names)}")
    leaked = actor_names & set(_CRITIC_ONLY)
    if leaked:
        raise RuntimeError(f"Critic-only terms leaked into the actor: {sorted(leaked)}")


@dataclass(kw_only=True)
class G1PerceptiveShadowingRayCasterEnvCfg(G1PerceptiveShadowingSmokeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        _use_actor_height_scan(self)


@dataclass(kw_only=True)
class G1PerceptiveShadowingRayCasterEnvCfg_PLAY(G1PerceptiveShadowingSmokeEnvCfg_PLAY):
    def __post_init__(self) -> None:
        super().__post_init__()
        _use_actor_height_scan(self)
        scanner = next(sensor for sensor in self.scene.sensors if sensor.name == "height_scanner")
        scanner.debug_vis = True
        scanner.viz.show_rays = True
