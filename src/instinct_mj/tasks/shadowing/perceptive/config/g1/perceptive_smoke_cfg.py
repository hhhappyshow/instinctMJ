"""Smoke-test perceptive shadowing on a local flat STL and standing motion.

The original ``MOTION_FOLDER`` is left unchanged. These configs retarget only
the smoke tasks at ``data/perceptive_smoke`` (or ``INSTINCT_PERCEPTIVE_SMOKE_DIR``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .perceptive_shadowing_cfg import (
    G1PerceptiveShadowingOneMotionEnvCfg,
    G1PerceptiveShadowingOneMotionEnvCfg_PLAY,
)

_REPO_ROOT = Path(__file__).resolve().parents[7]


def smoke_motion_folder() -> str:
    override = os.environ.get("INSTINCT_PERCEPTIVE_SMOKE_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return str(_REPO_ROOT / "data" / "perceptive_smoke")


def _retarget_smoke_scene(scene) -> None:
    motion_folder = smoke_motion_folder()
    motion_reference = next(sensor_cfg for sensor_cfg in scene.sensors if sensor_cfg.name == "motion_reference")
    motion_name = list(motion_reference.motion_buffers.keys())[0]
    motion_buffer = motion_reference.motion_buffers[motion_name]
    motion_buffer.path = motion_folder
    motion_buffer.metadata_yaml = os.path.join(motion_folder, "metadata.yaml")
    terrain_cfg = scene.terrain.terrain_generator.sub_terrains["motion_matched"]
    terrain_cfg.path = motion_folder
    terrain_cfg.metadata_yaml = motion_buffer.metadata_yaml
    # A convex slab does not need CoACD. Native mesh collision keeps the STL load path.
    terrain_cfg.collision_coacd = False
    terrain_cfg.collision_coacd_visualize_collision_hulls = False
    terrain_cfg.collision_coacd_prewarm_all = False
    scene.terrain.terrain_generator.num_rows = 1
    scene.terrain.terrain_generator.num_cols = 1


@dataclass(kw_only=True)
class G1PerceptiveShadowingSmokeEnvCfg(G1PerceptiveShadowingOneMotionEnvCfg):
    def __post_init__(self) -> None:
        _retarget_smoke_scene(self.scene)
        super().__post_init__()
        _retarget_smoke_scene(self.scene)
        self.scene.num_envs = 16


@dataclass(kw_only=True)
class G1PerceptiveShadowingSmokeEnvCfg_PLAY(G1PerceptiveShadowingOneMotionEnvCfg_PLAY):
    def __post_init__(self) -> None:
        _retarget_smoke_scene(self.scene)
        super().__post_init__()
        _retarget_smoke_scene(self.scene)
        self.scene.num_envs = 1
