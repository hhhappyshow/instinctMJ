"""Build a minimal perceptive-shadowing dataset: flat STL, standing motion, metadata."""

from __future__ import annotations

import os
import re
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import yaml

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH, _POPSICLE_INIT_STATE

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "perceptive_smoke"

TERRAIN_SIZE_XY = (30.0, 16.0)
TERRAIN_THICKNESS = 0.2
STAND_DURATION_S = 3.0
STAND_FRAMERATE = 50.0
ROOT_HEIGHT = float(_POPSICLE_INIT_STATE.pos[2])


def _repo_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            names.append(name)
    if len(names) != 29:
        raise RuntimeError(f"Expected 29 actuated joints in {G1_MJCF_PATH}, found {len(names)}: {names}")
    return names


def _default_joint_pos(joint_names: list[str]) -> np.ndarray:
    joint_pos = np.zeros(len(joint_names), dtype=np.float32)
    patterns = list(_POPSICLE_INIT_STATE.joint_pos.items())
    for index, joint_name in enumerate(joint_names):
        for pattern, value in patterns:
            if re.fullmatch(pattern, joint_name):
                joint_pos[index] = float(value)
    return joint_pos


def _make_terrain(output_path: Path) -> trimesh.Trimesh:
    # Box is centered at the origin. Shift it down so the walkable face is z=0
    # and the mesh has thickness below that face. MotionMatchedTerrainCfg treats
    # (0, 0, 0) as the surface center.
    mesh = trimesh.creation.box(extents=(TERRAIN_SIZE_XY[0], TERRAIN_SIZE_XY[1], TERRAIN_THICKNESS))
    mesh.apply_translation((0.0, 0.0, -TERRAIN_THICKNESS / 2.0))
    mesh.export(output_path)
    return mesh


def _make_motion(output_path: Path, joint_names: list[str], joint_pos: np.ndarray) -> None:
    num_frames = int(STAND_DURATION_S * STAND_FRAMERATE)
    base_pos_w = np.zeros((num_frames, 3), dtype=np.float32)
    base_pos_w[:, 2] = ROOT_HEIGHT
    # AmassMotion retargeted files use wxyz quaternions.
    base_quat_w = np.zeros((num_frames, 4), dtype=np.float32)
    base_quat_w[:, 0] = 1.0
    joint_trajectory = np.repeat(joint_pos[None, :], num_frames, axis=0)
    np.savez(
        output_path,
        framerate=np.float64(STAND_FRAMERATE),
        joint_names=np.asarray(joint_names),
        joint_pos=joint_trajectory,
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
    )


def _make_metadata(output_path: Path) -> None:
    # TerrainMotion casts terrain_id with int(), so the id must be numeric.
    payload = {
        "terrains": [
            {"terrain_id": 0, "terrain_file": "terrain.stl"},
        ],
        "motion_files": [
            {"terrain_id": 0, "motion_file": "stand.retargeted.npz", "weight": 1.0},
        ],
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _validate(output_dir: Path, joint_names: list[str]) -> None:
    terrain = trimesh.load(output_dir / "terrain.stl", force="mesh")
    if terrain.vertices.shape[0] == 0 or terrain.faces.shape[0] == 0:
        raise RuntimeError("terrain.stl has no vertices or faces.")
    top_z = float(terrain.vertices[:, 2].max())
    if abs(top_z) > 1e-5:
        raise RuntimeError(f"terrain.stl top surface is z={top_z}, expected 0.")
    if not terrain.is_watertight:
        raise RuntimeError("terrain.stl is not watertight.")

    with (output_dir / "metadata.yaml").open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    if int(metadata["terrains"][0]["terrain_id"]) != int(metadata["motion_files"][0]["terrain_id"]):
        raise RuntimeError("metadata.yaml terrain_id values do not match.")
    for entry in metadata["terrains"]:
        if not (output_dir / entry["terrain_file"]).is_file():
            raise RuntimeError(f"Missing terrain file {entry['terrain_file']}")
    for entry in metadata["motion_files"]:
        if not (output_dir / entry["motion_file"]).is_file():
            raise RuntimeError(f"Missing motion file {entry['motion_file']}")

    motion = np.load(output_dir / "stand.retargeted.npz", allow_pickle=True)
    required = ("framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w")
    missing = [key for key in required if key not in motion.files]
    if missing:
        raise RuntimeError(f"stand.retargeted.npz is missing fields: {missing}")
    loaded_names = motion["joint_names"].tolist()
    if loaded_names != joint_names:
        raise RuntimeError("NPZ joint_names do not match the MuJoCo model order.")
    joint_pos = np.asarray(motion["joint_pos"])
    base_pos_w = np.asarray(motion["base_pos_w"])
    base_quat_w = np.asarray(motion["base_quat_w"])
    if joint_pos.shape != (int(STAND_DURATION_S * STAND_FRAMERATE), len(joint_names)):
        raise RuntimeError(f"Unexpected joint_pos shape {joint_pos.shape}")
    if base_pos_w.shape[0] != joint_pos.shape[0] or base_quat_w.shape != (joint_pos.shape[0], 4):
        raise RuntimeError("Root pose shape does not match joint_pos.")
    quat_norm = np.linalg.norm(base_quat_w, axis=1)
    if not np.allclose(quat_norm, 1.0):
        raise RuntimeError("base_quat_w is not unit length.")
    if not np.allclose(base_quat_w[:, 0], 1.0):
        raise RuntimeError("base_quat_w is not wxyz identity.")
    if not np.allclose(base_pos_w[:, 2], ROOT_HEIGHT):
        raise RuntimeError(f"base height is not {ROOT_HEIGHT}.")
    print(f"[OK] wrote {output_dir}")
    print(f"[OK] joints={len(joint_names)} frames={joint_pos.shape[0]} root_z={ROOT_HEIGHT}")


def main() -> None:
    output_dir = Path(os.environ.get("INSTINCT_PERCEPTIVE_SMOKE_DIR", DEFAULT_OUTPUT_DIR)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(G1_MJCF_PATH)
    joint_names = _repo_joint_names(model)
    joint_pos = _default_joint_pos(joint_names)
    _make_terrain(output_dir / "terrain.stl")
    _make_motion(output_dir / "stand.retargeted.npz", joint_names, joint_pos)
    _make_metadata(output_dir / "metadata.yaml")
    _validate(output_dir, joint_names)


if __name__ == "__main__":
    main()
