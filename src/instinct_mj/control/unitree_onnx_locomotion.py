"""Official unitree_rl_mjlab G1 velocity policy.

The ONNX file maps a 98-dimensional observation to 29 joint targets. Observation
order, command limits and the 0.6 s gait phase follow deploy.yaml. The phase
clock is the caller's low-level time in seconds, advancing 0.02 s per control
tick. It must not use a 10 Hz navigation step.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
import yaml
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from instinct_mj.assets.unitree_g1 import beyondmimic_g1_29dof_actuator_cfgs

_REPO = Path(__file__).resolve().parents[3]
CANDIDATE_DIR = _REPO / "outputs/p4/checkpoint_candidates/unitree_rl_mjlab"
DEFAULT_ONNX = CANDIDATE_DIR / "policy.onnx"
DEFAULT_DEPLOY = CANDIDATE_DIR / "deploy.yaml"
OFFICIAL_XML = CANDIDATE_DIR / "source/src/assets/robots/unitree_g1/xmls/g1.xml"

# Body order in the official g1.xml. deploy.yaml joint_ids_map is identity.
OFFICIAL_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def official_g1_robot_cfg() -> EntityCfg:
    """Pelvis-base G1 used by the official policy, with robot geoms in group 1.

    Group 1 is only a ray-exclusion tag. Collision types, foot friction and
    contact dimensions match the configuration that reproduced the official gait.
    """
    if not OFFICIAL_XML.is_file():
        raise FileNotFoundError(f"Official G1 XML is missing: {OFFICIAL_XML}")

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(OFFICIAL_XML))
        for geom in spec.geoms:
            geom.group = 1
        return spec

    full_collision = CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 1},
        conaffinity={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 1},
        condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
        priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
        friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
    )
    home_keyframe = EntityCfg.InitialStateCfg(
        pos=(0, 0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.35,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "right_shoulder_roll_joint": -0.18,
        },
        joint_vel={".*": 0.0},
    )
    return EntityCfg(
        init_state=home_keyframe,
        collisions=(full_collision,),
        spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(copy.deepcopy(beyondmimic_g1_29dof_actuator_cfgs)),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


class UnitreeOnnxLocomotion(nn.Module):
    """Frozen official policy. ``command`` matches the frozen flat controller."""

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        super().__init__()
        import onnxruntime as ort

        onnx_path = Path(checkpoint).expanduser().resolve() if checkpoint else DEFAULT_ONNX
        deploy_path = onnx_path.parent / "deploy.yaml"
        if onnx_path.suffix != ".onnx" or not onnx_path.is_file() or not deploy_path.is_file():
            raise FileNotFoundError(f"Expected policy.onnx with a sibling deploy.yaml, got {onnx_path}")
        deploy = yaml.safe_load(deploy_path.read_text())
        ranges = deploy["commands"]["base_velocity"]["ranges"]
        limits = torch.tensor(
            [ranges[key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")], dtype=torch.float32
        )
        self.register_buffer("limits", limits)
        self.control_dt = float(deploy["step_dt"])
        if abs(self.control_dt - 0.02) > 1e-9:
            raise ValueError(f"Official policy control period must be 0.02 s, got {self.control_dt}")
        self.phase_period = float(deploy["observations"]["gait_phase"]["params"]["period"])
        self.stand_norm = 0.1
        self.default_joint_pos = torch.tensor(deploy["default_joint_pos"], dtype=torch.float32)
        if tuple(deploy["joint_ids_map"]) != tuple(range(29)):
            raise ValueError("deploy.yaml joint_ids_map is not identity; official joint order would be wrong")
        if self.default_joint_pos.shape != (29,):
            raise ValueError("deploy.yaml default_joint_pos must contain 29 values")
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.fingerprint = {"onnx_sha256": _sha256(onnx_path), "deploy_sha256": _sha256(deploy_path)}
        self.onnx_path = onnx_path
        self.eval()
        self.to(device)

    def train(self, mode: bool = True):
        return super().train(False)

    def command(self, nominal: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return (nominal + residual).clamp(self.limits[:, 0], self.limits[:, 1])

    def observe(
        self,
        data,
        velocity: torch.Tensor,
        previous: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble one 98-wide observation per environment.

        ``time_s`` is seconds since that environment's low-level clock was reset.
        """
        phase = torch.remainder(time_s, self.phase_period) / self.phase_period
        angle = phase * (2.0 * torch.pi)
        encoded = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        standing = torch.linalg.norm(velocity, dim=-1) < self.stand_norm
        encoded = torch.where(standing.unsqueeze(-1), torch.zeros_like(encoded), encoded)
        observation = torch.cat(
            (
                data.root_link_ang_vel_b,
                data.projected_gravity_b,
                velocity,
                encoded,
                data.joint_pos - data.default_joint_pos,
                data.joint_vel,
                previous,
            ),
            dim=-1,
        )
        if observation.shape[-1] != 98:
            raise ValueError(f"Official observation must be 98-dimensional, got {observation.shape[-1]}")
        return observation

    @torch.no_grad()
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != 98 or not torch.isfinite(observation).all():
            raise ValueError("Expected finite 98-dimensional official observations")
        rows = observation.detach().cpu().numpy().astype(np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]
        # The released graph fixes the leading dimension at 1.
        raw = np.stack([self.session.run(None, {self.input_name: row[None, :]})[0][0] for row in rows])
        return torch.as_tensor(raw, dtype=observation.dtype, device=observation.device)
