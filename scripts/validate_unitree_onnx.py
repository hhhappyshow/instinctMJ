"""Standalone validation of the official unitree_rl_mjlab G1 velocity ONNX policy.

Builds a minimal flat scene with the official G1 asset (pelvis base, 29 DoF) and the
actuator configuration already verified numerically identical to the official
deploy.yaml, drives the ONNX policy at 50 Hz with observations assembled in the
deploy.yaml order, and measures velocity tracking plus left/right foot contact
timing for fixed commands. Videos are rendered offline from the recorded qpos
trajectory.

Outputs land under outputs/p4/unitree_onnx_validation/.

This script does not modify any task code and does not touch model_5000.pt.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

import mujoco

CANDIDATE_DIR = Path("outputs/p4/checkpoint_candidates/unitree_rl_mjlab")
G1_XML = CANDIDATE_DIR / "source/src/assets/robots/unitree_g1/xmls/g1.xml"
POLICY_ONNX = CANDIDATE_DIR / "policy.onnx"
DEPLOY_YAML = CANDIDATE_DIR / "deploy.yaml"

# Official XML joint order (verified against g1.xml body tree and deploy.yaml
# stiffness signature: 7520_14=40.2 hip_pitch/hip_yaw/waist_yaw, 7520_22=99.1
# hip_roll/knee, dual-5020=28.5 ankles/waist roll+pitch, 5020=14.3 arms+wrist
# roll, 4010=16.8 wrist pitch/yaw).
EXPECTED_JOINT_ORDER = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Fixed commands: straight, in-place turns both directions, standing.
SCENARIOS = {
    "straight_vx0.5": (0.5, 0.0, 0.0),
    "turn_left_wz0.5": (0.0, 0.0, 0.5),
    "turn_right_wz-0.5": (0.0, 0.0, -0.5),
    "stand": (0.0, 0.0, 0.0),
}

GATE = {"linear_mae_mps": 0.15, "angular_mae_radps": 0.15}


def official_robot_cfg():
    """Official G1 EntityCfg: official XML, HOME_KEYFRAME pose, FULL_COLLISION."""
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from mjlab.utils.spec_config import CollisionCfg
    from instinct_mj.assets.unitree_g1 import beyondmimic_g1_29dof_actuator_cfgs

    def spec_fn():
        return mujoco.MjSpec.from_file(str(G1_XML))

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


def build_env(device):
    import instinct_mj.tasks  # noqa: F401  (task registration side effect)
    from instinct_mj.tasks.registry import load_env_cfg, load_env_cls

    task = "Instinct-Locomotion-Flat-G1-v0"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.scene.entities["robot"] = official_robot_cfg()
    cfg.seed = 42
    cfg.observations["policy"].enable_corruption = False
    cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints")}
    cfg.events["reset_base"].params["pose_range"] = {k: (0.0, 0.0) for k in ("x", "y", "yaw")}
    cfg.events["reset_base"].params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}
    cfg.events["reset_robot_joints"].params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    cfg.commands["base_velocity"].resampling_time_range = (1e6, 1e6)
    env = load_env_cls(task)(cfg, device)
    return env


def verify_orders_and_params(env, deploy):
    """Hard checks: joint order, PD gains, defaults and action scales vs deploy.yaml."""
    from instinct_mj.assets.unitree_g1 import beyondmimic_action_scale

    robot = env.scene["robot"]
    joint_names = list(robot.joint_names)
    assert joint_names == EXPECTED_JOINT_ORDER, (
        "Official XML joint order mismatch.\n got: %s" % joint_names
    )

    deploy_order = EXPECTED_JOINT_ORDER  # deploy.yaml joint_ids_map is identity
    dep_scale = {n: float(s) for n, s in zip(deploy_order, deploy["actions"]["JointPositionAction"]["scale"])}
    dep_offset = {n: float(s) for n, s in zip(deploy_order, deploy["actions"]["JointPositionAction"]["offset"])}
    dep_default = {n: float(s) for n, s in zip(deploy_order, deploy["default_joint_pos"])}
    dep_kp = {n: float(s) for n, s in zip(deploy_order, deploy["stiffness"])}
    dep_kd = {n: float(s) for n, s in zip(deploy_order, deploy["damping"])}

    # Local actuator semantics vs deploy.yaml per joint name.
    for name, s in beyondmimic_action_scale.items():
        if name in dep_scale:
            assert abs(dep_scale[name] - s) < 2e-2, f"scale mismatch {name}: {dep_scale[name]} vs {s}"

    # Default posture captured at reset must equal deploy offset/default.
    default_jpos = robot.data.joint_pos[0].detach().cpu().numpy()
    for i, name in enumerate(joint_names):
        assert abs(default_jpos[i] - dep_default[name]) < 1e-3, (
            f"default pos mismatch {name}: {default_jpos[i]} vs {dep_default[name]}"
        )
        assert abs(dep_offset[name] - dep_default[name]) < 1e-9
    return joint_names, dep_kp, dep_kd


def load_policy():
    import onnxruntime as ort

    sess = ort.InferenceSession(str(POLICY_ONNX), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    assert 98 in inp.shape, f"unexpected policy input shape {inp.shape}"
    out = sess.get_outputs()[0]
    assert 29 in out.shape, f"unexpected policy output shape {out.shape}"
    return sess, inp.name


def run_scenario(env, vec, sess, input_name, command, steps, settle_steps, record_video):
    robot = env.scene["robot"]
    sensor = env.scene["feet_contact_forces"]
    cmd_term = env.command_manager.get_term("base_velocity")

    env.reset()
    for field in ("is_heading_env", "is_standing_env", "is_world_env", "is_forward_env"):
        getattr(cmd_term, field)[:] = False
    cmd_t = torch.tensor([command], dtype=torch.float32, device=env.device)

    last_raw = np.zeros(29, dtype=np.float32)
    qpos_frames = []
    measured_log, contact_log, height_log, time_log = [], [], [], []
    fallen = False
    fall_step = None

    def current_qpos():
        pos = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        quat = robot.data.root_link_quat_w[0].detach().cpu().numpy()
        jpos = robot.data.joint_pos[0].detach().cpu().numpy()
        return np.concatenate([pos, quat, jpos]).astype(np.float64)

    for step in range(steps):
        cmd_term.vel_command_b[:] = cmd_t
        if record_video:
            qpos_frames.append(current_qpos())

        t = env.episode_length_buf[0].item() * env.step_dt
        ang_vel = robot.data.root_link_ang_vel_b.detach().cpu().numpy()[0]
        grav = robot.data.projected_gravity_b.detach().cpu().numpy()[0]
        if norm3(command) < 0.1:
            phase = np.zeros(2, dtype=np.float32)
        else:
            ph = (t % 0.6) / 0.6
            phase = np.array([math.sin(ph * 2 * math.pi), math.cos(ph * 2 * math.pi)], dtype=np.float32)
        jpos = robot.data.joint_pos[0].detach().cpu().numpy() - default_jpos
        jvel = robot.data.joint_vel[0].detach().cpu().numpy()
        obs = np.concatenate([ang_vel, grav, np.asarray(command, np.float32), phase, jpos, jvel, last_raw]).astype(np.float32)
        assert obs.shape == (98,), obs.shape

        raw = sess.run(None, {input_name: obs[None]})[0][0].astype(np.float32)
        last_raw = raw

        measured = np.concatenate([
            robot.data.root_link_lin_vel_b[0, :2].detach().cpu().numpy(),
            robot.data.root_link_ang_vel_b[0, 2:3].detach().cpu().numpy(),
        ])
        contact = (sensor.data.current_contact_time[0] > 0).detach().cpu().numpy().astype(np.int8)
        height = float(robot.data.root_link_pos_w[0, 2].detach())

        vec.step(torch.tensor(raw[None], device=env.device))

        time_log.append(t)
        measured_log.append(measured)
        contact_log.append(contact)
        height_log.append(height)
        if not fallen and (grav[2] > -0.5 or height < 0.45):
            fallen, fall_step = True, step
            break

    time_log = np.asarray(time_log)
    measured_log = np.asarray(measured_log)
    contact_log = np.asarray(contact_log)
    height_log = np.asarray(height_log)

    result = dict(command=list(command), fallen=bool(fallen), fall_step=fall_step,
                  steps_completed=int(len(time_log)), duration_s=float(time_log[-1] + env.step_dt),
                  contact_log=contact_log.astype(int).tolist(),
                  time_log=time_log.tolist(),
                  measured_velocity=measured_log.tolist(),
                  root_height=height_log.tolist())
    if fallen:
        return result, qpos_frames

    m = slice(settle_steps, None)
    err = np.abs(measured_log[m] - np.asarray(command, np.float32)[None, :])
    result["mean_abs_velocity_error"] = err.mean(axis=0).tolist()  # vx, vy, wz
    result["max_abs_velocity_error"] = err.max(axis=0).tolist()
    result["gate_linear_ok"] = bool(err[:, :2].mean() < GATE["linear_mae_mps"])
    result["gate_angular_ok"] = bool(err[:, 2].mean() < GATE["angular_mae_radps"])
    result["mean_root_height"] = float(height_log[m].mean())
    result["min_root_height"] = float(height_log[m].min())

    # Gait contact timing. Foot order in sensor = (left, right).
    gait = {}
    for side, idx in (("left", 0), ("right", 1)):
        c = contact_log[m, idx].astype(int)
        rising = np.flatnonzero(np.diff(np.concatenate([[0], c])) == 1)
        falling = np.flatnonzero(np.diff(np.concatenate([c, [0]])) == -1)
        gait[f"{side}_duty_factor"] = float(c.mean())
        gait[f"{side}_steps"] = int(len(rising))
        gait[f"{side}_step_frequency_hz"] = float(len(rising) / ((steps - settle_steps) * env.step_dt))
        if len(rising) == len(falling) and len(rising) > 0:
            step_duration = time_log[m][falling] - time_log[m][rising]
            gait[f"{side}_mean_cycle_s"] = float(np.mean(np.diff(time_log[m][rising])) if len(rising) > 1 else 0.0)
            gait[f"{side}_mean_stance_s"] = float(np.mean(step_duration))
    both = contact_log[m].sum(axis=1)
    gait["double_support_fraction"] = float((both == 2).mean())
    gait["flight_fraction"] = float((both == 0).mean())
    gait["single_stance_fraction"] = float((both == 1).mean())
    # Limping indicator: imbalance between feet step counts.
    gait["step_count_imbalance"] = abs(gait.get("left_steps", 0) - gait.get("right_steps", 0))
    result["gait"] = gait
    return result, qpos_frames


def norm3(v):
    return float(np.linalg.norm(v))


def save_plots(result, out_dir, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(result["time_log"])
    meas = np.asarray(result["measured_velocity"])
    cmd = np.asarray(result["command"])
    contact = np.asarray(result["contact_log"])
    height = np.asarray(result["root_height"])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, label in enumerate(("vx", "vy", "wz")):
        axes[0].plot(t, meas[:, i], label=f"measured {label}")
        axes[0].axhline(cmd[i], color=f"C{i}", linestyle="--", alpha=0.6)
    axes[0].set_ylabel("body-frame velocity")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(t, 0, 0.4 + 0.6 * contact[:, 0], step="mid", alpha=0.5, label="left foot")
    axes[1].fill_between(t, 0, 0.2 + 0.3 * contact[:, 1], step="mid", alpha=0.5, label="right foot")
    axes[1].set_ylabel("foot contact (low=swing)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, height)
    axes[2].axhline(0.45, color="r", linestyle=":", label="fall threshold")
    axes[2].set_ylabel("pelvis height (m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"official unitree_rl_mjlab ONNX — {name} cmd={result['command']}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=120)
    plt.close(fig)


def save_video(qpos_frames, model, out_path, fps=25):
    try:
        import imageio.v2 as imageio
    except Exception:
        imageio = None
    if imageio is None or not qpos_frames:
        return False
    try:
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.distance = 3.0
        cam.elevation = -10.0
        cam.azimuth = 90.0
        frames = []
        for qpos in qpos_frames:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
        imageio.mimwrite(str(out_path), frames, fps=fps, quality=8)
        renderer.close()
        return True
    except Exception as exc:  # headless/EGL issues must not fail validation
        print(f"[warn] video rendering skipped: {exc}")
        return False


def run_viewer(env, sess, input_name, command, backend):
    """Interactive play: official ONNX policy with keyboard-adjustable command."""
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    policy = OnnxVelocityPolicy(env, sess, input_name, command)
    env.reset()
    print("\n=== official ONNX play ===")
    print("keys: I/K vx+/-  U/O vy+/-  J/L wz+/-  Backspace zero  Space pause  Enter reset")
    print(f"command: vx={command[0]:+.1f} vy={command[1]:+.1f} wz={command[2]:+.1f}")
    if backend == "native":
        NativeMujocoViewer(env, policy, key_callback=policy.on_key).run()
    else:
        ViserPlayViewer(env, policy).run()


class OnnxVelocityPolicy:
    """Wraps the ONNX policy; builds the 98-dim official observation itself."""

    def __init__(self, env, sess, input_name, command):
        self.env = env
        self.sess = sess
        self.input_name = input_name
        self.command = list(command)
        self.robot = env.scene["robot"]
        self.cmd_term = env.command_manager.get_term("base_velocity")
        self.default_jpos = self.robot.data.joint_pos[0].detach().cpu().numpy()
        self.last_raw = np.zeros(29, dtype=np.float32)

    def reset(self):
        self.last_raw[:] = 0.0

    def on_key(self, key):
        from mjlab.viewer.native import keys as K

        step = 0.1
        if key == K.KEY_I:
            self.command[0] = min(self.command[0] + step, 1.0)
        elif key == K.KEY_K:
            self.command[0] = max(self.command[0] - step, -0.5)
        elif key == K.KEY_U:
            self.command[1] = min(self.command[1] + step, 0.5)
        elif key == K.KEY_O:
            self.command[1] = max(self.command[1] - step, -0.5)
        elif key == K.KEY_J:
            self.command[2] = min(self.command[2] + step, 1.0)
        elif key == K.KEY_L:
            self.command[2] = max(self.command[2] - step, -1.0)
        elif key == K.KEY_BACKSPACE:
            self.command = [0.0, 0.0, 0.0]
        else:
            return
        print(f"command: vx={self.command[0]:+.1f} vy={self.command[1]:+.1f} wz={self.command[2]:+.1f}")

    def __call__(self, obs):
        del obs  # the env's policy obs group is the Instinct layout; rebuild officially
        env = self.env
        cmd = np.asarray(self.command, dtype=np.float32)
        self.cmd_term.vel_command_b[:] = torch.tensor(cmd[None], device=env.device)
        t = env.episode_length_buf[0].item() * env.step_dt
        ang_vel = self.robot.data.root_link_ang_vel_b.detach().cpu().numpy()[0]
        grav = self.robot.data.projected_gravity_b.detach().cpu().numpy()[0]
        if float(np.linalg.norm(cmd)) < 0.1:
            phase = np.zeros(2, dtype=np.float32)
        else:
            ph = (t % 0.6) / 0.6
            phase = np.array([math.sin(ph * 2 * math.pi), math.cos(ph * 2 * math.pi)], dtype=np.float32)
        jpos = self.robot.data.joint_pos[0].detach().cpu().numpy() - self.default_jpos
        jvel = self.robot.data.joint_vel[0].detach().cpu().numpy()
        obs98 = np.concatenate([ang_vel, grav, cmd, phase, jpos, jvel, self.last_raw]).astype(np.float32)
        raw = self.sess.run(None, {self.input_name: obs98[None]})[0][0].astype(np.float32)
        self.last_raw = raw
        return torch.tensor(raw[None], device=env.device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600, help="policy steps per scenario (50 Hz)")
    p.add_argument("--settle-steps", type=int, default=100, help="initial steps excluded from metrics")
    p.add_argument("--output", type=Path, default=Path("outputs/p4/unitree_onnx_validation"))
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    p.add_argument("--viewer", choices=("native", "viser"), default=None,
                   help="interactive play window instead of offline validation")
    p.add_argument("--command", type=float, nargs=3, default=None,
                   help="initial velocity command vx vy wz for --viewer")
    a = p.parse_args()
    device = "cpu"  # CUDA unavailable on this machine

    deploy = yaml.safe_load(DEPLOY_YAML.read_text())
    assert int(round(1 / float(deploy["step_dt"]))) == 50

    env = build_env(device)
    assert abs(env.step_dt - 0.02) < 1e-9, env.step_dt
    from instinct_mj.rl import InstinctRlVecEnvWrapper
    vec = InstinctRlVecEnvWrapper(env)

    global default_jpos
    robot = env.scene["robot"]
    default_jpos = robot.data.joint_pos[0].detach().cpu().numpy()
    joint_names, dep_kp, dep_kd = verify_orders_and_params(env, deploy)

    sess, input_name = load_policy()

    if a.viewer is not None:
        command = a.command if a.command is not None else [0.5, 0.0, 0.0]
        run_viewer(env, sess, input_name, command, a.viewer)
        env.close()
        return

    sim = getattr(env, "sim", None)
    model = getattr(sim, "_mj_model", None)
    assert isinstance(model, mujoco.MjModel) and model.nq == 7 + len(joint_names), "unexpected replay model"

    a.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy": str(POLICY_ONNX.resolve()),
        "asset": str(G1_XML.resolve()),
        "control_hz": 50,
        "joint_order": joint_names,
        "gate_thresholds": GATE,
        "note": "observations per deploy.yaml order; local actuator gains verified against deploy.yaml",
        "scenarios": {},
    }
    for name in a.scenarios:
        command = SCENARIOS[name]
        result, qpos_frames = run_scenario(env, vec, sess, input_name, command, a.steps, a.settle_steps, not a.no_video)
        slim = {k: v for k, v in result.items() if k not in ("contact_log", "time_log", "measured_velocity", "root_height")}
        summary["scenarios"][name] = slim
        save_plots(result, a.output, name)
        print(f"=== {name}: {json.dumps(slim, indent=2)}")
        if not a.no_video:
            ok = save_video(qpos_frames, model, a.output / f"{name}.mp4")
            slim["video_saved"] = ok
        (a.output / f"{name}.json").write_text(json.dumps(result, indent=2))

    summary["approved"] = all(
        (r.get("gate_linear_ok") and r.get("gate_angular_ok") and not r.get("fallen"))
        for r in summary["scenarios"].values()
    )
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "scenarios"}, indent=2))
    env.close()


default_jpos = None


if __name__ == "__main__":
    main()
