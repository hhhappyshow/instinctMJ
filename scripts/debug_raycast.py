"""Measure the G1 height scanner on a flat slab, a step, and a ramp.

The robot is held at the popsicle pose. Physics is not stepped, so a fallen
zero-action pose cannot be mistaken for a sensor failure. Terrain geoms are
group 0. The scanner includes only that group, so it does not see the default
infinite plane or the robot visual meshes.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg

from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH, _POPSICLE_INIT_STATE
from instinct_mj.perception.height_scan import (
    HEIGHT_SCAN_ALIGNMENT,
    HEIGHT_SCAN_FRAME_BODY,
    HEIGHT_SCAN_GEOM_GROUPS,
    HEIGHT_SCAN_MAX_DISTANCE,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    height_scan_observation,
    make_height_scanner_cfg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs" / "raycast_debug"
STEP_TOP_Z = 0.10
STEP_X_MIN = 0.25
STEP_X_MAX = 1.25
SLOPE_RISE = 0.10
SLOPE_RUN = 1.20
Z_TOL = 0.03


class _SceneEnv:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene


def _box(spec: mujoco.MjSpec, name: str, half: tuple[float, float, float], pos: tuple[float, float, float]) -> None:
    spec.worldbody.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=half,
        pos=pos,
        group=0,
        rgba=(0.4, 0.4, 0.4, 1.0),
    )


def make_terrain_spec(kind: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    _box(spec, "terrain_ground", (3.0, 3.0, 0.1), (0.0, 0.0, -0.1))
    if kind == "step":
        _box(spec, "terrain_step", (0.5, 1.0, STEP_TOP_Z / 2.0), (0.75, 0.0, STEP_TOP_Z / 2.0))
    elif kind == "slope":
        # Center each slab on a scanner x sample so the ray hits the top, not a seam.
        for index, x in enumerate((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
            top_z = SLOPE_RISE * (x / SLOPE_RUN)
            _box(spec, f"terrain_slope_{index}", (0.04, 1.2, top_z / 2.0), (x, 0.0, top_z / 2.0))
    elif kind != "flat":
        raise ValueError(kind)
    return spec


def _stats(tensor: torch.Tensor) -> dict:
    values = tensor.detach().float().cpu()
    finite = values[torch.isfinite(values)]
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": None if finite.numel() == 0 else float(finite.min()),
        "max": None if finite.numel() == 0 else float(finite.max()),
        "mean": None if finite.numel() == 0 else float(finite.mean()),
        "std": None if finite.numel() == 0 else float(finite.std(unbiased=False)),
        "nan": int(torch.isnan(values).sum()),
        "inf": int(torch.isinf(values).sum()),
    }


def _measure(kind: str, device: str) -> dict:
    robot = EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(G1_MJCF_PATH),
        init_state=copy.deepcopy(_POPSICLE_INIT_STATE),
    )
    terrain = EntityCfg(spec_fn=lambda: make_terrain_spec(kind))
    scene_cfg = SceneCfg(
        num_envs=1,
        env_spacing=0.0,
        entities={"robot": robot, "terrain": terrain},
        sensors=(make_height_scanner_cfg(),),
    )
    scene = Scene(scene_cfg, device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=SimulationCfg(njmax=128, nconmax=20), model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    if scene.sensor_context is None:
        raise RuntimeError("RayCastSensor was not attached to a SensorContext")
    sim.set_sensor_context(scene.sensor_context)
    sim.reset()
    key_qpos = torch.tensor(sim.mj_model.key("init_state").qpos, device=device, dtype=sim.data.qpos.dtype)
    sim.data.qpos[:] = key_qpos
    sim.forward()

    sensor = scene["height_scanner"]
    torso_id = sensor._frame_infos[0][2]
    torso_z = float(sim.data.xpos[0, torso_id, 2].item())
    if torso_z < 0.5:
        raise RuntimeError(f"{kind}: torso_z={torso_z:.3f}; the held pose is not the standing pose")

    sim.sense()
    data = sensor.data
    origins = sensor._cached_world_origins[0].detach()
    directions = sensor._cached_world_rays[0].detach()
    distances = data.distances[0].detach()
    hit_pos = data.hit_pos_w[0].detach()
    valid = distances >= 0
    observation = height_scan_observation(_SceneEnv(scene), "height_scanner")[0].detach()

    geom_ids = wp.to_torch(sensor._ray_geomid)[0].detach().cpu()
    geom_names = []
    for geom_id in geom_ids.tolist():
        if int(geom_id) < 0:
            geom_names.append("miss")
        else:
            geom_names.append(sim.mj_model.geom(int(geom_id)).name)
    unique_geoms = sorted(set(geom_names))
    if any(name == "miss" or "terrain" not in name for name in unique_geoms):
        unexpected = [name for name in unique_geoms if "terrain" not in name]
    else:
        unexpected = []

    local = sensor._local_offsets.detach().cpu()
    xs = torch.unique(local[:, 0], sorted=True)
    ys = torch.unique(local[:, 1], sorted=True)
    if observation.numel() != xs.numel() * ys.numel():
        raise RuntimeError(f"grid size {xs.numel()}x{ys.numel()} does not match {observation.numel()} rays")
    heightmap = observation.cpu().view(ys.numel(), xs.numel())

    origin_x = origins[:, 0].cpu()
    hit_z = hit_pos[:, 2].cpu()
    result = {
        "kind": kind,
        "torso_z": torso_z,
        "num_rays": int(sensor.num_rays),
        "grid": [int(xs.numel()), int(ys.numel())],
        "x_range": [float(xs[0]), float(xs[-1])],
        "y_range": [float(ys[0]), float(ys[-1])],
        "valid_ratio": float(valid.float().mean()),
        "unique_geoms": unique_geoms,
        "unexpected_geoms": unexpected,
        "origins": _stats(origins),
        "directions": _stats(directions),
        "distances": _stats(distances),
        "hit_pos": _stats(hit_pos),
        "hit_z": _stats(hit_z),
        "observation": _stats(observation),
        "heightmap": heightmap.numpy(),
        "origin_x": origin_x.numpy(),
        "origin_z": origins[:, 2].cpu().numpy(),
        "hit_z_values": hit_z.numpy(),
        "valid": valid.cpu().numpy(),
        "passed": False,
        "checks": {},
    }
    _score(result)
    return result


def _score(result: dict) -> None:
    valid = result["valid"]
    hit_z = result["hit_z_values"]
    origin_x = result["origin_x"]
    checks = {}
    checks["all_rays_hit_terrain"] = bool(result["valid_ratio"] == 1.0 and not result["unexpected_geoms"])
    if result["kind"] == "flat":
        error = np.abs(hit_z - 0.0)
        checks["hit_z_error_m"] = float(error.mean())
        checks["passed_tolerance"] = bool(error.max() < Z_TOL)
    elif result["kind"] == "step":
        ground = origin_x < 0.15
        step = (origin_x > STEP_X_MIN + 0.05) & (origin_x < STEP_X_MAX - 0.05)
        ground_error = np.abs(hit_z[ground] - 0.0)
        step_error = np.abs(hit_z[step] - STEP_TOP_Z)
        contrast = float(hit_z[step].mean() - hit_z[ground].mean())
        checks["ground_error_m"] = float(ground_error.mean())
        checks["step_error_m"] = float(step_error.mean())
        checks["contrast_m"] = contrast
        checks["passed_tolerance"] = bool(
            ground_error.max() < Z_TOL and step_error.max() < Z_TOL and contrast > 0.06
        )
    elif result["kind"] == "slope":
        ramp = (origin_x > 0.05) & (origin_x < 0.85)
        expected = SLOPE_RISE * (origin_x[ramp] / SLOPE_RUN)
        error = np.abs(hit_z[ramp] - expected)
        near = origin_x[ramp] < 0.25
        far = origin_x[ramp] > 0.65
        rising = float(hit_z[ramp][far].mean() - hit_z[ramp][near].mean())
        checks["ramp_error_m"] = float(error.max())
        checks["rising_m"] = rising
        checks["passed_tolerance"] = bool(error.max() < Z_TOL and rising > 0.04)
    result["checks"] = checks
    result["passed"] = bool(checks["all_rays_hit_terrain"] and checks["passed_tolerance"] and result["observation"]["nan"] == 0)


def _save_heightmap(result: dict, path: Path) -> None:
    image = result["heightmap"]
    figure, axis = plt.subplots(figsize=(6, 4))
    mesh = axis.imshow(image, origin="lower", cmap="viridis", aspect="auto")
    axis.set_title(f"{result['kind']} height scan (m)")
    axis.set_xlabel("x index")
    axis.set_ylabel("y index")
    figure.colorbar(mesh, ax=axis, label="frame_z - hit_z")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _save_rays(result: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    xs = result["origin_x"]
    zs = result["hit_z_values"]
    valid = result["valid"]
    axis.scatter(xs[valid], zs[valid], s=8, c="tab:green", label="hit")
    if np.any(~valid):
        axis.scatter(xs[~valid], result["origin_z"][~valid], s=8, c="tab:red", label="miss")
    axis.set_xlabel("ray origin x (m)")
    axis.set_ylabel("hit z (m)")
    axis.set_title(f"{result['kind']} ray hits")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _jsonable(result: dict) -> dict:
    skip = {"heightmap", "origin_x", "origin_z", "hit_z_values", "valid"}
    payload = {}
    for key, value in result.items():
        if key in skip:
            continue
        if isinstance(value, np.ndarray):
            continue
        payload[key] = value
    return payload


def _write_report(results: list[dict], path: Path) -> None:
    lines = [
        "# RayCaster 验证",
        "",
        "## 射线配置",
        "",
        f"- 发射体：`robot/{HEIGHT_SCAN_FRAME_BODY}`",
        f"- 对齐：`{HEIGHT_SCAN_ALIGNMENT}`，方向只保留 yaw，原点跟随机体位置",
        f"- 网格：size={HEIGHT_SCAN_SIZE}，resolution={HEIGHT_SCAN_RESOLUTION}",
        f"- 最大距离：{HEIGHT_SCAN_MAX_DISTANCE} m",
        f"- 碰撞组：{HEIGHT_SCAN_GEOM_GROUPS}",
        "- 未命中：`distance < 0`，高度记为 `max_distance`",
        "- 观测：`frame_z - hit_z`，再裁剪到 [-20, 20]，不做归一化、不加噪声",
        "",
        "机器人保持 popsicle 站立姿态，没有执行 `--agent zero`，也没有做物理步进。",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['kind']}",
                "",
                f"- 射线数量：{result['num_rays']}，网格 x×y = {result['grid'][0]}×{result['grid'][1]}",
                f"- 有效射线比例：{result['valid_ratio']:.3f}",
                f"- 命中 geom：{', '.join(result['unique_geoms'])}",
                f"- 观测 min/max/mean：{result['observation']['min']:.4f} / {result['observation']['max']:.4f} / {result['observation']['mean']:.4f}",
                f"- NaN/Inf：{result['observation']['nan']} / {result['observation']['inf']}",
                f"- 检查：`{json.dumps(result['checks'], ensure_ascii=False)}`",
                f"- 通过：{result['passed']}",
                "",
            ]
        )
    lines.append(f"全部通过：{all(item['passed'] for item in results)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for kind in ("flat", "step", "slope"):
        print(f"[INFO] measuring {kind} on {device}")
        result = _measure(kind, device)
        results.append(result)
        _save_heightmap(result, OUTPUT_DIR / f"{kind}_heightmap.png")
        _save_rays(result, OUTPUT_DIR / f"{kind}_rays.png")
        print(f"[INFO] {kind} passed={result['passed']} valid={result['valid_ratio']:.3f} geoms={result['unique_geoms']}")
    payload = {item["kind"]: _jsonable(item) for item in results}
    (OUTPUT_DIR / "raycast_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(results, OUTPUT_DIR / "raycast_validation.md")
    if not all(item["passed"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
