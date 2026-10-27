"""Fixed-pose actual mjlab RayCastSensor checks, collision geoms, no walking."""
import copy
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
import warp as wp
from mjlab.entity import EntityCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation, SimulationCfg
from instinct_mj.assets.unitree_g1 import G1_MJCF_PATH, _POPSICLE_INIT_STATE
from instinct_mj.perception.forward_scan import make_forward_scanner_cfg, forward_scan, HEIGHT, WIDTH


_LIVE_SCENES = []


def measure(x=1.5, y=0., height=.45, translation=(0., 0.), yaw=0., empty=False):
    def robot_spec():
        spec = mujoco.MjSpec.from_file(G1_MJCF_PATH)
        for geom in spec.geoms:
            geom.group = 1
        return spec
    if empty:
        x = 10.0
    def obstacles():
        spec = mujoco.MjSpec()
        c, s = math.cos(yaw), math.sin(yaw)
        spec.worldbody.add_geom(name='box', type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(.2, .25, height/2), pos=(translation[0]+c*x-s*y, translation[1]+s*x+c*y, height/2),
            quat=(math.cos(yaw/2), 0, 0, math.sin(yaw/2)), group=0, contype=1, conaffinity=1)
        return spec
    initial = copy.deepcopy(_POPSICLE_INIT_STATE)
    initial.pos = (translation[0], translation[1], .82)
    initial.rot = (math.cos(yaw/2), 0., 0., math.sin(yaw/2))
    scene = Scene(SceneCfg(num_envs=1, env_spacing=0., entities={
        'robot': EntityCfg(spec_fn=robot_spec, init_state=initial), 'obstacles': EntityCfg(spec_fn=obstacles)},
        sensors=(make_forward_scanner_cfg(),)), 'cuda:0')
    sim = Simulation(num_envs=1, cfg=SimulationCfg(njmax=128, nconmax=20), model=scene.compile(), device='cuda:0')
    _LIVE_SCENES.append((scene, sim))
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.set_sensor_context(scene.sensor_context)
    sim.reset()
    sim.data.qpos[:] = torch.tensor(sim.mj_model.key('init_state').qpos, device='cuda:0')
    sim.forward()
    sim.sense()
    sensor = scene['forward_scanner']
    scan = forward_scan(sensor.data)
    valid = scan['ray_valid_mask'][0].cpu().numpy()
    distance = scan['ray_distances'][0].cpu().numpy()
    ids = wp.to_torch(sensor._ray_geomid)[0].cpu().tolist()
    names = sorted({sim.mj_model.geom(i).name for i in ids if i >= 0})
    assert all(name.startswith('obstacles/') for name in names), names
    if not empty:
        geom = sim.mj_model.geom('obstacles/box')
        assert geom.contype.item() and geom.conaffinity.item()
    assert np.isfinite(distance).all()
    assert np.all(distance[~valid] == 3.)
    grid = valid.reshape(HEIGHT, WIDTH)
    return dict(count=int(valid.sum()), minimum=float(distance.min()),
                mean_column=float(np.where(grid)[1].mean()) if valid.any() else None,
                hit_geoms=names), distance.reshape(HEIGHT, WIDTH)


def main():
    out = Path('outputs/p4/forward_raycast')
    out.mkdir(parents=True, exist_ok=True)
    cases = dict(front={}, left={'y': .5}, right={'y': -.5}, near={'x': 1.}, far={'x': 2.},
                 low={'height': .3}, high={'height': .6}, empty={'empty': True},
                 moved={'translation': (1., -2.), 'yaw': .8})
    stats, arrays = {}, {}
    fig, axes = plt.subplots(3, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (name, kw) in zip(axes.flat, cases.items()):
        print('Measuring', name, flush=True)
        stats[name], arrays[name] = measure(**kw)
        im = axis.imshow(arrays[name], vmin=0, vmax=3)
        axis.set_title(f'{name}: {stats[name]["count"]} hits')
    checks = dict(front_detected=stats['front']['count'] > 0,
        left_right=stats['left']['mean_column'] < stats['right']['mean_column'],
        range=stats['near']['minimum'] < stats['far']['minimum'],
        height=stats['low']['count'] < stats['high']['count'],
        misses=stats['empty']['count'] == 0,
        follows_pose=bool(np.allclose(arrays['front'], arrays['moved'], atol=1e-4)),
        self_excluded=all(all(n.startswith('obstacles/') for n in s['hit_geoms']) for s in stats.values()))
    fig.colorbar(im, ax=axes, label='Radial distance (m); miss=3')
    fig.savefig(out / 'distances.png', dpi=150)
    np.savez(out / 'distances.npz', **arrays)
    (out / 'statistics.json').write_text(json.dumps(dict(cases=stats, checks=checks), indent=2))
    print(json.dumps(checks, indent=2))
    assert all(checks.values()), checks


if __name__ == '__main__':
    main()
