"""Independent paired navigation tasks. Stage advances are explicitly gated externally."""
import copy
import os
from functools import partial
import mujoco
from mjlab.entity import EntityCfg
import mjlab.envs.mdp as base_mdp
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactSensorCfg, ContactMatch
from instinct_mj.assets.unitree_g1 import get_g1_spec
from instinct_mj.control.unitree_onnx_locomotion import DEFAULT_ONNX, official_g1_robot_cfg
from instinct_mj.perception.forward_scan import make_forward_scanner_cfg, ray_observation
from instinct_mj.tasks.locomotion.config.g1.flat_env_cfg import instinct_g1_locomotion_flat_env_cfg
from instinct_mj.tasks.locomotion.config.g1.rl_cfg import G1FlatPPORunnerCfg
from .control import NavigationActionCfg
from . import mdp


def robot_spec():
    spec = get_g1_spec()
    for geom in spec.geoms:
        geom.group = 1  # Sensor exclusion only; does not change collision masks.
    return spec


def box_spec(index):
    spec = mujoco.MjSpec()
    dimensions = ((.3, .3, .3), (.4, .45, .45), (.5, .6, .6))[index]
    body = spec.worldbody.add_body(name='box', mocap=True)
    body.add_geom(name='collision', type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=tuple(v/2 for v in dimensions), group=0, contype=1, conaffinity=1,
                  rgba=(.65, .25, .12, 1))
    return spec


def env_config(play=False, rays=True, official=False):
    cfg = instinct_g1_locomotion_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 64
    cfg.scene.env_spacing = 12.
    cfg.sim.njmax = 600
    cfg.sim.nconmax = 100
    cfg.decimation = 20  # 10 Hz residual, five 50 Hz locomotion updates.
    cfg.episode_length_s = 20.
    stage = int(os.environ.get('INSTINCT_NAVIGATION_STAGE', '0'))
    if stage not in range(4):
        raise ValueError('Navigation stage must be 0, 1, 2 or 3')
    if official:
        cfg.scene.entities['robot'] = official_g1_robot_cfg()
    else:
        cfg.scene.entities['robot'].spec_fn = robot_spec
    for index in range(3):
        cfg.scene.entities[f'obstacle_{index}'] = EntityCfg(spec_fn=partial(box_spec, index), init_state=EntityCfg.InitialStateCfg(pos=(5.+index, 0., .5)))
    contacts = tuple(ContactSensorCfg(name=f'obstacle_contact_{i}',
        primary=ContactMatch(mode='subtree', pattern='torso_link', entity='robot'),
        secondary=ContactMatch(mode='body', pattern='box', entity=f'obstacle_{i}'),
        fields=('found', 'force'), reduce='netforce', history_length=20) for i in range(3))
    feet = ()
    if official:
        feet = (ContactSensorCfg(name='feet_contact_forces',
            primary=ContactMatch(mode='body', pattern=('left_ankle_roll_link', 'right_ankle_roll_link'), entity='robot'),
            secondary=None, fields=('found', 'force'), reduce='netforce', track_air_time=True, history_length=3),)
    cfg.scene.sensors = (*contacts, *feet, make_forward_scanner_cfg())
    cfg.actions = {'navigation': NavigationActionCfg(
        entity_name='robot', stage=stage,
        backend='unitree_onnx' if official else 'frozen',
        checkpoint=str(DEFAULT_ONNX) if official else os.environ.get('INSTINCT_LOCOMOTION_CHECKPOINT', ''))}
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.events = {k: v for k, v in cfg.events.items() if k in ('reset_base', 'reset_robot_joints')}
    cfg.events['reset_base'].params['pose_range'] = {k: (0., 0.) for k in ('x','y','yaw')}
    cfg.events['reset_base'].params['velocity_range'] = {k: (0., 0.) for k in ('x','y','z','roll','pitch','yaw')}
    cfg.events['reset_robot_joints'].params.update(position_range=(1., 1.), velocity_range=(0., 0.))
    terms = dict(goal=ObservationTermCfg(func=mdp.goal), base_ang_vel=ObservationTermCfg(func=base_mdp.base_ang_vel),
                 projected_gravity=ObservationTermCfg(func=base_mdp.projected_gravity),
                 previous_action=ObservationTermCfg(func=base_mdp.last_action))
    if rays:
        terms['forward_scan'] = ObservationTermCfg(func=ray_observation)
    cfg.observations = {name: ObservationGroupCfg(terms=copy.deepcopy(terms), concatenate_terms=False, enable_corruption=False)
                        for name in ('policy', 'critic')}
    cfg.rewards = {kind: RewardTermCfg(func=mdp.reward, weight=weight, params={'kind': kind})
                   for kind, weight in dict(progress=2., success=10., collision=-10., fall=-10., time=-.1, smoothness=-.05).items()}
    cfg.terminations = {kind: TerminationTermCfg(func=mdp.outcome, params={'kind': kind}, time_out=kind=='timeout')
                        for kind in ('success','collision','fall','timeout')}
    cfg.run_name = f'{"Unitree" if official else ""}NavigationStage{stage}'
    return cfg


def runner_config(rays=True, official=False):
    cfg = G1FlatPPORunnerCfg()
    name = 'g1_navigation_raycast' if rays else 'g1_navigation_baseline'
    cfg.experiment_name = f'{name}_unitree' if official else name
    cfg.seed = 42
    return cfg
