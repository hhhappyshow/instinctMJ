"""Register Instinct Mj locomotion G1 tasks."""

from instinct_mj.tasks.registry import register_instinct_task


def _locomotion_flat_env_cfg(play: bool):
    from .flat_env_cfg import instinct_g1_locomotion_flat_env_cfg

    return instinct_g1_locomotion_flat_env_cfg(play=play)


def g1_locomotion_instinct_rl_cfg():
    from .rl_cfg import g1_locomotion_instinct_rl_cfg as _cfg

    return _cfg()


def g1_locomotion_rough_instinct_rl_cfg():
    from .rl_cfg import g1_locomotion_rough_instinct_rl_cfg as _cfg

    return _cfg()


def g1_locomotion_rough_raycast_instinct_rl_cfg():
    from .rl_cfg import g1_locomotion_rough_raycast_instinct_rl_cfg as _cfg

    return _cfg()


def _locomotion_rough_env_cfg(play: bool):
    from .rough_env_cfg import instinct_g1_locomotion_rough_env_cfg

    return instinct_g1_locomotion_rough_env_cfg(play=play, include_height_scan=False)


def _locomotion_rough_raycast_env_cfg(play: bool):
    from .rough_env_cfg import instinct_g1_locomotion_rough_env_cfg

    return instinct_g1_locomotion_rough_env_cfg(play=play, include_height_scan=True)


register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-v0",
    env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=False),
    play_env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Flat-G1-Play-v0",
    env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    play_env_cfg_factory=lambda: _locomotion_flat_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Rough-G1-v0",
    env_cfg_factory=lambda: _locomotion_rough_env_cfg(play=False),
    play_env_cfg_factory=lambda: _locomotion_rough_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_rough_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Rough-G1-Play-v0",
    env_cfg_factory=lambda: _locomotion_rough_env_cfg(play=True),
    play_env_cfg_factory=lambda: _locomotion_rough_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_rough_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Rough-RayCaster-G1-v0",
    env_cfg_factory=lambda: _locomotion_rough_raycast_env_cfg(play=False),
    play_env_cfg_factory=lambda: _locomotion_rough_raycast_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_rough_raycast_instinct_rl_cfg,
)

register_instinct_task(
    task_id="Instinct-Locomotion-Rough-RayCaster-G1-Play-v0",
    env_cfg_factory=lambda: _locomotion_rough_raycast_env_cfg(play=True),
    play_env_cfg_factory=lambda: _locomotion_rough_raycast_env_cfg(play=True),
    instinct_rl_cfg_factory=g1_locomotion_rough_raycast_instinct_rl_cfg,
)
