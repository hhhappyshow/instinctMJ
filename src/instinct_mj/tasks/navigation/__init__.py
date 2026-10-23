from instinct_mj.tasks.registry import register_instinct_task


def _env(play, rays, official=False):
    from .config import env_config
    return env_config(play, rays, official)


def _runner(rays, official=False):
    from .config import runner_config
    return runner_config(rays, official)


for _rays, _name in ((True, 'RayCaster'), (False, 'Baseline')):
    register_instinct_task(
        task_id=f'Instinct-G1-{_name}-Obstacle-Avoidance-v0',
        env_cfg_factory=lambda rays=_rays: _env(False, rays),
        play_env_cfg_factory=lambda rays=_rays: _env(True, rays),
        instinct_rl_cfg_factory=lambda rays=_rays: _runner(rays))
    register_instinct_task(
        task_id=f'Instinct-G1-{_name}-Obstacle-Avoidance-Unitree-v0',
        env_cfg_factory=lambda rays=_rays: _env(False, rays, True),
        play_env_cfg_factory=lambda rays=_rays: _env(True, rays, True),
        instinct_rl_cfg_factory=lambda rays=_rays: _runner(rays, True))
