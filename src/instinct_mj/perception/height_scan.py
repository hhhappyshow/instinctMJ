"""Ray hit processing and the actor/critic height-scan observation.

Sensor: mjlab ``RayCastSensor``. Rays start at the attachment body, point down,
and keep only yaw from that body.

Hit processing: a ray with ``distance < 0`` missed. Its height is replaced by
``max_distance``. A hit height is ``frame_z - hit_z`` (positive when the sensor
is above the surface). This matches ``mjlab.envs.mdp.observations.height_scan``.

Observation adapter: clip to ``HEIGHT_SCAN_CLIP``. No normalization and no noise
are applied here. Grid order is ``meshgrid(x, y, indexing="xy")`` then flatten,
so a reshape to ``(num_y, num_x)`` restores the scan.
"""

from __future__ import annotations

import torch
from mjlab.envs.mdp.observations import height_scan as mjlab_height_scan
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg

HEIGHT_SCAN_FRAME_BODY = "torso_link"
HEIGHT_SCAN_SIZE = (1.6, 1.0)
HEIGHT_SCAN_RESOLUTION = 0.1
HEIGHT_SCAN_ALIGNMENT = "yaw"
HEIGHT_SCAN_MAX_DISTANCE = 5.0
HEIGHT_SCAN_GEOM_GROUPS = (0,)
HEIGHT_SCAN_CLIP = (-20.0, 20.0)
# Fixed stand-in used by the perception ablation. It is not a ray hit.
HEIGHT_SCAN_MASK_VALUE = 0.0


def make_height_scanner_cfg(*, debug_vis: bool = False, show_rays: bool = False) -> RayCastSensorCfg:
    """Build the height scanner used by the new perceptive and rough tasks."""
    return RayCastSensorCfg(
        name="height_scanner",
        frame=ObjRef(type="body", name=HEIGHT_SCAN_FRAME_BODY, entity="robot"),
        pattern=GridPatternCfg(resolution=HEIGHT_SCAN_RESOLUTION, size=HEIGHT_SCAN_SIZE),
        ray_alignment=HEIGHT_SCAN_ALIGNMENT,
        max_distance=HEIGHT_SCAN_MAX_DISTANCE,
        exclude_parent_body=True,
        include_geom_groups=HEIGHT_SCAN_GEOM_GROUPS,
        debug_vis=debug_vis,
        viz=RayCastSensorCfg.VizCfg(show_rays=show_rays),
    )


def height_scan_observation(env, sensor_name: str = "height_scanner", offset: float = 0.0) -> torch.Tensor:
    """Return the clipped height scan that enters the observation manager.

    Shape is ``[num_envs, num_rays]``. Misses stay at ``max_distance`` before the
    clip, which does not change that value. Non-finite values are rejected
    instead of being replaced with a constant.
    """
    heights = mjlab_height_scan(env, sensor_name, offset=offset)
    if heights.ndim != 2:
        raise RuntimeError(f"height_scan expected [B, N], got shape {tuple(heights.shape)}")
    if not torch.isfinite(heights).all():
        raise RuntimeError("height_scan observation contains NaN or Inf")
    return heights.clamp(HEIGHT_SCAN_CLIP[0], HEIGHT_SCAN_CLIP[1])
