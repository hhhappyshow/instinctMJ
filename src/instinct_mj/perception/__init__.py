"""RayCaster height-scan contract shared by perceptive and locomotion tasks."""

from instinct_mj.perception.height_scan import (
    HEIGHT_SCAN_ALIGNMENT,
    HEIGHT_SCAN_CLIP,
    HEIGHT_SCAN_FRAME_BODY,
    HEIGHT_SCAN_MASK_VALUE,
    HEIGHT_SCAN_MAX_DISTANCE,
    HEIGHT_SCAN_RESOLUTION,
    HEIGHT_SCAN_SIZE,
    height_scan_observation,
    make_height_scanner_cfg,
)

__all__ = [
    "HEIGHT_SCAN_ALIGNMENT",
    "HEIGHT_SCAN_CLIP",
    "HEIGHT_SCAN_FRAME_BODY",
    "HEIGHT_SCAN_MASK_VALUE",
    "HEIGHT_SCAN_MAX_DISTANCE",
    "HEIGHT_SCAN_RESOLUTION",
    "HEIGHT_SCAN_SIZE",
    "height_scan_observation",
    "make_height_scanner_cfg",
]
