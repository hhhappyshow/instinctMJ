"""Forward G1 pinhole rays: torso +X forward, +Y left, +Z up.

Rows run top to bottom, columns left to right, then row-major flatten.
16x32, vertical FOV 45 degrees, horizontal FOV 79.28 degrees, pitched down
15 degrees to see 0.3 m boxes from the torso at z=0.82 m. Range is radial,
not camera depth. World hit positions preserve mjlab's origin-on-miss value;
the valid mask is authoritative. Invalid/nonfinite distances encode 3 m / 1.
All robot geoms must be excluded with groups: environment=0, robot=1 or 2.
"""
from dataclasses import dataclass
import math
import torch
from mjlab.sensor import PinholeCameraPatternCfg, RayCastSensorCfg, ObjRef

MAX_DISTANCE = 3.0
HEIGHT, WIDTH = 16, 32


@dataclass
class ForwardPattern(PinholeCameraPatternCfg):
    width: int = WIDTH
    height: int = HEIGHT
    fovy: float = 45.0

    def generate_rays(self, mj_model, device):
        origins, camera = super().generate_rays(mj_model, device)
        direction = torch.stack((-camera[:, 2], -camera[:, 0], -camera[:, 1]), -1)
        pitch = math.radians(15)
        rotation = direction.new_tensor([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        # API rotates directions but not pattern offsets: use physical torso origin.
        return origins, direction @ rotation.T


def make_forward_scanner_cfg(debug_vis=False):
    return RayCastSensorCfg(name='forward_scanner', frame=ObjRef(type='body', name='torso_link', entity='robot'),
                            pattern=ForwardPattern(), ray_alignment='base', max_distance=MAX_DISTANCE,
                            include_geom_groups=(0,), exclude_parent_body=True, debug_vis=debug_vis)


def forward_scan(data):
    d = data.distances
    valid = torch.isfinite(d) & (d >= 0) & (d <= MAX_DISTANCE)
    distances = torch.where(valid, d, torch.full_like(d, MAX_DISTANCE)).clamp(0, MAX_DISTANCE)
    return dict(ray_distances=distances, ray_hit_positions=data.hit_pos_w,
                ray_valid_mask=valid, ray_observation=distances / MAX_DISTANCE)


def ray_observation(env):
    return forward_scan(env.scene['forward_scanner'].data)['ray_observation']
