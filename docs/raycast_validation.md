# RayCaster 验证结果

探针脚本是 `scripts/debug_raycast.py`。机器人固定在 popsicle 站立姿态，只做 `reset`、`forward` 和 `sense`，没有物理步进，也没有使用 `--agent zero`。

原始记录在 `outputs/raycast_debug/`：`flat_heightmap.png`、`step_heightmap.png`、`slope_heightmap.png`、对应的 `*_rays.png`、`raycast_stats.json` 和 `raycast_validation.md`。

## 配置

187 条射线，从 `torso_link` 向下，只保留 yaw。碰撞组是 0。三次测试都没有无限平面，命中的 geom 名称都以 `terrain/` 开头。

未命中规则来自 mjlab：`distance < 0` 时高度写成 `max_distance`。这次三条场景的有效射线比例都是 1，没有未命中。观测是 `frame_z - hit_z`，再裁剪到 [-20, 20]。平地上这个值等于躯干高度 0.82 m。

## 实测

| 场景 | 命中 | 误差 | 结果 |
| --- | --- | --- | --- |
| 平地，顶面 z=0 | 全部打在 `terrain/terrain_ground`，hit z = 0 | 0 m | 通过 |
| 前方 10 cm 台阶 | 地面 hit z = 0，台阶 hit z = 0.10 | 台阶误差约 2e-8 m，高度差 0.10 m | 通过 |
| 坡道，0.10 m / 1.20 m | 8 块坡面和地面都被打到 | 相对设计高度的最大误差约 6e-8 m，远端比近端高 0.05 m | 通过 |

坡道由对齐到射线 x 采样点的方块组成。第一版方块接缝正好落在采样点上，射线打到侧面或地面，测试没有通过；对齐之后通过。

## Actor

`scripts/check_raycast_actor.py` 在 `Instinct-Perceptive-Shadowing-G1-RayCaster-Play-v0` 上确认：

- Actor 有本体历史、运动参考和 `height_scan (1, 187)`，没有 `depth_image`。
- Actor 没有 `base_lin_vel`、`link_pos`、`link_rot`。
- 打包后的 Actor 是 `(1, 1601)`，Critic 是 `(1, 1667)`。
- 第一层线性层输入是 1601，和观测宽度一致。
- 17 个可训练参数都在 PPO 优化器里。
- 零动作一步之后高度扫描仍是有限值。

这个 Play 环境站在平地 smoke 地形上，所以 187 维数值相同。平地和台阶的差异由上面的探针测得，使用的是同一个 `height_scan_observation`。
