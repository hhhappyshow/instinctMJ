# RayCaster 观测接口

UniLab 或 Isaac Lab 只需要复现这份观测。策略网络、奖励和地形课程不必在那边重新设计。本阶段没有改 UniLab 仓库。

实现位于 `src/instinct_mj/perception/height_scan.py`。`relative_heights()` 是和 mjlab `height_scan` 对齐的纯函数，可用合成命中点核对两个框架。

## 射线

- 挂载体：`robot/torso_link`。
- 排列：`GridPattern`，尺寸 (1.6, 1.0) m，间距 0.1 m。
- x 从 -0.8 到 0.8，共 17 个；y 从 -0.5 到 0.5，共 11 个。一共 187 条。
- `meshgrid(x, y, indexing="xy")` 后按 C 顺序展平。连续元素先沿 x 增加，y 较慢。`reshape(11, 17)` 能还原网格。
- 局部方向 `(0, 0, -1)`，再归一化。
- `ray_alignment="yaw"`：射线原点跟随躯干位置，朝向只保留偏航。射线在世界系里竖直向下。
- `max_distance=5.0`。`exclude_parent_body=True`。只检测 geom group 0。机器人可视 geom 在 group 2 和 3，不会被打中。地形 geom 默认是 group 0。

## 高度

高度等于传感器所在体的世界 z 减去命中点的世界 z。传感器在表面上方时高度为正。平地站立、躯干 z 为 0.82 m 时，高度约为 0.82，不是 0。

`distance < 0` 视为未命中，包括超过 `max_distance` 后被传感器改成 -1 的射线。未命中的高度替换成 5.0。然后裁剪到 [-20, 20]。

适配器不做归一化，也不加噪声。训练时其他本体观测有均匀噪声，高度项没有。Play 会关掉整组 policy 噪声。

每个控制步更新一次，周期 0.02 s。

## Actor 拼接

VecEnv 按观测项顺序展平。

RayCaster Actor，283 维：

1. `base_ang_vel`，3
2. `projected_gravity`，3
3. `velocity_commands`，3，顺序是 `vx, vy, wz`
4. `joint_pos`，29，相对默认姿态，MJCF 顺序
5. `joint_vel`，29
6. `actions`，29，上一时刻动作
7. `height_scan`，187

Baseline Actor 是前 96 维，没有第 7 段。

Critic 在最前面多一个 3 维 `base_lin_vel`。Baseline Critic 99 维，RayCaster Critic 286 维。

动作是 29 维关节位置目标，乘 `beyondmimic_action_scale`。关节顺序与 G1 29-DoF MJCF 一致：腰、左腿、右腿、左臂、右臂。不能直接加载 Isaac 肩关节在前的 checkpoint。

## 消融占位

`HEIGHT_SCAN_MASK_VALUE = 0.0`。屏蔽时填这个常数，不要填成“地面高度 0”的物理含义。加噪和延迟都在评估脚本里只改这 187 维，见 `docs/raycast_p3_evaluation.md`。

## 合成数据核对

已用合成命中点检查 `relative_heights()`：帧高 0.82、命中 z 为 0、距离为正时，高度是 0.82；距离为负时，高度是 5.0。UniLab 侧用同一组 `frame_z`、`hit_z` 和 `distance` 调用等价函数，结果应一致，再按上面的 17×11 顺序展平。
