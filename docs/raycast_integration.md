# RayCaster 接入

本地依赖是 `mjlab==1.6.0`（`../mjlab` 可编辑安装）。射线实现以这份源码为准。

## 原有感知任务

`Instinct-Perceptive-Shadowing-G1` 已经创建了 `height_scanner`，但 Actor 不读它。

- 射线从 `robot/torso_link` 发出，方向是机体系 `(0, 0, -1)`。
- `ray_alignment="yaw"`：原点跟随机体位置，方向只保留 yaw，忽略 pitch 和 roll。
- 网格 `size=(1.6, 1.0)`、`resolution=0.1`，x 从 -0.8 到 0.8、y 从 -0.5 到 0.5，共 17×11=187 条。排列是 `meshgrid(x, y, indexing="xy")` 后按行展开。
- 只和 geom group 0 碰撞。G1 视觉网格在 group 2/3，不会被打到。父体 `torso_link` 被排除。
- 未命中的定义是 `distance < 0`。高度记为 `max_distance`（5 m），不是另设距离阈值。
- 观测是 `frame_z - hit_z`。正值表示传感器在命中点上方。Critic 再裁剪到 [-20, 20]。没有归一化，也没有噪声。
- Actor 使用 `depth_image`（针孔射线、带噪声和缩放的深度图）。`height_scan` 只在 Critic。深度图是前向图像，高度扫描是身体周围的垂直地面高度。

## 这次新增的数据流

```text
RayCastSensor
    -> distance < 0 记为 max_distance，否则 frame_z - hit_z
    -> clip [-20, 20]
    -> Observation Manager 的 height_scan
    -> 与本体、指令或参考动作拼接
    -> Actor MLP
```

接口集中在 `src/instinct_mj/perception/height_scan.py`。

| 项目 | 取值 |
| --- | --- |
| 坐标系 | 传感器位置的世界 z 减去命中点世界 z |
| 网格 | 17×11，x 内层变化 |
| 高度符号 | 向上为正 |
| 未命中 | `distance < 0` 时使用 5 m |
| 归一化 | 无 |
| 更新 | 每个环境步，控制周期 0.02 s |
| Rough Actor 维度 | 96 + 187 = 283 |
| 感知 Shadowing Actor 维度 | 1601，其中高度扫描 187 |

## 修改的文件

- `src/instinct_mj/perception/height_scan.py`：传感器配置和观测适配。
- `src/instinct_mj/tasks/shadowing/perceptive/config/g1/perceptive_raycast_cfg.py`：Actor 去掉 `depth_image`，加入 `height_scan`。原任务配置未改。
- `src/instinct_mj/tasks/locomotion/config/g1/rough_env_cfg.py`：平地、缓坡、低台阶、轻微起伏四列地形，以及 Baseline / RayCaster 两种观测。
- `scripts/debug_raycast.py`：平地、台阶、坡道探针。
- `scripts/check_raycast_actor.py`：打印 Actor 观测并检查 MLP 输入。
- `scripts/eval_rough_terrain.py`：固定指令评估和高度观测消融。

## 运行

```bash
uv run python scripts/debug_raycast.py
uv run python scripts/check_raycast_actor.py
uv run instinct-train Instinct-Perceptive-Shadowing-G1-RayCaster-v0 --num-envs 4
uv run instinct-train Instinct-Locomotion-Rough-G1-v0 --num-envs 64
uv run instinct-train Instinct-Locomotion-Rough-RayCaster-G1-v0 --num-envs 64
```

## UniLab 迁移

仿真 RayCaster 不能直接部署到真机。迁移时保留上面的网格、符号、未命中值和 187 维顺序，在 UniLab 里重新实现射线碰撞，再把真实深度或 LiDAR 变成同一格式。策略只消费这个张量。

## 已知问题

- 原版 Perceptive Shadowing 的 Actor 仍然只用深度图。
- 高度扫描没有做 mjlab 速度任务里的 `1 / max_distance` 缩放。
- 感知 RayCaster 任务仍会创建深度相机，只是 Actor 不再读取它。
- 短程 checkpoint 不能说明机器人已经会越障。
