# RayCaster P3：Play 和评估

200 次更新的 checkpoint 还会在大约 1 秒内摔倒。下面的 Play 和短评估只用来确认流程。感知是否有效，要等 5000 次正式训练之后，用同一批种子再跑。

## Play

RayCaster 的 Play 会画出向下的射线和命中点。绿色竖线是高度扫描射线，青色点是这些射线的命中点，网格是 17×11。当前速度指令由命令可视化给出。摔倒原因在评估 JSON 里统计，不画进 MuJoCo 窗口。

```bash
uv run instinct-play Instinct-Locomotion-Rough-G1-Play-v0 \
  --agent trained --viewer native --num-envs 1 \
  --checkpoint-file logs/instinct_rl/g1_locomotion_rough/2026-10-15_17-58-22_G1RoughBaseline/model_200.pt

uv run instinct-play Instinct-Locomotion-Rough-RayCaster-G1-Play-v0 \
  --agent trained --viewer native --num-envs 1 \
  --checkpoint-file logs/instinct_rl/g1_locomotion_rough_raycast/2026-10-15_18-00-08_G1RoughRayCaster/model_200.pt
```

Play 会关掉课程和观测噪声，避免摆放被课程改掉。不要省略 `--checkpoint-file`，自动查找只认 `logs/instinct_rl/<experiment_name>/` 下最新的一次运行。

## 成功率

`scripts/eval_rough_terrain.py` 把所有并行环境放到同一列、同一难度行，并锁住同一条速度指令。成功的定义是：在 `max_steps` 个控制步内没有发生非超时终止。当前唯一的非超时终止是 `base_contact`。超时不算失败。

这个定义不是“到达目标区域”。正式对照时把 `--max-steps` 设为 1000，也就是完整的 20 s。若在 1000 步内摔倒，该集记为失败。

每种地形至少 100 集，Baseline 和 RayCaster 使用同一个 `--seed`、同一条 `--vx --vy --wz`。列是 `flat`、`slope`、`stairs`、`rough`。难度行 0、1、2 对应课程的三个阶段。第 2 行在策略还没被课程升上去之前，就是训练时没踩过的难度。生成器里没有第五种地形。

```bash
uv run python scripts/eval_rough_terrain.py \
  --variant baseline \
  --checkpoint <baseline_model.pt> \
  --terrain stairs --terrain-level 2 \
  --seed 0 --vx 0.4 --vy 0.0 --wz 0.0 \
  --episodes 100 --max-steps 1000 \
  --output outputs/raycast_p3/baseline_stairs_l2.json
```

RayCaster 把 `--variant` 换成 `raycaster`。输出包含成功率、跌倒率、平均集长、线速度误差、角速度误差、射线有效命中比例和策略推理时间。

## 消融

只作用于 187 维高度扫描，其余观测不变。

- `normal`：原始扫描。
- `masked`：整段填 0。0 是占位值，不是射线命中。
- `noisy`：加上标准差 `--noise-std` 的高斯噪声，默认 0.05 m，再裁剪到 [-20, 20]。
- `delayed`：使用 `--delay-steps` 个控制步之前的扫描，默认 5 步，也就是 0.1 s。历史不够时用最早的一帧。

```bash
uv run python scripts/eval_rough_terrain.py \
  --variant raycaster \
  --checkpoint <raycaster_model.pt> \
  --terrain rough --terrain-level 2 \
  --seed 0 --episodes 100 --max-steps 1000 \
  --ablation delayed --delay-steps 5
```

本轮只做了脚本冒烟：RayCaster 的 `model_200.pt`，平地第 0 行，`delayed`，2 集，30 步。两集都撑满了这 30 步，有效射线比例 1，单次推理约 0.14 ms。30 步只有 0.6 s，短于训练日志里约 1 s 的平均存活时间，所以成功率 1.0 只说明脚本能跑，不能说明策略会走路，也不能说明延迟没有影响。四种模式在五种地形上的 100 集对照，等正式 checkpoint 出来再跑。
