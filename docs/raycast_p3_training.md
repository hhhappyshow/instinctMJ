# RayCaster P3：训练配置、课程和中等规模测试

这一轮只做了配置核对、三阶段课程和 200 次 PPO 更新。没有启动 5000 次的正式训练。200 次更新后的策略还不能稳定行走，不能用来判断 RayCaster 是否有帮助。

## 配置核对

Baseline 和 RayCaster 共用同一套条件。差别只在 Actor 和 Critic 是否拼接 187 维高度扫描。

| 项目 | 两个任务的共同取值 |
| --- | --- |
| PPO | clip 0.2，entropy 0.008，学习率 1e-3 adaptive，gamma 0.99，lam 0.95，desired KL 0.01，5 epochs，4 minibatches，每环境 24 步 |
| 网络 | Actor 和 Critic 都是 ELU，`[256, 128, 128]`，初始动作噪声标准差 1.0 |
| 种子 | runner seed 42，地形生成 seed 0 |
| 仿真 | 物理步长 0.005 s，decimation 4，控制周期 0.02 s |
| 并行环境 | 训练默认 64，本轮实测后建议 512 |
| 速度指令 | `vx` [-0.5, 1.0]，`vy` [-0.5, 0.5]，`wz` [-1.5, 1.5]，20% 原地站立，每 10 s 重采样 |
| 动作 | 29 维关节位置目标，`beyondmimic_action_scale` |
| 关节驱动 | `beyondmimic_g1_29dof_actuator_cfgs` |
| 奖励和终止 | 与已经能在平地行走的 G1 flat 任务相同 |
| 随机化 | 同样的 reset 位姿、关节和推力事件 |
| 预算 | `max_iterations=5000`，`save_interval=500` |

奖励没有改。主要项是线速度和角速度跟踪各 1.0，腾空时间 1.0，脚滑 -0.1，躯干姿态 -1.0，原地站立 -0.8，关节限位 -1.0，垂直速度 -0.1，动作变化率 -0.05，终止 -200。终止条件是超时和 `base_contact`。一集 20 s，也就是 1000 个控制步。

mjlab 原生 G1 速度任务用更宽的网络 `[512, 256, 128]`，并训练 30000 次。这里没有改成那套。InstinctMJ 平地任务用现在这组 PPO 和网络，在 5000 次更新后已经能走。正式预算因此仍定为 5000 次。

P2 里有一个会阻碍收敛的问题：地形难度从 0.4 开始，第 0 行已经是坡和台阶，而且 `terrain_levels_vel` 没有接上。3 次更新里机器人马上摔倒，是因为开局地形太难，不是传感器失效。P3 只修了这一点。

## 三阶段课程

两个任务调用同一个 `make_rough_terrain_cfg()`。地块 8 m × 8 m，3 行 4 列。列是 `flat`、`slope`、`stairs`、`rough`。难度从 0.05 到 1.0，按行线性插值。初始 `max_init_terrain_level=0`。

| 行 | 难度 | 台阶高度 | 坡度 | 起伏噪声 |
| --- | --- | --- | --- | --- |
| 0 stage1_flat | 0.05 | 0.8 cm | 0.006 | 0.5–2 mm |
| 1 stage2_easy | 0.525 | 7.9 cm | 0.063 | 0.5–2.1 cm |
| 2 stage3_rough | 1.0 | 15 cm | 0.12 | 1–4 cm |

下面的 200 次更新是在台阶最高 10 cm 时跑的。当前代码已经把第 3 行改成 15 cm。

台阶和坡的平台宽 1.0 m，高度扫描能看到平台外沿。起伏噪声随难度从 0 放大到完整范围。

`terrain_levels_vel` 在每次 reset 时更新。走过超过 4 m 就升一行；走过的距离小于指令速度乘 10 s 就降一行。Play 和评估会清空这个课程，避免评估时的摆放被改掉。

200 次更新结束时，两套策略的平均地形等级都是 0。它们还走不到 4 m，所以还没有进入第二阶段。

## 吞吐量

机器是 RTX 4070 12 GB。显存峰值来自训练期间的 `nvidia-smi`。步数和迭代时间来自 8 次更新的日志，第一次迭代含编译，下表取后面的稳定值。

| 环境数 | 任务 | 显存峰值 | 仿真步/秒 | 每次迭代 |
| --- | --- | --- | --- | --- |
| 64 | Baseline | 2079 MiB | 约 4100 | 0.38 s |
| 256 | Baseline | 3118 MiB | 约 13000 | 0.46 s |
| 512 | Baseline | 4323 MiB | 约 23000 | 0.53 s |
| 512 | RayCaster | 4380 MiB | 约 24000 | 0.51 s |

512 个环境下，RayCaster 比 Baseline 多约 60 MiB。8 次更新里两者的采集时间都在 0.47–0.50 s，射线开销落在这次测量的波动里。显存还剩一半以上，正式训练用 512 个环境。

## 200 次更新

两次都是 512 个环境、seed 42、每 10 次迭代打印一次。损失全程有限，没有 NaN 或 Inf。

- Baseline：`logs/instinct_rl/g1_locomotion_rough/2026-10-15_17-58-22_G1RoughBaseline/`，到第 190 次共用 94.1 s。
- RayCaster：`logs/instinct_rl/g1_locomotion_rough_raycast/2026-10-15_18-00-08_G1RoughRayCaster/`，到第 190 次共用 95.8 s。

Checkpoint：各自的 `model_100.pt` 和 `model_200.pt`。完整日志在 `outputs/raycast_p3/medium_baseline_512.log` 和 `outputs/raycast_p3/medium_raycast_512.log`。

| 迭代 | Baseline 奖励 | Baseline 集长 | RayCaster 奖励 | RayCaster 集长 |
| --- | --- | --- | --- | --- |
| 0 | -4.64 | 10.1 | -4.75 | 12.4 |
| 40 | -4.21 | 5.2 | -4.18 | 5.0 |
| 100 | -4.72 | 22.7 | -4.83 | 29.7 |
| 190 | -5.24 | 46.0 | -5.32 | 51.7 |

第 30–40 次附近两套集长都掉到约 5 步，之后回升。第 190 次大约能站 0.9–1.0 s。完整一集是 1000 步。平均奖励变负，是因为活得稍长以后，逐步惩罚累加得更多；它不代表已经学会走路。

同期线速度误差从约 0.01 升到约 0.075。刚开始的误差小，是因为机器人在迈步前就摔倒。动作噪声标准差从 1.0 降到 Baseline 0.70、RayCaster 0.66。value loss 从约 7 降到约 0.02。surrogate loss 停在约 -0.02。这些说明优化器在更新，不是说步态已经形成。

第 190 次 RayCaster 的集长只比 Baseline 多大约 6 步，而且两者都停在第 0 行。这个差距不能当成感知收益。

## 正式训练

在这台 4070 上，稳定阶段每次迭代约 0.48 s。5000 次大约 40 分钟一个实验，两个实验大约 80 分钟，另加十几秒的首次编译。换 GPU 后按实测的每次迭代时间重估。

```bash
uv run instinct-train Instinct-Locomotion-Rough-G1-v0 \
  --num-envs 512 \
  --agent.max-iterations 5000 \
  --agent.save-interval 500 \
  --agent.log-interval 10

uv run instinct-train Instinct-Locomotion-Rough-RayCaster-G1-v0 \
  --num-envs 512 \
  --agent.max-iterations 5000 \
  --agent.save-interval 500 \
  --agent.log-interval 10
```

种子已经写在配置里，两次都会打印 `seed=42`。每 500 次保存一次，可用 `--agent.load-run` 和 checkpoint 接着训。确认收敛时看集长是否接近 1000、地形等级是否离开 0，而不是只看平均奖励。
