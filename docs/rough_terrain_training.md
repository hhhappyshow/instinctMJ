# 复杂地形速度跟踪

mjlab 1.6.0 已有 `unitree_g1_rough_env_cfg`，它的扫描挂在 pelvis，并用 mjlab 自己的训练入口。InstinctMJ 的 G1 根是 `torso_link`，训练入口是 `instinct-train`。因此在 InstinctMJ 里新增了独立任务，复用 mjlab 的地形生成和现有平地行走的奖励、指令、终止。

## 任务

- `Instinct-Locomotion-Rough-G1-v0`：Baseline。Actor 是角速度、重力投影、关节位置、关节速度、上一动作和三维速度指令，共 96 维。
- `Instinct-Locomotion-Rough-RayCaster-G1-v0`：同一场景上追加 187 维高度扫描。Actor 输入 283 维，Critic 输入 286 维。
- 对应 Play 任务把环境数设为 1，并关掉推扰。

两套任务共用四列地形：平地、缓坡、低台阶、轻微起伏。课程有 2 行，难度从 0.4 到 1.0，初始只放在第 0 行。速度指令是 `vx`、`vy`、`wz`。输出仍是 29 维关节位置目标。

## 奖励

沿用平地行走，没有新加奖励项。

- `track_lin_vel_xy_exp`、`track_ang_vel_z_exp`：跟踪速度指令。地形感知应帮助机器人在起伏上仍跟上指令。
- `flat_orientation_l2`、`lin_vel_z_l2`、`stand_still`：保持躯干稳定。
- `action_rate_l2`、`dof_acc_l2`：动作平滑。
- `feet_slide` 和 `base_contact` 终止：惩罚不合理接触和跌倒。
- `dof_pos_limits`、`dof_torques_l2`、关节偏离项：限制关节和力矩。

这些奖励不直接读取射线。射线只通过 Actor 改变动作，再影响速度跟踪和跌倒。

## 已运行的短程测试

各训练 8 个环境、3 次 PPO 更新，`save_interval=1`。损失都是有限值。机器人会跌倒，这不作为感知失败。

- Baseline：`logs/instinct_rl/g1_locomotion_rough/2026-10-13_16-28-03_G1RoughBaseline/model_3.pt`。第 0 次 `surrogate_loss=-0.1305`，`value_loss=6.3841`。Actor 输入 96。
- RayCaster：`logs/instinct_rl/g1_locomotion_rough_raycast/2026-10-13_16-32-31_G1RoughRayCaster/model_3.pt`。第 0 次 `surrogate_loss=-0.0657`，`value_loss=5.2499`。Actor 输入 283，其中包含 `height_scan (187,)`。

两个 checkpoint 都用 `--viewer none --max-steps 20` 重新加载并完成 Play。

## 评估

```bash
uv run python scripts/eval_rough_terrain.py \
  --variant raycaster \
  --checkpoint logs/instinct_rl/g1_locomotion_rough_raycast/2026-10-13_16-32-31_G1RoughRayCaster/model_3.pt \
  --terrain stairs --episodes 2 --max-steps 40 --ablation normal
```

`--ablation masked` 把高度扫描替换成 0，其他输入和权重不变。Baseline 没有这段观测，不能做 masked。

这次短程权重的结果只说明脚本能跑。2 个 episode、最多 40 步：Baseline 在平地的存活率 0.5，RayCaster 在低台阶上 normal 和 masked 都撑满 40 步，有效射线比例都是 1。推理约 0.14 ms。不能据此说感知优于 Baseline。

## 正式训练还缺什么

- 环境数要从 8 提到几十到几千，迭代要从 3 次提到和现有平地任务同级的数千次。
- 需要固定种子、多地形、多 episode 的完整评估，再比较 normal 和 masked。
- 真机部署需要用深度相机或 LiDAR 生成同样的 187 维高度，而不是直接使用仿真射线。
