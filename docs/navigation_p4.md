# G1 goal-conditioned obstacle avoidance — P4 implementation and status

## Status and scope

Independent tasks:
- `Instinct-G1-RayCaster-Obstacle-Avoidance-v0`
- `Instinct-G1-Baseline-Obstacle-Avoidance-v0`

Existing rough terrain and perceptive shadowing task files are unchanged. A pre-existing deleted README.md and untracked ws:/ were not touched.

No avoidance PPO training was started. No trained avoidance checkpoint or real-robot validation is claimed. The supplied request ended mid-P4-G; additional ablation/acceptance requirements remain unspecified.

## P4-A: audited low-level contract

Candidate: `logs/instinct_rl/g1_locomotion_flat/2026-09-30_20-19-38_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05/model_5000.pt`.
The local pretrained/ directory is empty. The [upstream README](https://github.com/project-instinct/InstinctMJ) links released shadowing and parkour weights; a compatible released flat-velocity checkpoint was not established. Do not substitute a tracking or different G1 asset checkpoint without an adapter.

Asset: local `g1_29dof_torsobase_popsicle.xml`, beyondmimic PD gains and action scales. Native observation joint order is waist pitch/roll/yaw, left leg, right leg, left arm, right arm. Full observed joint and actuator names are saved in the gate JSON; PD actuator order contains position/velocity actuators and must not be confused with the 29 action targets.

Actor: 96→256→128→128→29, ELU. Observations in order: body angular velocity(3), projected gravity(3), velocity command(3), joint position minus default(29), absolute joint velocity(29), previous low-level raw action(29). No observation history. The previous low-level action must NOT be replaced with the 3-dimensional navigation residual.

Checkpoint empirical normalization is loaded and frozen, including buffers. Missing weights, incompatible observation/history, asset identifiers, normalizer config or action scale are rejected. `model_30990.pt` is explicitly rejected. There is no default checkpoint or random-policy fallback.

Saved command ranges: vx [-0.5,1.0] m/s, vy [-0.5,0.5] m/s, wz [-1.5,1.5] rad/s. Simulation dt=.005, low-level period=.02 s (50 Hz). Upper period=.1 s (10 Hz): exactly five low-level evaluations, with each joint action held for four physics ticks. Velocity command remains held over the full upper period.

The 10-second fixed-command test survived all four scenarios with frozen weights/normalizer. Straight vx=.4 had vx MAE .0627 m/s. Combined vx=.3,wz=.3 had yaw MAE .0393 rad/s. Pure yaw=.4 had yaw MAE approximately .275–.277 rad/s across the two runs: this is a failed tracking gate, despite no fall. Gate thresholds (.15 m/s and .15 rad/s MAE) are initial engineering thresholds, not externally established safety limits. More seeds, both turning directions and visual gait inspection are still required. Do not start avoidance training on this candidate until this limitation is resolved or a different compatible model passes the gate.

## P4-B: forward perception

mjlab provides Grid, PinholeCamera and Ring patterns. ForwardPattern derives its ray directions from PinholeCameraPatternCfg: 32 columns ×16 rows, 45° vertical /79.28° horizontal FOV, 15° downward pitch, range 3 m. Torso physical origin, full body rotation; +X forward, +Y left, +Z up. Row-major flatten; rows top→bottom, columns left→right. Distances are radial Euclidean distance, not camera depth.

Environment geoms are group 0; robot geoms are group 1. This only controls sensing and does not disable collisions. Group-based exclusion removes all robot links, not just the attachment body. The infinite ground can appear in navigation scans; fixed-pose box tests intentionally isolate boxes.

`forward_scan` returns ray_distances, ray_hit_positions (world), ray_valid_mask and ray_observation. Nonfinite, negative and out-of-range distances are invalid; distance=3 and observation=1. Valid distance is clipped [0,3], divided by 3. On a miss, raw hit position is the sensor origin per mjlab; always consult the mask.

Nine fixed-pose GPU cases passed seven assertions. Results: `outputs/p4/forward_raycast/statistics.json`, `distances.npz`, `distances.png`. Includes joint robot/box translation and yaw rotation invariance. Does not establish hardware calibration, scan noise or motion latency behavior.

## P4-C/D/E: navigation interface

Actor and critic both use goal body xy+heading(3), body angular velocity(3), gravity(3), previous upper action(3); RayCaster adds 512 normalized ranges. Baseline=12, RayCaster=524. Neither receives obstacle poses. The critic also has no privileged obstacle state. Same low-level model, reward coefficients, scenes, RNG seed and PPO hyperparameters are used. Input layer parameter counts naturally differ.

Upper output is dimensionless residual, clipped [-1,1], scaled by half the saved velocity interval widths, then added to a bounded nominal command and clipped again to saved limits. Nominal command is proportional body goal xy and heading. It is a simple controller, not an obstacle planner.

Boxes have real collision geoms and mocap-fixed positions. Stages select active boxes; unused geoms are below ground. Stage 0 none; Stage 1 one medium box; Stage 2 one of three dimension presets (depth,width,height): (.3,.3,.3), (.4,.45,.45), (.5,.6,.6); Stage 3 three boxes. Stage 2 size randomization is discrete and correlated, not independent continuous size sampling. Positions and goal directions randomize symmetrically. Open flat terrain leaves both sides passable. Stage 3 goals are 4.5–5 m, beyond the box sequence.

Terminal causes are mutually exclusive, precedence collision→fall→success→timeout. Success requires planar distance <.3 m. Robot-subtree vs each box-body contact sensors exclude ground contacts; force history covers all 20 physics ticks of an upper step. Fall uses torso height<.45 m or gravity-z>-.5. Timeout=20 s and is marked truncation. Collision/fall/success are terminal states. No automatic distance-based curriculum exists.

Rewards use mjlab's dt scaling. Progress is **new closest approach** / dt, preventing repeat payment after backing away. Weighted progress=2 per meter; success +10 once; collision/fall -10 once; time -.1/s; smoothness -.05 times squared change in bounded upper residual per second. Raw and weighted values are logged under Navigation/*; episode weighted sums use the standard reward manager. These are initial scales requiring trained-policy evaluation. No gait shaping is applied upstairs.

## Reproduce verification

From repository root with the existing .venv:

```bash
export INSTINCT_LOCOMOTION_CHECKPOINT="$PWD/logs/instinct_rl/g1_locomotion_flat/2026-09-30_20-19-38_G1Flat_feetAirTime1.00_standStill0.80_actionRate0.05_jointDeviationKnee0.05/model_5000.pt"
.venv/bin/python scripts/debug_frozen_locomotion.py --checkpoint "$INSTINCT_LOCOMOTION_CHECKPOINT"
.venv/bin/python scripts/debug_forward_raycast.py
.venv/bin/python scripts/check_navigation_contracts.py
.venv/bin/python scripts/debug_navigation.py --checkpoint "$INSTINCT_LOCOMOTION_CHECKPOINT"
.venv/bin/python scripts/debug_navigation.py --checkpoint "$INSTINCT_LOCOMOTION_CHECKPOINT" --baseline
.venv/bin/python scripts/debug_navigation.py --checkpoint "$INSTINCT_LOCOMOTION_CHECKPOINT" --stage 1 --steps 150
```

The debug navigation script uses zero upper residual unless `--policy-checkpoint UPPER_MODEL.pt` is explicitly passed. It labels this distinction in JSON. Counts include completed episodes during the fixed duration and four current episodes; they are not an unbiased success-rate benchmark.

## P4-F/G: remaining acceptance

Stage 0 interface test: 4 worlds×100 upper steps; both variants produced 6 successes, zero collisions/falls. This is nominal goal control, NOT trained PPO. Stage 1 nominal test produced collisions, demonstrating that locomotion/goal control alone does not solve avoidance.

After a compatible locomotion model passes comprehensive tests, use standard entry points:

```bash
export INSTINCT_NAVIGATION_STAGE=0
.venv/bin/instinct-train Instinct-G1-RayCaster-Obstacle-Avoidance-v0
.venv/bin/instinct-train Instinct-G1-Baseline-Obstacle-Avoidance-v0
# For an actually trained upper checkpoint:
.venv/bin/instinct-play Instinct-G1-RayCaster-Obstacle-Avoidance-v0 --checkpoint-file /absolute/upper/model.pt
```

The environment variable only selects a stage; it does not certify readiness. Do not advance it until held-out full-episode success-rate evaluation passes. Automatic success-rate curriculum, paired multi-seed training, perception ablations, full-episode evaluation with confidence intervals, upper checkpoint provenance, real robot sensor/controller integration and hardware tests remain outstanding. No Stage 2/3 training acceptance is claimed. The training CLI can run these task interfaces but is not invoked by this implementation work.

Final checks: three semantic unit tests passed; syntax parsing and git diff whitespace checks passed. Stage 2 and Stage 3 each completed ten upper steps as construction/interface smoke tests only; this is not curriculum or task-performance acceptance.
