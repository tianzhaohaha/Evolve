# 流式稳定化：EMA 影子与任务无关经验回放

> 2026-09-06。针对 SEED-Agentstream Stage-3（异质任务流上逐批在线 OPD RL）的更新方向偏差问题。
> 两个方案各有独立开关，默认关闭；关闭时训练逻辑与原来逐 bit 一致。开启后算法不使用任何任务 / 域标签。

## 1. 问题与依据

Stage-3 每训练 step 只用 6 个任务（同一域或交错域）做若干次 Adam 更新，每批都会给策略带上该批任务的"口音"，
异质流上各批口音互相冲突。实验依据：OPD 系数减半训练明显变好。Adam 会归一化步长，所以整体缩放损失几乎不改
步长、只改各项损失的配比，减半 OPD 等于把更新方向从 OPD 侧挪向 GRPO 侧——说明 OPD 方向带的口音更重。根源是
OPD 的 teacher 就是「当前学生 + 本题 hindsight skill」：学生被上一批带偏，teacher 同样偏，偏差自我放大。

## 2. 方案 A：EMA 影子（`actor_rollout_ref.actor.ema_mode` / `ema_tau`）

**原理。** 每训练 step 后 `shadow <- tau * shadow + (1 - tau) * actor`。影子 ≈ 最近 1/(1-tau) 步的参数平均，
各批任务的专属偏向在平均中抵消。影子有两个用处：

| mode | 作用 | 要求 |
|---|---|---|
| `ref` | KL 参照从固定的 SFT 起点改为影子（绳子拴在"最近共识"上，允许长期趋势、不允许单批猛跳） | `use_kl_loss=True` |
| `teacher` | OPD 三路 teacher（episode / step / general）打分底座改为「影子 + skill prompt」，切断自蒸馏漂移复利 | 任一 OPD 系数 > 0 |
| `both` | 两者同时 | 两者 |

**实现（`verl/utils/ema.py: EmaShadow`）。**

- 影子 = 每 rank 自己的 FSDP 分片（`FlatParameter._local_shard`，无则 `.data`）的 fp32 副本，常驻 pinned CPU。
  fp32 是硬要求：lr 1e-6 下单步增量乘 (1-tau) 低于 bf16 分辨率，bf16 影子会原地不动。放 CPU 不占显存。
- 影子开口用参数交换 `swapped_in()`：逐张量把影子与分片对调、前向、finally 换回。暂存走 host 内存，GPU 上不分配
  任何临时张量。`swapped` 是脏标志：动手交换前置 True，只有完整换回成功才置 False，任何一步失败都留在 True。
  每次操作前 `_pairs()` 先整体校验（形状一致；`param.data` 与 `_local_shard` 共享存储，即模块已 reshard——FSDP1
  的根单元 forward 后不释放 unsharded buffer，若在该状态下换入，forward 会跳过 all-gather 而静默忽略写入的分片），
  校验通过后才写任何数据。`update()` 顺带返回 `||actor - shadow|| / ||shadow||`（跨 rank all-reduce）。
- worker（`fsdp_workers.py`）：`init_model` 按 `ema_mode` 建影子；`update_actor` 在 `update_policy` 后调
  `update()`（每训练 step 一次，tau 的语义即流上的 step 数）；`compute_log_prob` 读取 meta_info
  `use_ema_weights`（默认 False）决定是否在交换上下文内打分；`save/load_checkpoint` 另存 / 读取
  `ema_world_size_{ws}_rank_{r}.pt`，缺失则从当前权重重建。
- trainer（`ray_trainer.py`）：`ref` 模式下 ref 分支改调 `_compute_ema_log_prob`（输出改名 ref_log_prob，
  dp_actor 不动）；`teacher` 模式下打分闭包的 `teacher_meta_info` 加 `use_ema_weights`，三路一并生效。
- 安全防线：`update_actor` / `generate_sequences` / `start_rollout_generation_session`（vLLM 权重同步发生在此）/
  `save_checkpoint` 入口断言脏标志未立起，任何交换失败都会在下一个 RPC 立即报错，而不是带着混合权重继续。
  Ray 只保证同一 caller 对同一 actor 的任务顺序，openai 异步后端下后台 teacher 线程与主线程的 RPC 在各 rank
  上可能顺序不同，两套权重会被 all-gather 成混合模型；因此 `teacher` 模式下 teacher 打分强制走同步分支。

**指标。** `seed/ema_ref_enabled`、`seed/ema_teacher_enabled`（仅开启时上报）、`actor/ema_param_delta`。

**已知事项。** `teacher` 模式下技能池准入用的 gap（teacher − old_log_probs）混入权重滞后项，只影响候选排序。
V1 仅支持 fsdp1 且无 LoRA。

## 3. 方案 B：任务无关经验回放（`algorithm.seed.replay.*`）

**原理。** 每步除新题外，从缓冲区抽 `groups_per_step` 个旧任务组（8 条轨迹的全部 step 样本，advantage /
old / ref / teacher 张量已附带）一起算梯度，更新方向变成新旧混合。不重新与环境交互，首遍在线分不受影响。
off-policy 由 PPO 现有 ratio + dual clip 处理，回放行的 `old_log_probs` 是行为策略值，绝不重算。

**实现（`seed/replay.py`）。**

- 写入用蓄水池采样：前 `capacity` 组直接存，之后第 n 组以 capacity/n 的概率顶替随机位置。缓冲区始终 ≈ 整条流
  的均匀子集，各域按出现比例自然保留，不用标签（task-free）。只存 advantage 非全零的组。
- 每步顺序固定：`sample()` → `add_batch(live)` → 拼接，本步的组不会被本步回放。
- 拼接（`merge_for_update` + `RayPPOTrainer._mix_replay`）：键集或张量形状不一致则跳过本步并 warning（回放绝不
  拖垮主更新）；`DataProto.concat` 后复用现有 `adjust_batch` 补齐整除（它与原流程一样使用全局 numpy 随机数）；用回放专用 Generator 全局置换后复用现有 `_balance_batch`（否则回放行全落
  到最后一张卡的末尾 mini batch）；重算 `global_token_num`。合并后的 batch 只交给 `update_actor`，活 batch
  原样留给所有下游日志 / 指标。存入的组 `meta_info` 置空，不引用活 batch 的逐步 payload。
- 内存：每组约 100 MB，默认 48 组约 5 GB driver 内存；容量同时是回放样本的最大年龄。checkpoint 不持久化
  缓冲区，resume 后为空并从头累积。

**指标。** `replay/buffer_groups`、`replay/seen_groups`、`replay/sampled_groups`、`replay/sampled_samples`、
`replay/frac_of_batch`、`replay/skipped_key_mismatch`、`timing_s/replay`。回放健康度看全局 `actor/pg_clipfrac`
的前后变化。

## 4. 配置链路

`agentstream_full.env`（`AGENTSTREAM_SEED_EMA_MODE` / `_EMA_TAU` / `_REPLAY_ENABLE` / `_REPLAY_GROUPS_PER_STEP` /
`_REPLAY_CAPACITY`）→ `run_agentstream_sft_glm_self.sh`（export `SEED_*`）→ `_common/agentstream.sh`（hydra
override）→ `ppo_trainer.yaml` 默认值。`ema_mode` 在 yaml 里必须写 `"off"`（YAML 会把裸 `off` 解析成 False；
`normalize_ema_mode` 对两者都兜底）。

## 5. 验收与测试

- 关闭时不变量：不分配影子、不建缓冲区；batch 不增加键；样本顺序不变；不消耗随机数；指标集不变。
  验收：两开关全关、相同 seed 跑 2 step，`actor/pg_loss`、`kl_loss`、`opd_loss`、`grad_norm` 与旧代码逐项相同。
- 单元测试：`tests/utils/test_ema_shadow.py`（更新公式、交换还原、异常还原、嵌套拒绝、checkpoint 往返）、
  `tests/trainer/ppo/test_replay_buffer.py`（分组、零 advantage 过滤、蓄水池有界与等概率、先抽后写、键集拒绝、
  `_mix_replay` 全链路）。
- 推荐实验顺序：先 `ema_mode=teacher`（直接对应"减半 OPD 有效"的观察；若 OPD 系数回调到 0.01 不再变差即确认
  漂移主因）→ `both` → 叠加 replay（仅 sequential / interleaved）。
