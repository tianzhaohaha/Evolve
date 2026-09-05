# 双轨 OPD：SEED-Agentstream 的 spec + gen 两路技能蒸馏

> 2026-09-04。本文档记录当前训练框架的整体结构、双轨 OPD（On-Policy Distillation）扩展的设计与实现，以及全部代码改动的落点。设计参考：SGSD (arXiv:2605.28791)、UCOB (arXiv:2606.29502)、SkillRL (arXiv:2602.08234)、KDRL (arXiv:2506.02208)。

## 1. 训练框架总览

SEED-Agentstream 每个训练 step 的数据流：

```
rollout（多轮 agent-env 交互，plain prompt）
   │
   ├─► GRPO：advantage + PPO 策略损失（主 RL 信号）
   │
   └─► SEED 分析（LLM analyzer，逐轨迹）
          │  产出 episode_skill（hindsight 局部技能）
          │
          ├─► spec 通道：episode skill 注入 teacher prompt
          │      → 对学生 rollout 重打分得 teacher_log_prob → OPD_spec
          │
          └─► gen 通道（新增）：global skill 注入 "General Skill" 段
                 → 独立重打分得 gen_teacher_log_prob → OPD_gen

总损失：L = L_GRPO + c_spec · OPD_spec + c_gen · OPD_gen （+ 可选 skill_gen 损失）
```

要点：

- **teacher = 学生自己**。两路 teacher 都是当前策略在「skill 增强 prompt」下对学生已有 rollout 的重打分（一次 `compute_log_prob` forward，无额外采样），信号始终 on-policy。
- **OPD 损失形式**（`compute_opd_loss`，两通道共用）：
  ```
  gate = sigmoid(beta * (teacher_lp − student_lp))        # teacher、gate 均 detach
  loss = gate * (teacher_lp − student_lp)                  # 梯度只有 −gate·∇student_lp
  ```
  每个 token 的梯度权重 = gate ∈ (0,1)，天然有界；teacher 比学生更确信的 token 才被模仿。
- **teacher 信号构建的同步/异步取决于 analysis 后端**：`analysis_backend=openai` 时 SEED 分析 + 打分在 ThreadPoolExecutor 里与 reward / old_log_prob 计算重叠；**`policy_vllm`（当前 agentstream profile）时不提交异步任务**（分析要占用策略自己的 vLLM 引擎，不能与主线程的 rollout/log_prob RPC 并发），整条链路在 `seed_teacher` timer 内主线程同步执行（ray_trainer :3349 分支）。合并统一走 `_batch_source_idx` gather（兼容 dynamic-bsz 重排）。
- **互斥**：任一 OPD 系数 > 0 时 teacher-advantage 通道自动关闭。

## 2. 双轨设计

| | spec 通道（SEED 原有） | gen 通道（新增） |
|---|---|---|
| 技能来源 | 本轨迹的 episode_skill（局部） | global skill；**当前 = episode_skill 拷贝**，经验池上线后换成跨样本汇总 |
| prompt 注入 | "Episode-Level Skill" 段 | 独立的 "General Skill" 段（单独 prompt，不叠加） |
| 张量 | `teacher_log_prob` + `teacher_signal_mask` | `gen_teacher_log_prob` + `gen_skill_mask` |
| 系数 | `opd_loss_coef` | `opd_gen_loss_coef`（0 = 关闭，逐 bit 回退单轨） |
| gate 锐度 | `opd_gate_beta` | `opd_gen_gate_beta`（默认继承 spec） |

设计决策：

1. **两个独立损失而非合并 teacher**：gate 本身就是 per-token 路由器（每个 teacher 只在胜过学生的 token 上出力），且每路独立产出 gate 指标——这是判断 global skill 是否有信息量的唯一观测窗口。
2. **置信下限 `opd_gate_eps`**（SGSD confidence threshold）：|gap| < eps 的 token 不参与蒸馏，滤掉 gate≈0.5 的无信息噪声；两通道共用。
3. **支配路由 `opd_gen_dominance`**：`spec_first` 时 gen 的 token 权重乘 `(1 − spec_gate·spec_mask)`。实现上利用 `opd_step_mask` 的分数权重语义，损失函数无需改动。**注意语义是"重分配"而非"衰减"**：token-mean 按权重归一化（Σw·l/Σw），gen 总强度仍≈c_gen，只是移向 spec gate 低的 token。拷贝阶段两通道 gate 相同（=g），有效权重 ∝ g(1−g) 恰在无信息 token（g≈0.5、gap≈0）处最大，因此 **`spec_first` 消融必须搭配 `opd_gate_eps>0`**（eps 会先滤掉这些 gap≈0 的 token）。默认 `none`；正式双轨保持 none。
4. **约束**：`opd_gen_loss_coef > 0` 与 `skill_mode=step_only` 互斥（gen 依赖 episode skill，启动时 `_validate_config` 直接报错）。

推荐配方：双轨用 `c_spec = c_gen = 0.005`，与单轨 0.01 总蒸馏压强持平（拷贝阶段信号近似重复，不控总量等于变相翻倍蒸馏强度）。

## 3. 代码改动落点

| 文件 | 改动 |
|---|---|
| `verl/trainer/ppo/core_algos.py` | `compute_opd_loss` 加 `gate_eps` 参数（默认 0 = 原行为）；gap 计算提前到空 mask 早退之前 |
| `seed/prompting.py` | `build_augmented_observation_text` 加 `global_skill` 参数与 "General Skill" 段 |
| `verl/trainer/ppo/ray_trainer.py` | `_is_seed_opd_gen_loss_enabled` / `_is_seed_any_opd_loss_enabled`；`_prepare_seed_teacher_signals` 增加第三次打分 pass（`label="general-skill"`）；全部 5 条 zero-fill 路径与异步 merge 补 `gen_teacher_log_prob`/`gen_skill_mask`；fit 循环互斥判定改用 any；`_validate_config` 校验新配置；dump 增加 gen 字段 |
| `verl/workers/actor/dp_actor.py` | 读取 gen 系数/β/eps/dominance；第二次 `compute_opd_loss` 调用；`actor/opd_gen_*` 指标全套 |
| `verl/trainer/config/ppo_trainer.yaml` | `opd_gen_loss_coef: 0.0`、`opd_gen_gate_beta: null`、`opd_gate_eps: 0.0`、`opd_gen_dominance: none` |
| `examples/agentstream_trainer/_common/agentstream.sh` | `SEED_OPD_GEN_LOSS_COEF` 等 4 个变量 → hydra override |
| `examples/agentstream_trainer/run_agentstream_sft_glm_self.sh` | `AGENTSTREAM_SEED_OPD_GEN_*` → `SEED_OPD_GEN_*` 透传 |
| `examples/agentstream_trainer/agentstream_full.env` | 新参数及详细注释（含机制说明） |
| `tests/trainer/ppo/test_opd_loss.py` | `gate_eps` 过滤、分数权重 mask 新用例（dp_actor 级的双通道叠加与 spec_first 路由拼装无单测，已知缺口） |
| `tests/trainer/ppo/test_episode_skill_guidance.py` | "General Skill" 段注入位置用例 |

review 后追加的修正：`_prepare` 内的通道指标改名为 `seed/opd_spec_loss_enabled` / `seed/opd_gen_loss_enabled`（避免覆盖 fit 循环的合并指标 `seed/opd_loss_enabled`）；β 读取改显式 `is None` 判断；删除 step_only 死代码分支；`global_skill` 仅在 gen 启用时构造（dump 不再误导）。

## 4. 配置链路与监控

配置传递：`agentstream_full.env`（AGENTSTREAM_SEED_OPD_GEN_*）→ `run_agentstream_sft_glm_self.sh`（export SEED_OPD_GEN_*）→ `_common/agentstream.sh`（hydra `actor_rollout_ref.actor.*`）→ yaml 默认值兜底。

关键监控指标：

- `actor/opd_gen_loss` / `actor/opd_gen_gate_mean` / `actor/opd_gen_gate_active_ratio` / `actor/opd_gen_teacher_gap_mean`：gen 通道健康度。**gate_active_ratio 持续≈0 说明 global skill 无信息量**，应触发重新合成或降权。
- `seed/gen_skill_teacher/enabled`、`seed/gen_skill_teacher/gen_skills_applied`、`seed/gen_skill_teacher_log_prob_mean`：信号构建侧。
- 拷贝阶段预期：gen 与 spec 的 gate 统计高度相关；换真 global skill 后相关性下降幅度 = 信号独立性的度量。

## 5. 当前状态与后续计划

- **已完成**：双轨管道全链路（打分、异步、损失、配置、指标、测试）；`c_gen=0` 时与原单轨逐 bit 一致。
- **经验池已上线（2026-09-05，V1；同日完成一轮 review 修复）**：见 [GLOBAL_SKILL_POOL_V1.md](GLOBAL_SKILL_POOL_V1.md)。`algorithm.seed.global_pool.source=copy`（默认）保持本文的拷贝占位语义；`pool` 切换为"judge 准入 + embedding 检索"的跨任务技能来源，检索未命中时该轨迹 gen 通道跳过（不回退 copy）。修复轮要点：任务身份改由环境侧 `task_metadata()` 直供（prompt 文本在 tau2/QA 类 benchmark 上是常量，会让检索退化）、query = 任务全文+首步观测、池文件对齐 resume 语义、准入成功门控（`admit_failed`）、候选 per-task 去重按 gap 截断、EMA 每 skill 每步聚合、judge 解析抗 reasoning 前缀、load 全容错、admission 漏斗指标上报。
- **待做**：`opd_gen_start_after_steps` 独立调度、`spec_first` 消融、SGSD 极性项（失败轨迹上的反向蒸馏）、池 V2 杠杆（judge reason 文本重建索引）。

已知代价：teacher 打分从每 batch 一次 pass 变两次。**注意：当前 profile（`analysis_backend=policy_vllm`）下该成本完全在关键路径上**——policy_vllm 不走异步分支（见 §1），gen 的第三次 `compute_log_prob` 直接串行加进 `seed_teacher` timer；bootstrap 模式（非 failed_only）下 gen 行数≈全 batch，相当于每步多一遍 old_log_prob 量级的前向。"异步隐藏"仅对 `analysis_backend=openai` 成立。缓解手段：failed_only 模式下 gen 行数缩到失败轨迹子集；或将 gen 降频（每 k 步打分一次）；episode+gen 两次打分合并为一次 `compute_log_prob` 调用只省调度开销，FLOPs 不变。记录在案未修的小项（均不影响默认/推荐配置）：

- `c_spec=0` 且 `episode_skill_teacher_advantage_w>0` 时 episode 通道会白打一遍分（agentstream profile 不受影响）。
- dominance 开启时 gen 的 `active_token_ratio` 按 w>0 计、口径偏高；spec_first 路由用的 spec_gate 未应用 `gate_eps`（消融搭配 eps>0 即基本覆盖）。
- `dominance=spec_first` 且 `opd_loss_coef=0` 时静默退化为 none，validation 未拦。
- `seed/teacher_batch_size` 现含 gen 计数，双轨开启后与旧 run 面板不可比（gen 单独看 `seed/gen_skill_teacher/gen_skills_applied`）。
- teacher prompt 左截断（两路同享，SEED 预存在）：超长时先截掉的正是前部的 task/skill 段，无"截断命中"指标，只能看打分日志里 prompt_lengths.max 是否顶到上限。
- `opd_gate_eps` / `opd_gen_gate_beta` 无非负/正数校验（与 spec 的 `opd_gate_beta` 一致）。
- OPD 单调正向（gate 只缩放、不反向）与 gen/spec 共用调度、共用 eps：见上文待做（极性项、独立旋钮）。
- 运营提醒：开启 gen 改变训练目标，`RESUME_MODE=auto` 可能续到单轨 ckpt——双轨实验请换 `AGENTSTREAM_EXPERIMENT_PREFIX`。
