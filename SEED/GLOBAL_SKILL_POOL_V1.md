# Global Experience Pool V1：gen 通道的跨任务技能来源

> 2026-09-05 实现。承接 [DUAL_TRACK_OPD.md](DUAL_TRACK_OPD.md)：双轨 OPD 的 gen 通道
> 此前蒸馏 episode skill 的拷贝（占位），本文将其升级为"从跨任务经验池检索"。
> V1 为 filter-only：hindsight skill **原文**入池，不做任何改写。
> 设计参考：ReasoningBank（query-embedding 检索结构化经验）、SGSD（技能库准入筛选）、
> ExpeL（support 计数维护）、Voyager（embedding 技能检索）。

## 1. 总览与数据流

```
每个训练 step（pool 模式）：

  SEED 分析产出 episode_skill（local/hindsight）
      │
      ├─ [检索，关键路径] ─────────────────────────────────────┐
      │   每条成功分析的轨迹：query = 任务全文 + 首步观测         │
      │   （环境侧 task_metadata() 提供）→ embedding（一次批量） │
      │   → 与池内 ≤capacity 条 skill 向量算余弦 → top-1        │
      │   → 相似度 ≥ min_sim 且非同一任务（slug::task_id）⇒ 写入 │
      │     analysis["global_skill"] → gen teacher 打分与 OPD_Gen│
      │   → 否则该轨迹 gen 通道跳过（gen_skill_mask=False）      │
      │                                                        │
      └─ [准入，全异步，不阻塞训练] ──────────────────────────┘
          teacher merge 后收集本 batch 的 episode_skill 候选
          （默认仅成功轨迹，见 admit_failed）
          → spec teacher gap > 0 预过滤 + 每任务取 gap 最大 1 条
            + 按 gap 降序截断（免费，张量运算）
          → 后台线程：先 embed（本地免费，失败不花 judge 钱）
            → 批量 LLM judge（迁移性判定）→ 入池（去重/淘汰/落盘）
```

检索成本如实说明：本地 MiniLM 在 CPU 上编码一批 query 约 0.1–1s（首次的模型
加载已挪到 `_lazy_init` warmup、失败即刻报错），位于 policy_vllm 的同步路径上，
每步一次，相对 rollout/训练开销可忽略——但不是"毫秒级"。

**三条硬原则**：judge 与准入永不阻塞训练步（外部 API + 后台线程）；方法端零域标签
（检索只看任务文本，benchmark 标签仅用于评估读数）；检索选错的代价由损失层
per-token gate 兜底（这是敢用轻量检索的前提）。

**三个信号各管一段，不互相顶替**：

| 信号 | 回答的问题 | 用途 |
|---|---|---|
| embedding 余弦 | 这条 skill 与当前任务**相关**吗 | 检索（事前） |
| 损失层 per-token gate | 这次注入**真的有用**吗 | OPD_Gen 梯度（兜底） |
| gate EMA（标量/条） | 长期**值得留**吗 | 仅淘汰，**不参与检索** |

## 2. 准入流水线（Global 经验生成）

1. **成功门控**（候选收集处，`_select_seed_global_skills`）：默认只收成功轨迹的
   skill（`admit_failed=False`）。原因：失败轨迹上 spec-gap>0 预筛的语义是"skill
   让 teacher 更自信地执行**已导致失败**的动作"，恰好选中坏建议，而 judge 只判
   迁移性不判对错，挡不住。成功信号缺失（无 episode_success 的环境）时放行。
   **与 failed_only 的交互**：failed_only 模式下候选全部来自失败轨迹，默认组合
   会使准入恒为零（见 §5 必读第 5 条）。
2. **免费预过滤**（fit 循环 hook，`_update_seed_global_pool`）：轨迹的 spec teacher
   gap 均值 > 0（连本轨迹都帮不上的 skill 不送审；episode 通道未打分时该过滤自动
   放行，交给 judge 把关）+ skill 非空 + 未在池中（哈希查重）。随后
   `select_admission_candidates`：**每任务只留 gap 最大的 1 条**（GRPO 同组 8 副本
   产出近重复 skill，全部送审浪费 judge 调用且 dedup_sim 未必挡得住），再按 gap
   **降序**截断到 `max_candidates_per_step`（按 batch 顺序截断会系统性饿死排在
   batch 后部的任务）。
3. **批量 LLM Judge**（`seed/skill_judge.py`，后台线程）：**先 embed 后 judge**
   （embed 本地免费，先失败则一分 API 钱不花）。8–16 条打包一次调用 OpenRouter
   GLM（OpenAI 兼容端点，复用 `utils.openai_api` 的 client 与重试），逐条返回
   `{transferable, score∈[0,1], tag, reason}`；判定标准写在 system prompt：剥离
   领域名词后是否仍是可执行策略。`score ≥ score_threshold` 且 `transferable`
   才准入。解析用"从文本末尾起在每个 `[` 处做真 JSON 解码"（`_last_json_array`）：
   GLM 路由会把 reasoning 前缀拼在正文前、prose 里必然出现 `[1]` 式括号，贪婪
   正则会整批解析失败。
4. **fail-open + 可见**：API key 缺失（一次性告警后静默停用准入）、调用失败、
   解析失败——都只是跳过该批，训练不受影响；但每次后台任务的结果计入
   `admission_*` 计数器、下一步随 wandb 上报，"准入管线死了"与"正常冷启动"
   在曲线上可区分（见 §6）。
5. **防同组泄漏**：准入天然延迟一步（异步）+ 检索排除同一任务条目（`task_key`
   = 环境侧 `slug::task_id`），GRPO 同组 rollout 的 skill 不会被注回同任务的
   兄弟样本。注意 task_key **必须**由环境提供：从 prompt 文本推任务身份在 tau2
   （task 整段是常量）和 QA 类 benchmark（首行常量）上会退化成 benchmark 级
   常量，同任务排除变成同 benchmark 封禁、query 向量全同。

## 3. 池与检索（Key 设计）

- **存储 key** = `skill_id`（归一化文本哈希）：去重、统计归因、淘汰。
- **任务身份**：manager 在 reset 时就持有每 slot 的 slug/task_id/任务全文/首步
  观测，经 `task_metadata()` → rollout_loop non_tensor 列 → teacher 快照白名单
  到达 trainer（`_build_seed_traj_task_meta` 按轨迹取首行）。`task_key` =
  `slug::task_id`；无 task_metadata 的环境退回任务文本哈希（再退 traj_uid），
  永不崩溃。
- **检索 key** = skill 文本的 embedding；**query** = `build_retrieval_query(任务
  全文, 首步观测)`——两段各截 600 字符保证都落进 MiniLM 的 256-token 窗口。
  首步观测必须参与：tau2 的任务身份只在用户开场白里，任务文本本身是常量。
  余弦 top-1，`min_sim` 下限，排除同 `task_key`。
- **embedding 模型 = 冻结小模型**（默认 MiniLM，transformers CPU mean-pooling，
  无新依赖；可配 `http` 后端接任意 embedding 服务）。刻意**不用**被训策略自身的
  hidden state：策略权重每步在变，池里的旧向量和新查询会漂移出同一空间。
- **条目 schema**：

```json
{
  "skill_id": "…", "text": "<原文>",
  "source": {"task_key": "tau2::airline_3", "task_slug": "tau2", "task_id": "airline_3",
             "traj_uid": "…", "global_step": 120},
  "judge": {"score": 0.8, "tag": "error-recovery", "reason": "…"},
  "stats": {"times_injected": 49, "gate_ema": 0.31, "last_used_step": 300},
  "support": 3, "status": "active"
}
```

- **维护**：容量 `capacity`（默认 64），满员时按统一效用分淘汰
  `min(gate_ema 或 0.5 中性先验, last_used_step)`：验证过且 EMA<0.5 的确认
  死重先走，验证过且 EMA>0.5 的好成员比从未注入者更受保护，同分淘汰最旧
  （挂很久没被检索到 ≈ 真死重）。近重复（余弦 ≥ `dedup_sim`）合并为
  `support += 1`（ExpeL 式支持计数）。`gate_ema` 由 fit 循环 hook 用
  `σ(β·(gen_teacher_lp − old_log_probs))` 更新——**每 skill 每步聚合一次**
  （同任务 8 副本命中同一条 skill 时先取均值再更新，否则 α=0.1 下 8 次连乘
  会把 ~57% 的 EMA 权重压在单个任务上）；纯张量运算，零额外前向。
- **持久化与 resume 语义**：`<default_local_dir>/global_skill_pool.json`（原子写）
  + 同名 `.npy` 向量矩阵（记录 `embed_model`，换 embedder 不兼容即弃用重建）；
  每次准入后落盘。加载对齐 `trainer.resume_mode`：**从头跑**（disable，或 auto
  没找到 ckpt）把同名旧池改名 `.bak` 后空池启动（否则冷启动曲线被上次实验
  污染）；**真 resume** 按恢复步裁掉之后准入的条目（准入即落盘，崩溃时文件
  超前于 ckpt）。任何文件损坏（JSON 坏、npy 行数不齐）都告警后空池启动，
  **辅助文件永远无权炸训练**。
- **冷启动**：池空 / 检索不达标 ⇒ 该轨迹 gen 通道跳过（空 mask 管道天然支持），
  **不回退 copy**——一条曲线只有一种 gen 语义。

## 4. 代码落点

| 文件 | 内容 |
|---|---|
| `seed/global_pool.py`（新） | `GlobalPoolConfig` / `TextEmbedder`（local=HF mean-pooling、http）/ `GlobalSkillPool`（add/retrieve/record_usage/save/load，线程安全，load 全容错 + resume 裁剪）/ `build_retrieval_query` / `select_admission_candidates` |
| `seed/skill_judge.py`（新） | `SkillJudge` 批量判定 + `parse_judge_response`（`_last_json_array`：末尾起逐 `[` 真 JSON 解码，抗 reasoning 前缀），复用 `utils.openai_api` |
| `seed/analysis.py` | `_infer_task_description` 重构为模块级 `infer_task_description`（仅作无 task_metadata 环境的兜底） |
| `agent_system/.../agentstream/manager.py` | `task_metadata()`：每 slot 的 slug/task_id/任务全文/首步观测（reset 时快照） |
| `agent_system/multi_turn_rollout/rollout_loop.py` | reset 后探测 `task_metadata`（getattr，无此方法的 manager 零影响），长度校验后作为 non_tensor 列随每步附带 |
| `verl/trainer/ppo/ray_trainer.py` | `_lazy_init_seed_global_pool`（resume 语义 + embedder warmup fail-fast）；`_build_seed_traj_task_meta`；`_select_seed_global_skills`（env 级 task_key、query 重构、成功门控）；`_update_seed_global_pool`（usage 每 skill 聚合 + 候选 per-task 去重按 gap 截断 + admission 计数上报）；`_admit_seed_global_skills`（先 embed 后 judge，计数器）；快照白名单加 task_* 四键 |
| `verl/trainer/config/ppo_trainer.yaml` | `algorithm.seed.global_pool.*` 全部默认值 + `admit_failed`（source=copy ⇒ 存量行为逐 bit 不变） |
| `_common/agentstream.sh` / `run_agentstream_sft_glm_self.sh` / `agentstream_full.env` | `SEED_GLOBAL_POOL_{SOURCE,MIN_SIM,SCORE_THRESHOLD,ADMIT_FAILED,JUDGE_MODEL}` 链路与详注 |
| `tests/trainer/ppo/test_global_skill_pool.py` | 去重/合并、淘汰序（好成员保护/死重先走/staleness）、min_sim 与同任务排除、EMA、存取回环、load 容错（npy 截断/embedder 不符/JSON 损坏）、resume 裁剪、query 构造、候选选择、judge 解析（reasoning 前缀/围栏/乱序容错）（16 例） |
| `tests/agentstream/test_task_metadata.py`（新） | manager 元信息形状与拷贝语义 |

## 5. 配置速查

链路：`agentstream_full.env`（AGENTSTREAM_SEED_GLOBAL_POOL_*）→ run 脚本 export →
`agentstream.sh`（SEED_GLOBAL_POOL_*）→ hydra `algorithm.seed.global_pool.*` → yaml 默认。

| 参数 | 默认 | 说明 |
|---|---|---|
| `source` | copy | copy = gen 蒸 episode 拷贝（基线）；pool = 本文机制 |
| `min_sim` | 0.35 | 检索余弦下限，低于则跳过 gen |
| `score_threshold` | 0.6 | judge 迁移性准入门槛 |
| `admit_failed` | False | False = 仅成功轨迹的 skill 可参加准入（失败轨迹的 skill 可能背书失败动作） |
| `capacity` / `dedup_sim` / `ema_alpha` | 64 / 0.9 / 0.1 | 池容量 / 近重复合并 / EMA 步长 |
| `judge_model` / `judge_base_url` / `judge_api_key_env` | z-ai/glm-5.2 / openrouter / OPENROUTER_API_KEY | 缺 key ⇒ 准入静默停用 |
| `embed_backend` / `embed_model` / `embed_url` | local / MiniLM / null | local 需 HF 缓存或可下载（或给本地路径）；warmup 失败即刻终止训练（见下方必读第 4 条） |

**开 pool 前必读（操作清单）**——训练命令不变，条件逐项过：

1. **唯一必须改的一行**：`AGENTSTREAM_SEED_GLOBAL_POOL_SOURCE=pool`。注意 pool 只在
   gen 通道开启时激活（代码判定 = `opd_gen_loss_coef>0 且 source=pool`）；env 默认
   `OPD_GEN_LOSS_COEF=0.005` 已满足，若你把 gen 系数调成 0，pool 配置会被静默忽略。
2. **换 `AGENTSTREAM_EXPERIMENT_PREFIX`**：训练目标变更，防 auto-resume 续到 copy
   语义的旧 ckpt；新实验目录同时保证池文件与事件目录从零开始。
3. **`OPENROUTER_API_KEY` 在启动 shell 里**（run 脚本缺失时会打警告）：缺 key 训练
   照跑，但准入静默停用——wandb 上表现为 `judge_available=0`、池 size 恒 0。
4. **embedder 就绪（fail-fast 新行为）**：`_lazy_init` 会在首次使用时对 embedder 做
   warmup，失败**直接 RuntimeError 终止训练**（宁可 step 1 报错，不让实验静默空转）。
   MiniLM 通常已在本机 HF 缓存（`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`）；
   离线环境可设 `HF_HUB_OFFLINE=1` 强制走缓存，或把 `embed_model` 指到本地路径。
5. **failed_only 交互**：`failed_only` 模式下只分析失败轨迹 ⇒ 全部候选都来自失败
   轨迹 ⇒ 默认 `admit_failed=False` 会把候选全部滤掉，**准入恒为零、池永远不长**
   （wandb 上 `candidates_success_filtered` ≈ 候选数、`candidates_kept=0`）。要在
   failed_only 下用池，必须显式 `AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=1`。
6. **开跑后头几步健康检查**（wandb `seed/global_pool/*`）：`judge_available=1`；
   `admission_jobs_ok≥1` 且 `admission_jobs_failed=0`；`size` 开始增长；池非空后
   `retrieval_hit_ratio` 抬升。任一不满足按 §6 的指标语义定位。

## 6. 监控指标（`seed/global_pool/*`）

全部随训练 metrics 每步自动上报 wandb（pool 模式下出现）：`size` / `support_mean` /
`gate_ema_mean`（池状态）；`candidates` / `candidates_success_filtered` /
`candidates_kept`（准入漏斗前段）；`admission_jobs_ok` / `admission_jobs_failed` /
`admission_judged` / `admission_accepted` / `admission_added` / `judge_available`
（准入漏斗后段，后台任务计数、下一步上报——**`admission_jobs_failed` 持续 >0 =
准入管线故障，与"正常冷启动"（judged>0 但 accepted=0）从此可区分**）；
`retrieval_hit_ratio` / `retrieval_sim_mean` / `retrieval_failed`（检索侧——
hit_ratio 长期偏低 ⇒ 调低 `min_sim` 或池太小）；`injected_trajs` /
`unique_skills_injected` / `usage_gate_mean`（注入效果——usage_gate_mean 持续
≈0.5 以下说明池内技能对被注入任务无信息量；unique 远小于 injected 说明少数
skill 垄断命中）。

**逐条触发明细**（wandb 标量装不下的部分）：`save_events: True`（默认开）时每步写
`<default_local_dir>/global_pool_events/step_XXXXXXXX.jsonl`，逐轨迹记录
{traj_uid, task_key, query_preview, hit, skill_id, similarity, skill_preview,
pool_size}——离线分析"哪条 skill 在哪些任务上被触发"，与池文件
`global_skill_pool.json` 联表（source 里有 task_slug/task_id）即可按 benchmark
拆跨域命中分布。评估侧据此量化跨域迁移。

## 7. 验证顺序

1. **离线**：用已有 run 的 `_dump_seed_analysis` 产物单独跑 judge 与检索（人工抽查
   判定与命中合理性，迭代 judge prompt，不花训练资源）。
2. `source=copy` 回归：与改动前逐 bit 一致（新代码路径完全不触发）。
3. `source=pool` 小步数 run：看 `admission_*` 漏斗、池增长、`retrieval_hit_ratio`、
   `seed_teacher` timer（检索/embedding 在关键路径上的增量，预期亚秒级）。
4. 三臂正式对比：单轨 / 双轨-copy / 双轨-pool，核心读数 = 异域样本 gen gate 与
   val 成功率。

## 8. V1 边界与 V2 杠杆

不做：skill 改写、top-k>1、tag 匹配检索、per-sample 粒度选择、gen 独立调度。
已知偏置：任务文本 embedding 检索偏向同域表面相似（评估读数会量化它）；若跨域
引用太少，V2 第一杠杆 = 索引侧改用 judge 的域中性 `reason`/trigger 文本重建索引
（字段已入库，改一处索引文本即可）。
