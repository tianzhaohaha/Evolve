# SEED × AgentStream 集成 — 交接说明（供后续 Agent 使用）

> **本文档的用途**：本仓库在原版 SEED 之上完成了一套"任务流模式（Isolated / Sequential /
> Interleaved）× 多 benchmark"的集成，全部代码已通过静态检查与纯逻辑自测，但**尚未在真实
> 训练环境中运行过**（编写代码的服务器没有 GPU / conda 训练环境 / benchmark 数据）。
> 你的任务是在已按两个 repo README 配好环境的机器上，按下文 Phase 顺序运行验证并修复
> 运行期问题。修复时请遵守本集成的设计原则（见 §1）。

---

## 0. 背景一句话

- **SEED**（本仓库）：RL+OPD 自进化训练框架（权重空间进化）。原生一次训练只跑一个 benchmark，
  任务在 `envs.reset()` 内随机采样。
- **AgentStream**（兄弟目录 `../AgentStream`）：测试时自进化评测框架（exgentic），把 6 个
  benchmark（bfcl / tau2 / appworld / hle / browsecompplus / swebench）组织成可配置任务流，
  三种流模式：`isolated` / `sequential` / `interleaved`，任务**选择**固定 seed=42，**排序**用
  run seed。
- **本集成目标**：让 SEED 能在 AgentStream 任务流上训练（把"测试时任务流"映射为"RL 训练数据
  流"），同时保留 SEED 原有数据处理逻辑（`stream_mode=random` 对照组），并提供 ALFWorld
  6 任务类型作 domain 的廉价对照实验（`alfworld_stream`）。

## 1. 设计原则（修复问题时必须遵守）

1. **只增不改**：所有新逻辑位于独立包内。对既有文件的修改仅 3 处（见 §2.3），且均为
   默认行为不变的增量。不要为修 bug 而改写既有 SEED 模块的逻辑。
2. **对照可比性**：`stream_mode=random` 必须保持与 SEED 原版随机采样等价；奖励尺度统一
   `10.0 * success`；GRPO/OPD 超参与现有脚本一致（launcher 里已对齐）。
3. **AgentStream 协议保真**：任务选择 seed=42、块内顺序跨模式一致、interleaved 保序交错
   ——这些不变量已有自测（`task_stream.py` 直跑），改调度逻辑后必须重跑自测。
4. **精简**：不要加防御性冗余代码；错误应显式暴露而不是静默吞掉（worker 内的容错是刻意的
   例外：为了不让单个 session 故障杀死整个向量化 rollout）。

## 2. 模块结构

### 2.1 核心包 `agent_system/environments/env_package/agentstream/`

| 文件 | 职责 | 依赖 |
|---|---|---|
| `as_config.py` | 解析/校验 `env.agentstream` 配置块 → `AgentStreamConfig`；`_select` 嵌套取值工具 | 无重依赖 |
| `task_stream.py` | **调度核心**（纯标准库，可 `python3` 直跑自测）：seed-42 选择、三模式排序（移植 AgentStream `task_ordering.py`）、train/val 切分（`build_splits`）、批粒度游标+pass 追踪（`TaskStreamScheduler`）、确定性 val 循环（`ValTaskCycler`）、`random` 模式（SEED 原逻辑） | 无 |
| `exgentic_client.py` | exgentic 桥接：`bootstrap_exgentic`（sys.path 注入 `<root>/src`）；`BenchmarkHub`（宿主侧：每 slug 一个 Benchmark+Evaluator，`list_tasks` / `session_kwargs`）；`SessionDriver`（worker 侧：session 生命周期 reset/step/score，容忍 step-after-done 与 reset 失败） | exgentic（运行时） |
| `projection.py` | 策略文本 → 动作 dict：解析 `<action>{"name":...,"arguments":{...}}</action>`（容忍 \`\`\`json 围栏），可选 `<think>` 校验；无效输出 `valids[i]=0` 走 trainer 的 invalid-action penalty | 无 |
| `prompts.py` | 各 benchmark intro + `render_prompt()` **唯一渲染入口**（RL manager 与 SFT 管线共用） | 无 |
| `envs.py` | Ray 向量化环境：`AgentStreamWorker`（1 actor = 1 `SessionDriver`）+ `AgentStreamEnvs`（reset 从调度器取任务批，group_n 复制；step 扇出；终局算奖励） | ray |
| `manager.py` | `AgentStreamEnvironmentManager(EnvironmentManagerBase)`：SEED rollout 契约实现；`_process_batch` 输出 `success_rate` + `<slug>_success_rate` / `<slug>_score`（per-benchmark val 曲线 = 遗忘/迁移测量）；训练相在线指标记录 | agent_system.base / SimpleMemory |
| `metrics.py` | `OnlineMetricsRecorder`：每完成一个 episode 追加一行 JSONL（对齐 AgentStream `record_online_metrics`；重复 pass 标 `first_pass=false` 不计入累计均值） | 无 |
| `factory.py` | `make_agentstream_envs(config)` 唯一入口：hub → 任务全集 → 切分 → 调度器/循环器 → train/val envs + managers + recorder | — |

### 2.2 对照包 `agent_system/environments/env_package/alfworld_stream/`

ALFWorld 6 任务类型作为 domain 跑三模式（零新增依赖的受控对照）：

| 文件 | 职责 |
|---|---|
| `envs.py` | `collect_game_files`（扫 train split、按任务类型分组、solvable 过滤）；`AlfWorldStreamWorker`（每 reset 用 textworld 注册**单游戏** env，先例：`scripts/sft/_common/pipeline.py::FixedGameFileBatchEnv`）；`AlfWorldStreamEnvs`（返回形状与 `AlfworldEnvs` 完全兼容） |
| `factory.py` | 复用 agentstream 包的调度器/选择/Recorder；manager 为**现有 `AlfWorldEnvironmentManager` 的子类**（只加约 30 行在线指标记录，任务类型 val 指标是父类 `_process_gamefile` 自带的）；`task_types` 子集过滤供 isolated 单类型 run |

注意：此包的 val 集是 **train split 内 held-out 游戏文件**（与 stream 集在同一 seed-42
shuffle 中不相交），**不是**标准 `eval_in_distribution`。与原版 SEED alfworld 结果对比时
要说明这一差异。

### 2.3 对既有文件的修改（仅 3 处，均默认行为不变）

1. `agent_system/environments/env_manager.py` — `make_envs` 增加两个 `elif` 分支：
   - `"alfworld_stream" in env_name`（**必须排在 `"alfworld"` 分支之前**，子串包含）
   - `"agentstream" in env_name`（在最后的 else 之前）
2. `verl/trainer/config/ppo_trainer.yaml` — 增加 `env.alfworld_stream` 与 `env.agentstream`
   两个默认配置块（各键含义见块内注释与 `as_config.py`）。
3. `examples/seed_trainer/_common/alfworld.sh` — `env.env_name` 参数化为
   `${ALFWORLD_ENV_NAME:-alfworld/AlfredTWEnv}`（不设该变量时与原版完全相同）。

> ⚠️ 历史事故：本工作区曾两次出现文件被外部删除（`env_manager.py`、`ppo_trainer.yaml`
> 被删后已从 git 恢复并重打挂载；未跟踪的新文件 `agentstream/metrics.py` 被删后已重建）。
> **接手后请先跑完整性自检**：
>
> ```bash
> # 1) 挂载都在（应有 4 处命中）
> grep -n "agentstream\|alfworld_stream" \
>     agent_system/environments/env_manager.py verl/trainer/config/ppo_trainer.yaml | head
> # 2) 新增文件齐全且可编译 + 调度器自测
> python3 -m py_compile agent_system/environments/env_package/agentstream/*.py \
>     agent_system/environments/env_package/alfworld_stream/*.py \
>     scripts/sft/agentstream/pipeline.py examples/agentstream_trainer/*.py \
>   && python3 agent_system/environments/env_package/agentstream/task_stream.py
> ```
>
> 若有缺失，优先 `git status` 区分"改动的既有文件"（可 checkout 恢复后按 §2.3 重打挂载）
> 与"未跟踪的新文件"（需从本说明的模块清单重建）。建议接手后立刻 `git add` + commit 全部
> 新文件以获得版本保护。

### 2.4 入口脚本与工具 `examples/agentstream_trainer/`

| 文件 | 用途 |
|---|---|
| `smoke_env.py` | Phase 0 冒烟：exgentic 导入 → list_tasks → 单 session 生命周期（`--step` 可选执行一个动作）。无需 GPU/Ray |
| `_common/agentstream.sh` | RL 共享 launcher（.env/conda 头部 + 与 alfworld.sh 完全一致的 SEED 算法参数 + `env.agentstream.*` 覆盖；`AS_*` 环境变量见文件头注释） |
| `run_agentstream_random.sh` | 对照组：SEED 原始随机采样 + split 协议 |
| `run_agentstream_isolated.sh` | 按 benchmark 循环，每个独立一次训练（对齐 AgentStream isolated 语义） |
| `run_agentstream_sequential.sh` / `run_agentstream_interleaved.sh` | 单次训练跑整条流 |
| `run_alfworld_stream.sh` | alfworld_stream 全模式入口（isolated 自动按任务类型循环）；包装 `seed_trainer/_common/alfworld.sh` |
| `analyze_results.py` | 跨 run 对比在线 JSONL：GRPO 组内**先按任务平均**、仅 `first_pass` 计入、`--csv` 长格式导出。agentstream 与 alfworld_stream 共用 schema |

### 2.5 SFT Stage-1 `scripts/sft/agentstream/`

| 文件 | 用途 |
|---|---|
| `pipeline.py` | 采样（与 RL 相同 seed-42 选择，不漏进 holdout）→ 基线 rollout（线程池，每线程一个 `SessionDriver`，`session_kwargs` 主线程预取）→ 技能标注（**直接复用** `scripts/sft/_common/pipeline.py::build_candidate_skill_record`，即 SEED 原版分析器提示）→ parquet 导出。轨迹 `task_id` 用 `slug/id` 前缀防跨 benchmark 的 skill_id 冲突 |
| `prepare_data.sh` | 环境变量约定与 alfworld 版一致（RUN_MODE=smoke\|full、POLICY_*/SKILL_*、teacher_naming）。**刻意简化：不管理 vLLM 生命周期**，要求 `POLICY_BASE_URL` 已就绪（不可达即报错并打印启动命令） |
| `train_sft.sh` | `_common/trainer.sh` 薄包装（`DATASET_NAME=agentstream`, `MODEL_TAG=qwen25_3b`）；输出目录命名与 prepare 默认输出对接 |

## 3. 关键协议 / 概念映射（理解代码前必读）

- **三种流模式 → SEED 训练数据流**：isolated = 每 benchmark 单独一次训练；sequential =
  单次训练按 benchmark 分块推进（`block_passes` 控制每块重复次数）；interleaved = 每个
  训练 batch 按保序交错的全局流取任务。GRPO 组内归一化天然兼容混合 batch。
- **两种评测协议并存**：`protocol=split`（SEED 式：train 流与 val 集不相交）/
  `protocol=online`（AgentStream 式：流即评测，`val_source=holdout|stream|both`）。
  两者都自动获得周期性 val（trainer 现有 `test_freq` 机制 + per-slug 指标键）。
- **在线指标**：训练相每个 episode 完成即写一行 JSONL（默认
  `<trainer.default_local_dir>/agentstream_online_metrics.jsonl`），`pass_idx>0` 的重复
  pass 不计入累计均值。
- **SEED rollout 契约**（新 manager 必须满足）：`reset(kwargs)` →
  `({text,text_base,image,anchor}, infos)`；`step(text_actions)` →
  `(obs, rewards, dones, infos)`，info 含 `won` / `is_action_valid`；rollout 循环对已 done
  的 env 仍会持续 step（worker 需容忍，返回 `post_done`）；`success_evaluator` 由基类驱动
  `_process_batch`，每个 batch idx 恰好 append 一条 `success_rate`。
- **exgentic 侧**：`Session.start()/step(action)/done()/score()`；
  `Evaluator.get_session_kwargs(SessionIndex(task_id, session_id))` 返回可序列化 dict
  （因此可跨 Ray 边界传递）；benchmark 重依赖隔离在各自 uv venv（HTTP runner）。

## 4. 静态已验证 vs 待运行验证

### 4.1 已验证（本机，无环境依赖）

- 全部 py 文件 `py_compile` 通过；shell 脚本 `bash -n` 通过；yaml 可解析。
- `task_stream.py` 自测：选择确定性、切分不相交、跨模式块内序一致、游标/pass 环绕、
  block 重复、random 模式组复制。
- 配置解析（含 isolated 单 benchmark 守卫、benchmark_kwargs 覆盖合并）、投影解析
  （围栏/坏 JSON/缺 think）、`render_prompt` 三分支、`analyze_results.py`
  （合成数据：组内平均/首 pass 过滤/CSV）。

### 4.2 需要实际运行验证的点（按 Phase 顺序，附预期与风险）

**Phase 0 — 冒烟（无 GPU，seed conda env）**

```bash
python examples/agentstream_trainer/smoke_env.py \
    --exgentic-root $AGENTSTREAM_EXGENTIC_ROOT --benchmarks bfcl --step
```

1. `import exgentic`：宿主轻依赖（pydantic v2 等）与 verl 依赖树共存性。若冲突，考虑
   pin 版本或在 `bootstrap_exgentic` 中做兼容处理，**不要**把 exgentic 装进 site-packages。
2. venv runner 启动与 `Evaluator.list_tasks()`（各 benchmark 已 `exgentic install` 过）。
3. **最大的未实测接缝**：`session.actions` 经 runner 代理返回的 `ActionType` 能否在宿主
   进程 `build_action`（exgentic 自家 agent 走同一路径，预期可行）。若代理对象缺 `cls`，
   fallback 方案：在 `SessionDriver._build_action` 里改用
   `exgentic.core.actions.build_unknown_action` + 让 session 侧校验。
4. `run_scope` 上下文在长驻进程中的行为（会话输出目录是否落在 `exgentic_output_dir`）。
5. 对其余 5 个 benchmark 重复冒烟（tau2 需 user-simulator 端点可达；swebench 需 Docker）。

**Phase 1 — RL 小规模冒烟（GPU）**

```bash
AS_BENCHMARKS=bfcl TRAIN_DATA_SIZE=2 VAL_DATA_SIZE=4 GROUP_SIZE=2 \
TOTAL_EPOCHS=2 TEST_FREQ=1 N_GPUS_PER_NODE=2 \
bash examples/agentstream_trainer/run_agentstream_random.sh
```

6. hydra CLI：`env.agentstream.benchmarks=[bfcl]` 列表解析、`${oc.env:AGENTSTREAM_EXGENTIC_ROOT,""}`
   resolver（若 hydra struct 模式报未知键，检查 yaml 块是否在位）。
7. Ray actor 内 exgentic 导入（`bootstrap_exgentic` 在 actor 进程内执行 sys.path 注入
   ——actor 不继承 driver 的 sys.path 修改）。
8. 并发 session 资源：`train_batch_size × group_n` 个 venv session 同时存在的内存/端口/
   文件句柄；每个 RL step 全批 session 重建的**吞吐**（预算：单 reset 应在秒级；若过慢，
   优先考虑 exgentic 的 service/persistent runner 复用）。
9. 奖励链路：`score() → won → 10.0`；wandb 里 `val/success_rate` 与 `<slug>_success_rate`
   曲线出现且非全零/全一异常。
10. 在线 JSONL 落盘（driver 进程单线程写，无并发问题；确认 `first_pass` 与 pass 环绕正确）。
11. `data.max_prompt_length=8192` 是否够 bfcl 工具 schema + 历史（`truncation=left` 会截头，
    观察 `filter_overlong_prompts` 的丢弃率）。
12. **科学风险探针**：基座（或 SFT 后）模型在各 benchmark 的成功率应落在 ~5%–60%；
    全零 → GRPO 组内 advantage 全零无学习信号，需换子集（如 `tau2 subset=mock/retail`、
    bfcl simple 类）或加大基座。tau2/hle 的 LLM 依赖（user-sim / judge）建议指到本地 vLLM。

**Phase 2 — SFT 管线**

```bash
RUN_MODE=smoke AS_BENCHMARKS=bfcl bash scripts/sft/agentstream/prepare_data.sh
# 通过后 full，再:
bash scripts/sft/agentstream/train_sft.sh
```

13. `build_candidate_skill_record` 对 agentstream 轨迹的兼容：它调用
    `seed.analysis.SEEDEpisodeAnalyzer._build_episode_analysis_prompt`（私有 API），传入的
    steps 字段（`observation`/`observation_prompt`/`response`/`action_valid`）已按
    `trajectory_to_seed_steps` 的期望构造；跑一条真实轨迹确认 prompt 合理、
    `_parse_analysis_response` 能解析。
14. 线程本地 `SessionDriver` 的并发稳定性（`--parallel-sessions 8`）；`session_kwargs`
    主线程预取的总耗时（任务多时是串行 HTTP 调用）。
15. parquet → `train_sft.sh` 的 `DATA_DIR` 命名对接（`SKILL_MODEL` 决定
    `SFT_SELF_DIR_SUFFIX`，prepare 与 train 两边必须一致的 `SKILL_MODEL`/teacher 配置）。

**Phase 3 — alfworld_stream（依赖 ALFWORLD_DATA）**

```bash
AS_STREAM_MODE=sequential bash examples/agentstream_trainer/run_alfworld_stream.sh
```

16. `collect_game_files` 在真实 `$ALFWORLD_DATA/json_2.1.1/train` 上每类型的可用数量
    （`num_tasks_per_type=30` + `val_tasks_per_type=8` 需要每类型 ≥38 个 solvable 游戏）。
17. 每 reset 注册单游戏 textworld env 的开销（对比原版批量 env 的 reset 时延）与长训练下
    textworld/gym registry 增长（目前接受；若成为问题，在 worker 内做 env 缓存）。
18. 现有 `AlfWorldEnvironmentManager` 与新 envs 的兼容闭环：admissible commands 投影、
    `extra.gamefile` → 任务类型 val 指标。
19. **回归**：不设 `ALFWORLD_ENV_NAME` 时原 `examples/seed_trainer/run_alfworld_*.sh`
    行为与集成前一致。

**收尾 — 分析**

```bash
python examples/agentstream_trainer/analyze_results.py \
    <run1>/agentstream_online_metrics.jsonl <run2>/... --csv all.csv
```

## 5. 已知取舍与后续工作（非 bug）

- swebench / hle / browsecompplus 为 Tier 2/3：分别有 Docker 并发、LLM judge 成本/噪声、
  检索服务依赖问题；建议 Tier 1（bfcl / tau2 / appworld）先全链路打通。
- `on_exhausted=cycle` 下 verl 以 `trainer.total_epochs` 计步，流长度与总步数解耦——
  正式实验需按流长度换算 `TOTAL_EPOCHS`（`stream_length = Σ num_tasks × block_passes`，
  每 step 消耗 `train_batch_size` 个流位置）。
- 在线协议的严格单 pass 对齐：若要完全模拟 AgentStream 单 pass 评测，设
  `on_exhausted=stop` 并让总步数 = `ceil(stream_length / train_batch_size)`。
- SFT 阶段与 RL 阶段的 `benchmark_kwargs` 需人工保持一致（两处配置无共享来源）。
