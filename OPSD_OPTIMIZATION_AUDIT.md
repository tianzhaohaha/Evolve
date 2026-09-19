# RL + OPSD 优化方向：当前代码复核与研究证据

## 0. 范围与结论

- 代码基准：工作树 `4f78436e270070df1920b7a3c55771c03d9e4cb1`，2026-09-19 审计。保留用户已有的两个启动配置修改；未修改训练算法、未启动训练。
- 历史 W&B 运行使用更早的 `1edea1f` 与 SFT-v3；不能拿当前 v5/debug 默认值解释历史实验。
- 本记录的验证是源代码检查、论文原文核验与 CPU 小张量验证，不是对论文或训练收益的复现。
- **RL + OPSD 有直接研究依据，但“技能被接受/teacher 更自信”不等于“学生任务能力提高”。优先检查信号是否正确、是否与更好的决策对齐，再增加池、replay 与梯度手术复杂度。**
- 下文“直接证据”指相关机制在相近训练范式中经过实验，不表示在本 AgentStream 异质流上已验证。2026 年新预印本也不能与多次独立复现等同。

三个需要分别回答的问题：

1. 检索的经验是否适用于当前决策状态？
2. 经验条件化 teacher 是否能更好地区分局部正确与错误行为？
3. 这类监督能否被 plain-prompt student 内化，并且不损害其他任务？

## 1. 必须修正的当前代码认识

### 1.1 已有开关，不应作为待发明功能

- Global pool 已有 `gate_ema/lru/window` 淘汰策略，已有 `min_sim` 拒绝检索；尚无 top-k reranker 和语义适用性判定。
- 已有带结果方向的入池分数：成功/未知用 `spec_gap`，失败 avoidance 用 `-spec_gap`。这不是对同一个 skill 的成功/失败对照效用。
- 已有 `failed_skill_positive` 路径：失败经验作为正向规则时跳过上述 gap 门槛、降级排序；并非“失败样本整条反向训练”。仍受 `admit_failed` 控制。
- 已有 signed episode/step teacher advantage，但任一独立 OPD 系数大于零就禁用这条旧路径；旧路径没有 gen advantage。
- 已有 EMA `off/ref/teacher/both`，但当前 full/debug 的 `ref` 不是 teacher EMA。历史三个运行 EMA 均关闭。

位置：[池策略与入池分数](SEED/seed/global_pool.py#L43-L176)、[旧 teacher advantage](SEED/gigpo/core_gigpo.py#L253-L316)、[双通道 actor loss](SEED/verl/workers/actor/dp_actor.py#L948-L1049)。

### 1.2 真正缺少的能力

- Analyzer 外层已有 JSON，但 `episode_skill` 与 step skill 内容是字符串，不是带约束验证的 skill schema。直接输出嵌套字典会被 `str()` 字符串化，而不是自动理解 `when/do/unless`。
- Global pool 保存历史单轨迹 episode skill 原文；近重复增加 support，不合并、重写或抽象出多轨迹规则。
- Query 使用任务文本和 first observation，分别截取 600 字符；不是当前回合状态。BFCL 后续用户回合不会自动进入这个 query。
- Judge 只看 skill 文本的可迁移性，不看当前状态适用性，也不估计因果奖励收益。
- Teacher 给原 student response 重算 sampled log-prob，不生成、执行修正 action；输出是 `[B,R]`，不是完整词表分布。
- Gen 构建目前依赖本地分析成功，不是完全独立可用的历史经验通道。
- Pool usage 记录 live batch 的 teacher-old gap gate，先轨迹内 token 平均，再同 skill 的轨迹平均；不使用成功/失败 contrast，也不等同 actor 的过滤/归约。
- `policy_vllm` 分析和 teacher scoring 在同步关键路径；只有 pool judge admission 异步。

位置：[分析协议](SEED/seed/analysis.py#L734-L779)、[解析器](SEED/seed/analysis.py#L899-L914)、[检索与存储](SEED/seed/global_pool.py#L291-L348)、[judge](SEED/seed/skill_judge.py#L20-L33)、[teacher scoring](SEED/verl/trainer/ppo/ray_trainer.py#L3430-L3515)、[pool usage](SEED/verl/trainer/ppo/ray_trainer.py#L2480-L2559)。

### 1.3 归约问题：先定义目标，再谈 task balance

1. 一行是一个环境 step response，**不是一条完整 episode**。`seq-mean` 不是 episode/task 等权。
2. `seq-mean-token-mean` 对每行除以 mask token 数，无零分母保护。CPU 提取当前函数验证：一个有监督行加一个全零 mask 行，结果为 NaN。OPD 的整 microbatch 全空保护不能覆盖这种部分空行情况。
3. 历史默认是 `token-mean`；上述 NaN **不是历史 crashed 的已知原因**。
4. 默认 episode reward normalization 按展开后的 step 行统计；长轨迹的 episode reward 重复进入组均值/方差。函数已有按 `(uid,traj_uid)` 去重分支，但正常 trainer 调用未暴露该选择。
5. `adjust_batch` 会复制行补齐，而且发生在 advantage 构建前；复制行可能改变组统计及 loss 权重。
6. 当前 microbatch 局部归约再累积，不一般等于全局有效 token 归约；不同 spec/gen 覆盖率与 rank 切分可改变实际权重。FSDP 确实同步梯度，不能将此误述为“rank 不同步”。

CPU 示例：回报 0/10、step 数 1/3 的两条轨迹，在标准差归一化下，当前 cross-step 统计得到约 `[-1.5,.5,.5,.5]`，episode 去重统计得到约 `[-.7071,.7071,.7071,.7071]`。另一个纯算术示例：两个 microbatch 有效 token 数 1/3、loss 均值 1/3，均值再平均是 2，全局 token 均值是 2.5。

这些结果证明实现权重的差别，不证明 episode 等权必然提高分数。

位置：[agg_loss](SEED/verl/trainer/ppo/core_algos.py#L395-L430)、[episode 统计](SEED/gigpo/core_gigpo.py#L624-L692)、[复制补齐](SEED/agent_system/multi_turn_rollout/utils.py#L87-L159)、[梯度累积](SEED/verl/workers/actor/dp_actor.py#L1044-L1049)。

## 2. 对当前 gate 的正确数学解释

令 sampled student log-prob 为 $l$、detached teacher log-prob 为 $t$：

$$
\Delta=t-l,\quad g=\operatorname{sg}[\sigma(\beta\Delta)],\quad
L=g(\operatorname{sg}[t]-l).
$$

当前 [compute_opd_loss](SEED/verl/trainer/ppo/core_algos.py#L495-L593) 对单个被选 token 的直接梯度是 $-g/D$。负 gap token 在未启用 `positive_only` 时仍有正的模仿权重。

**但这不推出策略分布无法降低错误动作概率。**在固定状态、共同固定 teacher、严格 on-policy、没有额外筛选的简化条件下：

$$
\mathbb E\left[\frac{\partial L}{\partial z_j}\right]
=p_j\left(\mathbb E_p[g]-g_j\right).
$$

相对较低的 gate 可以降低相应动作的概率。Teacher=student 时，sample loss 为零、sample gradient 可不为零，但期望梯度为零。不能以 loss 值或单个 sample 的正权重判定整个目标错误。

[SEED 原论文](https://arxiv.org/html/2607.14777v1) Appendix A 进一步将轨迹相关 hindsight 纳入条件期望权重 $w(c,v)$，定义有效目标 $r(v|c)=p(v|c)w(c,v)/Z(c)$，其局部价值变化为：

$$
\mathbb E_r[Q]-\mathbb E_p[Q]
=\frac{\operatorname{Cov}_p(Q,w)}{Z}.
$$

真正值得验证的是 **teacher support 与优质行为的对齐**，而不是 mean gate 是否超过 0.5。论文也不保证单调 reward 改进。

另一个重要限制：固定 teacher、on-policy 下 $\mathbb E_p[\log q-\log p]=-D_{KL}(p\|q)\le0$。因此负平均 gap 本身不能证明 teacher 无用；实际 hindsight、过滤、old-policy 与 replay 又会使日志不再满足该简化解释。

- `g-.5` 是常数 score-function baseline：在严格 on-policy、固定状态且无额外动作依赖归约时可保持期望梯度；不保证降方差。常数 baseline 的零均值性质不要求 gate 本身不依赖 hindsight。
- `2g-1` 还将尺度翻倍，不能不调系数就当成纯居中对照。
- `positive_only` 只过滤 loss 分子，`gate_eps` 过滤 mask 和分母；两者不等价。
- fractional mask 在 weighted mean 中主要重新分配权重；统一缩小所有权重可能被分母抵消。
- PPO clipping 不是严格 trust-region 保证；现有独立 OPD loss 也不处于 PPO clipping 内。

## 3. 逐项方向审计：表示、检索、效用

证据等级：**A** 相近参数训练中的直接机制/实验；**B** agent 上下文记忆或较远训练范式的支持；**C** 合理假设但未找到对该具体设计的直接验证。一个方向可同时包含 A/B/C 层次。

| # | 方向及当前差距 | 研究支持与边界 | 修正后的判断与最小验证 |
|---|---|---|---|
| 1 | 条件 skill：`when/do/unless`；当前值为自由字符串 | SkillRL 用 `principle/when_to_apply`；Skill-SD 用成功分析、错误分析、理想流程。A/B：支持明确条件与结构；没有完整三字段 schema 的独立优越性证明 | 先保留原字符串协议，要求表达适用条件和例外。真正 schema 化要联动 parser、render、hash、embedding、judge、持久化；旧 SFT analyzer 未必适应新格式 |
| 2 | 将“跨任意领域通用”放宽为“在明确作用域内复用”；当前 judge 偏广泛迁移 | SkillRL 分环境通用/任务特定；Skill-SD 是 task-local bank。支持有边界的知识，而非跨领域万能规则 | 较高优先级。按工具族、状态前提判断适用性；不要简单降低 judge 阈值后直接全部蒸馏 |
| 3 | 当前状态检索；当前仅 task+first observation，整轨迹同 skill | Voyager 使用环境反馈与技能；Self-RAG 支持条件检索，后者是文档 RAG，属 B | 对多轮 BFCL 有明确机制动机。只用当前动作发生前可见的状态查询；需要 row-level skill provenance、usage 和 replay 路由，不是仅改 query 一行 |
| 4 | top-k 候选、重排、允许不选；已有 top1+min_sim | SkillRL 使用阈值/top-k；Self-RAG 支持适用性判断；ReasoningBank 多取经验反而下降 | 先检索 top-k、最终选 0/1，而非全拼 prompt。与现有阈值 top1 比适用率、teacher token 成本和状态级正确性；不能保证 reranker 自评可靠 |
| 5 | 同一 skill 的 success-minus-failure gap；当前仅单轨迹 signed admission 和 usage gate EMA | ExpeL/ReasoningBank 支持成功失败对照，但没有验证这个数值效用。C | 先作为诊断、不用于硬淘汰。必须同任务、同 skill、去 padding 重复、跨轨迹验证；全成功/全失败组是不可估计，不是效用为零 |
| 6 | LRU、candidate/proven tiers、不同任务证据；已有 LRU/window、全部 active | ExpeL 有规则修订/投票；Voyager 有验证后纳入；不能据此声称 LRU 胜过 gate EMA。LRU 为 C，验证式生命周期为 B | 先用现成 LRU 做廉价控制，检验 used<.5 与 unused=.5 的保留偏置。暂缓复杂 proven tiers；support 次数不等于独立收益证据 |
| 7 | local=.005/gen=0 控制；未运行 | 实验识别需求，不需要论文背书 | 必须保留待办，冻结历史设置。local .01→.005 与新增 gen 的影响目前混杂。gen=0 时用 `pool.source=copy`，不能保持 `source=pool` 绕过配置约束 |
| 8 | 同任务多轨迹对比提炼；当前单轨迹 analyzer | ExpeL 同任务成功/失败比较；ReasoningBank MaTTS 并行经验对照。B；SDPO 成功 sibling 条件化有 A，但不是抽象 skill 提炼的独立消融 | 推荐。先只替换 pool 候选来源，保留 local SEED；固定分析 token 预算，避免把更多观察量误认为抽象算法收益。多轨迹输入改变 analyzer 的 SFT 分布 |
| 9 | 短经验片段 vs 抽象 skill vs skill+example；当前仅字符串 skill | ExpeL 检索成功轨迹；SkillRL raw trajectories ablation较差；SDPO 反馈/示范有效，但拼回完整原尝试会加重模仿偏置 | 做等 token 三路对照，不推荐直接增加上下文。片段必须含必要前置状态；不能把不同状态的正确 action 直接复制到当前状态 |
| 10 | 局部知识留 memory、复用规则进参数；当前 pool 只给 teacher | ExpeL/ReasoningBank 支持上下文记忆；Skill-SD/SDPO/持续学习自蒸馏支持参数内化；没有该确切“按频率分流策略”的验证 | 作为设计假设。若 student 也读 memory，要明确改变部署协议，并确保行为 log-prob 对应实际 rollout prompt；不能训练有记忆、评估移除后仍称等价 |

### 同 skill outcome-gap 仅作 proxy 的定义

令 $d(\tau,k)$ 为同一 skill $k$ 对轨迹 $\tau$ 的平均 teacher-student log-gap，可记录：

$$
U(k)=\mathbb E[d(\tau,k)\mid success]-\mathbb E[d(\tau,k)\mid failure].
$$

它测量相对区分度，而非因果 reward 增益。Gen 在同组命中同 skill 且存在混合结果时可能直接收集；local 每条轨迹生成的 skill 不同，不能直接相减，必须将候选 skill 对 sibling 轨迹 cross-score。候选生成与验证尽量分开，避免“在自己解释过的数据上自证正确”。即使这样，轨迹状态、长度与选择偏差仍存在。

如果需分离 EMA lag 与 prompt 的效应，应该在同一 teacher 权重下比较 skill/no-skill，而不是只看 EMA teacher 与当前 student 的差。

## 4. 逐项方向审计：蒸馏目标与 teacher

| # | 方向及当前差距 | 研究支持与边界 | 修正后的判断与最小验证 |
|---|---|---|---|
| 11 | Signed teacher advantage / ratio surrogate；旧 local 路径有，双通道没有 | SDPO 有正负 log-ratio credit；Skill-SD 有 importance-weighted reverse-KL。A，明显强于只引用通用 KD | 高优先级候选，但不是“修复数学上错误的 gate”。先比较目标梯度、符号和尺度，再做 local-only 受控对照。若保留双通道，要独立实现，不可只切旧 advantage 权重 |
| 12 | centered gate；当前未实现 | 常数 baseline 有理论解释，没有该系统直接性能证据。C | 降级探索。无需优先于已有直接依据的 signed/distribution loss；过滤、replay、归约改变后不能照搬理想无偏结论 |
| 13 | action-span-only 蒸馏；当前 response mask覆盖整个模型回答 | 未找到“只蒸馏 tool call 比推理+action 更好”的直接支持；SOD 只排除环境 observation，仍含推理与最终回答 | 降级。工具名/参数/结束动作有实用动机，但需 token offset 对齐、多格式与空 span 处理；保留完整 attention，只改 loss mask，不能删上下文 |
| 14 | 覆盖率/梯度预算控制；当前固定系数、局部归约 | GradNorm 支持多任务幅度平衡，Skill-SD 有系数消融；不是当前动态调权规则的直接验证 | 先记录有效 token 数、梯度范数和方向。系数和相等不意味着梯度预算相等；不直接把符号变化、接近零的 RL loss 代入 GradNorm 相对学习速度公式 |
| 15 | full-vocab / top-k+tail divergence；当前只有 `[B,R]` LP | GKD、SDPO A；SDPO 有 logit/token/sequence 监督粒度消融。MiniLLM 的序列目标更复杂 | 更强但成本更高的方向。需 teacher 返回 support ID、对应 log-prob 和 tail mass，student 在同 support 对齐。当前 `kl_penalty='full'` 未实现，不是可用开关；top-k 截断后仅重归一化不是 SDPO 的 top-k+tail |
| 16 | teacher 修正动作/偏好监督；当前无修正生成 | Re-ReST 在 HotpotQA/ALFWorld 等执行“失败→修正→环境复验→SFT”，提供 A；SDPO 则直接重评原轨迹、不生成修正 | 有依据，但成本较高。Re-ReST 主要是整任务新尝试，不验证任意同状态分支。局部修正必须恢复真实状态、重新执行后续流程；成功/失败轨迹并非每个 action 都构成可靠 preference |
| 17 | EMA teacher；已有模式但 ref≠teacher | SDPO §4.3 有 EMA/冻结/即时 teacher 独立消融；持续学习自蒸馏也评估 EMA。Skill-SD 用每轮同步，不是 EMA | 优先于 centered gate。固定 objective 与 context，只变 teacher；随后另测 ref。核对系数约定和更新频率：本仓库每次完整 actor update 后更新，不是每 minibatch；analyzer 仍是 live policy |

### 4.1 比 centered gate 更明确的中等成本候选

Skill-SD 使用 sampled-token 目标，可复用当前三组标量 log-prob：

$$
\ell=\log p_\theta-\log q,\qquad
\rho=\exp(\log p_\theta-\log p_{old}),
$$

$$
L_{SDL}=\rho\left(e^{-\ell}-1+\ell\right),\qquad
\nabla L_{SDL}=\rho\ell\nabla\log p_\theta.
$$

- Teacher 和 old policy detach；**ratio 的 current-student 分子保持可微**。只乘一个 detached ratio 或裸 `low_var_kl` 不等价。
- Scalar loss 非负并不意味着只能正向模仿；梯度由 signed log-ratio 决定。
- CPU 三点验证涵盖正 gap、负 gap、相等，autograd 与 $\rho\ell$ 一致，相等时 sample gradient为零。
- 论文的条件无偏结论要求固定 prefix、固定 teacher 分布及匹配的采样分布/support；不能据此声称修复整个 replay 状态占用偏差。
- 特别是当前 local skill 来自待评分的同一完整轨迹，teacher context 依赖 sampled action/未来，不能不加限定地套用固定 teacher 的条件无偏解释。历史池/跨样本产生且在决策前固定的 skill 更接近该假设。
- 数值稳定性、ratio/gap 裁剪、微批归约仍需测试；裁剪会改变上述精确梯度关系。
- 不照搬其最佳 `lambda=.001`：损失定义、规模、任务和归约都不同。

### 4.2 更有证据的 teacher context 对照

SDPO 给 teacher 的主要额外信息是环境反馈或同题成功样本，并重评 student 原输出；这与“先把一切压缩成一段抽象 skill”不同。推荐未来对比：

1. 原始 local skill；
2. 有限、可核验的反馈/成功 sibling 证据；
3. 有适用条件的 skill + 最小证据片段。

固定 context token 预算。Siblings 必须同任务；多轮环境下还要区分前置状态。Teacher 完整 rollout 成功率改善是有用诊断，但**不是有用局部 credit 的必要条件**：SDPO 的困难题实验中，初始 teacher 几乎不能一次解出，仍报告后续学习收益。

### 4.3 Step 软加权：替代武断硬 mask 的候选

SOD 在 Python-tool math/science/code 上比较 step 软加权与 uniform、固定衰减、首个错误后全 mask；硬 mask 更差。其 step 是工具 observation 之间的完整模型输出，不是纯 action span。

可借鉴“保留错误后恢复段，不整条丢弃”，但小 teacher-student gap/低熵不是正确性证明。SOD 使用强外部 teacher，不是本项目同模型 skill teacher；其经验不能直接移植。本文仅引用经验消融，不依赖其关于 divergence proxy/SNR 的普遍理论保证。当前 fractional mask 与分母耦合，若要真正削弱总体监督，需明确独立 loss weight 与归一化规则。

## 5. 逐项方向审计：异质任务、replay、干扰

| # | 方向及当前差距 | 研究支持与边界 | 修正后的判断与最小验证 |
|---|---|---|---|
| 18 | task/group/episode-balanced aggregation；当前 row/token/microbatch 混合权重 | DAPO 与 Dr.GRPO 说明归约会改变训练；两者目标并不相同，不能共同当成“任务等权最优”背书 | 优先修正目标定义与实现一致性，再将 task 等权作为独立实验。带上 task/traj metadata；分别定义 PPO/spec/gen 分母；测试空行、padding、不同 microbatch/rank 切分 |
| 19 | successful-action SFT replay 或重新评分 teacher；当前缓存旧 PPO/OPD 信号 | Re-ReST 支持验证成功样本自训练；DER 支持旧 logits 作为保留目标；SDPO 反例说明 teacher-success SFT 可能比 OPD 更遗忘 | 不再称 SFT replay“更安全”。先明确 retention 与 improvement 目标。当前 skill-gen auxiliary 是分析 JSON 的 reward-weighted LM，不是 action SFT replay；默认关闭 |
| 20 | replay token/age/dedup 预算；当前按 uid group reservoir，容量不等于 age TTL | ER/DER 是通用持续学习支持，具体 TTL/token 方案仍属工程假设 | 较低风险优先。记录行为版本、teacher版本、插入步、token数与来源；固定 optimizer update 数以免把更多训练量混进 replay 收益 |
| 21 | 梯度诊断后 PCGrad/A-GEM/projection；当前无拆分梯度统计 | PCGrad/A-GEM 有多任务/持续学习原始证据，但主要非 LLM；Self-Distillation Enables Continual Learning 更直接支持 LLM 自蒸馏抗遗忘，不验证 projection | 先稀疏测 RL/spec/gen 及 domain 间梯度范数、cosine；确认持续冲突与 held-out 损害后才投影。A-GEM 要当前参数下 memory gradient，不是缓存旧梯度；FSDP、多次反传与 Adam 更新都增加复杂度 |

### Replay refresh 并非再调用一次 teacher

当前 [replay buffer](SEED/seed/replay.py) 缓存普通输入、old/ref/teacher LP、advantage 和 masks，但清理 `meta_info`，不保存完整 augmented teacher inputs、skill-gen payload 或完整 row-level skill provenance。`has_policy_signal` 检查 nonzero advantage，不等于成功，也不等于有 OPD 信号。

若刷新 teacher，需要：

1. 去重 padding，恢复轨迹与 step 顺序；当前 step index helper 按行出现顺序计数，不能默认 replay 排序正确。
2. 明确保留旧 skill、重新生成还是重新检索，记录来源。
3. 用同 tokenizer、template、truncation 还原 teacher prompt。
4. 重算所有使用中的 spec/gen LP、有效 masks；若使用 teacher advantage，也重算对应 advantage。
5. EMA ref 是否更新是另一个选择，不与 teacher refresh 混为一谈。
6. **保留真实行为 `old_log_probs`。不能重写成当前 LP，再声称历史数据恢复 on-policy。**
7. 明确保留历史 teacher 是 retention target 还是追求更好决策的 improvement teacher；前者不应一律被“刷新”。

## 6. 最相关论文与可支持的具体判断

所有数值为作者报告，未在本项目复现。未注明会议的 2026 项按所读 arXiv 版本处理。

| 论文 | 关键证据 | 不能据此推出 |
|---|---|---|
| [SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning, 2026](https://arxiv.org/html/2607.14777v1) | teacher-only hindsight skills、gated sampled loss；理论要求相对权重与好行为对齐 | Appendix B.1 按 benchmark 分别 SFT/RL，不是单模型异质持续任务流；没有单调回报保证 |
| [Reinforcement Learning via Self-Distillation / SDPO, 2026](https://arxiv.org/html/2601.20802v2) | 反馈/成功 sibling 条件化；top-k+tail；EMA/冻结/即时 teacher、监督粒度、反馈内容消融；LCBv6 报告 48.8 vs GRPO 41.2 | ToolAlpaca 不是长程工具 agent；弱模型可不如 GRPO；hybrid GRPO 帮助弱模型但不总帮助强模型 |
| [Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents, 2026](https://arxiv.org/html/2604.10674v1) | AppWorld/Sokoban；teacher-only skill；可微 ratio × k3；动态/冻结 teacher 消融；AppWorld 64.9 vs GRPO 50.9 | task-local UCB，不是跨任务语义池；整体提升不是任意结构化 skill 或本 loss 单因素收益；纯OPD对照同时使用冻结teacher |
| [Self-Distillation Enables Continual Learning, 2026](https://arxiv.org/html/2601.19897v2) | 同一 LLM 顺序学习科学、工具、医学；student生成prefix，teacher额外看可靠 demonstration；较 SFT 遗忘小；EMA消融 | 不是无外部信息：仍需示范；非长程环境流、非RL+历史replay。v2理论用reverse KL但实现说明称forward KL最好，不可简化为“reverse KL证明抗遗忘” |
| [Re-ReST: Reflection-Reinforced Self-Training for Language Agents, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.861/) | 失败输出+反馈→reflector修正→环境复验→成功样本SFT；涉及ALFWorld等 | 主要任务级新尝试，不提供任意中间状态branch证明；不验证异质流SFT replay抗遗忘 |
| [SOD: Step-wise On-policy Distillation for Small Language Model Agents, 2026](https://arxiv.org/html/2605.07725v3) | step软加权优于首错后全部mask；强teacher教小模型，Python工具任务 | 不是纯tool span loss，不证明低gap就是可靠，不证明API/web流同样收益 |
| [ExpeL: LLM Agents Are Experiential Learners, AAAI 2024](https://arxiv.org/html/2308.10144v3) | 同任务成功/失败比较、成功案例检索、文本规则修订 | 不更新权重；检索的是轨迹范例而非论文已实现的动态insight检索；不支持特定数值gap utility |
| [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory, 2025](https://arxiv.org/html/2509.25140v2) | 成败经验抽象、MaTTS多轨迹对照；Shopping中加失败经验46.5→49.7 | 不更新权重；其他记忆基线加失败经验可下降；多检索不总更好；实验consolidation以append为主，不是已验证LRU |
| [SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning, 2026](https://arxiv.org/html/2602.08234v1) | 条件化skill、环境通用/任务特定层次、动态库；有参数RL训练及raw-trajectory对照 | skills在actor context，训练/部署协议不同于SEED；框架消融不能隔离证明 `when_to_apply` 字段本身增益 |
| [GKD: On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes, ICLR 2024](https://arxiv.org/html/2306.13649v3) | student prefix上的分布散度，含RL+GKD实验 | 主要LM任务，不证明异质agent流/当前gate替换一定好 |
| [MiniLLM: Knowledge Distillation of Large Language Models, ICLR 2024版本](https://arxiv.org/html/2306.08543v4) | reverse-KL、student生成、梯度估计与方差控制 | 不是把single-token gap改符号就等价；包含未来return、混合采样等细节 |
| [Self-RAG, ICLR 2024](https://arxiv.org/html/2310.11511v1) | 学习检索必要性、相关性/支持度等判断 | 文档RAG；不是不训练即可获得可靠skill适用性judge |
| [Reflexion, NeurIPS 2023](https://arxiv.org/html/2303.11366v4) / [Voyager, 2023](https://arxiv.org/html/2305.16291v2) | 反馈反思、可执行技能、环境验证与复用 | 不更新权重；Reflexion最近若干记忆是窗口而非LRU；反思有负例 |
| [DAPO, 2025](https://arxiv.org/html/2503.14476v2) / [Dr.GRPO, 2025](https://arxiv.org/html/2503.20783v2) | 归约/长度/奖励标准差会影响优化 | DAPO实际token归约并保留std；Dr.GRPO用固定分母并去std；不能混称同一方法 |
| [PCGrad, NeurIPS 2020](https://arxiv.org/html/2001.06782v4) / [A-GEM, ICLR 2019](https://arxiv.org/html/1812.00420v2) | 冲突梯度投影、历史平均约束 | 非本项目LLM/FSDP实证；负cosine不充分证明有害，平均约束不保护每个旧任务 |
| [DER, NeurIPS 2020](https://arxiv.org/html/2004.07211v2) / [Tiny Episodic Memories, 2019](https://arxiv.org/html/1902.10486v4) | reservoir、历史logits保留、简单ER强基线 | 非LLM；不能替旧PPO advantage/replay off-policy风险背书 |
| [GradNorm, ICML 2018](https://arxiv.org/html/1711.02257v4) / [Unlikelihood Training, 2019](https://arxiv.org/html/1908.04319v2) | 多任务梯度幅度平衡 / 对可靠负候选做局部负监督 | 前者不修正方向，后者不允许把失败轨迹所有token当负例 |

SEED 论文措辞将 teacher/student 评估放在当前参数上；本仓库 teacher LP 在 actor minibatch 更新前缓存。多 minibatch/epoch/replay 时二者时序不同，这是需要控制的实现差别，不自动意味着 bug。

## 7. 验证顺序：少量可解释实验，不一次堆叠所有改动

### P0：可识别性与实现一致性

- 保留待运行的历史设置 `local=.005/gen=0`。三个已提供运行实际为 local SEED、Global、Global+Replay，**没有纯 RL**；不能据此证明 SEED<RL。
- 在固定任务 holdout 上追踪分域表现，区分在线任务难度变化与遗忘；按 task 聚类统计不把同任务重复当独立样本。
- 检查 teacher skill 在真正 tokenization/truncation 后是否仍可见、response 对齐是否正确；pretokenization dump不足。
- 为 loss 写明期望权重与分母，覆盖部分空mask、所有空mask、padding去重、microbatch/rank划分一致性；不要直接切不安全的 seq mean。
- 预算至少同时报告环境交互量、teacher/analyzer tokens、optimizer updates 与 wall-clock。相等 `lambda_spec+lambda_gen` 不是等梯度预算。

### P1：先提高信息质量

- 原字符串协议内加入条件与例外，允许有界的工具族/任务族复用。
- 用同任务多轨迹提炼改进 pool 候选；独立比较反馈/成功 sibling 证据作为 teacher context。
- 测 teacher 对已验证局部行为的区分、适用/不适用检索和 held-out迁移，不以 mean gate 验收。
- 现成 LRU 只作为低成本对照；不在 utility 尚未校准时投入复杂 candidate/proven系统。

### P2：目标与 teacher 稳定性

- 固定 context/数据，比较原 gate 与明确 signed/importance-weighted sampled reverse-KL；先 local-only，再双通道。
- 独立比较 live vs EMA teacher，reference保持不变；避免把 teacher lag、prompt效果、ref约束一并改变。
- 若 sampled方案有效且成本允许，再实现 SDPO式top-k+tail；不要先扩成full vocab。
- centered gate、action-only、step soft weighting属于后续单项消融，不预设收益。

### P3：持续学习与 replay

- 先加年龄/版本/有效token与优化步预算，明确retention/improvement目标。
- 验证成功SFT、旧teacher保留、新teacher重评应分成不同方案；不是三者同时叠加。
- 稀疏梯度诊断证实可重复干扰后，才考虑PCGrad/A-GEM。优先用plain-prompt student的固定旧任务评估检验遗忘。

**进入下一阶段的依据应是：teacher监督更能区分好决策、student holdout确有改善、其他域损害可控且成本可接受；不是池更大、分析成功率更高或gate更高。**

历史实证与待办见 [outputs/wandb_diagnostics_20260919/REPORT.md](outputs/wandb_diagnostics_20260919/REPORT.md)。