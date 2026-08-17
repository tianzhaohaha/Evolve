# AgentStream 分阶段 nohup 运行手册

本文档用于分阶段运行 SEED x AgentStream 正式验证。每个阶段都可以单独复制执行，检查完成后再进入下一阶段。

SEED 论文流程与脚本阶段的对应关系：

| 脚本阶段 | SEED 阶段 | 内容 |
|---|---|---|
| Stage 1/3 | Stage 1 数据构造 | rollout -> hindsight skill -> SFT parquet |
| Stage 2/3 | Stage 1 模型训练 | SFT 训练并导出 Hugging Face 模型 |
| Stage 3/3 | Stage 2 | Self-Evolving OPD RL |

默认正式配置位于 `examples/agentstream_trainer/agentstream_full.env`：

- benchmark：`bfcl,appworld,tau2`（bfcl 使用 `multi_turn_base` 子集，与原始 AgentStream 对齐；tau2 使用 `retail`）
- 每个 benchmark 50 个任务（seed-42 选择），每个任务 8 条 Stage-1 rollout
- SFT：2 epoch
- RL_OPD：sequential/interleaved 各 80 步，isolated 每个 benchmark 20 步
- 步数上限：全局 40，按 benchmark 覆盖 `{"bfcl":25,"tau2":30,"appworld":40}`（`AGENTSTREAM_MAX_STEPS_JSON`）
- 奖励：`10 × success + score`（`reward_use_score=True`，纳入 benchmark 的连续分数）
- GPU：2–7；Stage 1 本地 vLLM 只使用 GPU 2
- RL 模式：`sequential`、`interleaved`、`isolated`

> **配置断代提醒**：bfcl 子集、prompt 模板（带历史模板现每步携带 Task context）、
> 步数与奖励定义近期同时变更。旧的 Stage 1 数据与 Stage 2 SFT 模型和新配置不兼容，
> 需从 Stage 1 重跑；重跑时建议更换输出目录版本号（`AGENTSTREAM_SFT_DATA_DIR`、
> `AGENTSTREAM_SFT_MODEL_DIR`、`AGENTSTREAM_EXPERIMENT_PREFIX`），不要用
> `AGENTSTREAM_PREPARE_RESUME=true` 在旧数据目录上续跑，否则新旧任务集会混在一起。

所有命令均从 SEED 根目录执行：

```bash
cd /home/jcgu/qyliu/OPDevolve/SEED
mkdir -p logs/agentstream
```

> 不要同时运行 Stage 2 和 Stage 3，也不要并行运行多个 RL 模式。默认配置会让它们争用 GPU 2–7。

## 运行前检查

检查 GPU 和关键路径：

```bash
nvidia-smi

test -f /home/jcgu/qyliu/LLMs/Qwen2.5-3B-Instruct/config.json && echo "base model: OK"
test -d /home/jcgu/qyliu/OPDevolve/AgentStream/exgentic/src/exgentic && echo "exgentic: OK"
conda run -n seed python -c 'import litellm, ray, torch; print("seed environment: OK")'
```

检查配置展开结果：

```bash
bash -c '
source examples/agentstream_trainer/agentstream_full.env
printf "benchmarks=%s\ntasks/domain=%s\nsft_epochs=%s\nrl_modes=%s\nrl_epochs=%s\n" \
  "$AGENTSTREAM_BENCHMARKS" \
  "$AGENTSTREAM_NUM_TASKS" \
  "$AGENTSTREAM_SFT_EPOCHS" \
  "$AGENTSTREAM_RL_MODES" \
  "$AGENTSTREAM_RL_EPOCHS"
'
```

## Stage 1：生成 Hindsight-Skill SFT 数据

该命令会自动启动 Stage-1 本地 vLLM，生成 rollout 和 skill 数据，完成后停止该 vLLM。外部 skill teacher 的地址和凭据从仓库 `.env` 读取，不会写入日志命令。

```bash
nohup env \
  AGENTSTREAM_RUN_PREPARE=true \
  AGENTSTREAM_RUN_SFT=false \
  AGENTSTREAM_RUN_RL=false \
  AGENTSTREAM_PREPARE_RESUME=true \
  bash scripts/sft/agentstream/run_all.sh \
  > logs/agentstream/stage1_prepare.log 2>&1 &
echo $! | tee logs/agentstream/stage1_prepare.pid
```

查看进度：

```bash
tail -f logs/agentstream/stage1_prepare.log
```

检查进程是否仍在运行：

```bash
ps -fp "$(cat logs/agentstream/stage1_prepare.pid)"
```

检查 Stage 1 结果：

```bash
DATA_DIR=/home/jcgu/qyliu/OPDevolve/SEED/outputs/agentstream_episode_skill_pipeline_formal_glm_self

ls -lh \
  "$DATA_DIR/baseline_rollouts.jsonl" \
  "$DATA_DIR/candidate_skills.jsonl" \
  "$DATA_DIR/inspection/rollouts.json" \
  "$DATA_DIR/inspection/skills.json" \
  "$DATA_DIR/sft_episode_skill_train.parquet" \
  "$DATA_DIR/sft_episode_skill_val.parquet" \
  "$DATA_DIR/metrics.json"

conda run -n seed python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/agentstream_episode_skill_pipeline_formal_glm_self")
metrics = json.loads((root / "metrics.json").read_text())
print(json.dumps(metrics, indent=2, ensure_ascii=False))
assert metrics["baseline_rollouts"] > 0
assert metrics["parse_ok_skills"] > 0
assert metrics["sft_records"] > 1
print("Stage 1: PASS")
PY
```

`inspection/rollouts.json` 和 `inspection/skills.json` 是稳定排序后的少量检查样本；完整数据仍在对应 JSONL。检查样本不包含完整 action-schema prompt，并会脱敏疑似密钥、长 base64，单字段默认截断到 4000 字符。通过 `INSPECTION_SAMPLES` 和 `INSPECTION_MAX_CHARS` 调整，设 `INSPECTION_SAMPLES=0` 可关闭。

正常结束时日志应包含：

```text
Pipeline complete.
AgentStream full pipeline finished.
```

如果需要从头重新生成数据，将命令中的 `AGENTSTREAM_PREPARE_RESUME=true` 替换为：

```text
AGENTSTREAM_PREPARE_RESUME=false
AGENTSTREAM_PREPARE_OVERWRITE=true
```

这会覆盖 Stage 1 正式数据目录，执行前应确认旧数据不再需要。

## Stage 2：训练并导出 SFT 模型

只有 Stage 1 检查通过后再执行：

```bash
nohup env \
  AGENTSTREAM_RUN_PREPARE=false \
  AGENTSTREAM_RUN_SFT=true \
  AGENTSTREAM_RUN_RL=false \
  bash scripts/sft/agentstream/run_all.sh \
  > logs/agentstream/stage2_sft.log 2>&1 &
echo $! | tee logs/agentstream/stage2_sft.pid
```

查看训练日志：

```bash
tail -f logs/agentstream/stage2_sft.log
```

WandB 的 `sft/data_samples` 在第 1 step 及之后每 10 step 上传最多 3 组固定 `prompt/target`，用于检查监督数据是否正确。SFT 不运行 AgentStream 环境，因此这些不是 rollout，也不应解释为当前模型生成。可用 `SFT_LOG_DATA_SAMPLES`、`SFT_LOG_DATA_SAMPLES_FREQ` 和 `SFT_LOG_DATA_SAMPLES_MAX_CHARS` 控制。

检查进程和 GPU：

```bash
ps -fp "$(cat logs/agentstream/stage2_sft.pid)"
nvidia-smi
```

检查导出的 Hugging Face 模型：

```bash
SFT_MODEL=/home/jcgu/qyliu/LLMs/Qwen2.5-3B-Instruct-agentstream-episode-skill-sft-glm-self

test -f "$SFT_MODEL/config.json" && echo "config.json: OK"
find "$SFT_MODEL" -maxdepth 1 -type f \
  \( -name '*.safetensors' -o -name 'tokenizer*' -o -name 'config.json' \) \
  -printf '%f %s bytes\n' | sort
```

正常结束时日志应包含：

```text
Exported SFT model to /home/jcgu/qyliu/LLMs/Qwen2.5-3B-Instruct-agentstream-episode-skill-sft-glm-self
AgentStream full pipeline finished.
```

## Stage 3：Sequential Self-Evolving OPD RL

```bash
nohup env \
  bash examples/agentstream_trainer/run_agentstream_sft_glm_self.sh sequential \
  > logs/agentstream/stage3_sequential.log 2>&1 &
echo $! | tee logs/agentstream/stage3_sequential.pid
```

查看日志：

```bash
tail -f logs/agentstream/stage3_sequential.log
```

检查 checkpoint 和在线指标：

```bash
RUN_DIR=/home/jcgu/qyliu/OPDevolve/SEED/checkpoints/seed_qwen2.5_3b_agentstream_sft_glm_self_sequential_online_s44

find "$RUN_DIR" -maxdepth 2 -type f \
  \( -name 'latest_checkpointed_iteration.txt' -o -name 'agentstream_online_metrics.jsonl' \) \
  -printf '%p %s bytes\n'

tail -n 5 "$RUN_DIR/agentstream_online_metrics.jsonl"
```

Stage 3 的完整 rollout 继续按 step 写入 `$RUN_DIR/<step>.jsonl`；WandB 的 `rollout/samples` 默认在第 1 step 及之后每 10 step 上传最多 3 条真实环境交互样本，包括轨迹编号、turn、observation、response、score 和 action validity。文本会脱敏并截断到 4000 字符。可用 `WANDB_ROLLOUT_SAMPLES`、`WANDB_ROLLOUT_SAMPLE_FREQ` 和 `WANDB_ROLLOUT_SAMPLE_MAX_CHARS` 控制，样本数设为 0 可关闭上传。

## Stage 3：Interleaved Self-Evolving OPD RL

Sequential 完成并检查通过后再执行：

```bash
nohup env \
  bash examples/agentstream_trainer/run_agentstream_sft_glm_self.sh interleaved \
  > logs/agentstream/stage3_interleaved.log 2>&1 &
echo $! | tee logs/agentstream/stage3_interleaved.pid
```

查看日志：

```bash
tail -f logs/agentstream/stage3_interleaved.log
```

输出目录：

```text
/home/jcgu/qyliu/OPDevolve/SEED/checkpoints/seed_qwen2.5_3b_agentstream_sft_glm_self_interleaved_online_s44
```

## Stage 3：Isolated Self-Evolving OPD RL

Interleaved 完成并检查通过后再执行。Isolated 会按 benchmark 列表依次启动多个独立训练（默认 bfcl、appworld、tau2 三个），权重互不共享，各自产生独立的 checkpoint 目录和 WandB run：

```bash
nohup env \
  bash examples/agentstream_trainer/run_agentstream_sft_glm_self.sh isolated \
  > logs/agentstream/stage3_isolated.log 2>&1 &
echo $! | tee logs/agentstream/stage3_isolated.pid
```

查看日志：

```bash
tail -f logs/agentstream/stage3_isolated.log
```

预期每个 benchmark 各产生一个目录（默认三个）：

```bash
find /home/jcgu/qyliu/OPDevolve/SEED/checkpoints -maxdepth 1 -type d \
  -name 'seed_qwen2.5_3b_agentstream_sft_glm_self_isolated_online_s44_*' \
  -print | sort
```

## 可选：Random 对照

`random` 是 SEED 原始随机采样对照，不属于 AgentStream 的三种正式 stream 模式：

```bash
nohup env \
  bash examples/agentstream_trainer/run_agentstream_sft_glm_self.sh random \
  > logs/agentstream/stage3_random.log 2>&1 &
echo $! | tee logs/agentstream/stage3_random.pid
```

## RL_OPD 验收

对每个 Stage-3 日志执行以下检查。以 sequential 为例：

```bash
LOG=logs/agentstream/stage3_sequential.log

grep -E \
  'seed/analysis_enabled|seed/analysis_num_requests|seed/teacher_enabled|seed/teacher_available|seed/teacher_batch_size|seed/opd_loss_enabled|actor/opd_active_token_ratio|actor/opd_loss|training/global_step' \
  "$LOG" | tail -n 20
```

至少应确认：

- `seed/analysis_enabled:1`
- `seed/analysis_num_requests > 0`
- `seed/teacher_enabled:1`
- `seed/teacher_available:1`
- `seed/teacher_batch_size > 0`
- `seed/opd_loss_enabled:1`
- `actor/opd_active_token_ratio > 0`
- `actor/opd_loss` 是有限值
- `training/global_step` 最终达到配置的总步数（默认 sequential/interleaved 为 80，isolated 每个 benchmark 20）

检查致命异常：

```bash
grep -nE 'Error executing job|RayTaskError|Traceback|CUDA out of memory|OutOfMemoryError' "$LOG" | tail -n 30
```

Python 3.12 下进程退出时偶尔会出现以下 `multiprocess` warning：

```text
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
```

如果它只出现在训练完成后的 `Exception ignored in ResourceTracker.__del__` 中，并且已经打印最终 global step，则不属于训练失败。

## 停止任务

先向 nohup 主进程发送 `TERM`：

```bash
kill "$(cat logs/agentstream/stage3_sequential.pid)"
```

检查相关进程是否退出：

```bash
ps -fp "$(cat logs/agentstream/stage3_sequential.pid)"
pgrep -af 'verl.trainer.main_ppo|ray::|vllm serve'
```

不要直接使用 `kill -9`，除非普通 `TERM` 无法结束进程。Stage-1 总入口收到 `TERM` 时会通过 trap 停止它自己启动的本地 vLLM。

## 常用覆盖

临时增加训练规模时，把变量放在 `nohup env` 后面即可：

```bash
nohup env \
  AGENTSTREAM_NUM_TASKS=20 \
  AGENTSTREAM_RL_EPOCHS=10 \
  AGENTSTREAM_RL_SAVE_FREQ=5 \
  AGENTSTREAM_RL_TEST_FREQ=2 \
  bash examples/agentstream_trainer/run_agentstream_sft_glm_self.sh sequential \
  > logs/agentstream/stage3_sequential_ep10.log 2>&1 &
echo $! | tee logs/agentstream/stage3_sequential_ep10.pid
```

当前 sequential/interleaved checkpoint 尚未保存 `TaskStreamScheduler` 的 cursor/pass/RNG，因此默认保持 `AGENTSTREAM_RL_RESUME_MODE=disable`。在补齐 scheduler checkpoint 前，不建议用 resume 验证严格连续的任务流。

## 环境 straggler 超时

Stage 3 的每次向量化 reset/step 都会并发驱动全部 env worker（tau2 的 reset 还依赖外部 user-simulator API）。为避免单个卡死的环境阻塞整个 batch，默认启用以下超时；超时的 worker 会被强杀重建，该 slot 以零奖励错误 episode 降级：

```bash
AGENTSTREAM_RL_RESET_TIMEOUT=600   # 单批 reset 预算（秒），<=0 关闭
AGENTSTREAM_RL_STEP_TIMEOUT=600    # 单批 step 预算（秒），<=0 关闭
```

exgentic 侧的配套开关（通过环境变量传入训练进程即可，会自动透传给 benchmark 子进程）：

```bash
EXGENTIC_PROXY_OBSERVATION_TIMEOUT=600   # tau2 代理会合：等待下一条 observation
EXGENTIC_PROXY_ACTION_TIMEOUT=3600       # tau2 代理会合：等待下一条 action
EXGENTIC_VENV_TRANSPORT_TIMEOUT=1800     # venv 服务单次 RPC 上限
```

如果日志出现 `[agentstream] worker N unresponsive`，说明发生了一次 straggler 降级；偶发属正常（外部 API 抖动、端口竞态重试失败），频繁出现时应检查对应 benchmark 的 session 日志。

注意：benchmark venv 内安装的是 exgentic 的**拷贝**（非 editable）。修改 `AgentStream/exgentic` 源码后（包括上述 proxy 超时、health token），需要把新版 exgentic 重装进已存在的 venv 才会在 `exgentic serve` 子进程中生效：

```bash
for v in ~/.exgentic/benchmarks/*/venv; do
  uv pip install --python "$v/bin/python" --no-cache --reinstall "$AGENTSTREAM_EXGENTIC_ROOT"
done
```
