#!/usr/bin/env bash

# Internal implementation shared by the public AgentStream launchers.
# Adapted from examples/seed_trainer/_common/alfworld.sh; the SEED algorithm
# surface is identical — only the environment block changes.
#
# Required environment:
#   MODEL_PATH                     policy checkpoint (HF format)
#   AGENTSTREAM_EXGENTIC_ROOT      path to AgentStream/exgentic checkout
# Stream controls (all optional, sensible defaults below):
#   AS_BENCHMARKS   comma list, e.g. "bfcl,tau2,appworld"
#   AS_STREAM_MODE  random | isolated | sequential | interleaved
#   AS_PROTOCOL     split | online
#   AS_STREAM_SEED  AgentStream run seed (ordering; selection fixed at 42)
#   AS_NUM_TASKS    tasks per benchmark in the stream (AgentStream NUM_TASKS)
#   AS_VAL_TASKS    held-out validation tasks per benchmark
#   AS_VAL_SOURCE   online protocol validation source: holdout | stream | both
#   AS_BLOCK_PASSES sequential/isolated block repetition factor

set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_ENV="${CONDA_ENV:-seed}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

cd "$PROJECT_ROOT"

ENGINE=vllm

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

MODELS_ROOT=${MODELS_ROOT:-}
if [[ -z "${MODEL_PATH:-}" ]]; then
    : "${MODELS_ROOT:?Please set MODEL_PATH through a public launcher, or set MODELS_ROOT}"
    MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct"
fi

# Thinking stays disabled pipeline-wide. agentstream_full.env normally derives
# this switch; when the launcher is invoked without it, fall back to the same
# model-name rule so hybrid-thinking Qwen3 checkpoints never roll out with
# native thinking silently enabled.
if [[ -z "${AGENTSTREAM_DISABLE_THINKING:-}" ]]; then
    case "$(basename "$MODEL_PATH")" in
        Qwen3-*-2507*) AGENTSTREAM_DISABLE_THINKING=false ;;
        Qwen3-*)       AGENTSTREAM_DISABLE_THINKING=true ;;
        *)             AGENTSTREAM_DISABLE_THINKING=false ;;
    esac
fi
: "${AGENTSTREAM_EXGENTIC_ROOT:?Please set AGENTSTREAM_EXGENTIC_ROOT to the AgentStream/exgentic checkout}"
export AGENTSTREAM_EXGENTIC_ROOT

# Stream configuration.
AS_BENCHMARKS=${AS_BENCHMARKS:-bfcl}
AS_STREAM_MODE=${AS_STREAM_MODE:-random}
AS_PROTOCOL=${AS_PROTOCOL:-split}
AS_STREAM_SEED=${AS_STREAM_SEED:-44}
AS_NUM_TASKS=${AS_NUM_TASKS:-50}
AS_VAL_TASKS=${AS_VAL_TASKS:-16}
AS_VAL_SOURCE=${AS_VAL_SOURCE:-holdout}
AS_BLOCK_PASSES=${AS_BLOCK_PASSES:-1}
AS_ON_EXHAUSTED=${AS_ON_EXHAUSTED:-cycle}
# Multi-pass streams: additionally record every repeat pass K as its own
# cumulative wandb subtree online/pass<K>/* (first-pass online/* unaffected).
AS_TRACK_REPEAT_PASSES=${AS_TRACK_REPEAT_PASSES:-false}
if [[ -z "${AS_BENCHMARK_KWARGS_JSON:-}" ]]; then
    AS_BENCHMARK_KWARGS_JSON='{}'
fi
AS_BENCHMARK_KWARGS_HYDRA=$(python3 -c '
import json
import re
import sys

def render(value):
    if isinstance(value, dict):
        fields = []
        for key, item in value.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", str(key)):
                raise ValueError(f"Unsupported Hydra mapping key: {key!r}")
            fields.append(f"{key}:{render(item)}")
        return "{" + ",".join(fields) + "}"
    if isinstance(value, list):
        return "[" + ",".join(render(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=True)

print(render(json.loads(sys.argv[1])))
' "$AS_BENCHMARK_KWARGS_JSON")
AS_RUNNER=${AS_RUNNER:-venv}
AS_MAX_STEPS=${AS_MAX_STEPS:-30}
# Straggler handling: env workers missing these budgets are killed + replaced
# and their slot becomes a zero-reward error episode (<=0 disables).
AS_RESET_TIMEOUT=${AS_RESET_TIMEOUT:-600}
AS_STEP_TIMEOUT=${AS_STEP_TIMEOUT:-600}

# Rollout scale. AgentStream sessions are heavier than ALFWorld games:
# default to a smaller parallel width; tune per benchmark tier.
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-8}
# Validation slots: every holdout task once per repeat. ValTaskCycler serves the
# fixed holdout set round-robin, so slots = benchmarks x AS_VAL_TASKS x AS_VAL_REPEATS
# gives each task AS_VAL_REPEATS attempts (mean@k). Do NOT use
# actor_rollout_ref.rollout.val_kwargs.n for this: verl-agent sizes validation
# envs as val_batch_size x 1 and asserts batch == env count.
AS_VAL_REPEATS=${AS_VAL_REPEATS:-1}
_as_n_bench=$(awk -F',' '{print NF}' <<< "$AS_BENCHMARKS")
VAL_DATA_SIZE=${VAL_DATA_SIZE:-$(( _as_n_bench * AS_VAL_TASKS * AS_VAL_REPEATS ))}
GROUP_SIZE=${GROUP_SIZE:-8}
POLICY_ROLLOUT_N=${POLICY_ROLLOUT_N:-1}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-32}
# False skips the full-vocab entropy pass inside compute_log_prob (old-log-prob
# recompute + OPD teacher scoring). Saves a large transient memory spike on long
# micro-batches; only cost is losing the actor/entropy_loss metric. Gradients
# are unaffected (update-phase entropy is gated by ENTROPY_COEFF separately).
LOG_PROB_CALCULATE_ENTROPY=${LOG_PROB_CALCULATE_ENTROPY:-True}
# Token-budget micro-batching for mixed-length domains (bfcl ~6k vs appworld
# ~20k prompt tokens): fixed micro batches must be sized for the worst case,
# dynamic batching packs by real token count instead. When enabled the fixed
# PPO/LOG_PROB micro batch sizes above are ignored; log_prob and ref paths
# follow via ppo_trainer.yaml interpolation. Incompatible only with the SP/ID
# env aux loss (unused here); the OPD teacher tensors travel with the batch.
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-False}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-160}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-5}
# VAL_BEFORE_TRAIN=True logs the step-0 baseline on the same holdout set.
# ACTOR_LR=0 gives a frozen-policy control run over the same stream.
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
ACTOR_LR=${ACTOR_LR:-1e-6}
# Entropy bonus coefficient (ppo_trainer.yaml default 0.001). Non-zero makes the
# update pass materialize full-vocab fp32 softmax/logsumexp for entropy (~16 GiB
# per 29k-token sample with a 152k vocab) and disables in-place backward for the
# log-prob cross entropy; 0 skips that path. actor/entropy_loss stays logged
# either way (it comes from the no-grad old-log-prob pass).
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}
RL_RESUME_MODE=${RL_RESUME_MODE:-auto}
RL_RESUME_FROM_PATH=${RL_RESUME_FROM_PATH:-null}
WANDB_ROLLOUT_SAMPLES=${WANDB_ROLLOUT_SAMPLES:-3}
WANDB_ROLLOUT_SAMPLE_FREQ=${WANDB_ROLLOUT_SAMPLE_FREQ:-10}
WANDB_ROLLOUT_SAMPLE_MAX_CHARS=${WANDB_ROLLOUT_SAMPLE_MAX_CHARS:-4000}

# Long tool-schema prompts need a wider context than ALFWorld.
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

# SEED advantage and OPD teacher/OPD loss schedule (same defaults as alfworld).
SEED_MODE=${SEED_MODE:-mean_std_norm}
SEED_STEP_ADV_W=${SEED_STEP_ADV_W:-0.0}
SEED_EPISODE_SKILL_TEACHER_ADV_W=${SEED_EPISODE_SKILL_TEACHER_ADV_W:-0.0}
SEED_STEP_SKILL_TEACHER_ADV_W=${SEED_STEP_SKILL_TEACHER_ADV_W:-0.0}
SEED_SKILL_MODE=${SEED_SKILL_MODE:-episode_step}
SEED_SKILL_TEACHER_MODE=${SEED_SKILL_TEACHER_MODE:-step_priority}
SEED_OPD_START_AFTER_STEPS=${SEED_OPD_START_AFTER_STEPS:-null}
SEED_OPD_STOP_AFTER_STEPS=${SEED_OPD_STOP_AFTER_STEPS:-null}
SEED_OPD_LOSS_COEF=${SEED_OPD_LOSS_COEF:-0.01}
SEED_OPD_GATE_BETA=${SEED_OPD_GATE_BETA:-5.0}
SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU=${SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU:-${SEED_SKILL_GEN_MICRO_BATCH_SIZE:-1}}
SEED_SKILL_GEN_MAX_SAMPLES=${SEED_SKILL_GEN_MAX_SAMPLES:-all}
SEED_SKILL_GEN_VALID_JSON_BONUS=${SEED_SKILL_GEN_VALID_JSON_BONUS:-0.0}
SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS=${SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS:-0.0}
SEED_SKILL_GEN_TOO_LONG_PENALTY=${SEED_SKILL_GEN_TOO_LONG_PENALTY:-0.0}
SEED_SKILL_GEN_MAX_OUTPUT_CHARS=${SEED_SKILL_GEN_MAX_OUTPUT_CHARS:-1200}
SEED_SKILL_GEN_REWARD_CLIP=${SEED_SKILL_GEN_REWARD_CLIP:-2.0}
SEED_SKILL_GEN_FAILED_REWARD_MODE=${SEED_SKILL_GEN_FAILED_REWARD_MODE:-zero}

SEED_FAILED_ONLY=${SEED_FAILED_ONLY:-False}
SEED_FAILED_ONLY_AFTER_STEPS=${SEED_FAILED_ONLY_AFTER_STEPS:-null}
SEED_FAILURE_SUCCESS_THRESHOLD=${SEED_FAILURE_SUCCESS_THRESHOLD:-1.0}

SEED_ENABLE_ANALYSIS=${SEED_ENABLE_ANALYSIS:-True}
SEED_SELECTOR=${SEED_SELECTOR:-llm}
SEED_ANALYSIS_BACKEND=${SEED_ANALYSIS_BACKEND:-policy_vllm}
SEED_ANALYSIS_NUM_WORKERS=${SEED_ANALYSIS_NUM_WORKERS:-1}
SEED_ANALYSIS_CONTEXT_LENGTH=${SEED_ANALYSIS_CONTEXT_LENGTH:-16384}
SEED_ANALYSIS_MAX_COMPLETION_TOKENS=${SEED_ANALYSIS_MAX_COMPLETION_TOKENS:-4096}
SEED_ANALYSIS_MAX_MODEL_LEN=${SEED_ANALYSIS_MAX_MODEL_LEN:-20480}
SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ=${SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ:-5}
SEED_ANALYSIS_PROMPT_VERSION=${SEED_ANALYSIS_PROMPT_VERSION:-seed}
SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY=${SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY:-True}

PROJECT_NAME=${PROJECT_NAME:-agentic_agentstream}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-seed_agentstream_${AS_STREAM_MODE}_${AS_PROTOCOL}_s${AS_STREAM_SEED}}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${CHECKPOINTS_ROOT:-$MODELS_ROOT/ckpt}/$EXPERIMENT_NAME}

history_length=${history_length:-5}

real_train_batch_size=$((TRAIN_DATA_SIZE * POLICY_ROLLOUT_N))
if (( real_train_batch_size % N_GPUS_PER_NODE != 0 )); then
    echo "TRAIN_DATA_SIZE * POLICY_ROLLOUT_N ($real_train_batch_size) must be divisible by N_GPUS_PER_NODE ($N_GPUS_PER_NODE)." >&2
    exit 1
fi

python3 -m examples.data_preprocess.prepare \
    --mode text \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

# Hybrid-thinking policies (Qwen3 non-2507) roll out with thinking disabled;
# env_manager then auto-relaxes projection require_think for qwen3 models.
if [[ "${AGENTSTREAM_DISABLE_THINKING:-false}" == "true" ]]; then
    set -- "$@" "+data.apply_chat_template_kwargs.enable_thinking=false"
fi

# Action-hardness switches from agentstream_full.env (default off = strict).
if [[ "${AGENTSTREAM_RELAX_REQUIRE_THINK:-false}" == "true" ]]; then
    set -- "$@" "+env.projection_require_think=false"
fi
if [[ "${AGENTSTREAM_ACCEPT_TOOL_CALL:-false}" == "true" ]]; then
    set -- "$@" "+env.projection_accept_tool_call=true"
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=seed \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.opd_loss_coef=$SEED_OPD_LOSS_COEF \
    actor_rollout_ref.actor.opd_gate_beta=$SEED_OPD_GATE_BETA \
    actor_rollout_ref.actor.skill_gen_micro_batch_size_per_gpu=$SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.log_prob_calculate_entropy=$LOG_PROB_CALCULATE_ENTROPY \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.n=$POLICY_ROLLOUT_N \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=$SEED_ANALYSIS_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$SEED_ANALYSIS_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.seed.step_advantage_w=$SEED_STEP_ADV_W \
    algorithm.seed.episode_skill_teacher_advantage_w=$SEED_EPISODE_SKILL_TEACHER_ADV_W \
    algorithm.seed.step_skill_teacher_advantage_w=$SEED_STEP_SKILL_TEACHER_ADV_W \
    algorithm.seed.skill_mode=$SEED_SKILL_MODE \
    algorithm.seed.skill_teacher_mode=$SEED_SKILL_TEACHER_MODE \
    algorithm.seed.opd_start_after_steps=$SEED_OPD_START_AFTER_STEPS \
    algorithm.seed.opd_stop_after_steps=$SEED_OPD_STOP_AFTER_STEPS \
    algorithm.seed.failed_only=$SEED_FAILED_ONLY \
    algorithm.seed.failed_only_after_steps=$SEED_FAILED_ONLY_AFTER_STEPS \
    algorithm.seed.failure_success_threshold=$SEED_FAILURE_SUCCESS_THRESHOLD \
    algorithm.seed.mode=$SEED_MODE \
    algorithm.seed.enable_analysis=$SEED_ENABLE_ANALYSIS \
    algorithm.seed.selector=$SEED_SELECTOR \
    algorithm.seed.analysis_backend=$SEED_ANALYSIS_BACKEND \
    algorithm.seed.analysis_num_workers=$SEED_ANALYSIS_NUM_WORKERS \
    algorithm.seed.analysis_context_length=$SEED_ANALYSIS_CONTEXT_LENGTH \
    algorithm.seed.analysis_max_completion_tokens=$SEED_ANALYSIS_MAX_COMPLETION_TOKENS \
    algorithm.seed.analysis_max_step_skills_per_traj=$SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ \
    algorithm.seed.analysis_prompt_version=$SEED_ANALYSIS_PROMPT_VERSION \
    algorithm.seed.analysis_include_episode_summary=$SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY \
    algorithm.seed.skill_gen.max_samples=$SEED_SKILL_GEN_MAX_SAMPLES \
    algorithm.seed.skill_gen.valid_json_bonus=$SEED_SKILL_GEN_VALID_JSON_BONUS \
    algorithm.seed.skill_gen.non_empty_skill_bonus=$SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS \
    algorithm.seed.skill_gen.too_long_penalty=$SEED_SKILL_GEN_TOO_LONG_PENALTY \
    algorithm.seed.skill_gen.max_output_chars=$SEED_SKILL_GEN_MAX_OUTPUT_CHARS \
    algorithm.seed.skill_gen.reward_clip=$SEED_SKILL_GEN_REWARD_CLIP \
    algorithm.seed.skill_gen.failed_reward_mode=$SEED_SKILL_GEN_FAILED_REWARD_MODE \
    algorithm.seed.normalize_teacher_adv=False \
    env.history_length=$history_length \
    env.env_name=agentstream/mixed \
    env.seed=0 \
    env.max_steps=$AS_MAX_STEPS \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    env.agentstream.exgentic_root=$AGENTSTREAM_EXGENTIC_ROOT \
    env.agentstream.runner=$AS_RUNNER \
    "env.agentstream.benchmarks=[$AS_BENCHMARKS]" \
    "+env.agentstream.benchmark_kwargs=$AS_BENCHMARK_KWARGS_HYDRA" \
    env.agentstream.stream_mode=$AS_STREAM_MODE \
    env.agentstream.stream_seed=$AS_STREAM_SEED \
    env.agentstream.num_tasks_per_benchmark=$AS_NUM_TASKS \
    env.agentstream.protocol=$AS_PROTOCOL \
    env.agentstream.val_tasks_per_benchmark=$AS_VAL_TASKS \
    env.agentstream.val_source=$AS_VAL_SOURCE \
    env.agentstream.block_passes=$AS_BLOCK_PASSES \
    env.agentstream.on_exhausted=$AS_ON_EXHAUSTED \
    env.agentstream.online_track_repeat_passes=$AS_TRACK_REPEAT_PASSES \
    env.agentstream.reset_timeout_s=$AS_RESET_TIMEOUT \
    env.agentstream.step_timeout_s=$AS_STEP_TIMEOUT \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.resume_mode=$RL_RESUME_MODE \
    trainer.resume_from_path=$RL_RESUME_FROM_PATH \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.log_rollout_samples=$WANDB_ROLLOUT_SAMPLES \
    trainer.log_rollout_samples_freq=$WANDB_ROLLOUT_SAMPLE_FREQ \
    trainer.log_rollout_samples_max_chars=$WANDB_ROLLOUT_SAMPLE_MAX_CHARS \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    "$@"
