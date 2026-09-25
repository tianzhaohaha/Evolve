# AgentStream baselines on the SEED task stream

Runs the AgentStream self-evolving baselines (`reasoning_bank`, `ace`) and the no-learning
reference (`tool_calling`) on **exactly the SEED RL stream** — bfcl (multi_turn_base), tau2 (retail),
browsecompplus; 64 stream + 32 held-out tasks per benchmark; selection seed 42, order seed 44;
40 steps per episode; GLM user simulator / judge — so their curves are comparable with the RL runs.
Each run = an online pass (learning on, temperature 1.0) + a frozen holdout pass (learning off,
temperature 0.4). Metrics stream to wandb with the SEED names (`online/*` per task, `val/*`).

Files: `registry_ours.py` (benchmark settings + holdout split), `stream_ours.py` (runner: `online` /
`holdout`), `wandb_stream.py` (logger), `run_matrix_ours.sh` (matrix launcher), `serve_policy_vllm.sh`
(Qwen policy endpoint), `check_stream_alignment.py` (asserts the stream equals SEED's),
`summarize_matrix.py` (table). Agents: `learning_enabled` flag added to `ReasoningBankAgent` /
`ACEAgent` (frozen evaluation); everything else in `src/` is untouched.

## Setup (cluster)
Either a dedicated uv venv (`cd AgentStream/exgentic && uv sync --extra amem && uv pip install wandb`)
or the training conda env with the runner deps added; the second must keep verl's transformers pin:
```bash
conda activate seed-as
pip install "litellm>=1.65,<2,!=1.82.7,!=1.82.8" "sentence-transformers>=3,<5" "transformers<=4.57.3"
pip check   # only the pre-existing decord note may remain
export OPENROUTER_API_KEY=... WANDB_API_KEY=...  # GLM judges + API policies; wandb
export OPENAI_API_KEY=EMPTY                      # litellm needs a value for the local openai/ endpoint
```
`run_matrix_ours.sh` runs with `EXGENTIC_PYTHON` (default `uv run python`); the PBS script picks the
venv if present, else the active conda env. `EXGENTIC_LITELLM_CACHING=false` must be exported for the
runners (the PBS script does): exgentic's default litellm disk cache is an NFS SQLite that corrupts
under concurrent writers and replays sampled responses across runs.
Shared services (one node, 2 GPUs):
```bash
# GPU 1: browsecomp retriever (all runs share it)
DEVICE=1 PORT=60100 bash ../../SEED/examples/agentstream_trainer/serve_browsecomp_retriever.sh
# GPU 0: base policy for the qwen rows
MODEL_PATH=/path/LLMs/Qwen3-4B-Instruct-2507 GPU=0 PORT=8000 bash scripts/ours/serve_policy_vllm.sh
```

## Checks before the matrix
```bash
uv run python scripts/ours/check_stream_alignment.py             # task order + holdout == SEED stream
NUM_TASKS=2 NUM_HOLDOUT=1 ONLY=tool_calling:qwen:interleaved bash scripts/ours/run_matrix_ours.sh   # smoke
```

## Matrix
```bash
bash scripts/ours/run_matrix_ours.sh                        # 3 agents x {qwen,luna,dsflash} x {interleaved,isolated}
MODELS=qwen bash scripts/ours/run_matrix_ours.sh            # local rows only
ONLY=ace:dsflash:isolated bash scripts/ours/run_matrix_ours.sh
python scripts/ours/summarize_matrix.py outputs_ours --csv matrix.csv
```
Runs are independent; `PARALLEL` (default 6) bounds concurrency — keep it at or below 8 for the
shared retriever and the OpenRouter / GLM rate limits. A killed run resumes at its next task
(`progress.json`, store checkpoints, same wandb run id).

## Notes
- wandb run name `as_<agent>_<model>_<mode>_s44`, project `agentic_agentstream`; x-axis
  `online/task_index` (1..192), `online/stream_step` = task_index // 10 + 1 aligns with the RL steps.
- Protocol differs from the RL runs (native tool calling vs the SEED prompt template; 1 attempt per
  task vs 6): compare methods to their own-model `tool_calling` reference, and to the RL runs'
  `online/single/*` columns.
- `--reasoning-effort low` is passed to the API models; override with `EXTRA_ARGS_LUNA` /
  `EXTRA_ARGS_DSFLASH` if a provider rejects it.
