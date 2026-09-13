<h1 align="center">
<img src="figs/seed_icon.svg" alt="SEED icon" width="42" style="vertical-align:middle;">
SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning
</h1>

<p align="center">
  <a href="https://jinyangwu.github.io/seed/">
    <img src="https://img.shields.io/badge/Project-Page-1F6FEB?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://huggingface.co/papers/2607.14777">
    <img src="https://img.shields.io/badge/HF-Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=111827" alt="Hugging Face Paper">
  </a>
  <a href="https://arxiv.org/abs/2607.14777">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
  <a href="https://huggingface.co/Jinyang23/Seed-AlfWorld-3B">
    <img src="https://img.shields.io/badge/Model-Checkpoint-F59E0B?style=for-the-badge&logo=huggingface&logoColor=white" alt="Model Checkpoint">
  </a>
</p>

## News

- **2026-07-16**: We have released our paper and code.

If you have any questions ❓ or are interested in collaboration 🤝, please feel free to contact me at 
wu-jy23@mails.tsinghua.edu.cn.

## Overview

**SEED** is a **Self-Evolving On-Policy Distillation** framework for long-horizon
LLM agents. Its core idea is to make one policy checkpoint play two synchronized
roles: it acts in the environment to collect on-policy trajectories, then analyzes
its own completed trajectories into hindsight skills, and finally distills the
skill-induced change in action probabilities back into the ordinary policy. After
each update, the improved policy becomes the next analyzer and generates the next
round of hindsight skills, so the decision policy and hindsight supervision
evolve together.

SEED has two stages:

1. **Hindsight-skill SFT.** Collect ordinary agent trajectories, annotate each
   completed trajectory with an episode-level hindsight skill, and fine-tune the
   backbone so the same model can analyze trajectories.
2. **Self-evolving OPD during RL.** At each RL update, the frozen current policy
   both samples on-policy trajectories and serves as the synchronized analyzer
   that extracts hindsight skills. The same sampled action tokens are re-scored
   under ordinary and skill-augmented contexts. The skill-induced log-probability
   shift gates a dense OPD loss, which is optimized jointly with GRPO.

At inference time, SEED uses only the learned policy. It requires no analyzer,
no skill bank, no retrieval module, and no skill-augmented prompt at deployment.

<div align="center">
  <img src="figs/pipeline.png" alt="SEED pipeline" style="width:100%;">
  <br>
  <em>Figure 1: Overview of SEED.</em>
</div>

## Main Results

Across ALFWorld, Search-based QA, and WebShop, the experiments show three main
findings:

1. **Dense hindsight supervision improves outcome-only RL.** SEED consistently
   outperforms GRPO by converting trajectory-level hindsight into token-level
   OPD signals.
2. **Internalizing skills is better than prompting with skills.** SEED uses
   hindsight skills only during training and still outperforms skill-prompted
   evaluation baselines.
3. **Self-evolving distillation beats static distillation.** Refreshing the
   analyzer from the latest policy keeps hindsight supervision aligned with the
   policy's evolving behaviors and failure modes.

<div align="center">
  <img src="figs/results.png" alt="SEED results" style="width:100%;">
  <br>
  <em>Figure 2: Main results.</em>
</div>

## Installation

### Install veRL

```bash
conda create -n seed python==3.12 -y
conda activate seed

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

### Install Supported Environments

#### 1. ALFWorld

```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
```
Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in ~/.cache/alfworld/):
```bash
alfworld-download -f
```
#### 2. WebShop

WebShop requires Python <= 3.10, so begin by creating a separate environment:

```bash
conda create -n seed-webshop python==3.10 -y
conda activate seed-webshop

cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```


#### 3. Search-Based QA

```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare the Search-R1 style dataset:

```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```

The processed data is saved under `~/data/searchR1_processed_direct` by default.

Build a separate retrieval environment for the local search server:

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

Download the index:

```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server:

```bash
conda activate retriever

bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log
```

## Training

The training and data-construction scripts load environment variables from the
repo-level `.env` file by default. Put model paths, analyzer endpoints, and
optional logging keys there before running the scripts:

```bash
# Runtime devices and logging.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WANDB_MODE=offline

# Local model and data roots.
MODELS_ROOT=/path/to/models # the parent dir of models
DATA_ROOT=/path/to/data-root # used for search-qa data

# Used by Stage 1 hindsight-skill annotation.
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=glm-5.2
OPENAI_API_RETRIES=5
OPENAI_API_RETRY_DELAY=1.0
```



### Stage 1: Build Hindsight-Skill SFT Checkpoints

The paper-style workflow first builds episode-level skill SFT data and trains an
analyzer-capable policy checkpoint.

```bash
# ALFWorld
bash scripts/sft/alfworld/prepare_data.sh
bash scripts/sft/alfworld/train_sft.sh

# WebShop
bash scripts/sft/webshop/prepare_data.sh
bash scripts/sft/webshop/train_sft.sh

# Search-based QA
bash scripts/sft/search/prepare_data.sh
bash scripts/sft/search/train_sft.sh

# EZPoints
bash scripts/sft/ezpoints/prepare_data.sh
bash scripts/sft/ezpoints/train_sft.sh

# Sokoban
bash scripts/sft/sokoban/prepare_data.sh
bash scripts/sft/sokoban/train_sft.sh
```

The default scale follows the paper: 180 tasks and 8 rollouts per task. The SFT
scripts train for 3 epochs and export Hugging Face checkpoints under
`$MODELS_ROOT`. All five datasets use the same
`prepare_data.sh` / `train_sft.sh` interface; see
[`scripts/sft/README.md`](scripts/sft/README.md) for defaults and overrides.

### Stage 2: Run Self-Evolving OPD RL

All SEED RL scripts live under `examples/seed_trainer/` and assume the repo root
as the working directory.

```bash
bash examples/seed_trainer/run_alfworld_sft_glm_self.sh
bash examples/seed_trainer/run_webshop_sft_glm_self.sh
bash examples/seed_trainer/run_search_sft_glm_self.sh
```

Other teacher-self SFT entrypoints use the same naming convention:

```bash
bash examples/seed_trainer/run_alfworld_sft_qwen_self.sh
bash examples/seed_trainer/run_ezpoints_sft_gemini_self.sh
bash examples/seed_trainer/run_sokoban_sft_gemini_self.sh
```




## 可选改进开关（AgentStream / global pool，默认全部关闭）

三个开关都在 `examples/agentstream_trainer/agentstream_full.env`（§4 末尾），经
`run_agentstream_sft_glm_self.sh` → `_common/agentstream.sh` → hydra 传入；关闭时训练逻辑与
原始实现逐 bit 一致。共同背景：OPD 损失 `loss = gate·(teacher_lp − student_lp)`，
`gate = sigmoid(β·(teacher_lp − student_lp))`，其梯度对每个 token 都是 `−gate`，只会推高学生
已采样的 token，从不推低。三个开关分别处理由此产生的三个问题。

| 开关（env 变量 → hydra 键） | 默认 | 开启后的行为 |
|---|---|---|
| `AGENTSTREAM_SEED_FAILED_SKILL_POSITIVE` → `algorithm.seed.failed_skill_positive` | False | 失败轨迹的 episode_skill 从 avoidance 规则改为"本应遵循的规则"（正向 workflow），spec 与 gen 两通道同时可用；global pool 准入对失败候选不再做 spec_gap 门控，固定排在所有成功候选之后，judge 是唯一过滤器，per-step 上限先截失败候选。池侧效果依赖 `AGENTSTREAM_SEED_GLOBAL_POOL_ADMIT_FAILED=True`（full env 默认 True，`agentstream.sh` 单独默认 False）：为 False 时失败候选在收集阶段即被丢弃，开关只改 prompt，启动时会打 warning |
| `AGENTSTREAM_SEED_GLOBAL_POOL_EVICT_POLICY` → `algorithm.seed.global_pool.evict_policy`（配 `..._WINDOW_STEPS` → `window_steps`） | gate_ema / 48 | 满员淘汰策略。`gate_ema`（原始）按 gate EMA 最低淘汰；`lru` 淘汰最久未被检索者；`window` 每步删除入池早于 `当前步 − window_steps` 的条目（检索不续命），满员时按入池步 FIFO |
| `AGENTSTREAM_SEED_OPD_POSITIVE_ONLY` → `actor_rollout_ref.actor.opd_positive_only` | False | spec 与 gen 两通道只在 `teacher_lp − student_lp > 0` 的 token 上施加损失；token-mean 分母与全部 `actor/opd_*` 指标仍按原 mask 统计，系数语义和曲线口径不变，`opd_loss` 变为非负；可与 `opd_gate_eps` 叠加 |

为什么需要它们：

- **失败 skill 的正向措辞**。avoidance 措辞的信号落在需要推低的 token 上，单边损失下 gate 归零，
  失败轨迹只剩 gap≈0 的自我模仿；上游 SEED 的 avoidance prompt 只在 teacher-advantage 模式
  （`opd_loss_coef=0`，带符号 gap 直接进 advantage）下语义成立，而上游与本仓库默认都跑 OPD 损失模式。
  正向措辞让失败轨迹里做对的部分获得正 gap；准入随之放行是必要配套，否则正向 skill 在失败轨迹上
  gap 为正会被原有的 `-spec_gap>0` 规则全部丢弃。若启用 skill_gen 且 `failed_reward_mode=negate`，
  与本开关矛盾，需改回 `zero`。
- **淘汰策略**。gate EMA 是一致性量而非有用性量，实际值聚在 0.5 略下，原始策略会先淘汰被检索过的
  活跃条目、保留从未命中的死重。`lru` 直接针对死重；`window` 是近因先验，会清掉跨域旧 skill，
  建议作为命名消融而非默认。
- **正 gap 限定**。gap<0 的 token 仍以 gate∈(0,0.5) 的权重被推高，方向与 teacher 相反，等价于对
  自己 rollout 的半权重 BC（失败轨迹的错误动作也在内）。置零后 `opd_loss` 的下降只剩"正 gap token
  被学会"一种解释，无关的 pool 命中梯度趋近零，使放宽准入没有下行风险。

观测指标：`seed/global_pool/never_injected_ratio`（从未被检索条目占比）、`evicted_total`（累计满员
淘汰数）、`expired`（window 每步过期条数）；`actor/opd_loss`、`actor/opd_teacher_gap_mean`、
`actor/opd_gate_active_ratio` 及 gen 通道对应项口径不变。建议启用顺序：先 `OPD_POSITIVE_ONLY`，
再 `FAILED_SKILL_POSITIVE`，最后比较 `evict_policy`。代码落点：`verl/trainer/ppo/core_algos.py`
（`compute_opd_loss(positive_only)`）、`seed/analysis.py`（failure 分支）、`seed/global_pool.py`
（`select_admission_candidates` / `_evict_locked` / `expire`）、`verl/trainer/ppo/ray_trainer.py`
（开关装配与 `expire` 调用）；测试见 `tests/trainer/ppo/test_{opd_loss,global_skill_pool,seed_analyzer}.py`。
更多池机制见 [GLOBAL_SKILL_POOL_V1.md](GLOBAL_SKILL_POOL_V1.md)。

## Merge Checkpoints

See `scripts/model_merger.py` for FSDP/Megatron merge examples using paths under
`./checkpoints/...`.

## ⭐ Citation

If you find this project useful, welcome to cite us.

```bibtex
@article{wu2026seed,
  title={SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning},
  author={Wu, Jinyang and Yang, Shuo and Lu, Zhengxi and Zhang, Fan and
          Shen, Yuhao and Feng, Lang and Luo, Haoran and Lian, Zheng and
          Zhang, Shuai and Wen, Zhengqi and Tao, Jianhua},
  journal={arXiv preprint arXiv:2607.14777},
  year={2026}
}
```

## Acknowledgement

This project builds on
[veRL](https://github.com/volcengine/verl),
[verl-agent](https://github.com/langfengQ/verl-agent),
[SDAR](https://github.com/ZJU-REAL/SDAR),
and
[OPID](https://github.com/jinyangwu/OPID). We thank the authors of those
projects.
