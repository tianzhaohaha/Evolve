# Copyright 2026 SEED x AgentStream integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed view over the ``env.agentstream`` config block.

This module is self-contained: it only reads the OmegaConf/dict config that
the trainer already carries and normalizes it into a plain dataclass, so the
rest of the package never touches Hydra/OmegaConf directly.

Design goals (kept in sync with the integration plan):
  * ``protocol='split'``  -> SEED-style train/eval split. Train stream tasks
    and the held-out validation tasks are disjoint.
  * ``protocol='online'`` -> AgentStream-style online protocol. The stream
    itself is the evaluation; online per-episode metrics are recorded and
    validation re-visits the stream task set (retention) and/or a holdout.
  * ``stream_mode`` in {'random', 'isolated', 'sequential', 'interleaved'}.
    'random' replicates SEED's existing per-reset random sampling so results
    stay comparable with the original SEED data-processing logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

VALID_STREAM_MODES = ("random", "isolated", "sequential", "interleaved")
VALID_PROTOCOLS = ("split", "online")
VALID_VAL_SOURCES = ("holdout", "stream", "both")
VALID_ON_EXHAUSTED = ("cycle", "stop")

DEFAULT_BENCHMARK_KWARGS: Dict[str, Dict[str, Any]] = {
    # Mirrors AgentStream/exgentic/scripts/*/run_experiment.py BENCHMARK_REGISTRY
    # (subset choices), minus API-model specific fields which the user overrides
    # via env.agentstream.benchmark_kwargs.
    "bfcl": {},
    "tau2": {"subset": "telecom"},
    "appworld": {},
    "hle": {},
    "browsecompplus": {},
    "swebench": {},
}


def _select(cfg: Any, key: str, default: Any = None) -> Any:
    """Best-effort nested select working for OmegaConf nodes and dicts."""
    current = cfg
    for part in key.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue
        if not hasattr(current, part):
            return default
        current = getattr(current, part)
    return default if current is None else current


def _as_container(value: Any) -> Any:
    """Convert OmegaConf nodes to plain python containers when possible."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


@dataclass
class AgentStreamConfig:
    """Normalized ``env.agentstream`` settings."""

    # --- exgentic bootstrap -------------------------------------------------
    # Root of the AgentStream checkout; ``<root>/src`` is put on sys.path so
    # the host-side (lightweight) exgentic package is importable without
    # installing anything into the SEED conda env.
    exgentic_root: str = ""
    # Default runner used to isolate benchmark deps (venv per README install).
    runner: str = "venv"
    # Output dir used by exgentic sessions (trajectories, per-session results).
    exgentic_output_dir: str = "outputs/agentstream_sessions"

    # --- stream composition -------------------------------------------------
    benchmarks: List[str] = field(default_factory=lambda: ["bfcl"])
    benchmark_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stream_mode: str = "random"  # random | isolated | sequential | interleaved
    stream_seed: int = 44  # ordering seed (AgentStream's SEED)
    num_tasks_per_benchmark: int = 50  # AgentStream default (NUM_TASKS)

    # --- protocol -----------------------------------------------------------
    protocol: str = "split"  # split | online
    # split protocol: number of held-out val tasks per benchmark, drawn from
    # the seed-42 shuffled remainder (disjoint from the train stream set).
    val_tasks_per_benchmark: int = 16
    # online protocol: where validation tasks come from.
    val_source: str = "holdout"  # holdout | stream | both
    # What to do when the (mode-ordered) stream is exhausted:
    #   cycle -> start the next pass over the same ordered stream (default;
    #            verl trainers run for trainer.total_epochs steps regardless).
    #   stop  -> keep returning the last window (training effectively plateaus;
    #            use together with a matching trainer.total_epochs).
    on_exhausted: str = "cycle"
    # sequential mode: repeat each benchmark block this many times before
    # advancing to the next block (RL usually needs multiple passes per block).
    block_passes: int = 1

    # --- episode shaping ------------------------------------------------------
    max_steps: Optional[int] = None  # falls back to env.max_steps
    reward_success: float = 10.0  # same scale as ALFWorld/AppWorld in SEED
    # If true, add the raw benchmark score (0..1) on top of the success bonus.
    reward_use_score: bool = False
    # Penalty applied by projection when the action cannot be parsed. The
    # trainer already supports invalid-action penalties; this stays 0 here.
    format_penalty: float = 0.0

    # --- robustness -----------------------------------------------------------
    # Wall-clock budget for one vectorized reset()/step() fan-out over all env
    # workers. A worker missing the deadline is killed and recreated and its
    # slot degrades to a zero-reward error episode instead of blocking the
    # whole batch on ray.get (<= 0 restores the unbounded wait).
    reset_timeout_s: float = 600.0
    step_timeout_s: float = 600.0

    # --- online metrics -------------------------------------------------------
    online_metrics_enable: bool = True
    online_metrics_path: str = ""  # defaults to <trainer.default_local_dir>/agentstream_online_metrics.jsonl

    # --- validation stream ------------------------------------------------------
    # Validation always evaluates a fixed per-benchmark task set (cycled
    # deterministically), so per-benchmark success curves over training steps
    # directly yield forgetting / transfer measurements at trainer.test_freq.
    val_stream_seed: int = 1042

    def resolved_benchmark_kwargs(self, slug: str) -> Dict[str, Any]:
        merged = dict(DEFAULT_BENCHMARK_KWARGS.get(slug, {}))
        merged.update(self.benchmark_kwargs.get(slug, {}) or {})
        return merged


def parse_agentstream_config(config: Any) -> AgentStreamConfig:
    """Build an :class:`AgentStreamConfig` from the trainer config object."""
    node = _select(config, "env.agentstream")
    node = _as_container(node) or {}
    if not isinstance(node, dict):
        raise ValueError(f"env.agentstream must be a mapping, got: {type(node)}")

    cfg = AgentStreamConfig()

    def pick(key: str, default: Any) -> Any:
        value = node.get(key, default)
        return default if value is None else value

    cfg.exgentic_root = os.path.expanduser(str(pick("exgentic_root", cfg.exgentic_root)))
    cfg.runner = str(pick("runner", cfg.runner))
    cfg.exgentic_output_dir = str(pick("exgentic_output_dir", cfg.exgentic_output_dir))

    benchmarks = pick("benchmarks", cfg.benchmarks)
    if isinstance(benchmarks, str):
        benchmarks = [s.strip() for s in benchmarks.split(",") if s.strip()]
    cfg.benchmarks = sorted(str(b) for b in benchmarks)

    benchmark_kwargs = pick("benchmark_kwargs", {})
    if not isinstance(benchmark_kwargs, dict):
        raise ValueError("env.agentstream.benchmark_kwargs must be a mapping")
    cfg.benchmark_kwargs = {str(k): dict(v or {}) for k, v in benchmark_kwargs.items()}

    cfg.stream_mode = str(pick("stream_mode", cfg.stream_mode)).lower()
    cfg.stream_seed = int(pick("stream_seed", cfg.stream_seed))
    cfg.num_tasks_per_benchmark = int(pick("num_tasks_per_benchmark", cfg.num_tasks_per_benchmark))

    cfg.protocol = str(pick("protocol", cfg.protocol)).lower()
    cfg.val_tasks_per_benchmark = int(pick("val_tasks_per_benchmark", cfg.val_tasks_per_benchmark))
    cfg.val_source = str(pick("val_source", cfg.val_source)).lower()
    cfg.on_exhausted = str(pick("on_exhausted", cfg.on_exhausted)).lower()
    cfg.block_passes = max(1, int(pick("block_passes", cfg.block_passes)))

    max_steps = node.get("max_steps")
    cfg.max_steps = int(max_steps) if max_steps is not None else None
    cfg.reward_success = float(pick("reward_success", cfg.reward_success))
    cfg.reward_use_score = bool(pick("reward_use_score", cfg.reward_use_score))
    cfg.format_penalty = float(pick("format_penalty", cfg.format_penalty))

    cfg.reset_timeout_s = float(pick("reset_timeout_s", cfg.reset_timeout_s))
    cfg.step_timeout_s = float(pick("step_timeout_s", cfg.step_timeout_s))

    cfg.online_metrics_enable = bool(pick("online_metrics_enable", cfg.online_metrics_enable))
    cfg.online_metrics_path = str(pick("online_metrics_path", cfg.online_metrics_path))
    cfg.val_stream_seed = int(pick("val_stream_seed", cfg.val_stream_seed))

    # --- validation -----------------------------------------------------------
    if cfg.stream_mode not in VALID_STREAM_MODES:
        raise ValueError(f"env.agentstream.stream_mode must be one of {VALID_STREAM_MODES}, got '{cfg.stream_mode}'")
    if cfg.protocol not in VALID_PROTOCOLS:
        raise ValueError(f"env.agentstream.protocol must be one of {VALID_PROTOCOLS}, got '{cfg.protocol}'")
    if cfg.val_source not in VALID_VAL_SOURCES:
        raise ValueError(f"env.agentstream.val_source must be one of {VALID_VAL_SOURCES}, got '{cfg.val_source}'")
    if cfg.on_exhausted not in VALID_ON_EXHAUSTED:
        raise ValueError(f"env.agentstream.on_exhausted must be one of {VALID_ON_EXHAUSTED}, got '{cfg.on_exhausted}'")
    if not cfg.benchmarks:
        raise ValueError("env.agentstream.benchmarks must not be empty")
    if cfg.stream_mode == "isolated" and len(cfg.benchmarks) != 1:
        raise ValueError(
            "stream_mode='isolated' expects exactly one benchmark per run "
            "(launch one run per benchmark, mirroring AgentStream's run_experiment.sh); "
            f"got: {cfg.benchmarks}"
        )
    if not cfg.exgentic_root:
        raise ValueError(
            "env.agentstream.exgentic_root must point to the AgentStream/exgentic checkout "
            "(the directory that contains 'src/exgentic')."
        )
    return cfg
