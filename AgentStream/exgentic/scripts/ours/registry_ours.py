# SPDX-License-Identifier: Apache-2.0
"""Benchmark registry and stream constants shared by every AgentStream baseline run that must be
comparable with the SEED RL runs (three benchmarks, 64 stream + 32 holdout tasks per benchmark,
seed 44, 40 steps per episode, GLM judges, the shared BrowseComp retriever service).

The task selection / ordering code (``scripts/utils/task_ordering.py``) is the one the SEED task
stream was ported from, so ``get_unified_task_order(configs, 64, 44, mode)`` reproduces the RL
stream task by task and ``holdout_task_ids`` reproduces its holdout split.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

BENCHMARKS = ("bfcl", "tau2", "browsecompplus")
SELECTION_SEED = 42  # task selection (fixed, as in AgentStream and SEED)
STREAM_SEED = 44  # task ordering / interleaving (the SEED runs' AGENTSTREAM_STREAM_SEED)
NUM_STREAM_TASKS = 64
NUM_HOLDOUT_TASKS = 32
MAX_STEPS = 40  # per-episode cap, = the RL runs' AGENTSTREAM_MAX_STEPS
ONLINE_TEMPERATURE = 1.0  # = rollout temperature of the RL runs
HOLDOUT_TEMPERATURE = 0.4  # = validation temperature of the RL runs
DEFAULT_JUDGE_MODEL = "openrouter/z-ai/glm-5.2"  # tau2 user simulator + browsecomp answer judge
DEFAULT_RETRIEVER_URL = "http://127.0.0.1:60100"


def build_configs(
    judge_model: str | None = None, retriever_url: str | None = None, benchmarks: Sequence[str] = BENCHMARKS
) -> Dict[str, Dict[str, Any]]:
    """``{slug: {"bm_kwargs", "agent_kwargs"}}`` matching the SEED environment's benchmark settings."""
    judge_model = judge_model or os.environ.get("AGENTSTREAM_API_MODEL", DEFAULT_JUDGE_MODEL)
    retriever_url = retriever_url or os.environ.get("AGENTSTREAM_BROWSECOMP_RETRIEVER_URL", DEFAULT_RETRIEVER_URL)
    registry = {
        "bfcl": {"subset": "multi_turn_base"},
        "tau2": {"subset": "retail", "user_simulator_model": judge_model},
        "browsecompplus": {
            "include_get_document": True,
            "eval_model_id": judge_model,
            "retriever_url": retriever_url,
            "use_cache": False,  # a shared on-disk search cache corrupted earlier RL runs
        },
    }
    unknown = sorted(set(benchmarks) - set(registry))
    if unknown:
        raise ValueError(f"Unsupported benchmarks {unknown}; choose from {sorted(registry)}.")
    return {slug: {"bm_kwargs": dict(registry[slug]), "agent_kwargs": {}} for slug in benchmarks}


def holdout_task_ids(
    configs: Dict[str, Dict[str, Any]],
    num_stream: int = NUM_STREAM_TASKS,
    num_holdout: int = NUM_HOLDOUT_TASKS,
    select: Callable[..., Dict[str, List[str]]] | None = None,
) -> Dict[str, List[str]]:
    """Held-out ids per benchmark: the ``num_holdout`` ids after the stream selection in the same
    seed-42 shuffle (SEED ``select_holdout_tasks``). ``select`` defaults to AgentStream's selector."""
    if select is None:
        from task_ordering import select_tasks_only as select
    selected = select(configs, num_stream + num_holdout, SELECTION_SEED)
    return {slug: ids[num_stream : num_stream + num_holdout] for slug, ids in selected.items()}
