# SPDX-License-Identifier: Apache-2.0
"""Assert that the AgentStream task order / holdout split used by scripts/ours equals the SEED RL
stream (same benchmarks, 64 + 32 tasks, seed 44). Needs the benchmarks installed (run on the cluster):

    uv run python scripts/ours/check_stream_alignment.py [--seed-root ../../SEED]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "utils"))

from registry_ours import NUM_HOLDOUT_TASKS, NUM_STREAM_TASKS, STREAM_SEED, build_configs, holdout_task_ids  # noqa: E402


def load_seed_task_stream(seed_root: Path):
    path = seed_root / "agent_system/environments/env_package/agentstream/task_stream.py"
    spec = importlib.util.spec_from_file_location("seed_task_stream", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def task_universe(configs):
    from exgentic.interfaces.registry import load_benchmark

    universe = {}
    for slug, cfg in configs.items():
        bm = load_benchmark(slug)(**cfg["bm_kwargs"])
        evaluator = bm.get_evaluator()
        try:
            universe[slug] = [str(t) for t in evaluator.list_tasks()]
        finally:
            evaluator.close()
            bm.close()
    return universe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-root", default=str(HERE.parents[3] / "SEED"))
    args = parser.parse_args()
    from task_ordering import get_unified_task_order

    configs = build_configs()
    seed_ts = load_seed_task_stream(Path(args.seed_root))
    universe = task_universe(configs)
    stream = seed_ts.select_tasks(universe, NUM_STREAM_TASKS)
    for mode in ("interleaved", "isolated"):
        ours = get_unified_task_order(configs, NUM_STREAM_TASKS, STREAM_SEED, mode)
        theirs = [tuple(ref) for ref in seed_ts.order_tasks(stream, mode, STREAM_SEED)]
        assert ours == theirs, f"{mode}: task order differs at index {next(i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b)}"
        print(f"{mode}: {len(ours)} tasks identical; first 5 = {ours[:5]}")
    holdout = holdout_task_ids(configs)
    seed_holdout = seed_ts.select_holdout_tasks(universe, NUM_STREAM_TASKS, NUM_HOLDOUT_TASKS)
    assert holdout == seed_holdout, "holdout split differs"
    print("holdout: " + ", ".join(f"{slug}={len(ids)}" for slug, ids in holdout.items()) + " identical")


if __name__ == "__main__":
    main()
