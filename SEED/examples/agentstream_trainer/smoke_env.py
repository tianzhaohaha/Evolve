#!/usr/bin/env python3
"""Phase-0 smoke test for the SEED x AgentStream bridge (no GPU / Ray needed).

Validates, in order:
  1. host-side exgentic import inside the SEED conda env;
  2. task discovery (Evaluator.list_tasks) per benchmark;
  3. one full session lifecycle: reset -> (optional) one step -> score/close.

Run inside the SEED conda env, e.g.:
  python examples/agentstream_trainer/smoke_env.py \
      --exgentic-root $AGENTSTREAM_EXGENTIC_ROOT --benchmarks bfcl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agent_system.environments.env_package.agentstream.as_config import (  # noqa: E402
    resolve_benchmark_kwargs,
)
from agent_system.environments.env_package.agentstream.exgentic_client import (  # noqa: E402
    BenchmarkHub,
    SessionDriver,
)
from agent_system.environments.env_package.agentstream.task_stream import (  # noqa: E402
    select_tasks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exgentic-root", default=os.environ.get("AGENTSTREAM_EXGENTIC_ROOT", ""))
    parser.add_argument("--benchmarks", default="bfcl", help="comma list, e.g. bfcl,tau2")
    parser.add_argument("--benchmark-kwargs-json", default="{}", help='e.g. {"tau2": {"subset": "retail"}}')
    parser.add_argument("--runner", default="venv")
    parser.add_argument("--num-tasks", type=int, default=5, help="tasks to select/preview per benchmark")
    parser.add_argument("--step", action="store_true", help="also execute one (first-listed) action")
    parser.add_argument("--output-dir", default="outputs/agentstream_smoke")
    args = parser.parse_args()

    if not args.exgentic_root:
        print("error: --exgentic-root (or AGENTSTREAM_EXGENTIC_ROOT) is required", file=sys.stderr)
        return 2

    slugs = sorted(s.strip() for s in args.benchmarks.split(",") if s.strip())
    overrides = json.loads(args.benchmark_kwargs_json)
    bm_kwargs = {slug: resolve_benchmark_kwargs(slug, overrides.get(slug)) for slug in slugs}

    print(f"[1/3] importing exgentic from {args.exgentic_root} ...")
    hub = BenchmarkHub(
        exgentic_root=args.exgentic_root,
        slugs=slugs,
        benchmark_kwargs=bm_kwargs,
        runner=args.runner,
        output_dir=args.output_dir,
        run_id="smoke",
    )
    print("      OK")

    print(f"[2/3] listing tasks for {slugs} ...")
    universe = hub.list_all_tasks()
    selected = select_tasks(universe, args.num_tasks)
    for slug in slugs:
        print(f"      {slug}: {len(universe[slug])} tasks total, seed-42 selection: {selected[slug]}")

    slug = slugs[0]
    task_id = selected[slug][0]
    print(f"[3/3] session lifecycle on {slug}/{task_id} ...")
    driver = SessionDriver(
        exgentic_root=args.exgentic_root,
        runner=args.runner,
        output_dir=args.output_dir,
        run_id="smoke_driver",
        max_steps=3,
    )
    try:
        payload = driver.reset(
            slug=slug,
            task_id=task_id,
            bm_kwargs=bm_kwargs.get(slug, {}),
            session_kwargs=hub.session_kwargs(slug, task_id),
        )
        print(f"      task: {payload['task'][:300]}")
        print(f"      context: {payload['context'][:200]}")
        print(f"      actions:\n{payload['actions_text'][:800]}")
        print(f"      observation: {payload['observation'][:300]}")
        if payload.get("reset_error"):
            print("      RESET FAILED — see observation above", file=sys.stderr)
            return 1

        if args.step:
            first_action = payload["actions_text"].splitlines()[0].lstrip("- ").split(":")[0].strip()
            print(f"      stepping with action '{first_action}' (empty arguments) ...")
            obs_text, done, info = driver.step({"name": first_action, "arguments": {}})
            print(f"      -> done={done} info={ {k: info[k] for k in ('won', 'score', 'step_count') if k in info} }")
            print(f"      -> observation: {obs_text[:300]}")
    finally:
        driver.close()
        hub.close()

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
