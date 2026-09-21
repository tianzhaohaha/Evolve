#!/usr/bin/env python3
"""Which wandb runs lost the browsecompplus search path, and at which step?

A dead search (retriever unreachable or the searcher cache unopenable) is invisible to the env error
rate: the failure comes back as an observation string, the episode continues and scores 0. Its
signature in the logged metrics is a step that has browsecompplus episodes but whose longest prompt is
far shorter than a step with real search results (observations no longer carry retrieved documents):
`prompt_length/max` 9k-13k healthy vs 4k-6k dead on the mixed stream, and every such step has
browsecompplus success exactly 0.

  python examples/agentstream_trainer/check_browsecomp_health.py --since 2026-09-18
  python examples/agentstream_trainer/check_browsecomp_health.py --name-regex 69818 --threshold 8000

Needs the `seed` conda env (wandb, ~/.netrc). Verdict per run: healthy / dead@<first step> (n dead steps).
"""

from __future__ import annotations

import argparse
import re

import wandb

KEYS = ["_step", "prompt_length/max", "online/browsecompplus/first_pass_episodes", "episode/browsecompplus_success_rate"]


def dead_steps(rows, threshold):
    """Steps that ran browsecompplus episodes with a prompt max below the threshold."""
    dead, prev = [], 0.0
    for r in sorted(rows, key=lambda r: r["_step"]):
        eps = r.get("online/browsecompplus/first_pass_episodes")
        if eps is None or r.get("prompt_length/max") is None or r["_step"] < 1:
            continue
        ran_bcp, prev = eps > prev, eps
        if ran_bcp and r["prompt_length/max"] < threshold:
            dead.append(int(r["_step"]))
    return dead


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="ql2505-columbia-university/agentic_agentstream")
    p.add_argument("--since", default="2026-09-18", help="created_at lower bound (UTC, YYYY-MM-DD)")
    p.add_argument("--name-regex", default="", help="only runs whose display name matches")
    p.add_argument("--threshold", type=float, default=8000, help="prompt_length/max below this = no documents in context")
    args = p.parse_args()
    api = wandb.Api(timeout=180)
    runs = [r for r in api.runs(args.project, order="-created_at") if str(r.created_at) >= args.since and re.search(args.name_regex, r.name)]
    print(f"{'created (UTC)':16s} {'host':13s} {'steps':>5s} {'bcp':>4s} {'verdict':22s} name")
    for r in sorted(runs, key=lambda r: str(r.created_at)):
        rows = list(r.scan_history(keys=KEYS))
        bcp_steps = sum(1 for a, b in zip([0.0] + [x.get("online/browsecompplus/first_pass_episodes") or 0 for x in rows], [x.get("online/browsecompplus/first_pass_episodes") or 0 for x in rows]) if b > a)
        dead = dead_steps(rows, args.threshold)
        verdict = "no browsecomp" if not bcp_steps else (f"dead@{dead[0]} ({len(dead)}/{bcp_steps} steps)" if dead else "healthy")
        print(f"{str(r.created_at)[:16]:16s} {str((r.metadata or {}).get('host', '?')):13s} {str(r.summary.get('_step', '?')):>5s} {bcp_steps:>4d} {verdict:22s} {r.name}")


if __name__ == "__main__":
    main()
