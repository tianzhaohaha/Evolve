#!/usr/bin/env python3
"""Offline gap check: does a reference solution make the policy's OWN C0 trajectories more likely?

The delta test (delta_test.py) measures outcomes; this script measures what SEED's distillation loss
sees. For every step of every C0 rollout of a task that has a reference, the response is scored
twice with the frozen SFT checkpoint:
  lp0 = log p(response | plain prompt)            lp2 = log p(response | prompt + reference)
gap = lp2 - lp0 per token (nats). A one-sided OPD loss can only use the reference if the gap has
structure: on failed rollouts it should be positive up to the step where the rollout diverges from
the reference's action sequence and negative after it. The reference's own source rollout overlaps
the reference verbatim and is reported separately. Tokens inside <think> / <reasoning> spans and the
rest (action JSON) are reported apart, because the earlier action-only skeleton pushed exactly that
split the wrong way.

Needs the checkpoint on one GPU (transformers, bf16); no vLLM, no exgentic. The prompt is rebuilt
with the tokenizer's chat template (add_generation_prompt, enable_thinking=false as in RL); the
response is re-tokenized, so token boundaries can differ slightly from the sampled ones.

Usage (seed conda env, after delta_test.py produced C0 and materials):
  python examples/agentstream_trainer/delta_gap.py --output-dir outputs/delta_test_v5 --model "$AGENTSTREAM_SFT_MODEL_DIR"
Outputs under --output-dir: gap_report.md, gap_per_trajectory.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seed.prompting import build_augmented_observation_text  # noqa: E402
from seed.sibling import extract_action_text  # noqa: E402

THINK_RE = re.compile(r"<think>.*?</think>|<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)
CLASSES = ("source", "success", "failure")


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def divergence_index(actions: Sequence[str], ref_actions: Sequence[str]) -> int:
    """First step whose action differs from the reference's action at the same index (a rollout longer
    than the reference diverges at the reference's length). Equal to ``len(actions)`` when it never diverges."""
    for i, action in enumerate(actions):
        if i >= len(ref_actions) or action != ref_actions[i]:
            return i
    return len(actions)


def think_spans(response: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in THINK_RE.finditer(response or "")]


def token_is_think(offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int]]) -> List[bool]:
    """Per token: does its first character fall inside a reasoning span?"""
    return [any(start <= a < end for start, end in spans) for a, _ in offsets]


def _wmean(pairs: Sequence[Tuple[float, int]]) -> float:
    """Token-weighted mean of (sum, count) pairs."""
    total = sum(c for _, c in pairs)
    return sum(s for s, _ in pairs) / total if total else float("nan")


def aggregate(trajs: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Per-trajectory rows -> two tables (by benchmark x class; failure prefix / suffix split)."""
    slugs = ["all"] + sorted({t["slug"] for t in trajs})
    by_class, failures = [], []
    for slug in slugs:
        rows = [t for t in trajs if slug == "all" or t["slug"] == slug]
        for cls in CLASSES:
            group = [t for t in rows if t["cls"] == cls]
            if not group:
                continue
            by_class.append(
                {
                    "slug": slug, "cls": cls, "trajs": len(group),
                    "gap": _wmean([(t["gap_sum"], t["tokens"]) for t in group]),
                    "frac_positive": sum(1 for t in group if t["gap_sum"] > 0) / len(group),
                    "gap_think": _wmean([(t["think_gap_sum"], t["think_tokens"]) for t in group]),
                    "gap_other": _wmean([(t["other_gap_sum"], t["other_tokens"]) for t in group]),
                }
            )
        group = [t for t in rows if t["cls"] == "failure"]
        if group:
            failures.append(
                {
                    "slug": slug, "trajs": len(group),
                    "diverge_step_mean": sum(t["diverge_step"] for t in group) / len(group),
                    "frac_diverge_at_0": sum(1 for t in group if t["diverge_step"] == 0) / len(group),
                    "gap_prefix": _wmean([(t["prefix_gap_sum"], t["prefix_tokens"]) for t in group]),
                    "gap_suffix": _wmean([(t["suffix_gap_sum"], t["suffix_tokens"]) for t in group]),
                    "frac_prefix_gt_suffix": sum(
                        1 for t in group if t["prefix_tokens"] and t["suffix_tokens"]
                        and t["prefix_gap_sum"] / t["prefix_tokens"] > t["suffix_gap_sum"] / t["suffix_tokens"]
                    ) / len(group),
                }
            )
    return {"by_class": by_class, "failures": failures}


def format_report(tables: Dict[str, List[Dict[str, Any]]], condition: str, n_steps: int) -> str:
    lines = [
        f"# Gap check: log p(response | prompt + {condition} reference) - log p(response | prompt), nats per token",
        "",
        f"Scored {n_steps} C0 steps. 'source' = the rollout the reference was rendered from (verbatim overlap, inflated by construction).",
        "",
        "| benchmark | class | trajs | gap | trajs with gap > 0 | gap on think tokens | gap on other tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in tables["by_class"]:
        lines.append(f"| {r['slug']} | {r['cls']} | {r['trajs']} | {r['gap']:+.4f} | {r['frac_positive']:.2f} | {r['gap_think']:+.4f} | {r['gap_other']:+.4f} |")
    lines += [
        "",
        "Failed rollouts, split at the first step whose action differs from the reference's action sequence:",
        "",
        "| benchmark | trajs | mean divergence step | diverge at step 0 | gap before divergence | gap after divergence | prefix gap > suffix gap |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in tables["failures"]:
        lines.append(
            f"| {r['slug']} | {r['trajs']} | {r['diverge_step_mean']:.1f} | {r['frac_diverge_at_0']:.2f} | {r['gap_prefix']:+.4f} | {r['gap_suffix']:+.4f} | {r['frac_prefix_gt_suffix']:.2f} |"
        )
    lines += [
        "",
        "Reading: the one-sided OPD loss is usable when failures show 'gap before divergence' > 0 and 'gap after divergence' < 0",
        "(the reference endorses the shared prefix and withdraws support after the wrong turn). A flat or unstructured failure",
        "gap means the model follows a demonstration when generating but the reference does not re-rank its old samples;",
        "distill from reference-conditioned rollouts instead (context distillation). A think-token gap far below the other-token",
        "gap is the action-only skeleton failure mode.",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class Scorer:
    """Per-token log-probs of a response under a prompt with the frozen checkpoint (batch of one)."""

    def __init__(self, model_path: str, device: str = "cuda", enable_thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device).eval()
        except TypeError:  # transformers < 4.56
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device).eval()
        self.device = device
        self.enable_thinking = enable_thinking

    def prompt_ids(self, prompt: str) -> List[int]:
        # Single user message as the delta-test endpoint receives it; enable_thinking is ignored by templates
        # without the switch (Qwen3 *-2507) and mirrors RL / Stage-1 (thinking off) for hybrid Qwen3 models.
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, enable_thinking=self.enable_thinking
        )

    def response_ids(self, response: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        enc = self.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def logprobs(self, prompt_ids: Sequence[int], response_ids: Sequence[int]) -> List[float]:
        torch = self.torch
        ids = torch.tensor([list(prompt_ids) + list(response_ids)], device=self.device)
        keep = len(response_ids) + 1
        with torch.inference_mode():
            try:
                logits = self.model(input_ids=ids, logits_to_keep=keep).logits[0]
            except TypeError:  # older transformers: full logits
                logits = self.model(input_ids=ids).logits[0, -keep:]
        logp = torch.log_softmax(logits[:-1].float(), dim=-1)
        target = torch.tensor(list(response_ids), device=self.device)
        return logp.gather(-1, target[:, None]).squeeze(-1).tolist()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_trajectories(scorer: Scorer, c0_records: Sequence[Dict[str, Any]], materials: Sequence[Dict[str, Any]],
                       *, max_tokens: int, limit: Optional[int]) -> Tuple[List[Dict[str, Any]], int]:
    """One row per C0 rollout of a task that has a reference; returns (rows, scored steps)."""
    by_task_material = {m["task_id"]: m for m in materials}
    by_task_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in c0_records:
        if r.get("num_steps", 0) > 0 and r["task_id"] in by_task_material:
            by_task_records[r["task_id"]].append(r)
    ref_actions: Dict[str, List[str]] = {}
    for task_id, m in by_task_material.items():
        source = next((r for r in c0_records if r["task_id"] == m["source_task_id"] and r["rollout_id"] == m["source_rollout_id"]), None)
        ref_actions[task_id] = [extract_action_text(s.get("model_response", "")) for s in (source or {}).get("steps", [])]
    rows, scored = [], 0
    for task_id, records in by_task_records.items():
        m = by_task_material[task_id]
        for record in records:
            if limit is not None and len(rows) >= limit:
                return rows, scored
            actions = [extract_action_text(s.get("model_response", "")) for s in record["steps"]]
            diverge = divergence_index(actions, ref_actions[task_id]) if not record.get("success") else len(actions)
            row = {"task_id": task_id, "slug": record["task_type"], "rollout_id": record["rollout_id"], "success": bool(record.get("success")),
                   "cls": "source" if (m["source_task_id"] == task_id and record["rollout_id"] == m["source_rollout_id"]) else ("success" if record.get("success") else "failure"),
                   "steps": len(record["steps"]), "diverge_step": diverge, "tokens": 0, "gap_sum": 0.0, "think_tokens": 0, "think_gap_sum": 0.0,
                   "other_tokens": 0, "other_gap_sum": 0.0, "prefix_tokens": 0, "prefix_gap_sum": 0.0, "suffix_tokens": 0, "suffix_gap_sum": 0.0, "skipped_steps": 0}
            for step_idx, step in enumerate(record["steps"]):
                prompt, response = step.get("observation_prompt", ""), step.get("model_response", "")
                if not prompt or not response:
                    row["skipped_steps"] += 1
                    continue
                p0 = scorer.prompt_ids(prompt)
                p2 = scorer.prompt_ids(build_augmented_observation_text(observation=prompt, reference_solution=m["text"]))
                r_ids, offsets = scorer.response_ids(response)
                if not r_ids or len(p2) + len(r_ids) > max_tokens:
                    row["skipped_steps"] += 1
                    continue
                gap = [b - a for a, b in zip(scorer.logprobs(p0, r_ids), scorer.logprobs(p2, r_ids))]
                is_think = token_is_think(offsets, think_spans(response))
                scored += 1
                row["tokens"] += len(gap); row["gap_sum"] += sum(gap)
                row["think_tokens"] += sum(is_think); row["think_gap_sum"] += sum(g for g, t in zip(gap, is_think) if t)
                row["other_tokens"] += len(gap) - sum(is_think); row["other_gap_sum"] += sum(g for g, t in zip(gap, is_think) if not t)
                part = "prefix" if step_idx < diverge else "suffix"
                row[f"{part}_tokens"] += len(gap); row[f"{part}_gap_sum"] += sum(gap)
            rows.append(row)
            logging.info("scored %s:%s cls=%s steps=%d gap=%+.4f", task_id, record["rollout_id"], row["cls"], row["steps"],
                         row["gap_sum"] / row["tokens"] if row["tokens"] else float("nan"))
    return rows, scored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="the SFT checkpoint directory (RL start)")
    parser.add_argument("--condition", default="C2", choices=["C2"],
                        help="materials file providing the reference; only the same-task C2 reference makes the divergence split meaningful")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-tokens", type=int, default=16384, help="skip steps whose prompt + reference + response exceed this")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rollouts (smoke)")
    parser.add_argument("--enable-thinking", action="store_true", help="RL templates use enable_thinking=false")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.output_dir)
    c0 = read_jsonl(out / "C0" / "baseline_rollouts.jsonl")
    materials_file = out / f"materials_{args.condition}.jsonl"
    if args.condition == "C2" and not materials_file.exists():
        materials_file = out / "materials.jsonl"
    materials = read_jsonl(materials_file)
    if not c0 or not materials:
        raise SystemExit("gap: needs C0/baseline_rollouts.jsonl and the condition's materials file (run delta_test.py first)")
    scorer = Scorer(args.model, device=args.device, enable_thinking=args.enable_thinking)
    rows, scored = score_trajectories(scorer, c0, materials, max_tokens=args.max_tokens, limit=args.limit)
    rows = [r for r in rows if r["tokens"] > 0]
    tables = aggregate(rows)
    with open(out / "gap_per_trajectory.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "gap_report.md").write_text(format_report(tables, args.condition, scored), encoding="utf-8")
    print((out / "gap_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
