"""Pure parts of examples/agentstream_trainer/delta_gap.py (divergence, think-token split, aggregation, report)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "delta_gap", Path(__file__).resolve().parents[2] / "examples" / "agentstream_trainer" / "delta_gap.py"
)
delta_gap = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(delta_gap)


def test_divergence_index_is_the_first_mismatching_step():
    assert delta_gap.divergence_index(["a", "b", "c"], ["a", "b", "c"]) == 3
    assert delta_gap.divergence_index(["a", "x", "c"], ["a", "b", "c"]) == 1
    assert delta_gap.divergence_index(["a", "b", "c", "d"], ["a", "b", "c"]) == 3   # longer than the reference
    assert delta_gap.divergence_index(["x"], []) == 0
    assert delta_gap.divergence_index([], ["a"]) == 0


def test_think_spans_and_token_classification():
    response = "<think>plan</think>\n<action>{\"name\": \"go\"}</action>"
    spans = delta_gap.think_spans(response)
    assert spans == [(0, len("<think>plan</think>"))]
    offsets = [(0, 7), (7, 11), (11, 19), (19, 20), (20, 28)]
    assert delta_gap.token_is_think(offsets, spans) == [True, True, True, False, False]
    assert delta_gap.think_spans("<reasoning>r</reasoning><action>a</action>") == [(0, len("<reasoning>r</reasoning>"))]
    assert delta_gap.think_spans("<action>a</action>") == []


def _traj(slug, cls, *, gap_sum, tokens, think=(0.0, 0), prefix=(0.0, 0), diverge=0):
    other_sum, other_tokens = gap_sum - think[0], tokens - think[1]
    return {
        "task_id": f"{slug}/t", "slug": slug, "rollout_id": 0, "success": cls != "failure", "cls": cls, "steps": 2,
        "diverge_step": diverge, "tokens": tokens, "gap_sum": gap_sum,
        "think_tokens": think[1], "think_gap_sum": think[0], "other_tokens": other_tokens, "other_gap_sum": other_sum,
        "prefix_tokens": prefix[1], "prefix_gap_sum": prefix[0], "suffix_tokens": tokens - prefix[1], "suffix_gap_sum": gap_sum - prefix[0],
        "skipped_steps": 0,
    }


def test_aggregate_is_token_weighted_and_splits_failures_at_divergence():
    trajs = [
        _traj("bfcl", "source", gap_sum=4.0, tokens=4),
        _traj("bfcl", "success", gap_sum=-1.0, tokens=2, think=(-1.5, 1)),
        _traj("bfcl", "failure", gap_sum=0.0, tokens=4, prefix=(1.0, 2), diverge=1),   # prefix +0.5, suffix -0.5
        _traj("tau2", "failure", gap_sum=-2.0, tokens=4, prefix=(0.0, 0), diverge=0),  # diverges immediately
    ]
    tables = delta_gap.aggregate(trajs)
    by = {(r["slug"], r["cls"]): r for r in tables["by_class"]}
    assert by[("all", "source")]["gap"] == pytest.approx(1.0)
    assert by[("bfcl", "success")]["gap_think"] == pytest.approx(-1.5) and by[("bfcl", "success")]["gap_other"] == pytest.approx(0.5)
    assert by[("all", "failure")]["trajs"] == 2 and by[("all", "failure")]["gap"] == pytest.approx(-0.25)
    assert by[("all", "failure")]["frac_positive"] == 0.0
    failures = {r["slug"]: r for r in tables["failures"]}
    assert failures["bfcl"]["gap_prefix"] == pytest.approx(0.5) and failures["bfcl"]["gap_suffix"] == pytest.approx(-0.5)
    assert failures["bfcl"]["frac_prefix_gt_suffix"] == 1.0
    assert failures["all"]["frac_diverge_at_0"] == pytest.approx(0.5) and failures["all"]["diverge_step_mean"] == pytest.approx(0.5)
    assert failures["all"]["gap_prefix"] == pytest.approx(0.5) and failures["all"]["gap_suffix"] == pytest.approx(-3.0 / 6)
    report = delta_gap.format_report(tables, "C2", n_steps=8)
    assert "Scored 8 C0 steps" in report and "| bfcl | failure | 1 |" in report and "| tau2 | 1 | 0.0 | 1.00 |" in report


class _StubScorer:
    """Character tokens; the reference-augmented prompt (longer) raises every token's log-prob by 0.1."""

    def prompt_ids(self, prompt):
        return list(range(len(prompt)))

    def response_ids(self, response):
        return list(range(len(response))), [(i, i + 1) for i in range(len(response))]

    def logprobs(self, prompt_ids, response_ids):
        return [-1.0 + (0.1 if len(prompt_ids) > 200 else 0.0)] * len(response_ids)


def _record(task, rollout_id, success, actions, *, slug="bfcl"):
    steps = [
        {"step_idx": i, "observation_prompt": f"prompt {task} {i}", "model_response": f"<think>t{i}</think><action>{a}</action>"}
        for i, a in enumerate(actions)
    ]
    return {"task_id": f"{slug}/{task}", "task_type": slug, "rollout_id": rollout_id, "success": success, "num_steps": len(steps), "steps": steps}


def test_score_trajectories_classifies_rollouts_and_splits_at_divergence():
    c0 = [
        _record("t1", 0, True, ["a", "b"]),            # the reference source
        _record("t1", 1, True, ["a", "c", "b"]),       # another success
        _record("t1", 2, False, ["a", "x", "y"]),      # diverges at step 1
        _record("t2", 0, False, ["q"]),                # no material -> ignored
    ]
    materials = [{"task_id": "bfcl/t1", "source_task_id": "bfcl/t1", "source_rollout_id": 0, "text": "R" * 300}]
    rows, scored = delta_gap.score_trajectories(_StubScorer(), c0, materials, max_tokens=10_000, limit=None)
    assert scored == 8 and [r["cls"] for r in rows] == ["source", "success", "failure"]
    failure = rows[2]
    assert failure["diverge_step"] == 1
    assert failure["prefix_tokens"] == len(c0[2]["steps"][0]["model_response"]) and failure["suffix_tokens"] == failure["tokens"] - failure["prefix_tokens"]
    for row in rows:
        assert row["gap_sum"] / row["tokens"] == pytest.approx(0.1)
        assert row["think_tokens"] + row["other_tokens"] == row["tokens"] and row["think_tokens"] > 0
    rows, _ = delta_gap.score_trajectories(_StubScorer(), c0, materials, max_tokens=10_000, limit=1)
    assert len(rows) == 1
    rows, scored = delta_gap.score_trajectories(_StubScorer(), c0, materials, max_tokens=10, limit=None)
    assert scored == 0 and all(r["skipped_steps"] == r["steps"] for r in rows)
