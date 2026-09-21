"""Sibling-success local teacher and sample routing (seed/sibling.py) plus the actor-side PG weight."""

import numpy as np
import pytest
import torch

from seed.sibling import (
    build_action_skeleton,
    build_reference_solution,
    compute_pg_row_weights,
    extract_action_text,
    group_outcomes,
    select_sibling_references,
)
from tests.trainer.ppo.test_opd_spec_first import run_update  # noqa: F401  (pytest fixture)
from verl.trainer.ppo.core_algos import agg_loss, compute_opd_loss, compute_policy_loss, kl_penalty


def _step(idx, obs, action=None, valid=None, raw=None):
    if raw is None:
        raw = f"<think>x</think><action>{action}</action>" if action else "<think>no action yet</think>"
    step = {"step_index": idx, "observation": obs, "response": raw}
    if valid is not None:
        step["action_valid"] = valid
    return step


FINISH = '{"name":"finish","arguments":{}}'
SEARCH = '{"name":"search","arguments":{"q":"mug"}}'


# ---------------------------------------------------------------- action extraction / skeleton


def test_extract_action_text_compacts_json_and_falls_back():
    assert extract_action_text('<action>{"name": "search", "arguments": {"q": "mug"}}</action>') == SEARCH
    assert extract_action_text('<action>```json\n{"name": "finish", "arguments": {}}\n```</action>') == FINISH
    assert extract_action_text('<tool_call>{"name": "get_user", "arguments": {"id": 1}}</tool_call>') == (
        '{"name":"get_user","arguments":{"id":1}}'
    )
    assert extract_action_text("<action>not json\n  at   all</action>") == "not json at all"
    assert extract_action_text("<think>only thinking</think>") == ""
    assert extract_action_text("") == ""
    assert extract_action_text("<action>" + "x" * 50 + "</action>", max_chars=10) == "x" * 7 + "..."


def test_build_action_skeleton_formats_lines_and_skips_steps_without_actions():
    steps = [
        _step(0, "Welcome   to\nthe shop", SEARCH),
        _step(1, "no action here"),
        _step(2, "results", '{"name":"click","arguments":{"i":2}}', valid=False),
        _step(3, "cart", FINISH),
    ]
    text, rendered = build_action_skeleton(steps, obs_chars=12, action_chars=300, max_chars=6000)

    assert rendered == 2
    assert text == f"Step 1 | obs: Welcome t... | action: {SEARCH}\nStep 4 | obs: cart | action: {FINISH}"
    assert build_action_skeleton([]) == ("", 0)
    assert build_action_skeleton([_step(0, "x")]) == ("", 0)


def test_build_action_skeleton_cap_keeps_head_and_final_step():
    steps = [_step(i, f"obs{i}", '{"name":"a","arguments":{"k":%d}}' % i) for i in range(20)]
    full, rendered_full = build_action_skeleton(steps)
    assert rendered_full == 20 and full.count("\n") == 19

    line_len = len(full.split("\n")[0]) + 1
    text, rendered = build_action_skeleton(steps, max_chars=line_len * 6)
    lines = text.split("\n")

    assert len(text) <= line_len * 6
    assert lines[0].startswith("Step 1 |")
    assert lines[-1].startswith("Step 20 |")
    assert sum("steps omitted" in line for line in lines) == 1
    assert rendered == len(lines) - 1  # rendered lines = kept head + final step


def test_build_reference_solution_keeps_full_responses_and_shares_the_cap_rules():
    steps = [
        _step(0, "Welcome   to\nthe shop", SEARCH, raw=f"<think>look\n around</think>\n<action>{SEARCH}</action>"),
        _step(1, "no action here"),
        _step(2, "results", '{"name":"click","arguments":{"i":2}}', valid=False),
        _step(3, "cart", FINISH),
    ]
    text, rendered = build_reference_solution(steps, obs_chars=12, response_chars=300, max_chars=6000)
    assert rendered == 3  # the step without an action still carries a response; the invalid step is skipped
    lines = text.split("\n")
    assert lines[0] == f"Step 1 | obs: Welcome t... | response: <think>look around</think> <action>{SEARCH}</action>"
    assert lines[1] == "Step 2 | obs: no action... | response: <think>no action yet</think>"
    assert lines[2].startswith("Step 4 | obs: cart | response: <think>x</think><action>")
    assert build_reference_solution(steps, response_chars=10)[0].split("\n")[0].endswith("response: <think>...")
    assert build_reference_solution([]) == ("", 0)

    long = [_step(i, f"obs{i}", '{"name":"a","arguments":{"k":%d}}' % i) for i in range(20)]
    line_len = len(build_reference_solution(long)[0].split("\n")[0]) + 1
    capped, kept = build_reference_solution(long, max_chars=line_len * 6)
    capped_lines = capped.split("\n")
    assert capped_lines[0].startswith("Step 1 |") and capped_lines[-1].startswith("Step 20 |")
    assert sum("steps omitted" in line for line in capped_lines) == 1 and kept == len(capped_lines) - 1


# ---------------------------------------------------------------- group outcomes / references

UIDS = ["g1"] * 6 + ["g2"] * 2 + ["g3"] * 2
TRAJ_UIDS = ["s1", "s1", "s2", "f1", "f1", "f2", "a1", "a2", "b1", "b2"]
SUCCESS = {"s1": 1.0, "s2": 1.0, "f1": 0.0, "f2": 0.0, "a1": 0.0, "a2": 0.0, "b1": 1.0}  # b2 unlabeled


def _episodes(s2_has_action=True):
    return {
        "s1": [_step(0, "o", SEARCH), _step(1, "o", FINISH)],
        "s2": [_step(0, "o", FINISH if s2_has_action else None)],
        "f1": [_step(0, "o", SEARCH), _step(1, "o", SEARCH)],
        "f2": [_step(0, "o", SEARCH)],
        "a1": [_step(0, "o", SEARCH)],
        "a2": [_step(0, "o", SEARCH)],
        "b1": [_step(0, "o", FINISH)],
        "b2": [_step(0, "o", FINISH)],
    }


def test_group_outcomes_splits_groups_and_skips_unlabeled():
    assert group_outcomes(UIDS, TRAJ_UIDS, SUCCESS) == {
        "g1": (["s1", "s2"], ["f1", "f2"]),
        "g2": ([], ["a1", "a2"]),
        "g3": (["b1"], []),
    }


def test_select_sibling_references_uses_shortest_success_for_failed_siblings_only():
    refs, metrics = select_sibling_references(_episodes(), UIDS, TRAJ_UIDS, SUCCESS)

    assert set(refs) == {"f1", "f2"}
    assert refs["f1"] is refs["f2"]
    assert refs["f1"].traj_uid == "s2"
    assert refs["f1"].text == f"Step 1 | obs: o | action: {FINISH}"
    assert (refs["f1"].rendered_steps, refs["f1"].total_steps) == (1, 1)
    assert metrics["seed/sibling/groups_total"] == 3.0
    assert metrics["seed/sibling/mixed_group_ratio"] == pytest.approx(1 / 3)
    assert metrics["seed/sibling/allfail_group_ratio"] == pytest.approx(1 / 3)
    assert metrics["seed/sibling/allsuccess_group_ratio"] == pytest.approx(1 / 3)
    assert metrics["seed/sibling/failed_trajs_masked"] == 2.0
    assert metrics["seed/sibling/ref_steps_mean"] == 1.0
    assert metrics["seed/sibling/ref_chars_mean"] == float(len(refs["f1"].text))


def test_select_sibling_references_skips_empty_skeletons_and_breaks_ties_by_reward():
    refs, _ = select_sibling_references(_episodes(s2_has_action=False), UIDS, TRAJ_UIDS, SUCCESS)
    assert refs["f1"].traj_uid == "s1" and refs["f1"].rendered_steps == 2

    episodes = _episodes()
    episodes["s1"] = [_step(0, "o", SEARCH)]  # same length as s2 -> reward decides
    refs, _ = select_sibling_references(episodes, UIDS, TRAJ_UIDS, SUCCESS, traj_rewards={"s1": 11.0, "s2": 10.0})
    assert refs["f1"].traj_uid == "s1"
    refs, _ = select_sibling_references(episodes, UIDS, TRAJ_UIDS, SUCCESS, traj_rewards={"s1": 10.0, "s2": 10.0})
    assert refs["f1"].traj_uid == "s1"  # then traj_uid order

    refs, metrics = select_sibling_references(_episodes(), UIDS, TRAJ_UIDS, {})
    assert refs == {} and metrics["seed/sibling/groups_total"] == 0.0


# ---------------------------------------------------------------- routing weights


def test_compute_pg_row_weights_only_down_weights_failed_rows_of_mixed_groups():
    weights, metrics = compute_pg_row_weights(UIDS, TRAJ_UIDS, SUCCESS, failed_weight=0.25)

    np.testing.assert_allclose(weights, [1, 1, 1, 0.25, 0.25, 0.25, 1, 1, 1, 1])
    assert weights.dtype == np.float32
    assert metrics["seed/route/rows_pg_weighted_ratio"] == pytest.approx(0.3)
    assert metrics["seed/route/rows_pg_full_ratio"] == pytest.approx(0.7)
    assert metrics["seed/route/mixed_group_ratio"] == pytest.approx(1 / 3)
    assert metrics["seed/route/failed_weight"] == 0.25

    unlabeled, metrics = compute_pg_row_weights(UIDS, TRAJ_UIDS, {}, failed_weight=0.0)
    assert (unlabeled == 1.0).all() and metrics["seed/route/mixed_group_ratio"] == 0.0


def test_policy_loss_respects_fractional_row_weights():
    old = torch.full((2, 3), -1.0)
    adv = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 1.0, 0.0]])
    lp = torch.full((2, 3), -1.1, requires_grad=True)
    weighted_mask = torch.ones(2, 3) * torch.tensor([[1.0], [0.0]])

    loss, *_ = compute_policy_loss(old, lp, adv, weighted_mask, cliprange=0.2)
    loss.backward()
    assert torch.all(lp.grad[1] == 0) and torch.any(lp.grad[0] != 0)

    lp_row0 = torch.full((1, 3), -1.1, requires_grad=True)
    loss_row0, *_ = compute_policy_loss(old[:1], lp_row0, adv[:1], torch.ones(1, 3), cliprange=0.2)
    torch.testing.assert_close(loss, loss_row0)


def test_actor_pg_row_weight_scales_only_the_policy_gradient(run_update):
    weight = torch.tensor([1.0, 0.0])
    result = run_update(dominance="none", gen_coef=0.0, extra_tensors={"pg_row_weight": weight})
    data, mask, config = result.batch.batch, result.mask, result.config

    # Reference: PG on the weighted mask, spec OPD and KL on the plain mask.
    lp = torch.full((2, 4), -2.0, requires_grad=True)
    pg_mask = mask.to(torch.float32) * weight[:, None]
    loss = compute_policy_loss(data["old_log_probs"], lp, data["advantages"], pg_mask, cliprange=0.2)[0]
    spec = compute_opd_loss(lp, data["teacher_log_prob"], mask, data["teacher_signal_mask"], gate_beta=config.opd_gate_beta)
    loss = loss + config.opd_loss_coef * spec[0]
    kl = kl_penalty(lp, data["ref_log_prob"], config.kl_loss_type)
    loss = loss + config.kl_loss_coef * agg_loss(kl, mask, config.loss_agg_mode)
    loss.backward()

    torch.testing.assert_close(result.gradient, lp.grad, rtol=0, atol=0)
    assert result.metrics["actor/pg_row_weight_mean"][0] == pytest.approx(0.5)
    # The OPD call still received the un-weighted response mask.
    torch.testing.assert_close(result.calls[0]["response_mask"], mask, rtol=0, atol=0)


def test_actor_without_pg_row_weight_logs_no_routing_metric(run_update):
    result = run_update(dominance="none", gen_coef=0.0)
    assert "actor/pg_row_weight_mean" not in result.metrics
