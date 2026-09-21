"""Pure parts of examples/agentstream_trainer/delta_test.py (materials, prompt hook, paired summary)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from seed.prompting import build_augmented_observation_text

_SPEC = importlib.util.spec_from_file_location(
    "delta_test", Path(__file__).resolve().parents[2] / "examples" / "agentstream_trainer" / "delta_test.py"
)
delta_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(delta_test)


def _record(task, rollout_id, success, num_steps, *, slug="bfcl", score=None, reset_error=False):
    steps = [
        {
            "step_idx": i,
            "observation": f"obs {i}",
            "model_response": f"<think>step {i}</think>\n<action>{{\"name\": \"a{i}\", \"arguments\": {{}}}}</action>",
            "info": {"is_action_valid": i != 1 or num_steps < 3},
        }
        for i in range(num_steps)
    ]
    return {
        "task_id": f"{slug}/{task}", "benchmark_task_id": task, "task_type": slug, "rollout_id": rollout_id,
        "success": success, "score": 11.0 if success else 0.0, "num_steps": num_steps, "reset_error": reset_error, "steps": steps,
    } | ({"score": score} if score is not None else {})


C0 = [
    _record("t1", 0, False, 4), _record("t1", 1, True, 3), _record("t1", 2, True, 2), _record("t1", 3, True, 2, score=10.5),
    _record("t2", 0, False, 3), _record("t2", 1, False, 5),                       # all fail -> no material
    _record("t3", 0, True, 2), _record("t3", 1, True, 2),                         # all success -> no material
    _record("t4", 0, True, 2, reset_error=True), _record("t4", 1, False, 3),      # only usable rollout fails
    _record("t5", 0, True, 3, slug="tau2"), _record("t5", 1, False, 6, slug="tau2"),
]


def test_build_materials_selects_mixed_tasks_and_the_shortest_best_success():
    materials = delta_test.build_materials(C0, obs_chars=20, response_chars=80, max_chars=6000)
    assert [m["task_id"] for m in materials] == ["bfcl/t1", "tau2/t5"]
    t1 = materials[0]
    assert (t1["c0_rollouts"], t1["c0_successes"], t1["source_rollout_id"], t1["source_num_steps"]) == (4, 3, 2, 2)
    assert t1["rendered_steps"] == 2 and t1["chars"] == len(t1["text"])
    assert t1["text"].startswith("Step 1 | obs: obs 0 | response: <think>step 0</think> <action>")
    assert materials[1]["slug"] == "tau2" and materials[1]["source_num_steps"] == 3


def test_prompt_hook_matches_the_trainer_teacher_prompt_and_rejects_unknown_tasks():
    materials = delta_test.build_materials(C0)
    hook = delta_test.make_prompt_hook(materials)
    prompt = "You are an expert agent.\nYour task is: buy a mug\nYour current observation is: shop\n\nNow it's your turn to take an action."
    injected = hook(prompt, {"slug": "bfcl", "task_id": "t1", "rollout_id": 0}, 0)
    assert injected == build_augmented_observation_text(observation=prompt, reference_solution=materials[0]["text"])
    assert "Reference Solution" in injected and injected != prompt
    with pytest.raises(KeyError):
        hook(prompt, {"slug": "bfcl", "task_id": "t2", "rollout_id": 0}, 0)


def test_summarize_delta_pairs_tasks_present_in_both_conditions():
    c2 = [
        _record("t1", 0, True, 2), _record("t1", 1, True, 2), _record("t1", 2, False, 4), _record("t1", 3, True, 2),  # 0.75 vs 0.75
        _record("t5", 0, True, 2, slug="tau2"), _record("t5", 1, True, 2, slug="tau2"),                                # 1.0 vs 0.5
        _record("t9", 0, True, 2),  # not in C0 -> ignored
    ]
    summary = delta_test.summarize_delta(C0, c2)
    rows = {r["group"]: r for r in summary["rows"]}
    assert set(rows) == {"all", "bfcl", "tau2"}
    assert rows["tau2"]["tasks"] == 1 and rows["tau2"]["delta"] == pytest.approx(0.5)
    assert rows["bfcl"]["tasks"] == 1 and rows["bfcl"]["delta"] == pytest.approx(0.0)
    assert rows["all"]["tasks"] == 2 and rows["all"]["delta"] == pytest.approx(0.25)
    assert rows["all"]["se"] == pytest.approx(0.25) and (rows["all"]["tasks_up"], rows["all"]["tasks_down"]) == (1, 0)
    assert rows["tau2"]["num_steps_c0"] == pytest.approx(4.5) and rows["tau2"]["num_steps_c2"] == pytest.approx(2.0)
    report = delta_test.format_report(summary, delta_test.build_materials(C0))
    assert "| all | 2 |" in report and "| tau2 | 1 |" in report


def test_episode_stats_counts_invalid_steps_and_response_length():
    stats = delta_test.episode_stats(_record("t1", 0, True, 4))
    assert stats["success"] == 1.0 and stats["num_steps"] == 4.0
    assert stats["invalid_ratio"] == pytest.approx(0.25)
    assert stats["response_chars"] > 0
