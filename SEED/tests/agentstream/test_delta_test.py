"""Pure parts of examples/agentstream_trainer/delta_test.py (materials for C2 / C4 / C3, prompt hook, paired summary)."""

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


def _record(task, rollout_id, success, num_steps, *, slug="bfcl", score=None, reset_error=False, description=None):
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
        "task_description": description or f"task {task}",
        "success": success, "score": 11.0 if success else 0.0, "num_steps": num_steps, "reset_error": reset_error, "steps": steps,
    } | ({"score": score} if score is not None else {})


C0 = [
    _record("t1", 0, False, 4), _record("t1", 1, True, 3), _record("t1", 2, True, 2), _record("t1", 3, True, 2, score=10.5),
    _record("t2", 0, False, 3), _record("t2", 1, False, 5),                       # all fail -> no C2 material
    _record("t3", 0, True, 2), _record("t3", 1, True, 2),                         # all success -> never a target
    _record("t4", 0, True, 2, reset_error=True), _record("t4", 1, False, 3),      # only usable rollout fails
    _record("t5", 0, True, 3, slug="tau2"), _record("t5", 1, False, 6, slug="tau2"),
]
TRAJ = delta_test.trajectory_renderer(obs_chars=20, response_chars=80, max_chars=6000)


def test_build_materials_self_selects_mixed_tasks_and_the_shortest_best_success():
    materials = delta_test.build_materials(C0, render=TRAJ)
    assert [m["task_id"] for m in materials] == ["bfcl/t1", "tau2/t5"]
    t1 = materials[0]
    assert (t1["c0_rollouts"], t1["c0_successes"], t1["source_rollout_id"], t1["source_num_steps"]) == (4, 3, 2, 2)
    assert (t1["source"], t1["material"], t1["source_task_id"], t1["similarity"]) == ("self", "trajectory", "bfcl/t1", 1.0)
    assert t1["rendered_steps"] == 2 and t1["chars"] == len(t1["text"])
    assert t1["text"].startswith("Step 1 | obs: obs 0 | response: <think>step 0</think> <action>")
    assert materials[1]["slug"] == "tau2" and materials[1]["source_num_steps"] == 3


def test_select_neighbor_prefers_the_most_similar_solved_task_of_the_same_benchmark():
    c0 = [
        _record("a", 0, False, 3, description="book a flight to paris in may"),
        _record("b", 0, True, 3, description="book a flight to rome in may"),          # closest solved sibling
        _record("c", 0, True, 2, description="cancel the hotel booking"),
        _record("d", 0, False, 2, description="book a flight to paris in may"),        # identical text but no success
        _record("e", 0, True, 2, slug="tau2", description="book a flight to paris in may"),  # other benchmark
    ]
    by_task = delta_test.group_by_task(c0)
    other, sim = delta_test.select_neighbor("bfcl/a", by_task)
    assert other == "bfcl/b" and 0.5 < sim < 1.0
    assert delta_test.select_neighbor("tau2/e", by_task) is None
    assert delta_test.cosine(delta_test._bag("x y"), delta_test._bag("x y")) == pytest.approx(1.0)
    assert delta_test.cosine(delta_test._bag(""), delta_test._bag("x")) == 0.0


def test_build_materials_neighbor_targets_unsolved_tasks_and_drops_the_final_step():
    materials = delta_test.build_materials(C0, render=TRAJ, source="neighbor", drop_final_step=True)
    by_id = {m["task_id"]: m for m in materials}
    # t1 (mixed), t2 (all fail), t4 (fail) are targets; t3 (all success) is not; t5 has no same-benchmark partner.
    assert set(by_id) == {"bfcl/t1", "bfcl/t2", "bfcl/t4"}
    for m in by_id.values():
        assert m["source"] == "neighbor" and m["source_task_id"] != m["task_id"]
    # descriptions are all "task tX" -> equal similarity -> lowest task_id with a success: t1 for t2/t4, t3 for t1.
    assert by_id["bfcl/t2"]["source_task_id"] == "bfcl/t1" and by_id["bfcl/t1"]["source_task_id"] == "bfcl/t3"
    # t1's shortest success has 2 steps; with the final step dropped only step 1 is rendered.
    assert by_id["bfcl/t2"]["source_num_steps"] == 2 and by_id["bfcl/t2"]["rendered_steps"] == 1
    assert "Step 2" not in by_id["bfcl/t2"]["text"] and "Step 1 | obs: obs 0" in by_id["bfcl/t2"]["text"]
    with pytest.raises(ValueError):
        delta_test.build_materials(C0, render=TRAJ, source="pool")


def test_skill_material_shares_neighbours_and_keeps_parse_failures():
    calls = []

    def build_record(*, trajectory, skill_endpoint, skill_mode, max_step_skills):  # stub of build_candidate_skill_record
        calls.append((trajectory["task_id"], trajectory["rollout_id"], skill_endpoint, skill_mode, max_step_skills))
        ok = trajectory["task_id"] == "bfcl/t1"
        return {
            "analysis_prompt": {"messages": [{"role": "user", "content": "Analyze " + trajectory["task_id"]}]},
            "episode_skill": "  check the cart before paying  " if ok else "",
            "episode_summary": "bought a mug" if ok else "",
            "parse_ok": ok,
            "analysis_error": None if ok else "ValueError: No JSON object found",
        }

    render = delta_test.skill_renderer("endpoint", build_record=build_record)
    materials = delta_test.build_materials(C0, render=render, **delta_test.CONDITIONS["C3"])
    by_id = {m["task_id"]: m for m in materials}
    # Same targets and neighbours as C4; the neighbour shared by t2 / t4 is analysed once, with the trainer's settings.
    c4 = delta_test.build_materials(C0, render=TRAJ, **delta_test.CONDITIONS["C4"])
    assert [(m["task_id"], m["source_task_id"]) for m in materials] == [(m["task_id"], m["source_task_id"]) for m in c4]
    assert calls == [("bfcl/t3", 0, "endpoint", "episode_only", 0), ("bfcl/t1", 2, "endpoint", "episode_only", 0)]
    t2 = by_id["bfcl/t2"]
    assert (t2["material"], t2["text"], t2["parse_ok"], t2["chars"]) == ("skill", "check the cart before paying", True, 28)
    assert (t2["rendered_steps"], t2["episode_summary"], t2["prompt_chars"]) == (2, "bought a mug", len("Analyze bfcl/t1"))
    t1 = by_id["bfcl/t1"]  # neighbour t3 -> parse failure -> kept with empty text, excluded from sampling
    assert t1["text"] == "" and t1["parse_ok"] is False and "ValueError" in t1["analysis_error"]
    assert [m["task_id"] for m in delta_test.active_materials(materials)] == ["bfcl/t2", "bfcl/t4"]

    def broken(**kwargs):  # an exception in the analyzer call becomes a failed row, not an aborted phase
        raise RuntimeError("endpoint down")

    rows = delta_test.build_materials(C0, render=delta_test.skill_renderer("endpoint", build_record=broken), **delta_test.CONDITIONS["C3"])
    assert [(m["text"], m["parse_ok"], m["analysis_error"]) for m in rows] == [("", False, "RuntimeError: endpoint down")] * 3


def test_prompt_hook_injects_each_material_into_the_trainer_section():
    prompt = "You are an expert agent.\nYour task is: buy a mug\nYour current observation is: shop\n\nNow it's your turn to take an action."
    materials = delta_test.build_materials(C0, render=TRAJ)
    hook = delta_test.make_prompt_hook(materials)
    injected = hook(prompt, {"slug": "bfcl", "task_id": "t1", "rollout_id": 0}, 0)
    assert injected == build_augmented_observation_text(observation=prompt, reference_solution=materials[0]["text"])
    assert "Reference Solution" in injected and injected != prompt
    with pytest.raises(KeyError):
        hook(prompt, {"slug": "bfcl", "task_id": "t2", "rollout_id": 0}, 0)
    skill = [{"task_id": "bfcl/t1", "material": "skill", "text": "check the cart before paying"},
             {"task_id": "bfcl/t2", "material": "skill", "text": ""}]
    injected = delta_test.make_prompt_hook(skill)(prompt, {"slug": "bfcl", "task_id": "t1", "rollout_id": 0}, 0)
    assert injected == build_augmented_observation_text(observation=prompt, episode_skill="check the cart before paying")
    assert "Episode-Level Skill" in injected and "Reference Solution" not in injected
    with pytest.raises(KeyError):  # empty-text rows are not sampled
        delta_test.make_prompt_hook(skill)(prompt, {"slug": "bfcl", "task_id": "t2", "rollout_id": 0}, 0)


def test_summarize_delta_pairs_tasks_and_reports_c0_strata():
    cx = [
        _record("t1", 0, True, 2), _record("t1", 1, True, 2), _record("t1", 2, False, 4), _record("t1", 3, True, 2),  # 0.75 vs 0.75
        _record("t2", 0, True, 2), _record("t2", 1, False, 3),                                                          # 0.5 vs 0.0 (all-fail stratum)
        _record("t5", 0, True, 2, slug="tau2"), _record("t5", 1, True, 2, slug="tau2"),                                # 1.0 vs 0.5
        _record("t9", 0, True, 2),  # not in C0 -> ignored
    ]
    summary = delta_test.summarize_delta(C0, cx)
    rows = {r["group"]: r for r in summary["rows"]}
    assert set(rows) == {"all", "bfcl", "tau2", "c0_all_fail", "c0_mixed"}
    assert rows["tau2"]["tasks"] == 1 and rows["tau2"]["delta"] == pytest.approx(0.5)
    assert rows["bfcl"]["tasks"] == 2 and rows["bfcl"]["delta"] == pytest.approx(0.25)
    assert rows["c0_all_fail"]["tasks"] == 1 and rows["c0_all_fail"]["delta"] == pytest.approx(0.5)
    assert rows["c0_mixed"]["tasks"] == 2 and rows["c0_mixed"]["delta"] == pytest.approx(0.25)
    assert rows["all"]["tasks"] == 3 and rows["all"]["delta"] == pytest.approx(1.0 / 3)
    assert (rows["all"]["tasks_up"], rows["all"]["tasks_down"]) == (2, 0)
    assert rows["tau2"]["num_steps_c0"] == pytest.approx(4.5) and rows["tau2"]["num_steps_cx"] == pytest.approx(2.0)
    strata = {t["task_id"]: t["stratum"] for t in summary["per_task"]}
    assert strata == {"bfcl/t1": "c0_mixed", "bfcl/t2": "c0_all_fail", "tau2/t5": "c0_mixed"}
    c4 = delta_test.build_materials(C0, render=TRAJ, **delta_test.CONDITIONS["C4"])
    c3 = [dict(m, material="skill", text="" if m["task_id"] == "bfcl/t4" else "rule", chars=4) for m in c4]
    report = delta_test.format_report([("C4", summary, c4), ("C3", summary, c3)])
    assert "## C4: source=neighbor, material=trajectory, final step dropped=True" in report
    assert "Tasks with a reference: 3;" in report
    assert "## C3: source=neighbor, material=skill, final step dropped=False" in report
    assert "Tasks with a reference: 2 (of 3 attempted, skill parse rate 0.67)" in report
    assert "| all | 3 |" in report and "| c0_all_fail | 1 |" in report


def test_episode_stats_counts_invalid_steps_and_response_length():
    stats = delta_test.episode_stats(_record("t1", 0, True, 4))
    assert stats["success"] == 1.0 and stats["num_steps"] == 4.0
    assert stats["invalid_ratio"] == pytest.approx(0.25)
    assert stats["response_chars"] > 0
