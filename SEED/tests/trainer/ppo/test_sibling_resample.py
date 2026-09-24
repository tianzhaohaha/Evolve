"""seed/resample.py and the group-level reference selector in seed/sibling.py (pure, no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from seed.prompting import build_augmented_observation_text
from seed.resample import (
    augment_observations,
    build_resample_requests,
    reconcile_non_tensor_keys,
    summarize_resample_batch,
)
from seed.sibling import (
    SiblingReference,
    build_action_skeleton,
    build_reference_solution,
    select_group_references,
    select_sibling_references,
)


def _step(idx, obs, action="{\"name\": \"go\"}"):
    return {"step_index": idx, "observation": obs, "response": f"<think>t{idx}</think><action>{action}</action>", "action_valid": True}


UIDS = ["g1"] * 6 + ["g2"] * 2 + ["g3"] * 2
TRAJ_UIDS = ["s1", "s1", "s2", "f1", "f1", "f2", "a1", "a2", "b1", "b2"]
SUCCESS = {"s1": 1.0, "s2": 1.0, "f1": 0.0, "f2": 0.0, "a1": 0.0, "a2": 0.0, "b1": 1.0}  # b2 unlabeled
EPISODES = {
    "s1": [_step(0, "o0"), _step(1, "o1"), _step(2, "o2")],
    "s2": [_step(0, "o0"), _step(1, "o1")],  # shortest success of g1
    "f1": [_step(0, "o0")], "f2": [_step(0, "o0")], "a1": [_step(0, "o0")], "a2": [_step(0, "o0")], "b1": [_step(0, "o0")],
}


# ---- select_group_references ----------------------------------------------------------------

def test_select_group_references_keys_mixed_groups_by_uid_and_uses_the_render():
    refs, metrics = select_group_references(EPISODES, UIDS, TRAJ_UIDS, SUCCESS, render=build_reference_solution)
    assert list(refs) == ["g1"] and refs["g1"].traj_uid == "s2" and refs["g1"].total_steps == 2
    assert refs["g1"].text == build_reference_solution(EPISODES["s2"])[0] and "response:" in refs["g1"].text
    assert metrics == {
        "groups_total": 3.0, "mixed_groups": 1.0, "mixed_group_ratio": pytest.approx(1 / 3),
        "allfail_group_ratio": pytest.approx(1 / 3), "allsuccess_group_ratio": pytest.approx(1 / 3),
        "ref_chars_mean": float(len(refs["g1"].text)), "ref_steps_mean": 2.0,
    }
    # a renderer that cannot show the shortest success falls back to the next candidate; none -> no reference
    refs, _ = select_group_references(EPISODES, UIDS, TRAJ_UIDS, SUCCESS, render=lambda steps: ("", 0) if len(steps) == 2 else ("x", 1))
    assert refs["g1"].traj_uid == "s1"
    refs, metrics = select_group_references(EPISODES, UIDS, TRAJ_UIDS, SUCCESS, render=lambda steps: ("", 0))
    assert refs == {} and metrics["mixed_group_ratio"] == pytest.approx(1 / 3) and metrics["ref_chars_mean"] == 0.0


def test_select_sibling_references_wrapper_matches_the_group_selector():
    refs, metrics = select_sibling_references(EPISODES, UIDS, TRAJ_UIDS, SUCCESS)
    group_refs, _ = select_group_references(EPISODES, UIDS, TRAJ_UIDS, SUCCESS, render=build_action_skeleton)
    assert set(refs) == {"f1", "f2"} and refs["f1"] is refs["f2"] and refs["f1"] == group_refs["g1"]
    assert metrics["seed/sibling/failed_trajs_masked"] == 2.0 and metrics["seed/sibling/groups_total"] == 3.0


# ---- resample requests / augmentation --------------------------------------------------------

def _ref(uid, text="ref " + "x" * 10, steps=2):
    return SiblingReference(f"succ-{uid}", text, steps, steps)


def test_build_resample_requests_orders_by_sample_id_caps_and_skips_groups_without_identity():
    references = {"g7": _ref("g7"), "g2": _ref("g2"), "g5": _ref("g5"), "g9": _ref("g9")}
    task_refs = {"g7": (7, "bfcl", "t7"), "g2": (2, "tau2", "t2"), "g5": (5, "bfcl", "t5")}  # g9 has no identity
    requests = build_resample_requests(references, task_refs, max_groups=2)
    assert [(r["uid"], r["sample_id"], r["task_slug"], r["task_id"]) for r in requests] == [("g2", 2, "tau2", "t2"), ("g5", 5, "bfcl", "t5")]
    assert requests[0]["reference_solution"] == references["g2"].text and requests[0]["reference_traj_uid"] == "succ-g2"
    assert build_resample_requests(references, task_refs, max_groups=0) == []
    assert len(build_resample_requests(references, task_refs, max_groups=10)) == 3


def test_augment_observations_injects_only_non_empty_references():
    prompt = "You are an agent.\nYour task is: buy a mug\nYour current observation is: shop\n\nNow it's your turn to take an action."
    out = augment_observations([prompt, prompt], ["", "step 1 | go"])
    assert out[0] == prompt
    assert out[1] == build_augmented_observation_text(observation=prompt, reference_solution="step 1 | go")
    assert "Reference Solution" in out[1] and "step 1 | go" in out[1]


# ---- key reconciliation / summary ------------------------------------------------------------

def _is_metric(key):
    return "success_rate" in key or key.endswith("_score") or key.endswith("_rate")


def _objects(values):  # 1-D object array of arbitrary Python values (the rollout loop's _object_array)
    array = np.empty(len(values), dtype=object)
    for i, v in enumerate(values):
        array[i] = v
    return array


def test_reconcile_non_tensor_keys_aligns_to_the_main_batch_and_overwrites_broadcast_columns():
    main = {
        "uid": np.array(["a", "a", "b"], dtype=object), "step_num": np.array([0, 1, 0]),
        "success_rate": np.array([0.5, 0.5, 0.5], dtype=np.float64), "tau2_score": np.array([0.2, 0.2, 0.2]),
        "history": _objects([["h0"], ["h0", "h1"], []]),
    }
    extra = {
        "uid": np.array(["c", "c"], dtype=object), "step_num": np.array([0, 1]),
        "success_rate": np.array([1.0, 1.0]), "bfcl_score": np.array([1.0, 1.0]),  # bfcl only appeared in the resample pass
    }
    columns, dropped, filled = reconcile_non_tensor_keys(main, extra, is_broadcast_key=_is_metric)
    assert set(columns) == set(main) and dropped == ["bfcl_score"] and filled == ["success_rate", "tau2_score", "history"]
    assert list(columns["uid"]) == ["c", "c"] and list(columns["step_num"]) == [0, 1]
    assert list(columns["success_rate"]) == [0.5, 0.5] and columns["success_rate"].dtype == np.float64
    assert list(columns["tau2_score"]) == [0.2, 0.2] and list(columns["history"]) == [["h0"], ["h0"]]
    assert all(len(v) == 2 for v in columns.values())


def test_summarize_resample_batch_counts_trajectories_groups_and_uniform_groups():
    uids = ["g1"] * 4 + ["g2"] * 2
    trajs = ["t1", "t1", "t2", "t3", "t4", "t5"]
    rewards = [10.0, 10.0, 0.0, 10.0, 0.0, 0.0]
    m = summarize_resample_batch(uids, trajs, rewards, threshold=1.0)
    assert m == {"rows": 6.0, "trajs": 5.0, "groups": 2.0, "success_rate": pytest.approx(2 / 5), "uniform_group_ratio": 0.5}
    assert summarize_resample_batch([], [], [], threshold=1.0) == {"rows": 0.0, "trajs": 0.0, "groups": 0.0, "success_rate": 0.0, "uniform_group_ratio": 0.0}


# ---- pool requests / request sections (global-skill channel 2) --------------------------------

def test_build_pool_requests_orders_by_sample_id_caps_and_marks_the_section():
    from seed.resample import POOL_SECTION, SIBLING_SECTION, build_pool_requests

    hits = {"g7": ("id7", "skill 7"), "g2": ("id2", "skill 2"), "g9": ("id9", "skill 9")}
    task_refs = {"g7": (7, "bfcl", "t7"), "g2": (2, "tau2", "t2"), "g5": (5, "bfcl", "t5")}
    requests = build_pool_requests(["g7", "g5", "g2", "g9"], hits, task_refs, max_groups=5)  # g5: no hit; g9: no identity
    assert [(r["uid"], r["sample_id"], r["task_slug"], r["section"], r[POOL_SECTION], r["skill_id"]) for r in requests] == [
        ("g2", 2, "tau2", POOL_SECTION, "skill 2", "id2"), ("g7", 7, "bfcl", POOL_SECTION, "skill 7", "id7"),
    ]
    assert [r["uid"] for r in build_pool_requests(["g7", "g2"], hits, task_refs, max_groups=1)] == ["g2"]
    assert build_pool_requests([], hits, task_refs, max_groups=3) == []
    sibling = build_resample_requests({"g2": _ref("g2")}, task_refs, max_groups=1)[0]
    assert sibling["section"] == SIBLING_SECTION == "reference_solution" and sibling[SIBLING_SECTION] == _ref("g2").text
    for request in requests + [sibling]:  # every request carries its context under its section key
        assert request[request["section"]]


def test_augment_observations_routes_each_slot_to_its_section():
    from seed.resample import POOL_SECTION, SIBLING_SECTION

    out = augment_observations(["obs"] * 3, ["ref", "skill", ""], [SIBLING_SECTION, POOL_SECTION, POOL_SECTION])
    assert out[0] == build_augmented_observation_text(observation="obs", reference_solution="ref")
    assert out[1] == build_augmented_observation_text(observation="obs", global_skill="skill")
    assert "Reference Solution" not in out[1] and "General Skill" in out[1] and out[2] == "obs"
    assert augment_observations(["obs"], ["ref"]) == augment_observations(["obs"], ["ref"], [SIBLING_SECTION])
