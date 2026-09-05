import json

import numpy as np
import pytest

from seed.analysis import infer_task_description
from seed.global_pool import (
    GlobalPoolConfig,
    GlobalSkillPool,
    build_retrieval_query,
    select_admission_candidates,
    skill_id_for,
)
from seed.skill_judge import parse_judge_response


def _unit(x, y):
    vector = np.asarray([x, y], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def _make_pool(**overrides):
    defaults = dict(source="pool", capacity=3, min_sim=0.5, dedup_sim=0.95, ema_alpha=0.5)
    defaults.update(overrides)
    return GlobalSkillPool(GlobalPoolConfig(**defaults))


def _add(pool, text, embedding, task_key="task-a", score=0.9, step=1):
    return pool.add(
        text=text,
        embedding=embedding,
        source={"task_key": task_key, "traj_uid": "t", "global_step": step},
        judge={"score": score, "tag": "other", "reason": ""},
        global_step=step,
    )


def test_add_deduplicates_and_merges_near_duplicates():
    pool = _make_pool()
    assert _add(pool, "Verify state before acting.", _unit(1, 0)) == "added"
    assert _add(pool, "verify   STATE before acting.", _unit(1, 0)) == "duplicate"
    assert _add(pool, "Check state before you act.", _unit(0.999, 0.01)) == "merged"
    assert len(pool) == 1


def test_capacity_eviction_prefers_lowest_gate_ema():
    pool = _make_pool()
    for index, name in enumerate(["a", "b", "c"]):
        _add(pool, f"skill {name}", _unit(np.cos(index), np.sin(index)), task_key=f"task-{name}")
    weak_id = skill_id_for("skill a")
    strong_id = skill_id_for("skill b")
    pool.record_usage(weak_id, 0.05, global_step=2)
    pool.record_usage(strong_id, 0.9, global_step=2)
    assert _add(pool, "skill d", _unit(-1, 0), task_key="task-d") == "added"
    assert not pool.has(weak_id)
    assert pool.has(strong_id)
    assert len(pool) == 3


def test_retrieve_applies_min_sim_and_same_task_exclusion():
    pool = _make_pool(min_sim=0.6)
    _add(pool, "skill a", _unit(1, 0), task_key="task-a")
    _add(pool, "skill b", _unit(0, 1), task_key="task-b")

    hits = pool.retrieve(np.stack([_unit(1, 0.1), _unit(1, 0.1), _unit(1, 1)]), ["task-x", "task-a", "task-z"])
    assert hits[0] is not None and hits[0].skill_id == skill_id_for("skill a")
    # Same-task entries are excluded; the remaining skill b is below min_sim.
    assert hits[1] is None
    # Diagonal query matches both at ~0.707 >= 0.6; top-1 is deterministic by best similarity.
    assert hits[2] is not None


def test_record_usage_updates_ema():
    pool = _make_pool()
    _add(pool, "skill a", _unit(1, 0))
    skill_id = skill_id_for("skill a")
    pool.record_usage(skill_id, 0.4, global_step=1)
    pool.record_usage(skill_id, 0.8, global_step=2)
    hits = pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"])
    assert hits[0].skill_id == skill_id
    metrics = pool.snapshot_metrics()
    assert metrics["seed/global_pool/gate_ema_mean"] == pytest.approx(0.5 * 0.4 + 0.5 * 0.8)


def test_save_and_load_roundtrip(tmp_path):
    path = str(tmp_path / "pool.json")
    pool = _make_pool()
    pool.save_path = path
    _add(pool, "skill a", _unit(1, 0), task_key="task-a")
    pool.record_usage(skill_id_for("skill a"), 0.7, global_step=3)
    pool.save()

    restored = GlobalSkillPool(GlobalPoolConfig(source="pool", capacity=3, min_sim=0.5), save_path=path)
    assert len(restored) == 1
    hits = restored.retrieve(np.stack([_unit(1, 0)]), ["task-z"])
    assert hits[0] is not None and hits[0].text == "skill a"


def test_parse_judge_response_handles_order_and_garbage():
    text = """Here is my audit:
[
 {"idx": 2, "transferable": true, "score": 0.8, "tag": "planning", "reason": "general"},
 {"idx": 1, "transferable": false, "score": 0.2, "tag": "other", "reason": "domain-bound"},
 {"idx": 99, "transferable": true, "score": 1.0},
 "garbage"
]"""
    verdicts = parse_judge_response(text, expected=2)
    assert verdicts[0] is not None and verdicts[0].transferable is False
    assert verdicts[1] is not None and verdicts[1].score == pytest.approx(0.8)

    assert parse_judge_response("no json here", expected=2) == [None, None]


def test_eviction_protects_proven_good_over_never_used():
    pool = _make_pool(capacity=2)
    _add(pool, "skill a", _unit(1, 0), task_key="task-a", step=1)
    _add(pool, "skill b", _unit(0, 1), task_key="task-b", step=2)
    pool.record_usage(skill_id_for("skill a"), 0.9, global_step=3)
    # A validated skill (EMA 0.9) must outlive the never-injected one (0.5 prior).
    assert _add(pool, "skill c", _unit(-1, 0), task_key="task-c", step=4) == "added"
    assert pool.has(skill_id_for("skill a"))
    assert not pool.has(skill_id_for("skill b"))


def test_eviction_breaks_never_used_ties_by_staleness():
    pool = _make_pool(capacity=2)
    _add(pool, "skill old", _unit(1, 0), task_key="task-a", step=1)
    _add(pool, "skill new", _unit(0, 1), task_key="task-b", step=5)
    assert _add(pool, "skill c", _unit(-1, 0), task_key="task-c", step=6) == "added"
    assert not pool.has(skill_id_for("skill old"))
    assert pool.has(skill_id_for("skill new"))


def _saved_pool(tmp_path, entries=(("skill a", (1, 0), 1), ("skill b", (0, 1), 5))):
    path = str(tmp_path / "pool.json")
    pool = _make_pool()
    pool.save_path = path
    for text, (x, y), step in entries:
        _add(pool, text, _unit(x, y), task_key=f"task-{text[-1]}", step=step)
    pool.save()
    return path


def test_load_skips_stale_npy_sidecar(tmp_path):
    path = _saved_pool(tmp_path)
    matrix = np.load(path + ".npy")
    np.save(path + ".npy", matrix[:1])  # crash between JSON and sidecar writes
    restored = GlobalSkillPool(GlobalPoolConfig(source="pool"), save_path=path)
    assert len(restored) == 0


def test_load_rejects_embedder_mismatch(tmp_path):
    path = _saved_pool(tmp_path)
    restored = GlobalSkillPool(GlobalPoolConfig(source="pool", embed_model="other/encoder"), save_path=path)
    assert len(restored) == 0


def test_load_survives_corrupt_json(tmp_path):
    path = str(tmp_path / "pool.json")
    with open(path, "w") as f:
        f.write("not json {{{")
    restored = GlobalSkillPool(GlobalPoolConfig(source="pool"), save_path=path)
    assert len(restored) == 0


def test_load_respects_resume_step_and_load_existing(tmp_path):
    path = _saved_pool(tmp_path)  # entries admitted at steps 1 and 5

    resumed = GlobalSkillPool(GlobalPoolConfig(source="pool"), save_path=path, max_global_step=3)
    assert len(resumed) == 1 and resumed.has(skill_id_for("skill a"))

    fresh = GlobalSkillPool(GlobalPoolConfig(source="pool"), save_path=path, load_existing=False)
    assert len(fresh) == 0


def test_build_retrieval_query_keeps_first_obs_visible():
    assert build_retrieval_query("Book a flight", "Hi, I need help") == "Book a flight\nHi, I need help"
    assert build_retrieval_query("Task only", "") == "Task only"
    query = build_retrieval_query("T" * 5000, "OPENING LINE OF THE USER")
    head, tail = query.split("\n", 1)
    # A long task must not push the observation (tau2's only task identity) out.
    assert len(head) == 600 and tail == "OPENING LINE OF THE USER"


def test_select_admission_candidates_dedups_per_task_and_ranks_by_gap():
    c = [{"task_key": key, "skill": key + s} for key, s in
         [("t1", "x"), ("t1", "y"), ("t2", "x"), ("t3", "x"), ("t4", "x")]]
    scored = [(c[0], 0.2), (c[1], 0.9), (c[2], -0.1), (c[3], 0.5), (c[4], None)]

    kept = select_admission_candidates(scored, limit=10)
    # t1 keeps its best copy, t2 (gap<=0) drops, ranking is by gap, None ranks last.
    assert [item["skill"] for item in kept] == ["t1y", "t3x", "t4x"]

    assert [item["skill"] for item in select_admission_candidates(scored, limit=2)] == ["t1y", "t3x"]
    assert select_admission_candidates([], limit=4) == []


def test_parse_judge_response_survives_reasoning_preamble():
    preamble = "Let me audit skills [1] and [2]. Skill [1] is about state checks.\n"
    payload = [
        {"idx": 1, "transferable": True, "score": 0.9, "tag": "planning", "reason": "use the [verify] pattern"},
        {"idx": 2, "transferable": False, "score": 0.1, "tag": "other", "reason": "domain-bound"},
    ]
    verdicts = parse_judge_response(preamble + json.dumps(payload), expected=2)
    assert verdicts[0] is not None and verdicts[0].transferable is True
    assert verdicts[1] is not None and verdicts[1].transferable is False

    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert parse_judge_response(fenced, expected=2)[0] is not None


def test_infer_task_description_reads_step_fields():
    steps = [
        {"observation": "some ambient text"},
        {"task_description": "  Find the   cheapest flight  "},
    ]
    assert infer_task_description(steps) == "Find the cheapest flight"
    assert infer_task_description([{"observation": "Your task is to: put a clean mug on the desk\nGo."}]) == "put a clean mug on the desk"
