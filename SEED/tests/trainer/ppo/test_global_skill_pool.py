import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

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

    results = pool.retrieve(np.stack([_unit(1, 0.1), _unit(1, 0.1), _unit(1, 1)]), ["task-x", "task-a", "task-z"])
    assert results[0].hit is not None and results[0].hit.skill_id == skill_id_for("skill a")
    # Same-task entries are excluded; the remaining skill b is below min_sim.
    assert results[1].hit is None
    assert results[1].top_similarity == pytest.approx(float(np.dot(_unit(1, 0.1), _unit(0, 1))))
    # Diagonal query matches both at ~0.707 >= 0.6; top-1 is deterministic by best similarity.
    assert results[2].hit is not None


def test_record_usage_updates_ema():
    pool = _make_pool()
    _add(pool, "skill a", _unit(1, 0))
    skill_id = skill_id_for("skill a")
    pool.record_usage(skill_id, 0.4, global_step=1)
    pool.record_usage(skill_id, 0.8, global_step=2)
    results = pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"])
    assert results[0].hit.skill_id == skill_id
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
    results = restored.retrieve(np.stack([_unit(1, 0)]), ["task-z"])
    assert results[0].hit is not None and results[0].hit.text == "skill a"


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


@pytest.mark.parametrize("policy", ["gate_ema", "window", "lru"])
@pytest.mark.parametrize("resume_step", [None, 0, 9])
def test_load_clamps_future_recency_only_for_bounded_lru_resume(tmp_path, policy, resume_step):
    pool = _make_pool(capacity=5)
    for index, (admitted, used) in enumerate([(0, 20), (1, 4), (2, 9), (3, 20), (15, 20)]):
        text = f"skill {index}"
        _add(pool, text, np.eye(5, dtype=np.float32)[index], step=admitted)
        pool.record_usage(skill_id_for(text), 0.1 * (index + 1), global_step=used)
    path = str(tmp_path / "pool.json")
    pool.save(path)

    restored = GlobalSkillPool(
        GlobalPoolConfig(source="pool", capacity=5, evict_policy=policy),
        save_path=path, max_global_step=resume_step,
    )
    expected = {
        sid: deepcopy(entry) for sid, entry in pool._entries.items()
        if resume_step is None or entry["source"]["global_step"] <= resume_step
    }
    if policy == "lru" and resume_step is not None:
        for entry in expected.values():
            entry["stats"]["last_used_step"] = min(entry["stats"]["last_used_step"], resume_step)
    # Only LRU recency changes: EMA, usage counts, source metadata and vectors stay intact.
    assert restored._entries == expected
    assert restored._embeddings.keys() == expected.keys()
    for sid in expected:
        np.testing.assert_array_equal(restored._embeddings[sid], pool._embeddings[sid])


def test_lru_resume_restores_eviction_order_after_new_hits(tmp_path):
    pool = _make_pool(capacity=2, evict_policy="lru")
    _add(pool, "skill a", _unit(1, 0), step=1)
    _add(pool, "skill b", _unit(0, 1), step=2)
    # The eagerly saved pool ran beyond the model checkpoint at step 9.
    pool.retrieve(np.stack([_unit(0, 1)]), ["other-task"], current_step=12)
    pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"], current_step=20)
    path = str(tmp_path / "pool.json")
    pool.save(path)
    restored = GlobalSkillPool(pool.config, save_path=path, max_global_step=9)

    for text, vector, step in [("skill a", _unit(1, 0), 10), ("skill b", _unit(0, 1), 11)]:
        hit = restored.retrieve(np.stack([vector]), ["other-task"], current_step=step)[0].hit
        assert hit.skill_id == skill_id_for(text)
        restored.record_usage(hit.skill_id, 0.7, global_step=step)
    _add(restored, "skill c", _unit(-1, 0), step=11)
    assert not restored.has(skill_id_for("skill a"))
    assert restored.has(skill_id_for("skill b"))
    assert restored.has(skill_id_for("skill c"))


def test_build_retrieval_query_keeps_first_obs_visible():
    assert build_retrieval_query("Book a flight", "Hi, I need help") == "Book a flight\nHi, I need help"
    assert build_retrieval_query("Task only", "") == "Task only"
    query = build_retrieval_query("T" * 5000, "OPENING LINE OF THE USER")
    head, tail = query.split("\n", 1)
    # A long task must not push the observation (tau2's only task identity) out.
    assert len(head) == 600 and tail == "OPENING LINE OF THE USER"


def test_select_admission_candidates_uses_signed_utility_and_dedups_per_task():
    c = [
        {"task_key": "t1", "skill": "success-weak", "episode_success": True},
        {"task_key": "t1", "skill": "success-strong", "episode_success": True},
        {"task_key": "t2", "skill": "failure-helpful", "episode_success": False},
        {"task_key": "t3", "skill": "failure-harmful", "episode_success": False},
        {"task_key": "t4", "skill": "failure-unscored", "episode_success": False},
        {"task_key": "t5", "skill": "unknown-unscored", "episode_success": None},
    ]
    scored = [(c[0], 0.2), (c[1], 0.9), (c[2], -0.7), (c[3], 0.5), (c[4], None), (c[5], None)]

    kept = select_admission_candidates(scored, limit=10)
    assert [item["skill"] for item in kept] == ["success-strong", "failure-helpful", "unknown-unscored"]
    assert [(item["spec_gap"], item["admission_utility"]) for item in kept] == [
        (0.9, 0.9),
        (-0.7, 0.7),
        (None, None),
    ]
    assert "spec_gap" not in c[0]  # Selection must not mutate reusable candidate metadata.

    assert [item["skill"] for item in select_admission_candidates(scored, limit=2)] == ["success-strong", "failure-helpful"]
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


def test_lru_eviction_drops_least_recently_retrieved():
    pool = _make_pool(capacity=2, evict_policy="lru")
    _add(pool, "skill a", _unit(1, 0), task_key="task-a", step=1)
    _add(pool, "skill b", _unit(0, 1), task_key="task-b", step=2)
    # A low gate no longer matters: "a" was retrieved recently, "b" never was.
    pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"], current_step=3)
    pool.record_usage(skill_id_for("skill a"), 0.05, global_step=3)
    assert pool.snapshot_metrics()["seed/global_pool/never_injected_ratio"] == pytest.approx(0.5)
    assert _add(pool, "skill c", _unit(-1, 0), task_key="task-c", step=4) == "added"
    assert pool.has(skill_id_for("skill a"))
    assert not pool.has(skill_id_for("skill b"))
    assert pool.snapshot_metrics()["seed/global_pool/evicted_total"] == 1.0
    assert pool.expire(current_step=100) == 0  # expiry is a no-op outside the window policy


def test_window_policy_expires_by_admission_step_and_evicts_fifo():
    pool = _make_pool(capacity=2, evict_policy="window", window_steps=3)
    _add(pool, "skill a", _unit(1, 0), task_key="task-a", step=1)
    _add(pool, "skill b", _unit(0, 1), task_key="task-b", step=2)
    pool.record_usage(skill_id_for("skill a"), 0.9, global_step=5)  # retrieval does not extend life
    assert pool.expire(current_step=5) == 1  # cutoff = 2: "a" (step 1) expires, "b" (step 2) stays
    assert not pool.has(skill_id_for("skill a"))
    assert pool.has(skill_id_for("skill b"))
    _add(pool, "skill c", _unit(-1, 0), task_key="task-c", step=6)
    assert _add(pool, "skill d", _unit(0, -1), task_key="task-d", step=7) == "added"  # full: oldest admission goes
    assert not pool.has(skill_id_for("skill b"))
    assert pool.has(skill_id_for("skill c")) and pool.has(skill_id_for("skill d"))


def test_global_pool_config_validates_evict_policy():
    with pytest.raises(ValueError):
        GlobalPoolConfig(evict_policy="bogus").validate()
    with pytest.raises(ValueError):
        GlobalPoolConfig(evict_policy="window", window_steps=0).validate()
    assert GlobalPoolConfig(evict_policy="window", window_steps=1).validate().evict_policy == "window"


def test_select_admission_candidates_failed_skill_positive_bypasses_gap_gate():
    c = [
        {"task_key": "t1", "skill": "success-strong", "episode_success": True},
        {"task_key": "t2", "skill": "failure-negative-gap", "episode_success": False},
        {"task_key": "t3", "skill": "failure-positive-gap", "episode_success": False},
        {"task_key": "t4", "skill": "failure-unscored", "episode_success": False},
        {"task_key": "t5", "skill": "unknown-unscored", "episode_success": None},
    ]
    scored = [(c[0], 0.9), (c[1], -0.7), (c[2], 0.5), (c[3], None), (c[4], None)]

    kept = select_admission_candidates(scored, limit=10, failed_skill_positive=True)
    # Successful/unknown candidates keep the original gate and always rank first; failed ones
    # pass regardless of gap sign, ordered by gap among themselves.
    assert [item["skill"] for item in kept] == [
        "success-strong", "unknown-unscored", "failure-positive-gap", "failure-negative-gap", "failure-unscored",
    ]
    assert [item["admission_utility"] for item in kept] == [0.9, None, 0.5, -0.7, None]
    # The per-step cap truncates failed candidates first.
    assert [item["skill"] for item in select_admission_candidates(scored, limit=2, failed_skill_positive=True)] == [
        "success-strong", "unknown-unscored",
    ]
    # Switch off: original signed-utility semantics.
    assert [item["skill"] for item in select_admission_candidates(scored, limit=10)] == [
        "success-strong", "failure-negative-gap", "unknown-unscored",
    ]


@pytest.mark.parametrize("policy", ["window", "lru"])
def test_time_based_retrieval_requires_current_step(policy):
    pool = _make_pool(evict_policy=policy)
    with pytest.raises(ValueError, match="current_step is required"):
        pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"])


def test_window_retrieval_rechecks_delayed_admissions_and_keeps_cutoff_inclusive():
    pool = _make_pool(evict_policy="window", window_steps=3)
    _add(pool, "already stale", _unit(-1, 0), step=1)
    _add(pool, "at cutoff", _unit(0.8, 0.6), step=7)
    assert pool.expire(current_step=10) == 1
    # A queued admission finishes after explicit maintenance but before retrieval.
    _add(pool, "late stale", _unit(1, 0), step=6)
    result = pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"], current_step=10)[0]
    assert result.hit.text == "at cutoff"
    assert result.top_similarity == pytest.approx(0.8)
    assert not pool.has(skill_id_for("late stale"))
    assert pool.snapshot_metrics()["seed/global_pool/expired_total"] == 2.0
    assert pool.expire(current_step=10) == 0  # no double counting
    # Retrieval must not renew a window entry's admission age.
    result = pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"], current_step=11)[0]
    assert result.hit is None and result.top_similarity is None
    assert pool.snapshot_metrics()["seed/global_pool/expired_total"] == 3.0


def test_lru_hit_is_protected_before_scoring_without_double_counting_usage():
    pool = _make_pool(capacity=2, evict_policy="lru")
    _add(pool, "skill a", _unit(1, 0), step=1)
    _add(pool, "skill b", _unit(0, 1), step=2)
    results = pool.retrieve(np.stack([_unit(1, 0)] * 2), ["other-task"] * 2, current_step=3)
    skill_id = skill_id_for("skill a")
    assert all(result.hit.skill_id == skill_id for result in results)
    stats = pool._entries[skill_id]["stats"]
    assert stats == {"times_injected": 0, "gate_ema": None, "last_used_step": 3}
    # Previous step's admission arrives before this step's teacher scores.
    _add(pool, "skill c", _unit(-1, 0), step=2)
    assert pool.has(skill_id) and not pool.has(skill_id_for("skill b"))
    # Score completion neither rolls back nor advances the retrieval timestamp.
    pool.record_usage(skill_id, 0.8, global_step=2)
    pool.record_usage(skill_id, 0.4, global_step=5)
    assert stats["last_used_step"] == 3
    assert stats["times_injected"] == 2
    assert stats["gate_ema"] == pytest.approx(0.6)


@pytest.mark.parametrize("query,task_key", [((1, 0), "task-a"), ((-1, 0), "other-task")])
def test_lru_does_not_touch_same_task_or_below_threshold_matches(query, task_key):
    pool = _make_pool(evict_policy="lru")
    _add(pool, "skill a", _unit(1, 0), step=1)
    before = deepcopy(pool._entries)
    result = pool.retrieve(np.stack([_unit(*query)]), [task_key], current_step=10)[0]
    assert result.hit is None
    assert pool._entries == before


@pytest.mark.parametrize("current_step", [None, 100])
def test_gate_ema_retrieval_remains_read_only(current_step):
    pool = _make_pool(capacity=2)
    _add(pool, "skill a", _unit(1, 0), step=1)
    _add(pool, "skill b", _unit(0, 1), step=2)
    before = deepcopy(pool._entries)
    result = pool.retrieve(np.stack([_unit(1, 0)]), ["other-task"], current_step=current_step)[0]
    assert result.hit.text == "skill a"
    assert pool._entries == before
    _add(pool, "skill c", _unit(-1, 0), step=3)
    assert not pool.has(skill_id_for("skill a"))  # original neutral-prior tie break
    assert pool.snapshot_metrics()["seed/global_pool/expired_total"] == 0.0


@pytest.mark.parametrize("policy", ["window", "lru", "gate_ema"])
def test_trainer_retrieval_handles_admission_during_query_embedding(policy):
    # Exercise the real trainer call boundary without workers, API calls or file I/O.
    import torch
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    pool = _make_pool(evict_policy=policy, window_steps=3)
    _add(pool, "at cutoff", _unit(0.8, 0.6), step=7)
    _add(pool, "older entry", _unit(-1, 0), step=1)

    def encode(queries):
        _add(pool, "delayed entry", _unit(1, 0), step=6)
        return np.stack([_unit(1, 0)] * len(queries))

    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.global_steps = 10
    trainer._lazy_init_seed_global_pool = lambda: (
        pool, SimpleNamespace(available=True), SimpleNamespace(encode=encode)
    )
    trainer._dump_seed_global_pool_events = lambda events: None
    batch = DataProto.from_dict(tensors={"responses": torch.ones((1, 1), dtype=torch.long)})
    analysis = {"trajectory": {"episode_skill": ""}}
    metrics = {}
    result = trainer._select_seed_global_skills(
        batch=batch, episodes={"trajectory": [{"observation": "another task"}]},
        episode_analysis=analysis, traj_success={}, metrics=metrics,
    )
    expected_skill = "at cutoff" if policy == "window" else "delayed entry"
    assert analysis["trajectory"]["global_skill"] == expected_skill
    assert result["injections"] == {"trajectory": skill_id_for(expected_skill)}
    assert metrics["seed/global_pool/retrieval_failed"] == 0.0
    assert metrics["seed/global_pool/expired"] == (2.0 if policy == "window" else 0.0)
    stats = pool._entries[skill_id_for(expected_skill)]["stats"]
    assert stats["last_used_step"] == (10 if policy == "lru" else (7 if policy == "window" else 6))
    assert stats["times_injected"] == 0 and stats["gate_ema"] is None


@pytest.mark.parametrize("policy", ["window", "lru"])
def test_retrieval_keeps_maintenance_and_hit_selection_locked_against_admission(policy, monkeypatch):
    pool = _make_pool(capacity=2, evict_policy=policy, window_steps=3)
    _add(pool, "skill a", _unit(1, 0), step=1)
    _add(pool, "skill b", _unit(0, 1), step=2)
    selecting, admission_probed = Event(), Event()
    nearest = pool._nearest_locked

    def pause_selection(query, exclude_task_key):
        if exclude_task_key is not None:  # add() also calls _nearest_locked
            selecting.set()
            assert admission_probed.wait(timeout=5)
        return nearest(query, exclude_task_key)

    def admit():
        assert selecting.wait(timeout=5)
        acquired = pool._lock.acquire(blocking=False)
        if acquired:
            pool._lock.release()
        admission_probed.set()
        assert not acquired  # expiry/selection/touch must not have released the lock
        return _add(pool, "skill c", _unit(-1, 0), step=3)

    monkeypatch.setattr(pool, "_nearest_locked", pause_selection)
    with ThreadPoolExecutor(max_workers=2) as executor:
        retrieval = executor.submit(pool.retrieve, np.stack([_unit(1, 0)]), ["other-task"], current_step=5)
        admission = executor.submit(admit)
        result = retrieval.result(timeout=10)[0]
        assert admission.result(timeout=10) == "added"
    if policy == "window":
        assert result.hit is None  # skill a was over-age, skill b is below threshold
        assert not pool.has(skill_id_for("skill a"))
        assert pool.snapshot_metrics()["seed/global_pool/expired_total"] == 1.0
    else:
        assert result.hit.skill_id == skill_id_for("skill a")
        assert pool.has(skill_id_for("skill a"))
        assert not pool.has(skill_id_for("skill b"))
