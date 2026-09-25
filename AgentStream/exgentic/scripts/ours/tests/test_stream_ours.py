"""CPU-only tests of the scripts/ours runner pieces (no benchmarks, no API, no wandb)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

import registry_ours as reg  # noqa: E402
import stream_ours as so  # noqa: E402
from wandb_stream import OnlineTally, WandbStream, holdout_summary  # noqa: E402


def test_registry_matches_the_seed_environment_settings():
    configs = reg.build_configs("judge/x", "http://r:1")
    assert list(configs) == ["bfcl", "tau2", "browsecompplus"]
    assert configs["tau2"]["bm_kwargs"] == {"subset": "retail", "user_simulator_model": "judge/x"}
    assert configs["browsecompplus"]["bm_kwargs"] == {"include_get_document": True, "eval_model_id": "judge/x", "retriever_url": "http://r:1", "use_cache": False}
    assert (reg.MAX_STEPS, reg.STREAM_SEED, reg.NUM_STREAM_TASKS, reg.NUM_HOLDOUT_TASKS) == (40, 44, 64, 32)
    with pytest.raises(ValueError):
        reg.build_configs(benchmarks=["hle"])


def test_holdout_ids_are_the_next_slice_of_the_selection_shuffle():
    def select(configs, num, seed):
        assert (num, seed) == (96, 42)
        return {slug: [f"{slug}{i}" for i in range(num)] for slug in configs}
    holdout = reg.holdout_task_ids({"bfcl": {}, "tau2": {}}, select=select)
    assert holdout["bfcl"] == [f"bfcl{i}" for i in range(64, 96)] and len(holdout["tau2"]) == 32


def test_tally_and_holdout_summary_use_seed_metric_names():
    tally = OnlineTally()
    tally.add("bfcl", 1.0, True); tally.add("tau2", 0.5, False); tally.add("bfcl", 0.0, False)
    m = tally.metrics()
    assert m["online/cumulative_avg_score"] == pytest.approx(0.5) and m["online/cumulative_success_rate"] == pytest.approx(1 / 3)
    assert m["online/bfcl/cumulative_success_rate"] == 0.5 and m["online/tau2/cumulative_avg_score"] == 0.5
    rows = [{"benchmark_slug": "bfcl", "score": 1.0, "success": True, "status": "success"},
            {"benchmark_slug": "bfcl", "score": 0.0, "success": False, "status": "error"},
            {"benchmark_slug": "tau2", "score": 0.3, "success": False, "status": "unsuccessful"}]
    s = holdout_summary(rows)
    assert s["val/success_rate"] == pytest.approx(1 / 3) and s["val/bfcl_success_rate"] == 0.5 and s["val/tau2_score"] == pytest.approx(0.3)
    assert s["val/env_error_rate"] == pytest.approx(1 / 3) and holdout_summary([]) == {}


def test_wandb_stream_is_a_noop_when_disabled(tmp_path):
    stream = WandbStream("x", {}, tmp_path, enabled=False)
    stream.log_task({"session_index": 0, "score": 1, "success": True}, OnlineTally()); stream.log_holdout({"val/success_rate": 1}, 3); stream.finish()


def test_agent_specs_store_keys_follow_the_agentstream_convention():
    rb, ace, ref = so.AGENTS["reasoning_bank"], so.AGENTS["ace"], so.AGENTS["tool_calling"]
    assert rb.store_key("isolated", "tau2") == "rb_isolated_tau2" and rb.store_key("interleaved", "tau2") == "rb_interleaved_global"
    assert so.store_keys(ace, "isolated", ["bfcl", "tau2"]) == ["ace_isolated_bfcl", "ace_isolated_tau2"]
    assert so.store_keys(ace, "interleaved", ["bfcl", "tau2"]) == ["ace_interleaved_global"] and so.store_keys(ref, "isolated", ["bfcl"]) == []


def test_progress_truncates_records_beyond_the_committed_count(tmp_path):
    p = so.Progress(tmp_path, "online_metrics")
    assert p.load() == []
    p.append({"i": 0}, 1); p.append({"i": 1}, 2)
    (tmp_path / "online_metrics.jsonl").open("a").write(json.dumps({"i": 2}) + "\n")  # crash after the record, before progress
    assert [r["i"] for r in p.load()] == [0, 1]
    assert (tmp_path / "online_metrics.jsonl").read_text().count("\n") == 2


def _fake_sr(success, score=None, status="success"):
    return SimpleNamespace(success=success, score=score, steps=3, action_count=2, agent_cost=0.01, execution_time=1.0,
                           status=SimpleNamespace(value=status), cost_reports={"m": {"input_tokens": 10, "output_tokens": 2}})


def _args(tmp_path, agent="tool_calling", mode="interleaved", **over):
    base = dict(pass_="online", agent=agent, model="openai/m", mode=mode, output_dir=str(tmp_path), seed=44, num_tasks=2, num_holdout=1,
                benchmarks="bfcl,tau2", api_base=None, judge_model="j", retriever_url="http://r", max_tokens=None, reasoning_effort=None,
                run_name=None, wandb_project="p", no_wandb=True)
    base.update(over)
    return SimpleNamespace(**base)


def test_online_then_holdout_end_to_end_with_stubbed_tasks(tmp_path, monkeypatch):
    order = [("bfcl", "b0"), ("tau2", "t0"), ("bfcl", "b1"), ("tau2", "t1")]
    calls = []

    def fake_run_task(spec, configs, bm, tid, *, model, mode, settings, learning_enabled, sessions_dir):
        calls.append((bm, tid, learning_enabled, settings.temperature))
        return _fake_sr(success=bm == "bfcl", score=1.0 if bm == "bfcl" else 0.25)

    monkeypatch.setattr(so, "run_task", fake_run_task)
    monkeypatch.setattr(so, "model_settings", lambda args, t: SimpleNamespace(temperature=t))
    import task_ordering
    monkeypatch.setattr(task_ordering, "get_unified_task_order", lambda configs, n, seed, mode: order)
    monkeypatch.setattr(reg, "holdout_task_ids", lambda configs, n, k: {"bfcl": ["hb"], "tau2": ["ht"]})
    monkeypatch.setattr(so, "holdout_task_ids", reg.holdout_task_ids)

    so.run_online(_args(tmp_path))
    records = [json.loads(l) for l in (tmp_path / "online_metrics.jsonl").read_text().splitlines()]
    assert [(r["benchmark_slug"], r["task_id"], r["success"], r["score"]) for r in records] == [("bfcl", "b0", True, 1.0), ("tau2", "t0", False, 0.25), ("bfcl", "b1", True, 1.0), ("tau2", "t1", False, 0.25)]
    assert json.loads((tmp_path / "online_summary.json").read_text())["online/cumulative_success_rate"] == 0.5
    assert all(le and t == 1.0 for _, _, le, t in calls)

    so.run_holdout(_args(tmp_path, pass_="holdout"))
    hold = calls[4:]
    assert [(b, t) for b, t, _, _ in hold] == [("bfcl", "hb"), ("tau2", "ht")] and all(not le and t == 0.4 for _, _, le, t in hold)
    summary = json.loads((tmp_path / "holdout_summary.json").read_text())
    assert summary["val/success_rate"] == 0.5 and summary["val/bfcl_success_rate"] == 1.0

    calls.clear()  # both passes resume as no-ops
    so.run_online(_args(tmp_path)); so.run_holdout(_args(tmp_path, pass_="holdout"))
    assert calls == []


def test_learning_stores_round_trip_and_the_freeze_flag_is_a_pydantic_field(tmp_path):
    pytest.importorskip("exgentic.agents.reasoning_bank.rb_store")
    spec = so.AGENTS["reasoning_bank"]
    store = so.get_store(spec, "isolated", "bfcl")
    store.increment_session()
    so.save_stores(spec, tmp_path)
    assert (tmp_path / "store_rb_isolated_bfcl.json").exists()
    assert so.load_stores(spec, "isolated", ["bfcl"], tmp_path, required=False) == 1
    with pytest.raises(FileNotFoundError):
        so.load_stores(spec, "isolated", ["tau2"], tmp_path, required=True)
    assert so.memory_stats(spec, "isolated", "bfcl") == (0, 0) and so.memory_stats(so.AGENTS["tool_calling"], "isolated", "bfcl") == (0, 0)
    try:
        from exgentic.agents.reasoning_bank.rb_agent import ReasoningBankAgent
    except ImportError as exc:  # litellm missing on the dev box
        pytest.skip(f"agent import needs the runtime deps: {exc}")
    dumped = ReasoningBankAgent(model="m", learning_enabled=False).model_dump()
    assert dumped["learning_enabled"] is False


def test_token_counts_tolerate_missing_and_none_values():
    reports = {"a": {"input_tokens": 5, "output_tokens": None}, "b": SimpleNamespace(input_tokens=None, output_tokens=2), "c": {}}
    assert so.token_counts(reports) == (5, 2)


def test_ace_memory_stats_count_tagged_bullets():
    pytest.importorskip("exgentic.agents.ace.playbook_store")
    spec = so.AGENTS["ace"]
    spec.store_cls().reset_all()
    store = so.get_store(spec, "interleaved", "bfcl")
    store._playbook = "## strategies\n[str-00001] helpful=1 harmful=0 :: check the schema first\n[str-00002] helpful=0 harmful=0 :: submit once\n"
    entries, tokens = so.memory_stats(spec, "interleaved", "bfcl")
    assert entries == 2 and tokens > 0


def test_holdout_is_logged_one_step_after_the_last_online_task(tmp_path):
    logged = []
    stream = WandbStream("x", {}, tmp_path, enabled=False)
    stream.run = SimpleNamespace(log=lambda payload, step: logged.append((payload, step)), finish=lambda: None)
    stream.log_holdout({"val/success_rate": 0.5}, task_index=192)
    assert logged == [({"val/success_rate": 0.5, "online/task_index": 192}, 193)]
