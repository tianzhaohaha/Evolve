"""AgentStream manager in sibling-resample mode: reset(kwargs) re-runs explicit tasks with the
reference injected into ``text`` only, keeps ``text_base`` plain, and records no online metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="manager module imports the trainer-side env base")

from agent_system.environments.env_package.agentstream.manager import AgentStreamEnvironmentManager  # noqa: E402
from agent_system.environments.env_package.agentstream.metrics import OnlineMetricsRecorder  # noqa: E402
from agent_system.memory import SimpleMemory  # noqa: E402


class _FakeEnvs:
    """Vectorised env stub: reset() serves two stream tasks, reset_refs() whatever it is asked."""

    def __init__(self):
        self.calls = []

    def _payloads(self, refs, stream):
        payloads = [{"slug": s, "task_id": t, "task": f"do {t}", "context": "", "actions_text": "- noop", "observation": f"obs {t}"} for s, t in refs]
        infos = [{"slug": s, "task_id": t, "stream_index": i if stream else -1, "pass_idx": 0 if stream else -1,
                  "rollout_slot": i, "won": False, "reset_error": False} for i, (s, t) in enumerate(refs)]
        return payloads, infos

    def reset(self):
        self.calls.append(("reset", None))
        return self._payloads([("bfcl", "t1"), ("bfcl", "t2")], stream=True)

    def reset_refs(self, refs):
        self.calls.append(("reset_refs", list(refs)))
        return self._payloads(list(refs), stream=False)

    def step(self, payloads):
        n = len(payloads)
        info = {"won": True, "score": 1.0, "step_count": 1, "action_error": False, "post_done": False}
        return ["next"] * n, [10.0] * n, [True] * n, [dict(info, slug="bfcl", task_id=f"t{i}") for i in range(n)]


def _manager(tmp_path):
    m = AgentStreamEnvironmentManager.__new__(AgentStreamEnvironmentManager)
    m.envs = _FakeEnvs()
    m.config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    m.phase, m._global_step = "train", 1
    m.recorder = OnlineMetricsRecorder(str(tmp_path / "online.jsonl"), group_n=1)
    m.projection_f = lambda texts: (
        [{"name": "noop", "arguments": {}}] * len(texts), [True] * len(texts),
        [{"reason": "", "think_present": True, "used_tool_call_alias": False}] * len(texts),
    )
    m.memory = SimpleMemory()
    m._reference_solutions, m._resample_mode = [], False
    return m


def test_resample_reset_injects_the_reference_into_text_only_and_records_nothing(tmp_path):
    m = _manager(tmp_path)
    requests = np.array(
        [{"task_slug": "bfcl", "task_id": "t9", "reference_solution": "Step 1 | obs: o | response: go"},
         {"task_slug": "tau2", "task_id": "t3", "reference_solution": ""}],
        dtype=object,
    )
    obs, infos = m.reset(kwargs=requests)
    assert m.envs.calls[-1] == ("reset_refs", [("bfcl", "t9"), ("tau2", "t3")])
    assert m._resample_mode and m.task_metadata()["task_id"] == ["t9", "t3"]
    assert "Reference Solution" in obs["text"][0] and "Step 1 | obs: o | response: go" in obs["text"][0]
    assert "Reference Solution" not in obs["text_base"][0] and obs["text"][1] == obs["text_base"][1]
    assert infos[0]["pass_idx"] == -1
    next_obs, _, dones, _ = m.step(["<action>noop</action>"] * 2)
    assert all(dones) and "Reference Solution" in next_obs["text"][0] and "Reference Solution" not in next_obs["text_base"][0]
    assert "online/cumulative_avg_score" not in m.recorder.snapshot()  # the resample pass is not an online episode

    obs, _ = m.reset(kwargs=None)  # the next ordinary step restores the plain path
    assert m.envs.calls[-1] == ("reset", None) and not m._resample_mode
    assert obs["text"] == obs["text_base"] and m.task_metadata()["task_id"] == ["t1", "t2"]
    m.step(["<action>noop</action>"] * 2)
    assert m.recorder.snapshot()["online/cumulative_avg_score"] == 1.0


def test_empty_or_foreign_kwargs_take_the_plain_path(tmp_path):
    m = _manager(tmp_path)
    obs, _ = m.reset(kwargs=np.array([], dtype=object))
    assert m.envs.calls == [("reset", None)] and obs["text"] == obs["text_base"] and not m._resample_mode
    foreign = np.array([{"question": "q1"}, {"question": "q2"}], dtype=object)  # a dataset's own env_kwargs
    obs, _ = m.reset(kwargs=foreign)
    assert m.envs.calls == [("reset", None), ("reset", None)] and obs["text"] == obs["text_base"] and not m._resample_mode


def test_pool_requests_inject_the_skill_into_the_general_skill_section(tmp_path):
    m = _manager(tmp_path)
    requests = np.array(
        [{"task_slug": "bfcl", "task_id": "t9", "section": "global_skill", "global_skill": "Check the schema first."},
         {"task_slug": "tau2", "task_id": "t3", "section": "reference_solution", "reference_solution": "Step 1 | go"}],
        dtype=object,
    )
    obs, _ = m.reset(kwargs=requests)
    assert "General Skill" in obs["text"][0] and "Check the schema first." in obs["text"][0] and "Reference Solution" not in obs["text"][0]
    assert "Reference Solution" in obs["text"][1] and "Step 1 | go" in obs["text"][1]
    assert all("Check the schema first." not in text and "Step 1 | go" not in text for text in obs["text_base"])
    next_obs, _, _, _ = m.step(["<action>noop</action>"] * 2)
    assert "Check the schema first." in next_obs["text"][0] and "Check the schema first." not in next_obs["text_base"][0]
    assert "online/cumulative_avg_score" not in m.recorder.snapshot()
