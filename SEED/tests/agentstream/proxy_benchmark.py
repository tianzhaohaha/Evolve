# Copyright 2026 SEED x AgentStream integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process stand-in for tau2's proxy session (exgentic/benchmarks/tau2).

A daemon runner thread plays the benchmark side of the queue rendezvous: it
answers every action with an observation and, like tau2's ``run_domain``,
produces the result only after being told to stop (``wait_for_action`` ->
None). ``score()`` therefore fails while the runner is still alive, which is
exactly the state SEED's step budget leaves a tau2 session in.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from exgentic.adapters.executors.proxy import BaseProxySession
from exgentic.core.types import ActionType, SessionIndex, SessionScore, SingleObservation
from exgentic.testing.agent import GOOD_ACTION_TYPE
from exgentic.testing.benchmark import TestBenchmark, TestEvaluator


class ProxySession(BaseProxySession):
    def __init__(self, *, task_id: str, session_id: str) -> None:
        self._task_id = task_id
        self._session_id = session_id
        self._result: SessionScore | None = None
        super().__init__()
        self._runner_thread = threading.Thread(target=self._runner, daemon=True)
        self._runner_thread.start()

    def _runner(self) -> None:
        self.put_observation(SingleObservation(result="hello"))
        turns = 0
        while self.wait_for_action() is not None:
            turns += 1
            self.put_observation(SingleObservation(result=f"reply {turns}"))
        self._result = SessionScore(score=float(turns > 0), success=turns > 0, is_finished=True)
        self.put_observation(None)

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def task(self) -> str:
        return "Proxy task"

    @property
    def context(self) -> dict[str, Any]:
        return {}

    @property
    def actions(self) -> list[ActionType]:
        return [GOOD_ACTION_TYPE]

    def get_config(self) -> dict[str, Any]:
        return {"task_id": self._task_id}

    def score(self) -> SessionScore:
        if self._result is None:
            raise FileNotFoundError("runner has not written its results yet")
        return self._result

    def close(self) -> None:
        super().close()
        self._runner_thread.join(timeout=5.0)


class ProxyEvaluator(TestEvaluator):
    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        return {"task_id": str(index.task_id), "session_id": index.session_id}


class ProxyBenchmark(TestBenchmark):
    display_name: ClassVar[str] = "Proxy Benchmark"
    slug_name: ClassVar[str] = "proxy_benchmark"

    @classmethod
    def _get_evaluator_class(cls):
        return ProxyEvaluator

    @classmethod
    def _get_session_class(cls):
        return ProxySession
