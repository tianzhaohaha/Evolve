# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ...core.actions import ActionsHandler, extract_argument
from ...core.benchmark import Benchmark
from ...core.evaluator import Evaluator
from ...core.session import Session
from ...core.types import (
    Action,
    ActionType,
    BenchmarkResults,
    EmptyObservation,
    FinishAction,
    Observation,
    SessionIndex,
    SessionScore,
    SingleAction,
    SingleObservation,
)
from ...utils.cost import CostReport, LiteLLMCostReport
from ...utils.settings import RunnerName

HLE_TOTAL_TASKS = 2500

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|%| and 100|%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True]


class HLEFinishArgs(BaseModel):
    explanation: str = Field(..., description="Your explanation/reasoning for your answer.")
    answer: str = Field(..., description="Your final answer to the question.")
    confidence: int = Field(..., description="Your confidence score between 0 and 100.", ge=0, le=100)


class HLEFinishAction(FinishAction):
    name: Literal["finish"] = "finish"
    arguments: HLEFinishArgs

class HLESession(Session):

    def __init__(
        self,
        task_idx: int,
        judge_model: str = "o3-mini-2025-01-31",
        session_id: str | None = None,
        agent_timeout: int = 900,
        **_kwargs: Any,
    ) -> None:
        if session_id is not None:
            self._session_id = session_id
        from datasets import load_dataset

        dataset = load_dataset("cais/hle", split="test")
        row = dataset[task_idx]

        self._question = row["question"]
        self._gold_answer = row["answer"]
        self._image = self._encode_image(row.get("image", None))
        self._task_id = task_idx
        self._judge_model = judge_model
        self._agent_timeout = agent_timeout
        self._start_time: float | None = None
        self._done = False
        self._final_answer: str | None = None
        self._final_explanation: str | None = None
        self._final_confidence: int | None = None
        self._judge_result: dict[str, Any] | None = None
        self._judge_input_tokens = 0
        self._judge_output_tokens = 0

        self._registry = ActionsHandler(logger=self.logger)
        self._registry.add_action(
            name="finish",
            description="Submit your final answer with explanation and confidence score.",
            action_cls=HLEFinishAction,
            handler=self._handle_finish,
            is_finish=True,
        )
        super().__init__()

    @staticmethod
    def _encode_image(image) -> str | None:
        if image is None:
            return None
        if isinstance(image, str):  
            return image if image else None
        
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    @property
    def task(self) -> str:
        return (
            "Answer the following question. Submit your answer by calling the 'finish' action "
            "with your explanation, final answer, and confidence score (0-100).\n\n"
            f"Question: {self._question}"
        )

    @property
    def context(self) -> dict[str, Any]:
        if self._image:
            return {"image": {"type": "image_url", "data": self._image}}
        return {}

    @property
    def task_id(self) -> str:
        return str(self._task_id)

    @property
    def actions(self) -> list[ActionType]:
        return self._registry.actions

    def start(self) -> Observation | None:
        import time
        self._start_time = time.time()
        return EmptyObservation()

    def step(self, action: Action) -> Observation | None:
        import time
        if action is None:
            self._done = True
        if self._done:
            return None
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            if elapsed >= self._agent_timeout:
                self._done = True
                return SingleObservation(
                    result=f"[Agent timeout reached ({self._agent_timeout}s). Session ending.]"
                )
        observation = self._registry.execute(action)
        return observation

    def done(self) -> bool:
        return self._done

    def _run_judge(self) -> dict[str, Any] | None:
        if self._final_answer is None:
            return None
        
        response_text = (
            f"Explanation: {self._final_explanation or ''}\n"
            f"Answer: {self._final_answer}\n"
            f"Confidence: {self._final_confidence or 100}%"
        )

        prompt = JUDGE_PROMPT.format(
            question=self._question,
            correct_answer=self._gold_answer,
            response=response_text,
        )

        async def _judge_async() -> dict[str, Any] | None:
            import litellm

            try:
                resp = await litellm.acompletion(
                    model=self._judge_model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=ExtractedAnswer,
                )
                import json as _json

                content = _json.loads(resp.choices[0].message.content)
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    self._judge_input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                    self._judge_output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
                return {
                    "correct_answer": self._gold_answer,
                    "model_answer": content["extracted_final_answer"],
                    "reasoning": content["reasoning"],
                    "correct": content["correct"],
                    "confidence": content["confidence"],
                }
            except Exception as e:
                self.logger.warning(f"Judge failed: {e}")
                return None

        return asyncio.run(_judge_async())

    def score(self) -> SessionScore:
        if self._judge_result is None:
            self._judge_result = self._run_judge()

        if self._judge_result is not None:
            correct = self._judge_result["correct"] == "yes"
            score = 1.0 if correct else 0.0
        else:
            score = 0.0

        finished = self._final_answer is not None
        return SessionScore(
            score=score,
            success=score == 1.0,
            is_finished=finished,
            session_metrics={
                "confidence": self._final_confidence,
                "judge_result": self._judge_result,
            },
        )

    def close(self):
        super().close()
        sc = self.score()
        self.save_standard_results(sc)

    def _handle_finish(self, action: SingleAction) -> None:
        self._final_explanation = extract_argument(action.arguments, "explanation", None)
        self._final_answer = extract_argument(action.arguments, "answer", None)
        self._final_confidence = extract_argument(action.arguments, "confidence", 100)
        self._done = True
        return None

    def get_cost(self) -> CostReport:
        if self._judge_input_tokens == 0 and self._judge_output_tokens == 0:
            return LiteLLMCostReport.initialize_empty(model_name=self._judge_model)
        return LiteLLMCostReport.from_token_counts(
            self._judge_model,
            self._judge_input_tokens,
            self._judge_output_tokens,
        )



def calib_err(confidence, correct, p="2", beta=100):
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if not bins:
        return 0.0
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))
            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ("infty", "infinity", "max"):
                cerr = np.maximum(cerr, difference)

    if p == "2":
        cerr = np.sqrt(cerr)

    return float(cerr)


class HLEEvaluator(Evaluator):

    def __init__(self, subset: str = "test", judge_model: str = "o3-mini-2025-01-31", agent_timeout: int = 900) -> None:
        self._subset = subset
        self._judge_model = judge_model
        self._agent_timeout = agent_timeout
        self._dataset = None

    def _ensure_dataset(self) -> None:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset("cais/hle", split="test")

    def list_tasks(self) -> list[str]:
        self._ensure_dataset()
        return [str(i) for i in range(len(self._dataset))]

    def get_session_kwargs(self, index: SessionIndex) -> dict[str, Any]:
        self._ensure_dataset()
        idx = int(index.task_id)
        if idx < 0 or idx >= len(self._dataset):
            raise IndexError(f"Task id {index.task_id} out of range for HLE.")
        return {
            "task_idx": idx,
            "judge_model": self._judge_model,
            "agent_timeout": self._agent_timeout,
            "session_id": index.session_id,
        }

    def aggregate_sessions(self, sessions: list[SessionIndex]) -> BenchmarkResults:
        scores: list[float] = []
        confidences: list[float] = []
        corrects: list[float] = []

        for paths in self.get_sessions_paths(sessions):
            with open(paths.benchmark_results, encoding="utf-8-sig") as f:
                payload = json.load(f)
            s = float(payload["score"])
            scores.append(s)
            corrects.append(s)
            metrics = payload.get("session_metrics", {})
            conf = metrics.get("confidence")
            if conf is not None:
                confidences.append(float(conf) / 100.0)
            else:
                confidences.append(1.0)

        n = len(scores)
        accuracy = 100 * sum(scores) / n if n else 0.0
        confidence_half_width = 1.96 * math.sqrt(accuracy * (100 - accuracy) / n) if n else 0.0

        cal_err = 0.0
        if n > 0:
            cal_err = 100 * calib_err(
                np.array(confidences), np.array(corrects), p="2", beta=100
            )

        return BenchmarkResults(
            benchmark_name="hle",
            total_tasks=n,
            score=accuracy / 100.0,
            metrics={
                "accuracy_pct": round(accuracy, 2),
                "confidence_interval": round(confidence_half_width, 2),
                "calibration_error": round(cal_err, 2),
            },
        )


class HLEBenchmark(Benchmark, BaseModel):
    display_name: ClassVar[str] = "HLE"
    slug_name: ClassVar[str] = "hle"
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def _get_evaluator_class(cls):
        return HLEEvaluator

    @classmethod
    def _get_session_class(cls):
        return HLESession

    subset: Literal["test"] = "test"
    judge_model: str = "o3-mini-2025-01-31"
    agent_timeout: int = 900
    runner: RunnerName | None = None

    def _get_evaluator_kwargs(self) -> dict[str, Any]:
        return {
            "subset": self.subset,
            "judge_model": self.judge_model,
            "agent_timeout": self.agent_timeout,
        }
