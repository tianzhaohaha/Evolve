"""LLM-as-judge transferability screen for global-pool admission.

Batches several candidate skills into one OpenAI-compatible chat call (the
OpenRouter GLM route used elsewhere in the project) and parses a JSON array
verdict. Everything fails open: a missing API key, a failed call, or an
unparseable reply just skips admission for that batch — training never blocks
on the judge.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """You are auditing "skills" extracted from an agent's past episodes.
A skill is TRANSFERABLE when it would still be actionable advice after stripping
domain-specific nouns (tool names, product names, dataset fields): it describes a
general strategy (e.g. verify state before acting, recover from a failed call,
clarify ambiguous user intent) rather than a fact about one environment.

For EACH numbered skill, output one JSON object with:
  "idx": the skill's number,
  "transferable": true/false,
  "score": 0.0-1.0 confidence that it transfers across task domains,
  "tag": one of ["tool-arg-format","search-verify","user-intent","error-recovery","budget-control","answer-format","planning","other"],
  "reason": one short sentence.

Reply with ONLY a JSON array of these objects, one per skill, in order."""


@dataclass(frozen=True)
class SkillJudgeVerdict:
    transferable: bool
    score: float
    tag: str
    reason: str


class SkillJudge:
    def __init__(self, *, model: str, base_url: str, api_key_env: str = "OPENROUTER_API_KEY",
                 batch_size: int = 12, max_completion_tokens: int = 1024, retries: int = 3):
        # Accept litellm-style route names ("openrouter/z-ai/glm-5.2") for config reuse.
        self.model = model.split("openrouter/", 1)[1] if model.startswith("openrouter/") else model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.batch_size = max(int(batch_size), 1)
        self.max_completion_tokens = int(max_completion_tokens)
        self.retries = int(retries)
        self._client = None
        self._warned_unavailable = False

    @property
    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def _get_client(self):
        if self._client is None:
            from utils import create_openai_client

            self._client = create_openai_client(api_key=os.environ.get(self.api_key_env), base_url=self.base_url)
        return self._client

    def judge(self, skills: Sequence[str]) -> List[Optional[SkillJudgeVerdict]]:
        """One verdict per skill; None where the judge could not decide (fail-open)."""
        if not skills:
            return []
        if not self.available:
            if not self._warned_unavailable:
                logger.warning("Skill judge disabled: env %s is not set; global-pool admission is skipped.", self.api_key_env)
                self._warned_unavailable = True
            return [None] * len(skills)

        verdicts: List[Optional[SkillJudgeVerdict]] = []
        for start in range(0, len(skills), self.batch_size):
            chunk = list(skills[start:start + self.batch_size])
            verdicts.extend(self._judge_chunk(chunk))
        return verdicts

    def _judge_chunk(self, chunk: List[str]) -> List[Optional[SkillJudgeVerdict]]:
        from utils import chat_completion_with_retry

        numbered = "\n\n".join(f"[{i + 1}] {skill}" for i, skill in enumerate(chunk))
        try:
            text = chat_completion_with_retry(
                client=self._get_client(),
                model=self.model,
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                user_prompt=f"Skills to audit:\n\n{numbered}",
                temperature=0.0,
                max_completion_tokens=self.max_completion_tokens,
                retries=self.retries,
            )
            return parse_judge_response(text, expected=len(chunk))
        except Exception as exc:
            logger.warning("Skill judge chunk failed (%s skills): %s", len(chunk), exc)
            return [None] * len(chunk)


def _last_json_array(text: str) -> Optional[list]:
    """The last JSON array of objects in ``text``, or None.

    The client glues any ``reasoning_content`` in front of the reply, and that
    prose freely contains brackets (``[1]``-style skill indices), so a greedy
    regex over-captures from the first bracket in the reasoning to the last in
    the answer. Instead, attempt a real JSON decode at each ``[`` from the end
    backwards — the verdict array always sits after the preamble — and take the
    first decode that yields objects (a bare ``[1]`` from prose parses too, but
    only to a list of ints, which the object check rejects).
    """
    decoder = json.JSONDecoder()
    for position in range(len(text) - 1, -1, -1):
        if text[position] != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text, position)
        except ValueError:
            continue
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            return value
    return None


def parse_judge_response(text: str, expected: int) -> List[Optional[SkillJudgeVerdict]]:
    """Parse the judge's JSON array; unmatched or malformed items become None."""
    items = _last_json_array(str(text or ""))
    if items is None:
        return [None] * expected

    verdicts: List[Optional[SkillJudgeVerdict]] = [None] * expected
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("idx", position + 1)) - 1
            if not 0 <= idx < expected or verdicts[idx] is not None:
                continue
            verdicts[idx] = SkillJudgeVerdict(
                transferable=bool(item["transferable"]),
                score=max(0.0, min(1.0, float(item["score"]))),
                tag=str(item.get("tag", "other")),
                reason=str(item.get("reason", "")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return verdicts
