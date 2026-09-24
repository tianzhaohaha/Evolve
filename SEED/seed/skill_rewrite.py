"""Policy-side judge + rewrite of pool candidates (``algorithm.seed.global_pool.judge_backend=policy_vllm``).

One generation per candidate returns a JSON object ``{"score": 0-10, "generalized_skill": "..."}``:
the score is the transferability verdict (compared with ``score_threshold`` after dividing by 10),
the text is the version that enters the pool. ``rewrite`` modes:

* ``none``: score only; the raw skill is stored.
* ``deinstantiate``: strip instance details (names, ids, values, item lists) and keep the decision
  rule and action order, so the skill applies to any task of the same kind.
* ``aggregate``: the candidate is merged with its nearest pool entries into one general rule.

Pure (string / json) so it is unit-testable without the trainer.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple

REWRITE_MODES = ("none", "deinstantiate", "aggregate")

_HEADER = (
    "You are auditing a \"skill\" extracted from an agent's successful episode. A skill is useful when "
    "an agent facing a different task of the same kind could follow it; it is useless when it only "
    "restates what happened in that one episode or depends on details specific to it.\n\n"
    "Candidate skill:\n[{skill}]\n"
)
_NEIGHBORS = (
    "\nSkills already stored for similar tasks:\n{neighbors}\n"
)
_TASKS = {
    "none": (
        "\nRate the candidate's transferability from 0 (episode-specific) to 10 (a general rule). "
        "Copy the candidate unchanged into generalized_skill."
    ),
    "deinstantiate": (
        "\nRate the candidate's transferability from 0 to 10. Then rewrite it so it applies to any task "
        "of the same kind: remove names, ids, values, item lists and other instance details; keep the "
        "decision rule and the order of actions; at most 60 words; imperative voice."
    ),
    "aggregate": (
        "\nRate the candidate's transferability from 0 to 10. Then write ONE general rule that covers the "
        "candidate together with the stored skills above: keep only the decision rules and action order "
        "they share or that generalize them; no names, ids, values or item lists; at most 80 words; "
        "imperative voice."
    ),
}
_FOOTER = (
    "\n\nReply with ONLY a JSON object: {\"score\": <integer 0-10>, \"generalized_skill\": \"<text>\"}"
)


def build_judge_rewrite_prompt(skill: str, *, mode: str = "none", neighbors: Sequence[str] = ()) -> str:
    if mode not in REWRITE_MODES:
        raise ValueError(f"rewrite mode must be one of {REWRITE_MODES}, got {mode!r}")
    prompt = _HEADER.format(skill=str(skill).strip())
    if mode == "aggregate" and neighbors:
        prompt += _NEIGHBORS.format(neighbors="\n".join(f"- {str(n).strip()}" for n in neighbors))
    return prompt + _TASKS[mode] + _FOOTER


def _last_json_object(text: str) -> Optional[dict]:
    """The last complete JSON object in ``text`` (policy outputs may prepend reasoning)."""
    decoder = json.JSONDecoder(strict=False)
    start = text.rfind("{")
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        start = text.rfind("{", 0, start)
    return None


def parse_judge_rewrite(text: str, *, raw_skill: str = "") -> Optional[Tuple[float, str]]:
    """``(score in [0, 1], generalized skill)`` or None when the reply is unusable.

    An empty ``generalized_skill`` falls back to ``raw_skill`` (a judge-only reply), so ``none``
    mode never loses the skill to a lazy model."""
    obj = _last_json_object(str(text or ""))
    if obj is None or "score" not in obj:
        return None
    try:
        score = float(obj["score"])
    except (TypeError, ValueError):
        return None
    score = min(max(score / 10.0, 0.0), 1.0)
    skill = str(obj.get("generalized_skill") or "").strip() or str(raw_skill).strip()
    if not skill:
        return None
    return score, skill
