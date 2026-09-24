"""seed/skill_rewrite.py: policy-side judge / rewrite prompts and reply parsing."""

import pytest

from seed.skill_rewrite import REWRITE_MODES, build_judge_rewrite_prompt, parse_judge_rewrite

RAW = "Call get_user(id=42) before update_user."


def test_prompt_modes_share_the_header_and_footer_and_differ_in_the_task():
    none, deinst, agg = (build_judge_rewrite_prompt(RAW, mode=m, neighbors=["n1", "n2"]) for m in REWRITE_MODES)
    for prompt in (none, deinst, agg):
        assert RAW in prompt and prompt.rstrip().endswith('"generalized_skill": "<text>"}')
    assert "Copy the candidate unchanged" in none and "already stored" not in none
    assert "remove names, ids, values" in deinst and "already stored" not in deinst
    assert "- n1\n- n2" in agg and "ONE general rule" in agg
    assert "already stored" not in build_judge_rewrite_prompt(RAW, mode="aggregate")  # no neighbours yet
    with pytest.raises(ValueError):
        build_judge_rewrite_prompt(RAW, mode="rewrite")


def test_parse_scales_the_score_and_falls_back_to_the_raw_skill():
    assert parse_judge_rewrite('{"score": 7, "generalized_skill": "Verify before acting."}') == (0.7, "Verify before acting.")
    reasoning = 'First {not json}. Then:\n{"score": 12, "generalized_skill": ""}'
    assert parse_judge_rewrite(reasoning, raw_skill=RAW) == (1.0, RAW)  # clamped, judge-only reply keeps the raw text
    assert parse_judge_rewrite('{"score": "3.5", "generalized_skill": " x "}') == (0.35, "x")


@pytest.mark.parametrize("reply", ["no json here", '{"generalized_skill": "x"}', '{"score": "high"}', '{"score": 5, "generalized_skill": ""}', ""])
def test_parse_rejects_unusable_replies(reply):
    assert parse_judge_rewrite(reply) is None
