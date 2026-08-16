# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_SLUG_MAP = {
    "strategies_and_insights": "str",
    "formulas_and_calculations": "calc",
    "code_snippets_and_templates": "code",
    "common_mistakes_to_avoid": "err",
    "problem_solving_heuristics": "prob",
    "context_clues_and_indicators": "ctx",
    "others": "misc",
    "meta_strategies": "meta",
}


def get_section_slug(section_name: str) -> str:
    clean = section_name.lower().strip().replace(" ", "_").replace("&", "and")
    if clean in _SLUG_MAP:
        return _SLUG_MAP[clean]
    words = clean.split("_")
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[0] for w in words[:5])


_LINE_RE = re.compile(
    r"\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)"
)


def parse_playbook_line(line: str) -> Optional[Dict[str, Any]]:
    m = _LINE_RE.match(line.strip())
    if m:
        return {
            "id": m.group(1),
            "helpful": int(m.group(2)),
            "harmful": int(m.group(3)),
            "content": m.group(4),
            "raw_line": line,
        }
    return None


def format_playbook_line(
    bullet_id: str, helpful: int, harmful: int, content: str
) -> str:
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"

def update_bullet_counts(playbook_text: str, bullet_tags: List[Dict]) -> str:
    tag_map: Dict[str, str] = {}
    for tag in bullet_tags:
        if not isinstance(tag, dict):
            continue
        bid = tag.get("id") or tag.get("bullet", "")
        tval = tag.get("tag", "neutral")
        if bid:
            tag_map[bid] = tval

    if not tag_map:
        return playbook_text

    lines = playbook_text.split("\n")
    updated: List[str] = []
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in tag_map:
            t = tag_map[parsed["id"]]
            if t == "helpful":
                parsed["helpful"] += 1
            elif t == "harmful":
                parsed["harmful"] += 1
            updated.append(
                format_playbook_line(
                    parsed["id"], parsed["helpful"], parsed["harmful"], parsed["content"]
                )
            )
        else:
            updated.append(line)
    return "\n".join(updated)


def apply_curator_operations(
    playbook_text: str,
    operations: List[Dict[str, Any]],
    next_id: int,
) -> Tuple[str, int]:
    lines = playbook_text.split("\n")
    sections: Dict[str, int] = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("##"):
            header = line.strip()[2:].strip()
            norm = header.lower().replace(" ", "_").replace("&", "and")
            sections[norm] = i

    bullets_to_add: List[Tuple[str, str]] = []

    for op in operations:
        if op.get("type") != "ADD":
            continue
        section_raw = op.get("section", "others")
        section_norm = section_raw.lower().replace(" ", "_").replace("&", "and")
        if section_norm not in sections:
            section_norm = "others"

        slug = get_section_slug(section_norm)
        new_id = f"{slug}-{next_id:05d}"
        next_id += 1
        content = op.get("content", "")
        new_line = format_playbook_line(new_id, 0, 0, content)
        bullets_to_add.append((section_norm, new_line))

    final: List[str] = []
    current_section: Optional[str] = None

    for line in lines:
        if line.strip().startswith("##"):
            if current_section is not None:
                for sec, bline in bullets_to_add:
                    if sec == current_section:
                        final.append(bline)
                bullets_to_add = [
                    (s, b) for s, b in bullets_to_add if s != current_section
                ]
            header = line.strip()[2:].strip()
            current_section = header.lower().replace(" ", "_").replace("&", "and")
        final.append(line)

    if current_section is not None:
        for sec, bline in bullets_to_add:
            if sec == current_section:
                final.append(bline)
        bullets_to_add = [(s, b) for s, b in bullets_to_add if s != current_section]

    for _, bline in bullets_to_add:
        final.append(bline)

    return "\n".join(final), next_id

def get_playbook_stats(playbook_text: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "total_bullets": 0,
        "high_performing": 0,
        "problematic": 0,
        "unused": 0,
        "by_section": {},
    }
    current_section = "general"
    for line in playbook_text.split("\n"):
        if line.strip().startswith("##"):
            current_section = line.strip()[2:].strip()
            continue
        parsed = parse_playbook_line(line)
        if parsed:
            stats["total_bullets"] += 1
            h, d = parsed["helpful"], parsed["harmful"]
            if h > 5 and d < 2:
                stats["high_performing"] += 1
            elif d >= h and d > 0:
                stats["problematic"] += 1
            elif h + d == 0:
                stats["unused"] += 1
            sec = stats["by_section"].setdefault(
                current_section, {"count": 0, "helpful": 0, "harmful": 0}
            )
            sec["count"] += 1
            sec["helpful"] += h
            sec["harmful"] += d
    return stats

def extract_playbook_bullets(
    playbook_text: str, bullet_ids: List[str]
) -> str:
    if not bullet_ids:
        return "(No bullets used by generator)"
    found: List[str] = []
    for line in playbook_text.split("\n"):
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in bullet_ids:
            found.append(
                format_playbook_line(
                    parsed["id"], parsed["helpful"], parsed["harmful"], parsed["content"]
                )
            )
    return "\n".join(found) if found else "(No matching bullets found)"


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.I):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            continue

    i = 0
    while i < len(text):
        if text[i] == "{":
            depth, start = 1, i
            i += 1
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                elif text[i] == '"':
                    i += 1
                    while i < len(text) and text[i] != '"':
                        if text[i] == "\\":
                            i += 1
                        i += 1
                i += 1
            if depth == 0:
                try:
                    return json.loads(text[start:i])
                except json.JSONDecodeError:
                    pass
        else:
            i += 1

    return None
