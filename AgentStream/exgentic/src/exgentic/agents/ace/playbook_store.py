# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_PLAYBOOK = """\
## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""


@dataclass
class LearningEvent:
    session_id: str
    task_id: str
    step: int
    was_correct_before: bool
    was_correct_after: bool
    summary: str
    benchmark_id: str = ""


class PlaybookStore:

    _instances: Dict[str, "PlaybookStore"] = {}
    _global_lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        shuffle_mode: str = "isolated",
        task_group: Optional[str] = None,
        initial_playbook: Optional[str] = None,
        benchmark_id: Optional[str] = None,
    ) -> "PlaybookStore":
        if shuffle_mode == "isolated":
            bm = benchmark_id or task_group or "default"
            key = f"ace_isolated_{bm}"
        elif shuffle_mode == "sequential":
            key = "ace_sequential_global"
        elif shuffle_mode == "interleaved":
            key = "ace_interleaved_global"
        else:
            raise ValueError(f"Unknown shuffle_mode: {shuffle_mode!r}")

        with cls._global_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    store_id=key,
                    initial_playbook=initial_playbook or DEFAULT_PLAYBOOK,
                )
            return cls._instances[key]

    @classmethod
    def reset_all(cls) -> None:
        with cls._global_lock:
            cls._instances.clear()

    @classmethod
    def list_stores(cls) -> Dict[str, "PlaybookStore"]:
        with cls._global_lock:
            return dict(cls._instances)

    def __init__(self, store_id: str, initial_playbook: str) -> None:
        self.store_id = store_id
        self._lock = threading.Lock()
        self._playbook: str = initial_playbook
        self._next_global_id: int = 1
        self._session_count: int = 0
        self._history: List[LearningEvent] = []
        self._benchmark_counts: Dict[str, int] = {}  

    @property
    def playbook(self) -> str:
        with self._lock:
            return self._playbook

    @playbook.setter
    def playbook(self, value: str) -> None:
        with self._lock:
            self._playbook = value

    @property
    def next_global_id(self) -> int:
        with self._lock:
            return self._next_global_id

    @next_global_id.setter
    def next_global_id(self, value: int) -> None:
        with self._lock:
            self._next_global_id = value

    @property
    def session_count(self) -> int:
        with self._lock:
            return self._session_count

    def increment_session(self) -> int:
        with self._lock:
            self._session_count += 1
            return self._session_count

    def record_learning(
        self,
        session_id: str,
        task_id: str,
        was_correct_before: bool,
        was_correct_after: bool,
        summary: str,
        benchmark_id: str = "",
    ) -> None:
        with self._lock:
            self._history.append(
                LearningEvent(
                    session_id=session_id,
                    task_id=task_id,
                    step=self._session_count,
                    was_correct_before=was_correct_before,
                    was_correct_after=was_correct_after,
                    summary=summary[:500],
                    benchmark_id=benchmark_id,
                )
            )
            if benchmark_id:
                self._benchmark_counts[benchmark_id] = (
                    self._benchmark_counts.get(benchmark_id, 0) + 1
                )

    def save_checkpoint(self, path: str) -> None:
        with self._lock:
            payload = {
                "store_id": self.store_id,
                "playbook": self._playbook,
                "next_global_id": self._next_global_id,
                "session_count": self._session_count,
                "history_len": len(self._history),
                "benchmark_counts": dict(self._benchmark_counts),
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        with self._lock:
            self._playbook = payload["playbook"]
            self._next_global_id = payload["next_global_id"]
            self._session_count = payload.get("session_count", 0)
            self._benchmark_counts = payload.get("benchmark_counts", {})

    def save_playbook_text(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.playbook)
