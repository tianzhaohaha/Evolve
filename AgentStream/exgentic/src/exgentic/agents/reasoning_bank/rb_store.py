# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:

    task_id: str
    query: str
    think_list: list[str]
    action_list: list[str]
    status: str
    memory_items: list[str]
    template_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryEntry":
        return cls(
            task_id=d["task_id"],
            query=d["query"],
            think_list=d.get("think_list", []),
            action_list=d.get("action_list", []),
            status=d["status"],
            memory_items=d.get("memory_items", []),
            template_id=d.get("template_id"),
        )


class ReasoningBankStore:

    _instances: ClassVar[dict[str, "ReasoningBankStore"]] = {}
    _class_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, store_id: str) -> None:
        self._store_id = store_id
        self._lock = threading.Lock()
        self._entries: list[MemoryEntry] = []
        self._embeddings: list[list[float]] = []
        self._session_count: int = 0

    @classmethod
    def get_or_create(
        cls,
        shuffle_mode: str = "isolated",
        benchmark_id: str | None = None,
    ) -> "ReasoningBankStore":
        if shuffle_mode == "isolated":
            store_id = f"rb_isolated_{benchmark_id or 'default'}"
        elif shuffle_mode == "sequential":
            store_id = "rb_sequential_global"
        elif shuffle_mode == "interleaved":
            store_id = "rb_interleaved_global"
        else:
            store_id = f"rb_{shuffle_mode}_{benchmark_id or 'default'}"

        with cls._class_lock:
            if store_id not in cls._instances:
                cls._instances[store_id] = cls(store_id)
            return cls._instances[store_id]

    @classmethod
    def list_stores(cls) -> dict[str, "ReasoningBankStore"]:
        with cls._class_lock:
            return dict(cls._instances)

    @classmethod
    def reset_all(cls) -> None:
        with cls._class_lock:
            cls._instances.clear()

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def session_count(self) -> int:
        with self._lock:
            return self._session_count

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def increment_session(self) -> int:
        with self._lock:
            self._session_count += 1
            return self._session_count

    def add_entry(self, entry: MemoryEntry, embedding: list[float]) -> None:
        with self._lock:
            self._entries.append(entry)
            self._embeddings.append(embedding)

    def get_entries(self) -> list[MemoryEntry]:
        with self._lock:
            return list(self._entries)

    def get_embeddings_array(self) -> np.ndarray | None:
        with self._lock:
            if not self._embeddings:
                return None
            return np.array(self._embeddings, dtype=np.float32)

    def get_entry_ids(self) -> list[str]:
        with self._lock:
            return [e.task_id for e in self._entries]

    def save_checkpoint(self, path: str) -> None:
        with self._lock:
            data = {
                "store_id": self._store_id,
                "session_count": self._session_count,
                "entries": [e.to_dict() for e in self._entries],
                "embeddings": self._embeddings,
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("ReasoningBankStore[%s]: saved checkpoint (%d entries) to %s",
                    self._store_id, len(self._entries), path)

    def save_memories_text(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            lines = [
                f"# ReasoningBank Memory Store: {self._store_id}",
                f"# Entries: {len(self._entries)}",
                f"# Sessions: {self._session_count}",
                "",
            ]
            for i, entry in enumerate(self._entries):
                lines.append(f"--- Entry {i + 1} [{entry.task_id[:20]}] status={entry.status} ---")
                for item in entry.memory_items:
                    lines.append(item.strip())
                lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def load_checkpoint(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self._session_count = data.get("session_count", 0)
            self._entries = [MemoryEntry.from_dict(d) for d in data.get("entries", [])]
            self._embeddings = data.get("embeddings", [])
        logger.info("ReasoningBankStore[%s]: loaded checkpoint (%d entries) from %s",
                    self._store_id, len(self._entries), path)
