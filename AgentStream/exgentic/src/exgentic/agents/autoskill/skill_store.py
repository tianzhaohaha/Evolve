# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class SkillEntry:

    id: str
    name: str
    description: str
    instructions: str
    triggers: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "0.1.0"
    confidence: float = 0.5
    created_at: str = ""
    updated_at: str = ""

    def to_search_text(self) -> str:
        parts = [self.name, self.description]
        parts.extend(self.triggers)
        parts.extend(self.tags)
        parts.append(self.instructions)
        return " ".join(parts)

    def bump_version(self) -> None:
        parts = self.version.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
            self.version = ".".join(parts)
        else:
            self.version = "0.1.1"
        self.updated_at = datetime.now().isoformat()


@dataclass
class LearningEvent:

    session_id: str
    task_id: str
    benchmark_id: str
    action: str
    skill_name: str
    timestamp: str = ""


class SkillStore:

    _instances: Dict[str, "SkillStore"] = {}
    _class_lock = threading.Lock()

    def __init__(self, store_id: str) -> None:
        self._store_id = store_id
        self._lock = threading.Lock()
        self._skills: Dict[str, SkillEntry] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._session_count: int = 0
        self._history: List[LearningEvent] = []

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def session_count(self) -> int:
        with self._lock:
            return self._session_count

    @property
    def skill_count(self) -> int:
        with self._lock:
            return len(self._skills)

    @classmethod
    def get_or_create(
        cls,
        shuffle_mode: str = "isolated",
        benchmark_id: Optional[str] = None,
        task_group: Optional[str] = None,
    ) -> "SkillStore":
        if shuffle_mode == "isolated":
            key = f"autoskill_isolated_{benchmark_id or task_group or 'default'}"
        elif shuffle_mode == "sequential":
            key = "autoskill_sequential_global"
        elif shuffle_mode == "interleaved":
            key = "autoskill_interleaved_global"
        else:
            key = f"autoskill_{shuffle_mode}"

        with cls._class_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(store_id=key)
            return cls._instances[key]

    @classmethod
    def list_stores(cls) -> Dict[str, "SkillStore"]:
        with cls._class_lock:
            return dict(cls._instances)

    @classmethod
    def reset_all(cls) -> None:
        with cls._class_lock:
            cls._instances.clear()

    def increment_session(self) -> int:
        with self._lock:
            self._session_count += 1
            return self._session_count

    def add_skill(self, skill: SkillEntry) -> None:
        with self._lock:
            if not skill.id:
                skill.id = str(uuid.uuid4())
            if not skill.created_at:
                skill.created_at = datetime.now().isoformat()
            skill.updated_at = skill.created_at
            self._skills[skill.id] = skill

    def update_skill(self, skill: SkillEntry) -> None:
        with self._lock:
            skill.updated_at = datetime.now().isoformat()
            self._skills[skill.id] = skill

    def list_skills(self) -> List[SkillEntry]:
        with self._lock:
            return list(self._skills.values())

    def set_embedding(self, skill_id: str, embedding: List[float]) -> None:
        with self._lock:
            self._embeddings[skill_id] = embedding

    def get_embeddings(self) -> Dict[str, List[float]]:
        with self._lock:
            return dict(self._embeddings)

    def record_learning(
        self,
        session_id: str,
        task_id: str,
        benchmark_id: str,
        action: str,
        skill_name: str = "",
    ) -> None:
        with self._lock:
            self._history.append(LearningEvent(
                session_id=session_id,
                task_id=task_id,
                benchmark_id=benchmark_id,
                action=action,
                skill_name=skill_name,
                timestamp=datetime.now().isoformat(),
            ))

    def save_checkpoint(self, path: str) -> None:
        with self._lock:
            data = {
                "store_id": self._store_id,
                "session_count": self._session_count,
                "skills": {sid: asdict(s) for sid, s in self._skills.items()},
                "embeddings": self._embeddings,
                "history": [asdict(e) for e in self._history],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self._session_count = data.get("session_count", 0)
            self._skills = {}
            for sid, sdata in data.get("skills", {}).items():
                self._skills[sid] = SkillEntry(**{
                    k: v for k, v in sdata.items()
                    if k in SkillEntry.__dataclass_fields__
                })
            self._embeddings = data.get("embeddings", {})
            self._history = [
                LearningEvent(**{
                    k: v for k, v in e.items()
                    if k in LearningEvent.__dataclass_fields__
                })
                for e in data.get("history", [])
            ]

    def save_skills_text(self, path: str) -> None:
        with self._lock:
            skills = list(self._skills.values())
        lines = [f"# SkillBank: {self._store_id} ({len(skills)} skills)\n"]
        for s in skills:
            lines.append(f"## {s.name} (v{s.version})")
            lines.append(f"   {s.description}")
            lines.append(f"   Tags: {', '.join(s.tags)}")
            lines.append(f"   Triggers: {', '.join(s.triggers)}")
            lines.append(f"   Instructions: {s.instructions}")
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
