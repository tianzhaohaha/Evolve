# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import copy
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


DEFAULT_SYSTEM_PROMPT = """\
You are an expert agent that completes tasks using available tools.
Think step-by-step before acting.
Use available tools to interact with the environment.
When you are confident in your solution, use the finish/submit tool."""

DEFAULT_MEMORY = ""


@dataclass
class HarnessSkill:

    name: str
    description: str
    body: str
    last_used_session: int = 0
    created_session: int = 0


@dataclass
class VersionEntry:

    version: int
    session_id: str
    session_count: int
    ops_summary: List[str]
    timestamp: str = ""
    system_prompt: str = ""
    memory: str = ""
    skills: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningEvent:

    session_id: str
    task_id: str
    benchmark_id: str
    ops_applied: int
    ops_summary: List[str] = field(default_factory=list)
    timestamp: str = ""


class HarnessStore:

    _instances: Dict[str, "HarnessStore"] = {}
    _class_lock = threading.Lock()

    def __init__(self, store_id: str) -> None:
        self._store_id = store_id
        self._lock = threading.Lock()
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.memory: str = DEFAULT_MEMORY
        self.skills: Dict[str, HarnessSkill] = {}
        self.skill_embeddings: Dict[str, List[float]] = {}
        self._session_count: int = 0
        self._history: List[LearningEvent] = []
        self._versions: List[VersionEntry] = []
        self._next_version: int = 0

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
            return len(self.skills)

    @classmethod
    def get_or_create(
        cls,
        shuffle_mode: str = "isolated",
        benchmark_id: Optional[str] = None,
        task_group: Optional[str] = None,
    ) -> "HarnessStore":
        if shuffle_mode == "isolated":
            key = f"harness_isolated_{benchmark_id or task_group or 'default'}"
        elif shuffle_mode == "sequential":
            key = "harness_sequential_global"
        elif shuffle_mode == "interleaved":
            key = "harness_interleaved_global"
        else:
            key = f"harness_{shuffle_mode}"

        with cls._class_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(store_id=key)
            return cls._instances[key]

    @classmethod
    def list_stores(cls) -> Dict[str, "HarnessStore"]:
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

    def add_skill(self, name: str, description: str, body: str) -> None:
        with self._lock:
            self.skills[name] = HarnessSkill(
                name=name,
                description=description,
                body=body,
                last_used_session=self._session_count,
                created_session=self._session_count,
            )

    def edit_skill(
        self, name: str, description: Optional[str] = None, body: Optional[str] = None
    ) -> bool:
        with self._lock:
            skill = self.skills.get(name)
            if skill is None:
                return False
            if description is not None:
                skill.description = description
            if body is not None:
                skill.body = body
            return True

    def delete_skill(self, name: str) -> bool:
        with self._lock:
            if name in self.skills:
                del self.skills[name]
                self.skill_embeddings.pop(name, None)
                return True
            return False

    def touch_skills(self, names: List[str]) -> None:
        with self._lock:
            for name in names:
                skill = self.skills.get(name)
                if skill:
                    skill.last_used_session = self._session_count

    def get_skill_index(self) -> List[Dict[str, str]]:
        with self._lock:
            return [
                {"name": s.name, "description": s.description}
                for s in self.skills.values()
            ]

    def list_skills(self) -> List[HarnessSkill]:
        with self._lock:
            return list(self.skills.values())

    def edit_prompt(self, new_prompt: str) -> None:
        with self._lock:
            self.system_prompt = new_prompt

    def edit_memory(self, new_memory: str) -> None:
        with self._lock:
            self.memory = new_memory

    def set_embedding(self, skill_name: str, embedding: List[float]) -> None:
        with self._lock:
            self.skill_embeddings[skill_name] = embedding

    def get_embeddings(self) -> Dict[str, List[float]]:
        with self._lock:
            return dict(self.skill_embeddings)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "system_prompt": self.system_prompt,
                "memory": self.memory,
                "skills": copy.deepcopy(self.skills),
                "skill_embeddings": copy.deepcopy(self.skill_embeddings),
            }

    def rollback(self, snap: Dict[str, Any]) -> None:
        with self._lock:
            self.system_prompt = snap["system_prompt"]
            self.memory = snap["memory"]
            self.skills = snap["skills"]
            self.skill_embeddings = snap["skill_embeddings"]

    def commit_version(self, session_id: str, ops_summary: List[str]) -> int:
        with self._lock:
            version = self._next_version
            self._next_version += 1
            entry = VersionEntry(
                version=version,
                session_id=session_id,
                session_count=self._session_count,
                ops_summary=ops_summary,
                timestamp=datetime.now().isoformat(),
                system_prompt=self.system_prompt,
                memory=self.memory,
                skills={name: asdict(s) for name, s in self.skills.items()},
            )
            self._versions.append(entry)
            return version

    def record_learning(
        self,
        session_id: str,
        task_id: str,
        benchmark_id: str,
        ops_applied: int,
        ops_summary: Optional[List[str]] = None,
    ) -> None:
        with self._lock:
            self._history.append(LearningEvent(
                session_id=session_id,
                task_id=task_id,
                benchmark_id=benchmark_id,
                ops_applied=ops_applied,
                ops_summary=ops_summary or [],
                timestamp=datetime.now().isoformat(),
            ))

    def save_checkpoint(self, path: str) -> None:
        with self._lock:
            data = {
                "store_id": self._store_id,
                "session_count": self._session_count,
                "system_prompt": self.system_prompt,
                "memory": self.memory,
                "skills": {name: asdict(s) for name, s in self.skills.items()},
                "skill_embeddings": self.skill_embeddings,
                "history": [asdict(e) for e in self._history],
                "versions": [
                    {
                        "version": v.version,
                        "session_id": v.session_id,
                        "session_count": v.session_count,
                        "ops_summary": v.ops_summary,
                        "timestamp": v.timestamp,
                        "skill_names": list(v.skills.keys()),
                    }
                    for v in self._versions
                ],
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        with self._lock:
            self._session_count = data.get("session_count", 0)
            self.system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            self.memory = data.get("memory", DEFAULT_MEMORY)
            self.skills = {}
            for name, sdata in data.get("skills", {}).items():
                self.skills[name] = HarnessSkill(**{
                    k: v for k, v in sdata.items()
                    if k in HarnessSkill.__dataclass_fields__
                })
            self.skill_embeddings = data.get("skill_embeddings", {})
            self._history = [
                LearningEvent(**{
                    k: v for k, v in e.items()
                    if k in LearningEvent.__dataclass_fields__
                })
                for e in data.get("history", [])
            ]
            self._versions = []
            for vdata in data.get("versions", []):
                self._versions.append(VersionEntry(
                    version=vdata["version"],
                    session_id=vdata.get("session_id", ""),
                    session_count=vdata.get("session_count", 0),
                    ops_summary=vdata.get("ops_summary", []),
                    timestamp=vdata.get("timestamp", ""),
                    skills={name: {} for name in vdata.get("skill_names", [])},
                ))
            self._next_version = (
                self._versions[-1].version + 1 if self._versions else 0
            )

    def save_harness_text(self, path: str) -> None:
        with self._lock:
            skills = list(self.skills.values())
        lines = [
            f"# Harness State: {self._store_id}",
            f"## System Prompt",
            self.system_prompt,
            "",
            f"## Memory",
            self.memory,
            "",
            f"## Skills ({len(skills)})",
        ]
        for s in skills:
            lines.append(f"### {s.name}")
            lines.append(f"Description: {s.description}")
            lines.append(f"Last used: session {s.last_used_session}")
            lines.append(s.body)
            lines.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
