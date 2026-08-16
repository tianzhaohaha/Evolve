# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .memory_note import MemoryNote
from .prompts import (
    EVOLUTION_DECISION_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
)
from .retriever import EmbeddingRetriever

logger = logging.getLogger("amem")


@dataclass
class LearningEvent:

    session_id: str
    task_id: str
    step: int
    memories_added: int
    evolutions_triggered: int
    summary: str
    benchmark_id: str = ""


class MemoryStore:

    _instances: Dict[str, "MemoryStore"] = {}
    _global_lock = threading.Lock()

    @classmethod
    def get_or_create(
        cls,
        shuffle_mode: str = "isolated",
        task_group: Optional[str] = None,
        benchmark_id: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        evo_threshold: int = 100,
    ) -> "MemoryStore":
        if shuffle_mode == "isolated":
            bm = benchmark_id or task_group or "default"
            key = f"amem_isolated_{bm}"
        elif shuffle_mode == "sequential":
            key = "amem_sequential_global"
        elif shuffle_mode == "interleaved":
            key = "amem_interleaved_global"
        else:
            raise ValueError(f"Unknown shuffle_mode: {shuffle_mode!r}")

        with cls._global_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    store_id=key,
                    embedding_model=embedding_model,
                    evo_threshold=evo_threshold,
                )
            return cls._instances[key]

    @classmethod
    def reset_all(cls) -> None:
        with cls._global_lock:
            cls._instances.clear()

    @classmethod
    def list_stores(cls) -> Dict[str, "MemoryStore"]:
        with cls._global_lock:
            return dict(cls._instances)

    def __init__(
        self,
        store_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        evo_threshold: int = 100,
    ) -> None:
        self.store_id = store_id
        self._lock = threading.Lock()

        self._memories: Dict[str, MemoryNote] = {}
        self._retriever = EmbeddingRetriever(embedding_model)
        self._embedding_model = embedding_model
        self._evo_threshold = evo_threshold
        self._evo_cnt: int = 0

        self._session_count: int = 0
        self._history: List[LearningEvent] = []
        self._benchmark_counts: Dict[str, int] = {}


    @property
    def memory_count(self) -> int:
        with self._lock:
            return len(self._memories)

    def get_all_memories(self) -> List[MemoryNote]:
        with self._lock:
            return list(self._memories.values())


    @property
    def session_count(self) -> int:
        with self._lock:
            return self._session_count

    def increment_session(self) -> int:
        with self._lock:
            self._session_count += 1
            return self._session_count

    def add_memory(
        self,
        note: MemoryNote,
        llm_call: Callable[[str], str],
    ) -> bool:
        with self._lock:
            evolved = self._process_and_add(note, llm_call)
            return evolved

    def find_related_with_neighbors(
        self,
        query: str,
        k: int = 10,
    ) -> List[MemoryNote]:
        with self._lock:
            if not self._memories:
                return []
            indices = self._retriever.search(query, k)
            all_memories = list(self._memories.values())
            seen: set[int] = set()
            results: List[MemoryNote] = []

            for i in indices:
                if i >= len(all_memories) or i in seen:
                    continue
                seen.add(i)
                note = all_memories[i]
                note.retrieval_count += 1
                note.last_accessed = datetime.now().strftime("%Y%m%d%H%M")
                results.append(note)
                for link_idx in note.links:
                    if link_idx < len(all_memories) and link_idx not in seen:
                        seen.add(link_idx)
                        linked = all_memories[link_idx]
                        linked.retrieval_count += 1
                        results.append(linked)

            return results

    def _process_and_add(
        self,
        note: MemoryNote,
        llm_call: Callable[[str], str],
    ) -> bool:
        neighbor_str, indices = self._find_neighbors_for_evolution(
            note.content, k=5
        )

        evolved = False
        if indices:
            try:
                evolved = self._run_evolution(note, neighbor_str, indices, llm_call)
            except Exception as e:
                logger.error(
                    "Evolution failed for note %s: %s -- storing without evolution",
                    note.id[:8],
                    e,
                )

        self._memories[note.id] = note
        self._retriever.add_documents([note.to_retrieval_document()])

        if evolved:
            self._evo_cnt += 1
            if self._evo_cnt % self._evo_threshold == 0:
                self._consolidate()

        return evolved

    def _find_neighbors_for_evolution(
        self, query: str, k: int = 5
    ) -> Tuple[str, List[int]]:
        if not self._memories:
            return "", []

        indices = self._retriever.search(query, k)
        all_memories = list(self._memories.values())
        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                f"memory index:{i}"
                f"\t talk start time:{m.timestamp}"
                f"\t memory content: {m.content}"
                f"\t memory context: {m.context}"
                f"\t memory keywords: {m.keywords}"
                f"\t memory tags: {m.tags}\n"
            )
        return memory_str, indices

    def _run_evolution(
        self,
        note: MemoryNote,
        neighbor_str: str,
        indices: List[int],
        llm_call: Callable[[str], str],
    ) -> bool:
        decision_prompt = EVOLUTION_DECISION_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_str,
        )
        decision_response = llm_call(decision_prompt)
        decision = parse_evolution_decision(decision_response)
        logger.debug("Evolution decision: %s", decision)

        if decision["decision"] == "NO_EVOLUTION":
            return False

        should_strengthen = decision["decision"] in (
            "STRENGTHEN", "STRENGTHEN_AND_UPDATE"
        )
        should_update = decision["decision"] in (
            "UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE"
        )

        if should_strengthen:
            strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_str,
            )
            strengthen_response = llm_call(strengthen_prompt)
            strengthen = parse_strengthen_details(strengthen_response)
            logger.debug("Strengthen details: %s", strengthen)

            note.links.extend(strengthen["connections"])
            if strengthen["tags"]:
                note.tags = strengthen["tags"]

        if should_update:
            update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                content=note.content,
                context=note.context,
                nearest_neighbors_memories=neighbor_str,
                max_neighbor_idx=len(indices) - 1,
                neighbor_count=len(indices),
            )
            update_response = llm_call(update_prompt)
            neighbor_updates = parse_update_neighbors(
                update_response, len(indices)
            )
            logger.debug("Neighbor updates: %s", neighbor_updates)

            noteslist = list(self._memories.values())
            notes_id = list(self._memories.keys())
            for i in range(min(len(indices), len(neighbor_updates))):
                upd = neighbor_updates[i]
                memorytmp_idx = indices[i]
                if memorytmp_idx >= len(noteslist):
                    continue
                notetmp = noteslist[memorytmp_idx]
                if upd["tags"]:
                    notetmp.tags = upd["tags"]
                if upd["context"]:
                    notetmp.context = upd["context"]
                self._memories[notes_id[memorytmp_idx]] = notetmp

        return True

    def _consolidate(self) -> None:
        logger.info(
            "Consolidating memory retriever (%d memories, %d evolutions)",
            len(self._memories),
            self._evo_cnt,
        )
        documents = [m.to_retrieval_document() for m in self._memories.values()]
        self._retriever.reset(documents)

    def record_learning(
        self,
        session_id: str,
        task_id: str,
        memories_added: int,
        evolutions_triggered: int,
        summary: str,
        benchmark_id: str = "",
    ) -> None:
        with self._lock:
            self._history.append(
                LearningEvent(
                    session_id=session_id,
                    task_id=task_id,
                    step=self._session_count,
                    memories_added=memories_added,
                    evolutions_triggered=evolutions_triggered,
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
                "memory_count": len(self._memories),
                "evo_cnt": self._evo_cnt,
                "session_count": self._session_count,
                "history_len": len(self._history),
                "benchmark_counts": dict(self._benchmark_counts),
                "memories": {
                    mid: note.to_dict()
                    for mid, note in self._memories.items()
                },
            }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def load_checkpoint(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        with self._lock:
            self._evo_cnt = payload.get("evo_cnt", 0)
            self._session_count = payload.get("session_count", 0)
            self._benchmark_counts = payload.get("benchmark_counts", {})
            memories_data = payload.get("memories", {})
            self._memories = {
                mid: MemoryNote.from_dict(mdata)
                for mid, mdata in memories_data.items()
            }
            if self._memories:
                documents = [
                    m.to_retrieval_document() for m in self._memories.values()
                ]
                self._retriever.reset(documents)

    def save_memories_text(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            lines = [
                f"# A-Mem Memory Store: {self.store_id}",
                f"# Memories: {len(self._memories)}",
                f"# Sessions: {self._session_count}",
                f"# Evolutions: {self._evo_cnt}",
                "",
            ]
            for i, note in enumerate(self._memories.values()):
                lines.append(f"--- Memory {i + 1} [{note.id[:8]}] ---")
                lines.append(f"Content: {note.content}")
                lines.append(f"Context: {note.context}")
                lines.append(f"Keywords: {', '.join(note.keywords)}")
                lines.append(f"Tags: {', '.join(note.tags)}")
                links_str = (
                    ", ".join(str(l) for l in note.links) if note.links else "none"
                )
                lines.append(f"Links: {links_str}")
                lines.append(
                    f"Importance: {note.importance_score:.2f}  "
                    f"Retrieved: {note.retrieval_count} times"
                )
                lines.append("")

        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._memories)
            if total == 0:
                return {
                    "total_memories": 0,
                    "total_evolutions": self._evo_cnt,
                    "avg_links": 0.0,
                    "avg_keywords": 0.0,
                    "avg_importance": 0.0,
                    "most_retrieved": 0,
                }
            links_count = sum(
                len(m.links) for m in self._memories.values()
            )
            kw_count = sum(
                len(m.keywords) for m in self._memories.values()
            )
            imp_sum = sum(
                m.importance_score for m in self._memories.values()
            )
            max_retr = max(
                m.retrieval_count for m in self._memories.values()
            )
            return {
                "total_memories": total,
                "total_evolutions": self._evo_cnt,
                "avg_links": links_count / total,
                "avg_keywords": kw_count / total,
                "avg_importance": imp_sum / total,
                "most_retrieved": max_retr,
            }
