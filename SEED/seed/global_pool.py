"""Global experience pool for SEED dual-track OPD.

V1 semantics (filter-only): local hindsight skills that pass an LLM-as-judge
transferability screen are stored verbatim and retrieved for other tasks by
ReasoningBank-style query-embedding similarity. Retrieval answers "is this
skill relevant to the current task"; the per-token OPD gate in the loss stays
the safety net for wrong retrievals; a single scalar gate EMA per skill is
kept only to evict dead weight (it never participates in retrieval).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POOL_SCHEMA_VERSION = 1
GLOBAL_POOL_SOURCES = ("copy", "pool")
EMBED_BACKENDS = ("local", "http")


@dataclass(frozen=True)
class GlobalPoolConfig:
    source: str = "copy"
    capacity: int = 64
    score_threshold: float = 0.6
    min_sim: float = 0.35
    dedup_sim: float = 0.9
    ema_alpha: float = 0.1
    max_candidates_per_step: int = 16
    admit_failed: bool = False
    judge_model: str = "z-ai/glm-5.2"
    judge_base_url: str = "https://openrouter.ai/api/v1"
    judge_api_key_env: str = "OPENROUTER_API_KEY"
    judge_batch_size: int = 12
    embed_backend: str = "local"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_url: Optional[str] = None
    save_dir: Optional[str] = None

    def validate(self) -> "GlobalPoolConfig":
        if self.source not in GLOBAL_POOL_SOURCES:
            raise ValueError(f"global_pool.source must be one of {GLOBAL_POOL_SOURCES}, got {self.source!r}.")
        if self.embed_backend not in EMBED_BACKENDS:
            raise ValueError(f"global_pool.embed_backend must be one of {EMBED_BACKENDS}, got {self.embed_backend!r}.")
        if self.embed_backend == "http" and not self.embed_url:
            raise ValueError("global_pool.embed_backend='http' requires global_pool.embed_url.")
        if self.capacity <= 0:
            raise ValueError("global_pool.capacity must be positive.")
        return self


def normalize_skill_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def skill_id_for(text: object) -> str:
    return hashlib.sha1(normalize_skill_text(text).encode("utf-8")).hexdigest()[:16]


_QUERY_SEGMENT_CHARS = 600  # both halves must fit MiniLM's ~256-token window


def build_retrieval_query(task_text: object, first_obs: object) -> str:
    """Embedding query for pool retrieval: task text plus the first observation.

    Several benchmarks keep the task identity outside the task string (tau2's
    lives entirely in the user's opening message; browsecompplus/hle put the
    question after a constant first line), so the first observation must
    participate. The per-segment cap keeps the observation inside the encoder
    window even when the task text is long (appworld instructions).
    """
    head = str(task_text or "").strip()[:_QUERY_SEGMENT_CHARS]
    tail = str(first_obs or "").strip()[:_QUERY_SEGMENT_CHARS]
    return f"{head}\n{tail}".strip()


def select_admission_candidates(
    scored: Sequence[Tuple[dict, Optional[float]]], limit: int
) -> List[dict]:
    """Pick admission candidates from (candidate, spec_gap) pairs.

    A GRPO group's same-task copies produce near-duplicate skills, so only the
    highest-gap candidate per ``task_key`` survives, and the per-step cap then
    truncates by gap rather than by batch order (which would starve tasks that
    happen to sit late in the batch). ``gap`` semantics: ``<= 0`` drops the
    candidate (the skill did not help the spec teacher), ``None`` means no spec
    evidence — such candidates pass through but rank behind every scored one.
    """
    best: Dict[str, Tuple[float, dict]] = {}
    for candidate, gap in scored:
        if gap is not None and gap <= 0:
            continue
        sort_key = float("-inf") if gap is None else float(gap)
        task_key = str(candidate.get("task_key", ""))
        current = best.get(task_key)
        if current is None or sort_key > current[0]:
            best[task_key] = (sort_key, candidate)
    ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked[: max(int(limit), 0)]]


class TextEmbedder:
    """Small frozen-model text embedder (unit-normalized vectors).

    ``local`` runs a HF encoder with mean pooling on CPU; ``http`` POSTs
    {"texts": [...]} to ``embed_url`` and expects {"embeddings": [[...]]}.
    A frozen model keeps pool keys and queries in one stable vector space
    (the training policy's own hidden states would drift every step).
    """

    def __init__(self, backend: str = "local", model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 url: Optional[str] = None, device: str = "cpu", max_length: int = 256):
        if backend not in EMBED_BACKENDS:
            raise ValueError(f"Unsupported embed backend: {backend!r}")
        self.backend = backend
        self.model_name = model
        self.url = url
        self.device = device
        self.max_length = int(max_length)
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _load_local(self):
        if self._model is None:
            import torch  # noqa: F401  (ensures torch present before transformers loads weights)
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        return self._tokenizer, self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t or "") for t in texts]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "http":
            vectors = self._encode_http(texts)
        else:
            vectors = self._encode_local(texts)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.clip(norms, 1e-8, None)).astype(np.float32)

    def _encode_http(self, texts: List[str]) -> np.ndarray:
        import requests

        response = requests.post(self.url, json={"texts": texts}, timeout=30)
        response.raise_for_status()
        return np.asarray(response.json()["embeddings"], dtype=np.float32)

    def _encode_local(self, texts: List[str]) -> np.ndarray:
        import torch

        with self._lock:  # HF modules are not thread-safe; retrieval and admission may race
            tokenizer, model = self._load_local()
            with torch.no_grad():
                batch = tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        return pooled.cpu().numpy()


_EVICT_NEUTRAL_PRIOR = 0.5  # never-injected entries compete as a coin flip, not as immortal


@dataclass
class RetrievalHit:
    skill_id: str
    text: str
    similarity: float


class GlobalSkillPool:
    """Bounded skill store with embedding retrieval and gate-EMA eviction.

    Thread-safe: retrieval runs on the trainer thread while admission runs on
    a background executor.
    """

    def __init__(
        self,
        config: GlobalPoolConfig,
        save_path: Optional[str] = None,
        *,
        load_existing: bool = True,
        max_global_step: Optional[int] = None,
    ):
        self.config = config.validate()
        self.save_path = save_path
        self._entries: Dict[str, dict] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        # load_existing=False keeps a fresh run (trainer.resume_mode=disable) from
        # hot-starting off a stale same-name pool; max_global_step trims a resumed
        # pool back to the checkpointed step (admission saves eagerly, so on a
        # crash the file runs ahead of the checkpoint).
        if load_existing and save_path and os.path.exists(save_path):
            self.load(save_path, max_global_step=max_global_step)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def has(self, skill_id: str) -> bool:
        with self._lock:
            return skill_id in self._entries

    def add(self, *, text: str, embedding: np.ndarray, source: dict, judge: dict, global_step: int) -> str:
        """Insert one judged skill. Returns 'added' | 'duplicate' | 'merged'."""
        skill_id = skill_id_for(text)
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        with self._lock:
            if skill_id in self._entries:
                self._entries[skill_id]["support"] += 1
                return "duplicate"
            near = self._nearest_locked(embedding, exclude_task_key=None)
            if near is not None and near[1] >= self.config.dedup_sim:
                self._entries[near[0]]["support"] += 1
                return "merged"
            if len(self._entries) >= self.config.capacity:
                self._evict_locked()
            self._entries[skill_id] = {
                "skill_id": skill_id,
                "text": str(text),
                "source": dict(source),
                "judge": dict(judge),
                "stats": {"times_injected": 0, "gate_ema": None, "last_used_step": int(global_step)},
                "support": 1,
                "status": "active",
            }
            self._embeddings[skill_id] = embedding
            return "added"

    def retrieve(self, query_embeddings: np.ndarray, task_keys: Sequence[str]) -> List[Optional[RetrievalHit]]:
        """Top-1 cosine hit per query above min_sim, excluding same-task entries."""
        query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
        hits: List[Optional[RetrievalHit]] = []
        with self._lock:
            for query, task_key in zip(query_embeddings, task_keys):
                near = self._nearest_locked(query, exclude_task_key=str(task_key))
                if near is not None and near[1] >= self.config.min_sim:
                    hits.append(RetrievalHit(skill_id=near[0], text=self._entries[near[0]]["text"], similarity=near[1]))
                else:
                    hits.append(None)
        return hits

    def record_usage(self, skill_id: str, gate_value: float, global_step: int) -> None:
        with self._lock:
            entry = self._entries.get(skill_id)
            if entry is None:
                return
            stats = entry["stats"]
            stats["times_injected"] += 1
            stats["last_used_step"] = int(global_step)
            previous = stats["gate_ema"]
            alpha = float(self.config.ema_alpha)
            stats["gate_ema"] = float(gate_value) if previous is None else (1 - alpha) * float(previous) + alpha * float(gate_value)

    def _nearest_locked(self, query: np.ndarray, exclude_task_key: Optional[str]):
        best = None
        for skill_id, entry in self._entries.items():
            if exclude_task_key is not None and entry["source"].get("task_key") == exclude_task_key:
                continue
            similarity = float(np.dot(self._embeddings[skill_id], query))
            if best is None or similarity > best[1]:
                best = (skill_id, similarity)
        return best

    def _evict_locked(self) -> None:
        # One utility scale for everyone: entries with a proven-bad gate EMA
        # (below the neutral prior) go before never-injected ones, proven-good
        # entries outlive them, and ties fall to the stalest entry. A tiered
        # ordering ("injected always dies first") would churn validated skills
        # while never-retrieved ones squat forever.
        def _utility(item):
            stats = item[1]["stats"]
            ema = stats["gate_ema"]
            return (_EVICT_NEUTRAL_PRIOR if ema is None else float(ema), stats["last_used_step"])

        victim = min(self._entries.items(), key=_utility)[0]
        self._entries.pop(victim)
        self._embeddings.pop(victim, None)
        logger.info("Global skill pool evicted skill_id=%s at capacity=%s.", victim, self.config.capacity)

    def snapshot_metrics(self) -> Dict[str, float]:
        with self._lock:
            size = len(self._entries)
            emas = [e["stats"]["gate_ema"] for e in self._entries.values() if e["stats"]["gate_ema"] is not None]
            supports = [e["support"] for e in self._entries.values()]
        return {
            "seed/global_pool/size": float(size),
            "seed/global_pool/gate_ema_mean": float(np.mean(emas)) if emas else 0.0,
            "seed/global_pool/support_mean": float(np.mean(supports)) if supports else 0.0,
        }

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.save_path
        if not path:
            return
        with self._lock:
            payload = {
                "version": POOL_SCHEMA_VERSION,
                "embed_model": self.config.embed_model,
                "entries": list(self._entries.values()),
            }
            order = list(self._entries.keys())
            matrix = np.stack([self._embeddings[k] for k in order]) if order else np.zeros((0, 1), dtype=np.float32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Atomic JSON write + sidecar .npy keyed by entry order.
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump({**payload, "embedding_order": order}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        np.save(path + ".npy", matrix)

    def load(self, path: str, max_global_step: Optional[int] = None) -> None:
        """Restore entries and embeddings written by :meth:`save`.

        ``max_global_step`` drops entries admitted after that trainer step
        (resume alignment). Any inconsistency — schema or embedder mismatch,
        corrupt JSON, an .npy sidecar whose row count disagrees (the JSON write
        is atomic but the sidecar follows separately, so a kill between the two
        leaves them out of sync) — starts empty instead of raising: a damaged
        auxiliary file must never take training down.
        """
        try:
            with open(path) as f:
                payload = json.load(f)
            if int(payload.get("version", 0)) != POOL_SCHEMA_VERSION:
                logger.warning("Global skill pool at %s has version %s != %s; starting empty.", path, payload.get("version"), POOL_SCHEMA_VERSION)
                return
            embed_model = str(payload.get("embed_model", ""))
            if embed_model != self.config.embed_model:
                logger.warning(
                    "Global skill pool at %s was embedded with %r but %r is configured; starting empty.",
                    path, embed_model, self.config.embed_model,
                )
                return
            entries = list(payload.get("entries", []))
            order = list(payload.get("embedding_order", []))
            matrix = np.load(path + ".npy") if os.path.exists(path + ".npy") else None
            if matrix is None or matrix.ndim != 2 or matrix.shape[0] != len(order) or len(order) != len(entries):
                logger.warning("Global skill pool at %s is missing consistent embeddings; starting empty.", path)
                return
            if max_global_step is not None:
                entries = [
                    entry for entry in entries
                    if int((entry.get("source") or {}).get("global_step", 0)) <= int(max_global_step)
                ]
            row_of = {skill_id: i for i, skill_id in enumerate(order)}
            with self._lock:
                self._embeddings = {
                    entry["skill_id"]: matrix[row_of[entry["skill_id"]]]
                    for entry in entries
                    if entry["skill_id"] in row_of
                }
                # An entry without an embedding can never be retrieved and would
                # crash _nearest_locked, so it does not survive the load either.
                self._entries = {entry["skill_id"]: entry for entry in entries if entry["skill_id"] in self._embeddings}
            logger.info("Loaded global skill pool with %s entries from %s.", len(self._entries), path)
        except Exception:
            with self._lock:
                self._entries, self._embeddings = {}, {}
            logger.exception("Failed to load global skill pool from %s; starting empty.", path)
