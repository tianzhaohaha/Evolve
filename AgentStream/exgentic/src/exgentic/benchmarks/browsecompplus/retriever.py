# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""Shared retriever service for BrowseCompPlus search index.

The Retriever loads the heavy search index once and serves queries.
It can run via any runner (direct, service, docker, etc.), allowing
a single index copy to be shared across all sessions.
"""

import argparse
import json
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue
from typing import Any, Callable


class _QueryBatcher:
    """Coalesces concurrent ``search`` calls into one encoder forward.

    The HTTP host serves calls from a thread pool and a lock-step rollout
    (SEED) sends one query per environment slot at the same moment. A
    single-query forward of a large encoder is bound by reading its weights
    (on CPU especially), so embedding B queries together costs about the same
    as one. Requests arriving within ``window_s`` of the first one (up to
    ``max_batch``) are embedded in one call and searched with the largest k
    of the batch; each caller gets exactly its own top-k.
    """

    def __init__(self, run_batch: Callable[[list[str], int], list[list]], max_batch: int, window_s: float) -> None:
        self._run_batch = run_batch
        self._max_batch = max_batch
        self._window_s = window_s
        self._queue: Queue = Queue()
        threading.Thread(target=self._loop, name="retriever-batcher", daemon=True).start()

    def submit(self, query: str, k: int) -> list:
        future: Future = Future()
        self._queue.put((query, k, future))
        return future.result()

    def _loop(self) -> None:
        while True:
            batch = [self._queue.get()]
            time.sleep(self._window_s)
            while len(batch) < self._max_batch:
                try:
                    batch.append(self._queue.get_nowait())
                except Empty:
                    break
            try:
                results = self._run_batch([q for q, _, _ in batch], max(k for _, k, _ in batch))
                for (_, k, future), result in zip(batch, results):
                    future.set_result(result[:k])
            except Exception as exc:  # every waiting caller sees the failure
                for _, _, future in batch:
                    future.set_exception(exc)


class Retriever:
    """Loads a search index and serves queries.

    Designed to run via ``with_runner()`` in any runner. Dense (faiss)
    searchers answer concurrent queries through :class:`_QueryBatcher`;
    ``max_batch=1`` restores one serialized forward per query.
    """

    def __init__(
        self,
        searcher_type: str,
        *,
        max_batch: int = 32,
        batch_window_s: float = 0.02,
        **searcher_args: Any,
    ) -> None:
        # Import torch/safetensors before the searcher module to avoid a
        # native-library initialisation conflict with FAISS that causes
        # segfaults on Apple Silicon (faiss_searcher.py imports faiss
        # before torch at module level).
        import safetensors  # noqa: F401
        import torch  # noqa: F401
        from searcher.searchers import SearcherType

        searcher_class = SearcherType.get_searcher_class(searcher_type)
        parser = argparse.ArgumentParser()
        searcher_class.parse_args(parser)

        cli: list[str] = []
        for key, value in searcher_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli.append(flag)
            else:
                cli.extend([flag, str(value)])

        args = parser.parse_args(cli)
        self._searcher = searcher_class(args)
        self._search_lock = threading.Lock()
        # FaissSearcher (tokenizer + tevatron DenseModel) is the only searcher
        # whose per-query work is an encoder forward worth batching.
        self._batcher = None
        if getattr(self._searcher, "tokenizer", None) is not None:
            self._batcher = _QueryBatcher(self._search_dense_batch, max(1, max_batch), batch_window_s)

    def search(self, query: str, k: int) -> list:
        if self._batcher is not None:
            return self._batcher.submit(query, k)
        with self._search_lock:
            return self._searcher.search(query, k)

    def _search_dense_batch(self, queries: list[str], k: int) -> list[list]:
        """Batched twin of ``FaissSearcher.search`` (same prefix, tokenisation, pooling, lookup)."""
        import torch

        s = self._searcher
        batch = s.tokenizer(
            [s.args.task_prefix + q for q in queries],
            padding=True,
            truncation=True,
            max_length=s.args.max_length,
            return_tensors="pt",
        )
        batch = {name: t.to(s.device) for name, t in batch.items()}
        with s.get_autocast_ctx(), torch.no_grad():
            q_reps = s.model.encode_query(batch).float().cpu().numpy()  # faiss wants float32
        scores, indices = s.retriever.search(q_reps, k)
        return [
            [
                {"docid": s.lookup[i], "score": float(sc), "text": s.docid_to_text.get(s.lookup[i], "Text not found")}
                for sc, i in zip(row_scores, row_indices)
            ]
            for row_scores, row_indices in zip(scores, indices)
        ]

    def get_document(self, docid: str) -> dict | None:
        return self._searcher.get_document(docid)


class RetrieverClient:
    """Lazy HTTP client to a remote Retriever service.

    Picklable — stores only the URL. Connects on first use.
    This allows it to survive serialization into Docker containers.
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._proxy: Any = None

    def _connect(self) -> None:
        if self._proxy is None:
            from ...adapters.runners.service import HTTPTransport
            from ...adapters.runners.transport import ObjectProxy

            self._proxy = ObjectProxy(HTTPTransport(self._url))

    def search(self, query: str, k: int) -> list:
        self._connect()
        return self._proxy.search(query, k)

    def get_document(self, docid: str) -> dict | None:
        self._connect()
        return self._proxy.get_document(docid)

    def close(self) -> None:
        if self._proxy is not None:
            try:
                self._proxy.close()
            except Exception:
                pass
            self._proxy = None

    def __getstate__(self) -> dict:
        return {"url": self._url}

    def __setstate__(self, state: dict) -> None:
        self._url = state["url"]
        self._proxy = None


# ── Shared retriever cache ────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {}


def get_shared_retriever(
    runner: str,
    runner_kwargs: dict[str, Any] | None = None,
    **retriever_kwargs: Any,
) -> Any:
    """Get or create a shared Retriever running in the specified runner."""
    from ...adapters.runners import with_runner

    key = json.dumps(retriever_kwargs, sort_keys=True, default=str)
    if key not in _cache:
        with _cache_lock:
            if key not in _cache:
                _cache[key] = with_runner(
                    Retriever,
                    runner=runner,
                    **(runner_kwargs or {}),
                    **retriever_kwargs,
                )
    return _cache[key]


def get_retriever_url(proxy: Any) -> str:
    """Extract the HTTP URL from a retriever proxy."""
    from ...adapters.runners.service import HTTPTransport

    transport = object.__getattribute__(proxy, "_transport")
    if isinstance(transport, HTTPTransport):
        return transport._base_url
    raise ValueError(
        "Cannot extract URL from non-HTTP retriever. Use runner='service' or runner='docker' for the retriever."
    )
