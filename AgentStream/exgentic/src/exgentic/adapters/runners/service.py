# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

"""HTTPTransport + serve() — run any object as an HTTP service."""

from __future__ import annotations

import base64
import os
import threading
import time
from typing import Any, Optional

import cloudpickle as cp
import httpx
from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel

from .transport import ObjectHost, ObjectProxy, Transport, deserialize_error, serialize_error

# ── HTTP models ──────────────────────────────────────────────────────


class CallRequest(PydanticBaseModel):
    method: str
    args: str  # base64(cloudpickle)
    kwargs: str  # base64(cloudpickle)


class GetRequest(PydanticBaseModel):
    name: str


class SetRequest(PydanticBaseModel):
    name: str
    value: str  # base64(cloudpickle)


class RPCResponse(PydanticBaseModel):
    status: str  # "ok" | "error"
    result: Optional[str] = None  # base64(cloudpickle)
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    error_tb: Optional[str] = None
    error_pickled: Optional[str] = None  # base64(cloudpickle'd exception)


# ── helpers ──────────────────────────────────────────────────────────


def _encode(obj: Any) -> str:
    return base64.b64encode(cp.dumps(obj)).decode("ascii")


def _decode(data: str) -> Any:
    return cp.loads(base64.b64decode(data))


def _error_response(exc: Exception) -> RPCResponse:
    data = serialize_error(exc)
    pickled_b64 = None
    if data["pickled"] is not None:
        pickled_b64 = base64.b64encode(data["pickled"]).decode("ascii")
    return RPCResponse(
        status="error",
        error_type=data["type"],
        error_msg=data["msg"],
        error_tb=data["tb"],
        error_pickled=pickled_b64,
    )


# ── FastAPI app ──────────────────────────────────────────────────────


def create_app(host: ObjectHost) -> FastAPI:
    app = FastAPI()

    # Optional identity token (set by subprocess runners): lets the spawning
    # client verify that the /health answer comes from ITS server, not from a
    # different process that won a find_free_port() race on the same port.
    identity_token = os.environ.get("EXGENTIC_SERVE_TOKEN") or None

    @app.get("/health")
    def health():
        payload = {"status": "ok"}
        if identity_token:
            payload["token"] = identity_token
        return payload

    @app.post("/call")
    def handle_call(req: CallRequest) -> RPCResponse:
        try:
            result = host.handle("call", req.method, *_decode(req.args), **_decode(req.kwargs))
            return RPCResponse(status="ok", result=_encode(result))
        except Exception as exc:
            return _error_response(exc)

    @app.post("/get")
    def handle_get(req: GetRequest) -> RPCResponse:
        try:
            return RPCResponse(status="ok", result=_encode(host.handle("get", req.name)))
        except Exception as exc:
            return _error_response(exc)

    @app.post("/set")
    def handle_set(req: SetRequest) -> RPCResponse:
        try:
            host.handle("set", req.name, _decode(req.value))
            return RPCResponse(status="ok")
        except Exception as exc:
            return _error_response(exc)

    return app


# ── serve() ──────────────────────────────────────────────────────────


def serve(obj: Any, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Serve an object over HTTP (blocking)."""
    import uvicorn

    uvicorn.run(create_app(ObjectHost(obj)), host=host, port=port, log_level="warning")


# ── HTTPTransport — client side ──────────────────────────────────────


class HTTPTransport(Transport):
    """Talks to an HTTP server hosting an ObjectHost."""

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def _rpc(self, endpoint: str, payload: dict, timeout: Any = None) -> Any:
        resp = self._client.post(
            f"{self._base_url}{endpoint}",
            json=payload,
            timeout=timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT,
        )
        resp.raise_for_status()
        data = RPCResponse(**resp.json())
        if data.status == "error":
            pickled = base64.b64decode(data.error_pickled) if data.error_pickled else None
            raise deserialize_error(
                {
                    "type": data.error_type or "RuntimeError",
                    "msg": data.error_msg or "",
                    "tb": data.error_tb or "",
                    "pickled": pickled,
                }
            )
        return _decode(data.result) if data.result is not None else None

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self._rpc(
            "/call",
            {
                "method": method,
                "args": _encode(args),
                "kwargs": _encode(kwargs),
            },
        )

    def call_bounded(self, method: str, timeout: float, *args: Any, **kwargs: Any) -> Any:
        """Like call(), but with a per-request timeout overriding the client default.

        Used for teardown paths (e.g. close) that must not inherit the long
        transport timeout when the server is wedged.
        """
        return self._rpc(
            "/call",
            {
                "method": method,
                "args": _encode(args),
                "kwargs": _encode(kwargs),
            },
            timeout=timeout,
        )

    def get(self, name: str) -> Any:
        return self._rpc("/get", {"name": name})

    def set(self, name: str, value: Any) -> None:
        self._rpc("/set", {"name": name, "value": _encode(value)})

    def close(self) -> None:
        self._client.close()

    def __repr__(self) -> str:
        return f"HTTPTransport({self._base_url!r})"


# ── Utilities ────────────────────────────────────────────────────────


class ServiceIdentityError(RuntimeError):
    """The service answering on a port is not the one we spawned (port collision)."""


def _wait_for_health(url: str, timeout: float = 15.0, token: str | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                if token is None:
                    return
                try:
                    served_token = resp.json().get("token")
                except Exception:
                    served_token = None
                # Servers predating the token protocol return no token; accept
                # them. A *different* token means another process owns the port
                # (find_free_port raced) — waiting longer can never fix that.
                if served_token is None or served_token == token:
                    return
                raise ServiceIdentityError(
                    f"Service at {url} answered with a foreign identity token; "
                    "the port is owned by another exgentic serve process"
                )
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Service at {url} did not become healthy within {timeout}s")


# ── ServiceRunner ────────────────────────────────────────────────────


class ServiceRunner:
    """Starts an HTTP service in a background thread and returns an ObjectProxy."""

    def __init__(
        self,
        target_cls: type,
        *args: Any,
        port: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._target_cls = target_cls
        self._args = args
        self._kwargs = kwargs
        from ._utils import find_free_port

        self._port = port or find_free_port()
        self._server = None

    def start(self) -> ObjectProxy:
        import uvicorn

        from ...core.context import set_context_fallback, try_get_context

        # Set process-wide fallback so context is available in uvicorn's
        # request handler threads (which don't inherit ContextVar).
        set_context_fallback(try_get_context())

        obj = self._target_cls(*self._args, **self._kwargs)
        app = create_app(ObjectHost(obj))

        config = uvicorn.Config(app, host="127.0.0.1", port=self._port, log_level="warning")
        self._server = uvicorn.Server(config)
        threading.Thread(target=self._server.run, daemon=True).start()

        url = f"http://127.0.0.1:{self._port}"
        _wait_for_health(url)

        transport = HTTPTransport(url)
        proxy = ObjectProxy(transport)

        server_ref = self._server

        def _close() -> None:
            try:
                transport.call("close")
            except AttributeError:
                pass
            transport.close()
            server_ref.should_exit = True
            set_context_fallback(None)

        object.__setattr__(proxy, "close", _close)
        return proxy
