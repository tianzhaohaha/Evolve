"""The host-side RPC decoder must survive objects whose class only exists in the service venv."""

import base64
import sys
import types
from pathlib import Path

import cloudpickle
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from exgentic.adapters.runners import service  # noqa: E402


def _payload_with_foreign_object():
    # A class from a module that exists only "over there": define it, pickle by reference, drop the module.
    mod = types.ModuleType("bfcl_eval_fake")
    class CheckerFailure:  # noqa: D401
        def __init__(self, reason):
            self.reason = reason
    CheckerFailure.__module__ = "bfcl_eval_fake"
    CheckerFailure.__qualname__ = "CheckerFailure"
    mod.CheckerFailure = CheckerFailure
    sys.modules["bfcl_eval_fake"] = mod
    payload = {"score": 0.0, "success": False, "session_metadata": {"checker_result": {"valid": False, "error": CheckerFailure("bad call")}}}
    encoded = base64.b64encode(cloudpickle.dumps(payload)).decode("ascii")
    del sys.modules["bfcl_eval_fake"]
    return encoded


def test_decode_substitutes_placeholders_for_missing_modules():
    encoded = _payload_with_foreign_object()
    with pytest.raises(ModuleNotFoundError):
        cloudpickle.loads(base64.b64decode(encoded))
    decoded = service._decode(encoded)
    assert decoded["score"] == 0.0 and decoded["success"] is False
    placeholder = decoded["session_metadata"]["checker_result"]["error"]
    assert isinstance(placeholder, service._Unavailable)
    assert placeholder.qualname == "bfcl_eval_fake.CheckerFailure" and placeholder.state == {"reason": "bad call"}
    assert "unavailable" in repr(placeholder) and str(placeholder)  # json.dump(default=str) can serialise it


def test_decode_is_unchanged_for_ordinary_payloads():
    payload = {"a": [1, 2.5, "x"], "b": {"nested": (None, True)}}
    assert service._decode(base64.b64encode(cloudpickle.dumps(payload)).decode("ascii")) == payload
