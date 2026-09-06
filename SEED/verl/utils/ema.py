"""EMA shadow of a module's parameters, sharding-aware and GPU-memory free.

Each rank keeps an fp32 copy of *its own* parameter shards (FSDP1 keeps the
resharded local shard in ``FlatParameter._local_shard``; plain modules use
``.data``), so the shadow follows FSDP sharding for free and needs no extra
GPU memory: swaps stage through host memory one parameter at a time.

The shadow is exposed to the model through parameter swapping: inside
``swapped_in()`` the live shards hold the EMA values and the shadow holds the
live values; leaving the context (normally or via exception) swaps back.
``swapped`` is a dirty flag: it is raised before the first write and lowered
only after a complete swap-back, so any failure leaves it raised and callers
that must never see swapped weights (see ``_assert_not_swapped``) fail loudly.
"""

from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

EMA_MODES = ("off", "ref", "teacher", "both")


def normalize_ema_mode(value) -> str:
    """Map the config value onto EMA_MODES (YAML parses a bare ``off`` as False)."""
    if value is None or value is False:
        return "off"
    mode = str(value).strip().lower()
    if mode not in EMA_MODES:
        raise ValueError(f"ema_mode must be one of {EMA_MODES}, got {value!r}")
    return mode


def ema_applies_to(mode: str, target: str) -> bool:
    """Whether ``mode`` enables the EMA weights for ``target`` ('ref' or 'teacher')."""
    return mode == "both" or mode == target


def _shard(param: torch.Tensor) -> torch.Tensor:
    local = getattr(param, "_local_shard", None)
    return param.data if local is None else local


class EmaShadow:
    def __init__(self, module: nn.Module, tau: float, device: str = "cpu"):
        if not 0.0 < float(tau) < 1.0:
            raise ValueError(f"EMA tau must lie in (0, 1), got {tau}")
        self.tau = float(tau)
        self._params: List[torch.Tensor] = list(module.parameters())
        self._device = device
        self._shadow: List[torch.Tensor] = [self._fresh_copy(p) for p in self._params]
        self.swapped = False

    def _fresh_copy(self, param: torch.Tensor) -> torch.Tensor:
        copy = _shard(param).detach().to(device=self._device, dtype=torch.float32, copy=True)
        if self._device == "cpu" and torch.cuda.is_available():
            copy = copy.pin_memory()
        return copy

    def _pairs(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """(shadow, live shard) pairs, fully validated before any caller mutates anything.

        Besides the shape check this requires the module to be resharded (``param.data``
        aliases the local shard): FSDP1 leaves the root unit unsharded after a forward,
        and a forward started in that state would skip the all-gather and silently
        ignore whatever was written into the shard.
        """
        pairs = []
        for shadow, param in zip(self._shadow, self._params):
            live = _shard(param)
            if live.shape != shadow.shape:
                raise RuntimeError(f"EMA shadow shape {tuple(shadow.shape)} does not match live shard {tuple(live.shape)}")
            if live.data_ptr() != param.data.data_ptr():
                raise RuntimeError("EMA shadow requires a resharded module (param.data must alias its local shard)")
            pairs.append((shadow, live))
        return pairs

    @torch.no_grad()
    def reset(self) -> None:
        """Re-initialize the shadow from the live weights."""
        self._assert_not_swapped("reset")
        for shadow, live in self._pairs():
            shadow.copy_(live)

    @torch.no_grad()
    def update(self) -> float:
        """shadow <- tau * shadow + (1 - tau) * live. Returns the relative distance
        ||live - shadow|| / ||shadow|| measured before the update, reduced over all
        ranks when torch.distributed is initialized."""
        self._assert_not_swapped("update")
        num = torch.zeros((), dtype=torch.float64)
        den = torch.zeros((), dtype=torch.float64)
        for shadow, live in self._pairs():
            live_f32 = live.detach().to(device=shadow.device, dtype=shadow.dtype)
            num += (live_f32 - shadow).pow(2).sum(dtype=torch.float64)
            den += shadow.pow(2).sum(dtype=torch.float64)
            shadow.mul_(self.tau).add_(live_f32, alpha=1.0 - self.tau)
        if dist.is_available() and dist.is_initialized():
            stats = torch.stack([num, den]).to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
            dist.all_reduce(stats)
            num, den = stats[0].cpu(), stats[1].cpu()
        return float((num / den.clamp_min(1e-12)).sqrt())

    @torch.no_grad()
    def _swap(self) -> None:
        for shadow, live in self._pairs():
            staged = live.detach().to("cpu", copy=True)  # host staging: no GPU allocation
            live.copy_(shadow)
            shadow.copy_(staged)

    @contextmanager
    def swapped_in(self):
        """Temporarily load the EMA weights into the module."""
        self._assert_not_swapped("swapped_in")
        self.swapped = True  # dirty until a complete swap-back succeeds
        self._swap()
        try:
            yield self
        finally:
            self._swap()
            self.swapped = False

    def _assert_not_swapped(self, where: str) -> None:
        if self.swapped:
            raise RuntimeError(f"EMA shadow weights may still be loaded in the module during {where}")

    def state_dict(self) -> Dict[str, object]:
        return {"tau": self.tau, "shadow": [s.cpu() for s in self._shadow]}

    @torch.no_grad()
    def load_state_dict(self, state: Dict[str, object]) -> None:
        self._assert_not_swapped("load_state_dict")
        saved = list(state["shadow"])
        if len(saved) != len(self._shadow):
            raise RuntimeError(f"EMA checkpoint holds {len(saved)} tensors, module has {len(self._shadow)}")
        for shadow, tensor in zip(self._shadow, saved):
            if tensor.shape != shadow.shape:
                raise RuntimeError(f"EMA checkpoint shape {tuple(tensor.shape)} does not match {tuple(shadow.shape)}")
            shadow.copy_(tensor)
