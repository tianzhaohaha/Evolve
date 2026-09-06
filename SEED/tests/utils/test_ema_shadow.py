import pytest
import torch
import torch.nn as nn

from verl.utils.ema import EmaShadow, ema_applies_to, normalize_ema_mode


def _module(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))


def _flat(module):
    return torch.cat([p.detach().flatten().clone() for p in module.parameters()])


def _shadow_flat(shadow):
    return torch.cat([s.flatten() for s in shadow.state_dict()["shadow"]])


def test_normalize_ema_mode_accepts_yaml_bool_and_strings():
    assert normalize_ema_mode(None) == "off"
    assert normalize_ema_mode(False) == "off"
    assert normalize_ema_mode("OFF") == "off"
    assert normalize_ema_mode("teacher") == "teacher"
    for bad in ("student", "none", "false", True):
        with pytest.raises(ValueError):
            normalize_ema_mode(bad)
    assert ema_applies_to("both", "ref") and ema_applies_to("teacher", "teacher")
    assert not ema_applies_to("ref", "teacher") and not ema_applies_to("off", "ref")


def test_update_follows_ema_formula_and_reports_delta():
    module = _module()
    shadow = EmaShadow(module, tau=0.75)
    before = _flat(module)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(1.0)
    delta = shadow.update()
    torch.testing.assert_close(_shadow_flat(shadow), 0.75 * before + 0.25 * _flat(module))
    assert delta == pytest.approx(((_flat(module) - before).norm() / before.norm()).item(), rel=1e-5)


def test_swapped_in_exposes_shadow_and_restores_live_weights():
    module = _module()
    shadow = EmaShadow(module, tau=0.5)
    initial = _flat(module)
    with torch.no_grad():
        for p in module.parameters():
            p.mul_(3.0)
    live = _flat(module)
    with shadow.swapped_in():
        assert shadow.swapped
        torch.testing.assert_close(_flat(module), initial)
    assert not shadow.swapped
    torch.testing.assert_close(_flat(module), live)


def test_swapped_in_restores_on_exception_and_rejects_nesting():
    module = _module()
    shadow = EmaShadow(module, tau=0.5)
    live = _flat(module)
    with pytest.raises(RuntimeError, match="boom"):
        with shadow.swapped_in():
            raise RuntimeError("boom")
    assert not shadow.swapped
    torch.testing.assert_close(_flat(module), live)
    with shadow.swapped_in():
        with pytest.raises(RuntimeError):
            shadow.update()
        with pytest.raises(RuntimeError):
            with shadow.swapped_in():
                pass


def test_failed_swap_leaves_dirty_flag_raised_and_weights_untouched():
    module = _module()
    shadow = EmaShadow(module, tau=0.5)
    live = _flat(module)
    shadow._shadow[1] = torch.zeros(1)  # corrupt one entry: validation fails before any write
    with pytest.raises(RuntimeError, match="shape"):
        with shadow.swapped_in():
            pass
    assert shadow.swapped
    torch.testing.assert_close(_flat(module), live)
    with pytest.raises(RuntimeError):
        shadow.update()


def test_failed_swap_back_leaves_dirty_flag_raised(monkeypatch):
    module = _module()
    shadow = EmaShadow(module, tau=0.5)
    calls = {"n": 0}
    original = shadow._swap

    def flaky_swap():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("cuda gone")
        original()

    monkeypatch.setattr(shadow, "_swap", flaky_swap)
    with pytest.raises(RuntimeError, match="cuda gone"):
        with shadow.swapped_in():
            pass
    assert shadow.swapped


def test_swap_requires_param_data_to_alias_its_local_shard():
    module = _module()
    param = next(module.parameters())
    param._local_shard = param.data  # FSDP1-style alias: accepted
    shadow = EmaShadow(module, tau=0.5)
    with shadow.swapped_in():
        pass
    param._local_shard = param.data.clone()  # unsharded state: refused before any write
    with pytest.raises(RuntimeError, match="resharded"):
        with shadow.swapped_in():
            pass


def test_state_dict_round_trip_and_reset():
    module = _module()
    shadow = EmaShadow(module, tau=0.9)
    with torch.no_grad():
        for p in module.parameters():
            p.add_(0.5)
    shadow.update()
    saved = shadow.state_dict()
    other = EmaShadow(_module(seed=1), tau=0.9)
    other.load_state_dict(saved)
    torch.testing.assert_close(_shadow_flat(other), torch.cat([s.flatten() for s in saved["shadow"]]))
    shadow.reset()
    torch.testing.assert_close(_shadow_flat(shadow), _flat(module))
    with pytest.raises(RuntimeError):
        EmaShadow(nn.Linear(2, 2), tau=0.9).load_state_dict(saved)


def test_tau_must_be_in_open_unit_interval():
    with pytest.raises(ValueError):
        EmaShadow(_module(), tau=1.0)
