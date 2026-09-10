"""Loss-triggered stand-down — causality and rule tests (no engine, no network)."""
import pandas as pd

from engine.risk.stand_down import LossStandDown


def T(s):
    return pd.Timestamp(s, tz="UTC")


def test_two_losses_in_window_trigger_pause():
    sd = LossStandDown(k=2, window_hours=48, pause_hours=24)
    sd.record_leg("a", -100); sd.finalize("a", T("2026-01-01 00:00"))
    assert not sd.blocked(T("2026-01-01 01:00"))          # one loss: still active
    sd.record_leg("b", -50); sd.finalize("b", T("2026-01-02 12:00"))
    assert sd.blocked(T("2026-01-02 13:00"))              # paused
    assert sd.blocked(T("2026-01-03 11:59"))              # 24h from 2nd loss
    assert not sd.blocked(T("2026-01-03 12:01"))          # pause expired


def test_losses_outside_window_do_not_trigger():
    sd = LossStandDown(k=2, window_hours=48, pause_hours=24)
    sd.record_leg("a", -100); sd.finalize("a", T("2026-01-01 00:00"))
    sd.record_leg("b", -100); sd.finalize("b", T("2026-01-04 00:00"))  # 72h later
    assert not sd.blocked(T("2026-01-04 01:00"))


def test_net_position_pnl_not_legs():
    """A stopped runner whose scale-outs banked more than the stop lost is NOT a loss."""
    sd = LossStandDown(k=2, window_hours=48, pause_hours=24)
    sd.record_leg("a", +300); sd.record_leg("a", -200)     # net +100
    sd.finalize("a", T("2026-01-01 00:00"))
    sd.record_leg("b", -100); sd.finalize("b", T("2026-01-01 06:00"))
    assert not sd.blocked(T("2026-01-01 07:00"))           # only ONE net loss
    assert sd.losses_registered == 1


def test_winner_does_not_reset_pause():
    """green-never-red is NOT the rule: a win during the pause doesn't lift it."""
    sd = LossStandDown(k=2, window_hours=48, pause_hours=24)
    for pid, ts in (("a", "2026-01-01 00:00"), ("b", "2026-01-01 06:00")):
        sd.record_leg(pid, -100); sd.finalize(pid, T(ts))
    assert sd.blocked(T("2026-01-01 07:00"))
    sd.record_leg("c", +500); sd.finalize("c", T("2026-01-01 08:00"))
    assert sd.blocked(T("2026-01-01 09:00"))


def test_third_loss_extends_pause():
    sd = LossStandDown(k=2, window_hours=48, pause_hours=24)
    for pid, ts in (("a", "2026-01-01 00:00"), ("b", "2026-01-01 06:00"),
                    ("c", "2026-01-01 20:00")):
        sd.record_leg(pid, -100); sd.finalize(pid, T(ts))
    assert sd.blocked(T("2026-01-02 19:00"))               # 24h after 3rd loss
    assert not sd.blocked(T("2026-01-02 21:00"))
