"""Wyckoff event detection must be causal: batch results over full history must
match what a live walk (which only ever sees past bars) would have produced.

Guards the bug class found 2026-08-04: detect_spring_type_b read future closes
(`close.shift(-offset)`) with a "shift back" comment whose shift was never applied,
so wyckoff_spring_b repainted in batch runs while live (no future bars) could not
reproduce it.
"""
import numpy as np
import pandas as pd
import pytest

from engine.wyckoff.events import detect_all_wyckoff_events, detect_spring_type_b

EVENT_COLS = [
    'wyckoff_sc', 'wyckoff_bc', 'wyckoff_ar', 'wyckoff_as', 'wyckoff_st',
    'wyckoff_sos', 'wyckoff_sow', 'wyckoff_spring_a', 'wyckoff_spring_b',
    'wyckoff_ut', 'wyckoff_utad', 'wyckoff_lps', 'wyckoff_lpsy',
]


def _range_df(n=160, base=100.0, half_width=2.0, volume=1000.0):
    """Sideways range: lows ~base-2, highs ~base+2, flat volume (volume_z ~ 0)."""
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    rng = np.random.default_rng(42)
    close = base + rng.normal(0, 0.3, n)
    high = close + half_width * 0.5 + rng.uniform(0, 0.3, n)
    low = close - half_width * 0.5 - rng.uniform(0, 0.3, n)
    low = np.clip(low, base - half_width, None)
    high = np.clip(high, None, base + half_width)
    vol = volume + rng.uniform(-50, 50, n)
    return pd.DataFrame(
        {'open': close, 'high': high, 'low': low, 'close': close, 'volume': vol},
        index=idx,
    )


def _inject_shallow_spring(df, at, breakdown=0.008, recover_close=100.5):
    """Shallow breakdown below the rolling low at bar `at`, recovery on later bars.

    The candidate bar closes BELOW the range's lower quartile (no same-bar
    recovery) so only the following bars provide the recovery — exactly the
    shape that exposes a future-read in the recovery check.
    """
    rolling_low = df['low'].rolling(20).min().shift(1)
    ref_low = float(rolling_low.iloc[at])
    df.iloc[at, df.columns.get_loc('low')] = ref_low * (1 - breakdown)
    df.iloc[at, df.columns.get_loc('close')] = ref_low * 1.003  # below lower quartile
    df.iloc[at, df.columns.get_loc('open')] = ref_low * 1.01
    for k in range(1, 4):
        df.iloc[at + k, df.columns.get_loc('close')] = recover_close
        df.iloc[at + k, df.columns.get_loc('low')] = recover_close - 0.5
        df.iloc[at + k, df.columns.get_loc('high')] = recover_close + 0.5
    return df


def _fixture():
    df = _range_df()
    df = _inject_shallow_spring(df, at=60)
    df = _inject_shallow_spring(df, at=110)
    return df


@pytest.mark.parametrize('sm_enabled', [False, True],
                         ids=['raw-detectors', 'with-state-machine'])
def test_batch_matches_truncated_walk(sm_enabled):
    """For sampled bars t, detect_all on df[:t+1] must agree at bar t with
    detect_all on the full df — i.e., no detector may use future bars."""
    df = _fixture()
    cfg = {'state_machine_enabled': sm_enabled}

    full = detect_all_wyckoff_events(df.copy(), cfg=dict(cfg))

    # Sample every bar around the injected episodes plus a tail scatter
    sample_ts = list(range(58, 70)) + list(range(108, 120)) + [140, 150, 159]
    for t in sample_ts:
        trunc = detect_all_wyckoff_events(df.iloc[:t + 1].copy(), cfg=dict(cfg))
        for col in EVENT_COLS + ['wyckoff_phase_abc']:
            if col not in full.columns:
                continue
            full_val = full[col].iloc[t]
            trunc_val = trunc[col].iloc[-1]
            assert (full_val == trunc_val) or (
                pd.isna(full_val) and pd.isna(trunc_val)
            ), (
                f"{col} not causal at bar {t} (sm={sm_enabled}): "
                f"batch={full_val!r} vs truncated-walk={trunc_val!r}"
            )


def test_spring_b_fires_on_confirmation_bar_not_breakdown_bar():
    """detect_spring_type_b must fire on the confirmation bar (breakdown +
    recovery_bars), never on the breakdown bar itself (that would require
    knowing future closes)."""
    df = _fixture()
    detected, confidence = detect_spring_type_b(df.copy(), cfg={})

    fired = list(np.flatnonzero(detected.values))
    assert fired, "fixture should produce at least one spring_b detection"

    breakdown_bars = {60, 110}
    for b in breakdown_bars:
        assert not detected.iloc[b], (
            f"spring_b fired on the breakdown bar {b} — only possible by "
            f"reading future closes (look-ahead)"
        )

    recovery_bars = 3  # default
    expected = {b + recovery_bars for b in breakdown_bars}
    assert expected.issubset(set(fired)), (
        f"expected confirmation-bar firings at {sorted(expected)}, got {fired}"
    )
    # Confidence must sit on the firing bars only
    assert (confidence[detected].fillna(0) > 0).all()
    assert float(confidence[~detected].abs().max()) == 0.0


def test_spring_b_truncation_direct():
    """Function-level causality: truncating the frame at the breakdown bar must
    not change whether that bar is (not) a detection."""
    df = _fixture()
    full_det, _ = detect_spring_type_b(df.copy(), cfg={})
    for t in [60, 61, 62, 63, 110, 111, 112, 113]:
        trunc_det, _ = detect_spring_type_b(df.iloc[:t + 1].copy(), cfg={})
        assert bool(full_det.iloc[t]) == bool(trunc_det.iloc[-1]), (
            f"spring_b repaints at bar {t}: batch={bool(full_det.iloc[t])} "
            f"vs walk={bool(trunc_det.iloc[-1])}"
        )
