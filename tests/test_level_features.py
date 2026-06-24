"""
Tests for engine/features/level_features.py

Covers the non-negotiables from the wick_trap rebuild spec:
(a) no lookahead — truncating the series never changes already-emitted values
(b) batch vs incremental (LevelFeatureTracker) parity — one code path
(c) hand-built sweep + acceptance scenario (LuxAlgo criterion)
(d) day_type trend_down behavioral classifier (incl. UTC-day-boundary reset)
"""

import numpy as np
import pandas as pd
import pytest

from engine.features.level_features import (
    LEVEL_COLUMNS,
    LevelFeatureTracker,
    compute_level_features,
)


def _make_ohlcv(n: int, seed: int = 7, start: str = "2024-01-01") -> pd.DataFrame:
    """Seeded synthetic 1H random-walk OHLCV."""
    rng = np.random.default_rng(seed)
    close = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.0, 0.002, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(10.0, 100.0, n)
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _assert_rows_equal(row_a, row_b, label) -> None:
    """NaN-aware exact-ish comparison of two feature rows (dict or Series)."""
    for col in LEVEL_COLUMNS:
        va, vb = row_a[col], row_b[col]
        if pd.isna(va) and pd.isna(vb):
            continue
        assert va == pytest.approx(vb, rel=1e-12, abs=1e-12), (
            f"{col} mismatch at {label}: {va!r} != {vb!r}"
        )


def test_no_lookahead():
    """Features at bar t are identical whether or not bars > t exist."""
    df = _make_ohlcv(600, seed=7)
    full = compute_level_features(df)
    rng = np.random.default_rng(42)
    for t in rng.choice(np.arange(40, 600), size=20, replace=False):
        part = compute_level_features(df.iloc[: t + 1])
        _assert_rows_equal(part.iloc[-1], full.iloc[t], label=f"t={t}")


def test_batch_incremental_parity():
    """LevelFeatureTracker.update returns the batch values bar-for-bar."""
    df = _make_ohlcv(300, seed=11)
    full = compute_level_features(df)
    tracker = LevelFeatureTracker()
    for i, (ts, row) in enumerate(df.iterrows()):
        out = tracker.update(
            {
                "timestamp": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
        _assert_rows_equal(out, full.iloc[i], label=f"bar={i}")


def test_trimmed_buffer_parity_with_prime():
    """Parity must hold even after the live buffer trims to max_buffer_bars
    (1300 bars > 1000 buffer): prime() on history, update() on the tail."""
    df = _make_ohlcv(1300, seed=23)
    full = compute_level_features(df)
    tracker = LevelFeatureTracker(max_buffer_bars=1000)
    tracker.prime(df.iloc[:1200])
    for i in range(1200, 1300):
        ts = df.index[i]
        row = df.iloc[i]
        out = tracker.update(
            {
                "timestamp": ts,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
        _assert_rows_equal(out, full.iloc[i], label=f"bar={i}")


def test_sweep_low_and_acceptance():
    """Wick below a confirmed swing low + close back above => sweep event,
    then acceptance_bars_above counts consecutive holding closes."""
    n = 32
    idx = pd.date_range("2024-01-02", periods=n, freq="1h")  # single UTC day start
    open_ = np.full(n, 100.0)
    close = np.full(n, 100.0)
    low = np.full(n, 96.0)
    high = np.full(n, 101.0)

    low[10] = 95.0  # swing pivot low (min of bars 0..20), confirmed at bar 20
    # Sweep bar: wick below the 95.0 level, close back above it.
    low[21] = 94.5
    open_[21] = 99.0
    close[21] = 96.0
    high[21] = 100.0
    close[22:25] = 97.0  # holds the level => acceptance increments

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.ones(n)},
        index=idx,
    )
    feats = compute_level_features(df)

    # Pivot confirmation lag: invisible at bar 19, visible from bar 20.
    assert np.isnan(feats["swing_low_50"].iloc[19])
    assert feats["swing_low_50"].iloc[20] == 95.0

    # Sweep fires exactly at the wick-and-reclaim bar.
    assert feats["sweep_low_event"].iloc[21] == 1
    assert feats["sweep_low_event"].iloc[:21].sum() == 0

    # Acceptance: 1 on the sweep bar (close held), then 2, 3, ...
    assert feats["acceptance_bars_above"].iloc[20] == 0
    assert feats["acceptance_bars_above"].iloc[21] == 1
    assert feats["acceptance_bars_above"].iloc[22] == 2
    assert feats["acceptance_bars_above"].iloc[23] == 3

    # Interface contract: exact column set.
    assert list(feats.columns) == LEVEL_COLUMNS


def test_day_type_trend_down():
    """trend_down only after >=4 consecutive closes below prior_day_low within
    the current UTC day; run resets at the day boundary."""
    n = 72  # three UTC days, starting Monday 2024-01-01 00:00
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.full(n, 100.0)
    close[28:48] = 97.0  # day 2: below day-1 low (99.5) from bar 28 onward
    close[48:] = 96.0    # day 3: below day-2 low (96.5) from its first bar
    open_ = np.r_[close[0], close[:-1]]
    low = close - 0.5
    high = close + 0.5

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.ones(n)},
        index=idx,
    )
    feats = compute_level_features(df)
    dt = feats["day_type"].to_numpy()

    # Day 1: no prior day => range.
    assert (dt[:24] == 0).all()
    # Day 2: closes below PDL start at bar 28; trend_down from the 4th (bar 31).
    assert (dt[24:31] == 0).all()
    assert (dt[31:48] == -1).all()
    # Day 3: still below the (new) prior-day low, but the run resets at the
    # UTC-day boundary — needs 4 fresh closes (bars 48-50 range, 51 trend_down).
    assert (dt[48:51] == 0).all()
    assert (dt[51:] == -1).all()
    # No spurious trend_up anywhere.
    assert (dt != 1).all()


def test_rejects_tz_aware_index():
    df = _make_ohlcv(50)
    df.index = df.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="tz-naive"):
        compute_level_features(df)
