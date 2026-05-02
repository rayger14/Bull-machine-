"""Tests for scripts/rebuild_4h_wyckoff_features.py.

The most safety-critical property is no look-ahead bias when forward-filling
4H Wyckoff scores onto the 1H index. These tests build a synthetic feature
store and exercise the rebuild pipeline end-to-end.
"""
from __future__ import annotations

import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rebuild_4h_wyckoff_features import (  # noqa: E402
    UPDATE_COLS,
    compute_4h_event_confidences,
    look_ahead_check,
    map_4h_scores_to_1h,
    resample_1h_to_4h,
    rolling_directional_scores,
)


def _make_synthetic_1h(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Random-walk 1H OHLCV with two synthetic distribution tops."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2021-01-01', periods=n, freq='1h', tz='UTC')
    ret = rng.normal(0, 0.005, size=n)
    for t0 in (n // 3, 2 * n // 3):
        ret[t0:t0 + 30] += 0.02
        ret[t0 + 30:t0 + 50] -= 0.03
    price = 30000 * np.exp(np.cumsum(ret))
    high = price * (1 + np.abs(rng.normal(0, 0.003, size=n)))
    low = price * (1 - np.abs(rng.normal(0, 0.003, size=n)))
    open_ = np.concatenate([[price[0]], price[:-1]])
    volume = np.abs(rng.normal(1000, 400, size=n))
    for t0 in (n // 3, 2 * n // 3):
        volume[t0 + 25:t0 + 55] *= 4
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': price, 'volume': volume},
        index=idx,
    )


def test_resample_4h_uses_left_close_left_label():
    """Confirm the resample convention matches live (closed='left', label='left')."""
    idx = pd.date_range('2021-01-01 00:00', periods=12, freq='1h', tz='UTC')
    df = pd.DataFrame({
        'open': range(12), 'high': range(12), 'low': range(12),
        'close': range(12), 'volume': [1] * 12,
    }, index=idx)
    df_4h = resample_1h_to_4h(df)
    # 12 hourly bars → 3 4H bars labeled 00:00, 04:00, 08:00.
    assert list(df_4h.index) == [
        pd.Timestamp('2021-01-01 00:00', tz='UTC'),
        pd.Timestamp('2021-01-01 04:00', tz='UTC'),
        pd.Timestamp('2021-01-01 08:00', tz='UTC'),
    ]
    # First 4H bar covers hours 0..3 → close == 3.
    assert df_4h.iloc[0]['close'] == 3
    assert df_4h.iloc[0]['volume'] == 4


def test_no_look_ahead_close_time_mapping():
    """A 1H bar at time t must NOT see scores from a 4H bar that closes after t.

    Construct a deterministic 4H score series and confirm that the 1H value
    at each hour t equals the most recent 4H bar with close <= t.
    """
    idx_1h = pd.date_range('2021-01-01 00:00', periods=24, freq='1h', tz='UTC')
    idx_4h = pd.date_range('2021-01-01 00:00', periods=6, freq='4h', tz='UTC')
    scores_4h = pd.DataFrame({
        'tf4h_wyckoff_bullish_score': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        'tf4h_wyckoff_bearish_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'tf4h_wyckoff_phase_score':   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    }, index=idx_4h)

    aligned = map_4h_scores_to_1h(scores_4h, idx_1h)

    # 4H bar at 00:00 covers hours 0..3 and closes at 04:00.
    # Its value (0.1) should only be visible from hour 04:00 onward.
    # Hours 0..3 have no preceding 4H close, so they should be 0 (filled).
    for h in range(4):
        assert aligned.iloc[h]['tf4h_wyckoff_bullish_score'] == 0.0, \
            f"hour {h} sees future 4H bar"

    # Hour 04:00 should show the first 4H bar's score (0.1).
    for h in range(4, 8):
        assert aligned.iloc[h]['tf4h_wyckoff_bullish_score'] == 0.1

    # Hour 08:00 should show the second 4H bar's score (0.2), and so on.
    for h in range(8, 12):
        assert aligned.iloc[h]['tf4h_wyckoff_bullish_score'] == 0.2
    for h in range(12, 16):
        assert aligned.iloc[h]['tf4h_wyckoff_bullish_score'] == 0.3


def test_rolling_lookback_matches_truncated_pipeline():
    """Truncated end-to-end pipeline must agree with the full pipeline at t."""
    df_1h = _make_synthetic_1h(n=4000, seed=3)
    df_4h = resample_1h_to_4h(df_1h)
    df_4h_ev = compute_4h_event_confidences(df_4h)
    scores_4h = rolling_directional_scores(df_4h_ev)
    scores_1h_full = map_4h_scores_to_1h(scores_4h, df_1h.index)

    # Pick 3 sample hours after burn-in.
    burn_in = 250 * 4
    candidates = df_1h.index[burn_in:]
    sample = sorted(candidates[:: max(1, len(candidates) // 3)][:3])

    ok, results = look_ahead_check(
        df_1h[['open', 'high', 'low', 'close', 'volume']],
        scores_1h_full,
        list(sample),
    )
    # Each per-sample result reports `ok`; aggregate must be True too.
    for r in results:
        assert r['ok'], f"Look-ahead diff at {r['t']}: {r['diffs']}"
    assert ok


def test_cli_end_to_end_preserves_other_columns():
    """Run the rebuild script CLI on a synthetic parquet and verify that
    every non-update column is bit-for-bit identical after the rewrite."""
    df = _make_synthetic_1h(n=3000, seed=5)
    rng = np.random.default_rng(5)
    for i in range(40):
        df[f'dummy_{i}'] = rng.normal(0, 1, size=len(df))
    df['tf4h_wyckoff_bullish_score'] = 0.5
    df['tf4h_wyckoff_bearish_score'] = 0.5
    df['tf4h_wyckoff_phase_score'] = 0.5

    with tempfile.TemporaryDirectory() as tmp:
        pq = Path(tmp) / 'feat.parquet'
        df.to_parquet(pq, engine='pyarrow', compression='snappy')
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/rebuild_4h_wyckoff_features.py'),
             '--parquet', str(pq), '--look-ahead-samples', '3'],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr[-800:]

        post = pd.read_parquet(pq)
        assert post.shape == df.shape

        # OHLCV + 40 dummies preserved bit-for-bit
        for c in ['open', 'high', 'low', 'close', 'volume'] + [f'dummy_{i}' for i in range(40)]:
            assert np.array_equal(df[c].values, post[c].values), f"{c} changed"

        # Update columns now have variance
        for c in UPDATE_COLS:
            assert post[c].std() > 0.05, f"{c} std too low: {post[c].std()}"
            assert post[c].nunique() > 2, f"{c} only {post[c].nunique()} unique values"


def test_dry_run_does_not_write():
    df = _make_synthetic_1h(n=2500, seed=8)
    df['tf4h_wyckoff_bullish_score'] = 0.5
    df['tf4h_wyckoff_bearish_score'] = 0.5
    df['tf4h_wyckoff_phase_score'] = 0.5
    with tempfile.TemporaryDirectory() as tmp:
        pq = Path(tmp) / 'feat.parquet'
        df.to_parquet(pq, engine='pyarrow', compression='snappy')
        mtime_before = pq.stat().st_mtime_ns
        proc = subprocess.run(
            [sys.executable, str(ROOT / 'scripts/rebuild_4h_wyckoff_features.py'),
             '--parquet', str(pq), '--dry-run', '--look-ahead-samples', '2'],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0
        assert pq.stat().st_mtime_ns == mtime_before, "Dry-run modified the file"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
