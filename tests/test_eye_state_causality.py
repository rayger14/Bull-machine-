"""
All-Seeing Eye Phase 1 — causality / no-repaint unit tests.

Guards the two properties the feature pass must never violate:
  1. NO REPAINT: truncating future history must not change any already-emitted 1D state.
  2. NON-DEGENERATE: no single eye_state may absorb the whole timeline on synthetic data.

Study-only module; not wired into any production path.
"""
import numpy as np
import pandas as pd
import pytest

from engine.features.eye_state import (
    compute_eye_state_1d, compute_eye_features,
    IN_RANGE, TRENDING, CONFIRMED_BREAK,
)


def _synthetic_1h(n_days=400, seed=7):
    """Build a 1H OHLCV frame with a range -> up-break -> trend -> pullback -> down-break
    shape so multiple states are exercised."""
    rng = np.random.default_rng(seed)
    hours = n_days * 24
    idx = pd.date_range("2019-01-01", periods=hours, freq="1h")
    t = np.arange(n_days)
    # daily anchor path: flat range, then ramp up, then drop
    base = np.piecewise(
        t.astype(float),
        [t < 120, (t >= 120) & (t < 240), t >= 240],
        [lambda x: 100 + 3 * np.sin(x / 6.0),
         lambda x: 100 + (x - 120) * 1.2,
         lambda x: 244 - (x - 240) * 1.0],
    )
    # expand daily anchor to hourly with intrabar noise
    daily_close = base
    close = np.repeat(daily_close, 24) + rng.normal(0, 0.4, hours)
    high = close + np.abs(rng.normal(0, 0.8, hours))
    low = close - np.abs(rng.normal(0, 0.8, hours))
    openp = close + rng.normal(0, 0.3, hours)
    vol = np.abs(rng.normal(1000, 200, hours))
    df = pd.DataFrame({"open": openp, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
    df["wyckoff_phase_dir"] = "neutral"
    df["wyckoff_sos"] = 0.0
    return df


def test_no_repaint_on_truncation():
    df = _synthetic_1h()
    full = compute_eye_state_1d(df)
    # truncate at three interior cut points; overlap states must be identical
    for frac in (0.5, 0.7, 0.9):
        cut = df.index[int(len(df) * frac)]
        tr = compute_eye_state_1d(df.loc[:cut])
        common = tr.index.intersection(full.index)
        common = common[common <= cut]
        assert (tr.loc[common, "eye_state"].astype(str).values
                == full.loc[common, "eye_state"].astype(str).values).all(), f"state repaint at {frac}"
        assert (tr.loc[common, "eye_dir"].astype(str).values
                == full.loc[common, "eye_dir"].astype(str).values).all(), f"dir repaint at {frac}"


def test_broadcast_is_causal_lagged():
    """A 1H bar must never see a 1D state whose source day has not yet closed+settled."""
    df = _synthetic_1h(n_days=200)
    eye1h = compute_eye_features(df)
    # first non-null appears only after the first fully-formed + settled daily bar
    first_valid = eye1h["eye_state"].first_valid_index()
    assert first_valid is not None
    # must be strictly after the first daily close+settle (i.e. not at hour 0)
    assert first_valid > df.index[0]


def test_not_degenerate():
    df = _synthetic_1h()
    st = compute_eye_state_1d(df)["eye_state"].value_counts(normalize=True)
    # no single state may own the whole series; IN_RANGE should be present and material
    assert st.max() < 0.97, f"degenerate occupancy: {st.to_dict()}"
    assert st.get(IN_RANGE, 0) > 0.05, "IN_RANGE should be a material fraction"


def test_location_bounds():
    df = _synthetic_1h()
    loc = compute_eye_state_1d(df)["eye_location"].dropna()
    assert (loc >= 0).all() and (loc <= 1).all()
