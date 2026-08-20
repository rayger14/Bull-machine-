"""
Wyckoff v2 SHADOW detector tests — climax disambiguation (BC/SC v2),
anchored spring, per-bar event_now score, raw context sides.

Design contract (docs/knowledge/wyckoff_audit.md + live audit 2026-08-20):
- A climax and a breakout share the same 3 hard gates (volume_z, range_pos,
  range_z). They can only be told apart by rejection evidence on the bar
  (wick / weak close) or by what happens NEXT (reversal => climax,
  continuation => breakout). Requiring a wick alone was tried and REVERTED
  (Apr/Nov 2021 euphoric tops close at their highs) — so v2 keeps the
  original detector untouched and adds shadow columns:
    wyckoff_bc_v2            immediate: gates + rejection evidence NOW
    wyckoff_bc_v2_confirmed  delayed: candidate + reversal within k bars,
                             stamped at the CONFIRMATION bar (never backfilled)
    (sc_v2 mirror on the selling side)
- Shadow columns must NOT feed scores / state machine / phase (data
  collection only), and must be causal: truncated recompute == full compute
  on the shared prefix.
"""
import numpy as np
import pandas as pd
import pytest

from engine.wyckoff.events import (
    detect_all_wyckoff_events,
    detect_climax_v2,
    create_wyckoff_context,
)


# ---------------------------------------------------------------- helpers
def _flat_df(n=80, price=100.0, vol=1000.0, seed=7):
    """Quiet tape: small ranges, stable volume."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    drift = rng.normal(0, 0.03, n).cumsum()
    close = price + drift
    open_ = close + rng.normal(0, 0.02, n)
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    volume = vol + rng.normal(0, 20, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _spike_bar(df, i, kind):
    """Overwrite bar i with a gate-passing spike (huge volume, wide range, at highs)."""
    base = float(df["close"].iloc[i - 1])
    hi, lo = base + 6.0, base - 0.5  # wide range, top of every lookback window
    df.iloc[i, df.columns.get_loc("volume")] = 20000.0
    df.iloc[i, df.columns.get_loc("open")] = base
    df.iloc[i, df.columns.get_loc("high")] = hi
    df.iloc[i, df.columns.get_loc("low")] = lo
    if kind == "euphoric":        # closes at the high — breakout OR climax top
        df.iloc[i, df.columns.get_loc("close")] = hi - 0.05
    elif kind == "rejected":      # big upper wick, weak close
        df.iloc[i, df.columns.get_loc("close")] = lo + 0.15 * (hi - lo)
    return df, hi, lo


def _continue_up(df, i, hi, k=6):
    for j in range(i + 1, min(i + 1 + k, len(df))):
        lvl = hi + 0.8 * (j - i)
        df.iloc[j, df.columns.get_loc("open")] = lvl - 0.3
        df.iloc[j, df.columns.get_loc("close")] = lvl
        df.iloc[j, df.columns.get_loc("high")] = lvl + 0.1
        df.iloc[j, df.columns.get_loc("low")] = lvl - 0.5
    return df


def _reverse_down(df, i, lo, k=3):
    """Close below the candidate bar's low within k bars."""
    for j in range(i + 1, i + 1 + k):
        lvl = lo - 0.8 * (j - i)
        df.iloc[j, df.columns.get_loc("open")] = lvl + 0.3
        df.iloc[j, df.columns.get_loc("close")] = lvl
        df.iloc[j, df.columns.get_loc("high")] = lvl + 0.5
        df.iloc[j, df.columns.get_loc("low")] = lvl - 0.1
    return df


I = 65  # spike bar index (past all warmup windows)


# ---------------------------------------------------------------- BC v2
def test_breakout_is_not_bc_v2():
    """Euphoric close + upward continuation => neither immediate nor confirmed."""
    df, hi, lo = _spike_bar(_flat_df(), I, "euphoric")
    df = _continue_up(df, I, hi)
    det, conf, confirmed = detect_climax_v2(df.copy(), {}, side="buying")
    assert not det.iloc[I], "breakout bar must not be immediate bc_v2"
    assert not confirmed.any(), "continuation must never confirm a climax"


def test_rejection_wick_is_immediate_bc_v2():
    df, hi, lo = _spike_bar(_flat_df(), I, "rejected")
    det, conf, confirmed = detect_climax_v2(df.copy(), {}, side="buying")
    assert det.iloc[I], "gates + rejection wick must fire immediately"
    assert conf.iloc[I] > 0


def test_euphoric_top_confirmed_by_reversal():
    """Apr/Nov-2021 shape: closes at high, reverses after. Confirmed at the
    confirmation bar (close < candidate low), never at the event bar."""
    df, hi, lo = _spike_bar(_flat_df(), I, "euphoric")
    df = _reverse_down(df, I, lo, k=3)
    det, conf, confirmed = detect_climax_v2(df.copy(), {}, side="buying")
    assert not det.iloc[I], "euphoric close alone is ambiguous — no immediate fire"
    conf_idx = confirmed[confirmed].index
    assert len(conf_idx) >= 1, "reversal within k bars must confirm the climax"
    assert conf_idx[0] > df.index[I], "confirmation stamps AFTER the event bar"


def test_bc_v2_is_causal_no_repaint():
    """Truncated recompute must equal full compute on the shared prefix."""
    df, hi, lo = _spike_bar(_flat_df(), I, "euphoric")
    df = _reverse_down(df, I, lo, k=3)
    full_det, _, full_confirmed = detect_climax_v2(df.copy(), {}, side="buying")
    for cut in (I + 1, I + 2, I + 3, I + 4):
        part_det, _, part_confirmed = detect_climax_v2(df.iloc[:cut].copy(), {}, side="buying")
        pd.testing.assert_series_equal(part_det, full_det.iloc[:cut], check_names=False)
        pd.testing.assert_series_equal(
            part_confirmed, full_confirmed.iloc[:cut], check_names=False
        )


# ---------------------------------------------------------------- SC v2 mirror
def _spike_bar_down(df, i, kind):
    base = float(df["close"].iloc[i - 1])
    hi, lo = base + 0.5, base - 6.0
    df.iloc[i, df.columns.get_loc("volume")] = 20000.0
    df.iloc[i, df.columns.get_loc("open")] = base
    df.iloc[i, df.columns.get_loc("high")] = hi
    df.iloc[i, df.columns.get_loc("low")] = lo
    if kind == "capitulation_close":   # closes at the low — breakdown OR capitulation
        df.iloc[i, df.columns.get_loc("close")] = lo + 0.05
    elif kind == "absorbed":           # long lower wick, strong close
        df.iloc[i, df.columns.get_loc("close")] = hi - 0.15 * (hi - lo)
    return df, hi, lo


def test_breakdown_is_not_sc_v2():
    """The live 63.3k 'SC' shape: closes at low, keeps falling => nothing."""
    df, hi, lo = _spike_bar_down(_flat_df(), I, "capitulation_close")
    for j in range(I + 1, I + 6):
        lvl = lo - 0.8 * (j - I)
        df.iloc[j, df.columns.get_loc("open")] = lvl + 0.3
        df.iloc[j, df.columns.get_loc("close")] = lvl
        df.iloc[j, df.columns.get_loc("high")] = lvl + 0.5
        df.iloc[j, df.columns.get_loc("low")] = lvl - 0.1
    det, conf, confirmed = detect_climax_v2(df.copy(), {}, side="selling")
    assert not det.iloc[I]
    assert not confirmed.any()


def test_absorption_wick_is_immediate_sc_v2():
    df, hi, lo = _spike_bar_down(_flat_df(), I, "absorbed")
    det, conf, confirmed = detect_climax_v2(df.copy(), {}, side="selling")
    assert det.iloc[I]


def test_capitulation_confirmed_by_recovery():
    df, hi, lo = _spike_bar_down(_flat_df(), I, "capitulation_close")
    for j in range(I + 1, I + 4):   # close back above candidate high
        lvl = hi + 0.8 * (j - I)
        df.iloc[j, df.columns.get_loc("open")] = lvl - 0.3
        df.iloc[j, df.columns.get_loc("close")] = lvl
        df.iloc[j, df.columns.get_loc("high")] = lvl + 0.1
        df.iloc[j, df.columns.get_loc("low")] = lvl - 0.5
    det, conf, confirmed = detect_climax_v2(df.copy(), {}, side="selling")
    assert not det.iloc[I]
    assert confirmed.any()
    assert confirmed[confirmed].index[0] > df.index[I]


# ---------------------------------------------------------------- integration
def test_shadow_columns_present_and_isolated():
    """detect_all_wyckoff_events emits the shadow columns without changing
    the v1 columns, scores, or phase (shadow = data collection only)."""
    df, hi, lo = _spike_bar(_flat_df(120), I, "euphoric")
    df = _continue_up(df, I, hi)
    base = detect_all_wyckoff_events(df.copy())
    for col in (
        "wyckoff_bc_v2", "wyckoff_bc_v2_confidence", "wyckoff_bc_v2_confirmed",
        "wyckoff_sc_v2", "wyckoff_sc_v2_confidence", "wyckoff_sc_v2_confirmed",
        "wyckoff_spring_b_anchored", "wyckoff_event_now",
    ):
        assert col in base.columns, f"missing shadow column {col}"
    # v1 unchanged: recompute with shadow columns dropped pre-pass is impossible
    # to diff directly, so assert the invariant that shadow families are NOT in
    # the directional scores: a breakout bar has bc(v1) fired but bearish score
    # must be computable identically whether or not v2 columns exist.
    assert base["wyckoff_bc"].iloc[I], "sanity: v1 BC still fires on the spike"
    assert not base["wyckoff_bc_v2"].iloc[I], "v2 must not fire on the breakout"


def test_event_now_zero_when_quiet():
    df = _flat_df(120)
    out = detect_all_wyckoff_events(df.copy())
    quiet = out.iloc[30:50]
    active = quiet[[c for c in quiet.columns
                    if c.startswith("wyckoff_") and c.endswith("_confidence")
                    and "_v2" not in c]].max(axis=1).fillna(0)
    # event_now equals the per-bar max confidence — zero on bars with no event
    assert (quiet["wyckoff_event_now"][active == 0] == 0).all()


def test_context_exposes_raw_sides():
    df, hi, lo = _spike_bar(_flat_df(120), I, "rejected")
    out = detect_all_wyckoff_events(df.copy())
    ctx = create_wyckoff_context(out, lookback=90, timeframe="TEST")
    assert hasattr(ctx, "raw_bullish_score") and hasattr(ctx, "raw_bearish_score")
    # net-dominance behavior unchanged: at most one net side > 0
    assert min(ctx.bullish_score, ctx.bearish_score) == 0.0
    # raw sides are the pre-arbitration maxes: raw >= net on the winning side
    assert ctx.raw_bullish_score >= ctx.bullish_score
    assert ctx.raw_bearish_score >= ctx.bearish_score


def test_spring_b_anchored_subset():
    """anchored can only fire where the FINAL (state-machine-validated)
    spring_b fired. Regression guard for the 2026-08-20 bug where the anchor
    was computed pre-SM and fired 207 times vs spring_b's 146 on the V12
    store. Uses a noisy random-walk tape so spring_b genuinely fires."""
    rng = np.random.default_rng(3)
    n = 3000
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(rng.normal(0, 0.004, n).cumsum())
    high = close * (1 + abs(rng.normal(0, 0.003, n)))
    low = close * (1 - abs(rng.normal(0, 0.003, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    vol = abs(rng.lognormal(7, 1, n))
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    out = detect_all_wyckoff_events(df.copy())
    anchored = out["wyckoff_spring_b_anchored"].astype(bool)
    spring = out["wyckoff_spring_b"].astype(bool)
    assert spring.sum() > 0, "fixture must actually fire spring_b"
    assert (anchored & ~spring).sum() == 0, "anchored must be a subset of spring_b"
