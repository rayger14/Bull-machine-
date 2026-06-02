"""
Regression test for wyckoff_regime_bug_audit_2026_06_02.

Asserts that WyckoffHTFContext bullish_score and bearish_score are
mutually exclusive — they cannot both exceed 0.5 unless the underlying
confidences are very nearly tied (in which case downstream phase is
"transition" and direction is "neutral").

Without the fix, 28/94 live trades had bull > 0.5 AND bear > 0.5 with
correlation -0.27.
"""

import numpy as np
import pandas as pd
import pytest

from engine.wyckoff.events import create_wyckoff_context, _ACCUM_EVENTS, _DISTRIB_EVENTS


def _make_df_with_events(bull_conf: float, bear_conf: float, n_bars: int = 30):
    """Build a synthetic DF with all wyckoff_*_confidence columns set.

    Plants a single accumulation event (spring_a) on the last bar with bull_conf,
    and a single distribution event (utad) on the second-to-last bar with bear_conf.
    """
    df = pd.DataFrame({
        'open':  np.linspace(100, 110, n_bars),
        'high':  np.linspace(101, 111, n_bars),
        'low':   np.linspace(99, 109, n_bars),
        'close': np.linspace(100, 110, n_bars),
        'volume': np.ones(n_bars) * 1000.0,
    })
    all_events = _ACCUM_EVENTS + _DISTRIB_EVENTS
    for e in all_events:
        df[f'wyckoff_{e}_confidence'] = 0.0
    if bull_conf > 0:
        df.loc[df.index[-1], 'wyckoff_spring_a_confidence'] = bull_conf
    if bear_conf > 0:
        df.loc[df.index[-2], 'wyckoff_utad_confidence'] = bear_conf
    return df


def test_strong_bullish_suppresses_bearish():
    """raw_bull=0.9, raw_bear=0.4 → net bull=0.5, net bear=0."""
    df = _make_df_with_events(bull_conf=0.9, bear_conf=0.4)
    ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H')
    # net-dominance: winner = raw_winner - raw_loser
    assert ctx.bullish_score == pytest.approx(0.5, abs=1e-6)
    assert ctx.bearish_score == pytest.approx(0.0, abs=1e-6)
    assert ctx.dominant_direction == 'bullish'
    assert ctx.phase == 'accumulation'


def test_strong_bearish_suppresses_bullish():
    """raw_bear=0.9, raw_bull=0.4 → net bear=0.5, net bull=0."""
    df = _make_df_with_events(bull_conf=0.4, bear_conf=0.9)
    ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H')
    assert ctx.bearish_score == pytest.approx(0.5, abs=1e-6)
    assert ctx.bullish_score == pytest.approx(0.0, abs=1e-6)
    assert ctx.dominant_direction == 'bearish'
    assert ctx.phase == 'distribution'


def test_close_tie_collapses_both_to_zero():
    """raw_bull=0.85, raw_bear=0.80 → net bull=0.05, net bear=0.

    The remaining net of 0.05 reflects the tiny edge bullish has, but it
    won't pass the 0.5 downstream gate. Crucially, BOTH scores cannot exceed
    0.5 — the invariant downstream archetypes assume.
    """
    df = _make_df_with_events(bull_conf=0.85, bear_conf=0.80)
    ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H')
    assert ctx.bullish_score == pytest.approx(0.05, abs=1e-6)
    assert ctx.bearish_score == pytest.approx(0.0, abs=1e-6)
    # Neither passes the > 0.5 gate
    assert not (ctx.bullish_score > 0.5 and ctx.bearish_score > 0.5)


def test_exact_tie_collapses_both_to_zero():
    """raw_bull=raw_bear=0.9 → net both = 0 (truly ambiguous regime)."""
    df = _make_df_with_events(bull_conf=0.9, bear_conf=0.9)
    ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H')
    assert ctx.bullish_score == pytest.approx(0.0, abs=1e-6)
    assert ctx.bearish_score == pytest.approx(0.0, abs=1e-6)
    assert ctx.phase == 'none'
    assert ctx.dominant_direction == 'neutral'


def test_mutual_exclusion_invariant_across_grid():
    """Sweep raw bull/bear from 0 to 1.0. The invariant: NEVER both > 0.5.
    """
    fails = []
    for raw_bull in np.linspace(0.0, 1.0, 11):
        for raw_bear in np.linspace(0.0, 1.0, 11):
            df = _make_df_with_events(bull_conf=raw_bull, bear_conf=raw_bear)
            ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H')
            if ctx.bullish_score > 0.5 and ctx.bearish_score > 0.5:
                fails.append((raw_bull, raw_bear, ctx.bullish_score, ctx.bearish_score))
    assert not fails, f"Mutual exclusion violations (both > 0.5): {fails[:5]}"


def test_legacy_opt_out_preserves_bug():
    """Sanity: mutual_exclusion=False reproduces the original bug (independent maxes).
    Used to verify the flag actually toggles behavior.
    """
    df = _make_df_with_events(bull_conf=0.97, bear_conf=0.85)
    ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H', mutual_exclusion=False)
    # Legacy: both are raw maxes — the bug we're fixing
    assert ctx.bullish_score == pytest.approx(0.97, abs=1e-6)
    assert ctx.bearish_score == pytest.approx(0.85, abs=1e-6)


def test_no_events_returns_zero():
    df = _make_df_with_events(bull_conf=0.0, bear_conf=0.0)
    ctx = create_wyckoff_context(df, lookback=len(df), timeframe='4H')
    assert ctx.bullish_score == 0.0
    assert ctx.bearish_score == 0.0
    assert ctx.phase == 'none'
    assert ctx.dominant_direction == 'neutral'
