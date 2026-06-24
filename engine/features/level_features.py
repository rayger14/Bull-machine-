"""
Level-Awareness Features (single shared path: batch backfill + live bar-by-bar)
===============================================================================

Implements the level inventory + sweep/acceptance/day-type primitives specified by:
- docs/knowledge/trader_knowledge_failed_breakdown_es_transfer_2026_06_11.md
- docs/knowledge/industry_study_wick_trap_detection_2026_06_11.md

Design rules (non-negotiable):
- Point-in-time correct: every feature at bar t uses ONLY bars <= t.
  Swing pivots use a symmetric 10-bar window and only become visible
  10 bars after the pivot bar (confirmation lag, no repaint).
- One code path: `compute_level_features` is the only implementation.
  `LevelFeatureTracker` (live) calls it on a trailing buffer; because every
  feature has finite trailing data requirements (see BUFFER note below),
  the incremental output is identical to the batch output.

Interface contract (column names are consumed by the wick_trap rebuild detector):
    prior_day_high, prior_day_low, week_open,
    swing_low_50, swing_high_50, swing_low_touches,
    eq_low_pool, eq_high_pool,
    round_number_below, round_number_above,
    dist_to_support_atr, dist_to_resistance_atr,
    level_quality_low,
    sweep_low_event, sweep_high_event, acceptance_bars_above,
    day_type

ATR note: ATR14 here is the simple (SMA) 14-bar mean of true range, NOT
Wilder's EWMA. A finite window is required for exact batch/live parity
(EWMA has infinite memory, so a trimmed live buffer would drift).

BUFFER note: with the default 1000-bar live buffer, all features are exact
as long as (a) the most recent confirmed swing pivot, and (b) the most recent
sweep_low_event, occurred within the buffer. On BTC 1H both happen every few
dozen bars; pathological flat data could violate this (documented limitation).
"""

import numpy as np
import pandas as pd

# --- Parameters (per industry study: pivot window well below 50 to cut lag) ---
PIVOT_WINDOW = 10            # symmetric pivot half-window; confirmation lag = 10 bars
TOUCH_LOOKBACK = 500         # bars scanned for level touches / pivot pools
TOUCH_ATR_TOL = 0.25         # |low - level| <= 0.25*ATR14 counts as a touch
EQ_POOL_ATR_TOL = 0.5        # pivot cluster tolerance (LuxAlgo ATR-scaled margin)
CONFLUENCE_ATR_TOL = 0.5     # prior_day_low vs swing_low confluence tolerance
ATR_PERIOD = 14
DAY_TYPE_MIN_RUN = 4         # consecutive hourly closes beyond prior-day extreme
TOUCH_COUNT_CAP = 5          # touch-count normalization cap for level_quality_low

LEVEL_COLUMNS = [
    "prior_day_high", "prior_day_low", "week_open",
    "swing_low_50", "swing_high_50", "swing_low_touches",
    "eq_low_pool", "eq_high_pool",
    "round_number_below", "round_number_above",
    "dist_to_support_atr", "dist_to_resistance_atr",
    "level_quality_low",
    "sweep_low_event", "sweep_high_event", "acceptance_bars_above",
    "day_type",
]

_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _validate_input(ohlcv: pd.DataFrame) -> None:
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("ohlcv must have a DatetimeIndex")
    if ohlcv.index.tz is not None:
        raise ValueError(
            "ohlcv index must be tz-naive UTC (use idx.tz_localize(None))"
        )
    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError("ohlcv index must be sorted ascending")
    missing = [c for c in _REQUIRED_COLS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv missing required columns: {missing}")


def _consecutive_run(cond: np.ndarray, reset: np.ndarray) -> np.ndarray:
    """Length of the consecutive True-run of `cond` ending at each bar.

    The run breaks on False AND on `reset` (e.g. UTC-day boundary), so runs
    never span a reset point. Pure trailing computation (no lookahead).
    """
    key = np.cumsum((~cond) | reset)
    return pd.Series(cond.astype(np.int64)).groupby(key).cumsum().to_numpy()


def compute_level_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Vectorized level-awareness features over 1H OHLCV bars.

    Args:
        ohlcv: DataFrame with tz-naive UTC DatetimeIndex and columns
            open/high/low/close/volume (1H bars).

    Returns:
        DataFrame indexed identically to `ohlcv` with exactly LEVEL_COLUMNS.
        All values at bar t depend only on bars <= t (pivots carry a
        PIVOT_WINDOW-bar confirmation lag).
    """
    _validate_input(ohlcv)
    idx = ohlcv.index
    n = len(ohlcv)

    h = ohlcv["high"].to_numpy(dtype=np.float64)
    l = ohlcv["low"].to_numpy(dtype=np.float64)
    c = ohlcv["close"].to_numpy(dtype=np.float64)

    # --- ATR14 (SMA of true range; finite window => exact live parity) ---
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.fmax(h - l, np.fmax(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = (
        pd.Series(tr, index=idx)
        .rolling(ATR_PERIOD, min_periods=ATR_PERIOD)
        .mean()
        .to_numpy()
    )

    # --- Prior UTC-day extremes (prior day is complete when current day starts) ---
    day_key = idx.normalize()
    daily_high = ohlcv["high"].groupby(day_key).max()
    daily_low = ohlcv["low"].groupby(day_key).min()
    pdh = daily_high.shift(1).reindex(day_key).to_numpy()
    pdl = daily_low.shift(1).reindex(day_key).to_numpy()

    # --- Week open (Monday 00:00 UTC open; first bar of the Mon-Sun week if
    #     the 00:00 bar is missing — still point-in-time) ---
    week_key = idx.to_period("W-SUN")  # weeks ending Sunday => Monday start
    week_open = ohlcv["open"].groupby(week_key).transform("first").to_numpy()

    # --- Confirmed swing pivots (symmetric 10-bar window, visible at p+10) ---
    win = 2 * PIVOT_WINDOW + 1
    lo_s, hi_s = ohlcv["low"], ohlcv["high"]
    piv_lo = lo_s == lo_s.rolling(win, center=True, min_periods=win).min()
    piv_hi = hi_s == hi_s.rolling(win, center=True, min_periods=win).max()
    # Pivot price placed at its CONFIRMATION bar (pivot bar + PIVOT_WINDOW).
    vis_piv_low = lo_s.where(piv_lo).shift(PIVOT_WINDOW)
    vis_piv_high = hi_s.where(piv_hi).shift(PIVOT_WINDOW)
    swing_low = vis_piv_low.ffill().to_numpy()
    swing_high = vis_piv_high.ffill().to_numpy()
    vis_pl = vis_piv_low.to_numpy()
    vis_ph = vis_piv_high.to_numpy()

    # --- Touch counts + equal-low/high pivot pools (trailing 500-bar scan).
    #     Slice-shifted accumulation: no lookahead, no large temporaries. ---
    touch = np.zeros(n, dtype=np.int64)
    eq_lo_cnt = np.zeros(n, dtype=np.int64)
    eq_hi_cnt = np.zeros(n, dtype=np.int64)
    tol_touch = TOUCH_ATR_TOL * atr
    tol_eq = EQ_POOL_ATR_TOL * atr
    for k in range(min(TOUCH_LOOKBACK, n)):
        src = slice(0, n - k)
        dst = slice(k, n)
        touch[dst] += np.abs(l[src] - swing_low[dst]) <= tol_touch[dst]
        eq_lo_cnt[dst] += np.abs(vis_pl[src] - swing_low[dst]) <= tol_eq[dst]
        eq_hi_cnt[dst] += np.abs(vis_ph[src] - swing_high[dst]) <= tol_eq[dst]
    eq_low_pool = (eq_lo_cnt >= 2).astype(np.int8)
    eq_high_pool = (eq_hi_cnt >= 2).astype(np.int8)

    # --- Round numbers (Osler stop clustering): magnitude-aware grid =
    #     1/10th of the price decade (1000s for BTC in [10k, 100k)). ---
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = 10.0 ** np.floor(np.log10(np.where(c > 0, c, np.nan))) / 10.0
    rn_below = np.floor(c / grid) * grid
    rn_above = rn_below + grid

    # --- Nearest structural support/resistance + ATR-relative distance ---
    support = np.fmax(pdl, swing_low)        # fmax/fmin ignore NaN sides
    resistance = np.fmin(pdh, swing_high)
    with np.errstate(invalid="ignore"):
        dist_support = (c - support) / atr
        dist_resistance = (resistance - c) / atr

    # --- Level quality for the support below (0-1 composite) ---
    norm_touch = np.minimum(touch, TOUCH_COUNT_CAP) / float(TOUCH_COUNT_CAP)
    confluence = (np.abs(pdl - swing_low) <= CONFLUENCE_ATR_TOL * atr).astype(
        np.float64
    )
    quality = norm_touch * 0.5 + eq_low_pool * 0.25 + confluence * 0.25
    level_quality_low = np.where(np.isnan(swing_low), np.nan, quality)

    # --- Sweep events (LuxAlgo criterion: wick beyond level, close back inside) ---
    sweep_low = (l < support) & (c > support)
    sweep_high = (h > resistance) & (c < resistance)

    # --- Acceptance counter: consecutive closes (incl. current) holding >= the
    #     level swept by the most recent sweep_low_event. Resets to 0 once a
    #     close gives the level back; 0 before any sweep. ---
    ev = pd.Series(sweep_low.astype(np.int64), index=idx)
    gid = ev.cumsum()
    swept_level = pd.Series(np.where(sweep_low, support, np.nan), index=idx)
    lvl_g = swept_level.groupby(gid).transform("first").to_numpy()
    with np.errstate(invalid="ignore"):
        held = pd.Series((c >= lvl_g).astype(np.int64), index=idx)
    alive = held.groupby(gid).cumprod()              # 1 until first failed close
    pos = gid.groupby(gid).cumcount()                # bars since sweep (0-based)
    acceptance = (alive * (pos + 1)).where(gid > 0, 0).to_numpy()

    # --- Day type (behavioral, concurrent-state): trend only once the prior-day
    #     extreme is broken on a CLOSE basis and held for >= 4 consecutive hourly
    #     closes WITHIN the current UTC day; reverts to range when reclaimed. ---
    day_vals = day_key.values
    new_day = np.r_[True, day_vals[1:] != day_vals[:-1]]
    with np.errstate(invalid="ignore"):
        below_pdl = c < pdl
        above_pdh = c > pdh
    run_below = _consecutive_run(below_pdl, new_day)
    run_above = _consecutive_run(above_pdh, new_day)
    day_type = np.zeros(n, dtype=np.int8)
    day_type[run_below >= DAY_TYPE_MIN_RUN] = -1
    day_type[run_above >= DAY_TYPE_MIN_RUN] = 1

    out = pd.DataFrame(
        {
            "prior_day_high": pdh,
            "prior_day_low": pdl,
            "week_open": week_open,
            "swing_low_50": swing_low,
            "swing_high_50": swing_high,
            "swing_low_touches": touch.astype(np.int32),
            "eq_low_pool": eq_low_pool,
            "eq_high_pool": eq_high_pool,
            "round_number_below": rn_below,
            "round_number_above": rn_above,
            "dist_to_support_atr": dist_support,
            "dist_to_resistance_atr": dist_resistance,
            "level_quality_low": level_quality_low,
            "sweep_low_event": sweep_low.astype(np.int8),
            "sweep_high_event": sweep_high.astype(np.int8),
            "acceptance_bars_above": acceptance.astype(np.int32),
            "day_type": day_type,
        },
        index=idx,
    )
    return out[LEVEL_COLUMNS]


class LevelFeatureTracker:
    """Incremental live wrapper around `compute_level_features`.

    Maintains a trailing bar buffer (default 1000 bars ~ 41 days on 1H) and
    recomputes the shared vectorized function on the buffer each update —
    one code path for backtest and live (parity study item 6). All features
    only need trailing data, so the last row equals the batch value exactly
    (see BUFFER note in the module docstring).
    """

    def __init__(self, max_buffer_bars: int = 1000):
        if max_buffer_bars < TOUCH_LOOKBACK + 2 * PIVOT_WINDOW:
            raise ValueError(
                f"max_buffer_bars must be >= {TOUCH_LOOKBACK + 2 * PIVOT_WINDOW} "
                "to cover the touch lookback + pivot confirmation lag"
            )
        self._max_buffer = max_buffer_bars
        self._rows = []  # list of (ts, open, high, low, close, volume)

    @staticmethod
    def _coerce_ts(ts_raw) -> pd.Timestamp:
        ts = pd.Timestamp(ts_raw)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts

    def prime(self, ohlcv: pd.DataFrame) -> None:
        """Warm-start the buffer from historical bars (e.g. on live restart)."""
        _validate_input(ohlcv)
        for ts, row in ohlcv.iterrows():
            self._rows.append(
                (
                    self._coerce_ts(ts),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                )
            )
        self._trim()

    def update(self, bar: dict) -> dict:
        """Ingest one closed bar and return its level features.

        Args:
            bar: dict with keys timestamp, open, high, low, close, volume.

        Returns:
            dict of LEVEL_COLUMNS values for this bar (identical to the batch
            function's row for the same data).
        """
        self._rows.append(
            (
                self._coerce_ts(bar["timestamp"]),
                float(bar["open"]),
                float(bar["high"]),
                float(bar["low"]),
                float(bar["close"]),
                float(bar["volume"]),
            )
        )
        self._trim()
        buf = pd.DataFrame(
            self._rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        ).set_index("timestamp")
        feats = compute_level_features(buf)
        return {col: feats[col].iloc[-1] for col in LEVEL_COLUMNS}

    def _trim(self) -> None:
        if len(self._rows) > self._max_buffer:
            del self._rows[: len(self._rows) - self._max_buffer]
