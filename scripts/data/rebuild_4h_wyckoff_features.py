#!/usr/bin/env python3
"""
Rebuild broken 4H Wyckoff score columns in the BTC feature store.

Reuses the existing detector logic in `engine/wyckoff/events.py` (no
modifications) to compute per-bar 4H Wyckoff bullish/bearish/phase scores
and forward-fills them onto the 1H index of the feature parquet.

Behavior — preserves all other ~298 columns bit-for-bit:
    1. Loads `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet`.
    2. Snapshots a hash of every column NOT in the update set.
    3. Resamples 1H OHLCV → 4H using the same convention as
       `bin/live/live_feature_computer.py::_resample_to_tf` (pandas default
       `closed='left', label='left'`).
    4. Runs `detect_all_wyckoff_events` on the 4H bars.
    5. Computes a rolling 250-bar (≈ live's 1000-1H buffer ≈ 41 days of 4H)
       max over accumulation events (= bullish_score) and distribution events
       (= bearish_score) at each 4H bar — matching the live engine's
       `create_wyckoff_context(buf_4h, lookback=len(buf_4h))` call when the
       live buffer is full.
    6. Forward-fills onto the 1H index using a shift-by-bar-duration
       construction so each 1H bar reads the most recent 4H bar whose CLOSE
       is at or before the 1H bar's timestamp — this is the look-ahead-safe
       mapping (a 4H bar labeled `T` covers [T, T+4h) and only becomes
       readable at T+4h).
    7. Atomically writes the parquet back (temp + os.replace).
    8. Verifies all non-updated columns are bit-identical to the snapshot.

Usage:
    python3 scripts/rebuild_4h_wyckoff_features.py --dry-run
    python3 scripts/rebuild_4h_wyckoff_features.py
    python3 scripts/rebuild_4h_wyckoff_features.py \
        --parquet data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# --- Make engine importable when run from anywhere -------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.wyckoff.events import detect_all_wyckoff_events  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — derived from live engine, NOT new tunables
# ---------------------------------------------------------------------------

# Match _CFG_4H from bin/live/live_feature_computer.py:1602
CFG_4H: Dict[str, float] = {
    'sc_volume_z_min': 1.8,
    'sc_range_z_min': 1.2,
    'bc_volume_z_min': 1.8,
    'bc_range_z_min': 1.2,
    'sos_volume_z_min': 1.2,
    'sow_volume_z_min': 1.2,
    'spring_a_volume_z_min': 0.3,
    'ut_volume_z_min': 0.2,
    'st_lookback': 8,
    'st_volume_z_max': 0.0,
    'st_low_proximity': 0.04,
    'st_min_spacing': 3,
    'spring_b_breakdown_min': 0.002,
    'spring_b_breakdown_max': 0.02,
    'spring_b_recovery_bars': 2,
    'sm_st_max_count': 2,
    'sm_spring_tolerance': 0.015,
    'sm_ut_tolerance': 0.015,
}

# Match _ACCUM_EVENTS / _DISTRIB_EVENTS from engine/wyckoff/events.py:1192
ACCUM_EVENTS = ['sc', 'ar', 'st', 'spring_a', 'spring_b', 'sos', 'lps']
DISTRIB_EVENTS = ['bc', 'as', 'sow', 'ut', 'utad', 'lpsy']
ALL_EVENTS = ACCUM_EVENTS + DISTRIB_EVENTS

# Live engine uses `lookback=len(buf_4h_copy)` where buf_4h_copy comes from
# resampling a buffer capped at 1000 1H bars → ~250 4H bars (~41 days).
# We approximate that rolling behaviour here so per-bar scores in the parquet
# match what live would have produced at that bar.
ROLLING_LOOKBACK_4H_BARS = 250

# Columns we will UPDATE (and overwrite) in the parquet.
UPDATE_COLS = (
    'tf4h_wyckoff_bullish_score',
    'tf4h_wyckoff_bearish_score',
    'tf4h_wyckoff_phase_score',
)

DEFAULT_PARQUET = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rebuild_4h_wyckoff")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H OHLCV to 4H using the same convention as the live engine.

    Pandas default for `resample('4h')` is closed='left', label='left'.
    A 4H bar labeled `T` covers the half-open interval [T, T+4h).
    """
    needed = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in needed if c not in df_1h.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns in 1H frame: {missing}")
    df_4h = df_1h[needed].resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()
    return df_4h


def compute_4h_event_confidences(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Run the existing detector on 4H bars (returns df with wyckoff_*_confidence cols)."""
    df = df_4h.copy()
    df = detect_all_wyckoff_events(df, cfg=dict(CFG_4H), htf_context=None)
    return df


def rolling_directional_scores(
    df_4h_with_events: pd.DataFrame,
    lookback: int = ROLLING_LOOKBACK_4H_BARS,
) -> pd.DataFrame:
    """Compute per-4H-bar rolling-max bullish/bearish/phase scores.

    For each 4H bar, scan the trailing `lookback` bars (inclusive) for the
    maximum confidence across accumulation / distribution events. This
    mirrors the live engine call:
        create_wyckoff_context(buf_4h_copy, lookback=len(buf_4h_copy))
    where buf_4h_copy is bounded by the live buffer (~250 4H bars).
    """
    out = pd.DataFrame(index=df_4h_with_events.index)

    bull_cols = [f'wyckoff_{e}_confidence' for e in ACCUM_EVENTS
                 if f'wyckoff_{e}_confidence' in df_4h_with_events.columns]
    bear_cols = [f'wyckoff_{e}_confidence' for e in DISTRIB_EVENTS
                 if f'wyckoff_{e}_confidence' in df_4h_with_events.columns]
    all_cols = [f'wyckoff_{e}_confidence' for e in ALL_EVENTS
                if f'wyckoff_{e}_confidence' in df_4h_with_events.columns]

    def _row_max(cols: List[str]) -> pd.Series:
        if not cols:
            return pd.Series(0.0, index=df_4h_with_events.index)
        return df_4h_with_events[cols].fillna(0.0).max(axis=1)

    bull_per_bar = _row_max(bull_cols)
    bear_per_bar = _row_max(bear_cols)
    all_per_bar = _row_max(all_cols)

    win = max(1, lookback)
    out['tf4h_wyckoff_bullish_score'] = bull_per_bar.rolling(
        window=win, min_periods=1
    ).max().astype(float)
    out['tf4h_wyckoff_bearish_score'] = bear_per_bar.rolling(
        window=win, min_periods=1
    ).max().astype(float)
    out['tf4h_wyckoff_phase_score'] = all_per_bar.rolling(
        window=win, min_periods=1
    ).max().astype(float)

    return out


def map_4h_scores_to_1h(
    scores_4h: pd.DataFrame,
    index_1h: pd.DatetimeIndex,
    bar_duration: pd.Timedelta = pd.Timedelta(hours=4),
) -> pd.DataFrame:
    """Forward-fill 4H scores onto a 1H index without look-ahead.

    A 4H bar at label `T` covers [T, T+4h) and its values only become
    knowable at the bar's CLOSE = T+4h. So:
        score_for_1h_bar(t) = score from the most recent 4H bar with
                              T + 4h <= t  (i.e. close <= t)
    Implementation: shift the 4H label to its close time (T → T+4h),
    reindex onto the 1H index, then forward-fill.
    """
    closed_idx = scores_4h.index + bar_duration
    closed_scores = scores_4h.copy()
    closed_scores.index = closed_idx
    # Combine + sort, then forward-fill onto the 1H index.
    combined = closed_scores.reindex(
        closed_scores.index.union(index_1h)
    ).sort_index().ffill()
    aligned = combined.reindex(index_1h)
    aligned = aligned.fillna(0.0)
    return aligned


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def hash_columns(df: pd.DataFrame, cols: List[str]) -> Dict[str, str]:
    """Hash each column's bytes (via pandas->numpy) for bit-for-bit comparison."""
    out: Dict[str, str] = {}
    for c in cols:
        if c not in df.columns:
            continue
        arr = df[c].to_numpy()
        h = hashlib.sha1()
        # Include dtype + shape so a dtype change is also caught.
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        try:
            h.update(arr.tobytes())
        except (TypeError, ValueError):
            # Object dtype — fall back to deterministic stringification
            h.update(np.array([repr(v) for v in arr.tolist()]).tobytes())
        out[c] = h.hexdigest()
    return out


def hash_index(df: pd.DataFrame) -> str:
    h = hashlib.sha1()
    h.update(str(df.index.dtype).encode())
    h.update(str(df.index.shape).encode())
    if isinstance(df.index, pd.DatetimeIndex):
        # Use UTC nanosecond ints for a deterministic, value-based hash.
        # `asi8` returns int64 ns since epoch (UTC) regardless of tz.
        h.update(df.index.asi8.tobytes())
        h.update(str(df.index.tz).encode())
    else:
        try:
            h.update(np.ascontiguousarray(np.asarray(df.index)).tobytes())
        except (TypeError, ValueError):
            h.update(np.array([repr(v) for v in df.index]).tobytes())
    return h.hexdigest()


def look_ahead_check(
    df_1h_ohlcv: pd.DataFrame,
    final_1h_scores: pd.DataFrame,
    sample_hours: List[pd.Timestamp],
) -> Tuple[bool, List[Dict]]:
    """For each sample hour t, recompute 4H scores using only bars up to t and
    verify they match the parquet value bit-for-bit.

    The "only data up to t" check uses 1H OHLCV truncated at t, resampled
    to 4H, fed through the detector, then the rolling-max → close-time
    mapping. The score at t in the truncated pipeline must equal the
    score at t in the full pipeline.
    """
    results: List[Dict] = []
    overall_ok = True

    for t in sample_hours:
        if t not in final_1h_scores.index:
            continue
        df_trunc = df_1h_ohlcv.loc[:t]
        if len(df_trunc) < 100:
            continue
        df_4h = resample_1h_to_4h(df_trunc)
        if len(df_4h) < 30:
            continue
        df_4h_ev = compute_4h_event_confidences(df_4h)
        scores_4h = rolling_directional_scores(df_4h_ev)
        scores_1h = map_4h_scores_to_1h(scores_4h, df_trunc.index)
        truncated_at_t = scores_1h.loc[t]
        full_at_t = final_1h_scores.loc[t]

        diffs = {}
        ok = True
        for c in UPDATE_COLS:
            a = float(truncated_at_t[c])
            b = float(full_at_t[c])
            if not np.isclose(a, b, atol=1e-9, equal_nan=True):
                ok = False
                diffs[c] = (a, b)
        results.append({'t': t, 'ok': ok, 'diffs': diffs})
        if not ok:
            overall_ok = False
            log.warning("Look-ahead check FAILED at %s: %s", t, diffs)
        else:
            log.info("Look-ahead check OK at %s", t)
    return overall_ok, results


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, engine='pyarrow', compression='snappy')
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--parquet', default=DEFAULT_PARQUET,
        help=f"Path to feature parquet (default: {DEFAULT_PARQUET})",
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help="Compute scores and run validation, but DO NOT write the parquet.",
    )
    p.add_argument(
        '--look-ahead-samples', type=int, default=8,
        help="Number of random 1H bars to spot-check for look-ahead bias.",
    )
    p.add_argument(
        '--seed', type=int, default=42,
        help="RNG seed for reproducible spot checks.",
    )
    args = p.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        log.error("Feature parquet not found at %s", parquet_path)
        return 2

    log.info("Loading parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("Loaded shape=%s, dtype-of-index=%s", df.shape, df.index.dtype)

    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to promote a known column to the index
        for cand in ('timestamp', 'datetime', 'time'):
            if cand in df.columns:
                df = df.set_index(pd.to_datetime(df[cand], utc=True))
                df.index.name = cand
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        log.error("Parquet index is not a DatetimeIndex; cannot resample.")
        return 3
    if df.index.tz is None:
        # Match downstream expectation; live uses UTC. Don't mutate persisted
        # tz unless the parquet has none (avoid silently changing column data).
        try:
            df_for_resample = df.copy()
            df_for_resample.index = df_for_resample.index.tz_localize('UTC')
        except Exception:
            df_for_resample = df.copy()
    else:
        df_for_resample = df

    # 1) Snapshot non-update columns for byte-equality verification
    non_update_cols = [c for c in df.columns if c not in UPDATE_COLS]
    log.info(
        "Snapshotting %d non-update columns (will preserve bit-for-bit)",
        len(non_update_cols),
    )
    pre_hashes = hash_columns(df, non_update_cols)
    pre_index_hash = hash_index(df)

    # Distribution before
    log.info("--- BEFORE distribution of update columns ---")
    for c in UPDATE_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce')
            log.info(
                "  %s: count=%d nan=%d mean=%.6f std=%.6f min=%.6f max=%.6f "
                "pct_nonzero=%.2f%% n_unique=%d",
                c, s.notna().sum(), s.isna().sum(),
                float(s.mean()) if s.notna().any() else float('nan'),
                float(s.std()) if s.notna().any() else float('nan'),
                float(s.min()) if s.notna().any() else float('nan'),
                float(s.max()) if s.notna().any() else float('nan'),
                100.0 * float((s.fillna(0) != 0).sum()) / max(1, len(s)),
                int(s.nunique(dropna=True)),
            )
        else:
            log.info("  %s: ABSENT (will be added)", c)

    # 2) Resample 1H → 4H
    log.info("Resampling 1H OHLCV → 4H (closed='left', label='left')")
    df_4h = resample_1h_to_4h(df_for_resample)
    log.info("4H bars: %d (%s → %s)", len(df_4h), df_4h.index[0], df_4h.index[-1])

    # 3) Run Wyckoff event detector on 4H
    log.info("Running detect_all_wyckoff_events on 4H bars")
    df_4h_ev = compute_4h_event_confidences(df_4h)

    # 4) Rolling-max → 4H-per-bar scores
    log.info("Computing rolling %d-bar max scores", ROLLING_LOOKBACK_4H_BARS)
    scores_4h = rolling_directional_scores(df_4h_ev)

    # 5) Forward-fill onto 1H index using close-time mapping (no look-ahead)
    log.info("Mapping 4H scores → 1H index via close-time forward-fill")
    scores_1h = map_4h_scores_to_1h(scores_4h, df_for_resample.index)

    # Re-align scores_1h's index to match the original (untouched) df.index
    # so the column write preserves the original row alignment.
    scores_1h.index = df.index

    # 6) Look-ahead bias check on a few random hours
    log.info("Running look-ahead bias spot checks on %d sample hours",
             args.look_ahead_samples)
    rng = np.random.default_rng(args.seed)
    valid_idx = df.index
    # Skip the first ~250 4H bars (≈ 1000 1H bars) so the rolling window is
    # fully populated; otherwise the lookback isn't apples-to-apples.
    burn_in = ROLLING_LOOKBACK_4H_BARS * 4
    if len(valid_idx) > burn_in + 100:
        candidates = valid_idx[burn_in:]
    else:
        candidates = valid_idx[len(valid_idx) // 2:]
    sample_count = min(args.look_ahead_samples, len(candidates))
    sample_pos = rng.choice(len(candidates), size=sample_count, replace=False)
    sample_hours = sorted([candidates[i] for i in sample_pos])
    look_ok, look_results = look_ahead_check(
        df_for_resample[['open', 'high', 'low', 'close', 'volume']],
        scores_1h,
        sample_hours,
    )
    if not look_ok:
        log.error("LOOK-AHEAD BIAS DETECTED — refusing to write parquet.")
        return 4

    # Distribution after
    log.info("--- AFTER distribution of update columns (1H index) ---")
    for c in UPDATE_COLS:
        s = pd.to_numeric(scores_1h[c], errors='coerce')
        log.info(
            "  %s: count=%d nan=%d mean=%.6f std=%.6f min=%.6f max=%.6f "
            "pct_nonzero=%.2f%% n_unique=%d",
            c, s.notna().sum(), s.isna().sum(),
            float(s.mean()), float(s.std()),
            float(s.min()), float(s.max()),
            100.0 * float((s.fillna(0) != 0).sum()) / max(1, len(s)),
            int(s.nunique(dropna=True)),
        )

    # Sanity: top-5 highest bearish bars (expect 2021 tops)
    top_bear = scores_1h['tf4h_wyckoff_bearish_score'].nlargest(5)
    log.info("Top 5 bearish_score bars (sanity check, expect ~2021 tops):")
    for ts, v in top_bear.items():
        log.info("  %s -> %.4f", ts, v)
    top_bull = scores_1h['tf4h_wyckoff_bullish_score'].nlargest(5)
    log.info("Top 5 bullish_score bars (sanity check, expect ~2020/2022 bottoms):")
    for ts, v in top_bull.items():
        log.info("  %s -> %.4f", ts, v)

    # Apply updates
    df_out = df.copy()
    for c in UPDATE_COLS:
        df_out[c] = scores_1h[c].astype(np.float64).values

    # 7) Verify non-update columns are bit-identical
    post_hashes = hash_columns(df_out, non_update_cols)
    post_index_hash = hash_index(df_out)
    if post_index_hash != pre_index_hash:
        log.error("INDEX changed during update — aborting.")
        return 5
    mismatched = [c for c in non_update_cols if pre_hashes.get(c) != post_hashes.get(c)]
    if mismatched:
        log.error("Non-update columns changed during processing: %s", mismatched[:10])
        return 6
    log.info("Verified %d non-update columns are bit-identical to source.",
             len(non_update_cols))

    # 8) Atomic write
    if args.dry_run:
        log.warning("DRY RUN — skipping parquet write.")
    else:
        log.info("Writing parquet (atomic) → %s", parquet_path)
        atomic_write_parquet(df_out, parquet_path)
        # Reload + verify shape and column hashes
        reloaded = pd.read_parquet(parquet_path)
        if reloaded.shape != df_out.shape:
            log.error("Reload shape mismatch: %s vs %s", reloaded.shape, df_out.shape)
            return 7
        re_hashes = hash_columns(reloaded, non_update_cols)
        bad = [c for c in non_update_cols if pre_hashes.get(c) != re_hashes.get(c)]
        if bad:
            log.error("Reload reveals %d changed non-update columns: %s",
                      len(bad), bad[:10])
            return 8
        log.info("Reload verified: shape=%s, all non-update columns intact.",
                 reloaded.shape)

    log.info("DONE.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
