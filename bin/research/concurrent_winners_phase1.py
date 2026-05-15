#!/usr/bin/env python3
"""
Phase 1+2: Concurrent winning archetype analysis.

Captures pre-dedup signal coincidence on every bar by monkey-patching
the engine's _deduplicate_signals to log all signals BEFORE dedup runs.

Then computes forward-return-based outcome proxies:
  - For each archetype signal, record fwd_24h, fwd_72h, fwd_168h close-to-close return
  - Compare outcomes:
      * fired alone (no other concurrent same-direction winning archetype)
      * fired with 1+ concurrent winning archetype
      * fired with 2+ concurrent winning archetype

A 'winning archetype' = one of the archetypes with PF>=1.5 in the
post-Optuna production results (MEMORY.md table). This filters out noise
from chronic losers (long_squeeze, oi_divergence, order_block_retest).

This is a SIGNAL-LEVEL analysis. It does NOT execute trades, exits, or
sizing — it answers ONE question:

  Are bars where multiple winning archetypes fire concurrently
  predictive of better forward returns than bars where any single
  winning archetype fires alone?

Run:
  python3 bin/research/concurrent_winners_phase1.py \
    --start 2018-01-01 --end 2024-12-31 \
    --out results/cross_archetype/concurrent_conviction/phase1_pre_dedup
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

# Archetypes with PF >= 1.5 in post-Optuna production (from MEMORY.md):
#   liquidity_sweep (2.21), retest_cluster (1.92), wick_trap (1.59),
#   failed_continuation (13.47 + 0.35 weighted), trap_within_trend (3.54),
#   liquidity_vacuum (inf), funding_divergence (2.61), spring (1.62)
# fvg_continuation (1.36) and others are sub-1.5 PF and excluded.
WINNERS: Set[str] = {
    'liquidity_sweep',
    'retest_cluster',
    'wick_trap',
    'failed_continuation',
    'trap_within_trend',
    'liquidity_vacuum',
    'funding_divergence',
    'spring',
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/bull_machine_isolated_v11_fixed.json')
    p.add_argument('--feature-store', default='data/features_mtf/BTC_1H_LATEST.parquet')
    p.add_argument('--start', default='2018-01-01')
    p.add_argument('--end', default='2024-12-31')
    p.add_argument('--out', required=True, help='Output directory for analysis files')
    p.add_argument('--max-bars', type=int, default=None,
                   help='Limit number of bars (for quick smoke tests)')
    return p.parse_args()


def build_engine(config_path: str, features_df: pd.DataFrame):
    """Build the IsolatedArchetypeEngine WITH dedup disabled so we see all signals."""
    from bin.backtest_v11_standalone import StandaloneBacktestEngine

    with open(config_path) as f:
        cfg = json.load(f)

    # CRITICAL: disable dedup so we capture pre-dedup concurrency
    cfg.setdefault('signal_dedup', {})['mode'] = 'disabled'

    # Don't enable any optional ML
    cfg['use_ml_fusion'] = False

    engine = StandaloneBacktestEngine(
        config=cfg,
        features_df=features_df,
        initial_cash=100_000.0,
        commission_rate=0.0002,
        slippage_bps=3.0,
        signal_mode='fusion',
    )
    return engine


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 1+2: Concurrent Winners Analysis ===")
    print(f"Date range: {args.start} to {args.end}")
    print(f"Winner archetypes ({len(WINNERS)}): {sorted(WINNERS)}")
    print(f"Output: {out_dir}\n")

    # Load feature store
    fs_path = REPO_ROOT / args.feature_store
    print(f"Loading feature store: {fs_path}")
    df = pd.read_parquet(fs_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[(df.index >= args.start) & (df.index <= args.end)]
    print(f"  Bars: {len(df):,}  Cols: {len(df.columns)}")
    print(f"  Range: {df.index.min()} → {df.index.max()}\n")

    # Pre-compute forward-return targets (close-to-close, % return)
    df['fwd_24h']  = df['close'].pct_change(periods=24).shift(-24)
    df['fwd_72h']  = df['close'].pct_change(periods=72).shift(-72)
    df['fwd_168h'] = df['close'].pct_change(periods=168).shift(-168)

    engine = build_engine(args.config, df)
    df_engine = engine.features_df  # may have derived cols added

    # Snapshot all signal firings: (timestamp, archetype_id, direction, fusion_score)
    firings: List[Dict] = []

    n_bars = len(df_engine)
    if args.max_bars:
        n_bars = min(n_bars, args.max_bars)

    print(f"Scanning {n_bars:,} bars for signals...")
    log_every = max(1, n_bars // 20)
    t_start = pd.Timestamp.utcnow()

    for bar_idx in range(n_bars):
        bar = df_engine.iloc[bar_idx]
        prev_row = df_engine.iloc[bar_idx - 1] if bar_idx > 0 else None
        lookback_start = max(0, bar_idx - 500)
        lookback_df = df_engine.iloc[lookback_start:bar_idx + 1]

        signals = engine.engine.get_signals(
            bar=bar,
            bar_index=bar_idx,
            prev_row=prev_row,
            lookback_df=lookback_df,
            signal_mode='fusion',
        )

        if not signals:
            continue

        ts = bar.name
        fwd24 = df_engine.at[ts, 'fwd_24h'] if 'fwd_24h' in df_engine.columns else np.nan
        fwd72 = df_engine.at[ts, 'fwd_72h'] if 'fwd_72h' in df_engine.columns else np.nan
        fwd168 = df_engine.at[ts, 'fwd_168h'] if 'fwd_168h' in df_engine.columns else np.nan
        regime = bar.get('regime_label', 'neutral')

        for s in signals:
            firings.append({
                'timestamp': ts,
                'bar_idx': bar_idx,
                'archetype': s.archetype_id,
                'direction': s.direction,
                'fusion_score': s.fusion_score,
                'entry_price': s.entry_price,
                'regime': regime,
                'fwd_24h': fwd24,
                'fwd_72h': fwd72,
                'fwd_168h': fwd168,
                'is_winner': s.archetype_id in WINNERS,
            })

        if bar_idx % log_every == 0 and bar_idx > 0:
            pct = 100.0 * bar_idx / n_bars
            elapsed = (pd.Timestamp.utcnow() - t_start).total_seconds()
            print(f"  [{bar_idx:>6}/{n_bars}] {pct:5.1f}% | {len(firings)} firings | {elapsed:.0f}s")

    elapsed = (pd.Timestamp.utcnow() - t_start).total_seconds()
    print(f"\nDone. {len(firings):,} firings captured in {elapsed:.0f}s.\n")

    sig_df = pd.DataFrame(firings)
    sig_df.to_parquet(out_dir / 'signal_firings.parquet')
    print(f"Wrote {out_dir / 'signal_firings.parquet'}")

    # =========================================================================
    # Phase 1: Identify winning archetype PAIRS that co-fire most often
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 — Winner pair co-firing frequency (long signals only)")
    print("=" * 70)

    long_win = sig_df[(sig_df['direction'] == 'long') & sig_df['is_winner']]
    # Group by bar, list winners that fired
    grouped = long_win.groupby('timestamp')['archetype'].apply(set)
    # Forward returns at bar (use the median fwd return across signals that bar)
    # but fwd is constant per bar so just take first
    fwd_by_bar = long_win.groupby('timestamp')[['fwd_24h', 'fwd_72h', 'fwd_168h']].first()
    regime_by_bar = long_win.groupby('timestamp')['regime'].first()

    pair_counts: Counter = Counter()
    pair_fwd_72: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    pair_fwd_168: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    for ts, winset in grouped.items():
        if len(winset) < 2:
            continue
        winners = sorted(winset)
        fwd72 = fwd_by_bar.at[ts, 'fwd_72h']
        fwd168 = fwd_by_bar.at[ts, 'fwd_168h']
        for i in range(len(winners)):
            for j in range(i + 1, len(winners)):
                pair = (winners[i], winners[j])
                pair_counts[pair] += 1
                if pd.notna(fwd72):
                    pair_fwd_72[pair].append(fwd72)
                if pd.notna(fwd168):
                    pair_fwd_168[pair].append(fwd168)

    rows = []
    for pair, n in pair_counts.most_common(30):
        mean72 = float(np.mean(pair_fwd_72[pair])) if pair_fwd_72[pair] else np.nan
        med72 = float(np.median(pair_fwd_72[pair])) if pair_fwd_72[pair] else np.nan
        mean168 = float(np.mean(pair_fwd_168[pair])) if pair_fwd_168[pair] else np.nan
        win72 = (np.array(pair_fwd_72[pair]) > 0).mean() if pair_fwd_72[pair] else np.nan
        rows.append({
            'archetype_a': pair[0],
            'archetype_b': pair[1],
            'co_fire_count': n,
            'mean_fwd_72h_pct': mean72 * 100 if pd.notna(mean72) else np.nan,
            'median_fwd_72h_pct': med72 * 100 if pd.notna(med72) else np.nan,
            'win_rate_72h': win72 * 100 if pd.notna(win72) else np.nan,
            'mean_fwd_168h_pct': mean168 * 100 if pd.notna(mean168) else np.nan,
        })

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(out_dir / 'winner_pair_frequency.csv', index=False)
    print(pair_df.to_string(index=False))
    print(f"\nWrote {out_dir / 'winner_pair_frequency.csv'}")

    # =========================================================================
    # Phase 2: Outcome by concurrency level (per archetype)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2 — Outcome by concurrent-winner count (per archetype, long)")
    print("=" * 70)

    # Build per-bar concurrency map: timestamp -> number of distinct winners firing long
    long_win['n_concurrent_winners'] = long_win.groupby('timestamp')['archetype'].transform('nunique')

    # Outcome bins
    def conc_bin(n: int) -> str:
        if n <= 1:
            return '1_alone'
        elif n == 2:
            return '2_with_1_other'
        else:
            return '3plus'

    long_win['conc_bin'] = long_win['n_concurrent_winners'].apply(conc_bin)

    out_rows = []
    for arch in sorted(WINNERS):
        sub = long_win[long_win['archetype'] == arch]
        if len(sub) == 0:
            continue
        for bin_name in ['1_alone', '2_with_1_other', '3plus']:
            b = sub[sub['conc_bin'] == bin_name]
            if len(b) == 0:
                continue
            mean72 = b['fwd_72h'].mean() * 100
            mean168 = b['fwd_168h'].mean() * 100
            win72 = (b['fwd_72h'] > 0).mean() * 100
            win168 = (b['fwd_168h'] > 0).mean() * 100
            out_rows.append({
                'archetype': arch,
                'concurrency': bin_name,
                'n': len(b),
                'mean_fwd_72h_%': round(mean72, 3),
                'mean_fwd_168h_%': round(mean168, 3),
                'win_rate_72h_%': round(win72, 1),
                'win_rate_168h_%': round(win168, 1),
            })

    outcome_df = pd.DataFrame(out_rows)
    outcome_df.to_csv(out_dir / 'outcome_by_concurrency.csv', index=False)
    print(outcome_df.to_string(index=False))
    print(f"\nWrote {out_dir / 'outcome_by_concurrency.csv'}")

    # =========================================================================
    # Phase 2b: Aggregate across ALL winners — is more concurrent better?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2b — Aggregate outcome by concurrency level (all winners pooled)")
    print("=" * 70)
    agg_rows = []
    for bin_name in ['1_alone', '2_with_1_other', '3plus']:
        b = long_win[long_win['conc_bin'] == bin_name]
        if len(b) == 0:
            continue
        agg_rows.append({
            'concurrency': bin_name,
            'n_firings': len(b),
            'n_unique_bars': b['timestamp'].nunique(),
            'mean_fwd_24h_%': round(b['fwd_24h'].mean() * 100, 3),
            'mean_fwd_72h_%': round(b['fwd_72h'].mean() * 100, 3),
            'mean_fwd_168h_%': round(b['fwd_168h'].mean() * 100, 3),
            'median_fwd_72h_%': round(b['fwd_72h'].median() * 100, 3),
            'win_rate_72h_%': round((b['fwd_72h'] > 0).mean() * 100, 1),
            'win_rate_168h_%': round((b['fwd_168h'] > 0).mean() * 100, 1),
        })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(out_dir / 'aggregate_outcome_by_concurrency.csv', index=False)
    print(agg_df.to_string(index=False))
    print(f"\nWrote {out_dir / 'aggregate_outcome_by_concurrency.csv'}")

    # =========================================================================
    # Phase 2c: Train (2018-2022) vs OOS (2023-2024) split
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2c — Train (2018-2022) vs OOS (2023-2024) split")
    print("=" * 70)
    splits = []
    train = long_win[long_win['timestamp'] < pd.Timestamp('2023-01-01', tz='UTC')]
    test = long_win[long_win['timestamp'] >= pd.Timestamp('2023-01-01', tz='UTC')]

    # Tz-aware comparison: try both
    if len(train) == 0 and len(test) == 0:
        train = long_win[long_win['timestamp'] < pd.Timestamp('2023-01-01')]
        test = long_win[long_win['timestamp'] >= pd.Timestamp('2023-01-01')]

    for label, subset in [('TRAIN 2018-2022', train), ('OOS 2023-2024', test)]:
        for bin_name in ['1_alone', '2_with_1_other', '3plus']:
            b = subset[subset['conc_bin'] == bin_name]
            if len(b) == 0:
                splits.append({
                    'split': label,
                    'concurrency': bin_name,
                    'n_firings': 0,
                    'mean_fwd_72h_%': np.nan,
                    'win_rate_72h_%': np.nan,
                    'mean_fwd_168h_%': np.nan,
                })
                continue
            splits.append({
                'split': label,
                'concurrency': bin_name,
                'n_firings': len(b),
                'mean_fwd_72h_%': round(b['fwd_72h'].mean() * 100, 3),
                'win_rate_72h_%': round((b['fwd_72h'] > 0).mean() * 100, 1),
                'mean_fwd_168h_%': round(b['fwd_168h'].mean() * 100, 3),
            })
    split_df = pd.DataFrame(splits)
    split_df.to_csv(out_dir / 'train_oos_split.csv', index=False)
    print(split_df.to_string(index=False))
    print(f"\nWrote {out_dir / 'train_oos_split.csv'}")

    # =========================================================================
    # Phase 2d: Regime stratification — does concurrency boost help in bull AND bear?
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2d — Regime stratification")
    print("=" * 70)
    regime_rows = []
    for regime in sorted(long_win['regime'].dropna().unique()):
        rsub = long_win[long_win['regime'] == regime]
        for bin_name in ['1_alone', '2_with_1_other', '3plus']:
            b = rsub[rsub['conc_bin'] == bin_name]
            if len(b) == 0:
                continue
            regime_rows.append({
                'regime': regime,
                'concurrency': bin_name,
                'n_firings': len(b),
                'mean_fwd_72h_%': round(b['fwd_72h'].mean() * 100, 3),
                'win_rate_72h_%': round((b['fwd_72h'] > 0).mean() * 100, 1),
                'mean_fwd_168h_%': round(b['fwd_168h'].mean() * 100, 3),
            })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(out_dir / 'regime_stratified.csv', index=False)
    print(regime_df.to_string(index=False))
    print(f"\nWrote {out_dir / 'regime_stratified.csv'}")

    print(f"\n=== Phase 1+2 complete. All files in {out_dir} ===")
    return 0


if __name__ == '__main__':
    sys.exit(main())
