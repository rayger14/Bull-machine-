"""Cross-archetype anti-signals analysis.

Hypothesis #1: when chronic losers (oi_divergence, long_squeeze,
order_block_retest, fvg_continuation) FIRE at/near the same bar as
winning archetype entries, do those winners produce worse outcomes?

Inputs:
  - signal_log.csv (all raw signal fires)
  - trade_log.csv (executed trades w/ position_id, archetype, pnl, etc.)

Outputs:
  - cooccurrence_matrix.csv  (concurrent-fire counts by winner/loser/window)
  - conditional_outcomes.csv (mean/median R, win-rate, deltas)
  - top_patterns.json        (top 3 most-negative concurrent patterns)

Run:
  python3 scripts/cross_archetype/analyze_anti_signals.py \
      --signal-log results/cross_archetype/anti_signals/baseline_with_log/signal_log.csv \
      --trade-log  results/cross_archetype/anti_signals/baseline_with_log/trade_log.csv \
      --output-dir results/cross_archetype/anti_signals/analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Chronic losers per MEMORY.md + production results
LOSERS = ['oi_divergence', 'long_squeeze', 'order_block_retest', 'fvg_continuation']

# Top profitable archetypes (current production winners)
WINNERS = [
    'wick_trap', 'liquidity_sweep', 'retest_cluster', 'failed_continuation',
    'trap_within_trend', 'liquidity_vacuum', 'funding_divergence',
    'fvg_continuation', 'spring', 'confluence_breakout',
    'liquidity_compression', 'exhaustion_reversal',
]


def load_logs(signal_log_path: str, trade_log_path: str):
    sl = pd.read_csv(signal_log_path, parse_dates=['timestamp'])
    tl = pd.read_csv(trade_log_path, parse_dates=['timestamp', 'exit_timestamp'])

    # Normalize timezones (sometimes tz-aware, sometimes naive)
    if sl['timestamp'].dt.tz is not None:
        sl['timestamp'] = sl['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
    if tl['timestamp'].dt.tz is not None:
        tl['timestamp'] = tl['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
        tl['exit_timestamp'] = tl['exit_timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    # Deduplicate trade entries (scale-outs share position_id but represent
    # the same entry — keep first scale-out row per position)
    tl = tl.sort_values(['position_id', 'timestamp'])
    entries = tl.drop_duplicates(subset='position_id', keep='first').copy()

    # Aggregate PnL per position (sum across scale-out rows) for outcome
    pos_pnl = tl.groupby('position_id', as_index=False)['pnl'].sum()
    pos_pnl.rename(columns={'pnl': 'position_pnl'}, inplace=True)
    entries = entries.merge(pos_pnl, on='position_id', how='left')

    # Position notional risk: use position_size_usd and stop_loss to compute R
    # R-multiple = pnl / risk_at_entry, where risk = qty * |entry - stop|
    entries['risk_dollars'] = (
        entries['quantity'].abs() * (entries['entry_price'] - entries['stop_loss']).abs()
    )
    entries['r_multiple'] = entries['position_pnl'] / entries['risk_dollars'].replace(0, np.nan)
    entries['is_win'] = entries['position_pnl'] > 0

    return sl, entries


def build_cooccurrence_for_winner_loser(
    winner_entries: pd.DataFrame,
    loser_fires: pd.DataFrame,
    windows_hours=(0, 4, 12),
):
    """For each winner-archetype entry, mark whether the loser fired within
    each window. Windows are in BARS (1H feature store)."""
    if len(winner_entries) == 0 or len(loser_fires) == 0:
        return None

    loser_ts = loser_fires['timestamp'].sort_values().to_numpy()
    rows = []
    for ts in winner_entries['timestamp']:
        ts_np = np.datetime64(ts)
        row = {}
        for w in windows_hours:
            window_start = ts_np - np.timedelta64(w, 'h')
            window_end = ts_np  # up to (and including) the winner entry bar
            mask = (loser_ts >= window_start) & (loser_ts <= window_end)
            row[f'loser_in_last_{w}h'] = int(mask.any())
        rows.append(row)
    return pd.DataFrame(rows)


def conditional_outcomes(winner_entries: pd.DataFrame, label_col: str):
    """Return summary stats for trades grouped by label_col (0/1)."""
    g = winner_entries.groupby(label_col)
    out = g.agg(
        n=('position_pnl', 'size'),
        pnl_sum=('position_pnl', 'sum'),
        pnl_mean=('position_pnl', 'mean'),
        pnl_median=('position_pnl', 'median'),
        r_mean=('r_multiple', 'mean'),
        r_median=('r_multiple', 'median'),
        win_rate=('is_win', 'mean'),
    )
    return out


def analyze(signal_log_path, trade_log_path, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sl, entries = load_logs(signal_log_path, trade_log_path)
    print(f"Loaded {len(sl)} signal fires, {len(entries)} unique trade entries")

    fires_by_archetype = sl.groupby('archetype')

    # Pre-compute fire timestamps per loser
    loser_fires = {}
    for la in LOSERS:
        if la in fires_by_archetype.groups:
            loser_fires[la] = fires_by_archetype.get_group(la)[['timestamp']].copy()
        else:
            loser_fires[la] = pd.DataFrame(columns=['timestamp'])
        print(f"  loser {la}: {len(loser_fires[la])} fires")

    cooc_rows = []
    cond_rows = []

    windows = [0, 4, 12]

    for winner in WINNERS:
        we = entries[entries['archetype'] == winner].copy()
        if len(we) == 0:
            continue

        # Baseline (no loser concurrent in any window)
        baseline_stats = {
            'n': len(we),
            'pnl_sum': we['position_pnl'].sum(),
            'r_mean': we['r_multiple'].mean(),
            'r_median': we['r_multiple'].median(),
            'win_rate': we['is_win'].mean(),
        }

        for loser, lf in loser_fires.items():
            if loser == winner:
                continue
            flags = build_cooccurrence_for_winner_loser(we, lf, windows)
            if flags is None or len(flags) == 0:
                continue
            # Attach flags
            we_aug = we.reset_index(drop=True).copy()
            for col in flags.columns:
                we_aug[col] = flags[col].values

            for w in windows:
                col = f'loser_in_last_{w}h'
                n_concurrent = int(we_aug[col].sum())
                n_baseline = int((1 - we_aug[col]).sum())

                cooc_rows.append({
                    'winner': winner,
                    'loser': loser,
                    'window_hours': w,
                    'n_winner_entries': len(we_aug),
                    'n_concurrent': n_concurrent,
                    'pct_concurrent': n_concurrent / len(we_aug),
                })

                if n_concurrent >= 5:  # only meaningful if some samples
                    g = we_aug.groupby(col)
                    concurrent_pnl = we_aug.loc[we_aug[col] == 1, 'position_pnl']
                    baseline_pnl = we_aug.loc[we_aug[col] == 0, 'position_pnl']
                    concurrent_r = we_aug.loc[we_aug[col] == 1, 'r_multiple'].dropna()
                    baseline_r = we_aug.loc[we_aug[col] == 0, 'r_multiple'].dropna()

                    cond_rows.append({
                        'winner': winner,
                        'loser': loser,
                        'window_hours': w,
                        'n_concurrent': n_concurrent,
                        'n_baseline': n_baseline,
                        'pnl_concurrent_sum': concurrent_pnl.sum(),
                        'pnl_baseline_sum': baseline_pnl.sum(),
                        'pnl_concurrent_mean': concurrent_pnl.mean(),
                        'pnl_baseline_mean': baseline_pnl.mean(),
                        'r_concurrent_mean': concurrent_r.mean(),
                        'r_baseline_mean': baseline_r.mean(),
                        'r_concurrent_median': concurrent_r.median(),
                        'r_baseline_median': baseline_r.median(),
                        'wr_concurrent': we_aug.loc[we_aug[col] == 1, 'is_win'].mean(),
                        'wr_baseline': we_aug.loc[we_aug[col] == 0, 'is_win'].mean(),
                        'delta_pnl_mean': concurrent_pnl.mean() - baseline_pnl.mean(),
                        'delta_r_mean': concurrent_r.mean() - baseline_r.mean(),
                        'delta_wr': we_aug.loc[we_aug[col] == 1, 'is_win'].mean()
                                    - we_aug.loc[we_aug[col] == 0, 'is_win'].mean(),
                    })

    cooc_df = pd.DataFrame(cooc_rows)
    cond_df = pd.DataFrame(cond_rows)

    cooc_df.to_csv(out / 'cooccurrence_matrix.csv', index=False)
    cond_df.to_csv(out / 'conditional_outcomes.csv', index=False)

    # Top 3 most-negative patterns: rank by delta_pnl_mean (most negative)
    # with minimum sample sizes for credibility
    top = cond_df[cond_df['n_concurrent'] >= 10].sort_values('delta_pnl_mean').head(10)
    top_path = out / 'top_negative_patterns.csv'
    top.to_csv(top_path, index=False)

    # Concurrent-multiple-losers analysis
    # For each winner entry, count how many distinct losers concurrent within 4h
    multi_rows = []
    for winner in WINNERS:
        we = entries[entries['archetype'] == winner].copy()
        if len(we) == 0:
            continue
        we = we.reset_index(drop=True)
        we['multi_loser_count_4h'] = 0
        for loser, lf in loser_fires.items():
            if loser == winner or len(lf) == 0:
                continue
            flags = build_cooccurrence_for_winner_loser(we, lf, [4])
            we['multi_loser_count_4h'] = we['multi_loser_count_4h'] + flags['loser_in_last_4h'].values

        for count in sorted(we['multi_loser_count_4h'].unique()):
            sub = we[we['multi_loser_count_4h'] == count]
            multi_rows.append({
                'winner': winner,
                'concurrent_loser_count': int(count),
                'n': len(sub),
                'pnl_sum': sub['position_pnl'].sum(),
                'pnl_mean': sub['position_pnl'].mean(),
                'r_mean': sub['r_multiple'].mean(),
                'r_median': sub['r_multiple'].median(),
                'win_rate': sub['is_win'].mean(),
            })
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv(out / 'multi_loser_concurrent.csv', index=False)

    print(f"\nSaved: {out / 'cooccurrence_matrix.csv'}")
    print(f"Saved: {out / 'conditional_outcomes.csv'}")
    print(f"Saved: {out / 'top_negative_patterns.csv'}")
    print(f"Saved: {out / 'multi_loser_concurrent.csv'}")

    return cooc_df, cond_df, top, multi_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-log', required=True)
    parser.add_argument('--trade-log', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    analyze(args.signal_log, args.trade_log, args.output_dir)
