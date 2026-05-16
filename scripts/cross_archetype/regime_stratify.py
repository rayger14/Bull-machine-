"""Regime-stratified breakdown of candidate anti-signal patterns.

For each (winner, loser, window) candidate, split outcomes by entry_regime.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_anti_signals import load_logs, build_cooccurrence_for_winner_loser  # noqa


def stratify(signal_log, trade_log, output_dir,
             candidates):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sl, entries = load_logs(signal_log, trade_log)
    fires_by_archetype = sl.groupby('archetype')

    rows = []
    for winner, loser, window_hours in candidates:
        we = entries[entries['archetype'] == winner].copy().reset_index(drop=True)
        if loser not in fires_by_archetype.groups or len(we) == 0:
            continue
        lf = fires_by_archetype.get_group(loser)[['timestamp']]
        flags = build_cooccurrence_for_winner_loser(we, lf, [window_hours])
        we[f'concurrent_{window_hours}h'] = flags[f'loser_in_last_{window_hours}h'].values

        for regime, sub in we.groupby('entry_regime'):
            for concurrent_flag, ssub in sub.groupby(f'concurrent_{window_hours}h'):
                rows.append({
                    'winner': winner,
                    'loser': loser,
                    'window_hours': window_hours,
                    'regime': regime,
                    'concurrent': int(concurrent_flag),
                    'n': len(ssub),
                    'pnl_sum': ssub['position_pnl'].sum(),
                    'pnl_mean': ssub['position_pnl'].mean(),
                    'r_mean': ssub['r_multiple'].mean(),
                    'win_rate': ssub['is_win'].mean(),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out / 'regime_stratified.csv', index=False)
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal-log', required=True)
    parser.add_argument('--trade-log', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    candidates = [
        ('confluence_breakout', 'oi_divergence', 4),
        ('confluence_breakout', 'oi_divergence', 12),
        ('confluence_breakout', 'long_squeeze', 12),
    ]
    stratify(args.signal_log, args.trade_log, args.output_dir, candidates)
