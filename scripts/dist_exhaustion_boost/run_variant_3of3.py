"""Distribution-exhaustion 3-of-3 sizing-boost shim.

Hypothesis (final form, with OI capitulation gate active):
When ALL three are true at a long entry:
  1. tf4h_wyckoff_bearish_score >= 0.6  (distribution confirmed)
  2. oi_change_24h <= -0.02              (OI capitulating)
  3. range_position_20 < 0.40            (price at support)
multiply position size by X.

This is the FULL rule from commit 977e6bf ("the highest-conviction long entry
our engine can identify"). The 2-of-3 version (without OI) was REJECTED
because it fired on 41% of long entries — too broad. With the OI gate active
now (backfilled from data.binance.vision, May 2026), the rule should be much
more selective.

NO production code, configs, or YAMLs modified — monkey-patches only.

Usage:
    python3 scripts/dist_exhaustion_boost/run_variant_3of3.py \
        --boost 1.25 \
        --bearish-min 0.6 \
        --range-pos-max 0.40 \
        --oi-change-max -0.02 \
        --start-date 2023-01-01 \
        --end-date 2024-12-31 \
        --output-dir results/dist_exhaustion_boost_3of3/X_1.25/test_2023_2024
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import bin.backtest_v11_standalone as bt_module  # noqa: E402
from bin.backtest_v11_standalone import StandaloneBacktestEngine  # noqa: E402

logger = logging.getLogger(__name__)


def install_shim(boost_x: float, bearish_min: float, range_pos_max: float,
                 oi_change_max: float):
    """Monkey-patch _open_position to apply 3-of-3 distribution_exhaustion boost."""
    state = {
        'n_eligible_entries': 0,
        'n_boost_triggered': 0,
        'n_failed_bearish': 0,
        'n_failed_oi': 0,
        'n_failed_range_pos': 0,
        'n_missing_data': 0,
        'boost_x': boost_x,
        'bearish_min': bearish_min,
        'range_pos_max': range_pos_max,
        'oi_change_max': oi_change_max,
        'feature_lookup': None,
    }

    original_open = StandaloneBacktestEngine._open_position

    def patched_open(self, *args, **kwargs):
        if state['feature_lookup'] is None and hasattr(self, 'features_df'):
            df = self.features_df
            required = ['tf4h_wyckoff_bearish_score', 'range_position_20', 'oi_change_24h']
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise RuntimeError(f"Required features missing from parquet: {missing}")
            state['feature_lookup'] = df[required]
            logger.info(f"[3of3_SHIM] Feature lookup ready: {len(df):,} rows")

        direction = kwargs.get('direction') or (args[2] if len(args) > 2 else None)
        timestamp = kwargs.get('timestamp') or (args[0] if len(args) > 0 else None)

        if direction == 'long' and timestamp is not None and state['feature_lookup'] is not None:
            try:
                row = state['feature_lookup'].loc[timestamp]
                bearish = float(row['tf4h_wyckoff_bearish_score'])
                range_pos = float(row['range_position_20'])
                oi24 = float(row['oi_change_24h'])
                missing = False
            except (KeyError, ValueError, TypeError):
                missing = True
                bearish, range_pos, oi24 = 0.0, 1.0, 0.0

            state['n_eligible_entries'] += 1
            if missing or bearish != bearish or range_pos != range_pos or oi24 != oi24:
                state['n_missing_data'] += 1
            else:
                # Track which gates fail (for diagnostics)
                pass_b = bearish >= state['bearish_min']
                pass_o = oi24 <= state['oi_change_max']
                pass_p = range_pos < state['range_pos_max']
                if not pass_b: state['n_failed_bearish'] += 1
                if not pass_o: state['n_failed_oi'] += 1
                if not pass_p: state['n_failed_range_pos'] += 1
                if pass_b and pass_o and pass_p:
                    state['n_boost_triggered'] += 1
                    if 'allocated_size_pct' in kwargs:
                        kwargs['allocated_size_pct'] *= state['boost_x']
                    logger.debug(
                        f"[DIST_EX_3of3] {timestamp} long: "
                        f"bear={bearish:.3f}, oi24={oi24:.3f}, rp={range_pos:.3f} → size×{state['boost_x']:.2f}"
                    )
        return original_open(self, *args, **kwargs)

    StandaloneBacktestEngine._open_position = patched_open
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boost', type=float, required=True)
    parser.add_argument('--bearish-min', type=float, default=0.6)
    parser.add_argument('--range-pos-max', type=float, default=0.40)
    parser.add_argument('--oi-change-max', type=float, default=-0.02)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--config', type=str,
                        default='configs/bull_machine_isolated_v11_fixed.json')
    parser.add_argument('--feature-store', type=str,
                        default='data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet')
    parser.add_argument('--initial-cash', type=float, default=100_000.0)
    parser.add_argument('--commission-rate', type=float, default=0.0002)
    parser.add_argument('--slippage-bps', type=float, default=3.0)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if not args.verbose:
        for noisy in ['engine.archetypes.archetype_instance',
                      'engine.portfolio.archetype_allocator',
                      'engine.config.archetype_config_loader',
                      'engine.context.regime_service',
                      'engine.archetypes.exit_logic',
                      'engine.portfolio.regime_allocator']:
            logging.getLogger(noisy).setLevel(logging.WARNING)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = install_shim(args.boost, args.bearish_min, args.range_pos_max, args.oi_change_max)

    print('=' * 80)
    print('DISTRIBUTION-EXHAUSTION (3-of-3) SIZING BOOST')
    print('=' * 80)
    print(f'Boost X:                 {args.boost:.2f}')
    print(f'bearish_min:             {args.bearish_min:.2f}')
    print(f'range_pos_max:           {args.range_pos_max:.2f}')
    print(f'oi_change_max:           {args.oi_change_max:.4f}')
    print(f'Date Range:              {args.start_date or "start"} → {args.end_date or "end"}')
    print(f'Output dir:              {output_dir}')
    print('=' * 80)

    with open(Path(args.config), 'r') as f:
        config = json.load(f)
    feature_store_path = Path(args.feature_store)
    if not feature_store_path.exists():
        logger.error(f'Feature store not found: {feature_store_path}')
        sys.exit(1)

    engine = StandaloneBacktestEngine(
        config=config,
        feature_store_path=str(feature_store_path),
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        signal_mode='fusion',
        sizing_mode='fixed',
        health_mode='off',
        invalidation_mode=False,
    )
    engine.run(start_date=args.start_date, end_date=args.end_date)
    engine.print_summary()

    engine.save_trade_log(str(output_dir / 'trade_log.csv'))
    engine.save_equity_curve(str(output_dir / 'equity_curve.csv'))

    stats = engine.get_performance_stats()
    stats['dist_exhaustion_3of3'] = {
        'boost_x': args.boost,
        'bearish_min': args.bearish_min,
        'range_pos_max': args.range_pos_max,
        'oi_change_max': args.oi_change_max,
        'n_eligible_entries': state['n_eligible_entries'],
        'n_missing_data': state['n_missing_data'],
        'n_failed_bearish': state['n_failed_bearish'],
        'n_failed_oi': state['n_failed_oi'],
        'n_failed_range_pos': state['n_failed_range_pos'],
        'n_boost_triggered': state['n_boost_triggered'],
    }

    import numpy as np
    stats_clean = {}
    for k, v in stats.items():
        if isinstance(v, (list, set)):
            stats_clean[k] = list(v)
        elif isinstance(v, (np.floating, np.integer)):
            stats_clean[k] = float(v)
        elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            stats_clean[k] = str(v)
        else:
            stats_clean[k] = v
    with open(output_dir / 'performance_stats.json', 'w') as f:
        json.dump(stats_clean, f, indent=2, default=str)

    breakdown = engine.get_archetype_breakdown()
    if len(breakdown) > 0:
        breakdown.to_csv(str(output_dir / 'archetype_breakdown.csv'), index=False)

    print()
    print('=' * 80)
    print('3-of-3 BOOST TELEMETRY')
    print('=' * 80)
    print(f'Eligible long entries:    {state["n_eligible_entries"]:,}')
    print(f'Missing data:             {state["n_missing_data"]:,} ({state["n_missing_data"]/max(state["n_eligible_entries"],1)*100:.1f}%)')
    print(f'Failed bearish gate:      {state["n_failed_bearish"]:,}')
    print(f'Failed OI gate:           {state["n_failed_oi"]:,}')
    print(f'Failed range_pos gate:    {state["n_failed_range_pos"]:,}')
    print(f'Boost-triggered:          {state["n_boost_triggered"]:,}')
    if state['n_eligible_entries'] > 0:
        pct = 100.0 * state['n_boost_triggered'] / state['n_eligible_entries']
        print(f'  → {pct:.1f}% of eligible entries had 3-of-3 condition met')


if __name__ == '__main__':
    main()
