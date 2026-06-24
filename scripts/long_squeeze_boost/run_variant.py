"""long_squeeze 2-of-2 sizing-boost shim (RSI extreme + volume climax).

Hypothesis (from May 18 forensic on live trades):
The 1 winning live long_squeeze trade had RSI 73.9 AND vol_z 2.81 at entry.
The 3 losing trades had mid-range RSI (29.5–64) and mid-range vol_z (−0.59 to 1.36).

Boost test: when archetype is `long_squeeze`, direction is short, AND
  rsi_14 >= 70 AND volume_z >= 2.0
multiply position size by X.

NOT a filter — does NOT block low-conviction long_squeeze trades.

Mirror of scripts/dist_exhaustion_boost/run_variant_3of3.py wiring.
NO production code, configs, or YAMLs modified — monkey-patches only.

Usage:
    python3 scripts/long_squeeze_boost/run_variant.py \
        --boost 1.25 \
        --rsi-min 70.0 \
        --volz-min 2.0 \
        --start-date 2023-01-01 \
        --end-date 2024-12-31 \
        --output-dir results/long_squeeze_boost/X_1.25/test_2023_2024
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


def install_shim(boost_x: float, rsi_min: float, volz_min: float, fund_z_min: float | None = None):
    """Monkey-patch _open_position to apply long_squeeze 2-of-2 boost (3-of-3 if fund_z_min set)."""
    state = {
        'n_long_squeeze_entries': 0,
        'n_boost_triggered': 0,
        'n_failed_rsi': 0,
        'n_failed_volz': 0,
        'n_failed_fund_z': 0,
        'n_missing_data': 0,
        'boost_x': boost_x,
        'rsi_min': rsi_min,
        'volz_min': volz_min,
        'fund_z_min': fund_z_min,
        'feature_lookup': None,
    }

    original_open = StandaloneBacktestEngine._open_position

    def patched_open(self, *args, **kwargs):
        if state['feature_lookup'] is None and hasattr(self, 'features_df'):
            df = self.features_df
            cols_needed = ['rsi_14']
            # Try common volume_z column names
            volz_col = None
            for c in ['volume_z', 'volume_zscore', 'volume_z_7d']:
                if c in df.columns:
                    volz_col = c
                    cols_needed.append(c)
                    break
            if volz_col is None:
                raise RuntimeError("No volume_z column found in features_df")
            if 'funding_Z' in df.columns:
                cols_needed.append('funding_Z')
            state['feature_lookup'] = df[cols_needed]
            state['volz_col'] = volz_col
            logger.info(f"[LS_BOOST_SHIM] Feature lookup ready: {len(df):,} rows, volz_col={volz_col}")

        # Determine archetype + direction + timestamp
        archetype = kwargs.get('archetype') or (args[1] if len(args) > 1 else None)
        direction = kwargs.get('direction') or (args[2] if len(args) > 2 else None)
        timestamp = kwargs.get('timestamp') or (args[0] if len(args) > 0 else None)

        if archetype == 'long_squeeze' and direction == 'short' and timestamp is not None \
           and state['feature_lookup'] is not None:
            try:
                row = state['feature_lookup'].loc[timestamp]
                rsi = float(row['rsi_14'])
                volz = float(row[state['volz_col']])
                fund_z = float(row['funding_Z']) if 'funding_Z' in row else 0.0
                missing = False
            except (KeyError, ValueError, TypeError):
                missing = True
                rsi, volz, fund_z = 0.0, 0.0, 0.0

            state['n_long_squeeze_entries'] += 1
            if missing or rsi != rsi or volz != volz:
                state['n_missing_data'] += 1
            else:
                pass_r = rsi >= state['rsi_min']
                pass_v = volz >= state['volz_min']
                pass_f = (state['fund_z_min'] is None) or (fund_z == fund_z and fund_z >= state['fund_z_min'])
                if not pass_r: state['n_failed_rsi'] += 1
                if not pass_v: state['n_failed_volz'] += 1
                if not pass_f: state['n_failed_fund_z'] += 1
                if pass_r and pass_v and pass_f:
                    state['n_boost_triggered'] += 1
                    if 'allocated_size_pct' in kwargs:
                        kwargs['allocated_size_pct'] *= state['boost_x']
                    logger.debug(
                        f"[LS_BOOST] {timestamp} long_squeeze short: "
                        f"rsi={rsi:.2f} volz={volz:.2f} fund_z={fund_z:.2f} → size×{state['boost_x']:.2f}"
                    )
        return original_open(self, *args, **kwargs)

    StandaloneBacktestEngine._open_position = patched_open
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boost', type=float, required=True)
    parser.add_argument('--rsi-min', type=float, default=70.0)
    parser.add_argument('--volz-min', type=float, default=2.0)
    parser.add_argument('--fund-z-min', type=float, default=None,
                        help='Optional 3rd condition. Default: None (2-of-2).')
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

    state = install_shim(args.boost, args.rsi_min, args.volz_min, args.fund_z_min)

    print('=' * 80)
    print('LONG_SQUEEZE 2-of-2 SIZING BOOST (3-of-3 if fund_z_min set)')
    print('=' * 80)
    print(f'Boost X:        {args.boost:.2f}')
    print(f'rsi_min:        {args.rsi_min:.2f}')
    print(f'volz_min:       {args.volz_min:.2f}')
    print(f'fund_z_min:     {args.fund_z_min}')
    print(f'Date Range:     {args.start_date or "start"} → {args.end_date or "end"}')
    print(f'Output dir:     {output_dir}')
    print('=' * 80)

    with open(Path(args.config), 'r') as f:
        config = json.load(f)
    if not Path(args.feature_store).exists():
        logger.error(f'Feature store not found: {args.feature_store}')
        sys.exit(1)

    engine = StandaloneBacktestEngine(
        config=config,
        feature_store_path=args.feature_store,
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
    stats['long_squeeze_boost'] = {
        'boost_x': args.boost,
        'rsi_min': args.rsi_min,
        'volz_min': args.volz_min,
        'fund_z_min': args.fund_z_min,
        'n_long_squeeze_entries': state['n_long_squeeze_entries'],
        'n_missing_data': state['n_missing_data'],
        'n_failed_rsi': state['n_failed_rsi'],
        'n_failed_volz': state['n_failed_volz'],
        'n_failed_fund_z': state['n_failed_fund_z'],
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
    print('LONG_SQUEEZE BOOST TELEMETRY')
    print('=' * 80)
    print(f'long_squeeze entries:      {state["n_long_squeeze_entries"]:,}')
    print(f'Missing data:              {state["n_missing_data"]:,}')
    print(f'Failed RSI gate:           {state["n_failed_rsi"]:,}')
    print(f'Failed vol_z gate:         {state["n_failed_volz"]:,}')
    print(f'Failed funding_Z gate:     {state["n_failed_fund_z"]:,}')
    print(f'Boost-triggered:           {state["n_boost_triggered"]:,}')
    if state['n_long_squeeze_entries'] > 0:
        pct = 100.0 * state['n_boost_triggered'] / state['n_long_squeeze_entries']
        print(f'  → {pct:.1f}% of long_squeeze entries had RSI+volz boost condition met')


if __name__ == '__main__':
    main()
