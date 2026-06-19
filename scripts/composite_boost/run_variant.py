"""Generic composite-feature sizing-boost shim.

Takes a JSON spec describing a feature combination (which features must hit
which bins) and applies a size multiplier to any position whose entry bar
satisfies ALL conditions. Pure monkey-patch — NO production code or YAML changes.

Spec format (--spec FILE):
{
    "name": "macro_triple",
    "description": "taker_imbalance=H & yield_curve=H & funding_z=H",
    "boost_x": 1.5,
    "direction_filter": null,             // or "long" / "short" to restrict
    "conditions": [
        {"feature": "taker_imbalance", "op": "min", "value": 0.213},
        {"feature": "yield_curve",     "op": "min", "value": 0.357},
        {"feature": "funding_z",       "op": "min", "value": 1.117}
    ]
}

Cutoffs come from live trade-data terciles (see /tmp/composite_output.txt) so
they reflect what was empirically winning in live, not refit per backtest.

Usage:
    python3 scripts/composite_boost/run_variant.py \\
        --spec specs/pair_taker_oil.json \\
        --start-date 2020-01-01 \\
        --output-dir results/composite_boost/pair_taker_oil_X1.5
"""
from __future__ import annotations
import argparse, json, logging, sys, math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import bin.backtest_v11_standalone as bt_module  # noqa: E402
from bin.backtest_v11_standalone import StandaloneBacktestEngine  # noqa: E402

logger = logging.getLogger(__name__)

OP_FUNCS = {
    'min':   lambda val, target: val >= target,
    'max':   lambda val, target: val <= target,
    'eq':    lambda val, target: val == target,
    'lt':    lambda val, target: val < target,
    'gt':    lambda val, target: val > target,
}


def install_shim(spec: dict):
    """Monkey-patch _open_position to apply composite-feature size boost."""
    name = spec['name']
    boost_x = float(spec['boost_x'])
    conditions = spec['conditions']
    dir_filter = spec.get('direction_filter')  # None, "long", "short"
    required_features = [c['feature'] for c in conditions]

    state = {
        'name': name,
        'boost_x': boost_x,
        'n_entries_seen': 0,
        'n_eligible': 0,
        'n_boost_triggered': 0,
        'n_missing_data': 0,
        'per_condition_pass_count': {c['feature']: 0 for c in conditions},
        'feature_lookup': None,
    }

    original_open = StandaloneBacktestEngine._open_position

    def patched_open(self, *args, **kwargs):
        # Initialize feature lookup on first call
        if state['feature_lookup'] is None and hasattr(self, 'features_df'):
            df = self.features_df
            missing = [c for c in required_features if c not in df.columns]
            if missing:
                raise RuntimeError(
                    f"[{name}] Required features missing from parquet: {missing}"
                )
            state['feature_lookup'] = df[required_features]
            logger.info(f"[BOOST {name}] Feature lookup ready: {len(df):,} rows, "
                        f"boost_x={boost_x}, conditions={len(conditions)}")

        # Resolve direction + timestamp from kwargs/args
        direction = kwargs.get('direction') or (args[2] if len(args) > 2 else None)
        timestamp = kwargs.get('timestamp') or (args[0] if len(args) > 0 else None)

        # Apply direction filter
        if dir_filter is not None and direction != dir_filter:
            return original_open(self, *args, **kwargs)

        if timestamp is not None and state['feature_lookup'] is not None:
            state['n_entries_seen'] += 1
            try:
                row = state['feature_lookup'].loc[timestamp]
                # Get all values
                values = {c['feature']: float(row[c['feature']]) for c in conditions}
                # NaN check
                any_nan = any(v != v for v in values.values())
            except (KeyError, ValueError, TypeError):
                any_nan = True
                values = {}

            if any_nan or not values:
                state['n_missing_data'] += 1
            else:
                state['n_eligible'] += 1
                # Evaluate ALL conditions
                all_pass = True
                for cond in conditions:
                    feat = cond['feature']
                    op = cond['op']
                    target = float(cond['value'])
                    val = values[feat]
                    if not OP_FUNCS[op](val, target):
                        all_pass = False
                    else:
                        state['per_condition_pass_count'][feat] += 1

                if all_pass:
                    state['n_boost_triggered'] += 1
                    if 'allocated_size_pct' in kwargs:
                        kwargs['allocated_size_pct'] *= boost_x

        return original_open(self, *args, **kwargs)

    StandaloneBacktestEngine._open_position = patched_open
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=str, required=True,
                        help='JSON file with composite boost specification')
    parser.add_argument('--start-date', type=str, default='2020-01-01')
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

    spec = json.load(open(args.spec))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = install_shim(spec)

    print('=' * 80)
    print(f'COMPOSITE BOOST: {spec["name"]}')
    print(f'  {spec.get("description","")}')
    print(f'  boost_x = {spec["boost_x"]}')
    print(f'  direction_filter = {spec.get("direction_filter","None")}')
    print(f'  conditions:')
    for c in spec['conditions']:
        print(f'    {c["feature"]} {c["op"]} {c["value"]}')
    print('=' * 80)

    config = json.load(open(args.config))
    engine = StandaloneBacktestEngine(
        config=config, initial_cash=args.initial_cash,
        commission_rate=args.commission_rate, slippage_bps=args.slippage_bps,
        feature_store_path=args.feature_store,
    )
    engine.run(start_date=args.start_date, end_date=args.end_date)
    stats = engine.get_performance_stats()

    # Save standard outputs
    engine.save_trade_log(str(output_dir / 'trade_log.csv'))
    engine.save_equity_curve(str(output_dir / 'equity_curve.csv'))

    stats_out = dict(stats)
    stats_out['composite_boost'] = {
        'spec': spec,
        'n_entries_seen': state['n_entries_seen'],
        'n_eligible': state['n_eligible'],
        'n_boost_triggered': state['n_boost_triggered'],
        'n_missing_data': state['n_missing_data'],
        'per_condition_pass_count': state['per_condition_pass_count'],
        'fire_pct': (100.0 * state['n_boost_triggered'] / state['n_entries_seen']
                     if state['n_entries_seen'] else 0.0),
    }

    # JSON-safe
    import numpy as np
    clean = {}
    for k, v in stats_out.items():
        if isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        elif isinstance(v, (list, set)):
            clean[k] = list(v)
        elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = str(v)
        else:
            clean[k] = v
    with open(output_dir / 'performance_stats.json', 'w') as f:
        json.dump(clean, f, indent=2, default=str)

    breakdown = engine.get_archetype_breakdown()
    if len(breakdown) > 0:
        breakdown.to_csv(str(output_dir / 'archetype_breakdown.csv'), index=False)

    print()
    print('=' * 80)
    print(f'BOOST TELEMETRY: {spec["name"]}')
    print('=' * 80)
    print(f'  Entries seen:        {state["n_entries_seen"]:,}')
    print(f'  Missing data:        {state["n_missing_data"]:,}')
    print(f'  Eligible:            {state["n_eligible"]:,}')
    print(f'  Boost-triggered:     {state["n_boost_triggered"]:,}')
    if state['n_entries_seen'] > 0:
        pct = 100.0 * state['n_boost_triggered'] / state['n_entries_seen']
        print(f'  Fire rate:           {pct:.2f}% of all entries')
    print(f'  Per-condition pass counts (of eligible):')
    for feat, cnt in state['per_condition_pass_count'].items():
        pct = 100.0 * cnt / max(1, state['n_eligible'])
        print(f'    {feat:<22s} {cnt:>5d} ({pct:>4.1f}%)')
    print()
    print(f'Output: {output_dir}/')


if __name__ == '__main__':
    main()
