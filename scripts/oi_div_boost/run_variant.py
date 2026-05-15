"""OI-divergence concurrent-fire sizing-boost shim.

Hypothesis #1b (follow-on from losers_as_anti_signals study):
When `oi_divergence` fired (raw signal) within the last K bars BEFORE a long
entry from {wick_trap, liquidity_sweep, funding_divergence}, multiply
position-size by X.

This is a scratch/non-production runner. It imports the production backtester
and monkey-patches:
  1. IsolatedArchetypeEngine.get_signals — records raw oi_divergence fires
     (concurrent-fire detection, same logic as
     scripts/cross_archetype/analyze_anti_signals.py from
     agent-aa5eaef71a131fa32 branch quant/losers-as-anti-signals).
  2. StandaloneBacktestEngine._open_position — applies the X multiplier to
     `allocated_size_pct` when concurrence is detected.

NO production code, configs, or YAMLs are modified.

Usage:
    python3 scripts/oi_div_boost/run_variant.py \
        --boost 1.25 \
        --window-bars 12 \
        --start-date 2023-01-01 \
        --end-date 2024-12-31 \
        --output-dir results/oi_div_boost/X_1.25/test_2023_2024

The X=1.0 variant is the null/baseline (no boost) and should produce identical
results to the unpatched backtester — used as a parity check.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure we can import the production modules
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import production modules (READ-ONLY)
import bin.backtest_v11_standalone as bt_module  # noqa: E402
from bin.backtest_v11_standalone import StandaloneBacktestEngine  # noqa: E402
from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shim configuration (set via CLI)
# ---------------------------------------------------------------------------
ELIGIBLE_ARCHETYPES = {'wick_trap', 'liquidity_sweep', 'funding_divergence'}
ELIGIBLE_DIRECTION = 'long'
TRIGGER_ARCHETYPE = 'oi_divergence'


def install_shim(boost_x: float, window_bars: int, eligible_archetypes: set,
                 results_log_path: str | None = None):
    """Monkey-patch the backtester to apply the OI-div sizing boost.

    Args:
        boost_x: position-size multiplier (1.0 = null hypothesis).
        window_bars: K bars before entry within which oi_divergence must have
            fired (inclusive of the entry bar). Matches the prior study's
            "loser in last K hours" using bar_index-based check.
        eligible_archetypes: set of archetype names eligible for boost.
        results_log_path: optional path to write boost-fire records as
            line-delimited JSON for audit.
    """
    # ----- Shared state -----
    # We hang state on the IsolatedArchetypeEngine instance via attribute
    # injection in the patched get_signals. This is per-run state, reset
    # on engine init.
    boost_state: dict = {
        'last_oi_div_bar_idx': None,        # int | None
        'eligible_archetypes': set(eligible_archetypes),
        'window_bars': int(window_bars),
        'boost_x': float(boost_x),
        'n_oi_div_fires': 0,
        'n_eligible_entries': 0,            # eligible archetype + long
        'n_boost_triggered': 0,             # eligible + recent oi_div
        'boost_records': [],                # list of dicts (audit log)
        'results_log_path': results_log_path,
    }

    # ----- Patch IsolatedArchetypeEngine.get_signals -----
    _orig_get_signals = IsolatedArchetypeEngine.get_signals

    def _patched_get_signals(self, bar, regime_probs=None, bar_index=None,
                             prev_row=None, lookback_df=None,
                             signal_mode='fusion'):
        signals = _orig_get_signals(
            self, bar,
            regime_probs=regime_probs,
            bar_index=bar_index,
            prev_row=prev_row,
            lookback_df=lookback_df,
            signal_mode=signal_mode,
        )
        # Record raw fires for the trigger archetype. Note: signals here are
        # AFTER per-archetype detect() but BEFORE the backtester's threshold
        # filter and entry execution. That matches the "raw fire" semantics
        # in the original cooccurrence analysis (analyze_anti_signals.py
        # used signal_log which captured all post-detect raw fires).
        if signals and bar_index is not None:
            for s in signals:
                if s.archetype_id == TRIGGER_ARCHETYPE:
                    boost_state['last_oi_div_bar_idx'] = int(bar_index)
                    boost_state['n_oi_div_fires'] += 1
                    break  # one record per bar is sufficient
        return signals

    IsolatedArchetypeEngine.get_signals = _patched_get_signals

    # ----- Patch StandaloneBacktestEngine._open_position -----
    _orig_open_position = StandaloneBacktestEngine._open_position

    def _patched_open_position(self, *args, **kwargs):
        # Extract decision inputs
        archetype = kwargs.get('archetype') if 'archetype' in kwargs else (args[1] if len(args) > 1 else None)
        direction = kwargs.get('direction') if 'direction' in kwargs else (args[2] if len(args) > 2 else None)
        bar_idx = kwargs.get('bar_idx', 0)
        timestamp = kwargs.get('timestamp') if 'timestamp' in kwargs else (args[0] if len(args) > 0 else None)

        is_eligible = (
            archetype in boost_state['eligible_archetypes']
            and direction == ELIGIBLE_DIRECTION
        )
        if is_eligible:
            boost_state['n_eligible_entries'] += 1

        boost_applied = False
        recent_oi_div_bar = boost_state['last_oi_div_bar_idx']
        bars_since_oi = (
            (bar_idx - recent_oi_div_bar)
            if recent_oi_div_bar is not None else None
        )
        if (
            is_eligible
            and recent_oi_div_bar is not None
            and bars_since_oi is not None
            and 0 <= bars_since_oi <= boost_state['window_bars']
        ):
            # Apply boost: multiply allocated_size_pct
            if 'allocated_size_pct' in kwargs:
                pre = kwargs['allocated_size_pct']
                kwargs['allocated_size_pct'] = pre * boost_state['boost_x']
                post = kwargs['allocated_size_pct']
            else:
                # Find positional (this shouldn't happen given backtester code,
                # but handle defensively)
                pre = None
                post = None
            boost_state['n_boost_triggered'] += 1
            boost_applied = True
            rec = {
                'bar_idx': int(bar_idx),
                'timestamp': str(timestamp) if timestamp is not None else None,
                'archetype': archetype,
                'direction': direction,
                'oi_div_bar_idx': int(recent_oi_div_bar),
                'bars_since_oi_div': int(bars_since_oi),
                'size_pct_pre': float(pre) if pre is not None else None,
                'size_pct_post': float(post) if post is not None else None,
                'boost_x': boost_state['boost_x'],
            }
            boost_state['boost_records'].append(rec)
            if boost_state['results_log_path']:
                with open(boost_state['results_log_path'], 'a') as f:
                    f.write(json.dumps(rec) + '\n')

        return _orig_open_position(self, *args, **kwargs)

    StandaloneBacktestEngine._open_position = _patched_open_position

    # Hang state on the module so the main runner can access stats after run
    bt_module._OI_DIV_BOOST_STATE = boost_state
    return boost_state


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='OI-div sizing-boost shim runner')
    parser.add_argument('--boost', type=float, required=True,
                        help='Size multiplier X (1.0 = null hypothesis)')
    parser.add_argument('--window-bars', type=int, default=12,
                        help='Lookback window K (bars). Default 12 = match prior study.')
    parser.add_argument('--eligible-archetypes', type=str, default=None,
                        help='Comma-separated archetype names eligible for boost. '
                             'Default = wick_trap,liquidity_sweep,funding_divergence')
    parser.add_argument('--config', type=str,
                        default='configs/bull_machine_isolated_v11_fixed.json')
    parser.add_argument('--feature-store', type=str,
                        default='data/features_mtf/BTC_1H_LATEST.parquet')
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--initial-cash', type=float, default=100_000.0)
    parser.add_argument('--commission-rate', type=float, default=0.0002)
    parser.add_argument('--slippage-bps', type=float, default=3.0)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    # Mute noisy sub-loggers for clean output
    if not args.verbose:
        logging.getLogger('engine.archetypes.archetype_instance').setLevel(logging.WARNING)
        logging.getLogger('engine.portfolio.archetype_allocator').setLevel(logging.WARNING)
        logging.getLogger('engine.config.archetype_config_loader').setLevel(logging.WARNING)
        logging.getLogger('engine.context.regime_service').setLevel(logging.WARNING)
        logging.getLogger('engine.archetypes.exit_logic').setLevel(logging.WARNING)
        logging.getLogger('engine.portfolio.regime_allocator').setLevel(logging.WARNING)

    eligible = (
        set(s.strip() for s in args.eligible_archetypes.split(','))
        if args.eligible_archetypes
        else set(ELIGIBLE_ARCHETYPES)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boost_log_path = output_dir / 'boost_records.jsonl'
    # Ensure clean log
    if boost_log_path.exists():
        boost_log_path.unlink()

    # ----- Install monkey-patches -----
    state = install_shim(
        boost_x=args.boost,
        window_bars=args.window_bars,
        eligible_archetypes=eligible,
        results_log_path=str(boost_log_path),
    )

    print('=' * 80)
    print('OI-DIV SIZING-BOOST SHIM')
    print('=' * 80)
    print(f'Boost X:               {args.boost:.2f}')
    print(f'Window K (bars):       {args.window_bars}')
    print(f'Eligible archetypes:   {sorted(eligible)}')
    print(f'Direction:             {ELIGIBLE_DIRECTION}')
    print(f'Trigger archetype:     {TRIGGER_ARCHETYPE}')
    print(f'Date Range:            {args.start_date or "start"} to {args.end_date or "end"}')
    print(f'Output dir:            {output_dir}')
    print('=' * 80)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f'Config not found: {config_path}')
        sys.exit(1)
    with open(config_path, 'r') as f:
        config = json.load(f)

    feature_store_path = Path(args.feature_store)
    if not feature_store_path.exists():
        logger.error(f'Feature store not found: {feature_store_path}')
        sys.exit(1)

    # ----- Run -----
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

    # Save artifacts
    engine.save_trade_log(str(output_dir / 'trade_log.csv'))
    engine.save_equity_curve(str(output_dir / 'equity_curve.csv'))

    stats = engine.get_performance_stats()
    # Add boost telemetry
    stats['oi_div_boost'] = {
        'boost_x': args.boost,
        'window_bars': args.window_bars,
        'eligible_archetypes': sorted(eligible),
        'n_oi_div_fires': state['n_oi_div_fires'],
        'n_eligible_entries': state['n_eligible_entries'],
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

    # Save archetype breakdown
    breakdown = engine.get_archetype_breakdown()
    if len(breakdown) > 0:
        breakdown.to_csv(str(output_dir / 'archetype_breakdown.csv'), index=False)

    # Final telemetry to stdout
    print()
    print('=' * 80)
    print('OI-DIV BOOST TELEMETRY')
    print('=' * 80)
    print(f'oi_divergence raw fires:        {state["n_oi_div_fires"]:,}')
    print(f'Eligible long entries:          {state["n_eligible_entries"]:,}')
    print(f'Boost-triggered entries:        {state["n_boost_triggered"]:,}')
    if state['n_eligible_entries'] > 0:
        pct = 100.0 * state['n_boost_triggered'] / state['n_eligible_entries']
        print(f'  → {pct:.1f}% of eligible entries had concurrent oi_div')
    print(f'Boost records written:          {boost_log_path}')
    print()


if __name__ == '__main__':
    main()
