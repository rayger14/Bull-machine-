#!/usr/bin/env python3
"""
True Parameter Sweep Using Real Hybrid Runner

Instead of the simplified vectorized optimizer, this runs the ACTUAL
hybrid_runner.py with full production logic:
- Smart exits with trailing
- Partial TP logic
- Macro veto system
- Real position sizing
- Regime adaptation

This gives ACCURATE performance numbers we can trust.
"""

import subprocess
import json
import itertools
from pathlib import Path
from datetime import datetime
import pandas as pd
import re

# Parameter grid to test
PARAM_GRID = {
    'fusion_threshold': [0.55, 0.60, 0.65, 0.70],
    'wyckoff_weight': [0.25, 0.30, 0.35],
    'smc_weight': [0.10, 0.15, 0.20],
    'momentum_weight': [0.25, 0.30, 0.35],
    # HOB weight derived: 1.0 - others
}

def create_test_config(threshold, w_wyck, w_smc, w_mom):
    """Create config with test parameters"""
    w_hob = 1.0 - (w_wyck + w_smc + w_mom)

    if w_hob < 0.15 or w_hob > 0.35:
        return None  # Skip invalid weight combinations

    config = {
        "version": "1.8.6",
        "asset": "BTC",
        "profile": "test",
        "fast_signals": {
            "enabled": True,
            "mode": "execute_only_if_fusion_confirms",
            "min_confidence": 0.66,
            "adx_threshold": 20,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "price_extension_pct": 2.0
        },
        "fusion": {
            "entry_threshold_confidence": threshold,
            "full_engine_validation_interval_bars": 4,
            "weights": {
                "wyckoff": w_wyck,
                "liquidity": w_hob,
                "momentum": w_mom,
                "smc": w_smc
            }
        },
        "safety": {
            "loss_streak_threshold": 3,
            "atr_floor_percentile": 0.20,
            "atr_cap_percentile": 0.95
        },
        "context": {
            "macro_veto_threshold": 0.85,
            "vix_regime_switch_threshold": 22.0,
            "vix_hysteresis": {"on": 22.0, "off": 18.0},
            "funding_weight_cap": 0.20,
            "move_weight_cap": 0.25,
            "vix_panic_threshold": 30,
            "move_panic_threshold": 120,
            "dxy_extreme_threshold": 105,
            "crisis_fuse": {
                "enabled": True,
                "lookback_hours": 24,
                "allow_one_trade_if_fusion_confidence_ge": 0.80,
                "requires_mtf_alignment": True
            }
        },
        "entries": {
            "pullback_bars": 5,
            "pullback_pct": 0.01
        },
        "exits": {
            "use_structure_stops": True,
            "atr_k": 1.0,
            "tp1_r": 1.0,
            "tp1_pct": 0.5,
            "trail_method": "atr",
            "trail_atr_k": 1.2
        },
        "risk": {
            "base_risk_pct": 0.075,
            "max_position_size_pct": 0.20,
            "max_portfolio_risk_pct": 0.15
        },
        "mtf": {
            "timeframes": ["1H", "4H", "1D"],
            "require_alignment": True,
            "nested_threshold": 0.02
        },
        "pnl_tracker": {
            "leverage": 5.0,
            "risk_per_trade": 0.02,
            "atr_period": 20,
            "stop_buffer_multiplier": 2.0,
            "r_multiple_target": 2.0,
            "fees_bps": 10.0,
            "slippage_bps": 5.0,
            "max_margin_util": 0.50,
            "funding_rate_bps": 0.0,
            "exits": {
                "enable_partial": True,
                "scale_out_rr": 1.0,
                "scale_out_pct": 0.5,
                "move_sl_to_be_on_tp1": True,
                "trail_after_tp1": True,
                "trail_mode": "atr",
                "trail_atr_mult": 1.0,
                "regime_adaptive": True,
                "adx_period": 14,
                "adx_trend_hi": 25.0,
                "adx_range_lo": 20.0,
                "range_stop_factor": 0.75,
                "trend_stop_factor": 1.25,
                "liquidity_trap_protect": True,
                "liquidity_z_min": 1.3,
                "liquidity_lookback": 20,
                "macro_exit_enabled": True,
                "macro_exit_threshold": 0.8,
                "vix_exit_level": 30.0,
                "max_bars_in_trade": 96
            }
        }
    }

    return config

def parse_hybrid_output(output):
    """Parse P&L summary from hybrid_runner output"""
    metrics = {
        'trades': 0,
        'win_rate': 0.0,
        'total_return': 0.0,
        'sharpe': 0.0,
        'max_dd': 0.0,
        'profit_factor': 1.0,
        'avg_r': 0.0
    }

    lines = output.split('\n')

    for line in lines:
        # Match patterns like "Trades: 42"
        if 'Trades:' in line:
            match = re.search(r'Trades:\s*(\d+)', line)
            if match:
                metrics['trades'] = int(match.group(1))

        # Match "Win Rate: 45.2%"
        elif 'Win Rate:' in line or 'Win-rate:' in line:
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                metrics['win_rate'] = float(match.group(1))

        # Match "Return: +12.5%" or "Total Return: +12.5%"
        elif 'Return:' in line:
            match = re.search(r'([+-]?\d+\.?\d*)%', line)
            if match:
                metrics['total_return'] = float(match.group(1))

        # Match "Sharpe: 0.85"
        elif 'Sharpe:' in line or 'Sharpe Ratio:' in line:
            match = re.search(r'(\d+\.?\d*)', line)
            if match:
                metrics['sharpe'] = float(match.group(1))

        # Match "Max DD: -15.2%"
        elif 'Max DD:' in line or 'MaxDD:' in line:
            match = re.search(r'([+-]?\d+\.?\d*)%', line)
            if match:
                metrics['max_dd'] = abs(float(match.group(1)))

        # Match "PF: 1.45" or "Profit Factor: 1.45"
        elif 'PF:' in line or 'Profit Factor:' in line:
            match = re.search(r'(\d+\.?\d*)', line)
            if match:
                metrics['profit_factor'] = float(match.group(1))

        # Match "Avg R: 0.85"
        elif 'Avg R:' in line:
            match = re.search(r'(\d+\.?\d*)', line)
            if match:
                metrics['avg_r'] = float(match.group(1))

    return metrics

def run_backtest(config_path, asset, start, end):
    """Run hybrid_runner backtest and parse results"""
    cmd = [
        'python3', 'bin/live/hybrid_runner.py',
        '--asset', asset,
        '--start', start,
        '--end', end,
        '--config', config_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr  # Sometimes metrics are in stderr

        metrics = parse_hybrid_output(output)
        return metrics

    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Timeout (15 min) - skipping")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    print("🔬 Bull Machine True Parameter Sweep")
    print("Using REAL hybrid_runner.py with full execution logic")
    print("=" * 70)

    # Test period (18 months - good balance)
    asset = 'BTC'
    start = '2024-04-01'
    end = '2025-10-01'

    print(f"Asset: {asset}")
    print(f"Period: {start} → {end} (18 months)")
    print()

    # Generate all combinations
    combos = []
    for threshold in PARAM_GRID['fusion_threshold']:
        for w_wyck in PARAM_GRID['wyckoff_weight']:
            for w_smc in PARAM_GRID['smc_weight']:
                for w_mom in PARAM_GRID['momentum_weight']:
                    config = create_test_config(threshold, w_wyck, w_smc, w_mom)
                    if config:
                        combos.append((threshold, w_wyck, w_smc, w_mom, config))

    print(f"📋 Testing {len(combos)} parameter combinations")
    print(f"⏱️  Estimated time: {len(combos) * 5} minutes (~{len(combos) * 5 / 60:.1f} hours)\n")

    results = []

    for i, (threshold, w_wyck, w_smc, w_mom, config) in enumerate(combos, 1):
        w_hob = config['fusion']['weights']['liquidity']

        print(f"[{i}/{len(combos)}] Testing: T={threshold:.2f}, "
              f"W={w_wyck:.2f}, S={w_smc:.2f}, H={w_hob:.2f}, M={w_mom:.2f}")

        # Save temp config
        config_path = '/tmp/test_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Run backtest
        metrics = run_backtest(config_path, asset, start, end)

        if metrics and metrics['trades'] > 0:
            result = {
                'threshold': threshold,
                'w_wyckoff': w_wyck,
                'w_smc': w_smc,
                'w_hob': w_hob,
                'w_momentum': w_mom,
                **metrics
            }
            results.append(result)

            print(f"   ✅ Trades: {metrics['trades']}, "
                  f"WR: {metrics['win_rate']:.1f}%, "
                  f"Return: {metrics['total_return']:+.1f}%, "
                  f"Sharpe: {metrics['sharpe']:.2f}, "
                  f"PF: {metrics['profit_factor']:.2f}")
        else:
            print(f"   ⚠️  No trades or parse error")

        print()

    # Save results
    if len(results) > 0:
        df = pd.DataFrame(results)
        df.to_csv('hybrid_sweep_results.csv', index=False)

        # Print top configs
        print("\n" + "=" * 70)
        print("🏆 TOP 10 CONFIGURATIONS (by Sharpe Ratio)")
        print("=" * 70)

        top_10 = df.nlargest(10, 'sharpe')
        print(top_10[['threshold', 'w_wyckoff', 'w_momentum', 'trades', 'win_rate', 'total_return', 'sharpe', 'profit_factor']].to_string(index=False))

        print("\n" + "=" * 70)
        print("🎯 TOP 10 CONFIGURATIONS (by Total Return)")
        print("=" * 70)

        top_return = df.nlargest(10, 'total_return')
        print(top_return[['threshold', 'w_wyckoff', 'w_momentum', 'trades', 'win_rate', 'total_return', 'sharpe', 'profit_factor']].to_string(index=False))

        print(f"\n💾 Full results ({len(df)} configs) saved to: hybrid_sweep_results.csv")
        print("\n✅ Sweep complete!")
    else:
        print("\n❌ No successful results - check hybrid_runner output")

if __name__ == '__main__':
    main()
