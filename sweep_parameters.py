#!/usr/bin/env python3
"""
Simple Parameter Sweep using existing btc_simple_backtest.py

Tests different entry thresholds, ADX levels, and risk management
on 1-2 years of BTC data. Runs overnight, results by morning.

Usage:
    python sweep_parameters.py --asset BTC --configs 20
"""

import subprocess
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import time

# Parameter grid to test
PARAM_GRID = {
    'entry_threshold': [0.35, 0.40, 0.45, 0.50],
    'adx_threshold': [15, 20, 25],
    'stop_loss_pct': [0.06, 0.08, 0.10],
    'take_profit_pct': [0.12, 0.15, 0.18],
    'risk_pct': [0.05, 0.075]
}

def generate_configs(mode='quick'):
    """Generate parameter combinations to test"""
    configs = []

    if mode == 'quick':
        # 12 configs (~1 hour total)
        for entry in [0.40, 0.45, 0.50]:
            for adx in [20, 25]:
                for stop in [0.08]:
                    for tp in [0.15]:
                        for risk in [0.05]:
                            configs.append({
                                'entry_threshold': entry,
                                'adx_threshold': adx,
                                'stop_loss_pct': stop,
                                'take_profit_pct': tp,
                                'risk_pct': risk
                            })

    elif mode == 'standard':
        # ~50 configs (~4 hours total)
        for entry in PARAM_GRID['entry_threshold']:
            for adx in PARAM_GRID['adx_threshold']:
                for stop in PARAM_GRID['stop_loss_pct']:
                    if entry >= 0.45:  # Higher threshold = wider stops
                        tp_options = [0.15, 0.18]
                    else:
                        tp_options = [0.12, 0.15]

                    for tp in tp_options:
                        for risk in [0.05]:  # Fixed risk
                            configs.append({
                                'entry_threshold': entry,
                                'adx_threshold': adx,
                                'stop_loss_pct': stop,
                                'take_profit_pct': tp,
                                'risk_pct': risk
                            })

    print(f"Generated {len(configs)} configurations ({mode} mode)")
    return configs

def run_backtest_with_params(asset, params):
    """
    Modify btc_simple_backtest.py with params and run it.
    Returns results dict.
    """
    # For simplicity, we'll pass params as a JSON config file
    # and modify the backtest script to read from config

    # Create temp config
    config_path = f"temp_config_{int(time.time())}.json"
    with open(config_path, 'w') as f:
        json.dump(params, f)

    try:
        # Run backtest (assumes script prints JSON results)
        result = subprocess.run(
            ['python3', 'btc_simple_backtest.py', '--config', config_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 min timeout
        )

        # Parse output (assuming last line is JSON)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in reversed(lines):
                if line.startswith('{'):
                    return json.loads(line)

        return {'error': result.stderr}

    except subprocess.TimeoutExpired:
        return {'error': 'Timeout'}
    except Exception as e:
        return {'error': str(e)}
    finally:
        # Cleanup
        if Path(config_path).exists():
            Path(config_path).unlink()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset', default='BTC')
    parser.add_argument('--mode', choices=['quick', 'standard'], default='quick')
    parser.add_argument('--output', default='sweep_results.json')
    args = parser.parse_args()

    print(f"🎯 Bull Machine Parameter Sweep")
    print(f"Asset: {args.asset}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")

    configs = generate_configs(args.mode)

    results = []
    start_time = time.time()

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Testing: entry={config['entry_threshold']}, "
              f"adx={config['adx_threshold']}, stop={config['stop_loss_pct']}, "
              f"tp={config['take_profit_pct']}")

        result = run_backtest_with_params(args.asset, config)
        result.update(config)  # Add params to result
        results.append(result)

        if 'error' not in result:
            trades = result.get('trades', 0)
            ret = result.get('return', 0)
            wr = result.get('win_rate', 0)
            print(f"   → {trades} trades, {ret:+.2f}% return, {wr:.1f}% WR")
        else:
            print(f"   ❌ Error: {result['error']}")

        # Save intermediate results
        if i % 5 == 0:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    elapsed = time.time() - start_time

    # Final save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Analysis
    df = pd.DataFrame(results)
    df = df[df.get('error').isna()]  # Filter errors

    if len(df) > 0:
        df = df.sort_values('return', ascending=False)

        print(f"\n{'='*60}")
        print("TOP 5 CONFIGURATIONS")
        print(f"{'='*60}")
        print(df[['entry_threshold', 'adx_threshold', 'stop_loss_pct', 'take_profit_pct',
                  'trades', 'win_rate', 'return']].head(5).to_string(index=False))

        print(f"\n💾 Full results saved to: {args.output}")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    else:
        print("\n❌ No successful runs")

if __name__ == '__main__':
    main()
