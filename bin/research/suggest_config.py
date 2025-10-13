#!/usr/bin/env python3
"""
Config Suggestion CLI for Bull Machine v1.8.6
=============================================

Use trained ML model to suggest optimal configs based on current regime.

Usage:
    python bin/research/suggest_config.py --model models/sharpe_model --asset BTC --top-n 5

Example:
    # Suggest top 5 configs for BTC in current regime
    python bin/research/suggest_config.py --model models/sharpe_model --asset BTC --top-n 5

    # Suggest with custom macro snapshot
    python bin/research/suggest_config.py --model models/sharpe_model --asset BTC --vix 25.0 --dxy 103.5
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from engine.ml.models import ConfigSuggestionModel, rank_configs_by_prediction
from engine.ml.featurize import build_regime_vector


def generate_candidate_configs(mode: str = 'grid') -> list:
    """
    Generate candidate configs to evaluate.

    Args:
        mode: 'grid' (broad sweep) or 'targeted' (focused on best ranges)

    Returns:
        List of config dicts
    """
    configs = []

    if mode == 'grid':
        # Broad grid sweep
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75]
        stop_atrs = [0.8, 1.0, 1.2, 1.5]
        trail_atrs = [1.0, 1.2, 1.5]
        adx_thresholds = [18, 20, 22, 25]

    else:  # targeted
        # Focused on historically successful ranges
        thresholds = [0.65, 0.68, 0.70]
        stop_atrs = [1.0, 1.2]
        trail_atrs = [1.2, 1.4]
        adx_thresholds = [20, 22]

    # Generate all combinations
    for threshold in thresholds:
        for stop_atr in stop_atrs:
            for trail_atr in trail_atrs:
                for adx_threshold in adx_thresholds:
                    config = {
                        'fusion': {
                            'entry_threshold_confidence': threshold,
                            'weights': {
                                'wyckoff': 0.30,
                                'smc': 0.15,
                                'liquidity': 0.25,
                                'momentum': 0.30
                            }
                        },
                        'exits': {
                            'atr_k': stop_atr,
                            'trail_atr_k': trail_atr,
                            'tp1_r': 1.0,
                            'tp1_pct': 0.5,
                            'move_sl_to_be_on_tp1': True,
                            'trail_after_tp1': True
                        },
                        'risk': {
                            'base_risk_pct': 0.0075
                        },
                        'fast_signals': {
                            'adx_threshold': adx_threshold
                        }
                    }
                    configs.append(config)

    print(f"📋 Generated {len(configs)} candidate configs ({mode} mode)")
    return configs


def build_macro_snapshot_from_args(args) -> dict:
    """
    Build macro snapshot from CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Macro snapshot dict
    """
    snapshot = {
        'VIX': {'value': args.vix, 'stale': False},
        'MOVE': {'value': args.move, 'stale': False},
        'DXY': {'value': args.dxy, 'stale': False},
        'WTI': {'value': args.oil, 'stale': False},
        'GOLD': {'value': args.gold, 'stale': False},
        'US2Y': {'value': args.us2y, 'stale': False},
        'US10Y': {'value': args.us10y, 'stale': False},
        'TOTAL': {'value': np.nan, 'stale': True},  # Not specified
        'TOTAL2': {'value': np.nan, 'stale': True},
        'TOTAL3': {'value': np.nan, 'stale': True},
        'USDT.D': {'value': np.nan, 'stale': True},
        'BTC.D': {'value': 0.55, 'stale': False}
    }

    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Suggest optimal configs using ML model")

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (without extension)')
    parser.add_argument('--asset', type=str, default='BTC',
                        help='Asset to optimize for (default: BTC)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top configs to suggest (default: 5)')
    parser.add_argument('--mode', type=str, default='targeted', choices=['grid', 'targeted'],
                        help='Config generation mode (default: targeted)')

    # Macro snapshot (current regime)
    parser.add_argument('--vix', type=float, default=20.0,
                        help='VIX level (default: 20.0)')
    parser.add_argument('--move', type=float, default=80.0,
                        help='MOVE level (default: 80.0)')
    parser.add_argument('--dxy', type=float, default=100.0,
                        help='DXY level (default: 100.0)')
    parser.add_argument('--oil', type=float, default=70.0,
                        help='Oil (WTI) level (default: 70.0)')
    parser.add_argument('--gold', type=float, default=2500.0,
                        help='Gold level (default: 2500.0)')
    parser.add_argument('--us2y', type=float, default=4.0,
                        help='US 2Y yield (default: 4.0)')
    parser.add_argument('--us10y', type=float, default=4.0,
                        help='US 10Y yield (default: 4.0)')

    # Output options
    parser.add_argument('--output', type=str, default=None,
                        help='Save suggested config to JSON file')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed regime analysis')

    args = parser.parse_args()

    print("=" * 80)
    print("🔮 Bull Machine v1.8.6 - Config Suggestion")
    print("=" * 80)
    print()

    # Load model
    print(f"📂 Loading model from {args.model}...")
    try:
        model = ConfigSuggestionModel.load(args.model)
        print(f"   Model type: {model.model_type}")
        print(f"   Target: {model.target}")
        print()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\nHave you trained a model yet?")
        print("   python bin/research/train_ml.py --target sharpe")
        return 1

    # Build current regime snapshot
    print("🌍 Current Regime:")
    macro_snapshot = build_macro_snapshot_from_args(args)
    regime_vector = build_regime_vector(macro_snapshot, lookback_window=None)

    print(f"   VIX: {args.vix:.1f}")
    print(f"   MOVE: {args.move:.1f}")
    print(f"   DXY: {args.dxy:.1f}")
    print(f"   Oil: {args.oil:.1f}")
    print(f"   Gold: {args.gold:.0f}")
    print(f"   US 2Y: {args.us2y:.2f}%")
    print(f"   US 10Y: {args.us10y:.2f}%")
    print(f"   Yield Spread: {regime_vector['features']['yield_spread']:.2f}%")
    print()

    if args.detailed:
        print("📊 Regime Classification:")
        vix_regime = ['Calm (<18)', 'Elevated (18-30)', 'Panic (>30)'][regime_vector['features']['vix_regime']]
        dxy_regime = ['Weak (<100)', 'Neutral (100-105)', 'Strong (>105)'][regime_vector['features']['dxy_regime']]
        curve_regime = ['Inverted', 'Flat', 'Steep'][regime_vector['features']['curve_regime']]
        print(f"   VIX Regime: {vix_regime}")
        print(f"   DXY Regime: {dxy_regime}")
        print(f"   Curve Regime: {curve_regime}")
        print()

    # Generate candidate configs
    print(f"🔧 Generating candidate configs ({args.mode} mode)...")
    candidate_configs = generate_candidate_configs(mode=args.mode)
    print()

    # Rank configs
    print(f"🤖 Ranking configs using {model.target} predictor...")
    top_configs = rank_configs_by_prediction(
        model,
        candidate_configs,
        regime_vector,
        top_n=args.top_n
    )
    print()

    # Display results
    print("=" * 80)
    print(f"✅ TOP {args.top_n} RECOMMENDED CONFIGS")
    print("=" * 80)
    print()

    for i, (config, predicted_score) in enumerate(top_configs, 1):
        print(f"Rank {i}: Predicted {model.target} = {predicted_score:.3f}")
        print("-" * 40)
        print(f"  Fusion Threshold: {config['fusion']['entry_threshold_confidence']:.2f}")
        print(f"  Stop ATR: {config['exits']['atr_k']:.2f}")
        print(f"  Trail ATR: {config['exits']['trail_atr_k']:.2f}")
        print(f"  ADX Threshold: {config['fast_signals']['adx_threshold']}")
        print(f"  Base Risk: {config['risk']['base_risk_pct']:.4f} ({config['risk']['base_risk_pct']*100:.2f}%)")
        print()

    # Save top config if requested
    if args.output:
        best_config = top_configs[0][0]
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        full_config = {
            'version': '1.8.6',
            'asset': args.asset,
            'profile': 'ml_suggested',
            'ml_metadata': {
                'model': str(args.model),
                'target': model.target,
                'predicted_score': float(top_configs[0][1]),
                'regime': {
                    'vix': args.vix,
                    'move': args.move,
                    'dxy': args.dxy,
                    'oil': args.oil,
                    'gold': args.gold,
                    'us2y': args.us2y,
                    'us10y': args.us10y
                }
            }
        }
        full_config.update(best_config)

        with open(output_path, 'w') as f:
            json.dump(full_config, f, indent=2)

        print(f"💾 Saved best config to: {output_path}")
        print()

    print("Next steps:")
    print(f"  1. Backtest suggested config:")
    print(f"     python bin/optimize_v19.py --config {args.output if args.output else 'configs/v18/BTC_live.json'} --years 2")
    print()
    print(f"  2. Paper trade with suggested config:")
    print(f"     python bin/bull-live-paper --config {args.output if args.output else 'configs/v18/BTC_live.json'} --balance 25000")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
