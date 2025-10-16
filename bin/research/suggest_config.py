#!/usr/bin/env python3
"""
Config Suggestion CLI for Bull Machine v1.8.6 - WITH DIRICHLET SAMPLING
=======================================================================

Use trained ML model to suggest optimal configs based on current regime.

Features:
- Dirichlet sampling for domain weights (ensures they sum to 1.0)
- Threshold grid search (0.45 to 0.65)
- MaxDD guard (discard configs with predicted DD > 15%)
- Trade count filter (>= 100 trades)
- Top-N ranking by predicted PF/Sharpe

Usage:
    python bin/research/suggest_config.py --model models/sharpe_model --top-n 5

Example:
    # Suggest top 10 configs with 1000 weight samples
    python bin/research/suggest_config.py \
      --model models/sharpe_model \
      --n-weights 1000 \
      --top-n 10 \
      --output suggested_configs.json
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


def sample_dirichlet_weights(n_samples: int = 1000, alpha: float = 5.0) -> np.ndarray:
    """
    Sample domain weights using Dirichlet distribution.

    Dirichlet ensures weights are positive and sum to 1.0.

    Args:
        n_samples: Number of weight vectors to sample
        alpha: Concentration parameter (higher = more uniform, lower = more peaked)
               alpha=5.0 gives reasonable diversity around [0.25, 0.25, 0.25, 0.25]

    Returns:
        (n_samples, 4) array of [wyckoff, smc, hob, momentum] weights
    """
    # 4 domains: Wyckoff, SMC, HOB, Momentum
    alphas = np.array([alpha, alpha, alpha, alpha])

    # Sample from Dirichlet
    samples = np.random.dirichlet(alphas, size=n_samples)

    return samples


def generate_candidate_configs_dirichlet(
    thresholds: list,
    n_weight_samples: int = 1000,
    alpha: float = 5.0
) -> list:
    """
    Generate candidate configs using Dirichlet-sampled weights.

    Args:
        thresholds: List of fusion thresholds to test (e.g., [0.45, 0.50, 0.55, 0.60, 0.65])
        n_weight_samples: Number of weight vectors to sample
        alpha: Dirichlet concentration (5.0 = moderate diversity)

    Returns:
        List of config dicts
    """
    configs = []

    # Sample weights once
    weight_samples = sample_dirichlet_weights(n_weight_samples, alpha=alpha)

    for threshold in thresholds:
        for weights in weight_samples:
            config = {
                'fusion': {
                    'entry_threshold_confidence': threshold,
                    'weights': {
                        'wyckoff': float(weights[0]),
                        'smc': float(weights[1]),
                        'liquidity': float(weights[2]),  # HOB
                        'momentum': float(weights[3])
                    }
                },
                'exits': {
                    'atr_k': 1.0,
                    'trail_atr_k': 1.2,
                    'tp1_r': 1.0,
                    'tp1_pct': 0.5,
                    'move_sl_to_be_on_tp1': True,
                    'trail_after_tp1': True
                },
                'risk': {
                    'base_risk_pct': 0.0075
                },
                'fast_signals': {
                    'adx_threshold': 20
                }
            }
            configs.append(config)

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
        'TOTAL': {'value': np.nan, 'stale': True},
        'TOTAL2': {'value': np.nan, 'stale': True},
        'TOTAL3': {'value': np.nan, 'stale': True},
        'USDT.D': {'value': np.nan, 'stale': True},
        'BTC.D': {'value': 0.55, 'stale': False}
    }

    return snapshot


def main():
    parser = argparse.ArgumentParser(description="Suggest optimal configs using ML model + Dirichlet weights")

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (without extension)')
    parser.add_argument('--asset', type=str, default='BTC',
                        help='Asset to optimize for (default: BTC)')
    parser.add_argument('--top-n', type=int, default=10,
                        help='Number of top configs to suggest (default: 10)')

    # Config generation options
    parser.add_argument('--thresholds', type=str, default='0.45,0.48,0.50,0.52,0.55,0.58,0.60,0.62,0.65',
                        help='Comma-separated fusion thresholds (default: 0.45 to 0.65)')
    parser.add_argument('--n-weights', type=int, default=1000,
                        help='Number of Dirichlet weight samples (default: 1000)')
    parser.add_argument('--alpha', type=float, default=5.0,
                        help='Dirichlet concentration parameter (default: 5.0)')

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
                        help='Save suggested configs to JSON file')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed regime analysis')

    args = parser.parse_args()

    print("=" * 80)
    print("🔮 Bull Machine v1.8.6 - Config Suggestion (Dirichlet Sampling)")
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
    print(f"🔧 Generating candidate configs (Dirichlet sampling)...")
    thresholds = [float(t) for t in args.thresholds.split(',')]
    print(f"   Thresholds: {thresholds}")
    print(f"   Weight samples: {args.n_weights}")
    print(f"   Alpha: {args.alpha}")

    candidate_configs = generate_candidate_configs_dirichlet(
        thresholds=thresholds,
        n_weight_samples=args.n_weights,
        alpha=args.alpha
    )
    print(f"   ✅ Generated {len(candidate_configs)} candidates")
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
        weights = config['fusion']['weights']
        print(f"Rank {i}: Predicted {model.target} = {predicted_score:.3f}")
        print("-" * 40)
        print(f"  Fusion Threshold: {config['fusion']['entry_threshold_confidence']:.2f}")
        print(f"  Weights:")
        print(f"    Wyckoff:  {weights['wyckoff']:.3f}")
        print(f"    SMC:      {weights['smc']:.3f}")
        print(f"    HOB:      {weights['liquidity']:.3f}")
        print(f"    Momentum: {weights['momentum']:.3f}")
        print(f"  Stop ATR: {config['exits']['atr_k']:.2f}")
        print(f"  Trail ATR: {config['exits']['trail_atr_k']:.2f}")
        print()

    # Save configs if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save all top configs (not just best)
        configs_to_save = []
        for rank, (config, score) in enumerate(top_configs, 1):
            config_with_meta = config.copy()
            config_with_meta['version'] = '1.8.6'
            config_with_meta['asset'] = args.asset
            config_with_meta['profile'] = f'ml_suggested_rank{rank}'
            config_with_meta['ml_metadata'] = {
                'model': str(args.model),
                'target': model.target,
                'predicted_score': float(score),
                'rank': rank,
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
            configs_to_save.append(config_with_meta)

        with open(output_path, 'w') as f:
            json.dump(configs_to_save, f, indent=2)

        print(f"💾 Saved {len(configs_to_save)} configs to: {output_path}")
        print()

    print("Next steps:")
    print(f"  1. Verify top configs with backtest:")
    if args.output:
        print(f"     python bin/optimize_v19.py --mode verify --configs {args.output}")
    print()
    print(f"  2. Paper trade best config:")
    print(f"     python bin/bull-live-paper --config <best_config.json> --balance 25000")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
