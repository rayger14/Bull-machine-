#!/usr/bin/env python3
"""
Config Suggestion Tool - Phase 2 Meta-Optimizer

Uses trained config optimizer model + Bayesian optimization to suggest
promising configurations without running full backtests.

This is an intelligent config search that learns from historical trials
to predict which parameter combinations are likely to yield high PF.

Usage:
    # Suggest 10 configs based on learned landscape
    python3 bin/suggest_configs.py \
        --model models/btc_config_optimizer_v1.pkl \
        --n-suggestions 10 \
        --output reports/ml/suggested_configs.json \
        --min-pf 9.0

    # Then validate on 2024
    python3 bin/backtest_knowledge_v2.py \
        --asset BTC --start 2024-01-01 --end 2024-12-31 \
        --config reports/ml/suggested_configs_001.json
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import json
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')


# Config parameter bounds (from Optuna trials)
PARAM_BOUNDS = {
    'final_fusion_floor': (0.20, 0.45),
    'neutralize_fusion_drop': (0.08, 0.20),
    'neutralize_min_bars': (3, 10),
    'neutralize_pti_margin': (0.15, 0.35),
    'min_liquidity': (0.08, 0.20),
    'w_wyckoff': (0.25, 0.50),
    'w_liquidity': (0.10, 0.30),
    'w_momentum': (0.30, 0.55),
    'size_min': (0.50, 0.90),
    'size_max': (1.05, 1.40),
    'B_fusion': (0.28, 0.40),
    'C_fusion': (0.40, 0.60),
    'H_fusion': (0.50, 0.70),
    'K_fusion': (0.38, 0.55),
    'L_fusion': (0.32, 0.48),
    'trail_atr_mult': (1.0, 1.5),
    'max_bars': (60, 96),
    'range_stop_factor': (0.70, 0.95),
    'trend_stop_factor': (1.15, 1.40)
}


def load_model(model_path: str):
    """Load trained config optimizer model"""
    print(f"Loading config optimizer from {model_path}")
    model_data = joblib.load(model_path)

    model = model_data['model']
    feature_names = model_data['feature_names']
    metrics = model_data.get('metrics', {})

    print(f"  Features: {len(feature_names)}")
    print(f"  Test R²: {metrics.get('test_r2', 'N/A')}")
    print(f"  Test MAE: {metrics.get('test_mae', 'N/A')}")

    return model, feature_names, metrics


def config_to_features(config_dict: dict, feature_names: list) -> np.ndarray:
    """Convert config dict to feature vector"""
    features = []
    for fname in feature_names:
        features.append(config_dict.get(fname, 0.0))
    return np.array(features).reshape(1, -1)


def features_to_config(features: np.ndarray, feature_names: list) -> dict:
    """Convert feature vector to config dict"""
    return {fname: float(features[i]) for i, fname in enumerate(feature_names)}


def objective_function(x, model, feature_names):
    """Objective function for optimization (negative because we minimize)"""
    config_dict = features_to_config(x, feature_names)
    X = config_to_features(config_dict, feature_names)
    predicted_pf = model.predict(X)[0]
    return -predicted_pf  # Negative because differential_evolution minimizes


def suggest_configs_differential_evolution(model, feature_names, n_suggestions=10, random_state=42):
    """Use Differential Evolution to find optimal configs"""
    print(f"\n{'='*60}")
    print(f"SUGGESTING CONFIGS VIA DIFFERENTIAL EVOLUTION")
    print(f"{'='*60}\n")

    # Extract bounds for features in order
    bounds = []
    for fname in feature_names:
        if fname in PARAM_BOUNDS:
            bounds.append(PARAM_BOUNDS[fname])
        else:
            # Unknown feature - use default range
            bounds.append((0.0, 1.0))

    suggestions = []

    for i in range(n_suggestions):
        print(f"Optimizing config {i+1}/{n_suggestions}...", end=' ')

        # Use differential evolution to find optimal config
        result = differential_evolution(
            objective_function,
            bounds,
            args=(model, feature_names),
            seed=random_state + i,
            maxiter=100,
            popsize=15,
            atol=0.01,
            tol=0.01,
            workers=1
        )

        optimal_features = result.x
        predicted_pf = -result.fun  # Negate back to positive PF

        config_dict = features_to_config(optimal_features, feature_names)

        # Normalize fusion weights to sum to ~1.0
        fusion_keys = ['w_wyckoff', 'w_liquidity', 'w_momentum']
        if all(k in config_dict for k in fusion_keys):
            total_weight = sum(config_dict[k] for k in fusion_keys)
            for k in fusion_keys:
                config_dict[k] = config_dict[k] / total_weight

        suggestions.append({
            'config': config_dict,
            'predicted_pf': float(predicted_pf),
            'suggestion_id': i + 1
        })

        print(f"Predicted PF: {predicted_pf:.2f}")

    # Sort by predicted PF descending
    suggestions.sort(key=lambda x: x['predicted_pf'], reverse=True)

    return suggestions


def suggest_configs_random_search(model, feature_names, n_suggestions=10, random_state=42):
    """Baseline: Random search within param bounds"""
    print(f"\n{'='*60}")
    print(f"BASELINE: RANDOM SEARCH")
    print(f"{'='*60}\n")

    np.random.seed(random_state)

    suggestions = []

    for i in range(n_suggestions):
        config_dict = {}

        for fname in feature_names:
            if fname in PARAM_BOUNDS:
                low, high = PARAM_BOUNDS[fname]
                config_dict[fname] = np.random.uniform(low, high)
            else:
                config_dict[fname] = np.random.uniform(0.0, 1.0)

        # Normalize fusion weights
        fusion_keys = ['w_wyckoff', 'w_liquidity', 'w_momentum']
        if all(k in config_dict for k in fusion_keys):
            total_weight = sum(config_dict[k] for k in fusion_keys)
            for k in fusion_keys:
                config_dict[k] = config_dict[k] / total_weight

        X = config_to_features(config_dict, feature_names)
        predicted_pf = model.predict(X)[0]

        suggestions.append({
            'config': config_dict,
            'predicted_pf': float(predicted_pf),
            'suggestion_id': i + 1
        })

    suggestions.sort(key=lambda x: x['predicted_pf'], reverse=True)

    return suggestions


def export_to_json_configs(suggestions: list, output_dir: Path, base_config_path: str = None):
    """Export each suggestion as a standalone config JSON"""
    print(f"\n{'='*60}")
    print(f"EXPORTING CONFIG FILES")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base config if provided
    base_config = {}
    if base_config_path:
        with open(base_config_path, 'r') as f:
            base_config = json.load(f)

    exported_paths = []

    for suggestion in suggestions:
        sid = suggestion['suggestion_id']
        config = suggestion['config']
        predicted_pf = suggestion['predicted_pf']

        # Merge with base config
        final_config = base_config.copy() if base_config else {}

        # Update fusion weights
        if 'fusion' not in final_config:
            final_config['fusion'] = {}
        final_config['fusion']['weights'] = {
            'wyckoff': config.get('w_wyckoff', 0.38),
            'liquidity': config.get('w_liquidity', 0.19),
            'momentum': config.get('w_momentum', 0.43),
            'smc': 0.0
        }

        # Update archetype thresholds
        if 'archetypes' not in final_config:
            final_config['archetypes'] = {'thresholds': {}}
        if 'thresholds' not in final_config['archetypes']:
            final_config['archetypes']['thresholds'] = {}

        final_config['archetypes']['thresholds']['min_liquidity'] = config.get('min_liquidity', 0.12)
        final_config['archetypes']['thresholds']['B'] = {'fusion': config.get('B_fusion', 0.32)}
        final_config['archetypes']['thresholds']['C'] = {'fusion': config.get('C_fusion', 0.50)}
        final_config['archetypes']['thresholds']['H'] = {'fusion': config.get('H_fusion', 0.56)}
        final_config['archetypes']['thresholds']['K'] = {'fusion': config.get('K_fusion', 0.44)}
        final_config['archetypes']['thresholds']['L'] = {'fusion': config.get('L_fusion', 0.38)}

        # Update neutralization params
        if 'neutralization' not in final_config:
            final_config['neutralization'] = {}
        final_config['neutralization']['fusion_drop_threshold'] = config.get('neutralize_fusion_drop', 0.13)
        final_config['neutralization']['min_bars_since_entry'] = int(config.get('neutralize_min_bars', 6))
        final_config['neutralization']['pti_margin'] = config.get('neutralize_pti_margin', 0.25)

        # Update exit params
        if 'pnl_tracker' not in final_config:
            final_config['pnl_tracker'] = {'exits': {}}
        if 'exits' not in final_config['pnl_tracker']:
            final_config['pnl_tracker']['exits'] = {}

        final_config['pnl_tracker']['exits']['trail_atr_mult'] = config.get('trail_atr_mult', 1.2)
        final_config['pnl_tracker']['exits']['max_bars_in_trade'] = int(config.get('max_bars', 78))
        final_config['pnl_tracker']['exits']['range_stop_factor'] = config.get('range_stop_factor', 0.80)
        final_config['pnl_tracker']['exits']['trend_stop_factor'] = config.get('trend_stop_factor', 1.25)

        # Update decision_gates sizing
        if 'decision_gates' not in final_config:
            final_config['decision_gates'] = {}
        final_config['decision_gates']['size_min'] = config.get('size_min', 0.75)
        final_config['decision_gates']['size_max'] = config.get('size_max', 1.35)

        # Add metadata
        final_config['version'] = f'2.0.0-ml-suggested-{sid:03d}'
        final_config['description'] = f'ML-suggested config (predicted PF: {predicted_pf:.2f})'
        final_config['ml_metadata'] = {
            'suggested_by': 'config_optimizer_v1',
            'predicted_pf': predicted_pf,
            'suggestion_id': sid
        }

        # Export
        output_path = output_dir / f'suggested_config_{sid:03d}.json'
        with open(output_path, 'w') as f:
            json.dump(final_config, f, indent=2)

        exported_paths.append(str(output_path))
        print(f"  Config {sid:03d}: {output_path.name} (predicted PF: {predicted_pf:.2f})")

    return exported_paths


def main():
    parser = argparse.ArgumentParser(description='Suggest optimal configs using trained model')
    parser.add_argument('--model', required=True, help='Path to trained config optimizer (.pkl)')
    parser.add_argument('--n-suggestions', type=int, default=10, help='Number of configs to suggest')
    parser.add_argument('--output', required=True, help='Output directory for config JSONs')
    parser.add_argument('--base-config', type=str, default=None,
                        help='Base config to merge suggestions into (optional)')
    parser.add_argument('--method', type=str, default='differential_evolution',
                        choices=['differential_evolution', 'random'],
                        help='Optimization method')
    parser.add_argument('--min-pf', type=float, default=None,
                        help='Filter suggestions below this predicted PF')

    args = parser.parse_args()

    # Load model
    model, feature_names, metrics = load_model(args.model)

    # Check if model has reasonable performance
    test_r2 = metrics.get('test_r2', -999)
    if test_r2 < -0.5:
        print(f"\n⚠️  WARNING: Model has poor generalization (Test R² = {test_r2:.3f})")
        print("  Suggestions may not be reliable. Consider:")
        print("    1. Collecting more training data (200+ trials)")
        print("    2. Using feature selection to reduce dimensions")
        print("    3. Using simpler models (linear regression)")
        print("\n  Proceeding anyway for proof-of-concept...\n")

    # Generate suggestions
    if args.method == 'differential_evolution':
        suggestions = suggest_configs_differential_evolution(
            model, feature_names, args.n_suggestions
        )
    elif args.method == 'random':
        suggestions = suggest_configs_random_search(
            model, feature_names, args.n_suggestions
        )

    # Filter by min PF if specified
    if args.min_pf:
        suggestions = [s for s in suggestions if s['predicted_pf'] >= args.min_pf]
        print(f"\nFiltered to {len(suggestions)} suggestions with predicted PF >= {args.min_pf}")

    # Export to JSON config files
    output_dir = Path(args.output)
    exported_paths = export_to_json_configs(suggestions, output_dir, args.base_config)

    print(f"\n{'='*60}")
    print("CONFIG SUGGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {len(exported_paths)} config files in {output_dir}")
    print(f"\nTop 3 predicted configs:")
    for i, s in enumerate(suggestions[:3]):
        print(f"  {i+1}. Config {s['suggestion_id']:03d}: Predicted PF = {s['predicted_pf']:.2f}")

    print(f"\nNext step: Validate on 2024")
    print(f"  python3 bin/backtest_knowledge_v2.py \\")
    print(f"    --asset BTC --start 2024-01-01 --end 2024-12-31 \\")
    print(f"    --config {exported_paths[0]}")


if __name__ == '__main__':
    main()
