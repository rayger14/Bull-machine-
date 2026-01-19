#!/usr/bin/env python3
"""
Bull Machine v1.8.6 - Production-Faithful Optimizer v19

Uses pre-built feature store and Smart Exits simulator to match
production behavior exactly while maintaining speed.

Key improvements over v18:
- Feature store with causal computation (no future leak)
- Smart Exits matching SmartExitPortfolio exactly
- Dynamic position sizing with ADX/fusion adaptation
- Macro veto and macro exit integration
- TP1 partials + trailing stops + BE moves

Speed: ~10-20 seconds per 100 configs on 18 months of data
Accuracy: Matches hybrid_runner.py within ±2%
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import argparse
import time
import multiprocessing as mp
from functools import partial

from smart_exits_sim import simulate_trades_batch, calculate_metrics

# Phase 2: Regime adaptation
from engine.context.regime_classifier import RegimeClassifier
from engine.context.regime_policy import RegimePolicy

# ML dataset logging
try:
    from engine.ml.dataset import OptimizationDataset
    from engine.ml.featurize import build_regime_vector, build_training_row
    ML_LOGGING_AVAILABLE = True
except ImportError:
    ML_LOGGING_AVAILABLE = False
    print("⚠️  ML logging not available (engine.ml not found)")


def load_feature_store(path: str) -> Dict[str, np.ndarray]:
    """Load pre-computed feature store"""
    df = pd.read_parquet(path)

    # Validate no NaN values (all features computed causally)
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        print(f"⚠️  Warning: NaN values in columns: {nan_cols}")
        df = df.fillna(method='ffill').fillna(0.5)  # Forward fill, then neutral

    return {
        'close': df['close'].to_numpy(),
        'high': df['high'].to_numpy(),
        'low': df['low'].to_numpy(),
        'atr_20': df['atr_20'].to_numpy(),
        'atr_14': df['atr_14'].to_numpy(),
        'adx_14': df['adx_14'].to_numpy(),
        'rsi_14': df['rsi_14'].to_numpy(),
        'wyckoff': df['wyckoff'].to_numpy(),
        'smc': df['smc'].to_numpy(),
        'hob': df['hob'].to_numpy(),
        'momentum': df['momentum'].to_numpy(),
        'temporal': df['temporal'].to_numpy(),
        'mtf_align': df['mtf_align'].to_numpy(dtype=bool),
        'macro_veto': df['macro_veto'].to_numpy(dtype=bool),
        'macro_exit_flag': df['macro_exit_flag'].to_numpy(dtype=bool),
        'timestamp': df.index.to_numpy()
    }


def compute_fusion_scores(features: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """
    Compute fusion scores for all bars using given weights

    Vectorized operation - very fast!
    """
    X = np.stack([
        features['wyckoff'],
        features['smc'],
        features['hob'],
        features['momentum'],
        features['temporal']
    ], axis=1).astype('float32')

    w = np.array([
        weights['wyckoff'],
        weights['smc'],
        weights['hob'],
        weights['momentum'],
        weights.get('temporal', 0.0)
    ], dtype='float32')

    # Normalize weights
    w = w / w.sum()

    return X @ w


def apply_cooldown(entry_bool: np.ndarray, cooldown_bars: int = 8) -> np.ndarray:
    """
    Apply cooldown period after each entry to prevent immediate re-entry

    Args:
        entry_bool: Boolean array of potential entries
        cooldown_bars: Bars to wait before allowing new entry

    Returns:
        Filtered entry array with cooldown applied
    """
    keep = np.zeros_like(entry_bool, dtype=bool)
    next_unlock = 0

    for t in range(len(entry_bool)):
        if entry_bool[t] and t >= next_unlock:
            keep[t] = True
            next_unlock = t + cooldown_bars

    return keep


def generate_entry_indices(
    features: Dict[str, np.ndarray],
    fusion_scores: np.ndarray,
    threshold: float,
    cooldown: int = 8
) -> np.ndarray:
    """
    Generate entry indices with all filters applied

    Filters:
    - Fusion score >= threshold
    - MTF alignment = True
    - Macro veto = False
    - Cooldown period enforced
    """
    raw_entries = (
        (fusion_scores >= threshold) &
        features['mtf_align'] &
        (~features['macro_veto'])
    )

    # Apply cooldown
    filtered = apply_cooldown(raw_entries, cooldown)

    return np.flatnonzero(filtered)


def backtest_config(
    features: Dict[str, np.ndarray],
    config: Dict,
    regime_classifier=None,
    regime_policy=None,
    asset: str = "BTC"
) -> Dict:
    """
    Run backtest for single configuration

    Args:
        features: Pre-loaded feature store
        config: Configuration dict with weights, threshold, exits
        regime_classifier: Optional RegimeClassifier for Phase 2
        regime_policy: Optional RegimePolicy for Phase 2

    Returns:
        Performance metrics dict
    """
    start_time = time.time()

    # Phase 2: Apply regime adaptation if enabled
    adjusted_config = config.copy()
    regime_label = None
    regime_confidence = None

    if regime_classifier is not None and regime_policy is not None:
        # Build macro snapshot at start of backtest window
        macro_snapshot = build_macro_snapshot(features, start_idx=0, asset=asset)

        # Extract values for regime classification (flatten nested dict)
        macro_row = {k: v['value'] if isinstance(v, dict) else v
                     for k, v in macro_snapshot.items()}

        # Classify regime
        result = regime_classifier.classify(macro_row)
        regime_label = result['regime']
        regime_confidence = result['proba'][regime_label]

        # Apply policy adjustments
        adjustment = regime_policy.apply(config, result)

        # Apply threshold adjustment
        adjusted_config['fusion_threshold'] = np.clip(
            config['fusion_threshold'] + adjustment['enter_threshold_delta'],
            0.45,  # Min threshold
            0.80   # Max threshold
        )

        # Apply weight nudges
        weight_nudges = adjustment.get('weight_nudges', {})
        adjusted_config['wyckoff_weight'] = np.clip(
            config['wyckoff_weight'] + weight_nudges.get('wyckoff', 0.0),
            0.15, 0.45
        )
        adjusted_config['smc_weight'] = np.clip(
            config['smc_weight'] + weight_nudges.get('smc', 0.0),
            0.10, 0.30
        )
        adjusted_config['hob_weight'] = np.clip(
            config['hob_weight'] + weight_nudges.get('liquidity', 0.0),
            0.15, 0.40
        )
        adjusted_config['momentum_weight'] = np.clip(
            config['momentum_weight'] + weight_nudges.get('momentum', 0.0),
            0.20, 0.40
        )

        # Renormalize weights to sum to 1.0
        total = (adjusted_config['wyckoff_weight'] + adjusted_config['smc_weight'] +
                 adjusted_config['hob_weight'] + adjusted_config['momentum_weight'])
        adjusted_config['wyckoff_weight'] /= total
        adjusted_config['smc_weight'] /= total
        adjusted_config['hob_weight'] /= total
        adjusted_config['momentum_weight'] /= total

        # Apply risk multiplier
        base_risk = config.get('base_risk_pct', 0.0075)
        adjusted_config['base_risk_pct'] = base_risk * adjustment['risk_multiplier']

    # Extract config (use adjusted if regime applied)
    weights = {
        'wyckoff': adjusted_config['wyckoff_weight'],
        'smc': adjusted_config['smc_weight'],
        'hob': adjusted_config['hob_weight'],
        'momentum': adjusted_config['momentum_weight'],
        'temporal': 0.0  # Placeholder
    }
    threshold = adjusted_config['fusion_threshold']

    # Compute fusion scores (vectorized - fast!)
    fusion_scores = compute_fusion_scores(features, weights)

    # Generate entries with filters
    entry_indices = generate_entry_indices(features, fusion_scores, threshold)

    if len(entry_indices) == 0:
        result = {
            **config,
            'trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 1.0,
            'max_drawdown': 0.0,
            'avg_r': 0.0,
            'backtest_seconds': time.time() - start_time
        }
        # Add regime tracking if enabled
        if regime_label is not None:
            result['regime_label'] = regime_label
            result['regime_confidence'] = regime_confidence
        return result

    # Get fusion scores at entry points
    entry_fusion_scores = fusion_scores[entry_indices]

    # Simulate trades with Smart Exits
    exit_config = {
        'fusion_threshold': threshold,
        'stop_atr': config.get('stop_atr', 1.0),
        'tp1_r': config.get('tp1_r', 1.0),
        'scale_out_pct': config.get('scale_out_pct', 0.5),
        'move_sl_to_be_on_tp1': config.get('move_sl_to_be_on_tp1', True),
        'trail_after_tp1': config.get('trail_after_tp1', True),
        'trail_atr_mult': config.get('trail_atr_mult', 1.0),
        'adx_trend_hi': config.get('adx_trend_hi', 25.0),
        'adx_range_lo': config.get('adx_range_lo', 20.0),
        'trend_stop_factor': config.get('trend_stop_factor', 1.25),
        'range_stop_factor': config.get('range_stop_factor', 0.75),
        'max_bars_in_trade': config.get('max_bars_in_trade', 96),
        'base_risk_pct': config.get('base_risk_pct', 0.0075),
        'leverage': config.get('leverage', 5.0),
        'fees_bps': config.get('fees_bps', 10.0),
        'slippage_bps': config.get('slippage_bps', 5.0)
    }

    trades = simulate_trades_batch(
        entry_indices,
        features,
        entry_fusion_scores,
        side=1,  # Long only for now
        config=exit_config
    )

    # Calculate metrics
    metrics = calculate_metrics(trades)
    metrics.update(config)
    metrics['backtest_seconds'] = time.time() - start_time

    # Add regime tracking if enabled
    if regime_label is not None:
        metrics['regime_label'] = regime_label
        metrics['regime_confidence'] = regime_confidence

    return metrics


# Global macro data cache
_MACRO_CACHE = {}


def load_macro_data(asset: str) -> pd.DataFrame:
    """
    Load macro feature dataset from parquet (cached)

    Args:
        asset: BTC or ETH

    Returns:
        DataFrame with macro features
    """
    if asset in _MACRO_CACHE:
        return _MACRO_CACHE[asset]

    macro_path = f"data/macro/{asset}_macro_features.parquet"

    try:
        macro_df = pd.read_parquet(macro_path)
        _MACRO_CACHE[asset] = macro_df
        return macro_df
    except Exception as e:
        print(f"⚠️  Failed to load macro data from {macro_path}: {e}")
        return None


def build_macro_snapshot(features: Dict[str, np.ndarray], start_idx: int = 0, asset: str = "BTC") -> Dict:
    """
    Build macro snapshot from macro dataset at given timestamp.

    This captures the regime at the START of the backtest window.

    Args:
        features: Feature store dict (must contain 'timestamp')
        start_idx: Index to sample (default: 0 = start of window)
        asset: BTC or ETH (for loading macro data)

    Returns:
        Macro snapshot dict ready for regime classification
    """
    # Load macro data
    macro_df = load_macro_data(asset)

    if macro_df is None or 'timestamp' not in features:
        # Fallback to synthetic defaults
        return {
            'VIX': {'value': 20.0, 'stale': True},
            'DXY': {'value': 104.0, 'stale': True},
            'MOVE': {'value': 80.0, 'stale': True},
            'YIELD_2Y': {'value': 4.5, 'stale': True},
            'YIELD_10Y': {'value': 4.3, 'stale': True},
            'USDT.D': {'value': 4.5, 'stale': True},
            'BTC.D': {'value': 55.0, 'stale': True},
            'TOTAL': {'value': np.nan, 'stale': True},
            'TOTAL2': {'value': np.nan, 'stale': True},
            'funding': {'value': 0.01, 'stale': True},
            'oi': {'value': np.nan, 'stale': True},
            'rv_20d': {'value': 40.0, 'stale': True},
            'rv_60d': {'value': 45.0, 'stale': True}
        }

    # Get timestamp at start_idx
    target_ts = features['timestamp'][start_idx]

    # Find closest macro record
    macro_df['ts_diff'] = abs((macro_df['timestamp'] - target_ts).dt.total_seconds())
    closest_idx = macro_df['ts_diff'].idxmin()
    macro_row = macro_df.loc[closest_idx]

    # Build snapshot from macro data
    snapshot = {
        'VIX': {'value': float(macro_row['VIX']), 'stale': False},
        'DXY': {'value': float(macro_row['DXY']), 'stale': False},
        'MOVE': {'value': float(macro_row['MOVE']), 'stale': False},
        'YIELD_2Y': {'value': float(macro_row['YIELD_2Y']), 'stale': False},
        'YIELD_10Y': {'value': float(macro_row['YIELD_10Y']), 'stale': False},
        'USDT.D': {'value': float(macro_row['USDT.D']), 'stale': False},
        'BTC.D': {'value': float(macro_row['BTC.D']), 'stale': False},
        'TOTAL': {'value': float(macro_row['TOTAL']) if not pd.isna(macro_row['TOTAL']) else np.nan, 'stale': pd.isna(macro_row['TOTAL'])},
        'TOTAL2': {'value': float(macro_row['TOTAL2']) if not pd.isna(macro_row['TOTAL2']) else np.nan, 'stale': pd.isna(macro_row['TOTAL2'])},
        'funding': {'value': float(macro_row['funding']), 'stale': False},
        'oi': {'value': float(macro_row['oi']), 'stale': False},
        'rv_20d': {'value': float(macro_row['rv_20d']) if not pd.isna(macro_row['rv_20d']) else 40.0, 'stale': pd.isna(macro_row['rv_20d'])},
        'rv_60d': {'value': float(macro_row['rv_60d']) if not pd.isna(macro_row['rv_60d']) else 45.0, 'stale': pd.isna(macro_row['rv_60d'])}
    }

    return snapshot


def generate_configs(mode: str = 'quick') -> List[Dict]:
    """Generate parameter configurations to test"""
    configs = []

    if mode == 'quick':
        # Quick test - 24 configs
        thresholds = [0.55, 0.60, 0.65, 0.70]
        wyckoff_weights = [0.25, 0.30, 0.35]
        smc_weights = [0.15]
        momentum_weights = [0.30]

    elif mode == 'aggressive':
        # Aggressive mode - lower thresholds for more entries
        thresholds = [0.45, 0.48, 0.50, 0.53]
        wyckoff_weights = [0.25, 0.30, 0.35]
        smc_weights = [0.15]
        momentum_weights = [0.30]

    elif mode == 'grid':
        # Full grid - 144 configs
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        wyckoff_weights = [0.20, 0.25, 0.30, 0.35, 0.40]
        smc_weights = [0.10, 0.15, 0.20]
        momentum_weights = [0.25, 0.30, 0.35]

    else:
        # Exhaustive - 432 configs
        thresholds = [0.50, 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74]
        wyckoff_weights = [0.20, 0.25, 0.30, 0.35, 0.40]
        smc_weights = [0.10, 0.13, 0.16, 0.19]
        momentum_weights = [0.23, 0.27, 0.31, 0.35]

    for threshold in thresholds:
        for w_wyck in wyckoff_weights:
            for w_smc in smc_weights:
                for w_mom in momentum_weights:
                    # Calculate HOB weight (must sum to 1.0)
                    w_hob = 1.0 - (w_wyck + w_smc + w_mom)

                    # Valid range for HOB
                    if 0.15 <= w_hob <= 0.40:
                        configs.append({
                            'fusion_threshold': threshold,
                            'wyckoff_weight': w_wyck,
                            'smc_weight': w_smc,
                            'hob_weight': w_hob,
                            'momentum_weight': w_mom
                        })

    return configs


def main():
    parser = argparse.ArgumentParser(description='Production-Faithful Optimizer v19')
    parser.add_argument('--asset', default='BTC', help='Asset (BTC, ETH, SOL)')
    parser.add_argument('--mode', default='quick', choices=['quick', 'aggressive', 'grid', 'exhaustive'])
    parser.add_argument('--workers', type=int, default=mp.cpu_count() - 1, help='Parallel workers')
    parser.add_argument('--output', default='optimization_results_v19.json', help='Output file')

    # Phase 2: Regime adaptation
    parser.add_argument('--regime', type=str, choices=['true', 'false'], default='false',
                        help='Enable Phase 2 regime adaptation (default: false)')
    parser.add_argument('--start', type=str, help='Start date filter (YYYY-MM-DD, optional)')
    parser.add_argument('--end', type=str, help='End date filter (YYYY-MM-DD, optional)')

    args = parser.parse_args()

    print("🎯 Bull Machine v1.9 Optimizer (Production-Faithful)")
    print(f"Asset: {args.asset}")
    print(f"Mode: {args.mode}")
    print("=" * 70)

    # Load feature store
    feature_path = f"data/features/v18/{args.asset}_1H.parquet"
    if not Path(feature_path).exists():
        print(f"\n❌ Feature store not found: {feature_path}")
        print(f"   Run: python bin/build_feature_store.py --asset {args.asset}")
        return 1

    print(f"\n📊 Loading feature store: {feature_path}")
    features = load_feature_store(feature_path)
    print(f"   ✅ Loaded {len(features['close'])} bars × {len(features)} features")

    # Phase 2: Date filtering
    if args.start or args.end:
        print(f"\n📅 Applying date filter...")
        if 'timestamp' in features:
            timestamps = features['timestamp']
            mask = np.ones(len(timestamps), dtype=bool)

            if args.start:
                start_ts = pd.Timestamp(args.start)
                # Handle tz-aware vs tz-naive
                if hasattr(timestamps[0], 'tz') and timestamps[0].tz is not None:
                    start_ts = start_ts.tz_localize('UTC')
                mask &= (timestamps >= start_ts)
                print(f"   Start: {args.start}")

            if args.end:
                end_ts = pd.Timestamp(args.end)
                # Handle tz-aware vs tz-naive
                if hasattr(timestamps[0], 'tz') and timestamps[0].tz is not None:
                    end_ts = end_ts.tz_localize('UTC')
                mask &= (timestamps <= end_ts)
                print(f"   End: {args.end}")

            # Apply filter to all features
            for key in features:
                features[key] = features[key][mask]

            print(f"   ✅ Filtered to {len(features['close'])} bars")
        else:
            print(f"   ⚠️  No timestamp column in feature store, skipping date filter")

    # Phase 2: Load regime components
    regime_enabled = (args.regime == 'true')
    regime_classifier = None
    regime_policy = None

    if regime_enabled:
        print(f"\n🧠 Loading Phase 2 regime components...")
        feature_order = [
            "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
            "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
            "funding", "oi", "rv_20d", "rv_60d"
        ]
        try:
            regime_classifier = RegimeClassifier.load("models/regime_classifier_gmm.pkl", feature_order)
            regime_policy = RegimePolicy.load("configs/v19/regime_policy.json")
            print(f"   ✅ Regime adaptation ENABLED")
        except Exception as e:
            print(f"   ⚠️  Regime classifier load failed: {e}")
            print(f"   Falling back to baseline (regime disabled)")
            regime_enabled = False
    else:
        print(f"\n📊 Regime adaptation: DISABLED (baseline mode)")

    # Generate configs
    configs = generate_configs(args.mode)
    print(f"\n📋 Generated {len(configs)} configurations ({args.mode} mode)")

    # Run optimization
    print(f"\n🚀 Starting parallel optimization with {args.workers} workers...")
    start_time = time.time()

    with mp.Pool(processes=args.workers) as pool:
        backtest_fn = partial(
            backtest_config,
            features,
            regime_classifier=regime_classifier,
            regime_policy=regime_policy,
            asset=args.asset
        )
        results = pool.map(backtest_fn, configs)

    elapsed = time.time() - start_time

    print(f"\n✅ Optimization complete in {elapsed:.1f}s ({len(configs)/elapsed:.1f} configs/sec)")

    # Filter valid results
    valid_results = [r for r in results if r['trades'] > 0]
    print(f"   Valid results: {len(valid_results)}/{len(results)}")

    if len(valid_results) == 0:
        print("\n⚠️  No valid results (no trades generated)")
        return 1

    # Save results
    with open(args.output, 'w') as f:
        json.dump(valid_results, f, indent=2)

    # ========================================================================
    # ML DATASET LOGGING (v1.8.6+)
    # ========================================================================
    if ML_LOGGING_AVAILABLE and len(valid_results) > 0:
        try:
            print(f"\n📊 Logging {len(valid_results)} results to ML dataset...")

            # Build macro snapshot (regime at start of window)
            macro_snapshot = build_macro_snapshot(features, start_idx=0, asset=args.asset)

            # Build regime vector
            regime_vector = build_regime_vector(macro_snapshot, lookback_window=None)

            # Get window metadata
            start_date = str(features['timestamp'][0]) if len(features['timestamp']) > 0 else 'unknown'
            end_date = str(features['timestamp'][-1]) if len(features['timestamp']) > 0 else 'unknown'

            # Convert results to training rows
            training_rows = []
            for result in valid_results:
                # Build config dict
                config = {
                    'fusion': {
                        'entry_threshold_confidence': result['fusion_threshold'],
                        'weights': {
                            'wyckoff': result['wyckoff_weight'],
                            'smc': result['smc_weight'],
                            'liquidity': result['hob_weight'],
                            'momentum': result['momentum_weight']
                        }
                    },
                    'exits': {
                        'atr_k': 1.0,  # TODO: Add to result dict
                        'trail_atr_k': 1.2,
                        'tp1_r': 1.0
                    },
                    'risk': {
                        'base_risk_pct': 0.0075  # TODO: Add to result dict
                    },
                    'fast_signals': {
                        'adx_threshold': 20  # TODO: Add to result dict
                    }
                }

                # Build metrics dict
                metrics = {
                    'profit_factor': result.get('profit_factor', 0.0),
                    'sharpe': result.get('sharpe_ratio', 0.0),
                    'max_drawdown': result.get('max_drawdown', 0.0),
                    'total_trades': result.get('trades', 0),
                    'win_rate': result.get('win_rate', 0.0),
                    'total_return_pct': result.get('total_return', 0.0),
                    'avg_r_multiple': result.get('avg_r', 0.0)
                }

                # Build metadata
                metadata = {
                    'asset': args.asset,
                    'start_date': start_date,
                    'end_date': end_date
                }

                # Build training row
                row = build_training_row(regime_vector, config, metrics, metadata)
                training_rows.append(row)

            # Append to dataset
            dataset = OptimizationDataset()
            dataset.append_results(training_rows)

            print(f"   ✅ ML dataset updated ({len(training_rows)} rows appended)")

        except Exception as e:
            print(f"   ⚠️  ML logging failed: {e}")
            print("   (Results still saved to JSON)")

    # Display top results
    df = pd.DataFrame(valid_results)

    print("\n" + "=" * 70)
    print("🏆 TOP 10 CONFIGURATIONS (by Sharpe Ratio)")
    print("=" * 70)

    top_10 = df.nlargest(10, 'sharpe_ratio')
    print(top_10[['fusion_threshold', 'wyckoff_weight', 'momentum_weight', 'trades',
                  'win_rate', 'total_return', 'sharpe_ratio', 'profit_factor', 'avg_r']].to_string(index=False))

    print(f"\n💾 Full results ({len(valid_results)} configs) saved to: {args.output}")
    print("\nNext steps:")
    print(f"  python bin/analyze_optimization.py {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
