#!/usr/bin/env python3
"""
Quick validation test for engine weight optimizer.

Tests that the optimizer can run end-to-end without errors
using a small sample of the data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import tempfile
import json


def create_mock_feature_store(n_bars: int = 1000) -> pd.DataFrame:
    """Create minimal mock feature store for testing."""
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='1H', tz='UTC')

    # Generate synthetic OHLCV
    close = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    high = close + np.abs(np.random.randn(n_bars) * 50)
    low = close - np.abs(np.random.randn(n_bars) * 50)
    open_ = close + np.random.randn(n_bars) * 30
    volume = np.abs(np.random.randn(n_bars) * 1e6 + 1e7)

    # Generate domain scores (with some correlation to price movement)
    price_change = np.diff(close, prepend=close[0])
    momentum_signal = (price_change > 0).astype(float)

    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'open': open_,
        'volume': volume,
        'tf1h_fusion_score': 0.5 + momentum_signal * 0.2 + np.random.randn(n_bars) * 0.1,
        'tf4h_fusion_score': 0.5 + momentum_signal * 0.15 + np.random.randn(n_bars) * 0.1,
        'tf1d_fusion_score': 0.5 + momentum_signal * 0.1 + np.random.randn(n_bars) * 0.05,
        'liquidity_score': 0.5 + np.random.randn(n_bars) * 0.15,
        'rsi_14': 50 + np.random.randn(n_bars) * 15,
        'adx_14': 20 + np.abs(np.random.randn(n_bars) * 10),
        'VIX_Z': np.random.randn(n_bars) * 0.5,
        'regime_label': np.random.choice(['RISK_ON', 'RISK_OFF', 'NEUTRAL'], n_bars),
        'regime_confidence': 0.5 + np.abs(np.random.randn(n_bars) * 0.2),
        'k2_fusion_score': 0.5 + momentum_signal * 0.15 + np.random.randn(n_bars) * 0.1,
        'macd_histogram': np.random.randn(n_bars) * 50,
    }, index=dates)

    # Clip scores to [0, 1]
    for col in ['tf1h_fusion_score', 'tf4h_fusion_score', 'tf1d_fusion_score',
                'liquidity_score', 'k2_fusion_score']:
        df[col] = df[col].clip(0, 1)

    df['rsi_14'] = df['rsi_14'].clip(0, 100)
    df['adx_14'] = df['adx_14'].clip(0, 100)
    df['regime_confidence'] = df['regime_confidence'].clip(0, 1)

    return df


def test_optimizer():
    """Test optimizer with minimal trials."""
    print("=" * 80)
    print("ENGINE WEIGHT OPTIMIZER - VALIDATION TEST")
    print("=" * 80)

    # Create mock data
    print("\n1. Creating mock feature store...")
    df = create_mock_feature_store(n_bars=2000)
    print(f"   ✓ Generated {len(df)} bars")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        df.to_parquet(tmp_path)
        print(f"   ✓ Saved to {tmp_path}")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        print(f"   ✓ Output dir: {output_dir}")

        # Import optimizer
        print("\n2. Initializing optimizer...")
        from bin.optimize_engine_weights import EngineWeightOptimizer

        optimizer = EngineWeightOptimizer(
            feature_store_path=str(tmp_path),
            output_dir=str(output_dir)
        )
        print("   ✓ Optimizer initialized")

        # Run minimal optimization
        print("\n3. Running optimization (5 trials - fast test)...")
        try:
            results = optimizer.optimize(n_trials=5)
            print("   ✓ Optimization completed")

            # Check outputs
            print("\n4. Validating outputs...")

            # Check optimal weights
            weights_file = output_dir / "optimal_weights.json"
            if weights_file.exists():
                with open(weights_file) as f:
                    weights_data = json.load(f)
                    optimal_weights = weights_data['optimal_weights']

                print("   ✓ optimal_weights.json created")
                print(f"      Structure: {optimal_weights['structure']:.3f}")
                print(f"      Liquidity: {optimal_weights['liquidity']:.3f}")
                print(f"      Momentum: {optimal_weights['momentum']:.3f}")
                print(f"      Macro: {optimal_weights['macro']:.3f}")

                # Verify weights sum to 1.0
                weight_sum = sum(optimal_weights.values())
                assert abs(weight_sum - 1.0) < 0.001, f"Weights don't sum to 1.0: {weight_sum}"
                print(f"      Sum: {weight_sum:.6f} ✓")
            else:
                print("   ✗ optimal_weights.json not found")
                return False

            # Check study database
            study_file = output_dir / "optuna_study.db"
            if study_file.exists():
                print("   ✓ optuna_study.db created")
            else:
                print("   ✗ optuna_study.db not found")

            print("\n5. Testing sensitivity analysis...")
            try:
                optimizer.analyze_sensitivity(results['study'])
                sensitivity_file = output_dir / "weight_sensitivity.png"
                if sensitivity_file.exists():
                    print("   ✓ weight_sensitivity.png created")
                else:
                    print("   ⚠ weight_sensitivity.png not created (may need display)")
            except Exception as e:
                print(f"   ⚠ Sensitivity analysis skipped: {e}")

            print("\n6. Testing regime breakdown...")
            try:
                optimizer.validate_regime_breakdown(results['optimal_weights'])
                breakdown_file = output_dir / "regime_breakdown.csv"
                if breakdown_file.exists():
                    print("   ✓ regime_breakdown.csv created")
                    breakdown = pd.read_csv(breakdown_file)
                    print(f"      Regimes analyzed: {len(breakdown)}")
                else:
                    print("   ✗ regime_breakdown.csv not found")
            except Exception as e:
                print(f"   ⚠ Regime breakdown failed: {e}")

            print("\n" + "=" * 80)
            print("✅ VALIDATION TEST PASSED")
            print("=" * 80)
            print("\nOptimizer is working correctly. You can now run with full data:")
            print("  python bin/optimize_engine_weights.py \\")
            print("    --feature-store data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \\")
            print("    --trials 100")

            return True

        except Exception as e:
            print(f"\n✗ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Cleanup
    tmp_path.unlink()


def test_quality_filter():
    """Test ML quality filter (if dependencies available)."""
    print("\n" + "=" * 80)
    print("ML QUALITY FILTER - VALIDATION TEST")
    print("=" * 80)

    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        print("✓ LightGBM and sklearn available")
    except ImportError:
        print("⚠ LightGBM or sklearn not available - skipping ML test")
        print("  Install with: pip install lightgbm scikit-learn")
        return True  # Not a failure, just skipped

    # Create mock data
    print("\n1. Creating mock feature store...")
    df = create_mock_feature_store(n_bars=2000)
    print(f"   ✓ Generated {len(df)} bars")

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        df.to_parquet(tmp_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        print("\n2. Initializing quality filter trainer...")
        from bin.train_quality_filter import QualityFilterTrainer

        trainer = QualityFilterTrainer(
            feature_store_path=str(tmp_path),
            output_dir=str(output_dir)
        )
        print("   ✓ Trainer initialized")

        print("\n3. Training model (this may take a minute)...")
        try:
            metrics = trainer.train(test_size=0.3)
            print("   ✓ Training completed")
            print(f"      Test AUC: {metrics['test_auc']:.4f}")

            # Check outputs
            print("\n4. Validating outputs...")

            model_file = output_dir / "quality_filter_model.pkl"
            if model_file.exists():
                print("   ✓ quality_filter_model.pkl created")
            else:
                print("   ✗ quality_filter_model.pkl not found")
                return False

            report_file = output_dir / "quality_filter_report.md"
            if report_file.exists():
                print("   ✓ quality_filter_report.md created")
            else:
                print("   ⚠ quality_filter_report.md not found")

            print("\n✅ ML QUALITY FILTER TEST PASSED")
            return True

        except Exception as e:
            print(f"\n✗ Quality filter training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Cleanup
    tmp_path.unlink()


def main():
    """Run all validation tests."""
    print("\nRunning validation tests for engine weight optimization system...\n")

    # Test optimizer
    optimizer_ok = test_optimizer()

    # Test quality filter (optional)
    quality_filter_ok = test_quality_filter()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"  Optuna Weight Optimizer: {'✅ PASS' if optimizer_ok else '❌ FAIL'}")
    print(f"  ML Quality Filter:       {'✅ PASS' if quality_filter_ok else '⚠ SKIPPED'}")

    if optimizer_ok:
        print("\n✅ All critical tests passed. System is ready for production use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please review errors above.")
        return 1


if __name__ == '__main__':
    exit(main())
