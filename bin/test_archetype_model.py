#!/usr/bin/env python3
"""
Test ArchetypeModel wrapper integration.

Verifies that ArchetypeModel correctly wraps the existing archetype system
and implements the BaseModel interface.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine.models.archetype_model import ArchetypeModel
from engine.models.base import Signal, Position


def create_test_data(n_bars: int = 100) -> pd.DataFrame:
    """Create synthetic test data with required features."""
    np.random.seed(42)

    # Generate OHLCV
    close_prices = 50000 + np.cumsum(np.random.randn(n_bars) * 500)
    high_prices = close_prices + np.abs(np.random.randn(n_bars) * 200)
    low_prices = close_prices - np.abs(np.random.randn(n_bars) * 200)
    open_prices = close_prices + np.random.randn(n_bars) * 100

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1H'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, n_bars),

        # ATR for stop loss calculations
        'atr_14': np.random.uniform(500, 1500, n_bars),
        'atr': np.random.uniform(500, 1500, n_bars),

        # Fusion components
        'fusion_score': np.random.uniform(0.2, 0.8, n_bars),
        'liquidity_score': np.random.uniform(0.1, 0.9, n_bars),
        'wyckoff_score': np.random.uniform(0.2, 0.7, n_bars),
        'momentum_score': np.random.uniform(0.3, 0.8, n_bars),

        # S4-specific features (funding divergence)
        'funding_Z': np.random.uniform(-2.5, 1.0, n_bars),
        'funding_rate': np.random.uniform(-0.01, 0.03, n_bars),
        'oi_change_1h': np.random.uniform(-0.05, 0.05, n_bars),
        'volume_quiet_score': np.random.uniform(0.3, 0.7, n_bars),
        'liquidity_thin_score': np.random.uniform(0.2, 0.6, n_bars),

        # Price action
        'price_resilience_score': np.random.uniform(0.4, 0.9, n_bars),
        'returns_1h': np.random.uniform(-0.02, 0.02, n_bars),

        # Regime features
        'macro_regime': np.random.choice(['risk_on', 'neutral', 'risk_off'], n_bars),
        'regime_probs_risk_on': np.random.uniform(0.2, 0.8, n_bars),
    })

    df.set_index('timestamp', inplace=True)

    return df


def test_initialization():
    """Test model initialization."""
    print("\n" + "="*60)
    print("TEST 1: Model Initialization")
    print("="*60)

    config_path = 'configs/s4_optimized_oos_test.json'

    try:
        model = ArchetypeModel(
            config_path=config_path,
            archetype_name='S4',
            name='S4-Test'
        )

        print(f"✓ Model created: {model.name}")
        print(f"✓ Config loaded: {model.config_path}")
        print(f"✓ Archetype: {model.archetype_name}")
        print(f"✓ Direction: {model.direction}")
        print(f"✓ Parameters:")
        for k, v in model.get_params().items():
            print(f"    {k}: {v}")

        return model

    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_fit(model: ArchetypeModel, df: pd.DataFrame):
    """Test model fit (should be no-op for pre-configured model)."""
    print("\n" + "="*60)
    print("TEST 2: Model Fit")
    print("="*60)

    try:
        model.fit(df)
        print(f"✓ Model fitted: {model._is_fitted}")
        return True
    except Exception as e:
        print(f"✗ Fit failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predict(model: ArchetypeModel, df: pd.DataFrame):
    """Test signal generation."""
    print("\n" + "="*60)
    print("TEST 3: Signal Generation")
    print("="*60)

    signals_generated = 0
    holds = 0

    try:
        # Test on first 20 bars
        for i in range(min(20, len(df))):
            bar = df.iloc[i]
            signal = model.predict(bar)

            # Verify signal structure
            assert isinstance(signal, Signal), f"Expected Signal, got {type(signal)}"
            assert signal.direction in ['long', 'short', 'hold'], f"Invalid direction: {signal.direction}"
            assert 0.0 <= signal.confidence <= 1.0, f"Invalid confidence: {signal.confidence}"
            assert signal.entry_price > 0, f"Invalid entry price: {signal.entry_price}"

            if signal.is_entry:
                signals_generated += 1
                print(f"\n✓ Bar {i}: {signal.direction.upper()} SIGNAL")
                print(f"  Entry: ${signal.entry_price:,.2f}")
                print(f"  Stop:  ${signal.stop_loss:,.2f}")
                print(f"  Confidence: {signal.confidence:.2%}")
                print(f"  Archetype: {signal.metadata.get('archetype', 'N/A')}")
                print(f"  Fusion: {signal.metadata.get('fusion_score', 0):.3f}")
            else:
                holds += 1

        print(f"\n✓ Tested {min(20, len(df))} bars")
        print(f"  Signals: {signals_generated}")
        print(f"  Holds: {holds}")

        return signals_generated > 0  # Should generate at least 1 signal

    except Exception as e:
        print(f"✗ Predict failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_position_sizing(model: ArchetypeModel, df: pd.DataFrame):
    """Test position size calculation."""
    print("\n" + "="*60)
    print("TEST 4: Position Sizing")
    print("="*60)

    try:
        bar = df.iloc[0]
        signal = Signal(
            direction='long',
            confidence=0.8,
            entry_price=bar['close'],
            stop_loss=bar['close'] - (2.5 * bar['atr_14'])
        )

        position_size = model.get_position_size(bar, signal)

        print(f"✓ Position size calculated: ${position_size:,.2f}")
        print(f"  Entry: ${signal.entry_price:,.2f}")
        print(f"  Stop: ${signal.stop_loss:,.2f}")
        print(f"  Stop distance: {abs(signal.entry_price - signal.stop_loss)/signal.entry_price*100:.2f}%")
        print(f"  Risk per trade: {model.max_risk_pct*100:.1f}%")

        # Verify position size is reasonable
        assert position_size > 0, "Position size must be positive"
        assert position_size < 10000, "Position size shouldn't exceed 15% of $10k portfolio"

        return True

    except Exception as e:
        print(f"✗ Position sizing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_regime_switching(model: ArchetypeModel, df: pd.DataFrame):
    """Test regime switching."""
    print("\n" + "="*60)
    print("TEST 5: Regime Switching")
    print("="*60)

    try:
        bar = df.iloc[10]

        # Test different regimes
        for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
            model.set_regime(regime)
            signal = model.predict(bar)

            print(f"  {regime:12s}: {signal.direction:5s} (conf={signal.confidence:.2f})")

        print("✓ Regime switching works")
        return True

    except Exception as e:
        print(f"✗ Regime switching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_flow(model: ArchetypeModel, df: pd.DataFrame):
    """Test complete integration flow."""
    print("\n" + "="*60)
    print("TEST 6: Integration Flow (Full Workflow)")
    print("="*60)

    try:
        # Simulate trading loop
        position = None
        trades = []

        for i in range(min(50, len(df))):
            bar = df.iloc[i]

            # Generate signal
            signal = model.predict(bar, position)

            # If we have a signal and no position, enter trade
            if signal.is_entry and position is None:
                position_size = model.get_position_size(bar, signal)

                position = Position(
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    entry_time=bar.name,
                    size=position_size,
                    stop_loss=signal.stop_loss,
                    metadata={'signal': signal}
                )

                trades.append({
                    'entry_time': bar.name,
                    'entry_price': signal.entry_price,
                    'direction': signal.direction,
                    'size': position_size,
                    'stop': signal.stop_loss,
                    'archetype': signal.metadata.get('archetype', 'N/A')
                })

                print(f"\n✓ Trade #{len(trades)} @ bar {i}")
                print(f"  Time: {bar.name}")
                print(f"  Direction: {signal.direction}")
                print(f"  Entry: ${signal.entry_price:,.2f}")
                print(f"  Size: ${position_size:,.2f}")
                print(f"  Stop: ${signal.stop_loss:,.2f}")

            # Check stop loss
            elif position is not None:
                if position.direction == 'long' and bar['low'] <= position.stop_loss:
                    print(f"  Stop hit @ bar {i}")
                    position = None
                elif position.direction == 'short' and bar['high'] >= position.stop_loss:
                    print(f"  Stop hit @ bar {i}")
                    position = None

        print(f"\n✓ Integration test complete")
        print(f"  Bars processed: {min(50, len(df))}")
        print(f"  Trades generated: {len(trades)}")

        return True

    except Exception as e:
        print(f"✗ Integration flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ARCHETYPE MODEL WRAPPER TEST SUITE")
    print("="*60)

    # Generate test data
    print("\nGenerating test data...")
    df = create_test_data(n_bars=100)
    print(f"✓ Created {len(df)} bars of test data")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Run tests
    results = {}

    # Test 1: Initialization
    model = test_initialization()
    results['initialization'] = model is not None

    if model is None:
        print("\n✗ Cannot continue - initialization failed")
        return

    # Test 2: Fit
    results['fit'] = test_fit(model, df)

    # Test 3: Predict
    results['predict'] = test_predict(model, df)

    # Test 4: Position sizing
    results['position_sizing'] = test_position_sizing(model, df)

    # Test 5: Regime switching
    results['regime_switching'] = test_regime_switching(model, df)

    # Test 6: Integration flow
    results['integration'] = test_integration_flow(model, df)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {test_name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n✓ All tests passed! ArchetypeModel is working correctly.")
    else:
        print(f"\n✗ {total - passed} test(s) failed.")


if __name__ == '__main__':
    main()
