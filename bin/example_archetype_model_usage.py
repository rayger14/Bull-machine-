#!/usr/bin/env python3
"""
Example usage of ArchetypeModel wrapper.

Shows how to use the new BaseModel interface with the existing archetype system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from engine.models.archetype_model import ArchetypeModel


def load_sample_data():
    """Load or create sample data for testing."""
    # In production, load from feature store
    # For demo, create synthetic data
    n_bars = 100
    np.random.seed(42)

    close_prices = 50000 + np.cumsum(np.random.randn(n_bars) * 500)

    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='h'),
        'open': close_prices + np.random.randn(n_bars) * 100,
        'high': close_prices + np.abs(np.random.randn(n_bars) * 200),
        'low': close_prices - np.abs(np.random.randn(n_bars) * 200),
        'close': close_prices,
        'volume': np.random.uniform(100, 1000, n_bars),
        'atr_14': np.random.uniform(500, 1500, n_bars),
        'atr': np.random.uniform(500, 1500, n_bars),

        # Fusion components
        'fusion_score': np.random.uniform(0.2, 0.8, n_bars),
        'liquidity_score': np.random.uniform(0.1, 0.9, n_bars),

        # S4 features
        'funding_Z': np.random.uniform(-2.5, 1.0, n_bars),
        'price_resilience_score': np.random.uniform(0.4, 0.9, n_bars),
    })

    df.set_index('timestamp', inplace=True)
    return df


def main():
    """Demonstrate ArchetypeModel usage."""
    print("="*70)
    print("ARCHETYPE MODEL - EXAMPLE USAGE")
    print("="*70)

    # 1. Initialize model with config
    print("\n1. Initialize Model")
    print("-" * 70)

    model = ArchetypeModel(
        config_path='configs/s4_optimized_oos_test.json',
        archetype_name='S4',
        name='S4-Production'
    )

    print(f"Model: {model}")
    print(f"Parameters: {model.get_params()}")

    # 2. Load training data and fit (no-op for pre-configured models)
    print("\n2. Fit Model (using pre-configured parameters)")
    print("-" * 70)

    train_data = load_sample_data()
    model.fit(train_data)

    print(f"Fitted: {model._is_fitted}")

    # 3. Generate signals on test data
    print("\n3. Generate Signals on Test Data")
    print("-" * 70)

    test_data = load_sample_data()

    signals_found = 0
    for i in range(min(20, len(test_data))):
        bar = test_data.iloc[i]
        signal = model.predict(bar)

        if signal.is_entry:
            signals_found += 1
            print(f"\nSignal #{signals_found} at bar {i} ({bar.name}):")
            print(f"  Direction: {signal.direction}")
            print(f"  Entry: ${signal.entry_price:,.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
            print(f"  Confidence: {signal.confidence:.1%}")
            print(f"  Archetype: {signal.metadata.get('archetype', 'N/A')}")

            # Calculate position size
            position_size = model.get_position_size(bar, signal)
            print(f"  Position Size: ${position_size:,.2f}")

    if signals_found == 0:
        print("No signals generated (expected with random synthetic data)")
        print("In production, use real feature data for actual signals")

    # 4. Use in backtest loop
    print("\n4. Example Backtest Loop")
    print("-" * 70)

    print("""
# Pseudo-code for backtest integration:

from engine.models.archetype_model import ArchetypeModel

# Initialize
model = ArchetypeModel('configs/s4_optimized.json', 'S4')
model.fit(train_data)

# Backtest loop
portfolio_value = 10000
position = None

for bar in test_data.iterrows():
    # Generate signal
    signal = model.predict(bar, position)

    # Entry logic
    if signal.is_entry and position is None:
        position_size = model.get_position_size(bar, signal)

        position = Position(
            direction=signal.direction,
            entry_price=signal.entry_price,
            entry_time=bar.name,
            size=position_size,
            stop_loss=signal.stop_loss
        )

        print(f"ENTER {signal.direction} @ ${signal.entry_price:,.2f}")

    # Exit logic (stop loss, take profit, etc.)
    elif position is not None:
        # Check stop loss hit
        if position.direction == 'long' and bar['low'] <= position.stop_loss:
            print(f"STOP LOSS HIT @ ${position.stop_loss:,.2f}")
            position = None
    """)

    # 5. Get model state
    print("\n5. Model State & Introspection")
    print("-" * 70)

    state = model.get_state()
    print("State:")
    for k, v in state.items():
        print(f"  {k}: {v}")

    # 6. Regime switching
    print("\n6. Regime Adaptation")
    print("-" * 70)

    test_bar = test_data.iloc[0]

    for regime in ['risk_on', 'neutral', 'risk_off', 'crisis']:
        model.set_regime(regime)
        signal = model.predict(test_bar)
        print(f"  {regime:12s}: {signal.direction:5s} (conf={signal.confidence:.2f})")

    print("\n" + "="*70)
    print("✓ Example complete!")
    print("="*70)


if __name__ == '__main__':
    main()
