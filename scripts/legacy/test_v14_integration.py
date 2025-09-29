#!/usr/bin/env python3
"""
Test v1.4 Integration with v1.3 Engine
Quick validation that the backtest framework can call v1.3
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, ".")


def test_basic_integration():
    """Test basic v1.4 components load and work"""
    print("🧪 Testing v1.4 Basic Integration")
    print("=" * 50)

    try:
        # Test imports
        from bull_machine.backtest.strategy_adapter_v13_integrated import strategy_from_df
        from bull_machine.backtest.broker import PaperBroker
        from bull_machine.backtest.portfolio import Portfolio

        print("✅ All v1.4 imports successful")

        # Test strategy adapter with sample data
        sample_data = pd.DataFrame(
            {
                "open": np.random.uniform(50000, 52000, 200),
                "high": np.random.uniform(51000, 53000, 200),
                "low": np.random.uniform(49000, 51000, 200),
                "close": np.random.uniform(50000, 52000, 200),
                "volume": np.random.uniform(1000, 5000, 200),
            }
        )

        # Ensure realistic OHLC relationships
        for i in range(len(sample_data)):
            high = max(
                sample_data.iloc[i]["open"], sample_data.iloc[i]["close"]
            ) + np.random.uniform(0, 500)
            low = min(
                sample_data.iloc[i]["open"], sample_data.iloc[i]["close"]
            ) - np.random.uniform(0, 500)
            sample_data.iloc[i, sample_data.columns.get_loc("high")] = high
            sample_data.iloc[i, sample_data.columns.get_loc("low")] = low

        print("✅ Sample data generated (200 bars)")

        # Test strategy adapter (should not crash)
        signal = strategy_from_df("BTCUSD", "1H", sample_data, balance=10000)
        print(f"✅ Strategy adapter returned: {signal.get('action', 'unknown')}")

        # Test broker
        broker = PaperBroker(fee_bps=10, slippage_bps=5)
        fill = broker.submit(
            ts=1234567890, symbol="BTCUSD", side="long", size=1.0, price_hint=50000
        )
        print(f"✅ Broker fill: ${fill['price']:.2f} size {fill['size_filled']}")

        # Test portfolio
        portfolio = Portfolio(100000)
        portfolio.on_fill("BTCUSD", "long", 50000, 1.0, 50)
        equity = portfolio.equity()
        print(f"✅ Portfolio equity: ${equity:,.2f}")

        return True

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_v13_engine_call():
    """Test calling v1.3 engine with real data if available"""
    print(f"\n🔗 Testing v1.3 Engine Integration")
    print("=" * 50)

    chart_dir = "/Users/raymondghandchi/Downloads/Chart logs 2"
    csv_path = f"{chart_dir}/COINBASE_BTCUSD, 60_50ad4.csv"

    if not os.path.exists(csv_path):
        print(f"⚠️  Chart data not found at {csv_path}")
        print("   Using synthetic data instead")

        # Create synthetic data
        dates = pd.date_range(start="2024-01-01", periods=500, freq="1H")
        price = 50000
        data = []

        for date in dates:
            # Random walk
            price += np.random.normal(0, 100)
            open_price = price + np.random.normal(0, 50)
            high_price = max(open_price, price) + abs(np.random.normal(0, 200))
            low_price = min(open_price, price) - abs(np.random.normal(0, 200))
            close_price = price

            data.append(
                {
                    "time": int(date.timestamp()),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": np.random.uniform(100, 1000),
                }
            )

        df = pd.DataFrame(data)

    else:
        print(f"✅ Loading real BTC data from {csv_path}")
        df = pd.read_csv(csv_path)

    try:
        from bull_machine.backtest.strategy_adapter_v13_integrated import strategy_from_df

        # Take a recent window
        window = df.tail(300)  # Last 300 bars
        print(f"✅ Data window: {len(window)} bars")

        # Call strategy
        signal = strategy_from_df("BTCUSD", "1H", window, balance=100000)
        print(f"✅ v1.3 engine result: {signal}")

        if signal.get("action") not in ["flat"]:
            print(f"   📈 Signal Details:")
            print(f"      Action: {signal.get('action')}")
            print(f"      Size: {signal.get('size', 'N/A')}")
            print(f"      Confidence: {signal.get('confidence', 'N/A')}")
            print(f"      Stop: {signal.get('stop', 'N/A')}")
            if signal.get("reasons"):
                print(f"      Reasons: {signal.get('reasons')}")

        return True

    except Exception as e:
        print(f"❌ v1.3 engine test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_backtest_pipeline():
    """Test a mini backtest run"""
    print(f"\n🏃‍♂️ Testing Full Backtest Pipeline")
    print("=" * 50)

    try:
        from bull_machine.backtest.engine import BacktestEngine
        from bull_machine.backtest.datafeed import DataFeed
        from bull_machine.backtest.broker import PaperBroker
        from bull_machine.backtest.portfolio import Portfolio
        from bull_machine.backtest.strategy_adapter_v13_integrated import strategy_from_df

        # Create mini config
        config = {
            "run_id": "v14_test",
            "engine": {"lookback_bars": 50, "seed": 42},
            "portfolio": {"starting_cash": 10000},
        }

        # Create synthetic data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        price = 50000
        data = []

        for date in dates:
            price += np.random.normal(0, 100)
            data.append(
                {
                    "time": date,
                    "open": price + np.random.normal(0, 50),
                    "high": price + abs(np.random.normal(0, 200)),
                    "low": price - abs(np.random.normal(0, 200)),
                    "close": price,
                    "volume": np.random.uniform(100, 1000),
                }
            )

        df = pd.DataFrame(data).set_index("time")

        # Save test data to temporary CSV
        temp_csv = "test_btc_data.csv"
        df.reset_index().to_csv(temp_csv, index=False)

        # Setup components
        datafeed = DataFeed({"BTCUSD": temp_csv})
        broker = PaperBroker(fee_bps=10)
        portfolio = Portfolio(10000)
        engine = BacktestEngine(config, datafeed, broker, portfolio)

        # Strategy function
        def test_strategy(symbol, tf, window):
            # Simple strategy: always flat (for testing)
            return {"action": "flat"}

        # Run mini backtest
        result = engine.run(test_strategy, ["BTCUSD"], ["1H"], out_dir="test_output")
        print(f"✅ Backtest completed:")
        print(f"   Metrics: {result['metrics']}")
        print(f"   Artifacts: {result['artifacts']}")

        # Cleanup
        os.remove(temp_csv)

        return True

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("🚀 BULL MACHINE v1.4 INTEGRATION TESTS")
    print("=" * 60)
    print("Validating v1.4 backtest framework integration with v1.3 engine\n")

    tests = [
        ("Basic Integration", test_basic_integration),
        ("v1.3 Engine Call", test_v13_engine_call),
        ("Full Pipeline", test_full_backtest_pipeline),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print(f"\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(f"🎉 ALL TESTS PASSED - v1.4 integration ready!")
    else:
        print(f"⚠️  Some tests failed - review implementation")

    return passed == total


if __name__ == "__main__":
    main()
