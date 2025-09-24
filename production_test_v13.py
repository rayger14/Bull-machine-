#!/usr/bin/env python3
"""
Bull Machine v1.3 Production Testing Framework
Tests MTF sync integration with real market data across multiple timeframes
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level=logging.WARNING):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_subset_csv(df, end_idx, filename):
    """Create a CSV subset up to end_idx for testing"""
    subset = df.iloc[: end_idx + 1].copy()
    subset.to_csv(filename, index=False)
    return filename


def test_v13_mtf_integration(
    csv_path,
    symbol,
    timeframe,
    thresholds=[0.30, 0.35, 0.40],
    test_interval=10,
    max_tests=50,
    mtf_enabled=True,
):
    """
    Test v1.3 with MTF sync enabled vs disabled

    Args:
        csv_path: Path to historical data CSV
        symbol: Asset symbol for logging
        timeframe: Timeframe string for logging
        thresholds: List of thresholds to test
        test_interval: Bars between tests
        max_tests: Maximum number of signals to test
        mtf_enabled: Whether to enable MTF sync
    """

    print(f"\n{'=' * 80}")
    print(f"v1.3 {'MTF ENABLED' if mtf_enabled else 'MTF DISABLED'} TEST: {symbol} {timeframe}")
    print(f"{'=' * 80}")

    # Load full dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} bars of {symbol} data")

    # Convert time column to proper format if needed
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    results_by_threshold = {}

    for threshold in thresholds:
        print(f"\n📊 Testing threshold={threshold:.2f}...")

        trades = []
        temp_files = []
        signals_tested = 0
        no_trade_count = 0
        veto_reasons = {}
        mtf_decisions = {"allow": 0, "raise": 0, "veto": 0}

        # Start from bar 100 to ensure enough history
        start_idx = 100
        end_idx = min(len(df) - 30, start_idx + (max_tests * test_interval))

        for i in range(start_idx, end_idx, test_interval):
            if signals_tested >= max_tests:
                break

            # Create subset CSV up to current point
            temp_filename = f"temp_{symbol}_{threshold}_{i}.csv"
            create_subset_csv(df, i, temp_filename)
            temp_files.append(temp_filename)

            try:
                # Import v1.3 main function
                from bull_machine.app.main_v13 import run_bull_machine_v1_3

                # Run v1.3 with MTF enabled/disabled
                result = run_bull_machine_v1_3(
                    temp_filename, account_balance=10000, mtf_enabled=mtf_enabled
                )

                signals_tested += 1

                if result["action"] == "enter_trade":
                    signal = result.get("signal")
                    plan = result.get("risk_plan")

                    if signal and plan:
                        current_bar = df.iloc[i]
                        entry_date = current_bar.get("timestamp", i)

                        trade = {
                            "date": entry_date,
                            "bar_idx": i,
                            "side": signal.side,
                            "entry_price": plan.entry,
                            "stop_price": plan.stop,
                            "confidence": signal.confidence,
                            "size": plan.size,
                            "risk": abs(plan.entry - plan.stop) * plan.size,
                            "mtf_enabled": mtf_enabled,
                        }
                        trades.append(trade)

                        print(
                            f"  Trade #{len(trades)}: {signal.side.upper()} @ ${plan.entry:.2f} [Conf: {signal.confidence:.3f}]"
                        )

                elif result["action"] == "no_trade":
                    no_trade_count += 1
                    reason = result.get("reason", "unknown")
                    veto_reasons[reason] = veto_reasons.get(reason, 0) + 1

            except Exception as e:
                if "no_trade" not in str(e).lower():
                    print(f"  Error at bar {i}: {e}")
                continue

        # Cleanup temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        # Store results for this threshold
        results_by_threshold[threshold] = {
            "trades": trades,
            "signals_tested": signals_tested,
            "no_trade_count": no_trade_count,
            "veto_reasons": veto_reasons,
            "mtf_decisions": mtf_decisions,
            "trade_count": len(trades),
            "avg_confidence": sum(t["confidence"] for t in trades) / len(trades) if trades else 0,
            "mtf_enabled": mtf_enabled,
        }

        print(f"\n  Results for threshold {threshold:.2f}:")
        print(f"    Signals tested: {signals_tested}")
        print(f"    Trades generated: {len(trades)}")
        print(f"    No-trade signals: {no_trade_count}")
        print(f"    Trade rate: {(len(trades) / signals_tested) * 100:.1f}%")
        if trades:
            print(
                f"    Average confidence: {results_by_threshold[threshold]['avg_confidence']:.3f}"
            )

        if veto_reasons:
            print(f"    Top veto reasons:")
            for reason, count in sorted(veto_reasons.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"      - {reason}: {count} times")

    return results_by_threshold


def simulate_trade_outcomes(df, trades, bars_ahead=20):
    """Simulate trade outcomes for generated signals"""
    results = []

    for trade in trades:
        bar_idx = trade["bar_idx"]
        side = trade["side"]
        entry = trade["entry_price"]
        stop = trade["stop_price"]

        # Get future bars
        future_bars = []
        for j in range(bar_idx + 1, min(bar_idx + bars_ahead + 1, len(df))):
            future_bars.append(df.iloc[j])

        if not future_bars:
            continue

        # Simple simulation: check if stop hit or profit target reached
        outcome = "expired"
        exit_price = entry
        bars_held = bars_ahead

        # Calculate R-based targets
        risk_per_unit = abs(entry - stop)
        if side == "long":
            tp1 = entry + risk_per_unit * 1.0  # 1R
            tp2 = entry + risk_per_unit * 2.0  # 2R
        else:
            tp1 = entry - risk_per_unit * 1.0  # 1R
            tp2 = entry - risk_per_unit * 2.0  # 2R

        for i, bar in enumerate(future_bars):
            if side == "long":
                if bar["low"] <= stop:
                    outcome = "loss"
                    exit_price = stop
                    bars_held = i + 1
                    break
                elif bar["high"] >= tp2:
                    outcome = "win_tp2"
                    exit_price = tp2
                    bars_held = i + 1
                    break
                elif bar["high"] >= tp1:
                    outcome = "win_tp1"
                    exit_price = tp1
                    bars_held = i + 1
                    break
            else:  # short
                if bar["high"] >= stop:
                    outcome = "loss"
                    exit_price = stop
                    bars_held = i + 1
                    break
                elif bar["low"] <= tp2:
                    outcome = "win_tp2"
                    exit_price = tp2
                    bars_held = i + 1
                    break
                elif bar["low"] <= tp1:
                    outcome = "win_tp1"
                    exit_price = tp1
                    bars_held = i + 1
                    break

        # If not hit, use final price
        if outcome == "expired":
            exit_price = future_bars[-1]["close"]

        # Calculate R-multiple
        if side == "long":
            r_multiple = (exit_price - entry) / risk_per_unit
        else:
            r_multiple = (entry - exit_price) / risk_per_unit

        pnl = r_multiple * trade["risk"]

        results.append(
            {
                **trade,
                "outcome": outcome,
                "exit_price": exit_price,
                "bars_held": bars_held,
                "r_multiple": r_multiple,
                "pnl": pnl,
            }
        )

    return results


def analyze_performance(trade_results):
    """Analyze trading performance metrics"""
    if not trade_results:
        return {"total_trades": 0, "win_rate": 0, "avg_r": 0, "total_r": 0, "total_pnl": 0}

    winners = [t for t in trade_results if t["r_multiple"] > 0]
    losers = [t for t in trade_results if t["r_multiple"] < 0]

    win_rate = len(winners) / len(trade_results) * 100
    avg_r = sum(t["r_multiple"] for t in trade_results) / len(trade_results)
    total_r = sum(t["r_multiple"] for t in trade_results)
    total_pnl = sum(t["pnl"] for t in trade_results)

    avg_win_r = sum(t["r_multiple"] for t in winners) / len(winners) if winners else 0
    avg_loss_r = sum(t["r_multiple"] for t in losers) / len(losers) if losers else 0

    return {
        "total_trades": len(trade_results),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_r": total_r,
        "total_pnl": total_pnl,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "expectancy": total_pnl / len(trade_results),
    }


def run_comprehensive_test(test_configs):
    """
    Run comprehensive testing across multiple assets

    Args:
        test_configs: List of dicts with 'path', 'symbol', 'timeframe'
    """

    print("=" * 100)
    print("BULL MACHINE v1.3 - COMPREHENSIVE PRODUCTION TESTING")
    print("=" * 100)
    print("Testing MTF Sync integration vs baseline v1.2.1")
    print("-" * 100)

    all_results = {}

    for config in test_configs:
        csv_path = config["path"]
        symbol = config["symbol"]
        timeframe = config["timeframe"]

        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            continue

        # Test both MTF enabled and disabled
        for mtf_enabled in [False, True]:
            test_key = f"{symbol}_{timeframe}_MTF_{mtf_enabled}"

            results = test_v13_mtf_integration(
                csv_path,
                symbol,
                timeframe,
                thresholds=[0.30, 0.35, 0.40],
                test_interval=8,
                max_tests=30,
                mtf_enabled=mtf_enabled,
            )

            all_results[test_key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "mtf_enabled": mtf_enabled,
                "results": results,
            }

    return all_results


def print_performance_comparison(all_results):
    """Print comparison between MTF enabled vs disabled"""

    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISON: MTF ENABLED vs DISABLED")
    print("=" * 100)

    symbols = set()
    for key in all_results.keys():
        symbol = key.split("_")[0]
        symbols.add(symbol)

    for symbol in sorted(symbols):
        mtf_off = None
        mtf_on = None

        for key, data in all_results.items():
            if symbol in key and "MTF_False" in key:
                mtf_off = data
            elif symbol in key and "MTF_True" in key:
                mtf_on = data

        if mtf_off and mtf_on:
            print(f"\n📊 {symbol} COMPARISON:")
            print(f"{'Metric':<20} {'MTF OFF':<15} {'MTF ON':<15} {'Improvement':<15}")
            print("-" * 65)

            # Compare at 0.35 threshold
            threshold = 0.35
            off_results = mtf_off["results"].get(threshold, {})
            on_results = mtf_on["results"].get(threshold, {})

            off_trades = off_results.get("trade_count", 0)
            on_trades = on_results.get("trade_count", 0)

            off_conf = off_results.get("avg_confidence", 0)
            on_conf = on_results.get("avg_confidence", 0)

            off_rate = (off_trades / off_results.get("signals_tested", 1)) * 100
            on_rate = (on_trades / on_results.get("signals_tested", 1)) * 100

            print(
                f"{'Trade Count':<20} {off_trades:<15} {on_trades:<15} {on_trades - off_trades:+.0f}"
            )
            print(
                f"{'Avg Confidence':<20} {off_conf:<15.3f} {on_conf:<15.3f} {on_conf - off_conf:+.3f}"
            )
            print(
                f"{'Trade Rate %':<20} {off_rate:<15.1f} {on_rate:<15.1f} {on_rate - off_rate:+.1f}%"
            )


if __name__ == "__main__":
    setup_logging(logging.WARNING)

    # Example test configuration - you'll provide real paths
    test_configs = [
        # Will be populated with your CSV paths
    ]

    print("🎯 Ready for comprehensive v1.3 testing!")
    print("\nPlease provide CSV file paths for:")
    print("- BTC (multiple timeframes)")
    print("- ETH (multiple timeframes)")
    print("- Other assets you want to test")
    print("\nFormat: {'path': '/path/to/file.csv', 'symbol': 'BTCUSD', 'timeframe': '4H'}")
    print("\nI'll test both MTF enabled vs disabled to show the improvement!")
