#!/usr/bin/env python3
"""
Test broker-managed TP/SL system to ensure every entry becomes a round-trip trade.
"""

import sys
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bull_machine.backtest.broker import PaperBroker, TPLevel, Position

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_auto_tp_sl_generation():
    """Test that broker auto-generates TP/SL when not provided."""
    print("\n" + "="*60)
    print("TEST: Auto TP/SL Generation")
    print("="*60)

    broker = PaperBroker()

    # Submit entry without risk plan
    fill = broker.submit(
        ts=pd.Timestamp.now(),
        symbol="TESTBTC",
        side="long",
        size=1.0,
        price_hint=50000.0
    )

    print(f"Entry fill: {fill}")

    # Check that position has auto-generated TP/SL
    pos = broker.positions.get("TESTBTC")
    assert pos is not None, "Position should exist"
    assert pos.stop is not None, "Stop should be auto-generated"
    assert pos.tp_levels is not None, "TP levels should be auto-generated"
    assert len(pos.tp_levels) == 3, "Should have 3 TP levels"

    print(f"✅ Auto-generated stop: {pos.stop}")
    print(f"✅ Auto-generated TPs: {[(tp.price, tp.size_pct, tp.r_multiple) for tp in pos.tp_levels]}")

    return True

def test_stop_loss_execution():
    """Test that stop loss closes position completely."""
    print("\n" + "="*60)
    print("TEST: Stop Loss Execution")
    print("="*60)

    broker = PaperBroker()

    # Open long position
    broker.submit(
        ts=pd.Timestamp.now(),
        symbol="TESTBTC",
        side="long",
        size=1.0,
        price_hint=50000.0
    )

    pos = broker.positions.get("TESTBTC")
    print(f"Position opened: entry={pos.entry}, stop={pos.stop}")

    # Price drops to trigger stop
    stop_price = pos.stop - 100  # Below stop
    fills = broker.mark(pd.Timestamp.now(), "TESTBTC", stop_price)

    print(f"Stop triggered at {stop_price}")
    print(f"Stop fills: {fills}")

    # Position should be closed
    remaining_pos = broker.positions.get("TESTBTC")
    assert remaining_pos is None, "Position should be closed after stop"
    assert fills is not None, "Stop fill should be generated"
    assert fills[0]['side'] == 'stop', "Fill should be marked as stop"
    assert fills[0]['pnl'] < 0, "Stop should generate loss"

    print(f"✅ Stop executed with PnL: {fills[0]['pnl']:.2f}")

    return True

def test_tp_ladder_execution():
    """Test that TP ladder executes properly with breakeven move."""
    print("\n" + "="*60)
    print("TEST: TP Ladder Execution")
    print("="*60)

    broker = PaperBroker()

    # Open long position
    broker.submit(
        ts=pd.Timestamp.now(),
        symbol="TESTBTC",
        side="long",
        size=1.0,
        price_hint=50000.0
    )

    pos = broker.positions.get("TESTBTC")
    original_stop = pos.stop
    tp1_price = pos.tp_levels[0].price
    tp2_price = pos.tp_levels[1].price
    tp3_price = pos.tp_levels[2].price

    print(f"Position: entry={pos.entry}, stop={original_stop}")
    print(f"TP levels: {tp1_price}, {tp2_price}, {tp3_price}")

    # Hit TP1
    fills = broker.mark(pd.Timestamp.now(), "TESTBTC", tp1_price + 10)
    print(f"TP1 hit: {fills}")

    pos = broker.positions.get("TESTBTC")
    assert pos is not None, "Position should still exist after TP1"
    assert pos.be_moved, "Breakeven should be moved"
    assert pos.stop == pos.entry, "Stop should be at breakeven"
    assert fills[0]['side'] == 'tp1', "Should be TP1 fill"

    # Hit TP2
    fills = broker.mark(pd.Timestamp.now(), "TESTBTC", tp2_price + 10)
    print(f"TP2 hit: {fills}")

    pos = broker.positions.get("TESTBTC")
    assert pos is not None, "Position should still exist after TP2"
    assert fills[0]['side'] == 'tp2', "Should be TP2 fill"

    # Hit TP3 - should close completely
    fills = broker.mark(pd.Timestamp.now(), "TESTBTC", tp3_price + 10)
    print(f"TP3 hit: {fills}")

    final_pos = broker.positions.get("TESTBTC")
    assert final_pos is None, "Position should be completely closed after all TPs"

    # Should have TP3 fill and possibly a close_remaining fill
    assert any(fill['side'] == 'tp3' for fill in fills), "Should have TP3 fill"

    print(f"✅ All TPs executed, position closed")

    return True

def test_position_size_tracking():
    """Test that position size reduces correctly with partial TPs."""
    print("\n" + "="*60)
    print("TEST: Position Size Tracking")
    print("="*60)

    broker = PaperBroker()

    # Open position
    broker.submit(
        ts=pd.Timestamp.now(),
        symbol="TESTBTC",
        side="long",
        size=1.0,
        price_hint=50000.0
    )

    pos = broker.positions.get("TESTBTC")
    original_size = pos.size
    tp1_size_pct = pos.tp_levels[0].size_pct

    print(f"Original size: {original_size}")
    print(f"TP1 will close {tp1_size_pct}%")

    # Hit TP1
    tp1_price = pos.tp_levels[0].price
    fills = broker.mark(pd.Timestamp.now(), "TESTBTC", tp1_price + 10)

    pos = broker.positions.get("TESTBTC")
    expected_remaining = original_size * (1 - tp1_size_pct/100)

    print(f"Size after TP1: {pos.size}")
    print(f"Expected remaining: {expected_remaining}")

    assert abs(pos.size - expected_remaining) < 0.001, f"Size should be {expected_remaining}, got {pos.size}"

    print(f"✅ Position size correctly reduced")

    return True

def test_complete_round_trip():
    """Test complete round-trip trade flow."""
    print("\n" + "="*60)
    print("TEST: Complete Round-Trip Trade")
    print("="*60)

    broker = PaperBroker()

    # Track all fills
    all_fills = []

    # Open position
    entry_fill = broker.submit(
        ts=pd.Timestamp.now(),
        symbol="TESTBTC",
        side="long",
        size=1.0,
        price_hint=50000.0
    )
    all_fills.append(entry_fill)

    pos = broker.positions.get("TESTBTC")

    # Simulate price moving through all TP levels
    for i, tp in enumerate(pos.tp_levels):
        fills = broker.mark(pd.Timestamp.now(), "TESTBTC", tp.price + 10)
        if fills:
            all_fills.extend(fills)
            print(f"TP{i+1} executed: {fills}")

    # Position should be closed
    final_pos = broker.positions.get("TESTBTC")
    assert final_pos is None, "Position should be fully closed"

    # Calculate total PnL
    total_pnl = sum(fill.get('pnl', 0) for fill in all_fills if 'pnl' in fill)
    entry_count = sum(1 for fill in all_fills if fill.get('side') in ['long', 'short'])
    exit_count = sum(1 for fill in all_fills if fill.get('side') in ['tp1', 'tp2', 'tp3', 'stop', 'close_remaining'])

    print(f"\nROUND-TRIP SUMMARY:")
    print(f"Total fills: {len(all_fills)}")
    print(f"Entry fills: {entry_count}")
    print(f"Exit fills: {exit_count}")
    print(f"Total PnL: {total_pnl:.2f}")

    assert entry_count > 0, "Should have entry fills"
    assert exit_count > 0, "Should have exit fills"
    assert total_pnl != 0, "Should have non-zero PnL"

    print(f"✅ Complete round-trip trade executed")

    return True

def main():
    """Run all broker tests."""
    print("🧪 BROKER MANAGED TP/SL TESTS")
    print("=" * 80)

    tests = [
        ("Auto TP/SL Generation", test_auto_tp_sl_generation),
        ("Stop Loss Execution", test_stop_loss_execution),
        ("TP Ladder Execution", test_tp_ladder_execution),
        ("Position Size Tracking", test_position_size_tracking),
        ("Complete Round-Trip", test_complete_round_trip)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"\n{test_name}: {'✅ PASSED' if result else '❌ FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n{test_name}: ❌ ERROR - {e}")
            logging.error(f"Test {test_name} failed with error: {e}", exc_info=True)

    print("\n" + "="*80)
    print("BROKER TEST RESULTS")
    print("="*80)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOVERALL: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All broker tests passed! Auto-exit system working.")
        return True
    else:
        print("⚠️ Some broker tests failed. Check logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)