#!/usr/bin/env python3
"""
Unit tests for Router v10

Tests all decision logic paths, edge cases, and telemetry.
"""

import sys
from pathlib import Path
import pandas as pd
import json
import tempfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from engine.router_v10 import RouterV10


def test_confidence_veto():
    """Test that low confidence triggers CASH."""
    router = RouterV10()

    # Test below threshold
    decision = router.select_config(
        timestamp=pd.Timestamp('2024-01-15 14:00:00', tz='UTC'),
        regime_label='risk_on',
        regime_confidence=0.55,
        event_flag=False
    )

    assert decision['action'] == 'CASH', "Low confidence should trigger CASH"
    assert 'low_confidence' in decision['reason']

    # Test at threshold (should pass)
    decision = router.select_config(
        timestamp=pd.Timestamp('2024-01-15 14:00:00', tz='UTC'),
        regime_label='risk_on',
        regime_confidence=0.60,
        event_flag=False
    )

    assert decision['action'] != 'CASH', "Confidence at threshold should allow trade"

    print("✅ Confidence veto test passed")


def test_event_suppression():
    """Test that event windows trigger CASH."""
    router = RouterV10()

    # Test with event suppression enabled (default)
    decision = router.select_config(
        timestamp=pd.Timestamp('2024-01-15 14:00:00', tz='UTC'),
        regime_label='risk_on',
        regime_confidence=0.85,
        event_flag=True
    )

    assert decision['action'] == 'CASH', "Event window should trigger CASH"
    assert decision['reason'] == 'event_suppression'

    # Test with event suppression disabled
    router_no_suppress = RouterV10(event_suppression=False)
    decision = router_no_suppress.select_config(
        timestamp=pd.Timestamp('2024-01-15 14:00:00', tz='UTC'),
        regime_label='risk_on',
        regime_confidence=0.85,
        event_flag=True
    )

    assert decision['action'] != 'CASH', "Event flag should be ignored when suppression disabled"

    print("✅ Event suppression test passed")


def test_regime_selection():
    """Test that regime correctly determines config."""
    router = RouterV10()
    ts = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    # Risk-off → Bear
    decision = router.select_config(
        timestamp=ts,
        regime_label='risk_off',
        regime_confidence=0.75,
        event_flag=False
    )
    assert decision['action'] == 'BEAR', "risk_off should select BEAR config"

    # Crisis → Bear
    decision = router.select_config(
        timestamp=ts,
        regime_label='crisis',
        regime_confidence=0.80,
        event_flag=False
    )
    assert decision['action'] == 'BEAR', "crisis should select BEAR config"

    # Risk-on → Bull
    decision = router.select_config(
        timestamp=ts,
        regime_label='risk_on',
        regime_confidence=0.75,
        event_flag=False
    )
    assert decision['action'] == 'BULL', "risk_on should select BULL config"

    # Neutral → Bull
    decision = router.select_config(
        timestamp=ts,
        regime_label='neutral',
        regime_confidence=0.75,
        event_flag=False
    )
    assert decision['action'] == 'BULL', "neutral should select BULL config"

    print("✅ Regime selection test passed")


def test_decision_priority():
    """Test that decision rules apply in correct priority."""
    router = RouterV10()
    ts = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    # Confidence veto should take priority over regime
    decision = router.select_config(
        timestamp=ts,
        regime_label='risk_on',
        regime_confidence=0.50,
        event_flag=False
    )
    assert decision['action'] == 'CASH', "Confidence veto should override regime"

    # Event suppression should take priority over regime (but after confidence)
    decision = router.select_config(
        timestamp=ts,
        regime_label='risk_on',
        regime_confidence=0.85,
        event_flag=True
    )
    assert decision['action'] == 'CASH', "Event suppression should override regime"

    # Both vetoes active - confidence should be reported
    decision = router.select_config(
        timestamp=ts,
        regime_label='risk_on',
        regime_confidence=0.50,
        event_flag=True
    )
    assert decision['action'] == 'CASH'
    assert 'low_confidence' in decision['reason'], "Confidence veto checked first"

    print("✅ Decision priority test passed")


def test_hysteresis():
    """Test regime hysteresis to prevent whipsaw."""
    # No hysteresis (default)
    router = RouterV10(hysteresis_bars=0)
    ts = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    decision1 = router.select_config(ts, 'risk_on', 0.75, False)
    assert decision1['action'] == 'BULL'

    decision2 = router.select_config(ts, 'risk_off', 0.75, False)
    assert decision2['action'] == 'BEAR', "Should switch immediately with no hysteresis"

    # With hysteresis = 2 bars
    router_hyst = RouterV10(hysteresis_bars=2)

    # Start in risk_on
    decision1 = router_hyst.select_config(ts, 'risk_on', 0.75, False)
    assert decision1['action'] == 'BULL'

    # First bar of risk_off (hysteresis not met)
    decision2 = router_hyst.select_config(ts, 'risk_off', 0.75, False)
    assert decision2['action'] == 'BULL', "Should stay BULL (hysteresis bar 1)"

    # Second bar of risk_off (hysteresis met)
    decision3 = router_hyst.select_config(ts, 'risk_off', 0.75, False)
    assert decision3['action'] == 'BEAR', "Should switch to BEAR (hysteresis met)"

    # Back to risk_on immediately (resets counter)
    decision4 = router_hyst.select_config(ts, 'risk_on', 0.75, False)
    assert decision4['action'] == 'BEAR', "Should stay BEAR (reset counter)"

    print("✅ Hysteresis test passed")


def test_config_loading():
    """Test that configs are loaded and validated."""
    router = RouterV10()

    # Verify configs loaded
    assert router.bull_config is not None
    assert router.bear_config is not None

    # Verify config structure
    assert 'fusion' in router.bull_config
    assert 'fusion' in router.bear_config
    assert 'entry_threshold_confidence' in router.bull_config['fusion']
    assert 'entry_threshold_confidence' in router.bear_config['fusion']

    # Verify different configs (bull is aggressive, bear is defensive)
    bull_threshold = router.bull_config['fusion']['entry_threshold_confidence']
    bear_threshold = router.bear_config['fusion']['entry_threshold_confidence']
    assert bull_threshold < bear_threshold, \
        f"Bull threshold ({bull_threshold}) should be lower than bear ({bear_threshold})"

    print("✅ Config loading test passed")


def test_invalid_configs():
    """Test that invalid config paths raise errors."""
    try:
        router = RouterV10(
            bull_config_path='nonexistent_bull.json',
            bear_config_path='configs/v10_bases/btc_bear_v10_best.json'
        )
        assert False, "Should raise FileNotFoundError for missing bull config"
    except FileNotFoundError:
        pass

    try:
        router = RouterV10(
            bull_config_path='configs/v10_bases/btc_bull_v10_best.json',
            bear_config_path='nonexistent_bear.json'
        )
        assert False, "Should raise FileNotFoundError for missing bear config"
    except FileNotFoundError:
        pass

    print("✅ Invalid config test passed")


def test_telemetry():
    """Test decision logging and statistics."""
    router = RouterV10()
    ts = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    # Make several decisions
    router.select_config(ts, 'risk_on', 0.85, False)   # BULL
    router.select_config(ts, 'risk_off', 0.75, False)  # BEAR
    router.select_config(ts, 'risk_on', 0.50, False)   # CASH (low conf)
    router.select_config(ts, 'risk_on', 0.85, True)    # CASH (event)

    # Check decision history
    assert len(router.decision_history) == 4

    # Check stats
    stats = router.get_stats()
    assert stats['total_decisions'] == 4
    assert stats['action_distribution']['BULL'] == 25.0
    assert stats['action_distribution']['BEAR'] == 25.0
    assert stats['action_distribution']['CASH'] == 50.0

    # Check confidence stats
    assert stats['confidence_stats']['min'] == 0.50
    assert stats['confidence_stats']['max'] == 0.85

    print("✅ Telemetry test passed")


def test_decision_export():
    """Test exporting decision log to JSON."""
    router = RouterV10()
    ts = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    # Make some decisions
    router.select_config(ts, 'risk_on', 0.85, False)
    router.select_config(ts, 'risk_off', 0.75, False)

    # Export to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    router.export_decision_log(temp_path)

    # Verify file exists and is valid JSON
    with open(temp_path, 'r') as f:
        log_data = json.load(f)

    assert len(log_data) == 2
    assert 'timestamp' in log_data[0]
    assert 'action' in log_data[0]
    assert 'reason' in log_data[0]

    # Clean up
    Path(temp_path).unlink()

    print("✅ Decision export test passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    router = RouterV10()
    ts = pd.Timestamp('2024-01-15 14:00:00', tz='UTC')

    # Confidence exactly at threshold
    decision = router.select_config(ts, 'risk_on', 0.60, False)
    assert decision['action'] == 'BULL', "Confidence at threshold should pass"

    # Confidence just below threshold
    decision = router.select_config(ts, 'risk_on', 0.5999, False)
    assert decision['action'] == 'CASH', "Confidence below threshold should veto"

    # Confidence = 1.0 (perfect)
    decision = router.select_config(ts, 'risk_on', 1.0, False)
    assert decision['action'] == 'BULL'

    # Confidence = 0.0 (minimum)
    decision = router.select_config(ts, 'risk_on', 0.0, False)
    assert decision['action'] == 'CASH'

    print("✅ Edge cases test passed")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*80)
    print("ROUTER V10 - UNIT TESTS")
    print("="*80)

    tests = [
        test_confidence_veto,
        test_event_suppression,
        test_regime_selection,
        test_decision_priority,
        test_hysteresis,
        test_config_loading,
        test_invalid_configs,
        test_telemetry,
        test_decision_export,
        test_edge_cases
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"\nRunning {test_func.__name__}...")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "="*80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
