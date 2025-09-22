#!/usr/bin/env python3
"""
Quick test to debug quality gates with detailed logging.
"""

import json
import tempfile
import subprocess
from pathlib import Path

def test_quality_gates_debug():
    """Test quality gates with detailed fusion logging."""

    # Create test strategy config with moderate floors
    test_config = {
        "fusion": {
            "quality_floors": {
                "wyckoff": 0.30,     # Low floor to allow signals through
                "liquidity": 0.25,   # Low floor to allow signals through
                "structure": 0.35,   # Low floor to allow signals through
                "momentum": 0.12,
                "volume": 0.15,
                "context": 0.15
            },
            "variance_guards": {"enabled": True, "min_variance": 0.001},
            "debug": {"emit_quality_scores": True, "emit_floor_rejections": True}
        },
        "features": {
            "structure": {"enabled": True, "min_distance_bps": 50, "max_age_bars": 24},
            "wyckoff": {"enabled": True, "lookback": 10, "confidence_threshold": 0.7},
            "liquidity": {"enabled": True, "volume_ma": 20, "price_action_weight": 0.6}
        },
        "mtf": {"enabled": True, "anchor_tf": "4H", "confirm_tf": "1H"},
        "risk": {"base_pct": 0.008, "max_pct": 0.02},
        "exit_rules": {
            "enabled": True,
            "choch_against": {"enabled": True, "swing_lookback": 3, "bars_confirm": 2},
            "momentum_fade": {"enabled": True, "lookback": 6, "drop_pct": 0.20},
            "time_stop": {"enabled": True, "max_bars_1h": 24}
        }
    }

    # Create temp directory and save test config
    temp_dir = tempfile.mkdtemp(prefix="quality_debug_")
    temp_dir_path = Path(temp_dir)

    # Write test strategy config
    test_strategy_path = temp_dir_path / "test_strategy.json"
    with open(test_strategy_path, 'w') as f:
        json.dump(test_config, f, indent=2)

    print(f"📝 Created test strategy config: {test_strategy_path}")
    print(f"🎯 Test floors: wyckoff=0.30, liquidity=0.25, structure=0.35")

    # Create test backtest config that references the temp strategy
    test_backtest_config = {
        "run_id": "quality_gates_debug",
        "data": {
            "sources": {
                "BTCUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv"
            },
            "timeframes": ["1H"],
            "schema": {
                "timestamp": {"name": "time", "unit": "s"},
                "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"
            }
        },
        "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2, "partial_fill": True},
        "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.60, "max_positions": 4},
        "engine": {"lookback_bars": 100, "seed": 42},
        "strategy": {
            "version": "v1.4",
            "config": str(test_strategy_path.absolute())  # ABSOLUTE PATH
        },
        "risk": {"base_risk_pct": 0.008, "max_risk_per_trade": 0.02, "min_stop_pct": 0.001},
        "exit_signals": {"enabled": True, "enabled_exits": ["choch_against", "momentum_fade", "time_stop"]},
        "logging": {"level": "INFO", "emit_fusion_debug": True, "emit_exit_debug": True}  # INFO level to see quality gates
    }

    # Write test backtest config
    test_backtest_path = temp_dir_path / "test_backtest.json"
    with open(test_backtest_path, 'w') as f:
        json.dump(test_backtest_config, f, indent=2)

    print(f"📝 Created test backtest config: {test_backtest_path}")

    # Run a brief backtest to test quality gates
    print(f"\n🧪 Running quality gates debug test...")
    result = subprocess.run([
        "python3", "-m", "bull_machine.app.main_backtest",
        "--config", str(test_backtest_path.absolute()),
        "--out", str(temp_dir_path / "debug_results"),
        "--debug"  # Enable debug logging
    ], capture_output=True, text=True, timeout=60)

    print(f"Return code: {result.returncode}")

    # Look for quality gate evidence in output
    quality_floors_logged = False
    gates_applied = False
    layers_kept = False
    layers_masked = False

    print(f"\n🔍 Analyzing output for quality gate evidence...")
    lines = result.stdout.split('\n')

    for line in lines:
        # Look for quality floors config
        if "🎯 QUALITY_FLOORS:" in line:
            quality_floors_logged = True
            print(f"✅ Quality floors config: {line.strip()}")

        # Look for layer kept/masked decisions
        if "✅ KEPT" in line:
            layers_kept = True
            print(f"✅ Layer kept: {line.strip()}")

        if "❌ MASKED" in line:
            layers_masked = True
            print(f"❌ Layer masked: {line.strip()}")

        # Look for insufficient layers message
        if "Insufficient quality layers:" in line:
            gates_applied = True
            print(f"🚫 Gates applied: {line.strip()}")

    if result.stderr:
        print(f"\n❌ Stderr output:")
        print(result.stderr[:500])

    print(f"\n📊 Quality Gates Test Results:")
    print(f"   Quality floors logged: {quality_floors_logged}")
    print(f"   Gates applied: {gates_applied}")
    print(f"   Layers kept: {layers_kept}")
    print(f"   Layers masked: {layers_masked}")

    status = "PASS" if quality_floors_logged and (layers_kept or layers_masked) else "FAIL"
    print(f"   Test status: {status}")

    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"🗑️  Cleaned up: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")

if __name__ == "__main__":
    test_quality_gates_debug()