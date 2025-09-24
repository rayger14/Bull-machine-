#!/usr/bin/env python3
"""
Debug script to verify config loading mechanism in calibration sweeps.
"""

import json
import tempfile
import subprocess
from pathlib import Path


def test_config_loading():
    """Test if modified strategy configs are properly loaded."""

    # Create a test strategy config with unique quality floors
    test_config = {
        "fusion": {
            "quality_floors": {
                "wyckoff": 0.99,  # Unique value to test loading
                "liquidity": 0.88,  # Unique value to test loading
                "structure": 0.77,  # Unique value to test loading
                "momentum": 0.12,
                "volume": 0.15,
                "context": 0.15,
            },
            "variance_guards": {"enabled": True, "min_variance": 0.001},
            "debug": {"emit_quality_scores": True, "emit_floor_rejections": True},
        },
        "features": {
            "structure": {"enabled": True, "min_distance_bps": 50, "max_age_bars": 24},
            "wyckoff": {"enabled": True, "lookback": 10, "confidence_threshold": 0.7},
            "liquidity": {"enabled": True, "volume_ma": 20, "price_action_weight": 0.6},
        },
        "mtf": {"enabled": True, "anchor_tf": "4H", "confirm_tf": "1H"},
        "risk": {"base_pct": 0.008, "max_pct": 0.02},
        "exit_rules": {
            "enabled": True,
            "choch_against": {"enabled": True, "swing_lookback": 3, "bars_confirm": 2},
            "momentum_fade": {"enabled": True, "lookback": 6, "drop_pct": 0.20},
            "time_stop": {"enabled": True, "max_bars_1h": 24},
        },
    }

    # Create temp directory and save test config
    temp_dir = tempfile.mkdtemp(prefix="config_debug_")
    temp_dir_path = Path(temp_dir)

    # Write test strategy config
    test_strategy_path = temp_dir_path / "test_strategy.json"
    with open(test_strategy_path, "w") as f:
        json.dump(test_config, f, indent=2)

    print(f"📝 Created test strategy config: {test_strategy_path}")
    print(f"🎯 Test floors: wyckoff=0.99, liquidity=0.88, structure=0.77")

    # Create test backtest config that references the temp strategy
    test_backtest_config = {
        "run_id": "config_debug_test",
        "data": {
            "sources": {
                "BTCUSD_1H": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv"
            },
            "timeframes": ["1H"],
            "schema": {
                "timestamp": {"name": "time", "unit": "s"},
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
        },
        "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2, "partial_fill": True},
        "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.60, "max_positions": 4},
        "engine": {"lookback_bars": 100, "seed": 42},
        "strategy": {
            "version": "v1.4",
            "config": str(test_strategy_path.absolute()),  # ABSOLUTE PATH
        },
        "risk": {"base_risk_pct": 0.008, "max_risk_per_trade": 0.02, "min_stop_pct": 0.001},
        "exit_signals": {
            "enabled": True,
            "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
        },
        "logging": {"level": "DEBUG", "emit_fusion_debug": True, "emit_exit_debug": True},
    }

    # Write test backtest config
    test_backtest_path = temp_dir_path / "test_backtest.json"
    with open(test_backtest_path, "w") as f:
        json.dump(test_backtest_config, f, indent=2)

    print(f"📝 Created test backtest config: {test_backtest_path}")
    print(f"🔗 Strategy config path in backtest: {test_backtest_config['strategy']['config']}")

    # Run a minimal backtest to test config loading
    print("\n🧪 Running debug backtest...")
    result = subprocess.run(
        [
            "python3",
            "-m",
            "bull_machine.app.main_backtest",
            "--config",
            str(test_backtest_path.absolute()),
            "--out",
            str(temp_dir_path / "debug_results"),
            "--debug",  # Enable debug logging
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    print(f"Return code: {result.returncode}")

    # Check for config loading evidence in output
    config_loaded = False
    floors_applied = False

    print("\n🔍 Analyzing output for config loading evidence...")
    for line in result.stdout.split("\n"):
        # Look for strategy config loading
        if "Loaded strategy config from:" in line or "Loading config from" in line:
            config_loaded = True
            print(f"✅ Config loading: {line.strip()}")

        # Look for our unique floor values being applied
        if "0.99" in line or "0.88" in line or "0.77" in line:
            floors_applied = True
            print(f"✅ Floor evidence: {line.strip()}")

        # Look for quality floor rejections
        if "FLOOR" in line or "quality" in line.lower():
            print(f"🎯 Quality info: {line.strip()}")

        # Look for fusion debug output
        if "FUSION" in line or "fusion" in line.lower():
            print(f"🔥 Fusion info: {line.strip()}")

    if result.stderr:
        print(f"\n❌ Stderr output:")
        print(result.stderr[:1000])

    print(f"\n📊 Config Loading Test Results:")
    print(f"   Config loaded: {config_loaded}")
    print(f"   Floors applied: {floors_applied}")
    print(f"   Test status: {'PASS' if config_loaded and floors_applied else 'FAIL'}")

    # Cleanup
    import shutil

    try:
        shutil.rmtree(temp_dir)
        print(f"🗑️  Cleaned up: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Cleanup failed: {e}")


if __name__ == "__main__":
    test_config_loading()
