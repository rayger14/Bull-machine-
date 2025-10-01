#!/usr/bin/env python3
"""
Telemetry Test: Verify enhanced exit parameter telemetry

Runs a quick test to ensure the new telemetry files are generated:
- exits_applied.json
- parameter_usage.json
- layer_masks.json
- exit_counts.json (legacy)
- exit_cfg_applied.json (config dump)
"""

import subprocess
import tempfile
import json
from pathlib import Path


def test_telemetry():
    """Test the enhanced telemetry system."""

    # Create test config with telemetry enabled
    config = {
        "run_id": "telemetry_test",
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
        "engine": {"lookback_bars": 50, "seed": 42},
        "strategy": {
            "version": "v1.4",
            "config": "bull_machine/configs/diagnostic_v14_step4_config.json",
        },
        "risk": {"base_risk_pct": 0.008, "max_risk_per_trade": 0.02, "min_stop_pct": 0.001},
        "exit_signals": {
            "enabled": True,
            "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
            "emit_exit_edge_logs": True,
            "choch_against": {
                "swing_lookback": 3,
                "bars_confirm": 1,  # Aggressive for testing
                "min_break_strength": 0.02,
            },
            "momentum_fade": {
                "lookback": 6,
                "drop_pct": 0.15,  # Aggressive for testing
                "min_bars_in_pos": 2,
            },
            "time_stop": {
                "max_bars_1h": 12,  # Short for testing
                "max_bars_4h": 8,
                "max_bars_1d": 4,
            },
            "emit_exit_debug": True,
        },
        "logging": {"level": "INFO", "emit_fusion_debug": False, "emit_exit_debug": True},
    }

    # Create temporary files
    temp_dir = tempfile.mkdtemp(prefix="telemetry_test_")
    config_path = Path(temp_dir) / "test_config.json"
    result_dir = Path(temp_dir) / "results"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("🧪 TESTING ENHANCED EXIT TELEMETRY")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {result_dir}")
    print()

    try:
        # Run backtest
        result = subprocess.run(
            [
                "python3",
                "-m",
                "bull_machine.app.main_backtest",
                "--config",
                str(config_path),
                "--out",
                str(result_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode == 0:
            print("✅ Backtest completed successfully")

            # Check telemetry files
            expected_files = [
                "exits_applied.json",
                "parameter_usage.json",
                "layer_masks.json",
                "exit_counts.json",
                "exit_cfg_applied.json",
            ]

            print(f"\n🔍 CHECKING TELEMETRY FILES:")
            for filename in expected_files:
                file_path = result_dir / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    print(f"  ✅ {filename} ({file_size} bytes)")

                    # Show sample content for key files
                    if filename in ["parameter_usage.json", "exits_applied.json"]:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        print(f"     Preview: {list(data.keys())}")
                else:
                    print(f"  ❌ {filename} - NOT FOUND")

            # Show detailed parameter usage
            param_usage_path = result_dir / "parameter_usage.json"
            if param_usage_path.exists():
                with open(param_usage_path, "r") as f:
                    param_data = json.load(f)

                print(f"\n📊 PARAMETER EFFECTIVENESS:")
                for exit_type, data in param_data["parameter_effectiveness"].items():
                    print(f"  {exit_type}:")
                    print(f"    Evaluated: {data['evaluated_count']}")
                    print(f"    Triggered: {data['triggered_count']}")
                    print(f"    Rate: {data['trigger_rate']:.1%}")
                    print(f"    Params: {list(data['parameters_applied'].keys())}")

            # Show layer masks summary
            layer_masks_path = result_dir / "layer_masks.json"
            if layer_masks_path.exists():
                with open(layer_masks_path, "r") as f:
                    layer_data = json.load(f)

                print(f"\n🎭 FUSION LAYERS:")
                for layer, info in layer_data["fusion_layers"].items():
                    print(f"  {layer}: Active={info['active']}, Triggered={info['trigger_count']}")

                print(
                    f"  Fusion Effectiveness: {layer_data['layer_interaction']['fusion_effectiveness']:.1%}"
                )

            print(f"\n🎉 TELEMETRY TEST COMPLETE")
            print(f"All enhanced telemetry files generated successfully!")

        else:
            print(f"❌ Backtest failed with code {result.returncode}")
            print(f"Error: {result.stderr[:500]}")

    except subprocess.TimeoutExpired:
        print(f"❌ Test timeout")
    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        # Cleanup (optional - leave for inspection)
        print(f"\n📁 Results saved in: {result_dir}")
        print(f"Config used: {config_path}")


if __name__ == "__main__":
    test_telemetry()
