#!/usr/bin/env python3
"""
Exit Parameter Sweep - 3x3x3 Grid Calibration

Implements surgical exit calibration:
- choch_against.bars_confirm ∈ {1,2,3}
- momentum_fade.drop_pct ∈ {0.25, 0.35, 0.45}
- time_stop.bars_max ∈ {48,72,96}

Records exit metrics, counts, and performance for each configuration.
"""

import json
import copy
import subprocess
import csv
import time
import pathlib
from pathlib import Path

# Create output directory
OUTDIR = pathlib.Path("reports/exit_sweep")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load working baseline config and update it for sweep
with open("bull_machine/configs/_baseline_data.json") as f:
    BASE_CONFIG = json.load(f)

# Update for exit sweep
BASE_CONFIG.update({
    "run_id": "exit_sweep_base",
    "broker": {
        "fee_bps": 10,
        "slippage_bps": 5,
        "spread_bps": 2,
        "partial_fill": True
    },
    "portfolio": {
        "starting_cash": 100000,
        "exposure_cap_pct": 0.60,
        "max_positions": 8
    },
    "engine": {
        "lookback_bars": 100,
        "seed": 42
    },
    "strategy": {
        "version": "v1.4",
        "config": "bull_machine/configs/diagnostic_v14_step4_config.json"
    },
    "risk": {
        "base_risk_pct": 0.008,
        "max_risk_per_trade": 0.02,
        "min_stop_pct": 0.0008,
        "tp_ladder": {
            "tp1": {"r_multiple": 1.0, "size_pct": 33, "action": "move_stop_to_breakeven"},
            "tp2": {"r_multiple": 2.0, "size_pct": 33, "action": "trail_stop"},
            "tp3": {"r_multiple": 3.0, "size_pct": 34, "action": "hold"}
        }
    },
    "exit_signals": {
        "enabled": True,
        "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
        "choch_against": {
            "ltf_tf": "1H",
            "mtf_tf": "4H",
            "bars_confirm": 2,
            "swing_lookback": 3,
            "use_intrabar_wick_break": True,
            "min_break_strength": 0.10,
            "min_volume_factor": 0.80,
            "priority": 100
        },
        "momentum_fade": {
            "lookback": 6,
            "compare_window": 6,
            "min_body_atr": 0.20,
            "drop_pct": 0.35,
            "min_rsi_div": 4.0,
            "action": "PARTIAL_33",
            "priority": 60
        },
        "time_stop": {
            "bars_max": 72,
            "grace_bars_after_tp1": 12,
            "tighten_stop_to": 0.50,
            "priority": 10
        }
    },
    "logging": {
        "level": "INFO",
        "emit_fusion_debug": True,
        "emit_exit_debug": True
    }
})

# Parameter grid
GRID = [
    {"choch_against.bars_confirm": b, "momentum_fade.drop_pct": d, "time_stop.bars_max": t}
    for b in (1, 2, 3)
    for d in (0.25, 0.35, 0.45)
    for t in (48, 72, 96)
]

def set_in(dct, dotted, val):
    """Set value in nested dict using dotted path."""
    node = dct
    *path, leaf = dotted.split(".")
    for p in path:
        node = node[p]
    node[leaf] = val

def main():
    print("🔬 Exit Parameter Sweep - 3x3x3 Grid Calibration")
    print("=" * 50)

    rows = []
    total_tests = len(GRID)

    print(f"Running {total_tests} configurations...")
    print("choch_against.bars_confirm: [1, 2, 3]")
    print("momentum_fade.drop_pct: [0.25, 0.35, 0.45]")
    print("time_stop.bars_max: [48, 72, 96]")
    print()

    for i, params in enumerate(GRID, 1):
        # Create modified config
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["run_id"] = f"exit_sweep_{i:02d}"

        # Apply parameter changes
        for k, v in params.items():
            set_in(cfg["exit_signals"], k, v)

        # Write config file
        cfg_path = OUTDIR / f"run_{i:02d}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Set up output directory
        outdir = OUTDIR / f"o_{i:02d}"

        print(f"🧪 Run {i:02d}/{total_tests}: bars_confirm={params['choch_against.bars_confirm']}, "
              f"drop_pct={params['momentum_fade.drop_pct']:.2f}, "
              f"bars_max={params['time_stop.bars_max']}")

        # Run backtest
        cmd = [
            "python3", "-m", "bull_machine.app.main_backtest",
            "--config", str(cfg_path),
            "--out", str(outdir)
        ]

        start = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            elapsed = time.time() - start

            if result.returncode == 0:
                # Parse main results
                try:
                    summary_data = json.loads(result.stdout.strip())
                    metrics = summary_data.get("metrics", {})
                    success = True
                except json.JSONDecodeError:
                    metrics = {}
                    success = False
                    print(f"❌ Failed to parse JSON output")

                # Parse exit counts
                exit_counts = {}
                try:
                    counts_file = outdir / "exit_counts.json"
                    if counts_file.exists():
                        with open(counts_file) as f:
                            exit_counts = json.load(f)
                except Exception as e:
                    print(f"⚠️  Could not read exit counts: {e}")

                if success:
                    print(f"✅ {metrics.get('trades', 0)} trades, "
                          f"{metrics.get('win_rate', 0.0):.1%} win rate, "
                          f"{metrics.get('expectancy', 0.0):.0f} expectancy, "
                          f"exits: {sum(exit_counts.values()) if exit_counts else 0}")
                else:
                    print(f"❌ Parse error")

            else:
                print(f"❌ Backtest failed: {result.stderr}")
                metrics = {}
                exit_counts = {}
                elapsed = time.time() - start
                success = False

        except subprocess.TimeoutExpired:
            print(f"❌ Test timeout")
            metrics = {}
            exit_counts = {}
            elapsed = 120
            success = False

        # Record results
        row = {
            **params,
            "elapsed_s": round(elapsed, 2),
            "success": success,
            "trades": metrics.get("trades", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "expectancy": metrics.get("expectancy", 0.0),
            "max_dd": metrics.get("max_dd", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "exits_total": sum(exit_counts.values()) if exit_counts else 0,
            "exits_choch": exit_counts.get("choch_against", 0),
            "exits_mom": exit_counts.get("momentum_fade", 0),
            "exits_time": exit_counts.get("time_stop", 0),
            "exits_none": exit_counts.get("none", 0)
        }
        rows.append(row)

        # Cleanup temp config
        try:
            cfg_path.unlink()
        except:
            pass

    # Save results
    print("\\n" + "=" * 50)
    print("📊 EXIT SWEEP RESULTS")
    print("=" * 50)

    # Write CSV
    csv_path = OUTDIR / "exit_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print(f"Saved {len(rows)} rows to {csv_path}")

    # Filter successful tests
    successful_tests = [r for r in rows if r.get("success", False)]

    if not successful_tests:
        print("❌ No successful tests!")
        return

    # Sort by expectancy (best first)
    successful_tests.sort(key=lambda x: x["expectancy"], reverse=True)

    print(f"\\n🎯 TOP 5 CONFIGURATIONS (by expectancy):")
    print("-" * 100)
    print(f"{'Rank':<4} {'Bars':<5} {'Drop%':<6} {'TimeMax':<7} {'Trades':<6} {'WinRate':<8} "
          f"{'Expectancy':<10} {'MaxDD':<8} {'ExitsTotal':<10}")
    print("-" * 100)

    for i, result in enumerate(successful_tests[:5]):
        print(f"{i+1:<4} {result['choch_against.bars_confirm']:<5} "
              f"{result['momentum_fade.drop_pct']:<6.2f} {result['time_stop.bars_max']:<7} "
              f"{result['trades']:<6} {result['win_rate']:<8.1%} "
              f"{result['expectancy']:<10.0f} {result['max_dd']:<8.0f} {result['exits_total']:<10}")

    # Find configurations with meaningful exit activity
    active_exits = [r for r in successful_tests if r["exits_total"] > 0]

    if active_exits:
        print(f"\\n🎯 ACTIVE EXIT CONFIGURATIONS ({len(active_exits)} total):")
        print("-" * 100)
        for i, result in enumerate(active_exits[:3]):
            exit_pct = (result['exits_total'] / max(result['trades'], 1)) * 100
            print(f"{i+1:<4} {result['choch_against.bars_confirm']:<5} "
                  f"{result['momentum_fade.drop_pct']:<6.2f} {result['time_stop.bars_max']:<7} "
                  f"{result['trades']:<6} {result['win_rate']:<8.1%} "
                  f"{result['expectancy']:<10.0f} {exit_pct:<8.1f}% "
                  f"C:{result['exits_choch']} M:{result['exits_mom']} T:{result['exits_time']}")

        best_active = active_exits[0]
        print(f"\\n🏆 RECOMMENDED ACTIVE CONFIGURATION:")
        print(f"   choch_against.bars_confirm: {best_active['choch_against.bars_confirm']}")
        print(f"   momentum_fade.drop_pct: {best_active['momentum_fade.drop_pct']:.2f}")
        print(f"   time_stop.bars_max: {best_active['time_stop.bars_max']}")
        print(f"   Performance: {best_active['trades']} trades, {best_active['win_rate']:.1%} win rate, "
              f"{best_active['expectancy']:.0f} expectancy")
        print(f"   Exit breakdown: CHoCH={best_active['exits_choch']}, "
              f"Momentum={best_active['exits_mom']}, Time={best_active['exits_time']}")
    else:
        print("⚠️  No configurations showed exit activity - parameters too conservative")
        print("📋 Using best expectancy regardless of exits:")
        best_overall = successful_tests[0]
        print(f"   choch_against.bars_confirm: {best_overall['choch_against.bars_confirm']}")
        print(f"   momentum_fade.drop_pct: {best_overall['momentum_fade.drop_pct']:.2f}")
        print(f"   time_stop.bars_max: {best_overall['time_stop.bars_max']}")

    print(f"\\n💾 Detailed results saved to: {csv_path}")

if __name__ == "__main__":
    main()