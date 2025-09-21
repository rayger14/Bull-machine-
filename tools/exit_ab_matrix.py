#!/usr/bin/env python3
"""
Exit A/B Test Matrix - 4 configurations to bracket sensitivity
Tests aggressive exit parameters to achieve 10-40% exit coverage.
"""

import json
import copy
import subprocess
import csv
import time
import pathlib
from pathlib import Path

# Create output directory
OUTDIR = pathlib.Path("reports/exit_ab_matrix")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Load tuned baseline config
with open("bull_machine/configs/exits_tuned.json") as f:
    BASE_CONFIG = json.load(f)

# A/B test matrix - 4 configurations
MATRIX = [
    {"name": "A", "choch_against.bars_confirm": 1, "momentum_fade.drop_pct": 0.20, "time_stop.bars_max": 36},
    {"name": "B", "choch_against.bars_confirm": 1, "momentum_fade.drop_pct": 0.15, "time_stop.bars_max": 36},
    {"name": "C", "choch_against.bars_confirm": 1, "momentum_fade.drop_pct": 0.20, "time_stop.bars_max": 24},
    {"name": "D", "choch_against.bars_confirm": 2, "momentum_fade.drop_pct": 0.20, "time_stop.bars_max": 36},
]

def set_in(dct, dotted, val):
    """Set value in nested dict using dotted path."""
    node = dct
    *path, leaf = dotted.split(".")
    for p in path:
        node = node[p]
    node[leaf] = val

def main():
    print("🔬 Exit A/B Test Matrix - Aggressive Tuning")
    print("=" * 60)
    print("Target: 10-40% exit coverage (exits/total terminations)")
    print("Baseline: 37 trades, 32.4% win rate, -690 expectancy")
    print()

    rows = []
    total_tests = len(MATRIX)

    print(f"Running {total_tests} configurations...")
    print("-" * 60)
    print("Run | CHoCH | Mom% | Time | Status")
    print("-" * 60)

    for params in MATRIX:
        # Create modified config
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["run_id"] = f"exit_ab_{params['name']}"

        # Apply parameter changes
        for k, v in params.items():
            if k != "name":
                set_in(cfg["exit_signals"], k, v)

        # Write config file
        cfg_path = OUTDIR / f"config_{params['name']}.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # Set up output directory
        outdir = OUTDIR / f"run_{params['name']}"

        print(f"{params['name']:^3} | {params['choch_against.bars_confirm']:^5} | "
              f"{params['momentum_fade.drop_pct']:^4.2f} | {params['time_stop.bars_max']:^4} | ", end="", flush=True)

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
                # Parse main results - extract JSON from mixed output
                try:
                    json_line = None
                    for line in result.stdout.split('\n'):
                        if line.strip().startswith('{"ok":'):
                            json_line = line.strip()
                            break

                    if json_line:
                        summary_data = json.loads(json_line)
                        metrics = summary_data.get("metrics", {})
                        success = True
                    else:
                        raise ValueError("No JSON output found")

                except (json.JSONDecodeError, ValueError) as e:
                    metrics = {}
                    success = False
                    print(f"❌ Parse error", flush=True)

                # Parse exit counts
                exit_counts = {}
                try:
                    counts_file = outdir / "exit_counts.json"
                    if counts_file.exists():
                        with open(counts_file) as f:
                            exit_counts = json.load(f)
                except Exception:
                    pass

                if success:
                    exits_total = sum([exit_counts.get(k, 0) for k in ['choch_against', 'momentum_fade', 'time_stop']])
                    trades = metrics.get('trades', 0)
                    exit_coverage = (exits_total / max(trades, 1)) * 100 if trades > 0 else 0

                    print(f"✅ {trades} trades, {metrics.get('win_rate', 0.0):.1%} WR, "
                          f"{exits_total} exits ({exit_coverage:.0f}%)", flush=True)
                else:
                    print(f"❌ Failed", flush=True)

            else:
                print(f"❌ Error", flush=True)
                metrics = {}
                exit_counts = {}
                elapsed = time.time() - start
                success = False

        except subprocess.TimeoutExpired:
            print(f"❌ Timeout", flush=True)
            metrics = {}
            exit_counts = {}
            elapsed = 120
            success = False

        # Record results
        exits_total = sum([exit_counts.get(k, 0) for k in ['choch_against', 'momentum_fade', 'time_stop']])
        trades = metrics.get('trades', 0)
        exit_coverage = (exits_total / max(trades, 1)) * 100 if trades > 0 else 0

        row = {
            "run": params["name"],
            "choch_bars": params["choch_against.bars_confirm"],
            "mom_drop_pct": params["momentum_fade.drop_pct"],
            "time_bars_max": params["time_stop.bars_max"],
            "elapsed_s": round(elapsed, 2),
            "success": success,
            "trades": trades,
            "win_rate": metrics.get("win_rate", 0.0),
            "expectancy": metrics.get("expectancy", 0.0),
            "max_dd": metrics.get("max_dd", 0.0),
            "sharpe": metrics.get("sharpe", 0.0),
            "exits_total": exits_total,
            "exit_coverage_pct": exit_coverage,
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

    # Analysis
    print("\n" + "=" * 60)
    print("📊 A/B MATRIX RESULTS")
    print("=" * 60)

    # Write CSV
    csv_path = OUTDIR / "exit_ab_matrix.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print(f"Saved {len(rows)} results to {csv_path}\n")

    # Filter successful tests
    successful_tests = [r for r in rows if r.get("success", False)]

    if not successful_tests:
        print("❌ No successful tests!")
        return

    # Baseline comparison
    baseline_metrics = {"trades": 37, "win_rate": 0.324, "expectancy": -690, "max_dd": 110199}

    print("📈 PERFORMANCE vs BASELINE")
    print("-" * 80)
    print(f"{'Run':<4} {'Trades':<7} {'WinRate':<8} {'Δ WR':<7} {'Expect':<10} {'Δ Exp':<8} "
          f"{'MaxDD':<10} {'Δ DD%':<7} {'Exits':<6} {'Coverage':<8}")
    print("-" * 80)

    for result in successful_tests:
        wr_delta = (result['win_rate'] - baseline_metrics['win_rate']) * 100
        exp_delta = result['expectancy'] - baseline_metrics['expectancy']
        dd_delta = ((result['max_dd'] - baseline_metrics['max_dd']) / baseline_metrics['max_dd']) * 100

        print(f"{result['run']:<4} {result['trades']:<7} {result['win_rate']:<8.1%} "
              f"{wr_delta:+6.1f}pp {result['expectancy']:<10.0f} {exp_delta:+7.0f} "
              f"{result['max_dd']:<10.0f} {dd_delta:+6.1f}% {result['exits_total']:<6} "
              f"{result['exit_coverage_pct']:<7.1f}%")

    # Find configurations with meaningful exit activity
    active_exits = [r for r in successful_tests if r["exits_total"] > 0]

    if active_exits:
        print(f"\n🎯 EXITS BREAKDOWN ({len(active_exits)} configs with exits)")
        print("-" * 60)
        print(f"{'Run':<4} {'Total':<7} {'CHoCH':<7} {'Momentum':<10} {'TimeStop':<9} {'None':<6}")
        print("-" * 60)

        for result in active_exits:
            print(f"{result['run']:<4} {result['exits_total']:<7} {result['exits_choch']:<7} "
                  f"{result['exits_mom']:<10} {result['exits_time']:<9} {result['exits_none']:<6}")

        # Find best by exit coverage in target range
        in_range = [r for r in active_exits if 10 <= r['exit_coverage_pct'] <= 40]
        if in_range:
            best = max(in_range, key=lambda x: x['expectancy'])
            print(f"\n🏆 RECOMMENDED: Run {best['run']}")
            print(f"   Exit Coverage: {best['exit_coverage_pct']:.1f}% (TARGET: 10-40%)")
            print(f"   Performance: {best['trades']} trades, {best['win_rate']:.1%} win rate")
            print(f"   Lift vs baseline: {(best['win_rate']-baseline_metrics['win_rate'])*100:+.1f}pp WR, "
                  f"{best['expectancy']-baseline_metrics['expectancy']:+.0f} expectancy")
            print(f"   Exit breakdown: CHoCH={best['exits_choch']}, Mom={best['exits_mom']}, Time={best['exits_time']}")
    else:
        print("\n⚠️  No configurations triggered exits - need more aggressive parameters")
        print("\n📋 NEXT STEPS:")
        print("   1. Lower momentum_fade.drop_pct to 0.15")
        print("   2. Lower choch_against.min_break_strength to 0.03")
        print("   3. Lower time_stop.bars_max to 24")

    print(f"\n💾 Full results: {csv_path}")

if __name__ == "__main__":
    main()