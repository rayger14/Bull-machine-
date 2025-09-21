import json, copy, time, pathlib, sys, os
from statistics import mean
from typing import List, Dict, Any
import subprocess

# Add bull_machine to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

RANGE = [round(x, 2) for x in [0.15 + 0.02*i for i in range(16)]]  # 0.15..0.45

def run_backtest_subprocess(cfg_path: str, out_dir: str) -> Dict[str, Any]:
    """Run backtest as subprocess and parse results"""
    cmd = [
        "python3", "-m", "bull_machine.app.main_backtest",
        "--config", cfg_path,
        "--out", out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=pathlib.Path(__file__).parent.parent.parent)

    if result.returncode != 0:
        print(f"Backtest failed: {result.stderr}")
        return {"metrics": {}, "trades": []}

    # Try to parse JSON output from stdout
    try:
        # Look for the JSON output line
        lines = result.stdout.strip().split('\n')
        json_line = None
        for line in lines:
            if line.startswith('{"ok":'):
                json_line = line
                break

        if json_line:
            backtest_result = json.loads(json_line)

            # Load trades and equity files if they exist
            artifacts = backtest_result.get("artifacts", {})
            trades_path = artifacts.get("trades")
            summary_path = artifacts.get("summary")

            trades = []
            if trades_path and pathlib.Path(trades_path).exists():
                import pandas as pd
                try:
                    trades_df = pd.read_csv(trades_path)
                    trades = trades_df.to_dict('records')
                except:
                    pass

            metrics = backtest_result.get("metrics", {})
            if summary_path and pathlib.Path(summary_path).exists():
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        metrics = summary.get("metrics", metrics)
                except:
                    pass

            return {"metrics": metrics, "trades": trades}
        else:
            print(f"No JSON output found in: {result.stdout}")
            return {"metrics": {}, "trades": []}

    except Exception as e:
        print(f"Failed to parse backtest output: {e}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return {"metrics": {}, "trades": []}

def run_once(base_cfg: Dict[str, Any], thr: float, tag: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # Read the strategy config and modify threshold
    strategy_config_path = cfg.get("strategy", {}).get("config", "config/threshold_calibration.json")
    strategy_config_full_path = pathlib.Path(strategy_config_path)
    if not strategy_config_full_path.exists():
        # Try relative to repo root
        strategy_config_full_path = pathlib.Path(__file__).parent.parent.parent / strategy_config_path

    if strategy_config_full_path.exists():
        strategy_cfg = json.loads(strategy_config_full_path.read_text())
        # Modify the threshold
        if "fusion" in strategy_cfg:
            strategy_cfg["fusion"]["enter_threshold"] = thr
            # Disable quality floors for baseline calibration
            strategy_cfg["fusion"]["quality_floors"] = {
                "wyckoff": 0.0,
                "liquidity": 0.0,
                "structure": 0.0,
                "momentum": 0.0,
                "volume": 0.0,
                "context": 0.0
            }
        elif "mode" in strategy_cfg:
            # Handle diagnostic config format
            strategy_cfg["mode"]["enter_threshold"] = thr
            if "signals" in strategy_cfg:
                strategy_cfg["signals"]["enter_threshold"] = thr
            # Set minimal quality floors
            strategy_cfg["quality_floors"] = {
                "wyckoff": 0.0,
                "liquidity": 0.0,
                "structure": 0.0,
                "momentum": 0.0,
                "volume": 0.0,
                "context": 0.0
            }

        # Write temporary strategy config
        temp_strategy_path = f"/tmp/calib_strategy_{thr:.2f}.json"
        with open(temp_strategy_path, 'w') as f:
            json.dump(strategy_cfg, f, indent=2)

        # Update main config to point to temp strategy config
        cfg["strategy"]["config"] = temp_strategy_path
    else:
        print(f"Warning: Could not find strategy config at {strategy_config_path}")

    # Write temporary main config
    temp_cfg_path = f"/tmp/calib_thr_{thr:.2f}.json"
    with open(temp_cfg_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    out_dir = f"/tmp/calib_thr_{thr:.2f}_results"

    start = time.time()
    result = run_backtest_subprocess(temp_cfg_path, out_dir)
    elapsed = time.time() - start

    metrics = result.get("metrics", {})
    trades = result.get("trades", [])

    # --- Degradation proxies ---
    # 1) Entry score vs outcome: correlation-ish by buckets
    entry_scores = []
    pnls = []
    wins = []
    losses = []

    for t in trades:
        pnl = t.get("pnl", 0.0)
        if isinstance(pnl, (int, float)):
            pnls.append(pnl)
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)

        # Try to get entry score from metadata if available
        metadata = t.get("metadata", {})
        if isinstance(metadata, dict):
            score = metadata.get("entry_fusion_score")
            if score is not None:
                entry_scores.append(score)

    # 2) Holding-quality: MFE/MAE and bars_held
    mfes = []
    maes = []
    held = []

    for t in trades:
        metadata = t.get("metadata", {})
        if isinstance(metadata, dict):
            mfe = metadata.get("max_favorable_excursion")
            mae = metadata.get("max_adverse_excursion")
            bars = metadata.get("bars_held")

            if mfe is not None:
                mfes.append(mfe)
            if mae is not None:
                maes.append(mae)
            if bars is not None:
                held.append(bars)

    def safe_avg(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(mean(xs), 4) if xs else None

    def safe_metric(metrics, key, default=None):
        val = metrics.get(key, default)
        if isinstance(val, (int, float)):
            return round(val, 4)
        return val

    row = {
        "tag": tag,
        "threshold": thr,
        "elapsed_sec": round(elapsed, 3),
        # headline
        "trades": safe_metric(metrics, "trades"),
        "win_rate": safe_metric(metrics, "win_rate"),
        "profit_factor": safe_metric(metrics, "profit_factor"),
        "avg_R": safe_metric(metrics, "avg_r"),
        "max_dd": safe_metric(metrics, "max_dd"),
        "expectancy": safe_metric(metrics, "expectancy"),
        # degradation-ish
        "avg_entry_score": safe_avg(entry_scores),
        "avg_hold_bars": safe_avg(held),
        "avg_mfe_R": safe_avg(mfes),
        "avg_mae_R": safe_avg(maes),
        "avg_win": safe_avg(wins),
        "avg_loss": safe_avg(losses),
        "entries": safe_metric(metrics, "entries", len(pnls)),
    }

    # Clean up temp files
    try:
        os.unlink(temp_cfg_path)
        temp_strategy_path = f"/tmp/calib_strategy_{thr:.2f}.json"
        if os.path.exists(temp_strategy_path):
            os.unlink(temp_strategy_path)
        import shutil
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
    except:
        pass

    return row

def main():
    base_cfg_path = pathlib.Path("bull_machine/configs/calib_baseline_entries.json")
    if not base_cfg_path.exists():
        print(f"Config file not found: {base_cfg_path}")
        return

    base_cfg = json.loads(base_cfg_path.read_text())

    out_path = pathlib.Path("reports") / f"threshold_sweep_{int(time.time())}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["tag","threshold","elapsed_sec","trades","win_rate","profit_factor",
              "avg_R","max_dd","expectancy","avg_entry_score","avg_hold_bars","avg_mfe_R","avg_mae_R",
              "avg_win","avg_loss","entries"]

    lines = [",".join(header)]

    print("🔍 Starting threshold sweep from 0.30 to 0.50...")
    print(f"Base config: {base_cfg_path}")
    print(f"Output: {out_path}")
    print()

    for i, thr in enumerate(RANGE):
        print(f"[{i+1:2d}/{len(RANGE)}] Testing threshold {thr:.2f}...", end=" ", flush=True)

        try:
            row = run_once(base_cfg, thr, tag="baseline_no_exits")
            line = ",".join("" if row[k] is None else str(row[k]) for k in header)
            lines.append(line)

            # Print key metrics
            trades = row.get("trades", 0)
            win_rate = row.get("win_rate", 0)
            expectancy = row.get("expectancy", 0)
            print(f"✅ {trades} trades, {win_rate:.1%} win rate, {expectancy:.0f} expectancy")

        except Exception as e:
            print(f"❌ Failed: {e}")
            # Add empty row
            empty_row = {k: None for k in header}
            empty_row["tag"] = "baseline_no_exits"
            empty_row["threshold"] = thr
            line = ",".join("" if empty_row[k] is None else str(empty_row[k]) for k in header)
            lines.append(line)

    out_path.write_text("\n".join(lines))
    print(f"\n🎯 Sweep complete! Results saved to: {out_path}")

    # Print summary
    print("\n📊 Quick Summary:")
    print("threshold,trades,win_rate,expectancy")
    for line in lines[1:]:  # Skip header
        parts = line.split(",")
        thr = parts[1]
        trades = parts[3] if parts[3] else "0"
        win_rate = parts[4] if parts[4] else "0"
        expectancy = parts[8] if parts[8] else "0"
        print(f"{thr},{trades},{win_rate},{expectancy}")

if __name__ == "__main__":
    main()