#!/usr/bin/env python3
"""
Check determinism by running the same test twice and comparing outputs
"""
import subprocess
import json
import sys
import time
from pathlib import Path
from datetime import datetime

def run_paper_trade(run_id: int):
    """Run paper trading with fixed seed and return output files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cmd = [
        "python3", "bin/live/paper_trading.py",
        "--asset", "BTC",
        "--start", "2025-09-01",
        "--end", "2025-09-03",  # 3-day window for quick test
        "--balance", "10000",
        "--config", "configs/live/presets/BTC_vanilla.json"
    ]

    print(f"Run {run_id}: Executing paper trading...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Run {run_id} failed: {result.stderr}")
        return None

    # Parse output to find generated files
    output_files = {}
    for line in result.stdout.split('\n'):
        if "Signals:" in line:
            output_files["signals"] = line.split("Signals:")[1].strip()
        elif "Summary:" in line:
            output_files["summary"] = line.split("Summary:")[1].strip()

    return output_files

def compare_files(file1: Path, file2: Path, file_type: str):
    """Compare two files and report differences"""
    if file_type == "signals":
        # Compare last 5 lines of signals
        with open(file1) as f1:
            lines1 = f1.readlines()[-5:] if f1.readable() else []
        with open(file2) as f2:
            lines2 = f2.readlines()[-5:] if f2.readable() else []

        if lines1 == lines2:
            return True, "Last 5 signal lines match"
        else:
            return False, f"Signal mismatch:\nRun 1: {lines1}\nRun 2: {lines2}"

    elif file_type == "summary":
        # Compare key metrics from portfolio summary
        with open(file1) as f1:
            data1 = json.load(f1)
        with open(file2) as f2:
            data2 = json.load(f2)

        pts1 = data1.get("paper_trading_summary", {})
        pts2 = data2.get("paper_trading_summary", {})

        # Extract key metrics
        metrics1 = {
            "return_pct": pts1.get("portfolio", {}).get("return_pct"),
            "total_trades": pts1.get("trading", {}).get("total_trades"),
            "win_rate": pts1.get("trading", {}).get("win_rate"),
            "profit_factor": pts1.get("trading", {}).get("profit_factor")
        }

        metrics2 = {
            "return_pct": pts2.get("portfolio", {}).get("return_pct"),
            "total_trades": pts2.get("trading", {}).get("total_trades"),
            "win_rate": pts2.get("trading", {}).get("win_rate"),
            "profit_factor": pts2.get("trading", {}).get("profit_factor")
        }

        if metrics1 == metrics2:
            return True, f"Metrics match: {metrics1}"
        else:
            return False, f"Metrics differ:\nRun 1: {metrics1}\nRun 2: {metrics2}"

    return False, "Unknown file type"

def main():
    print("🔬 Bull Machine v1.7.3 - Determinism Check")
    print("==========================================")
    print("Running same paper trade twice with identical parameters...\n")

    # Run 1
    print("🚀 Starting Run 1...")
    files1 = run_paper_trade(1)
    if not files1:
        print("❌ Run 1 failed")
        sys.exit(1)
    print(f"   ✓ Run 1 complete: {files1['summary']}")

    # Small delay to ensure different timestamps
    time.sleep(2)

    # Run 2
    print("\n🚀 Starting Run 2...")
    files2 = run_paper_trade(2)
    if not files2:
        print("❌ Run 2 failed")
        sys.exit(1)
    print(f"   ✓ Run 2 complete: {files2['summary']}")

    # Compare results
    print("\n📊 Comparing Results...")
    all_match = True

    # Compare signals
    if files1.get("signals") and files2.get("signals"):
        match, msg = compare_files(
            Path(files1["signals"]),
            Path(files2["signals"]),
            "signals"
        )
        print(f"   Signals: {'✅ MATCH' if match else '❌ DIFFER'}")
        if not match:
            print(f"     {msg}")
            all_match = False

    # Compare summaries
    if files1.get("summary") and files2.get("summary"):
        match, msg = compare_files(
            Path(files1["summary"]),
            Path(files2["summary"]),
            "summary"
        )
        print(f"   Portfolio: {'✅ MATCH' if match else '❌ DIFFER'}")
        if not match:
            print(f"     {msg}")
            all_match = False
        elif match:
            print(f"     {msg}")

    # Final verdict
    print("\n" + "="*50)
    if all_match:
        print("✅ DETERMINISM CONFIRMED: Identical results across runs")
        sys.exit(0)
    else:
        print("❌ NON-DETERMINISTIC: Results differ between runs")
        sys.exit(1)

if __name__ == "__main__":
    main()