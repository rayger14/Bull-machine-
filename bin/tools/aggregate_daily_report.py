#!/usr/bin/env python3
"""
Aggregate daily paper trading reports into single summary JSON
"""
import json
import glob
import os
import sys
from pathlib import Path
from datetime import datetime

def main():
    # Get output directory from args or use today's date
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results") / datetime.now().strftime("%Y%m%d")

    if not out_dir.exists():
        print(f"❌ Directory not found: {out_dir}")
        sys.exit(1)

    summary = {
        "date": out_dir.name,
        "timestamp": datetime.now().isoformat(),
        "assets": {},
        "health": {
            "alerts": [],
            "all_green": True
        }
    }

    # Process all portfolio summary files
    for path in sorted(out_dir.glob("portfolio_summary_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)

            # Extract asset name from filename or data
            if "paper_trading_summary" in data:
                pts = data["paper_trading_summary"]
                asset = pts.get("asset", "UNKNOWN")

                summary["assets"][asset] = {
                    # Portfolio metrics
                    "initial_balance": pts.get("initial_balance"),
                    "final_equity": pts["portfolio"].get("total_equity"),
                    "return_pct": pts["portfolio"].get("return_pct"),
                    "realized_pnl": pts["portfolio"].get("realized_pnl"),
                    "unrealized_pnl": pts["portfolio"].get("unrealized_pnl"),

                    # Trading metrics
                    "total_trades": pts["trading"].get("total_trades", 0),
                    "win_rate": pts["trading"].get("win_rate", 0),
                    "profit_factor": pts["trading"].get("profit_factor", float('inf')),
                    "avg_win": pts["trading"].get("avg_win", 0),
                    "avg_loss": pts["trading"].get("avg_loss", 0),
                    "total_fees": pts["trading"].get("total_fees", 0),

                    # Health metrics (if available)
                    "health_status": pts["health"].get("status", "unknown"),
                    "period": pts.get("period")
                }

                # Check for alerts based on health bands
                metrics = summary["assets"][asset]

                # Profit factor check
                if metrics["profit_factor"] <= 1.0 and metrics["total_trades"] > 0:
                    summary["health"]["alerts"].append(f"{asset}: PF <= 1.0 ({metrics['profit_factor']:.2f})")
                    summary["health"]["all_green"] = False

                # Win rate check (warning if < 40%)
                if metrics["win_rate"] < 40 and metrics["total_trades"] > 5:
                    summary["health"]["alerts"].append(f"{asset}: Win rate < 40% ({metrics['win_rate']:.1f}%)")
                    summary["health"]["all_green"] = False

                # Return check (warning if negative)
                if metrics["return_pct"] < 0:
                    summary["health"]["alerts"].append(f"{asset}: Negative return ({metrics['return_pct']:.2f}%)")
                    summary["health"]["all_green"] = False

        except Exception as e:
            print(f"⚠️  Error processing {path}: {e}")
            continue

    # Also check for health summary files
    for path in sorted(out_dir.glob("health_summary_*.json")):
        try:
            with open(path) as f:
                health_data = json.load(f)

            # Extract asset from filename
            asset = path.stem.split("_")[2]  # health_summary_BTC_timestamp.json

            if asset in summary["assets"]:
                if "current_metrics" in health_data:
                    cm = health_data["current_metrics"]
                    summary["assets"][asset].update({
                        "macro_veto_rate": cm.get("macro_veto_rate"),
                        "smc_2hit_rate": cm.get("smc_2hit_rate"),
                        "hob_relevance_rate": cm.get("hob_relevance_rate"),
                        "delta_breaches": cm.get("delta_breaches", 0)
                    })

                    # Check health band violations
                    if cm.get("macro_veto_rate") is not None:
                        rate = cm["macro_veto_rate"]
                        if not (5.0 <= rate <= 15.0):
                            summary["health"]["alerts"].append(f"{asset}: Macro veto rate {rate:.1f}% out of band [5-15%]")
                            summary["health"]["all_green"] = False

                    if cm.get("smc_2hit_rate") is not None:
                        rate = cm["smc_2hit_rate"]
                        if rate < 30.0:
                            summary["health"]["alerts"].append(f"{asset}: SMC 2+ hit rate {rate:.1f}% < 30%")
                            summary["health"]["all_green"] = False

                    if cm.get("hob_relevance_rate") is not None:
                        rate = cm["hob_relevance_rate"]
                        if rate > 30.0:
                            summary["health"]["alerts"].append(f"{asset}: HOB relevance {rate:.1f}% > 30%")
                            summary["health"]["all_green"] = False

                    if cm.get("delta_breaches", 0) > 0:
                        summary["health"]["alerts"].append(f"{asset}: {cm['delta_breaches']} delta breaches detected")
                        summary["health"]["all_green"] = False

        except Exception as e:
            print(f"⚠️  Error processing health file {path}: {e}")
            continue

    # Write aggregate summary
    out_file = out_dir / f"daily_aggregate_{out_dir.name}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n📊 Daily Aggregate Report")
    print(f"   Date: {summary['date']}")
    print(f"   Assets analyzed: {', '.join(summary['assets'].keys())}")

    if summary["health"]["all_green"]:
        print("   ✅ Health Status: ALL GREEN")
    else:
        print("   ⚠️  Health Alerts:")
        for alert in summary["health"]["alerts"]:
            print(f"      - {alert}")

    print(f"\n📁 Report saved to: {out_file}")

    # Return non-zero exit code if alerts present (for CI)
    sys.exit(0 if summary["health"]["all_green"] else 1)

if __name__ == "__main__":
    main()