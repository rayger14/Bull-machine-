#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Deterministic Backtest CLI
Clean, isolated interface for grid optimization
No LLMs, no streaming output corruption
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Config load failed: {e}"}

def parse_runtime_config(runtime_cfg: str) -> dict:
    """Parse runtime configuration string"""
    params = {}
    if not runtime_cfg:
        return params

    parts = runtime_cfg.split("_")
    for part in parts:
        if part.startswith("thresh"):
            params["entry_threshold"] = float(part[6:])
        elif part.startswith("min"):
            params["min_active_domains"] = int(part[3:])
        elif part.startswith("cd"):
            params["cooldown_days"] = int(part[2:])
        elif part.startswith("r"):
            params["risk_pct"] = float(part[1:])
        elif part.startswith("sl"):
            params["sl_atr_multiplier"] = float(part[2:])
        elif part.startswith("tp"):
            params["tp_atr_multiplier"] = float(part[2:])
        elif part.startswith("tr"):
            params["trail_atr_multiplier"] = float(part[2:])

    return params

def run_deterministic_backtest(args) -> dict:
    """Run a single deterministic backtest"""
    start_time = time.time()

    try:
        # Load base config
        config = load_config(args.config)
        if "error" in config:
            return config

        # Apply runtime overrides
        runtime_params = parse_runtime_config(os.environ.get("BM_RTCFG", ""))
        config.update(runtime_params)

        # Apply CLI overrides
        if args.start:
            config["start_date"] = args.start
        if args.end:
            config["end_date"] = args.end
        if args.risk_pct:
            config["risk_pct"] = args.risk_pct
        if args.fee_bps:
            config["fee_bps"] = args.fee_bps
        if args.slip_bps:
            config["slip_bps"] = args.slip_bps

        # Set seed for reproducibility
        import random
        import numpy as np
        seed = args.seed or 42
        random.seed(seed)
        np.random.seed(seed)

        # Import and run backtest
        from run_complete_confluence_system import (
            load_multi_timeframe_data,
            run_complete_confluence_backtest
        )

        # Extract asset from config or filename
        asset = config.get("asset") or args.asset
        if not asset:
            config_name = Path(args.config).stem
            if "ETH" in config_name.upper():
                asset = "ETH"
            elif "BTC" in config_name.upper():
                asset = "BTC"
            else:
                asset = "ETH"  # default

        # Load data
        data = load_multi_timeframe_data(asset)
        if not data or '1D' not in data:
            return {
                "status": "error",
                "error": f"No data available for {asset}",
                "asset": asset,
                "dur_s": round(time.time() - start_time, 2)
            }

        # Run backtest
        result = run_complete_confluence_backtest(asset, data, config)

        if not result:
            return {
                "status": "error",
                "error": "Backtest returned empty result",
                "asset": asset,
                "config": config,
                "dur_s": round(time.time() - start_time, 2)
            }

        # Format clean output
        clean_result = {
            "status": "ok",
            "asset": asset,
            "start_date": config.get("start_date", "unknown"),
            "end_date": config.get("end_date", "unknown"),
            "total_trades": result.get("total_trades", 0),
            "win_rate": round(result.get("win_rate", 0.0), 4),
            "total_pnl_pct": round(result.get("total_pnl_pct", 0.0), 4),
            "profit_factor": round(result.get("profit_factor", 0.0), 4),
            "max_drawdown_pct": round(result.get("max_drawdown_pct", 0.0), 4),
            "sharpe_ratio": round(result.get("sharpe_ratio", 0.0), 4),
            "trades_per_month": round(result.get("trades_per_month", 0.0), 2),
            "avg_trade_pct": round(result.get("avg_trade_pct", 0.0), 4),
            "best_trade_pct": round(result.get("best_trade_pct", 0.0), 4),
            "worst_trade_pct": round(result.get("worst_trade_pct", 0.0), 4),
            "seed": seed,
            "dur_s": round(time.time() - start_time, 2),
            "config": {k: v for k, v in config.items() if k not in ["data", "trades"]},
            "timestamp": datetime.now().isoformat()
        }

        return clean_result

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "asset": args.asset or "unknown",
            "dur_s": round(time.time() - start_time, 2),
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bull Machine Deterministic Backtest CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest
  python bull_machine_cli.py --config configs/v160/assets/ETH.json --start 2023-01-01 --end 2024-01-01

  # With runtime config override
  BM_RTCFG="thresh0.3_min3_cd7_r0.025_sl1.4_tp2.5" python bull_machine_cli.py --config configs/v160/assets/ETH.json

  # Quiet mode for grid optimization
  python bull_machine_cli.py --config configs/v160/assets/ETH.json --quiet --seed 42
        """
    )

    # Required arguments
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")

    # Optional arguments
    parser.add_argument("--asset", help="Asset symbol (ETH, BTC) - auto-detected if not provided")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--risk-pct", type=float, help="Risk percentage per trade")
    parser.add_argument("--fee-bps", type=float, help="Fee in basis points")
    parser.add_argument("--slip-bps", type=float, help="Slippage in basis points")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Suppress output if quiet mode
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.ERROR)

    # Run backtest
    result = run_deterministic_backtest(args)

    # Output clean JSON (no extra text)
    print(json.dumps(result))

    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "ok" else 1)

if __name__ == "__main__":
    main()