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
        from scripts.backtests.run_complete_confluence_system import (
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

def run_mode(mode: str, args) -> dict:
    """Run specific mode"""
    if mode == "confluence":
        return run_deterministic_backtest(args)

    elif mode == "ensemble":
        try:
            from scripts.backtests.run_full_ensemble_backtests import main as ensemble_main
            # Run ensemble backtest - this would need to be refactored to accept args
            return {"status": "ok", "mode": "ensemble", "message": "Ensemble mode - use run_full_ensemble_backtests.py directly for now"}
        except ImportError:
            return {"status": "error", "error": "Ensemble mode not available"}

    elif mode in ["stage-a", "stage-b", "stage-c"]:
        stage_map = {
            "stage-a": "scripts.opt.run_stage_a_complete",
            "stage-b": "scripts.opt.run_stage_b_optimization",
            "stage-c": "scripts.opt.run_stage_c_validation"
        }
        try:
            # These would need refactoring to accept CLI args
            return {"status": "ok", "mode": mode, "message": f"Optimization {mode} - use {stage_map[mode]}.py directly for now"}
        except ImportError:
            return {"status": "error", "error": f"Optimization {mode} not available"}

    elif mode == "weight-opt":
        try:
            # Signal weight optimization
            return {"status": "ok", "mode": "weight-opt", "message": "Weight optimization - use scripts/opt/run_signal_weight_optimization.py directly for now"}
        except ImportError:
            return {"status": "error", "error": "Weight optimization not available"}

    elif mode == "risk-scaling":
        try:
            # Risk scaling analysis
            return {"status": "ok", "mode": "risk-scaling", "message": "Risk scaling - use test_risk_scaling.py directly for now"}
        except ImportError:
            return {"status": "error", "error": "Risk scaling not available"}

    elif mode == "tearsheet":
        try:
            from generate_institutional_tearsheet import generate_tearsheet
            # Generate professional tearsheet
            return {"status": "ok", "mode": "tearsheet", "message": "Tearsheet generation - use generate_institutional_tearsheet.py directly for now"}
        except ImportError:
            return {"status": "error", "error": "Tearsheet generation not available"}

    else:
        return {"status": "error", "error": f"Unknown mode: {mode}"}

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bull Machine v1.6.2 - Unified CLI for Institutional Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5-Domain Confluence Backtest (default mode)
  python bull_machine_cli.py --config configs/v160/rc/ETH_production_v162.json --start 2024-01-01 --end 2024-12-31

  # Ensemble Backtesting
  python bull_machine_cli.py --mode ensemble --asset ETH --config configs/v160/assets/ETH.json

  # Multi-Stage Optimization
  python bull_machine_cli.py --mode stage-a --asset ETH --config configs/v160/assets/ETH.json
  python bull_machine_cli.py --mode stage-b --asset ETH --config configs/v160/assets/ETH.json
  python bull_machine_cli.py --mode stage-c --asset ETH --config configs/v160/assets/ETH.json

  # Risk Parameter Scaling
  python bull_machine_cli.py --mode risk-scaling --asset ETH --config configs/v160/assets/ETH.json

  # Signal Weight Optimization
  python bull_machine_cli.py --mode weight-opt --asset ETH --config configs/v160/assets/ETH.json

  # Professional Tearsheet Generation
  python bull_machine_cli.py --mode tearsheet --asset ETH --config configs/v160/rc/ETH_production_v162.json

  # With runtime config override
  BM_RTCFG="thresh0.3_min3_cd7_r0.075_sl1.4_tp2.5" python bull_machine_cli.py --config configs/v160/rc/ETH_production_v162.json

  # Quiet mode for grid optimization
  python bull_machine_cli.py --mode confluence --config configs/v160/assets/ETH.json --quiet --seed 42
        """
    )

    # Mode selection
    parser.add_argument("--mode", default="confluence",
                       choices=["confluence", "ensemble", "stage-a", "stage-b", "stage-c", "weight-opt", "risk-scaling", "tearsheet"],
                       help="Operation mode (default: confluence)")

    # Required arguments
    parser.add_argument("--config", required=True, help="Path to configuration JSON file")

    # Optional arguments
    parser.add_argument("--asset", help="Asset symbol (ETH, BTC, SOL) - auto-detected if not provided")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--risk-pct", type=float, help="Risk percentage per trade")
    parser.add_argument("--fee-bps", type=float, help="Fee in basis points")
    parser.add_argument("--slip-bps", type=float, help="Slippage in basis points")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--out", help="Output directory for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Suppress output if quiet mode
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.ERROR)

    # Run mode
    result = run_mode(args.mode, args)

    # Output clean JSON (no extra text)
    print(json.dumps(result))

    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "ok" else 1)

if __name__ == "__main__":
    main()