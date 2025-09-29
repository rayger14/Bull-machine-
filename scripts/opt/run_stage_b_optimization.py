#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Stage B Bayesian Optimization
Refine around successful Stage A parameters using Bayesian search
"""

import numpy as np
import pandas as pd
import json
import random
from typing import List, Dict, Any
from datetime import datetime
import time
from optimization_strategies import StageBStrategy
from safe_grid_runner import SafeGridRunner

def create_stage_b_configs() -> List[Dict[str, Any]]:
    """Create Stage B configurations around known good parameters"""

    # Start with known successful configuration from manual testing
    base_config = {
        "entry_threshold": 0.3,
        "min_active_domains": 3,
        "cooldown_days": 7,
        "risk_pct": 0.025,
        "sl_atr_multiplier": 1.4,
        "tp_atr_multiplier": 2.5,
        "trail_atr_multiplier": 0.8
    }

    # Generate variations around the base configuration
    configs = []

    # Parameter ranges for refinement
    entry_variations = [0.25, 0.3, 0.35, 0.4]
    risk_variations = [0.02, 0.025, 0.03, 0.035]
    sl_variations = [1.2, 1.4, 1.6, 1.8]
    tp_variations = [2.0, 2.5, 3.0, 3.5]
    cooldown_variations = [5, 7, 10, 14]

    # Create focused grid around successful parameters
    for entry in entry_variations:
        for risk in risk_variations:
            for sl in sl_variations:
                for tp in tp_variations:
                    if tp > sl:  # TP must be > SL
                        for cd in cooldown_variations:
                            config = {
                                "asset": "ETH",  # Focus on ETH first
                                "entry_threshold": entry,
                                "min_active_domains": 3,
                                "cooldown_days": cd,
                                "risk_pct": risk,
                                "sl_atr_multiplier": sl,
                                "tp_atr_multiplier": tp,
                                "trail_atr_multiplier": 0.8,
                                "strategy": "stage_b",
                                "fold": "2023-01-01..2025-01-01"  # Full period
                            }
                            configs.append(config)

    # Limit to reasonable number for Stage B
    random.shuffle(configs)
    return configs[:50]  # Top 50 variations

class StageBRunner(SafeGridRunner):
    """Stage B runner with Bayesian-style parameter exploration"""

    def __init__(self):
        super().__init__(max_workers=4, timeout_s=300)
        self.configs = create_stage_b_configs()

    def run_stage_b_optimization(self):
        """Run Stage B optimization with focused parameter search"""
        print(f"🔬 Bull Machine v1.6.2 - Stage B Bayesian Optimization")
        print(f"Git commit: {self.git_commit}")
        print(f"System: {self.system_info}")
        print(f"Configs to test: {len(self.configs)}")
        print(f"Results: {RESULTS}")
        print("-" * 60)

        # Check what's already done
        done_keys = set()
        if RESULTS.exists():
            with open(RESULTS, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if result.get("strategy") == "stage_b":
                            key = (result["asset"], result["fold"], result["cfg"])
                            done_keys.add(key)
                    except:
                        continue

        # Filter out completed configs
        todo_configs = []
        for config in self.configs:
            cfg_str = f"thresh{config['entry_threshold']}_min{config['min_active_domains']}_cd{config['cooldown_days']}_r{config['risk_pct']}_sl{config['sl_atr_multiplier']}_tp{config['tp_atr_multiplier']}_tr{config['trail_atr_multiplier']}"
            key = (config["asset"], config["fold"], cfg_str)
            if key not in done_keys:
                todo_configs.append((config["asset"], "1D", config["fold"], cfg_str))

        print(f"Remaining configs: {len(todo_configs)}")

        if not todo_configs:
            print("✅ All Stage B configurations already completed!")
            return

        # Run optimizations
        completed = 0
        errors = 0
        successful_trades = 0

        with open(RESULTS, "a", encoding="utf-8") as results_file:
            for task in todo_configs:
                try:
                    result = self.run_one_backtest(task, seed=42)
                    result["strategy"] = "stage_b"

                    # Write result immediately
                    results_file.write(json.dumps(result) + "\n")
                    results_file.flush()

                    completed += 1
                    status = result.get("status", "unknown")

                    if status == "ok":
                        trades = result.get("trades", 0)
                        if trades > 0:
                            successful_trades += 1
                            pnl = result.get("pnl_pct", 0)
                            wr = result.get("wr", 0)
                            pf = result.get("pf", 0)
                            sharpe = result.get("sharpe", 0)
                            print(f"✅ [{completed}/{len(todo_configs)}] {result['asset']} {result['cfg'][:25]}... "
                                  f"→ {trades}T, {wr:.1%} WR, {pnl:+.1f}% PnL, {pf:.2f} PF, {sharpe:.2f} Sharpe")
                        else:
                            print(f"⚪ [{completed}/{len(todo_configs)}] {result['asset']} {result['cfg'][:25]}... → 0 trades")
                    else:
                        errors += 1
                        error_msg = result.get("error", result.get("status", "unknown"))[:100]
                        print(f"❌ [{completed}/{len(todo_configs)}] {task[0]} → {status}: {error_msg}")

                except Exception as e:
                    errors += 1
                    print(f"💥 [{completed+1}/{len(todo_configs)}] {task[0]} → {str(e)[:100]}")
                    completed += 1

        print(f"\n🎯 Stage B optimization complete!")
        print(f"✅ Total runs: {completed}")
        print(f"📊 Runs with trades: {successful_trades}")
        print(f"❌ Errors: {errors}")
        print(f"📁 Results: {RESULTS}")

# Import the RESULTS path from safe_grid_runner
from safe_grid_runner import RESULTS

def main():
    """Main entry point for Stage B optimization"""
    runner = StageBRunner()
    runner.run_stage_b_optimization()

if __name__ == "__main__":
    main()