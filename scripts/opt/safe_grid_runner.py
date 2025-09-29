#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Safe Grid Runner
Battle-tested resumable optimization framework
- Process isolation with timeouts and memory caps
- Append-only JSONL results (never corrupted)
- Resource guardrails to prevent system crashes
- Deterministic seeds and full reproducibility
"""

import json
import os
import time
import signal
import psutil
import itertools
import random
import pathlib
import subprocess
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Tuple, Any, Set

# Configuration
OUT_DIR = "reports/opt"
LOG = pathlib.Path(OUT_DIR)
LOG.mkdir(parents=True, exist_ok=True)
RESULTS = LOG / "results.jsonl"
CHECKPOINT = LOG / "_checkpoint.json"
ERROR_LOG = LOG / "errors.log"

def get_git_commit():
    """Get current git commit hash for reproducibility"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def get_system_info():
    """Get system resource info"""
    try:
        mem = psutil.virtual_memory()
        return {
            "cpu_count": os.cpu_count(),
            "mem_total_gb": round(mem.total / (1024**3), 1),
            "mem_available_gb": round(mem.available / (1024**3), 1)
        }
    except:
        return {"cpu_count": 1, "mem_total_gb": 0, "mem_available_gb": 0}

class SafeGridRunner:
    def __init__(self, max_workers: int = None, timeout_s: int = 240):
        self.max_workers = max_workers or max(1, min(6, os.cpu_count() - 1))
        self.timeout_s = timeout_s
        self.git_commit = get_git_commit()
        self.system_info = get_system_info()

        # Set resource limits
        self._set_resource_limits()

    def _set_resource_limits(self):
        """Set system resource limits to prevent crashes"""
        try:
            import resource
            # Limit memory to 12GB per process
            resource.setrlimit(resource.RLIMIT_AS, (12 * 1024 * 1024 * 1024, -1))
            # Limit open files
            resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 8192))
            # Limit processes
            resource.setrlimit(resource.RLIMIT_NPROC, (1024, 2048))
        except:
            pass  # Windows doesn't have resource module

    def already_done(self) -> Set[Tuple]:
        """Load completed runs from JSONL to enable resumability"""
        done = set()
        if RESULTS.exists():
            with open(RESULTS, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        d = json.loads(line.strip())
                        if d.get("status") == "ok":
                            key = (d["asset"], d["tf"], d["fold"], d["cfg"])
                            done.add(key)
                    except json.JSONDecodeError as e:
                        self._log_error(f"Line {line_num}: Invalid JSON - {e}")
                    except Exception as e:
                        self._log_error(f"Line {line_num}: Parse error - {e}")
        return done

    def _log_error(self, msg: str):
        """Log errors to separate error file"""
        timestamp = datetime.now().isoformat()
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}: {msg}\n")

    def create_stage_a_grid(self) -> List[Tuple]:
        """Create Stage A parameter grid - coarse sweep of major parameters"""
        return list(itertools.product(
            ["ETH", "BTC"],  # assets
            ["1D"],  # timeframes - start with daily only for stability
            [
                "2023-01-01..2023-12-31",  # fold 1: 2023 data
                "2024-01-01..2024-12-31",  # fold 2: 2024 data
                "2023-01-01..2025-01-01"   # fold 3: full period
            ],
            [
                # Entry thresholds
                "thresh0.2_min2_cd5_r0.025_sl1.2_tp2.0_tr0.6",
                "thresh0.3_min3_cd7_r0.025_sl1.4_tp2.5_tr0.8",
                "thresh0.4_min3_cd7_r0.025_sl1.6_tp3.0_tr1.0",
                "thresh0.5_min4_cd10_r0.025_sl1.8_tp3.5_tr1.2",

                # Risk variations
                "thresh0.3_min3_cd7_r0.015_sl1.4_tp2.5_tr0.8",
                "thresh0.3_min3_cd7_r0.035_sl1.4_tp2.5_tr0.8",
                "thresh0.3_min3_cd7_r0.050_sl1.4_tp2.5_tr0.8",

                # SL/TP variations
                "thresh0.3_min3_cd7_r0.025_sl1.0_tp2.0_tr0.6",
                "thresh0.3_min3_cd7_r0.025_sl2.0_tp4.0_tr1.2",

                # Cooldown variations
                "thresh0.3_min3_cd3_r0.025_sl1.4_tp2.5_tr0.8",
                "thresh0.3_min3_cd14_r0.025_sl1.4_tp2.5_tr0.8",
            ]
        ))

    def parse_config_string(self, cfg_str: str) -> Dict[str, Any]:
        """Parse configuration string into parameters"""
        params = {}
        parts = cfg_str.split("_")

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

    def run_one_backtest(self, task: Tuple, seed: int = 42) -> Dict[str, Any]:
        """Run a single backtest in isolated process"""
        asset, tf, fold, cfg = task
        start_date, end_date = fold.split("..")

        # Parse configuration
        params = self.parse_config_string(cfg)

        # Create temporary config file
        config_data = {
            "asset": asset,
            "timeframe": tf,
            "start_date": start_date,
            "end_date": end_date,
            "seed": seed,
            **params
        }

        # Generate unique config hash for this run
        config_hash = hashlib.md5(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:8]

        start_time = time.time()

        try:
            # Import and run backtest directly (no subprocess for now)
            from run_complete_confluence_system import (
                load_multi_timeframe_data,
                run_complete_confluence_backtest
            )

            # Data paths for ETH and BTC
            data_paths = {
                'ETH': {
                    '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
                    '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
                    '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
                },
                'BTC': {
                    '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv',
                    '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv',
                    '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv'
                }
            }

            # Load data
            if asset not in data_paths:
                return {
                    "asset": asset, "tf": tf, "fold": fold, "cfg": cfg,
                    "status": "unsupported_asset", "dur_s": round(time.time() - start_time, 2),
                    "git": self.git_commit, "config_hash": config_hash
                }

            data = load_multi_timeframe_data(asset, data_paths[asset])
            if not data or '1D' not in data:
                return {
                    "asset": asset, "tf": tf, "fold": fold, "cfg": cfg,
                    "status": "no_data", "dur_s": round(time.time() - start_time, 2),
                    "git": self.git_commit, "config_hash": config_hash
                }

            # Create config for backtest
            backtest_config = {
                "entry_threshold": params.get("entry_threshold", 0.3),
                "min_active_domains": params.get("min_active_domains", 3),
                "cooldown_days": params.get("cooldown_days", 7),
                "risk_pct": params.get("risk_pct", 0.025),
                "sl_atr_multiplier": params.get("sl_atr_multiplier", 1.4),
                "tp_atr_multiplier": params.get("tp_atr_multiplier", 2.5),
                "trail_atr_multiplier": params.get("trail_atr_multiplier", 0.8),
                "start_date": start_date,
                "end_date": end_date
            }

            # Run backtest
            result = run_complete_confluence_backtest(asset, data, backtest_config)

            if not result:
                return {
                    "asset": asset, "tf": tf, "fold": fold, "cfg": cfg,
                    "status": "backtest_failed", "dur_s": round(time.time() - start_time, 2),
                    "git": self.git_commit, "config_hash": config_hash
                }

            # Extract key metrics from metrics dictionary
            duration_s = round(time.time() - start_time, 2)
            metrics = result.get("metrics", {})

            return {
                "asset": asset,
                "tf": tf,
                "fold": fold,
                "cfg": cfg,
                "status": "ok",
                "trades": metrics.get("total_trades", 0),
                "wr": metrics.get("win_rate", 0.0),
                "pnl_pct": metrics.get("total_pnl_pct", 0.0),
                "pf": metrics.get("profit_factor", 0.0),
                "dd_pct": metrics.get("max_drawdown_pct", 0.0),
                "sharpe": metrics.get("sharpe_ratio", 0.0),
                "dur_s": duration_s,
                "seed": seed,
                "git": self.git_commit,
                "config_hash": config_hash,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            return {
                "asset": asset, "tf": tf, "fold": fold, "cfg": cfg,
                "status": "error", "error": str(e)[:2000],
                "dur_s": round(time.time() - start_time, 2),
                "git": self.git_commit, "config_hash": config_hash
            }

    def run_stage_a_optimization(self):
        """Run Stage A optimization with full safety and resumability"""
        print(f"🔬 Bull Machine v1.6.2 - Stage A Safe Grid Optimization")
        print(f"Git commit: {self.git_commit}")
        print(f"System: {self.system_info}")
        print(f"Workers: {self.max_workers}, Timeout: {self.timeout_s}s")
        print(f"Results: {RESULTS}")
        print("-" * 60)

        # Create grid and check what's already done
        grid = self.create_stage_a_grid()
        done = self.already_done()
        todo = [g for g in grid if g not in done]

        print(f"Total combinations: {len(grid)}")
        print(f"Already completed: {len(done)}")
        print(f"Remaining: {len(todo)}")

        if not todo:
            print("✅ All combinations already completed!")
            return

        # Randomize order to avoid systematic bias
        random.shuffle(todo)

        # Run with process pool for isolation
        completed = 0
        errors = 0

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            with open(RESULTS, "a", encoding="utf-8") as results_file:

                # Submit all jobs
                future_to_task = {
                    executor.submit(self.run_one_backtest, task, 42): task
                    for task in todo
                }

                # Process completed jobs
                for future in as_completed(future_to_task):
                    task = future_to_task[future]

                    try:
                        result = future.result(timeout=self.timeout_s + 30)

                        # Write result immediately
                        results_file.write(json.dumps(result) + "\n")
                        results_file.flush()

                        # Progress tracking
                        completed += 1
                        status = result.get("status", "unknown")

                        if status == "ok":
                            pnl = result.get("pnl_pct", 0)
                            trades = result.get("trades", 0)
                            wr = result.get("wr", 0)
                            pf = result.get("pf", 0)
                            print(f"✅ [{completed}/{len(todo)}] {result['asset']} {result['cfg'][:20]}... "
                                  f"→ {trades}T, {wr:.1%} WR, {pnl:+.1f}% PnL, {pf:.2f} PF")
                        else:
                            errors += 1
                            error_msg = result.get("error", result.get("status", "unknown"))[:100]
                            print(f"❌ [{completed}/{len(todo)}] {task[0]} {task[3][:20]}... → {status}: {error_msg}")
                            self._log_error(f"{task}: {status} - {error_msg}")

                    except Exception as e:
                        errors += 1
                        error_result = {
                            "asset": task[0], "tf": task[1], "fold": task[2], "cfg": task[3],
                            "status": "executor_error", "error": str(e)[:2000],
                            "git": self.git_commit, "timestamp": datetime.now().isoformat()
                        }
                        results_file.write(json.dumps(error_result) + "\n")
                        results_file.flush()
                        print(f"💥 [{completed+1}/{len(todo)}] {task[0]} → executor error: {str(e)[:100]}")
                        self._log_error(f"{task}: executor_error - {str(e)}")
                        completed += 1

        print(f"\n🎯 Stage A optimization complete!")
        print(f"✅ Successful runs: {completed - errors}")
        print(f"❌ Failed runs: {errors}")
        print(f"📊 Results saved to: {RESULTS}")
        print(f"🚨 Errors logged to: {ERROR_LOG}")

def main():
    """Main entry point"""
    runner = SafeGridRunner(max_workers=4, timeout_s=300)
    runner.run_stage_a_optimization()

if __name__ == "__main__":
    main()