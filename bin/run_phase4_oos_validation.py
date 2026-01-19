#!/usr/bin/env python3
"""
Phase 4: Out-of-Sample (OOS) Validation

Loads Pareto frontier results from Phase 2/3, selects top candidates,
and validates on 2024 data (out-of-sample) to detect overfitting.

Usage:
    python3 bin/run_phase4_oos_validation.py [--optuna-db PATH] [--top-n N]

Workflow:
    1. Load Pareto frontier results from Optuna database
    2. Select top N trials (conservative, balanced, aggressive profiles)
    3. Test each on 2024 data (OOS)
    4. Compare in-sample (2022-2023) vs OOS (2024) metrics
    5. Flag overfitting (>30% degradation in PF or Sharpe)
    6. Recommend production config
    7. Save comparison report to results/phase4_oos_validation/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Tuple
import pandas as pd
import optuna
from optuna.study import Study


# Constants
DEFAULT_OPTUNA_DB = "sqlite:///optuna_archetypes_v2.db"
DEFAULT_TOP_N = 5
OOS_START = "2024-01-01"
OOS_END = "2024-09-30"
INSAMPLE_START = "2022-01-01"
INSAMPLE_END = "2023-12-31"
OVERFITTING_THRESHOLD = 0.30  # 30% degradation flags overfitting


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Out-of-Sample Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--optuna-db",
        default=DEFAULT_OPTUNA_DB,
        help=f"Path to Optuna database (default: {DEFAULT_OPTUNA_DB})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of top trials to validate (default: {DEFAULT_TOP_N})"
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Specific study name to load (default: best available)"
    )
    parser.add_argument(
        "--base-config",
        default="configs/mvp_bull_market_v1.json",
        help="Base config to use for OOS testing"
    )
    parser.add_argument(
        "--output-dir",
        default="results/phase4_oos_validation",
        help="Output directory for reports"
    )
    return parser.parse_args()


def load_pareto_frontier(optuna_db: str, study_name: str = None) -> Tuple[Study, List[optuna.trial.FrozenTrial]]:
    """
    Load Pareto frontier trials from Optuna database.

    Returns:
        (study, pareto_trials): Study object and list of Pareto-optimal trials
    """
    print(f"Loading Optuna database: {optuna_db}")

    # Load study
    storage = optuna.storages.RDBStorage(url=optuna_db)

    if study_name is None:
        # Find best study (most trials or best value)
        study_summaries = optuna.study.get_all_study_summaries(storage)
        if not study_summaries:
            raise ValueError("No studies found in database")

        # Sort by number of trials
        study_summaries.sort(key=lambda s: s.n_trials, reverse=True)
        study_name = study_summaries[0].study_name
        print(f"Auto-selected study: {study_name} ({study_summaries[0].n_trials} trials)")

    study = optuna.load_study(study_name=study_name, storage=storage)
    print(f"Loaded study: {study_name}")
    print(f"  Trials: {len(study.trials)}")
    print(f"  Best value: {study.best_value:.2f}")

    # Get Pareto frontier (top trials by objective value)
    trials = study.best_trials
    if not trials:
        # Fallback: get all completed trials and sort by value
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        trials.sort(key=lambda t: t.value if t.value else 0, reverse=True)

    print(f"  Pareto frontier trials: {len(trials)}")

    return study, trials


def select_diverse_trials(trials: List[optuna.trial.FrozenTrial], top_n: int) -> List[Dict]:
    """
    Select diverse trials from Pareto frontier.

    Strategy:
        - Conservative: Highest PF, lower risk
        - Balanced: Middle ground
        - Aggressive: More trades, acceptable PF
    """
    if len(trials) <= top_n:
        selected = trials[:top_n]
    else:
        # Select diverse set
        selected = []

        # 1. Best overall (highest value)
        selected.append(trials[0])

        # 2. Most diverse parameters
        used_indices = {0}
        for trial in trials[1:]:
            if len(selected) >= top_n:
                break

            # Simple diversity: avoid duplicate parameters
            params_str = str(sorted(trial.params.items()))
            is_duplicate = any(
                str(sorted(t.params.items())) == params_str
                for t in selected
            )

            if not is_duplicate:
                selected.append(trial)

        # Fill remaining with next best
        if len(selected) < top_n:
            for trial in trials:
                if trial not in selected:
                    selected.append(trial)
                    if len(selected) >= top_n:
                        break

    # Convert to dict format
    result = []
    for i, trial in enumerate(selected):
        result.append({
            "trial_number": trial.number,
            "profile": _infer_profile(trial.params, i),
            "params": trial.params,
            "insample_value": trial.value if trial.value else 0.0
        })

    return result


def _infer_profile(params: Dict, rank: int) -> str:
    """Infer trading profile from parameters."""
    if rank == 0:
        return "conservative"
    elif rank <= 2:
        return "balanced"
    else:
        return "aggressive"


def run_backtest(config_path: str, start: str, end: str, output_dir: Path) -> Dict:
    """
    Run backtest and extract metrics.

    Returns:
        dict with metrics: trades, pf, wr, dd, sharpe, return
    """
    log_file = output_dir / f"backtest_{start}_{end}.log"
    csv_file = output_dir / f"trades_{start}_{end}.csv"

    cmd = [
        "python3",
        "bin/backtest_knowledge_v2.py",
        "--asset", "BTC",
        "--start", start,
        "--end", end,
        "--config", str(config_path),
        "--export-trades", str(csv_file)
    ]

    print(f"  Running backtest: {start} to {end}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=Path(__file__).parent.parent
        )

        # Save log
        with open(log_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)

        # Extract metrics from output
        metrics = _extract_metrics(result.stdout)
        metrics["status"] = "success" if result.returncode == 0 else "failed"
        metrics["log_file"] = str(log_file)

        return metrics

    except subprocess.TimeoutExpired:
        print("  WARNING: Backtest timeout")
        return {
            "status": "timeout",
            "trades": 0,
            "pf": 0.0,
            "wr": 0.0,
            "dd": 0.0,
            "sharpe": 0.0,
            "return": 0.0,
            "log_file": str(log_file)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "status": "error",
            "trades": 0,
            "pf": 0.0,
            "wr": 0.0,
            "dd": 0.0,
            "sharpe": 0.0,
            "return": 0.0,
            "log_file": str(log_file)
        }


def _extract_metrics(output: str) -> Dict:
    """Extract metrics from backtest output."""
    import re

    metrics = {
        "trades": 0,
        "pf": 0.0,
        "wr": 0.0,
        "dd": 0.0,
        "sharpe": 0.0,
        "return": 0.0
    }

    patterns = {
        "trades": r"Total Trades:\s*(\d+)",
        "pf": r"Profit Factor:\s*([\d.]+)",
        "wr": r"Win Rate:\s*([\d.]+)%?",
        "dd": r"Max Drawdown:\s*([\d.]+)%?",
        "sharpe": r"Sharpe Ratio:\s*([-\d.]+)",
        "return": r"Total Return:\s*([-\d.]+)%?"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            if key == "trades":
                metrics[key] = int(value)
            else:
                metrics[key] = float(value)

    return metrics


def create_config_from_trial(base_config_path: str, trial: Dict, output_path: Path) -> Path:
    """
    Create a config file from trial parameters.

    Merges trial params with base config.
    """
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    # Update config with trial parameters
    params = trial["params"]

    # Map Optuna params to config structure
    # (This is simplified - adjust based on actual param structure)
    if "w_wyckoff" in params:
        config["fusion"]["weights"]["wyckoff"] = params["w_wyckoff"]
    if "w_liquidity" in params:
        config["fusion"]["weights"]["liquidity"] = params["w_liquidity"]
    if "w_momentum" in params:
        config["fusion"]["weights"]["momentum"] = params["w_momentum"]

    # Save config
    config_path = output_path / f"trial_{trial['trial_number']}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def calculate_degradation(insample: float, oos: float) -> float:
    """Calculate performance degradation percentage."""
    if insample == 0:
        return 0.0
    return (insample - oos) / insample


def generate_report(results: List[Dict], output_dir: Path):
    """Generate OOS validation report."""
    report_path = output_dir / "oos_validation_report.md"
    summary_path = output_dir / "oos_validation_summary.csv"

    # Create summary DataFrame
    df = pd.DataFrame(results)
    df.to_csv(summary_path, index=False)

    # Generate markdown report
    with open(report_path, 'w') as f:
        f.write("# Phase 4: Out-of-Sample Validation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**In-Sample Period**: {INSAMPLE_START} to {INSAMPLE_END}\n")
        f.write(f"**Out-of-Sample Period**: {OOS_START} to {OOS_END}\n")
        f.write(f"**Overfitting Threshold**: {OVERFITTING_THRESHOLD * 100:.0f}% degradation\n\n")

        f.write("---\n\n")
        f.write("## Summary\n\n")

        # Results table
        f.write("| Trial | Profile | IS PF | OOS PF | Degradation | IS Sharpe | OOS Sharpe | Overfitting? |\n")
        f.write("|-------|---------|-------|--------|-------------|-----------|------------|-------------|\n")

        for result in results:
            overfitting = "YES" if result["overfitting"] else "no"
            flag = "⚠️" if result["overfitting"] else "✓"

            f.write(f"| {result['trial_number']} | "
                   f"{result['profile']} | "
                   f"{result['insample_pf']:.2f} | "
                   f"{result['oos_pf']:.2f} | "
                   f"{result['pf_degradation'] * 100:.1f}% | "
                   f"{result['insample_sharpe']:.2f} | "
                   f"{result['oos_sharpe']:.2f} | "
                   f"{flag} {overfitting} |\n")

        f.write("\n---\n\n")
        f.write("## Detailed Results\n\n")

        for result in results:
            f.write(f"### Trial {result['trial_number']} ({result['profile']})\n\n")
            f.write(f"**In-Sample Metrics** ({INSAMPLE_START} to {INSAMPLE_END}):\n")
            f.write(f"- Trades: {result['insample_trades']}\n")
            f.write(f"- Profit Factor: {result['insample_pf']:.2f}\n")
            f.write(f"- Win Rate: {result['insample_wr']:.1f}%\n")
            f.write(f"- Max Drawdown: {result['insample_dd']:.1f}%\n")
            f.write(f"- Sharpe Ratio: {result['insample_sharpe']:.2f}\n")
            f.write(f"- Return: {result['insample_return']:.1f}%\n\n")

            f.write(f"**Out-of-Sample Metrics** ({OOS_START} to {OOS_END}):\n")
            f.write(f"- Trades: {result['oos_trades']}\n")
            f.write(f"- Profit Factor: {result['oos_pf']:.2f}\n")
            f.write(f"- Win Rate: {result['oos_wr']:.1f}%\n")
            f.write(f"- Max Drawdown: {result['oos_dd']:.1f}%\n")
            f.write(f"- Sharpe Ratio: {result['oos_sharpe']:.2f}\n")
            f.write(f"- Return: {result['oos_return']:.1f}%\n\n")

            f.write(f"**Degradation Analysis**:\n")
            f.write(f"- PF Degradation: {result['pf_degradation'] * 100:.1f}%\n")
            f.write(f"- Sharpe Degradation: {result['sharpe_degradation'] * 100:.1f}%\n")

            if result["overfitting"]:
                f.write(f"\n⚠️ **OVERFITTING DETECTED** - Degradation exceeds {OVERFITTING_THRESHOLD * 100:.0f}% threshold\n\n")
            else:
                f.write(f"\n✓ Performance stable - Suitable for production\n\n")

            f.write("---\n\n")

        # Recommendations
        f.write("## Recommendations\n\n")

        # Find best non-overfitted trial
        valid_trials = [r for r in results if not r["overfitting"]]
        if valid_trials:
            best = max(valid_trials, key=lambda r: r["oos_pf"])
            f.write(f"**Recommended for Production**: Trial {best['trial_number']} ({best['profile']})\n\n")
            f.write(f"- OOS Profit Factor: {best['oos_pf']:.2f}\n")
            f.write(f"- OOS Sharpe Ratio: {best['oos_sharpe']:.2f}\n")
            f.write(f"- PF Degradation: {best['pf_degradation'] * 100:.1f}%\n")
            f.write(f"- Config: `{best['config_file']}`\n\n")
        else:
            f.write("⚠️ **WARNING**: All trials show overfitting. Consider:\n")
            f.write("- Increasing regularization in Phase 2\n")
            f.write("- Reducing parameter complexity\n")
            f.write("- Using walk-forward validation\n\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Summary saved to: {summary_path}")


def main():
    args = parse_args()

    print("=" * 80)
    print("Phase 4: Out-of-Sample Validation")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Pareto frontier
    study, trials = load_pareto_frontier(args.optuna_db, args.study_name)

    # Select diverse trials
    selected_trials = select_diverse_trials(trials, args.top_n)
    print(f"\nSelected {len(selected_trials)} trials for OOS validation:")
    for trial in selected_trials:
        print(f"  Trial {trial['trial_number']}: {trial['profile']} (IS value: {trial['insample_value']:.2f})")

    print()

    # Validate each trial
    results = []

    for i, trial in enumerate(selected_trials):
        print(f"\n[{i+1}/{len(selected_trials)}] Validating Trial {trial['trial_number']} ({trial['profile']})")
        print("-" * 80)

        # Create trial-specific output directory
        trial_dir = output_dir / f"trial_{trial['trial_number']}"
        trial_dir.mkdir(exist_ok=True)

        # Create config from trial parameters
        config_path = create_config_from_trial(args.base_config, trial, trial_dir)
        print(f"  Config: {config_path}")

        # Run in-sample backtest (for verification)
        print("\n  In-Sample Test:")
        insample_metrics = run_backtest(config_path, INSAMPLE_START, INSAMPLE_END, trial_dir)

        # Run out-of-sample backtest
        print("\n  Out-of-Sample Test:")
        oos_metrics = run_backtest(config_path, OOS_START, OOS_END, trial_dir)

        # Calculate degradation
        pf_degradation = calculate_degradation(insample_metrics["pf"], oos_metrics["pf"])
        sharpe_degradation = calculate_degradation(insample_metrics["sharpe"], oos_metrics["sharpe"])

        # Flag overfitting
        overfitting = (
            pf_degradation > OVERFITTING_THRESHOLD or
            sharpe_degradation > OVERFITTING_THRESHOLD
        )

        # Store results
        result = {
            "trial_number": trial["trial_number"],
            "profile": trial["profile"],
            "config_file": str(config_path),
            "insample_trades": insample_metrics["trades"],
            "insample_pf": insample_metrics["pf"],
            "insample_wr": insample_metrics["wr"],
            "insample_dd": insample_metrics["dd"],
            "insample_sharpe": insample_metrics["sharpe"],
            "insample_return": insample_metrics["return"],
            "oos_trades": oos_metrics["trades"],
            "oos_pf": oos_metrics["pf"],
            "oos_wr": oos_metrics["wr"],
            "oos_dd": oos_metrics["dd"],
            "oos_sharpe": oos_metrics["sharpe"],
            "oos_return": oos_metrics["return"],
            "pf_degradation": pf_degradation,
            "sharpe_degradation": sharpe_degradation,
            "overfitting": overfitting
        }
        results.append(result)

        # Print summary
        print("\n  Results:")
        print(f"    In-Sample:  PF={insample_metrics['pf']:.2f}, Sharpe={insample_metrics['sharpe']:.2f}")
        print(f"    Out-Sample: PF={oos_metrics['pf']:.2f}, Sharpe={oos_metrics['sharpe']:.2f}")
        print(f"    Degradation: PF={pf_degradation*100:.1f}%, Sharpe={sharpe_degradation*100:.1f}%")
        if overfitting:
            print(f"    ⚠️  OVERFITTING DETECTED")
        else:
            print(f"    ✓  Performance stable")

    # Generate report
    print("\n" + "=" * 80)
    print("Generating Report")
    print("=" * 80)
    generate_report(results, output_dir)

    print("\n" + "=" * 80)
    print("Phase 4 OOS Validation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
