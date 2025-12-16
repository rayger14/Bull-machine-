#!/usr/bin/env python3
"""
S5 (Long Squeeze) Production Config Generator

Reads Optuna study results and generates production-ready configs from Pareto frontier.
Selects 3 configs representing different risk/reward profiles:

1. CONSERVATIVE: High PF, lower trade count (strict thresholds)
2. BALANCED: Middle ground between PF and trade count
3. AGGRESSIVE: More trades, slightly lower PF (relaxed thresholds)

OUTPUT:
- configs/optimized/s5_conservative.json
- configs/optimized/s5_balanced.json
- configs/optimized/s5_aggressive.json

Author: Claude Code (Backend Architect)
Date: 2025-11-20
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_pareto_results() -> pd.DataFrame:
    """
    Load Pareto frontier results from optimization.

    Returns:
        DataFrame with Pareto-optimal solutions
    """
    results_file = Path('results/optimization/s5_calibration_pareto_frontier.csv')

    if not results_file.exists():
        # Fallback: Load all trials and filter locally
        all_trials_file = Path('results/optimization/s5_calibration_all_trials.csv')
        if not all_trials_file.exists():
            raise FileNotFoundError(
                "Optimization results not found. Run bin/optimize_s5_calibration.py first."
            )

        logger.warning("Pareto frontier file not found, using all trials")
        df = pd.read_csv(all_trials_file)

        # Simple Pareto filtering: Top 20 by PF
        df = df.sort_values('profit_factor', ascending=False).head(20)
    else:
        df = pd.read_csv(results_file)

    logger.info(f"Loaded {len(df)} Pareto-optimal solutions")

    return df


def select_configs(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Select 3 representative configs from Pareto frontier.

    Selection strategy:
    - Conservative: Highest PF among solutions with trades < median
    - Balanced: Best PF/WR combination near median trades
    - Aggressive: Highest WR among solutions with trades > median

    Args:
        df: Pareto frontier dataframe

    Returns:
        Dictionary with 'conservative', 'balanced', 'aggressive' configs
    """
    if len(df) == 0:
        raise ValueError("No solutions available")

    # Filter: Only consider solutions with reasonable metrics
    df_viable = df[
        (df['profit_factor'] >= 1.3) &
        (df['win_rate'] >= 50.0) &
        (df['total_trades'] >= 5)
    ].copy()

    if len(df_viable) == 0:
        logger.warning("No viable solutions found, using top 3 by PF")
        df_viable = df.sort_values('profit_factor', ascending=False).head(3)

    # Compute trade count median
    median_trades = df_viable['total_trades'].median()

    # CONSERVATIVE: Highest PF with below-median trades
    conservative_candidates = df_viable[df_viable['total_trades'] <= median_trades]
    if len(conservative_candidates) > 0:
        conservative = conservative_candidates.nlargest(1, 'profit_factor').iloc[0]
    else:
        conservative = df_viable.nlargest(1, 'profit_factor').iloc[0]

    # BALANCED: Best combined score near median trades
    df_viable['combined_score'] = df_viable['profit_factor'] * 0.6 + df_viable['win_rate'] * 0.01
    balanced = df_viable.nlargest(1, 'combined_score').iloc[0]

    # AGGRESSIVE: Highest WR with above-median trades
    aggressive_candidates = df_viable[df_viable['total_trades'] >= median_trades]
    if len(aggressive_candidates) > 0:
        aggressive = aggressive_candidates.nlargest(1, 'win_rate').iloc[0]
    else:
        aggressive = df_viable.nlargest(1, 'win_rate').iloc[0]

    return {
        'conservative': conservative,
        'balanced': balanced,
        'aggressive': aggressive
    }


def create_config_json(params: pd.Series, profile_name: str, description: str) -> Dict:
    """
    Create production config JSON from parameters.

    Args:
        params: Parameter row from Pareto frontier
        profile_name: Config profile name
        description: Human-readable description

    Returns:
        Complete config dictionary
    """
    config = {
        "version": f"s5_optimized_{profile_name}",
        "profile": f"S5 Long Squeeze - {profile_name.capitalize()}",
        "description": description,
        "_optimization_metadata": {
            "trial_number": int(params['trial_number']),
            "train_pf": float(params.get('train_pf', 0.0)),
            "val_pf": float(params.get('val_pf', 0.0)),
            "avg_pf": float(params['profit_factor']),
            "win_rate": float(params['win_rate']),
            "total_trades": int(params['total_trades']),
            "trades_per_6mo": float(params['trades_per_6mo']),
            "optimized_date": pd.Timestamp.now().strftime('%Y-%m-%d')
        },
        "adaptive_fusion": True,
        "regime_classifier": {
            "model_path": "models/regime_classifier_gmm.pkl",
            "feature_order": [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ],
            "zero_fill_missing": False,
            "regime_override": {
                "_comment": "S5 fires in risk_on (bull) and crisis regimes only"
            }
        },
        "ml_filter": {
            "enabled": False,
            "_comment": "Disabled for pure archetype testing - can enable in production"
        },
        "fusion": {
            "entry_threshold_confidence": 0.30,
            "weights": {
                "wyckoff": 0.35,
                "liquidity": 0.30,
                "momentum": 0.35,
                "smc": 0.0
            }
        },
        "archetypes": {
            "use_archetypes": True,
            "max_trades_per_day": 6,
            "enable_A": False, "enable_B": False, "enable_C": False, "enable_D": False,
            "enable_E": False, "enable_F": False, "enable_G": False, "enable_H": False,
            "enable_K": False, "enable_L": False, "enable_M": False,
            "enable_S1": False, "enable_S2": False, "enable_S3": False, "enable_S4": False,
            "enable_S5": True,
            "enable_S6": False, "enable_S7": False, "enable_S8": False,
            "thresholds": {
                "min_liquidity": 0.10,
                "long_squeeze": {
                    "direction": "short",
                    "fusion_threshold": float(params['fusion_threshold']),
                    "funding_z_min": float(params['funding_z_min']),
                    "rsi_min": float(params['rsi_min']),
                    "liquidity_max": float(params['liquidity_max']),
                    "oi_change_min": float(params['oi_change_min']),
                    "max_risk_pct": 0.015,
                    "atr_stop_mult": float(params['atr_stop_mult'])
                }
            },
            "long_squeeze": {
                "archetype_weight": 2.2,
                "final_fusion_gate": float(params['fusion_threshold']),
                "cooldown_bars": int(params['cooldown_bars']),
                "_comment": "Optimized thresholds from multi-objective calibration"
            },
            "routing": {
                "risk_on": {
                    "weights": {"long_squeeze": 2.0},
                    "final_gate_delta": 0.0,
                    "_comment": "Primary regime for S5"
                },
                "neutral": {
                    "weights": {"long_squeeze": 1.5},
                    "final_gate_delta": 0.0
                },
                "risk_off": {
                    "weights": {"long_squeeze": 0.0},
                    "final_gate_delta": 0.0,
                    "_comment": "Disabled in bear markets"
                },
                "crisis": {
                    "weights": {"long_squeeze": 2.5},
                    "final_gate_delta": 0.0,
                    "_comment": "Highest weight during capitulation phases"
                }
            },
            "exits": {
                "long_squeeze": {
                    "trail_atr": float(params['trail_atr_mult']),
                    "time_limit_hours": 24
                }
            }
        },
        "risk": {
            "base_risk_pct": 0.015,
            "max_position_size_pct": 0.15,
            "max_portfolio_risk_pct": 0.08
        }
    }

    return config


def generate_configs(selected_configs: Dict[str, pd.Series]) -> None:
    """
    Generate and save production config files.

    Args:
        selected_configs: Dictionary of selected configs (conservative/balanced/aggressive)
    """
    output_dir = Path('configs/optimized')
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptions = {
        'conservative': (
            "Conservative S5 config optimized for high profit factor. "
            f"Strict thresholds, fewer trades (~{selected_configs['conservative']['total_trades']}/year), "
            f"PF={selected_configs['conservative']['profit_factor']:.2f}"
        ),
        'balanced': (
            "Balanced S5 config with optimal PF/WR combination. "
            f"Moderate thresholds, ~{selected_configs['balanced']['total_trades']}/year trades, "
            f"PF={selected_configs['balanced']['profit_factor']:.2f}, "
            f"WR={selected_configs['balanced']['win_rate']:.1f}%"
        ),
        'aggressive': (
            "Aggressive S5 config optimized for higher trade frequency. "
            f"Relaxed thresholds, more trades (~{selected_configs['aggressive']['total_trades']}/year), "
            f"WR={selected_configs['aggressive']['win_rate']:.1f}%"
        )
    }

    for profile_name, params in selected_configs.items():
        config = create_config_json(params, profile_name, descriptions[profile_name])

        output_file = output_dir / f's5_{profile_name}.json'
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Generated {profile_name} config: {output_file}")
        logger.info(
            f"  Trial {params['trial_number']}: "
            f"PF={params['profit_factor']:.2f}, "
            f"WR={params['win_rate']:.1f}%, "
            f"Trades={params['total_trades']}"
        )


def generate_comparison_report(selected_configs: Dict[str, pd.Series]) -> str:
    """
    Generate markdown comparison report.

    Args:
        selected_configs: Dictionary of selected configs

    Returns:
        Markdown report string
    """
    report = [
        "# S5 (Long Squeeze) Optimized Configs Comparison",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "Three production configs generated from Pareto frontier optimization:",
        "",
        "| Profile | Trial | Fusion Threshold | Funding Z Min | RSI Min | Trades | PF | Win Rate |",
        "|---------|-------|-----------------|---------------|---------|--------|-----|----------|"
    ]

    for profile_name in ['conservative', 'balanced', 'aggressive']:
        params = selected_configs[profile_name]
        report.append(
            f"| {profile_name.capitalize():11s} | "
            f"{params['trial_number']:5d} | "
            f"{params['fusion_threshold']:16.3f} | "
            f"{params['funding_z_min']:13.2f} | "
            f"{params['rsi_min']:7.1f} | "
            f"{params['total_trades']:6d} | "
            f"{params['profit_factor']:3.2f} | "
            f"{params['win_rate']:8.1f}% |"
        )

    report.extend([
        "",
        "## Detailed Parameters",
        ""
    ])

    for profile_name in ['conservative', 'balanced', 'aggressive']:
        params = selected_configs[profile_name]
        report.extend([
            f"### {profile_name.capitalize()}",
            "",
            "```json",
            "{",
            f'  "fusion_threshold": {params["fusion_threshold"]:.4f},',
            f'  "funding_z_min": {params["funding_z_min"]:.2f},',
            f'  "rsi_min": {params["rsi_min"]:.1f},',
            f'  "liquidity_max": {params["liquidity_max"]:.3f},',
            f'  "oi_change_min": {params["oi_change_min"]:.3f},',
            f'  "cooldown_bars": {int(params["cooldown_bars"])},',
            f'  "atr_stop_mult": {params["atr_stop_mult"]:.2f},',
            f'  "trail_atr_mult": {params["trail_atr_mult"]:.2f}',
            "}",
            "```",
            ""
        ])

    report.extend([
        "## Usage",
        "",
        "**Conservative:** Use when prioritizing high profit factor over trade frequency",
        "```bash",
        "python3 bin/backtest_knowledge_v2.py --config configs/optimized/s5_conservative.json --start 2024-01-01 --end 2024-12-31",
        "```",
        "",
        "**Balanced:** Recommended for most production use cases",
        "```bash",
        "python3 bin/backtest_knowledge_v2.py --config configs/optimized/s5_balanced.json --start 2024-01-01 --end 2024-12-31",
        "```",
        "",
        "**Aggressive:** Use when seeking more trading opportunities",
        "```bash",
        "python3 bin/backtest_knowledge_v2.py --config configs/optimized/s5_aggressive.json --start 2024-01-01 --end 2024-12-31",
        "```",
        "",
        "## Validation",
        "",
        "All configs optimized on 2023 data (H1 train + H2 validation).",
        "Test on 2024 OOS data before production deployment.",
        ""
    ])

    return "\n".join(report)


def main():
    """Main config generation routine"""

    print("="*80)
    print("S5 (LONG SQUEEZE) PRODUCTION CONFIG GENERATOR")
    print("="*80)
    print()

    try:
        # Load Pareto results
        df_pareto = load_pareto_results()

        if len(df_pareto) == 0:
            logger.error("No Pareto-optimal solutions found")
            sys.exit(1)

        # Select 3 representative configs
        print("\n" + "-"*80)
        print("SELECTING REPRESENTATIVE CONFIGS")
        print("-"*80)

        selected = select_configs(df_pareto)

        print(f"\nSelected configs:")
        for profile_name, params in selected.items():
            print(f"  {profile_name.upper():12s}: "
                  f"Trial {params['trial_number']:3d}, "
                  f"PF={params['profit_factor']:.2f}, "
                  f"WR={params['win_rate']:.1f}%, "
                  f"Trades={params['total_trades']}")

        # Generate config files
        print("\n" + "-"*80)
        print("GENERATING CONFIG FILES")
        print("-"*80)

        generate_configs(selected)

        # Generate comparison report
        report = generate_comparison_report(selected)
        report_file = Path('configs/optimized/S5_CONFIGS_COMPARISON.md')
        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"\nComparison report saved: {report_file}")

        print("\n" + "="*80)
        print("CONFIG GENERATION COMPLETE")
        print("="*80)
        print("\nGenerated files:")
        print("  - configs/optimized/s5_conservative.json")
        print("  - configs/optimized/s5_balanced.json")
        print("  - configs/optimized/s5_aggressive.json")
        print("  - configs/optimized/S5_CONFIGS_COMPARISON.md")
        print("\nNext step: Validate on 2024 OOS data")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Config generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
