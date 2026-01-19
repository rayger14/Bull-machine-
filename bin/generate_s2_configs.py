#!/usr/bin/env python3
"""
S2 Config Generator from Pareto Frontier

Reads Optuna study results and generates production-ready configs:
- Conservative: Highest PF with trade count >= 8/year
- Balanced: Best trade-off between PF and trade frequency
- Aggressive: Most trades while maintaining PF > 1.3

Usage:
    python3 bin/generate_s2_configs.py

Output:
    - configs/optimized/s2_conservative.json
    - configs/optimized/s2_balanced.json
    - configs/optimized/s2_aggressive.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path("results/s2_calibration")
DB_PATH = RESULTS_DIR / "optuna_s2_calibration.db"
STUDY_NAME = "s2_failed_rally_calibration_v1"

OUTPUT_DIR = Path("configs/optimized")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ConfigProfile:
    """Configuration profile metadata"""
    name: str
    description: str
    trial_number: int
    harmonic_pf: float
    annual_trades: float
    max_drawdown: float
    win_rate: float
    sharpe_ratio: float
    parameters: Dict


def load_pareto_trials(study: optuna.Study) -> List[ConfigProfile]:
    """
    Load Pareto-optimal trials from study.

    Args:
        study: Optuna study

    Returns:
        List of ConfigProfile sorted by PF
    """
    if not study.best_trials:
        logger.error("No Pareto-optimal trials found")
        return []

    profiles = []

    for trial in study.best_trials:
        profile = ConfigProfile(
            name=f"trial_{trial.number}",
            description=f"Pareto solution from trial {trial.number}",
            trial_number=trial.number,
            harmonic_pf=trial.user_attrs.get('harmonic_pf', 0.0),
            annual_trades=trial.user_attrs.get('mean_annual_trades', 0.0),
            max_drawdown=trial.user_attrs.get('mean_max_dd', 0.0),
            win_rate=trial.user_attrs.get('mean_win_rate', 0.0),
            sharpe_ratio=trial.user_attrs.get('mean_sharpe', 0.0),
            parameters={
                'fusion_threshold': trial.params['fusion_threshold'],
                'wick_ratio_min': trial.params['wick_ratio_min'],
                'rsi_min': trial.params['rsi_min'],
                'volume_z_max': trial.params['volume_z_max'],
                'liquidity_max': trial.params['liquidity_max'],
                'cooldown_bars': trial.params['cooldown_bars'],
            }
        )
        profiles.append(profile)

    # Sort by PF descending
    profiles.sort(key=lambda p: p.harmonic_pf, reverse=True)

    return profiles


def select_conservative(profiles: List[ConfigProfile]) -> Optional[ConfigProfile]:
    """
    Select conservative config: Highest PF with trade count >= 8/year.

    Conservative = maximize quality over quantity.
    """
    candidates = [p for p in profiles if p.annual_trades >= 8.0]

    if not candidates:
        logger.warning("No profiles with >= 8 trades/year, using highest PF overall")
        return profiles[0] if profiles else None

    # Highest PF among candidates
    return max(candidates, key=lambda p: p.harmonic_pf)


def select_balanced(profiles: List[ConfigProfile]) -> Optional[ConfigProfile]:
    """
    Select balanced config: Best trade-off between PF and trade frequency.

    Balanced = minimize distance to ideal point (PF=2.0, trades=10).
    """
    if not profiles:
        return None

    # Normalize and compute distance to ideal
    ideal_pf = 2.0
    ideal_trades = 10.0

    best_profile = None
    best_score = float('inf')

    for p in profiles:
        # Normalize (0-1 scale)
        pf_norm = min(p.harmonic_pf / ideal_pf, 1.0)
        trades_norm = min(p.annual_trades / ideal_trades, 1.0)

        # Distance to ideal (lower is better)
        distance = ((1.0 - pf_norm) ** 2 + (1.0 - trades_norm) ** 2) ** 0.5

        if distance < best_score:
            best_score = distance
            best_profile = p

    return best_profile


def select_aggressive(profiles: List[ConfigProfile]) -> Optional[ConfigProfile]:
    """
    Select aggressive config: Most trades while maintaining PF > 1.3.

    Aggressive = maximize trade frequency while keeping quality acceptable.
    """
    candidates = [p for p in profiles if p.harmonic_pf >= 1.3]

    if not candidates:
        logger.warning("No profiles with PF >= 1.3, using highest trade count overall")
        return max(profiles, key=lambda p: p.annual_trades) if profiles else None

    # Most trades among candidates
    return max(candidates, key=lambda p: p.annual_trades)


def create_production_config(profile: ConfigProfile, config_type: str) -> Dict:
    """
    Create production config from profile.

    Args:
        profile: Selected config profile
        config_type: 'conservative', 'balanced', or 'aggressive'

    Returns:
        Production config dict
    """
    return {
        "version": f"s2_{config_type}_v1",
        "profile": f"s2_{config_type}",
        "description": f"S2 (Failed Rally) {config_type} config - Optuna calibrated (Trial {profile.trial_number})",

        "_optimization_metadata": {
            "trial_number": profile.trial_number,
            "profile_type": config_type,
            "performance": {
                "harmonic_pf": round(profile.harmonic_pf, 3),
                "annual_trades": round(profile.annual_trades, 1),
                "max_drawdown_pct": round(profile.max_drawdown, 2),
                "win_rate_pct": round(profile.win_rate, 2),
                "sharpe_ratio": round(profile.sharpe_ratio, 3)
            },
            "optimized_on": "2022 bear market + 2023 H1",
            "cv_folds": ["2022_H1", "2022_H2", "2023_H1"]
        },

        "adaptive_fusion": True,

        "regime_classifier": {
            "model_path": "models/regime_classifier_gmm.pkl",
            "feature_order": [
                "VIX", "DXY", "MOVE", "YIELD_2Y", "YIELD_10Y",
                "USDT.D", "BTC.D", "TOTAL", "TOTAL2",
                "funding", "oi", "rv_20d", "rv_60d"
            ],
            "zero_fill_missing": False
        },

        "ml_filter": {
            "enabled": False
        },

        "fusion": {
            "entry_threshold_confidence": 0.36,
            "weights": {
                "wyckoff": 0.35,
                "liquidity": 0.30,
                "momentum": 0.35,
                "smc": 0.0
            }
        },

        "archetypes": {
            "use_archetypes": True,
            "max_trades_per_day": 8,

            # ONLY S2 enabled (production should enable other archetypes)
            "enable_A": False,
            "enable_B": False,
            "enable_C": False,
            "enable_D": False,
            "enable_E": False,
            "enable_F": False,
            "enable_G": False,
            "enable_H": False,
            "enable_K": False,
            "enable_L": False,
            "enable_M": False,
            "enable_S1": False,
            "enable_S2": True,
            "enable_S3": False,
            "enable_S4": False,
            "enable_S5": False,
            "enable_S6": False,
            "enable_S7": False,
            "enable_S8": False,

            "thresholds": {
                "min_liquidity": round(profile.parameters['liquidity_max'], 3)
            },

            "failed_rally": {
                "direction": "short",
                "archetype_weight": 2.0,
                "fusion_threshold": round(profile.parameters['fusion_threshold'], 3),
                "final_fusion_gate": round(profile.parameters['fusion_threshold'], 3),
                "cooldown_bars": int(profile.parameters['cooldown_bars']),
                "max_risk_pct": 0.015,
                "atr_stop_mult": 2.0,
                "wick_ratio_min": round(profile.parameters['wick_ratio_min'], 2),
                "rsi_min": round(profile.parameters['rsi_min'], 1),
                "volume_z_max": round(profile.parameters['volume_z_max'], 2),
                "require_rsi_divergence": False,
                "use_runtime_features": True,

                "weights": {
                    "ob_retest": 0.25,
                    "wick_rejection": 0.25,
                    "rsi_signal": 0.20,
                    "volume_fade": 0.15,
                    "tf4h_confirm": 0.15
                }
            },

            "routing": {
                "risk_on": {
                    "weights": {"failed_rally": 0.0},
                    "final_gate_delta": 0.0,
                    "_comment": "S2 disabled in bull markets"
                },
                "neutral": {
                    "weights": {"failed_rally": 0.5},
                    "final_gate_delta": 0.0,
                    "_comment": "S2 reduced weight in neutral markets"
                },
                "risk_off": {
                    "weights": {"failed_rally": 2.0},
                    "final_gate_delta": 0.02,
                    "_comment": "S2 full weight in bear markets"
                },
                "crisis": {
                    "weights": {"failed_rally": 2.5},
                    "final_gate_delta": 0.04,
                    "_comment": "S2 max weight in crisis"
                }
            },

            "exits": {
                "failed_rally": {
                    "enable_trail": True,
                    "trail_atr_mult": 1.5,
                    "time_limit_hours": 48
                }
            }
        },

        "context": {
            "crisis_fuse": {"enabled": False}
        },

        "risk": {
            "base_risk_pct": 0.015,
            "max_position_size_pct": 0.15,
            "max_portfolio_risk_pct": 0.08
        }
    }


def main():
    """Generate S2 configs from Pareto frontier"""
    print("="*80)
    print("S2 CONFIG GENERATOR")
    print("="*80)
    print()

    # Load study
    if not DB_PATH.exists():
        logger.error(f"Database not found: {DB_PATH}")
        logger.error("Run bin/optimize_s2_calibration.py first")
        return 1

    logger.info(f"Loading study from: {DB_PATH}")
    storage = f"sqlite:///{DB_PATH}"

    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=storage
        )
    except KeyError:
        logger.error(f"Study '{STUDY_NAME}' not found in database")
        return 1

    logger.info(f"Loaded study with {len(study.trials)} trials")
    logger.info(f"Pareto solutions: {len(study.best_trials)}")
    print()

    if len(study.best_trials) == 0:
        logger.error("No Pareto-optimal solutions found")
        return 1

    # Load profiles
    profiles = load_pareto_trials(study)
    logger.info(f"Extracted {len(profiles)} Pareto profiles")
    print()

    # Select configs
    print("Selecting optimal configurations...")
    print()

    conservative = select_conservative(profiles)
    balanced = select_balanced(profiles)
    aggressive = select_aggressive(profiles)

    if not conservative or not balanced or not aggressive:
        logger.error("Failed to select all config profiles")
        return 1

    # Display selections
    print("Selected Configurations:")
    print()

    configs = [
        ("Conservative", conservative),
        ("Balanced", balanced),
        ("Aggressive", aggressive)
    ]

    for name, profile in configs:
        print(f"{name:12} (Trial {profile.trial_number:3}): "
              f"PF={profile.harmonic_pf:.2f}, "
              f"Trades={profile.annual_trades:.1f}/yr, "
              f"DD={profile.max_drawdown:.1f}%, "
              f"WR={profile.win_rate:.1f}%")

    print()

    # Generate config files
    print("Generating config files...")
    print()

    for config_type, profile in configs:
        config_type_lower = config_type.lower()

        config = create_production_config(profile, config_type_lower)

        output_path = OUTPUT_DIR / f"s2_{config_type_lower}.json"

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"  [{config_type:12}] {output_path}")

    print()
    print("="*80)
    print("CONFIG GENERATION COMPLETE")
    print("="*80)
    print()
    print("Files generated:")
    for config_type, _ in configs:
        config_type_lower = config_type.lower()
        print(f"  - configs/optimized/s2_{config_type_lower}.json")
    print()
    print("Next steps:")
    print("  1. Review generated configs")
    print("  2. Run validation: bin/backtest_knowledge_v2.py --config configs/optimized/s2_balanced.json")
    print("  3. Update S2_CALIBRATION_RESULTS.md with findings")

    return 0


if __name__ == '__main__':
    sys.exit(main())
