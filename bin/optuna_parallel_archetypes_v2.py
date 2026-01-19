#!/usr/bin/env python3
"""
Parallel Optuna Archetype Optimization with Correct Two-Layer Threshold Architecture

ARCHITECTURAL PRINCIPLES:
=======================
1. GLOBAL SAFETY RAILS (FIXED - Never Optimize):
   - min_liquidity_floor: 0.05 (hard minimum)
   - vix_panic_threshold: 30.0 (crisis detection)
   - funding_z_extreme: 3.0 (extreme crowding)
   - crisis_fuse_enabled: true

2. ARCHETYPE-SPECIFIC THRESHOLDS (Optimize Per Pattern):
   - fusion_threshold: Entry confidence per archetype
   - min_liquidity: Minimum liquidity requirement per archetype
   - archetype_weight: Pattern scoring multiplier
   - Pattern filters: funding_z_min, rsi_min, vol_z_min, etc.

3. CONFIG COMPATIBILITY:
   - Reads from MVP config structure: configs/mvp/mvp_bull_market_v1.json
   - Applies parameters to correct nested paths:
     * archetypes.thresholds[pattern].fusion_threshold
     * archetypes[pattern].archetype_weight
     * archetypes[pattern].final_fusion_gate (merged with fusion_threshold)

PARALLEL EXECUTION:
==================
- 4 independent Optuna studies (one per archetype group)
- Hyperband pruning for early stopping
- Multi-fidelity evaluation: 1mo → 3mo → 9mo
- Expected runtime: 6-8 hours

ARCHETYPE GROUPS:
================
1. Trap Within Trend (A, G, K) - momentum-based reversals
2. Order Block Retest (B, H, L) - structure-based entries
3. BOS/CHOCH (C) - continuation patterns
4. Long Squeeze (S5) - funding rate cascades

Usage:
    python bin/optuna_parallel_archetypes_v2.py --trials 100 --base-config configs/mvp/mvp_bull_market_v1.json
    python bin/optuna_parallel_archetypes_v2.py --resume  # Resume from checkpoints
    python bin/optuna_parallel_archetypes_v2.py --test-trial  # Run single trial to test config application
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Global Safety Rails (FIXED - Never Optimize)
# ============================================================================

GLOBAL_SAFETY_RAILS = {
    'min_liquidity_floor': 0.05,          # Hard minimum, never go below
    'vix_panic_threshold': 30.0,           # Crisis detection
    'move_panic_threshold': 120.0,         # Bond volatility crisis
    'dxy_extreme_threshold': 105.0,        # Dollar crisis
    'funding_z_extreme': 3.0,              # Extreme crowding
    'crisis_fuse_enabled': True,           # Always enabled for safety
    'crisis_fuse_lookback_hours': 24,      # Standard lookback
    'max_portfolio_risk_pct': 0.10,        # Never risk more than 10% total
}


# ============================================================================
# Archetype Group Definitions
# ============================================================================

ARCHETYPE_GROUPS = {
    # CORRECTED MAPPINGS - Now match runtime behavior
    # A = spring (trap_reversal), G = liquidity_sweep (re_accumulate),
    # H = trap_within_trend, K = wick_trap
    'spring_utad': {
        'archetypes': ['A'],
        'canonical': ['spring'],  # A queries 'spring' in runtime
        'description': 'PTI-based spring and UTAD patterns',
        'trader_type': 'Moneytaur',
    },
    'liquidity_sweep': {
        'archetypes': ['G'],
        'canonical': ['liquidity_sweep'],  # G queries 'liquidity_sweep' in runtime
        'description': 'Liquidity sweep and re-accumulation patterns',
        'trader_type': 'Moneytaur',
    },
    'trap_within_trend': {
        'archetypes': ['H'],  # H is the REAL trap_within_trend!
        'canonical': ['trap_within_trend'],  # H queries 'trap_within_trend' in runtime
        'description': 'Trap-within-trend continuation patterns',
        'trader_type': 'Moneytaur',
    },
    'wick_trap': {
        'archetypes': ['K'],  # K is wick_trap, not spring!
        'canonical': ['wick_trap_moneytaur'],  # K queries 'wick_trap_moneytaur' in runtime
        'description': 'Wick-based trap and momentum exhaustion',
        'trader_type': 'Moneytaur',
    },
    'order_block_retest': {
        'archetypes': ['B', 'L'],  # Removed H (moved to trap_within_trend)
        'canonical': ['order_block_retest', 'volume_exhaustion'],
        'description': 'Structure-based order block retests',
        'trader_type': 'Zeroika',
    },
    'bos_choch': {
        'archetypes': ['C'],
        'canonical': ['bos_choch_reversal'],  # Corrected canonical name
        'description': 'Break of Structure and Change of Character',
        'trader_type': 'Generic',
    },
    'long_squeeze': {
        'archetypes': ['S5'],
        'canonical': ['long_squeeze'],
        'description': 'Funding rate cascade patterns',
        'trader_type': 'Moneytaur',
    }
}


# ============================================================================
# Config Generator with MVP Structure Compatibility
# ============================================================================

class MVPConfigGenerator:
    """
    Generates trial configs compatible with MVP config structure.

    Applies parameters to correct nested paths:
    - archetypes.thresholds[pattern].fusion_threshold
    - archetypes[pattern].archetype_weight
    - archetypes[pattern].final_fusion_gate
    """

    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()

    def _load_base_config(self) -> dict:
        """Load and validate base MVP config."""
        try:
            with open(self.base_config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded base config: {self.base_config_path}")

            # Validate MVP structure
            if 'archetypes' not in config:
                raise ValueError("Missing 'archetypes' section in config")

            return config
        except FileNotFoundError:
            logger.error(f"Base config not found: {self.base_config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in base config: {e}")
            raise

    def generate(
        self,
        global_params: Dict[str, float],
        archetype_params: Dict[str, Dict[str, float]],
        group_name: str
    ) -> str:
        """
        Generate trial config with suggested parameters.

        Args:
            global_params: Global fusion weights (NOT global safety rails)
            archetype_params: Per-archetype threshold parameters
            group_name: Archetype group being optimized

        Returns:
            Path to temporary config file
        """
        # Deep copy base config
        trial_config = deepcopy(self.base_config)

        # 1. ENFORCE GLOBAL SAFETY RAILS (always fixed)
        self._apply_safety_rails(trial_config)

        # 2. Apply global fusion weights
        self._apply_fusion_weights(trial_config, global_params)

        # 3. Enable only this group's archetypes
        self._enable_group_archetypes(trial_config, group_name)

        # 4. Apply archetype-specific parameters
        self._apply_archetype_params(trial_config, archetype_params)

        # 5. Validate config structure
        self._validate_config(trial_config)

        # Write to temp file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False,
            prefix=f'optuna_{group_name}_'
        )

        json.dump(trial_config, temp_file, indent=2)
        temp_file.close()

        return temp_file.name

    def _apply_safety_rails(self, config: dict):
        """Apply fixed global safety rails (never optimized)."""
        # Ensure context section exists
        if 'context' not in config:
            config['context'] = {}

        # Apply crisis fuse settings
        config['context']['crisis_fuse'] = {
            'enabled': GLOBAL_SAFETY_RAILS['crisis_fuse_enabled'],
            'lookback_hours': GLOBAL_SAFETY_RAILS['crisis_fuse_lookback_hours'],
            'allow_one_trade_if_fusion_confidence_ge': 0.8  # Conservative default
        }

        # Apply risk limits
        if 'risk' not in config:
            config['risk'] = {}

        config['risk']['max_portfolio_risk_pct'] = GLOBAL_SAFETY_RAILS['max_portfolio_risk_pct']

        # Note: VIX/DXY/MOVE thresholds are typically in 'context' section
        # but MVP configs may not have explicit crisis thresholds yet
        # These are applied at runtime by the engine's crisis detection logic

        logger.debug("Applied global safety rails")

    def _apply_fusion_weights(self, config: dict, global_params: Dict[str, float]):
        """Apply global fusion weights (NOT safety rails)."""
        if 'fusion' not in config:
            config['fusion'] = {}

        if 'weights' not in config['fusion']:
            config['fusion']['weights'] = {}

        # Normalize weights to sum to 1.0
        total = sum(global_params[k] for k in ['w_wyckoff', 'w_liquidity', 'w_momentum'])

        config['fusion']['weights']['wyckoff'] = global_params['w_wyckoff'] / total
        config['fusion']['weights']['liquidity'] = global_params['w_liquidity'] / total
        config['fusion']['weights']['momentum'] = global_params['w_momentum'] / total
        config['fusion']['weights']['smc'] = 0.0  # SMC disabled for MVP

        logger.debug(f"Applied fusion weights: {config['fusion']['weights']}")

    def _enable_group_archetypes(self, config: dict, group_name: str):
        """Enable only archetypes in this group."""
        group = ARCHETYPE_GROUPS[group_name]
        archetypes_cfg = config.get('archetypes', {})

        # Disable all archetypes first
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M',
                       'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
            archetypes_cfg[f'enable_{letter}'] = False

        # Enable only this group's archetypes
        for letter in group['archetypes']:
            archetypes_cfg[f'enable_{letter}'] = True

        logger.debug(f"Enabled archetypes: {group['archetypes']}")

    def _apply_archetype_params(
        self,
        config: dict,
        archetype_params: Dict[str, Dict[str, float]]
    ):
        """
        Apply archetype-specific parameters to MVP config structure.

        MVP configs use:
        - archetypes.thresholds[pattern].fusion_threshold
        - archetypes.thresholds[pattern].max_risk_pct
        - archetypes[pattern].archetype_weight
        - archetypes[pattern].final_fusion_gate
        """
        archetypes_cfg = config.get('archetypes', {})

        # Ensure thresholds section exists
        if 'thresholds' not in archetypes_cfg:
            archetypes_cfg['thresholds'] = {}

        thresholds = archetypes_cfg['thresholds']

        for pattern_name, params in archetype_params.items():
            # 1. Apply to thresholds section
            if pattern_name not in thresholds:
                thresholds[pattern_name] = {}

            pattern_thresholds = thresholds[pattern_name]

            # Apply fusion threshold
            if 'fusion_threshold' in params:
                pattern_thresholds['fusion_threshold'] = params['fusion_threshold']

            # Apply pattern-specific filters
            for key in ['funding_z_min', 'rsi_min', 'vol_z_min', 'vol_z_max',
                       'liquidity_max', 'adx_threshold', 'boms_strength_min',
                       'wyckoff_min', 'disp_atr_multiplier', 'pti_score_threshold']:
                if key in params:
                    pattern_thresholds[key] = params[key]

            # 2. Apply to top-level archetype config (where archetype_weight lives)
            if pattern_name not in archetypes_cfg:
                archetypes_cfg[pattern_name] = {}

            pattern_cfg = archetypes_cfg[pattern_name]

            # Apply archetype weight
            if 'archetype_weight' in params:
                pattern_cfg['archetype_weight'] = params['archetype_weight']

            # Apply final fusion gate (same as fusion_threshold in MVP)
            if 'fusion_threshold' in params:
                pattern_cfg['final_fusion_gate'] = params['fusion_threshold']

            # Apply cooldown if specified
            if 'cooldown_bars' in params:
                pattern_cfg['cooldown_bars'] = params['cooldown_bars']

        logger.debug(f"Applied params for {len(archetype_params)} archetypes")

    def _validate_config(self, config: dict):
        """Validate final config structure."""
        required_sections = ['fusion', 'archetypes', 'context', 'risk']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        # Validate at least one archetype is enabled
        archetypes_cfg = config.get('archetypes', {})
        enabled = [k for k, v in archetypes_cfg.items()
                   if k.startswith('enable_') and v is True]

        if not enabled:
            raise ValueError("No archetypes enabled in trial config")

        logger.debug(f"Config validated: {len(enabled)} archetypes enabled")


# ============================================================================
# Backtest Execution
# ============================================================================

def run_backtest(
    config_path: str,
    start_date: str,
    end_date: str,
    asset: str = "BTC",
    timeout: int = 120
) -> Optional[Dict]:
    """
    Run backtest_knowledge_v2.py and extract metrics.

    Args:
        config_path: Path to config JSON
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        asset: Asset symbol (BTC, ETH, etc.)
        timeout: Timeout in seconds

    Returns:
        Dict with metrics or None on error
    """
    cmd = [
        "python3",
        "bin/backtest_knowledge_v2.py",
        "--asset", asset,
        "--start", start_date,
        "--end", end_date,
        "--config", config_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            cwd=Path(__file__).parent.parent
        )

        output = result.stdout + result.stderr

        # Extract metrics
        metrics = {
            'pnl': 0.0,
            'trades': 0,
            'roi': 0.0,
            'win_rate': 0.0,
            'drawdown': 0.0,
            'profit_factor': 0.0,
            'sharpe': 0.0,
            'avg_pnl_per_trade': 0.0
        }

        # Parse metrics
        pnl_match = re.search(r'Total PNL:\s+\$?([-\d,\.]+)', output)
        if pnl_match:
            metrics['pnl'] = float(pnl_match.group(1).replace(',', ''))

        trades_match = re.search(r'Total Trades:\s+(\d+)', output)
        if trades_match:
            metrics['trades'] = int(trades_match.group(1))

        roi_match = re.search(r'ROI:\s+([-\d\.]+)%', output)
        if roi_match:
            metrics['roi'] = float(roi_match.group(1))

        wr_match = re.search(r'Win Rate:\s+([\d\.]+)%', output)
        if wr_match:
            metrics['win_rate'] = float(wr_match.group(1))

        dd_match = re.search(r'Max Drawdown:\s+([\d\.]+)%', output)
        if dd_match:
            metrics['drawdown'] = float(dd_match.group(1))

        pf_match = re.search(r'Profit Factor:\s+([\d\.]+)', output)
        if pf_match:
            metrics['profit_factor'] = float(pf_match.group(1))

        sharpe_match = re.search(r'Sharpe:\s+([-\d\.]+)', output)
        if sharpe_match:
            metrics['sharpe'] = float(sharpe_match.group(1))

        if metrics['trades'] > 0:
            metrics['avg_pnl_per_trade'] = metrics['pnl'] / metrics['trades']

        return metrics

    except subprocess.TimeoutExpired:
        logger.warning(f"Backtest timeout for {start_date} to {end_date}")
        return None
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return None


# ============================================================================
# Multi-Fidelity Training
# ============================================================================

def get_training_periods(fidelity: int) -> Tuple[str, str]:
    """
    Get training period based on fidelity level.

    Fidelity levels:
    - 0: 1 month (2024-01-01 to 2024-01-31) - fast pruning
    - 1: 3 months (2024-01-01 to 2024-03-31) - medium validation
    - 2: 9 months (2024-01-01 to 2024-09-30) - full evaluation

    Args:
        fidelity: Fidelity level (0-2)

    Returns:
        (start_date, end_date) tuple
    """
    periods = {
        0: ("2024-01-01", "2024-01-31"),   # 1 month
        1: ("2024-01-01", "2024-03-31"),   # 3 months
        2: ("2024-01-01", "2024-09-30"),   # 9 months
    }
    return periods.get(fidelity, periods[2])


# ============================================================================
# Parameter Space Definitions (ARCHETYPE-SPECIFIC ONLY)
# ============================================================================

def suggest_global_fusion_weights(trial: optuna.Trial) -> dict:
    """
    Suggest ONLY global fusion weights (NOT safety rails).

    Safety rails are NEVER optimized - they're fixed constants.

    Args:
        trial: Optuna trial

    Returns:
        Dict with fusion weights
    """
    w_wyckoff = trial.suggest_float('w_wyckoff', 0.15, 0.55, step=0.05)
    w_liquidity = trial.suggest_float('w_liquidity', 0.15, 0.55, step=0.05)
    w_momentum = trial.suggest_float('w_momentum', 0.05, 0.45, step=0.05)

    return {
        'w_wyckoff': w_wyckoff,
        'w_liquidity': w_liquidity,
        'w_momentum': w_momentum
    }


def suggest_archetype_params(
    trial: optuna.Trial,
    group_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Suggest archetype-specific parameters for a group.

    Each archetype gets:
    - fusion_threshold: Entry confidence
    - archetype_weight: Scoring multiplier
    - Pattern-specific filters

    Args:
        trial: Optuna trial
        group_name: Archetype group name

    Returns:
        Dict mapping pattern_name -> {param: value}
    """
    params = {}

    if group_name == 'trap_within_trend':
        # Trap Within Trend (primary pattern)
        params['trap_within_trend'] = {
            'fusion_threshold': trial.suggest_float('trap_fusion', 0.30, 0.48, step=0.01),
            'archetype_weight': trial.suggest_float('trap_weight', 0.85, 1.30, step=0.05),
            'adx_threshold': trial.suggest_float('trap_adx', 18.0, 35.0, step=1.0),
            'cooldown_bars': trial.suggest_int('trap_cooldown', 10, 20, step=2),
        }

        # Liquidity Sweep
        params['liquidity_sweep'] = {
            'fusion_threshold': trial.suggest_float('sweep_fusion', 0.28, 0.48, step=0.01),
            'archetype_weight': trial.suggest_float('sweep_weight', 0.90, 1.25, step=0.05),
            'boms_strength_min': trial.suggest_float('sweep_boms', 0.25, 0.55, step=0.05),
        }

        # Spring (Wyckoff)
        params['spring'] = {
            'fusion_threshold': trial.suggest_float('spring_fusion', 0.26, 0.44, step=0.01),
            'archetype_weight': trial.suggest_float('spring_weight', 0.95, 1.20, step=0.05),
            'pti_score_threshold': trial.suggest_float('spring_pti', 0.25, 0.55, step=0.05),
            'disp_atr_multiplier': trial.suggest_float('spring_disp', 0.5, 1.3, step=0.1),
        }

    elif group_name == 'order_block_retest':
        # Order Block Retest (primary)
        params['order_block_retest'] = {
            'fusion_threshold': trial.suggest_float('ob_fusion', 0.28, 0.48, step=0.01),
            'archetype_weight': trial.suggest_float('ob_weight', 0.85, 1.35, step=0.05),
            'boms_strength_min': trial.suggest_float('ob_boms', 0.20, 0.50, step=0.02),
            'wyckoff_min': trial.suggest_float('ob_wyckoff', 0.25, 0.55, step=0.02),
            'cooldown_bars': trial.suggest_int('ob_cooldown', 8, 16, step=2),
        }

        # Momentum Continuation
        params['momentum_continuation'] = {
            'fusion_threshold': trial.suggest_float('mom_fusion', 0.30, 0.50, step=0.01),
            'archetype_weight': trial.suggest_float('mom_weight', 0.90, 1.25, step=0.05),
            'adx_threshold': trial.suggest_float('mom_adx', 22.0, 40.0, step=2.0),
        }

        # Volume Exhaustion
        params['volume_exhaustion'] = {
            'fusion_threshold': trial.suggest_float('volexh_fusion', 0.30, 0.48, step=0.01),
            'archetype_weight': trial.suggest_float('volexh_weight', 0.85, 1.20, step=0.05),
            'vol_z_min': trial.suggest_float('volexh_volz', 0.6, 1.8, step=0.1),
            'rsi_min': trial.suggest_float('volexh_rsi', 62.0, 78.0, step=2.0),
        }

    elif group_name == 'bos_choch':
        # Wick Trap (BOS/CHOCH continuation)
        params['wick_trap'] = {
            'fusion_threshold': trial.suggest_float('bos_fusion', 0.32, 0.52, step=0.01),
            'archetype_weight': trial.suggest_float('bos_weight', 0.85, 1.25, step=0.05),
            'disp_atr_multiplier': trial.suggest_float('bos_disp', 0.6, 1.6, step=0.1),
            'cooldown_bars': trial.suggest_int('bos_cooldown', 10, 18, step=2),
        }

    elif group_name == 'long_squeeze':
        # Long Squeeze (funding cascade)
        params['long_squeeze'] = {
            'fusion_threshold': trial.suggest_float('squeeze_fusion', 0.26, 0.46, step=0.01),
            'archetype_weight': trial.suggest_float('squeeze_weight', 0.40, 0.80, step=0.05),
            'funding_z_min': trial.suggest_float('squeeze_funding', 0.8, 2.2, step=0.1),
            'rsi_min': trial.suggest_float('squeeze_rsi', 62.0, 78.0, step=2.0),
            'liquidity_max': trial.suggest_float('squeeze_liq_max', 0.15, 0.35, step=0.02),
            'cooldown_bars': trial.suggest_int('squeeze_cooldown', 6, 12, step=2),
        }

    return params


# ============================================================================
# Objective Function
# ============================================================================

def compute_objective_score(metrics: Dict, fidelity: int) -> float:
    """
    Compute objective score for optimization.

    Scoring formula:
    - Base: Profit Factor × (1 + win_rate/100) × sqrt(trades)
    - Penalties: -drawdown/10, -overtrading
    - Bonuses: +sharpe, +consistency

    Args:
        metrics: Backtest metrics dict
        fidelity: Fidelity level (0-2)

    Returns:
        Objective score (higher is better)
    """
    if not metrics or metrics['trades'] < 3:
        return -1000.0  # Penalize configs with < 3 trades

    pf = max(metrics['profit_factor'], 0.1)
    wr = metrics['win_rate']
    trades = metrics['trades']
    dd = metrics['drawdown']
    sharpe = metrics.get('sharpe', 0.0)

    # Base score: PF × win rate × trade consistency
    base_score = pf * (1 + wr / 100.0) * (trades ** 0.5)

    # Drawdown penalty (lighter at low fidelity)
    dd_penalty = dd / (10.0 if fidelity == 2 else 20.0)

    # Overtrading penalty (scale with fidelity)
    max_trades = [30, 60, 100][fidelity]
    overtrade_penalty = max(0, (trades - max_trades) * 0.1)

    # Sharpe bonus (only at full fidelity)
    sharpe_bonus = max(sharpe, 0) * 0.5 if fidelity == 2 else 0

    # Final score
    score = base_score - dd_penalty - overtrade_penalty + sharpe_bonus

    return score


# ============================================================================
# Optuna Study Runner (per archetype group)
# ============================================================================

def optimize_archetype_group(
    group_name: str,
    base_config_path: str,
    n_trials: int,
    storage_path: str,
    progress_queue: mp.Queue
) -> Dict:
    """
    Run Optuna optimization for a single archetype group.

    Args:
        group_name: Archetype group name
        base_config_path: Path to base MVP config
        n_trials: Number of trials to run
        storage_path: Path to SQLite storage
        progress_queue: Multiprocessing queue for progress updates

    Returns:
        Best trial parameters dict
    """
    group = ARCHETYPE_GROUPS[group_name]
    study_name = f"archetype_{group_name}_v2"

    logger.info(f"Starting optimization for {group_name}: {group['description']}")
    logger.info(f"Archetypes: {', '.join(group['archetypes'])}")

    # Create config generator
    config_gen = MVPConfigGenerator(base_config_path)

    # Create Optuna study with Hyperband pruner
    sampler = TPESampler(seed=42, n_startup_trials=10)
    pruner = HyperbandPruner(
        min_resource=1,
        max_resource=3,
        reduction_factor=3
    )

    # Use separate database file per group to avoid SQLite concurrency issues
    group_storage_path = storage_path.replace('.db', f'_{group_name}.db')

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{group_storage_path}",
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    def objective(trial: optuna.Trial) -> float:
        """Objective function for this archetype group."""
        # Multi-fidelity: start at fidelity 0, increase based on performance
        fidelity = trial.suggest_int('_fidelity', 0, 2)
        start_date, end_date = get_training_periods(fidelity)

        # Suggest parameters
        global_params = suggest_global_fusion_weights(trial)
        archetype_params = suggest_archetype_params(trial, group_name)

        # Generate config
        config_path = config_gen.generate(global_params, archetype_params, group_name)

        try:
            # Run backtest
            metrics = run_backtest(config_path, start_date, end_date, asset="BTC")

            if metrics is None:
                return -1000.0

            # Compute score
            score = compute_objective_score(metrics, fidelity)

            # Report for pruning
            trial.report(score, step=fidelity)

            # Update progress
            progress_queue.put({
                'group': group_name,
                'trial': trial.number,
                'score': score,
                'pf': metrics['profit_factor'],
                'trades': metrics['trades'],
                'fidelity': fidelity
            })

            # Prune if needed
            if trial.should_prune():
                logger.info(f"[{group_name}] Trial {trial.number} pruned at fidelity {fidelity}")
                raise optuna.TrialPruned()

            return score

        finally:
            # Cleanup temp config
            try:
                os.unlink(config_path)
            except:
                pass

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )

    # Return best params
    best_trial = study.best_trial
    logger.info(f"[{group_name}] Best trial: {best_trial.number}, score: {best_trial.value:.2f}")

    return {
        'group': group_name,
        'best_params': best_trial.params,
        'best_score': best_trial.value,
        'trial_number': best_trial.number,
        'n_trials': len(study.trials),
        'storage_path': group_storage_path
    }


# ============================================================================
# Progress Monitor
# ============================================================================

def monitor_progress(progress_queue: mp.Queue, total_trials: int, n_groups: int):
    """Monitor and display progress from all worker processes."""
    logger.info(f"Monitoring {n_groups} parallel studies with {total_trials} trials each")

    group_progress = defaultdict(lambda: {'trials': 0, 'best_score': -float('inf')})
    start_time = time.time()

    while True:
        try:
            update = progress_queue.get(timeout=1)

            if update == 'DONE':
                break

            group = update['group']
            trial = update['trial']
            score = update['score']

            # Update progress
            group_progress[group]['trials'] = trial + 1
            if score > group_progress[group]['best_score']:
                group_progress[group]['best_score'] = score

            # Display summary
            elapsed = time.time() - start_time
            total_completed = sum(p['trials'] for p in group_progress.values())

            logger.info(
                f"Progress: {total_completed}/{total_trials * n_groups} trials | "
                f"Elapsed: {elapsed/3600:.1f}h | "
                f"Group: {group} | "
                f"Trial: {trial} | "
                f"Score: {score:.2f} (PF={update['pf']:.2f}, trades={update['trades']})"
            )

        except mp.queues.Empty:
            continue
        except KeyboardInterrupt:
            logger.info("Progress monitor interrupted")
            break


# ============================================================================
# Result Aggregation
# ============================================================================

def aggregate_results(
    results: List[Dict],
    base_config_path: str,
    output_path: str
):
    """
    Aggregate best parameters from all groups into unified config.

    Args:
        results: List of optimization results per group
        base_config_path: Path to base MVP config
        output_path: Path to save unified config
    """
    logger.info("Aggregating results from all archetype groups...")

    # Load base config
    with open(base_config_path, 'r') as f:
        unified = json.load(f)

    # Extract best global params from best overall group
    best_group = max(results, key=lambda x: x['best_score'])
    global_params = {k: v for k, v in best_group['best_params'].items()
                     if k.startswith('w_')}

    # Apply global fusion weights
    if all(k in global_params for k in ['w_wyckoff', 'w_liquidity', 'w_momentum']):
        total = sum(global_params[k] for k in ['w_wyckoff', 'w_liquidity', 'w_momentum'])
        unified['fusion']['weights'] = {
            'wyckoff': global_params['w_wyckoff'] / total,
            'liquidity': global_params['w_liquidity'] / total,
            'momentum': global_params['w_momentum'] / total,
            'smc': 0.0
        }

    # Apply per-archetype params from each group
    for result in results:
        group_name = result['group']
        group = ARCHETYPE_GROUPS[group_name]
        params = result['best_params']

        # Extract archetype-specific params (exclude global and internal)
        arch_params = {k: v for k, v in params.items()
                       if not k.startswith(('w_', '_'))}

        # Map trial params back to archetype configs
        for canonical_name in group['canonical']:
            # Find params for this archetype
            pattern_params = {}
            prefix = canonical_name.split('_')[0]  # e.g., 'trap', 'ob', 'spring'

            for key, value in arch_params.items():
                if key.startswith(prefix):
                    param_name = key.replace(f'{prefix}_', '')
                    pattern_params[param_name] = value

            if pattern_params:
                # Map short parameter names to canonical long-form names
                # This prevents shadowing where baseline has 'fusion_threshold'
                # but optimizer writes 'fusion', causing runtime to use wrong value
                PARAM_NAME_MAP = {
                    'fusion': 'fusion_threshold',
                    'adx': 'adx_threshold',
                    'weight': 'archetype_weight',
                    'cooldown': 'cooldown_bars',
                }

                # Expand short names to canonical names
                expanded_params = {}
                for key, value in pattern_params.items():
                    canonical_key = PARAM_NAME_MAP.get(key, key)
                    expanded_params[canonical_key] = value

                # Apply to thresholds section
                if 'thresholds' not in unified['archetypes']:
                    unified['archetypes']['thresholds'] = {}

                if canonical_name not in unified['archetypes']['thresholds']:
                    unified['archetypes']['thresholds'][canonical_name] = {}

                unified['archetypes']['thresholds'][canonical_name].update(expanded_params)

                # Apply to top-level archetype section (using expanded_params)
                if canonical_name not in unified['archetypes']:
                    unified['archetypes'][canonical_name] = {}

                if 'archetype_weight' in expanded_params:
                    unified['archetypes'][canonical_name]['archetype_weight'] = expanded_params['archetype_weight']

                if 'fusion_threshold' in expanded_params:
                    unified['archetypes'][canonical_name]['final_fusion_gate'] = expanded_params['fusion_threshold']

    # Enable all optimized archetypes
    for result in results:
        group = ARCHETYPE_GROUPS[result['group']]
        for letter in group['archetypes']:
            unified['archetypes'][f'enable_{letter}'] = True

    # Add metadata
    unified['_optimization_metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'optimizer_version': 'v2_two_layer_architecture',
        'groups_optimized': [r['group'] for r in results],
        'total_trials': sum(r['n_trials'] for r in results),
        'best_scores': {r['group']: r['best_score'] for r in results},
        'global_safety_rails': GLOBAL_SAFETY_RAILS,
        '_note': 'Global safety rails are FIXED and were NOT optimized'
    }

    # Save unified config
    with open(output_path, 'w') as f:
        json.dump(unified, f, indent=2)

    logger.info(f"Unified config saved to {output_path}")

    # Generate comparison table
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"{'Group':<25} {'Best Score':>12} {'Trials':>8} {'Archetypes':<30}")
    print("-" * 80)
    for result in sorted(results, key=lambda x: x['best_score'], reverse=True):
        group = ARCHETYPE_GROUPS[result['group']]
        print(
            f"{result['group']:<25} "
            f"{result['best_score']:>12.2f} "
            f"{result['n_trials']:>8} "
            f"{', '.join(group['archetypes']):<30}"
        )
    print("=" * 80)


# ============================================================================
# Single Trial Test
# ============================================================================

def test_single_trial(base_config_path: str):
    """
    Run a single trial to test config generation and application.

    Validates:
    1. Config can be generated with trial parameters
    2. Parameters are applied to correct locations
    3. Backtest can run with generated config
    """
    logger.info("=" * 80)
    logger.info("SINGLE TRIAL TEST")
    logger.info("=" * 80)

    # Test with trap_within_trend group
    group_name = 'trap_within_trend'
    group = ARCHETYPE_GROUPS[group_name]

    logger.info(f"Testing group: {group_name}")
    logger.info(f"Archetypes: {', '.join(group['archetypes'])}")

    # Create config generator
    config_gen = MVPConfigGenerator(base_config_path)

    # Create dummy trial params
    global_params = {
        'w_wyckoff': 0.40,
        'w_liquidity': 0.30,
        'w_momentum': 0.30
    }

    archetype_params = {
        'trap_within_trend': {
            'fusion_threshold': 0.42,
            'archetype_weight': 1.15,
            'adx_threshold': 25.0,
            'cooldown_bars': 14
        },
        'liquidity_sweep': {
            'fusion_threshold': 0.38,
            'archetype_weight': 1.10,
            'boms_strength_min': 0.40
        },
        'spring': {
            'fusion_threshold': 0.35,
            'archetype_weight': 1.05,
            'pti_score_threshold': 0.40,
            'disp_atr_multiplier': 0.8
        }
    }

    # Generate config
    logger.info("\nGenerating trial config...")
    config_path = config_gen.generate(global_params, archetype_params, group_name)

    # Read and display config
    with open(config_path, 'r') as f:
        trial_config = json.load(f)

    logger.info("\nGenerated config validation:")
    logger.info(f"- Fusion weights: {trial_config['fusion']['weights']}")
    logger.info(f"- Crisis fuse enabled: {trial_config['context']['crisis_fuse']['enabled']}")

    # Check archetype params
    logger.info("\nArchetype parameters:")
    for pattern_name in ['trap_within_trend', 'liquidity_sweep', 'spring']:
        if pattern_name in trial_config['archetypes']:
            params = trial_config['archetypes'][pattern_name]
            logger.info(f"- {pattern_name}:")
            logger.info(f"    archetype_weight: {params.get('archetype_weight', 'MISSING')}")
            logger.info(f"    final_fusion_gate: {params.get('final_fusion_gate', 'MISSING')}")

        if pattern_name in trial_config['archetypes'].get('thresholds', {}):
            thresholds = trial_config['archetypes']['thresholds'][pattern_name]
            logger.info(f"    fusion_threshold: {thresholds.get('fusion_threshold', 'MISSING')}")

    # Run backtest
    logger.info("\nRunning test backtest (1 month)...")
    metrics = run_backtest(
        config_path,
        start_date="2024-01-01",
        end_date="2024-01-31",
        asset="BTC",
        timeout=120
    )

    if metrics:
        logger.info("\nBacktest Results:")
        logger.info(f"- Trades: {metrics['trades']}")
        logger.info(f"- Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"- Win Rate: {metrics['win_rate']:.1f}%")
        logger.info(f"- Max Drawdown: {metrics['drawdown']:.1f}%")
        logger.info("\nTEST PASSED: Config generation and backtest execution successful")
    else:
        logger.error("\nTEST FAILED: Backtest did not return metrics")

    # Cleanup
    try:
        os.unlink(config_path)
    except:
        pass

    logger.info("=" * 80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Parallel Optuna Archetype Optimization with Two-Layer Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full optimization with MVP bull config
  python bin/optuna_parallel_archetypes_v2.py --trials 100 --base-config configs/mvp/mvp_bull_market_v1.json

  # Test single trial to verify config application
  python bin/optuna_parallel_archetypes_v2.py --test-trial --base-config configs/mvp/mvp_bull_market_v1.json

  # Resume from checkpoint
  python bin/optuna_parallel_archetypes_v2.py --resume

  # Optimize specific groups only
  python bin/optuna_parallel_archetypes_v2.py --groups trap_within_trend order_block_retest
        """
    )

    parser.add_argument('--trials', type=int, default=50,
                        help='Trials per archetype group (default: 50)')
    parser.add_argument('--base-config', type=str, default='configs/mvp/mvp_bull_market_v1.json',
                        help='Base MVP config path')
    parser.add_argument('--storage', type=str, default='optuna_archetypes_v2.db',
                        help='SQLite storage path')
    parser.add_argument('--output', type=str, default='configs/optimized_archetypes_v2.json',
                        help='Output config path')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Specific groups to optimize (default: all)')
    parser.add_argument('--test-trial', action='store_true',
                        help='Run single trial test to verify config application')

    args = parser.parse_args()

    # Validate base config exists
    if not Path(args.base_config).exists():
        logger.error(f"Base config not found: {args.base_config}")
        return 1

    # Test mode
    if args.test_trial:
        test_single_trial(args.base_config)
        return 0

    # Select groups to optimize
    groups_to_run = args.groups or list(ARCHETYPE_GROUPS.keys())

    logger.info("=" * 80)
    logger.info("OPTUNA PARALLEL ARCHETYPE OPTIMIZATION V2")
    logger.info("Two-Layer Threshold Architecture")
    logger.info("=" * 80)
    logger.info(f"Base config: {args.base_config}")
    logger.info(f"Groups to optimize: {len(groups_to_run)}")
    logger.info(f"Trials per group: {args.trials}")
    logger.info(f"Total expected trials: {args.trials * len(groups_to_run)}")
    logger.info(f"Estimated runtime: 6-8 hours")
    logger.info("")
    logger.info("GLOBAL SAFETY RAILS (FIXED - Not Optimized):")
    for key, value in GLOBAL_SAFETY_RAILS.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    # Create progress queue using Manager for macOS compatibility
    manager = mp.Manager()
    progress_queue = manager.Queue()

    # Start progress monitor in separate process
    monitor_proc = mp.Process(
        target=monitor_progress,
        args=(progress_queue, args.trials, len(groups_to_run))
    )
    monitor_proc.start()

    # Create worker pool
    with mp.Pool(processes=len(groups_to_run)) as pool:
        # Start optimization for each group
        async_results = []
        for group_name in groups_to_run:
            result = pool.apply_async(
                optimize_archetype_group,
                args=(group_name, args.base_config, args.trials, args.storage, progress_queue)
            )
            async_results.append(result)

        # Wait for all to complete
        results = []
        for async_result in async_results:
            try:
                result = async_result.get(timeout=28800)  # 8 hour timeout
                results.append(result)
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    # Signal monitor to stop
    progress_queue.put('DONE')
    monitor_proc.join(timeout=5)

    # Aggregate results
    if results:
        aggregate_results(results, args.base_config, args.output)
        logger.info("Optimization complete!")
        logger.info(f"Optimized config: {args.output}")
    else:
        logger.error("No results to aggregate")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
