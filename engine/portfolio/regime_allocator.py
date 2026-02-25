"""
RegimeWeightAllocator - Regime-conditioned portfolio allocation (mixture of experts)

Implements soft gating with empirical Bayes shrinkage and guardrails to prevent:
1. Hard zeros (always maintain min 1% floor for exploration)
2. Concentration risk (cap negative edge at 20%)
3. Overfitting (sample size shrinkage)

Author: Production Quant Implementation
Date: 2026-01-10
Spec: SOFT_GATING_PHASE1_SPEC.md
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger(__name__)


class RegimeWeightAllocator:
    """
    Compute regime-conditioned portfolio weights using empirical edge data.

    Core Formula (4-step process):
    1. Shrink edge by sample size: edge_shrunk = edge * (N / (N + k))
    2. Map to positive strength: strength = sigmoid(alpha * edge_shrunk)
    3. Apply guardrails: cap negative edge at 20%, floor at 1%
    4. Enforce directional caps: scale down if sum(long) > long_cap or sum(short) > short_cap

    Budget Semantics:
    - Budgets represent SPOT EXPOSURE (% of equity in positions)
    - NOT margin (separate constraint handled by broker)
    - NOT notional risk (that's stop_distance based)

    Two-Level Budget Enforcement:
    1. Total regime budget (overall exposure per regime, e.g., 80% in risk_on)
    2. Directional splits within budget (e.g., 60% long / 20% short)

    Example (risk_on regime):
    - Total budget: 80% of equity
    - Long cap: 60% of equity (HARD CAP - enforced by _apply_directional_caps)
    - Short cap: 20% of equity (HARD CAP - enforced by _apply_directional_caps)
    - Enforcement: sum(long_weights) <= 60%, sum(short_weights) <= 20%

    If archetypes stack beyond caps, proportional scaling reduces all weights
    on that side to fit within the cap while preserving relative signal strength.

    Ensures smooth, continuous allocation with no hard cliffs and strict cap enforcement.
    """

    # Regime-specific risk budgets (max total exposure per regime)
    REGIME_RISK_BUDGETS = {
        'crisis': 0.30,    # Max 30% total exposure in CRISIS
        'risk_off': 0.50,  # Max 50% in RISK_OFF
        'neutral': 0.70,   # Max 70% in NEUTRAL
        'risk_on': 0.80    # Max 80% in RISK_ON
    }

    # Regime-directional budgets (aligns with trader psychology)
    # Crisis: aggressive shorts, cautious longs
    # Risk_on: aggressive longs, minimal shorts
    REGIME_DIRECTIONAL_BUDGETS = {
        'crisis': {'long': 0.15, 'short': 0.60},
        'risk_off': {'long': 0.25, 'short': 0.50},
        'neutral': {'long': 0.40, 'short': 0.40},
        'risk_on': {'long': 0.60, 'short': 0.20}
    }

    def __init__(
        self,
        edge_table_path: str,
        config_path: Optional[str] = None,
        config_override: Optional[Dict] = None,
        registry_path: Optional[str] = None
    ):
        """
        Initialize the allocator with edge data and configuration.

        Args:
            edge_table_path: Path to archetype_regime_edge_table.csv
            config_path: Path to regime_allocator_config.json (optional)
            config_override: Direct config dict (overrides config_path)
            registry_path: Path to archetype_registry.yaml (optional, defaults to root)
        """
        self.edge_table_path = edge_table_path
        self.config_path = config_path

        # Load configuration
        self.config = self._load_config(config_path, config_override)

        # Extract parameters
        self.k_shrinkage = self.config.get('k_shrinkage', 30)
        self.min_weight = self.config.get('min_weight', 0.01)
        self.neg_edge_cap = self.config.get('neg_edge_cap', 0.20)
        self.min_trades = self.config.get('min_trades', 5)
        self.alpha = self.config.get('alpha', 4.0)

        # Load directional budgets from config if present
        config_directional_budgets = self.config.get('regime_directional_budgets')
        if config_directional_budgets:
            # Update class constant with config values (removing 'note' fields)
            for regime, budgets in config_directional_budgets.items():
                if isinstance(budgets, dict):
                    self.REGIME_DIRECTIONAL_BUDGETS[regime] = {
                        'long': budgets.get('long', 0.5),
                        'short': budgets.get('short', 0.5)
                    }

        # Load archetype-specific directional budgets from config (NEW: support archetype-level scaling)
        # Format: {"archetype_name": {"crisis": 0.0, "risk_off": 0.3, "neutral": 0.8, "risk_on": 1.0}}
        self.archetype_budgets = self.config.get('directional_budgets', {})
        if self.archetype_budgets:
            logger.info(f"Loaded archetype-specific budgets for {len(self.archetype_budgets)} archetypes")

        # Load archetype registry to get direction metadata
        self.archetype_directions = self._load_archetype_directions(registry_path)

        # Load edge data
        self.edge_data = self._load_edge_table()

        # Cache for computed weights
        self._weight_cache: Dict[Tuple[str, str], float] = {}

        # Track current regime exposures (updated externally during backtesting)
        self.regime_exposures: Dict[str, float] = {
            'crisis': 0.0,
            'risk_off': 0.0,
            'neutral': 0.0,
            'risk_on': 0.0
        }

        logger.info(
            f"RegimeWeightAllocator initialized: "
            f"k_shrinkage={self.k_shrinkage}, alpha={self.alpha}, "
            f"min_weight={self.min_weight}, neg_edge_cap={self.neg_edge_cap}, "
            f"directional_budgets_enabled=True"
        )

    def _load_config(
        self,
        config_path: Optional[str],
        config_override: Optional[Dict]
    ) -> Dict:
        """Load configuration from file or use override."""
        if config_override is not None:
            logger.info("Using config override")
            return config_override

        if config_path is not None and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
            return config

        # Use defaults
        default_config = {
            'k_shrinkage': 30,
            'min_weight': 0.01,
            'neg_edge_cap': 0.20,
            'min_trades': 5,
            'alpha': 4.0
        }
        logger.info("Using default configuration")
        return default_config

    def _load_archetype_directions(self, registry_path: Optional[str] = None) -> Dict[str, str]:
        """
        Load archetype direction metadata from registry.

        Args:
            registry_path: Path to archetype_registry.yaml (defaults to root)

        Returns:
            Dictionary mapping archetype_id -> direction ('long' or 'short')
        """
        if registry_path is None:
            # Try to find registry in common locations
            possible_paths = [
                Path.cwd() / 'archetype_registry.yaml',
                Path(__file__).parent.parent.parent / 'archetype_registry.yaml',
            ]
            registry_path = None
            for path in possible_paths:
                if path.exists():
                    registry_path = str(path)
                    break

        if registry_path is None or not Path(registry_path).exists():
            logger.warning(
                "Archetype registry not found. Using default direction=long for all archetypes. "
                "This will disable regime-directional bias."
            )
            return {}

        try:
            with open(registry_path, 'r') as f:
                registry = yaml.safe_load(f)

            directions = {}
            for archetype in registry.get('archetypes', []):
                archetype_id = archetype.get('id')
                direction = archetype.get('direction', 'long')
                if archetype_id:
                    directions[archetype_id] = direction

            logger.info(
                f"Loaded archetype directions from {registry_path}: "
                f"{len(directions)} archetypes"
            )
            return directions

        except Exception as e:
            logger.error(f"Failed to load archetype registry: {e}")
            return {}

    def _load_edge_table(self) -> pd.DataFrame:
        """Load the edge table from CSV with sanity checks for dummy data."""
        if not Path(self.edge_table_path).exists():
            raise FileNotFoundError(
                f"Edge table not found at {self.edge_table_path}. "
                f"Run edge computation script first:\n"
                f"  python bin/compute_regime_stratified_performance.py \\\n"
                f"    --trades results/v11_standalone/trade_log.csv \\\n"
                f"    --feature-store data/features_mtf/"
                f"BTC_1H_CANONICAL_FUSION_FIXED_20260202.parquet \\\n"
                f"    --force-feature-store-regime"
            )

        df = pd.read_csv(self.edge_table_path)

        # Validate required columns
        required_cols = ['archetype', 'regime', 'n_trades', 'sharpe_like']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Edge table missing required columns: {missing_cols}"
            )

        logger.info(
            f"Loaded edge table: {len(df)} archetype-regime pairs from "
            f"{self.edge_table_path}"
        )

        # Sanity check: warn if edge data looks like dummy/placeholder data
        self._validate_edge_table(df)

        return df

    def _validate_edge_table(self, df: pd.DataFrame) -> None:
        """
        Validate edge table for signs of dummy or degenerate data.

        Checks for:
        1. Low variance in sharpe_like (all values same -> dummy data)
        2. Trades concentrated in a single regime (regime inference failed)
        3. No regime has any trades (completely empty edge table)
        """
        has_trades = df[df['n_trades'] > 0]

        if len(has_trades) == 0:
            logger.warning(
                "EDGE TABLE WARNING: No archetype-regime combos have trades. "
                "This is likely a placeholder edge table. "
                "Run bin/compute_regime_stratified_performance.py with real trade data."
            )
            return

        # Check Sharpe variance
        sharpe_values = has_trades['sharpe_like'].values
        sharpe_variance = float(np.var(sharpe_values))
        if sharpe_variance < 1e-6 and len(has_trades) > 4:
            logger.warning(
                f"EDGE TABLE WARNING: Sharpe-like variance is very low "
                f"({sharpe_variance:.8f}). This suggests dummy or uniform data. "
                f"All combos have sharpe_like ~ {sharpe_values[0]:.4f}."
            )

        # Check regime concentration
        regimes_with_trades = has_trades['regime'].unique()
        if len(regimes_with_trades) == 1:
            sole_regime = regimes_with_trades[0]
            logger.warning(
                f"EDGE TABLE WARNING: All trades are in a single regime "
                f"('{sole_regime}'). This suggests regime inference did not "
                f"distribute trades across regimes. Consider re-running with "
                f"--force-feature-store-regime."
            )
        else:
            # Log healthy distribution
            regime_trade_counts = (
                has_trades.groupby('regime')['n_trades'].sum().to_dict()
            )
            logger.info(
                f"Edge table regime distribution: {regime_trade_counts}"
            )

    def get_directional_budget(self, archetype: str, regime: str) -> float:
        """
        Get the regime-directional budget for an archetype.

        This implements trader psychology: be MORE aggressive on regime-aligned directions.
        - Crisis: boost shorts (0.60), shrink longs (0.15)
        - Risk_on: boost longs (0.60), shrink shorts (0.20)

        Args:
            archetype: Archetype ID (e.g., 'S1', 'S5', 'H')
            regime: Regime label (crisis, risk_off, neutral, risk_on)

        Returns:
            Directional budget ∈ [0.0, 1.0]
        """
        # Get archetype direction from registry
        direction = self.archetype_directions.get(archetype, 'long')

        # Get regime-directional budget
        regime_budgets = self.REGIME_DIRECTIONAL_BUDGETS.get(regime, {'long': 0.5, 'short': 0.5})
        budget = regime_budgets.get(direction, 0.5)

        logger.debug(
            f"Directional budget: archetype={archetype}, regime={regime}, "
            f"direction={direction}, budget={budget:.2f}"
        )

        return budget

    def get_directional_budget_probabilistic(
        self,
        archetype: str,
        regime_probs: Dict[str, float]
    ) -> float:
        """
        Get directional budget using probabilistic regime blend (SOFT CONTROLS).

        Instead of binary gate (pick ONE regime), this blends directional budgets
        weighted by regime probabilities. Enables smooth transitions.

        Example:
            regime_probs = {'crisis': 0.10, 'neutral': 0.30, 'risk_on': 0.60}
            direction = 'long'

            blended_budget =
                0.10 * 0.15 (crisis long)  +
                0.30 * 0.40 (neutral long) +
                0.60 * 0.60 (risk_on long)
              = 0.015 + 0.12 + 0.36
              = 0.495 (49.5% long cap)

        This is TRULY probabilistic - no hard gates, smooth regime transitions.

        Args:
            archetype: Archetype ID (e.g., 'S1', 'H', 'A')
            regime_probs: Probability distribution over regimes
                e.g., {'crisis': 0.1, 'risk_off': 0.0, 'neutral': 0.3, 'risk_on': 0.6}

        Returns:
            Blended directional budget ∈ [0.0, 1.0]
        """
        # Get archetype direction from registry
        direction = self.archetype_directions.get(archetype, 'long')

        # Weighted average of directional budgets
        blended_budget = 0.0
        for regime, prob in regime_probs.items():
            if prob > 0:  # Skip zero probabilities
                regime_budgets = self.REGIME_DIRECTIONAL_BUDGETS.get(
                    regime,
                    {'long': 0.5, 'short': 0.5}
                )
                budget = regime_budgets.get(direction, 0.5)
                blended_budget += prob * budget

        logger.debug(
            f"Probabilistic directional budget: archetype={archetype}, "
            f"direction={direction}, regime_probs={regime_probs}, "
            f"blended_budget={blended_budget:.3f}"
        )

        return blended_budget

    def get_archetype_budget_probabilistic(
        self,
        archetype: str,
        regime_probs: Dict[str, float]
    ) -> float:
        """
        Get archetype-specific budget using probabilistic regime blend.

        This allows per-archetype scaling in each regime, supporting configs like:
            {"spring": {"crisis": 0.0, "risk_off": 0.3, "neutral": 0.8, "risk_on": 1.0}}

        Example:
            archetype = 'spring'
            regime_probs = {'neutral': 0.70, 'risk_on': 0.30}
            archetype_budgets = {'spring': {'neutral': 0.8, 'risk_on': 1.0}}

            blended_budget = 0.70 * 0.8 + 0.30 * 1.0 = 0.56 + 0.30 = 0.86

        Args:
            archetype: Archetype name (e.g., 'spring', 'wick_trap')
            regime_probs: Probability distribution over regimes

        Returns:
            Blended archetype budget ∈ [0.0, 1.0] (defaults to 1.0 if not configured)
        """
        # If no archetype-specific budgets configured, return 1.0 (no scaling)
        if not self.archetype_budgets or archetype not in self.archetype_budgets:
            return 1.0

        # Weighted average of archetype budgets across regime probabilities
        archetype_regime_budgets = self.archetype_budgets[archetype]
        blended_budget = 0.0

        for regime, prob in regime_probs.items():
            if prob > 0:  # Skip zero probabilities
                # Get archetype budget for this regime (default to 1.0 if missing)
                budget = archetype_regime_budgets.get(regime, 1.0)
                blended_budget += prob * budget

        logger.debug(
            f"Archetype-specific budget: archetype={archetype}, "
            f"regime_probs={regime_probs}, blended_budget={blended_budget:.3f}"
        )

        return blended_budget

    def _apply_directional_caps(
        self,
        weights: Dict[str, float],
        regime: str
    ) -> Dict[str, float]:
        """
        Apply directional budget caps to archetype weights.

        CRITICAL FIX: Enforces directional budgets as HARD CAPS, not per-archetype multipliers.

        If total long allocation exceeds long_budget, scale down all long weights
        proportionally. Same for shorts.

        This ensures:
        - sum(long_weights) <= REGIME_DIRECTIONAL_BUDGETS[regime]['long']
        - sum(short_weights) <= REGIME_DIRECTIONAL_BUDGETS[regime]['short']

        Args:
            weights: Dictionary of archetype -> weight
            regime: Regime label

        Returns:
            Capped weights dictionary
        """
        # Get directional budgets
        budgets = self.REGIME_DIRECTIONAL_BUDGETS.get(regime, {'long': 0.5, 'short': 0.5})
        long_budget = budgets['long']
        short_budget = budgets['short']

        # Split weights by direction
        long_weights = {}
        short_weights = {}

        for archetype, weight in weights.items():
            direction = self.archetype_directions.get(archetype, 'long')
            if direction == 'short':
                short_weights[archetype] = weight
            else:
                long_weights[archetype] = weight

        # Calculate totals
        long_total = sum(long_weights.values())
        short_total = sum(short_weights.values())

        # Apply caps (scale down if exceeded)
        capped_weights = {}

        if long_total > long_budget:
            # Scale down all long weights proportionally
            scale_factor = long_budget / long_total
            logger.info(
                f"Directional cap applied for {regime}: "
                f"Long total {long_total:.1%} exceeds budget {long_budget:.1%}, "
                f"scaling by {scale_factor:.3f}"
            )
            for archetype, weight in long_weights.items():
                capped_weights[archetype] = weight * scale_factor
        else:
            capped_weights.update(long_weights)

        if short_total > short_budget:
            # Scale down all short weights proportionally
            scale_factor = short_budget / short_total
            logger.info(
                f"Directional cap applied for {regime}: "
                f"Short total {short_total:.1%} exceeds budget {short_budget:.1%}, "
                f"scaling by {scale_factor:.3f}"
            )
            for archetype, weight in short_weights.items():
                capped_weights[archetype] = weight * scale_factor
        else:
            capped_weights.update(short_weights)

        # Log final totals
        final_long_total = sum(w for a, w in capped_weights.items()
                               if self.archetype_directions.get(a, 'long') == 'long')
        final_short_total = sum(w for a, w in capped_weights.items()
                                if self.archetype_directions.get(a, 'long') == 'short')

        logger.debug(
            f"Directional caps for {regime}: "
            f"long={final_long_total:.3f}/{long_budget:.3f}, "
            f"short={final_short_total:.3f}/{short_budget:.3f}"
        )

        return capped_weights

    def compute_weight(
        self,
        edge: float,
        N: int,
        archetype: str,
        regime: str,
        is_entry: bool = True
    ) -> float:
        """
        Compute regime-conditioned weight with guardrails AND directional bias.

        UPDATED IMPLEMENTATION: Applies regime-directional budget scaling.

        The weight is computed in 4 steps:
        1. Shrink edge by sample size (empirical Bayes)
        2. Map to positive strength using sigmoid (smooth, no cliffs)
        3. Apply guardrails (neg edge cap, exploration floor)
        4. Scale by regime-directional budget (NEW: aligns with trader psychology)

        Args:
            edge: Sharpe-like metric (risk-adjusted return)
            N: Number of trades (sample size)
            archetype: Archetype name (for logging and direction lookup)
            regime: Regime label (for directional budget)
            is_entry: True for entry decisions, False for exits
                When True and bypass_entry_filtering is enabled, returns 1.0
                When False, always applies regime filtering

        Returns:
            weight ∈ [0.0, 1.0] for allocation (may be very small if counter-regime)
        """
        # Check bypass flag for entries
        if is_entry and self.config.get('bypass_entry_filtering', False):
            logger.debug(
                f"Bypass entry filtering enabled for {archetype} - returning weight=1.0"
            )
            return 1.0

        # Step 1: Shrink edge by sample size (empirical Bayes)
        edge_shrunk = edge * (N / (N + self.k_shrinkage))

        # Step 2: Map to positive strength using sigmoid (smooth, no cliffs)
        strength = 1.0 / (1.0 + np.exp(-self.alpha * edge_shrunk))

        # Step 3: Apply guardrails
        weight = strength

        # Guardrail A: Cap negative edge
        if edge_shrunk < 0:
            weight = min(weight, self.neg_edge_cap)

        # Guardrail B: Floor for exploration
        if edge_shrunk < -0.10 and N >= self.min_trades:
            # Strongly negative with enough data → minimal allocation
            weight = max(weight, self.min_weight)
        elif N >= self.min_trades:
            # Normal case → standard floor
            weight = max(weight, self.min_weight)
        else:
            # Small sample → softer floor based on sample size
            sample_floor = self.min_weight * (N / self.min_trades)
            weight = max(weight, sample_floor)

        # Step 4: Scale by regime-directional budget (NEW)
        # This implements trader psychology: be MORE aggressive on regime-aligned direction
        directional_budget = self.get_directional_budget(archetype, regime)
        weight_before_scaling = weight
        weight = weight * directional_budget

        # Log weight computation for debugging
        logger.debug(
            f"Weight computation: archetype={archetype}, regime={regime}, is_entry={is_entry}, "
            f"edge={edge:.4f}, N={N}, edge_shrunk={edge_shrunk:.4f}, "
            f"strength={strength:.4f}, weight_base={weight_before_scaling:.4f}, "
            f"directional_budget={directional_budget:.2f}, weight_final={weight:.4f}"
        )

        return weight

    def compute_weight_probabilistic(
        self,
        edge: float,
        N: int,
        archetype: str,
        regime_probs: Dict[str, float],
        is_entry: bool = True
    ) -> float:
        """
        Compute weight with probabilistic regime blending.

        Same as compute_weight() but uses probabilistic directional budget.

        Args:
            edge: Sharpe-like metric (risk-adjusted return)
            N: Number of trades (sample size)
            archetype: Archetype name (for logging and direction lookup)
            regime_probs: Probability distribution over regimes
                e.g., {'crisis': 0.1, 'risk_off': 0.0, 'neutral': 0.3, 'risk_on': 0.6}
            is_entry: True for entry decisions, False for exits
                When True and bypass_entry_filtering is enabled, returns 1.0
                When False, always applies regime filtering

        Returns:
            weight ∈ [0.0, 1.0] for allocation (probabilistically blended)
        """
        # Check bypass flag for entries (CRITICAL DIAGNOSTIC LOGGING)
        bypass_enabled = self.config.get('bypass_entry_filtering', False)
        logger.info(
            f"[REGIME ALLOCATOR] compute_weight_probabilistic: "
            f"archetype={archetype}, is_entry={is_entry}, "
            f"bypass_enabled={bypass_enabled}, regime_probs={regime_probs}"
        )

        if is_entry and bypass_enabled:
            logger.info(
                f"[BYPASS ACTIVE] Returning weight=1.0 for {archetype} (entry bypass)"
            )
            return 1.0

        # Steps 1-3: Same as discrete version
        edge_shrunk = edge * (N / (N + self.k_shrinkage))
        strength = 1.0 / (1.0 + np.exp(-self.alpha * edge_shrunk))
        weight = strength

        # Guardrails
        if edge_shrunk < 0:
            weight = min(weight, self.neg_edge_cap)

        if edge_shrunk < -0.10 and N >= self.min_trades:
            weight = max(weight, self.min_weight)
        elif N >= self.min_trades:
            weight = max(weight, self.min_weight)
        else:
            sample_floor = self.min_weight * (N / self.min_trades)
            weight = max(weight, sample_floor)

        # Step 4: PROBABILISTIC directional budget scaling
        directional_budget = self.get_directional_budget_probabilistic(
            archetype, regime_probs
        )
        weight_before_scaling = weight
        weight = weight * directional_budget

        # Step 5: ARCHETYPE-SPECIFIC budget scaling (NEW!)
        # Apply archetype-level regime scaling if configured
        archetype_budget = self.get_archetype_budget_probabilistic(
            archetype, regime_probs
        )
        # TEMPORARY FIX: Don't double-scale with both directional AND archetype budgets
        # weight = weight * archetype_budget  # DISABLED - too aggressive

        logger.debug(
            f"Probabilistic weight: archetype={archetype}, is_entry={is_entry}, "
            f"edge={edge:.4f}, N={N}, weight_base={weight_before_scaling:.4f}, "
            f"directional_budget={directional_budget:.3f}, "
            f"archetype_budget={archetype_budget:.3f} (not applied), weight_final={weight:.4f}"
        )

        return weight

    def get_weight(self, archetype: str, regime: str) -> float:
        """
        Get the allocation weight for a given archetype-regime pair.

        Args:
            archetype: Archetype name
            regime: Regime label (crisis, risk_off, neutral, risk_on)

        Returns:
            Weight ∈ [0.01, 1.0] for allocation
        """
        # Check cache first
        cache_key = (archetype, regime)
        if cache_key in self._weight_cache:
            return self._weight_cache[cache_key]

        # Look up edge data
        mask = (
            (self.edge_data['archetype'] == archetype) &
            (self.edge_data['regime'] == regime)
        )

        if not mask.any():
            # No data for this pair → use minimal weight
            logger.warning(
                f"No edge data for {archetype} in {regime}. "
                f"Using minimal weight {self.min_weight}"
            )
            weight = self.min_weight
        else:
            row = self.edge_data[mask].iloc[0]
            edge = row['sharpe_like']
            N = int(row['n_trades'])

            # Compute weight using exact formula
            weight = self.compute_weight(edge, N, archetype, regime)

        # Cache result
        self._weight_cache[cache_key] = weight

        return weight

    def get_sqrt_weight(self, archetype: str, regime: str) -> float:
        """
        Get the square-root of allocation weight for a given archetype-regime pair.

        This is used for the SQUARE-ROOT SPLIT approach to prevent double-weight bug.
        Both score layer and sizing layer apply sqrt(weight), giving combined impact
        of sqrt(w) * sqrt(w) = w (correct!).

        Args:
            archetype: Archetype name
            regime: Regime label (crisis, risk_off, neutral, risk_on)

        Returns:
            sqrt(weight) ∈ [0.1, 1.0] for allocation
        """
        weight = self.get_weight(archetype, regime)
        sqrt_weight = np.sqrt(weight)

        logger.debug(
            f"Sqrt weight: {archetype} in {regime}: "
            f"weight={weight:.3f}, sqrt={sqrt_weight:.3f}"
        )

        return sqrt_weight

    def get_edge_metrics(self, archetype: str, regime: str) -> Dict:
        """
        Get detailed edge metrics for a given archetype-regime pair.

        Args:
            archetype: Archetype name
            regime: Regime label

        Returns:
            Dictionary with edge metrics and computed weight
        """
        mask = (
            (self.edge_data['archetype'] == archetype) &
            (self.edge_data['regime'] == regime)
        )

        if not mask.any():
            return {
                'archetype': archetype,
                'regime': regime,
                'has_data': False,
                'weight': self.min_weight,
                'reason': 'no_data'
            }

        row = self.edge_data[mask].iloc[0]
        edge = row['sharpe_like']
        N = int(row['n_trades'])

        # Compute derived metrics
        edge_shrunk = edge * (N / (N + self.k_shrinkage))
        strength = 1.0 / (1.0 + np.exp(-self.alpha * edge_shrunk))
        weight = self.compute_weight(edge, N, archetype, regime)

        return {
            'archetype': archetype,
            'regime': regime,
            'has_data': True,
            'n_trades': N,
            'edge_raw': float(edge),
            'edge_shrunk': float(edge_shrunk),
            'strength': float(strength),
            'weight': float(weight),
            'total_pnl': float(row['total_pnl']),
            'expectancy': float(row['expectancy']),
            'win_rate': float(row['win_rate']),
            'profit_factor': float(row['profit_factor'])
        }

    def get_regime_distribution(self, regime: str) -> Dict[str, float]:
        """
        Get the weight distribution for all archetypes in a given regime.

        UPDATED: Now enforces directional budget caps after computing weights.

        Args:
            regime: Regime label

        Returns:
            Dictionary mapping archetype -> weight (unnormalized, with directional caps enforced)
        """
        mask = self.edge_data['regime'] == regime
        regime_data = self.edge_data[mask]

        distribution = {}
        for _, row in regime_data.iterrows():
            archetype = row['archetype']
            weight = self.get_weight(archetype, regime)
            distribution[archetype] = weight

        # CRITICAL FIX: Enforce directional caps
        distribution = self._apply_directional_caps(distribution, regime)

        # Defensive check - invariants MUST hold
        long_total = sum(w for a, w in distribution.items()
                        if self.archetype_directions.get(a, 'long') == 'long')
        short_total = sum(w for a, w in distribution.items()
                         if self.archetype_directions.get(a, 'long') == 'short')

        budgets = self.REGIME_DIRECTIONAL_BUDGETS[regime]
        tolerance = 1e-6

        assert long_total <= budgets['long'] + tolerance, \
            f"INVARIANT VIOLATION: {regime} long {long_total:.1%} > {budgets['long']:.1%}"
        assert short_total <= budgets['short'] + tolerance, \
            f"INVARIANT VIOLATION: {regime} short {short_total:.1%} > {budgets['short']:.1%}"

        return distribution

    def normalize_weights_per_regime(
        self,
        regime: str,
        allow_cash_bucket: bool = True
    ) -> Dict[str, float]:
        """
        Normalize weights within a regime with optional cash bucket.

        CRITICAL: This method implements the cash bucket concept.

        When allow_cash_bucket=True (default):
        - If archetype weights sum > 1.0 → normalize to 1.0 (no cash)
        - If archetype weights sum < 1.0 → keep as-is (remainder becomes cash)
        - Cash bucket represents unused allocation when edge is weak

        When allow_cash_bucket=False (legacy behavior):
        - Always normalize to sum exactly 1.0 (forces full allocation)

        Args:
            regime: Regime label
            allow_cash_bucket: If True, allows weights to sum < 1.0 (default True)

        Returns:
            Dictionary mapping archetype -> weight (may sum < 1.0 if cash bucket exists)
        """
        weights = self.get_regime_distribution(regime)

        total = sum(weights.values())

        if total == 0:
            # No archetypes → empty dict (cash bucket gets 100%)
            logger.warning(f"No archetypes with weight in {regime} regime - 100% cash")
            return {}

        if allow_cash_bucket:
            # Cash bucket mode: only normalize if over-allocated
            if total > 1.0:
                # Over-allocated → normalize to 1.0
                normalized = {k: v / total for k, v in weights.items()}
                cash_pct = 0.0
                logger.info(
                    f"Regime {regime} over-allocated ({total:.2f} > 1.0), "
                    f"normalized to 1.0 - no cash bucket"
                )
            else:
                # Under-allocated → keep weights as-is, remainder is cash
                normalized = weights.copy()
                cash_pct = 1.0 - total
                logger.info(
                    f"Regime {regime} under-allocated (sum={total:.2f}), "
                    f"cash_bucket={cash_pct:.1%}"
                )
        else:
            # Legacy mode: force renormalize to 1.0
            normalized = {k: v / total for k, v in weights.items()}
            cash_pct = 0.0
            logger.info(
                f"Regime {regime} force-normalized to 1.0 (legacy mode)"
            )

        logger.info(
            f"Regime {regime} weights: "
            f"{', '.join(f'{k}={v:.2%}' for k, v in normalized.items())}"
            + (f", cash={cash_pct:.1%}" if cash_pct > 0 else "")
        )

        return normalized

    def get_cash_bucket_weight(self, regime: str) -> float:
        """
        Get the cash bucket weight for a regime.

        Cash bucket = 1.0 - sum(archetype_weights)

        This represents unused allocation when archetypes have weak edge.
        If archetypes only deserve 40% allocation → 60% stays in cash.

        Args:
            regime: Regime label

        Returns:
            Cash bucket weight ∈ [0.0, 1.0]
        """
        weights = self.get_regime_distribution(regime)
        total = sum(weights.values())

        # Cash bucket is the remainder after archetype allocation
        cash_bucket = max(0.0, 1.0 - total)

        return cash_bucket

    def get_effective_allocation(
        self,
        regime: str,
        include_cash: bool = True
    ) -> Dict[str, float]:
        """
        Get effective allocation including cash bucket.

        Returns dictionary with archetype weights + cash bucket.
        Guaranteed to sum to 1.0 when include_cash=True.

        Args:
            regime: Regime label
            include_cash: If True, include 'CASH' entry (default True)

        Returns:
            Dictionary mapping archetype/CASH -> weight (sums to 1.0)
        """
        # Get archetype weights (may sum < 1.0)
        weights = self.get_regime_distribution(regime)
        cash_bucket = self.get_cash_bucket_weight(regime)

        # Build effective allocation
        allocation = weights.copy()

        if include_cash and cash_bucket > 0:
            allocation['CASH'] = cash_bucket

        # Verify sum (should be 1.0 or very close due to floating point)
        total = sum(allocation.values())
        if include_cash and abs(total - 1.0) > 1e-6:
            logger.warning(
                f"Effective allocation does not sum to 1.0: "
                f"total={total:.6f} for regime={regime}"
            )

        return allocation

    def clear_cache(self):
        """Clear the weight cache (call if edge table is updated)."""
        self._weight_cache.clear()
        logger.info("Weight cache cleared")

    def get_all_weights(self) -> pd.DataFrame:
        """
        Get all computed weights as a DataFrame for analysis.

        Returns:
            DataFrame with columns: archetype, regime, weight, edge_metrics
        """
        weights = []

        for _, row in self.edge_data.iterrows():
            archetype = row['archetype']
            regime = row['regime']
            weight = self.get_weight(archetype, regime)
            metrics = self.get_edge_metrics(archetype, regime)

            weights.append({
                'archetype': archetype,
                'regime': regime,
                'weight': weight,
                'n_trades': metrics['n_trades'],
                'edge_raw': metrics['edge_raw'],
                'edge_shrunk': metrics['edge_shrunk'],
                'strength': metrics['strength']
            })

        return pd.DataFrame(weights)

    # ========================================================================
    # Regime Risk Budget Methods (NEW - for position sizing integration)
    # ========================================================================

    def get_regime_budget(self, regime: str) -> float:
        """
        Get risk budget for regime.

        Args:
            regime: Regime label

        Returns:
            Maximum total exposure allowed (as fraction of portfolio)
        """
        return self.REGIME_RISK_BUDGETS.get(regime, 0.80)

    def get_dynamic_regime_budget(
        self,
        regime: str,
        current_vol: float,
        baseline_vol: float
    ) -> float:
        """
        Scale regime budget by current volatility relative to long-term baseline.

        When volatility spikes above the baseline, exposure is reduced to protect
        capital. When volatility is at or below baseline, the full static budget
        is used (no increase beyond the base budget).

        Formula:
            vol_ratio = current_vol / baseline_vol
            vol_discount = max(0.7, 1.0 - 0.3 * (vol_ratio - 1.0))
            adjusted = base_budget * min(1.0, vol_discount)

        The discount is clamped:
            - Floor at 0.7 (max 30% reduction even in extreme vol)
            - Ceiling at 1.0 (never increase beyond static budget)

        Args:
            regime: Regime label (crisis, risk_off, neutral, risk_on)
            current_vol: Current 7-day realized volatility (annualized, e.g. 0.85)
            baseline_vol: Long-term average volatility (annualized, e.g. 0.60)

        Returns:
            Adjusted budget as fraction of portfolio, e.g.:
                - Normal vol (ratio=1.0): budget unchanged
                - 1.5x vol: budget * 0.85 (15% reduction)
                - 2.0x vol: budget * 0.70 (30% reduction, the maximum)
                - 0.5x vol: budget * 1.0 (capped, no increase beyond base)

        Examples:
            >>> allocator.get_dynamic_regime_budget('risk_on', 0.60, 0.60)
            0.80  # ratio=1.0, no change
            >>> allocator.get_dynamic_regime_budget('risk_on', 0.90, 0.60)
            0.68  # ratio=1.5, 0.80 * 0.85
            >>> allocator.get_dynamic_regime_budget('risk_on', 1.20, 0.60)
            0.56  # ratio=2.0, 0.80 * 0.70 (max reduction)
            >>> allocator.get_dynamic_regime_budget('crisis', 0.90, 0.60)
            0.255  # ratio=1.5, 0.30 * 0.85
        """
        base_budget = self.get_regime_budget(regime)

        # Guard against zero or negative baseline
        if baseline_vol <= 0:
            logger.warning(
                f"Invalid baseline_vol={baseline_vol:.4f}, using base budget "
                f"for {regime}: {base_budget:.2f}"
            )
            return base_budget

        vol_ratio = current_vol / baseline_vol

        # Compute discount: linear scaling with floor at 0.7, ceiling at 1.0
        vol_discount = max(0.7, 1.0 - 0.3 * (vol_ratio - 1.0))
        vol_discount = min(1.0, vol_discount)

        adjusted_budget = base_budget * vol_discount

        logger.info(
            f"Dynamic regime budget: regime={regime}, "
            f"current_vol={current_vol:.4f}, baseline_vol={baseline_vol:.4f}, "
            f"vol_ratio={vol_ratio:.2f}, vol_discount={vol_discount:.3f}, "
            f"base_budget={base_budget:.2f}, adjusted_budget={adjusted_budget:.3f}"
        )

        return adjusted_budget

    def get_regime_budget_with_vol(
        self,
        regime: str,
        current_vol: float,
        baseline_vol: float
    ) -> float:
        """
        Convenience wrapper for get_dynamic_regime_budget.

        Identical to get_dynamic_regime_budget() but provides a shorter name
        for use in hot-path position sizing code where brevity is preferred.

        Args:
            regime: Regime label (crisis, risk_off, neutral, risk_on)
            current_vol: Current 7-day realized volatility (annualized)
            baseline_vol: Long-term average volatility (annualized)

        Returns:
            Volatility-adjusted regime budget (same as get_dynamic_regime_budget)
        """
        return self.get_dynamic_regime_budget(regime, current_vol, baseline_vol)

    def get_dynamic_directional_budgets(
        self,
        regime: str,
        current_vol: float,
        baseline_vol: float
    ) -> Dict[str, float]:
        """
        Get volatility-adjusted directional budgets for a regime.

        Applies the same volatility discount used by get_dynamic_regime_budget()
        to BOTH the long and short directional budgets. This ensures that when
        volatility spikes, both sides of the book are reduced proportionally.

        Formula (per direction):
            vol_ratio = current_vol / baseline_vol
            vol_discount = max(0.7, min(1.0, 1.0 - 0.3 * (vol_ratio - 1.0)))
            adjusted_long  = base_long  * vol_discount
            adjusted_short = base_short * vol_discount

        Args:
            regime: Regime label (crisis, risk_off, neutral, risk_on)
            current_vol: Current 7-day realized volatility (annualized)
            baseline_vol: Long-term average volatility (annualized)

        Returns:
            Dict with 'long' and 'short' budgets, scaled by volatility.

        Examples:
            >>> allocator.get_dynamic_directional_budgets('risk_on', 0.60, 0.60)
            {'long': 0.60, 'short': 0.20}  # ratio=1.0, unchanged
            >>> allocator.get_dynamic_directional_budgets('risk_on', 0.90, 0.60)
            {'long': 0.51, 'short': 0.17}  # ratio=1.5, 15% reduction
            >>> allocator.get_dynamic_directional_budgets('crisis', 1.20, 0.60)
            {'long': 0.105, 'short': 0.42}  # ratio=2.0, 30% reduction (max)
        """
        # Get base directional budgets
        base_budgets = self.REGIME_DIRECTIONAL_BUDGETS.get(
            regime, {'long': 0.5, 'short': 0.5}
        )

        # Guard against zero or negative baseline
        if baseline_vol <= 0:
            logger.warning(
                f"Invalid baseline_vol={baseline_vol:.4f}, using base directional "
                f"budgets for {regime}: {base_budgets}"
            )
            return {'long': base_budgets['long'], 'short': base_budgets['short']}

        vol_ratio = current_vol / baseline_vol

        # Same discount formula as get_dynamic_regime_budget
        vol_discount = max(0.7, 1.0 - 0.3 * (vol_ratio - 1.0))
        vol_discount = min(1.0, vol_discount)

        adjusted_long = base_budgets['long'] * vol_discount
        adjusted_short = base_budgets['short'] * vol_discount

        logger.info(
            f"Dynamic directional budgets: regime={regime}, "
            f"vol_ratio={vol_ratio:.2f}, vol_discount={vol_discount:.3f}, "
            f"long={base_budgets['long']:.2f}->{adjusted_long:.3f}, "
            f"short={base_budgets['short']:.2f}->{adjusted_short:.3f}"
        )

        return {'long': adjusted_long, 'short': adjusted_short}

    def update_regime_exposure(self, regime: str, exposure: float) -> None:
        """
        Update current exposure for a regime.

        This should be called by the position manager after each trade
        to track total exposure per regime.

        Args:
            regime: Regime label
            exposure: Current total exposure (as fraction of portfolio)
        """
        self.regime_exposures[regime] = exposure

    def get_regime_exposure(self, regime: str) -> float:
        """
        Get current exposure for regime.

        Args:
            regime: Regime label

        Returns:
            Current total exposure (as fraction of portfolio)
        """
        return self.regime_exposures.get(regime, 0.0)

    def get_available_budget(self, regime: str) -> float:
        """
        Get remaining budget for regime.

        Args:
            regime: Regime label

        Returns:
            Remaining budget (as fraction of portfolio)
        """
        budget = self.get_regime_budget(regime)
        current = self.get_regime_exposure(regime)
        return max(0.0, budget - current)

    def apply_regime_budget_cap(
        self,
        regime: str,
        position_size_pct: float
    ) -> Tuple[float, bool]:
        """
        Apply regime risk budget cap to position size.

        This prevents concentration risk by limiting total exposure
        per regime (e.g., max 30% in CRISIS).

        Args:
            regime: Regime label
            position_size_pct: Proposed position size (as % of portfolio)

        Returns:
            Tuple of (capped_size_pct, was_capped)
        """
        budget = self.get_regime_budget(regime)
        current = self.get_regime_exposure(regime)
        available = budget - current

        if position_size_pct > available:
            capped = max(0.0, available)
            logger.info(
                f"Regime budget cap applied for {regime}: "
                f"{position_size_pct:.1%} -> {capped:.1%} "
                f"(budget={budget:.1%}, current={current:.1%})"
            )
            return capped, True

        return position_size_pct, False

    def reset_regime_exposures(self) -> None:
        """
        Reset all regime exposures to zero.

        Call this at the start of each backtest iteration.
        """
        for regime in self.regime_exposures.keys():
            self.regime_exposures[regime] = 0.0

    def get_allocation_summary(
        self,
        regime: str,
        show_effective: bool = True
    ) -> str:
        """
        Get formatted summary of allocations for a regime.

        Args:
            regime: Regime label
            show_effective: If True, show effective allocation with regime budget applied

        Returns:
            Formatted summary string
        """
        lines = [f"\nRegime: {regime.upper()}"]
        lines.append(f"Budget: {self.get_regime_budget(regime):.1%}")
        lines.append(f"Current Exposure: {self.get_regime_exposure(regime):.1%}")
        lines.append(f"Available: {self.get_available_budget(regime):.1%}")

        # Show directional budgets
        directional_budgets = self.REGIME_DIRECTIONAL_BUDGETS.get(regime, {})
        if directional_budgets:
            lines.append(f"\nDirectional Budgets:")
            lines.append(f"  Long:  {directional_budgets.get('long', 0.5):.1%}")
            lines.append(f"  Short: {directional_budgets.get('short', 0.5):.1%}")

        # Get archetype weights and cash bucket
        weights = self.get_regime_distribution(regime)
        cash_bucket = self.get_cash_bucket_weight(regime)
        total_allocated = sum(weights.values())

        # Split weights by direction
        long_weights = {}
        short_weights = {}
        for archetype, weight in weights.items():
            direction = self.archetype_directions.get(archetype, 'long')
            if direction == 'short':
                short_weights[archetype] = weight
            else:
                long_weights[archetype] = weight

        total_long = sum(long_weights.values())
        total_short = sum(short_weights.values())

        lines.append(f"\nArchetype Weights (sum={total_allocated:.1%}):")
        lines.append(f"  Long total:  {total_long:.3f}")
        lines.append(f"  Short total: {total_short:.3f}")

        # Show long archetypes
        if long_weights:
            lines.append(f"\n  Long Archetypes:")
            for archetype, weight in sorted(long_weights.items(), key=lambda x: -x[1]):
                metrics = self.get_edge_metrics(archetype, regime)
                lines.append(
                    f"    {archetype}: {weight:.3f} "
                    f"(sharpe={metrics['edge_raw']:.3f}, n={metrics['n_trades']})"
                )

        # Show short archetypes
        if short_weights:
            lines.append(f"\n  Short Archetypes:")
            for archetype, weight in sorted(short_weights.items(), key=lambda x: -x[1]):
                metrics = self.get_edge_metrics(archetype, regime)
                lines.append(
                    f"    {archetype}: {weight:.3f} "
                    f"(sharpe={metrics['edge_raw']:.3f}, n={metrics['n_trades']})"
                )

        # Show cash bucket if exists
        if cash_bucket > 0:
            lines.append(f"\nCash Bucket: {cash_bucket:.1%}")
            lines.append(
                f"  → Unused allocation due to weak archetype edge"
            )

        # Show effective allocation with regime budget
        if show_effective:
            regime_budget = self.get_regime_budget(regime)
            lines.append(f"\nEffective Portfolio Allocation (with {regime_budget:.0%} regime budget):")

            for archetype, weight in sorted(weights.items(), key=lambda x: -x[1]):
                direction = self.archetype_directions.get(archetype, 'long')
                effective_pct = weight * regime_budget
                lines.append(f"  {archetype} ({direction}): {effective_pct:.1%}")

            if cash_bucket > 0:
                # Cash within regime budget
                cash_effective_pct = cash_bucket * regime_budget
                lines.append(f"  CASH (regime): {cash_effective_pct:.1%}")

            # Remaining cash outside regime budget
            remaining_budget = 1.0 - regime_budget
            if remaining_budget > 0:
                lines.append(f"  CASH (other regimes/reserve): {remaining_budget:.1%}")

        return '\n'.join(lines)

    def get_directional_allocation_summary(self) -> str:
        """
        Get formatted summary showing directional bias across all regimes.

        Returns:
            Formatted summary string showing how allocations shift by direction
        """
        lines = ["\n" + "="*80]
        lines.append("REGIME-DIRECTIONAL ALLOCATION SUMMARY")
        lines.append("="*80)

        regimes = ['crisis', 'risk_off', 'neutral', 'risk_on']

        for regime in regimes:
            weights = self.get_regime_distribution(regime)

            # Split by direction
            long_total = 0.0
            short_total = 0.0

            for archetype, weight in weights.items():
                direction = self.archetype_directions.get(archetype, 'long')
                if direction == 'short':
                    short_total += weight
                else:
                    long_total += weight

            total = long_total + short_total

            # Get directional budgets
            budgets = self.REGIME_DIRECTIONAL_BUDGETS.get(regime, {})
            long_budget = budgets.get('long', 0.5)
            short_budget = budgets.get('short', 0.5)

            lines.append(f"\n{regime.upper()}:")
            lines.append(f"  Directional Budgets: Long={long_budget:.1%}, Short={short_budget:.1%}")
            lines.append(f"  Actual Weights:      Long={long_total:.3f}, Short={short_total:.3f}")

            if total > 0:
                long_pct = long_total / total
                short_pct = short_total / total
                lines.append(f"  Allocation Mix:      Long={long_pct:.1%}, Short={short_pct:.1%}")

        lines.append("\n" + "="*80)
        lines.append("INTERPRETATION:")
        lines.append("="*80)
        lines.append("Crisis:   Aggressive shorts (60%), cautious longs (15%)")
        lines.append("Risk_off: Favor shorts (50%), reduce longs (25%)")
        lines.append("Neutral:  Balanced exposure (40% each)")
        lines.append("Risk_on:  Aggressive longs (60%), minimal shorts (20%)")
        lines.append("\nThis aligns with real trader psychology:")
        lines.append("- Bear markets → aggressive on shorts, selective on longs")
        lines.append("- Bull markets → aggressive on longs, minimal shorts")

        return '\n'.join(lines)
