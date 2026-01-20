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


logger = logging.getLogger(__name__)


class RegimeWeightAllocator:
    """
    Compute regime-conditioned portfolio weights using empirical edge data.

    Core Formula (3-step process):
    1. Shrink edge by sample size: edge_shrunk = edge * (N / (N + k))
    2. Map to positive strength: strength = sigmoid(alpha * edge_shrunk)
    3. Apply guardrails: cap negative edge at 20%, floor at 1%

    Ensures smooth, continuous allocation with no hard cliffs.
    """

    # Regime-specific risk budgets (max total exposure per regime)
    REGIME_RISK_BUDGETS = {
        'crisis': 0.30,    # Max 30% total exposure in CRISIS
        'risk_off': 0.50,  # Max 50% in RISK_OFF
        'neutral': 0.70,   # Max 70% in NEUTRAL
        'risk_on': 0.80    # Max 80% in RISK_ON
    }

    def __init__(
        self,
        edge_table_path: str,
        config_path: Optional[str] = None,
        config_override: Optional[Dict] = None
    ):
        """
        Initialize the allocator with edge data and configuration.

        Args:
            edge_table_path: Path to archetype_regime_edge_table.csv
            config_path: Path to regime_allocator_config.json (optional)
            config_override: Direct config dict (overrides config_path)
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
            f"min_weight={self.min_weight}, neg_edge_cap={self.neg_edge_cap}"
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

    def _load_edge_table(self) -> pd.DataFrame:
        """Load the edge table from CSV."""
        if not Path(self.edge_table_path).exists():
            raise FileNotFoundError(
                f"Edge table not found at {self.edge_table_path}. "
                f"Run edge computation script first."
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

        return df

    def compute_weight(
        self,
        edge: float,
        N: int,
        archetype: str,
        regime: str
    ) -> float:
        """
        Compute regime-conditioned weight with guardrails.

        EXACT IMPLEMENTATION per SOFT_GATING_PHASE1_SPEC.md

        Args:
            edge: Sharpe-like metric (risk-adjusted return)
            N: Number of trades (sample size)
            archetype: Archetype name (for logging)
            regime: Regime label (for logging)

        Returns:
            weight ∈ [0.01, 1.0] for allocation
        """
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

        # Log weight computation for debugging
        logger.debug(
            f"Weight computation: archetype={archetype}, regime={regime}, "
            f"edge={edge:.4f}, N={N}, edge_shrunk={edge_shrunk:.4f}, "
            f"strength={strength:.4f}, weight={weight:.4f}"
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

        Args:
            regime: Regime label

        Returns:
            Dictionary mapping archetype -> weight (unnormalized)
        """
        mask = self.edge_data['regime'] == regime
        regime_data = self.edge_data[mask]

        distribution = {}
        for _, row in regime_data.iterrows():
            archetype = row['archetype']
            weight = self.get_weight(archetype, regime)
            distribution[archetype] = weight

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

        # Get archetype weights and cash bucket
        weights = self.get_regime_distribution(regime)
        cash_bucket = self.get_cash_bucket_weight(regime)
        total_allocated = sum(weights.values())

        lines.append(f"\nArchetype Weights (sum={total_allocated:.1%}):")

        # Sort by weight (descending)
        for archetype, weight in sorted(weights.items(), key=lambda x: -x[1]):
            metrics = self.get_edge_metrics(archetype, regime)
            lines.append(
                f"  {archetype}: {weight:.1%} "
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
                effective_pct = weight * regime_budget
                lines.append(f"  {archetype}: {effective_pct:.1%}")

            if cash_bucket > 0:
                # Cash within regime budget
                cash_effective_pct = cash_bucket * regime_budget
                lines.append(f"  CASH (regime): {cash_effective_pct:.1%}")

            # Remaining cash outside regime budget
            remaining_budget = 1.0 - regime_budget
            if remaining_budget > 0:
                lines.append(f"  CASH (other regimes/reserve): {remaining_budget:.1%}")

        return '\n'.join(lines)
