"""
ArchetypeInstance - Self-Contained Archetype with Isolated Fusion Calculation

Each archetype calculates its OWN fusion score using archetype-specific weights
from YAML configs (configs/archetypes/*.yaml), with hard gates evaluated before
fusion scoring and whale conflict penalties applied after.

Key Properties:
- Own fusion calculation (archetype-specific weights from YAML)
- YAML-driven hard gates (evaluated BEFORE fusion, fail = reject signal)
- Whale conflict penalty (direction-aware 4-signal OI/funding/LS/taker check)
- Per-archetype base thresholds (not uniform)
- Can be backtested in ISOLATION

Pipeline: hard_gates → fusion_score → whale_penalty → threshold_check → signal
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd

from engine.models.base import Signal

logger = logging.getLogger(__name__)


def _safe_float(val) -> float:
    """Convert value to float, returning 0.0 for NaN/None/non-numeric."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return f if f == f else 0.0  # NaN check
    except (ValueError, TypeError):
        return 0.0


# Derived features computed from OHLC for hard gate evaluation
DERIVED_FEATURES = {
    'lower_wick_pct': lambda f: (min(_safe_float(f.get('open', 0)), _safe_float(f.get('close', 0))) - _safe_float(f.get('low', 0))) / max(_safe_float(f.get('high', 0)) - _safe_float(f.get('low', 0)), 0.01),
    'upper_wick_pct': lambda f: (_safe_float(f.get('high', 0)) - max(_safe_float(f.get('open', 0)), _safe_float(f.get('close', 0)))) / max(_safe_float(f.get('high', 0)) - _safe_float(f.get('low', 0)), 0.01),
    'wick_anomaly': lambda f: (
        (min(_safe_float(f.get('open', 0)), _safe_float(f.get('close', 0))) - _safe_float(f.get('low', 0))) / max(_safe_float(f.get('high', 0)) - _safe_float(f.get('low', 0)), 0.01) > 0.35
        or (_safe_float(f.get('high', 0)) - max(_safe_float(f.get('open', 0)), _safe_float(f.get('close', 0)))) / max(_safe_float(f.get('high', 0)) - _safe_float(f.get('low', 0)), 0.01) > 0.35
    ),
    'any_bos_1h': lambda f: bool(_safe_float(f.get('tf1h_bos_bullish', 0))) or bool(_safe_float(f.get('tf1h_bos_bearish', 0))),
    'any_bos_any_tf': lambda f: (
        bool(_safe_float(f.get('tf1h_bos_bullish', 0))) or bool(_safe_float(f.get('tf1h_bos_bearish', 0)))
        or bool(_safe_float(f.get('tf4h_bos_bullish', 0))) or bool(_safe_float(f.get('tf4h_bos_bearish', 0)))
    ),
    'any_fvg': lambda f: _safe_float(f.get('tf1h_fvg_present', 0)) > 0 or _safe_float(f.get('tf4h_fvg_present', 0)) > 0 or bool(f.get('fvg_present', False)),
    'rsi_extreme_65': lambda f: _safe_float(f.get('rsi_14', 50)) > 65 or _safe_float(f.get('rsi_14', 50)) < 35,
    'upper_wick_body_ratio': lambda f: min(
        (_safe_float(f.get('high', 0)) - max(_safe_float(f.get('open', 0)), _safe_float(f.get('close', 0)))) / max(abs(_safe_float(f.get('close', 0)) - _safe_float(f.get('open', 0))), 0.01),
        10.0
    ),
    # Phase 2: Strategy notebook derived features
    'wyckoff_in_accumulation': lambda f: str(f.get('wyckoff_context', '')).lower() == 'accumulation',
    'wyckoff_in_distribution': lambda f: str(f.get('wyckoff_context', '')).lower() == 'distribution',
    'wyckoff_phase_c': lambda f: str(f.get('wyckoff_phase_abc', '')).upper() == 'C',
    'lower_wick_dominant': lambda f: _safe_float(f.get('wick_lower_ratio', 0)) > 1.5,
    'bb_tight_compression': lambda f: _safe_float(f.get('bb_width', 1.0)) < 0.02,
    'rsi_divergence_bullish': lambda bar: 1 if bar.get('rsi_divergence', 0) > 0.1 else 0,
    # Post-rally exhaustion: reads pre-computed column injected by backtester/live computer
    'prior_12h_return': lambda f: _safe_float(f.get('prior_12h_return', 0.0)),
    # Wyckoff phase + price position context gates
    # "Don't long at resistance in distribution, don't short at support in accumulation"
    'distribution_at_resistance': lambda f: (
        _safe_float(f.get('tf4h_wyckoff_bearish_score', 0.0)) > 0.5
        and _safe_float(f.get('range_position_20', f.get('range_position', 0.5))) > 0.70
    ),
    'accumulation_at_support': lambda f: (
        _safe_float(f.get('tf4h_wyckoff_bullish_score', 0.0)) > 0.5
        and _safe_float(f.get('range_position_20', f.get('range_position', 0.5))) < 0.30
    ),
    # Distribution exhaustion: the highest-conviction long entry signal
    # 4H bearish (distribution) + OI declining (capitulation) + price at support (bounce zone)
    # Walk-forward validated: PF 3.74 at bearish 0.6+, OI declining p<0.0001 (+2.3% fwd)
    'distribution_exhaustion': lambda f: (
        _safe_float(f.get('tf4h_wyckoff_bearish_score', 0.0)) >= 0.6
        and _safe_float(f.get('oi_change_24h', 0.0)) < -0.02
        and _safe_float(f.get('range_position_20', f.get('range_position', 0.5))) < 0.40
    ),
}

# Features previously frozen in V12 feature store — NOW ALL PATCHED with real values.
# bin/patch_frozen_features.py computed: fusion_smc, tf4h_fvg_present,
# tf4h_squiggle_confidence, tf1h_frvp_distance_to_poc, tf4h_choch_flag.
# boms_strength, tf1d_boms_strength, tf4h_boms_displacement already had real values.
FROZEN_FEATURES: set = set()


@dataclass
class ArchetypeConfig:
    """
    Configuration for a single archetype instance.

    Each archetype is self-contained with its own:
    - Fusion weights (domain emphasis)
    - Thresholds (entry/exit criteria)
    - Exit rules (stop loss, take profit, time limits)
    - Position sizing parameters
    """

    # Identity
    name: str
    direction: str  # "long" or "short"

    # Fusion calculation (archetype-specific)
    # Example: {"wyckoff": 0.40, "liquidity": 0.50, "momentum": 0.10}
    fusion_weights: Dict[str, float] = field(default_factory=dict)

    # Entry thresholds (archetype-specific)
    entry_threshold: float = 0.35
    min_liquidity: float = 0.12

    # Exit rules (archetype-specific)
    atr_stop_mult: float = 2.5
    atr_tp_mult: float = 2.5
    max_hold_hours: int = 72
    trailing_stop: bool = True

    # Position sizing (archetype-specific)
    max_risk_pct: float = 0.02  # 2% risk per trade

    # Regime sensitivity (for portfolio allocator)
    regime_weights: Dict[str, float] = field(default_factory=dict)
    # Example: {"crisis": 0.10, "risk_off": 0.30, "neutral": 0.80, "risk_on": 1.00}

    # Pattern-specific parameters (optional)
    pattern_params: Dict = field(default_factory=dict)

    # Hard pattern gates (declarative, evaluated BEFORE fusion computation)
    hard_gates: List[Dict] = field(default_factory=list)

    # Gate mode: "hard" (block on failure), "soft" (penalize fusion score on failure)
    gate_mode: str = "hard"

    def __post_init__(self):
        """Set default fusion weights if not provided."""
        if not self.fusion_weights:
            # Default: Equal weighting across domains
            self.fusion_weights = {
                'wyckoff': 0.25,
                'liquidity': 0.25,
                'momentum': 0.25,
                'smc': 0.25
            }

        if not self.regime_weights:
            # Default: Equal weighting across regimes
            self.regime_weights = {
                'crisis': 0.50,
                'risk_off': 0.70,
                'neutral': 0.85,
                'risk_on': 1.00
            }

    def validate(self):
        """Validate configuration parameters."""
        assert self.name, "Archetype name required"
        assert self.direction in ['long', 'short', 'neutral'], f"Invalid direction: {self.direction}"

        # Validate fusion weights sum close to 1.0
        total_weight = sum(self.fusion_weights.values())
        assert 0.99 <= total_weight <= 1.01, f"Fusion weights sum to {total_weight}, expected ~1.0"

        # Validate thresholds
        assert 0.0 <= self.entry_threshold <= 1.0, "Entry threshold must be in [0, 1]"
        assert 0.0 <= self.min_liquidity <= 1.0, "Min liquidity must be in [0, 1]"

        # Validate exit parameters
        assert self.atr_stop_mult > 0, "ATR stop multiplier must be positive"
        assert self.max_hold_hours > 0, "Max hold hours must be positive"

        # Validate risk
        assert 0.0 < self.max_risk_pct < 0.1, "Risk % must be in (0, 10%)"


class ArchetypeInstance:
    """
    Self-contained trading strategy with NO dependencies on other archetypes.

    This solves the coupling issue where all archetypes shared fusion_total,
    causing changes to one archetype to affect all others.

    Key Properties:
    - Own fusion calculation (archetype-specific weights)
    - Own thresholds (independent entry/exit criteria)
    - Own position sizing (before portfolio scaling)
    - Can be backtested in ISOLATION

    Example:
        >>> config = ArchetypeConfig(
        ...     name='spring',
        ...     direction='long',
        ...     fusion_weights={'wyckoff': 0.70, 'liquidity': 0.15, 'momentum': 0.10, 'smc': 0.05}
        ... )
        >>> spring = ArchetypeInstance(config)
        >>> fusion = spring.compute_fusion_score(features)
        >>> # This fusion score is INDEPENDENT of other archetypes
    """

    def __init__(self, config: ArchetypeConfig):
        self.config = config
        self.name = config.name
        self.direction = config.direction

        # Cooling period state (for overtrading prevention)
        self.last_signal_bar = None
        self.cooling_period_bars = config.pattern_params.get('cooling_period_bars', 24)  # Default 24H for 1H data

        # Validate config
        config.validate()

        logger.info(f"Initialized archetype: {self.name}")
        logger.info(f"  Direction: {self.direction}")
        logger.info(f"  Fusion weights: {self.config.fusion_weights}")
        logger.info(f"  Entry threshold: {self.config.entry_threshold}")
        logger.info(f"  Cooling period: {self.cooling_period_bars} bars")

    def compute_fusion_score(self, features: Dict, regime: str = 'neutral') -> float:
        """
        Calculate archetype-specific fusion score with penalties for:
        - Funding rate overcrowding (Moneytaur/CryptoChase signal)
        - Fibonacci premium zone (overextended longs)
        - Setup staleness >89 bars (Moneytaur time decay)

        Args:
            features: Dict of feature values from feature store
            regime: Current regime label (applies regime_preferences multiplier)

        Returns:
            Fusion score in [0, 1]
        """
        # Extract domain scores from features
        wyckoff_score = self._get_wyckoff_score(features)
        liquidity_score = self._get_liquidity_score(features)
        momentum_score = self._get_momentum_score(features)
        smc_score = self._get_smc_score(features)

        # Apply archetype-specific weights
        fusion = (
            self.config.fusion_weights.get('wyckoff', 0.0) * wyckoff_score +
            self.config.fusion_weights.get('liquidity', 0.0) * liquidity_score +
            self.config.fusion_weights.get('momentum', 0.0) * momentum_score +
            self.config.fusion_weights.get('smc', 0.0) * smc_score
        )

        # Apply penalties (fakeouts, PTI, etc.)
        if features.get('tf1h_fakeout_detected', False):
            fusion -= 0.10

        # PTI penalty (applies to all archetypes)
        pti_penalty = max(
            features.get('tf1d_pti_score', 0.0),
            features.get('tf1h_pti_score', 0.0)
        )
        fusion -= 0.10 * pti_penalty

        # --- WHALE CONFLICT PENALTY ---
        # Direction-aware institutional data conflict detection.
        # Checks 4 signals: funding overcrowding, OI momentum, taker flow, L/S ratio.
        # Only penalties, never bonuses (lesson #1). NaN-safe (absence = neutral).
        whale_mult = self._compute_whale_conflict(features)
        fusion *= whale_mult

        # --- FIBONACCI PREMIUM ZONE PENALTY ---
        # Penalize longs in premium zone (price above 0.618 retracement = overextended)
        fib_in_premium = features.get('fib_in_premium', 0)
        if not pd.isna(fib_in_premium) and fib_in_premium and self.direction == 'long':
            fusion *= 0.92  # 8% penalty for longs in premium zone

        # --- STALENESS PENALTY (Moneytaur-style time decay) ---
        # Only apply if bars_since_pivot data is actually in the feature store
        # Setups older than 89 bars lose edge (stale wicks lose edge)
        bars_since = features.get('bars_since_pivot', None)
        if bars_since is not None and not pd.isna(bars_since) and bars_since > 89:
            fusion *= 0.80  # Heavy decay: stale setup past fib-89 window

        # NOTE: regime_preferences (self.config.regime_weights) are NOT applied as
        # fusion multipliers. The soft regime system works through
        # fusion_thresholds_by_regime in the backtester/strategy, which requires
        # higher fusion scores in crisis/risk_off. Applying regime_preferences ON TOP
        # of varying thresholds double-gates and hurts archetypes that perform well
        # across regimes (e.g., trap_within_trend PF=2.19 in neutral was killed by 0.7x).

        # Clip to [0, 1]
        return max(0.0, min(1.0, fusion))

    def _get_wyckoff_score(self, features: Dict) -> float:
        """
        Extract Wyckoff domain score using MTF confluence scoring.

        Architecture:
        1. Collect direction-appropriate scores from 1H, 4H, 1D
        2. Compute weighted average: 0.50 * 1H + 0.30 * 4H + 0.20 * 1D
        3. Apply confluence multiplier: bonus when multiple TFs agree
           - 1 TF active: 1.0x (no bonus)
           - 2 TFs active: 1.15x
           - 3 TFs active: 1.30x
        4. Fallback: direction-aware event confidence (no BC leak into longs)

        This rewards multi-timeframe alignment over a single loud TF.
        """
        # --- Primary: Graded directional scores (diversity-weighted, SM-validated) ---
        bullish_1h = _safe_float(features.get('wyckoff_bullish_score', 0.0))
        bearish_1h = _safe_float(features.get('wyckoff_bearish_score', 0.0))
        bullish_4h = _safe_float(features.get('tf4h_wyckoff_bullish_score', 0.0))
        bearish_4h = _safe_float(features.get('tf4h_wyckoff_bearish_score', 0.0))
        bullish_1d = _safe_float(features.get('tf1d_wyckoff_bullish_score', 0.0))
        bearish_1d = _safe_float(features.get('tf1d_wyckoff_bearish_score', 0.0))

        # Select direction-appropriate scores
        if self.direction == 'long':
            s_1h, s_4h, s_1d = bullish_1h, bullish_4h, bullish_1d
        else:
            s_1h, s_4h, s_1d = bearish_1h, bearish_4h, bearish_1d

        # MTF confluence scoring
        tf_scores = [s_1h, s_4h, s_1d]
        tf_weights = [0.50, 0.30, 0.20]  # 1H primary, 4H secondary, 1D context
        n_active = sum(1 for s in tf_scores if s > 0)

        if n_active > 0:
            # Weighted average of active timeframes
            weighted_sum = sum(s * w for s, w in zip(tf_scores, tf_weights))
            weight_total = sum(w for s, w in zip(tf_scores, tf_weights) if s > 0)
            # Normalize by active weights only (don't dilute with zero TFs)
            confluence_avg = weighted_sum / weight_total if weight_total > 0 else 0.0

            # Confluence multiplier: reward multi-TF alignment
            if n_active >= 3:
                confluence_mult = 1.30  # All 3 TFs agree — strong signal
            elif n_active == 2:
                confluence_mult = 1.15  # 2 TFs agree — moderate boost
            else:
                confluence_mult = 1.00  # Single TF — no bonus

            directional = min(1.0, confluence_avg * confluence_mult)
            return directional

        # --- Fallback: Direction-aware event confidence (no BC leak) ---
        # Prefer directional composite if available (live system provides these)
        if self.direction == 'long':
            dir_conf = _safe_float(features.get('wyckoff_bullish_event_confidence', 0.0))
        else:
            dir_conf = _safe_float(features.get('wyckoff_bearish_event_confidence', 0.0))
        if dir_conf > 0:
            return dir_conf

        # Non-directional fallback (feature store backward compat)
        # Only use if no directional scores exist at all
        tf4h_phase = _safe_float(features.get('tf4h_wyckoff_phase_score', 0.0))
        wyckoff_conf = _safe_float(features.get('wyckoff_event_confidence', 0.0))
        if tf4h_phase > 0 or wyckoff_conf > 0:
            return max(tf4h_phase, wyckoff_conf)

        # Last resort: M1/M2 binary signals
        if features.get('tf1d_wyckoff_m1_signal') and self.direction == 'long':
            return 0.6
        if features.get('tf1d_wyckoff_m2_signal') and self.direction == 'short':
            return 0.6

        return 0.0

    def _get_liquidity_score(self, features: Dict) -> float:
        """
        Extract liquidity domain score.

        Uses pre-computed liquidity_score from feature store (real data, mean=0.485).
        Fallback: computes from non-frozen components only (volume, ATR, OI).
        Old formula used frozen BOMS/FVG (all zero) → capped at 0.333. Fixed.
        """
        # Try pre-computed liquidity score first (feature store has real values)
        if 'liquidity_score' in features and not pd.isna(features['liquidity_score']):
            return features['liquidity_score']

        # Fallback for live system: use non-frozen components only
        components = []
        weights = []

        # Volume Z-score (real data, measures liquidity demand)
        vol_z = _safe_float(features.get('volume_z_7d', features.get('volume_zscore', 0.0)))
        vol_score = min(max(vol_z, 0.0) / 2.5, 1.0)  # Normalize: Z>2.5 = max liquidity
        components.append(vol_score)
        weights.append(0.35)

        # ATR percentile (real data, higher ATR = more movement/liquidity)
        atr_pct = _safe_float(features.get('atr_percentile', 0.5))
        components.append(atr_pct)
        weights.append(0.25)

        # FVG presence (tf1h has some real data, tf4h is frozen)
        fvg = _safe_float(features.get('tf1h_fvg_present', 0.0))
        components.append(fvg)
        weights.append(0.20)

        # OI-based liquidity (new Binance features, if available)
        oi_change = _safe_float(features.get('oi_change_4h', 0.0))
        oi_score = min(abs(oi_change) * 10.0, 1.0)  # Large OI changes = institutional activity
        components.append(oi_score)
        weights.append(0.20)

        liquidity = sum(c * w for c, w in zip(components, weights))

        return max(0.0, min(1.0, liquidity))

    def _get_momentum_score(self, features: Dict) -> float:
        """
        Extract momentum domain score.

        Combines ADX, RSI, and squiggle confidence.
        """
        # ADX strength (normalized to 0-1)
        adx = features.get('adx_14', 20.0) / 100.0

        # RSI momentum (distance from 50)
        rsi = features.get('rsi_14', 50.0)
        rsi_momentum = abs(rsi - 50.0) / 50.0

        # Squiggle confidence (real values from feature store)
        squiggle_conf = features.get('tf4h_squiggle_confidence', 0.0)
        # Frozen bypass (no-op when FROZEN_FEATURES is empty)
        if 'tf4h_squiggle_confidence' in FROZEN_FEATURES:
            squiggle_conf = 0.0

        momentum = (adx + rsi_momentum + squiggle_conf) / 3.0

        return max(0.0, min(1.0, momentum))

    def _get_smc_score(self, features: Dict) -> float:
        """
        Extract SMC (Smart Money Concepts) domain score.

        Computed from BOS/CHOCH/FVG component features. The parquet column
        fusion_smc is frozen at 0.5 (builder.py placeholder, never computed)
        and is intentionally ignored here.
        """
        # Always compute from SMC component features
        # Use directional BOS scores (real data, ~3-4% active)
        if self.direction == 'long':
            bos_score = _safe_float(features.get('tf1h_bos_bullish', 0.0))
        else:
            bos_score = _safe_float(features.get('tf1h_bos_bearish', 0.0))

        # CHOCH detection
        choch_score = 1.0 if features.get('tf1h_choch_detected', False) else 0.0

        # FVG — tf1h has real data, tf4h is frozen at 0
        fvg_score = _safe_float(features.get('tf1h_fvg_present', 0.0))

        # Only count non-frozen components
        components = [bos_score, choch_score, fvg_score]
        active = [c for c in components if c > 0]

        if not active:
            return 0.0  # No SMC signal = no boost (not neutral 0.5)

        # Score based on how many SMC components are active
        smc = sum(components) / len(components)

        return max(0.0, min(1.0, smc))

    def _compute_whale_conflict(self, features: Dict) -> float:
        """
        Compute whale conflict penalty based on institutional data.

        Checks 4 direction-aware signals for conflict between trade direction
        and what whale/institutional data suggests. Each signal that conflicts
        adds a penalty tier. Only penalties, NEVER bonuses.

        For LONGS (conflict = institutional data says "don't go long"):
          - funding_Z > 2.0: Longs overcrowded
          - oi_change_4h < -0.03: Mass position closing
          - taker_imbalance < -0.3: Aggressive selling
          - ls_ratio_extreme > 2.0: L/S ratio extreme long crowding

        For SHORTS (conflict = institutional data says "don't go short"):
          - funding_Z < -2.0: Shorts overcrowded
          - oi_change_4h > 0.05 AND funding > 0: Strong long accumulation
          - taker_imbalance > 0.3: Aggressive buying
          - ls_ratio_extreme < -2.0: L/S ratio extreme short crowding

        Returns:
            Multiplier in [0.70, 1.00] — lower = more conflict
        """
        PENALTY_TIERS = [1.00, 0.95, 0.90, 0.85, 0.80]  # 0,1,2,3,4 conflicts
        conflicts = 0

        # --- Signal 1: Funding rate overcrowding ---
        funding_z = _safe_float(features.get('funding_Z', 0.0))
        funding_z_raw = features.get('funding_Z')
        has_funding = funding_z_raw is not None and not (isinstance(funding_z_raw, float) and funding_z_raw != funding_z_raw)

        if has_funding:
            if self.direction == 'long' and funding_z > 2.0:
                conflicts += 1
            elif self.direction == 'short' and funding_z < -2.0:
                conflicts += 1

        # --- Signal 2: OI momentum (mass position closing/building) ---
        # Thresholds set extreme to avoid false penalties during normal volatility
        oi_4h_raw = features.get('oi_change_4h')
        has_oi = oi_4h_raw is not None and not (isinstance(oi_4h_raw, float) and oi_4h_raw != oi_4h_raw)

        if has_oi:
            oi_4h = _safe_float(oi_4h_raw)
            if self.direction == 'long' and oi_4h < -0.05:
                # OI dropping >5% in 4h = mass exodus, bad for longs
                conflicts += 1
            elif self.direction == 'short':
                # Strong long accumulation: OI surging + positive funding
                funding_rate = _safe_float(features.get('binance_funding_rate', 0.0))
                if oi_4h > 0.07 and funding_rate > 0:
                    conflicts += 1

        # --- Signal 3: Taker flow imbalance ---
        taker_raw = features.get('taker_imbalance')
        has_taker = taker_raw is not None and not (isinstance(taker_raw, float) and taker_raw != taker_raw)

        if has_taker:
            taker = _safe_float(taker_raw)
            if self.direction == 'long' and taker < -0.5:
                # Extreme aggressive selling, bad for longs
                conflicts += 1
            elif self.direction == 'short' and taker > 0.5:
                # Extreme aggressive buying, bad for shorts
                conflicts += 1

        # --- Signal 4: L/S ratio extreme ---
        ls_raw = features.get('ls_ratio_extreme')
        has_ls = ls_raw is not None and not (isinstance(ls_raw, float) and ls_raw != ls_raw)

        if has_ls:
            ls_ratio = _safe_float(ls_raw)
            if self.direction == 'long' and ls_ratio > 2.0:
                # Extreme long crowding, bad for longs
                conflicts += 1
            elif self.direction == 'short' and ls_ratio < -2.0:
                # Extreme short crowding, bad for shorts
                conflicts += 1

        # Cap at max index
        idx = min(conflicts, len(PENALTY_TIERS) - 1)
        multiplier = PENALTY_TIERS[idx]

        if conflicts > 0:
            logger.debug(
                f"[WHALE_CONFLICT] {self.name} ({self.direction}): "
                f"{conflicts} conflicts -> {multiplier:.2f}x penalty"
            )

        return multiplier

    def can_signal(self, current_bar_idx: int) -> bool:
        """
        Check if archetype can signal (cooling period check).

        Args:
            current_bar_idx: Current bar index (monotonically increasing)

        Returns:
            True if cooling period has elapsed, False otherwise
        """
        if self.last_signal_bar is None:
            return True
        return (current_bar_idx - self.last_signal_bar) >= self.cooling_period_bars

    def _check_hard_gates(self, features: Dict) -> Tuple[bool, Optional[str]]:
        """
        Evaluate hard pattern gates BEFORE computing fusion score.
        Gate definitions loaded from YAML configs (configs/archetypes/*.yaml).

        Returns:
            (passed, reason) - passed=True if all gates pass, reason=None on pass
        """
        passed, reason, _ = self._evaluate_gates(features)
        return passed, reason

    def _evaluate_gates(self, features: Dict) -> Tuple[bool, Optional[str], float]:
        """
        Evaluate gates and return pass/fail + penalty multiplier.

        Returns:
            (all_passed, failed_description, penalty_multiplier)
            - penalty_multiplier: 1.0 if all pass, reduced for each failure
              e.g. 2 gates, 1 fails -> 0.50 penalty
        """
        if not self.config.hard_gates:
            return True, None, 1.0

        total_gates = 0
        failed_gates = 0
        first_failure = None

        for gate in self.config.hard_gates:
            feature_key = gate.get('feature', '')
            op = gate.get('op', 'bool_true')
            value = gate.get('value')
            nan_policy = gate.get('nan_policy', 'fail')
            frozen_bypass = gate.get('frozen_bypass', False)
            description = gate.get('description', feature_key)

            # Resolve feature value
            if feature_key.startswith('derived:'):
                derived_name = feature_key.split(':', 1)[1]
                compute_fn = DERIVED_FEATURES.get(derived_name)
                if compute_fn is None:
                    logger.warning(f"[HARD_GATE] Unknown derived feature: {derived_name}")
                    if nan_policy == 'fail':
                        total_gates += 1
                        failed_gates += 1
                        if first_failure is None:
                            first_failure = f"unknown_derived:{derived_name}"
                    continue
                try:
                    feat_val = compute_fn(features)
                except Exception as e:
                    logger.debug(f"[HARD_GATE] Error computing {derived_name}: {e}")
                    if nan_policy == 'fail':
                        total_gates += 1
                        failed_gates += 1
                        if first_failure is None:
                            first_failure = f"compute_error:{derived_name}"
                    continue
            else:
                feat_val = features.get(feature_key)

                # Check frozen bypass
                if frozen_bypass and feature_key in FROZEN_FEATURES:
                    continue

            # NaN / None handling
            is_nan = feat_val is None or (isinstance(feat_val, float) and feat_val != feat_val)
            if is_nan:
                if nan_policy == 'fail':
                    total_gates += 1
                    failed_gates += 1
                    if first_failure is None:
                        first_failure = f"nan:{description}"
                else:  # 'skip' or 'pass'
                    pass
                continue

            # Convert to float for comparison
            feat_val = _safe_float(feat_val) if not isinstance(feat_val, bool) else feat_val

            # Evaluate gate condition
            total_gates += 1
            gate_failed = False
            if op == 'min' and feat_val < value:
                gate_failed = True
            elif op == 'max' and feat_val > value:
                gate_failed = True
            elif op == 'bool_true' and not feat_val:
                gate_failed = True
            elif op == 'bool_false' and feat_val:
                gate_failed = True
            elif op == 'in_range':
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    if feat_val < value[0] or feat_val > value[1]:
                        gate_failed = True
            elif op == 'eq' and feat_val != value:
                gate_failed = True

            if gate_failed:
                failed_gates += 1
                if first_failure is None:
                    first_failure = description

        # Calculate penalty: each failed gate reduces score proportionally
        if total_gates == 0:
            return True, None, 1.0

        all_passed = (failed_gates == 0)
        # Penalty: e.g. 1 of 2 gates fail -> 0.50, 2 of 3 fail -> 0.33
        penalty = 1.0 - (failed_gates / total_gates) if total_gates > 0 else 1.0
        return all_passed, first_failure, penalty

    def detect(
        self,
        features: Dict,
        regime: str,
        current_bar_idx: Optional[int] = None,
        prev_row: Optional[pd.Series] = None,
        lookback_df: Optional[pd.DataFrame] = None,
        structural_checker=None,
        signal_mode: str = 'fusion',
    ) -> Optional[Signal]:
        """
        Generate entry signal if archetype conditions met.

        Args:
            features: Feature dict from feature store
            regime: Current regime (for metadata, not filtering)
            current_bar_idx: Current bar index (for cooling period tracking)
            prev_row: Previous bar's features (for structural checks needing lookback)
            lookback_df: DataFrame of recent bars (for structural checks needing history)
            structural_checker: StructuralChecker instance for pattern validation

        Returns:
            Signal object or None if no signal
        """
        # Structural pattern check (BEFORE cooling/gates/fusion)
        # This makes each archetype a genuine independent strategy
        if structural_checker is not None:
            row_series = pd.Series(features) if isinstance(features, dict) else features
            passed, reason = structural_checker.check_structure(
                archetype_name=self.name,
                row=row_series,
                prev_row=prev_row,
                lookback_df=lookback_df,
                bar_index=current_bar_idx or 0,
            )
            if not passed:
                logger.debug(f"[STRUCTURE] {self.name} rejected: {reason}")
                return None

        # Check cooling period (if bar index provided)
        if current_bar_idx is not None and not self.can_signal(current_bar_idx):
            logger.debug(
                f"[COOLING] {self.name} blocked by cooling period "
                f"(last={self.last_signal_bar}, current={current_bar_idx}, "
                f"need_wait={self.cooling_period_bars})"
            )
            return None

        # Check pattern gates (BEFORE fusion computation to save CPU)
        gate_passed, gate_failed, gate_penalty = self._evaluate_gates(features)

        if self.config.gate_mode == 'hard':
            # Hard mode: block signal entirely if any gate fails
            if not gate_passed:
                logger.debug(
                    f"[HARD_GATE] {self.name} rejected: {gate_failed}"
                )
                return None
            gate_penalty = 1.0  # No penalty if all passed
        else:
            # Soft mode: apply penalty multiplier to fusion score
            if not gate_passed:
                logger.debug(
                    f"[SOFT_GATE] {self.name} penalty={gate_penalty:.2f}: {gate_failed}"
                )

        # Compute archetype-specific fusion score (with regime soft multiplier)
        fusion = self.compute_fusion_score(features, regime=regime)

        # Apply gate penalty in soft mode
        fusion *= gate_penalty

        # Signal mode: controls how fusion score is used for trade selection
        if signal_mode == 'fusion':
            # Production: weighted fusion score vs dynamic threshold
            if fusion < self.config.entry_threshold:
                return None
        elif signal_mode == 'structural':
            # Structural: skip fusion threshold — hard gates + cooling are sufficient
            pass
        elif signal_mode == 'composite':
            # N-of-M: count domains with score > 0.25, require >= 3 of 4
            domain_scores = {
                'wyckoff': self._get_wyckoff_score(features),
                'liquidity': self._get_liquidity_score(features),
                'momentum': self._get_momentum_score(features),
                'smc': self._get_smc_score(features),
            }
            domains_active = sum(1 for s in domain_scores.values() if s > 0.25)
            if domains_active < 3:
                return None
        else:
            raise ValueError(f"Unknown signal_mode: {signal_mode}")

        # Check liquidity requirement
        liquidity = self._get_liquidity_score(features)
        if liquidity < self.config.min_liquidity:
            return None

        # Calculate stop loss and take profit
        entry_price = features['close']
        atr = features.get('atr_14', features.get('atr', entry_price * 0.02))

        # For neutral archetypes, skip (they need separate detection logic)
        if self.direction == 'neutral':
            return None

        if self.direction == 'long':
            stop_loss = entry_price - (self.config.atr_stop_mult * atr)
            take_profit = entry_price + (self.config.atr_tp_mult * atr)
        else:  # short
            stop_loss = entry_price + (self.config.atr_stop_mult * atr)
            take_profit = entry_price - (self.config.atr_tp_mult * atr)

        # ATR minimum distance validation: stop must be at least 0.5x ATR from entry
        min_atr_distance = atr * 0.5
        actual_distance = abs(entry_price - stop_loss)
        if actual_distance < min_atr_distance:
            logger.debug(
                "[GATE] %s: stop too close (%.2f < %.2f ATR min)",
                self.name, actual_distance, min_atr_distance
            )
            return None

        # Compute confidence (fusion score scaled)
        confidence = min(1.0, fusion / max(self.config.entry_threshold, 0.01))

        # Update cooling period state
        if current_bar_idx is not None:
            self.last_signal_bar = current_bar_idx
            logger.debug(f"[SIGNAL] {self.name} fired at bar {current_bar_idx}, next available at {current_bar_idx + self.cooling_period_bars}")

        # Compute domain scores for metadata
        wyckoff_score = self._get_wyckoff_score(features)
        momentum_score = self._get_momentum_score(features)
        smc_score = self._get_smc_score(features)

        # Build signal (no regime filtering here - that's allocator's job)
        return Signal(
            direction=self.direction,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            regime_label=regime,  # For allocator's use
            metadata={
                'archetype': self.name,
                'fusion_score': fusion,
                'gate_penalty': gate_penalty,
                'wyckoff_score': wyckoff_score,
                'liquidity_score': liquidity,
                'momentum_score': momentum_score,
                'smc_score': smc_score,
                'atr': atr,
                'atr_stop_mult': self.config.atr_stop_mult,
                'atr_tp_mult': self.config.atr_tp_mult
            }
        )

    def get_position_size(
        self,
        portfolio_value: float,
        signal: Signal,
        regime: str
    ) -> float:
        """
        Calculate position size for this archetype.

        Returns RAW size (before portfolio allocation).
        Portfolio allocator may scale this down based on:
        - Regime weight
        - Total portfolio exposure
        - Other archetype positions

        Args:
            portfolio_value: Total portfolio value ($)
            signal: Entry signal
            regime: Current regime

        Returns:
            Position size in dollars (before portfolio scaling)
        """
        # ATR-based risk sizing
        stop_distance_pct = abs(
            signal.entry_price - signal.stop_loss
        ) / signal.entry_price

        # Risk amount in dollars
        risk_dollars = portfolio_value * self.config.max_risk_pct

        # Base position size
        # Example: $10k portfolio, 2% risk = $200 risk
        #          Stop 5% away = $200 / 0.05 = $4000 position
        base_size = risk_dollars / stop_distance_pct

        # Cap at reasonable max (e.g., 12% of portfolio)
        max_size = portfolio_value * 0.12
        base_size = min(base_size, max_size)

        # IMPORTANT: Don't apply regime scaling here
        # That's the portfolio allocator's job
        return base_size

    def __repr__(self) -> str:
        return (
            f"ArchetypeInstance(name='{self.name}', direction='{self.direction}', "
            f"fusion_weights={self.config.fusion_weights})"
        )
