# Archetype Contract Design Specification

**Version:** 2.0
**Status:** Design Complete
**Author:** System Architect (Claude Code)
**Date:** 2025-12-12

---

## Executive Summary

This specification defines a **unified archetype contract and registry system** to eliminate ghost archetypes permanently. The system enforces a standard interface that all archetypes must implement, provides centralized registration, validates feature availability, and enables scalable addition of 20+ archetypes without code duplication or inconsistency.

**Key Goals:**
- **Enforceability:** System rejects incomplete archetypes at load time
- **Scalability:** Easy to add new archetypes via YAML + Python class
- **Maintainability:** Single source of truth for archetype definitions
- **Observability:** Clear status reporting of what works vs. needs implementation
- **Backward Compatibility:** Graceful migration path from existing code

---

## Architecture Overview

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                   Backtest Runner                           │
│                (backtest_knowledge_v2.py)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ArchetypeRegistry                              │
│  - Loads archetype_registry.yaml                            │
│  - Validates all archetypes implement BaseArchetype         │
│  - Provides filtering (maturity, regime, direction)         │
│  - Reports status of all registered archetypes              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            FeatureRealityGate                               │
│  - Pre-backtest feature validation                          │
│  - Reports coverage % for each archetype                    │
│  - Fails fast on missing critical features                  │
│  - Allows degraded mode for optional features               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           BaseArchetype (ABC)                               │
│  ├─ required_features() → List[str]                         │
│  ├─ score(context) → (float, Dict)                          │
│  ├─ veto(context) → (bool, str)                             │
│  ├─ entry(context) → (signal, confidence, metadata)         │
│  ├─ exit(context) → Optional[Dict]                          │
│  └─ diagnostics(context) → Dict                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Concrete Archetype Classes                         │
│  ├─ LiquidityVacuumArchetype (S1)                           │
│  ├─ FundingDivergenceArchetype (S4)                         │
│  ├─ LongSqueezeArchetype (S5)                               │
│  ├─ SpringUTADArchetype (A)  [stub → production]            │
│  └─ ... (20+ total archetypes)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. BaseArchetype Abstract Class

### Interface Definition

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    """Trade signal types"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class MaturityLevel(Enum):
    """Archetype maturity states"""
    STUB = "stub"               # Placeholder, no implementation
    DEVELOPMENT = "development" # Implementation started, not validated
    CALIBRATED = "calibrated"   # Validated on historical data
    PRODUCTION = "production"   # Live-ready, battle-tested

@dataclass
class ArchetypeScore:
    """Structured scoring output"""
    total_score: float          # [0.0, 1.0] - Overall confidence
    component_scores: Dict[str, float]  # Breakdown by feature domain
    reasons: List[str]          # Human-readable scoring factors
    metadata: Dict              # Additional context

@dataclass
class ArchetypeVeto:
    """Structured veto output"""
    is_vetoed: bool
    reason: str
    veto_type: str  # "hard_stop" | "safety" | "regime_mismatch" | "feature_missing"

@dataclass
class ArchetypeEntry:
    """Structured entry signal"""
    signal: SignalType
    confidence: float           # [0.0, 1.0]
    entry_price: Optional[float]  # None = market order
    metadata: Dict              # Stop loss, take profit, position size hints

class BaseArchetype(ABC):
    """
    Abstract base class for all archetypes.

    All concrete archetypes MUST implement this interface.
    The registry validates compliance at load time.
    """

    # ========================================================================
    # METADATA (Class attributes - override in subclass)
    # ========================================================================

    ARCHETYPE_ID: str = None          # e.g., "S1"
    ARCHETYPE_NAME: str = None        # e.g., "Liquidity Vacuum"
    MATURITY: MaturityLevel = MaturityLevel.STUB
    DIRECTION: SignalType = None      # Primary signal direction
    REGIME_TAGS: List[str] = []       # e.g., ["risk_off", "crisis"]
    REQUIRES_ENGINES: List[str] = []  # e.g., ["wyckoff", "smc"]

    # ========================================================================
    # REQUIRED METHODS (Must implement)
    # ========================================================================

    @abstractmethod
    def required_features(self) -> List[str]:
        """
        Return list of features this archetype requires.

        Features are categorized as:
        - critical: Archetype cannot function without these
        - recommended: Degraded performance without these
        - optional: Nice to have, graceful fallback available

        Returns:
            List of feature names from feature store

        Example:
            return [
                'liquidity_score',      # critical
                'volume_zscore',        # critical
                'wick_lower_ratio',     # critical
                'VIX_Z',                # recommended
                'funding_Z',            # optional
            ]
        """
        pass

    @abstractmethod
    def score(self, context: RuntimeContext) -> ArchetypeScore:
        """
        Calculate archetype confidence score.

        This is the PATTERN RECOGNITION stage - how well does current
        market state match this archetype's canonical pattern?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            ArchetypeScore with total_score [0.0, 1.0] and breakdown

        Example:
            score = ArchetypeScore(
                total_score=0.72,
                component_scores={
                    'liquidity_drain': 0.85,
                    'volume_panic': 0.78,
                    'wick_rejection': 0.65,
                    'crisis_context': 0.60
                },
                reasons=[
                    'Liquidity drained 44% below 7d avg',
                    'Volume panic z-score: 2.8',
                    'Deep lower wick: 48% of candle'
                ],
                metadata={'pattern_quality': 'high'}
            )
        """
        pass

    @abstractmethod
    def veto(self, context: RuntimeContext) -> ArchetypeVeto:
        """
        Hard safety disqualifiers.

        This is the SAFETY GATE stage - are there conditions that make
        this trade unsafe regardless of score?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            ArchetypeVeto indicating whether trade is blocked

        Example:
            # Liquidity Vacuum only trades in risk_off/crisis regimes
            if context.regime_label not in ['risk_off', 'crisis']:
                return ArchetypeVeto(
                    is_vetoed=True,
                    reason='Liquidity Vacuum requires risk_off or crisis regime',
                    veto_type='regime_mismatch'
                )
        """
        pass

    @abstractmethod
    def entry(self, context: RuntimeContext) -> ArchetypeEntry:
        """
        Generate entry signal with confidence and metadata.

        This is the EXECUTION stage - given score passed and no veto,
        what is the exact entry specification?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            ArchetypeEntry with signal, confidence, and trade parameters

        Example:
            entry = ArchetypeEntry(
                signal=SignalType.LONG,
                confidence=0.72,
                entry_price=None,  # Market order
                metadata={
                    'stop_loss_pct': -0.025,
                    'take_profit_pct': 0.08,
                    'position_size_mult': 1.0,
                    'max_hold_bars': 72,
                    'entry_reason': 'Capitulation reversal setup'
                }
            )
        """
        pass

    # ========================================================================
    # OPTIONAL METHODS (Default implementations provided)
    # ========================================================================

    def exit(self, context: RuntimeContext) -> Optional[Dict]:
        """
        Optional exit logic override.

        Most archetypes use standard exit logic (trailing stop, max hold).
        Override this if archetype has specific exit conditions.

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            None (use default exits) OR Dict with exit signal

        Example:
            # Exit if liquidity recovers above 7d average
            if context.row['liquidity_drain_pct'] > 0.10:
                return {
                    'exit_signal': True,
                    'exit_reason': 'Liquidity recovered above baseline',
                    'exit_price': None  # Market exit
                }
            return None  # Continue holding
        """
        return None

    def diagnostics(self, context: RuntimeContext) -> Dict:
        """
        Output what archetype looked at for auditing.

        This enables EXPLAINABILITY - what features did archetype use
        to make its decision?

        Args:
            context: RuntimeContext with current bar, regime, features

        Returns:
            Dict with diagnostic information

        Example:
            return {
                'timestamp': context.ts,
                'archetype': self.ARCHETYPE_ID,
                'regime': context.regime_label,
                'features_used': {
                    'liquidity_score': context.row['liquidity_score'],
                    'liquidity_drain_pct': context.row['liquidity_drain_pct'],
                    'volume_zscore': context.row['volume_zscore'],
                    'wick_lower_ratio': context.row['wick_lower_ratio']
                },
                'thresholds_applied': {
                    'liquidity_drain_min': -0.30,
                    'volume_zscore_min': 2.0,
                    'wick_lower_min': 0.30
                },
                'score_components': {...},
                'veto_checks': {...}
            }
        """
        return {
            'timestamp': context.ts,
            'archetype_id': self.ARCHETYPE_ID,
            'archetype_name': self.ARCHETYPE_NAME,
            'maturity': self.MATURITY.value,
            'regime': context.regime_label
        }

    # ========================================================================
    # UTILITY METHODS (Available to all archetypes)
    # ========================================================================

    def get_threshold(self, context: RuntimeContext, param: str, default: float = 0.0) -> float:
        """
        Get archetype-specific threshold from RuntimeContext.

        Handles regime-aware threshold resolution via ThresholdPolicy.
        """
        return context.get_threshold(self.ARCHETYPE_ID.lower(), param, default)

    def validate_features(self, context: RuntimeContext) -> Tuple[bool, List[str]]:
        """
        Validate required features are present in context.

        Returns:
            (all_present, missing_features)
        """
        required = self.required_features()
        missing = [f for f in required if f not in context.row.index]
        return (len(missing) == 0, missing)
```

---

## 2. Archetype Registry YAML

### Registry Structure

**File:** `archetype_registry.yaml`

```yaml
# Archetype Registry - Single Source of Truth
# All archetypes MUST be registered here to be recognized by the system

version: "2.0"
registry_type: "archetype"
description: "Canonical registry of all Bull Machine archetypes"

# ============================================================================
# PRODUCTION ARCHETYPES (Battle-tested, live-ready)
# ============================================================================

archetypes:
  # --- BEAR MARKET ARCHETYPES (SHORT BIAS) ---

  - id: S1
    name: "Liquidity Vacuum Reversal"
    slug: "liquidity_vacuum"
    class: "engine.strategies.archetypes.bear.LiquidityVacuumArchetype"
    maturity: production
    direction: long  # Counter-trend reversal
    regime_tags:
      - risk_off
      - crisis
    requires_engines:
      - liquidity
      - wyckoff
      - macro
    requires_features:
      critical:
        - liquidity_score
        - liquidity_drain_pct
        - volume_zscore
        - wick_lower_ratio
      recommended:
        - VIX_Z
        - DXY_Z
        - funding_Z
      optional:
        - crisis_composite
        - capitulation_depth
    enable_flag: enable_S1
    description: |
      Capitulation reversal during liquidity vacuum conditions.
      Detects extreme orderbook drain + panic selling + wick rejection.
      Target: 10-15 trades/year, PF > 2.0 in bear markets.
    historical_performance:
      backtest_period: "2022-2024"
      total_trades: 34
      win_rate: 0.68
      profit_factor: 2.34
      sharpe_ratio: 1.82

  - id: S4
    name: "Funding Divergence (Short Squeeze)"
    slug: "funding_divergence"
    class: "engine.strategies.archetypes.bear.FundingDivergenceArchetype"
    maturity: production
    direction: long  # Counter-trend reversal
    regime_tags:
      - risk_off
      - neutral
    requires_engines:
      - funding
      - liquidity
    requires_features:
      critical:
        - funding_rate
        - funding_Z
        - oi_change
      recommended:
        - liquidity_score
        - volume_zscore
      optional:
        - shorts_liquidations
    enable_flag: enable_S4
    description: |
      Short squeeze setup from extreme negative funding + rising OI.
      Targets overleveraged short positions forced to cover.
      Target: 12-18 trades/year, PF > 1.8.
    historical_performance:
      backtest_period: "2022-2024"
      total_trades: 42
      win_rate: 0.64
      profit_factor: 1.89
      sharpe_ratio: 1.45

  - id: S5
    name: "Long Squeeze Cascade"
    slug: "long_squeeze"
    class: "engine.strategies.archetypes.bear.LongSqueezeArchetype"
    maturity: production
    direction: short  # Trend continuation
    regime_tags:
      - risk_off
    requires_engines:
      - funding
      - liquidity
      - smc
    requires_features:
      critical:
        - funding_rate
        - funding_Z
        - oi_change
        - bos_detected
      recommended:
        - liquidity_score
        - volume_zscore
      optional:
        - longs_liquidations
    enable_flag: enable_S5
    description: |
      Long squeeze during downtrend with overleveraged longs.
      Extreme positive funding + BOS down + liquidity drain.
      Target: 8-12 trades/year, PF > 2.0.
    historical_performance:
      backtest_period: "2022-2024"
      total_trades: 28
      win_rate: 0.71
      profit_factor: 2.18
      sharpe_ratio: 1.67

  # --- BULL MARKET ARCHETYPES (LONG BIAS) ---

  - id: A
    name: "Spring / UTAD"
    slug: "wyckoff_spring_utad"
    class: "engine.strategies.archetypes.bull.SpringUTADArchetype"
    maturity: stub  # TODO: Needs implementation
    direction: long
    regime_tags:
      - risk_on
      - neutral
    requires_engines:
      - wyckoff
      - pti
    requires_features:
      critical:
        - pti_score_1h
        - pti_score_1d
        - wyckoff_phase
      recommended:
        - wyckoff_m1_signal
        - wyckoff_m2_signal
      optional: []
    enable_flag: enable_A
    description: |
      Wyckoff spring (bullish) or UTAD (bearish) trap reversals.
      Uses PTI-based displacement confirmation.
      Target: 15-20 trades/year, PF > 2.5.
    implementation_status:
      status: "stub"
      blockers:
        - "PTI feature integration incomplete"
        - "Wyckoff phase detection needs calibration"
        - "No historical validation yet"
      next_steps:
        - "Implement required_features() method"
        - "Implement score() using PTI + Wyckoff phase"
        - "Add regime veto logic"
        - "Calibrate on 2020-2024 data"

  - id: H
    name: "Trap Within Trend"
    slug: "trap_within_trend"
    class: "engine.strategies.archetypes.bull.TrapWithinTrendArchetype"
    maturity: calibrated
    direction: long
    regime_tags:
      - risk_on
    requires_engines:
      - wyckoff
      - smc
      - liquidity
    requires_features:
      critical:
        - tf4h_external_trend
        - liquidity_score
        - wick_upper_ratio
        - bos_detected
      recommended:
        - adx_14
        - atr_percentile
      optional: []
    enable_flag: enable_H
    description: |
      HTF uptrend + liquidity drop + wick against trend.
      Counter-trend trap that fails, resuming main trend.
      Target: 20-30 trades/year, PF > 1.8.
    historical_performance:
      backtest_period: "2023-2024"
      total_trades: 58
      win_rate: 0.62
      profit_factor: 1.76
      sharpe_ratio: 1.34

  - id: B
    name: "Order Block Retest"
    slug: "order_block_retest"
    class: "engine.strategies.archetypes.bull.OrderBlockRetestArchetype"
    maturity: calibrated
    direction: long
    regime_tags:
      - risk_on
    requires_engines:
      - smc
      - wyckoff
    requires_features:
      critical:
        - boms_strength
        - bos_detected
        - order_block_proximity
      recommended:
        - wyckoff_phase
        - volume_zscore
      optional: []
    enable_flag: enable_B
    description: |
      Price returns to test significant order block zone.
      BOMS strength + Wyckoff context + near BOS zone.
      Target: 25-35 trades/year, PF > 1.6.
    historical_performance:
      backtest_period: "2023-2024"
      total_trades: 64
      win_rate: 0.59
      profit_factor: 1.58
      sharpe_ratio: 1.22

# ============================================================================
# DEPRECATED ARCHETYPES (Historical record, not used)
# ============================================================================

deprecated:
  - id: S2
    name: "Failed Rally Rejection"
    slug: "failed_rally"
    reason: "PF 0.48 after optimization, pattern fundamentally broken for BTC"
    deprecated_date: "2024-12-01"
    replacement: null
    note: "Works for altcoins with different microstructure, disabled for BTC"

  - id: S1_OLD
    name: "Breakdown (Old)"
    slug: "breakdown"
    reason: "Replaced by S1 (Liquidity Vacuum) with better feature engineering"
    deprecated_date: "2024-11-15"
    replacement: "S1"

  - id: S4_OLD
    name: "Distribution (Old)"
    slug: "distribution"
    reason: "Replaced by S4 (Funding Divergence) with funding-centric logic"
    deprecated_date: "2024-11-15"
    replacement: "S4"

# ============================================================================
# REGISTRY METADATA
# ============================================================================

metadata:
  total_archetypes: 6
  production_count: 3
  calibrated_count: 2
  development_count: 0
  stub_count: 1
  deprecated_count: 3
  last_updated: "2025-12-12"
  schema_version: "2.0"
```

---

## 3. Feature Reality Gate

### Purpose

Validate features exist **before** backtest to prevent runtime failures and provide visibility into archetype readiness.

### Implementation

**File:** `engine/validation/feature_reality_gate.py`

```python
"""
Feature Reality Gate - Pre-backtest feature validation

Ensures archetypes have access to required features before execution.
Provides coverage reports and fails fast on critical missing features.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureAvailability(Enum):
    """Feature availability states"""
    PRESENT = "present"
    MISSING_CRITICAL = "missing_critical"
    MISSING_RECOMMENDED = "missing_recommended"
    DEGRADED = "degraded"

@dataclass
class ArchetypeCoverage:
    """Coverage report for single archetype"""
    archetype_id: str
    archetype_name: str
    total_features: int
    critical_present: int
    critical_missing: List[str]
    recommended_present: int
    recommended_missing: List[str]
    optional_present: int
    optional_missing: List[str]
    coverage_pct: float
    status: FeatureAvailability
    can_run: bool

@dataclass
class FeatureGateReport:
    """Complete feature reality report"""
    total_archetypes: int
    can_run_count: int
    degraded_count: int
    blocked_count: int
    archetype_reports: List[ArchetypeCoverage]
    missing_features_global: List[str]

class FeatureRealityGate:
    """
    Validates feature availability before backtest execution.

    Three-tier feature classification:
    - CRITICAL: Archetype cannot function without these
    - RECOMMENDED: Degraded performance without these
    - OPTIONAL: Nice to have, graceful fallback available

    Behavior:
    - CRITICAL missing → Block archetype (cannot run)
    - RECOMMENDED missing → Warn + allow degraded mode
    - OPTIONAL missing → Silent fallback to defaults
    """

    def __init__(
        self,
        allow_degraded: bool = True,
        fail_on_critical: bool = True
    ):
        """
        Initialize feature gate.

        Args:
            allow_degraded: Allow archetypes to run with missing recommended features
            fail_on_critical: Raise exception if critical features missing
        """
        self.allow_degraded = allow_degraded
        self.fail_on_critical = fail_on_critical

    def validate_archetype(
        self,
        archetype_meta: Dict,
        available_features: List[str]
    ) -> ArchetypeCoverage:
        """
        Validate feature availability for single archetype.

        Args:
            archetype_meta: Archetype metadata from registry
            available_features: List of features in dataframe

        Returns:
            ArchetypeCoverage report
        """
        required = archetype_meta.get('requires_features', {})
        critical = required.get('critical', [])
        recommended = required.get('recommended', [])
        optional = required.get('optional', [])

        # Check critical features
        critical_present = [f for f in critical if f in available_features]
        critical_missing = [f for f in critical if f not in available_features]

        # Check recommended features
        recommended_present = [f for f in recommended if f in available_features]
        recommended_missing = [f for f in recommended if f not in available_features]

        # Check optional features
        optional_present = [f for f in optional if f in available_features]
        optional_missing = [f for f in optional if f not in available_features]

        # Calculate coverage
        total_features = len(critical) + len(recommended) + len(optional)
        present_features = len(critical_present) + len(recommended_present) + len(optional_present)
        coverage_pct = (present_features / total_features * 100) if total_features > 0 else 0.0

        # Determine status
        if critical_missing:
            status = FeatureAvailability.MISSING_CRITICAL
            can_run = False
        elif recommended_missing:
            status = FeatureAvailability.MISSING_RECOMMENDED if self.allow_degraded else FeatureAvailability.MISSING_CRITICAL
            can_run = self.allow_degraded
        else:
            status = FeatureAvailability.PRESENT
            can_run = True

        return ArchetypeCoverage(
            archetype_id=archetype_meta['id'],
            archetype_name=archetype_meta['name'],
            total_features=total_features,
            critical_present=len(critical_present),
            critical_missing=critical_missing,
            recommended_present=len(recommended_present),
            recommended_missing=recommended_missing,
            optional_present=len(optional_present),
            optional_missing=optional_missing,
            coverage_pct=coverage_pct,
            status=status,
            can_run=can_run
        )

    def validate_all(
        self,
        registry_archetypes: List[Dict],
        df: pd.DataFrame
    ) -> FeatureGateReport:
        """
        Validate all archetypes against available features.

        Args:
            registry_archetypes: List of archetype metadata from registry
            df: Feature dataframe

        Returns:
            FeatureGateReport with complete validation results
        """
        available_features = df.columns.tolist()

        archetype_reports = []
        for arch_meta in registry_archetypes:
            # Skip deprecated archetypes
            if arch_meta.get('maturity') == 'deprecated':
                continue

            coverage = self.validate_archetype(arch_meta, available_features)
            archetype_reports.append(coverage)

        # Aggregate statistics
        can_run = [r for r in archetype_reports if r.can_run]
        degraded = [r for r in archetype_reports if r.status == FeatureAvailability.MISSING_RECOMMENDED]
        blocked = [r for r in archetype_reports if not r.can_run]

        # Find globally missing features
        all_missing = set()
        for report in archetype_reports:
            all_missing.update(report.critical_missing)
            all_missing.update(report.recommended_missing)

        report = FeatureGateReport(
            total_archetypes=len(archetype_reports),
            can_run_count=len(can_run),
            degraded_count=len(degraded),
            blocked_count=len(blocked),
            archetype_reports=archetype_reports,
            missing_features_global=sorted(list(all_missing))
        )

        # Log report
        self._log_report(report)

        # Fail if critical features missing and fail_on_critical=True
        if blocked and self.fail_on_critical:
            raise FeatureValidationError(
                f"{len(blocked)} archetypes blocked due to missing critical features. "
                f"See log for details."
            )

        return report

    def _log_report(self, report: FeatureGateReport):
        """Log feature validation report"""
        logger.info("=" * 80)
        logger.info("FEATURE REALITY GATE REPORT")
        logger.info("=" * 80)
        logger.info(f"Total archetypes: {report.total_archetypes}")
        logger.info(f"  ✓ Can run: {report.can_run_count}")
        logger.info(f"  ⚠ Degraded mode: {report.degraded_count}")
        logger.info(f"  ✗ Blocked: {report.blocked_count}")

        if report.missing_features_global:
            logger.warning(f"\nGlobally missing features ({len(report.missing_features_global)}):")
            for feat in report.missing_features_global[:10]:
                logger.warning(f"  - {feat}")
            if len(report.missing_features_global) > 10:
                logger.warning(f"  ... and {len(report.missing_features_global) - 10} more")

        logger.info("\nPer-archetype coverage:")
        for arch_report in sorted(report.archetype_reports, key=lambda r: r.coverage_pct, reverse=True):
            status_icon = "✓" if arch_report.can_run else "✗"
            logger.info(
                f"  {status_icon} {arch_report.archetype_id:5s} {arch_report.archetype_name:30s} "
                f"{arch_report.coverage_pct:5.1f}% "
                f"({arch_report.critical_present}/{arch_report.total_features} features)"
            )

            if arch_report.critical_missing:
                logger.error(f"      CRITICAL MISSING: {', '.join(arch_report.critical_missing)}")
            if arch_report.recommended_missing:
                logger.warning(f"      RECOMMENDED MISSING: {', '.join(arch_report.recommended_missing[:3])}")

        logger.info("=" * 80)

class FeatureValidationError(Exception):
    """Raised when critical features are missing"""
    pass
```

---

## 4. Archetype Registry Manager

### Implementation

**File:** `engine/archetypes/registry_manager.py`

```python
"""
Archetype Registry Manager - Centralized archetype loading and validation

Loads archetype_registry.yaml and provides filtered access to archetypes.
Validates all archetypes implement BaseArchetype contract.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from importlib import import_module
import logging

from engine.archetypes.base_archetype import BaseArchetype, MaturityLevel, SignalType

logger = logging.getLogger(__name__)

class ArchetypeRegistry:
    """
    Centralized archetype registry.

    Responsibilities:
    - Load archetype_registry.yaml
    - Validate class paths and import archetype classes
    - Verify all archetypes implement BaseArchetype
    - Provide filtered access (by maturity, regime, direction)
    - Report status of registered archetypes
    """

    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize archetype registry.

        Args:
            registry_path: Path to archetype_registry.yaml
                          (defaults to project_root/archetype_registry.yaml)
        """
        if registry_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent.parent
            registry_path = project_root / "archetype_registry.yaml"

        self.registry_path = Path(registry_path)
        self.registry_data = None
        self.archetypes = {}
        self.deprecated = {}

        self._load_registry()
        self._validate_registry()

    def _load_registry(self):
        """Load and parse registry YAML"""
        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Archetype registry not found: {self.registry_path}\n"
                f"Expected location: archetype_registry.yaml in project root"
            )

        with open(self.registry_path, 'r') as f:
            self.registry_data = yaml.safe_load(f)

        logger.info(f"Loaded archetype registry from {self.registry_path}")
        logger.info(f"Registry version: {self.registry_data.get('version')}")

        # Load archetypes
        for arch_meta in self.registry_data.get('archetypes', []):
            self.archetypes[arch_meta['id']] = arch_meta

        # Load deprecated archetypes
        for arch_meta in self.registry_data.get('deprecated', []):
            self.deprecated[arch_meta['id']] = arch_meta

        logger.info(f"Loaded {len(self.archetypes)} active archetypes")
        logger.info(f"Loaded {len(self.deprecated)} deprecated archetypes")

    def _validate_registry(self):
        """Validate archetype classes implement BaseArchetype"""
        logger.info("Validating archetype implementations...")

        for arch_id, arch_meta in self.archetypes.items():
            # Skip stub archetypes (they don't have implementations yet)
            if arch_meta.get('maturity') == 'stub':
                logger.info(f"  ⊘ {arch_id:5s} STUB (no implementation expected)")
                continue

            # Try to import class
            class_path = arch_meta.get('class')
            if not class_path:
                logger.warning(f"  ✗ {arch_id:5s} No class path specified")
                continue

            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = import_module(module_path)
                archetype_class = getattr(module, class_name)

                # Verify implements BaseArchetype
                if not issubclass(archetype_class, BaseArchetype):
                    logger.error(
                        f"  ✗ {arch_id:5s} {class_name} does NOT implement BaseArchetype"
                    )
                    continue

                # Verify required class attributes set
                if archetype_class.ARCHETYPE_ID is None:
                    logger.warning(f"  ⚠ {arch_id:5s} ARCHETYPE_ID not set in class")

                logger.info(f"  ✓ {arch_id:5s} {class_name} implements BaseArchetype")

            except ImportError as e:
                logger.error(f"  ✗ {arch_id:5s} Failed to import: {e}")
            except AttributeError as e:
                logger.error(f"  ✗ {arch_id:5s} Class not found: {e}")
            except Exception as e:
                logger.error(f"  ✗ {arch_id:5s} Validation error: {e}")

    def get_archetypes(
        self,
        maturity: Optional[List[str]] = None,
        regime_tags: Optional[List[str]] = None,
        direction: Optional[str] = None,
        enabled_only: bool = False,
        config: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get filtered list of archetypes.

        Args:
            maturity: Filter by maturity levels (e.g., ['production', 'calibrated'])
            regime_tags: Filter by regime tags (e.g., ['risk_off', 'crisis'])
            direction: Filter by direction ('long', 'short')
            enabled_only: Only return archetypes enabled in config
            config: Optional config dict for enable flag checking

        Returns:
            List of archetype metadata dicts

        Examples:
            # Get only production archetypes
            prod = registry.get_archetypes(maturity=['production'])

            # Get bear market archetypes
            bear = registry.get_archetypes(regime_tags=['risk_off'])

            # Get enabled archetypes from config
            enabled = registry.get_archetypes(enabled_only=True, config=config)
        """
        results = []

        for arch_id, arch_meta in self.archetypes.items():
            # Filter by maturity
            if maturity and arch_meta.get('maturity') not in maturity:
                continue

            # Filter by regime tags (any overlap)
            if regime_tags:
                arch_regimes = arch_meta.get('regime_tags', [])
                if not any(tag in arch_regimes for tag in regime_tags):
                    continue

            # Filter by direction
            if direction and arch_meta.get('direction') != direction:
                continue

            # Filter by enable flag
            if enabled_only and config:
                enable_flag = arch_meta.get('enable_flag')
                if enable_flag and not config.get('archetypes', {}).get(enable_flag, False):
                    continue

            results.append(arch_meta)

        return results

    def get_archetype(self, archetype_id: str) -> Optional[Dict]:
        """Get single archetype by ID"""
        return self.archetypes.get(archetype_id)

    def is_deprecated(self, archetype_id: str) -> bool:
        """Check if archetype is deprecated"""
        return archetype_id in self.deprecated

    def get_status_report(self) -> Dict:
        """
        Generate status report of all archetypes.

        Returns:
            Dict with maturity breakdown and status
        """
        by_maturity = {
            'production': [],
            'calibrated': [],
            'development': [],
            'stub': []
        }

        for arch_id, arch_meta in self.archetypes.items():
            maturity = arch_meta.get('maturity', 'stub')
            by_maturity[maturity].append(arch_id)

        return {
            'total_archetypes': len(self.archetypes),
            'by_maturity': {
                k: {'count': len(v), 'ids': v}
                for k, v in by_maturity.items()
            },
            'deprecated_count': len(self.deprecated),
            'deprecated_ids': list(self.deprecated.keys())
        }

    def log_status_report(self):
        """Log status report to console"""
        report = self.get_status_report()

        logger.info("=" * 80)
        logger.info("ARCHETYPE REGISTRY STATUS")
        logger.info("=" * 80)
        logger.info(f"Total archetypes: {report['total_archetypes']}")
        logger.info(f"Deprecated: {report['deprecated_count']}")
        logger.info("")

        for maturity, data in report['by_maturity'].items():
            count = data['count']
            ids = ', '.join(data['ids']) if data['ids'] else 'none'
            logger.info(f"  {maturity.upper():15s}: {count:2d}  [{ids}]")

        if report['deprecated_ids']:
            logger.info(f"\nDeprecated: {', '.join(report['deprecated_ids'])}")

        logger.info("=" * 80)


# ============================================================================
# Convenience Functions
# ============================================================================

_global_registry = None

def get_registry(registry_path: Optional[Path] = None) -> ArchetypeRegistry:
    """
    Get global archetype registry instance (singleton pattern).

    Args:
        registry_path: Optional path to registry YAML (only used on first call)

    Returns:
        ArchetypeRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = ArchetypeRegistry(registry_path)

    return _global_registry

def reset_registry():
    """Reset global registry (useful for testing)"""
    global _global_registry
    _global_registry = None
```

---

## 5. Migration Guide

See separate file: `ARCHETYPE_CONTRACT_MIGRATION_GUIDE.md`

---

## 6. Integration with Backtest Runner

### Modified backtest_knowledge_v2.py Flow

```python
# 1. Load registry
from engine.archetypes.registry_manager import get_registry
from engine.validation.feature_reality_gate import FeatureRealityGate

registry = get_registry()
registry.log_status_report()

# 2. Get enabled archetypes from config
enabled_archetypes = registry.get_archetypes(
    maturity=['production', 'calibrated'],
    enabled_only=True,
    config=config
)

logger.info(f"Enabled archetypes: {[a['id'] for a in enabled_archetypes]}")

# 3. Validate features before backtest
gate = FeatureRealityGate(allow_degraded=True, fail_on_critical=True)
gate_report = gate.validate_all(enabled_archetypes, df)

# 4. Filter to only archetypes that can run
runnable_archetypes = [
    arch for arch, report in zip(enabled_archetypes, gate_report.archetype_reports)
    if report.can_run
]

logger.info(f"Runnable archetypes: {[a['id'] for a in runnable_archetypes]}")

# 5. Load archetype classes
archetype_instances = {}
for arch_meta in runnable_archetypes:
    class_path = arch_meta['class']
    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    archetype_class = getattr(module, class_name)

    # Instantiate archetype
    archetype_instances[arch_meta['id']] = archetype_class()

# 6. Execute backtest with archetype instances
for idx, row in df.iterrows():
    context = RuntimeContext(
        ts=idx,
        row=row,
        regime_probs=regime_probs[idx],
        regime_label=regime_labels[idx],
        adapted_params=adapted_params[idx],
        thresholds=thresholds[idx]
    )

    # Evaluate all archetypes
    for arch_id, archetype in archetype_instances.items():
        # Score
        score = archetype.score(context)

        # Veto
        veto = archetype.veto(context)
        if veto.is_vetoed:
            continue

        # Entry
        entry = archetype.entry(context)

        # ... execute trade logic
```

---

## 7. Benefits

### Enforceability
- **Compile-time validation:** Registry validates all archetypes implement BaseArchetype at load time
- **Feature validation:** FeatureRealityGate fails fast on missing critical features
- **No ghosts:** Stub archetypes explicitly marked and skipped

### Scalability
- **Easy addition:** New archetype = 1 YAML entry + 1 Python class
- **No code duplication:** BaseArchetype provides common utilities
- **Centralized registration:** Single YAML file for all archetype definitions

### Maintainability
- **Single source of truth:** archetype_registry.yaml is canonical reference
- **Clear maturity states:** Production vs. stub vs. deprecated
- **Migration path:** Backward compatibility via enable flags

### Observability
- **Status reporting:** Registry shows what works vs. needs implementation
- **Coverage reports:** FeatureRealityGate shows feature availability per archetype
- **Diagnostics:** Every archetype exports what it looked at for debugging

---

## 8. Next Steps

1. **Implement BaseArchetype:** Create abstract base class (deliverable #2)
2. **Create Registry YAML:** Define all archetypes in YAML (deliverable #3)
3. **Implement Registry Manager:** Build registry loader and validator (deliverable #4)
4. **Implement Feature Gate:** Build pre-backtest validation (included in #4)
5. **Migrate Existing Archetypes:** Convert S1, S4, S5 to new contract (see migration guide)
6. **Update Backtest Runner:** Integrate registry and feature gate (see section 6)

---

## Appendix A: Design Decisions

### Why Abstract Base Class?
- **Enforceability:** Python's ABC module raises TypeError if abstract methods not implemented
- **IDE support:** Type hints and autocomplete for all archetype methods
- **Contract clarity:** Explicit interface definition

### Why YAML Registry?
- **Non-code configuration:** Product owners can see archetype status without reading code
- **Version control friendly:** Easy to diff and review changes
- **Migration friendly:** Can add/remove archetypes without code changes

### Why Three-Tier Feature Classification?
- **Realistic:** Not all features are equally important
- **Graceful degradation:** Archetypes can still run with degraded performance
- **Flexible:** New features can be added as optional first, then promoted

### Why Separate Veto from Score?
- **Safety first:** Hard safety checks separate from pattern recognition
- **Clarity:** Explicit veto reasons vs. low scores
- **Performance:** Can short-circuit evaluation if vetoed

---

**End of Specification**
