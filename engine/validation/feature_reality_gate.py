"""
Feature Reality Gate - Pre-backtest feature validation

Ensures archetypes have access to required features before execution.
Provides coverage reports and fails fast on critical missing features.

Three-tier feature classification:
- CRITICAL: Archetype cannot function without these (block execution)
- RECOMMENDED: Degraded performance without these (warn + allow)
- OPTIONAL: Nice to have, graceful fallback available (silent)

Author: System Architect (Claude Code)
Date: 2025-12-12
Version: 2.0
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

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

    def __repr__(self):
        status_icon = "✓" if self.can_run else "✗"
        return (
            f"{status_icon} {self.archetype_id:5s} {self.archetype_name:30s} "
            f"{self.coverage_pct:5.1f}% "
            f"({self.critical_present}/{self.total_features} features)"
        )


@dataclass
class FeatureGateReport:
    """Complete feature reality report"""
    total_archetypes: int
    can_run_count: int
    degraded_count: int
    blocked_count: int
    archetype_reports: List[ArchetypeCoverage]
    missing_features_global: List[str]

    def get_runnable_archetypes(self) -> List[str]:
        """Get list of archetype IDs that can run"""
        return [r.archetype_id for r in self.archetype_reports if r.can_run]

    def get_blocked_archetypes(self) -> List[str]:
        """Get list of archetype IDs that are blocked"""
        return [r.archetype_id for r in self.archetype_reports if not r.can_run]


# ============================================================================
# Feature Reality Gate
# ============================================================================

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

    Usage:
        gate = FeatureRealityGate(allow_degraded=True, fail_on_critical=True)
        report = gate.validate_all(registry_archetypes, df)

        if report.blocked_count > 0:
            logger.warning(f"Blocked archetypes: {report.get_blocked_archetypes()}")

        # Get runnable archetypes
        runnable = [
            arch for arch in registry_archetypes
            if arch['id'] in report.get_runnable_archetypes()
        ]
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

        Raises:
            FeatureValidationError: If critical features missing and fail_on_critical=True
        """
        available_features = df.columns.tolist()

        archetype_reports = []
        for arch_meta in registry_archetypes:
            # Skip deprecated archetypes
            if arch_meta.get('maturity') == 'deprecated':
                continue

            # Skip stub archetypes (they don't have feature requirements yet)
            if arch_meta.get('maturity') == 'stub':
                logger.debug(f"Skipping stub archetype {arch_meta['id']} (no feature validation needed)")
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
                f"Blocked: {[r.archetype_id for r in blocked]}\n"
                f"See log for details."
            )

        return report

    def _log_report(self, report: FeatureGateReport):
        """Log feature validation report"""
        logger.info("=" * 80)
        logger.info("FEATURE REALITY GATE REPORT")
        logger.info("=" * 80)
        logger.info(f"Total archetypes validated: {report.total_archetypes}")
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

    def get_missing_features_by_archetype(
        self,
        report: FeatureGateReport
    ) -> Dict[str, List[str]]:
        """
        Get missing features grouped by archetype.

        Args:
            report: FeatureGateReport from validate_all()

        Returns:
            Dict mapping archetype_id -> list of missing critical features
        """
        missing_by_archetype = {}

        for arch_report in report.archetype_reports:
            if arch_report.critical_missing:
                missing_by_archetype[arch_report.archetype_id] = arch_report.critical_missing

        return missing_by_archetype

    def suggest_feature_priorities(
        self,
        report: FeatureGateReport
    ) -> List[Tuple[str, int]]:
        """
        Suggest which missing features to add first (by impact).

        Args:
            report: FeatureGateReport from validate_all()

        Returns:
            List of (feature_name, num_archetypes_blocked) sorted by impact

        Example:
            priorities = gate.suggest_feature_priorities(report)
            # [('liquidity_drain_pct', 3), ('volume_zscore', 2), ...]

            print("Missing features blocking most archetypes:")
            for feature, count in priorities[:5]:
                print(f"  {feature}: blocks {count} archetypes")
        """
        feature_impact = {}

        for arch_report in report.archetype_reports:
            for feature in arch_report.critical_missing:
                feature_impact[feature] = feature_impact.get(feature, 0) + 1

        # Sort by impact (descending)
        priorities = sorted(
            feature_impact.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return priorities


class FeatureValidationError(Exception):
    """Raised when critical features are missing"""
    pass


# ============================================================================
# CLI Tool for Feature Validation
# ============================================================================

if __name__ == '__main__':
    """
    CLI tool for validating feature availability.

    Usage:
        python -m engine.validation.feature_reality_gate \\
            --registry archetype_registry.yaml \\
            --features data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet
    """
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(description='Validate feature availability for archetypes')
    parser.add_argument(
        '--registry',
        type=str,
        default='archetype_registry.yaml',
        help='Path to archetype registry YAML'
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to feature store parquet file'
    )
    parser.add_argument(
        '--allow-degraded',
        action='store_true',
        default=True,
        help='Allow degraded mode (missing recommended features)'
    )
    parser.add_argument(
        '--fail-on-critical',
        action='store_true',
        default=False,
        help='Fail if critical features missing'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("FEATURE REALITY GATE VALIDATOR")
    print("=" * 80)
    print()

    try:
        # Load registry
        from engine.archetypes.registry_manager import get_registry

        print(f"Loading registry: {args.registry}")
        registry = get_registry(Path(args.registry))

        # Load features
        print(f"Loading features: {args.features}")
        df = pd.read_parquet(args.features)
        print(f"  Features: {len(df.columns)} columns, {len(df)} rows")
        print()

        # Get archetypes
        archetypes = registry.get_archetypes()
        print(f"Validating {len(archetypes)} archetypes")
        print()

        # Validate
        gate = FeatureRealityGate(
            allow_degraded=args.allow_degraded,
            fail_on_critical=args.fail_on_critical
        )

        report = gate.validate_all(archetypes, df)

        # Show priorities
        print("\n" + "=" * 80)
        print("FEATURE PRIORITY RECOMMENDATIONS")
        print("=" * 80)

        priorities = gate.suggest_feature_priorities(report)
        if priorities:
            print("\nMissing features blocking most archetypes:")
            for feature, count in priorities[:10]:
                print(f"  {feature:40s} blocks {count:2d} archetype(s)")
        else:
            print("\nNo missing critical features - all archetypes can run!")

        print("\n" + "=" * 80)

        # Exit status
        if report.blocked_count > 0:
            print(f"\nWARNING: {report.blocked_count} archetypes blocked")
            if args.fail_on_critical:
                sys.exit(1)
        else:
            print("\nSUCCESS: All archetypes can run")
            sys.exit(0)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
