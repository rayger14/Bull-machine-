"""
Validation Module - Pre-execution validation and reality checks

Components:
- FeatureRealityGate: Validates features exist before backtest
- ConfigValidator: Validates config structure and parameters
- DataQualityGate: Validates data quality (coverage, integrity)
"""

from engine.validation.feature_reality_gate import (
    FeatureRealityGate,
    FeatureAvailability,
    ArchetypeCoverage,
    FeatureGateReport,
    FeatureValidationError
)

__all__ = [
    'FeatureRealityGate',
    'FeatureAvailability',
    'ArchetypeCoverage',
    'FeatureGateReport',
    'FeatureValidationError'
]
