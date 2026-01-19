"""Bull Machine Fusion Engine - Domain Integration"""
from .domain_fusion import analyze_fusion, FusionSignal
from .knowledge_hooks import (
    apply_knowledge_hooks,
    assert_feature_contract,
    FusionDelta
)

__all__ = [
    'analyze_fusion',
    'FusionSignal',
    'apply_knowledge_hooks',
    'assert_feature_contract',
    'FusionDelta'
]
