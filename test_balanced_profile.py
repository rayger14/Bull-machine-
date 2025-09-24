#!/usr/bin/env python3
"""
Quick test of Balanced profile with relaxed thresholds
"""

import json
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bull_machine.scoring.fusion import FusionEngineV141


def test_balanced_config():
    """Test that Balanced profile loads with correct thresholds."""

    # Load Balanced profile
    balanced_path = "configs/v141/profile_balanced.json"
    with open(balanced_path) as f:
        balanced_config = json.load(f)

    # Load base system config
    system_path = "configs/v141/system_config.json"
    with open(system_path) as f:
        system_config = json.load(f)

    # Merge configs (simplified)
    merged_config = {**system_config}

    # Apply overrides from Balanced profile
    if "signals" in balanced_config:
        merged_config.setdefault("signals", {})
        merged_config["signals"].update(balanced_config["signals"])

    if "quality_floors" in balanced_config:
        merged_config.setdefault("quality_floors", {})
        merged_config["quality_floors"].update(balanced_config["quality_floors"])

    print("🎯 Balanced Profile Configuration:")
    print(f"   Enter Threshold: {merged_config['signals']['enter_threshold']}")
    print(f"   Wyckoff Floor: {merged_config['quality_floors']['wyckoff']}")
    print(f"   Liquidity Floor: {merged_config['quality_floors']['liquidity']}")

    # Test fusion engine with new thresholds
    engine = FusionEngineV141(merged_config)

    # Mock layer scores that should trigger with relaxed thresholds
    mock_scores = {
        "wyckoff": 0.70,      # Above new 0.37 floor
        "liquidity": 0.75,    # Above new 0.32 floor, triggers MTF override at 0.70
        "structure": 0.65,
        "momentum": 0.60,
        "volume": 0.65,
        "context": 0.50,
        "mtf": 0.65          # Below 0.70 MTF threshold, but liquidity should override
    }

    result = engine.fuse_scores(mock_scores)

    print(f"\n🧪 Test Fusion Result:")
    print(f"   Weighted Score: {result['weighted_score']:.3f}")
    print(f"   Enter Decision: {'✅ ENTER' if result['weighted_score'] >= merged_config['signals']['enter_threshold'] else '❌ SKIP'}")
    print(f"   Global Veto: {result['global_veto']}")
    print(f"   MTF Gate: {result['mtf_gate']}")

    should_enter = engine.should_enter(result)
    print(f"   Final Decision: {'✅ TRADE' if should_enter else '❌ NO TRADE'}")

    return result


if __name__ == "__main__":
    test_balanced_config()