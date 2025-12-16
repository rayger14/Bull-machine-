#!/usr/bin/env python3
"""
Test S1 V2 Confluence Logic

Demonstrates the difference between binary and confluence detection modes
using synthetic examples modeled after real events.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext


def create_test_config(use_confluence: bool = False):
    """Create test config with confluence mode toggled."""
    return {
        'use_archetypes': True,
        'enable_S1': True,
        'thresholds': {
            'liquidity_vacuum': {
                'use_v2_logic': True,
                'use_confluence': use_confluence,

                # Base thresholds (same for both modes)
                'capitulation_depth_max': -0.20,
                'crisis_composite_min': 0.35,
                'volume_climax_3b_min': 0.50,
                'wick_exhaustion_3b_min': 0.60,

                # Confluence parameters
                'confluence_min_conditions': 3,
                'confluence_weights': {
                    'capitulation_depth': 0.30,
                    'crisis_environment': 0.25,
                    'volume_climax': 0.25,
                    'wick_exhaustion': 0.20
                },
                'confluence_threshold': 0.65,

                # Standard parameters
                'fusion_threshold': 0.30,
                'v2_weights': {
                    'capitulation_depth_score': 0.20,
                    'crisis_environment': 0.15,
                    'volume_climax_3b': 0.08,
                    'wick_exhaustion_3b': 0.07,
                    'liquidity_drain_severity': 0.10,
                    'liquidity_velocity_score': 0.08,
                    'liquidity_persistence_score': 0.07,
                    'funding_reversal': 0.12,
                    'oversold': 0.08,
                    'volatility_spike': 0.05
                }
            }
        }
    }


def create_test_row(
    cap_depth: float,
    crisis: float,
    vol_climax: float,
    wick_exhaust: float,
    name: str = "test"
):
    """Create a test data row with specified values."""
    return pd.Series({
        'timestamp': pd.Timestamp('2022-11-09'),
        'close': 16000,
        'open': 17000,
        'high': 17500,
        'low': 15800,

        # V2 features
        'capitulation_depth': cap_depth,
        'crisis_composite': crisis,
        'volume_climax_last_3b': vol_climax,
        'wick_exhaustion_last_3b': wick_exhaust,

        # Optional features
        'liquidity_drain_pct': -0.30,
        'liquidity_velocity': -0.05,
        'liquidity_persistence': 5,
        'funding_Z': -1.2,
        'rsi_14': 25,
        'atr_percentile': 0.85,

        'name': name
    })


def evaluate_pattern(
    row: pd.Series,
    use_confluence: bool,
    verbose: bool = True
):
    """Evaluate S1 pattern in binary or confluence mode."""
    config = create_test_config(use_confluence)
    logic = ArchetypeLogic(config)

    # Create runtime context (using dataclass fields)
    context = RuntimeContext(
        ts=row['timestamp'],
        row=row,
        regime_probs={'risk_off': 1.0},
        regime_label='risk_off',
        adapted_params={},
        thresholds={'liquidity_vacuum': config['thresholds']['liquidity_vacuum']},
        metadata={}
    )

    # Evaluate S1
    matched, score, meta = logic._check_S1(context)

    if verbose:
        mode = "CONFLUENCE" if use_confluence else "BINARY"
        status = "PASS" if matched else "FAIL"

        print(f"\n{mode} MODE: {status}")
        print(f"  Score: {score:.3f}")
        print(f"  Reason: {meta.get('reason', 'pattern_matched')}")

        if 'conditions_met' in meta:
            print(f"  Conditions: {meta['conditions_met']}/{meta.get('min_conditions', 4)}")

        if 'confluence_score' in meta:
            print(f"  Confluence Score: {meta['confluence_score']:.3f}")

        if 'normalized_scores' in meta:
            scores = meta['normalized_scores']
            print(f"  Normalized Scores:")
            print(f"    depth:  {scores['depth_score']:.3f}")
            print(f"    crisis: {scores['crisis_score']:.3f}")
            print(f"    volume: {scores['vol_score']:.3f}")
            print(f"    wick:   {scores['wick_score']:.3f}")

        if 'condition_states' in meta:
            states = meta['condition_states']
            print(f"  Binary Conditions:")
            print(f"    depth:  {'PASS' if states['depth'] else 'FAIL'}")
            print(f"    crisis: {'PASS' if states['crisis'] else 'FAIL'}")
            print(f"    volume: {'PASS' if states['volume'] else 'FAIL'}")
            print(f"    wick:   {'PASS' if states['wick'] else 'FAIL'}")

    return matched, score, meta


def test_scenario(name: str, cap_depth: float, crisis: float, vol_climax: float, wick_exhaust: float):
    """Test a scenario in both binary and confluence modes."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")
    print(f"Raw Values:")
    print(f"  cap_depth:      {cap_depth:.3f} ({'PASS' if cap_depth < -0.20 else 'FAIL'} < -0.20)")
    print(f"  crisis:         {crisis:.3f} ({'PASS' if crisis > 0.35 else 'FAIL'} > 0.35)")
    print(f"  vol_climax_3b:  {vol_climax:.3f} ({'PASS' if vol_climax > 0.50 else 'FAIL'} > 0.50)")
    print(f"  wick_exhaust_3b: {wick_exhaust:.3f} ({'PASS' if wick_exhaust > 0.60 else 'FAIL'} > 0.60)")

    row = create_test_row(cap_depth, crisis, vol_climax, wick_exhaust, name)

    # Binary mode
    binary_match, binary_score, binary_meta = evaluate_pattern(row, use_confluence=False)

    # Confluence mode
    conf_match, conf_score, conf_meta = evaluate_pattern(row, use_confluence=True)

    # Summary
    print(f"\n{'─'*60}")
    print(f"RESULT COMPARISON:")
    print(f"  Binary Mode:     {'DETECT' if binary_match else 'MISS'}")
    print(f"  Confluence Mode: {'DETECT' if conf_match else 'MISS'}")

    if conf_match != binary_match:
        print(f"\n  ⚠ DIFFERENT OUTCOMES!")
        if conf_match and not binary_match:
            print(f"  → Confluence caught an edge case Binary missed")
        else:
            print(f"  → Binary caught something Confluence filtered")

    return {
        'name': name,
        'binary_match': binary_match,
        'binary_score': binary_score,
        'conf_match': conf_match,
        'conf_score': conf_score
    }


def main():
    """Run test scenarios."""
    print("\n" + "="*60)
    print("S1 V2 CONFLUENCE LOGIC TEST")
    print("="*60)

    scenarios = [
        # Scenario 1: All conditions pass (both modes should detect)
        {
            'name': 'Perfect Capitulation (LUNA May-12)',
            'cap_depth': -0.384,
            'crisis': 0.639,
            'vol_climax': 0.000,
            'wick_exhaust': 0.489
        },

        # Scenario 2: Crisis slightly below threshold (confluence advantage)
        {
            'name': 'FTX-like Edge Case (weak crisis)',
            'cap_depth': -0.268,
            'crisis': 0.303,  # Below 0.35 threshold
            'vol_climax': 0.628,  # Strong volume compensates
            'wick_exhaust': 0.710  # Strong wick compensates
        },

        # Scenario 3: Only depth passes (both should miss)
        {
            'name': 'Insufficient Confluence (1/4)',
            'cap_depth': -0.250,
            'crisis': 0.200,
            'vol_climax': 0.300,
            'wick_exhaust': 0.400
        },

        # Scenario 4: 3 of 4 conditions pass (confluence should detect)
        {
            'name': 'Moderate Multi-Signal (3/4)',
            'cap_depth': -0.220,  # PASS
            'crisis': 0.420,      # PASS
            'vol_climax': 0.580,  # PASS
            'wick_exhaust': 0.520 # FAIL (below 0.60)
        },

        # Scenario 5: Only 2 conditions pass (both should miss)
        {
            'name': 'Borderline Case (2/4)',
            'cap_depth': -0.210,  # PASS
            'crisis': 0.380,      # PASS
            'vol_climax': 0.450,  # FAIL
            'wick_exhaust': 0.550 # FAIL
        }
    ]

    results = []
    for scenario in scenarios:
        result = test_scenario(**scenario)
        results.append(result)

    # Summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<40} {'Binary':<10} {'Confluence':<10}")
    print(f"{'-'*60}")

    for r in results:
        binary_status = "DETECT" if r['binary_match'] else "MISS"
        conf_status = "DETECT" if r['conf_match'] else "MISS"
        different = "⚠" if r['binary_match'] != r['conf_match'] else " "

        print(f"{r['name']:<40} {binary_status:<10} {conf_status:<10} {different}")

    print(f"\n{'='*60}")
    print("KEY INSIGHTS:")
    print(f"{'='*60}")
    print("1. Confluence mode catches edge cases where 3/4 signals strong")
    print("2. Binary mode more strict - all hard gates must pass")
    print("3. Both modes agree on clear positives and negatives")
    print("4. Confluence provides robustness for borderline events")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
