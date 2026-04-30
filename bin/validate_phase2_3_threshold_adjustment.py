#!/usr/bin/env python3
"""
Validate Phase 2-3 Threshold Adjustments

This script validates the impact of adjusting:
- volume_fade_threshold: -1.0 → 0.0
- mtf_momentum_threshold: 0.6 (unchanged)

Compares filtering impact on Q1 2023 B archetype signals.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_feature_store():
    """Load feature store data."""
    feature_store_path = PROJECT_ROOT / "data" / "features" / "btc_usdt_1h.parquet"

    if not feature_store_path.exists():
        raise FileNotFoundError(f"Feature store not found: {feature_store_path}")

    df = pd.read_parquet(feature_store_path)
    return df


def get_b_archetype_candidates(df, start_date, end_date):
    """Get B archetype candidate signals."""
    df_period = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    conditions = (
        (df_period.get("bos_bullish", False) == True) &
        (df_period.get("boms_strength", 0.0) >= 0.30) &
        (df_period.get("wyckoff_score", 0.0) >= 0.35) &
        (df_period.get("regime_confidence", 1.0) >= 0.50) &
        (df_period.get("regime_label", "neutral") != "risk_on")
    )

    return df_period[conditions].copy()


def apply_phase2_filter(candidates, threshold):
    """Apply Phase 2 volume fade filter with given threshold."""
    vol_z = candidates.index.to_series().apply(
        lambda idx: candidates.loc[idx, 'volume_zscore'] if 'volume_zscore' in candidates.columns else
                   candidates.loc[idx, 'vol_z'] if 'vol_z' in candidates.columns else 0.0
    )

    # Crisis exception
    is_crisis = candidates.index.to_series().apply(
        lambda idx: (candidates.loc[idx, 'crisis_composite'] if 'crisis_composite' in candidates.columns else 0.0) >= 0.30 or
                   (candidates.loc[idx, 'regime_label'] if 'regime_label' in candidates.columns else 'neutral') in ['crisis', 'bear']
    )

    # Pass if vol_z <= threshold OR crisis
    passed_mask = (vol_z <= threshold) | is_crisis

    return candidates[passed_mask], candidates[~passed_mask]


def apply_phase3_filter(candidates, mtf_threshold=0.6, wyckoff_exception=0.80):
    """Apply Phase 3 MTF confirmation filter."""

    def check_mtf_confirmation(idx):
        row = candidates.loc[idx]

        # 4H BOS check
        tf4h_bos = row.get('tf4h_bos_bullish', False)

        # Momentum alignment (with fallback to fusion_momentum)
        momentum = row.get('momentum_alignment', None)
        if momentum is None:
            momentum = row.get('fusion_momentum', 0.0)

        # Wyckoff score
        wyckoff = row.get('wyckoff_score', 0.0)

        # MTF confirmation logic
        has_mtf = tf4h_bos or (momentum > mtf_threshold)
        has_wyckoff_exception = wyckoff >= wyckoff_exception

        return has_mtf or has_wyckoff_exception

    passed_mask = candidates.index.to_series().apply(check_mtf_confirmation)

    return candidates[passed_mask], candidates[~passed_mask]


def compare_configurations():
    """Compare old vs new threshold configurations."""

    print("=" * 80)
    print("PHASE 2-3 THRESHOLD ADJUSTMENT VALIDATION")
    print("=" * 80)
    print()
    print("Testing period: Q1 2023 (2023-01-01 to 2023-03-31)")
    print()

    # Load data
    print("Loading feature store...")
    df = load_feature_store()
    print(f"Loaded {len(df)} rows")
    print()

    # Get candidates
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    candidates = get_b_archetype_candidates(df, start_date, end_date)

    print(f"Baseline B archetype candidates: {len(candidates)}")
    print()

    if len(candidates) == 0:
        print("No candidates found!")
        return

    # Configuration comparison
    configs = [
        {
            "name": "OLD (Original)",
            "vol_threshold": -1.0,
            "mtf_threshold": 0.6,
            "wyckoff_exception": 0.80,
        },
        {
            "name": "NEW (Adjusted)",
            "vol_threshold": 0.0,
            "mtf_threshold": 0.6,
            "wyckoff_exception": 0.80,
        },
        {
            "name": "ALTERNATIVE (-0.5)",
            "vol_threshold": -0.5,
            "mtf_threshold": 0.6,
            "wyckoff_exception": 0.80,
        },
    ]

    results = []

    for config in configs:
        # Apply Phase 2
        p2_passed, p2_filtered = apply_phase2_filter(candidates, config["vol_threshold"])

        # Apply Phase 3 to Phase 2 passed
        combined_passed, p3_filtered = apply_phase3_filter(p2_passed, config["mtf_threshold"], config["wyckoff_exception"])

        total_filtered = len(candidates) - len(combined_passed)
        filter_rate = (total_filtered / len(candidates) * 100) if len(candidates) > 0 else 0

        results.append({
            "config": config["name"],
            "vol_threshold": config["vol_threshold"],
            "baseline": len(candidates),
            "p2_passed": len(p2_passed),
            "p2_filtered": len(p2_filtered),
            "p3_passed": len(combined_passed),
            "total_filtered": total_filtered,
            "filter_rate": filter_rate,
        })

    # Print comparison table
    print("=" * 80)
    print("FILTER IMPACT COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Configuration':<25} {'Vol Thresh':<12} {'Baseline':<10} {'P2 Pass':<10} {'P3 Pass':<10} {'Filter %':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['config']:<25} {r['vol_threshold']:<12.1f} {r['baseline']:<10} {r['p2_passed']:<10} {r['p3_passed']:<10} {r['filter_rate']:<10.1f}")

    print()

    # Detailed breakdown for NEW config
    print("=" * 80)
    print("DETAILED ANALYSIS: NEW CONFIGURATION (vol_threshold = 0.0)")
    print("=" * 80)
    print()

    new_config = configs[1]
    p2_passed, p2_filtered = apply_phase2_filter(candidates, new_config["vol_threshold"])
    combined_passed, p3_filtered = apply_phase3_filter(p2_passed, new_config["mtf_threshold"], new_config["wyckoff_exception"])

    print(f"Phase 2 (Volume Fade Filter):")
    print(f"  Threshold: vol_z <= {new_config['vol_threshold']:.1f}")
    print(f"  Passed: {len(p2_passed)}/{len(candidates)} ({len(p2_passed)/len(candidates)*100:.1f}%)")
    print(f"  Filtered: {len(p2_filtered)} ({len(p2_filtered)/len(candidates)*100:.1f}%)")
    print()

    print(f"Phase 3 (MTF Confirmation Filter):")
    print(f"  MTF Threshold: momentum > {new_config['mtf_threshold']:.1f} OR 4H BOS")
    print(f"  Wyckoff Exception: wyckoff >= {new_config['wyckoff_exception']:.2f}")
    print(f"  Passed: {len(combined_passed)}/{len(p2_passed)} ({len(combined_passed)/len(p2_passed)*100 if len(p2_passed) > 0 else 0:.1f}%)")
    print(f"  Filtered: {len(p3_filtered)} ({len(p3_filtered)/len(p2_passed)*100 if len(p2_passed) > 0 else 0:.1f}%)")
    print()

    print(f"Combined Filter:")
    print(f"  Total Passed: {len(combined_passed)}/{len(candidates)} ({len(combined_passed)/len(candidates)*100:.1f}%)")
    print(f"  Total Filtered: {len(candidates) - len(combined_passed)} ({(len(candidates) - len(combined_passed))/len(candidates)*100:.1f}%)")
    print()

    # Show which signals were filtered and why
    print("=" * 80)
    print("FILTERED SIGNALS BREAKDOWN")
    print("=" * 80)
    print()

    if len(p2_filtered) > 0:
        print(f"Filtered by Phase 2 (Volume Fade): {len(p2_filtered)} signals")
        print(f"{'Timestamp':<20} {'Vol_z':<10} {'Wyckoff':<10} {'Reason':<30}")
        print("-" * 80)

        for idx in p2_filtered.index:
            vol_z = p2_filtered.loc[idx, 'volume_zscore'] if 'volume_zscore' in p2_filtered.columns else \
                    p2_filtered.loc[idx, 'vol_z'] if 'vol_z' in p2_filtered.columns else 0.0
            wyckoff = p2_filtered.loc[idx, 'wyckoff_score'] if 'wyckoff_score' in p2_filtered.columns else 0.0

            print(f"{str(idx):<20} {vol_z:<10.2f} {wyckoff:<10.3f} {'Volume not fading enough':<30}")
        print()

    if len(p3_filtered) > 0:
        print(f"Filtered by Phase 3 (MTF Confirmation): {len(p3_filtered)} signals")
        print(f"{'Timestamp':<20} {'4H BOS':<10} {'Momentum':<12} {'Wyckoff':<10} {'Reason':<30}")
        print("-" * 80)

        for idx in p3_filtered.index:
            tf4h_bos = p3_filtered.loc[idx, 'tf4h_bos_bullish'] if 'tf4h_bos_bullish' in p3_filtered.columns else False
            momentum = p3_filtered.loc[idx, 'momentum_alignment'] if 'momentum_alignment' in p3_filtered.columns else \
                      p3_filtered.loc[idx, 'fusion_momentum'] if 'fusion_momentum' in p3_filtered.columns else 0.0
            wyckoff = p3_filtered.loc[idx, 'wyckoff_score'] if 'wyckoff_score' in p3_filtered.columns else 0.0

            print(f"{str(idx):<20} {str(tf4h_bos):<10} {momentum:<12.3f} {wyckoff:<10.3f} {'No MTF confirmation':<30}")
        print()

    # Success criteria check
    print("=" * 80)
    print("SUCCESS CRITERIA VALIDATION")
    print("=" * 80)
    print()

    target_filter_rate_min = 10.0
    target_filter_rate_max = 30.0

    new_result = results[1]  # NEW config

    checks = []

    # Check 1: Filter rate in target range
    if target_filter_rate_min <= new_result["filter_rate"] <= target_filter_rate_max:
        checks.append(("✓", f"Filter rate ({new_result['filter_rate']:.1f}%) in target range ({target_filter_rate_min:.0f}-{target_filter_rate_max:.0f}%)"))
    else:
        checks.append(("✗", f"Filter rate ({new_result['filter_rate']:.1f}%) outside target range ({target_filter_rate_min:.0f}-{target_filter_rate_max:.0f}%)"))

    # Check 2: Improvement over old config
    old_result = results[0]  # OLD config
    if new_result["p3_passed"] > old_result["p3_passed"]:
        improvement = new_result["p3_passed"] - old_result["p3_passed"]
        checks.append(("✓", f"Passes {improvement} more signals than old config ({new_result['p3_passed']} vs {old_result['p3_passed']})"))
    else:
        checks.append(("✗", f"Does not pass more signals than old config"))

    # Check 3: Not too lenient
    if new_result["filter_rate"] > 0:
        checks.append(("✓", f"Still filters some signals ({new_result['filter_rate']:.1f}%)"))
    else:
        checks.append(("⚠", "Warning: Filters no signals (may be too lenient)"))

    for status, message in checks:
        print(f"{status} {message}")

    print()

    # Recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if new_result["filter_rate"] < target_filter_rate_min:
        print("RECOMMENDATION: Volume threshold of 0.0 is good but slightly lenient.")
        print("Consider using -0.5 for more balanced filtering if needed.")
        print()
        alt_result = results[2]
        print(f"Alternative config (-0.5) would filter {alt_result['filter_rate']:.1f}% of signals.")
    elif new_result["filter_rate"] > target_filter_rate_max:
        print("RECOMMENDATION: Volume threshold of 0.0 is too strict for Q1 2023.")
        print("Consider increasing to 0.5 or higher.")
    else:
        print("RECOMMENDATION: ✓ Volume threshold of 0.0 is OPTIMAL for Q1 2023.")
        print()
        print("This configuration:")
        print(f"  - Passes {new_result['p3_passed']}/{new_result['baseline']} signals ({new_result['p3_passed']/new_result['baseline']*100:.1f}%)")
        print(f"  - Filters {new_result['filter_rate']:.1f}% (balanced filtering)")
        print(f"  - Improvement over old config: +{new_result['p3_passed'] - old_result['p3_passed']} signals")

    print()

    # Save results
    output_dir = PROJECT_ROOT / "artifacts" / "b_archetype_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"threshold_validation_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "period": {
            "start": start_date,
            "end": end_date,
        },
        "baseline_signals": len(candidates),
        "configurations": results,
        "success_criteria": {
            "target_filter_rate_min": target_filter_rate_min,
            "target_filter_rate_max": target_filter_rate_max,
            "checks": [{"status": c[0], "message": c[1]} for c in checks],
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


def main():
    """Main entry point."""
    try:
        compare_configurations()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
