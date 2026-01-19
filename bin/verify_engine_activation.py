#!/usr/bin/env python3
"""
Verify Domain Engine Activation in Variant Configs

Checks if domain engines will activate based on config structure.
Tests both feature_flags and engine-specific config sections.
"""
import json
import sys
from pathlib import Path


def check_engine_activation(config_path: str) -> dict:
    """
    Check which domain engines will activate based on config structure.

    Returns:
        dict with engine names and activation status
    """
    with open(config_path) as f:
        config = json.load(f)

    results = {
        'config_file': Path(config_path).name,
        'engines': {}
    }

    # Feature flags (used by domain boost/veto logic)
    feature_flags = config.get('feature_flags', {})

    # 1. Temporal Fusion Engine (needs both feature flag AND config section)
    temporal_cfg = config.get('temporal_fusion', {})
    temporal_enabled = (
        feature_flags.get('enable_temporal', False) and
        temporal_cfg.get('enabled', False)
    )
    results['engines']['Temporal Fusion'] = {
        'active': temporal_enabled,
        'feature_flag': feature_flags.get('enable_temporal', False),
        'config_section': temporal_cfg.get('enabled', False)
    }

    # 2. Wyckoff Events (needs config section with enabled=true)
    wyckoff_cfg = config.get('wyckoff_events', {})
    wyckoff_enabled = wyckoff_cfg.get('enabled', False)
    results['engines']['Wyckoff Events'] = {
        'active': wyckoff_enabled,
        'feature_flag': feature_flags.get('enable_wyckoff', False),
        'config_section': wyckoff_cfg.get('enabled', False)
    }

    # 3. SMC Engine (feature flag controls domain boost logic)
    smc_cfg = config.get('smc_engine', {})
    smc_enabled = feature_flags.get('enable_smc', False)
    results['engines']['SMC (Domain Boost)'] = {
        'active': smc_enabled,
        'feature_flag': smc_enabled,
        'config_section': smc_cfg.get('enabled', False) if smc_cfg else None
    }

    # 4. HOB Engine (feature flag controls domain boost logic)
    hob_cfg = config.get('hob_engine', {})
    hob_enabled = feature_flags.get('enable_hob', False)
    results['engines']['HOB (Domain Boost)'] = {
        'active': hob_enabled,
        'feature_flag': hob_enabled,
        'config_section': hob_cfg.get('enabled', False) if hob_cfg else None
    }

    # 5. Fusion Layer (feature flag)
    fusion_enabled = feature_flags.get('use_fusion_layer', False)
    results['engines']['Fusion Layer'] = {
        'active': fusion_enabled,
        'feature_flag': fusion_enabled,
        'config_section': None
    }

    # 6. Macro Regime (feature flag)
    macro_enabled = feature_flags.get('enable_macro', False)
    results['engines']['Macro Regime'] = {
        'active': macro_enabled,
        'feature_flag': macro_enabled,
        'config_section': None
    }

    return results


def print_results(results: dict):
    """Print engine activation results in readable format."""
    config_name = results['config_file']
    print(f"\n{'='*80}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*80}")

    active_count = sum(1 for e in results['engines'].values() if e['active'])
    total_count = len(results['engines'])

    for engine_name, status in results['engines'].items():
        active = status['active']
        flag = status['feature_flag']
        cfg = status['config_section']

        symbol = "✅" if active else "❌"
        print(f"\n{symbol} {engine_name}: {'ACTIVE' if active else 'INACTIVE'}")
        print(f"   Feature Flag: {flag}")
        if cfg is not None:
            print(f"   Config Section: {cfg}")

    print(f"\n{'-'*80}")
    print(f"SUMMARY: {active_count}/{total_count} engines will activate")
    print(f"{'='*80}\n")

    return active_count, total_count


def main():
    """Verify engine activation for all variant configs."""
    base_path = Path(__file__).parent.parent / 'configs' / 'variants'

    configs = [
        base_path / 's1_full.json',
        base_path / 's4_full.json',
        base_path / 's5_full.json'
    ]

    print("\n" + "="*80)
    print("DOMAIN ENGINE ACTIVATION VERIFICATION")
    print("="*80)

    all_results = []
    for config_path in configs:
        if not config_path.exists():
            print(f"\n❌ ERROR: Config not found: {config_path}")
            continue

        results = check_engine_activation(str(config_path))
        active, total = print_results(results)
        all_results.append((config_path.name, active, total))

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    all_pass = True
    for config_name, active, total in all_results:
        status = "✅ PASS" if active == total else "❌ FAIL"
        print(f"{status} {config_name}: {active}/{total} engines active")
        if active != total:
            all_pass = False

    print("="*80)

    if all_pass:
        print("\n✅ SUCCESS: All domain engines will activate in FULL variants")
        return 0
    else:
        print("\n❌ FAILURE: Some engines will not activate")
        return 1


if __name__ == '__main__':
    sys.exit(main())
