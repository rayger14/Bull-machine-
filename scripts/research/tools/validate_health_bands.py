#!/usr/bin/env python3
"""
Bull Machine Health Band Validator
Validates long-run results against institutional health thresholds
"""

import argparse
import json
import sys
from typing import Dict, Any, List, Tuple

def load_manifest(filepath: str) -> Dict[str, Any]:
    """Load merged manifest file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading manifest: {e}")
        return {}

def validate_profit_factor(pf: float, min_pf: float) -> Tuple[bool, str]:
    """Validate profit factor against minimum threshold."""
    if pf >= min_pf:
        return True, f"✅ Profit Factor: {pf:.2f} >= {min_pf:.2f}"
    else:
        return False, f"❌ Profit Factor: {pf:.2f} < {min_pf:.2f} (FAIL)"

def validate_drawdown(dd: float, max_dd_pct: float) -> Tuple[bool, str]:
    """Validate maximum drawdown against threshold."""
    if dd <= max_dd_pct:
        return True, f"✅ Max Drawdown: {dd:.2f}% <= {max_dd_pct:.2f}%"
    else:
        return False, f"❌ Max Drawdown: {dd:.2f}% > {max_dd_pct:.2f}% (FAIL)"

def validate_macro_veto_rate(engine_util: Dict[str, int], total_trades: int, min_rate: float, max_rate: float) -> Tuple[bool, str]:
    """Validate macro veto rate within acceptable range."""
    macro_vetos = engine_util.get('macro_veto', 0)
    total_signals = sum(engine_util.values())

    if total_signals == 0:
        return False, "❌ Macro Veto Rate: No signals generated"

    veto_rate = macro_vetos / total_signals

    if min_rate <= veto_rate <= max_rate:
        return True, f"✅ Macro Veto Rate: {veto_rate:.1%} ({min_rate:.1%}-{max_rate:.1%})"
    else:
        return False, f"❌ Macro Veto Rate: {veto_rate:.1%} outside range ({min_rate:.1%}-{max_rate:.1%}) (FAIL)"

def validate_smc_2hit_rate(engine_util: Dict[str, int], min_rate: float) -> Tuple[bool, str]:
    """Validate SMC 2-hit rate (approximated by SMC utilization)."""
    smc_signals = engine_util.get('smc', 0)
    total_signals = sum(engine_util.values())

    if total_signals == 0:
        return False, "❌ SMC 2-Hit Rate: No signals generated"

    smc_rate = smc_signals / total_signals

    if smc_rate >= min_rate:
        return True, f"✅ SMC 2-Hit Rate: {smc_rate:.1%} >= {min_rate:.1%}"
    else:
        return False, f"❌ SMC 2-Hit Rate: {smc_rate:.1%} < {min_rate:.1%} (FAIL)"

def validate_hob_relevance(engine_util: Dict[str, int], max_rate: float) -> Tuple[bool, str]:
    """Validate HOB relevance doesn't exceed maximum."""
    hob_signals = engine_util.get('hob', 0)
    total_signals = sum(engine_util.values())

    if total_signals == 0:
        return True, "✅ HOB Relevance: No signals to validate"

    hob_rate = hob_signals / total_signals

    if hob_rate <= max_rate:
        return True, f"✅ HOB Relevance: {hob_rate:.1%} <= {max_rate:.1%}"
    else:
        return False, f"❌ HOB Relevance: {hob_rate:.1%} > {max_rate:.1%} (FAIL)"

def validate_win_rate(win_rate: float, min_wr: float) -> Tuple[bool, str]:
    """Validate win rate meets minimum threshold."""
    if win_rate >= min_wr:
        return True, f"✅ Win Rate: {win_rate:.1f}% >= {min_wr:.1f}%"
    else:
        return False, f"❌ Win Rate: {win_rate:.1f}% < {min_wr:.1f}% (FAIL)"

def validate_trade_frequency(total_trades: int, period_days: int, min_monthly: int, max_monthly: int) -> Tuple[bool, str]:
    """Validate trade frequency within acceptable range."""
    months = period_days / 30.44  # Average days per month
    monthly_rate = total_trades / months

    if min_monthly <= monthly_rate <= max_monthly:
        return True, f"✅ Trade Frequency: {monthly_rate:.1f} trades/month ({min_monthly}-{max_monthly})"
    else:
        return False, f"❌ Trade Frequency: {monthly_rate:.1f} trades/month outside range ({min_monthly}-{max_monthly}) (FAIL)"

def calculate_period_days(period_start: str, period_end: str) -> int:
    """Calculate number of days in period."""
    try:
        from datetime import datetime
        start = datetime.fromisoformat(period_start.replace('Z', '+00:00'))
        end = datetime.fromisoformat(period_end.replace('Z', '+00:00'))
        return (end - start).days
    except:
        return 365  # Default to 1 year

def main():
    parser = argparse.ArgumentParser(description='Validate Bull Machine health bands')
    parser.add_argument('--manifest', required=True, help='Merged manifest JSON file')
    parser.add_argument('--pf_min', type=float, default=1.30, help='Minimum profit factor')
    parser.add_argument('--dd_max_pct', type=float, default=8.0, help='Maximum drawdown percentage')
    parser.add_argument('--macro_veto_min', type=float, default=0.05, help='Minimum macro veto rate')
    parser.add_argument('--macro_veto_max', type=float, default=0.15, help='Maximum macro veto rate')
    parser.add_argument('--smc_2hit_min', type=float, default=0.30, help='Minimum SMC 2-hit rate')
    parser.add_argument('--hob_rel_max', type=float, default=0.30, help='Maximum HOB relevance')
    parser.add_argument('--win_rate_min', type=float, default=55.0, help='Minimum win rate percentage')
    parser.add_argument('--trades_monthly_min', type=int, default=8, help='Minimum trades per month')
    parser.add_argument('--trades_monthly_max', type=int, default=25, help='Maximum trades per month')
    parser.add_argument('--fail_fast', action='store_true', help='Exit on first failure')

    args = parser.parse_args()

    print("🏥 BULL MACHINE HEALTH BAND VALIDATOR")
    print("=" * 50)
    print(f"Manifest: {args.manifest}")

    # Load manifest
    manifest = load_manifest(args.manifest)
    if not manifest:
        print("❌ Failed to load manifest")
        sys.exit(1)

    summary = manifest.get('summary', {})
    performance = manifest.get('performance', {})
    engine_util = manifest.get('engine_utilization', {})

    print(f"\n📊 VALIDATING: {summary.get('asset', 'Unknown')} ({summary.get('period_start', 'Unknown')} → {summary.get('period_end', 'Unknown')})")

    # Extract metrics
    total_trades = performance.get('total_trades', 0)
    profit_factor = performance.get('profit_factor', 0)
    max_drawdown = performance.get('max_drawdown', 0)
    win_rate = performance.get('win_rate', 0)

    # Calculate period
    period_days = calculate_period_days(
        summary.get('period_start', '2024-01-01'),
        summary.get('period_end', '2024-12-31')
    )

    # Run validations
    validations = []
    failed_count = 0

    print("\n🔍 HEALTH BAND VALIDATIONS:")
    print("-" * 50)

    # Core performance validations
    is_valid, msg = validate_profit_factor(profit_factor, args.pf_min)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    is_valid, msg = validate_drawdown(max_drawdown, args.dd_max_pct)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    is_valid, msg = validate_win_rate(win_rate, args.win_rate_min)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    is_valid, msg = validate_trade_frequency(total_trades, period_days, args.trades_monthly_min, args.trades_monthly_max)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    # Engine utilization validations
    print("\n🔧 ENGINE HEALTH VALIDATIONS:")
    print("-" * 50)

    is_valid, msg = validate_macro_veto_rate(engine_util, total_trades, args.macro_veto_min, args.macro_veto_max)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    is_valid, msg = validate_smc_2hit_rate(engine_util, args.smc_2hit_min)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    is_valid, msg = validate_hob_relevance(engine_util, args.hob_rel_max)
    validations.append((is_valid, msg))
    if not is_valid:
        failed_count += 1
    print(msg)

    # Summary
    total_validations = len(validations)
    passed_count = total_validations - failed_count

    print(f"\n📋 VALIDATION SUMMARY:")
    print("-" * 50)
    print(f"Total Validations: {total_validations}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if failed_count == 0:
        print("\n🎉 ALL HEALTH BANDS PASSED!")
        print("✅ Bull Machine v1.7.1 meets institutional standards")
        print("✅ System approved for production deployment")
        exit_code = 0
    else:
        print(f"\n⚠️  {failed_count} HEALTH BAND VIOLATIONS DETECTED")
        print("❌ System requires refinement before production")

        # Provide specific recommendations
        print("\n🔧 RECOMMENDED ACTIONS:")
        if profit_factor < args.pf_min:
            print("   • Improve signal quality or reduce false positives")
        if max_drawdown > args.dd_max_pct:
            print("   • Strengthen risk management and position sizing")
        if win_rate < args.win_rate_min:
            print("   • Enhance entry timing and confluence requirements")

        exit_code = 1

    # Additional insights
    print(f"\n📈 PERFORMANCE INSIGHTS:")
    print("-" * 50)
    print(f"Period analyzed: {period_days} days ({period_days/365:.1f} years)")
    print(f"Total return: {performance.get('total_return', 0):.2f}%")
    print(f"Sharpe ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"Average R-multiple: {performance.get('avg_r_multiple', 0):.2f}")

    if args.fail_fast and failed_count > 0:
        print("\n🚨 FAIL_FAST mode: Exiting due to validation failures")
        sys.exit(exit_code)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()