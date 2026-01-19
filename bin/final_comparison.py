#!/usr/bin/env python3
"""
Final Comparison: Archetypes vs Baselines
Step 9 of validation process - make deployment decision
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_baseline_results():
    """Load baseline results from quant suite."""
    baseline_file = 'results/quant_suite/quant_suite_results_20251207_184821.csv'
    df = pd.read_csv(baseline_file)

    # Filter for test period (2H2023)
    test_df = df[df['period'] == 'test'].copy()

    # Rename columns to match our comparison format
    test_df = test_df.rename(columns={
        'model_name': 'System',
        'profit_factor': 'Test_PF',
        'win_rate': 'Test_WR',
        'num_trades': 'Test_Trades',
        'sharpe_ratio': 'Test_Sharpe',
        'total_return_pct': 'Test_Return_Pct'
    })

    # Filter out broken/reference baselines
    test_df = test_df[~test_df['System'].isin(['Baseline0_BuyAndHold', 'Baseline5_Cash'])]

    # Add metadata
    test_df['Type'] = 'Baseline'
    test_df['Complexity'] = 'Simple'

    return test_df[['System', 'Type', 'Test_PF', 'Test_WR', 'Test_Trades', 'Test_Sharpe', 'Complexity']]


def load_archetype_results():
    """Load archetype results from unified comparison."""
    comparison_file = 'results/unified_comparison_table.csv'
    df = pd.read_csv(comparison_file)

    # Filter for archetypes only
    arch_df = df[df['Type'] == 'Archetype'].copy()

    # Extract test period data
    arch_df = arch_df.rename(columns={
        'Model': 'System',
        'Test_Sharpe': 'Test_Sharpe'
    })

    # Add complexity
    arch_df['Complexity'] = 'Complex'

    return arch_df[['System', 'Type', 'Test_PF', 'Test_Trades', 'Test_Sharpe', 'Complexity']]


def create_unified_comparison():
    """Create comprehensive comparison of all systems."""

    print("\n" + "="*100)
    print("FINAL SYSTEM COMPARISON - ARCHETYPES VS BASELINES")
    print("="*100 + "\n")

    # Load both datasets
    baselines = load_baseline_results()
    archetypes = load_archetype_results()

    print(f"Loaded {len(baselines)} baselines, {len(archetypes)} archetypes\n")

    # Combine
    df = pd.concat([baselines, archetypes], ignore_index=True)

    # Sort by Test PF
    df = df.sort_values('Test_PF', ascending=False).reset_index(drop=True)

    # Calculate ranks and gaps
    df['Rank'] = range(1, len(df) + 1)
    best_pf = df['Test_PF'].max()
    df['Gap_vs_Best'] = df['Test_PF'] - best_pf
    df['Gap_Pct'] = (df['Gap_vs_Best'] / best_pf * 100).round(1)

    # Save
    output_file = 'results/validation/final_comparison.csv'
    Path('results/validation').mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print("FULL SYSTEM RANKINGS (Test Period 2H2023):")
    print("-" * 100)
    display_df = df[['Rank', 'System', 'Type', 'Test_PF', 'Gap_Pct', 'Test_Trades', 'Test_Sharpe']].copy()
    print(display_df.to_string(index=False))
    print(f"\nSaved to: {output_file}\n")

    return df


def determine_deployment_scenario(df):
    """Determine deployment scenario based on results."""

    print("\n" + "="*100)
    print("DEPLOYMENT DECISION MATRIX")
    print("="*100 + "\n")

    # Get best of each type
    baselines = df[df['Type'] == 'Baseline'].copy()
    archetypes = df[df['Type'] == 'Archetype'].copy()

    if len(baselines) == 0 or len(archetypes) == 0:
        print("ERROR: Missing baseline or archetype data")
        return None

    best_baseline = baselines.iloc[0]
    best_archetype = archetypes.iloc[0]

    best_baseline_pf = best_baseline['Test_PF']
    best_archetype_pf = best_archetype['Test_PF']

    print(f"Best Baseline:  {best_baseline['System']:35s} PF {best_baseline_pf:.2f}")
    print(f"Best Archetype: {best_archetype['System']:35s} PF {best_archetype_pf:.2f}")
    print(f"Gap: {best_archetype_pf - best_baseline_pf:+.2f} ({(best_archetype_pf / best_baseline_pf - 1) * 100:+.1f}%)\n")

    # Scenario determination
    if best_archetype_pf > best_baseline_pf + 0.1:
        scenario = "A"
        recommendation = "Deploy archetypes as MAIN engine, baselines as diversifiers"
        allocation = "70% Archetypes, 30% Baselines"
        reason = "Archetypes clearly outperform baselines"
    elif best_archetype_pf >= best_baseline_pf * 0.9:  # Within 10%
        scenario = "B"
        recommendation = "Deploy HYBRID (archetypes competitive with baselines)"
        allocation = "50% Archetypes, 50% Baselines"
        reason = "Archetypes competitive, leverage diversification"
    else:
        scenario = "C"
        recommendation = "Deploy BASELINES ONLY (archetypes underperform)"
        allocation = "0% Archetypes, 100% Baselines"
        reason = "Baselines significantly outperform archetypes"

    print(f"SCENARIO: {scenario}")
    print(f"REASON: {reason}")
    print(f"RECOMMENDATION: {recommendation}")
    print(f"CAPITAL ALLOCATION: {allocation}\n")

    # Check individual archetype targets
    print("="*100)
    print("INDIVIDUAL ARCHETYPE ASSESSMENT")
    print("="*100 + "\n")

    # Target PFs from requirements
    targets = {
        'S4_FundingDivergence': 2.2,
        'S1_LiquidityVacuum': 1.8,
        'S5_LongSqueeze': 1.6
    }

    deploy_archetypes = []
    for archetype, target in targets.items():
        arch_row = df[df['System'] == archetype]
        if len(arch_row) > 0:
            pf = arch_row.iloc[0]['Test_PF']
            status = "PASS" if pf >= target else "FAIL"
            emoji = "✓" if pf >= target else "✗"
            print(f"{emoji} {archetype:30s} PF {pf:.2f} (Target: {target:.2f}) [{status}]")
            if pf >= target:
                deploy_archetypes.append(archetype)
        else:
            print(f"✗ {archetype:30s} NO DATA [FAIL]")

    if deploy_archetypes:
        print(f"\nDeploy-Ready Archetypes: {', '.join(deploy_archetypes)}")
    else:
        print(f"\nDeploy-Ready Archetypes: NONE")

    return {
        'scenario': scenario,
        'reason': reason,
        'recommendation': recommendation,
        'allocation': allocation,
        'deploy_archetypes': deploy_archetypes,
        'best_baseline': best_baseline['System'],
        'best_baseline_pf': best_baseline_pf,
        'best_archetype': best_archetype['System'],
        'best_archetype_pf': best_archetype_pf,
    }


def create_deployment_report(decision, df):
    """Generate final deployment decision report."""

    report = f"""# Final Deployment Decision
## Archetype Validation Complete

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** VALIDATION COMPLETE

---

## EXECUTIVE SUMMARY

**Scenario:** {decision['scenario']}
**Reason:** {decision['reason']}
**Recommendation:** {decision['recommendation']}
**Capital Allocation:** {decision['allocation']}

**Best Baseline:** {decision['best_baseline']} (PF {decision['best_baseline_pf']:.2f})
**Best Archetype:** {decision['best_archetype']} (PF {decision['best_archetype_pf']:.2f})
**Performance Gap:** {decision['best_archetype_pf'] - decision['best_baseline_pf']:+.2f} ({(decision['best_archetype_pf'] / decision['best_baseline_pf'] - 1) * 100:+.1f}%)

---

## FULL SYSTEM RANKINGS (Test Period 2H2023)

"""

    # Add rankings table
    display_cols = ['Rank', 'System', 'Type', 'Test_PF', 'Gap_Pct', 'Test_Trades', 'Test_Sharpe']

    # Format as simple table (without tabulate dependency)
    report += "| " + " | ".join(display_cols) + " |\n"
    report += "|" + "|".join(["-" * (len(col) + 2) for col in display_cols]) + "|\n"
    for _, row in df[display_cols].iterrows():
        report += "| " + " | ".join([str(row[col]) for col in display_cols]) + " |\n"
    report += "\n"

    report += "---\n\n## DEPLOYMENT PLAN\n\n"

    if decision['scenario'] == 'A':
        expected_pf = decision['best_archetype_pf'] * 0.7 + decision['best_baseline_pf'] * 0.3
        report += f"""
### SCENARIO A: Archetypes Win

**Deploy Archetypes:** {', '.join(decision['deploy_archetypes']) if decision['deploy_archetypes'] else 'None'}

**Phase 1 (Week 1-2): Paper Trading**
- Deploy archetypes in paper trading mode
- Monitor vs baseline performance
- Validate live vs backtest alignment
- Track execution quality

**Phase 2 (Week 3-4): Live Small**
- 10% capital to archetypes
- 5% capital to baselines (safety net)
- Monitor for 2 weeks
- Maximum 1% risk per trade

**Phase 3 (Week 5-8): Scale Up**
- Scale to 70% archetypes, 30% baselines
- Monthly rebalancing based on performance
- Continue monitoring execution quality

**Expected Portfolio PF:** {expected_pf:.2f}
**Risk:** Medium (archetypes more complex than baselines)
"""

    elif decision['scenario'] == 'B':
        expected_pf = (decision['best_archetype_pf'] + decision['best_baseline_pf']) / 2
        report += f"""
### SCENARIO B: Hybrid Deployment

**Deploy Both:** Archetypes and Baselines

**Phase 1 (Week 1-2): Paper Trading**
- Deploy both archetypes and baselines
- Monitor correlation and diversification
- Validate both systems independently

**Phase 2 (Week 3-4): Live Small**
- 10% capital split 50/50 (5% each)
- Validate both systems in live environment
- Monitor correlation benefits

**Phase 3 (Week 5-8): Balanced Portfolio**
- 50% archetypes, 50% baselines
- Leverage diversification benefits
- Quarterly rebalancing

**Expected Portfolio PF:** {expected_pf:.2f}
**Diversification Benefit:** Lower variance, smoother equity curve
**Risk:** Lower (systems complement each other)
"""

    else:  # Scenario C
        report += f"""
### SCENARIO C: Baselines Only

**Deploy:** Baselines only (archetypes did not meet targets)

**Why Archetypes Failed:**
1. Performance below targets on test period
2. Complexity not justified by returns
3. Simpler baselines achieve better risk-adjusted returns

**Archetype Post-Mortem Actions:**
1. Review why archetypes underperformed targets
2. Consider: Temporal domain missing, regime classification accuracy
3. Analyze: Are features predictive or overfit?
4. Options:
   - Rework with enhanced features (4+ weeks)
   - Archive and focus on baseline optimization
   - Hybrid approach: Use archetype signals as filters for baselines

**Baseline Deployment:**
- Week 1-2: Paper trading best baseline
- Week 3-4: Live 10% capital
- Week 5-8: Scale to 100%

**Expected Portfolio PF:** {decision['best_baseline_pf']:.2f}
**Risk:** Low (simple, proven strategies)
"""

    report += f"""
---

## STATISTICAL ANALYSIS

### Performance Metrics Comparison

| Metric | Best Baseline | Best Archetype | Winner |
|--------|--------------|----------------|--------|
| Profit Factor | {decision['best_baseline_pf']:.2f} | {decision['best_archetype_pf']:.2f} | {'Baseline' if decision['best_baseline_pf'] > decision['best_archetype_pf'] else 'Archetype'} |
| Relative Performance | 100% | {(decision['best_archetype_pf'] / decision['best_baseline_pf'] * 100):.1f}% | - |

### Key Insights

1. **Simplicity vs Complexity:** {'Complexity justified' if decision['best_archetype_pf'] > decision['best_baseline_pf'] else 'Simplicity wins'}
2. **Risk-Adjusted Returns:** See Sharpe ratios in rankings table above
3. **Trade Frequency:** Higher trade count may indicate more opportunities or overfitting

---

## NEXT ACTIONS

1. **Review this report** with team
2. **Approve deployment scenario** (A/B/C)
3. **Execute deployment plan** following phased approach
4. **Monitor performance** vs expectations
5. **Setup alerting** for performance degradation

---

## RISK WARNINGS

- All backtested performance is historical and may not repeat
- Live execution will differ from backtest due to slippage, fees, latency
- Start small (10% capital max) during validation phase
- Monitor for at least 2 weeks before scaling
- Be prepared to kill systems that underperform

---

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Framework:** Bull Machine Validation Suite v2.0
"""

    # Save report
    report_file = 'results/validation/FINAL_DEPLOYMENT_DECISION.md'
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n{'='*100}")
    print(f"DEPLOYMENT DECISION REPORT SAVED")
    print(f"{'='*100}\n")
    print(f"Report: {report_file}\n")

    return report_file


def main():
    """Execute final comparison and create deployment decision."""

    print("\n" + "="*100)
    print("STEP 9: FINAL COMPARISON - ARCHETYPES VS BASELINES")
    print("="*100 + "\n")

    # Create comparison
    df = create_unified_comparison()

    # Determine scenario
    decision = determine_deployment_scenario(df)

    if decision:
        # Create report
        report_file = create_deployment_report(decision, df)

        print("\n" + "="*100)
        print("FINAL VERDICT")
        print("="*100 + "\n")
        print(f"Scenario: {decision['scenario']}")
        print(f"Recommendation: {decision['recommendation']}")
        print(f"Allocation: {decision['allocation']}")
        print(f"\nRead full report: {report_file}\n")
    else:
        print("\nERROR: Could not determine deployment scenario")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
