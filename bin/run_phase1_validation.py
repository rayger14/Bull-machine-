#!/usr/bin/env python3
"""
Phase 1 Quick Validation - Run 5 threshold configs on 2022 bear market data.
"""
import subprocess
import json
import pandas as pd
from pathlib import Path
import sys

# Config definitions
CONFIGS = [
    ("ultra_strict", "configs/quick_test/ultra_strict.json", "5-10 trades"),
    ("strict", "configs/quick_test/strict.json", "15-20 trades"),
    ("moderate", "configs/quick_test/moderate.json", "25-35 trades"),
    ("relaxed", "configs/quick_test/relaxed.json", "40-50 trades"),
    ("ultra_relaxed", "configs/quick_test/ultra_relaxed.json", "60+ trades"),
]

ASSET = "BTC"
START = "2022-01-01"
END = "2022-12-31"
OUTPUT_DIR = Path("results/phase1_quick_validation")

def run_backtest(name, config_path, expected):
    """Run single backtest configuration."""
    output_csv = OUTPUT_DIR / f"{name}_trades.csv"
    output_log = OUTPUT_DIR / f"{name}.log"

    print(f"\n{'='*80}")
    print(f"Running: {name} (expected: {expected})")
    print(f"Config: {config_path}")
    print(f"{'='*80}")

    # Run backtest
    cmd = [
        "python3", "bin/backtest_knowledge_v2.py",
        "--config", config_path,
        "--asset", ASSET,
        "--start", START,
        "--end", END,
        "--export-trades", str(output_csv)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )

        # Write log
        with open(output_log, 'w') as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode != 0:
            print(f"❌ FAILED: {name}")
            print(f"   See log: {output_log}")
            return None

        # Parse CSV
        if not output_csv.exists():
            print(f"⚠️  No trades file generated: {output_csv}")
            return None

        df = pd.read_csv(output_csv)

        # Extract metrics
        total_trades = len(df)
        wins = df['trade_won'].sum()
        wr = wins / total_trades if total_trades > 0 else 0
        total_r = df['r_multiple'].sum()

        win_r = df[df['trade_won'] == 1]['r_multiple'].sum()
        loss_r = abs(df[df['trade_won'] == 0]['r_multiple'].sum())
        pf = win_r / loss_r if loss_r > 0 else 0

        # Count archetypes
        s2_col = 'archetype_failed_rally'
        s5_col = 'archetype_long_squeeze'

        s2_count = df[s2_col].sum() if s2_col in df.columns else 0
        s5_count = df[s5_col].sum() if s5_col in df.columns else 0

        # Count bull archetypes (for comparison)
        bull_archs = ['archetype_trap', 'archetype_retest', 'archetype_continuation',
                      'archetype_failed_continuation', 'archetype_compression',
                      'archetype_exhaustion', 'archetype_reaccumulation',
                      'archetype_trap_within_trend', 'archetype_wick_trap',
                      'archetype_volume_exhaustion', 'archetype_ratio_coil_break',
                      'archetype_false_break_reversal']
        bull_count = sum(df[col].sum() for col in bull_archs if col in df.columns)

        # Load config to get thresholds
        with open(config_path) as f:
            cfg = json.load(f)
        s2_fusion = cfg['archetypes']['thresholds']['failed_rally']['fusion_threshold']
        s5_fusion = cfg['archetypes']['thresholds']['long_squeeze']['fusion_threshold']

        print(f"✅ Complete: {name}")
        print(f"   Trades: {total_trades}")
        print(f"   Win Rate: {wr:.1%}")
        print(f"   Profit Factor: {pf:.2f}")
        print(f"   Total R: {total_r:.2f}")
        print(f"   S2: {s2_count}, S5: {s5_count}, Bull: {bull_count}")

        return {
            'name': name,
            'expected': expected,
            'trades': total_trades,
            'win_rate': wr,
            'profit_factor': pf,
            'total_r': total_r,
            's2_count': s2_count,
            's5_count': s5_count,
            'bull_count': bull_count,
            's2_fusion': s2_fusion,
            's5_fusion': s5_fusion,
        }

    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {name} (>5 min)")
        return None
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   {e}")
        return None

def main():
    """Run all Phase 1 validation tests."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PHASE 1 QUICK VALIDATION")
    print("="*80)
    print(f"Asset: {ASSET}")
    print(f"Period: {START} to {END} (2022 Bear Market)")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Output: {OUTPUT_DIR}")
    print("="*80)

    results = []
    for name, config_path, expected in CONFIGS:
        result = run_backtest(name, config_path, expected)
        if result:
            results.append(result)

    # Summary table
    if results:
        print("\n" + "="*80)
        print("PHASE 1 VALIDATION SUMMARY")
        print("="*80)

        df = pd.DataFrame(results)

        # Format table
        print(f"\n{'Config':<15} {'Trades':<8} {'WR':<8} {'PF':<8} {'R':<8} {'S2':<6} {'S5':<6} {'Bull':<6}")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['name']:<15} {row['trades']:<8} {row['win_rate']:<8.1%} "
                  f"{row['profit_factor']:<8.2f} {row['total_r']:<8.2f} "
                  f"{row['s2_count']:<6.0f} {row['s5_count']:<6.0f} {row['bull_count']:<6.0f}")

        # Export summary
        summary_path = OUTPUT_DIR / "phase1_summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary exported: {summary_path}")

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)

        target = df[(df['trades'] >= 25) & (df['trades'] <= 40)]
        if not target.empty:
            best = target.loc[target['profit_factor'].idxmax()]
            print(f"✅ Best config in target range (25-40 trades): {best['name']}")
            print(f"   Trades: {best['trades']}, PF: {best['profit_factor']:.2f}, WR: {best['win_rate']:.1%}")
            print(f"   S2 fusion: {best['s2_fusion']}, S5 fusion: {best['s5_fusion']}")
            print(f"\n   Use this as starting point for Phase 2 optimization.")
        else:
            print("⚠️  No config in target range (25-40 trades)")
            print("   Adjust thresholds or proceed with closest config.")

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("1. Review results in results/phase1_quick_validation/")
        print("2. If satisfied, proceed to Phase 2:")
        print("   python3 bin/optimize_bear_s2_s5_multiobjective.py --trials 50")
        print("="*80)
    else:
        print("\n❌ No successful results")
        sys.exit(1)

if __name__ == "__main__":
    main()
