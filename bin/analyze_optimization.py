#!/usr/bin/env python3
"""
Bull Machine v1.8.6 - Optimization Results Analyzer

Analyzes optimization results and generates actionable insights:
- Statistical significance testing
- Overfitting detection
- Parameter sensitivity analysis
- Production recommendations

Usage:
    python bin/analyze_optimization.py optimization_results.json
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse


class OptimizationAnalyzer:
    """Analyze optimization results and generate insights"""

    def __init__(self, results_path: str):
        self.results_path = results_path
        self.df = pd.read_json(results_path)

        # Filter minimum sample size
        self.df = self.df[self.df['sample_size'] >= 10]

        print(f"📊 Loaded {len(self.df)} configurations")

    def analyze_statistical_significance(self):
        """Test statistical significance of results"""
        print(f"\n{'='*60}")
        print("STATISTICAL SIGNIFICANCE ANALYSIS")
        print(f"{'='*60}")

        high_sig = self.df[self.df['statistical_significance'] == 'high']
        med_sig = self.df[self.df['statistical_significance'] == 'medium']
        low_sig = self.df[self.df['statistical_significance'] == 'low']

        print(f"High significance (≥100 trades): {len(high_sig)} configs")
        print(f"Medium significance (30-99 trades): {len(med_sig)} configs")
        print(f"Low significance (<30 trades): {len(low_sig)} configs")

        if len(high_sig) > 0:
            print(f"\n✅ RECOMMENDED: Use high-significance configs only")
            print(f"   Top Sharpe: {high_sig['sharpe_ratio'].max():.3f}")
            print(f"   Avg return: {high_sig['total_return'].mean():.2f}%")
            print(f"   Avg trades: {high_sig['sample_size'].mean():.0f}")
        else:
            print(f"\n⚠️  WARNING: No high-significance results")
            print(f"   Consider: Lower fusion threshold OR longer backtest period")

    def detect_overfitting(self):
        """Detect potential overfitting"""
        print(f"\n{'='*60}")
        print("OVERFITTING DETECTION")
        print(f"{'='*60}")

        # Red flags for overfitting:
        # 1. Extremely high win rate (>75%)
        # 2. Very few trades (<20)
        # 3. Profit factor >5.0 with <30 trades

        overfit_flags = []

        extreme_wr = self.df[(self.df['win_rate'] > 75) & (self.df['sample_size'] < 30)]
        if len(extreme_wr) > 0:
            overfit_flags.append(f"🚩 {len(extreme_wr)} configs with >75% WR and <30 trades (likely overfit)")

        extreme_pf = self.df[(self.df['profit_factor'] > 5.0) & (self.df['sample_size'] < 30)]
        if len(extreme_pf) > 0:
            overfit_flags.append(f"🚩 {len(extreme_pf)} configs with PF>5.0 and <30 trades (likely overfit)")

        few_trades = self.df[self.df['sample_size'] < 20]
        if len(few_trades) > len(self.df) * 0.5:
            overfit_flags.append(f"🚩 {len(few_trades)}/{len(self.df)} configs have <20 trades (thresholds too high)")

        if len(overfit_flags) > 0:
            print("⚠️  OVERFITTING RISKS DETECTED:")
            for flag in overfit_flags:
                print(f"   {flag}")
            print(f"\n   RECOMMENDATION: Use walk-forward validation")
        else:
            print("✅ No obvious overfitting detected")

    def analyze_parameter_sensitivity(self):
        """Analyze which parameters matter most"""
        print(f"\n{'='*60}")
        print("PARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*60}")

        # Group by fusion threshold
        print("\n1️⃣  FUSION THRESHOLD IMPACT")
        threshold_groups = self.df.groupby('fusion_threshold').agg({
            'sharpe_ratio': 'mean',
            'total_return': 'mean',
            'sample_size': 'mean',
            'win_rate': 'mean'
        }).round(2)
        print(threshold_groups.to_string())

        best_threshold = threshold_groups['sharpe_ratio'].idxmax()
        print(f"\n   🎯 BEST THRESHOLD: {best_threshold:.2f}")

        # Weight sensitivity
        print("\n2️⃣  DOMAIN WEIGHT SENSITIVITY")

        # Bin weights and see impact
        for weight_col in ['wyckoff_weight', 'smc_weight', 'hob_weight', 'momentum_weight']:
            self.df[f'{weight_col}_bin'] = pd.cut(self.df[weight_col], bins=3,
                                                   labels=['low', 'med', 'high'])

            weight_impact = self.df.groupby(f'{weight_col}_bin')['sharpe_ratio'].mean()
            best_bin = weight_impact.idxmax()

            print(f"\n   {weight_col.replace('_weight', '').upper()}: Best at {best_bin} weight")
            print(f"   {weight_impact.to_string()}")

    def generate_production_recommendations(self):
        """Generate actionable production recommendations"""
        print(f"\n{'='*60}")
        print("PRODUCTION RECOMMENDATIONS")
        print(f"{'='*60}")

        # Filter high-quality configs
        quality_df = self.df[
            (self.df['statistical_significance'] == 'high') &
            (self.df['sharpe_ratio'] > 0.5) &
            (self.df['profit_factor'] > 1.2) &
            (self.df['max_drawdown'] > -15)
        ]

        if len(quality_df) == 0:
            print("⚠️  NO CONFIGS MEET PRODUCTION QUALITY CRITERIA")
            print("\n   Minimum criteria:")
            print("   - ≥100 trades (statistical significance)")
            print("   - Sharpe ratio >0.5")
            print("   - Profit factor >1.2")
            print("   - Max drawdown <15%")
            print("\n   RECOMMENDATION: Lower fusion threshold OR add more data")
            return

        # Rank by Sharpe ratio
        quality_df = quality_df.sort_values('sharpe_ratio', ascending=False)

        print(f"\n✅ FOUND {len(quality_df)} PRODUCTION-QUALITY CONFIGS")
        print(f"\nTOP 3 RECOMMENDED CONFIGURATIONS:\n")

        for i, (idx, row) in enumerate(quality_df.head(3).iterrows(), 1):
            print(f"{'─'*60}")
            print(f"RANK #{i}")
            print(f"{'─'*60}")
            print(f"Fusion Threshold:  {row['fusion_threshold']:.2f}")
            print(f"Domain Weights:")
            print(f"  - Wyckoff:  {row['wyckoff_weight']:.2f}")
            print(f"  - SMC:      {row['smc_weight']:.2f}")
            print(f"  - HOB:      {row['hob_weight']:.2f}")
            print(f"  - Momentum: {row['momentum_weight']:.2f}")
            print(f"\nPerformance:")
            print(f"  - Total Return:  {row['total_return']:+.2f}%")
            print(f"  - Win Rate:      {row['win_rate']:.1f}%")
            print(f"  - Sharpe Ratio:  {row['sharpe_ratio']:.3f}")
            print(f"  - Profit Factor: {row['profit_factor']:.2f}")
            print(f"  - Max Drawdown:  {row['max_drawdown']:.2f}%")
            print(f"  - Trades:        {row['sample_size']}")
            print(f"  - Avg Trade:     {row['avg_trade_pct']:+.2f}%")
            print(f"  - Avg R-Multiple: {row['avg_r_multiple']:+.2f}")
            print()

        # Generate config file
        best_config = quality_df.iloc[0]
        self._generate_config_file(best_config)

    def _generate_config_file(self, config_row):
        """Generate optimized config file"""
        config = {
            "version": "1.8.6-optimized",
            "asset": config_row['asset'],
            "profile": "optimized",
            "optimization_date": pd.Timestamp.now().strftime("%Y-%m-%d"),

            "fusion": {
                "entry_threshold_confidence": float(config_row['fusion_threshold']),
                "weights": {
                    "wyckoff": float(config_row['wyckoff_weight']),
                    "smc": float(config_row['smc_weight']),
                    "liquidity": float(config_row['hob_weight']),
                    "momentum": float(config_row['momentum_weight'])
                }
            },

            "backtest_performance": {
                "total_return_pct": float(config_row['total_return']),
                "win_rate_pct": float(config_row['win_rate']),
                "sharpe_ratio": float(config_row['sharpe_ratio']),
                "profit_factor": float(config_row['profit_factor']),
                "max_drawdown_pct": float(config_row['max_drawdown']),
                "total_trades": int(config_row['sample_size']),
                "backtest_period": config_row['period']
            }
        }

        output_path = f"configs/v18/{config_row['asset']}_optimized.json"
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"💾 Optimized config saved to: {output_path}")

    def generate_comparison_chart(self):
        """Generate parameter comparison data"""
        print(f"\n{'='*60}")
        print("PARAMETER COMPARISON (for visualization)")
        print(f"{'='*60}")

        # Export data for plotting
        export_df = self.df[self.df['statistical_significance'] == 'high'][[
            'fusion_threshold', 'wyckoff_weight', 'momentum_weight',
            'sharpe_ratio', 'total_return', 'win_rate', 'sample_size'
        ]]

        export_path = self.results_path.replace('.json', '_comparison.csv')
        export_df.to_csv(export_path, index=False)
        print(f"📊 Comparison data exported to: {export_path}")
        print(f"   Use this for visualization in Excel/Python")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        self.analyze_statistical_significance()
        self.detect_overfitting()
        self.analyze_parameter_sensitivity()
        self.generate_production_recommendations()
        self.generate_comparison_chart()

        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE ✅")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Bull Machine optimization results')
    parser.add_argument('results_file', help='Path to optimization results JSON')

    args = parser.parse_args()

    if not Path(args.results_file).exists():
        print(f"❌ Error: File not found: {args.results_file}")
        sys.exit(1)

    analyzer = OptimizationAnalyzer(args.results_file)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
