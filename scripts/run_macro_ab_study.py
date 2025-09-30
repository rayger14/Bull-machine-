"""
Comprehensive A/B and Ablation Study for Macro Pulse

Runs complete comparison:
1. A/B: v1.6.2 (no macro) vs v1.7 (with macro)
2. Regime Analysis: Risk-On vs Risk-Off vs Neutral performance
3. Ablation Studies: Individual veto/boost impact measurement
4. Statistical Significance Testing
"""

import subprocess
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroABStudy:
    """Complete A/B and ablation study framework"""

    def __init__(self, config_path: str, output_dir: str):
        self.config_path = config_path
        self.output_dir = output_dir
        self.results = {}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def run_full_study(self, assets: List[str], start: str, end: str):
        """Run complete A/B and ablation study"""
        logger.info("Starting comprehensive macro pulse study...")

        # 1. A/B Test: Macro On vs Off
        logger.info("Running A/B test: Macro On vs Off")
        self._run_ab_test(assets, start, end)

        # 2. Ablation Studies
        logger.info("Running ablation studies")
        self._run_ablation_studies(assets, start, end)

        # 3. Generate comparative analysis
        logger.info("Generating comparative analysis")
        self._generate_analysis()

        # 4. Create visualizations
        logger.info("Creating visualizations")
        self._create_visualizations()

        logger.info(f"Study complete. Results saved to {self.output_dir}")

    def _run_ab_test(self, assets: List[str], start: str, end: str):
        """Run A/B test comparing macro on vs off"""

        # Run with macro enabled (v1.7)
        logger.info("Running v1.7 (macro enabled)...")
        cmd_macro = [
            'python3', 'scripts/run_macro_backtest.py',
            '--assets'] + assets + [
            '--start', start,
            '--end', end,
            '--config', self.config_path,
            '--output_dir', self.output_dir,
            '--ablation', 'none'
        ]

        result = subprocess.run(cmd_macro, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Macro enabled run failed: {result.stderr}")

        # Run with macro disabled (v1.6.2 baseline)
        logger.info("Running v1.6.2 (macro disabled)...")
        cmd_no_macro = [
            'python3', 'scripts/run_macro_backtest.py',
            '--assets'] + assets + [
            '--start', start,
            '--end', end,
            '--config', self.config_path,
            '--output_dir', self.output_dir,
            '--disable_macro'
        ]

        result = subprocess.run(cmd_no_macro, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Macro disabled run failed: {result.stderr}")

    def _run_ablation_studies(self, assets: List[str], start: str, end: str):
        """Run ablation studies for each macro component"""

        ablations = [
            'no_dxy',
            'no_vix',
            'no_oil_dxy',
            'no_yields',
            'no_usdjpy',
            'no_hyg',
            'no_usdt_sfp'
        ]

        for ablation in ablations:
            logger.info(f"Running ablation: {ablation}")

            cmd = [
                'python3', 'scripts/run_macro_backtest.py',
                '--assets'] + assets + [
                '--start', start,
                '--end', end,
                '--config', self.config_path,
                '--output_dir', self.output_dir,
                '--ablation', ablation
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Ablation {ablation} failed: {result.stderr}")

    def _load_results(self) -> Dict[str, pd.DataFrame]:
        """Load all backtest results"""
        results = {}

        # Find all summary CSV files
        for file in Path(self.output_dir).glob('summary*.csv'):
            key = file.stem.replace('summary_', '').replace('summary', 'baseline')
            try:
                df = pd.read_csv(file)
                results[key] = df
                logger.info(f"Loaded {key}: {len(df)} assets")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

        return results

    def _generate_analysis(self):
        """Generate comprehensive comparative analysis"""
        results = self._load_results()

        if not results:
            logger.error("No results found for analysis")
            return

        # Generate comparison tables
        self._generate_ab_comparison(results)
        self._generate_ablation_analysis(results)
        self._generate_regime_analysis(results)

    def _generate_ab_comparison(self, results: Dict[str, pd.DataFrame]):
        """Generate A/B comparison between macro on/off"""
        try:
            macro_on = results.get('macro', pd.DataFrame())
            macro_off = results.get('no_macro', pd.DataFrame())

            if macro_on.empty or macro_off.empty:
                logger.warning("Missing A/B test data")
                return

            # Merge on asset
            comparison = macro_on.merge(macro_off, on='asset', suffixes=('_macro', '_no_macro'))

            # Calculate improvements
            comparison['pnl_improvement'] = comparison['pnl_percent_macro'] - comparison['pnl_percent_no_macro']
            comparison['pf_improvement'] = comparison['profit_factor_macro'] / comparison['profit_factor_no_macro'] - 1
            comparison['dd_improvement'] = comparison['max_drawdown_no_macro'] - comparison['max_drawdown_macro']
            comparison['winrate_improvement'] = comparison['win_rate_macro'] - comparison['win_rate_no_macro']

            # Summary statistics
            summary = {
                'Metric': ['PnL Improvement (%)', 'Profit Factor Lift (%)', 'Max DD Reduction (%)', 'Win Rate Improvement (%)'],
                'Mean': [
                    comparison['pnl_improvement'].mean(),
                    comparison['pf_improvement'].mean() * 100,
                    comparison['dd_improvement'].mean() * 100,
                    comparison['winrate_improvement'].mean() * 100
                ],
                'Median': [
                    comparison['pnl_improvement'].median(),
                    comparison['pf_improvement'].median() * 100,
                    comparison['dd_improvement'].median() * 100,
                    comparison['winrate_improvement'].median() * 100
                ],
                'Best_Asset': [
                    comparison.loc[comparison['pnl_improvement'].idxmax(), 'asset'],
                    comparison.loc[comparison['pf_improvement'].idxmax(), 'asset'],
                    comparison.loc[comparison['dd_improvement'].idxmax(), 'asset'],
                    comparison.loc[comparison['winrate_improvement'].idxmax(), 'asset']
                ]
            }

            summary_df = pd.DataFrame(summary)

            # Save results
            comparison.to_csv(os.path.join(self.output_dir, 'ab_comparison.csv'), index=False)
            summary_df.to_csv(os.path.join(self.output_dir, 'ab_summary.csv'), index=False)

            # Print key findings
            print("\n=== A/B TEST RESULTS (v1.6.2 vs v1.7) ===")
            print(summary_df.to_string(index=False))

            # Statistical significance test
            from scipy import stats
            pnl_macro = comparison['pnl_percent_macro'].values
            pnl_no_macro = comparison['pnl_percent_no_macro'].values

            if len(pnl_macro) > 1:
                t_stat, p_value = stats.ttest_rel(pnl_macro, pnl_no_macro)
                print(f"\nStatistical Significance (paired t-test):")
                print(f"t-statistic: {t_stat:.3f}")
                print(f"p-value: {p_value:.3f}")
                print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")

        except Exception as e:
            logger.error(f"Error in A/B comparison: {e}")

    def _generate_ablation_analysis(self, results: Dict[str, pd.DataFrame]):
        """Analyze impact of individual ablations"""
        try:
            baseline = results.get('macro', pd.DataFrame())
            if baseline.empty:
                logger.warning("No baseline macro results for ablation analysis")
                return

            ablations = ['no_dxy', 'no_vix', 'no_oil_dxy', 'no_yields', 'no_usdjpy', 'no_hyg', 'no_usdt_sfp']

            ablation_impacts = []

            for ablation in ablations:
                ablation_results = results.get(ablation, pd.DataFrame())
                if ablation_results.empty:
                    continue

                # Merge with baseline
                merged = baseline.merge(ablation_results, on='asset', suffixes=('_baseline', '_ablation'))

                # Calculate impact of removing this component
                pf_impact = (merged['profit_factor_baseline'] / merged['profit_factor_ablation'] - 1).mean()
                dd_impact = (merged['max_drawdown_ablation'] - merged['max_drawdown_baseline']).mean()
                pnl_impact = (merged['pnl_percent_baseline'] - merged['pnl_percent_ablation']).mean()
                trades_impact = (merged['total_trades_baseline'] - merged['total_trades_ablation']).mean()

                ablation_impacts.append({
                    'Component': ablation.replace('no_', '').upper(),
                    'PF_Impact_Pct': pf_impact * 100,
                    'DD_Impact_Pct': dd_impact * 100,
                    'PnL_Impact_Pct': pnl_impact,
                    'Trades_Impact': trades_impact,
                    'Net_Benefit': pf_impact * 100 - dd_impact * 50  # Combined score
                })

            ablation_df = pd.DataFrame(ablation_impacts)
            ablation_df = ablation_df.sort_values('Net_Benefit', ascending=False)

            # Save results
            ablation_df.to_csv(os.path.join(self.output_dir, 'ablation_analysis.csv'), index=False)

            print("\n=== ABLATION STUDY RESULTS ===")
            print("Impact of removing each macro component:")
            print(ablation_df.to_string(index=False))

        except Exception as e:
            logger.error(f"Error in ablation analysis: {e}")

    def _generate_regime_analysis(self, results: Dict[str, pd.DataFrame]):
        """Analyze performance by macro regime"""
        try:
            # This would require loading individual trade files to get regime data
            # For now, create a placeholder analysis

            print("\n=== REGIME ANALYSIS ===")
            print("Performance breakdown by macro regime:")
            print("(Detailed regime analysis requires trade-level data)")

            # Load trade files if they exist
            trade_files = list(Path(self.output_dir).glob('*_trades_macro.csv'))

            if trade_files:
                all_trades = []
                for file in trade_files:
                    try:
                        df = pd.read_csv(file)
                        df['asset'] = file.stem.split('_')[0]
                        all_trades.append(df)
                    except Exception as e:
                        logger.error(f"Error loading {file}: {e}")

                if all_trades:
                    combined_trades = pd.concat(all_trades, ignore_index=True)
                    exits = combined_trades[combined_trades['event'] == 'EXIT'].copy()

                    if not exits.empty and 'macro_regime' in exits.columns:
                        regime_analysis = exits.groupby('macro_regime').agg({
                            'pnl': ['count', 'mean', 'std', lambda x: (x > 0).mean()]
                        }).round(4)

                        regime_analysis.columns = ['Trade_Count', 'Avg_Return', 'Return_Std', 'Win_Rate']
                        regime_analysis['Profit_Factor'] = exits.groupby('macro_regime').apply(
                            lambda x: x[x['pnl'] > 0]['pnl'].sum() / abs(x[x['pnl'] < 0]['pnl'].sum())
                            if (x['pnl'] < 0).any() else float('inf')
                        )

                        print(regime_analysis)
                        regime_analysis.to_csv(os.path.join(self.output_dir, 'regime_analysis.csv'))

        except Exception as e:
            logger.error(f"Error in regime analysis: {e}")

    def _create_visualizations(self):
        """Create visualization charts"""
        try:
            results = self._load_results()

            if not results:
                return

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Macro Pulse Backtest Analysis', fontsize=16)

            # 1. A/B Comparison
            if 'macro' in results and 'no_macro' in results:
                macro_df = results['macro']
                no_macro_df = results['no_macro']

                ax1 = axes[0, 0]
                x = np.arange(len(macro_df))
                width = 0.35

                ax1.bar(x - width/2, macro_df['profit_factor'], width, label='v1.7 (Macro)', alpha=0.8)
                ax1.bar(x + width/2, no_macro_df['profit_factor'], width, label='v1.6.2 (No Macro)', alpha=0.8)
                ax1.set_xlabel('Assets')
                ax1.set_ylabel('Profit Factor')
                ax1.set_title('A/B Test: Profit Factor Comparison')
                ax1.set_xticks(x)
                ax1.set_xticklabels(macro_df['asset'])
                ax1.legend()

            # 2. Ablation Impact
            ablation_file = os.path.join(self.output_dir, 'ablation_analysis.csv')
            if os.path.exists(ablation_file):
                ablation_df = pd.read_csv(ablation_file)
                ax2 = axes[0, 1]
                ax2.barh(ablation_df['Component'], ablation_df['Net_Benefit'])
                ax2.set_xlabel('Net Benefit Score')
                ax2.set_title('Ablation Study: Component Impact')

            # 3. PnL Distribution
            if 'macro' in results:
                ax3 = axes[1, 0]
                ax3.hist(results['macro']['pnl_percent'], bins=10, alpha=0.7, label='v1.7')
                if 'no_macro' in results:
                    ax3.hist(results['no_macro']['pnl_percent'], bins=10, alpha=0.7, label='v1.6.2')
                ax3.set_xlabel('PnL %')
                ax3.set_ylabel('Frequency')
                ax3.set_title('PnL Distribution')
                ax3.legend()

            # 4. Risk-Adjusted Returns
            if 'macro' in results:
                macro_df = results['macro']
                ax4 = axes[1, 1]
                # Risk-adjusted return = PnL / Max Drawdown
                risk_adj = macro_df['pnl_percent'] / macro_df['max_drawdown'].replace(0, 1)
                ax4.scatter(macro_df['max_drawdown'], macro_df['pnl_percent'], alpha=0.7)
                ax4.set_xlabel('Max Drawdown %')
                ax4.set_ylabel('PnL %')
                ax4.set_title('Risk vs Return (v1.7)')

                # Add trend line
                z = np.polyfit(macro_df['max_drawdown'], macro_df['pnl_percent'], 1)
                p = np.poly1d(z)
                ax4.plot(macro_df['max_drawdown'], p(macro_df['max_drawdown']), "r--", alpha=0.8)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'macro_analysis_charts.png'), dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Visualization saved to {self.output_dir}/macro_analysis_charts.png")

        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

def main():
    """Run comprehensive macro pulse study"""
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive Macro Pulse A/B Study')
    parser.add_argument('--assets', nargs='+', default=['ETH', 'BTC', 'SOL'])
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2025-01-01')
    parser.add_argument('--config', default='configs/v170/assets/ETH_v17_baseline.json')
    parser.add_argument('--output_dir', default='reports/macro_study')

    args = parser.parse_args()

    # Initialize study
    study = MacroABStudy(args.config, args.output_dir)

    # Run full study
    study.run_full_study(args.assets, args.start, args.end)

    print(f"\n=== STUDY COMPLETE ===")
    print(f"Results saved to: {args.output_dir}")
    print("\nKey files generated:")
    print("- ab_comparison.csv: Detailed A/B test results")
    print("- ab_summary.csv: A/B test summary statistics")
    print("- ablation_analysis.csv: Individual component impact")
    print("- regime_analysis.csv: Performance by macro regime")
    print("- macro_analysis_charts.png: Visualization dashboard")

if __name__ == '__main__':
    main()