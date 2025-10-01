"""
24-Month Macro Pulse Study - Automated Runner

Execute comprehensive backtest analysis automatically without user input.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_macro_study():
    """Run the complete 24-month macro study automatically"""

    logger.info("🚀 Starting 24-Month Macro Pulse Study")
    logger.info("=" * 60)

    # Study parameters
    assets = ['ETH', 'BTC', 'SOL']
    start_date = '2023-01-01'
    end_date = '2025-01-01'
    config_file = 'configs/v170/assets/ETH_v17_baseline.json'
    output_dir = 'reports/24_month_macro_study'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"📊 Study Configuration:")
    logger.info(f"  Assets: {', '.join(assets)}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Config: {config_file}")
    logger.info(f"  Output: {output_dir}")
    logger.info("")

    try:
        # Run the comprehensive A/B and ablation study
        logger.info("🔬 Executing comprehensive macro study...")

        cmd = [
            sys.executable, 'scripts/run_macro_ab_study.py',
            '--assets'] + assets + [
            '--start', start_date,
            '--end', end_date,
            '--config', config_file,
            '--output_dir', output_dir
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')

        if result.returncode == 0:
            logger.info("✅ Study completed successfully!")
            logger.info("")
            logger.info("📈 Study Output:")
            print(result.stdout)

            # List generated files
            output_path = Path(output_dir)
            if output_path.exists():
                files = list(output_path.glob('*.csv'))
                charts = list(output_path.glob('*.png'))

                logger.info(f"📁 Generated {len(files)} CSV files and {len(charts)} charts:")
                for file in sorted(files + charts):
                    logger.info(f"  • {file.name}")

            # Print summary of findings
            print_study_findings(output_dir)

            return True

        else:
            logger.error("❌ Study failed!")
            logger.error(f"Error output: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Error running study: {e}")
        return False

def print_study_findings(output_dir: str):
    """Print summary of study findings"""

    print("\n" + "🎉" * 20)
    print("MACRO PULSE STUDY COMPLETED!")
    print("🎉" * 20)

    # Try to load and summarize key results
    try:
        import pandas as pd

        ab_summary_file = Path(output_dir) / 'ab_summary.csv'
        if ab_summary_file.exists():
            df = pd.read_csv(ab_summary_file)
            print("\n📊 KEY A/B TEST RESULTS:")
            print(df.to_string(index=False))

        ablation_file = Path(output_dir) / 'ablation_analysis.csv'
        if ablation_file.exists():
            df = pd.read_csv(ablation_file)
            print("\n🔬 TOP MACRO COMPONENTS BY IMPACT:")
            print(df.head(3).to_string(index=False))

    except Exception as e:
        logger.warning(f"Could not load summary data: {e}")

    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review {output_dir}/ab_summary.csv for A/B results")
    print(f"2. Check ablation_analysis.csv for component impacts")
    print(f"3. Examine regime_analysis.csv for regime performance")
    print(f"4. View macro_analysis_charts.png for visualizations")
    print(f"\n💡 SUCCESS CRITERIA:")
    print(f"   • Profit Factor ↑ ≥10% vs v1.6.2")
    print(f"   • Max Drawdown ↓ ≥15% vs v1.6.2")
    print(f"   • p-value < 0.05 for statistical significance")
    print(f"\n🚀 If criteria met → Ready for live paper trading!")

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("24-MONTH MACRO PULSE VALIDATION STUDY")
    print("="*80)
    print("\n🎯 EXECUTING COMPREHENSIVE ANALYSIS:")
    print("   ✓ A/B Test: v1.6.2 vs v1.7")
    print("   ✓ Regime Analysis: Risk-On/Risk-Off/Neutral")
    print("   ✓ Ablation Studies: Individual component impact")
    print("   ✓ Statistical Validation & Explainability")
    print("\n" + "="*80)

    # Run the study
    success = run_macro_study()

    if not success:
        print("\n❌ Study failed. Check logs above for errors.")
        print("💡 Common issues:")
        print("   • Missing chart_logs data")
        print("   • Configuration file errors")
        print("   • Missing Python dependencies")

if __name__ == '__main__':
    main()