"""
24-Month Macro Pulse Study Runner

Execute comprehensive backtest analysis:
1. A/B: v1.6.2 (no macro) vs v1.7 (with macro)
2. Regime analysis: Risk-On vs Risk-Off vs Neutral
3. Ablation studies: Individual veto/boost impact
4. Statistical validation and explainability

Run this to get definitive v1.7 Macro Pulse validation.
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        logger.info("✓ All required dependencies found")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install pandas numpy matplotlib seaborn scipy")
        return False

def check_data_directory():
    """Check if chart_logs directory exists"""
    chart_logs = Path("chart_logs")
    if chart_logs.exists():
        logger.info(f"✓ Found chart_logs directory with {len(list(chart_logs.glob('*')))} files")
        return True
    else:
        logger.warning("⚠ chart_logs directory not found")
        logger.info("Will use synthetic data for testing")
        return False

def run_macro_study():
    """Run the complete 24-month macro study"""

    logger.info("🚀 Starting 24-Month Macro Pulse Study")
    logger.info("=" * 60)

    # Check prerequisites
    if not check_dependencies():
        return False

    check_data_directory()

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

            return True

        else:
            logger.error("❌ Study failed!")
            logger.error(f"Error output: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"❌ Error running study: {e}")
        return False

def print_study_summary():
    """Print what the study will measure"""

    print("\n" + "=" * 80)
    print("24-MONTH MACRO PULSE VALIDATION STUDY")
    print("=" * 80)

    print("\n🎯 WHAT WE'RE MEASURING:")
    print("\n1. A/B COMPARISON (v1.6.2 vs v1.7)")
    print("   • Profit Factor improvement")
    print("   • Maximum Drawdown reduction")
    print("   • Win Rate impact")
    print("   • Trade frequency changes")
    print("   • Statistical significance testing")

    print("\n2. REGIME SLICING")
    print("   • Risk-On regime performance")
    print("   • Risk-Off regime performance")
    print("   • Neutral regime performance")
    print("   • Regime transition handling")

    print("\n3. ABLATION STUDIES")
    print("   • DXY veto impact")
    print("   • VIX/MOVE spike protection")
    print("   • Oil+DXY stagflation detection")
    print("   • Yield spike filtering")
    print("   • USDJPY carry unwind protection")
    print("   • HYG credit stress detection")
    print("   • USDT.D SFP/Wolfe trap avoidance")

    print("\n4. EXPLAINABILITY")
    print("   • Per-trade macro context")
    print("   • Veto reason tracking")
    print("   • Boost factor attribution")
    print("   • Human-readable trade notes")

    print("\n📊 SUCCESS CRITERIA:")
    print("   ✓ Profit Factor ↑ by ≥10% vs v1.6.2")
    print("   ✓ Max Drawdown ↓ by ≥15% vs v1.6.2")
    print("   ✓ Risk-off regime DD materially lower")
    print("   ✓ Each major veto shows positive/neutral impact")
    print("   ✓ Statistical significance (p < 0.05)")

    print("\n🚀 EXPECTED INSIGHTS:")
    print("   • Which macro signals provide most value")
    print("   • How regime awareness improves risk management")
    print("   • Trade-off between signal frequency and quality")
    print("   • Macro timing vs pure technical timing")

    print("\n" + "=" * 80)

def main():
    """Main execution function"""

    print_study_summary()

    # Ask for confirmation
    try:
        response = input("\n🎮 Ready to run 24-month study? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Study cancelled.")
            return
    except KeyboardInterrupt:
        print("\nStudy cancelled.")
        return

    # Run the study
    success = run_macro_study()

    if success:
        print("\n" + "🎉" * 20)
        print("STUDY COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        print("\n📋 NEXT STEPS:")
        print("1. Review reports/24_month_macro_study/ab_summary.csv")
        print("2. Check ablation_analysis.csv for component impacts")
        print("3. Examine regime_analysis.csv for regime performance")
        print("4. View macro_analysis_charts.png for visualizations")
        print("\n💡 If results meet criteria (PF ↑10%, DD ↓15%), proceed to live paper!")

    else:
        print("\n❌ Study failed. Check logs above for errors.")
        print("💡 Common issues:")
        print("   • Missing chart_logs data")
        print("   • Configuration file errors")
        print("   • Missing Python dependencies")

if __name__ == '__main__':
    main()