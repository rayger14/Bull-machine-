#!/usr/bin/env python3
"""
Professional Quant Testing Suite

Runs standardized backtests on all models (baselines + Bull Machine archetypes)
with consistent train/test/OOS splits, metrics, and ranking.

This is the foundation of scientific strategy validation:
1. All strategies run on same data
2. All strategies use same costs
3. All strategies judged by same metrics
4. Baselines provide honesty check (can't beat buy-and-hold? Don't trade.)

Usage:
    # Run baselines only
    python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only

    # Run baselines + archetypes
    python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json

    # Verbose mode
    python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --verbose

    # Custom output directory
    python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --output results/my_test

Output:
    - CSV file with all metrics (ranked by test PF)
    - Markdown report with analysis and red flags
    - Console table with color-coded results
"""

import sys
import os
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.models.base import BaseModel
from engine.models.baselines import get_all_baselines
from engine.backtesting.engine import BacktestEngine, BacktestResults


@dataclass
class ModelResults:
    """Results for one model across all periods."""
    model_name: str
    train_results: BacktestResults
    test_results: BacktestResults
    oos_results: BacktestResults

    @property
    def overfit_score(self) -> float:
        """Train PF - Test PF (lower is better)."""
        return self.train_results.profit_factor - self.test_results.profit_factor

    @property
    def status_emoji(self) -> str:
        """Color code based on test PF."""
        pf = self.test_results.profit_factor
        if pf >= 2.0:
            return '✅'
        elif pf >= 1.5:
            return '🔧'
        else:
            return '❌'


class QuantSuite:
    """
    Professional quant testing suite.

    Runs standardized backtests on baselines and archetypes.
    """

    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize quant suite.

        Args:
            config_path: Path to experiment configuration JSON
            verbose: Enable verbose logging
        """
        self.config_path = config_path
        self.verbose = verbose

        # Load configuration
        self.config = self._load_config()

        # Load data
        self.data = self._load_data()

        # Build model list
        self.models: List[BaseModel] = []
        self._build_model_list()

        # Results storage
        self.results: List[ModelResults] = []

    def _load_config(self) -> dict:
        """Load experiment configuration from JSON."""
        config_path = PROJECT_ROOT / self.config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"\n{'='*80}")
        print(f"QUANT SUITE: {config['experiment_name']}")
        print(f"{'='*80}")
        print(f"Asset: {config['asset']} {config['timeframe']}")
        print(f"Train: {config['periods']['train']['start']} to {config['periods']['train']['end']}")
        print(f"Test:  {config['periods']['test']['start']} to {config['periods']['test']['end']}")
        print(f"OOS:   {config['periods']['oos']['start']} to {config['periods']['oos']['end']}")
        print(f"Costs: {config['costs']['total_bps']}bp ({config['costs']['slippage_bps']}bp slip + {config['costs']['fee_bps']}bp fees)")
        print(f"{'='*80}\n")

        return config

    def _load_data(self) -> pd.DataFrame:
        """Load feature store data."""
        data_path = PROJECT_ROOT / self.config['data_path']
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"Loading data from {data_path.name}...")
        data = pd.read_parquet(data_path)

        # Ensure timestamp index
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        data.index = pd.to_datetime(data.index)

        # Add basic indicators if missing (for baselines)
        data = self._add_basic_indicators(data)

        print(f"Loaded {len(data):,} bars ({data.index[0]} to {data.index[-1]})\n")

        # Validate date coverage
        self._validate_date_coverage(data)

        return data

    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic indicators required by baseline models.

        Args:
            data: Raw OHLCV data

        Returns:
            Data with SMA, RSI, ATR columns
        """
        print("Calculating basic indicators for baselines...")

        # SMA(50, 200)
        if 'sma_50' not in data.columns:
            data['sma_50'] = data['close'].rolling(window=50).mean()
        if 'sma_200' not in data.columns:
            data['sma_200'] = data['close'].rolling(window=200).mean()

        # RSI(14)
        if 'rsi_14' not in data.columns:
            data['rsi_14'] = self._calculate_rsi(data['close'], period=14)

        # ATR(14)
        if 'atr_14' not in data.columns:
            data['atr_14'] = self._calculate_atr(data, period=14)

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        return atr

    def _validate_date_coverage(self, data: pd.DataFrame) -> None:
        """Validate that data covers all required periods."""
        periods = self.config['periods']

        for period_name, period_config in periods.items():
            start = pd.Timestamp(period_config['start'])
            end = pd.Timestamp(period_config['end'])

            # Make timezone-aware if data index is timezone-aware
            if data.index.tz is not None:
                start = start.tz_localize('UTC')
                end = end.tz_localize('UTC')

            period_data = data[(data.index >= start) & (data.index <= end)]
            if len(period_data) == 0:
                raise ValueError(f"No data available for {period_name} period ({start} to {end})")

            print(f"  {period_name.upper()}: {len(period_data):,} bars")

        print()

    def _build_model_list(self) -> None:
        """Build list of models to test."""
        print("Building model roster...")

        # Add baselines
        if self.config.get('baseline_config', {}).get('enabled', True):
            baseline_classes = get_all_baselines()
            for baseline_cls in baseline_classes:
                model = baseline_cls()
                self.models.append(model)
                print(f"  + {model.name}")

        # Add archetypes (if enabled)
        if self.config.get('archetype_config', {}).get('enabled', False):
            # TODO: Add archetype model loading
            print("  (Archetypes not yet implemented)")

        print(f"\nTotal models: {len(self.models)}\n")

    def run(self) -> None:
        """Run backtests on all models."""
        print(f"{'='*80}")
        print(f"RUNNING BACKTESTS")
        print(f"{'='*80}\n")

        total_costs_pct = self.config['costs']['total_bps'] / 10000  # Convert bp to decimal

        for i, model in enumerate(self.models, 1):
            print(f"[{i}/{len(self.models)}] {model.name}")

            try:
                # Fit on train data
                train_period = self.config['periods']['train']
                train_data = self.data[train_period['start']:train_period['end']]
                model.fit(train_data)

                # Run train backtest
                train_results = self._run_backtest(model, 'train')

                # Run test backtest
                test_results = self._run_backtest(model, 'test')

                # Run OOS backtest
                oos_results = self._run_backtest(model, 'oos')

                # Store results
                self.results.append(ModelResults(
                    model_name=model.name,
                    train_results=train_results,
                    test_results=test_results,
                    oos_results=oos_results
                ))

                # Print summary
                print(f"  Train: PF={train_results.profit_factor:.2f}, Trades={train_results.total_trades}")
                print(f"  Test:  PF={test_results.profit_factor:.2f}, Trades={test_results.total_trades}")
                print(f"  OOS:   PF={oos_results.profit_factor:.2f}, Trades={oos_results.total_trades}")
                print()

            except Exception as e:
                print(f"  ERROR: {e}")
                print()
                continue

    def _run_backtest(self, model: BaseModel, period_name: str) -> BacktestResults:
        """
        Run backtest on specified period.

        Args:
            model: Trading model
            period_name: Period name (train/test/oos)

        Returns:
            BacktestResults
        """
        period_config = self.config['periods'][period_name]
        start = period_config['start']
        end = period_config['end']

        # Get costs
        total_costs_pct = self.config['costs']['total_bps'] / 10000

        # Run backtest
        engine = BacktestEngine(
            model=model,
            data=self.data,
            initial_capital=self.config['initial_capital'],
            commission_pct=total_costs_pct
        )

        results = engine.run(start=start, end=end, verbose=False)
        return results

    def generate_report(self, output_dir: str = None) -> None:
        """
        Generate ranked results table and report.

        Args:
            output_dir: Output directory (default: from config)
        """
        if output_dir is None:
            output_dir = self.config['output']['results_dir']

        output_path = PROJECT_ROOT / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Generate CSV
        if self.config['output']['save_csv']:
            csv_path = output_path / f'quant_suite_results_{timestamp}.csv'
            self._generate_csv(csv_path)

        # Generate report
        if self.config['output']['save_report']:
            report_path = output_path / f'quant_suite_report_{timestamp}.md'
            self._generate_markdown_report(report_path)

        # Print console table
        self._print_results_table()

    def _generate_csv(self, csv_path: Path) -> None:
        """Generate CSV with all results."""
        rows = []

        for model_result in self.results:
            # Add train results
            row = model_result.train_results.to_dict('train')
            row['overfit_score'] = model_result.overfit_score
            rows.append(row)

            # Add test results
            row = model_result.test_results.to_dict('test')
            row['overfit_score'] = model_result.overfit_score
            rows.append(row)

            # Add OOS results
            row = model_result.oos_results.to_dict('oos')
            row['overfit_score'] = model_result.overfit_score
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by test PF (descending)
        test_df = df[df['period'] == 'test'].sort_values('profit_factor', ascending=False)
        model_order = test_df['model_name'].tolist()

        # Reorder full dataframe
        df['model_order'] = df['model_name'].apply(lambda x: model_order.index(x) if x in model_order else 999)
        df = df.sort_values(['model_order', 'period'])
        df = df.drop(columns=['model_order'])

        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    def _print_results_table(self) -> None:
        """Print formatted results table to console."""
        # Sort by test PF
        sorted_results = sorted(
            self.results,
            key=lambda x: x.test_results.profit_factor,
            reverse=True
        )

        print(f"\n{'='*120}")
        print(f"RANKED RESULTS (by Test PF)")
        print(f"{'='*120}")
        print(f"{'Rank':<6}{'Model':<35}{'Train PF':<12}{'Test PF':<12}{'OOS PF':<12}{'Overfit':<12}{'Trades':<10}{'Status':<8}")
        print(f"{'-'*120}")

        for rank, model_result in enumerate(sorted_results, 1):
            train_pf = model_result.train_results.profit_factor
            test_pf = model_result.test_results.profit_factor
            oos_pf = model_result.oos_results.profit_factor
            overfit = model_result.overfit_score
            trades = model_result.test_results.total_trades
            status = model_result.status_emoji

            print(f"{rank:<6}{model_result.model_name:<35}{train_pf:<12.2f}{test_pf:<12.2f}{oos_pf:<12.2f}{overfit:<12.2f}{trades:<10}{status:<8}")

        print(f"{'='*120}\n")

        # Legend
        print("Legend:")
        print("  ✅ = Test PF >= 2.0 (excellent)")
        print("  🔧 = Test PF 1.5-2.0 (acceptable)")
        print("  ❌ = Test PF < 1.5 (needs work)")
        print()

    def _generate_markdown_report(self, report_path: Path) -> None:
        """Generate markdown report with analysis."""
        # Sort by test PF
        sorted_results = sorted(
            self.results,
            key=lambda x: x.test_results.profit_factor,
            reverse=True
        )

        # Calculate overfit ranking
        overfit_sorted = sorted(self.results, key=lambda x: x.overfit_score)

        # Find red flags
        red_flags = []
        for model_result in self.results:
            if model_result.overfit_score > 0.5:
                red_flags.append(f"- {model_result.model_name}: High overfit (Train-Test PF = {model_result.overfit_score:.2f})")
            if model_result.test_results.total_trades < 50:
                red_flags.append(f"- {model_result.model_name}: Low trade count ({model_result.test_results.total_trades} trades)")
            if model_result.oos_results.profit_factor < 1.0:
                red_flags.append(f"- {model_result.model_name}: Poor OOS performance (PF = {model_result.oos_results.profit_factor:.2f})")

        # Write report
        with open(report_path, 'w') as f:
            f.write(f"# Quant Suite Report\n\n")
            f.write(f"**Experiment:** {self.config['experiment_name']}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"## Configuration\n\n")
            f.write(f"- **Asset:** {self.config['asset']} {self.config['timeframe']}\n")
            f.write(f"- **Train Period:** {self.config['periods']['train']['start']} to {self.config['periods']['train']['end']}\n")
            f.write(f"- **Test Period:** {self.config['periods']['test']['start']} to {self.config['periods']['test']['end']}\n")
            f.write(f"- **OOS Period:** {self.config['periods']['oos']['start']} to {self.config['periods']['oos']['end']}\n")
            f.write(f"- **Costs:** {self.config['costs']['total_bps']}bp total\n")
            f.write(f"- **Initial Capital:** ${self.config['initial_capital']:,.0f}\n\n")

            f.write(f"## Summary\n\n")
            f.write(f"- **Models Tested:** {len(self.results)}\n")

            if sorted_results:
                best = sorted_results[0]
                f.write(f"- **Best Performer (Test PF):** {best.model_name} (PF={best.test_results.profit_factor:.2f})\n")

                least_overfit = overfit_sorted[0]
                f.write(f"- **Least Overfit:** {least_overfit.model_name} (Overfit={least_overfit.overfit_score:.2f})\n\n")

            f.write(f"## Ranked Results\n\n")
            f.write(f"| Rank | Model | Train PF | Test PF | OOS PF | Overfit | Trades | Sharpe | Status |\n")
            f.write(f"|------|-------|----------|---------|--------|---------|--------|--------|--------|\n")

            for rank, model_result in enumerate(sorted_results, 1):
                train_pf = model_result.train_results.profit_factor
                test_pf = model_result.test_results.profit_factor
                oos_pf = model_result.oos_results.profit_factor
                overfit = model_result.overfit_score
                trades = model_result.test_results.total_trades
                sharpe = model_result.test_results.sharpe_ratio
                status = model_result.status_emoji

                f.write(f"| {rank} | {model_result.model_name} | {train_pf:.2f} | {test_pf:.2f} | {oos_pf:.2f} | {overfit:.2f} | {trades} | {sharpe:.2f} | {status} |\n")

            f.write(f"\n## Red Flags\n\n")
            if red_flags:
                for flag in red_flags:
                    f.write(f"{flag}\n")
            else:
                f.write("No red flags detected.\n")

            f.write(f"\n## Acceptance Criteria\n\n")
            criteria = self.config['acceptance_criteria']
            f.write(f"- **Min Test PF:** {criteria['min_test_pf']}\n")
            f.write(f"- **Min Test Sharpe:** {criteria['min_test_sharpe']}\n")
            f.write(f"- **Max Overfit:** {criteria['max_overfit']}\n")
            f.write(f"- **Min Trades:** {criteria['min_trades']}\n\n")

            # Check which models pass
            passing = []
            for model_result in self.results:
                if (model_result.test_results.profit_factor >= criteria['min_test_pf'] and
                    model_result.test_results.sharpe_ratio >= criteria['min_test_sharpe'] and
                    model_result.overfit_score <= criteria['max_overfit'] and
                    model_result.test_results.total_trades >= criteria['min_trades']):
                    passing.append(model_result.model_name)

            if passing:
                f.write(f"**Models Passing All Criteria:** {', '.join(passing)}\n\n")
            else:
                f.write(f"**Models Passing All Criteria:** None\n\n")

        print(f"Report saved to {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Professional Quant Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to experiment configuration JSON'
    )

    parser.add_argument(
        '--baselines-only',
        action='store_true',
        help='Run only baseline models (skip archetypes)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Custom output directory (overrides config)'
    )

    args = parser.parse_args()

    # Initialize suite
    suite = QuantSuite(config_path=args.config, verbose=args.verbose)

    # Disable archetypes if baselines-only
    if args.baselines_only:
        suite.config['archetype_config']['enabled'] = False

    # Run backtests
    suite.run()

    # Generate report
    suite.generate_report(output_dir=args.output)

    print("\nQuant Suite complete.")


if __name__ == '__main__':
    main()
