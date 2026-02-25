"""
Model comparison framework.

Compare multiple models on the same data with clear train/test separation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

from engine.models.base import BaseModel
from .engine import BacktestEngine, BacktestResults

logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """Multi-model comparison results."""

    models: List[str]
    train_results: Dict[str, BacktestResults]
    test_results: Dict[str, BacktestResults]
    train_period: Tuple[str, str]
    test_period: Tuple[str, str]

    def summary_table(self) -> pd.DataFrame:
        """
        Create comparison table with key metrics.

        Returns:
            DataFrame with models as rows, metrics as columns
        """
        rows = []
        for model_name in self.models:
            train_res = self.train_results[model_name]
            test_res = self.test_results[model_name]

            rows.append({
                'Model': model_name,
                'Train_PF': train_res.profit_factor,
                'Train_WR': train_res.win_rate,
                'Train_Trades': train_res.total_trades,
                'Train_PnL': train_res.total_pnl,
                'Test_PF': test_res.profit_factor,
                'Test_WR': test_res.win_rate,
                'Test_Trades': test_res.total_trades,
                'Test_PnL': test_res.total_pnl,
                'Overfit': train_res.profit_factor - test_res.profit_factor
            })

        df = pd.DataFrame(rows)
        return df.set_index('Model')

    def print_summary(self) -> None:
        """Print formatted comparison summary."""
        print("="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        print(f"\nTrain Period: {self.train_period[0]} to {self.train_period[1]}")
        print(f"Test Period:  {self.test_period[0]} to {self.test_period[1]}")
        print(f"\nModels Compared: {len(self.models)}")

        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        table = self.summary_table()
        print(table.to_string())

        print("\n" + "="*80)
        print("WINNER ANALYSIS")
        print("="*80)

        # Best test PF
        best_pf_idx = table['Test_PF'].idxmax()
        print(f"Best Test Profit Factor: {best_pf_idx} (PF={table.loc[best_pf_idx, 'Test_PF']:.2f})")

        # Best test WR
        best_wr_idx = table['Test_WR'].idxmax()
        print(f"Best Test Win Rate: {best_wr_idx} (WR={table.loc[best_wr_idx, 'Test_WR']:.1f}%)")

        # Least overfit
        least_overfit_idx = table['Overfit'].idxmin()
        print(f"Least Overfit: {least_overfit_idx} (Overfit={table.loc[least_overfit_idx, 'Overfit']:.2f})")

        print("\n" + "="*80)


class ModelComparison:
    """
    Framework for comparing multiple models on same data.

    Usage:
        comparison = ModelComparison(data)
        results = comparison.compare(
            models=[baseline, archetype_s1, archetype_s4],
            train_period=('2022-01-01', '2022-12-31'),
            test_period=('2023-01-01', '2023-12-31')
        )
        results.print_summary()
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 10000.0):
        """
        Initialize comparison framework.

        Args:
            data: Full historical dataset
            initial_capital: Starting portfolio value for all models
        """
        self.data = data
        self.initial_capital = initial_capital

    def compare(
        self,
        models: List[BaseModel],
        train_period: Tuple[str, str],
        test_period: Tuple[str, str],
        fit_on_train: bool = True,
        verbose: bool = True
    ) -> ComparisonReport:
        """
        Compare models with train/test split.

        Process:
            1. Fit each model on train period
            2. Backtest each model on train period (in-sample performance)
            3. Backtest each model on test period (out-of-sample performance)
            4. Compare results

        Args:
            models: List of models to compare
            train_period: (start, end) for training
            test_period: (start, end) for testing
            fit_on_train: If True, call model.fit() on train data
            verbose: Log progress

        Returns:
            ComparisonReport with train/test results for all models
        """
        if verbose:
            logger.info("="*60)
            logger.info("MULTI-MODEL COMPARISON")
            logger.info("="*60)
            logger.info(f"Train period: {train_period[0]} to {train_period[1]}")
            logger.info(f"Test period:  {test_period[0]} to {test_period[1]}")
            logger.info(f"Models: {len(models)}")
            for i, model in enumerate(models, 1):
                logger.info(f"  {i}. {model.name}")

        # Extract train data
        train_data = self.data[
            (self.data.index >= train_period[0]) &
            (self.data.index <= train_period[1])
        ]

        train_results = {}
        test_results = {}

        for model in models:
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {model.name}")
                logger.info(f"{'='*60}")

            # Step 1: Fit on train data
            if fit_on_train:
                if verbose:
                    logger.info("Fitting model on train data...")
                model.fit(train_data)

            # Step 2: Backtest on train period (in-sample)
            if verbose:
                logger.info("Running train backtest...")
            train_engine = BacktestEngine(
                model=model,
                data=self.data,
                initial_capital=self.initial_capital
            )
            train_bt = train_engine.run(
                start=train_period[0],
                end=train_period[1],
                verbose=False
            )
            train_results[model.name] = train_bt

            if verbose:
                logger.info(f"Train Results: PF={train_bt.profit_factor:.2f}, WR={train_bt.win_rate:.1f}%, Trades={train_bt.total_trades}")

            # Step 3: Backtest on test period (out-of-sample)
            if verbose:
                logger.info("Running test backtest...")
            test_engine = BacktestEngine(
                model=model,
                data=self.data,
                initial_capital=self.initial_capital
            )
            test_bt = test_engine.run(
                start=test_period[0],
                end=test_period[1],
                verbose=False
            )
            test_results[model.name] = test_bt

            if verbose:
                logger.info(f"Test Results: PF={test_bt.profit_factor:.2f}, WR={test_bt.win_rate:.1f}%, Trades={test_bt.total_trades}")
                overfit = train_bt.profit_factor - test_bt.profit_factor
                logger.info(f"Overfit: {overfit:+.2f} (train PF - test PF)")

        # Build comparison report
        report = ComparisonReport(
            models=[m.name for m in models],
            train_results=train_results,
            test_results=test_results,
            train_period=train_period,
            test_period=test_period
        )

        if verbose:
            logger.info("\n")
            report.print_summary()

        return report
