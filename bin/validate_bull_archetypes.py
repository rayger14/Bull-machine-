#!/usr/bin/env python3
"""
Step 2: Bull Archetype Standalone Validation

Validates 7 bull archetypes (A, B, C, G, H, K, L) using the same quant suite framework
used for baseline validation.

Protocol:
- Train: 2020-2022 (3 years)
- Test: 2023 (1 year)
- OOS: 2024 (1 year)
- Same slippage/fees as baseline validation
- Asset: BTC/USDT 1H

Success Criteria:
- Test PF ≥ 1.5
- At least 30-50 trades total (10-20/year minimum)
- No catastrophic OOS collapse (OOS PF > 0.8)
- Overfit check: Train PF - Test PF < 1.0

Bull Archetypes:
1. A - Spring/UTAD (wyckoff_spring_utad)
2. B - Order Block Retest (order_block_retest)
3. C - BOS/CHOCH Reversal (bos_choch_reversal)
4. G - Liquidity Sweep & Reclaim (liquidity_sweep_reclaim)
5. H - Trap Within Trend (trap_within_trend)
6. K - Wick Trap Moneytaur (wick_trap_moneytaur)
7. L - Fakeout Real Move (fakeout_real_move)
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.backtesting.engine import BacktestEngine, BacktestResults
from engine.models.base import BaseModel, Signal, Position


# Archetype metadata
BULL_ARCHETYPES = {
    'A': {
        'name': 'Spring/UTAD',
        'canonical': 'wyckoff_spring_utad',
        'config_key': 'spring',
        'letter': 'A',
        'description': 'PTI-based spring/UTAD reversal'
    },
    'B': {
        'name': 'Order Block Retest',
        'canonical': 'order_block_retest',
        'config_key': 'order_block_retest',
        'letter': 'B',
        'description': 'BOMS + Wyckoff structure retest'
    },
    'C': {
        'name': 'BOS/CHOCH Reversal',
        'canonical': 'bos_choch_reversal',
        'config_key': 'wick_trap',  # Name mismatch!
        'letter': 'C',
        'description': 'Displacement + momentum + recent BOS'
    },
    'G': {
        'name': 'Liquidity Sweep & Reclaim',
        'canonical': 'liquidity_sweep_reclaim',
        'config_key': 'liquidity_sweep',
        'letter': 'G',
        'description': 'BOMS strength + rising liquidity'
    },
    'H': {
        'name': 'Trap Within Trend',
        'canonical': 'trap_within_trend',
        'config_key': 'trap_within_trend',
        'letter': 'H',
        'description': 'HTF trend + low liquidity trap'
    },
    'K': {
        'name': 'Wick Trap (Moneytaur)',
        'canonical': 'wick_trap_moneytaur',
        'config_key': 'wick_trap_moneytaur',
        'letter': 'K',
        'description': 'Wick rejection + ADX + liquidity'
    },
    'L': {
        'name': 'Fakeout Real Move',
        'canonical': 'fakeout_real_move',
        'config_key': 'volume_exhaustion',  # Name mismatch!
        'letter': 'L',
        'description': 'Fakeout followed by real move'
    }
}


class ArchetypeModel(BaseModel):
    """
    Wrapper to make a single archetype compatible with BacktestEngine.

    This loads the full Bull Machine configuration but only enables ONE archetype
    at a time for isolated testing.
    """

    def __init__(self, archetype_letter: str, config_path: str):
        """
        Initialize archetype model.

        Args:
            archetype_letter: Archetype letter code (A, B, C, G, H, K, L)
            config_path: Path to bull market config JSON
        """
        self.archetype = BULL_ARCHETYPES[archetype_letter]
        self.letter = archetype_letter
        self.name = f"Archetype_{archetype_letter}_{self.archetype['canonical']}"
        self.config_path = config_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Disable all archetypes except this one
        for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M']:
            self.config['archetypes'][f'enable_{letter}'] = (letter == archetype_letter)

        # Disable bear archetypes
        for s_letter in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']:
            self.config['archetypes'][f'enable_{s_letter}'] = False

        self._is_fitted = False

        # Import the actual engine components
        from engine.context.regime_classifier import RegimeClassifier
        from engine.archetypes.logic_v2_adapter import ArchetypeDispatcher

        # Initialize regime classifier if enabled
        self.regime_classifier = None
        if self.config.get('regime_classifier', {}).get('enabled', True):
            try:
                model_path = PROJECT_ROOT / self.config['regime_classifier'].get('model_path', '')
                if model_path.exists():
                    self.regime_classifier = RegimeClassifier(str(model_path))
            except Exception as e:
                print(f"Warning: Could not load regime classifier: {e}")

        # Initialize archetype dispatcher
        self.dispatcher = ArchetypeDispatcher(self.config)

    def fit(self, data: pd.DataFrame) -> None:
        """Fit on training data."""
        self._is_fitted = True
        print(f"  Archetype {self.letter} ({self.archetype['name']}) ready")

    def predict(self, bar: pd.Series, position: Position = None) -> Signal:
        """
        Generate signal for current bar.

        Args:
            bar: Current OHLCV bar with features
            position: Current open position (if any)

        Returns:
            Signal object
        """
        # If already in position, check for exit
        if position is not None:
            # Simple trailing stop exit (ATR-based)
            if 'atr_14' in bar and pd.notna(bar['atr_14']):
                trail_mult = self.config['archetypes'].get('exits', {}).get(
                    self.archetype['config_key'], {}
                ).get('trail_atr', 1.5)

                stop_price = position.entry_price - (trail_mult * bar['atr_14'])

                if bar['close'] <= stop_price:
                    return Signal(
                        direction='hold',
                        confidence=0.0,
                        entry_price=0.0,
                        metadata={'reason': 'stop_loss', 'exit': True}
                    )

            # Time-based exit
            time_limit_hours = self.config['archetypes'].get('exits', {}).get(
                self.archetype['config_key'], {}
            ).get('time_limit_hours', 72)

            hours_in_trade = (bar.name - position.entry_time).total_seconds() / 3600
            if hours_in_trade >= time_limit_hours:
                return Signal(
                    direction='hold',
                    confidence=0.0,
                    entry_price=0.0,
                    metadata={'reason': 'time_limit', 'exit': True}
                )

            # Hold position
            return Signal(direction='hold', confidence=0.0, entry_price=0.0)

        # Check for entry signal using archetype logic
        try:
            # Build context object (simplified version)
            from types import SimpleNamespace

            context = SimpleNamespace(
                bar=bar,
                config=self.config,
                regime='risk_on',  # Assume bull market for validation
                fusion_score=0.5,  # Placeholder
                wyckoff_phase='accumulation',  # Placeholder
                liquidity_score=bar.get('liquidity_score', 0.5),
                boms_strength=bar.get('boms_strength', 0.5),
                adx=bar.get('adx_14', 25.0),
                rsi=bar.get('rsi_14', 50.0),
                atr=bar.get('atr_14', 500.0),
                volume_z=bar.get('volume_z', 0.0),
                displacement=bar.get('displacement', 0.0)
            )

            # Call archetype dispatcher
            archetype_name, confidence, metadata = self.dispatcher.dispatch(context)

            # Only accept signals from our target archetype
            if archetype_name == self.archetype['canonical'] or \
               archetype_name == self.archetype['config_key'] or \
               archetype_name == self.letter:

                # Get entry threshold
                fusion_threshold = self.config['archetypes']['thresholds'].get(
                    self.archetype['config_key'], {}
                ).get('fusion_threshold', 0.40)

                if confidence >= fusion_threshold:
                    # Calculate stop loss (ATR-based)
                    atr_stop_mult = self.config['archetypes']['thresholds'].get(
                        self.archetype['config_key'], {}
                    ).get('atr_stop_mult', 2.0)

                    stop_loss = bar['close'] - (atr_stop_mult * bar.get('atr_14', 500.0))

                    return Signal(
                        direction='long',
                        confidence=confidence,
                        entry_price=bar['close'],
                        stop_loss=stop_loss,
                        metadata=metadata
                    )

        except Exception as e:
            # Silently fail for missing features
            pass

        # No signal
        return Signal(direction='hold', confidence=0.0, entry_price=0.0)

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        """Calculate position size based on risk."""
        if signal.direction == 'hold':
            return 0.0

        # Get risk parameters
        risk_pct = self.config['archetypes']['thresholds'].get(
            self.archetype['config_key'], {}
        ).get('max_risk_pct', 0.02)

        # Capital (fixed for validation)
        capital = 10000.0

        # Risk amount in dollars
        risk_amount = capital * risk_pct

        # Position size based on stop distance
        if signal.stop_loss > 0:
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            if stop_distance > 0:
                position_size = risk_amount / stop_distance
                # Cap at 50% of capital
                max_size = capital * 0.5
                return min(position_size, max_size)

        # Default to 2% of capital
        return capital * 0.02

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'archetype_letter': self.letter,
            'archetype_name': self.archetype['name'],
            'canonical': self.archetype['canonical'],
            'config_key': self.archetype['config_key']
        }

    def get_state(self) -> Dict:
        """Get model state."""
        return {
            'is_fitted': self._is_fitted,
            'archetype': self.archetype
        }


@dataclass
class ArchetypeValidationResult:
    """Validation results for one archetype."""
    letter: str
    name: str
    train_pf: float
    test_pf: float
    oos_pf: float
    train_trades: int
    test_trades: int
    oos_trades: int
    trades_per_year: float
    overfit_score: float
    test_sharpe: float
    test_max_dd: float
    status: str

    @property
    def status_emoji(self) -> str:
        """Status emoji."""
        if self.status == 'PASS':
            return '✅'
        elif self.status == 'MARGINAL':
            return '🔧'
        else:
            return '❌'


def run_archetype_validation(
    archetype_letter: str,
    config_path: str,
    data: pd.DataFrame,
    periods: Dict[str, Dict[str, str]],
    total_costs_pct: float
) -> ArchetypeValidationResult:
    """
    Run validation for a single archetype.

    Args:
        archetype_letter: Archetype letter (A, B, C, G, H, K, L)
        config_path: Path to bull market config
        data: Full feature store data
        periods: Period definitions (train/test/oos)
        total_costs_pct: Total transaction costs (decimal)

    Returns:
        ArchetypeValidationResult
    """
    archetype = BULL_ARCHETYPES[archetype_letter]

    print(f"\n{'='*80}")
    print(f"VALIDATING ARCHETYPE {archetype_letter}: {archetype['name']}")
    print(f"{'='*80}")

    # Create model
    model = ArchetypeModel(archetype_letter, config_path)

    # Fit on train data
    train_period = periods['train']
    train_data = data[train_period['start']:train_period['end']]
    model.fit(train_data)

    # Run backtests
    results = {}
    for period_name, period_config in periods.items():
        print(f"\nRunning {period_name.upper()} backtest...")

        engine = BacktestEngine(
            model=model,
            data=data,
            initial_capital=10000.0,
            commission_pct=total_costs_pct
        )

        result = engine.run(
            start=period_config['start'],
            end=period_config['end'],
            verbose=False
        )

        results[period_name] = result

        print(f"  PF: {result.profit_factor:.2f}")
        print(f"  Trades: {result.total_trades}")
        print(f"  Win Rate: {result.win_rate:.1f}%")

    # Calculate metrics
    train_res = results['train']
    test_res = results['test']
    oos_res = results['oos']

    # Trades per year (test period is 1 year)
    trades_per_year = test_res.total_trades

    # Overfit score
    overfit_score = train_res.profit_factor - test_res.profit_factor

    # Determine status
    status = 'FAIL'
    if (test_res.profit_factor >= 1.5 and
        test_res.total_trades >= 30 and
        oos_res.profit_factor > 0.8 and
        overfit_score < 1.0):
        status = 'PASS'
    elif (test_res.profit_factor >= 1.3 and
          test_res.total_trades >= 20):
        status = 'MARGINAL'

    return ArchetypeValidationResult(
        letter=archetype_letter,
        name=archetype['name'],
        train_pf=train_res.profit_factor,
        test_pf=test_res.profit_factor,
        oos_pf=oos_res.profit_factor,
        train_trades=train_res.total_trades,
        test_trades=test_res.total_trades,
        oos_trades=oos_res.total_trades,
        trades_per_year=trades_per_year,
        overfit_score=overfit_score,
        test_sharpe=test_res.sharpe_ratio,
        test_max_dd=test_res.max_drawdown,
        status=status
    )


def print_results_table(results: List[ArchetypeValidationResult]) -> None:
    """Print formatted results table."""
    print(f"\n{'='*120}")
    print(f"BULL ARCHETYPE VALIDATION RESULTS")
    print(f"{'='*120}\n")

    # Header
    print(f"| {'Code':<6} | {'Name':<25} | {'Train PF':<10} | {'Test PF':<10} | {'OOS PF':<10} | {'T/Y':<6} | {'Status':<12} |")
    print(f"|{'-'*8}|{'-'*27}|{'-'*12}|{'-'*12}|{'-'*12}|{'-'*8}|{'-'*14}|")

    # Rows
    for r in results:
        print(f"| {r.letter:<6} | {r.name:<25} | {r.train_pf:<10.2f} | {r.test_pf:<10.2f} | {r.oos_pf:<10.2f} | {r.trades_per_year:<6.0f} | {r.status_emoji} {r.status:<9} |")

    print(f"\n{'='*120}\n")

    # Summary
    passed = [r for r in results if r.status == 'PASS']
    marginal = [r for r in results if r.status == 'MARGINAL']
    failed = [r for r in results if r.status == 'FAIL']

    print(f"SUMMARY:")
    print(f"- Passed: {len(passed)}/{len(results)} archetypes ({', '.join([r.letter for r in passed]) if passed else 'None'})")
    print(f"- Marginal: {len(marginal)}/{len(results)} archetypes ({', '.join([r.letter for r in marginal]) if marginal else 'None'})")
    print(f"- Failed: {len(failed)}/{len(results)} archetypes ({', '.join([r.letter for r in failed]) if failed else 'None'})")

    if passed:
        print(f"\nTier 2 Candidates for ML Ensemble: {', '.join([r.letter for r in passed])}")
    else:
        print(f"\nNo archetypes passed all criteria. Review thresholds and feature availability.")

    print()


def save_results(results: List[ArchetypeValidationResult], output_dir: Path) -> None:
    """Save results to CSV and markdown."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save CSV
    csv_path = output_dir / f'bull_archetype_validation_{timestamp}.csv'
    df = pd.DataFrame([
        {
            'letter': r.letter,
            'name': r.name,
            'train_pf': r.train_pf,
            'test_pf': r.test_pf,
            'oos_pf': r.oos_pf,
            'train_trades': r.train_trades,
            'test_trades': r.test_trades,
            'oos_trades': r.oos_trades,
            'trades_per_year': r.trades_per_year,
            'overfit_score': r.overfit_score,
            'test_sharpe': r.test_sharpe,
            'test_max_dd': r.test_max_dd,
            'status': r.status
        }
        for r in results
    ])
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Save markdown report
    md_path = output_dir / f'bull_archetype_validation_{timestamp}.md'
    with open(md_path, 'w') as f:
        f.write(f"# Bull Archetype Validation Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"## Validation Protocol\n\n")
        f.write(f"- **Train Period:** 2022 (1 year - bear market)\n")
        f.write(f"- **Test Period:** 2023 (1 year - transition/recovery)\n")
        f.write(f"- **OOS Period:** 2024 (1 year - bull market)\n")
        f.write(f"- **Asset:** BTC/USDT 1H\n")
        f.write(f"- **Costs:** 20bp total (10bp slippage + 10bp fees)\n")
        f.write(f"- **Note:** Bull archetypes are long-only, so 2022 bear market is challenging train period\n\n")

        f.write(f"## Results Summary\n\n")
        f.write(f"| Code | Name | Train PF | Test PF | OOS PF | T/Y | Overfit | Sharpe | MaxDD | Status |\n")
        f.write(f"|------|------|----------|---------|--------|-----|---------|--------|-------|--------|\n")

        for r in results:
            f.write(f"| {r.letter} | {r.name} | {r.train_pf:.2f} | {r.test_pf:.2f} | {r.oos_pf:.2f} | {r.trades_per_year:.0f} | {r.overfit_score:.2f} | {r.test_sharpe:.2f} | {r.test_max_dd:.1f}% | {r.status_emoji} {r.status} |\n")

        f.write(f"\n## Acceptance Criteria\n\n")
        f.write(f"- **Min Test PF:** 1.5\n")
        f.write(f"- **Min Trades:** 30 total (10-20/year)\n")
        f.write(f"- **OOS PF:** > 0.8 (no catastrophic collapse)\n")
        f.write(f"- **Max Overfit:** < 1.0 (Train PF - Test PF)\n\n")

        passed = [r for r in results if r.status == 'PASS']
        f.write(f"## Tier 2 Candidates\n\n")
        if passed:
            f.write(f"Archetypes passing all criteria: **{', '.join([r.letter for r in passed])}**\n\n")
            for r in passed:
                f.write(f"- **{r.letter} ({r.name})**: Test PF={r.test_pf:.2f}, {r.trades_per_year:.0f} trades/year\n")
        else:
            f.write(f"No archetypes passed all criteria.\n")

    print(f"Report saved to {md_path}")


def main():
    """Main entry point."""
    # Configuration
    config_path = PROJECT_ROOT / 'configs/mvp/mvp_bull_market_v1.json'
    data_path = PROJECT_ROOT / 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'
    output_dir = PROJECT_ROOT / 'results/archetype_validation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validation periods (adjusted for available data: 2022-2024)
    # Train: 2022 (1 year - bear market)
    # Test: 2023 (1 year - transition/recovery)
    # OOS: 2024 (1 year - bull market)
    periods = {
        'train': {'start': '2022-01-01', 'end': '2022-12-31'},
        'test': {'start': '2023-01-01', 'end': '2023-12-31'},
        'oos': {'start': '2024-01-01', 'end': '2024-12-31'}
    }

    # Transaction costs
    total_costs_bp = 20  # 10bp slippage + 10bp fees
    total_costs_pct = total_costs_bp / 10000

    print(f"{'='*80}")
    print(f"BULL ARCHETYPE STANDALONE VALIDATION")
    print(f"{'='*80}")
    print(f"Config: {config_path.name}")
    print(f"Data: {data_path.name}")
    print(f"Costs: {total_costs_bp}bp")
    print(f"{'='*80}")

    # Load data
    print(f"\nLoading feature store data...")
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print(f"Please ensure feature store data is available.")
        return 1

    data = pd.read_parquet(data_path)
    if 'timestamp' in data.columns:
        data.set_index('timestamp', inplace=True)
    data.index = pd.to_datetime(data.index)
    print(f"Loaded {len(data):,} bars ({data.index[0]} to {data.index[-1]})")

    # Validate date coverage
    for period_name, period_config in periods.items():
        start = pd.Timestamp(period_config['start'])
        end = pd.Timestamp(period_config['end'])
        period_data = data[(data.index >= start) & (data.index <= end)]
        print(f"  {period_name.upper()}: {len(period_data):,} bars")

    # Run validation for each archetype
    results = []
    for letter in ['A', 'B', 'C', 'G', 'H', 'K', 'L']:
        try:
            result = run_archetype_validation(
                archetype_letter=letter,
                config_path=str(config_path),
                data=data,
                periods=periods,
                total_costs_pct=total_costs_pct
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR validating archetype {letter}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print results
    if results:
        print_results_table(results)
        save_results(results, output_dir)
    else:
        print("\nNo results to report.")
        return 1

    print("\nValidation complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
