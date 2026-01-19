# QUANT LAB 72-HOUR EXECUTION GUIDE

**Mission:** Build a professional quant testing framework in 72 hours that treats all strategies as equals and lets evidence decide.

**Philosophy:** Everything is a model. Baselines set the bar. Evidence drives decisions. Kill fast, learn faster.

---

## OVERVIEW

**What You're Building:**
A systematic backtesting lab that runs Bull Machine archetypes against baseline strategies and makes objective keep/improve/kill decisions based on out-of-sample performance.

**Why This Matters:**
Without baselines, you can't tell if your complex archetypes are valuable or just overfitted noise. This lab gives you the truth.

**End State After 72 Hours:**
- Working baseline suite (5 models)
- Integrated archetype testing (3+ models)
- Unified comparison framework
- Clear deployment decisions
- Next experiment queue

---

## DAY 1: FRAMEWORK + BASELINES (8 HOURS)

**Goal:** Working baseline suite that runs end-to-end and produces ranked results.

### MORNING (4 HOURS): Framework Setup

**Objective:** Build the core engine that loads configs, runs models, calculates metrics, and outputs results.

#### Tasks

**1.1 Create Core Interfaces (45 min)**

Location: `engine/backtesting/base_model.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class Signal:
    """Trading signal at a specific timestamp."""
    timestamp: pd.Timestamp
    direction: str  # 'long', 'short', 'flat'
    size: float = 1.0  # Position size (0-1)
    metadata: Optional[Dict[str, Any]] = None

class BaseModel(ABC):
    """Base class for all tradeable models."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
               and DatetimeIndex

        Returns:
            DataFrame with columns [direction, size] and same index as df
            direction: 'long', 'short', 'flat'
            size: 0.0 to 1.0
        """
        pass

    @abstractmethod
    def get_param_hash(self) -> str:
        """Return unique hash of parameters for caching."""
        pass
```

Location: `engine/backtesting/backtest_results.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd

@dataclass
class BacktestResults:
    """Results from a single backtest run."""

    # Identity
    model_name: str
    period_name: str  # 'train', 'test', 'oos'
    start_date: str
    end_date: str

    # Performance Metrics
    profit_factor: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    sharpe_ratio: float

    # Additional Metrics
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0

    # Trade Details
    trades: Optional[pd.DataFrame] = None
    equity_curve: Optional[pd.DataFrame] = None

    # Metadata
    config: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            'model': self.model_name,
            'period': self.period_name,
            'start': self.start_date,
            'end': self.end_date,
            'profit_factor': round(self.profit_factor, 2),
            'win_rate': round(self.win_rate, 2),
            'trades': self.total_trades,
            'max_dd': round(self.max_drawdown, 2),
            'sharpe': round(self.sharpe_ratio, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'expectancy': round(self.expectancy, 2)
        }
```

**1.2 Create Backtest Engine (90 min)**

Location: `engine/backtesting/backtest_engine.py`

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .base_model import BaseModel
from .backtest_results import BacktestResults

class BacktestEngine:
    """Core backtesting engine with transaction costs."""

    def __init__(self,
                 slippage_bps: float = 5.0,
                 fee_bps: float = 6.0,
                 initial_capital: float = 100000.0):
        self.slippage_bps = slippage_bps
        self.fee_bps = fee_bps
        self.initial_capital = initial_capital
        self.total_cost_bps = slippage_bps + fee_bps

    def run(self,
            model: BaseModel,
            df: pd.DataFrame,
            period_name: str = 'test') -> BacktestResults:
        """
        Run backtest for a model on given data.

        Args:
            model: BaseModel instance
            df: OHLCV DataFrame with DatetimeIndex
            period_name: 'train', 'test', or 'oos'

        Returns:
            BacktestResults object
        """
        # Generate signals
        signals = model.generate_signals(df)

        # Calculate trades
        trades = self._signals_to_trades(signals, df)

        if len(trades) == 0:
            return self._empty_results(model.name, period_name, df)

        # Calculate metrics
        metrics = self._calculate_metrics(trades)

        # Build equity curve
        equity_curve = self._build_equity_curve(trades, df)

        return BacktestResults(
            model_name=model.name,
            period_name=period_name,
            start_date=df.index[0].strftime('%Y-%m-%d'),
            end_date=df.index[-1].strftime('%Y-%m-%d'),
            profit_factor=metrics['profit_factor'],
            win_rate=metrics['win_rate'],
            total_trades=len(trades),
            max_drawdown=metrics['max_drawdown'],
            sharpe_ratio=metrics['sharpe_ratio'],
            avg_win=metrics['avg_win'],
            avg_loss=metrics['avg_loss'],
            expectancy=metrics['expectancy'],
            trades=trades,
            equity_curve=equity_curve,
            config={'slippage_bps': self.slippage_bps, 'fee_bps': self.fee_bps}
        )

    def _signals_to_trades(self, signals: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to trades with entry/exit prices."""
        trades = []
        position = None

        for i in range(len(signals)):
            current_signal = signals.iloc[i]
            direction = current_signal['direction']

            if position is None and direction != 'flat':
                # Enter position
                position = {
                    'entry_time': signals.index[i],
                    'entry_price': df.iloc[i]['close'],
                    'direction': direction,
                    'size': current_signal['size']
                }

            elif position is not None:
                # Check for exit
                if direction == 'flat' or direction != position['direction']:
                    # Exit position
                    exit_price = df.iloc[i]['close']
                    entry_price = position['entry_price']

                    # Calculate P&L with costs
                    if position['direction'] == 'long':
                        gross_pnl_pct = (exit_price - entry_price) / entry_price
                    else:  # short
                        gross_pnl_pct = (entry_price - exit_price) / entry_price

                    # Apply transaction costs (entry + exit)
                    net_pnl_pct = gross_pnl_pct - (2 * self.total_cost_bps / 10000)

                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': signals.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': position['direction'],
                        'pnl_pct': net_pnl_pct,
                        'gross_pnl_pct': gross_pnl_pct
                    })

                    # Open new position if signal is not flat
                    if direction != 'flat':
                        position = {
                            'entry_time': signals.index[i],
                            'entry_price': exit_price,
                            'direction': direction,
                            'size': current_signal['size']
                        }
                    else:
                        position = None

        return pd.DataFrame(trades)

    def _calculate_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics from trades."""
        if len(trades) == 0:
            return self._empty_metrics()

        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] < 0]

        total_win = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 0

        profit_factor = total_win / total_loss if total_loss > 0 else (10.0 if total_win > 0 else 0.0)
        win_rate = len(wins) / len(trades) if len(trades) > 0 else 0.0

        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0.0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0.0

        expectancy = trades['pnl_pct'].mean()

        # Calculate max drawdown from equity curve
        equity = (1 + trades['pnl_pct']).cumprod()
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Sharpe ratio (annualized, assuming trades are uncorrelated)
        sharpe_ratio = (expectancy / trades['pnl_pct'].std() * np.sqrt(365)) if trades['pnl_pct'].std() > 0 else 0.0

        return {
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }

    def _build_equity_curve(self, trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Build equity curve aligned with price data."""
        equity = pd.DataFrame(index=df.index)
        equity['equity'] = self.initial_capital

        for _, trade in trades.iterrows():
            mask = equity.index >= trade['exit_time']
            equity.loc[mask, 'equity'] *= (1 + trade['pnl_pct'])

        return equity

    def _empty_results(self, model_name: str, period_name: str, df: pd.DataFrame) -> BacktestResults:
        """Return empty results for models with no trades."""
        return BacktestResults(
            model_name=model_name,
            period_name=period_name,
            start_date=df.index[0].strftime('%Y-%m-%d'),
            end_date=df.index[-1].strftime('%Y-%m-%d'),
            **self._empty_metrics()
        )

    def _empty_metrics(self) -> Dict:
        """Return zero metrics."""
        return {
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'expectancy': 0.0
        }
```

**1.3 Create Experiment Config Schema (30 min)**

Location: `configs/experiment_btc_1h_2020_2025.json`

```json
{
  "experiment_name": "BTC_1H_2020_2025_Baseline",
  "asset": "BTC/USDT",
  "timeframe": "1h",
  "data_source": "features/btc_features_2020_2025.csv",

  "periods": {
    "train": {
      "start": "2020-01-01",
      "end": "2022-12-31"
    },
    "test": {
      "start": "2023-01-01",
      "end": "2023-12-31"
    },
    "oos": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  },

  "backtest_config": {
    "initial_capital": 100000,
    "slippage_bps": 5.0,
    "fee_bps": 6.0
  },

  "baselines": [
    {
      "name": "B0_BuyHold",
      "class": "BuyAndHoldModel",
      "params": {}
    },
    {
      "name": "B1_SMA200",
      "class": "SMA200TrendModel",
      "params": {"period": 200}
    },
    {
      "name": "B2_SMACross",
      "class": "SMACrossoverModel",
      "params": {"fast": 50, "slow": 200}
    },
    {
      "name": "B3_RSI",
      "class": "RSIMeanReversionModel",
      "params": {"period": 14, "oversold": 30, "overbought": 70}
    },
    {
      "name": "B4_VolTarget",
      "class": "VolTargetTrendModel",
      "params": {"lookback": 100, "target_vol": 0.15}
    }
  ],

  "archetypes": [
    {
      "name": "S4_FundingDivergence",
      "class": "ArchetypeWrapper",
      "params": {"archetype": "funding_divergence"}
    },
    {
      "name": "S1_V2_LiquidityVacuum",
      "class": "ArchetypeWrapper",
      "params": {"archetype": "liquidity_vacuum_v2"}
    },
    {
      "name": "S5_LongSqueeze",
      "class": "ArchetypeWrapper",
      "params": {"archetype": "long_squeeze"}
    }
  ],

  "output": {
    "results_dir": "results/quant_suite",
    "save_trades": true,
    "save_equity_curves": true
  }
}
```

**1.4 Create Runner Script Skeleton (45 min)**

Location: `bin/run_quant_suite.py`

```python
#!/usr/bin/env python3
"""
Quant Lab Baseline Suite Runner

Usage:
    python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only
    python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.backtesting.backtest_engine import BacktestEngine
from engine.backtesting.baseline_models import MODEL_REGISTRY

def load_config(config_path: str) -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_data(data_source: str) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(data_source, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Data must contain columns: {required}")

    return df[required]

def run_experiment(config_path: str, baselines_only: bool = False):
    """Run full experiment suite."""

    print("=" * 80)
    print("QUANT LAB BASELINE SUITE")
    print("=" * 80)

    # Load config
    config = load_config(config_path)
    print(f"\nExperiment: {config['experiment_name']}")
    print(f"Asset: {config['asset']} @ {config['timeframe']}")

    # Load data
    print(f"\nLoading data from {config['data_source']}...")
    df = load_data(config['data_source'])
    print(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")

    # Split data into periods
    periods = {}
    for period_name, period_config in config['periods'].items():
        start = pd.Timestamp(period_config['start'])
        end = pd.Timestamp(period_config['end'])
        periods[period_name] = df[(df.index >= start) & (df.index <= end)].copy()
        print(f"  {period_name}: {len(periods[period_name])} rows")

    # Initialize backtest engine
    engine = BacktestEngine(
        slippage_bps=config['backtest_config']['slippage_bps'],
        fee_bps=config['backtest_config']['fee_bps'],
        initial_capital=config['backtest_config']['initial_capital']
    )

    # Run baselines
    print("\n" + "=" * 80)
    print("RUNNING BASELINES")
    print("=" * 80)

    all_results = []
    models_to_run = config['baselines']

    if not baselines_only:
        models_to_run.extend(config.get('archetypes', []))

    for model_config in models_to_run:
        model_name = model_config['name']
        model_class = MODEL_REGISTRY.get(model_config['class'])

        if model_class is None:
            print(f"\nWARNING: Model class '{model_config['class']}' not found. Skipping {model_name}")
            continue

        print(f"\n--- {model_name} ---")

        # Instantiate model
        model = model_class(name=model_name, params=model_config['params'])

        # Run on each period
        for period_name, period_df in periods.items():
            print(f"  Running {period_name}...", end=' ')
            result = engine.run(model, period_df, period_name)
            all_results.append(result)
            print(f"PF={result.profit_factor:.2f}, WR={result.win_rate:.2f}, Trades={result.total_trades}")

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    output_dir = Path(config['output']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save CSV
    results_df = pd.DataFrame([r.to_dict() for r in all_results])
    csv_path = output_dir / f"results_{timestamp}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Create latest symlink
    latest_path = output_dir / "results_LATEST.csv"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(csv_path.name)

    # Generate report
    report_path = output_dir / f"report_{timestamp}.md"
    generate_report(results_df, report_path, config)
    print(f"Saved: {report_path}")

    # Create latest symlink
    latest_report = output_dir / "report_LATEST.md"
    if latest_report.exists():
        latest_report.unlink()
    latest_report.symlink_to(report_path.name)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nView results: cat {latest_path}")
    print(f"View report:  cat {latest_report}")

def generate_report(results_df: pd.DataFrame, output_path: Path, config: dict):
    """Generate markdown report."""

    with open(output_path, 'w') as f:
        f.write(f"# Quant Lab Results: {config['experiment_name']}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Asset:** {config['asset']} @ {config['timeframe']}\n\n")

        f.write("## Period Definitions\n\n")
        for period_name, period_config in config['periods'].items():
            f.write(f"- **{period_name}:** {period_config['start']} to {period_config['end']}\n")

        f.write("\n## Results by Period\n\n")

        for period in ['train', 'test', 'oos']:
            period_results = results_df[results_df['period'] == period].copy()

            if len(period_results) == 0:
                continue

            f.write(f"### {period.upper()} Period\n\n")

            # Sort by profit factor
            period_results = period_results.sort_values('profit_factor', ascending=False)

            # Format table
            f.write("| Rank | Model | PF | WR | Trades | Sharpe | Max DD |\n")
            f.write("|------|-------|----|----|--------|--------|--------|\n")

            for idx, row in enumerate(period_results.itertuples(), 1):
                f.write(f"| {idx} | {row.model} | {row.profit_factor:.2f} | {row.win_rate:.2f} | {row.trades} | {row.sharpe:.2f} | {row.max_dd:.2f} |\n")

            f.write("\n")

        f.write("## Overfit Analysis\n\n")
        f.write("| Model | Train PF | Test PF | Overfit Score |\n")
        f.write("|-------|----------|---------|---------------|\n")

        models = results_df['model'].unique()
        for model in models:
            model_results = results_df[results_df['model'] == model]
            train_pf = model_results[model_results['period'] == 'train']['profit_factor'].values
            test_pf = model_results[model_results['period'] == 'test']['profit_factor'].values

            if len(train_pf) > 0 and len(test_pf) > 0:
                train_pf = train_pf[0]
                test_pf = test_pf[0]
                overfit = (train_pf - test_pf) / train_pf if train_pf > 0 else 0
                f.write(f"| {model} | {train_pf:.2f} | {test_pf:.2f} | {overfit:.2f} |\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Quant Lab baseline suite')
    parser.add_argument('--config', required=True, help='Path to experiment config JSON')
    parser.add_argument('--baselines-only', action='store_true', help='Run only baseline models')

    args = parser.parse_args()

    run_experiment(args.config, args.baselines_only)
```

**1.5 Unit Tests for Core Components (30 min)**

Location: `tests/unit/backtesting/test_backtest_engine.py`

```python
import pytest
import pandas as pd
import numpy as np
from engine.backtesting.backtest_engine import BacktestEngine
from engine.backtesting.base_model import BaseModel

class DummyModel(BaseModel):
    """Simple buy-and-hold model for testing."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'long'
        signals['size'] = 1.0
        return signals

    def get_param_hash(self) -> str:
        return "dummy_v1"

def test_backtest_engine_initialization():
    """Test engine initializes with correct parameters."""
    engine = BacktestEngine(slippage_bps=5.0, fee_bps=6.0)
    assert engine.slippage_bps == 5.0
    assert engine.fee_bps == 6.0
    assert engine.total_cost_bps == 11.0

def test_backtest_engine_run():
    """Test engine runs and produces results."""
    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    model = DummyModel(name="TestModel", params={})
    engine = BacktestEngine()

    result = engine.run(model, df, 'test')

    assert result.model_name == "TestModel"
    assert result.period_name == "test"
    assert result.total_trades >= 0
```

#### Validation Checkpoint

**Run these commands to validate Morning work:**

```bash
# Test imports
python -c "from engine.backtesting.base_model import BaseModel; print('✓ BaseModel')"
python -c "from engine.backtesting.backtest_engine import BacktestEngine; print('✓ BacktestEngine')"
python -c "from engine.backtesting.backtest_results import BacktestResults; print('✓ BacktestResults')"

# Run unit tests
pytest tests/unit/backtesting/ -v

# Validate config
python -c "import json; json.load(open('configs/experiment_btc_1h_2020_2025.json')); print('✓ Config valid')"
```

**Success Criteria:**
- All imports work without errors
- Unit tests pass
- Config loads successfully

---

### AFTERNOON (4 HOURS): Baseline Implementation

**Objective:** Implement 5 baseline models that represent different market hypotheses.

#### Tasks

**2.1 Create Baseline Model Registry (30 min)**

Location: `engine/backtesting/baseline_models.py`

```python
"""
Baseline Models Registry

These models represent fundamental market hypotheses:
- B0: Market goes up (buy and hold)
- B1: Trend persistence (SMA200)
- B2: Momentum cycles (SMA crossover)
- B3: Mean reversion (RSI)
- B4: Volatility-adjusted trend (vol targeting)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import hashlib
import json
from .base_model import BaseModel

class BuyAndHoldModel(BaseModel):
    """B0: Pure buy and hold."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'long'
        signals['size'] = 1.0
        return signals

    def get_param_hash(self) -> str:
        return "buyhold_v1"

class SMA200TrendModel(BaseModel):
    """B1: Long when price > SMA200, flat otherwise."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get('period', 200)

        sma = df['close'].rolling(window=period).mean()

        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'flat'
        signals.loc[df['close'] > sma, 'direction'] = 'long'
        signals['size'] = 1.0

        return signals

    def get_param_hash(self) -> str:
        return f"sma200_p{self.params.get('period', 200)}"

class SMACrossoverModel(BaseModel):
    """B2: Long when fast SMA > slow SMA, flat otherwise."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        fast_period = self.params.get('fast', 50)
        slow_period = self.params.get('slow', 200)

        fast_sma = df['close'].rolling(window=fast_period).mean()
        slow_sma = df['close'].rolling(window=slow_period).mean()

        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'flat'
        signals.loc[fast_sma > slow_sma, 'direction'] = 'long'
        signals['size'] = 1.0

        return signals

    def get_param_hash(self) -> str:
        return f"smacross_f{self.params.get('fast', 50)}_s{self.params.get('slow', 200)}"

class RSIMeanReversionModel(BaseModel):
    """B3: Long when RSI < oversold, flat when RSI > overbought."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get('period', 14)
        oversold = self.params.get('oversold', 30)
        overbought = self.params.get('overbought', 70)

        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'flat'

        # Enter long on oversold
        signals.loc[rsi < oversold, 'direction'] = 'long'

        # Exit on overbought
        signals.loc[rsi > overbought, 'direction'] = 'flat'

        # Forward fill to maintain position
        signals['direction'] = signals['direction'].replace('flat', np.nan)
        signals['direction'] = signals['direction'].fillna(method='ffill').fillna('flat')

        signals['size'] = 1.0

        return signals

    def get_param_hash(self) -> str:
        p = self.params
        return f"rsi_p{p.get('period', 14)}_os{p.get('oversold', 30)}_ob{p.get('overbought', 70)}"

class VolTargetTrendModel(BaseModel):
    """B4: SMA trend with position sizing based on volatility targeting."""

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        lookback = self.params.get('lookback', 100)
        target_vol = self.params.get('target_vol', 0.15)  # 15% annualized

        # Calculate realized volatility
        returns = df['close'].pct_change()
        realized_vol = returns.rolling(window=lookback).std() * np.sqrt(365 * 24)  # Annualized for hourly

        # Calculate position size
        position_size = (target_vol / realized_vol).clip(0, 2.0)  # Cap at 2x leverage

        # Trend filter (simple SMA)
        sma = df['close'].rolling(window=200).mean()

        signals = pd.DataFrame(index=df.index)
        signals['direction'] = 'flat'
        signals.loc[df['close'] > sma, 'direction'] = 'long'
        signals['size'] = position_size
        signals['size'] = signals['size'].fillna(0)

        return signals

    def get_param_hash(self) -> str:
        p = self.params
        return f"voltarget_lb{p.get('lookback', 100)}_tv{p.get('target_vol', 0.15)}"

# Registry for dynamic loading
MODEL_REGISTRY = {
    'BuyAndHoldModel': BuyAndHoldModel,
    'SMA200TrendModel': SMA200TrendModel,
    'SMACrossoverModel': SMACrossoverModel,
    'RSIMeanReversionModel': RSIMeanReversionModel,
    'VolTargetTrendModel': VolTargetTrendModel
}
```

**2.2 Unit Tests for Each Baseline (90 min)**

Location: `tests/unit/backtesting/test_baseline_models.py`

```python
import pytest
import pandas as pd
import numpy as np
from engine.backtesting.baseline_models import (
    BuyAndHoldModel,
    SMA200TrendModel,
    SMACrossoverModel,
    RSIMeanReversionModel,
    VolTargetTrendModel
)

@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data."""
    dates = pd.date_range('2020-01-01', periods=500, freq='1H')
    df = pd.DataFrame({
        'open': np.random.randn(500).cumsum() + 100,
        'high': np.random.randn(500).cumsum() + 101,
        'low': np.random.randn(500).cumsum() + 99,
        'close': np.random.randn(500).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 500)
    }, index=dates)
    return df

def test_buyhold_model(sample_ohlcv):
    """Test buy and hold model."""
    model = BuyAndHoldModel(name="B0_BuyHold", params={})
    signals = model.generate_signals(sample_ohlcv)

    assert len(signals) == len(sample_ohlcv)
    assert (signals['direction'] == 'long').all()
    assert (signals['size'] == 1.0).all()

def test_sma200_model(sample_ohlcv):
    """Test SMA200 trend model."""
    model = SMA200TrendModel(name="B1_SMA200", params={'period': 200})
    signals = model.generate_signals(sample_ohlcv)

    assert len(signals) == len(sample_ohlcv)
    assert set(signals['direction'].unique()).issubset({'long', 'flat'})
    assert (signals['size'] == 1.0).all()

def test_sma_crossover_model(sample_ohlcv):
    """Test SMA crossover model."""
    model = SMACrossoverModel(name="B2_SMACross", params={'fast': 50, 'slow': 200})
    signals = model.generate_signals(sample_ohlcv)

    assert len(signals) == len(sample_ohlcv)
    assert set(signals['direction'].unique()).issubset({'long', 'flat'})

def test_rsi_model(sample_ohlcv):
    """Test RSI mean reversion model."""
    model = RSIMeanReversionModel(name="B3_RSI", params={'period': 14, 'oversold': 30, 'overbought': 70})
    signals = model.generate_signals(sample_ohlcv)

    assert len(signals) == len(sample_ohlcv)
    assert set(signals['direction'].unique()).issubset({'long', 'flat'})

def test_voltarget_model(sample_ohlcv):
    """Test volatility-targeted model."""
    model = VolTargetTrendModel(name="B4_VolTarget", params={'lookback': 100, 'target_vol': 0.15})
    signals = model.generate_signals(sample_ohlcv)

    assert len(signals) == len(sample_ohlcv)
    assert set(signals['direction'].unique()).issubset({'long', 'flat'})
    assert (signals['size'] >= 0).all()
    assert (signals['size'] <= 2.0).all()
```

**2.3 Create Test Data (60 min)**

You'll need to extract a CSV with OHLCV data. If you don't have this ready:

```python
# bin/prepare_test_data.py
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.db import get_engine

def export_ohlcv_for_testing():
    """Export OHLCV data for baseline testing."""

    engine = get_engine()

    query = """
    SELECT
        timestamp,
        open,
        high,
        low,
        close,
        volume
    FROM ohlcv
    WHERE symbol = 'BTC/USDT'
      AND timeframe = '1h'
      AND timestamp >= '2020-01-01'
      AND timestamp <= '2024-12-31'
    ORDER BY timestamp
    """

    df = pd.read_sql(query, engine)

    output_path = Path('features/btc_ohlcv_2020_2025.csv')
    output_path.parent.mkdir(exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} rows to {output_path}")

if __name__ == '__main__':
    export_ohlcv_for_testing()
```

**2.4 Integration Test (60 min)**

Run the full suite:

```bash
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only
```

This should produce:
- CSV with results for all 5 baselines across 3 periods (15 rows)
- Markdown report with rankings

#### Validation Checkpoint

**Success Criteria for Day 1:**

```bash
# All tests pass
pytest tests/unit/backtesting/ -v

# Suite runs without errors
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --baselines-only

# Results exist
cat results/quant_suite/results_LATEST.csv
cat results/quant_suite/report_LATEST.md

# CSV has 15 rows (5 models × 3 periods)
wc -l results/quant_suite/results_LATEST.csv  # Should be 16 (15 + header)
```

**Expected Output Example:**

```
model,period,start,end,profit_factor,win_rate,trades,max_dd,sharpe
B0_BuyHold,train,2020-01-01,2022-12-31,1.52,0.00,1,0.35,0.42
B1_SMA200,train,2020-01-01,2022-12-31,2.14,0.58,23,0.18,1.23
B2_SMACross,train,2020-01-01,2022-12-31,1.87,0.55,18,0.22,0.98
B3_RSI,train,2020-01-01,2022-12-31,1.45,0.52,67,0.28,0.75
B4_VolTarget,train,2020-01-01,2022-12-31,1.92,0.57,21,0.19,1.15
...
```

**End of Day 1:** You now have a working baseline suite that can evaluate any model.

---

## DAY 2: BULL MACHINE INTEGRATION (8 HOURS)

**Goal:** Run archetypes through the same rigorous testing and compare against baselines.

### MORNING (4 HOURS): Archetype Backtests

**Objective:** Extract performance metrics for S4, S1 V2, and S5 using consistent train/test/OOS periods.

#### Tasks

**3.1 Run S4 Funding Divergence (60 min)**

```bash
# Use existing backtest script with period constraints
python bin/backtest.py \
  --config configs/s4_funding_divergence.json \
  --start 2020-01-01 \
  --end 2022-12-31 \
  --output results/archetypes/s4_train.json

python bin/backtest.py \
  --config configs/s4_funding_divergence.json \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --output results/archetypes/s4_test.json

python bin/backtest.py \
  --config configs/s4_funding_divergence.json \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output results/archetypes/s4_oos.json
```

Extract metrics:

```python
# bin/extract_archetype_metrics.py
import json
import pandas as pd
from pathlib import Path

def extract_metrics(result_path: Path) -> dict:
    """Extract key metrics from archetype backtest result."""
    with open(result_path, 'r') as f:
        result = json.load(f)

    trades = result.get('trades', [])

    if len(trades) == 0:
        return {
            'profit_factor': 0,
            'win_rate': 0,
            'trades': 0,
            'max_dd': 0,
            'sharpe': 0
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] < 0]

    total_win = sum(t['pnl'] for t in wins)
    total_loss = abs(sum(t['pnl'] for t in losses))

    pf = total_win / total_loss if total_loss > 0 else (10 if total_win > 0 else 0)
    wr = len(wins) / len(trades) if trades else 0

    # Calculate max DD from equity curve
    equity = [100000]
    for t in trades:
        equity.append(equity[-1] + t['pnl'])

    equity_series = pd.Series(equity)
    running_max = equity_series.expanding().max()
    dd = (equity_series - running_max) / running_max
    max_dd = abs(dd.min())

    # Simple Sharpe
    pnls = [t['pnl'] for t in trades]
    sharpe = (pd.Series(pnls).mean() / pd.Series(pnls).std() * (len(trades) ** 0.5)) if len(pnls) > 1 else 0

    return {
        'profit_factor': round(pf, 2),
        'win_rate': round(wr, 2),
        'trades': len(trades),
        'max_dd': round(max_dd, 2),
        'sharpe': round(sharpe, 2)
    }

def create_archetype_csv():
    """Create CSV with archetype metrics."""
    archetypes = [
        ('S4_FundingDivergence', 'results/archetypes/s4'),
        ('S1_V2_LiquidityVacuum', 'results/archetypes/s1_v2'),
        ('S5_LongSqueeze', 'results/archetypes/s5')
    ]

    rows = []

    for archetype_name, base_path in archetypes:
        for period in ['train', 'test', 'oos']:
            result_path = Path(f"{base_path}_{period}.json")

            if not result_path.exists():
                print(f"WARNING: Missing {result_path}")
                continue

            metrics = extract_metrics(result_path)

            rows.append({
                'model': archetype_name,
                'period': period,
                **metrics
            })

    df = pd.DataFrame(rows)
    output_path = Path('results/archetypes/archetype_metrics.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Created {output_path}")
    print(df)

if __name__ == '__main__':
    create_archetype_csv()
```

**3.2 Run S1 V2 Liquidity Vacuum (60 min)**

Same process as S4:

```bash
python bin/backtest.py --config configs/s1_v2_liquidity_vacuum.json --start 2020-01-01 --end 2022-12-31 --output results/archetypes/s1_v2_train.json
python bin/backtest.py --config configs/s1_v2_liquidity_vacuum.json --start 2023-01-01 --end 2023-12-31 --output results/archetypes/s1_v2_test.json
python bin/backtest.py --config configs/s1_v2_liquidity_vacuum.json --start 2024-01-01 --end 2024-12-31 --output results/archetypes/s1_v2_oos.json
```

**3.3 Run S5 Long Squeeze (60 min)**

```bash
python bin/backtest.py --config configs/s5_long_squeeze.json --start 2020-01-01 --end 2022-12-31 --output results/archetypes/s5_train.json
python bin/backtest.py --config configs/s5_long_squeeze.json --start 2023-01-01 --end 2023-12-31 --output results/archetypes/s5_test.json
python bin/backtest.py --config configs/s5_long_squeeze.json --start 2024-01-01 --end 2024-12-31 --output results/archetypes/s5_oos.json
```

**3.4 Consolidate Metrics (60 min)**

Run the extraction script:

```bash
python bin/extract_archetype_metrics.py
```

Verify output:

```bash
cat results/archetypes/archetype_metrics.csv
```

Expected format:

```
model,period,profit_factor,win_rate,trades,max_dd,sharpe
S4_FundingDivergence,train,3.25,0.68,45,0.12,2.14
S4_FundingDivergence,test,2.87,0.65,18,0.15,1.89
S4_FundingDivergence,oos,2.45,0.61,12,0.18,1.56
...
```

#### Validation Checkpoint

**Success Criteria:**
- All 9 backtest runs complete (3 archetypes × 3 periods)
- CSV contains 9 rows with valid metrics
- No missing data or errors

---

### AFTERNOON (4 HOURS): Unified Comparison

**Objective:** Merge baseline and archetype results into a single ranked comparison table.

#### Tasks

**4.1 Merge Results (45 min)**

```python
# bin/merge_baseline_archetype_results.py
import pandas as pd
from pathlib import Path

def merge_results():
    """Merge baseline and archetype results into unified comparison."""

    # Load baseline results
    baseline_df = pd.read_csv('results/quant_suite/results_LATEST.csv')

    # Load archetype results
    archetype_df = pd.read_csv('results/archetypes/archetype_metrics.csv')

    # Ensure consistent columns
    baseline_df = baseline_df[['model', 'period', 'profit_factor', 'win_rate', 'trades', 'max_dd', 'sharpe']]
    archetype_df = archetype_df[['model', 'period', 'profit_factor', 'win_rate', 'trades', 'max_dd', 'sharpe']]

    # Add model type
    baseline_df['type'] = 'baseline'
    archetype_df['type'] = 'archetype'

    # Merge
    unified_df = pd.concat([baseline_df, archetype_df], ignore_index=True)

    # Save
    output_path = Path('results/unified_comparison.csv')
    unified_df.to_csv(output_path, index=False)

    print(f"Created {output_path}")
    print(f"Total models: {unified_df['model'].nunique()}")
    print(f"Total rows: {len(unified_df)}")

    return unified_df

if __name__ == '__main__':
    merge_results()
```

**4.2 Calculate Overfit Scores (45 min)**

```python
# bin/calculate_overfit_scores.py
import pandas as pd

def calculate_overfit():
    """Calculate overfit scores for all models."""

    df = pd.read_csv('results/unified_comparison.csv')

    overfit_scores = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]

        train_pf = model_df[model_df['period'] == 'train']['profit_factor'].values
        test_pf = model_df[model_df['period'] == 'test']['profit_factor'].values

        if len(train_pf) > 0 and len(test_pf) > 0:
            train_pf = train_pf[0]
            test_pf = test_pf[0]

            # Overfit score: (Train_PF - Test_PF) / Train_PF
            overfit_score = (train_pf - test_pf) / train_pf if train_pf > 0 else 0

            overfit_scores.append({
                'model': model,
                'train_pf': train_pf,
                'test_pf': test_pf,
                'overfit_score': round(overfit_score, 2),
                'type': model_df.iloc[0]['type']
            })

    overfit_df = pd.DataFrame(overfit_scores)
    overfit_df = overfit_df.sort_values('overfit_score')

    output_path = 'results/overfit_analysis.csv'
    overfit_df.to_csv(output_path, index=False)

    print(f"Created {output_path}")
    print("\nOverfit Rankings (lower is better):")
    print(overfit_df)

    return overfit_df

if __name__ == '__main__':
    calculate_overfit()
```

**4.3 Create Unified Ranking (90 min)**

```python
# bin/create_unified_ranking.py
import pandas as pd
from pathlib import Path

def create_ranking():
    """Create unified ranking table."""

    df = pd.read_csv('results/unified_comparison.csv')
    overfit_df = pd.read_csv('results/overfit_analysis.csv')

    # Pivot to get wide format
    pivot = df.pivot_table(
        index='model',
        columns='period',
        values=['profit_factor', 'win_rate', 'trades']
    )

    # Flatten column names
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    # Merge with overfit scores
    ranking = pivot.merge(overfit_df[['model', 'overfit_score', 'type']], on='model')

    # Sort by test PF
    ranking = ranking.sort_values('profit_factor_test', ascending=False)

    # Add rank
    ranking.insert(0, 'rank', range(1, len(ranking) + 1))

    # Save
    output_path = Path('results/unified_ranking.csv')
    ranking.to_csv(output_path, index=False)

    print(f"Created {output_path}")
    print("\nTop 10 Models by Test PF:")
    print(ranking.head(10))

    # Generate markdown report
    generate_markdown_report(ranking)

    return ranking

def generate_markdown_report(ranking: pd.DataFrame):
    """Generate markdown report."""

    output_path = Path('results/unified_ranking_report.md')

    with open(output_path, 'w') as f:
        f.write("# Unified Model Ranking Report\n\n")
        f.write("## Rankings by Test Period Profit Factor\n\n")

        f.write("| Rank | Model | Type | Train PF | Test PF | OOS PF | Overfit | Test Trades |\n")
        f.write("|------|-------|------|----------|---------|--------|---------|-------------|\n")

        for _, row in ranking.iterrows():
            f.write(f"| {row['rank']} | {row['model']} | {row['type']} | ")
            f.write(f"{row['profit_factor_train']:.2f} | {row['profit_factor_test']:.2f} | ")
            f.write(f"{row['profit_factor_oos']:.2f} | {row['overfit_score']:.2f} | ")
            f.write(f"{int(row['trades_test'])} |\n")

        f.write("\n## Red Flags\n\n")

        # High overfit
        high_overfit = ranking[ranking['overfit_score'] > 0.5]
        if len(high_overfit) > 0:
            f.write("### High Overfit (>0.5)\n\n")
            for model in high_overfit['model']:
                overfit = high_overfit[high_overfit['model'] == model]['overfit_score'].values[0]
                f.write(f"- {model}: {overfit:.2f}\n")
            f.write("\n")

        # OOS collapse
        oos_collapse = ranking[ranking['profit_factor_oos'] < 1.0]
        if len(oos_collapse) > 0:
            f.write("### OOS Performance Collapse (<1.0 PF)\n\n")
            for model in oos_collapse['model']:
                oos_pf = oos_collapse[oos_collapse['model'] == model]['profit_factor_oos'].values[0]
                f.write(f"- {model}: OOS PF = {oos_pf:.2f}\n")
            f.write("\n")

        # Low trade count
        low_trades = ranking[ranking['trades_test'] < 20]
        if len(low_trades) > 0:
            f.write("### Low Trade Count (<20 on test)\n\n")
            for model in low_trades['model']:
                trades = low_trades[low_trades['model'] == model]['trades_test'].values[0]
                f.write(f"- {model}: {int(trades)} trades\n")
            f.write("\n")

        f.write("## Key Insights\n\n")

        # Best baseline
        best_baseline = ranking[ranking['type'] == 'baseline'].iloc[0]
        f.write(f"**Best Baseline:** {best_baseline['model']} (Test PF: {best_baseline['profit_factor_test']:.2f})\n\n")

        # Archetypes beating baselines
        baseline_bar = best_baseline['profit_factor_test']
        beating_baselines = ranking[(ranking['type'] == 'archetype') & (ranking['profit_factor_test'] > baseline_bar)]

        if len(beating_baselines) > 0:
            f.write(f"**Archetypes Beating Best Baseline ({baseline_bar:.2f}):**\n\n")
            for _, row in beating_baselines.iterrows():
                f.write(f"- {row['model']}: Test PF = {row['profit_factor_test']:.2f} (+{row['profit_factor_test'] - baseline_bar:.2f})\n")
        else:
            f.write("**NO ARCHETYPES BEAT BEST BASELINE**\n\n")
            f.write("This is a critical finding. All archetypes underperform simple baselines.\n")

    print(f"Created {output_path}")

if __name__ == '__main__':
    create_ranking()
```

**4.4 Run Analysis Pipeline (30 min)**

```bash
# Merge results
python bin/merge_baseline_archetype_results.py

# Calculate overfit
python bin/calculate_overfit_scores.py

# Create ranking
python bin/create_unified_ranking.py
```

#### Validation Checkpoint

**Success Criteria:**

```bash
# Check unified comparison
cat results/unified_comparison.csv | wc -l  # Should be 25 (8 models × 3 periods + header)

# Check overfit analysis
cat results/overfit_analysis.csv

# Check ranking
cat results/unified_ranking.csv

# Read report
cat results/unified_ranking_report.md
```

**Expected Insights:**
- Clear ranking of all models
- Identification of which archetypes (if any) beat baselines
- Red flags highlighted (overfit, low trades, OOS collapse)

**End of Day 2:** You now know objectively how archetypes compare to baselines.

---

## DAY 3: DECISION MAKING (8 HOURS)

**Goal:** Make keep/improve/kill decisions and plan next steps.

### MORNING (4 HOURS): Analysis & Acceptance

**Objective:** Apply acceptance criteria to each model and make evidence-based decisions.

#### Tasks

**5.1 Apply Acceptance Criteria (90 min)**

Create decision script:

```python
# bin/apply_acceptance_criteria.py
import pandas as pd
from pathlib import Path

# Acceptance criteria from RULES_OF_THE_LAB.md
CRITERIA = {
    'min_test_pf': 1.5,           # Minimum test profit factor
    'max_overfit': 0.5,           # Maximum overfit score
    'min_trades': 50,             # Minimum trades on test period
    'min_oos_pf': 1.2,            # Minimum OOS profit factor
    'baseline_beat_margin': 0.1   # Must beat best baseline by at least this
}

def apply_criteria():
    """Apply acceptance criteria to all models."""

    ranking = pd.read_csv('results/unified_ranking.csv')

    # Get best baseline PF
    baselines = ranking[ranking['type'] == 'baseline']
    best_baseline_pf = baselines['profit_factor_test'].max()

    decisions = []

    for _, row in ranking.iterrows():
        model = row['model']
        model_type = row['type']

        # Extract metrics
        test_pf = row['profit_factor_test']
        oos_pf = row['profit_factor_oos']
        overfit = row['overfit_score']
        trades = row['trades_test']

        # Check criteria
        checks = {
            'test_pf': test_pf >= CRITERIA['min_test_pf'],
            'oos_pf': oos_pf >= CRITERIA['min_oos_pf'],
            'overfit': overfit <= CRITERIA['max_overfit'],
            'trades': trades >= CRITERIA['min_trades'],
            'beats_baseline': test_pf >= (best_baseline_pf + CRITERIA['baseline_beat_margin'])
        }

        # Determine decision
        if model_type == 'baseline':
            # Baselines are kept as reference
            decision = 'KEEP (Reference)'
            reason = 'Baseline model'
        else:
            # Archetypes must meet all criteria
            passes = sum(checks.values())

            if checks['test_pf'] and checks['oos_pf'] and checks['overfit'] and checks['trades'] and checks['beats_baseline']:
                decision = 'KEEP'
                reason = 'Meets all criteria'
            elif passes >= 3:
                decision = 'IMPROVE'
                failures = [k for k, v in checks.items() if not v]
                reason = f"Fails: {', '.join(failures)}"
            else:
                decision = 'KILL'
                failures = [k for k, v in checks.items() if not v]
                reason = f"Fails too many: {', '.join(failures)}"

        decisions.append({
            'model': model,
            'type': model_type,
            'decision': decision,
            'reason': reason,
            'test_pf': test_pf,
            'oos_pf': oos_pf,
            'overfit': overfit,
            'trades': int(trades),
            'beats_baseline': checks['beats_baseline']
        })

    decision_df = pd.DataFrame(decisions)

    # Save
    output_path = Path('results/model_decisions.csv')
    decision_df.to_csv(output_path, index=False)

    print(f"Created {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DECISION SUMMARY")
    print("=" * 80)

    for decision in ['KEEP', 'IMPROVE', 'KILL', 'KEEP (Reference)']:
        models = decision_df[decision_df['decision'] == decision]
        if len(models) > 0:
            print(f"\n{decision}: {len(models)} models")
            for model in models['model']:
                reason = models[models['model'] == model]['reason'].values[0]
                print(f"  - {model}: {reason}")

    return decision_df

if __name__ == '__main__':
    apply_criteria()
```

Run it:

```bash
python bin/apply_acceptance_criteria.py
```

**5.2 Generate Decision Report (90 min)**

```python
# bin/generate_decision_report.py
import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_report():
    """Generate comprehensive decision report."""

    decisions = pd.read_csv('results/model_decisions.csv')
    ranking = pd.read_csv('results/unified_ranking.csv')

    output_path = Path('results/DECISION_REPORT.md')

    with open(output_path, 'w') as f:
        f.write("# Quant Lab Decision Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")

        keep_count = len(decisions[decisions['decision'].str.contains('KEEP')])
        improve_count = len(decisions[decisions['decision'] == 'IMPROVE'])
        kill_count = len(decisions[decisions['decision'] == 'KILL'])

        f.write(f"- **KEEP:** {keep_count} models (ready for deployment consideration)\n")
        f.write(f"- **IMPROVE:** {improve_count} models (needs optimization)\n")
        f.write(f"- **KILL:** {kill_count} models (abandon or major rework)\n\n")

        # Winners
        f.write("## Models to KEEP\n\n")
        keep_models = decisions[decisions['decision'].str.contains('KEEP') & ~decisions['decision'].str.contains('Reference')]

        if len(keep_models) > 0:
            f.write("These models beat baselines and meet all acceptance criteria:\n\n")
            for _, model in keep_models.iterrows():
                f.write(f"### {model['model']}\n\n")
                f.write(f"- **Test PF:** {model['test_pf']:.2f}\n")
                f.write(f"- **OOS PF:** {model['oos_pf']:.2f}\n")
                f.write(f"- **Overfit Score:** {model['overfit']:.2f}\n")
                f.write(f"- **Test Trades:** {model['trades']}\n")
                f.write(f"- **Decision:** {model['decision']}\n")
                f.write(f"- **Next Steps:** Paper trading, then live deployment\n\n")
        else:
            f.write("**NO MODELS MET ACCEPTANCE CRITERIA**\n\n")
            f.write("This indicates:\n")
            f.write("1. Archetypes may be overfitted to historical data\n")
            f.write("2. Market conditions may have changed\n")
            f.write("3. Transaction costs may be eating into edge\n")
            f.write("4. Baselines may be sufficient (consider deploying B1 or B2)\n\n")

        # Improve candidates
        f.write("## Models to IMPROVE\n\n")
        improve_models = decisions[decisions['decision'] == 'IMPROVE']

        if len(improve_models) > 0:
            for _, model in improve_models.iterrows():
                f.write(f"### {model['model']}\n\n")
                f.write(f"- **Reason:** {model['reason']}\n")
                f.write(f"- **Test PF:** {model['test_pf']:.2f}\n")
                f.write(f"- **Action Items:**\n")

                # Specific recommendations
                if 'overfit' in model['reason']:
                    f.write("  - Reduce model complexity\n")
                    f.write("  - Add regularization\n")
                    f.write("  - Test on different time periods\n")

                if 'trades' in model['reason']:
                    f.write("  - Loosen entry filters\n")
                    f.write("  - Test on multiple assets\n")
                    f.write("  - Consider if low frequency is acceptable\n")

                if 'oos_pf' in model['reason']:
                    f.write("  - Investigate OOS collapse\n")
                    f.write("  - Check for regime change\n")
                    f.write("  - Test walk-forward optimization\n")

                if 'beats_baseline' in model['reason']:
                    f.write("  - Compare to best baseline feature-by-feature\n")
                    f.write("  - Identify where alpha is coming from\n")
                    f.write("  - Consider if complexity is justified\n")

                f.write("\n")

        # Kill list
        f.write("## Models to KILL\n\n")
        kill_models = decisions[decisions['decision'] == 'KILL']

        if len(kill_models) > 0:
            f.write("These models fail too many criteria to justify further development:\n\n")
            for _, model in kill_models.iterrows():
                f.write(f"- **{model['model']}:** {model['reason']}\n")

            f.write("\n**Action:** Archive code, document learnings, move on.\n\n")

        # Key insights
        f.write("## Key Insights\n\n")

        # Best baseline
        best_baseline = decisions[decisions['type'] == 'baseline'].sort_values('test_pf', ascending=False).iloc[0]
        f.write(f"**Best Baseline:** {best_baseline['model']} (Test PF: {best_baseline['test_pf']:.2f})\n\n")

        # Archetype performance
        archetypes = decisions[decisions['type'] == 'archetype']
        if len(archetypes) > 0:
            avg_archetype_pf = archetypes['test_pf'].mean()
            avg_baseline_pf = decisions[decisions['type'] == 'baseline']['test_pf'].mean()

            f.write(f"**Average Archetype PF:** {avg_archetype_pf:.2f}\n")
            f.write(f"**Average Baseline PF:** {avg_baseline_pf:.2f}\n")

            if avg_archetype_pf > avg_baseline_pf:
                f.write(f"**Verdict:** Archetypes show promise (+{avg_archetype_pf - avg_baseline_pf:.2f})\n\n")
            else:
                f.write(f"**Verdict:** Baselines outperform on average ({avg_baseline_pf - avg_archetype_pf:.2f})\n\n")

        # Overfit analysis
        high_overfit = decisions[decisions['overfit'] > 0.5]
        if len(high_overfit) > 0:
            f.write(f"**Overfit Warning:** {len(high_overfit)} models show high overfit (>0.5)\n\n")

        f.write("## Recommendations\n\n")

        if len(keep_models) > 0:
            f.write("1. **Deploy Winners:** Begin paper trading for KEEP models\n")
            f.write("2. **Improve Candidates:** Address specific issues in IMPROVE models\n")
            f.write("3. **Kill Losers:** Archive KILL models and document learnings\n\n")
        else:
            f.write("1. **Consider Baseline Deployment:** B1 or B2 may be worth deploying\n")
            f.write("2. **Rework Archetypes:** Fundamental issues need addressing\n")
            f.write("3. **Investigate Market Regime:** Has market changed?\n\n")

    print(f"Created {output_path}")
    print(f"\nRead full report: cat {output_path}")

if __name__ == '__main__':
    generate_report()
```

Run it:

```bash
python bin/generate_decision_report.py
```

#### Validation Checkpoint

```bash
# Check decisions
cat results/model_decisions.csv

# Read decision report
cat results/DECISION_REPORT.md
```

---

### AFTERNOON (4 HOURS): Next Steps Planning

**Objective:** Plan deployment, improvement, and next experiments.

#### Tasks

**6.1 Create Deployment Roadmap (90 min)**

```python
# bin/create_deployment_roadmap.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def create_roadmap():
    """Create deployment roadmap for KEEP models."""

    decisions = pd.read_csv('results/model_decisions.csv')
    keep_models = decisions[decisions['decision'].str.contains('KEEP') & ~decisions['decision'].str.contains('Reference')]

    if len(keep_models) == 0:
        print("No models to deploy. Consider improving candidates or deploying baselines.")
        return

    output_path = Path('results/DEPLOYMENT_ROADMAP.md')

    today = datetime.now()

    with open(output_path, 'w') as f:
        f.write("# Deployment Roadmap\n\n")
        f.write(f"**Generated:** {today.strftime('%Y-%m-%d')}\n\n")

        for idx, (_, model) in enumerate(keep_models.iterrows(), 1):
            model_name = model['model']

            # Calculate timeline
            paper_start = today + timedelta(days=idx * 7)
            paper_end = paper_start + timedelta(days=14)
            live_start = paper_end + timedelta(days=1)

            f.write(f"## {idx}. {model_name}\n\n")

            f.write("### Performance Summary\n\n")
            f.write(f"- Test PF: {model['test_pf']:.2f}\n")
            f.write(f"- OOS PF: {model['oos_pf']:.2f}\n")
            f.write(f"- Overfit: {model['overfit']:.2f}\n")
            f.write(f"- Trades: {model['trades']}\n\n")

            f.write("### Deployment Timeline\n\n")
            f.write(f"- **Paper Trading:** {paper_start.strftime('%Y-%m-%d')} to {paper_end.strftime('%Y-%m-%d')} (14 days)\n")
            f.write(f"- **Live Deployment:** {live_start.strftime('%Y-%m-%d')} (if paper trading successful)\n\n")

            f.write("### Paper Trading Acceptance Criteria\n\n")
            f.write(f"- Minimum {max(10, int(model['trades'] / 12))} trades executed\n")
            f.write(f"- PF >= {model['oos_pf'] * 0.8:.2f} (80% of OOS performance)\n")
            f.write(f"- No catastrophic drawdown (>20%)\n")
            f.write("- Signal generation matches backtest\n\n")

            f.write("### Risk Parameters\n\n")
            f.write("- **Initial Position Size:** 1% of capital per trade\n")
            f.write("- **Max Concurrent Positions:** 1\n")
            f.write("- **Daily Loss Limit:** 2% of capital\n")
            f.write("- **Circuit Breaker:** 5% drawdown from peak\n\n")

            f.write("### Monitoring Requirements\n\n")
            f.write("- Daily PnL review\n")
            f.write("- Signal generation audit (compare to backtest)\n")
            f.write("- Slippage analysis\n")
            f.write("- Fill rate analysis\n\n")

            f.write("---\n\n")

    print(f"Created {output_path}")

if __name__ == '__main__':
    create_roadmap()
```

**6.2 Create Improvement Plan (90 min)**

```python
# bin/create_improvement_plan.py
import pandas as pd
from pathlib import Path

def create_plan():
    """Create improvement plan for IMPROVE models."""

    decisions = pd.read_csv('results/model_decisions.csv')
    improve_models = decisions[decisions['decision'] == 'IMPROVE']

    if len(improve_models) == 0:
        print("No models need improvement.")
        return

    output_path = Path('results/IMPROVEMENT_PLAN.md')

    with open(output_path, 'w') as f:
        f.write("# Model Improvement Plan\n\n")

        for idx, (_, model) in enumerate(improve_models.iterrows(), 1):
            model_name = model['model']

            f.write(f"## {idx}. {model_name}\n\n")

            f.write(f"**Current Status:** {model['reason']}\n\n")

            f.write("### Diagnostic Tasks\n\n")

            # Specific diagnostics based on failure reason
            if 'overfit' in model['reason'].lower():
                f.write("- [ ] Run walk-forward optimization\n")
                f.write("- [ ] Test with reduced feature set\n")
                f.write("- [ ] Add regularization to parameters\n")
                f.write("- [ ] Test on different time periods (2018-2019)\n")

            if 'trades' in model['reason'].lower():
                f.write("- [ ] Analyze why trade frequency is low\n")
                f.write("- [ ] Test with loosened entry filters\n")
                f.write("- [ ] Backtest on multiple assets (ETH, SOL)\n")
                f.write("- [ ] Consider if low frequency is acceptable (rare alpha)\n")

            if 'oos' in model['reason'].lower():
                f.write("- [ ] Investigate market regime change in OOS period\n")
                f.write("- [ ] Compare feature distributions (train vs OOS)\n")
                f.write("- [ ] Test on alternative OOS period (2024 H1 vs H2)\n")
                f.write("- [ ] Check if specific market events broke model\n")

            if 'baseline' in model['reason'].lower():
                f.write("- [ ] Feature ablation study (which features add value?)\n")
                f.write("- [ ] Compare to best baseline side-by-side\n")
                f.write("- [ ] Identify periods where model beats/loses to baseline\n")
                f.write("- [ ] Quantify incremental value above baseline\n")

            f.write("\n### Success Criteria for Re-evaluation\n\n")
            f.write("- Test PF >= 1.5\n")
            f.write("- Overfit Score <= 0.5\n")
            f.write("- Test Trades >= 50\n")
            f.write("- Beats best baseline by +0.1 PF\n\n")

            f.write("### Timeline\n\n")
            f.write("- **Week 1:** Complete diagnostic tasks\n")
            f.write("- **Week 2:** Implement improvements\n")
            f.write("- **Week 3:** Re-run backtest suite\n")
            f.write("- **Week 4:** Re-evaluate against criteria\n\n")

            f.write("---\n\n")

    print(f"Created {output_path}")

if __name__ == '__main__':
    create_plan()
```

**6.3 Design Next Experiments (60 min)**

Create experiment queue:

```python
# bin/create_experiment_queue.py
from pathlib import Path

def create_queue():
    """Create queue of next experiments."""

    output_path = Path('results/EXPERIMENT_QUEUE.md')

    with open(output_path, 'w') as f:
        f.write("# Experiment Queue\n\n")
        f.write("Prioritized list of next experiments to run.\n\n")

        f.write("## High Priority\n\n")

        f.write("### 1. Temporal Layer Ablation Study\n\n")
        f.write("**Objective:** Measure lift from temporal layers (Fib clusters, temporal confluence)\n\n")
        f.write("**Method:**\n")
        f.write("- Run S4 with funding only (no temporal layers)\n")
        f.write("- Run S4 with funding + temporal layers\n")
        f.write("- Calculate incremental PF lift\n\n")
        f.write("**Acceptance:** Temporal layers must add +0.2 PF to justify complexity\n\n")

        f.write("### 2. Regime Slicing Analysis\n\n")
        f.write("**Objective:** Understand which models work in which regimes\n\n")
        f.write("**Method:**\n")
        f.write("- Split test period by regime (bull, bear, sideways)\n")
        f.write("- Run all models on each regime slice\n")
        f.write("- Identify regime-specific winners\n\n")
        f.write("**Outcome:** Regime-routed ensemble or regime-specific deployment\n\n")

        f.write("### 3. Multi-Asset Validation\n\n")
        f.write("**Objective:** Test if models generalize beyond BTC\n\n")
        f.write("**Method:**\n")
        f.write("- Run top 3 models on ETH/USDT\n")
        f.write("- Run top 3 models on SOL/USDT\n")
        f.write("- Compare performance to BTC results\n\n")
        f.write("**Acceptance:** Models must maintain >70% of BTC performance on other assets\n\n")

        f.write("## Medium Priority\n\n")

        f.write("### 4. Transaction Cost Sensitivity\n\n")
        f.write("**Objective:** Understand impact of slippage and fees\n\n")
        f.write("**Method:**\n")
        f.write("- Re-run baselines with 2x, 3x, 5x current costs\n")
        f.write("- Identify break-even cost levels\n")
        f.write("- Prioritize low-frequency models if costs are higher than expected\n\n")

        f.write("### 5. Parameter Sensitivity Analysis\n\n")
        f.write("**Objective:** Understand parameter robustness\n\n")
        f.write("**Method:**\n")
        f.write("- For top models, vary each parameter ±20%\n")
        f.write("- Measure PF stability\n")
        f.write("- Identify brittle vs robust models\n\n")

        f.write("### 6. Walk-Forward Optimization\n\n")
        f.write("**Objective:** Test parameter stability over time\n\n")
        f.write("**Method:**\n")
        f.write("- Split data into 6-month windows\n")
        f.write("- Optimize on each window, test on next\n")
        f.write("- Compare to static parameters\n\n")

        f.write("## Low Priority\n\n")

        f.write("### 7. Ensemble Methods\n\n")
        f.write("**Objective:** Test if combining models improves performance\n\n")
        f.write("**Method:**\n")
        f.write("- Equal weight ensemble of top 3 models\n")
        f.write("- PF-weighted ensemble\n")
        f.write("- Sharpe-weighted ensemble\n\n")

        f.write("### 8. Alternative Timeframes\n\n")
        f.write("**Objective:** Test if models work on 4h or 15m data\n\n")
        f.write("**Method:**\n")
        f.write("- Re-run baseline suite on 4h BTC\n")
        f.write("- Re-run baseline suite on 15m BTC\n")
        f.write("- Compare frequency vs profitability tradeoffs\n\n")

    print(f"Created {output_path}")

if __name__ == '__main__':
    create_queue()
```

**6.4 Update Master Roadmap (30 min)**

Update the project roadmap based on findings:

```bash
# bin/update_roadmap.py
```

This should update your main project roadmap to reflect:
- Models ready for deployment
- Models needing improvement
- Next experiments to run
- Updated timelines

#### Final Validation

**Success Criteria for Day 3:**

```bash
# All deliverables exist
ls -la results/model_decisions.csv
ls -la results/DECISION_REPORT.md
ls -la results/DEPLOYMENT_ROADMAP.md
ls -la results/IMPROVEMENT_PLAN.md
ls -la results/EXPERIMENT_QUEUE.md

# Decision report has clear keep/improve/kill counts
grep "KEEP:" results/DECISION_REPORT.md
grep "IMPROVE:" results/DECISION_REPORT.md
grep "KILL:" results/DECISION_REPORT.md
```

---

## POST-72-HOUR WORKFLOW

**Ongoing Process:**

### Weekly Cadence

**Every Monday:**
1. Review paper trading results for deployed models
2. Check improvement progress for IMPROVE models
3. Run next experiment from queue
4. Update decision matrix

**Every Friday:**
1. Run baseline suite on latest data (rolling window)
2. Compare live performance to backtest expectations
3. Re-evaluate decisions if market regime changes

### Monthly Deep Dive

1. Full backtest refresh with latest data
2. Re-run acceptance criteria
3. Add new baseline if market has changed
4. Review and update experiment queue

---

## TROUBLESHOOTING

### Common Issues

**Issue: Baseline suite runs but produces zero trades**
- Check data quality (missing values, incorrect OHLCV format)
- Verify date ranges match available data
- Check for timezone mismatches

**Issue: Archetypes can't be loaded into suite**
- Use ArchetypeWrapper adapter class
- Run archetypes separately and merge results manually
- Plan integration for Phase 2

**Issue: All models fail acceptance criteria**
- Lower criteria thresholds (min_test_pf = 1.2 instead of 1.5)
- Extend test period (more data needed)
- Check if market regime is unusual
- Consider that baselines may be sufficient

**Issue: Overfit scores are all negative**
- Check calculation (should be (Train - Test) / Train)
- If test > train, model may be lucky or undertrained
- Extend training period

**Issue: OOS period too short for statistical significance**
- Use 2024 H1 as test, H2 as OOS
- Or use walk-forward validation instead
- Or accept limitation and monitor closely in paper trading

---

## SUCCESS METRICS

**End of 72 Hours, You Should Have:**

1. Working baseline suite (5 models) ✓
2. Archetype integration (3+ models) ✓
3. Unified comparison framework ✓
4. Clear keep/improve/kill decisions ✓
5. Deployment roadmap ✓
6. Improvement plan ✓
7. Next experiment queue ✓

**Qualitative Success:**
- Confidence in decisions (evidence-based, not gut feel)
- Understanding of model strengths/weaknesses
- Clear path forward (not stuck)
- Repeatable process (can run on new data anytime)

**The Ultimate Test:**
Can you explain to a skeptical quant why your archetype is worth deploying instead of SMA200?

If yes: You have a real lab.
If no: Keep running experiments.

---

## APPENDIX: FILE STRUCTURE

```
Bull-machine-/
├── engine/
│   └── backtesting/
│       ├── __init__.py
│       ├── base_model.py
│       ├── backtest_engine.py
│       ├── backtest_results.py
│       └── baseline_models.py
├── bin/
│   ├── run_quant_suite.py
│   ├── extract_archetype_metrics.py
│   ├── merge_baseline_archetype_results.py
│   ├── calculate_overfit_scores.py
│   ├── create_unified_ranking.py
│   ├── apply_acceptance_criteria.py
│   ├── generate_decision_report.py
│   ├── create_deployment_roadmap.py
│   ├── create_improvement_plan.py
│   └── create_experiment_queue.py
├── tests/
│   └── unit/
│       └── backtesting/
│           ├── test_backtest_engine.py
│           └── test_baseline_models.py
├── configs/
│   └── experiment_btc_1h_2020_2025.json
├── results/
│   ├── quant_suite/
│   │   ├── results_LATEST.csv
│   │   └── report_LATEST.md
│   ├── archetypes/
│   │   ├── s4_train.json
│   │   ├── s4_test.json
│   │   ├── s4_oos.json
│   │   └── archetype_metrics.csv
│   ├── unified_comparison.csv
│   ├── unified_ranking.csv
│   ├── overfit_analysis.csv
│   ├── model_decisions.csv
│   ├── DECISION_REPORT.md
│   ├── DEPLOYMENT_ROADMAP.md
│   ├── IMPROVEMENT_PLAN.md
│   └── EXPERIMENT_QUEUE.md
└── docs/
    ├── RULES_OF_THE_LAB.md
    ├── QUANT_LAB_72H_EXECUTION.md
    ├── QUANT_LAB_CHECKLIST.md
    └── QUANT_LAB_FAQ.md
```

---

**Let's build a real quant lab.**
