# Quant Lab Architecture

**Version:** 1.0
**Date:** 2025-12-05
**Status:** Production Framework Design

## Executive Summary

This document defines the **Professional Quant Testing Framework** for Bull Machine - a standardized system that treats all trading strategies (baselines, archetypes, ML models) as "models" that must pass through rigorous testing gates before deployment.

**Core Philosophy:** Everything is a model. No special treatment. Evidence-based decisions only.

---

## Table of Contents

1. [Framework Philosophy](#framework-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Component Design](#component-design)
4. [Data Flow](#data-flow)
5. [Integration Points](#integration-points)
6. [Model Lifecycle](#model-lifecycle)
7. [Technical Specifications](#technical-specifications)
8. [Future Extensions](#future-extensions)

---

## Framework Philosophy

### Core Principles

**1. Everything is a Model**
- Simple baselines (buy drawdown, sell rally)
- Complex archetypes (Wyckoff patterns, SMC setups)
- ML models (ensemble, regime classifiers)
- Feature layers (temporal/Fib/Gann) are modifiers, not separate models

**2. Standardized Treatment**
- All models use same experiment configuration schema
- All models produce same metrics structure
- All models pass through same acceptance gates
- No exceptions, no shortcuts

**3. Evidence-Based Decisions**
- Test performance matters more than train performance
- Out-of-sample (OOS) validation is mandatory
- Overfit detection is automatic
- Sample size determines confidence

**4. Professional Standards**
- Mimic institutional quant fund workflows
- Reproducible experiments (versioned configs)
- Documented decisions (why accepted/rejected)
- Statistical rigor (min trade counts, significance tests)

**5. Fail Fast, Learn Fast**
- Kill underperforming models early
- Document failures (learn from them)
- Iterate quickly on improvements
- No emotional attachment to models

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     QUANT LAB FRAMEWORK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  EXPERIMENT  │───>│   BACKTEST   │───>│   METRICS    │     │
│  │    CONFIG    │    │    ENGINE    │    │  CALCULATOR  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                    │             │
│         │                    │                    │             │
│         v                    v                    v             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │    MODEL     │    │    TRADES    │    │   RANKING    │     │
│  │  REGISTRY    │    │   DATABASE   │    │    ENGINE    │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         │                    │                    │             │
│         │                    │                    │             │
│         v                    v                    v             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ ACCEPTANCE   │    │ COMPARISON   │    │  DEPLOYMENT  │     │
│  │    RULES     │    │   REPORTS    │    │    GATE      │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Layer Architecture

**Layer 1: Configuration & Data**
- Experiment configs (train/test/OOS splits, regime filters, costs)
- Feature stores (MTF technical, macro, fundamental)
- Model registry (baselines, archetypes, ML)

**Layer 2: Execution & Measurement**
- Backtest engine (unified simulator)
- Trade database (all executions logged)
- Metrics calculator (standardized outputs)

**Layer 3: Analysis & Decision**
- Ranking engine (compare models)
- Acceptance rules (pass/fail gates)
- Deployment workflow (research → paper → live)

---

## Component Design

### 1. Experiment Configuration Schema

**Purpose:** Define testing parameters once, reuse across all models

**File:** `configs/experiments/experiment_btc_1h_2020_2025.json`

```json
{
  "experiment_id": "exp_btc_1h_2020_2025_v1",
  "description": "BTC 1H full history test (2020-2025)",
  "created_at": "2025-12-05T00:00:00Z",

  "asset": {
    "symbol": "BTC",
    "timeframe": "1H",
    "data_source": "data/processed/features_mtf/BTC_1H_2020-01-01_to_2025-12-31.parquet"
  },

  "periods": {
    "train": {
      "start": "2020-01-01",
      "end": "2022-12-31",
      "description": "Bull cycle + bear market"
    },
    "test": {
      "start": "2023-01-01",
      "end": "2023-12-31",
      "description": "Recovery period"
    },
    "oos": {
      "start": "2024-01-01",
      "end": "2024-12-31",
      "description": "Out-of-sample validation"
    }
  },

  "regime_filters": {
    "enabled": true,
    "modes": ["all", "risk_on", "risk_off", "crisis"],
    "default": "all"
  },

  "cost_model": {
    "slippage_bps": 5.0,
    "fee_bps": 6.0,
    "total_cost_bps": 11.0
  },

  "position_sizing": {
    "initial_capital": 10000,
    "risk_per_trade_pct": 1.0,
    "max_leverage": 1.0,
    "compounding": true
  },

  "statistical_requirements": {
    "min_trades_train": 30,
    "min_trades_test": 20,
    "min_trades_oos": 15,
    "min_confidence_level": 0.80
  },

  "extensions": {
    "multi_asset": {
      "enabled": false,
      "assets": ["BTC", "ETH", "SOL"]
    },
    "walk_forward": {
      "enabled": false,
      "window_months": 6,
      "step_months": 1
    }
  }
}
```

**Key Features:**
- **Versioned:** Each config is immutable (append v2, v3)
- **Reusable:** Same config tests all models
- **Extensible:** Easy to add multi-asset, regime slicing
- **Documented:** Inline descriptions explain each section

---

### 2. Standard Metrics System

**Purpose:** All models output identical metrics for fair comparison

**Dataclass:** `engine/backtesting/metrics.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BacktestResults:
    """
    Standardized backtest results.

    ALL models (baselines, archetypes, ML) must produce this structure.
    """
    # Identification
    model_name: str
    experiment_id: str
    period: str  # "train", "test", "oos"
    start_date: str
    end_date: str

    # Core Performance
    total_trades: int
    win_rate: float  # 0.0 - 1.0
    profit_factor: float
    total_return_pct: float
    cagr: float

    # Risk Metrics
    max_drawdown: float  # Peak-to-trough %
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float  # CAGR / MaxDD

    # Trade Quality
    avg_win_r: float  # Average winner in R-multiples
    avg_loss_r: float  # Average loser in R-multiples
    avg_r: float  # Overall expectancy in R
    largest_win_r: float
    largest_loss_r: float

    # Statistical Validity
    sample_size_sufficient: bool  # Based on min_trades requirement
    confidence_score: float  # 0.0 - 1.0 (based on sample size + consistency)

    # Execution Stats
    avg_bars_held: float
    win_streak_max: int
    loss_streak_max: int

    # Cost Impact
    total_cost_paid: float  # In % of capital
    slippage_impact_pct: float
    fee_impact_pct: float

    # Timestamps
    backtest_completed_at: str

    # Optional: Regime Breakdown
    regime_breakdown: Optional[dict] = None  # {regime: {trades, pf, wr}}

    def to_dict(self):
        """Convert to dictionary for CSV/JSON export"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def overfit_score(self, train_results: 'BacktestResults') -> float:
        """
        Calculate overfit score vs training period.

        Returns: (Train PF - Test PF)
        Interpretation:
          - < 0.3: Excellent generalization
          - 0.3 - 0.5: Acceptable
          - > 0.5: Overfitting detected
        """
        return train_results.profit_factor - self.profit_factor
```

**Derived Metrics:**

```python
def calculate_derived_metrics(train: BacktestResults,
                              test: BacktestResults,
                              oos: BacktestResults) -> dict:
    """Calculate comparison metrics across periods"""
    return {
        # Overfit Detection
        "overfit_train_to_test": test.overfit_score(train),
        "overfit_train_to_oos": oos.overfit_score(train),

        # Degradation
        "pf_degradation_pct": ((train.profit_factor - oos.profit_factor)
                               / train.profit_factor * 100),
        "sharpe_degradation_pct": ((train.sharpe_ratio - oos.sharpe_ratio)
                                   / train.sharpe_ratio * 100),

        # Consistency
        "trade_count_variance": np.std([train.total_trades,
                                        test.total_trades,
                                        oos.total_trades]),

        # Risk-Adjusted
        "avg_sharpe": np.mean([train.sharpe_ratio,
                               test.sharpe_ratio,
                               oos.sharpe_ratio]),
        "avg_calmar": np.mean([train.calmar_ratio,
                               test.calmar_ratio,
                               oos.calmar_ratio])
    }
```

---

### 3. Model Ranking System

**Purpose:** Objective comparison of all models

**Primary Ranking Criteria:**
1. **Test Profit Factor** (most important - unseen data performance)
2. **Overfit Score** (secondary - generalization ability)
3. **Sample Size** (tertiary - statistical confidence)

**Ranking Algorithm:**

```python
def rank_models(results: List[BacktestResults]) -> pd.DataFrame:
    """
    Rank models by multi-objective scoring.

    Returns DataFrame with columns:
    - Model, Test_PF, Overfit, Trades, Confidence, Score, Rank, Status
    """
    df = pd.DataFrame([r.to_dict() for r in results])

    # Calculate composite score
    df['pf_score'] = df['test_profit_factor'] / df['test_profit_factor'].max()
    df['overfit_penalty'] = 1.0 - (df['overfit'] / 2.0).clip(0, 1)
    df['sample_score'] = (df['test_trades'] / 50).clip(0, 1)  # 50 trades = 100%

    # Weighted composite
    df['composite_score'] = (
        0.50 * df['pf_score'] +           # 50% weight on PF
        0.30 * df['overfit_penalty'] +    # 30% weight on generalization
        0.20 * df['sample_score']         # 20% weight on sample size
    )

    # Rank
    df['rank'] = df['composite_score'].rank(ascending=False)

    # Status flags
    df['status'] = df.apply(assign_status, axis=1)

    return df.sort_values('rank')


def assign_status(row: pd.Series) -> str:
    """
    Assign status based on acceptance rules.

    Returns: "✅ Keep", "🔧 Improve", "❌ Kill"
    """
    # Red flags (kill)
    if row['test_profit_factor'] < 1.0:
        return "❌ Kill (PF < 1.0)"
    if row['overfit'] > 0.5:
        return "❌ Kill (Overfit > 0.5)"
    if row['test_trades'] < 15:
        return "❌ Kill (Sample < 15)"

    # Needs work
    if row['test_profit_factor'] < 1.5:
        return "🔧 Improve (PF < 1.5)"
    if row['overfit'] > 0.3:
        return "🔧 Improve (Overfit > 0.3)"

    # Production ready
    if row['test_profit_factor'] >= 2.0 and row['overfit'] <= 0.3:
        return "✅ Keep (Production Ready)"

    return "🔧 Improve (Marginal)"
```

---

### 4. Comparison Framework

**Purpose:** Side-by-side model comparison with visual ranking

**Output Schema (CSV):**

```csv
Model,Train_PF,Train_WR,Train_Trades,Test_PF,Test_WR,Test_Trades,OOS_PF,OOS_WR,OOS_Trades,Overfit,Status,Rank
Baseline-BuyDip,-15pct,2.1,45.2,120,1.8,42.1,80,1.7,40.5,65,0.4,✅ Keep,2
Baseline-SellRally,+12pct,1.5,38.5,95,1.6,41.2,88,1.5,39.8,82,-0.1,✅ Keep,4
Archetype-S2-FailedRally,3.2,58.5,85,2.8,56.2,72,2.6,54.8,68,0.6,❌ Kill,7
Archetype-A-TrapReversal,4.5,62.1,42,3.8,59.3,38,3.6,58.7,35,0.9,❌ Kill,9
ML-RandomForest-v1,5.2,68.4,180,2.1,51.2,165,1.9,49.8,158,3.1,❌ Kill,12
```

**Visual Comparison Table:**

```
═══════════════════════════════════════════════════════════════════════════════
MODEL COMPARISON REPORT
Experiment: exp_btc_1h_2020_2025_v1
Generated: 2025-12-05 15:30:00
═══════════════════════════════════════════════════════════════════════════════

╔═══════════════════════════╦═══════╦═══════╦═══════╦═════════╦════════╦═══════╗
║ Model                     ║ Test  ║ OOS   ║ Over  ║ Trades  ║ Status ║ Rank  ║
║                           ║ PF    ║ PF    ║ fit   ║ (T/O)   ║        ║       ║
╠═══════════════════════════╬═══════╬═══════╬═══════╬═════════╬════════╬═══════╣
║ Baseline-BuyDip-15pct     ║ 1.8   ║ 1.7   ║ 0.4   ║ 80/65   ║ ✅ Keep║   2   ║
║ Baseline-SellRally+12pct  ║ 1.6   ║ 1.5   ║ -0.1  ║ 88/82   ║ ✅ Keep║   4   ║
║ Archetype-S2-FailedRally  ║ 2.8   ║ 2.6   ║ 0.6   ║ 72/68   ║ ❌ Kill║   7   ║
║ Archetype-A-TrapReversal  ║ 3.8   ║ 3.6   ║ 0.9   ║ 38/35   ║ ❌ Kill║   9   ║
║ ML-RandomForest-v1        ║ 2.1   ║ 1.9   ║ 3.1   ║ 165/158 ║ ❌ Kill║  12   ║
╚═══════════════════════════╩═══════╩═══════╩═══════╩═════════╩════════╩═══════╝

KEY INSIGHTS:
✓ Top Model: Baseline-BuyDip-15pct (Test PF 1.8, Overfit 0.4)
✓ 2 models production-ready
⚠ 3 models killed (overfit > 0.5 or PF degradation)

RECOMMENDATION:
Deploy Baseline-BuyDip-15pct to paper trading (2-week monitoring).
Kill complex models - they overfit despite high train performance.
```

---

## Data Flow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: CONFIGURATION                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  configs/experiments/experiment_btc_1h_2020_2025.json                   │
│         │                                                                │
│         v                                                                │
│  Load Experiment Config (train/test/OOS splits, costs, filters)         │
│         │                                                                │
│         v                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Baseline 1  │  │ Archetype S2 │  │  ML Model 1  │                  │
│  │  Buy Dip     │  │ FailedRally  │  │RandomForest  │                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: EXECUTION                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  For each model:                                                        │
│    For each period (train, test, oos):                                 │
│      1. Load data from feature store                                   │
│      2. Apply regime filter (if enabled)                               │
│      3. Run backtest with cost model                                   │
│      4. Log all trades to database                                     │
│      5. Calculate BacktestResults                                      │
│                                                                          │
│  Output: 3 BacktestResults per model (train, test, oos)                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: ANALYSIS                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Calculate derived metrics (overfit, degradation)                   │
│  2. Rank models (composite score)                                      │
│  3. Apply acceptance rules (pass/fail gates)                           │
│  4. Generate comparison report                                         │
│  5. Export CSV/JSON results                                            │
│                                                                          │
│  Output:                                                                │
│    - results/comparison_btc_1h_2020_2025.csv                           │
│    - results/comparison_btc_1h_2020_2025_report.md                     │
│    - results/trades_database.parquet                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                v
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: DECISION                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  IF model passes acceptance rules:                                     │
│    → Promote to paper trading (2-4 week monitoring)                    │
│    → Document why (checklist in RULES_OF_THE_LAB.md)                   │
│                                                                          │
│  ELSE:                                                                  │
│    → Kill or rework                                                     │
│    → Document why (failure log)                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### Current System Integration

**Short-Term: Dual-Engine Comparison (First 72 hours)**

```
┌──────────────────────────────────────────────────────┐
│              MANUAL MERGE WORKFLOW                   │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Step 1: Run Baselines in New Framework              │
│  ────────────────────────────────────────────        │
│    python bin/run_experiment.py \                    │
│      --experiment configs/experiments/exp_btc.json \ │
│      --models baseline_buy_dip baseline_sell_rally   │
│                                                       │
│    Output: results/baselines_btc_comparison.csv      │
│                                                       │
│  Step 2: Run Archetypes in Old Engine                │
│  ──────────────────────────────────────────          │
│    python bin/backtest_knowledge_v2.py \             │
│      --config configs/mvp/mvp_bull_market_v1.json    │
│                                                       │
│    Output: results/archetype_backtest.log            │
│                                                       │
│  Step 3: Manual Merge (extract metrics from log)     │
│  ─────────────────────────────────────────────       │
│    python bin/merge_comparison_results.py \          │
│      --baselines results/baselines_btc_comparison.csv\│
│      --archetypes results/archetype_backtest.log \   │
│      --output results/unified_comparison.csv         │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Apples-to-Apples Guarantees:**
- Same date ranges (2022-2024)
- Same cost model (11 bps total)
- Same initial capital (10k)
- Same feature store data

**Long-Term: Unified Suite (After wrapper fix)**

```python
# Single command runs all models through same framework
python bin/run_experiment.py \
  --experiment configs/experiments/exp_btc.json \
  --models all  # Baselines + Archetypes + ML
```

---

## Model Lifecycle

### From Research to Production

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL LIFECYCLE STAGES                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STAGE 1: RESEARCH                                                  │
│  ─────────────────────                                              │
│    • Develop model logic                                            │
│    • Run quick sanity tests                                         │
│    • Document hypothesis                                            │
│    • Checkpoint: Logic compiles + basic backtest works              │
│                                                                      │
│                     │                                                │
│                     v                                                │
│  STAGE 2: VALIDATION                                                │
│  ────────────────────                                               │
│    • Run full experiment (train/test/OOS)                           │
│    • Calculate all metrics                                          │
│    • Compare vs baselines                                           │
│    • Checkpoint: Passes acceptance rules in RULES_OF_THE_LAB.md     │
│                                                                      │
│                     │                                                │
│                     v                                                │
│  STAGE 3: PAPER TRADING                                             │
│  ───────────────────────                                            │
│    • Deploy to paper account                                        │
│    • Monitor 2-4 weeks                                              │
│    • Track live performance vs backtest                             │
│    • Checkpoint: OOS metrics hold (±20% tolerance)                  │
│                                                                      │
│                     │                                                │
│                     v                                                │
│  STAGE 4: PRODUCTION                                                │
│  ────────────────────                                               │
│    • Deploy to live trading (small size)                            │
│    • Ramp up over 1 month                                           │
│    • Continuous monitoring                                          │
│    • Checkpoint: Live PF > 80% of OOS PF                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Decision Tree

```
                     [Model Developed]
                            │
                            v
                  [Run Experiment Test]
                            │
                ┌───────────┴───────────┐
                v                       v
        [Test PF < 1.0]         [Test PF ≥ 1.0]
                │                       │
                v                       │
           ❌ KILL                      │
        Document why                    v
                              [Overfit > 0.5?]
                                        │
                            ┌───────────┴───────────┐
                            v                       v
                       [Yes - High]            [No - Low]
                            │                       │
                            v                       v
                      ❌ KILL                 [Sample Size ≥ 15?]
                   Document why                     │
                                        ┌───────────┴───────────┐
                                        v                       v
                                   [No - Low]             [Yes - OK]
                                        │                       │
                                        v                       v
                               🔧 REWORK                [Beats Baselines?]
                          (get more trades)                     │
                                                    ┌───────────┴───────────┐
                                                    v                       v
                                               [No - Worse]          [Yes - Better]
                                                    │                       │
                                                    v                       v
                                              ❌ KILL              ✅ PAPER TRADING
                                         Complex ≠ Better            (2-4 weeks)
                                                                            │
                                                                            v
                                                                    [Paper Holds?]
                                                                            │
                                                                ┌───────────┴───────────┐
                                                                v                       v
                                                           [No - Fails]          [Yes - Good]
                                                                │                       │
                                                                v                       v
                                                          🔧 REWORK              ✅ PRODUCTION
                                                       (find bug/leak)        (gradual ramp)
```

---

## Technical Specifications

### File Structure

```
Bull-machine-/
├── configs/
│   └── experiments/
│       ├── experiment_template.json          # Template with all options
│       ├── experiment_btc_1h_2020_2025.json  # BTC full history
│       ├── experiment_btc_1h_bear_2022.json  # 2022 bear market only
│       └── experiment_multi_asset_v1.json    # BTC+ETH+SOL (future)
│
├── engine/
│   ├── backtesting/
│   │   ├── metrics.py                 # BacktestResults dataclass
│   │   ├── experiment.py              # Experiment runner
│   │   ├── comparison.py              # Model comparison engine
│   │   └── ranking.py                 # Ranking algorithm
│   │
│   └── models/
│       ├── base.py                    # BaseTradingModel interface
│       ├── baseline.py                # Simple baseline models
│       ├── archetype_model.py         # Archetype wrapper
│       └── ml_model.py                # ML model wrapper
│
├── bin/
│   ├── run_experiment.py              # Main experiment runner
│   ├── merge_comparison_results.py    # Manual merge (short-term)
│   └── generate_comparison_report.py  # Report generator
│
├── results/
│   ├── experiments/
│   │   └── exp_btc_1h_2020_2025_v1/
│   │       ├── comparison.csv
│   │       ├── comparison_report.md
│   │       ├── trades_database.parquet
│   │       └── model_results/
│   │           ├── baseline_buy_dip/
│   │           │   ├── train_metrics.json
│   │           │   ├── test_metrics.json
│   │           │   └── oos_metrics.json
│   │           └── archetype_s2/
│   │               ├── train_metrics.json
│   │               ├── test_metrics.json
│   │               └── oos_metrics.json
│   │
│   └── failures/
│       └── killed_models_log.csv      # Graveyard (learn from failures)
│
└── docs/
    ├── QUANT_LAB_ARCHITECTURE.md      # This file
    ├── RULES_OF_THE_LAB.md            # Acceptance criteria
    ├── QUANT_LAB_QUICK_START.md       # 72-hour execution guide
    └── TESTING_FRAMEWORK_DIAGRAMS.md  # Visual diagrams
```

### API Contracts

**BaseTradingModel Interface:**

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd

class BaseTradingModel(ABC):
    """
    All trading models must implement this interface.

    Ensures consistent API across baselines, archetypes, ML.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate entry signals from data.

        Args:
            data: Feature store dataframe (OHLCV + features)

        Returns:
            List of Signal objects with entry_price, side, timestamp
        """
        pass

    @abstractmethod
    def get_exit_plan(self, signal: Signal, data: pd.DataFrame) -> ExitPlan:
        """
        Define exit strategy for a given signal.

        Args:
            signal: Entry signal
            data: Feature store data

        Returns:
            ExitPlan with stop_loss, take_profits, max_hold_bars
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for reporting"""
        pass

    @property
    def model_type(self) -> str:
        """Model category: 'baseline', 'archetype', 'ml'"""
        return "unknown"
```

**Example Implementation:**

```python
class BuyDipBaseline(BaseTradingModel):
    def __init__(self, dip_threshold_pct: float = -15.0):
        self.dip_threshold_pct = dip_threshold_pct

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        for idx, row in data.iterrows():
            if row['drawdown_pct'] <= self.dip_threshold_pct:
                signals.append(Signal(
                    timestamp=idx,
                    side='long',
                    entry_price=row['close'],
                    confidence=abs(row['drawdown_pct']) / 20.0  # 20% = max
                ))
        return signals

    def get_exit_plan(self, signal: Signal, data: pd.DataFrame) -> ExitPlan:
        return ExitPlan(
            stop_loss=signal.entry_price * 0.95,  # 5% stop
            take_profit=signal.entry_price * 1.08,  # 8% target
            max_hold_bars=168  # 1 week (1H bars)
        )

    @property
    def name(self) -> str:
        return f"Baseline-BuyDip-{abs(self.dip_threshold_pct):.0f}pct"

    @property
    def model_type(self) -> str:
        return "baseline"
```

---

## Future Extensions

### Multi-Asset Support

**Config Extension:**
```json
{
  "experiment_id": "exp_multi_asset_v1",
  "assets": [
    {"symbol": "BTC", "timeframe": "1H", "weight": 0.50},
    {"symbol": "ETH", "timeframe": "1H", "weight": 0.30},
    {"symbol": "SOL", "timeframe": "1H", "weight": 0.20}
  ],
  "portfolio_metrics": {
    "correlation_penalty": true,
    "max_concurrent_positions": 3
  }
}
```

### Regime Slicing

**Test same model across different regimes:**

```json
{
  "regime_slicing": {
    "enabled": true,
    "slices": [
      {"name": "risk_on", "filter": "regime == 'risk_on'"},
      {"name": "risk_off", "filter": "regime == 'risk_off'"},
      {"name": "crisis", "filter": "regime == 'crisis'"}
    ]
  }
}
```

**Output:**
- 3 separate BacktestResults per model (one per regime)
- Helps identify regime-specific winners

### Ensemble Models

**Combine multiple models:**

```python
class EnsembleModel(BaseTradingModel):
    def __init__(self, models: List[BaseTradingModel], voting: str = 'majority'):
        self.models = models
        self.voting = voting  # 'majority', 'unanimous', 'weighted'

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        # Collect signals from all sub-models
        all_signals = [m.generate_signals(data) for m in self.models]

        # Vote on which signals to take
        if self.voting == 'unanimous':
            return signals_agreed_by_all(all_signals)
        elif self.voting == 'majority':
            return signals_agreed_by_majority(all_signals)
        else:
            return weighted_vote(all_signals, weights=[m.confidence for m in self.models])
```

---

## Conclusion

This framework provides:

1. **Standardization:** All models tested identically
2. **Objectivity:** Evidence-based ranking, no bias
3. **Rigor:** Statistical requirements, overfit detection
4. **Scalability:** Easy to add new models/assets/regimes
5. **Professionalism:** Institutional-grade workflow

**Next Steps:**
1. Implement core components (see `QUANT_LAB_QUICK_START.md`)
2. Define acceptance rules (see `RULES_OF_THE_LAB.md`)
3. Run first experiment (baselines vs archetypes)
4. Make kill/keep decisions based on data

No more guessing. No more overfitting. Just solid quant research.
