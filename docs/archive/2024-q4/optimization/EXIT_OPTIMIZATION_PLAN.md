# Exit Strategy Optimization Plan

**Created**: 2025-10-21
**Status**: Ready for Implementation
**Goal**: Use ML (Bayesian Optimization) to find optimal exit strategy parameters across all phases

---

## TL;DR

Instead of manually testing exit configurations (Phase 2/3/4 Tier 1, 2, 3...), we can use **Bayesian optimization** (Optuna) to automatically search 10+ dimensional parameter space and find the sweet spot for each asset.

**Benefits**:
- Tests 100-200 configurations in hours (vs weeks of manual testing)
- Finds non-obvious parameter combinations humans miss
- Provides statistically robust top-10 configs ranked by composite score
- Can run in parallel for BTC, ETH, SPY

**Created Tool**: `bin/optimize_exit_strategies.py` - Ready to use (needs small integration first)

---

## What Gets Optimized

### Phase 2: Pattern Exits (2-Leg Pullback, Inside Bar Expansion)
- `pattern_confluence_threshold`: [1, 2, 3]
  - 1 = any 1 factor (too loose)
  - 2 = 2/3 factors (current)
  - 3 = all 3 factors (strict)

### Phase 2: Structure Invalidation
- `structure_min_hold_bars`: [8, 12, 16, 20]
  - Prevents exiting too early
  - Current: 12 bars
- `structure_rsi_long_threshold`: [20, 25, 30]
  - Lower = stricter (only exit if VERY oversold)
  - Current: 25
- `structure_rsi_short_threshold`: [70, 75, 80]
  - Higher = stricter (only exit if VERY overbought)
  - Current: 75
- `structure_vol_zscore_min`: [0.5, 1.0, 1.5, 2.0]
  - Volume spike requirement for structure break
  - Current: 1.0

### Phase 3: Regime-Aware Trailing Stops
- `trailing_stop_base_mult`: [1.5, 2.0, 2.5]
  - Base trailing stop distance (× ATR)
  - Current: 2.0
- `trailing_stop_trending_mult`: [2.0, 2.5, 3.0]
  - Wider stops in trending regimes
  - Current: 2.5

### Phase 4: Re-Entry Logic
- `reentry_confluence_threshold`: [0, 2, 3]
  - 0 = disable Phase 4 entirely
  - 2 = 2/3 confluence (RSI OR vol OR 4H)
  - 3 = 3/3 confluence (RSI AND vol AND 4H)
- `reentry_window_btc_eth`: [5, 7, 10] bars
  - How long after exit to allow re-entry
  - Current: 7
- `reentry_fusion_delta`: [0.03, 0.05, 0.10]
  - How much below threshold to re-enter
  - Current: 0.05

---

## Optimization Objective

**Maximize**: `Total_PNL × sqrt(Win_Rate) × sqrt(Profit_Factor) × Trade_Penalty`

**Why this metric**:
- `PNL`: Absolute returns (what we care about)
- `sqrt(Win_Rate)`: Consistency (50% win rate = 0.707 multiplier)
- `sqrt(Profit_Factor)`: Risk-adjusted returns (2.0 PF = 1.414 multiplier)
- `Trade_Penalty`: Penalize < 20 trades (insufficient sample size)

**Example**:
- Config A: $1,000 PNL, 40% WR, 1.5 PF, 30 trades → Score = 1000 × 0.632 × 1.225 × 1.0 = **774**
- Config B: $1,200 PNL, 30% WR, 1.2 PF, 25 trades → Score = 1200 × 0.548 × 1.095 × 1.0 = **720**
- **Config A wins** despite lower PNL (better balance)

---

## Implementation Steps

### Step 1: Add Environment Variable Support to Backtest

**File**: `bin/backtest_knowledge_v2.py`

**Add at top of `KnowledgeBacktester.__init__()`**:

```python
# Phase 2: Pattern Exit Parameters (read from env or use defaults)
import os
self.pattern_confluence_threshold = int(os.getenv('EXIT_PATTERN_CONFLUENCE', '2'))

# Phase 2: Structure Exit Parameters
self.structure_min_hold_bars = int(os.getenv('EXIT_STRUCT_MIN_HOLD', '12'))
self.structure_rsi_long_threshold = int(os.getenv('EXIT_STRUCT_RSI_LONG', '25'))
self.structure_rsi_short_threshold = int(os.getenv('EXIT_STRUCT_RSI_SHORT', '75'))
self.structure_vol_zscore_min = float(os.getenv('EXIT_STRUCT_VOL_Z', '1.0'))

# Phase 3: Trailing Stop Parameters
self.trailing_stop_base_mult = float(os.getenv('EXIT_TRAILING_BASE', '2.0'))
self.trailing_stop_trending_mult = float(os.getenv('EXIT_TRAILING_TREND', '2.5'))

# Phase 4: Re-Entry Parameters
self.reentry_confluence_threshold = int(os.getenv('EXIT_REENTRY_CONF', '3'))
self.reentry_window_btc_eth = int(os.getenv('EXIT_REENTRY_WINDOW', '7'))
self.reentry_fusion_delta = float(os.getenv('EXIT_REENTRY_DELTA', '0.05'))
```

**Then replace hard-coded values throughout the code with these variables**.

**Example Changes**:
```python
# BEFORE (line 527):
if bars_held < 12:

# AFTER:
if bars_held < self.structure_min_hold_bars:

# BEFORE (line 585):
if rsi < 25 and vol_z > 1.0:

# AFTER:
if rsi < self.structure_rsi_long_threshold and vol_z > self.structure_vol_zscore_min:

# BEFORE (line 806):
if confluence_score < 3:

# AFTER:
if confluence_score < self.reentry_confluence_threshold:
```

**Estimated Time**: 30-45 minutes (find/replace ~15 hard-coded values)

---

### Step 2: Test the Optimizer on BTC (Small Run)

```bash
python3 bin/optimize_exit_strategies.py \
  --asset BTC \
  --start 2024-07-01 \
  --end 2024-09-30 \
  --trials 20 \
  --output reports/exit_optimization
```

**Expected Runtime**: ~40 minutes (20 trials × 2 min per backtest)

**Output**: `reports/exit_optimization/BTC_2024-07-01_2024-09-30_exit_optimization.json`

**Example Output**:
```json
{
  "best_score": 1250.45,
  "best_params": {
    "pattern_confluence": 3,
    "struct_min_hold": 16,
    "struct_rsi_long": 20,
    "struct_rsi_short": 80,
    "struct_vol_z": 1.5,
    "trailing_base": 2.2,
    "trailing_trend": 2.8,
    "reentry_conf": 0,
    "reentry_window": 7,
    "reentry_delta": 0.05
  },
  "best_metrics": {
    "pnl": 2450.00,
    "trades": 45,
    "win_rate": 0.55,
    "profit_factor": 2.1,
    "max_drawdown": 0.04
  }
}
```

---

### Step 3: Full Optimization Run (All Assets)

```bash
# BTC - Full 2024 dataset, 200 trials
nohup python3 bin/optimize_exit_strategies.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --trials 200 \
  --output reports/exit_optimization \
  > logs/btc_exit_opt.log 2>&1 &

# ETH - Full 2024 dataset, 200 trials
nohup python3 bin/optimize_exit_strategies.py \
  --asset ETH \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --trials 200 \
  --output reports/exit_optimization \
  > logs/eth_exit_opt.log 2>&1 &

# SPY - Full 2024 dataset, 200 trials
nohup python3 bin/optimize_exit_strategies.py \
  --asset SPY \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --trials 200 \
  --output reports/exit_optimization \
  > logs/spy_exit_opt.log 2>&1 &
```

**Expected Runtime**: ~6-8 hours per asset (200 trials × 2 min per backtest)

**Can run in parallel** - all 3 assets complete in ~8 hours

---

### Step 4: Analyze Results & Apply Best Config

**Review Top 10 Configs**:
```python
import json

# Load BTC results
with open('reports/exit_optimization/BTC_2024-01-01_2024-12-31_exit_optimization.json') as f:
    results = json.load(f)

# Print top 10 ranked by composite score
for trial in results['top_10_trials']:
    print(f"Rank {trial['rank']}: Score={trial['score']:.2f}, PNL=${trial['metrics']['pnl']:.2f}, "
          f"WR={trial['metrics']['win_rate']:.1%}, PF={trial['metrics']['profit_factor']:.2f}")
    print(f"  Params: {trial['params']}")
    print()
```

**Apply Best Config to Backtest**:
```bash
# Test best config manually
export EXIT_PATTERN_CONFLUENCE=3
export EXIT_STRUCT_MIN_HOLD=16
export EXIT_STRUCT_RSI_LONG=20
export EXIT_STRUCT_RSI_SHORT=80
export EXIT_STRUCT_VOL_Z=1.5
export EXIT_TRAILING_BASE=2.2
export EXIT_TRAILING_TREND=2.8
export EXIT_REENTRY_CONF=0  # Disable Phase 4
export EXIT_REENTRY_WINDOW=7
export EXIT_REENTRY_DELTA=0.05

python3 bin/backtest_knowledge_v2.py --asset BTC --start 2024-01-01 --end 2024-12-31
```

**Verify Results Match Optimization**:
- If PNL/WR/PF match optimizer output → Success!
- If not → Debug environment variable reading

---

## Expected Outcomes

### Hypothesis 1: Phase 4 Will Be Disabled (reentry_conf=0)
Based on our manual testing, the optimizer will likely discover that disabling Phase 4 re-entry logic produces better results than any confluence configuration.

### Hypothesis 2: Pattern Confluence Will Be 3/3
Manual testing showed 2/3 is too loose (42% pattern exits). Optimizer should find 3/3 reduces pattern exits to <15% while improving PNL.

### Hypothesis 3: Structure Guards Will Be Tightened
- Min hold: 12 → 16-20 bars (prevent early exits)
- RSI thresholds: 25/75 → 20/80 (only exit on extreme moves)
- Vol z-score: 1.0 → 1.5-2.0 (require significant volume)

### Hypothesis 4: Trailing Stops Will Be Wider
- Base: 2.0 → 2.2-2.5 ATR (give winners more room)
- Trending: 2.5 → 2.8-3.0 ATR (even wider in trends)

### Hypothesis 5: Asset-Specific Sweet Spots
BTC, ETH, SPY may have different optimal configs due to volatility differences:
- BTC (high vol): Wider stops, stricter confluence
- ETH (medium vol): Moderate parameters
- SPY (low vol): Tighter stops, looser confluence

---

## Alternative: Grid Search (Simpler but Slower)

If Bayesian optimization is too complex, we can use grid search:

```python
# Define grid
grid = {
    'pattern_confluence': [2, 3],
    'struct_min_hold': [12, 16],
    'struct_rsi_long': [20, 25],
    'struct_vol_z': [1.0, 1.5],
    'trailing_base': [2.0, 2.5],
    'reentry_conf': [0, 3],
}

# Total combinations: 2 × 2 × 2 × 2 × 2 × 2 = 64
# Runtime: 64 × 2 min = 2 hours (faster than 200 trials Bayesian)
```

**Trade-off**: Grid search is faster but may miss optimal combinations between grid points.

---

## Next Steps (Priority Order)

### P0 (Do First):
1. Add environment variable support to `backtest_knowledge_v2.py` (30 min)
2. Test with manual env vars to verify it works (5 min)
3. Run small optimization test (20 trials, BTC Jul-Sep) (40 min)

### P1 (Do Next):
4. Run full optimization for all 3 assets (6-8 hours, can run overnight)
5. Analyze top 10 results per asset
6. Apply best config and verify results

### P2 (Nice-to-Have):
7. Create config files from best params (`configs/optimized/BTC_exits.json`)
8. Add optimizer to CI/CD for periodic re-optimization
9. Build visualization dashboard for comparing configs

---

**Status**: Optimizer tool ready (`bin/optimize_exit_strategies.py`). Awaiting Step 1 (environment variable integration) to unlock ML-driven exit optimization ⚡
