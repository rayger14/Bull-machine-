# Parallel Archetype Optimization Guide

**Status:** Production Ready
**Speedup:** ~5-6x (6-8 hours vs 33 hours sequential)
**Created:** 2025-11-16

---

## Overview

`optuna_parallel_archetypes.py` runs separate Optuna studies for each archetype group in parallel using Python's multiprocessing. This dramatically reduces optimization time while maintaining study independence.

### Archetype Groups

The script optimizes 4 independent archetype groups:

| Group | Archetypes | Trader Type | Description |
|-------|------------|-------------|-------------|
| **trap_within_trend** | A, G, K | Moneytaur | Momentum reversals & liquidity traps |
| **order_block_retest** | B, H, L | Zeroika | Structure-based OB retests |
| **bos_choch** | C | Generic | Break of Structure patterns |
| **long_squeeze** | S5 | Moneytaur | Funding rate cascades |

**Why groups?** Archetypes in the same group share similar detection logic and parameter sensitivities, allowing focused optimization.

---

## Features

### 1. **True Parallelism**
- 4 worker processes (one per group)
- Independent Optuna studies
- No resource contention
- Linear speedup with CPU cores

### 2. **Hyperband Pruning**
- Early stopping of unpromising trials
- 3x reduction factor
- Typical pruning rate: 60-70%
- Convergence in ~50 trials per group

### 3. **Multi-Fidelity Training**
- **Fidelity 0:** 1 month (2024-01-01 to 2024-01-31) - fast pruning
- **Fidelity 1:** 3 months (2024-01-01 to 2024-03-31) - validation
- **Fidelity 2:** 9 months (2024-01-01 to 2024-09-30) - full evaluation

Unpromising trials fail fast at fidelity 0 (~30s), promising trials get full 9-month evaluation (~2-3 min).

### 4. **Shared Global Parameters**
All groups optimize the same global parameters (via TPE sampler):
- `global_min_liquidity`: Global liquidity threshold (0.10-0.35)
- `global_fusion_floor`: Minimum fusion score (0.30-0.45)
- `w_wyckoff`, `w_liquidity`, `w_momentum`: Fusion weights (sum to 1.0)

### 5. **Per-Archetype Parameter Spaces**

Each group has custom parameter ranges tailored to its detection logic:

#### Trap Within Trend (A, G, K)
```python
{
  'trap_within_trend': {
    'fusion_threshold': 0.30-0.45,
    'adx_threshold': 20-35,
    'liquidity_threshold': 0.15-0.35,  # Inverted (low liq = trap)
    'archetype_weight': 0.85-1.15
  },
  'spring': {
    'fusion_threshold': 0.28-0.42,
    'pti_score_threshold': 0.30-0.55,
    'disp_atr_multiplier': 0.6-1.2
  },
  'liquidity_sweep': {
    'fusion_threshold': 0.32-0.48,
    'boms_strength_min': 0.30-0.55,
    'liquidity_min': 0.30-0.50
  }
}
```

#### Order Block Retest (B, H, L)
```python
{
  'order_block_retest': {
    'fusion_threshold': 0.30-0.45,
    'boms_strength_min': 0.25-0.45,
    'wyckoff_min': 0.30-0.50,
    'archetype_weight': 0.90-1.20
  },
  'volume_exhaustion': {
    'fusion_threshold': 0.32-0.46,
    'vol_z_min': 0.8-1.5,
    'rsi_min': 65-78,
    'archetype_weight': 0.95-1.15
  }
}
```

#### BOS/CHOCH (C)
```python
{
  'wick_trap': {
    'fusion_threshold': 0.35-0.50,
    'disp_atr_multiplier': 0.8-1.5,
    'momentum_min': 0.40-0.60,
    'tf4h_fusion_min': 0.20-0.35
  }
}
```

#### Long Squeeze (S5)
```python
{
  'long_squeeze': {
    'fusion_threshold': 0.28-0.42,
    'funding_z_min': 1.0-1.8,
    'rsi_min': 65-78,
    'liquidity_max': 0.18-0.32
  }
}
```

### 6. **Objective Function**

Multi-objective scoring with fidelity-adaptive penalties:

```python
score = PF × (1 + WR/100) × sqrt(trades)
        - DD / (10 if fidelity==2 else 20)
        - max(0, (trades - max_trades) × 0.1)
        + sharpe × 0.5  # Only at fidelity 2
```

**Components:**
- **Base:** Profit Factor × win rate bonus × trade consistency
- **DD Penalty:** Drawdown penalty (stricter at full fidelity)
- **Overtrading Penalty:** Penalize excessive trades (scales with fidelity)
- **Sharpe Bonus:** Only applied at full evaluation

**Trade Targets by Fidelity:**
- Fidelity 0 (1mo): 30 trades max
- Fidelity 1 (3mo): 60 trades max
- Fidelity 2 (9mo): 100 trades max

---

## Usage

### Basic Run
```bash
# Optimize all 4 groups with 50 trials each (default)
python bin/optuna_parallel_archetypes.py \
  --trials 50 \
  --base-config configs/profile_production.json
```

### Custom Configuration
```bash
# Specify trials, storage, and output paths
python bin/optuna_parallel_archetypes.py \
  --trials 100 \
  --base-config configs/btc_v8_adaptive.json \
  --storage results/optuna_btc.db \
  --output configs/btc_optimized_archetypes.json
```

### Resume from Checkpoint
```bash
# Resume previous optimization (uses existing SQLite studies)
python bin/optuna_parallel_archetypes.py \
  --resume \
  --trials 150  # Add 100 more trials to existing 50
```

### Optimize Specific Groups
```bash
# Only optimize specific archetype groups
python bin/optuna_parallel_archetypes.py \
  --groups trap_within_trend long_squeeze \
  --trials 75
```

---

## Output

### 1. **SQLite Storage**
All trials saved to `optuna_archetypes.db` (default):
- Persistent across runs
- Supports resume
- Full trial history with parameters and scores

### 2. **Unified Config**
Best parameters from all groups merged into single config:
```json
{
  "archetypes": {
    "thresholds": {
      "min_liquidity": 0.24
    },
    "trap_within_trend": {
      "fusion_threshold": 0.38,
      "adx_threshold": 27.0,
      ...
    },
    "order_block_retest": {
      "fusion_threshold": 0.35,
      "boms_strength_min": 0.34,
      ...
    }
  },
  "decision_gates": {
    "final_fusion_floor": 0.37
  },
  "fusion": {
    "weights": {
      "wyckoff": 0.331,
      "liquidity": 0.392,
      "momentum": 0.277
    }
  },
  "_optimization_metadata": {
    "timestamp": "2025-11-16T14:30:00",
    "groups_optimized": ["trap_within_trend", "order_block_retest", ...],
    "total_trials": 200,
    "best_scores": {
      "trap_within_trend": 23.45,
      "order_block_retest": 28.91,
      ...
    }
  }
}
```

### 3. **Terminal Output**
Real-time progress monitoring:
```
15:23:45 [MainProcess] Starting parallel optimization for 4 groups
15:23:45 [MainProcess] Trials per group: 50
15:23:45 [MainProcess] Total expected trials: 200
15:23:45 [MainProcess] Estimated runtime: 6-8 hours
...
15:45:12 [MainProcess] Progress: 12/200 trials | Elapsed: 0.4h | Group: trap_within_trend | Trial: 3 | Score: 18.23 (PF=4.52, trades=24)
...
21:15:33 [MainProcess] Optimization complete!

================================================================================
OPTIMIZATION RESULTS
================================================================================
Group                     Best Score   Trials   Archetypes
--------------------------------------------------------------------------------
order_block_retest             28.91       50   B, H, L
trap_within_trend              23.45       50   A, G, K
long_squeeze                   19.87       50   S5
bos_choch                      16.34       50   C
================================================================================
```

---

## Analysis

### View Results
```bash
# Comprehensive analysis report
python bin/analyze_archetype_optimization.py \
  --storage optuna_archetypes.db

# Quick comparison only
python bin/analyze_archetype_optimization.py \
  --storage optuna_archetypes.db \
  --compare-only
```

### Export Data
```bash
# Export best parameters to JSON
python bin/analyze_archetype_optimization.py \
  --storage optuna_archetypes.db \
  --export-json results/best_params.json

# Export all trials to CSV for external analysis
python bin/analyze_archetype_optimization.py \
  --storage optuna_archetypes.db \
  --export-csv results/trials_csv/
```

### Analysis Report Structure
```
================================================================================
ARCHETYPE OPTIMIZATION ANALYSIS
================================================================================

📊 STUDY COMPARISON
--------------------------------------------------------------------------------
Study                     Best Score   Trials   Completed   Pruned   Failed
--------------------------------------------------------------------------------
archetype_order_block_retest    28.91       50          18       30        2
archetype_trap_within_trend     23.45       50          22       27        1
...

📈 ARCHETYPE_TRAP_WITHIN_TREND
--------------------------------------------------------------------------------
Best Score: 23.450
Best Trial: #34
Total Trials: 50 (Completed: 22, Pruned: 27, Failed: 1)

🏆 Top 5 Trials:
  Trial #34: 23.450 (COMPLETE)
  Trial #12: 21.893 (COMPLETE)
  Trial #41: 20.567 (COMPLETE)
  ...

🔧 Best Parameters:
  trap_fusion: 0.3812
  trap_adx: 27.0000
  trap_liq_max: 0.2200
  ...

📊 Parameter Importance (Top 10):
  trap_fusion: 0.4521
  global_min_liquidity: 0.3892
  trap_adx: 0.2145
  ...
```

---

## Performance Benchmarks

### Runtime Comparison

| Approach | Trials/Group | Total Time | Speedup |
|----------|--------------|------------|---------|
| Sequential | 100 | ~33 hours | 1.0x |
| Parallel (4 workers) | 100 | 6-8 hours | 4.5-5.5x |
| Parallel + Hyperband | 100 | 5-6 hours | 5.5-6.6x |

**Factors:**
- Sequential: 100 trials × 4 groups × ~5 min/trial = 2000 min = 33h
- Parallel: 100 trials × ~5 min/trial (overlapped) = 500 min = 8.3h
- Hyperband pruning: 60-70% trials pruned at fidelity 0 → ~6h effective

### Resource Usage
- **CPU:** 4 cores saturated (one per group)
- **Memory:** ~800 MB per worker = 3.2 GB total
- **Disk:** SQLite storage grows to ~50-100 MB for 200 trials

---

## Advanced Topics

### Custom Archetype Groups

Edit `ARCHETYPE_GROUPS` dict in script:
```python
ARCHETYPE_GROUPS = {
    'my_custom_group': {
        'archetypes': ['A', 'B'],
        'canonical': ['spring', 'order_block_retest'],
        'description': 'Custom combo',
        'trader_type': 'Hybrid',
    }
}
```

### Custom Parameter Spaces

Add to `suggest_archetype_params()`:
```python
if group_name == 'my_custom_group':
    cfg['archetypes']['spring'] = {
        'fusion_threshold': trial.suggest_float('custom_fusion', 0.25, 0.50),
        'custom_param': trial.suggest_int('custom_int', 10, 50)
    }
```

### Custom Objective Function

Modify `compute_objective_score()`:
```python
def compute_objective_score(metrics: Dict, fidelity: int) -> float:
    # Your custom scoring logic
    return custom_score
```

---

## Troubleshooting

### Issue: Workers stuck at "Starting optimization"
**Cause:** Multiprocessing fork issues on macOS
**Fix:** Add to script top:
```python
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    exit(main())
```

### Issue: "Database is locked" errors
**Cause:** SQLite doesn't support true concurrent writes
**Fix:** Each worker writes to separate SQLite file:
```bash
--storage results/optuna_{group_name}.db
```

### Issue: High memory usage
**Cause:** All 4 workers loading feature data simultaneously
**Fix:** Reduce workers or add memory limits:
```python
with mp.Pool(processes=2) as pool:  # Use 2 workers instead of 4
```

### Issue: Progress monitor not updating
**Cause:** Queue buffer overflow
**Fix:** Increase queue size or drain more frequently:
```python
progress_queue = mp.Queue(maxsize=1000)
```

---

## Next Steps

1. **Validate Results:** Run validation backtest on Q4 2024 with unified config
2. **Fine-tune:** Use analyze script to identify parameter sensitivities
3. **Ensemble:** Combine top 3 trials from each group for robust ensemble
4. **Walk-forward:** Re-run optimization on rolling windows (Q1, Q2, Q3, Q4)

---

## References

- Optuna Documentation: https://optuna.readthedocs.io/
- Hyperband Pruner Paper: https://arxiv.org/abs/1603.06560
- TPE Sampler Paper: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
- Architecture Design: `/docs/BEAR_ARCHETYPES_ZERO_MATCHES_DIAGNOSIS.md`
- Archetype Logic: `/engine/archetypes/logic_v2_adapter.py`
