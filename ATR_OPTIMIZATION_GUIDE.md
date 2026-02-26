# ATR Multiplier Optimization Guide

## Problem

After fixing the `atr_tp_mult` bug and implementing per-archetype ATR multipliers, backtest PnL dropped from $119K to $59K. Analysis showed:

- **Dedup alone** reduced PnL by only 10% ($119K → $107K) — removed accidental leverage on duplicated trades
- **Hand-tuned ATR values** caused the real drop ($107K → $59K, -45%) — especially hurt wick_trap (PnL $42K→$15K)

The architecture is correct (per-archetype risk params + signal dedup). The specific ATR values need **data-driven optimization** rather than hand-tuning.

## Solution: Optuna ATR Optimization

### What Gets Optimized

**32 parameters for 16 archetypes:**
- `atr_stop_mult`: Stop loss distance in ATR multiples [1.0 - 3.5]
- `atr_tp_mult`: Take profit distance in ATR multiples [2.0 - 6.0]

**Constraint:** `atr_tp_mult > atr_stop_mult` (reward must exceed risk)

### Objective

Maximize **Profit Factor** on training period (2020-2022), validate on test period (2023-2024).

**Hard constraints:**
- Minimum 10 trades per archetype (statistical significance)
- ATR TP > ATR stop (positive risk:reward)

## Step-by-Step Optimization

### 1. Run Baseline Backtest (Current Hand-Tuned Values)

```bash
python3 bin/backtest_v11_standalone.py \
  --start-date 2020-01-01 \
  --commission-rate 0.0002 \
  --slippage-bps 3 \
  | tee results/baseline_hand_tuned.txt
```

**Expected baseline (2020-2024 with dedup + hand-tuned ATR):**
- Trades: ~2,309
- PF: ~1.094
- PnL: ~$59K
- MaxDD: -19.6%
- Sharpe: 0.900

Record these numbers for comparison.

### 2. Run Optuna Optimization (100 trials, ~2 hours)

```bash
python3 bin/optuna_optimize_atr_multipliers.py \
  --n-trials 100 \
  --n-jobs 1 \
  --seed 42 \
  --commission-rate 0.0002 \
  --slippage-bps 3 \
  | tee results/optuna_atr_log.txt
```

**Output files:**
- `configs/optimized/optuna_atr_best.json` — Best parameters + metadata
- `configs/optimized/optuna_atr_best_raw_params.json` — Raw parameter values
- `configs/optimized/optuna_atr_best_all_trials.json` — All trial results

**Performance tracking:**
- Each trial: ~60-120s (train + test backtest)
- Progress updates every 10 trials
- Best trial marked with `***BEST***`
- Pruning: ~50% of trials pruned (skip test backtest if train PF below median)

### 3. Review Results

```bash
# View optimization summary
tail -100 results/optuna_atr_log.txt

# Check top 5 trials
grep "TOP 5 TRIALS" -A 10 results/optuna_atr_log.txt

# See parameter importance (which ATR params matter most)
grep "PARAMETER IMPORTANCE" -A 30 results/optuna_atr_log.txt
```

**Key metrics to check:**
- **Train PF** — should be > 1.2 (baseline was 1.094)
- **Test PF** — should be > 1.0 (OOS validation)
- **WFE (Walk-Forward Efficiency)** — should be > 70% (test_pf / train_pf × 100)
- **Test trades** — should be > 100 (enough data)

**Warning signs:**
- WFE < 70% = overfitting (test performance much worse than train)
- Best params at search space boundaries (1.0, 3.5, 6.0) = need wider range
- Single archetype dominates (>80% of trades) = concentration risk

### 4. Apply Optimized Parameters

**WITH BACKUP (recommended first time):**
```bash
python3 bin/apply_optimized_atr_params.py \
  configs/optimized/optuna_atr_best.json \
  --backup
```

This creates `configs/archetypes_backup_YYYYMMDD_HHMMSS/` with copies of all YAML files.

**Without backup:**
```bash
python3 bin/apply_optimized_atr_params.py \
  configs/optimized/optuna_atr_best.json
```

### 5. Validate with Full Backtest

```bash
python3 bin/backtest_v11_standalone.py \
  --start-date 2020-01-01 \
  --commission-rate 0.0002 \
  --slippage-bps 3 \
  | tee results/post_optuna_full_backtest.txt
```

**Compare to baseline:**
- Should see PF improvement (>1.2)
- PnL should increase from $59K
- Trades may change ± 10% (different SL/TP distances)
- MaxDD should be similar or better

**If results are worse:**
- Check for WFE < 70% (overfitting)
- Restore from backup: `cp configs/archetypes_backup_*/\*.yaml configs/archetypes/`
- Try optimization with more trials (200+) or different seed

### 6. Deploy to Production

**After validation passes:**

```bash
# Build and deploy
cd dashboard && npm run build && cd ..
./deploy/deploy.sh

# Monitor on server
ssh -i ~/.ssh/oracle_bullmachine ubuntu@165.1.79.19
sudo journalctl -u coinbase-paper -f
```

**Watch for:**
- First few signals use new SL/TP distances
- Trade frequency (should be similar, ±20%)
- Win rate on live data (should match backtest within ±5pp)

## Expected Improvements

Based on initial isolation test showing hand-tuned ATR caused -45% PnL drop:

**Conservative estimate:**
- Train PF: 1.094 → **1.20-1.35** (+10-25%)
- OOS PF: ~1.10 → **1.15-1.25** (+5-15%)
- PnL (2020-2024): $59K → **$80K-$110K** (+35-85%)

**Best case (if optimization finds significantly better ATR values):**
- Train PF: **1.40-1.50**
- OOS PF: **1.25-1.35**
- PnL: **$110K-$130K**

**Key archetype to watch:**
- **wick_trap** — 800+ trades, hurt most by tight 1.8x stop. Optimal stop likely 2.3-2.8x.

## Advanced: Multi-Objective Optimization

If you want to optimize for multiple goals simultaneously (PF + Sharpe + low DD):

```python
# Edit bin/optuna_optimize_atr_multipliers.py line ~225
# Change return statement to return tuple:

return (train_pf, train_sharpe, -train_stats.get('max_drawdown', 0))

# And change study creation (line ~544):
study = optuna.create_study(
    study_name='bull_machine_atr_multipliers',
    directions=['maximize', 'maximize', 'maximize'],  # PF, Sharpe, -DD
    sampler=sampler,
    pruner=pruner,
)
```

This finds Pareto-optimal solutions balancing all three objectives.

## Troubleshooting

### Optimization crashes / hangs

**Symptom:** Trial fails with error or hangs indefinitely

**Fix:**
```bash
# Check feature store
python3 -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet'); print(df.shape, df.index.min(), df.index.max())"

# Run single backtest manually
python3 bin/backtest_v11_standalone.py --start-date 2020-01-01 --end-date 2022-12-31
```

### All trials pruned / rejected

**Symptom:** No trials complete, all show `REJECTED (too_few_trades)`

**Fix:**
- Check that `signal_dedup.mode` is NOT `disabled` in config
- Check that at least 6 archetypes are enabled (not in `disabled_archetypes`)
- Lower min_trades constraint in script (line ~337: change `< 10` to `< 5`)

### WFE very low (< 50%)

**Symptom:** Test PF much lower than train PF (overfitting)

**Fix:**
- Increase train period (2018-2022 instead of 2020-2022)
- Reduce search space precision (change `step=0.1` to `step=0.2`)
- Use MedianPruner to kill bad trials early (default: enabled)

### Parameters at boundaries

**Symptom:** Best stop_mult = 1.0 or 3.5, or best tp_mult = 6.0

**Fix:**
- Expand search range in script:
  - Line ~157: change `1.0, 3.5` to `0.8, 4.0`
  - Line ~163: change `6.0` to `8.0`
- Re-run optimization with wider range

## Comparison to Hand-Tuned Values

| Archetype | Hand-Tuned Stop | Hand-Tuned TP | Rationale |
|-----------|:---------------:|:-------------:|-----------|
| wick_trap | 1.8 | 3.0 | Fast reversal: tight stop below wick |
| trap_within_trend | 2.5 | 4.0 | Trend continuation: ride the move |
| spring | 2.0 | 3.5 | Reversal at accumulation: room for retest |
| exhaustion_reversal | 1.5 | 2.5 | Fastest resolution: tightest params |
| retest_cluster | 2.0 | 3.0 | Multi-confluence fakeout then real move |
| liquidity_sweep | 2.5 | 3.5 | Sweep + reclaim: standard stop, moderate target |
| liquidity_vacuum | 2.0 | 4.0 | Crisis reversal: bounces hard |
| failed_continuation | 2.0 | 3.5 | FVG-based reversal: stop below FVG |
| fvg_continuation | 2.5 | 4.5 | Displacement + BOS: let trend run |
| order_block_retest | 3.0 | 5.0 | OB needs wide stop; continuation target large |
| confluence_breakout | 2.5 | 5.0 | Coil break: medium stop, wide target |
| liquidity_compression | 3.0 | 5.0 | Compression range wide; breakout target wide |
| funding_divergence | 2.5 | 3.5 | Contrarian squeeze: medium params |
| long_squeeze | 2.8 | 3.8 | Short on overheated longs: wider stop to survive whipsaw |
| whipsaw | 1.5 | 2.0 | Scalp mean reversion: very tight |
| volume_fade_chop | 1.5 | 2.0 | Chop fade: very tight |

After optimization, compare Optuna values to these hand-tuned guesses. Expect:
- **wick_trap stop** to widen from 1.8 → 2.3-2.8 (data shows 2.5 was better)
- **order_block_retest** may need even wider stop (3.0 → 3.2-3.5)
- **Scalpers (whipsaw, volume_fade_chop)** may stay tight at 1.5/2.0

## Files Created

- `bin/optuna_optimize_atr_multipliers.py` — Main optimization script (32 params, PF objective)
- `bin/apply_optimized_atr_params.py` — Apply results to YAML files
- `ATR_OPTIMIZATION_GUIDE.md` — This guide

## Next Steps After Optimization

1. **Update MEMORY.md** — Add Optuna ATR results to Critical Lessons
2. **Commit optimized YAMLs** — `git add configs/archetypes/*.yaml && git commit -m "feat: apply Optuna-optimized ATR multipliers"`
3. **Update plan file** — Mark ATR optimization as COMPLETE
4. **Monitor live performance** — Watch first 20-30 live trades with new SL/TP
5. **Re-optimize quarterly** — Market conditions change, ATR optimal values may drift

---

**Last Updated:** 2026-02-23
**Status:** Ready for use
**Expected Runtime:** ~2 hours for 100 trials
