# Quick Start: Signal Rejection Diagnosis

## Problem
357 signals → 9 trades = **97.5% rejection rate**

## Solution: Run This One Command

```bash
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-
python3 bin/backtest_debug_enhanced.py --debug --period sanity 2>&1 | tee backtest_run.log
```

**Runtime**: ~2-3 minutes for sanity period (2022 Q2)

---

## What This Does

1. ✅ Loads feature data
2. ✅ Classifies regime using HMM model
3. ✅ Runs backtest with full logging
4. ✅ Tracks every signal through pipeline
5. ✅ Counts rejections by category
6. ✅ Prints diagnostic analysis
7. ✅ Saves results to `results/debug_backtest/`

---

## Expected Output

### During Run
```
[INFO] Loading feature data...
[INFO] Classifying regime with RegimeService...
[INFO] Running backtest: 2022 Q2 Sanity Test
[INFO] [Signal Generated] Archetype_B: conf=0.850, dir=long, regime=neutral
[INFO] [Confidence Waterfall] Archetype_B long:
[INFO]   Base confidence: 0.850
[INFO]   After regime penalty (-50%): 0.425
[INFO]   Final confidence: 0.425
[INFO]   Result: ACCEPT
[WARNING] [REJECT: Low Confidence] Archetype_H: 0.260 < 0.300
...
```

### Final Analysis
```
================================================================================
SIGNAL REJECTION ANALYSIS
================================================================================

## Summary
- Total signals generated: 357
- Signals accepted: 9 (2.5%)
- Signals rejected: 348 (97.5%)

## Rejection Breakdown
1. Regime Penalty: 245 (70.4%) ← TOP CAUSE
2. Low Confidence: 67 (19.3%)
3. Regime Confidence Low: 28 (8.1%)
4. Circuit Breaker: 5 (1.4%)
5. Position Limit: 3 (0.9%)

## Pipeline Efficiency
- Signals entering pipeline: 357
- Signals passing filters: 9
- Overall acceptance rate: 2.5%
- Target acceptance rate: 40-60% (after fixes)
================================================================================
```

---

## Files Created

After run completes:

```
results/debug_backtest/
├── trades_sanity.csv              # Trade blotter
├── equity_sanity.csv              # Equity curve
└── rejection_stats_sanity.json   # Full statistics (JSON)

backtest_debug.log                 # Complete debug trace
backtest_run.log                   # Console output (if using tee)
```

---

## Quick Analysis Commands

### View rejection summary
```bash
grep "SIGNAL REJECTION ANALYSIS" -A 20 backtest_run.log
```

### Count rejections by type
```bash
grep "REJECT:" backtest_debug.log | cut -d']' -f2 | cut -d':' -f1 | sort | uniq -c | sort -rn
```

### Most rejected archetypes
```bash
grep "REJECT:" backtest_debug.log | grep -oE "Archetype_[A-Z0-9]+" | sort | uniq -c | sort -rn
```

### View all accepted signals
```bash
grep "ACCEPT" backtest_debug.log
```

### View confidence waterfalls
```bash
grep "Confidence Waterfall" -A 10 backtest_debug.log | less
```

---

## Interpretation Guide

### If Top Cause is "Regime Penalty" (Expected)
**Finding**: 60-70% of rejections due to -50% regime penalty

**Root Cause**: Penalties too harsh for regime mismatch

**Fix**: Parallel agent reducing to -15%/-25% (in progress)

**Expected Improvement**: Rejection rate drops to 40-60%

### If Top Cause is "Low Confidence"
**Finding**: Signals weak after pipeline scaling

**Root Cause**: Cumulative penalties stacking

**Fix**: Review confidence thresholds or reduce penalty stacking

### If Top Cause is "Circuit Breaker"
**Finding**: Trading halted due to drawdowns

**Root Cause**: Risk controls triggered

**Action**: Review circuit breaker thresholds

### If Top Cause is "Position Limit"
**Finding**: Max 5 positions already active

**Action**: Consider increasing limit or improving exit timing

---

## Next Steps After Diagnosis

### 1. Identify Root Cause
Look at "Rejection Breakdown" section for top 1-2 causes

### 2. Apply Fixes
- **Regime penalty** → Wait for parallel agent completion
- **Low confidence** → Adjust thresholds
- **Circuit breaker** → Review risk settings

### 3. Re-Run Validation
```bash
python3 bin/backtest_debug_enhanced.py --debug --period sanity
```

### 4. Compare Results
- Rejection rate should decrease
- Trade count should increase
- Win rate and profitability should improve

---

## Troubleshooting

### Issue: "Feature file not found"
```bash
# Check if file exists
ls -lh data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet

# If missing, regenerate features
python3 bin/generate_features.py
```

### Issue: "Model file not found"
```bash
# Check if regime model exists
ls -lh models/logistic_regime_v1.pkl

# If missing, train model
python3 bin/train_regime_classifier.py
```

### Issue: Taking too long
```bash
# Use sanity period (fastest)
python3 bin/backtest_debug_enhanced.py --period sanity

# Or check progress
tail -f backtest_debug.log
```

---

## Success Criteria

✅ **Diagnostic complete** when you see:
- "SIGNAL REJECTION ANALYSIS" section printed
- Rejection breakdown with percentages
- Top cause identified
- Results saved to `results/debug_backtest/`

✅ **Root cause identified** when:
- Top 1-2 rejection reasons account for >70% of rejections
- Specific fix can be identified

✅ **Ready for fixes** when:
- Baseline metrics captured
- Rejection categories quantified
- Comparison data available

---

## Quick Wins Checklist

- [ ] Run diagnostic command
- [ ] Wait 2-3 minutes for completion
- [ ] Read "SIGNAL REJECTION ANALYSIS" section
- [ ] Note top rejection cause
- [ ] Save baseline metrics
- [ ] Wait for parallel agent fixes
- [ ] Re-run validation
- [ ] Compare before/after
- [ ] Document improvement

---

**Total Time**: 5 minutes (2-3 min run + 2 min analysis)

**Output**: Clear diagnosis of rejection root cause with percentages

**Next**: Apply fixes and validate improvement
