# Variant Configs Testing Guide

**Quick Start for Backtesting Simplified S1/S4/S5 Configs**

---

## TL;DR - Run All Variants

```bash
cd /Users/raymondghandchi/Bull-machine-/Bull-machine-

# S1 Liquidity Vacuum variants (expecting 40-60 trades in baseline, fewer in simpler)
python bin/backtest_knowledge_v2.py configs/variants/s1_core.json
python bin/backtest_knowledge_v2.py configs/variants/s1_core_plus_time.json
python bin/backtest_knowledge_v2.py configs/variants/s1_full.json

# S4 Funding Divergence variants (expecting 12 trades in full, more in core)
python bin/backtest_knowledge_v2.py configs/variants/s4_core.json
python bin/backtest_knowledge_v2.py configs/variants/s4_core_plus_macro.json
python bin/backtest_knowledge_v2.py configs/variants/s4_full.json

# S5 Long Squeeze variants (expecting 9 trades in full, more in core)
python bin/backtest_knowledge_v2.py configs/variants/s5_core.json
python bin/backtest_knowledge_v2.py configs/variants/s5_core_plus_wyckoff.json
python bin/backtest_knowledge_v2.py configs/variants/s5_full.json
```

---

## What Each Variant Tests

### S1 Variants (Liquidity Vacuum - Capitulation Reversals)

**Core (s1_core.json)** - MINIMAL COMPLEXITY
- Pure Wyckoff engine only
- No regime or time filtering
- Tests raw capitulation detection quality
- Expected: More trades, higher noise

**Core+Time (s1_core_plus_time.json)** - INTERMEDIATE
- Wyckoff + temporal confluence
- Time-of-day filtering added
- Tests whether time context improves signal quality
- Expected: Fewer false positives than core

**Full (s1_full.json)** - PRODUCTION
- All 6 domain engines enabled
- Full macro regime filtering
- Regime-aware weighting
- Production target: 40-60 trades/year, 50-60% win rate

---

### S4 Variants (Funding Divergence - Short Squeezes)

**Core (s4_core.json)** - MINIMAL COMPLEXITY
- Pure funding divergence logic
- No regime weighting
- Fires in ALL market conditions
- Expected: High trade frequency (not bear-specific)

**Core+Macro (s4_core_plus_macro.json)** - INTERMEDIATE
- Funding + macro regime routing
- Regime weighting applied (crisis=1.5, bear=1.0, bull=0.5)
- Tests regime discrimination impact
- Expected: Fewer trades in bull, more in bear

**Full (s4_full.json)** - PRODUCTION
- All optimizations applied (PF 2.22)
- Bear market specialist behavior
- Production target: 12 trades/year, 55.7% win rate, 0 in bull markets

---

### S5 Variants (Long Squeeze - Failed Rallies)

**Core (s5_core.json)** - MINIMAL COMPLEXITY
- Funding + RSI only
- No Wyckoff pattern detection
- Simple momentum + positioning setup
- Expected: High frequency, structure-agnostic

**Core+Wyckoff (s5_core_plus_wyckoff.json)** - INTERMEDIATE
- Funding + RSI + Wyckoff distribution
- Pattern recognition added
- Tests impact of structural market analysis
- Expected: More selective, fewer false positives

**Full (s5_full.json)** - PRODUCTION
- All features + macro regime routing
- Optimized routing (risk_off=2.2, crisis=2.5)
- Bear market specialist (disabled in bull markets)
- Production target: 9 trades/year, 55.6% win rate

---

## Performance Expectations by Complexity Level

### Expected Trade Frequency Scaling

**S1 Liquidity Vacuum:**
```
Core:              ~100-150 trades/year (high noise)
Core+Time:         ~60-80 trades/year (filtered)
Full (Production): 40-60 trades/year (optimized)
```

**S4 Funding Divergence:**
```
Core:              ~30-40 trades/year (all regimes)
Core+Macro:        ~15-20 trades/year (regime-aware)
Full (Production): ~12 trades/year (optimized bear-only)
```

**S5 Long Squeeze:**
```
Core:              ~20-30 trades/year (all regimes)
Core+Wyckoff:      ~12-15 trades/year (pattern-filtered)
Full (Production): ~9 trades/year (bear-optimized)
```

---

## Analysis Questions to Answer

### Complexity Efficiency
1. Does `core_plus` achieve 80%+ of full production performance with 50% of complexity?
2. What's the performance gain per added domain engine?
3. Is full complexity justified by measurable performance improvement?

### Regime Sensitivity
4. How much does regime routing reduce false positives?
5. Does full config correctly abstain in wrong regimes?
6. Is regime discrimination worth the added feature complexity?

### Filter Effectiveness
7. For S1: Does temporal confluence improve timing or just reduce frequency?
8. For S4: Does macro routing prevent bull market false positives effectively?
9. For S5: Does Wyckoff distribution detection improve signal quality?

### Practical Deployability
10. Which variant offers best risk/reward tradeoff?
11. Can simpler variant reduce operational overhead while maintaining profitability?
12. Where are diminishing returns in complexity?

---

## Comparing Results

### Quick Metrics Comparison

Create a comparison table after running all variants:

| Archetype | Variant | Trades/Yr | Win Rate | Sharpe | Profit Factor | Drawdown | Complexity |
|-----------|---------|-----------|----------|--------|---------------|----------|-----------|
| S1 | Core | ? | ? | ? | ? | ? | 1/6 |
| S1 | Core+Time | ? | ? | ? | ? | ? | 2/6 |
| S1 | Full | 40-60 | 50-60% | ? | ? | ? | 6/6 |
| S4 | Core | ? | ? | ? | ? | ? | 1/6 |
| S4 | Core+Macro | ? | ? | ? | ? | ? | 2/6 |
| S4 | Full | 12 | 55.7% | ? | 2.22 | ? | 6/6 |
| S5 | Core | ? | ? | ? | ? | ? | 2/6 |
| S5 | Core+Wyckoff | ? | ? | ? | ? | ? | 3/6 |
| S5 | Full | 9 | 55.6% | ? | 1.86 | ? | 6/6 |

---

## Testing Workflow

### Step 1: Run Single Variant
```bash
python bin/backtest_knowledge_v2.py configs/variants/s1_core.json
```

### Step 2: Capture Key Metrics
From backtest output, record:
- Total trades
- Trades per year
- Win rate / Profit factor
- Sharpe ratio / Sortino
- Max drawdown
- Average trade duration

### Step 3: Compare Across Variants
```bash
# Create simple comparison by running all and collecting results
for variant in s1_core s1_core_plus_time s1_full; do
  echo "Testing $variant..."
  python bin/backtest_knowledge_v2.py configs/variants/${variant}.json 2>&1 | grep -E "trades|win|sharpe|PF|drawdown"
done
```

### Step 4: Analyze Tradeoffs
- Plot complexity vs performance
- Identify diminishing returns
- Find optimal config for operational deployment

---

## File Locations

```
Active Variant Configs:
/configs/variants/
├── s1_core.json
├── s1_core_plus_time.json
├── s1_full.json
├── s4_core.json
├── s4_core_plus_macro.json
├── s4_full.json
├── s5_core.json
├── s5_core_plus_wyckoff.json
└── s5_full.json

Documentation:
/VARIANT_CONFIGS_SUMMARY.md      (detailed technical specs)
/VARIANT_TESTING_GUIDE.md        (this file)
```

---

## Key Insights from Config Design

### S1 Design Philosophy
- Regime filtering is critical (capitulations only in bear/crisis)
- Temporal confluence adds quality without much complexity
- Core threshold (confluence_threshold: 0.65) is well-tuned

### S4 Design Philosophy
- Regime discrimination is essential (bear specialist)
- Core funding + resilience logic is strong baseline
- Macro routing provides significant false positive reduction

### S5 Design Philosophy
- Wyckoff distribution detection adds structural confidence
- Regime routing critical (disabled in bull markets)
- Core funding + RSI is effective but needs pattern context

---

## Debugging: If Results Don't Match Expectations

### Check Config is Loading
```python
import json
with open('configs/variants/s1_core.json') as f:
    config = json.load(f)
    print(f"Variant: {config['profile']}")
    print(f"Feature flags: {config['feature_flags']}")
```

### Verify Feature Flags Active
Look for in backtest logs:
- `enable_wyckoff: true` → Wyckoff engine running
- `enable_macro: false` → Macro context disabled
- `use_regime_filter: false` → No regime gating

### Compare Against Production Baseline
If core performs unexpectedly well:
- May indicate production config is over-engineered
- Suggests opportunity to simplify operations
- Consider deploying simpler variant with monitoring

If core performs poorly:
- Validates complexity as necessary
- Each filter layer prevents real false positives
- Production complexity justified

---

## Next Steps After Testing

1. **Document Results**
   - Create comparative performance report
   - Identify complexity/performance tradeoff curves

2. **Recommend Config for Production**
   - If full config justified: Keep as-is
   - If core/core+ sufficient: Simplify operations
   - If variant outperforms: Update production baseline

3. **Update Operator Guide**
   - Document which variant running
   - Explain expected trade frequency/regime behavior
   - Alert to complexity changes

4. **Set Up Monitoring**
   - Track actual vs expected trade frequency
   - Alert if regime behavior deviates
   - Monitor signal quality metrics

---

## Success Criteria

Variant testing is successful when:
- ✓ All 9 configs run without errors
- ✓ Trade frequencies scale with expected complexity
- ✓ Regime behavior aligns with design (core fires all regimes, full is discriminant)
- ✓ Complexity/performance tradeoff clear
- ✓ Recommendation emerges for production optimization

Variant testing fails if:
- ✗ Results are inconsistent with config design
- ✗ Complexity doesn't correlate with performance
- ✗ Regime behavior doesn't match expectations
- ✗ No clear tradeoff pattern emerges
