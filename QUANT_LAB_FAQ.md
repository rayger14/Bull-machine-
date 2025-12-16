# QUANT LAB FAQ

**Common questions and answers for running the Quant Lab.**

---

## PHILOSOPHY & APPROACH

### Q: What if a baseline performs better than all my archetypes?

**A:** Deploy the baseline. This is not failure—this is the lab working as designed.

The baselines exist to establish the minimum bar for deployment. If SMA200 achieves 2.5 PF and your complex archetype achieves 2.3 PF, the market is telling you:
- Your complexity doesn't add value
- You may be overfitting
- The simple trend hypothesis is sufficient

**Action:**
1. Deploy the winning baseline (paper → live)
2. Investigate why archetypes underperform (overfit? wrong features? regime mismatch?)
3. Consider if archetypes add value in specific regimes (run regime slicing experiment)

**Remember:** The goal is profit, not complexity. A simple model that works beats a complex model that doesn't.

---

### Q: What if ALL models fail acceptance criteria?

**A:** You have several options, in order of preference:

1. **Lower the criteria temporarily** (min_test_pf = 1.2 instead of 1.5) and see what's closest
2. **Extend the test period** to get more statistical significance
3. **Check for market regime change** (2023-2024 may be fundamentally different from 2020-2022)
4. **Increase acceptable overfit threshold** if market is evolving rapidly
5. **Consider that the market may not be tradeable** with your current approach

**Before lowering criteria, ask:**
- Would I trade real money with these results?
- Is the market fundamentally different from training data?
- Are transaction costs too high for my strategy frequency?

**Critical insight:** If nothing works, that's valuable information. Don't force deployment.

---

### Q: How do I handle low-trade-count archetypes?

**A:** Low frequency strategies (< 50 trades/year) require special treatment.

**Tag them as "macro/rare" strategies:**
- Acceptance criteria: min_trades = 10 instead of 50
- Require multi-asset validation (must work on BTC + ETH + SOL)
- Extend OOS period (need 2-3 years to validate, not just 1 year)
- Higher bar for profit factor (min 3.0 instead of 1.5)
- Detailed manual review of each trade

**Why different criteria:**
- Small sample size = high variance in results
- Could be lucky on a few trades
- Harder to detect overfit with limited data
- Need broader validation to trust signal

**Example:**
```
S4 has 8 trades on test period with PF 4.5
→ Don't auto-deploy
→ Run on ETH, SOL, AVAX
→ If it gets 30+ trades total across assets with PF > 3.0 → Consider deployment
→ If it only works on BTC → Might be lucky, need more OOS data
```

---

### Q: What if the OOS period is too short for statistical significance?

**A:** You have several options:

**Option 1: Use multiple OOS periods**
- Test: 2023 H1
- OOS1: 2023 H2
- OOS2: 2024 H1
- OOS3: 2024 H2

Require model to pass on ALL OOS periods.

**Option 2: Walk-forward validation**
- Split data into 6-month windows
- Optimize on each window, test on next
- Require consistent performance across all folds

**Option 3: Accept limitation and monitor closely**
- Deploy to paper trading immediately
- Require 30 days paper trading before live
- Monitor for degradation

**Option 4: Multi-asset validation**
- If model works on BTC, ETH, and SOL → more confidence
- Generalizing across assets reduces risk of lucky OOS period

**Recommendation:** Use Option 1 (multiple OOS periods) for rare strategies, Option 3 (paper trading) for frequent strategies.

---

### Q: Can temporal layers (Fib clusters, etc.) be tested separately?

**A:** Yes, this is the temporal ablation experiment.

**Method:**
1. Run S4 with funding divergence only (disable temporal layers)
2. Run S4 with funding + temporal layers enabled
3. Compare results

**Analysis:**
```
S4_FundingOnly:     Test PF = 2.8, Trades = 45
S4_Funding+Temporal: Test PF = 3.1, Trades = 38

Temporal lift = 3.1 - 2.8 = +0.3 PF
Trade reduction = 45 - 38 = -7 trades

Conclusion: Temporal layer adds 0.3 PF but reduces frequency by 15%.
Decision: Keep temporal layer (quality > quantity)
```

**Acceptance criteria for layers:**
- Must add >= +0.2 PF to justify complexity
- Trade reduction acceptable if PF lift is significant
- Must not cause OOS collapse (check OOS performance too)

**Priority experiment:** This should be in your first batch after baseline suite is working.

---

## TECHNICAL QUESTIONS

### Q: How do I handle missing data in OHLCV?

**A:** Several strategies:

**Prevention (best):**
- Use high-quality data source
- Fill gaps during data collection
- Validate data quality before backtesting

**If gaps exist:**
- Forward-fill OHLCV for small gaps (< 3 bars)
- Exclude periods with large gaps (> 24 hours)
- Mark gap periods and analyze if trades cluster around them
- Consider if gaps represent real trading halts (handle appropriately)

**Red flag:** If your model only works during gap periods, it's likely exploiting data quality issues, not real market edge.

---

### Q: Should I include funding rate fees in transaction costs?

**A:** Yes, if you're trading perpetuals.

**Calculation:**
```python
# Typical funding rate: 0.01% every 8 hours = 0.03% per day
# For multi-day holds, this adds up

holding_days = (exit_time - entry_time).days
funding_cost_pct = holding_days * 0.03  # 0.03% per day

total_cost = slippage + trading_fees + funding_cost
```

**Impact:**
- Long-hold strategies (> 7 days) significantly impacted
- Short-duration strategies (< 24 hours) minimally impacted
- Can be positive or negative (you receive funding when counterparty pays you)

**Conservative approach:** Assume you always pay funding (worst case).

---

### Q: How do I handle overnight/weekend gaps in hourly data?

**A:** Depends on your strategy:

**For trend-following strategies:**
- Gaps are real market behavior
- Don't fill them artificially
- Model should handle them naturally

**For mean-reversion strategies:**
- Gaps can create false signals
- Consider excluding first bar after gap
- Or require gap < X% to enter trade

**For all strategies:**
- Analyze trade performance around gaps separately
- If model only works during gaps → suspicious (liquidity issue?)
- If model fails around gaps → add gap filter

---

### Q: What Sharpe ratio should I expect from baselines?

**A:** Rough benchmarks for crypto (BTC 1h):

```
B0 (Buy & Hold):        0.5 - 1.0  (market baseline)
B1 (SMA200):           0.8 - 1.5  (basic trend)
B2 (SMA Crossover):    0.6 - 1.2  (moderate trend)
B3 (RSI Mean Rev):     0.4 - 0.9  (lower, choppier)
B4 (Vol Target):       1.0 - 1.8  (higher, risk-managed)
```

**Archetypes should target:**
- Sharpe > 1.5 (meaningfully better than baselines)
- Sharpe > 2.0 (excellent, deployable with confidence)
- Sharpe > 3.0 (exceptional, validate carefully for overfit)

**Sharpe < 0.5:** Probably not tradeable (noise)
**Sharpe > 4.0:** Likely overfitted or lucky

---

## DECISION-MAKING QUESTIONS

### Q: What if a model passes on test but fails on OOS?

**A:** This is OOS collapse—a critical red flag.

**Common causes:**
1. **Regime change:** Market fundamentally different in OOS period
2. **Overfit:** Model memorized train data, doesn't generalize
3. **Lucky test period:** Test period happened to favor model
4. **Data leakage:** Future information leaked into features

**Diagnosis:**
```python
# Check feature distributions
train_features.describe()
test_features.describe()
oos_features.describe()

# Look for distributional shifts
# If mean/std drastically different → regime change
```

**Decision:**
- Tag as IMPROVE
- Investigate specific cause
- Test on alternative OOS period (2024 H1 vs H2)
- If still fails → KILL

**Do not deploy** models with OOS collapse, even if test performance is excellent.

---

### Q: What if a model has high overfit but good OOS performance?

**A:** Investigate carefully—this is unusual.

**Scenario:**
```
Train PF: 5.0
Test PF:  2.5  (overfit score = 0.5)
OOS PF:   2.6  (good!)
```

**Possible explanations:**
1. **Train period was exceptionally good** (2020-2022 bull run?)
2. **Model is robust** despite high train performance
3. **OOS is lucky** (need more validation)

**Action:**
- Check if train period had unusual regime
- Test on multiple OOS periods
- If OOS is consistently > 2.0 → Consider KEEP
- If OOS varies wildly → Tag as IMPROVE

**General rule:** Trust OOS more than overfit score, but validate carefully.

---

### Q: What if two models have similar performance? How do I choose?

**A:** Tiebreaker criteria (in order):

1. **OOS performance** (most important—future prediction)
2. **Lower overfit score** (more robust)
3. **Higher trade count** (more statistical significance)
4. **Lower complexity** (simpler is better, all else equal)
5. **Lower transaction cost sensitivity** (more realistic)
6. **Better Sharpe ratio** (risk-adjusted returns)

**Example:**
```
Model A: Test PF 2.5, OOS PF 2.4, Overfit 0.3, Trades 60, Sharpe 1.8
Model B: Test PF 2.6, OOS PF 2.2, Overfit 0.5, Trades 45, Sharpe 1.6

Winner: Model A
Reason: Better OOS, lower overfit, more trades, better Sharpe (even though Test PF is slightly lower)
```

**When in doubt:** Deploy both to paper trading and let live market decide.

---

### Q: Should I optimize parameters before running the baseline suite?

**A:** No for baselines. Yes for archetypes (but carefully).

**Baselines:**
- Use standard parameters (SMA200, not SMA187.3)
- Purpose is to establish known benchmarks
- Optimization would make them less interpretable

**Archetypes:**
- If parameters are already tuned → use them
- If parameters are arbitrary → run quick optimization on train period
- If parameters are unknown → start with sensible defaults, optimize later

**Critical rule:** Never optimize on test or OOS data. Ever.

---

## PROCESS QUESTIONS

### Q: How often should I re-run the baseline suite?

**A:** Depends on purpose:

**During development:**
- Weekly (Friday): Check if deployed models still work
- Monthly: Full refresh with latest data

**In production:**
- Daily: Monitor live vs backtest performance
- Weekly: Re-run baselines on rolling window
- Quarterly: Full re-validation with updated train/test/OOS splits

**Triggers for immediate re-run:**
- Live performance deviates > 20% from backtest
- Major market event (crash, regime change)
- Adding new model to comparison
- Data quality issue discovered

---

### Q: Can I test on intraday data (15m, 5m)?

**A:** Yes, but adjust expectations and costs.

**Considerations:**
- **Higher frequency = higher transaction costs** (more trades)
- **More sensitive to slippage** (lower liquidity at smaller timeframes)
- **Requires more data** (need longer history for same number of samples)
- **Harder to validate OOS** (market microstructure changes faster)

**Recommendations:**
- Increase slippage_bps to 10-15 for 15m
- Increase slippage_bps to 20-30 for 5m
- Require higher PF (min 2.0 instead of 1.5)
- Test fill rate in paper trading carefully

**Start with 1h or 4h** until framework is proven, then move to intraday if needed.

---

### Q: What if I want to test on multiple assets simultaneously?

**A:** Create a multi-asset experiment config.

**Approach 1: Separate backtests, combined results**
```json
{
  "assets": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  "run_separate": true,
  "combine_results": true
}
```

**Approach 2: Portfolio-level backtest**
```json
{
  "assets": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
  "position_sizing": "equal_weight",
  "rebalance_frequency": "daily"
}
```

**Analysis:**
- Models must work on >= 2 out of 3 assets
- Average performance across assets
- Check correlation of returns across assets

**Benefit:** More robust validation, less risk of lucky single-asset results.

---

### Q: How do I know if my transaction costs are realistic?

**A:** Validate against live trading data.

**Method:**
1. Run paper trading for 2 weeks
2. Measure actual slippage per trade
3. Measure actual fees per trade
4. Compare to backtest assumptions

**Typical findings:**
- **Backtest assumes:** 5 bps slippage + 6 bps fees = 11 bps total
- **Reality (good execution):** 8-15 bps total
- **Reality (poor execution):** 20-50 bps total

**Recommendation:**
- Start conservative (15 bps total)
- Adjust after paper trading validation
- If live costs > backtest assumptions → reduce position sizes or trade less frequently

**Red flag:** If model only works with 0 bps costs → not tradeable.

---

## ADVANCED TOPICS

### Q: How do I implement walk-forward optimization?

**A:** This is a medium-priority experiment after baseline suite works.

**Process:**
1. Split data into windows (e.g., 6 months each)
2. For each window:
   - Optimize parameters on current window (in-sample)
   - Test on next window (out-of-sample)
3. Combine all OOS results

**Code outline:**
```python
windows = [
    ("2020-01", "2020-06", "2020-07", "2020-12"),  # train, test
    ("2020-07", "2020-12", "2021-01", "2021-06"),
    ("2021-01", "2021-06", "2021-07", "2021-12"),
    # ...
]

for train_start, train_end, test_start, test_end in windows:
    # Optimize on train
    best_params = optimize(train_start, train_end)
    # Test on test
    result = backtest(best_params, test_start, test_end)
    results.append(result)
```

**Compare:**
- Walk-forward vs static parameters
- If walk-forward is better → parameters are time-varying, need dynamic optimization
- If static is similar → parameters are stable, simpler deployment

---

### Q: How do I test regime-specific models?

**A:** Create regime-sliced experiments.

**Step 1: Define regimes**
```python
regimes = {
    "bull": df[df['sma200_slope'] > 0.01],
    "bear": df[df['sma200_slope'] < -0.01],
    "sideways": df[abs(df['sma200_slope']) <= 0.01]
}
```

**Step 2: Run baselines on each regime**
```bash
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --regime bull
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --regime bear
python bin/run_quant_suite.py --config configs/experiment_btc_1h_2020_2025.json --regime sideways
```

**Step 3: Analyze regime-specific winners**
- Maybe SMA200 wins in bull regime
- Maybe RSI wins in sideways regime
- Maybe a bear archetype wins in bear regime

**Step 4: Consider regime-routed ensemble**
```python
if current_regime == 'bull':
    use_model = 'B1_SMA200'
elif current_regime == 'bear':
    use_model = 'S2_BearArchetype'
else:
    use_model = 'B3_RSI'
```

**Caution:** Adds complexity. Only do if regime detection is robust and regime routing adds significant value.

---

### Q: What's the best way to combine multiple models into an ensemble?

**A:** Start simple, add complexity only if justified.

**Approach 1: Equal weight**
```python
signal = (model_a_signal + model_b_signal + model_c_signal) / 3
```

**Approach 2: Performance-weighted**
```python
weights = {
    'model_a': pf_a / sum_of_pfs,
    'model_b': pf_b / sum_of_pfs,
    'model_c': pf_c / sum_of_pfs
}
signal = sum(weights[m] * signals[m] for m in models)
```

**Approach 3: Sharpe-weighted**
```python
weights = {
    'model_a': sharpe_a / sum_of_sharpes,
    'model_b': sharpe_b / sum_of_sharpes,
    'model_c': sharpe_c / sum_of_sharpes
}
```

**Test all approaches:**
- Ensemble must beat best individual model
- Must be more robust (lower drawdown, higher Sharpe)
- Must not just be diluting best model with mediocre models

**Rule:** Only ensemble models that are uncorrelated. If all models make the same trades, ensemble adds no value.

---

### Q: How do I handle models that work on some assets but not others?

**A:** Asset-specific deployment is acceptable if justified.

**Scenario:**
```
S5_LongSqueeze on BTC: Test PF 2.8, OOS PF 2.5 ✓
S5_LongSqueeze on ETH: Test PF 1.2, OOS PF 0.9 ✗
S5_LongSqueeze on SOL: Test PF 1.5, OOS PF 1.3 ~
```

**Decision:**
- Deploy on BTC only
- Tag as "BTC-specific" (not generalizable)
- Monitor closely for regime change
- Investigate why it doesn't work on ETH/SOL (feature difference? liquidity?)

**Caution:** Asset-specific models are riskier (could be overfitted to BTC quirks). Require higher bar for deployment.

---

**Remember: When in doubt, ask for evidence. If you can't measure it, you can't improve it.**
