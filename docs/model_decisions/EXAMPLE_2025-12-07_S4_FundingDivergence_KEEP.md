# MODEL ACCEPTANCE CHECKLIST

**Model Name:** S4_FundingDivergence
**Test Date:** 2025-12-07
**Analyst:** Lead Quant
**Experiment ID:** exp_btc_1h_2020_2025_funding_v1
**Config File:** configs/production/s4_funding_divergence_v1.json

---

## BASELINE CONTEXT

**Best Baseline on Test Set:**
- Baseline Name: BuyDip15pct
- Test PF: 2.12
- OOS PF: 1.95
- Max DD: 18%

**Required Threshold (Baseline + 0.1):** 2.22

---

## RULE 1: BEAT BASELINES

**Requirement:** Test PF > max(baseline_Test_PF) + 0.1

**Data:**
- Best Baseline Test PF: 2.12
- This Model Test PF: 2.35
- Delta: 0.23
- Required Delta: > 0.1

**Status:** [X] PASS [ ] FAIL

**Notes:**
Beats baseline by 0.23 PF on test set, well above the required 0.1 buffer.
This justifies the added complexity of funding rate + OI divergence logic
over simple "buy dips" baseline.

---

## RULE 2: GENERALIZATION (LOW OVERFIT)

**Requirement:** Overfit Score < 0.5

**Data:**
- Train PF: 2.45
- Test PF: 2.35
- Overfit Score (Train - Test): 0.10
- Required: < 0.5

**Interpretation:**
- [ ] Score < 0 (EXCELLENT - negative overfit)
- [X] Score 0-0.3 (GOOD - minimal overfit)
- [ ] Score 0.3-0.5 (ACCEPTABLE - moderate overfit)
- [ ] Score > 0.5 (FAIL - high overfit)
- [ ] Score > 1.5 (KILL IMMEDIATELY - extreme overfit)

**Status:** [X] PASS [ ] FAIL

**Notes:**
Overfit score of 0.10 indicates excellent generalization. The model
performs nearly identically on training and test data, suggesting the
logic captures true edge rather than training noise. This is a strong
indicator of robust parameter selection.

---

## RULE 3: STATISTICAL SIGNIFICANCE

**Requirement:** Total Trades >= 50 OR Low-Frequency Tag

**Data:**
- Train Trades: 95
- Test Trades: 78
- OOS Trades: 68
- Total Trades: 241
- Required: >= 50

**Low-Frequency Exception (if applicable):**
- [ ] Tagged as "low-frequency/macro"
- [ ] Multi-asset validation completed (BTC + ETH + SOL)
- [ ] OR walk-forward validation across 5+ regimes
- [ ] Documented why low frequency is inherent

**Status:** [X] PASS [ ] FAIL [ ] PASS (with exception)

**Notes:**
241 total trades far exceeds the 50-trade threshold. Sample size is
more than sufficient to trust performance metrics. Each period (train/test/OOS)
has 60+ trades individually, providing confidence in stability.

---

## RULE 4: OOS VALIDATION

**Requirement:** OOS_PF > 1.2 AND (OOS_PF / Test_PF) > 0.6

**Data:**
- OOS PF: 2.20
- Required OOS PF: > 1.2
- Test PF: 2.35
- OOS/Test Ratio: 0.94
- Required Ratio: > 0.6

**Interpretation:**
- [X] OOS/Test > 0.9 (EXCELLENT consistency)
- [ ] OOS/Test 0.7-0.9 (GOOD - expected degradation)
- [ ] OOS/Test 0.6-0.7 (ACCEPTABLE - investigate)
- [ ] OOS/Test < 0.6 (FAIL - overfit or regime shift)

**Status:** [X] PASS [ ] FAIL

**Notes:**
OOS performance (PF 2.20) is excellent - well above the 1.2 profitability
threshold. The 0.94 ratio (OOS/Test) indicates exceptional consistency,
with only 6% degradation from test to OOS. This suggests the strategy
will likely perform well on future unseen data.

---

## RULE 5: RISK-ADJUSTED PERFORMANCE

**Requirement:** Max_DD <= 2x Baseline_Max_DD OR PF > 3.0

**Data:**
- This Model Max DD: 15%
- Best Baseline Max DD: 18%
- 2x Baseline Max DD: 36%
- This Model PF: 2.35

**Check:**
- [X] Max DD <= 2x Baseline (passes risk test)
- [ ] OR PF > 3.0 (high alpha compensates for risk)

**Status:** [X] PASS [ ] FAIL

**Notes:**
Maximum drawdown of 15% is actually LOWER than the baseline's 18% DD.
This is remarkable - the model not only generates higher returns (PF 2.35 vs 2.12)
but also with LOWER risk. This is a clear win on both risk and return dimensions.

---

## RULE 6: COSTS INCLUDED

**Requirement:** All backtests MUST include realistic slippage and fees

**Data:**
- Slippage (bps per trade): 5
- Fees (bps per trade): 3
- Total Round-Trip Cost (bps): 16
- Standard Assumption: 16 bps (5 bps slippage + 3 bps fees x2)

**Verification:**
- [X] Costs applied in backtest configuration
- [X] Slippage >= 5 bps per trade
- [X] Fees >= 3 bps per trade
- [X] Results reflect post-cost performance

**Status:** [X] PASS [ ] FAIL

**Notes:**
All costs properly included in backtest. Used 5 bps slippage and 3 bps fees
per trade (16 bps round-trip). Performance metrics reflect realistic
post-cost returns. Verified in config file that cost parameters are active.

---

## SUMMARY SCORECARD

**Total Rules Passed:** 6 / 6

**Individual Results:**
- [X] Rule 1: Beat Baselines
- [X] Rule 2: Low Overfit
- [X] Rule 3: Statistical Significance
- [X] Rule 4: OOS Validation
- [X] Rule 5: Risk-Adjusted Performance
- [X] Rule 6: Costs Included

---

## DECISION

**Select One:**

### [X] ✅ KEEP (6/6 rules passed)
**Decision:** Deploy to paper trading → production
**Timeline:** Within 1 week
**Next Steps:**
1. Set up paper trading account with OKX testnet
2. Configure monitoring dashboard (track live PF, DD, trade frequency)
3. Run paper trading for 2-4 weeks (minimum 15 trades)
4. Compare paper results to OOS backtest (expect PF within ±20%)
5. If paper succeeds, ramp to production: Week1=25%, Week2=50%, Week3=75%, Week4=100%

---

### [ ] 🔧 IMPROVE (4-5 rules passed)
**Decision:** Specific improvements required, re-test in 1-2 weeks

**Failed Rules:**
1. N/A
2. N/A

**Root Causes:**
N/A

**Remediation Plan:**
N/A

**Re-test Deadline:** N/A

---

### [ ] ❌ KILL (< 4 rules passed)
**Decision:** Archive and document learnings

**Failed Rules:**
1. N/A
2. N/A
3. N/A

**Kill Reason:**
[ ] Worse than best baseline on ALL metrics
[ ] Extreme overfit (>1.5)
[ ] Negative OOS PF (losing money)
[ ] Logically flawed (implementation error)
[ ] Data leakage detected
[ ] Other: N/A

**Post-Mortem:**
N/A

**Lessons Learned:**
N/A

**Salvageable Components:**
N/A

---

## ADDITIONAL METRICS (OPTIONAL)

**Performance Metrics:**
- Win Rate (Train/Test/OOS): 52% / 51% / 49%
- Sharpe Ratio (Train/Test/OOS): 1.65 / 1.58 / 1.52
- Max DD (Train/Test/OOS): 12% / 15% / 14%
- Avg R per Trade: 0.58R
- Largest Win: 2.8R
- Largest Loss: -1.2R

**Trade Quality:**
- Average Trade Duration: 18.5 hours
- Win/Loss Ratio: 1.45
- Profit per Trade: $142 (on $10k position)

**Risk Metrics:**
- Calmar Ratio: 2.15 (CAGR / Max DD)
- Sortino Ratio: 2.10
- Value at Risk (95%): -0.8R

---

## REVIEWER SIGN-OFF

**Primary Analyst:**
- Name: Lead Quant
- Date: 2025-12-07
- Signature: [Signed]

**Secondary Reviewer:**
- Name: Senior Researcher
- Date: 2025-12-07
- Signature: [Signed]

**Final Approval (if KEEP):**
- Lead Quant: [Approved]
- Date: 2025-12-07

---

## ARCHIVE LOCATION

**Documentation Saved To:**
- Checklist: `/docs/model_decisions/2025-12-07_S4_FundingDivergence_KEEP.md`
- Results: `/results/validation_reports/s4_funding_divergence_validation.json`
- Config: `/configs/production/s4_funding_divergence_v1.json`

**Version Control:**
- Git Commit: abc123def456
- Branch: feature/s4-funding-divergence

---

## SPECIAL NOTES

**Regime-Specific Strategy:**
- [ ] Yes, regime-specific deployment
- Target Regime: N/A
- Regime Classifier: N/A
- Performance in Target Regime: N/A
- Performance in Other Regimes: N/A

**Low-Frequency Strategy:**
- [ ] Yes, low-frequency exception applied
- Expected Trade Frequency: N/A
- Multi-Asset Validation Results: N/A

**Ensemble Strategy:**
- [ ] Yes, portfolio of sub-strategies
- Components: N/A
- Component Correlation: N/A
- Portfolio Improvement Over Best Component: N/A

**Other Considerations:**
This strategy uses funding rate divergence from perpetual futures markets
combined with open interest changes. The logic is:
- Entry: When funding rate diverges from price momentum AND OI shows
  institutional accumulation/distribution
- Exit: Mean reversion to fair value OR stop-loss at -1.5R

Key strength: Works across both bull and bear regimes because it's based
on derivatives market microstructure rather than directional bias.

Monitoring notes:
- Watch funding rate data quality (check for API outages)
- OI data from OKX - ensure no gaps
- Strategy performs best when funding > 0.1% (high conviction signals)

---

**END OF CHECKLIST**

**Remember:** No exceptions. No shortcuts. No self-deception.

---

## PAPER TRADING PLAN

**Account Setup:**
- Platform: OKX Testnet
- Initial Capital: $10,000 (virtual)
- Position Size: $1,000 per trade (10% of capital)
- Leverage: 2x (conservative)

**Monitoring Schedule:**
- Daily: Check for execution issues, data gaps
- Weekly: Calculate PF, WR, DD vs OOS expectations
- Bi-weekly: Full performance review with team

**Success Criteria:**
- Paper PF within ±20% of OOS PF (1.76 - 2.64 range)
- No execution bugs (all orders fill correctly)
- No data quality issues (no missing bars)
- Minimum 15 trades completed

**Kill Switches:**
- Paper PF < 50% of OOS PF (< 1.10) → PAUSE & investigate
- Max DD > 150% of backtest DD (> 22.5%) → PAUSE
- Data feed issues lasting > 24h → PAUSE

**Timeline:**
- Week 1-2: Paper trading execution
- Week 3: Performance analysis
- Week 4: Decision (proceed to production or rework)

**Expected Outcome:**
Given the strong OOS validation, expect paper trading to confirm
backtest results. If paper succeeds, this becomes a core production
strategy with gradual capital ramp-up.
