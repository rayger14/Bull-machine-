# MODEL ACCEPTANCE CHECKLIST

**Model Name:** _______________
**Test Date:** _______________
**Analyst:** _______________
**Experiment ID:** _______________
**Config File:** _______________

---

## BASELINE CONTEXT

**Best Baseline on Test Set:**
- Baseline Name: _______________
- Test PF: _______________
- OOS PF: _______________
- Max DD: _______________

**Required Threshold (Baseline + 0.1):** _______________

---

## RULE 1: BEAT BASELINES

**Requirement:** Test PF > max(baseline_Test_PF) + 0.1

**Data:**
- Best Baseline Test PF: _______________
- This Model Test PF: _______________
- Delta: _______________
- Required Delta: > 0.1

**Status:** [ ] PASS [ ] FAIL

**Notes:**
_______________________________________________________________
_______________________________________________________________

---

## RULE 2: GENERALIZATION (LOW OVERFIT)

**Requirement:** Overfit Score < 0.5

**Data:**
- Train PF: _______________
- Test PF: _______________
- Overfit Score (Train - Test): _______________
- Required: < 0.5

**Interpretation:**
- [ ] Score < 0 (EXCELLENT - negative overfit)
- [ ] Score 0-0.3 (GOOD - minimal overfit)
- [ ] Score 0.3-0.5 (ACCEPTABLE - moderate overfit)
- [ ] Score > 0.5 (FAIL - high overfit)
- [ ] Score > 1.5 (KILL IMMEDIATELY - extreme overfit)

**Status:** [ ] PASS [ ] FAIL

**Notes:**
_______________________________________________________________
_______________________________________________________________

---

## RULE 3: STATISTICAL SIGNIFICANCE

**Requirement:** Total Trades >= 50 OR Low-Frequency Tag

**Data:**
- Train Trades: _______________
- Test Trades: _______________
- OOS Trades: _______________
- Total Trades: _______________
- Required: >= 50

**Low-Frequency Exception (if applicable):**
- [ ] Tagged as "low-frequency/macro"
- [ ] Multi-asset validation completed (BTC + ETH + SOL)
- [ ] OR walk-forward validation across 5+ regimes
- [ ] Documented why low frequency is inherent

**Status:** [ ] PASS [ ] FAIL [ ] PASS (with exception)

**Notes:**
_______________________________________________________________
_______________________________________________________________

---

## RULE 4: OOS VALIDATION

**Requirement:** OOS_PF > 1.2 AND (OOS_PF / Test_PF) > 0.6

**Data:**
- OOS PF: _______________
- Required OOS PF: > 1.2
- Test PF: _______________
- OOS/Test Ratio: _______________
- Required Ratio: > 0.6

**Interpretation:**
- [ ] OOS/Test > 0.9 (EXCELLENT consistency)
- [ ] OOS/Test 0.7-0.9 (GOOD - expected degradation)
- [ ] OOS/Test 0.6-0.7 (ACCEPTABLE - investigate)
- [ ] OOS/Test < 0.6 (FAIL - overfit or regime shift)

**Status:** [ ] PASS [ ] FAIL

**Notes:**
_______________________________________________________________
_______________________________________________________________

---

## RULE 5: RISK-ADJUSTED PERFORMANCE

**Requirement:** Max_DD <= 2x Baseline_Max_DD OR PF > 3.0

**Data:**
- This Model Max DD: _______________
- Best Baseline Max DD: _______________
- 2x Baseline Max DD: _______________
- This Model PF: _______________

**Check:**
- [ ] Max DD <= 2x Baseline (passes risk test)
- [ ] OR PF > 3.0 (high alpha compensates for risk)

**Status:** [ ] PASS [ ] FAIL

**Notes:**
_______________________________________________________________
_______________________________________________________________

---

## RULE 6: COSTS INCLUDED

**Requirement:** All backtests MUST include realistic slippage and fees

**Data:**
- Slippage (bps per trade): _______________
- Fees (bps per trade): _______________
- Total Round-Trip Cost (bps): _______________
- Standard Assumption: 16 bps (5 bps slippage + 3 bps fees x2)

**Verification:**
- [ ] Costs applied in backtest configuration
- [ ] Slippage >= 5 bps per trade
- [ ] Fees >= 3 bps per trade
- [ ] Results reflect post-cost performance

**Status:** [ ] PASS [ ] FAIL

**Notes:**
_______________________________________________________________
_______________________________________________________________

---

## SUMMARY SCORECARD

**Total Rules Passed:** ___ / 6

**Individual Results:**
- [ ] Rule 1: Beat Baselines
- [ ] Rule 2: Low Overfit
- [ ] Rule 3: Statistical Significance
- [ ] Rule 4: OOS Validation
- [ ] Rule 5: Risk-Adjusted Performance
- [ ] Rule 6: Costs Included

---

## DECISION

**Select One:**

### [ ] ✅ KEEP (6/6 rules passed)
**Decision:** Deploy to paper trading → production
**Timeline:** Within 1 week
**Next Steps:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________
5. _______________________________________________

---

### [ ] 🔧 IMPROVE (4-5 rules passed)
**Decision:** Specific improvements required, re-test in 1-2 weeks

**Failed Rules:**
1. _______________________________________________
2. _______________________________________________

**Root Causes:**
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

**Remediation Plan:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
4. _______________________________________________

**Re-test Deadline:** _______________

---

### [ ] ❌ KILL (< 4 rules passed)
**Decision:** Archive and document learnings

**Failed Rules:**
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

**Kill Reason:**
[ ] Worse than best baseline on ALL metrics
[ ] Extreme overfit (>1.5)
[ ] Negative OOS PF (losing money)
[ ] Logically flawed (implementation error)
[ ] Data leakage detected
[ ] Other: _______________________________________________

**Post-Mortem:**
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

**Lessons Learned:**
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

**Salvageable Components:**
_______________________________________________________________
_______________________________________________________________

---

## ADDITIONAL METRICS (OPTIONAL)

**Performance Metrics:**
- Win Rate (Train/Test/OOS): ___ / ___ / ___
- Sharpe Ratio (Train/Test/OOS): ___ / ___ / ___
- Max DD (Train/Test/OOS): ___ / ___ / ___
- Avg R per Trade: _______________
- Largest Win: _______________
- Largest Loss: _______________

**Trade Quality:**
- Average Trade Duration: _______________
- Win/Loss Ratio: _______________
- Profit per Trade: _______________

**Risk Metrics:**
- Calmar Ratio: _______________
- Sortino Ratio: _______________
- Value at Risk (95%): _______________

---

## REVIEWER SIGN-OFF

**Primary Analyst:**
- Name: _______________
- Date: _______________
- Signature: _______________

**Secondary Reviewer:**
- Name: _______________
- Date: _______________
- Signature: _______________

**Final Approval (if KEEP):**
- Lead Quant: _______________
- Date: _______________

---

## ARCHIVE LOCATION

**Documentation Saved To:**
- Checklist: `/docs/model_decisions/YYYY-MM-DD_ModelName_[KEEP|IMPROVE|KILL].md`
- Results: `/results/validation_reports/ModelName_validation.json`
- Config: `/configs/production/ModelName_v1.json` (if KEEP)

**Version Control:**
- Git Commit: _______________
- Branch: _______________

---

## SPECIAL NOTES

**Regime-Specific Strategy:**
- [ ] Yes, regime-specific deployment
- Target Regime: _______________
- Regime Classifier: _______________
- Performance in Target Regime: _______________
- Performance in Other Regimes: _______________

**Low-Frequency Strategy:**
- [ ] Yes, low-frequency exception applied
- Expected Trade Frequency: _______________
- Multi-Asset Validation Results: _______________

**Ensemble Strategy:**
- [ ] Yes, portfolio of sub-strategies
- Components: _______________
- Component Correlation: _______________
- Portfolio Improvement Over Best Component: _______________

**Other Considerations:**
_______________________________________________________________
_______________________________________________________________
_______________________________________________________________

---

**END OF CHECKLIST**

**Remember:** No exceptions. No shortcuts. No self-deception.
