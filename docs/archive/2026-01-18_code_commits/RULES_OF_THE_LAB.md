# RULES OF THE LAB
## Acceptance Criteria Framework for Bull Machine Models

**Version:** 2.0
**Last Updated:** 2025-12-07
**Status:** ACTIVE - All models must comply

---

## PHILOSOPHY

**Core Principle:** Everything is a model. Baselines, archetypes, ensembles - all are evaluated by the same rigorous standards.

**Guiding Beliefs:**
- Evidence beats intuition
- Complexity must justify itself against simplicity
- Overfitting is the enemy, generalization is the goal
- Kill fast, learn faster
- Baselines set the bar; beating them is not optional

**Purpose:** Prevent self-deception, ensure production readiness, maintain quant fund standards.

---

## THE SIX RULES OF DEPLOYMENT ELIGIBILITY

A model is eligible for production deployment **ONLY** if it passes **ALL SIX** rules.

### Rule 1: Beat Baselines
**Requirement:** Test PF > max(baseline_Test_PF) + 0.1

**Definition:**
- Identify the best performing baseline on the test set
- Add buffer of 0.1 to that baseline's Test PF
- Model must exceed this threshold

**Rationale:**
- Complexity must be justified by measurably better performance
- 0.1 buffer ensures improvement is meaningful, not noise
- If a simple SMA crossover works as well, use that instead

**Example:**
```
Best Baseline (SMA): Test PF = 2.12
Required Threshold: 2.12 + 0.1 = 2.22
Your Model Test PF: 2.35 ✓ PASS
Your Model Test PF: 2.18 ✗ FAIL
```

---

### Rule 2: Generalization (Low Overfit)
**Requirement:** Overfit Score < 0.5

**Definition:**
```
Overfit Score = Train_PF - Test_PF
```

**Interpretation:**
- **Score < 0:** Model performs BETTER on test than train (EXCELLENT - negative overfit)
- **Score 0-0.3:** Minimal overfit (GOOD)
- **Score 0.3-0.5:** Moderate overfit (ACCEPTABLE if other rules pass)
- **Score > 0.5:** High overfit (FAIL - likely curve-fitted)
- **Score > 1.5:** Extreme overfit (KILL IMMEDIATELY)

**Rationale:**
- Models that perform vastly better on training data are curve-fitted
- Real edge generalizes to unseen data
- Small negative overfit suggests robust, conservative tuning

**Example:**
```
Train PF: 2.50, Test PF: 2.35
Overfit = 2.50 - 2.35 = 0.15 ✓ PASS (good generalization)

Train PF: 3.80, Test PF: 2.10
Overfit = 3.80 - 2.10 = 1.70 ✗ FAIL (curve-fitted to train)
```

---

### Rule 3: Statistical Significance
**Requirement:** Total Trades >= 50 OR Low-Frequency Tag

**Definition:**
```
Total Trades = Train_Trades + Test_Trades + OOS_Trades
```

**Rationale:**
- Need sufficient sample size to trust performance metrics
- With <50 trades, PF can be dominated by 1-2 lucky/unlucky trades
- Exception for macro/rare-event strategies with explicit validation

**Low-Frequency Exception:**
If Total Trades < 50, model MUST:
1. Be explicitly tagged as "low-frequency/macro"
2. Pass multi-asset validation (BTC + ETH + SOL)
3. OR pass walk-forward validation across 5+ regimes
4. Document why low frequency is inherent to strategy logic

**Example:**
```
Regular Strategy:
Total Trades: 75 ✓ PASS

Low-Frequency Strategy:
Total Trades: 35 ✗ Would fail
BUT: Validated on BTC, ETH, SOL (3 assets × 35 = 105 effective trades)
AND: Tagged as "Crisis Detector" (rare events only)
✓ PASS with exception
```

---

### Rule 4: OOS Validation
**Requirement:** OOS_PF > 1.2 AND (OOS_PF / Test_PF) > 0.6

**Definition:**
- Out-of-Sample (OOS) is truly unseen future data
- Must be profitable (PF > 1.2 = 20% return after costs)
- Performance degradation must be reasonable (not catastrophic)

**Rationale:**
- OOS is the ultimate test of generalization
- Many strategies work in-sample but fail in real future markets
- Some degradation is normal (markets change), but not collapse

**Interpretation:**
- **OOS/Test ratio > 0.9:** Excellent consistency
- **OOS/Test ratio 0.7-0.9:** Good (expected degradation)
- **OOS/Test ratio 0.6-0.7:** Acceptable (investigate why)
- **OOS/Test ratio < 0.6:** Fail (likely overfit or regime shift)

**Example:**
```
Test PF: 2.35, OOS PF: 2.20
OOS > 1.2 ✓
OOS/Test = 2.20/2.35 = 0.94 ✓ PASS (excellent consistency)

Test PF: 2.35, OOS PF: 1.10
OOS > 1.2 ✗
OOS/Test = 1.10/2.35 = 0.47 ✗ FAIL (catastrophic degradation)
```

---

### Rule 5: Risk-Adjusted Performance
**Requirement:** Max_DD <= 2x Baseline_Max_DD OR PF > 3.0

**Definition:**
- Maximum Drawdown (DD) is the largest peak-to-trough decline
- Compare to best baseline's drawdown on same period
- High PF can compensate for higher DD

**Rationale:**
- Can't have unlimited risk even if returns are high
- 2x baseline drawdown is maximum acceptable risk increase
- Exception: If PF > 3.0, strategy is generating enough alpha to justify higher risk

**Example:**
```
Best Baseline Max DD: 18%
Your Model Max DD: 15% ✓ PASS (lower risk)
Your Model Max DD: 25%, PF = 3.5 ✓ PASS (high PF compensates)
Your Model Max DD: 40%, PF = 2.2 ✗ FAIL (excessive risk)
```

---

### Rule 6: Costs Included
**Requirement:** All backtests MUST include realistic slippage and fees

**Standard Assumptions:**
- **Slippage:** 5 bps (0.05%) per trade
- **Fees:** 3 bps (0.03%) per trade
- **Total Round-Trip Cost:** 16 bps (entry + exit)

**Rationale:**
- Real trading has costs; "theoretical" backtests are worthless
- Costs disproportionately hurt high-frequency strategies
- Must prove profitability AFTER costs

**Example:**
```
Model without costs: PF = 2.80
Model with costs (8 bps per trade): PF = 2.35 ✓ PASS (still beats baseline)

Model without costs: PF = 1.90
Model with costs: PF = 1.15 ✗ FAIL (barely profitable after costs)
```

---

## THREE-TIER CLASSIFICATION SYSTEM

### ✅ KEEP (Deploy to Production)

**Criteria:** Passes ALL 6 eligibility rules

**Decision:** Graduate to paper trading → live deployment

**Timeline:** Immediate (within 1 week)

**Action Items:**
1. Document model configuration
2. Set up paper trading monitoring
3. Define live deployment triggers
4. Establish kill-switches and risk limits
5. Schedule weekly performance review

**Example:**
```
Model: S4_FundingDivergence
✓ Rule 1: Test PF 2.35 > 2.22 (baseline + buffer)
✓ Rule 2: Overfit 0.15 < 0.5
✓ Rule 3: 75 trades >= 50
✓ Rule 4: OOS PF 2.20, ratio 0.94
✓ Rule 5: Max DD 15% < 36% (2x baseline)
✓ Rule 6: Costs included (8 bps)
→ ✅ KEEP - Deploy to paper trading this week
```

---

### 🔧 IMPROVE (Needs Work)

**Criteria:** Passes 4-5 rules (close but not ready)

**Decision:** Specific improvements required, re-test in 1-2 weeks

**Action Items:**
1. Identify which rules failed and why
2. Develop specific remediation plan
3. Re-run validation after changes
4. Document what was fixed
5. Re-evaluate with fresh eyes

**Common Failure Modes & Fixes:**

**High Overfit (Rule 2 Fail):**
- Reduce number of parameters
- Add regularization
- Simplify logic
- Use more conservative parameter selection
- Increase training data period

**Low Trades (Rule 3 Fail):**
- Relax overly strict filters
- Reduce confluence requirements
- Expand regime applicability
- OR pursue multi-asset validation if low-freq is inherent

**OOS Collapse (Rule 4 Fail):**
- Investigate regime changes between test and OOS
- Check for data leakage in training
- Recalibrate for new market conditions
- Consider walk-forward re-optimization

**High Drawdown (Rule 5 Fail):**
- Add position sizing logic
- Implement stop-losses
- Reduce leverage
- Add correlation filters

**Example:**
```
Model: S1_LiquidityVacuum
✗ Rule 1: Test PF 1.80 < 2.22 (below baseline + buffer)
✗ Rule 2: Overfit 1.20 > 0.5 (HIGH)
✗ Rule 3: 45 trades < 50
✓ Rule 4: OOS PF 1.50, ratio 0.83
✓ Rule 5: Max DD 12% < 36%
✓ Rule 6: Costs included
→ 🔧 IMPROVE
Action Plan:
1. Reduce parameters to fix overfit
2. Relax entry filters to get 60+ trades
3. Re-optimize more conservatively
Re-test deadline: 2025-12-21
```

---

### ❌ KILL (Abandon or Archive)

**Criteria:** Passes < 4 rules OR fundamentally worse than baselines

**Decision:** Archive code, document learnings, move on

**Action Items:**
1. Document why model failed
2. Extract any useful sub-components
3. Archive code in `/archive/failed_models/`
4. Add to "lessons learned" log
5. Free up resources for better ideas

**When to Kill Immediately:**
- Worse than best baseline on ALL metrics
- Extreme overfit (>1.5)
- Negative OOS PF (losing money)
- Logically flawed (discovered error in implementation)
- Data leakage detected

**Example:**
```
Model: S5_LongSqueeze
✗ Rule 1: Test PF 1.30 < 2.22 (FAR below baseline)
✗ Rule 2: Overfit 0.80 > 0.5
✗ Rule 3: 25 trades < 50
✗ Rule 4: OOS PF 0.90 < 1.2 (LOSING MONEY)
✗ Rule 5: Max DD 25% > 36% (but PF too low to compensate)
✓ Rule 6: Costs included
→ ❌ KILL
Reason: Worse than simple SMA baseline in every way, negative OOS, abandon immediately
Lesson Learned: "Long squeeze" signal needs complete redesign or is not viable
```

---

## SPECIAL CASES AND EXCEPTIONS

### Low-Frequency Strategies

**Scenario:** Total trades < 50 but strategy is inherently macro/rare-event based

**Requirements:**
1. **Multi-Asset Validation:**
   - Must work on BTC, ETH, and SOL (minimum)
   - Each asset must show positive PF independently
   - Portfolio PF must exceed single-asset PF

2. **OR Walk-Forward Validation:**
   - Divide history into 5+ non-overlapping periods
   - Re-optimize on each period's training data
   - Show consistency across all test periods

3. **Explicit Tagging:**
   - Document in model metadata: `"frequency": "low-freq/macro"`
   - Explain why low frequency is inherent to strategy
   - Define expected trade frequency range

**Example:**
```
Model: MacroCrisisDetector
Total Trades: 18 (too low for Rule 3)
BUT:
- Works on BTC (6 trades, PF 2.1), ETH (7 trades, PF 1.9), SOL (5 trades, PF 2.3)
- Total across assets: 18 trades, effective sample ~54 (3 independent markets)
- Strategy logic: Detects >3 sigma vol spikes (inherently rare)
- Tagged as "low-freq/macro"
→ ✓ PASS Rule 3 with exception
```

---

### Regime-Specific Strategies

**Scenario:** Strategy only works in specific regime (e.g., bear market only)

**Requirements:**
1. **Regime-Specific Metrics:**
   - Measure performance within target regime separately
   - Must pass all 6 rules within that regime

2. **Neutral Regime Performance:**
   - Must not LOSE money in neutral/other regimes
   - Minimum requirement: PF > 0.9 (near break-even)

3. **Regime Detection:**
   - Must have reliable regime classifier
   - Classifier must be validated out-of-sample
   - Document regime definitions clearly

4. **Documentation:**
   - Explicitly state regime dependency
   - Define when strategy should be active/inactive
   - Set up monitoring for regime transitions

**Example:**
```
Model: BearMarketTrapWithinTrend
Full Period: PF 1.45 (fails Rule 1)
BUT:
- Bear Regime Only: PF 2.85 (beats baseline in bear)
- Passes all 6 rules WITHIN bear regime
- Neutral/Bull Performance: PF 1.05 (not losing money)
- GMM regime classifier validated (95% accuracy OOS)
→ ✓ PASS with regime-specific deployment
Deploy: Only activate when GMM detects bear regime
```

---

### Ensemble/Portfolio Strategies

**Scenario:** Combining multiple sub-strategies into portfolio

**Requirements:**
1. **Component Validation:**
   - Each sub-strategy must pass all 6 rules individually
   - OR pass 5/6 rules with complementary failures

2. **Portfolio Improvement:**
   - Portfolio PF > max(component PF)
   - Portfolio Sharpe > max(component Sharpe)
   - Portfolio Max DD < average(component Max DD)

3. **Diversification Benefit:**
   - Document correlation between components (<0.6)
   - Show that combination improves risk-adjusted returns
   - Prove it's not just "adding more stuff"

4. **Simplicity Test:**
   - Portfolio must beat best single component by >0.15 PF
   - If not, just use the best single component

**Example:**
```
Components:
- S2_SMCBreakOfStructure: PF 2.20 (passes all 6 rules)
- S3_OrderBlockRetest: PF 2.15 (passes all 6 rules)
- Correlation: 0.42 (good diversification)

Portfolio (equal weight):
- PF: 2.45 > 2.20 (beats best component by 0.25)
- Sharpe: 1.85 > 1.65 (best component Sharpe)
- Max DD: 13% < 16% (average of components)
→ ✓ PASS - Portfolio adds value beyond components
```

---

## DECISION WORKFLOW

```
┌─────────────────────────┐
│    New Model Created    │
│  (Baseline/Archetype)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Run Standard Validation │
│ - Train: 2020-2021      │
│ - Test:  2022           │
│ - OOS:   2023           │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Calculate All Metrics  │
│ - PF, Overfit, Trades   │
│ - OOS, DD, Costs        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Apply 6 Eligibility     │
│ Rules (Score 0-6)       │
└───────────┬─────────────┘
            │
            ▼
┌───────────┴─────────────┐
│  How many rules passed? │
└───┬─────────┬───────┬───┘
    │         │       │
6/6 │     4-5 │    <4 │
    │         │       │
    ▼         ▼       ▼
┌────────┐ ┌────┐ ┌──────┐
│  KEEP  │ │IMPR│ │ KILL │
└───┬────┘ └─┬──┘ └──┬───┘
    │        │       │
    ▼        ▼       ▼
┌────────┐ ┌────┐ ┌──────┐
│ Paper  │ │Fix │ │Archiv│
│Trading │ │&   │ │e &   │
│→ Live  │ │Ret │ │Learn │
└────────┘ └────┘ └──────┘
```

---

## DOCUMENTATION REQUIREMENTS

Every model decision must be documented with:

### Required Artifacts
1. **Completed Acceptance Checklist** (see MODEL_ACCEPTANCE_CHECKLIST.md)
2. **Performance Summary** (all metrics on train/test/OOS)
3. **Rule-by-Rule Evaluation** (explicit pass/fail for each rule)
4. **Decision Rationale** (why KEEP/IMPROVE/KILL)
5. **Action Items** (if IMPROVE) or **Lessons Learned** (if KILL)

### Storage Location
```
/docs/model_decisions/
  ├── YYYY-MM-DD_ModelName_KEEP.md
  ├── YYYY-MM-DD_ModelName_IMPROVE.md
  └── YYYY-MM-DD_ModelName_KILL.md
```

### Review Process
1. Model creator fills out checklist
2. Second team member reviews metrics independently
3. Both sign off on decision
4. Archive in version control
5. Update model registry

---

## CONTINUOUS IMPROVEMENT

### Quarterly Rule Review
Every 3 months, review these rules:
- Are thresholds still appropriate?
- Have we discovered new failure modes?
- Do we need additional rules?
- Should we tighten standards?

### Kill-Switch Monitoring
Even KEEP models must be monitored:
- Weekly PF check on live/paper trading
- If live PF < 1.0 for 2 consecutive weeks → pause and investigate
- If live PF < 0.8 → KILL immediately and post-mortem

### Learning Log
Maintain `/docs/lessons_learned.md`:
- What caused each KILL decision?
- What patterns predict failure?
- What works vs what doesn't?
- Update rules based on learnings

---

## EMERGENCY OVERRIDES

**Who Can Override:** Lead Quant + CTO (both required)

**When Allowed:**
- Extremely low-frequency strategy with compelling economic logic
- Market regime has fundamentally changed (requires thesis document)
- Critical bug discovered in validation framework
- New data source invalidates historical baselines

**Process:**
1. Document override reason in writing
2. Get independent 3rd party review
3. Set explicit re-evaluation date (max 30 days)
4. Tighten monitoring (daily instead of weekly)
5. Archive override decision

**Override Template:**
```
EMERGENCY OVERRIDE: [Model Name]
Date: [Date]
Failed Rules: [List]
Override Reason: [Detailed explanation]
Risk Assessment: [What could go wrong]
Monitoring Plan: [How we'll catch failure]
Re-evaluation Date: [Max 30 days]
Approved By: [Lead Quant], [CTO]
```

---

## SUMMARY CHECKLIST

Before deploying ANY model to production, verify:

- [ ] All 6 rules evaluated explicitly
- [ ] Baseline comparison documented
- [ ] Overfit score calculated and acceptable
- [ ] Statistical significance confirmed
- [ ] OOS validation passed
- [ ] Risk metrics within bounds
- [ ] Costs properly included
- [ ] Decision documented and archived
- [ ] Second reviewer signed off
- [ ] Action plan created (if IMPROVE)
- [ ] Lessons logged (if KILL)
- [ ] Paper trading plan ready (if KEEP)

**No exceptions. No shortcuts. No self-deception.**

---

## VERSION HISTORY

**v2.0 (2025-12-07):** Comprehensive framework overhaul
- Refined 6 core rules with clear thresholds
- Added detailed 3-tier classification system (KEEP/IMPROVE/KILL)
- Created decision workflow diagram
- Added special cases for edge scenarios
- Added emergency override process

**v1.0 (2025-12-05):** Initial framework
- Established 6 core gates
- Defined acceptance workflow
- Created validation checklists

---

**"In God we trust. All others must bring data."**
**— W. Edwards Deming**

**"The best models are the ones that work."**
**— Bull Machine Lab Philosophy**
