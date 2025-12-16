# RULES OF THE LAB - COMPLETE FRAMEWORK
## Acceptance Criteria System for Trading Models

**Created:** 2025-12-07
**Purpose:** Hard rules to prevent overfitting, self-deception, and deployment of bad models

---

## EXECUTIVE SUMMARY

This framework establishes rigorous, unambiguous acceptance criteria for ALL trading models in the Bull Machine system - whether baselines, archetypes, ML models, or ensembles.

**Core Principle:** Everything is a model. No special treatment. Evidence-based decisions only.

**The 6 Rules:** Every model must pass ALL 6 rules to deploy to production:
1. Beat baselines by ≥0.1 PF
2. Low overfit (<0.5)
3. Statistical significance (≥50 trades)
4. OOS validation (PF >1.2, ratio >0.6)
5. Risk-adjusted (DD within limits)
6. Costs included (16 bps)

**Three Outcomes:**
- ✅ **KEEP** (6/6 rules) → Deploy to paper trading
- 🔧 **IMPROVE** (4-5/6 rules) → Fix specific issues, re-test
- ❌ **KILL** (<4/6 rules) → Archive, document learnings, move on

---

## DOCUMENT STRUCTURE

This framework consists of 4 core documents:

### 1. RULES_OF_THE_LAB.md (Main Document)
**Path:** `/RULES_OF_THE_LAB.md`

**Contents:**
- Philosophy and guiding beliefs
- The 6 Rules of Deployment Eligibility (detailed specs)
- Three-tier classification system (KEEP/IMPROVE/KILL)
- Special cases and exceptions (regime-specific, low-freq, ensemble)
- Decision workflow diagram
- Documentation requirements
- Emergency overrides
- Version history

**Use Case:** Primary reference for understanding and applying the acceptance criteria.

---

### 2. MODEL_ACCEPTANCE_CHECKLIST.md (Template)
**Path:** `/MODEL_ACCEPTANCE_CHECKLIST.md`

**Contents:**
- Blank checklist template for evaluating any model
- Section for each of the 6 rules with data fields
- Decision section (KEEP/IMPROVE/KILL)
- Action plan sections
- Reviewer sign-off
- Archive location tracking

**Use Case:** Copy this template for each model evaluation. Fill out systematically. Archive in `/docs/model_decisions/`.

**Example Usage:**
```bash
# Copy template
cp MODEL_ACCEPTANCE_CHECKLIST.md docs/model_decisions/2025-12-07_YourModel_[STATUS].md

# Fill out with model metrics
# Save with decision in filename: KEEP, IMPROVE, or KILL
```

---

### 3. ACCEPTANCE_CRITERIA_EXAMPLES.md (Case Studies)
**Path:** `/docs/ACCEPTANCE_CRITERIA_EXAMPLES.md`

**Contents:**
- 10+ detailed examples of model evaluations
- Clear winners (KEEP examples)
- Needs improvement (IMPROVE examples)
- Kill immediately (KILL examples)
- Edge cases (regime-specific, low-freq, multi-asset)
- Common failure modes and how to fix them

**Use Case:** Reference when evaluating models. Find similar scenarios to your model. Learn from examples.

**Highlighted Examples:**
- Example 1: Ideal archetype (passes all 6 rules)
- Example 2: Negative overfit (excellent generalization)
- Example 4: High overfit (needs parameter reduction)
- Example 6: OOS collapse (regime shift investigation)
- Example 7: Worse than baseline (immediate kill)
- Example 10: Regime-specific deployment
- Example 11: Low-frequency with multi-asset validation

---

### 4. QUANT_LAB_PHILOSOPHY.md (Principles)
**Path:** `/docs/QUANT_LAB_PHILOSOPHY.md`

**Contents:**
- Core philosophy ("Everything is a model")
- Guiding beliefs (Evidence > intuition, etc.)
- Operational principles
- Decision-making framework
- Psychological discipline
- Cultural norms
- The Bull Machine Way

**Use Case:** Onboarding document. Read first to understand WHY these rules exist. Reference when making tough decisions.

**Key Sections:**
- "Evidence Beats Intuition"
- "Complexity Must Justify Itself"
- "Kill Fast, Learn Faster"
- "No Self-Deception"
- Psychological traps to avoid
- Quotes to live by

---

## QUICK START GUIDE

### For First-Time Users

**Step 1: Read Philosophy**
- Start with `QUANT_LAB_PHILOSOPHY.md`
- Understand the "why" behind the rules
- Internalize the mindset (15 min read)

**Step 2: Review Main Rules**
- Read `RULES_OF_THE_LAB.md` sections:
  - The 6 Rules (detailed requirements)
  - Three-tier classification
  - Decision workflow
- Understand pass/fail criteria (20 min read)

**Step 3: Study Examples**
- Read `ACCEPTANCE_CRITERIA_EXAMPLES.md`
- Focus on 3-4 examples similar to your model type
- Note how decisions are made (30 min read)

**Step 4: Evaluate Your Model**
- Copy `MODEL_ACCEPTANCE_CHECKLIST.md` template
- Fill out systematically (rule by rule)
- Make decision (KEEP/IMPROVE/KILL)
- Archive completed checklist

**Total Time Investment:** ~1 hour initial, then 15 min per model evaluation

---

### For Model Evaluation

**Workflow:**
```
1. Run backtest (train/test/OOS splits)
   ↓
2. Calculate metrics (PF, overfit, DD, etc.)
   ↓
3. Copy MODEL_ACCEPTANCE_CHECKLIST.md
   ↓
4. Fill out each rule section
   ↓
5. Count passed rules (0-6)
   ↓
6. Make decision:
   - 6/6 → KEEP
   - 4-5 → IMPROVE
   - <4 → KILL
   ↓
7. Document action plan
   ↓
8. Get second reviewer sign-off
   ↓
9. Archive in /docs/model_decisions/
   ↓
10. Execute decision (deploy/fix/archive)
```

---

## THE 6 RULES (QUICK REFERENCE)

### Rule 1: Beat Baselines
**Requirement:** Test PF > (Best Baseline Test PF + 0.1)
**Why:** Complexity must justify itself
**Example:** Baseline PF 2.12 → Need 2.22+

### Rule 2: Low Overfit
**Requirement:** (Train PF - Test PF) < 0.5
**Why:** Generalization over curve-fitting
**Example:** Train 2.5, Test 2.35 → Overfit 0.15 ✓

### Rule 3: Statistical Significance
**Requirement:** Total Trades ≥ 50 OR low-freq exception
**Why:** Need sample size to trust metrics
**Example:** 25 trades → FAIL unless multi-asset validated

### Rule 4: OOS Validation
**Requirement:** OOS PF > 1.2 AND (OOS/Test) > 0.6
**Why:** Must work on truly unseen data
**Example:** OOS 2.2, Test 2.35 → Ratio 0.94 ✓

### Rule 5: Risk-Adjusted
**Requirement:** Max DD ≤ 2x Baseline DD OR PF > 3.0
**Why:** Can't have unlimited risk
**Example:** DD 25%, Baseline DD 18% → 25% < 36% ✓

### Rule 6: Costs Included
**Requirement:** Slippage ≥5 bps, Fees ≥3 bps (16 bps total)
**Why:** Real trading has costs
**Example:** Backtest includes 8 bps per trade ✓

---

## DECISION FLOWCHART

```
Model Evaluation
    ↓
How many rules passed?
    ↓
┌───────┬───────┬───────┐
│  6/6  │ 4-5/6 │ < 4/6 │
│ KEEP  │ IMPR  │ KILL  │
└───┬───┴───┬───┴───┬───┘
    │       │       │
    ▼       ▼       ▼
  Paper   Fix &   Archive
  Trade   Retest  & Learn
    ↓       ↓
  Prod    Success?
          ↓     ↓
        Paper  Kill
        Trade
```

---

## COMMON SCENARIOS

### Scenario 1: "Great on train, bad on test"
**Diagnosis:** High overfit (Rule 2 fail)
**Decision:** IMPROVE or KILL
**Fix:** Reduce parameters, add regularization, simplify logic

### Scenario 2: "Beats baseline but low sample"
**Diagnosis:** Fails Rule 3 (statistical significance)
**Decision:** IMPROVE (apply exception)
**Fix:** Multi-asset validation OR walk-forward across regimes

### Scenario 3: "Good everywhere except OOS"
**Diagnosis:** Rule 4 fail (regime shift or data leakage)
**Decision:** IMPROVE (investigate) or KILL
**Fix:** Check for data leaks, add regime filters, recalibrate

### Scenario 4: "Worse than baseline"
**Diagnosis:** Rule 1 fail (complexity doesn't add value)
**Decision:** KILL immediately
**Fix:** Use the baseline instead

### Scenario 5: "Complex model, marginally better"
**Diagnosis:** Beats baseline by 0.05 PF (below 0.1 buffer)
**Decision:** KILL (use simple baseline)
**Fix:** Simplicity wins when performance is similar

---

## FILE LOCATIONS

### Templates
```
/MODEL_ACCEPTANCE_CHECKLIST.md          # Blank template
```

### Documentation
```
/RULES_OF_THE_LAB.md                    # Main rules document
/docs/QUANT_LAB_PHILOSOPHY.md           # Philosophy & principles
/docs/ACCEPTANCE_CRITERIA_EXAMPLES.md   # Case studies
```

### Example Evaluations
```
/docs/model_decisions/
  └── EXAMPLE_2025-12-07_S4_FundingDivergence_KEEP.md
```

### Your Model Decisions (archive here)
```
/docs/model_decisions/
  ├── 2025-12-07_YourModel_KEEP.md
  ├── 2025-12-08_AnotherModel_IMPROVE.md
  └── 2025-12-09_FailedModel_KILL.md
```

---

## INTEGRATION WITH DAY 2 VALIDATION

**Day 2 Deliverable:** Unified comparison of baselines vs archetypes

**How Rules Apply:**
1. Run all baselines through validation → Apply 6 rules
2. Run all archetypes through validation → Apply 6 rules
3. Compare results side-by-side
4. Make KEEP/IMPROVE/KILL decisions for each
5. Document which models graduate to production

**Output Format:**
```
Model                    | R1 | R2 | R3 | R4 | R5 | R6 | Score | Decision
-------------------------|----|----|----|----|----|----|-------|----------
Baseline_BuyDip15        | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | 6/6   | KEEP
Baseline_SellRally12     | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | 6/6   | KEEP
S2_SMC_BreakOfStructure  | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | 6/6   | KEEP
S4_FundingDivergence     | ✓  | ✓  | ✓  | ✓  | ✓  | ✓  | 6/6   | KEEP
S1_LiquidityVacuum       | ✗  | ✗  | ✓  | ✓  | ✓  | ✓  | 4/6   | IMPROVE
S5_LongSqueeze           | ✗  | ✗  | ✗  | ✗  | ✓  | ✓  | 2/6   | KILL
H_TrapWithinTrend        | ✓  | ✓  | ✓  | ✗  | ✓  | ✓  | 5/6   | IMPROVE
```

**Next Steps After Day 2:**
- KEEP models → Paper trading queue
- IMPROVE models → Remediation plan + 2-week deadline
- KILL models → Archive + lessons learned log

---

## ANTI-PATTERNS TO AVOID

### ❌ Don't Do This

**1. Cherry-Picking Metrics**
```
WRONG: "Test PF fails BUT look at this other metric!"
RIGHT: "All 6 rules must pass. No exceptions."
```

**2. Rationalizing Failures**
```
WRONG: "It failed OOS but markets changed, so it's OK"
RIGHT: "Failed Rule 4. Need to add regime filter or KILL."
```

**3. Sunk Cost Fallacy**
```
WRONG: "I spent 2 weeks on this, can't kill it now"
RIGHT: "2 weeks are gone. Don't waste 2 more on a failed model."
```

**4. Complexity Bias**
```
WRONG: "This is so clever, it must be good"
RIGHT: "Test PF 2.18 < 2.22. Baseline wins. Use baseline."
```

**5. Hope-Based Trading**
```
WRONG: "It'll work in live, trust me"
RIGHT: "OOS PF 0.9. Data says no. KILL."
```

### ✅ Do This Instead

**1. Systematic Evaluation**
- Fill out checklist completely
- No skipping rules
- Document everything

**2. Honest Assessment**
- If it fails, it fails
- No rationalizing
- Data > opinions

**3. Fast Decisions**
- Kill bad ideas quickly
- Don't chase sunk costs
- Learn and move on

**4. Simplicity Preference**
- Use baseline if close
- Complexity must beat by ≥0.1 PF
- Robust > optimal

**5. Evidence-Based**
- OOS performance is truth
- Paper trading validates
- Live results are final judge

---

## MONITORING & CONTINUOUS IMPROVEMENT

### Weekly Reviews (for KEEP models in production)

**Metrics to Track:**
```python
weekly_check = {
    "live_pf": actual_pf,
    "expected_pf": oos_pf,
    "pf_ratio": actual_pf / oos_pf,  # Should be > 0.8

    "live_dd": current_dd,
    "expected_dd": backtest_dd,
    "dd_ratio": current_dd / backtest_dd,  # Should be < 1.5
}
```

**Alert Levels:**
- 🟢 Green: PF ratio > 0.9 (excellent)
- 🟡 Yellow: PF ratio 0.8-0.9 (investigate)
- 🔴 Red: PF ratio < 0.8 (pause and debug)
- ⚫ Kill: PF ratio < 0.5 for 2 weeks (KILL)

### Quarterly Rule Reviews

**Questions to Ask:**
1. Are thresholds still appropriate? (e.g., is 0.1 PF buffer right?)
2. Have we discovered new failure modes?
3. Do we need additional rules?
4. Should we tighten standards?

**Update Process:**
1. Review all KILL decisions from quarter
2. Identify patterns
3. Propose rule amendments
4. Document in version history
5. Communicate to team

---

## FAQ

**Q: What if my model passes 5/6 rules?**
A: IMPROVE. Identify the failed rule, create specific fix, re-test in 1-2 weeks.

**Q: Can I deploy with 5/6 if the failed rule is "minor"?**
A: No. All 6 rules exist for a reason. Fix it first.

**Q: What if baselines are really bad (PF < 1.5)?**
A: Build better baselines first. Archetypes compete with best baseline, not worst.

**Q: My strategy has only 30 trades but it's macro-focused. Can I deploy?**
A: Only if you pass multi-asset validation (BTC+ETH+SOL) OR walk-forward across 5+ regimes. Document exception.

**Q: Overfit is 0.6 (slightly above 0.5). Is that OK?**
A: No. 0.5 is the hard limit. Reduce parameters and re-optimize.

**Q: Can I skip paper trading if backtest is great?**
A: Absolutely not. Paper trading catches data leakage and execution bugs. Always required.

**Q: What if OOS period had a market crash that's rare?**
A: Investigate if regime-specific. If strategy only works in normal regimes, add regime filter. Re-evaluate with filter active.

**Q: Model works great but I forgot to include costs. Can I add them post-hoc?**
A: No. Re-run full backtest with costs from the start. Results will likely be very different.

**Q: Who can override the 6 rules?**
A: Lead Quant + CTO (both required). Must document override reason, set 30-day re-evaluation, tighten monitoring.

---

## VERSION HISTORY

**v1.0 (2025-12-07):** Initial framework release
- Created 4-document system
- Established 6 rules with clear thresholds
- Defined 3-tier classification (KEEP/IMPROVE/KILL)
- Added 10+ detailed examples
- Documented philosophy and principles

**Next Review:** 2025-03-07 (quarterly)

---

## SUPPORT & QUESTIONS

**Primary Contact:** Lead Quant
**Documentation Location:** `/docs/`
**Issue Tracking:** GitHub Issues (tag: `validation`)
**Team Channel:** #quant-lab (Slack/Discord)

**When in Doubt:**
1. Re-read QUANT_LAB_PHILOSOPHY.md
2. Check ACCEPTANCE_CRITERIA_EXAMPLES.md for similar case
3. Ask second reviewer
4. Default to conservative (when unsure, IMPROVE or KILL)

---

## FINAL REMINDER

**The Rules Are Non-Negotiable.**

- No shortcuts
- No exceptions without documentation
- No self-deception
- No "this time is different"

**The Goal Is Not:**
- Most complex model
- Most "clever" strategy
- Best train performance
- Coolest technology

**The Goal Is:**
- Models that work on unseen data
- Strategies that beat baselines
- Systems that survive regime changes
- Trading that makes money in production

**If it passes the 6 rules, ship it.**
**If it doesn't, fix it or kill it.**

**That's the law of the lab.**

---

**"In God we trust. All others must bring data."**
**— Bull Machine Quant Lab**
