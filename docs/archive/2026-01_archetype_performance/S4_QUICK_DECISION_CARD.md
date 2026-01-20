# S4 Quick Decision Card

**Status:** ANALYSIS COMPLETE | **Date:** 2026-01-07

---

## The Question

**Should we deploy S4 (Funding Divergence) to production?**

## The Answer

**NO. Deploy S1 (Liquidity Vacuum) instead.**

---

## Why Not S4?

```
❌ 70.7% OOS degradation (target: <20%)
❌ 0% OI data coverage (broken features)
❌ Backwards performance (fails in bear, succeeds in bull)
❌ Too risky for first production deployment
```

## Why S1?

```
✅ 1.5% OOS degradation (excellent)
✅ Proven through walk-forward validation
✅ Same regime target (bear/crisis)
✅ No dependency on broken OI data
```

---

## Quick Stats

| Metric | S4 | S1 |
|--------|----|----|
| **OOS Degradation** | 70.7% ❌ | 1.5% ✅ |
| **Bear Windows** | 0/4 ❌ | 2/4 ⚠️ |
| **Production Ready** | NO | CLOSE |
| **Data Issues** | OI 0% | None |

---

## What To Do

### TODAY
```bash
# 1. Update deployment config
# Remove S4 from Week 1
# Deploy S1 only

# 2. Document decision
# "S4 disabled due to 70.7% degradation + 0% OI data"
```

### WEEK 1
- Deploy S1 to paper trading
- Monitor execution quality
- Validate infrastructure

### WEEK 2-4
- Validate S5 (Long Squeeze)
- Test S2 (Failed Rally)
- Test S8 (Breakdown)

### MONTH 2
- Fix OI data pipeline
- Re-engineer S4 (2020-2024 training)
- Extended validation
- Deploy if <20% degradation

---

## If You Insist on Fixing S4 Now

### Option A: Re-optimize on Full 2022

**Command:**
```bash
python bin/optimize_s4_multi_objective_v2.py \
  --train-start 2022-01-01 \
  --train-end 2022-12-31 \
  --n-trials 50
```

**Time:** 1-2 hours
**Success Rate:** 40-60%
**Risk:** High

**Success Criteria:**
- OOS degradation <25% = deploy ✅
- OOS degradation 25-40% = caution ⚠️
- OOS degradation >40% = disable ❌

**Recommendation:** Only try if you have time to spare and accept the risk.

---

## Files Created

1. **`S4_EXECUTIVE_SUMMARY.md`** ← Start here
2. **`S4_OOS_DEGRADATION_FINAL_REPORT.md`** ← Full analysis
3. **`S4_BEFORE_AFTER_COMPARISON.md`** ← Detailed comparison
4. **`S4_OOS_DEGRADATION_FIX_PLAN.md`** ← All 4 options explained
5. **`bin/optimize_s4_multi_objective_v2.py`** ← Ready to run (if Option A)
6. **`S4_QUICK_DECISION_CARD.md`** ← This file

---

## Key Insights

**Root Cause:**
- Tiny training window (6 months crisis)
- Broken OI data (0% coverage)
- Backwards parameters (opposite of design)

**The Fix:**
- Expand training: 6 mo → 4 years
- Fix OI pipeline or remove dependency
- Extended validation (2020-2024)
- Timeline: 4-6 weeks (Month 2 project)

**Why Wait:**
- First production test is CRITICAL
- Need clean signal quality
- Don't confound execution vs alpha issues
- S1 works NOW, fix S4 later

---

## Decision Tree

```
Is S4 production-ready?
│
├─ OOS degradation <20%? NO (70.7%)
├─ Data quality good? NO (OI 0%)
├─ Performance correct? NO (backwards)
└─ Worth the risk? NO

Should we deploy S4?
└─ NO → Deploy S1 instead

When can we deploy S4?
└─ Month 2 (after re-engineering)
```

---

## TLDR

**Problem:** S4 has 70.7% degradation + 0% OI data + backwards performance

**Solution:** Disable S4, deploy S1 (1.5% degradation, proven)

**Timeline:** Re-engineer S4 in Month 2 (4-6 weeks)

**Command:** Update deployment config to remove S4

---

**Recommendation:** DISABLE S4, DEPLOY S1
**Confidence:** 85%
**Next Step:** Update production deployment plan

---

## Questions?

**Q: Can we fix S4 quickly?**
A: Maybe (40-60% success), but risky for first deployment.

**Q: What if Option A works?**
A: Great! Add S4 in Week 2 after S1 proves infrastructure.

**Q: When will S4 be ready?**
A: Month 2-3 (after data fixes + re-engineering).

---

**BOTTOM LINE: Don't risk your first production test on a broken strategy. Deploy what works (S1), fix what doesn't (S4).**
