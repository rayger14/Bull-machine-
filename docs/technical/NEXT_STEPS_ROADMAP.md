# Bull Machine: Strategic Roadmap & Next Steps

**Date:** 2026-01-15
**Status:** Phase 1 & 2 Complete → Foundation Fixes Required
**Priority:** Follow critical path before archetype tuning

---

## Executive Summary

**What We Just Completed:**
- ✅ Phase 1: Confidence calibration (R²=0.2471)
- ✅ Phase 2: Hybrid integration into RegimeService
- ✅ Comprehensive backtest validation
- ✅ Engine integration analysis
- ✅ Strategic roadmap planning

**Critical Finding:**
🚨 **Regime detection is 70% complete, NOT "fixed"**
🚨 **Archetype tuning is PREMATURE without validated foundation**

**Recommended Path:** 4-8 weeks to production readiness, then archetype optimization

---

## Current State Analysis

### What's Working ✅

| Component | Status | Details |
|-----------|--------|---------|
| **RegimeService** | ✅ COMPLETE | 3-layer stack, hybrid confidence |
| **Confidence Calibrator** | ✅ PRODUCTION | R²=0.2471, validated |
| **Archetypes (9 total)** | ✅ CALIBRATED | S1, S4, S5, H, B, C, K, G, A |
| **Wyckoff Engine** | ✅ EXCELLENT | 11 features fully wired |
| **Backtest Infrastructure** | ✅ COMPLETE | BaseModel-agnostic |

### Critical Gaps ⚠️

| Issue | Impact | Fix Time | Priority |
|-------|--------|----------|----------|
| **S5 short bug** | 0% short trades | 2-4 hours | **P0** |
| **Regime metadata not saved** | Can't analyze by regime | 30 min | **P0** |
| **No walk-forward validation** | Unknown if overfit | 4-8 hours | **P0** |
| **Position sizing (20%)** | 51% max DD | 30 min | **P0** |
| **SMC BOS unwired** | -20-30% alpha | 3-4 hours | **P1** |
| **FTX recall 32%** | Missed crisis signals | 4-6 hours | **P1** |

---

## Strategic Roadmap (4-8 Weeks to Production)

### Week 1: Foundation Fixes (15-20 hours) 🚨 CRITICAL

**Priority 0 - Blockers:**

1. **Fix Position Sizing (30 minutes)**
   ```python
   # Current: 20% per position → 51% max DD (UNACCEPTABLE)
   # Fix: Reduce to 12% per position

   File: engine/models/archetype_model.py (or config)
   Change: base_position_size = 0.20 → 0.12
   Expected: 51% DD → 30% DD
   Test: Re-run full backtest
   ```

2. **Fix S5 Short Bug (2-4 hours)**
   ```python
   # Current: S5 (long_squeeze) executing LONGS (should be SHORTS)
   # Result: 0% short trades in backtest (100% long bias)

   File: engine/strategies/archetypes/bear/long_squeeze.py
   Investigation:
   - Check signal direction mapping
   - Verify entry/exit logic
   - Compare with S1, S4 (working shorts)

   Expected: 30-40% of trades should be short
   Validation: Run smoke test on 2022 crisis period
   ```

3. **Fix Regime Metadata Saving (30 minutes)**
   ```python
   # Current: All trades show regime: "unknown"
   # Issue: Regime not propagated to Trade object

   Files:
   - engine/backtesting/engine.py (Trade dataclass)
   - engine/models/archetype_model.py (Signal object)

   Change: Add regime_label field to Signal/Trade
   Validation: Backtest output shows correct regimes
   ```

4. **Walk-Forward Validation (4-8 hours)** ← **GO/NO-GO GATE**
   ```bash
   # Test for overfitting using rolling windows

   Script: bin/walk_forward_validation.py (already exists)

   Run:
   python3 bin/walk_forward_validation.py \
     --train_window=12 \
     --test_window=3 \
     --step=3 \
     --period=2018-2024

   Success Criteria:
   - OOS degradation <20% vs in-sample
   - Sharpe remains >0.5 on all windows
   - No catastrophic losses (>40% DD)

   IF FAIL: STOP. Fix overfitting before proceeding.
   IF PASS: Continue to Week 2.
   ```

**Exit Criteria:**
- ✅ Position sizing safe (<35% max DD)
- ✅ Short trading works (20-40% of trades)
- ✅ Regime metadata saved correctly
- ✅ Walk-forward degradation <20%

---

### Week 2-3: Complete Regime Detection (10-15 hours)

**Priority 1 - Regime System:**

1. **Activate Adaptive HMM Regime (1 hour)**
   ```python
   # Current: Using static precomputed labels
   # Available: Agent 3 retrained HMM (AGENT3_HMM_RETRAINING_STATUS.md)

   File: engine/context/regime_service.py
   Change: mode='dynamic_ensemble' with HMM features
   Expected: 10-20% performance improvement
   ```

2. **Improve Crisis Detection (4-6 hours)**
   ```python
   # Current: FTX recall 32% (below 60% target)
   # Goal: Increase to 60%+ without false positives

   Files:
   - engine/context/regime_service.py (EventOverrideDetector)
   - Feature engineering scripts

   Improvements:
   - Better volatility shock detection
   - Improved crash frequency calculation
   - Liquidity drain persistence tracking

   Validation: Smoke test on FTX crisis period
   ```

3. **Circuit Breaker Strict Mode (30 minutes)**
   ```python
   # Current: Monitoring mode only (logs but doesn't halt)
   # Production: Need strict enforcement

   File: engine/risk/circuit_breaker.py
   Change: strict_mode = True
   Test: Simulate 20% DD, verify halt
   ```

4. **Full Backtest Re-run (1 hour)**
   ```bash
   # With all fixes applied

   Run full engine backtest with:
   - 12% position sizing
   - S5 short bug fixed
   - Adaptive regime enabled
   - Crisis detection improved

   Expected Results:
   - Sharpe 0.31 → 0.8-1.0
   - Max DD 51% → 25-35%
   - Short trades: 0% → 30-40%
   - Crisis recall: 75%/32% → 75%/60%
   ```

**Exit Criteria:**
- ✅ Crisis recall >60% on LUNA AND FTX
- ✅ Risk-on detection >15%
- ✅ Regime transitions 10-40/year (hysteresis working)
- ✅ Backtest Sharpe >0.8

**Decision Point:** If Sharpe <0.8 after fixes → Diagnose before archetype work

---

### Week 4-8: Paper Trading (2-4 weeks) 🎯 REQUIRED

**Priority 2 - Real Market Validation:**

1. **Deploy Paper Trading (2 hours setup)**
   ```bash
   # Infrastructure setup
   - Configure paper trading account
   - Deploy RegimeService + ArchetypeModel
   - Set up monitoring dashboard
   - Enable logging pipeline
   ```

2. **Monitor 50-100 Trades (2-4 weeks)**
   ```
   Key Metrics to Track:
   - Paper return vs backtest return
   - Max DD vs backtest DD
   - Sharpe vs backtest Sharpe
   - Slippage (assumed 0.08%, actual?)
   - Fill rate (assumed 100%, actual?)
   - Regime detection accuracy

   Minimum Trades: 50 (statistically meaningful)
   Ideal Trades: 100+ (higher confidence)
   ```

3. **Acceptance Criteria**
   ```
   ✅ PASS: Paper return within 80-120% of backtest
   ✅ PASS: Max DD within 100-120% of backtest
   ✅ PASS: Sharpe within 80-120% of backtest
   ✅ PASS: No catastrophic losses (>50% DD)
   ✅ PASS: All risk systems working (circuit breaker, position limits)

   ⚠️ CAUTION: Degradation 20-40% → Re-calibrate parameters
   ❌ FAIL: Degradation >40% → Major issues, do NOT go live
   ```

4. **GO/NO-GO Decision**
   ```
   IF degradation <20%:
     → Proceed to live trading (10% capital)
     → Monitor for 2 weeks
     → Scale to 25%, 50%, 100% over 2 months

   IF degradation 20-40%:
     → Re-calibrate parameters
     → Run another 50 trades
     → Re-evaluate

   IF degradation >40%:
     → ABORT live deployment
     → Investigate assumptions (slippage, fees, fill rate)
     → Fix issues and restart from Week 1
   ```

**Exit Criteria:**
- ✅ 50+ paper trades executed
- ✅ Degradation <20% vs backtest
- ✅ No catastrophic losses
- ✅ All risk systems working

---

### Month 3+: Archetype Optimization (ONLY if paper trading succeeds)

**Priority 3 - Regime-Aware Tuning:**

1. **Wire SMC Engine BOS Signals (3-4 hours)** ← Quick win, +20-30% alpha
   ```python
   # Current: SMC engine 1/7 features wired (smc_score only)
   # Missing: 6 BOS/FVG signals

   Features to wire:
   - tf1h_bos_bearish → S1, S4, S5 (capitulation confirmation)
   - tf1h_bos_bullish → S5 (trap reversal exits)
   - tf4h_bos_bearish → S1, S4
   - tf4h_bos_bullish → Bull archetypes (H, B, C)
   - tf1h_fvg_bear, tf1h_fvg_bull → Bull archetypes (confluence)

   Files: engine/strategies/archetypes/bear/*.py
          engine/strategies/archetypes/bull/*.py

   Expected Impact: +20-30% signal quality
   ```

2. **Regime-Specific Parameter Tuning (2-4 weeks)**
   ```python
   # Current: Same thresholds for all regimes
   # Goal: Different thresholds per regime

   Example (S1 Liquidity Vacuum):
   - Crisis regime: More aggressive (fusion_threshold 0.30 → 0.25)
   - Risk-off regime: Standard (0.35)
   - Neutral/risk-on: Disable (not applicable)

   Process:
   1. Split data by regime
   2. Optimize thresholds per regime
   3. Walk-forward validate per regime
   4. Deploy and monitor

   Expected Impact: +10-15% Sharpe improvement
   ```

3. **Multi-Objective Optimization (2-3 weeks)**
   ```python
   # Framework: Agent 3 completed (MULTI_OBJECTIVE_OPTIMIZATION_REPORT.md)

   Objectives:
   - Maximize Sharpe (primary)
   - Minimize max DD (risk constraint)
   - Maximize trade frequency (liquidity constraint)
   - Maintain win rate >55%

   Apply to:
   - Regime-specific archetype configs
   - Position sizing rules
   - Stop loss placement

   Validation: Walk-forward per regime
   ```

4. **Meta-Model (Overlap-as-Feature) (9-12 weeks)** - Optional
   ```python
   # Full implementation roadmap: META_MODEL_IMPLEMENTATION_ROADMAP.md

   Concept: Treat archetype overlap as feature
   - Current: 56.5% signal overlap (inefficient)
   - Meta-model: Learn from overlap patterns

   Expected Impact:
   - Win rate 55% → 60-65%
   - Sharpe +0.3
   - Better capital efficiency

   Complexity: HIGH (9-12 weeks)
   Dependency: Requires validated baseline first
   ```

**Exit Criteria:**
- ✅ Regime-specific configs improve Sharpe >+0.2
- ✅ Walk-forward validation per regime passes
- ✅ No degradation in crisis performance
- ✅ Live trading validation (paper → 10% → 100% capital)

---

## Dependency Chain

```
┌─────────────────────────────────────────────────────────────┐
│                   CRITICAL PATH GATES                        │
└─────────────────────────────────────────────────────────────┘

Week 1: Foundation Fixes (20 hours)
├── Position sizing fix (30 min)
├── S5 short bug fix (2-4 hours)
├── Regime metadata fix (30 min)
└── Walk-forward validation (4-8 hours)
                │
        ┌───────┴────────┐
        │ GATE 1         │  ← CRITICAL DECISION POINT
        │ Pass? (<20% DD)│
        └───────┬────────┘
                │ YES
                ▼
Week 2-3: Complete Regime Detection (15 hours)
├── Activate HMM regime (1 hour)
├── Improve crisis detection (4-6 hours)
├── Circuit breaker strict (30 min)
└── Full backtest re-run (1 hour)
                │
        ┌───────┴────────┐
        │ GATE 2         │  ← GO/NO-GO DECISION
        │ Sharpe >0.8?   │
        └───────┬────────┘
                │ YES
                ▼
Week 4-8: Paper Trading (2-4 weeks)
├── Deploy paper trading (2 hours)
├── Monitor 50-100 trades (2-4 weeks)
├── Acceptance testing
└── GO/NO-GO decision
                │
        ┌───────┴────────┐
        │ GATE 3         │  ← FINAL GATE TO PRODUCTION
        │ <20% degrad?   │
        └───────┬────────┘
                │ YES
                ▼
        ┌────────────────┐
        │ LIVE TRADING   │
        │ (10% capital)  │
        └────────────────┘
                │
                ▼
Month 3+: Archetype Optimization (ONLY if live succeeds)
├── Wire SMC BOS signals (3-4 hours)
├── Regime-specific parameters (2-4 weeks)
├── Multi-objective optimization (2-3 weeks)
└── Meta-model (9-12 weeks, optional)

IF ANY GATE FAILS → STOP and fix issues before proceeding
```

---

## Risk Assessment

### High-Probability Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Walk-forward fails** | 40% | CRITICAL | Must fix overfitting, retrain |
| **Paper trading >40% degrad** | 30% | CRITICAL | Invalidates backtest, restart |
| **S5 bug unfixable** | 10% | High | Disable S5, lose 30% of signals |
| **Regime shift in production** | 50% | Medium | Monthly retraining, monitoring |

### Technical Debt

**High Impact:**
- No domain engine integration (-10-20% performance)
- Feature engineering not audited for lookahead bias (risk of overfit)
- Archetype overlap 56.5% (inefficient capital use)

**Medium Impact:**
- Only 62% archetype utilization (A, K inactive)
- No structured rejection logging
- Circuit breaker in monitoring mode

---

## Answer to Your Question

### "If we fix regime detection, do we work towards archetype tuning based on regimes?"

**Short Answer:** NO - Wrong sequencing.

**Correct Sequence:**
1. **Fix foundation** (Week 1) ← YOU ARE HERE
2. **Complete regime detection** (Week 2-3)
3. **Paper trading** (Week 4-8) ← **REQUIRED**
4. **THEN** archetype tuning (Month 3+)

**Why This Order:**

1. **Foundation not validated**
   - No walk-forward test (unknown if overfit)
   - S5 bug blocks short trading (30-40% of opportunities)
   - Regime metadata not saved (can't analyze by regime)
   - Position sizing too aggressive (51% DD unacceptable)

2. **Regime detection not "fixed"**
   - 70% complete, not 100%
   - FTX recall 32% (below 60% target)
   - Risk-on detection 8.7% (below 15% target)
   - Missing walk-forward validation

3. **Paper trading skipped**
   - Industry best practice: NEVER skip paper trading
   - Unknown execution assumptions (slippage, fees, fill rate)
   - Risk: 40%+ degradation on live → catastrophic loss

4. **Archetype tuning premature**
   - Optimizing for unvalidated baseline = optimizing noise
   - Regime-specific parameters need stable regime labels
   - Meta-model needs 50k+ validated signals (don't have yet)

**Timeline to Production:**
- **4-8 weeks** following critical path
- **3-6 months** for full optimization (including meta-model)

---

## Immediate Next Actions

### Today (1-2 hours)

1. **Execute commit strategy**
   ```bash
   # Follow COMMIT_STRATEGY_GUIDE.md
   ./bin/cleanup_and_organize_docs.sh
   git commit -F commit_message_1_feature.txt
   git commit -F commit_message_2_docs.txt
   git commit -F commit_message_3_cleanup.txt
   git push origin feature/ghost-modules-to-live-v2
   ```

2. **Review strategic roadmap**
   - Read this document thoroughly
   - Understand gates and dependencies
   - Prepare for Week 1 foundation fixes

### Week 1 (15-20 hours)

1. **Fix position sizing** (30 min)
2. **Fix S5 short bug** (2-4 hours)
3. **Fix regime metadata** (30 min)
4. **Run walk-forward validation** (4-8 hours)
5. **GO/NO-GO decision** based on walk-forward results

### After Week 1

- If walk-forward passes → Proceed to Week 2-3
- If walk-forward fails → Fix overfitting, restart validation

---

## Key Insights

### What You've Built

✅ **Excellent infrastructure** (the hard part is done)
- Hybrid regime detection with confidence calibration
- 9 production/calibrated archetypes
- Clean architecture (RegimeService, ArchetypeModel, BacktestEngine)
- Comprehensive testing and documentation

### What You Need to Fix

⚠️ **Foundation validation** (relatively easy, 20 hours)
- Walk-forward test (proves it's not overfit)
- S5 short bug (enables bear market profitability)
- Regime metadata (enables regime-specific analysis)
- Position sizing (reduces risk to acceptable levels)

### What You Should NOT Do

❌ **Optimize before validating**
- Archetype tuning without walk-forward = optimizing noise
- Live trading without paper = gambling
- Regime-specific parameters without stable regimes = wasted effort

---

## Success Metrics

### Week 1 Success
- ✅ Walk-forward degradation <20%
- ✅ Position sizing →Max DD <35%
- ✅ S5 short trades 30-40%
- ✅ Regime metadata saved correctly

### Week 2-3 Success
- ✅ Crisis recall >60% (LUNA & FTX)
- ✅ Backtest Sharpe >0.8
- ✅ Regime transitions 10-40/year

### Week 4-8 Success
- ✅ Paper trading degradation <20%
- ✅ 50+ trades executed
- ✅ No catastrophic losses

### Month 3+ Success
- ✅ Live trading profitable (>Sharpe 0.8)
- ✅ Regime-specific tuning +0.2 Sharpe
- ✅ Production monitoring stable

---

## Resources

### Documentation Created
- `COMMIT_STRATEGY_GUIDE.md` - Execution guide for commits
- `HYBRID_CONFIDENCE_GUIDE.md` - Usage guide for hybrid metrics
- `PHASE_1_COMPLETE_FINDINGS.md` - Phase 1 research
- `PHASE_2_INTEGRATION_COMPLETE.md` - Phase 2 summary
- `NEXT_STEPS_ROADMAP.md` - This document

### Agent Reports
- Backtest validation: `HYBRID_CONFIDENCE_BACKTEST_REPORT.md`
- Engine analysis: Agent a46bf0c findings
- Cleanup strategy: Agent a5e2e0b recommendations
- Strategic guidance: Agent a646d4e roadmap

### Scripts Available
- `bin/cleanup_and_organize_docs.sh` - Repository cleanup
- `bin/walk_forward_validation.py` - Overfitting test
- `bin/test_hybrid_confidence_integration.py` - Integration tests
- `bin/backtest_hybrid_confidence_validation.py` - Backtest validation

---

## Bottom Line

**Status:** Phase 1 & 2 complete ✅, Foundation validation required ⚠️

**Next Step:** Execute commit strategy, then start Week 1 foundation fixes

**Timeline to Production:** 4-8 weeks (if all gates pass)

**Key Message:** You have excellent infrastructure. Don't skip validation to jump to optimization. Follow the critical path, pass the gates, and you'll have a production-ready system in 4-8 weeks.

**Remember:** "The bones are excellent. Don't skip validation to jump to optimization." - System Architect Agent

---

**Ready to proceed?** Start with `COMMIT_STRATEGY_GUIDE.md` → Week 1 foundation fixes.
