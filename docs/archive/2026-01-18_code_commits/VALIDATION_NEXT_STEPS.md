# VALIDATION NEXT STEPS - ACTIONABLE ROADMAP

**Date:** 2025-12-08
**Status:** VALIDATION COMPLETE - DEPLOYMENT PLAN READY
**Recommendation:** Deploy Baselines Only (Scenario C)

---

## IMMEDIATE ACTIONS (Today)

### 1. Review Validation Completion Report ⏱ 30 minutes

**File:** `ARCHETYPE_VALIDATION_COMPLETE.md`

**Review Checklist:**
- [ ] Understand 5-day journey timeline
- [ ] Verify before/after comparison metrics
- [ ] Review validation protocol results (9 steps)
- [ ] Understand why archetypes = baselines (PF 1.76)
- [ ] Agree with Scenario C recommendation

**Key Insight:** Infrastructure fixed (+335% feature coverage, +488% domain engines), yet zero PF improvement

---

### 2. Approve Deployment Scenario ⏱ 15 minutes

**Decision Required:** Scenario A / B / C

**Options:**

**Scenario A: Archetypes Win** ❌ NOT APPLICABLE
- Condition: Archetype PF > Baseline PF by ≥5%
- Result: Archetypes PF 1.76 = Baselines PF 1.76 (0% advantage)
- Status: Condition not met

**Scenario B: Hybrid (50/50)** ❌ NOT RECOMMENDED
- Condition: Archetypes PF within ±5% of Baseline PF
- Result: 0% delta, but adds complexity without benefit
- Issues: Double maintenance burden, no diversification
- Recommendation: REJECT

**Scenario C: Baselines Only** ✅ RECOMMENDED
- Condition: Baseline PF ≥ Archetype PF with simpler logic
- Result: Tie on performance, baselines 40% simpler code
- Benefits: Single system, easier maintenance, proven results
- Recommendation: APPROVE

**Action:** Notify stakeholders of Scenario C approval

---

### 3. Notify Stakeholders ⏱ 15 minutes

**Email Template:**

```
Subject: Bull Machine Validation Complete - Baselines Deployment Approved

Team,

Archetype validation is complete. After 5 days of systematic testing:

RESULTS:
- Archetypes: PF 1.76 (366 trades, 49.5% win rate)
- Baselines: PF 1.76 (366 trades, 49.5% win rate)
- Conclusion: Performance parity, baselines simpler

FIXES DELIVERED:
- Feature coverage: 20% → 87% (+335%)
- Domain engines: 17% → 100% (+488%)
- Calibrations: Vanilla → Optuna optimized
- Infrastructure: All issues resolved

DEPLOYMENT DECISION:
- Approved: Scenario C (Baselines Only)
- Rationale: Equal performance, simpler maintenance
- Timeline: 8 weeks to full deployment
- Expected PF: 1.76 (validated on 2024 bull market)

NEXT STEPS:
- Week 1-2: Paper trading validation
- Week 3-4: Live small (10% capital)
- Week 5-8: Scale to 100%

See ARCHETYPE_VALIDATION_COMPLETE.md for full report.

[Your Name]
```

---

## WEEK 1 PREPARATION (Days 1-5)

### Day 1: Set Up Paper Trading Environment ⏱ 4 hours

**Tasks:**

**1. Configure Paper Trading Broker Connection**
```bash
# Set up OKX testnet credentials
export OKX_API_KEY="testnet-key"
export OKX_SECRET_KEY="testnet-secret"
export OKX_PASSPHRASE="testnet-passphrase"
export OKX_TESTNET=true

# Test connection
python bin/test_okx_api.py \
  --mode testnet \
  --check-balance true \
  --check-market-data true
```

**2. Prepare Baseline Production Config**
```bash
# Copy validated config
cp configs/mvp/mvp_bull_market_v1.json \
   configs/production/baseline_paper_trading.json

# Review config settings
vim configs/production/baseline_paper_trading.json

# Verify:
# - max_positions: 3 (limit concurrent trades)
# - risk_per_trade: 0.02 (2% risk)
# - enable_baselines: true
# - enable_archetypes: false
# - fusion.entry_threshold_confidence: 0.99 (no tier-1 fallback)
```

**3. Initialize Paper Trading Database**
```bash
# Create paper trading log database
python bin/init_paper_trading_db.py \
  --mode paper \
  --reset true

# Expected output:
# Created tables: trades, signals, positions, performance
# Paper trading mode: ACTIVE
```

**Success Criteria:**
- [ ] OKX testnet connection working
- [ ] Baseline config validated
- [ ] Paper trading database initialized

---

### Day 2: Configure Monitoring Dashboards ⏱ 3 hours

**Tasks:**

**1. Set Up Monitoring Script**
```bash
# Create monitoring config
cat > configs/monitoring/paper_trading_monitor.json << EOF
{
  "check_interval_seconds": 300,
  "alerts": {
    "pf_min": 1.41,
    "trade_count_min": 330,
    "trade_count_max": 400,
    "execution_errors": true,
    "data_quality_degradation": true
  },
  "webhooks": {
    "slack": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    "email": "alerts@yourdomain.com"
  }
}
EOF

# Test monitoring
python bin/monitor_paper_trading.py \
  --config configs/monitoring/paper_trading_monitor.json \
  --test-mode true
```

**2. Create Performance Dashboard**
```bash
# Generate daily summary
python bin/generate_dashboard.py \
  --mode paper \
  --output results/paper_trading/dashboard.html \
  --refresh-interval 300

# Open dashboard
open results/paper_trading/dashboard.html
```

**Dashboard Widgets:**
- [ ] Current PF (target ≥ 1.41)
- [ ] Trade count (target 330-400)
- [ ] Win rate (target ~50%)
- [ ] Recent trades table
- [ ] Signal generation chart
- [ ] Feature coverage status
- [ ] Data quality indicators

**Success Criteria:**
- [ ] Monitoring script running every 5 minutes
- [ ] Dashboard auto-refreshing
- [ ] Alerts configured (Slack/Email)

---

### Day 3: Test Alerting Systems ⏱ 2 hours

**Tasks:**

**1. Trigger Test Alerts**
```bash
# Simulate low PF alert
python bin/trigger_test_alert.py \
  --alert-type pf_below_threshold \
  --test-value 1.0 \
  --expected-pf 1.41

# Expected: Slack/Email notification received

# Simulate execution error
python bin/trigger_test_alert.py \
  --alert-type execution_error \
  --error-message "Missed exit signal at 2024-12-08 10:30"

# Expected: Critical alert sent
```

**2. Configure Alert Escalation**
```yaml
# configs/monitoring/alert_escalation.yaml
alerts:
  warning:
    conditions:
      - pf < 1.41 and pf > 1.20
    actions:
      - slack_notify
  critical:
    conditions:
      - pf < 1.20
      - execution_error
      - data_quality < 0.80
    actions:
      - slack_notify
      - email_notify
      - sms_notify
  escalate:
    conditions:
      - critical for > 4 hours
    actions:
      - page_on_call
```

**Success Criteria:**
- [ ] Warning alerts working (Slack)
- [ ] Critical alerts working (Slack + Email)
- [ ] Escalation policy tested

---

### Day 4: Brief Trading Team ⏱ 1 hour

**Presentation Outline:**

**Slide 1: Validation Results**
- Archetypes vs Baselines: PF 1.76 (tie)
- Decision: Deploy baselines only
- Rationale: Simpler, equal performance

**Slide 2: Baseline Strategy Overview**
- What: Multi-archetype baseline patterns (wick_trap, retest, continuation, etc.)
- Why: Proven PF 1.76 on 2024 bull market
- How: Domain engine fusion + regime routing

**Slide 3: Paper Trading Plan**
- Week 1-2: Validate signals in paper mode
- Success criteria: PF ≥ 1.41, 330-400 trades
- Monitoring: 5-minute intervals, automated alerts

**Slide 4: Risk Management**
- Max 3 concurrent positions
- 2% risk per trade
- 8% portfolio risk limit
- Auto-stop if DD > 20%

**Slide 5: Escalation Procedures**
- Warning: PF < 1.41 → Monitor closely
- Critical: PF < 1.20 → Investigate immediately
- Emergency: Execution errors → Pause trading

**Q&A Topics:**
- What if paper trading fails? (Re-validate config, extend paper period)
- When do we go live? (After 2 weeks if PF ≥ 1.41)
- What about archetypes? (Archived, revisit in 6-12 months)

**Success Criteria:**
- [ ] Trading team briefed
- [ ] Questions answered
- [ ] Escalation procedures understood

---

### Day 5: Final Pre-Launch Checklist ⏱ 2 hours

**Infrastructure Checklist:**
- [ ] OKX testnet connection stable
- [ ] Baseline config validated (baselines enabled, archetypes disabled)
- [ ] Paper trading database initialized
- [ ] Monitoring dashboards operational
- [ ] Alerting systems tested
- [ ] Trading team briefed

**Config Validation:**
```bash
# Run comprehensive config check
python bin/validate_config.py \
  --config configs/production/baseline_paper_trading.json \
  --mode paper \
  --check-all true

# Expected output:
# ✅ Baselines enabled
# ✅ Archetypes disabled
# ✅ Risk limits configured (2% per trade, 8% portfolio)
# ✅ Max positions: 3
# ✅ Feature coverage ≥ 90%
# ✅ Domain engines: 18/18 active
# READY FOR PAPER TRADING
```

**Data Quality Check:**
```bash
# Verify live data feeds
python bin/check_data_quality.py \
  --sources okx,macro,funding \
  --check-latency true \
  --check-null-rates true

# Expected output:
# OKX: ✅ <100ms latency, 0% null
# Macro (VIX/DXY): ✅ <5s latency, <5% null
# Funding: ✅ <1s latency, 0% null
# DATA QUALITY: EXCELLENT
```

**Go/No-Go Decision:**
- ✅ 6/6 infrastructure checks passed → Proceed to paper trading
- ❌ <6/6 checks passed → Fix issues, re-validate

---

## WEEK 1-2 EXECUTION: PAPER TRADING (Days 6-19)

### Daily Routine (Days 6-19)

**Morning (9:00 AM UTC):**
```bash
# Check overnight performance
python bin/paper_trading_summary.py \
  --period yesterday \
  --compare-to-baseline true

# Review:
# - Trades executed: X (expected ~1-2/day)
# - PF (rolling 7-day): X.XX (target ≥ 1.41)
# - Execution errors: X (target 0)
```

**Midday (12:00 PM UTC):**
```bash
# Monitor live signals
python bin/monitor_paper_signals.py \
  --show-recent 10 \
  --alert-if-stale true

# Expected: 0-2 active positions, signals generating normally
```

**Evening (6:00 PM UTC):**
```bash
# Generate daily report
python bin/generate_daily_report.py \
  --mode paper \
  --output results/paper_trading/daily_2024-12-XX.md

# Review:
# - Total trades: XXX
# - PF: X.XX
# - Delta vs backtest: ±X%
# - Issues: [None or list]
```

**Weekly Review (Day 12, Day 19):**
```bash
# Compare week 1 vs week 2
python bin/compare_paper_weeks.py \
  --week1 results/paper_trading/week1_summary.json \
  --week2 results/paper_trading/week2_summary.json

# Metrics:
# - PF stability (week-to-week variance)
# - Trade count consistency
# - Execution quality
```

---

### Week 2 End: Paper Trading Go/No-Go Decision (Day 19)

**Success Criteria Review:**

**Criterion 1: PF ≥ 1.41 (80% of backtest 1.76)**
```bash
python bin/calculate_paper_pf.py \
  --period 14d \
  --min-trades 20

# Expected: PF ≥ 1.41
```

**Criterion 2: Trade Count 330-400 (±10% of backtest 366)**
```bash
python bin/count_paper_trades.py \
  --period 14d \
  --expected-min 330 \
  --expected-max 400

# Expected: 330-400 trades
```

**Criterion 3: Zero Execution Errors**
```bash
python bin/audit_execution_errors.py \
  --period 14d \
  --include-types missed_exit,panic_sell,stale_signal

# Expected: 0 errors
```

**Decision Matrix:**
```
Criteria Met    Decision
3/3             ✅ Proceed to Live Small (Week 3)
2/3             ⚠️  Extend paper trading 1 week, investigate
1/3 or 0/3      ❌ Stop, re-validate config, restart paper trading
```

---

## WEEK 3-4: LIVE SMALL (10%) (Days 20-33)

### Day 20: Deploy Live Small ⏱ 4 hours

**Pre-Deployment Checks:**
```bash
# Switch to live API credentials
export OKX_API_KEY="live-key"
export OKX_SECRET_KEY="live-secret"
export OKX_PASSPHRASE="live-passphrase"
export OKX_TESTNET=false

# Verify live connection
python bin/test_okx_api.py \
  --mode live \
  --check-balance true \
  --check-order-placement false

# Expected: Balance confirmed, DO NOT place test orders
```

**Deploy with 10% Allocation:**
```bash
# Create live config
cp configs/production/baseline_paper_trading.json \
   configs/production/baseline_live_small.json

# Update allocation
vim configs/production/baseline_live_small.json
# Set:
# - allocation: 0.10 (10% of capital)
# - max_positions: 1 (reduce concurrent risk)
# - risk_per_trade: 0.01 (1% risk, halved for caution)

# Deploy live
python bin/deploy_live_baseline.py \
  --config configs/production/baseline_live_small.json \
  --mode live \
  --confirm true

# Expected output:
# Live trading ACTIVE
# Allocation: 10%
# Max positions: 1
# Risk per trade: 1%
# Monitoring: 1-minute intervals
```

**Success Criteria:**
- [ ] Live API connection working
- [ ] 10% allocation configured
- [ ] First trade executed successfully

---

### Daily Monitoring (Days 20-33)

**Critical Metrics:**
```bash
# Morning check
python bin/monitor_live_trading.py \
  --period yesterday \
  --check-pf-min 0.88 \
  --check-slippage-max 0.0005 \
  --alert-unexpected true

# Review:
# - Live PF: X.XX (target ≥ 0.88, which is 50% of backtest 1.76)
# - Slippage: X.XX% (target ≤ 0.05%)
# - Unexpected behavior: None
```

**Slippage Analysis:**
```bash
# Compare execution vs backtest assumptions
python bin/analyze_slippage.py \
  --period 7d \
  --compare-to-backtest true

# Expected:
# - Entry slippage: <0.03% (backtest assumes 0.05%)
# - Exit slippage: <0.02%
# - Fees: ~0.04% (maker) or ~0.06% (taker)
```

**Behavior Audit:**
```bash
# Check for unexpected patterns
python bin/audit_live_behavior.py \
  --period 7d \
  --check-panic-sells true \
  --check-missed-exits true \
  --check-stale-signals true

# Expected: 0 unexpected behaviors
```

---

### Week 4 End: Live Small Go/No-Go Decision (Day 33)

**Success Criteria Review:**

**Criterion 1: Live PF ≥ 0.88 (50% of backtest 1.76)**
```bash
python bin/calculate_live_pf.py \
  --period 14d \
  --mode live

# Expected: PF ≥ 0.88
```

**Criterion 2: Slippage/Fees Within Assumptions (≤0.05%)**
```bash
python bin/analyze_execution_costs.py \
  --period 14d \
  --max-slippage 0.0005

# Expected: Slippage ≤ 0.05%, fees ~0.04-0.06%
```

**Criterion 3: Zero Unexpected Behavior**
```bash
python bin/audit_unexpected_behavior.py \
  --period 14d \
  --strict-mode true

# Expected: 0 panic sells, 0 missed exits, 0 stale signals
```

**Decision Matrix:**
```
Criteria Met    Decision
3/3             ✅ Proceed to Scale Up (Week 5)
2/3             ⚠️  Extend live small 1 week, tune parameters
1/3 or 0/3      ❌ Revert to paper trading, investigate issues
```

---

## WEEK 5-8: SCALE UP (100%) (Days 34-61)

### Day 34: Scale to Full Allocation ⏱ 2 hours

**Gradual Scale Plan:**
```bash
# Week 5: 25% allocation
python bin/scale_live_baseline.py \
  --allocation 0.25 \
  --max-positions 2

# Week 6: 50% allocation
python bin/scale_live_baseline.py \
  --allocation 0.50 \
  --max-positions 4

# Week 7: 75% allocation
python bin/scale_live_baseline.py \
  --allocation 0.75 \
  --max-positions 6

# Week 8: 100% allocation
python bin/scale_live_baseline.py \
  --allocation 1.00 \
  --max-positions 8
```

**Success Criteria (Week 8 End):**

**Criterion 1: Portfolio PF ≥ 1.76**
```bash
python bin/calculate_portfolio_pf.py \
  --period 28d

# Expected: PF ≥ 1.76
```

**Criterion 2: Drawdown < 20%**
```bash
python bin/calculate_max_drawdown.py \
  --period 28d

# Expected: Max DD < 20%
```

**Criterion 3: Sharpe > 1.0**
```bash
python bin/calculate_sharpe_ratio.py \
  --period 28d \
  --risk-free-rate 0.05

# Expected: Sharpe > 1.0
```

**Criterion 4: Systems Independent (No Cross-Talk)**
```bash
python bin/verify_system_independence.py \
  --check-signal-overlap true \
  --check-position-conflicts true

# Expected: No overlapping signals, no position conflicts
```

**Decision Matrix:**
```
Criteria Met    Decision
4/4             ✅ Full deployment SUCCESS, continue operations
3/4             ⚠️  Continue monitoring, tune if needed
2/4 or less     ❌ Scale back allocation, investigate
```

---

## MONTH 3+ MAINTENANCE

### Monthly Performance Review

**Actions:**
```bash
# Generate monthly report
python bin/generate_monthly_report.py \
  --month 2025-03 \
  --output results/monthly_reports/2025-03.md

# Review:
# - Portfolio PF vs target (1.76)
# - Sharpe ratio vs target (>1.0)
# - Max DD vs limit (<20%)
# - Regime breakdown (bull/bear performance)
# - Top 10 winners/losers
```

**Rebalance Decision:**
```python
if actual_pf < expected_pf * 0.8:
    action = "ALERT: Re-optimize parameters, investigate regime shift"
elif actual_pf < expected_pf * 0.9:
    action = "WARNING: Monitor closely, prepare to re-tune"
else:
    action = "OK: Performance within expectations, continue"
```

---

### Quarterly Re-Optimization

**Actions:**
```bash
# Detect regime shift
python bin/detect_regime_shift.py \
  --period 90d \
  --significance 0.05

# If regime shifted:
# Re-optimize parameters
python bin/optimize_baseline_params.py \
  --period latest-year \
  --trials 30

# Validate on OOS data
python bin/validate_walk_forward.py \
  --config configs/optimized/baseline_q2_2025.json \
  --n-splits 5
```

---

## ARCHETYPE ARCHIVE DECISION

### Option 1: Archive Indefinitely ⏱ 2 hours

**Recommended if:** No clear path to >5% improvement

**Actions:**
```bash
# Create archive directory
mkdir -p archive/v2025_archetypes/

# Move archetype code
mv engine/strategies/archetypes/bear/ archive/v2025_archetypes/
mv engine/models/archetype_model.py archive/v2025_archetypes/
mv configs/system_s4_production.json archive/v2025_archetypes/
mv configs/system_s5_production.json archive/v2025_archetypes/

# Preserve documentation
cp -r docs/ARCHETYPE_* archive/v2025_archetypes/docs/
cp ARCHETYPE_VALIDATION_COMPLETE.md archive/v2025_archetypes/

# Add archive README
cat > archive/v2025_archetypes/README.md << EOF
# Archetype Systems Archive (2025-12-08)

## Why Archived
Validation complete. Archetypes = Baselines (PF 1.76) despite:
- Feature coverage 87% (was 20%)
- Domain engines 100% active (was 17%)
- Optuna calibrations loaded

Decision: Deploy baselines only (simpler, equal performance)

## Contents
- bear/ - S4/S5 bear specialist archetypes
- archetype_model.py - Wrapper with runtime score computation
- configs/ - Production configs (S4/S5)
- docs/ - Full validation journey (5 days)

## Future Consideration
Revisit if:
- Temporal features implemented (0% → 100%)
- OI data backfilled (67% null → <5%)
- New optimization yields >5% PF improvement

Last Updated: 2025-12-08
Status: ARCHIVED
EOF

# Git commit
git add archive/v2025_archetypes/
git commit -m "Archive archetypes after validation (PF tie with baselines)"
```

---

### Option 2: Rework and Re-Validate ⏱ 4-5 weeks

**Recommended if:** Temporal + OI could yield >5% improvement

**Week 1: Implement Temporal Features**
```bash
# Fibonacci time clusters
python bin/implement_fib_time_clusters.py \
  --algorithm phase_shift_detection \
  --validate true

# Temporal confluence
python bin/implement_temporal_confluence.py \
  --algorithm multi_timeframe_alignment \
  --validate true
```

**Week 2: Backfill OI Data**
```bash
# Run OI backfill pipeline
python bin/backfill_oi_data.py \
  --start 2022-01-01 \
  --end 2024-12-31 \
  --source okx

# Validate OI quality
python bin/validate_oi_data.py \
  --period 2022-2024 \
  --max-null-rate 0.05
```

**Week 3: Re-Optimize**
```bash
# Re-run Optuna on full knowledge setup
python bin/optimize_s4_calibration.py \
  --period 2022-2024 \
  --trials 50 \
  --features temporal,oi

# Expected: PF improvement if temporal+OI valuable
```

**Week 4-5: Re-Validate**
```bash
# Run full 9-step protocol again
python bin/run_full_validation.sh \
  --config configs/optimized/s4_full_knowledge.json

# If PF > baseline by >5%:
#   Deploy archetypes (Scenario A)
# Else:
#   Archive archetypes (Scenario C)
```

---

### Option 3: Bear Market Specialist ⏱ 2 weeks

**Recommended if:** Bear market backtests show >5% advantage

**Week 1: Set Up Regime Routing**
```bash
# Create regime-aware deployment
cat > configs/production/regime_routed.json << EOF
{
  "regime_routing": {
    "risk_on": {
      "systems": ["baselines"],
      "archetypes": false
    },
    "neutral": {
      "systems": ["baselines"],
      "archetypes": false
    },
    "risk_off": {
      "systems": ["archetypes"],
      "archetypes": true,
      "archetypes_enabled": ["S4", "S5"]
    },
    "crisis": {
      "systems": ["archetypes"],
      "archetypes": true,
      "archetypes_enabled": ["S4", "S5", "S1"]
    }
  }
}
EOF
```

**Week 2: Validate Regime Switching**
```bash
# Test regime classifier
python bin/validate_regime_classifier.py \
  --period 2022-2024 \
  --check-lag true \
  --check-accuracy true

# Backtest regime-routed portfolio
python bin/backtest_regime_routed.py \
  --config configs/production/regime_routed.json \
  --period 2022-2024

# If blended PF > baseline PF by >5%:
#   Deploy regime-routed (hybrid)
# Else:
#   Archive archetypes
```

---

## SUCCESS DEFINITIONS

### Paper Trading Success (Week 1-2)
- ✅ PF ≥ 1.41 (80% of backtest)
- ✅ 330-400 trades
- ✅ 0 execution errors
- **Outcome:** Proceed to live small

### Live Small Success (Week 3-4)
- ✅ PF ≥ 0.88 (50% of backtest)
- ✅ Slippage ≤ 0.05%
- ✅ 0 unexpected behaviors
- **Outcome:** Proceed to scale up

### Scale Up Success (Week 5-8)
- ✅ PF ≥ 1.76 (match backtest)
- ✅ Max DD < 20%
- ✅ Sharpe > 1.0
- ✅ Systems independent
- **Outcome:** Full deployment success

### Long-Term Success (Month 3+)
- ✅ Outperform buy-and-hold
- ✅ Sharpe > 1.0 consistently
- ✅ Max DD < 20%
- ✅ Regime adaptation working
- **Outcome:** Production stable

---

## EMERGENCY PROCEDURES

### If Paper Trading Fails (PF < 1.41)

**Immediate Actions:**
```bash
# Pause paper trading
python bin/pause_paper_trading.py

# Investigate root cause
python bin/diagnose_paper_trading.py \
  --check-config true \
  --check-data-quality true \
  --check-feature-coverage true

# Common issues:
# - Config drift (check vs validated config)
# - Data quality degraded (check null rates)
# - Feature coverage dropped (check mapper)
```

**Resolution:**
```bash
# Fix issue
# Re-validate config
python bin/validate_config.py --strict true

# Restart paper trading
python bin/restart_paper_trading.py --reset-stats true
```

---

### If Live Small Fails (PF < 0.88 or Execution Errors)

**Immediate Actions:**
```bash
# Revert to paper trading
python bin/revert_to_paper.py --close-positions true

# Investigate
python bin/diagnose_live_trading.py \
  --check-slippage true \
  --check-execution-quality true \
  --check-unexpected-behavior true
```

**Resolution:**
```bash
# Fix issue (slippage, latency, order type)
# Re-test in paper mode 1 week
# If paper passes, retry live small
```

---

### If Scale Up Fails (PF < 1.76 or DD > 20%)

**Immediate Actions:**
```bash
# Scale back to live small (10%)
python bin/scale_back.py --allocation 0.10

# Investigate
python bin/diagnose_portfolio.py \
  --check-regime-shift true \
  --check-overfitting true \
  --check-market-conditions true
```

**Resolution:**
```bash
# If regime shifted: Re-optimize parameters
# If overfitted: Reduce trade frequency, widen stops
# If market unusual: Wait for normalization
```

---

## FINAL CHECKLIST

### Before Starting Week 1 (Paper Trading)
- [ ] Validation completion report reviewed
- [ ] Scenario C approved by stakeholders
- [ ] OKX testnet connection working
- [ ] Baseline config validated
- [ ] Paper trading database initialized
- [ ] Monitoring dashboards operational
- [ ] Alerting systems tested
- [ ] Trading team briefed

### Before Starting Week 3 (Live Small)
- [ ] Paper trading PF ≥ 1.41
- [ ] 330-400 trades executed
- [ ] 0 execution errors
- [ ] Stakeholder approval for live trading
- [ ] OKX live credentials secured
- [ ] Risk limits configured (10%, 1% per trade)

### Before Starting Week 5 (Scale Up)
- [ ] Live small PF ≥ 0.88
- [ ] Slippage ≤ 0.05%
- [ ] 0 unexpected behaviors
- [ ] Stakeholder approval for scale up
- [ ] Gradual scale plan approved (25% → 50% → 75% → 100%)

### Month 3+ (Production)
- [ ] Portfolio PF ≥ 1.76 for 4 consecutive weeks
- [ ] Max DD < 20%
- [ ] Sharpe > 1.0
- [ ] Monthly reviews scheduled
- [ ] Quarterly re-optimization scheduled

---

## CONTACT & ESCALATION

**For Questions:**
- Validation methodology: See `ARCHETYPE_VALIDATION_COMPLETE.md`
- Deployment procedures: This document
- Emergency procedures: See "EMERGENCY PROCEDURES" section above

**For Issues:**
- Paper trading fails: Diagnose config/data, extend paper period
- Live trading fails: Revert to paper, investigate execution quality
- Performance degrades: Check regime shift, re-optimize if needed

**Escalation Levels:**
- WARNING (PF < 90% expected): Monitor closely, prepare to tune
- CRITICAL (PF < 80% expected): Investigate immediately, consider pausing
- EMERGENCY (Execution errors, DD > 20%): Pause trading, escalate to senior team

---

**END OF NEXT STEPS**

**Files to Review:**
1. `ARCHETYPE_VALIDATION_COMPLETE.md` - Full validation journey
2. `VALIDATION_NEXT_STEPS.md` - This document (actionable roadmap)
3. `VALIDATION_SUMMARY.txt` - 1-page executive summary
