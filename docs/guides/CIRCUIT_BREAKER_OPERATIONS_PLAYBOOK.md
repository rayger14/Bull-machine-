# Circuit Breaker Operations Playbook

## Emergency Contact Information

**Primary On-Call**: [PHONE NUMBER]
**Secondary On-Call**: [PHONE NUMBER]
**CEO**: [PHONE NUMBER]
**Slack Channel**: #trading-alerts
**Email**: trading-team@company.com

---

## Quick Reference: What To Do When You Get Paged

### Tier 1 Alert (CRITICAL - SMS + Phone Call)

**Response Time SLA**: <5 minutes

**Steps**:

1. **Acknowledge Alert** (text "ACK" to alert system)
2. **Log into Dashboard** (https://trading-dashboard.company.com)
3. **Review Trigger**:
   - What triggered the halt?
   - Daily loss >5%? → Check market conditions
   - Flash crash? → Verify exchange data
   - System failure? → Check logs
4. **Assess Portfolio**:
   - Current equity
   - Open positions
   - Unrealized PnL
5. **Take Action**:
   - If legitimate halt → Follow recovery procedure
   - If false positive → Document and tune thresholds
6. **Document Incident** (fill out incident report template)
7. **Notify Team** (post to #trading-alerts)

**DO NOT**:
- Immediately force resume without analysis
- Close positions in panic (unless flash crash)
- Ignore the alert

---

### Tier 2 Alert (WARNING - Slack + Email)

**Response Time SLA**: <15 minutes

**Steps**:

1. **Acknowledge Alert** (react with 👀 in Slack)
2. **Review Metrics**:
   - Is condition improving or worsening?
   - How long has soft halt been active?
3. **Monitor**:
   - Set up real-time monitoring dashboard
   - Check every 15 minutes for escalation
4. **Decision** (after 2 hours):
   - If improving → Resume to 75% risk
   - If stable → Keep monitoring
   - If worsening → Escalate to Tier 1
5. **Document** (post update to #trading-alerts)

---

### Tier 3 Alert (INFO - Email Only)

**Response Time SLA**: <1 hour

**Steps**:

1. **Review Warning**
2. **Log for Post-Mortem**
3. **Monitor** (no immediate action needed)
4. **If Persists >24h** → Escalate to Tier 2

---

## Incident Response Procedures

### Tier 1 Instant Halt Response

**Timeline**: 1-2 hours to resume

#### Phase 1: Assessment (0-10 minutes)

```
[ ] Log into dashboard
[ ] Review trigger condition
[ ] Check portfolio state (positions, PnL, equity)
[ ] Verify market data is valid
[ ] Check recent trade history
[ ] Review system logs for errors
```

#### Phase 2: Root Cause Analysis (10-30 minutes)

**Question Checklist**:

- Was the halt correct?
- What caused the trigger?
- Is this a market event or system failure?
- Are there any code bugs?
- Is data corrupted?

**Common Triggers & Diagnosis**:

| Trigger | Root Cause | Action |
|---------|------------|--------|
| `daily_loss_5pct` | Market moved against us | Verify market data, assess if normal volatility or unusual event |
| `flash_crash_detected` | Rapid price move | Verify exchange data, check if market-wide or single asset |
| `fill_rate_below_85pct` | Exchange issues | Check exchange status, API latency |
| `metadata_integrity_failure` | Feature store corruption | Check feature store logs, verify data pipeline |
| `archetype_failure_cluster_8` | Code regression or market regime change | Review recent code changes, check if regime shifted |

#### Phase 3: Documentation (30-45 minutes)

**Fill Out Incident Report**:

```
Incident ID: TIER1-[DATE]-[001]
Date/Time: [YYYY-MM-DD HH:MM:SS UTC]
Trigger: [trigger_name]
Tier: 1 (Instant Halt)

Timeline:
- [HH:MM:SS] - Halt triggered
- [HH:MM:SS] - Alert acknowledged
- [HH:MM:SS] - Root cause identified
- [HH:MM:SS] - Recovery approved

Root Cause:
[Detailed explanation of what caused the halt]

Was Halt Correct? [YES/NO]
False Positive? [YES/NO]

Portfolio State at Halt:
- Open Positions: [N]
- Total Exposure: $[X]
- Unrealized PnL: $[X]
- Current Equity: $[X]

Actions Taken:
[ ] Kept positions open
[ ] Closed positions
[ ] Reduced position sizes
[ ] Updated thresholds
[ ] Fixed code bug
[ ] Other: [describe]

Lessons Learned:
[What can we improve?]

Approval to Resume:
- Primary: [Name, Signature]
- CEO: [Name, Signature]
```

#### Phase 4: Recovery Decision (45-60 minutes)

**Recovery Checklist**:

```
[ ] Root cause identified and resolved
[ ] Market data validated as correct
[ ] System health checks passed:
    [ ] All archetypes generating signals
    [ ] Metadata integrity verified
    [ ] Regime detection functioning
    [ ] Feature store operational
[ ] Manual approval obtained:
    [ ] Primary on-call: [Name]
    [ ] CEO: [Name]
[ ] Recovery plan documented
[ ] Team notified
```

**Decision Tree**:

```
Was halt due to market conditions?
├─ YES → Safe to resume with gradual ramp
└─ NO → Was it a system failure?
    ├─ YES → Fix system first, then resume
    └─ NO → False positive?
        ├─ YES → Tune thresholds, resume
        └─ NO → Investigate further
```

#### Phase 5: Gradual Resume (60 minutes - 24 hours)

**Protocol**:

```python
# Hour 0-6: 25% risk
circuit_breaker.trading_enabled = True
circuit_breaker.position_size_multiplier = 0.25
logger.info("Trading resumed at 25% risk")

# Hour 6-12: 50% risk
# (manually increase after 6 hours)
circuit_breaker.position_size_multiplier = 0.50

# Hour 12-18: 75% risk
circuit_breaker.position_size_multiplier = 0.75

# Hour 18-24: 100% risk (full resume)
circuit_breaker.position_size_multiplier = 1.00
```

**Monitoring During Resume**:

- Check every 15 minutes for first 2 hours
- Check every hour for next 6 hours
- Check every 4 hours for next 16 hours
- Return to normal monitoring after 24 hours

---

### Tier 2 Soft Halt Response

**Timeline**: 2-4 hours to resume or escalate

#### Phase 1: Assessment (0-15 minutes)

```
[ ] Review trigger metric
[ ] Check current value vs threshold
[ ] Assess trend (improving or worsening?)
[ ] Verify position size multiplier applied correctly
[ ] Check recent trade performance
```

#### Phase 2: Monitoring (15 minutes - 2 hours)

**Set Up Dashboard**:

- Real-time metric display (refresh every 1 minute)
- Equity curve
- Drawdown chart
- Recent trades table
- Alert log

**Check Every 15 Minutes**:

```
Time: [HH:MM]
Metric: [drawdown/sharpe/fill_rate/etc]
Current Value: [X]
Threshold: [Y]
Trend: [improving/stable/worsening]
Action: [continue monitoring/escalate/resume]
```

#### Phase 3: Decision (After 2 Hours)

**If Improving**:

```
[ ] Metric returned to normal range
[ ] No new warnings triggered
[ ] Resume to 75% risk:
    circuit_breaker.position_size_multiplier = 0.75
[ ] Monitor for 4 more hours
[ ] If stable, resume to 100%
```

**If Stable**:

```
[ ] Metric unchanged
[ ] Continue monitoring
[ ] Keep at current risk level
[ ] Re-assess in 2 hours
```

**If Worsening**:

```
[ ] Metric deteriorating
[ ] Escalate to Tier 1:
    circuit_breaker.tier1_instant_halt("tier2_escalation")
[ ] Follow Tier 1 response procedure
```

---

## Manual Controls

### Emergency Stop Button

**When to Use**:

- Observe unexpected system behavior
- Detect serious code bug
- Need to halt immediately for any reason

**How to Use**:

1. Navigate to dashboard: https://trading-dashboard.company.com
2. Click "EMERGENCY STOP" button (red, top right)
3. Authenticate (username + password + 2FA)
4. Select options:
   - [ ] Close all positions? (only if necessary)
5. Enter reason: "Why are you halting?"
6. Click "CONFIRM HALT"
7. Verify halt executed (trading status = HALTED)
8. Post to #trading-alerts: "Manual halt triggered: [reason]"

**API Version**:

```bash
curl -X POST https://api.trading.company.com/emergency-stop \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "user": "your_username",
    "password": "your_password",
    "reason": "Observed unexpected behavior",
    "close_positions": false
  }'
```

---

### Force Resume

**When to Use**:

- After Tier 1 halt resolved
- Override automatic halt (use with extreme caution)

**Requirements**:

- CEO approval (digital signature)
- Root cause documented
- Recovery plan in place

**How to Use**:

1. Complete incident report
2. Obtain CEO approval (email or Slack DM)
3. Navigate to dashboard
4. Click "FORCE RESUME"
5. Enter:
   - User: [your_username]
   - Justification: [why resuming]
   - CEO Approval Signature: [signature_from_ceo]
6. Click "CONFIRM RESUME"
7. Verify resume executed (trading status = LIVE)
8. Post to #trading-alerts: "Trading resumed by [user]: [justification]"

**API Version**:

```bash
curl -X POST https://api.trading.company.com/force-resume \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "user": "your_username",
    "justification": "Root cause identified and fixed. Market data validated.",
    "approval_signature": "CEO_SIGNATURE_HERE"
  }'
```

---

### Risk Dial

**When to Use**:

- Reduce risk during uncertain market conditions
- Gradual ramp-up after resume
- Conservative approach during high volatility

**How to Use**:

1. Navigate to dashboard
2. Select "RISK DIAL" section
3. Choose level:
   - ○ 25% (very conservative)
   - ○ 50% (conservative)
   - ○ 75% (moderate)
   - ● 100% (full risk)
4. Set duration: [24] hours (auto-reset to 100%)
5. Click "APPLY"
6. Verify multiplier updated

**API Version**:

```bash
curl -X POST https://api.trading.company.com/risk-dial \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "level": "50%",
    "duration_hours": 24
  }'
```

**Common Use Cases**:

- **50% during high volatility**: Reduce risk when VIX >30
- **25% after major loss**: Conservative approach after drawdown
- **75% during gradual resume**: Step up from 50% to 100%

---

## Pre-Shift Checklist

**Before each trading session**:

```
[ ] Check circuit breaker status (get_status API)
[ ] Verify alert channels working (send test alert)
[ ] Review recent events (past 24 hours)
[ ] Check on-call rotation (who's on duty?)
[ ] Verify dashboard accessible
[ ] Review portfolio state (equity, positions, drawdown)
[ ] Check market conditions (volatility, news)
[ ] Test emergency stop button (in staging)
[ ] Review team escalation path
```

---

## Post-Shift Checklist

**At end of trading session**:

```
[ ] Review circuit breaker events
[ ] Document any incidents
[ ] Update thresholds if needed
[ ] Hand off to next shift (if 24/7)
[ ] Post summary to #trading-alerts
```

---

## Common Scenarios & Responses

### Scenario 1: Market Flash Crash

**Symptoms**:
- Alert: "flash_crash_detected"
- Price moved >10% in <5 minutes

**Response**:
1. Verify exchange data (not a data error)
2. Check if market-wide or isolated to one asset
3. Review positions (likely closed at market automatically)
4. Assess damage (PnL impact)
5. Wait for market to stabilize (30-60 minutes)
6. Validate data feed
7. Resume trading with 50% risk

**Timeline**: 1-2 hours

---

### Scenario 2: Exchange API Outage

**Symptoms**:
- Alert: "exchange_outage"
- Cannot place/cancel orders
- High API latency

**Response**:
1. Check exchange status page
2. Test API connectivity
3. Verify WebSocket connection
4. If exchange-wide: wait for recovery
5. If local issue: restart services
6. Resume once connectivity restored
7. Verify all positions managed correctly

**Timeline**: 15 minutes - 2 hours (depends on exchange)

---

### Scenario 3: Metadata Integrity Failure

**Symptoms**:
- Alert: "metadata_integrity_failure"
- Feature store corruption
- All boost/veto counts = 0

**Response**:
1. Check feature store logs
2. Identify which features missing
3. Verify data pipeline status
4. Re-run feature computation if needed
5. Validate metadata restored
6. Test archetype signal generation
7. Resume trading

**Timeline**: 30 minutes - 2 hours

---

### Scenario 4: High Drawdown

**Symptoms**:
- Alert: "drawdown_20pct" (Tier 2)
- Portfolio down 20% from peak

**Response**:
1. Review recent trades (what went wrong?)
2. Check if market conditions changed
3. Verify regime classification correct
4. Assess if strategy still valid
5. Options:
   - Keep soft halt (50% risk) and monitor
   - Reduce to 25% risk if very uncertain
   - Full halt if strategy broken
6. Re-assess thresholds (were they appropriate?)

**Timeline**: Monitor for 2-4 hours before decision

---

## Escalation Decision Tree

```
Alert Received
     │
     ▼
Tier 1 (Critical)?
     ├─ YES → Page primary on-call immediately
     │         │
     │         ▼
     │    Acknowledged within 5 minutes?
     │         ├─ YES → Primary handles
     │         └─ NO → Page secondary on-call
     │                  │
     │                  ▼
     │            Acknowledged within 10 minutes?
     │                  ├─ YES → Secondary handles
     │                  └─ NO → Page CEO
     │
     └─ NO → Tier 2 or 3?
               │
               ▼
          Send to Slack + Email
          (no page required)
```

---

## Monthly Drills

**Conduct on first Monday of each month**:

### Drill 1: Simulate Tier 1 Halt

**Procedure**:

1. Announce drill in #trading-alerts
2. Manually trigger halt:
   ```python
   circuit_breaker.manual_emergency_stop(
       user="drill_coordinator",
       reason="Monthly drill - Tier 1 simulation",
       close_positions=False
   )
   ```
3. Primary on-call responds (measure response time)
4. Walk through recovery checklist
5. Practice manual resume
6. Document drill results

**Target Metrics**:
- Acknowledgement time: <5 minutes
- Root cause analysis: <15 minutes
- Recovery approval: <30 minutes
- Total time to resume: <1 hour

### Drill 2: Test All Alert Channels

**Procedure**:

1. Send test alert to SMS
2. Send test alert to phone (automated call)
3. Send test alert to Slack
4. Send test alert to email
5. Verify all channels received

**Pass Criteria**: All channels deliver within 2 minutes

### Drill 3: Emergency Stop Button

**Procedure**:

1. Log into dashboard
2. Click emergency stop
3. Verify authentication required
4. Complete halt
5. Verify audit log entry
6. Practice resume

---

## Troubleshooting

### Issue: Alert Not Received

**Diagnosis**:
- Check alert callback registered
- Verify notification service (Twilio, Slack, email) operational
- Check logs for errors

**Fix**:
```python
# Re-register callbacks
circuit_breaker.register_alert_callback("emergency", send_emergency_alert)
circuit_breaker.register_alert_callback("warning", send_warning_alert)
circuit_breaker.register_alert_callback("info", send_info_alert)
```

---

### Issue: False Positives

**Diagnosis**:
- Thresholds too tight for strategy
- Market volatility higher than expected

**Fix**:
```python
# Adjust thresholds
custom_thresholds = CircuitBreakerThresholds(
    daily_loss_pct=0.07,  # More lenient
    drawdown_tier1=0.30
)
circuit_breaker.thresholds = custom_thresholds
```

---

### Issue: Trading Not Resuming

**Diagnosis**:
- Check `circuit_breaker.trading_enabled` flag
- Verify no active halt

**Fix**:
```python
# Force enable trading
circuit_breaker.trading_enabled = True
circuit_breaker.position_size_multiplier = 1.0
```

---

## Audit & Compliance

**Event Retention**: 90 days minimum

**Audit Log Location**: `logs/circuit_breaker/audit_log_YYYYMMDD.jsonl`

**Event Log Location**: `logs/circuit_breaker/circuit_breaker_events_YYYYMMDD.jsonl`

**Required Reports**:

- Weekly: Event summary (Tier 1, 2, 3 counts)
- Monthly: Incident reports for all Tier 1 halts
- Quarterly: Threshold tuning recommendations
- Annual: Full system audit

---

## Team Training

**All team members must complete**:

1. Read this playbook
2. Complete circuit breaker training module
3. Participate in monthly drill
4. Shadow on-call for 1 week
5. Pass certification quiz

**Certification Quiz** (must score 100%):

1. What is the response time SLA for Tier 1 alerts?
2. When should you close all positions?
3. What is required for force resume?
4. How does the risk dial work?
5. What is the gradual resume protocol?

---

## Document Version

**Version**: 1.0
**Last Updated**: 2025-12-17
**Next Review**: 2026-01-17 (monthly)
**Owner**: Risk Management Team
**Approved By**: CEO, Head of Trading

---

## Emergency Contacts (Fill In)

| Role | Name | Phone | Email | Slack |
|------|------|-------|-------|-------|
| Primary On-Call | [NAME] | [PHONE] | [EMAIL] | @[username] |
| Secondary On-Call | [NAME] | [PHONE] | [EMAIL] | @[username] |
| Head of Trading | [NAME] | [PHONE] | [EMAIL] | @[username] |
| CEO | [NAME] | [PHONE] | [EMAIL] | @[username] |
| Engineering Lead | [NAME] | [PHONE] | [EMAIL] | @[username] |

---

**KEEP THIS PLAYBOOK ACCESSIBLE 24/7**

Print a copy and keep it next to your workstation.
Bookmark the digital version in your browser.
Review monthly to stay current.
