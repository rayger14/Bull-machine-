# Operational Playbook: Dual-System Trading

**Version:** 1.0.0
**Date:** 2025-12-03
**Status:** ACTIVE PROCEDURES
**Audience:** Trading operators, on-call engineers, monitoring team

---

## Executive Summary

This playbook defines daily operations for running two independent trading systems (B0 + Archetypes) in parallel.

**Purpose:** Ensure smooth operations, catch issues early, respond to incidents quickly.

**Key Principle:** Automate monitoring, human reviews for decisions, clear escalation paths.

---

## Daily Operations Checklist

### Morning Review (Start of Trading Day)

**Time:** 9:00 AM UTC (before market opens)

**Duration:** 15-20 minutes

**Owner:** Operator on duty

#### 1. System Health Check

```
[ ] Check dashboard: All systems online?
    - B0:  🟢 Online / 🔴 Offline
    - S4:  🟢 Online / 🔴 Offline
    - S5:  🟢 Online / 🔴 Offline
    - S1:  🟢 Online / 🔴 Offline (if enabled)

[ ] Check API connectivity
    - Exchange API: 🟢 Connected / 🔴 Disconnected
    - Data feed: 🟢 Live / 🔴 Stale
    - Last heartbeat: < 5 minutes ago

[ ] Check regime classification
    - Current regime: risk_on / neutral / risk_off / crisis
    - Last update: < 1 hour ago
    - Confidence: > 60%

[ ] Check open positions
    - B0 positions: X open (details: symbol, entry, PnL)
    - S4 positions: X open
    - S5 positions: X open
    - Total exposure: X% of capital (should be < 25%)
```

**Actions:**
- If any system offline: Escalate immediately (see Incident Response)
- If API disconnected: Check exchange status, reconnect, alert team
- If regime stale: Restart classifier, investigate
- If exposure > 25%: Review positions, consider reducing

#### 2. Performance Review (Last 24 Hours)

```
[ ] Check overnight performance
    - Portfolio PnL (last 24h): $XXX
    - Per-system PnL:
      - B0:  $XXX
      - S4:  $XXX
      - S5:  $XXX
    - Any stop losses hit? Yes / No (details if yes)
    - Any take profits hit? Yes / No

[ ] Check drawdown
    - Current portfolio DD: X%
    - Max DD (last 24h): X%
    - Alert if DD > 15%

[ ] Check signals fired
    - B0 signals (last 24h): X
    - S4 signals (last 24h): X
    - S5 signals (last 24h): X
    - Any unexpected signals? Yes / No
```

**Actions:**
- If PnL negative > -3%: Review trades, investigate losses
- If DD > 15%: Alert team, review risk management
- If unexpected signals: Investigate signal logic, check for bugs
- If no signals for 7 days: Normal (low frequency), verify system still active

#### 3. Alerts and Notifications Review

```
[ ] Check alert queue (Slack, email, dashboard)
    - Unresolved alerts: X
    - Critical alerts: X
    - Warnings: X

[ ] Acknowledge and triage
    - Critical: Address immediately
    - Warnings: Review and log
    - Info: Note and dismiss

[ ] Check log files for errors
    - Backend errors (last 24h): X
    - Execution errors: X
    - Data feed errors: X
```

**Actions:**
- Critical alerts: Escalate immediately
- Recurring warnings: Investigate root cause
- Log errors: Review, fix if needed

---

### Midday Check (Trading Session)

**Time:** 12:00 PM UTC (mid-session)

**Duration:** 5-10 minutes

**Owner:** Operator on duty

#### Quick Status Check

```
[ ] All systems still online?
[ ] Any new signals fired?
[ ] Any positions opened since morning?
[ ] Any alerts since morning check?
[ ] Exchange API still connected?
```

**Actions:**
- If new positions: Verify entry price, stop loss set correctly
- If alerts: Triage and respond
- If system offline: Restart, escalate if restart fails

---

### End of Day Review (Market Close)

**Time:** 5:00 PM UTC (after market closes)

**Duration:** 20-30 minutes

**Owner:** Operator on duty

#### 1. Daily Performance Summary

```
[ ] Calculate daily metrics
    - Portfolio PnL (today): $XXX
    - Portfolio PF (rolling 30d): X.XX
    - Portfolio WR (rolling 30d): XX%
    - Trades executed today: X

[ ] Per-system performance
    - B0:  PnL today: $XXX, Trades: X
    - S4:  PnL today: $XXX, Trades: X
    - S5:  PnL today: $XXX, Trades: X

[ ] Compare to targets
    - Is daily PnL within expectations? Yes / No
    - Is PF on track (>2.0)? Yes / No
    - Any anomalies? Yes / No (details if yes)
```

**Actions:**
- If PnL significantly off target: Investigate (market regime? Config change?)
- If anomalies: Document, escalate if needed
- Update daily log (see Logging section)

#### 2. Position Review

```
[ ] Review all open positions
    - Position age: How many hours/days open?
    - Current PnL: Winning or losing?
    - Stop loss distance: Still appropriate?
    - Take profit distance: On track?

[ ] Check for stale positions (open > 72 hours)
    - If stale: Consider manual exit or adjust stops

[ ] Verify stop losses set correctly
    - All positions have stops? Yes / No
    - Stop prices reasonable? Yes / No
```

**Actions:**
- If position stale (>3 days): Review, consider closing manually
- If stop loss missing: Set immediately (critical)
- If position deeply underwater: Review, may need manual intervention

#### 3. Risk Assessment

```
[ ] Daily risk metrics
    - Max exposure today: X% (should be < 25%)
    - Max drawdown today: X% (should be < 15%)
    - Stop losses hit today: X
    - Largest single loss: $XXX

[ ] Risk limit violations
    - Any exposure > 25%? Yes / No
    - Any DD > 15%? Yes / No
    - Any single loss > 5% of capital? Yes / No
```

**Actions:**
- If violations: Document, escalate, adjust risk limits if needed
- If multiple stop losses: Review strategy, market conditions

#### 4. System Health

```
[ ] Check system uptime
    - B0 uptime: XX hours (target: 24/24)
    - S4 uptime: XX hours
    - S5 uptime: XX hours

[ ] Check resource usage
    - CPU: XX% (should be < 70%)
    - Memory: XX% (should be < 80%)
    - Disk: XX% (should be < 90%)

[ ] Check logs for anomalies
    - Errors logged today: X
    - Warnings logged today: X
```

**Actions:**
- If uptime < 23 hours: Investigate outages
- If resource usage high: Investigate, may need optimization
- If many errors: Review logs, fix bugs

---

## How to Handle Each System's Signals

### System B0 (Baseline-Conservative)

**Signal Type:** Long entry when 30-day drawdown < -15%

#### When B0 Fires:

```
1. Verify Signal
   [ ] Current drawdown: -XX% (should be < -15%)
   [ ] Close price: $XXXXX
   [ ] ATR: $XXX
   [ ] Market conditions: Normal / Volatile / Flash crash

2. Calculate Position
   [ ] Allocation: 15% of B0 capital
   [ ] Entry price: Current close
   [ ] Stop loss: Entry - (2.5 * ATR) = $XXXXX
   [ ] Take profit: Entry * 1.08 = $XXXXX

3. Execute Order
   [ ] Place market buy order
   [ ] Set stop loss order
   [ ] Set take profit order
   [ ] Log entry in trade journal

4. Monitor Position
   [ ] Check every 4 hours (1H timeframe)
   [ ] Update stop loss if needed (trailing)
   [ ] Exit at TP or SL

5. Document
   [ ] Log signal timestamp
   [ ] Log entry price, stop, TP
   [ ] Log market conditions
   [ ] Log regime (for analysis)
```

**Expected Frequency:** 0-2 signals per week (low frequency)

**Typical Duration:** 2-7 days per trade

**Common Issues:**
- False signal (drawdown not actually -15%): Check calculation
- Gap in data: Verify close price is correct
- Stop loss too tight: Verify ATR calculation

### System S4 (Funding Divergence)

**Signal Type:** Long entry when funding divergence + regime = risk_off/neutral

#### When S4 Fires:

```
1. Verify Signal
   [ ] Fusion score: X.XX (should be > threshold, ~0.6)
   [ ] Regime: risk_off / neutral (should not be risk_on)
   [ ] Funding rate: XX% (should be negative, < -0.01%)
   [ ] Price resilience: OK
   [ ] Liquidity score: X.XX (should be > 0.5)

2. Calculate Position
   [ ] Allocation: 10% of S4 capital
   [ ] Entry price: Current close
   [ ] Stop loss: Entry - (ATR * mult) = $XXXXX (mult varies)
   [ ] Take profit: Dynamic (trailing or fixed)

3. Execute Order
   [ ] Place market buy order
   [ ] Set stop loss order
   [ ] Log entry in trade journal

4. Monitor Position
   [ ] Check every 4 hours
   [ ] Monitor funding rate (exit if turns positive?)
   [ ] Monitor fusion score (exit if drops below threshold?)
   [ ] Update trailing stop if configured

5. Document
   [ ] Log fusion score components
   [ ] Log regime and confidence
   [ ] Log funding rate at entry
   [ ] Log market conditions
```

**Expected Frequency:** 1-2 signals per week (when risk_off/neutral)

**Typical Duration:** 1-5 days per trade

**Common Issues:**
- Regime blocks signal: Check regime classifier (should not be stuck on 'neutral')
- Fusion score calculation fails: Check runtime enrichment (features present?)
- Funding data missing: System should fallback to 0.0, but signal weaker

### System S5 (Long Squeeze)

**Signal Type:** Long entry when positive funding + overleveraged longs + regime = risk_on

#### When S5 Fires:

```
1. Verify Signal
   [ ] Fusion score: X.XX (should be > threshold, ~0.5-0.6)
   [ ] Regime: risk_on / neutral (should not be risk_off)
   [ ] Funding rate: XX% (should be positive, > +0.01%)
   [ ] RSI: XX (should be overbought, > 70)
   [ ] Liquidity score: X.XX

2. Calculate Position
   [ ] Allocation: 10% of S5 capital
   [ ] Entry price: Current close
   [ ] Stop loss: Entry - (ATR * mult) = $XXXXX
   [ ] Take profit: Dynamic or fixed

3. Execute Order
   [ ] Place market buy order
   [ ] Set stop loss order
   [ ] Log entry in trade journal

4. Monitor Position
   [ ] Check every 4 hours
   [ ] Monitor funding rate (exit if turns negative?)
   [ ] Monitor RSI (exit if drops back to normal?)
   [ ] Update trailing stop if configured

5. Document
   [ ] Log fusion score components
   [ ] Log regime and confidence
   [ ] Log funding rate at entry
   [ ] Log RSI at entry
```

**Expected Frequency:** 0-1 signals per week (regime-dependent)

**Typical Duration:** 1-3 days per trade

**Common Issues:**
- Regime blocks signal: Normal in bear markets (S5 should abstain)
- OI data missing: System uses fallback (75% signal strength)
- RSI not overbought: Verify calculation

---

## Conflict Resolution (Multiple Systems Signal)

### Scenario 1: B0 and S4 Both Signal

**Action: Take BOTH positions** (independent strategies)

```
1. Verify both signals valid
   [ ] B0: Drawdown < -15%
   [ ] S4: Fusion > threshold, regime OK

2. Calculate combined exposure
   [ ] B0 position: 15% of B0 allocation = $XXX
   [ ] S4 position: 10% of S4 allocation = $XXX
   [ ] Total exposure: (B0 + S4) / Total capital = X%

3. Check exposure limit
   [ ] If total exposure > 25%: Reduce both proportionally
   [ ] Else: Take both positions at full size

4. Execute orders
   [ ] Place both orders
   [ ] Set stops for both
   [ ] Log both entries

5. Monitor independently
   [ ] B0 exits when TP/SL hit
   [ ] S4 exits when TP/SL hit
   [ ] Systems do not affect each other
```

### Scenario 2: S4, S5, and S1 All Signal (Rare)

**Action: Prioritize by regime fit**

```
1. Check current regime
   [ ] Regime: risk_on / neutral / risk_off / crisis

2. Prioritize archetypes
   - If crisis: S1 > S4 > S5
   - If risk_off: S4 > S1 > S5
   - If neutral: S4 = S5 > S1
   - If risk_on: S5 > S4 > S1

3. Take top 2 signals
   [ ] Primary: 60% of archetype allocation
   [ ] Secondary: 40% of archetype allocation
   [ ] Skip tertiary (avoid over-concentration)

4. Execute
   [ ] Place 2 orders
   [ ] Set stops for both
   [ ] Log entries
```

### Scenario 3: B0 Signals, But S4 Already Has Position

**Action: Take B0 position** (independent systems)

```
1. Check total exposure
   [ ] S4 open position: $XXX (X% of capital)
   [ ] B0 proposed position: $XXX (X% of capital)
   [ ] Total: X% of capital

2. If total > 25%:
   [ ] Skip B0 signal (exposure limit)
   [ ] Log: "B0 signal skipped due to exposure limit"

3. Else:
   [ ] Take B0 position
   [ ] Monitor both positions independently
```

---

## Emergency Procedures

### Kill Switch Activation

**Trigger:** Portfolio DD > 25% OR catastrophic event (flash crash, exchange hack, etc.)

**Action:**

```
1. IMMEDIATE: Pause all systems
   [ ] Stop B0 (no new entries)
   [ ] Stop S4 (no new entries)
   [ ] Stop S5 (no new entries)
   [ ] Stop S1 (if enabled)

2. Close all open positions (within 5 minutes)
   [ ] Place market sell orders for all positions
   [ ] Cancel all pending orders (stops, TPs)
   [ ] Confirm all positions closed

3. Alert team
   [ ] Send critical alert (Slack, SMS, email)
   [ ] Subject: "KILL SWITCH ACTIVATED - Portfolio DD > 25%"
   [ ] Include: Current DD, total loss, positions closed

4. Document incident
   [ ] Log timestamp
   [ ] Log trigger (DD exceeded, flash crash, etc.)
   [ ] Log positions closed (entry, exit, loss)
   [ ] Log market conditions

5. Post-mortem
   [ ] Schedule emergency meeting (within 1 hour)
   [ ] Review what happened
   [ ] Determine root cause
   [ ] Decide: Resume trading or pause indefinitely
```

**Recovery:**
- Do NOT resume trading until full post-mortem complete
- Do NOT resume trading until root cause fixed
- Resume with reduced capital (50%) until confidence restored

### API Disconnection

**Trigger:** Exchange API not responding for > 1 minute

**Action:**

```
1. Immediate check
   [ ] Exchange status page: Up / Down
   [ ] Internet connectivity: OK / Failed
   [ ] API keys: Valid / Expired

2. Attempt reconnection
   [ ] Restart API client
   [ ] Re-authenticate
   [ ] Test connection with ping request

3. If reconnection fails (after 3 attempts):
   [ ] Alert team (Slack, SMS)
   [ ] Check positions manually (via exchange UI)
   [ ] If positions open: Monitor manually until API restored

4. If prolonged (> 10 minutes):
   [ ] Consider moving to backup exchange
   [ ] Or: Close positions manually via UI
   [ ] Alert team for manual intervention
```

**Prevention:**
- Keep backup API keys ready
- Test backup exchange connectivity weekly
- Monitor API rate limits

### Data Feed Stale

**Trigger:** Last data update > 5 minutes ago

**Action:**

```
1. Check data feed status
   [ ] Data provider: Up / Down
   [ ] Last update: X minutes ago
   [ ] Internet connectivity: OK / Failed

2. Restart data feed
   [ ] Reconnect to data source
   [ ] Verify latest candle received
   [ ] Check timestamp (should be < 1 minute ago)

3. If restart fails:
   [ ] Switch to backup data source
   [ ] Alert team
   [ ] Pause trading if data unreliable

4. Verify historical data intact
   [ ] Check feature store (no gaps)
   [ ] Check regime classification (still working)
```

### System Crash

**Trigger:** System process terminated unexpectedly

**Action:**

```
1. Identify crashed system
   [ ] B0 / S4 / S5 / S1 / All

2. Check logs
   [ ] Last error message
   [ ] Stack trace
   [ ] Resource usage before crash (OOM? CPU spike?)

3. Restart system
   [ ] Reload config
   [ ] Re-initialize models
   [ ] Reconnect APIs
   [ ] Verify online

4. Check open positions
   [ ] Are positions still valid?
   [ ] Are stops still set?
   [ ] If stops missing: Set immediately

5. If crash persists:
   [ ] Disable crashed system
   [ ] Alert engineering team
   [ ] Keep other systems running (if independent)
```

---

## Performance Review Schedule

### Daily Review (Every Day)

**Owner:** Operator on duty

**Output:** Daily log entry

**Content:**
- Daily PnL (portfolio and per-system)
- Trades executed
- Alerts and incidents
- Notes and observations

**Template:**
```
Date: YYYY-MM-DD
Operator: Name

Performance:
- Portfolio PnL: $XXX
- B0 PnL: $XXX (Trades: X)
- S4 PnL: $XXX (Trades: X)
- S5 PnL: $XXX (Trades: X)

Positions:
- Open: X (details...)
- Closed: X (details...)

Alerts:
- Critical: X
- Warnings: X
- Resolved: X

Incidents:
- None / Details...

Notes:
- Observations, unusual market conditions, etc.
```

### Weekly Review (Every Monday)

**Owner:** Lead operator + Quant analyst

**Duration:** 30 minutes

**Output:** Weekly report (email to stakeholders)

**Agenda:**
1. Review last 7 days performance
   - Portfolio PF, WR, trade count
   - Per-system performance
   - Compare to targets
2. Review risk metrics
   - Max DD, exposure, stop losses hit
3. Review alerts and incidents
   - Any recurring issues?
   - Root causes resolved?
4. Plan for next week
   - Expected market conditions
   - Any config changes needed?

**Template:**
```
Week of: YYYY-MM-DD to YYYY-MM-DD

Performance Summary:
- Portfolio PF (7d): X.XX
- Portfolio WR (7d): XX%
- Portfolio PnL (7d): $XXX
- Trades: X

Per-System:
- B0:  PF X.XX, WR XX%, PnL $XXX, Trades X
- S4:  PF X.XX, WR XX%, PnL $XXX, Trades X
- S5:  PF X.XX, WR XX%, PnL $XXX, Trades X

Risk:
- Max DD: X%
- Max exposure: X%
- Stop losses hit: X
- Largest loss: $XXX

Alerts:
- Total alerts: X
- Critical: X (resolved: X)
- Warnings: X (resolved: X)

Incidents:
- None / Details...

Notes:
- Market conditions (bull/bear/neutral/crisis)
- Regime transitions
- Any observations
```

### Monthly Review (First Monday of Month)

**Owner:** Lead operator + Quant team + Management

**Duration:** 1 hour

**Output:** Monthly report + Rebalancing decision

**Agenda:**
1. Review 30-day performance
   - Portfolio PF, WR, Sharpe, DD
   - Per-system performance
   - Compare to targets and backtest
2. Rebalancing decision
   - Current allocation: X% B0, Y% S4, Z% S5
   - Recommended allocation: Based on performance
   - Rationale: Data-driven justification
3. Risk review
   - Were risk limits respected?
   - Any violations?
   - Should limits be adjusted?
4. Operational review
   - Uptime (target: 99.9%)
   - Incidents and root causes
   - Any process improvements?
5. Next month plan
   - Expected market conditions
   - Any strategy changes?
   - Any new systems to deploy?

**Template:**
```
Month: YYYY-MM

Performance Summary:
- Portfolio PF (30d): X.XX (target: >2.0)
- Portfolio WR (30d): XX% (target: >40%)
- Portfolio Sharpe (30d): X.XX (target: >1.0)
- Portfolio Max DD: X% (target: <15%)
- Total trades: X

Per-System:
- B0:  PF X.XX, WR XX%, Trades X, Allocation: X%
- S4:  PF X.XX, WR XX%, Trades X, Allocation: X%
- S5:  PF X.XX, WR XX%, Trades X, Allocation: X%

Rebalancing Decision:
- Current: X% B0, Y% S4, Z% S5
- Recommended: A% B0, B% S4, C% S5
- Rationale: [Data-driven explanation]
- Effective: YYYY-MM-DD

Risk:
- Risk limits respected: Yes / No (violations: X)
- Recommended limit adjustments: None / Details...

Operations:
- Uptime: XX.X% (target: 99.9%)
- Incidents: X (critical: X, resolved: X)
- Process improvements: [List...]

Next Month:
- Expected conditions: [Bull/bear/neutral/crisis]
- Strategy changes: [None / Details...]
- New deployments: [None / Details...]
```

### Quarterly Review (Every 3 Months)

**Owner:** All stakeholders

**Duration:** 2 hours

**Output:** Quarterly report + Strategic decisions

**Agenda:**
1. 90-day performance review
   - Full metrics analysis
   - Comparison to backtest and targets
   - Regime-specific performance
2. Strategy review
   - Is current strategy still valid?
   - Should we optimize parameters?
   - Should we add/remove systems?
3. Risk assessment
   - Is risk management working?
   - Should we adjust position sizing?
   - Should we adjust stop losses?
4. Operational review
   - System stability
   - Incident analysis
   - Team performance
5. Annual plan
   - Targets for next quarter
   - Capital allocation
   - New initiatives

---

## Logging and Documentation

### Trade Journal

**Location:** `logs/trade_journal.csv`

**Format:** CSV with columns:
```
timestamp, system, signal_type, entry_price, stop_loss, take_profit, position_size, regime, fusion_score (if archetype), exit_timestamp, exit_price, pnl, notes
```

**Example:**
```
2025-12-03 10:15:00, B0, long, 42500.00, 41000.00, 45900.00, 0.15, neutral, N/A, 2025-12-05 14:30:00, 45800.00, +777.00, "Drawdown -16.2%, clean entry"
```

**Usage:**
- Log every trade (entry and exit)
- Review monthly for analysis
- Export for backtesting comparison

### Daily Log

**Location:** `logs/daily_operations.md`

**Format:** Markdown, one entry per day

**Template:** See Daily Review section above

**Usage:**
- Document daily operations
- Track incidents and resolutions
- Handoff between operators (shift changes)

### Incident Log

**Location:** `logs/incidents.csv`

**Format:** CSV with columns:
```
timestamp, severity, system, description, root_cause, resolution, resolved_by, time_to_resolve
```

**Severity Levels:**
- **Critical:** System down, kill switch activated, API failure
- **High:** Stop loss failure, execution error, data feed issue
- **Medium:** Unexpected signal, alert not firing, monitoring issue
- **Low:** Minor bug, cosmetic issue, documentation gap

**Usage:**
- Track all incidents for post-mortem
- Identify recurring issues
- Measure MTTR (mean time to resolution)

---

## Contact and Escalation

### Escalation Matrix

| Issue Severity | Response Time | Contact | Method |
|----------------|---------------|---------|--------|
| **Critical** | Immediate (< 5 min) | On-call engineer | Phone + Slack |
| **High** | 30 minutes | Lead operator | Slack |
| **Medium** | 2 hours | Team lead | Slack |
| **Low** | Next business day | Team | Slack |

### On-Call Rotation

**Schedule:**
- Week 1: Engineer A
- Week 2: Engineer B
- Week 3: Engineer C
- Week 4: Engineer D

**On-Call Duties:**
- Respond to critical alerts within 5 minutes
- Available 24/7 (phone, Slack)
- Escalate to management if needed

### Escalation Paths

**Level 1: Operator**
- Handles routine operations
- Monitors dashboard
- Responds to low/medium severity issues
- Escalates critical issues

**Level 2: Lead Operator**
- Handles high severity issues
- Makes rebalancing decisions
- Coordinates weekly reviews
- Escalates to Level 3 if needed

**Level 3: Engineering Lead**
- Handles critical technical issues
- Makes architecture decisions
- Coordinates bug fixes
- Escalates to management for strategic decisions

**Level 4: Management**
- Handles strategic decisions
- Kill switch approval (if needed)
- Budget and resource allocation
- Final decision authority

---

## Key Takeaways

1. **Daily monitoring is essential** - 15 min morning check, 5 min midday, 20 min EOD
2. **Automate alerts** - Don't rely on manual checks for critical issues
3. **Document everything** - Trade journal, daily log, incident log
4. **Clear escalation** - Know who to call and when
5. **Practice emergency procedures** - Test kill switch quarterly
6. **Review regularly** - Daily, weekly, monthly, quarterly reviews
7. **Continuous improvement** - Learn from incidents, update playbook

**Remember:** Systems are independent - one system failing doesn't affect the others.

---

**Document Owner:** Operations Team
**Last Updated:** 2025-12-03
**Next Review:** Monthly (first Monday)
