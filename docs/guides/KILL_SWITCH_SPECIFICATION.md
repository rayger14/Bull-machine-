# Kill-Switch and Circuit Breaker Specification
## Capital Protection Framework for Multi-Archetype Trading System

**Version**: 1.0
**Status**: Production-Ready Design
**Last Updated**: 2025-12-17
**Owner**: Risk Management Team

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Circuit Breaker Hierarchy](#circuit-breaker-hierarchy)
3. [Performance-Based Kill Switches](#performance-based-kill-switches)
4. [System Health Kill Switches](#system-health-kill-switches)
5. [Execution-Based Kill Switches](#execution-based-kill-switches)
6. [Market Anomaly Detection](#market-anomaly-detection)
7. [Capital Protection Rules](#capital-protection-rules)
8. [Recovery & Resume Protocols](#recovery--resume-protocols)
9. [Manual Override Capabilities](#manual-override-capabilities)
10. [Monitoring & Alerting](#monitoring--alerting)
11. [Testing & Validation](#testing--validation)
12. [Implementation Guide](#implementation-guide)
13. [Operations Playbook](#operations-playbook)

---

## Executive Summary

### Philosophy

**CAPITAL PRESERVATION > PROFIT MAXIMIZATION**

This framework implements a fail-safe, layered defense system designed to protect live capital during automated trading operations. Every decision defaults to halting or reducing risk when uncertain.

### Design Principles

1. **Fail-Safe**: Default to halt when conditions are ambiguous
2. **Layered Defense**: 4-tier escalation (Info → Warning → Soft Halt → Instant Halt)
3. **No Single Point of Failure**: Redundant monitoring across multiple dimensions
4. **Auditability**: All decisions logged with timestamps, reasons, and responsible parties
5. **Graceful Degradation**: Reduce risk progressively before full shutdown
6. **Human-in-the-Loop**: Critical decisions require manual approval

### Key Metrics

| Tier | Response Time | Human Approval | Position Impact |
|------|--------------|----------------|-----------------|
| **Tier 4: Info** | N/A | No | None |
| **Tier 3: Warning** | Log only | No | None |
| **Tier 2: Soft Halt** | <5 seconds | Optional | Reduce 50-75% |
| **Tier 1: Instant Halt** | <1 second | Required | Stop all new entries |

---

## Circuit Breaker Hierarchy

### Decision Tree Overview

```
                      ┌─────────────────────┐
                      │  System Monitoring  │
                      │   (Continuous)      │
                      └──────────┬──────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Metric Evaluation     │
                    │  (Every 1-5 seconds)    │
                    └────────────┬────────────┘
                                 │
                ┌────────────────┼────────────────┐
                │                │                │
                ▼                ▼                ▼
         ┌──────────┐     ┌──────────┐    ┌──────────┐
         │  TIER 4  │     │  TIER 3  │    │  TIER 2  │
         │   INFO   │────▶│ WARNING  │───▶│SOFT HALT │
         │(Log Only)│     │ (Alert)  │    │(Reduce)  │
         └──────────┘     └──────────┘    └─────┬────┘
                                                 │
                                                 │ Escalate if
                                                 │ conditions worsen
                                                 ▼
                                          ┌──────────┐
                                          │  TIER 1  │
                                          │ INSTANT  │
                                          │  HALT    │
                                          └─────┬────┘
                                                │
                                                ▼
                                       ┌────────────────┐
                                       │ Manual Recovery│
                                       │   Required     │
                                       └────────────────┘
```

---

## 1. Circuit Breaker Hierarchy (Layered Defense)

### Tier 1: Instant Halt (Emergency Stop)

**Trigger Conditions** (ANY condition triggers immediate halt):

| Condition | Threshold | Response Time | Position Action |
|-----------|-----------|---------------|-----------------|
| **Daily Loss** | >5% of capital | <1 second | Stop all new entries |
| **Flash Crash** | >10% price move in <5min | <1 second | Close all at market |
| **Exchange API Failure** | Cannot place/cancel orders | <1 second | Stop trading |
| **Metadata Integrity Failure** | All boost/veto = 0 for >3 archetypes | <1 second | Stop trading |
| **Position Sizing Overflow** | Attempting >100% capital allocation | <1 second | Reject trade + halt |
| **Drawdown from Peak** | >25% | <1 second | Stop new entries |
| **Fill Rate Collapse** | <85% for >5 trades | <1 second | Stop trading |
| **Order Failure Cascade** | 10+ failed orders in 10 minutes | <1 second | Stop trading |
| **Data Feed Corruption** | Missing OHLCV >5 minutes | <1 second | Stop trading |
| **Leverage Breach** | Actual leverage >1.5x (if unlevered system) | <1 second | Close positions |

**Response Protocol**:

```python
def tier1_instant_halt(trigger_condition: str):
    """
    Instant halt - no human approval needed for initial stop.

    Actions:
    1. Stop all new entry signal generation (<1 second)
    2. Cancel all pending orders (<2 seconds)
    3. Alert all stakeholders (SMS + phone call)
    4. Log trigger condition with full context
    5. Await manual recovery approval
    """
    # 1. Set global trading flag
    TRADING_ENABLED = False

    # 2. Cancel all pending orders
    cancel_all_pending_orders()

    # 3. Log incident
    log_circuit_breaker_event(
        tier=1,
        trigger=trigger_condition,
        timestamp=now(),
        portfolio_state=get_portfolio_snapshot(),
        market_data=get_market_snapshot()
    )

    # 4. Send alerts
    send_emergency_alert(
        channels=["sms", "phone", "slack", "email"],
        message=f"TIER 1 HALT: {trigger_condition}",
        severity="CRITICAL"
    )

    # 5. Optional: Close all positions (if trigger is flash crash)
    if trigger_condition == "flash_crash_detected":
        close_all_positions_at_market()
```

**Position Handling**:

- **Flash Crash**: Close all positions at market immediately
- **All Other Triggers**: Keep existing positions, stop new entries
- **Rationale**: Panic selling often worse than riding out temporary issues

**Resume Requirements**:

1. Root cause identified and documented
2. Manual approval from 2 authorized persons (CEO + Head of Trading)
3. System health checks passed
4. Gradual resume (start at 25% position sizes, ramp over 24h)

---

### Tier 2: Soft Halt (Risk Reduction)

**Trigger Conditions**:

| Condition | Threshold | Response Time | Position Action |
|-----------|-----------|---------------|-----------------|
| **Drawdown from Peak** | 15-25% | <5 seconds | Reduce to 50% size |
| **Win Rate Degradation** | <45% for 72 hours | <5 seconds | Reduce to 50% size |
| **Fill Rate Degradation** | 85-95% | <5 seconds | Reduce to 75% size |
| **HMM Regime Thrashing** | >3 transitions in 1 hour | <5 seconds | Reduce to 50% size |
| **Sharpe Ratio Floor** | <0.5 for 5 consecutive days | <5 seconds | Reduce to 50% size |
| **Archetype Failure Cluster** | 3-7 archetypes with 0 signals for >24h | <5 seconds | Reduce to 50% size |
| **Signal Overlap Explosion** | 55-65% overlap | <5 seconds | Reduce to 75% size |
| **Slippage Explosion** | 0.30-0.50% per trade | <5 seconds | Reduce to 75% size |
| **Weekly Loss** | 7-10% of capital | <5 seconds | Reduce to 50% size |

**Response Protocol**:

```python
def tier2_soft_halt(trigger_condition: str, risk_reduction: float = 0.5):
    """
    Soft halt - reduce risk, alert human, continue monitoring.

    Actions:
    1. Reduce position sizes to risk_reduction % (default 50%)
    2. Stop new entries if condition severe
    3. Alert stakeholders (Slack + Email)
    4. Monitor for escalation to Tier 1
    """
    # 1. Reduce position sizing
    POSITION_SIZE_MULTIPLIER = risk_reduction

    # 2. Log event
    log_circuit_breaker_event(
        tier=2,
        trigger=trigger_condition,
        risk_reduction=risk_reduction,
        timestamp=now()
    )

    # 3. Send alerts
    send_alert(
        channels=["slack", "email"],
        message=f"TIER 2 SOFT HALT: {trigger_condition} - Risk reduced to {risk_reduction*100}%",
        severity="WARNING"
    )

    # 4. Monitor for escalation
    schedule_escalation_check(interval_minutes=15)
```

**Escalation Logic**:

```python
def check_soft_halt_escalation():
    """Check if Tier 2 should escalate to Tier 1."""

    # If conditions worsen after 2 hours → escalate
    if time_since_soft_halt() > 2 * 60 * 60:  # 2 hours
        if conditions_worsening():
            tier1_instant_halt("tier2_escalation")

    # If conditions improve → resume
    if conditions_improving() and time_since_soft_halt() > 2 * 60 * 60:
        resume_from_soft_halt()
```

**Resume Conditions**:

- Trigger metric returns to normal range for >2 hours
- No new warning signals
- Manual approval (optional, recommended for drawdown triggers)

---

### Tier 3: Warning Alerts (Monitor Closely)

**Trigger Conditions**:

| Condition | Threshold | Response Time | Action |
|-----------|-----------|---------------|--------|
| **Sharpe Ratio Degradation** | <1.0 for 48 hours | Log only | Alert + monitor |
| **Win Rate Drift** | <50% for 24 hours | Log only | Alert + monitor |
| **Archetype Performance Degradation** | 2 archetypes with >15% degradation | Log only | Alert + monitor |
| **Feature Drift (PSI)** | >0.25 for >3 features | Log only | Alert + monitor |
| **Signal Overlap Increase** | 45-55% | Log only | Alert + monitor |
| **Fill Rate Degradation** | 90-95% | Log only | Alert + monitor |
| **Slippage Increase** | 0.15-0.30% per trade | Log only | Alert + monitor |
| **Daily Loss** | 3-5% of capital | Log only | Alert + monitor |

**Response Protocol**:

```python
def tier3_warning(trigger_condition: str):
    """
    Warning alert - no action yet, just monitor.

    Actions:
    1. Log warning
    2. Send alert to monitoring channel
    3. Increase monitoring frequency
    4. Prepare for potential escalation
    """
    log_circuit_breaker_event(
        tier=3,
        trigger=trigger_condition,
        timestamp=now()
    )

    send_alert(
        channels=["email"],
        message=f"TIER 3 WARNING: {trigger_condition} - Monitoring closely",
        severity="INFO"
    )

    # Increase monitoring frequency
    set_monitoring_interval(seconds=60)  # Check every minute instead of every 5
```

**Escalation Conditions**:

- If condition persists for >24 hours → escalate to Tier 2
- If multiple Tier 3 warnings active → escalate to Tier 2
- If metric worsens rapidly → escalate to Tier 2

---

### Tier 4: Informational Logs (No Alert)

**Logged Metrics** (for post-mortem analysis):

- All trade executions (entry, exit, PnL)
- Signal generation (all archetypes, even rejected)
- Metadata state snapshots (every hour)
- Regime transitions (HMM state changes)
- Feature values (for all signals)
- Portfolio state (equity curve, positions, cash)
- Market data quality (missing bars, data gaps)
- API latency (exchange response times)

**No alerts triggered** - just logged for analysis.

---

## 2. Performance-Based Kill Switches

### Drawdown Thresholds

**Definition**: Drawdown measured from rolling equity peak.

| Drawdown Level | Action | Recovery Protocol |
|----------------|--------|-------------------|
| **0-15%** | Normal operation | None |
| **15-20%** | Tier 3 Warning | Monitor closely |
| **20-25%** | Tier 2 Soft Halt (50% risk) | Manual review required |
| **25-35%** | Tier 1 Instant Halt | Manual approval + RCA |
| **>35%** | Emergency Halt + Close All | CEO approval required |

**Measurement**:

```python
def calculate_drawdown() -> float:
    """Calculate current drawdown from equity peak."""
    equity_curve = get_equity_curve()
    peak = max(equity_curve)
    current = equity_curve[-1]
    drawdown = (peak - current) / peak
    return drawdown

def check_drawdown_circuit_breaker():
    """Check drawdown and trigger appropriate tier."""
    dd = calculate_drawdown()

    if dd > 0.35:
        tier1_instant_halt("drawdown_emergency_35pct")
        close_all_positions_at_market()
    elif dd > 0.25:
        tier1_instant_halt("drawdown_25pct")
    elif dd > 0.20:
        tier2_soft_halt("drawdown_20pct", risk_reduction=0.5)
    elif dd > 0.15:
        tier3_warning("drawdown_15pct")
```

**Time-Based Drawdown Limits**:

| Period | Max Drawdown | Action |
|--------|--------------|--------|
| **Intraday** | 5% | Tier 1 Halt (stop trading for rest of day) |
| **Daily** | 7% | Tier 1 Halt (stop trading for rest of day) |
| **Weekly** | 10% | Tier 1 Halt (manual recovery) |
| **Monthly** | 15% | Tier 1 Halt + full system audit |

---

### PnL-Based Kill Switches

**Daily Loss Limits**:

```python
def check_daily_loss_limit():
    """Check daily PnL against limits."""
    daily_pnl_pct = get_daily_pnl_percent()

    if daily_pnl_pct < -5.0:
        tier1_instant_halt("daily_loss_5pct")
    elif daily_pnl_pct < -3.0:
        tier3_warning("daily_loss_3pct")
```

**Weekly Loss Limits**:

```python
def check_weekly_loss_limit():
    """Check weekly PnL against limits."""
    weekly_pnl_pct = get_weekly_pnl_percent()

    if weekly_pnl_pct < -10.0:
        tier1_instant_halt("weekly_loss_10pct")
    elif weekly_pnl_pct < -7.0:
        tier2_soft_halt("weekly_loss_7pct", risk_reduction=0.5)
```

**Monthly Loss Limits**:

```python
def check_monthly_loss_limit():
    """Check monthly PnL against limits."""
    monthly_pnl_pct = get_monthly_pnl_percent()

    if monthly_pnl_pct < -15.0:
        tier1_instant_halt("monthly_loss_15pct")
        require_system_audit()
    elif monthly_pnl_pct < -12.0:
        tier2_soft_halt("monthly_loss_12pct", risk_reduction=0.5)
```

**NO DYNAMIC ADJUSTMENT**: Loss limits are fixed. No raising limits during winning streaks (prevents revenge trading mentality).

---

### Risk-Adjusted Performance Kill Switches

**Sharpe Ratio Floor**:

```python
def check_sharpe_ratio():
    """Check rolling Sharpe ratio."""
    sharpe_30d = calculate_rolling_sharpe(days=30)

    if sharpe_30d < 0.5 and days_below_threshold(sharpe_30d, 0.5) >= 5:
        tier2_soft_halt("sharpe_below_0.5_for_5days", risk_reduction=0.5)
    elif sharpe_30d < 1.0 and days_below_threshold(sharpe_30d, 1.0) >= 2:
        tier3_warning("sharpe_below_1.0_for_48h")
```

**Calmar Ratio Threshold**:

```python
def check_calmar_ratio():
    """Check Calmar ratio (return / max drawdown)."""
    calmar = calculate_calmar_ratio(days=90)

    if calmar < 0.5:
        tier3_warning("calmar_below_0.5")
```

**Sortino Ratio Degradation**:

```python
def check_sortino_ratio():
    """Check Sortino ratio (downside deviation)."""
    sortino_30d = calculate_sortino_ratio(days=30)

    if sortino_30d < 0.75 and days_below_threshold(sortino_30d, 0.75) >= 3:
        tier3_warning("sortino_degradation")
```

---

### Win Rate Degradation

**Expected Win Rate**: 55-60% (from backtest)
**Acceptable Deviation**: ±10% (49.5% - 66%)

```python
def check_win_rate():
    """Check rolling win rate."""
    win_rate_24h = calculate_win_rate(hours=24)
    win_rate_72h = calculate_win_rate(hours=72)

    if win_rate_72h < 0.45:
        tier1_instant_halt("win_rate_below_45pct_for_72h")
    elif win_rate_24h < 0.50:
        tier3_warning("win_rate_below_50pct_for_24h")
```

**Trade Frequency Check**:

```python
def check_trade_frequency():
    """Detect if system stopped generating signals."""
    trades_24h = count_trades(hours=24)

    # Expected: ~3-5 trades per day (based on backtest)
    if trades_24h == 0:
        tier3_warning("zero_trades_24h")
    elif trades_24h > 20:
        tier3_warning("excessive_trades_24h")  # Possible system malfunction
```

---

## 3. System Health Kill Switches

### Metadata Integrity Checks

**Critical Metadata Components**:

1. Domain boost/veto counts
2. Archetype metadata (confidence, reasons)
3. Regime labels
4. Feature values

```python
def check_metadata_integrity():
    """Verify metadata is not corrupted."""

    # Check 1: Domain boost/veto counts
    archetypes_with_zero_metadata = 0
    for archetype in get_all_archetypes():
        metadata = get_archetype_metadata(archetype)
        if sum(metadata.get("domain_boosts", {}).values()) == 0 and \
           sum(metadata.get("domain_vetoes", {}).values()) == 0:
            archetypes_with_zero_metadata += 1

    if archetypes_with_zero_metadata >= 3:
        tier1_instant_halt("metadata_integrity_failure_3plus_archetypes")

    # Check 2: Feature store corruption
    if detect_feature_store_corruption():
        tier1_instant_halt("feature_store_corruption")

    # Check 3: Regime labels missing
    if get_current_regime() is None:
        tier2_soft_halt("regime_label_missing", risk_reduction=0.5)
```

**Feature Store Validation**:

```python
def detect_feature_store_corruption() -> bool:
    """Check if feature store is corrupted."""

    # Check for missing critical features
    critical_features = [
        "smc_bos_bullish", "smc_choch_bearish",
        "wyckoff_phase", "liquidity_score",
        "funding_rate", "oi_pct_change"
    ]

    missing_features = []
    for feature in critical_features:
        if not feature_exists(feature):
            missing_features.append(feature)

    if len(missing_features) > 0:
        log_error(f"Missing critical features: {missing_features}")
        return True

    # Check for feature value anomalies
    if detect_feature_value_anomalies():
        return True

    return False
```

---

### Archetype Failure Detection

```python
def check_archetype_health():
    """Monitor archetype signal generation."""

    failed_archetypes = []
    for archetype in get_all_archetypes():
        signals_24h = count_signals(archetype, hours=24)

        # Define "failed": 0 signals for >24h (if archetype should be active)
        if signals_24h == 0 and is_archetype_expected_active(archetype):
            failed_archetypes.append(archetype)

    num_failed = len(failed_archetypes)

    if num_failed >= 8:
        tier1_instant_halt(f"archetype_failure_cluster_{num_failed}_archetypes")
    elif num_failed >= 3:
        tier2_soft_halt(f"archetype_degradation_{num_failed}_archetypes", risk_reduction=0.5)
    elif num_failed >= 1:
        tier3_warning(f"archetype_failure_{failed_archetypes[0]}")
```

**Expected Activity Matrix**:

| Regime | Expected Active Archetypes |
|--------|---------------------------|
| **Bull** | 5-8 bull archetypes |
| **Bear** | 4-6 bear archetypes |
| **Neutral** | 2-4 archetypes (any) |

---

### Regime Detection Failure

**HMM Regime Thrashing**:

```python
def check_regime_stability():
    """Detect if HMM is thrashing between regimes."""

    transitions_1h = count_regime_transitions(hours=1)

    if transitions_1h > 5:
        tier1_instant_halt("hmm_thrashing_5plus_transitions_1h")
    elif transitions_1h > 3:
        tier2_soft_halt("hmm_thrashing_3_transitions_1h", risk_reduction=0.5)
```

**HMM Frozen Detection**:

```python
def check_regime_frozen():
    """Detect if HMM stuck in one regime."""

    days_in_current_regime = get_days_in_current_regime()

    if days_in_current_regime > 14:
        tier3_warning("hmm_stuck_in_regime_14days")
        # Recommend manual regime override or model retrain
```

**HMM Reality Check**:

```python
def check_regime_vs_reality():
    """Compare HMM regime to market reality."""

    hmm_regime = get_hmm_regime()
    market_regime = infer_market_regime_from_price_action()

    if hmm_regime != market_regime:
        days_disagreement = get_days_regime_disagreement()

        if days_disagreement > 3:
            tier3_warning("hmm_disagrees_with_reality_3days")
            # Recommend manual override
```

---

### Signal Overlap Explosion

**Overlap Monitoring**:

```python
def check_signal_overlap():
    """Monitor signal overlap across archetypes."""

    overlap_pct = calculate_signal_overlap()

    if overlap_pct > 0.65:
        tier1_instant_halt("signal_overlap_65pct_redundancy_collapse")
    elif overlap_pct > 0.55:
        tier2_soft_halt("signal_overlap_55pct", risk_reduction=0.75)
    elif overlap_pct > 0.45:
        tier3_warning("signal_overlap_45pct")
```

**Overlap Calculation**:

```python
def calculate_signal_overlap() -> float:
    """Calculate percentage of overlapping signals."""

    all_signals = get_all_signals_last_24h()

    # Group signals by timestamp (within 1-hour window)
    signal_groups = group_signals_by_time(all_signals, window_minutes=60)

    overlapping_signals = sum(1 for group in signal_groups if len(group) > 1)
    total_signals = len(all_signals)

    return overlapping_signals / max(total_signals, 1)
```

---

## 4. Execution-Based Kill Switches

### Fill Rate Degradation

```python
def check_fill_rate():
    """Monitor order fill rate."""

    fill_rate = calculate_fill_rate(hours=24)

    if fill_rate < 0.85:
        tier1_instant_halt("fill_rate_below_85pct")
    elif fill_rate < 0.90:
        tier2_soft_halt("fill_rate_85_90pct", risk_reduction=0.75)
    elif fill_rate < 0.95:
        tier3_warning("fill_rate_90_95pct")
```

**Fill Rate Calculation**:

```python
def calculate_fill_rate(hours: int = 24) -> float:
    """Calculate percentage of orders that filled."""

    orders = get_orders_last_n_hours(hours)
    filled = sum(1 for order in orders if order.status == "filled")
    total = len(orders)

    return filled / max(total, 1)
```

---

### Slippage Explosion

```python
def check_slippage():
    """Monitor execution slippage."""

    avg_slippage = calculate_average_slippage(hours=24)

    if avg_slippage > 0.005:  # 0.5%
        tier1_instant_halt("slippage_above_0.5pct")
    elif avg_slippage > 0.003:  # 0.3%
        tier2_soft_halt("slippage_0.3_0.5pct", risk_reduction=0.75)
    elif avg_slippage > 0.0015:  # 0.15%
        tier3_warning("slippage_0.15_0.3pct")
```

**Slippage Calculation**:

```python
def calculate_average_slippage(hours: int = 24) -> float:
    """Calculate average slippage per trade."""

    trades = get_trades_last_n_hours(hours)

    slippages = []
    for trade in trades:
        expected_price = trade.signal_price
        actual_price = trade.fill_price
        slippage = abs(actual_price - expected_price) / expected_price
        slippages.append(slippage)

    return sum(slippages) / max(len(slippages), 1)
```

---

### Order Failures

```python
def check_order_failures():
    """Monitor order placement failures."""

    failures_10min = count_order_failures(minutes=10)

    if failures_10min >= 10:
        tier1_instant_halt("order_failures_10plus_in_10min")
    elif failures_10min >= 5:
        tier2_soft_halt("order_failures_5_in_10min", risk_reduction=0.5)
    elif failures_10min >= 3:
        tier3_warning("order_failures_3_in_10min")
```

**Order Failure Types**:

- API errors (500, 503, timeout)
- Insufficient balance
- Invalid order parameters
- Market closed / halted
- Rate limit exceeded

---

### Position Sizing Errors

```python
def validate_position_size(signal) -> bool:
    """Validate position size before placing order."""

    portfolio = get_portfolio_state()

    # Check 1: Not attempting >100% capital allocation
    total_allocated = portfolio.get_total_allocated_pct()
    new_allocation = calculate_position_allocation(signal)

    if total_allocated + new_allocation > 1.0:
        log_error("Position sizing overflow: attempting >100% allocation")
        send_alert("Position sizing error: rejected trade")
        return False  # Reject trade, don't halt

    # Check 2: Single trade risk not >10% of account
    if new_allocation > 0.10:
        log_error(f"Single trade risk too high: {new_allocation*100}%")
        send_alert("Position sizing error: rejected trade")
        return False

    # Check 3: Leverage not >1.5x (if system should be unlevered)
    if portfolio.get_leverage() > 1.5:
        tier1_instant_halt("leverage_breach_1.5x")
        return False

    return True
```

---

## 5. Market Anomaly Detection

### Flash Crash Protection

```python
def detect_flash_crash() -> bool:
    """Detect rapid price moves indicating flash crash."""

    price_5min_ago = get_price_n_minutes_ago(5)
    current_price = get_current_price()

    price_change_pct = abs(current_price - price_5min_ago) / price_5min_ago

    if price_change_pct > 0.10:  # 10% move in 5 minutes
        tier1_instant_halt("flash_crash_10pct_in_5min")
        close_all_positions_at_market()  # Exit immediately
        return True

    return False
```

**Flash Crash Response**:

1. Instant halt (<1 second)
2. Close all positions at market
3. Alert all stakeholders (SMS + phone call)
4. Manual validation of exchange data required before resume

---

### Liquidity Crisis Detection

```python
def detect_liquidity_crisis() -> bool:
    """Detect market liquidity issues."""

    # Check 1: Bid-ask spread
    bid_ask_spread = get_bid_ask_spread()
    mid_price = get_mid_price()
    spread_pct = bid_ask_spread / mid_price

    if spread_pct > 0.01:  # 1% spread (normal: <0.1%)
        tier2_soft_halt("liquidity_crisis_1pct_spread", risk_reduction=0.0)
        # Stop new entries, allow exits only
        return True

    # Check 2: Order book depth
    depth_1pct = get_orderbook_depth_within_pct(1.0)

    if depth_1pct < 100_000:  # <$100k depth (normal: >$1M)
        tier2_soft_halt("liquidity_crisis_low_depth", risk_reduction=0.0)
        return True

    return False
```

---

### Exchange Outage Detection

```python
def detect_exchange_outage() -> bool:
    """Detect exchange connectivity issues."""

    # Check 1: API latency
    latency_ms = get_api_latency()

    if latency_ms > 5000:  # 5 seconds (normal: <500ms)
        tier1_instant_halt("exchange_api_latency_5s")
        return True

    # Check 2: WebSocket disconnect
    ws_disconnect_seconds = get_websocket_disconnect_duration()

    if ws_disconnect_seconds > 30:
        tier1_instant_halt("websocket_disconnect_30s")
        return True

    # Check 3: Heartbeat failure
    if not exchange_heartbeat_ok():
        tier1_instant_halt("exchange_heartbeat_failure")
        return True

    return False
```

---

### Data Feed Corruption

```python
def detect_data_corruption() -> bool:
    """Detect corrupted market data."""

    # Check 1: Missing OHLCV data
    if has_missing_candles(minutes=5):
        tier1_instant_halt("missing_ohlcv_data_5min")
        return True

    # Check 2: Price discontinuity (>5% jump without trade)
    if detect_price_discontinuity(threshold_pct=0.05):
        tier2_soft_halt("price_discontinuity_5pct", risk_reduction=0.0)
        return True

    # Check 3: Zero volume (market closed?)
    if get_volume_last_n_minutes(10) == 0:
        tier2_soft_halt("zero_volume_10min", risk_reduction=0.0)
        return True

    return False
```

---

## 6. Capital Protection Rules

### Maximum Loss Per Trade

**Hard Limit**: 1% of capital per trade

```python
def enforce_max_loss_per_trade(signal):
    """Enforce maximum risk per trade."""

    risk_pct = calculate_trade_risk_pct(signal)

    if risk_pct > 0.01:  # 1% max risk
        log_error(f"Trade risk {risk_pct*100}% exceeds max 1%")
        send_alert("Trade rejected: risk too high")
        return None  # Reject trade

    return signal
```

**NO EXCEPTIONS**: Even if confluence score = 5.0, max risk is 1%.

---

### Maximum Loss Per Day

**Hard Limit**: 5% of capital

```python
def enforce_max_daily_loss():
    """Halt trading if daily loss exceeds 5%."""

    daily_loss_pct = get_daily_loss_pct()

    if daily_loss_pct > 0.05:
        tier1_instant_halt("daily_loss_5pct")
        halt_until_next_day()
```

**Resume**: Next day at 00:00 UTC with manual approval.

---

### Maximum Loss Per Week

**Hard Limit**: 10% of capital

```python
def enforce_max_weekly_loss():
    """Halt trading if weekly loss exceeds 10%."""

    weekly_loss_pct = get_weekly_loss_pct()

    if weekly_loss_pct > 0.10:
        tier1_instant_halt("weekly_loss_10pct")
        require_root_cause_analysis()
```

**Resume**: Requires root cause analysis before resuming.

---

### Maximum Drawdown from Peak

| Drawdown | Action |
|----------|--------|
| **15%** | Tier 2: Reduce to 50% position sizes |
| **25%** | Tier 1: Full halt, manual recovery |
| **35%** | Emergency: Close all positions, shut down system |

---

## 7. Recovery & Resume Protocols

### After Tier 1 Instant Halt

**Recovery Checklist**:

1. **Root Cause Analysis**
   - What triggered the halt?
   - Was it correct or false positive?
   - What conditions led to trigger?

2. **Data Validation**
   - Is market data valid and reliable?
   - Are exchange APIs functioning normally?
   - Is feature store intact?

3. **System Health Check**
   - All archetypes generating signals?
   - Metadata integrity verified?
   - Regime detection functioning?

4. **Manual Approval**
   - Require sign-off from 2 authorized persons
   - CEO + Head of Trading (or equivalent)
   - Document approval in audit log

5. **Gradual Resume**
   ```python
   def gradual_resume_from_tier1():
       """Resume trading gradually after Tier 1 halt."""

       # Hour 0-6: 25% position sizes
       set_position_size_multiplier(0.25)
       enable_trading()

       # Hour 6-12: 50% position sizes
       schedule_action(hours=6, action=lambda: set_position_size_multiplier(0.50))

       # Hour 12-18: 75% position sizes
       schedule_action(hours=12, action=lambda: set_position_size_multiplier(0.75))

       # Hour 18-24: 100% position sizes
       schedule_action(hours=18, action=lambda: set_position_size_multiplier(1.0))
   ```

---

### After Tier 2 Soft Halt

**Resume Conditions**:

1. Trigger metric returns to normal range for >2 hours
2. No new warning signals
3. Manual approval (optional, but recommended for drawdown triggers)

```python
def resume_from_tier2():
    """Resume from soft halt if conditions improved."""

    if conditions_improved() and time_since_soft_halt() > 2 * 60 * 60:
        # Gradually restore position sizes
        set_position_size_multiplier(0.75)  # Start at 75%

        schedule_action(hours=4, action=lambda: set_position_size_multiplier(1.0))

        log_event("Resumed from Tier 2 soft halt")
        send_alert("Trading resumed at reduced risk")
```

---

### After Tier 3 Warning

**No special resume needed** - just monitor and clear warning flag when resolved.

---

### Graceful Shutdown Procedure

**When shutting down** (e.g., end of trading, manual shutdown):

```python
def graceful_shutdown():
    """Gracefully shut down trading system."""

    # Step 1: Stop new entries
    ALLOW_NEW_ENTRIES = False
    log_event("Graceful shutdown initiated - no new entries")

    # Step 2: Let existing positions run to target/stop
    # (Don't panic close)

    # Step 3: After 12 hours, close any remaining positions
    schedule_action(hours=12, action=close_all_positions_at_market)

    # Step 4: Final shutdown after all positions closed
    wait_for_all_positions_closed()

    log_event("Graceful shutdown complete")
    disable_trading()
```

**Rationale**: Avoid panic selling. Let trades complete naturally.

---

## 8. Manual Override Capabilities

### Emergency Stop Button

**UI Component**:

```
┌─────────────────────────────────┐
│  EMERGENCY STOP                 │
│                                 │
│  [HALT TRADING IMMEDIATELY]     │
│                                 │
│  ⚠️  Requires authentication    │
│                                 │
│  [ ] Close all positions        │
│                                 │
│  Reason: ________________       │
│                                 │
│  [CONFIRM HALT]                 │
└─────────────────────────────────┘
```

**Implementation**:

```python
def emergency_stop_button(
    close_positions: bool = False,
    reason: str = "",
    authenticated_user: str = ""
):
    """Manual emergency stop triggered by user."""

    # Require authentication
    if not verify_authentication(authenticated_user):
        raise PermissionError("Authentication required")

    # Log who triggered and why
    log_audit_event(
        action="emergency_stop",
        user=authenticated_user,
        reason=reason,
        timestamp=now()
    )

    # Halt trading
    tier1_instant_halt(f"manual_emergency_stop_by_{authenticated_user}")

    # Optionally close positions
    if close_positions:
        close_all_positions_at_market()

    # Alert team
    send_alert(
        f"EMERGENCY STOP triggered by {authenticated_user}: {reason}",
        severity="CRITICAL"
    )
```

---

### Force Resume

**Requires CEO-level approval**:

```python
def force_resume(
    authenticated_user: str,
    justification: str,
    approval_signature: str
):
    """Override automatic halt and force resume."""

    # Require CEO-level authentication
    if not verify_ceo_authentication(authenticated_user, approval_signature):
        raise PermissionError("CEO approval required")

    # Log override
    log_audit_event(
        action="force_resume",
        user=authenticated_user,
        justification=justification,
        timestamp=now()
    )

    # Resume trading
    enable_trading()
    set_position_size_multiplier(1.0)

    # Alert team
    send_alert(
        f"FORCE RESUME by {authenticated_user}: {justification}",
        severity="WARNING"
    )
```

---

### Risk Dial

**Adjust position sizes on the fly**:

```python
def set_risk_dial(level: str, duration_hours: int = 24):
    """
    Manually adjust risk level.

    Args:
        level: "25%", "50%", "75%", "100%"
        duration_hours: Auto-reset to 100% after N hours
    """

    multipliers = {
        "25%": 0.25,
        "50%": 0.50,
        "75%": 0.75,
        "100%": 1.00
    }

    multiplier = multipliers.get(level, 1.0)
    set_position_size_multiplier(multiplier)

    log_event(f"Risk dial set to {level} for {duration_hours} hours")

    # Auto-reset after duration
    schedule_action(
        hours=duration_hours,
        action=lambda: set_position_size_multiplier(1.0)
    )
```

**UI Component**:

```
┌─────────────────────────────────┐
│  RISK DIAL                      │
│                                 │
│  Current: 100%                  │
│                                 │
│  ○ 25%  ○ 50%  ○ 75%  ● 100%   │
│                                 │
│  Duration: [24] hours           │
│                                 │
│  [APPLY]                        │
└─────────────────────────────────┘
```

---

## 9. Monitoring & Alerting

### Alert Channels

| Tier | Severity | Channels | Response SLA |
|------|----------|----------|--------------|
| **Tier 1** | CRITICAL | SMS + Phone Call + Slack + Email | Acknowledge <5 min |
| **Tier 2** | WARNING | Slack + Email | Acknowledge <15 min |
| **Tier 3** | INFO | Email | Acknowledge <1 hour |
| **Tier 4** | DEBUG | Logs only | N/A |

---

### Alert Frequency

```python
def send_alert_with_frequency_control(message: str, tier: int):
    """Send alerts with frequency throttling."""

    if tier == 1:
        # Instant notification, no throttling
        send_sms(message)
        make_phone_call(message)
        send_slack(message)
        send_email(message)

    elif tier == 2:
        # Alert every 15 minutes until resolved
        if time_since_last_alert(message) > 15 * 60:
            send_slack(message)
            send_email(message)

    elif tier == 3:
        # Alert once, then hourly summary
        if not already_alerted(message):
            send_email(message)
        elif time_since_last_alert(message) > 60 * 60:
            send_email_summary()

    elif tier == 4:
        # Log only
        log_info(message)
```

---

### On-Call Rotation

**Escalation Path**:

```
Tier 1 Alert
     │
     ▼
Primary On-Call (5 min SLA)
     │
     ├─ Acknowledged? ─→ Handle incident
     │
     ▼ No response after 5 min
Secondary On-Call (10 min SLA)
     │
     ├─ Acknowledged? ─→ Handle incident
     │
     ▼ No response after 10 min
CEO / Head of Trading (15 min SLA)
```

**On-Call Schedule**:

```python
ON_CALL_SCHEDULE = {
    "primary": {
        "weekdays": "head_of_trading@company.com",
        "weekends": "ceo@company.com"
    },
    "secondary": {
        "weekdays": "senior_trader@company.com",
        "weekends": "head_of_trading@company.com"
    },
    "escalation": "ceo@company.com"
}
```

---

## 10. Testing & Validation

### Pre-Live Checklist

**Before going live with real capital**:

- [ ] All Tier 1 kill switches tested in paper trading
- [ ] All Tier 2 soft halts tested and verified
- [ ] Tier 3 warnings tested and logged correctly
- [ ] Manual emergency stop button tested
- [ ] Force resume tested with CEO approval
- [ ] Risk dial tested at all levels (25%, 50%, 75%, 100%)
- [ ] Alert notifications delivered to all channels (SMS, phone, Slack, email)
- [ ] On-call rotation verified (test page to primary, secondary, escalation)
- [ ] Recovery protocols documented and rehearsed
- [ ] Team trained on escalation procedures
- [ ] Graceful shutdown tested
- [ ] Flash crash simulation tested
- [ ] Exchange outage simulation tested
- [ ] Data corruption simulation tested
- [ ] Metadata integrity checks tested

---

### Monthly Drills

**Conduct monthly circuit breaker drills**:

1. **Simulate Tier 1 Halt**
   - Manually trigger a fake Tier 1 condition
   - Verify halt executes <1 second
   - Verify alerts sent to all channels
   - Practice recovery procedure
   - Measure time to resume

2. **Simulate Tier 2 Soft Halt**
   - Trigger a fake Tier 2 condition
   - Verify position sizes reduced correctly
   - Verify alerts sent
   - Practice recovery

3. **Test Emergency Stop Button**
   - Click emergency stop
   - Verify authentication required
   - Verify audit log entry
   - Verify alerts sent

4. **Test Force Resume**
   - Attempt force resume without CEO approval (should fail)
   - Attempt with CEO approval (should succeed)
   - Verify audit log

**Drill Log Template**:

```
Drill Date: 2025-12-17
Drill Type: Tier 1 Instant Halt Simulation
Trigger: Manual (daily_loss_5pct simulated)

Timeline:
- 14:00:00 - Trigger initiated
- 14:00:00.8 - Trading halted (0.8 seconds)
- 14:00:01 - SMS sent
- 14:00:02 - Phone call initiated
- 14:00:05 - Primary on-call acknowledged
- 14:15:00 - Recovery checklist completed
- 14:20:00 - Manual approval obtained
- 14:25:00 - Gradual resume initiated

Pass/Fail: PASS
Issues: None
Action Items: None
```

---

### Post-Halt Analysis Template

**After every real halt**:

```
Incident Report: TIER-1-HALT-2025-12-17-001

Date/Time: 2025-12-17 14:32:15 UTC
Trigger: daily_loss_5pct
Tier: 1 (Instant Halt)

Timeline:
- 14:32:15 - Halt triggered
- 14:32:16 - Trading stopped
- 14:32:17 - Alerts sent
- 14:37:20 - Primary on-call acknowledged
- 15:00:00 - Root cause identified: adverse market move
- 15:30:00 - Recovery approved
- 15:35:00 - Gradual resume started

Root Cause:
- Market flash crash -8% in 10 minutes
- System correctly halted to prevent further losses
- No system malfunction

Was Halt Correct? YES
False Positive? NO

Positions at Halt:
- 3 open positions
- Total exposure: $4,200
- Unrealized PnL: -$520

Positions Closed?
- No (kept open, market recovered)

Time to Resume: 1 hour 3 minutes

Lessons Learned:
- Halt worked correctly
- Recovery process smooth
- No process improvements needed

Approval: CEO (John Doe), Head of Trading (Jane Smith)
```

---

## 11. Implementation Guide

### Core Circuit Breaker Engine

**File**: `/engine/risk/circuit_breaker.py`

```python
"""
Circuit Breaker System for Capital Protection.

This module implements a 4-tier escalation system to halt trading
when risk metrics exceed acceptable thresholds.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker trigger."""
    tier: int  # 1-4
    trigger: str  # e.g., "daily_loss_5pct"
    timestamp: datetime
    portfolio_state: Dict
    market_data: Dict
    action_taken: str
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None


class CircuitBreakerEngine:
    """
    Circuit breaker engine for kill-switch logic.

    Monitors:
    - Performance metrics (drawdown, PnL, Sharpe, win rate)
    - System health (metadata, archetypes, regime)
    - Execution quality (fill rate, slippage, order failures)
    - Market conditions (flash crash, liquidity, data quality)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.trading_enabled = True
        self.position_size_multiplier = 1.0
        self.events: List[CircuitBreakerEvent] = []

        # Thresholds
        self.thresholds = {
            "daily_loss_pct": 0.05,  # 5%
            "weekly_loss_pct": 0.10,  # 10%
            "drawdown_tier1": 0.25,  # 25%
            "drawdown_tier2": 0.20,  # 20%
            "fill_rate_tier1": 0.85,  # 85%
            "slippage_tier1": 0.005,  # 0.5%
            "flash_crash_pct": 0.10,  # 10% in 5 min
            "overlap_tier1": 0.65,  # 65%
        }

    def check_all_circuit_breakers(self, portfolio, market_data) -> Optional[str]:
        """
        Check all circuit breaker conditions.

        Returns:
            Trigger name if halt needed, None otherwise
        """
        # Performance checks
        if trigger := self._check_performance(portfolio):
            return trigger

        # System health checks
        if trigger := self._check_system_health():
            return trigger

        # Execution checks
        if trigger := self._check_execution_quality():
            return trigger

        # Market anomaly checks
        if trigger := self._check_market_anomalies(market_data):
            return trigger

        return None

    def _check_performance(self, portfolio) -> Optional[str]:
        """Check performance-based kill switches."""

        # Daily loss
        daily_pnl_pct = portfolio.get_daily_pnl_pct()
        if daily_pnl_pct < -self.thresholds["daily_loss_pct"]:
            return "daily_loss_5pct"

        # Drawdown
        dd = portfolio.calculate_drawdown()
        if dd > self.thresholds["drawdown_tier1"]:
            return "drawdown_25pct"
        elif dd > self.thresholds["drawdown_tier2"]:
            return "drawdown_20pct"

        # Win rate (if enough trades)
        if portfolio.get_trade_count(hours=72) >= 10:
            win_rate = portfolio.get_win_rate(hours=72)
            if win_rate < 0.45:
                return "win_rate_below_45pct"

        return None

    def _check_system_health(self) -> Optional[str]:
        """Check system health kill switches."""

        # Metadata integrity
        if self._detect_metadata_corruption():
            return "metadata_integrity_failure"

        # Archetype failures
        failed_archetypes = self._count_failed_archetypes()
        if failed_archetypes >= 8:
            return f"archetype_failure_cluster_{failed_archetypes}"

        # Regime thrashing
        transitions = self._count_regime_transitions(hours=1)
        if transitions > 5:
            return "hmm_thrashing_5plus_transitions"

        return None

    def _check_execution_quality(self) -> Optional[str]:
        """Check execution quality kill switches."""

        # Fill rate
        fill_rate = self._calculate_fill_rate(hours=24)
        if fill_rate < self.thresholds["fill_rate_tier1"]:
            return "fill_rate_below_85pct"

        # Slippage
        slippage = self._calculate_average_slippage(hours=24)
        if slippage > self.thresholds["slippage_tier1"]:
            return "slippage_above_0.5pct"

        # Order failures
        failures = self._count_order_failures(minutes=10)
        if failures >= 10:
            return "order_failures_10plus"

        return None

    def _check_market_anomalies(self, market_data) -> Optional[str]:
        """Check market condition kill switches."""

        # Flash crash
        if self._detect_flash_crash(market_data):
            return "flash_crash_detected"

        # Exchange outage
        if self._detect_exchange_outage():
            return "exchange_outage"

        # Data corruption
        if self._detect_data_corruption():
            return "data_corruption"

        return None

    def tier1_instant_halt(self, trigger: str, portfolio, market_data):
        """Execute Tier 1 instant halt."""

        logger.critical(f"TIER 1 INSTANT HALT: {trigger}")

        # Stop trading
        self.trading_enabled = False

        # Cancel pending orders
        self._cancel_all_pending_orders()

        # Log event
        event = CircuitBreakerEvent(
            tier=1,
            trigger=trigger,
            timestamp=datetime.now(),
            portfolio_state=portfolio.to_dict(),
            market_data=market_data,
            action_taken="instant_halt"
        )
        self.events.append(event)

        # Send alerts
        self._send_emergency_alert(trigger)

        # Close positions if flash crash
        if trigger == "flash_crash_detected":
            self._close_all_positions_at_market()

    def tier2_soft_halt(self, trigger: str, risk_reduction: float = 0.5):
        """Execute Tier 2 soft halt (reduce risk)."""

        logger.warning(f"TIER 2 SOFT HALT: {trigger}")

        # Reduce position sizes
        self.position_size_multiplier = risk_reduction

        # Log event
        event = CircuitBreakerEvent(
            tier=2,
            trigger=trigger,
            timestamp=datetime.now(),
            portfolio_state={},
            market_data={},
            action_taken=f"soft_halt_{risk_reduction*100}pct"
        )
        self.events.append(event)

        # Send alerts
        self._send_warning_alert(trigger, risk_reduction)

    def tier3_warning(self, trigger: str):
        """Log Tier 3 warning."""

        logger.info(f"TIER 3 WARNING: {trigger}")

        # Log event
        event = CircuitBreakerEvent(
            tier=3,
            trigger=trigger,
            timestamp=datetime.now(),
            portfolio_state={},
            market_data={},
            action_taken="warning_logged"
        )
        self.events.append(event)

        # Send alert
        self._send_info_alert(trigger)

    def manual_emergency_stop(self, user: str, reason: str, close_positions: bool = False):
        """Manual emergency stop button."""

        logger.critical(f"MANUAL EMERGENCY STOP by {user}: {reason}")

        # Halt trading
        self.tier1_instant_halt(f"manual_stop_by_{user}", {}, {})

        # Optionally close positions
        if close_positions:
            self._close_all_positions_at_market()

        # Audit log
        self._log_audit_event("emergency_stop", user, reason)

    def force_resume(self, user: str, justification: str, approval: str):
        """Force resume trading (requires CEO approval)."""

        if not self._verify_ceo_approval(user, approval):
            raise PermissionError("CEO approval required")

        logger.warning(f"FORCE RESUME by {user}: {justification}")

        # Resume trading
        self.trading_enabled = True
        self.position_size_multiplier = 1.0

        # Audit log
        self._log_audit_event("force_resume", user, justification)

    # Helper methods (implement based on your system)

    def _detect_metadata_corruption(self) -> bool:
        """Check if metadata is corrupted."""
        # TODO: Implement based on your feature store
        return False

    def _count_failed_archetypes(self) -> int:
        """Count archetypes with 0 signals."""
        # TODO: Implement based on your archetype system
        return 0

    def _count_regime_transitions(self, hours: int) -> int:
        """Count HMM regime transitions."""
        # TODO: Implement based on your regime system
        return 0

    def _calculate_fill_rate(self, hours: int) -> float:
        """Calculate order fill rate."""
        # TODO: Implement based on your execution system
        return 1.0

    def _calculate_average_slippage(self, hours: int) -> float:
        """Calculate average slippage."""
        # TODO: Implement based on your execution system
        return 0.0

    def _count_order_failures(self, minutes: int) -> int:
        """Count recent order failures."""
        # TODO: Implement based on your execution system
        return 0

    def _detect_flash_crash(self, market_data) -> bool:
        """Detect rapid price moves."""
        # TODO: Implement based on your market data
        return False

    def _detect_exchange_outage(self) -> bool:
        """Detect exchange connectivity issues."""
        # TODO: Implement based on your exchange API
        return False

    def _detect_data_corruption(self) -> bool:
        """Detect corrupted market data."""
        # TODO: Implement based on your data pipeline
        return False

    def _cancel_all_pending_orders(self):
        """Cancel all pending orders."""
        # TODO: Implement
        pass

    def _close_all_positions_at_market(self):
        """Close all positions at market price."""
        # TODO: Implement
        pass

    def _send_emergency_alert(self, message: str):
        """Send critical alert (SMS + phone + Slack + email)."""
        # TODO: Implement
        pass

    def _send_warning_alert(self, message: str, risk_reduction: float):
        """Send warning alert (Slack + email)."""
        # TODO: Implement
        pass

    def _send_info_alert(self, message: str):
        """Send info alert (email)."""
        # TODO: Implement
        pass

    def _log_audit_event(self, action: str, user: str, reason: str):
        """Log audit event."""
        # TODO: Implement
        pass

    def _verify_ceo_approval(self, user: str, approval: str) -> bool:
        """Verify CEO approval signature."""
        # TODO: Implement
        return False
```

---

### Integration with Trading Loop

**Example**:

```python
from engine.risk.circuit_breaker import CircuitBreakerEngine

# Initialize
circuit_breaker = CircuitBreakerEngine(config)

# In main trading loop
while True:
    # Update portfolio and market data
    portfolio = get_portfolio_state()
    market_data = get_market_data()

    # Check circuit breakers BEFORE generating signals
    trigger = circuit_breaker.check_all_circuit_breakers(portfolio, market_data)

    if trigger:
        # Determine tier and execute
        if trigger in TIER1_TRIGGERS:
            circuit_breaker.tier1_instant_halt(trigger, portfolio, market_data)
        elif trigger in TIER2_TRIGGERS:
            circuit_breaker.tier2_soft_halt(trigger, risk_reduction=0.5)
        elif trigger in TIER3_TRIGGERS:
            circuit_breaker.tier3_warning(trigger)

    # Only generate signals if trading enabled
    if circuit_breaker.trading_enabled:
        signals = generate_signals()

        # Apply position size multiplier
        for signal in signals:
            signal.size *= circuit_breaker.position_size_multiplier

        execute_signals(signals)

    time.sleep(5)  # Check every 5 seconds
```

---

## 12. Operations Playbook

### Incident Response Procedures

#### Tier 1 Instant Halt Response

**When you receive a Tier 1 alert**:

1. **Acknowledge Alert** (<5 minutes)
   - Respond to SMS/phone call
   - Log into monitoring dashboard

2. **Assess Situation** (5-10 minutes)
   - Review trigger condition
   - Check portfolio state (positions, PnL)
   - Verify market data is valid
   - Review recent trades

3. **Determine Root Cause** (10-20 minutes)
   - Was halt correct?
   - System malfunction or legitimate market event?
   - Check logs for errors

4. **Document Incident** (20-30 minutes)
   - Fill out incident report template
   - Capture screenshots
   - Export logs

5. **Recovery Decision** (30-60 minutes)
   - Can we resume safely?
   - Do we need to close positions?
   - Obtain approval from CEO + Head of Trading

6. **Execute Recovery** (60-90 minutes)
   - Gradual resume (25% → 50% → 75% → 100% over 24h)
   - Monitor closely for first 6 hours

**Total Time to Resume**: 1-2 hours

---

#### Tier 2 Soft Halt Response

**When you receive a Tier 2 alert**:

1. **Acknowledge Alert** (<15 minutes)
2. **Review Metrics** (15-30 minutes)
   - Is condition improving or worsening?
3. **Monitor** (30-120 minutes)
   - If improving: resume after 2 hours
   - If worsening: escalate to Tier 1
4. **Resume** (if conditions improved)
   - Restore to 75% risk, then 100% after 4 hours

---

### Escalation Decision Tree

```
Tier 1 Alert Received
     │
     ▼
Is trigger valid?
     ├─ YES → Proceed with halt
     │         │
     │         ▼
     │    Can we resume safely?
     │         ├─ YES → Gradual resume
     │         └─ NO → Keep halted, escalate to CEO
     │
     └─ NO → False positive
               │
               ▼
          Tune threshold to prevent
          future false positives
```

---

### Team Roles and Responsibilities

| Role | Responsibilities | Authority Level |
|------|------------------|-----------------|
| **Primary On-Call** | First responder, incident assessment | Acknowledge alerts |
| **Head of Trading** | Root cause analysis, recovery approval | Approve Tier 1 resume |
| **CEO** | Final approval for force resume | Override any halt |
| **Engineering** | System diagnostics, code fixes | Fix bugs causing halts |

---

## 13. Dashboard Design

### Real-Time Risk Metrics Display

```
┌─────────────────────────────────────────────────────────────────┐
│  BULL MACHINE - LIVE RISK DASHBOARD                             │
│                                                                  │
│  Trading Status: ● LIVE          Risk Level: ━━━━━━━━━━ 100%   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  CAPITAL PROTECTION METRICS                                     │
│                                                                  │
│  Daily PnL:        +$1,250 (+1.25%)      Limit: -5.0%          │
│  Drawdown:         -8.2%                 Tier 2: 20% | Tier 1: 25% │
│  Win Rate (72h):   58.3%                 Tier 1: <45%          │
│  Fill Rate (24h):  97.2%                 Tier 1: <85%          │
│  Slippage (24h):   0.12%                 Tier 1: >0.5%         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  SYSTEM HEALTH                                                   │
│                                                                  │
│  Archetypes Active:    12/16             Tier 1: <8            │
│  Metadata Integrity:   ✓ OK              Tier 1: FAIL          │
│  Regime Detection:     BULL (stable)     Tier 1: >5 trans/h    │
│  Signal Overlap:       42%               Tier 1: >65%          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  CIRCUIT BREAKER STATUS                                          │
│                                                                  │
│  ○ Tier 3 Warning  ○ Tier 2 Soft Halt  ○ Tier 1 Instant Halt   │
│                                                                  │
│  Last Alert: None                                                │
│  Alerts (24h): 0 Tier 1 | 1 Tier 2 | 3 Tier 3                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  MANUAL CONTROLS                                                 │
│                                                                  │
│  [EMERGENCY STOP]   Risk Dial: ○ 25% ○ 50% ○ 75% ● 100%        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### Alert History Log

```
┌─────────────────────────────────────────────────────────────────┐
│  ALERT HISTORY (Last 7 Days)                                     │
│                                                                  │
│  2025-12-17 14:32:15  │ TIER 1  │ daily_loss_5pct              │
│                       │         │ RESOLVED (1h 3m)              │
│                                                                  │
│  2025-12-16 09:15:00  │ TIER 2  │ fill_rate_85_90pct           │
│                       │         │ RESOLVED (2h 15m)             │
│                                                                  │
│  2025-12-15 18:45:30  │ TIER 3  │ sharpe_below_1.0             │
│                       │         │ RESOLVED (8h 30m)             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary

This kill-switch specification provides **comprehensive capital protection** through:

1. **4-Tier Escalation System**: Graceful degradation from warnings to instant halts
2. **Multi-Dimensional Monitoring**: Performance, system health, execution, market conditions
3. **Fast Response Times**: <1 second for Tier 1 halts
4. **Fail-Safe Design**: Default to halt when uncertain
5. **Human Oversight**: Manual approval required for recovery
6. **Auditability**: All events logged with timestamps and reasons
7. **Testing Framework**: Monthly drills and pre-live validation
8. **Operations Playbook**: Clear procedures for incident response

**Philosophy**: Protect capital first, ask questions later. It's better to halt unnecessarily 10 times than to fail to halt once when needed.

---

**Next Steps**:

1. Implement `CircuitBreakerEngine` class
2. Integrate with trading loop
3. Set up alert infrastructure (SMS, phone, Slack, email)
4. Conduct paper trading with all kill switches enabled
5. Run monthly drills
6. Get CEO approval for thresholds
7. Deploy to production with gradual ramp (25% → 50% → 75% → 100%)

---

**Document Version**: 1.0
**Status**: Ready for Implementation
**Approval Required**: CEO, Head of Trading, Risk Management
**Last Review**: 2025-12-17
