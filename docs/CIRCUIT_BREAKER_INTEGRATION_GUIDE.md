# Circuit Breaker Integration Guide

## Quick Start

### 1. Basic Setup

```python
from engine.risk.circuit_breaker import CircuitBreakerEngine, CircuitBreakerThresholds

# Initialize with default thresholds
circuit_breaker = CircuitBreakerEngine(
    config={"log_dir": "logs/circuit_breaker"}
)

# Or with custom thresholds
custom_thresholds = CircuitBreakerThresholds(
    daily_loss_pct=0.03,  # More conservative: 3% instead of 5%
    drawdown_tier1=0.20,  # Halt at 20% instead of 25%
)

circuit_breaker = CircuitBreakerEngine(
    config={"log_dir": "logs/circuit_breaker"},
    thresholds=custom_thresholds
)
```

### 2. Register Alert Callbacks

```python
def send_emergency_alert(message: str, event):
    """Send critical alerts via SMS, phone, Slack, email."""
    # Send SMS
    twilio_client.messages.create(
        to="+1234567890",
        from_="+1111111111",
        body=f"EMERGENCY: {message}"
    )

    # Make phone call
    twilio_client.calls.create(
        to="+1234567890",
        from_="+1111111111",
        twiml=f"<Response><Say>Emergency trading halt: {message}</Say></Response>"
    )

    # Send Slack
    slack_client.chat_postMessage(
        channel="#trading-alerts",
        text=f":rotating_light: **TIER 1 HALT**: {message}"
    )

    # Send email
    send_email(
        to="ceo@company.com",
        subject="EMERGENCY TRADING HALT",
        body=f"Circuit breaker triggered: {message}\n\nEvent: {event.to_dict()}"
    )

def send_warning_alert(message: str, event):
    """Send warning alerts via Slack, email."""
    slack_client.chat_postMessage(
        channel="#trading-alerts",
        text=f":warning: **TIER 2 SOFT HALT**: {message}"
    )

    send_email(
        to="trading-team@company.com",
        subject="Trading Risk Reduction",
        body=f"Circuit breaker soft halt: {message}\n\nEvent: {event.to_dict()}"
    )

def send_info_alert(message: str, event):
    """Send info alerts via email."""
    send_email(
        to="trading-team@company.com",
        subject="Trading Warning",
        body=f"Circuit breaker warning: {message}"
    )

# Register callbacks
circuit_breaker.register_alert_callback("emergency", send_emergency_alert)
circuit_breaker.register_alert_callback("warning", send_warning_alert)
circuit_breaker.register_alert_callback("info", send_info_alert)
```

### 3. Integrate with Trading Loop

```python
def trading_loop():
    """Main trading loop with circuit breaker protection."""

    while True:
        try:
            # Get portfolio and market data
            portfolio = get_portfolio_state()
            market_data = get_market_data()

            # CHECK CIRCUIT BREAKERS FIRST (before generating signals)
            trigger = circuit_breaker.check_all_circuit_breakers(portfolio, market_data)

            if trigger:
                # Execute appropriate circuit breaker action
                circuit_breaker.execute_circuit_breaker(trigger, portfolio, market_data)

            # Only generate signals if trading is enabled
            if circuit_breaker.trading_enabled:
                # Generate signals from all archetypes
                signals = generate_all_signals()

                # Apply position size multiplier (for soft halts)
                for signal in signals:
                    signal.size *= circuit_breaker.position_size_multiplier

                # Execute signals
                execute_signals(signals)

            else:
                logger.info("Trading halted - no new signals generated")

            # Sleep until next check
            time.sleep(5)  # Check every 5 seconds

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            # On unexpected error, halt as a safety measure
            circuit_breaker.manual_emergency_stop(
                user="system",
                reason=f"Unexpected error: {e}",
                close_positions=False
            )
            break
```

### 4. Position Validation Before Entry

```python
def validate_and_execute_trade(signal):
    """Validate trade against capital protection rules before execution."""

    portfolio = get_portfolio_state()

    # Check 1: Single trade risk not >1% of capital
    trade_risk_pct = calculate_trade_risk_pct(signal)
    if trade_risk_pct > 0.01:
        logger.warning(f"Trade rejected: risk {trade_risk_pct*100}% exceeds max 1%")
        return False

    # Check 2: Not attempting >100% capital allocation
    total_allocated = portfolio.get_total_allocated_pct()
    new_allocation = calculate_position_allocation(signal)

    if total_allocated + new_allocation > 1.0:
        logger.warning("Trade rejected: would exceed 100% capital allocation")
        return False

    # Check 3: Single position not >10% of account
    if new_allocation > 0.10:
        logger.warning(f"Trade rejected: position size {new_allocation*100}% exceeds max 10%")
        return False

    # Check 4: Leverage not >1.5x
    if portfolio.get_leverage() > 1.5:
        logger.error("Leverage breach detected - triggering circuit breaker")
        circuit_breaker.tier1_instant_halt(
            "leverage_breach",
            portfolio,
            {},
            category="capital_protection"
        )
        return False

    # All checks passed - execute trade
    execute_trade(signal)
    return True
```

### 5. Manual Controls (Dashboard Integration)

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/api/emergency-stop", methods=["POST"])
def emergency_stop():
    """Emergency stop button endpoint."""

    # Authenticate user
    user = request.json.get("user")
    password = request.json.get("password")
    reason = request.json.get("reason")
    close_positions = request.json.get("close_positions", False)

    if not authenticate(user, password):
        return jsonify({"error": "Authentication failed"}), 401

    # Trigger emergency stop
    circuit_breaker.manual_emergency_stop(
        user=user,
        reason=reason,
        close_positions=close_positions
    )

    return jsonify({
        "status": "success",
        "message": f"Emergency stop triggered by {user}"
    })

@app.route("/api/force-resume", methods=["POST"])
def force_resume():
    """Force resume endpoint (requires CEO approval)."""

    user = request.json.get("user")
    justification = request.json.get("justification")
    approval_signature = request.json.get("approval_signature")

    try:
        circuit_breaker.force_resume(
            user=user,
            justification=justification,
            approval_signature=approval_signature
        )

        return jsonify({
            "status": "success",
            "message": f"Trading resumed by {user}"
        })

    except PermissionError as e:
        return jsonify({"error": str(e)}), 403

@app.route("/api/risk-dial", methods=["POST"])
def set_risk_dial():
    """Risk dial adjustment endpoint."""

    level = request.json.get("level")  # "25%", "50%", "75%", "100%"
    duration_hours = request.json.get("duration_hours", 24)

    circuit_breaker.set_risk_dial(level, duration_hours)

    return jsonify({
        "status": "success",
        "risk_level": level,
        "duration_hours": duration_hours
    })

@app.route("/api/circuit-breaker/status", methods=["GET"])
def get_status():
    """Get circuit breaker status."""
    status = circuit_breaker.get_status()
    return jsonify(status)

@app.route("/api/circuit-breaker/events", methods=["GET"])
def get_events():
    """Get recent circuit breaker events."""
    hours = request.args.get("hours", 24, type=int)
    events = circuit_breaker.get_recent_events(hours=hours)

    return jsonify({
        "events": [e.to_dict() for e in events]
    })
```

### 6. Implement Portfolio Interface

Your portfolio object needs to implement these methods for the circuit breaker to work:

```python
class Portfolio:
    """Portfolio state tracker with circuit breaker interface."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.equity_curve = [initial_capital]
        self.trades = []

    def get_daily_pnl_pct(self) -> float:
        """Get daily PnL as percentage."""
        if len(self.equity_curve) < 2:
            return 0.0

        day_start_equity = self.equity_curve[-1]  # TODO: Get actual day start
        current_equity = self.current_equity

        return (current_equity - day_start_equity) / day_start_equity

    def get_weekly_pnl_pct(self) -> float:
        """Get weekly PnL as percentage."""
        # TODO: Implement based on your equity tracking
        pass

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        peak = max(self.equity_curve)
        current = self.current_equity

        return (peak - current) / peak

    def get_trade_count(self, hours: int) -> int:
        """Get number of trades in last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return sum(1 for trade in self.trades if trade.entry_time >= cutoff)

    def get_win_rate(self, hours: int) -> float:
        """Get win rate over last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_trades = [t for t in self.trades if t.exit_time >= cutoff]

        if not recent_trades:
            return None

        winning_trades = sum(1 for t in recent_trades if t.pnl > 0)
        return winning_trades / len(recent_trades)

    def calculate_sharpe_ratio(self, days: int) -> float:
        """Calculate rolling Sharpe ratio."""
        # Get daily returns for last N days
        daily_returns = []  # TODO: Calculate from equity curve

        if len(daily_returns) < 2:
            return None

        mean_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)

        if std_return == 0:
            return None

        # Annualized Sharpe (assuming daily returns)
        return (mean_return / std_return) * np.sqrt(365)

    def get_total_allocated_pct(self) -> float:
        """Get percentage of capital currently allocated."""
        total_position_value = sum(pos.value for pos in self.open_positions)
        return total_position_value / self.current_equity

    def get_leverage(self) -> float:
        """Get current leverage."""
        total_notional = sum(pos.notional_value for pos in self.open_positions)
        return total_notional / self.current_equity

    def to_dict(self) -> Dict:
        """Serialize portfolio state for logging."""
        return {
            "current_equity": self.current_equity,
            "drawdown": self.calculate_drawdown(),
            "daily_pnl_pct": self.get_daily_pnl_pct(),
            "open_positions": len(self.open_positions),
            "total_allocated_pct": self.get_total_allocated_pct(),
            "leverage": self.get_leverage()
        }
```

---

## Advanced Usage

### Custom Thresholds Per Regime

```python
def get_circuit_breaker_for_regime(regime: str) -> CircuitBreakerEngine:
    """Get circuit breaker with regime-specific thresholds."""

    if regime == "bull":
        # More lenient in bull markets
        thresholds = CircuitBreakerThresholds(
            daily_loss_pct=0.06,  # 6% instead of 5%
            drawdown_tier1=0.30   # 30% instead of 25%
        )
    elif regime == "bear":
        # More conservative in bear markets
        thresholds = CircuitBreakerThresholds(
            daily_loss_pct=0.03,  # 3% instead of 5%
            drawdown_tier1=0.20   # 20% instead of 25%
        )
    else:
        thresholds = CircuitBreakerThresholds()  # Default

    return CircuitBreakerEngine(
        config={"log_dir": f"logs/circuit_breaker/{regime}"},
        thresholds=thresholds
    )
```

### Gradual Resume After Tier 1 Halt

```python
def gradual_resume_from_tier1():
    """Resume trading gradually after Tier 1 halt."""

    # Require manual approval first
    approval = input("Approve resume? (yes/no): ")
    if approval.lower() != "yes":
        logger.info("Resume cancelled")
        return

    # Resume at 25% risk
    circuit_breaker.trading_enabled = True
    circuit_breaker.position_size_multiplier = 0.25
    logger.info("Trading resumed at 25% risk")

    # Schedule ramp-up (pseudo-code - use background scheduler)
    schedule_in_hours(6, lambda: circuit_breaker.set_risk_dial("50%"))
    schedule_in_hours(12, lambda: circuit_breaker.set_risk_dial("75%"))
    schedule_in_hours(18, lambda: circuit_breaker.set_risk_dial("100%"))
```

### Escalation Monitoring for Tier 2 Soft Halts

```python
def monitor_soft_halt_escalation():
    """Check if Tier 2 should escalate to Tier 1."""

    if circuit_breaker.soft_halt_start_time is None:
        return  # No soft halt active

    # Time since soft halt started
    duration = datetime.now() - circuit_breaker.soft_halt_start_time

    if duration.total_seconds() > 2 * 60 * 60:  # 2 hours
        # Check if conditions worsening
        portfolio = get_portfolio_state()
        current_dd = portfolio.calculate_drawdown()

        # If drawdown increased during soft halt → escalate
        if current_dd > 0.22:  # Was 0.20 when soft halt triggered
            logger.warning("Conditions worsening during soft halt - escalating to Tier 1")
            circuit_breaker.tier1_instant_halt(
                "tier2_escalation_worsening_conditions",
                portfolio,
                {},
                category="performance"
            )

        # If conditions improving → resume
        elif current_dd < 0.18:
            logger.info("Conditions improved - resuming from soft halt")
            circuit_breaker.position_size_multiplier = 0.75
            circuit_breaker.soft_halt_start_time = None

            # Full resume after 4 more hours
            schedule_in_hours(4, lambda: circuit_breaker.set_risk_dial("100%"))
```

---

## Testing

### 1. Unit Tests

```python
import pytest
from engine.risk.circuit_breaker import CircuitBreakerEngine, CircuitBreakerThresholds

def test_tier1_daily_loss_trigger():
    """Test Tier 1 triggers on 5% daily loss."""

    cb = CircuitBreakerEngine()

    # Mock portfolio with -5.1% daily loss
    class MockPortfolio:
        def get_daily_pnl_pct(self):
            return -0.051

    portfolio = MockPortfolio()
    trigger = cb.check_all_circuit_breakers(portfolio, {})

    assert trigger == "daily_loss_5pct"

def test_tier2_drawdown_trigger():
    """Test Tier 2 triggers on 20% drawdown."""

    cb = CircuitBreakerEngine()

    class MockPortfolio:
        def get_daily_pnl_pct(self):
            return 0.0
        def calculate_drawdown(self):
            return 0.21  # 21% drawdown

    portfolio = MockPortfolio()
    trigger = cb.check_all_circuit_breakers(portfolio, {})

    assert trigger == "drawdown_20pct"

def test_position_size_multiplier_applied():
    """Test position size multiplier after soft halt."""

    cb = CircuitBreakerEngine()

    # Trigger soft halt
    cb.tier2_soft_halt("test_trigger", risk_reduction=0.5)

    assert cb.position_size_multiplier == 0.5
    assert cb.trading_enabled == True  # Still trading, just reduced
```

### 2. Integration Tests

See `bin/test_circuit_breaker.py` (created below)

### 3. Manual Testing Checklist

**Before going live**:

- [ ] Test Tier 1 instant halt (mock daily loss >5%)
- [ ] Verify trading stops immediately
- [ ] Verify emergency alerts sent (SMS, phone, Slack, email)
- [ ] Test manual emergency stop button
- [ ] Test force resume with CEO approval
- [ ] Test force resume without approval (should fail)
- [ ] Test risk dial at all levels (25%, 50%, 75%, 100%)
- [ ] Test Tier 2 soft halt (mock 20% drawdown)
- [ ] Verify position sizes reduced to 50%
- [ ] Test escalation from Tier 2 to Tier 1
- [ ] Test graceful resume from Tier 2
- [ ] Test all alert channels working
- [ ] Verify events logged correctly
- [ ] Verify audit log for manual actions

---

## Troubleshooting

### Issue: Circuit breaker not triggering

**Diagnosis**:
1. Check if portfolio object implements required methods
2. Verify thresholds are set correctly
3. Check logs for errors

**Solution**:
```python
# Test if portfolio interface works
portfolio = get_portfolio_state()
print(f"Daily PnL: {portfolio.get_daily_pnl_pct()}")
print(f"Drawdown: {portfolio.calculate_drawdown()}")

# Test circuit breaker manually
trigger = circuit_breaker.check_all_circuit_breakers(portfolio, {})
print(f"Trigger: {trigger}")
```

### Issue: False positives (halting unnecessarily)

**Diagnosis**:
Thresholds too tight for your strategy

**Solution**:
```python
# Adjust thresholds to match your risk tolerance
custom_thresholds = CircuitBreakerThresholds(
    daily_loss_pct=0.07,  # Increase from 5% to 7%
    drawdown_tier1=0.30   # Increase from 25% to 30%
)

circuit_breaker = CircuitBreakerEngine(thresholds=custom_thresholds)
```

### Issue: Alerts not sending

**Diagnosis**:
Alert callbacks not registered

**Solution**:
```python
# Verify callbacks registered
print(circuit_breaker.alert_callbacks)

# Re-register if needed
circuit_breaker.register_alert_callback("emergency", send_emergency_alert)
```

---

## Next Steps

1. Implement portfolio interface methods
2. Set up alert infrastructure (Twilio, Slack, email)
3. Test all circuit breakers in paper trading
4. Conduct monthly drills
5. Deploy to production with gradual ramp

**Documentation**: See `/KILL_SWITCH_SPECIFICATION.md` for complete specification.
