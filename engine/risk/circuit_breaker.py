"""
Circuit Breaker System for Capital Protection.

This module implements a 4-tier escalation system to halt trading
when risk metrics exceed acceptable thresholds.

Tier 1: Instant Halt (CRITICAL) - <1 second response
Tier 2: Soft Halt (WARNING) - Reduce risk 50-75%
Tier 3: Warning (INFO) - Monitor closely
Tier 4: Log (DEBUG) - No action

Author: Bull Machine Risk Team
Version: 1.0
Date: 2025-12-17
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class CircuitBreakerTier(Enum):
    """Circuit breaker escalation tiers."""
    INSTANT_HALT = 1
    SOFT_HALT = 2
    WARNING = 3
    INFO = 4


class TriggerCategory(Enum):
    """Categories of circuit breaker triggers."""
    PERFORMANCE = "performance"
    SYSTEM_HEALTH = "system_health"
    EXECUTION = "execution"
    MARKET_ANOMALY = "market_anomaly"
    CAPITAL_PROTECTION = "capital_protection"
    MANUAL = "manual"


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker trigger."""
    tier: int  # 1-4
    trigger: str  # e.g., "daily_loss_5pct"
    category: str  # performance, system_health, etc.
    timestamp: datetime
    portfolio_state: Dict = field(default_factory=dict)
    market_data: Dict = field(default_factory=dict)
    action_taken: str = ""
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "tier": self.tier,
            "trigger": self.trigger,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "action_taken": self.action_taken,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class CircuitBreakerThresholds:
    """Configurable thresholds for circuit breaker triggers."""

    # Performance thresholds
    daily_loss_pct: float = 0.05  # 5% daily loss -> Tier 1
    daily_loss_warning_pct: float = 0.03  # 3% -> Tier 3
    weekly_loss_pct: float = 0.10  # 10% weekly loss -> Tier 1
    weekly_loss_soft_pct: float = 0.07  # 7% -> Tier 2
    monthly_loss_pct: float = 0.15  # 15% monthly loss -> Tier 1

    drawdown_tier1: float = 0.25  # 25% DD -> instant halt
    drawdown_tier2: float = 0.20  # 20% DD -> soft halt
    drawdown_tier3: float = 0.15  # 15% DD -> warning

    sharpe_tier2: float = 0.5  # Sharpe <0.5 for 5 days -> soft halt
    sharpe_tier3: float = 1.0  # Sharpe <1.0 for 2 days -> warning

    win_rate_tier1: float = 0.45  # <45% for 72h -> instant halt
    win_rate_tier3: float = 0.50  # <50% for 24h -> warning

    # Execution thresholds
    fill_rate_tier1: float = 0.85  # <85% -> instant halt
    fill_rate_tier2: float = 0.90  # 85-90% -> soft halt
    fill_rate_tier3: float = 0.95  # 90-95% -> warning

    slippage_tier1: float = 0.005  # >0.5% -> instant halt
    slippage_tier2: float = 0.003  # 0.3-0.5% -> soft halt
    slippage_tier3: float = 0.0015  # 0.15-0.3% -> warning

    order_failures_tier1: int = 10  # 10+ failures in 10 min -> instant halt
    order_failures_tier2: int = 5  # 5+ -> soft halt
    order_failures_tier3: int = 3  # 3+ -> warning

    # Market anomaly thresholds
    flash_crash_pct: float = 0.10  # 10% move in 5 min -> instant halt
    liquidity_spread_pct: float = 0.01  # 1% bid-ask spread -> soft halt
    api_latency_ms: float = 5000  # 5 second API latency -> instant halt

    # System health thresholds
    archetype_failure_tier1: int = 8  # 8+ failed archetypes -> instant halt
    archetype_failure_tier2: int = 3  # 3+ -> soft halt
    regime_transitions_tier1: int = 5  # >5 transitions in 1h -> instant halt
    signal_overlap_tier1: float = 0.65  # >65% overlap -> instant halt
    signal_overlap_tier2: float = 0.55  # 55-65% -> soft halt

    # Capital protection
    max_risk_per_trade_pct: float = 0.01  # 1% max risk per trade
    max_position_pct: float = 0.10  # 10% max single position
    max_leverage: float = 1.5  # 1.5x max leverage (if unlevered system)


class CircuitBreakerEngine:
    """
    Circuit breaker engine for kill-switch logic.

    Monitors:
    - Performance metrics (drawdown, PnL, Sharpe, win rate)
    - System health (metadata, archetypes, regime)
    - Execution quality (fill rate, slippage, order failures)
    - Market conditions (flash crash, liquidity, data quality)

    Example:
        >>> config = {"log_dir": "logs/circuit_breaker"}
        >>> cb = CircuitBreakerEngine(config)
        >>> trigger = cb.check_all_circuit_breakers(portfolio, market_data)
        >>> if trigger:
        ...     cb.execute_circuit_breaker(trigger, portfolio, market_data)
    """

    def __init__(self, config: Optional[Dict] = None, thresholds: Optional[CircuitBreakerThresholds] = None):
        """
        Initialize circuit breaker engine.

        Args:
            config: Configuration dict with settings
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.config = config or {}
        self.thresholds = thresholds or CircuitBreakerThresholds()

        # State
        self.trading_enabled = True
        self.position_size_multiplier = 1.0
        self.events: List[CircuitBreakerEvent] = []

        # Alert callbacks
        self.alert_callbacks: Dict[str, Callable] = {}

        # Monitoring state
        self.last_check_time = datetime.now()
        self.soft_halt_start_time: Optional[datetime] = None

        # Regime tracking for adaptive thresholds
        self.current_regime = 'neutral'
        self.regime_confidence = 1.0

        # Logging
        self.log_dir = Path(config.get("log_dir", "logs/circuit_breaker"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Circuit Breaker Engine initialized (regime-aware)")
        logger.info(f"  Base drawdown thresholds: T1={self.thresholds.drawdown_tier1*100}%, "
                   f"T2={self.thresholds.drawdown_tier2*100}%, T3={self.thresholds.drawdown_tier3*100}%")

    # ============================================================================
    # REGIME AWARENESS
    # ============================================================================

    def update_regime(self, regime: str, confidence: float = 1.0):
        """
        Update current regime for adaptive threshold adjustment.

        Args:
            regime: 'crisis', 'risk_off', 'neutral', or 'risk_on'
            confidence: Regime confidence (0-1)
        """
        if regime != self.current_regime:
            logger.info(f"[Circuit Breaker] Regime transition: {self.current_regime} → {regime}")
            self.current_regime = regime

        self.regime_confidence = confidence

    def get_adaptive_drawdown_threshold(self, tier: int) -> float:
        """
        Get regime-adjusted drawdown threshold.

        Crisis mode: Tighter thresholds (more sensitive)
        Risk-on mode: Looser thresholds (allow more drawdown)

        Args:
            tier: 1 (instant halt), 2 (soft halt), or 3 (warning)

        Returns:
            Adjusted threshold as decimal (e.g., 0.20 for 20%)
        """
        base_thresholds = {
            1: self.thresholds.drawdown_tier1,  # 25%
            2: self.thresholds.drawdown_tier2,  # 20%
            3: self.thresholds.drawdown_tier3   # 15%
        }

        base = base_thresholds.get(tier, 0.20)

        # Adjust based on regime
        if self.current_regime == 'crisis':
            multiplier = 0.80  # 20% tighter (e.g., 25% → 20%)
        elif self.current_regime == 'risk_off':
            multiplier = 0.90  # 10% tighter
        elif self.current_regime == 'neutral':
            multiplier = 1.00  # No adjustment
        elif self.current_regime == 'risk_on':
            multiplier = 1.10  # 10% looser (allow more drawdown in bull markets)
        else:
            multiplier = 1.00

        adjusted = base * multiplier

        return adjusted

    # ============================================================================
    # MAIN CHECK METHODS
    # ============================================================================

    def check_all_circuit_breakers(self, portfolio, market_data: Dict) -> Optional[str]:
        """
        Check all circuit breaker conditions.

        Returns:
            Trigger name if halt needed, None otherwise
        """
        self.last_check_time = datetime.now()

        # Check in order of severity (Tier 1 first)

        # Capital protection (highest priority)
        if trigger := self._check_capital_protection(portfolio):
            return trigger

        # Market anomalies (external threats)
        if trigger := self._check_market_anomalies(market_data):
            return trigger

        # Execution quality (can't trade safely)
        if trigger := self._check_execution_quality():
            return trigger

        # System health (internal issues)
        if trigger := self._check_system_health():
            return trigger

        # Performance degradation (slowest to trigger)
        if trigger := self._check_performance(portfolio):
            return trigger

        return None

    def execute_circuit_breaker(self, trigger: str, portfolio, market_data: Dict):
        """
        Execute appropriate circuit breaker based on trigger.

        Args:
            trigger: Trigger name (e.g., "daily_loss_5pct")
            portfolio: Portfolio state object
            market_data: Market data dict
        """
        tier = self._get_trigger_tier(trigger)
        category = self._get_trigger_category(trigger)

        if tier == 1:
            self.tier1_instant_halt(trigger, portfolio, market_data, category)
        elif tier == 2:
            risk_reduction = self._get_risk_reduction(trigger)
            self.tier2_soft_halt(trigger, risk_reduction, category)
        elif tier == 3:
            self.tier3_warning(trigger, category)
        else:
            self.tier4_log(trigger, category)

    # ============================================================================
    # PERFORMANCE CHECKS
    # ============================================================================

    def _check_performance(self, portfolio) -> Optional[str]:
        """Check performance-based kill switches."""

        # Daily loss
        daily_pnl_pct = self._get_daily_pnl_pct(portfolio)
        if daily_pnl_pct is not None:
            if daily_pnl_pct < -self.thresholds.daily_loss_pct:
                return "daily_loss_5pct"
            elif daily_pnl_pct < -self.thresholds.daily_loss_warning_pct:
                return "daily_loss_3pct_warning"

        # Weekly loss
        weekly_pnl_pct = self._get_weekly_pnl_pct(portfolio)
        if weekly_pnl_pct is not None:
            if weekly_pnl_pct < -self.thresholds.weekly_loss_pct:
                return "weekly_loss_10pct"
            elif weekly_pnl_pct < -self.thresholds.weekly_loss_soft_pct:
                return "weekly_loss_7pct"

        # Drawdown (regime-aware thresholds)
        dd = self._calculate_drawdown(portfolio)
        if dd is not None:
            dd_t1 = self.get_adaptive_drawdown_threshold(1)
            dd_t2 = self.get_adaptive_drawdown_threshold(2)
            dd_t3 = self.get_adaptive_drawdown_threshold(3)

            if dd > dd_t1:
                logger.critical(f"Drawdown {dd*100:.1f}% > threshold {dd_t1*100:.1f}% (regime={self.current_regime})")
                return "drawdown_25pct"
            elif dd > dd_t2:
                logger.warning(f"Drawdown {dd*100:.1f}% > threshold {dd_t2*100:.1f}% (regime={self.current_regime})")
                return "drawdown_20pct"
            elif dd > dd_t3:
                logger.info(f"Drawdown {dd*100:.1f}% > threshold {dd_t3*100:.1f}% (regime={self.current_regime})")
                return "drawdown_15pct_warning"

        # Win rate (if enough trades)
        trade_count_72h = self._get_trade_count(portfolio, hours=72)
        if trade_count_72h >= 10:
            win_rate = self._get_win_rate(portfolio, hours=72)
            if win_rate is not None:
                if win_rate < self.thresholds.win_rate_tier1:
                    return "win_rate_below_45pct_72h"

        trade_count_24h = self._get_trade_count(portfolio, hours=24)
        if trade_count_24h >= 5:
            win_rate_24h = self._get_win_rate(portfolio, hours=24)
            if win_rate_24h is not None:
                if win_rate_24h < self.thresholds.win_rate_tier3:
                    return "win_rate_below_50pct_24h"

        # Sharpe ratio degradation
        sharpe = self._calculate_sharpe_ratio(portfolio, days=30)
        if sharpe is not None:
            days_below_0_5 = self._days_below_threshold(portfolio, "sharpe", 0.5)
            if sharpe < self.thresholds.sharpe_tier2 and days_below_0_5 >= 5:
                return "sharpe_below_0.5_for_5days"

            days_below_1_0 = self._days_below_threshold(portfolio, "sharpe", 1.0)
            if sharpe < self.thresholds.sharpe_tier3 and days_below_1_0 >= 2:
                return "sharpe_below_1.0_for_48h"

        return None

    # ============================================================================
    # SYSTEM HEALTH CHECKS
    # ============================================================================

    def _check_system_health(self) -> Optional[str]:
        """Check system health kill switches."""

        # Metadata integrity
        if self._detect_metadata_corruption():
            return "metadata_integrity_failure"

        # Archetype failures
        failed_archetypes = self._count_failed_archetypes()
        if failed_archetypes >= self.thresholds.archetype_failure_tier1:
            return f"archetype_failure_cluster_{failed_archetypes}"
        elif failed_archetypes >= self.thresholds.archetype_failure_tier2:
            return f"archetype_degradation_{failed_archetypes}"
        elif failed_archetypes >= 1:
            return "archetype_failure_single"

        # Regime thrashing
        transitions = self._count_regime_transitions(hours=1)
        if transitions > self.thresholds.regime_transitions_tier1:
            return "hmm_thrashing_5plus_transitions"
        elif transitions > 3:
            return "hmm_thrashing_3_transitions"

        # Signal overlap
        overlap = self._calculate_signal_overlap()
        if overlap is not None:
            if overlap > self.thresholds.signal_overlap_tier1:
                return "signal_overlap_65pct"
            elif overlap > self.thresholds.signal_overlap_tier2:
                return "signal_overlap_55pct"
            elif overlap > 0.45:
                return "signal_overlap_45pct_warning"

        return None

    # ============================================================================
    # EXECUTION CHECKS
    # ============================================================================

    def _check_execution_quality(self) -> Optional[str]:
        """Check execution quality kill switches."""

        # Fill rate
        fill_rate = self._calculate_fill_rate(hours=24)
        if fill_rate is not None:
            if fill_rate < self.thresholds.fill_rate_tier1:
                return "fill_rate_below_85pct"
            elif fill_rate < self.thresholds.fill_rate_tier2:
                return "fill_rate_85_90pct"
            elif fill_rate < self.thresholds.fill_rate_tier3:
                return "fill_rate_90_95pct"

        # Slippage
        slippage = self._calculate_average_slippage(hours=24)
        if slippage is not None:
            if slippage > self.thresholds.slippage_tier1:
                return "slippage_above_0.5pct"
            elif slippage > self.thresholds.slippage_tier2:
                return "slippage_0.3_0.5pct"
            elif slippage > self.thresholds.slippage_tier3:
                return "slippage_0.15_0.3pct"

        # Order failures
        failures = self._count_order_failures(minutes=10)
        if failures >= self.thresholds.order_failures_tier1:
            return "order_failures_10plus_in_10min"
        elif failures >= self.thresholds.order_failures_tier2:
            return "order_failures_5_in_10min"
        elif failures >= self.thresholds.order_failures_tier3:
            return "order_failures_3_in_10min"

        return None

    # ============================================================================
    # MARKET ANOMALY CHECKS
    # ============================================================================

    def _check_market_anomalies(self, market_data: Dict) -> Optional[str]:
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

        # Liquidity crisis
        if self._detect_liquidity_crisis(market_data):
            return "liquidity_crisis"

        return None

    # ============================================================================
    # CAPITAL PROTECTION CHECKS
    # ============================================================================

    def _check_capital_protection(self, portfolio) -> Optional[str]:
        """Check capital protection rules."""

        # Position sizing overflow
        if self._detect_position_sizing_overflow(portfolio):
            return "position_sizing_overflow"

        # Leverage breach
        leverage = self._get_leverage(portfolio)
        if leverage is not None and leverage > self.thresholds.max_leverage:
            return "leverage_breach"

        return None

    # ============================================================================
    # TIER EXECUTION METHODS
    # ============================================================================

    def tier1_instant_halt(
        self,
        trigger: str,
        portfolio,
        market_data: Dict,
        category: str = TriggerCategory.PERFORMANCE.value
    ):
        """
        Execute Tier 1 instant halt.

        Actions:
        1. Stop trading immediately (<1 second)
        2. Cancel all pending orders
        3. Log event
        4. Send critical alerts (SMS + phone + Slack + email)
        5. Optionally close positions (flash crash only)
        """
        logger.critical(f"TIER 1 INSTANT HALT: {trigger}")

        # Stop trading
        self.trading_enabled = False

        # Cancel pending orders
        self._cancel_all_pending_orders()

        # Log event
        event = CircuitBreakerEvent(
            tier=1,
            trigger=trigger,
            category=category,
            timestamp=datetime.now(),
            portfolio_state=self._get_portfolio_snapshot(portfolio),
            market_data=market_data,
            action_taken="instant_halt"
        )
        self.events.append(event)
        self._save_event(event)

        # Send alerts
        self._send_emergency_alert(trigger, event)

        # Close positions if flash crash
        if trigger == "flash_crash_detected":
            logger.critical("Flash crash detected - closing all positions at market")
            self._close_all_positions_at_market()

    def tier2_soft_halt(
        self,
        trigger: str,
        risk_reduction: float = 0.5,
        category: str = TriggerCategory.PERFORMANCE.value
    ):
        """
        Execute Tier 2 soft halt (reduce risk).

        Actions:
        1. Reduce position sizes to risk_reduction %
        2. Log event
        3. Send warning alerts (Slack + email)
        4. Schedule escalation check
        """
        logger.warning(f"TIER 2 SOFT HALT: {trigger} - Risk reduced to {risk_reduction*100}%")

        # Reduce position sizes
        self.position_size_multiplier = risk_reduction
        self.soft_halt_start_time = datetime.now()

        # Log event
        event = CircuitBreakerEvent(
            tier=2,
            trigger=trigger,
            category=category,
            timestamp=datetime.now(),
            action_taken=f"soft_halt_{risk_reduction*100}pct"
        )
        self.events.append(event)
        self._save_event(event)

        # Send alerts
        self._send_warning_alert(trigger, risk_reduction, event)

    def tier3_warning(
        self,
        trigger: str,
        category: str = TriggerCategory.PERFORMANCE.value
    ):
        """
        Log Tier 3 warning.

        Actions:
        1. Log warning
        2. Send info alert (email)
        3. Increase monitoring frequency
        """
        logger.info(f"TIER 3 WARNING: {trigger}")

        # Log event
        event = CircuitBreakerEvent(
            tier=3,
            trigger=trigger,
            category=category,
            timestamp=datetime.now(),
            action_taken="warning_logged"
        )
        self.events.append(event)
        self._save_event(event)

        # Send alert
        self._send_info_alert(trigger, event)

    def tier4_log(
        self,
        trigger: str,
        category: str = TriggerCategory.PERFORMANCE.value
    ):
        """Log Tier 4 info (no alert)."""
        logger.debug(f"TIER 4 INFO: {trigger}")

        event = CircuitBreakerEvent(
            tier=4,
            trigger=trigger,
            category=category,
            timestamp=datetime.now(),
            action_taken="logged_only"
        )
        self.events.append(event)

    # ============================================================================
    # MANUAL CONTROLS
    # ============================================================================

    def manual_emergency_stop(
        self,
        user: str,
        reason: str,
        close_positions: bool = False
    ):
        """
        Manual emergency stop button.

        Args:
            user: User who triggered stop
            reason: Reason for stop
            close_positions: Whether to close all positions
        """
        logger.critical(f"MANUAL EMERGENCY STOP by {user}: {reason}")

        # Create fake portfolio/market data for logging
        portfolio_snapshot = {"manual_stop": True}
        market_data = {"manual_stop": True}

        # Halt trading
        self.tier1_instant_halt(
            f"manual_stop_by_{user}",
            portfolio_snapshot,
            market_data,
            category=TriggerCategory.MANUAL.value
        )

        # Optionally close positions
        if close_positions:
            self._close_all_positions_at_market()

        # Audit log
        self._log_audit_event("emergency_stop", user, reason)

    def force_resume(
        self,
        user: str,
        justification: str,
        approval_signature: str
    ):
        """
        Force resume trading (requires CEO approval).

        Args:
            user: User requesting resume
            justification: Written justification
            approval_signature: CEO approval signature
        """
        if not self._verify_ceo_approval(user, approval_signature):
            raise PermissionError("CEO approval required for force resume")

        logger.warning(f"FORCE RESUME by {user}: {justification}")

        # Resume trading
        self.trading_enabled = True
        self.position_size_multiplier = 1.0
        self.soft_halt_start_time = None

        # Audit log
        self._log_audit_event("force_resume", user, justification)

        # Alert team
        self._send_info_alert(f"force_resume_by_{user}", None)

    def set_risk_dial(self, level: str, duration_hours: int = 24):
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
        self.position_size_multiplier = multiplier

        logger.info(f"Risk dial set to {level} for {duration_hours} hours")

        # TODO: Schedule auto-reset (requires background task)
        # For now, caller must manually reset

    # ============================================================================
    # HELPER METHODS (TO BE IMPLEMENTED BY USER)
    # ============================================================================

    def _get_daily_pnl_pct(self, portfolio) -> Optional[float]:
        """Get daily PnL as percentage of capital."""
        # TODO: Implement based on your portfolio object
        if hasattr(portfolio, "get_daily_pnl_pct"):
            return portfolio.get_daily_pnl_pct()
        return None

    def _get_weekly_pnl_pct(self, portfolio) -> Optional[float]:
        """Get weekly PnL as percentage of capital."""
        if hasattr(portfolio, "get_weekly_pnl_pct"):
            return portfolio.get_weekly_pnl_pct()
        return None

    def _calculate_drawdown(self, portfolio) -> Optional[float]:
        """Calculate current drawdown from peak."""
        if hasattr(portfolio, "calculate_drawdown"):
            return portfolio.calculate_drawdown()
        return None

    def _get_trade_count(self, portfolio, hours: int) -> int:
        """Get number of trades in last N hours."""
        if hasattr(portfolio, "get_trade_count"):
            return portfolio.get_trade_count(hours=hours)
        return 0

    def _get_win_rate(self, portfolio, hours: int) -> Optional[float]:
        """Get win rate over last N hours."""
        if hasattr(portfolio, "get_win_rate"):
            return portfolio.get_win_rate(hours=hours)
        return None

    def _calculate_sharpe_ratio(self, portfolio, days: int) -> Optional[float]:
        """Calculate rolling Sharpe ratio."""
        if hasattr(portfolio, "calculate_sharpe_ratio"):
            return portfolio.calculate_sharpe_ratio(days=days)
        return None

    def _days_below_threshold(self, portfolio, metric: str, threshold: float) -> int:
        """Count consecutive days metric below threshold."""
        # TODO: Implement based on your metrics tracking
        return 0

    def _detect_metadata_corruption(self) -> bool:
        """Check if metadata is corrupted."""
        # TODO: Implement based on your feature store
        return False

    def _count_failed_archetypes(self) -> int:
        """Count archetypes with 0 signals in last 24h."""
        # TODO: Implement based on your archetype system
        return 0

    def _count_regime_transitions(self, hours: int) -> int:
        """Count HMM regime transitions in last N hours."""
        # TODO: Implement based on your regime system
        return 0

    def _calculate_signal_overlap(self) -> Optional[float]:
        """Calculate signal overlap percentage."""
        # TODO: Implement based on your signal system
        return None

    def _calculate_fill_rate(self, hours: int) -> Optional[float]:
        """Calculate order fill rate."""
        # TODO: Implement based on your execution system
        return None

    def _calculate_average_slippage(self, hours: int) -> Optional[float]:
        """Calculate average slippage."""
        # TODO: Implement based on your execution system
        return None

    def _count_order_failures(self, minutes: int) -> int:
        """Count recent order failures."""
        # TODO: Implement based on your execution system
        return 0

    def _detect_flash_crash(self, market_data: Dict) -> bool:
        """Detect rapid price moves indicating flash crash."""
        # TODO: Implement based on your market data
        # Example:
        # price_5min_ago = market_data.get("price_5min_ago")
        # current_price = market_data.get("current_price")
        # if price_5min_ago and current_price:
        #     change = abs(current_price - price_5min_ago) / price_5min_ago
        #     return change > self.thresholds.flash_crash_pct
        return False

    def _detect_exchange_outage(self) -> bool:
        """Detect exchange connectivity issues."""
        # TODO: Implement based on your exchange API
        return False

    def _detect_data_corruption(self) -> bool:
        """Detect corrupted market data."""
        # TODO: Implement based on your data pipeline
        return False

    def _detect_liquidity_crisis(self, market_data: Dict) -> bool:
        """Detect liquidity issues (wide spreads, thin orderbook)."""
        # TODO: Implement based on your market data
        return False

    def _detect_position_sizing_overflow(self, portfolio) -> bool:
        """Detect if attempting to allocate >100% capital."""
        if hasattr(portfolio, "get_total_allocated_pct"):
            return portfolio.get_total_allocated_pct() > 1.0
        return False

    def _get_leverage(self, portfolio) -> Optional[float]:
        """Get current portfolio leverage."""
        if hasattr(portfolio, "get_leverage"):
            return portfolio.get_leverage()
        return None

    def _cancel_all_pending_orders(self):
        """Cancel all pending orders."""
        # TODO: Implement based on your execution system
        logger.info("Cancelling all pending orders")

    def _close_all_positions_at_market(self):
        """Close all positions at market price."""
        # TODO: Implement based on your execution system
        logger.critical("Closing all positions at market")

    def _get_portfolio_snapshot(self, portfolio) -> Dict:
        """Get portfolio state snapshot for logging."""
        if hasattr(portfolio, "to_dict"):
            return portfolio.to_dict()
        return {}

    # ============================================================================
    # ALERT METHODS
    # ============================================================================

    def register_alert_callback(self, alert_type: str, callback: Callable):
        """
        Register callback for alerts.

        Args:
            alert_type: "emergency", "warning", "info"
            callback: Function to call with (message, event)
        """
        self.alert_callbacks[alert_type] = callback

    def _send_emergency_alert(self, message: str, event: CircuitBreakerEvent):
        """Send critical alert (SMS + phone + Slack + email)."""
        logger.critical(f"EMERGENCY ALERT: {message}")

        callback = self.alert_callbacks.get("emergency")
        if callback:
            callback(message, event)

    def _send_warning_alert(self, message: str, risk_reduction: float, event: CircuitBreakerEvent):
        """Send warning alert (Slack + email)."""
        logger.warning(f"WARNING ALERT: {message} - Risk reduced to {risk_reduction*100}%")

        callback = self.alert_callbacks.get("warning")
        if callback:
            callback(message, event)

    def _send_info_alert(self, message: str, event: Optional[CircuitBreakerEvent]):
        """Send info alert (email)."""
        logger.info(f"INFO ALERT: {message}")

        callback = self.alert_callbacks.get("info")
        if callback:
            callback(message, event)

    # ============================================================================
    # LOGGING & AUDIT
    # ============================================================================

    def _save_event(self, event: CircuitBreakerEvent):
        """Save event to log file."""
        log_file = self.log_dir / f"circuit_breaker_events_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def _log_audit_event(self, action: str, user: str, reason: str):
        """Log audit event for manual actions."""
        audit_file = self.log_dir / f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

        audit_entry = {
            "action": action,
            "user": user,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

        with open(audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.info(f"AUDIT: {action} by {user}: {reason}")

    def _verify_ceo_approval(self, user: str, approval_signature: str) -> bool:
        """Verify CEO approval signature."""
        # TODO: Implement real signature verification
        # For now, just check if signature is provided
        return len(approval_signature) > 0

    # ============================================================================
    # TRIGGER MAPPING
    # ============================================================================

    def _get_trigger_tier(self, trigger: str) -> int:
        """Get tier level for trigger."""
        tier1_triggers = [
            "daily_loss_5pct", "weekly_loss_10pct", "monthly_loss_15pct",
            "drawdown_25pct", "win_rate_below_45pct_72h",
            "fill_rate_below_85pct", "slippage_above_0.5pct",
            "order_failures_10plus_in_10min",
            "flash_crash_detected", "exchange_outage", "data_corruption",
            "metadata_integrity_failure", "archetype_failure_cluster_8",
            "hmm_thrashing_5plus_transitions", "signal_overlap_65pct",
            "position_sizing_overflow", "leverage_breach"
        ]

        tier2_triggers = [
            "weekly_loss_7pct", "drawdown_20pct",
            "sharpe_below_0.5_for_5days",
            "fill_rate_85_90pct", "slippage_0.3_0.5pct",
            "order_failures_5_in_10min",
            "liquidity_crisis",
            "archetype_degradation_3", "archetype_degradation_4",
            "archetype_degradation_5", "archetype_degradation_6",
            "archetype_degradation_7",
            "hmm_thrashing_3_transitions", "signal_overlap_55pct"
        ]

        tier3_triggers = [
            "daily_loss_3pct_warning", "drawdown_15pct_warning",
            "win_rate_below_50pct_24h", "sharpe_below_1.0_for_48h",
            "fill_rate_90_95pct", "slippage_0.15_0.3pct",
            "order_failures_3_in_10min",
            "archetype_failure_single", "signal_overlap_45pct_warning"
        ]

        if any(t in trigger for t in tier1_triggers):
            return 1
        elif any(t in trigger for t in tier2_triggers):
            return 2
        elif any(t in trigger for t in tier3_triggers):
            return 3
        else:
            return 4

    def _get_trigger_category(self, trigger: str) -> str:
        """Get category for trigger."""
        if "loss" in trigger or "drawdown" in trigger or "sharpe" in trigger or "win_rate" in trigger:
            return TriggerCategory.PERFORMANCE.value
        elif "fill_rate" in trigger or "slippage" in trigger or "order_failures" in trigger:
            return TriggerCategory.EXECUTION.value
        elif "flash_crash" in trigger or "exchange" in trigger or "data_corruption" in trigger or "liquidity" in trigger:
            return TriggerCategory.MARKET_ANOMALY.value
        elif "metadata" in trigger or "archetype" in trigger or "regime" in trigger or "overlap" in trigger:
            return TriggerCategory.SYSTEM_HEALTH.value
        elif "position_sizing" in trigger or "leverage" in trigger:
            return TriggerCategory.CAPITAL_PROTECTION.value
        elif "manual" in trigger:
            return TriggerCategory.MANUAL.value
        else:
            return "unknown"

    def _get_risk_reduction(self, trigger: str) -> float:
        """Get risk reduction multiplier for soft halt trigger."""
        # More severe triggers = more risk reduction
        if "archetype" in trigger or "overlap" in trigger:
            return 0.50  # 50% risk
        elif "drawdown_20" in trigger or "weekly_loss" in trigger:
            return 0.50  # 50% risk
        else:
            return 0.75  # 75% risk

    # ============================================================================
    # STATUS & REPORTING
    # ============================================================================

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            "trading_enabled": self.trading_enabled,
            "position_size_multiplier": self.position_size_multiplier,
            "soft_halt_active": self.soft_halt_start_time is not None,
            "soft_halt_duration_minutes": (
                (datetime.now() - self.soft_halt_start_time).total_seconds() / 60
                if self.soft_halt_start_time else 0
            ),
            "total_events": len(self.events),
            "events_last_24h": len([e for e in self.events if (datetime.now() - e.timestamp).total_seconds() < 86400]),
            "tier1_events_24h": len([e for e in self.events if e.tier == 1 and (datetime.now() - e.timestamp).total_seconds() < 86400]),
            "tier2_events_24h": len([e for e in self.events if e.tier == 2 and (datetime.now() - e.timestamp).total_seconds() < 86400]),
            "tier3_events_24h": len([e for e in self.events if e.tier == 3 and (datetime.now() - e.timestamp).total_seconds() < 86400]),
        }

    def get_recent_events(self, hours: int = 24) -> List[CircuitBreakerEvent]:
        """Get recent circuit breaker events."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self.events if e.timestamp >= cutoff]
