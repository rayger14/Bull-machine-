#!/usr/bin/env python3
"""
System B0 Real-Time Monitoring

Production monitoring system for System B0 with:
- Real-time market status tracking
- Entry signal detection
- Position monitoring (PnL, distance to targets)
- Alert system (console, file, webhook)
- Performance metrics dashboard

Architecture:
- Non-blocking updates
- Graceful error handling
- Comprehensive logging
- Alert escalation

Usage:
    # Basic monitoring
    python bin/monitor_system_b0.py

    # Custom check interval
    python bin/monitor_system_b0.py --interval 60

    # Enable webhook alerts
    python bin/monitor_system_b0.py --webhook https://your-webhook-url
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd
import requests

from engine.models.simple_classifier import BuyHoldSellClassifier, Signal, Position


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MarketStatus:
    """Current market status snapshot."""
    timestamp: datetime
    price: float
    capitulation_depth: float
    atr_14: float
    volume_z: float
    high_30d: float
    drawdown_pct: float
    distance_to_entry: float


@dataclass
class PositionStatus:
    """Current position status."""
    entry_time: datetime
    entry_price: float
    current_price: float
    stop_loss: float
    profit_target: float
    pnl_usd: float
    pnl_pct: float
    distance_to_stop_pct: float
    distance_to_target_pct: float
    bars_held: int
    r_multiple: float


@dataclass
class Alert:
    """Alert message."""
    timestamp: datetime
    severity: str  # INFO, WARNING, CRITICAL
    category: str  # SIGNAL, POSITION, RISK, SYSTEM
    message: str
    data: Dict[str, Any]


# =============================================================================
# Monitoring Engine
# =============================================================================

class MonitoringEngine:
    """Real-time monitoring engine with alerting."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_config = config['monitoring']
        
        # Initialize model
        strategy_params = config['strategy']['parameters']
        self.model = BuyHoldSellClassifier(**strategy_params)
        
        # State
        self.current_position: Optional[Position] = None
        self.alerts: List[Alert] = []
        self.last_signal_time: Optional[datetime] = None
        self.signal_count = 0
        self.check_count = 0
        
        # Setup logging
        self._setup_logging()
        
        self.log_info("Monitoring engine initialized")
    
    def _setup_logging(self):
        """Setup logging to console and file."""
        import logging
        
        log_level = self.monitoring_config.get('log_level', 'INFO')
        log_file = self.monitoring_config.get('log_file', 'logs/system_b0_monitor.log')
        
        self.logger = logging.getLogger('MonitorB0')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        ))
        self.logger.addHandler(console)
        
        # File handler
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(file_handler)
    
    def log_info(self, msg: str):
        self.logger.info(msg)
    
    def log_warning(self, msg: str):
        self.logger.warning(msg)
    
    def log_critical(self, msg: str):
        self.logger.critical(msg)
    
    def check_market(self, bar: pd.Series) -> MarketStatus:
        """Check current market conditions."""
        
        # Calculate metrics
        price = bar['close']
        capitulation_depth = bar.get('capitulation_depth', 0.0)
        atr_14 = bar.get('atr_14', bar.get('atr_20', price * 0.02))
        volume_z = bar.get('volume_z', 0.0)
        high_30d = bar.get('high_30d', price)
        
        drawdown_pct = capitulation_depth
        distance_to_entry = abs(drawdown_pct - self.config['strategy']['parameters']['buy_threshold'])
        
        status = MarketStatus(
            timestamp=bar.name if hasattr(bar, 'name') else datetime.now(),
            price=price,
            capitulation_depth=capitulation_depth,
            atr_14=atr_14,
            volume_z=volume_z,
            high_30d=high_30d,
            drawdown_pct=drawdown_pct,
            distance_to_entry=distance_to_entry
        )
        
        return status
    
    def check_signal(self, bar: pd.Series) -> Optional[Signal]:
        """Check for entry/exit signals."""
        
        signal = self.model.predict(bar, self.current_position)
        
        if signal.is_entry:
            self.signal_count += 1
            self.last_signal_time = datetime.now()
            
            # Create alert
            alert = Alert(
                timestamp=datetime.now(),
                severity='WARNING',
                category='SIGNAL',
                message=f"Entry signal detected: {signal.direction} @ ${signal.entry_price:.2f}",
                data={
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'confidence': signal.confidence,
                    'metadata': signal.metadata
                }
            )
            self._send_alert(alert)
            
            return signal
        
        return None
    
    def check_position(self, bar: pd.Series) -> Optional[PositionStatus]:
        """Check current position status."""
        
        if self.current_position is None:
            return None
        
        current_price = bar['close']
        profit_target_price = self.current_position.entry_price * (
            1 + self.config['strategy']['parameters']['profit_target']
        )
        
        pnl_usd = (current_price - self.current_position.entry_price) * self.current_position.size
        pnl_pct = (current_price - self.current_position.entry_price) / self.current_position.entry_price
        
        distance_to_stop = (current_price - self.current_position.stop_loss) / current_price
        distance_to_target = (profit_target_price - current_price) / current_price
        
        risk_per_unit = abs(self.current_position.entry_price - self.current_position.stop_loss)
        r_multiple = (current_price - self.current_position.entry_price) / risk_per_unit if risk_per_unit > 0 else 0
        
        bars_held = (datetime.now() - self.current_position.entry_time).total_seconds() / 3600
        
        status = PositionStatus(
            entry_time=self.current_position.entry_time,
            entry_price=self.current_position.entry_price,
            current_price=current_price,
            stop_loss=self.current_position.stop_loss,
            profit_target=profit_target_price,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            distance_to_stop_pct=distance_to_stop,
            distance_to_target_pct=distance_to_target,
            bars_held=int(bars_held),
            r_multiple=r_multiple
        )
        
        # Check for warning conditions
        warn_threshold = self.monitoring_config['thresholds'].get('profit_target_distance_warn', 0.02)
        if distance_to_target < warn_threshold and pnl_pct > 0:
            alert = Alert(
                timestamp=datetime.now(),
                severity='INFO',
                category='POSITION',
                message=f"Position near profit target: {distance_to_target:.1%} away",
                data=asdict(status)
            )
            self._send_alert(alert)
        
        # Check stop loss proximity
        if distance_to_stop < 0.03:  # Within 3% of stop
            alert = Alert(
                timestamp=datetime.now(),
                severity='WARNING',
                category='POSITION',
                message=f"Position near stop loss: {distance_to_stop:.1%} away",
                data=asdict(status)
            )
            self._send_alert(alert)
        
        return status
    
    def run_check(self, bar: pd.Series) -> Dict[str, Any]:
        """Run complete monitoring check."""
        
        self.check_count += 1
        
        # Check market conditions
        market_status = self.check_market(bar)
        
        # Check for signals
        signal = self.check_signal(bar)
        
        # Check position (if any)
        position_status = self.check_position(bar) if self.current_position else None
        
        # Build status report
        report = {
            'check_number': self.check_count,
            'timestamp': datetime.now().isoformat(),
            'market': asdict(market_status),
            'signal': asdict(signal) if signal else None,
            'position': asdict(position_status) if position_status else None,
            'stats': {
                'total_signals': self.signal_count,
                'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
                'has_position': self.current_position is not None
            }
        }
        
        return report
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        
        self.alerts.append(alert)
        
        # Console alert
        if self.monitoring_config['alerts'].get('console', True):
            level_map = {
                'INFO': self.log_info,
                'WARNING': self.log_warning,
                'CRITICAL': self.log_critical
            }
            log_func = level_map.get(alert.severity, self.log_info)
            log_func(f"[{alert.category}] {alert.message}")
        
        # File alert
        if self.monitoring_config['alerts'].get('file', True):
            alert_file = Path(self.monitoring_config.get('log_file', 'logs/system_b0_monitor.log')).parent / 'alerts.jsonl'
            alert_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(alert_file, 'a') as f:
                alert_data = asdict(alert)
                alert_data['timestamp'] = alert.timestamp.isoformat()
                f.write(json.dumps(alert_data) + '\n')
        
        # Webhook alert
        if self.monitoring_config['alerts'].get('webhook', False):
            webhook_url = self.monitoring_config['alerts'].get('webhook_url')
            if webhook_url:
                try:
                    payload = {
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity,
                        'category': alert.category,
                        'message': alert.message,
                        'data': alert.data
                    }
                    requests.post(webhook_url, json=payload, timeout=5)
                except Exception as e:
                    self.log_warning(f"Webhook alert failed: {e}")
    
    def print_dashboard(self, report: Dict[str, Any]):
        """Print monitoring dashboard."""
        
        print("\n" + "=" * 80)
        print(f"SYSTEM B0 MONITORING DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Market section
        market = report['market']
        print("\nMARKET STATUS:")
        print(f"  Price:              ${market['price']:,.2f}")
        print(f"  30d High:           ${market['high_30d']:,.2f}")
        print(f"  Drawdown:           {market['drawdown_pct']:.2%}")
        print(f"  Distance to Entry:  {market['distance_to_entry']:.2%}")
        print(f"  ATR(14):            ${market['atr_14']:.2f}")
        print(f"  Volume Z-Score:     {market['volume_z']:.2f}")
        
        # Entry threshold indicator
        buy_threshold = self.config['strategy']['parameters']['buy_threshold']
        if market['drawdown_pct'] < buy_threshold:
            print(f"\n  >> ENTRY CONDITION MET (Drawdown < {buy_threshold:.1%}) <<")
        else:
            distance_pct = (market['drawdown_pct'] - buy_threshold) / abs(buy_threshold) * 100
            print(f"\n  Waiting for entry ({distance_pct:.1f}% away from threshold)")
        
        # Position section
        if report['position']:
            pos = report['position']
            print("\nPOSITION STATUS:")
            print(f"  Entry Price:        ${pos['entry_price']:,.2f}")
            print(f"  Current Price:      ${pos['current_price']:,.2f}")
            print(f"  Stop Loss:          ${pos['stop_loss']:,.2f}")
            print(f"  Profit Target:      ${pos['profit_target']:,.2f}")
            print(f"  PnL:                ${pos['pnl_usd']:+,.2f} ({pos['pnl_pct']:+.2%})")
            print(f"  R-Multiple:         {pos['r_multiple']:+.2f}R")
            print(f"  Distance to Stop:   {pos['distance_to_stop_pct']:.2%}")
            print(f"  Distance to Target: {pos['distance_to_target_pct']:.2%}")
            print(f"  Bars Held:          {pos['bars_held']}")
        else:
            print("\nPOSITION STATUS: No open position")
        
        # Stats section
        stats = report['stats']
        print("\nSTATISTICS:")
        print(f"  Total Checks:       {self.check_count}")
        print(f"  Total Signals:      {stats['total_signals']}")
        print(f"  Last Signal:        {stats['last_signal_time'] or 'Never'}")
        
        print("=" * 80 + "\n")


# =============================================================================
# CLI Entry Point
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_latest_bar(config: Dict[str, Any]) -> pd.Series:
    """Load latest market data bar."""
    # TODO: Implement real-time data loading
    # For now, return dummy data
    return pd.Series({
        'close': 50000.0,
        'high': 50500.0,
        'low': 49500.0,
        'volume': 1000000,
        'capitulation_depth': -0.05,
        'atr_14': 2000.0,
        'volume_z': 1.5,
        'high_30d': 52500.0
    }, name=datetime.now())


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='System B0 Real-Time Monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/system_b0_production.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        help='Check interval in seconds (default: from config)'
    )
    
    parser.add_argument(
        '--webhook',
        type=str,
        help='Webhook URL for alerts'
    )
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (useful for testing)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override webhook if provided
        if args.webhook:
            config['monitoring']['alerts']['webhook'] = True
            config['monitoring']['alerts']['webhook_url'] = args.webhook
        
        # Get check interval
        interval = args.interval or config['monitoring'].get('check_interval_seconds', 300)
        
        # Initialize monitoring engine
        engine = MonitoringEngine(config)
        
        engine.log_info("=" * 80)
        engine.log_info("SYSTEM B0 MONITORING STARTED")
        engine.log_info("=" * 80)
        engine.log_info(f"Check interval: {interval}s")
        engine.log_info(f"Alerts: Console={config['monitoring']['alerts']['console']}, "
                       f"File={config['monitoring']['alerts']['file']}, "
                       f"Webhook={config['monitoring']['alerts']['webhook']}")
        
        # Monitoring loop
        while True:
            try:
                # Load latest bar
                bar = load_latest_bar(config)
                
                # Run monitoring check
                report = engine.run_check(bar)
                
                # Print dashboard
                engine.print_dashboard(report)
                
                # Exit if single run
                if args.once:
                    break
                
                # Wait for next check
                time.sleep(interval)
                
            except KeyboardInterrupt:
                engine.log_info("\nMonitoring stopped by user")
                break
                
            except Exception as e:
                engine.log_critical(f"Monitoring error: {e}")
                if args.once:
                    raise
                # Continue monitoring on error
                time.sleep(interval)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
