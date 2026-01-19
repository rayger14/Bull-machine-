#!/usr/bin/env python3
"""
System B0 Production Deployment Script

Production-ready deployment system for the Baseline-Conservative strategy.
Supports multiple operational modes with comprehensive error handling and monitoring.

Architecture:
- Data integrity: Validation at every stage
- Fault tolerance: Graceful degradation and recovery
- Security: No hardcoded credentials, secure configuration
- Observability: Comprehensive logging and metrics

Modes:
- backtest: Historical performance validation
- live_signal: Real-time signal generation (no execution)
- paper_trading: Simulated execution with real data
- live_trading: Real execution (requires additional safety checks)

Usage:
    # Backtest mode (default)
    python examples/baseline_production_deploy.py --mode backtest --period 2022-01-01:2024-12-31

    # Live signal monitoring
    python examples/baseline_production_deploy.py --mode live_signal

    # Paper trading
    python examples/baseline_production_deploy.py --mode paper_trading --duration 24h
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

from engine.models.simple_classifier import BuyHoldSellClassifier, Signal, Position
from engine.features.builder import FeatureStoreBuilder


# =============================================================================
# Configuration Management
# =============================================================================

class ConfigLoader:
    """Secure configuration loader with validation."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            ConfigLoader._validate(config)
            return config
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration: {e}")
    
    @staticmethod
    def _validate(config: Dict[str, Any]) -> None:
        """Validate configuration structure."""
        required_sections = ['strategy', 'risk_management', 'execution', 'monitoring']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate strategy parameters
        strategy = config['strategy']
        if 'parameters' not in strategy:
            raise ValueError("Missing strategy.parameters")
        
        # Validate risk management
        risk = config['risk_management']
        required_risk = ['portfolio_size', 'risk_per_trade_pct', 'max_concurrent_positions']
        for param in required_risk:
            if param not in risk:
                raise ValueError(f"Missing risk_management.{param}")


# =============================================================================
# Logging System
# =============================================================================

class ProductionLogger:
    """Production-grade logging with rotation and severity levels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure structured logging."""
        log_config = self.config.get('monitoring', {})
        log_level = getattr(logging, log_config.get('log_level', 'INFO'))
        log_file = log_config.get('log_file', 'logs/system_b0.log')
        
        # Create logger
        logger = logging.getLogger('SystemB0')
        logger.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler (if configured)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_format = logging.Formatter(
                '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            logger.addHandler(file_handler)
        
        return logger
    
    def info(self, msg: str, **kwargs):
        """Log info message."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"{msg}{extra}")
    
    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.warning(f"{msg}{extra}")
    
    def error(self, msg: str, **kwargs):
        """Log error message."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.error(f"{msg}{extra}")
    
    def critical(self, msg: str, **kwargs):
        """Log critical message."""
        extra = f" | {kwargs}" if kwargs else ""
        self.logger.critical(f"{msg}{extra}")


# =============================================================================
# Data Management
# =============================================================================

class DataManager:
    """Manages data loading with caching and validation."""
    
    def __init__(self, config: Dict[str, Any], logger: ProductionLogger):
        self.config = config
        self.logger = logger
        self.builder = FeatureStoreBuilder()
        self._cache = {}
    
    def load_historical_data(self, start: str, end: str) -> pd.DataFrame:
        """Load and validate historical data."""
        self.logger.info(f"Loading historical data: {start} to {end}")
        
        cache_key = f"{start}_{end}"
        if cache_key in self._cache:
            self.logger.info("Using cached data")
            return self._cache[cache_key]
        
        try:
            execution_config = self.config['execution']
            asset = execution_config['asset']
            
            # Load from feature store
            df = self.builder.load(asset, start, end, validate=True)
            
            # Validate required features
            self._validate_features(df)
            
            # Compute additional features if needed
            df = self._ensure_features(df)
            
            self._cache[cache_key] = df
            self.logger.info(f"Loaded {len(df)} bars with {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            raise
    
    def _validate_features(self, df: pd.DataFrame) -> None:
        """Validate required features are present."""
        required = self.config['validation']['required_features']
        missing = set(required) - set(df.columns)
        
        if missing:
            raise ValueError(f"Missing required features: {missing}")
    
    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute any missing derived features."""
        df = df.copy()
        
        # Ensure capitulation_depth (30-day drawdown)
        if 'capitulation_depth' not in df.columns:
            self.logger.info("Computing capitulation_depth feature")
            df['high_30d'] = df['high'].rolling(window=720, min_periods=1).max()  # 30 days in 1H
            df['capitulation_depth'] = (df['close'] - df['high_30d']) / df['high_30d']
        
        # Ensure ATR
        if 'atr_14' not in df.columns and 'atr_20' not in df.columns:
            self.logger.info("Computing ATR feature")
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr_14'] = true_range.rolling(window=14).mean()
        
        return df


# =============================================================================
# Position and Risk Management
# =============================================================================

@dataclass
class TradeResult:
    """Immutable trade result record."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    side: str
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str
    bars_held: int
    metadata: Dict[str, Any]


class RiskManager:
    """Production risk management with circuit breakers."""
    
    def __init__(self, config: Dict[str, Any], logger: ProductionLogger):
        self.config = config
        self.logger = logger
        self.risk_config = config['risk_management']
        self.emergency_config = config['emergency']
        
        # State tracking
        self.portfolio_value = self.risk_config['portfolio_size']
        self.positions: List[Position] = []
        self.trades: List[TradeResult] = []
        self.daily_trades = 0
        self.last_trade_date: Optional[datetime] = None
        self.consecutive_losses = 0
        
    def can_open_position(self, signal: Signal, current_time: datetime) -> tuple[bool, str]:
        """Check if new position can be opened."""
        
        # Check max concurrent positions
        if len(self.positions) >= self.risk_config['max_concurrent_positions']:
            return False, "Max concurrent positions reached"
        
        # Check daily trade limit
        if self.last_trade_date and self.last_trade_date.date() == current_time.date():
            if self.daily_trades >= self.risk_config['max_daily_trades']:
                return False, "Daily trade limit reached"
        else:
            self.daily_trades = 0
            self.last_trade_date = current_time
        
        # Check cooldown period
        if self.trades:
            last_trade_time = self.trades[-1].exit_time
            cooldown_hours = self.risk_config['cooldown_hours']
            if (current_time - last_trade_time).total_seconds() / 3600 < cooldown_hours:
                return False, "Cooldown period active"
        
        # Check portfolio risk
        current_risk = self._calculate_portfolio_risk()
        max_risk = self.risk_config['max_portfolio_risk_pct']
        if current_risk >= max_risk:
            return False, f"Portfolio risk limit reached ({current_risk:.1%})"
        
        # Check emergency kill switch
        if self.emergency_config['kill_switch_enabled']:
            if self._check_emergency_conditions():
                return False, "Emergency kill switch activated"
        
        return True, "OK"
    
    def calculate_position_size(self, signal: Signal) -> float:
        """Calculate position size based on risk parameters."""
        risk_per_trade = self.risk_config['risk_per_trade_pct']
        risk_amount = self.portfolio_value * risk_per_trade
        
        if signal.stop_loss is None:
            return risk_amount
        
        # Position size = Risk Amount / Risk per Unit
        risk_per_unit = abs(signal.entry_price - signal.stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        
        position_size = risk_amount / risk_per_unit
        
        # Cap at max position size
        max_size = self.config['deployment']['safety_checks'].get('max_position_size_usd', float('inf'))
        position_size = min(position_size, max_size)
        
        return position_size
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk exposure."""
        if not self.positions:
            return 0.0
        
        total_risk = sum(
            abs(pos.entry_price - pos.stop_loss) * pos.size 
            for pos in self.positions
        )
        return total_risk / self.portfolio_value
    
    def _check_emergency_conditions(self) -> bool:
        """Check if emergency stop conditions are met."""
        # Check consecutive losses
        max_losses = self.emergency_config['auto_stop_on_consecutive_losses']
        if self.consecutive_losses >= max_losses:
            self.logger.critical(f"Emergency stop: {self.consecutive_losses} consecutive losses")
            return True
        
        # Check drawdown
        if self.trades:
            peak = max([self.portfolio_value] + [
                self.portfolio_value + sum(t.pnl_usd for t in self.trades[:i+1])
                for i in range(len(self.trades))
            ])
            current = self.portfolio_value + sum(t.pnl_usd for t in self.trades)
            drawdown = (peak - current) / peak
            
            max_dd = self.emergency_config['auto_stop_on_drawdown_pct']
            if drawdown >= max_dd:
                self.logger.critical(f"Emergency stop: {drawdown:.1%} drawdown")
                return True
        
        return False
    
    def record_trade(self, trade: TradeResult) -> None:
        """Record completed trade and update state."""
        self.trades.append(trade)
        self.daily_trades += 1
        self.portfolio_value += trade.pnl_usd
        
        # Update consecutive losses
        if trade.pnl_usd < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        self.logger.info(
            f"Trade closed: {trade.exit_reason} | "
            f"PnL: ${trade.pnl_usd:.2f} ({trade.pnl_pct:.2%}) | "
            f"R: {trade.r_multiple:.2f}R"
        )


# =============================================================================
# Backtesting Engine
# =============================================================================

class BacktestEngine:
    """Production-grade backtesting with accurate simulation."""
    
    def __init__(
        self,
        model: BuyHoldSellClassifier,
        data: pd.DataFrame,
        risk_manager: RiskManager,
        config: Dict[str, Any],
        logger: ProductionLogger
    ):
        self.model = model
        self.data = data
        self.risk_manager = risk_manager
        self.config = config
        self.logger = logger
        self.execution_config = config['execution']
        
    def run(self) -> Dict[str, Any]:
        """Execute backtest with accurate trade simulation."""
        self.logger.info("Starting backtest")
        self.logger.info(f"Period: {self.data.index[0]} to {self.data.index[-1]}")
        self.logger.info(f"Bars: {len(self.data)}")
        
        current_position: Optional[Position] = None
        signals_generated = 0
        signals_rejected = 0
        
        for i, (timestamp, bar) in enumerate(self.data.iterrows()):
            # Skip if insufficient history
            if i < self.execution_config['min_bars_history']:
                continue
            
            try:
                # Generate signal
                signal = self.model.predict(bar, current_position)
                
                # Handle exit logic
                if current_position is not None:
                    exit_result = self._check_exit(current_position, bar, timestamp)
                    if exit_result:
                        self.risk_manager.record_trade(exit_result)
                        current_position = None
                        continue
                
                # Handle entry logic
                if signal.is_entry and current_position is None:
                    signals_generated += 1
                    
                    # Risk checks
                    can_trade, reason = self.risk_manager.can_open_position(signal, timestamp)
                    if not can_trade:
                        signals_rejected += 1
                        self.logger.warning(f"Signal rejected: {reason}")
                        continue
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(signal)
                    if position_size <= 0:
                        signals_rejected += 1
                        continue
                    
                    # Open position
                    current_position = Position(
                        direction=signal.direction,
                        entry_price=signal.entry_price,
                        entry_time=timestamp,
                        size=position_size,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        metadata=signal.metadata
                    )
                    
                    self.logger.info(
                        f"Position opened: {signal.direction} @ ${signal.entry_price:.2f} | "
                        f"Size: ${position_size:.2f} | Stop: ${signal.stop_loss:.2f}"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error at bar {i} ({timestamp}): {e}")
                continue
        
        # Close any remaining position
        if current_position is not None:
            final_bar = self.data.iloc[-1]
            final_time = self.data.index[-1]
            exit_result = self._force_exit(current_position, final_bar, final_time, "backtest_end")
            self.risk_manager.record_trade(exit_result)
        
        # Generate performance report
        results = self._calculate_performance()
        results['signals_generated'] = signals_generated
        results['signals_rejected'] = signals_rejected
        results['signal_fill_rate'] = (
            (signals_generated - signals_rejected) / signals_generated 
            if signals_generated > 0 else 0
        )
        
        return results
    
    def _check_exit(
        self, 
        position: Position, 
        bar: pd.Series, 
        timestamp: datetime
    ) -> Optional[TradeResult]:
        """Check exit conditions and simulate exit."""
        
        # Check stop loss
        if position.direction == 'long' and bar['low'] <= position.stop_loss:
            return self._create_trade_result(
                position, 
                position.stop_loss, 
                timestamp, 
                "stop_loss"
            )
        
        # Check profit target (from model parameters)
        profit_pct = (bar['close'] - position.entry_price) / position.entry_price
        profit_target = self.config['strategy']['parameters']['profit_target']
        
        if profit_pct >= profit_target:
            return self._create_trade_result(
                position,
                bar['close'],
                timestamp,
                "profit_target"
            )
        
        return None
    
    def _force_exit(
        self,
        position: Position,
        bar: pd.Series,
        timestamp: datetime,
        reason: str
    ) -> TradeResult:
        """Force exit position at market."""
        return self._create_trade_result(position, bar['close'], timestamp, reason)
    
    def _create_trade_result(
        self,
        position: Position,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> TradeResult:
        """Create trade result with accurate calculations."""
        
        # Apply slippage and commission
        slippage = self.execution_config['slippage_pct']
        commission = self.execution_config['commission_pct']
        
        if position.direction == 'long':
            exit_price_adjusted = exit_price * (1 - slippage - commission)
        else:
            exit_price_adjusted = exit_price * (1 + slippage + commission)
        
        # Calculate PnL
        if position.direction == 'long':
            pnl_usd = (exit_price_adjusted - position.entry_price) * position.size
            pnl_pct = (exit_price_adjusted - position.entry_price) / position.entry_price
        else:
            pnl_usd = (position.entry_price - exit_price_adjusted) * position.size
            pnl_pct = (position.entry_price - exit_price_adjusted) / position.entry_price
        
        # Calculate R-multiple
        risk_per_unit = abs(position.entry_price - position.stop_loss)
        if risk_per_unit > 0:
            if position.direction == 'long':
                r_multiple = (exit_price_adjusted - position.entry_price) / risk_per_unit
            else:
                r_multiple = (position.entry_price - exit_price_adjusted) / risk_per_unit
        else:
            r_multiple = 0.0
        
        bars_held = (exit_time - position.entry_time).total_seconds() / 3600  # Assuming 1H bars
        
        return TradeResult(
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price_adjusted,
            stop_loss=position.stop_loss,
            side=position.direction,
            size_usd=position.size,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            exit_reason=exit_reason,
            bars_held=int(bars_held),
            metadata=position.metadata
        )
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        trades = self.risk_manager.trades
        
        if not trades:
            return {
                'total_trades': 0,
                'error': 'No trades executed'
            }
        
        # Basic metrics
        winning_trades = [t for t in trades if t.pnl_usd > 0]
        losing_trades = [t for t in trades if t.pnl_usd <= 0]
        
        total_profit = sum(t.pnl_usd for t in winning_trades)
        total_loss = abs(sum(t.pnl_usd for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # R-multiple metrics
        avg_r = np.mean([t.r_multiple for t in trades])
        avg_win_r = np.mean([t.r_multiple for t in winning_trades]) if winning_trades else 0
        avg_loss_r = np.mean([t.r_multiple for t in losing_trades]) if losing_trades else 0
        
        # Drawdown calculation
        equity_curve = [self.risk_manager.risk_config['portfolio_size']]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl_usd)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Exit reason breakdown
        exit_reasons = {}
        for trade in trades:
            exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl_usd': sum(t.pnl_usd for t in trades),
            'total_pnl_pct': sum(t.pnl_usd for t in trades) / self.risk_manager.risk_config['portfolio_size'],
            'avg_win_usd': np.mean([t.pnl_usd for t in winning_trades]) if winning_trades else 0,
            'avg_loss_usd': np.mean([t.pnl_usd for t in losing_trades]) if losing_trades else 0,
            'avg_r_multiple': avg_r,
            'avg_win_r': avg_win_r,
            'avg_loss_r': avg_loss_r,
            'max_drawdown': max_drawdown,
            'max_consecutive_losses': max(
                sum(1 for _ in group) 
                for key, group in __import__('itertools').groupby(trades, key=lambda t: t.pnl_usd <= 0)
                if key
            ) if losing_trades else 0,
            'avg_bars_held': np.mean([t.bars_held for t in trades]),
            'exit_reasons': exit_reasons,
            'equity_curve': equity_curve,
            'final_portfolio_value': self.risk_manager.portfolio_value
        }


# =============================================================================
# Main Application
# =============================================================================

class SystemB0:
    """Main application orchestrator."""
    
    def __init__(self, config_path: str):
        self.config = ConfigLoader.load(config_path)
        self.logger = ProductionLogger(self.config)
        self.data_manager = DataManager(self.config, self.logger)
        self.risk_manager = RiskManager(self.config, self.logger)
        
        # Initialize model
        strategy_params = self.config['strategy']['parameters']
        self.model = BuyHoldSellClassifier(**strategy_params)
        
        self.logger.info("=" * 70)
        self.logger.info(f"System B0 Initialized: {self.config['system_name']}")
        self.logger.info("=" * 70)
    
    def run_backtest(self, start: str, end: str) -> Dict[str, Any]:
        """Run historical backtest."""
        self.logger.info(f"Mode: BACKTEST | Period: {start} to {end}")
        
        # Load data
        data = self.data_manager.load_historical_data(start, end)
        
        # Run backtest
        engine = BacktestEngine(
            self.model,
            data,
            self.risk_manager,
            self.config,
            self.logger
        )
        
        results = engine.run()
        
        # Print results
        self._print_results(results)
        
        # Validate against targets
        self._validate_performance(results)
        
        return results
    
    def run_live_signal(self, duration_hours: Optional[int] = None):
        """Run live signal generation (no execution)."""
        self.logger.info("Mode: LIVE_SIGNAL")
        self.logger.warning("Live signal monitoring not yet implemented")
        # TODO: Implement real-time data streaming and signal generation
    
    def run_paper_trading(self, duration_hours: int = 24):
        """Run paper trading simulation."""
        self.logger.info(f"Mode: PAPER_TRADING | Duration: {duration_hours}h")
        self.logger.warning("Paper trading not yet implemented")
        # TODO: Implement paper trading with simulated execution
    
    def _print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted results."""
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST RESULTS")
        self.logger.info("=" * 70)
        self.logger.info(f"Total Trades:        {results['total_trades']}")
        self.logger.info(f"Win Rate:            {results['win_rate']:.1%}")
        self.logger.info(f"Profit Factor:       {results['profit_factor']:.2f}")
        self.logger.info(f"Total PnL:           ${results['total_pnl_usd']:.2f} ({results['total_pnl_pct']:.1%})")
        self.logger.info(f"Max Drawdown:        {results['max_drawdown']:.1%}")
        self.logger.info(f"Avg R-Multiple:      {results['avg_r_multiple']:.2f}R")
        self.logger.info(f"Avg Win:             ${results['avg_win_usd']:.2f} ({results['avg_win_r']:.2f}R)")
        self.logger.info(f"Avg Loss:            ${results['avg_loss_usd']:.2f} ({results['avg_loss_r']:.2f}R)")
        self.logger.info(f"Max Consec Losses:   {results['max_consecutive_losses']}")
        self.logger.info(f"Avg Bars Held:       {results['avg_bars_held']:.1f}")
        self.logger.info("")
        self.logger.info("Exit Reasons:")
        for reason, count in results['exit_reasons'].items():
            self.logger.info(f"  {reason:20s} {count:3d} ({count/results['total_trades']:.1%})")
        self.logger.info("=" * 70)
    
    def _validate_performance(self, results: Dict[str, Any]) -> None:
        """Validate results against performance targets."""
        targets = self.config['performance_targets']
        
        self.logger.info("")
        self.logger.info("Performance Validation:")
        
        checks = [
            ('Profit Factor', results['profit_factor'], targets['min_profit_factor'], '>='),
            ('Win Rate', results['win_rate'] * 100, targets['min_win_rate_pct'], '>='),
            ('Max Drawdown', results['max_drawdown'] * 100, targets['max_drawdown_pct'], '<='),
        ]
        
        all_passed = True
        for name, actual, target, operator in checks:
            if operator == '>=':
                passed = actual >= target
            else:
                passed = actual <= target
            
            status = "PASS" if passed else "FAIL"
            self.logger.info(f"  {name:20s} {actual:8.2f} (target: {operator} {target:.2f}) [{status}]")
            
            if not passed:
                all_passed = False
        
        if all_passed:
            self.logger.info("")
            self.logger.info("All performance targets met")
        else:
            self.logger.warning("")
            self.logger.warning("Some performance targets not met")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='System B0 Production Deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for 2022-2024
  %(prog)s --mode backtest --period 2022-01-01:2024-12-31

  # Run backtest for specific regime
  %(prog)s --mode backtest --period 2022-01-01:2022-12-31  # Bear market

  # Live signal monitoring
  %(prog)s --mode live_signal

  # Paper trading for 24 hours
  %(prog)s --mode paper_trading --duration 24
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/system_b0_production.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['backtest', 'live_signal', 'paper_trading'],
        default='backtest',
        help='Operational mode'
    )
    
    parser.add_argument(
        '--period',
        type=str,
        help='Backtest period (YYYY-MM-DD:YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        help='Duration in hours (for live_signal and paper_trading modes)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        # Initialize system
        system = SystemB0(args.config)
        
        # Run appropriate mode
        if args.mode == 'backtest':
            if not args.period:
                # Default to full test period
                args.period = '2022-01-01:2024-09-30'
            
            start, end = args.period.split(':')
            results = system.run_backtest(start, end)
            
            return 0 if results['total_trades'] > 0 else 1
            
        elif args.mode == 'live_signal':
            system.run_live_signal(args.duration)
            return 0
            
        elif args.mode == 'paper_trading':
            duration = args.duration or 24
            system.run_paper_trading(duration)
            return 0
            
    except Exception as e:
        print(f"\nERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
