"""
Smart Exits Module - v1.8.2
Implements institutional-grade exit management:
- Partial exits at TP1
- Move-to-breakeven after TP1
- ATR trailing stops
- Regime-adaptive stops (ADX-based)
- Liquidity trap protection
- Macro/event safety exits
- Time-based exit guard
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


def atr(df: pd.DataFrame, period: int = 14, method: str = "ema") -> float:
    """Calculate ATR using True Range + EMA smoothing."""
    high = df['High'] if 'High' in df.columns else df['high']
    low = df['Low'] if 'Low' in df.columns else df['low']
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    if method == "ema":
        return tr.ewm(span=period, adjust=False).mean().iloc[-1]
    return tr.rolling(period).mean().iloc[-1]


def calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ADX (Average Directional Index)."""
    high = df['High'] if 'High' in df.columns else df['high']
    low = df['Low'] if 'Low' in df.columns else df['low']
    close = df['Close'] if 'Close' in df.columns else df['close']
    
    if len(df) < period * 2:
        return 20.0  # neutral default
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # True Range
    prev_close = close.shift()
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # Smoothed indicators
    atr_val = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val)
    
    # Directional Index
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx_val = dx.rolling(period).mean()
    
    return float(adx_val.iloc[-1]) if not pd.isna(adx_val.iloc[-1]) else 20.0


def volume_zscore(df: pd.DataFrame, lookback: int = 20) -> float:
    """Calculate volume z-score for liquidity trap detection."""
    if 'volume' not in df.columns and 'Volume' not in df.columns:
        return 0.0
    
    vol = df['volume'] if 'volume' in df.columns else df['Volume']
    mu = vol.rolling(lookback, min_periods=max(5, lookback//3)).mean()
    sd = vol.rolling(lookback, min_periods=max(5, lookback//3)).std().replace(0, np.nan)
    z = (vol - mu) / sd
    
    return float(z.iloc[-1].clip(-3.0, 5.0)) if not pd.isna(z.iloc[-1]) else 0.0


@dataclass
class SmartPosition:
    """Position with smart exit tracking."""
    asset: str
    side: str
    entry_price: float
    qty: float
    full_qty: float  # Original quantity before partials
    stop_price: float
    tp1_price: float
    tp2_price: float
    leverage: float
    entry_time: datetime
    fees_bps: float = 10.0
    slippage_bps: float = 5.0
    
    # Smart exit state
    partial_done: bool = False
    trail_stop: Optional[float] = None
    bars_held: int = 0
    be_triggered: bool = False
    
    # Exit tracking
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl: float = 0.0
    
    @property
    def notional_size(self) -> float:
        return self.entry_price * self.qty
    
    @property
    def current_r(self) -> float:
        """Current R-multiple based on unrealized P&L."""
        if self.stop_price == self.entry_price:
            return 0.0
        
        stop_dist = abs(self.entry_price - self.stop_price)
        if self.side == "long":
            current_dist = self.entry_price - self.stop_price  # Risk distance
        else:
            current_dist = self.stop_price - self.entry_price
        
        return current_dist / stop_dist if stop_dist > 0 else 0.0
    
    def unrealized_r(self, current_price: float) -> float:
        """Calculate unrealized R-multiple."""
        stop_dist = abs(self.entry_price - self.stop_price)
        if stop_dist == 0:
            return 0.0
        
        if self.side == "long":
            price_move = current_price - self.entry_price
        else:
            price_move = self.entry_price - current_price
        
        return price_move / stop_dist
    
    def exit_pnl(self, exit_price: float, qty: float) -> float:
        """Calculate P&L for exiting qty shares."""
        sign = 1 if self.side == "long" else -1
        gross_pnl = sign * (exit_price - self.entry_price) * qty
        
        # Calculate costs (fees + slippage)
        entry_notional = self.entry_price * qty
        exit_notional = exit_price * qty
        total_cost = (entry_notional + exit_notional) * (self.fees_bps + self.slippage_bps) / 10000
        
        return gross_pnl - total_cost


@dataclass
class TradeLog:
    """Complete trade log entry."""
    timestamp: datetime
    asset: str
    side: str
    event: str  # open, partial_exit, TP, SL, trail_hit, macro_event, time_out
    entry_price: float
    exit_price: Optional[float]
    qty: float
    pnl: float
    fees: float
    r_multiple: float
    trail_stop: Optional[float]
    adx_4h: float
    macro_veto: float
    config_hash: str
    
    def to_dict(self) -> Dict:
        return {
            'ts': self.timestamp.isoformat(),
            'asset': self.asset,
            'side': self.side,
            'event': self.event,
            'entry': self.entry_price,
            'exit': self.exit_price,
            'qty': self.qty,
            'pnl': self.pnl,
            'fees': self.fees,
            'r_multiple': self.r_multiple,
            'trail_stop': self.trail_stop,
            'adx_4h': self.adx_4h,
            'macro_veto': self.macro_veto,
            'config_hash': self.config_hash
        }


class SmartExitPortfolio:
    """Portfolio with smart exit management."""
    
    def __init__(self, initial_balance: float, config: Dict, macro_data: Optional[Dict] = None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.config = config
        self.macro_data = macro_data or {}
        
        # PNL tracker config
        self.cfg = config.get('pnl_tracker', {})
        self.exit_cfg = self.cfg.get('exits', {})
        
        # Core parameters
        self.risk_pct = self.cfg.get('risk_per_trade', 0.02)
        self.leverage = self.cfg.get('leverage', 5.0)
        self.atr_period = self.cfg.get('atr_period', 20)
        self.stop_mult = self.cfg.get('stop_buffer_multiplier', 2.0)
        self.r_mult_target = self.cfg.get('r_multiple_target', 2.0)
        self.fees_bps = self.cfg.get('fees_bps', 10.0)
        self.slippage_bps = self.cfg.get('slippage_bps', 5.0)
        self.max_margin_util = self.cfg.get('max_margin_util', 0.50)
        
        # Smart exit parameters
        self.enable_partial = self.exit_cfg.get('enable_partial', True)
        self.scale_out_rr = self.exit_cfg.get('scale_out_rr', 1.0)
        self.scale_out_pct = self.exit_cfg.get('scale_out_pct', 0.5)
        self.move_sl_to_be = self.exit_cfg.get('move_sl_to_be_on_tp1', True)
        
        self.trail_after_tp1 = self.exit_cfg.get('trail_after_tp1', True)
        self.trail_mode = self.exit_cfg.get('trail_mode', 'atr')
        self.trail_atr_mult = self.exit_cfg.get('trail_atr_mult', 1.0)
        
        self.regime_adaptive = self.exit_cfg.get('regime_adaptive', True)
        self.adx_period = self.exit_cfg.get('adx_period', 14)
        self.adx_trend_hi = self.exit_cfg.get('adx_trend_hi', 25.0)
        self.adx_range_lo = self.exit_cfg.get('adx_range_lo', 20.0)
        self.range_stop_factor = self.exit_cfg.get('range_stop_factor', 0.75)
        self.trend_stop_factor = self.exit_cfg.get('trend_stop_factor', 1.25)
        
        self.liq_trap_protect = self.exit_cfg.get('liquidity_trap_protect', True)
        self.liq_z_min = self.exit_cfg.get('liquidity_z_min', 1.3)
        self.liq_lookback = self.exit_cfg.get('liquidity_lookback', 20)
        
        self.macro_exit_enabled = self.exit_cfg.get('macro_exit_enabled', True)
        self.macro_exit_threshold = self.exit_cfg.get('macro_exit_threshold', 0.80)
        self.vix_exit_level = self.exit_cfg.get('vix_exit_level', 30.0)
        
        self.max_bars_in_trade = self.exit_cfg.get('max_bars_in_trade', 96)
        
        # State
        self.positions: Dict[str, SmartPosition] = {}
        self.closed_trades: List[TradeLog] = []
        self.equity_peak = initial_balance
        self.max_dd = 0.0
        
        # Ensure results directory
        Path('results').mkdir(exist_ok=True)
    
    def _log_debug(self, log_type: str, data: Dict):
        """Write debug log to JSONL."""
        log_path = Path('results') / f'{log_type}.jsonl'
        with open(log_path, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _compute_qty_and_levels(self, side: str, entry: float, df_1h: pd.DataFrame) -> tuple:
        """Calculate position size and stop/target levels."""
        import math

        # Calculate ATR
        atr_value = atr(df_1h, period=self.atr_period, method="ema")

        # Guard: ATR invalid
        if atr_value is None or not math.isfinite(atr_value) or atr_value <= 0:
            self._log_debug('open_fail', {
                'reason': 'atr_invalid',
                'atr': atr_value,
                'entry': entry,
                'side': side,
                'df_len': len(df_1h)
            })
            return None, None, None, None

        # Calculate stop distance
        stop_distance = self.stop_mult * atr_value

        # Guard: stop distance invalid
        if stop_distance <= 0:
            self._log_debug('open_fail', {
                'reason': 'stop_distance_nonpositive',
                'stop_mult': self.stop_mult,
                'atr': atr_value,
                'stop_distance': stop_distance
            })
            return None, None, None, None

        stop_price = entry - stop_distance if side == "long" else entry + stop_distance

        # Calculate stop as percentage
        stop_pct = abs(entry - stop_price) / entry

        # Guard: stop_pct invalid
        if stop_pct <= 0 or not math.isfinite(stop_pct):
            self._log_debug('open_fail', {
                'reason': 'stop_pct_invalid',
                'entry': entry,
                'stop_price': stop_price,
                'stop_pct': stop_pct
            })
            return None, None, None, None

        # Risk-based position sizing
        risk_dollars = self.balance * self.risk_pct
        notional = risk_dollars / stop_pct

        # Apply leverage to determine margin
        margin_needed = notional / self.leverage
        max_margin = self.balance * self.max_margin_util

        # Scale down if margin exceeds limit
        if margin_needed > max_margin:
            scale = max_margin / margin_needed
            notional *= scale
            margin_needed = notional / self.leverage

        # Calculate quantity
        qty = notional / entry

        # Guard: qty invalid
        if qty <= 0 or not math.isfinite(qty):
            self._log_debug('open_fail', {
                'reason': 'qty_nonpositive',
                'qty': qty,
                'notional': notional,
                'entry': entry,
                'risk_dollars': risk_dollars,
                'stop_pct': stop_pct,
                'balance': self.balance
            })
            return None, None, None, None

        # Guard: min notional check (avoid dust)
        min_notional = 10.0  # $10 minimum trade
        if notional < min_notional:
            self._log_debug('open_fail', {
                'reason': 'below_min_notional',
                'notional': notional,
                'min_notional': min_notional,
                'qty': qty
            })
            return None, None, None, None

        # Calculate TP levels
        tp1_price = entry + (self.scale_out_rr * stop_distance) if side == "long" else entry - (self.scale_out_rr * stop_distance)
        tp2_price = entry + (self.r_mult_target * stop_distance) if side == "long" else entry - (self.r_mult_target * stop_distance)

        # Log success
        self._log_debug('open_ok', {
            'entry': entry,
            'side': side,
            'atr': atr_value,
            'stop_distance': stop_distance,
            'stop_pct': stop_pct,
            'qty': qty,
            'notional': notional,
            'margin_needed': margin_needed,
            'balance': self.balance,
            'risk_pct': self.risk_pct
        })

        return qty, stop_price, tp1_price, tp2_price
    
    def open_position(self, asset: str, side: str, entry_price: float, df_1h: pd.DataFrame,
                     timestamp: datetime, config_hash: str = "default") -> bool:
        """Open new position with smart exit tracking."""
        if asset in self.positions:
            self._log_debug('open_fail', {
                'reason': 'position_already_exists',
                'asset': asset,
                'timestamp': str(timestamp)
            })
            return False

        qty, stop_price, tp1_price, tp2_price = self._compute_qty_and_levels(side, entry_price, df_1h)

        # Check if sizing failed
        if qty is None:
            self._log_debug('open_fail', {
                'reason': 'sizing_failed',
                'asset': asset,
                'side': side,
                'entry': entry_price,
                'timestamp': str(timestamp)
            })
            return False

        position = SmartPosition(
            asset=asset,
            side=side,
            entry_price=entry_price,
            qty=qty,
            full_qty=qty,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            leverage=self.leverage,
            entry_time=timestamp,
            fees_bps=self.fees_bps,
            slippage_bps=self.slippage_bps
        )

        self.positions[asset] = position

        # Log open
        trade_log = TradeLog(
            timestamp=timestamp,
            asset=asset,
            side=side,
            event='open',
            entry_price=entry_price,
            exit_price=None,
            qty=qty,
            pnl=0.0,
            fees=0.0,
            r_multiple=0.0,
            trail_stop=None,
            adx_4h=0.0,
            macro_veto=0.0,
            config_hash=config_hash
        )
        self._append_trade_log(trade_log)

        logger.info(f"✅ Opened {side} position: {asset} @ ${entry_price:.2f}, qty={qty:.4f}, stop=${stop_price:.2f}")

        return True
    
    def update_positions(self, asset: str, timestamp: datetime, high: float, low: float, 
                        current_price: float, df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                        macro_snapshot: Optional[Dict] = None, config_hash: str = "default"):
        """Update position with smart exit logic."""
        if asset not in self.positions:
            return
        
        pos = self.positions[asset]
        pos.bars_held += 1
        
        # Calculate context
        atr_value = atr(df_1h, period=self.atr_period)
        adx_4h = calc_adx(df_4h, period=self.adx_period)
        vol_z = volume_zscore(df_1h, lookback=self.liq_lookback)
        macro_veto = macro_snapshot.get('veto_strength', 0.0) if macro_snapshot else 0.0
        vix = macro_snapshot.get('vix', 0.0) if macro_snapshot else 0.0
        
        # 1. Check macro/event safety exit FIRST
        if self.macro_exit_enabled:
            if macro_veto > self.macro_exit_threshold or (vix > 0 and vix > self.vix_exit_level):
                self._close_position(pos, current_price, timestamp, 'macro_event', adx_4h, macro_veto, config_hash)
                return
        
        # 2. Partial exit at TP1
        if self.enable_partial and not pos.partial_done:
            unrealized_r = pos.unrealized_r(current_price)
            if unrealized_r >= self.scale_out_rr:
                # Close partial
                partial_qty = pos.full_qty * self.scale_out_pct
                pnl = pos.exit_pnl(current_price, partial_qty)
                fees = partial_qty * current_price * (self.fees_bps + self.slippage_bps) / 10000
                
                self.balance += pnl
                pos.qty -= partial_qty
                pos.partial_done = True
                pos.realized_pnl += pnl
                
                # Log partial
                trade_log = TradeLog(
                    timestamp=timestamp,
                    asset=asset,
                    side=pos.side,
                    event='partial_exit',
                    entry_price=pos.entry_price,
                    exit_price=current_price,
                    qty=partial_qty,
                    pnl=pnl,
                    fees=fees,
                    r_multiple=unrealized_r,
                    trail_stop=pos.trail_stop,
                    adx_4h=adx_4h,
                    macro_veto=macro_veto,
                    config_hash=config_hash
                )
                self._append_trade_log(trade_log)
                
                # Move stop to BE if enabled
                if self.move_sl_to_be:
                    pos.stop_price = pos.entry_price
                    pos.be_triggered = True
        
        # 3. Initialize/update trailing stop after TP1
        if self.trail_after_tp1 and pos.partial_done and self.trail_mode == 'atr':
            trail_dist = self.trail_atr_mult * atr_value
            if pos.side == "long":
                new_trail = current_price - trail_dist
                if pos.trail_stop is None:
                    pos.trail_stop = max(pos.stop_price, new_trail)
                else:
                    pos.trail_stop = max(pos.trail_stop, new_trail)
            else:  # short
                new_trail = current_price + trail_dist
                if pos.trail_stop is None:
                    pos.trail_stop = min(pos.stop_price, new_trail)
                else:
                    pos.trail_stop = min(pos.trail_stop, new_trail)
        
        # 4. Regime-adaptive stop adjustment
        effective_stop = pos.stop_price
        if self.regime_adaptive:
            stop_dist = abs(pos.entry_price - pos.stop_price)
            if adx_4h < self.adx_range_lo:
                # Range: tighten stop
                adjusted_dist = stop_dist * self.range_stop_factor
                effective_stop = pos.entry_price - adjusted_dist if pos.side == "long" else pos.entry_price + adjusted_dist
            elif adx_4h > self.adx_trend_hi:
                # Trend: loosen stop
                adjusted_dist = stop_dist * self.trend_stop_factor
                effective_stop = pos.entry_price - adjusted_dist if pos.side == "long" else pos.entry_price + adjusted_dist
        
        # 5. Liquidity trap protection
        if self.liq_trap_protect and vol_z > self.liq_z_min:
            # Tighten stop near liquidity
            trap_dist = 0.5 * atr_value
            trap_stop = current_price - trap_dist if pos.side == "long" else current_price + trap_dist
            if pos.side == "long":
                effective_stop = max(effective_stop, trap_stop)
            else:
                effective_stop = min(effective_stop, trap_stop)
        
        # 6. Use trailing stop if tighter (long) or wider (short)
        if pos.trail_stop is not None:
            if pos.side == "long":
                effective_stop = max(effective_stop, pos.trail_stop)
            else:
                effective_stop = min(effective_stop, pos.trail_stop)
        
        # 7. Check stop loss
        if pos.side == "long" and low <= effective_stop:
            self._close_position(pos, effective_stop, timestamp, 'SL' if pos.trail_stop is None else 'trail_hit', 
                               adx_4h, macro_veto, config_hash)
            return
        elif pos.side == "short" and high >= effective_stop:
            self._close_position(pos, effective_stop, timestamp, 'SL' if pos.trail_stop is None else 'trail_hit',
                               adx_4h, macro_veto, config_hash)
            return
        
        # 8. Check TP2
        if pos.side == "long" and high >= pos.tp2_price:
            self._close_position(pos, pos.tp2_price, timestamp, 'TP', adx_4h, macro_veto, config_hash)
            return
        elif pos.side == "short" and low <= pos.tp2_price:
            self._close_position(pos, pos.tp2_price, timestamp, 'TP', adx_4h, macro_veto, config_hash)
            return
        
        # 9. Time-based exit guard
        if pos.bars_held >= self.max_bars_in_trade:
            self._close_position(pos, current_price, timestamp, 'time_out', adx_4h, macro_veto, config_hash)
            return
    
    def _close_position(self, pos: SmartPosition, exit_price: float, timestamp: datetime,
                       reason: str, adx_4h: float, macro_veto: float, config_hash: str):
        """Close remaining position."""
        remaining_qty = pos.qty
        pnl = pos.exit_pnl(exit_price, remaining_qty)
        fees = remaining_qty * exit_price * (self.fees_bps + self.slippage_bps) / 10000
        
        self.balance += pnl
        total_pnl = pos.realized_pnl + pnl
        
        # Calculate final R-multiple
        r_mult = pos.unrealized_r(exit_price)
        
        # Log final close
        trade_log = TradeLog(
            timestamp=timestamp,
            asset=pos.asset,
            side=pos.side,
            event=reason,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=remaining_qty,
            pnl=pnl,
            fees=fees,
            r_multiple=r_mult,
            trail_stop=pos.trail_stop,
            adx_4h=adx_4h,
            macro_veto=macro_veto,
            config_hash=config_hash
        )
        self._append_trade_log(trade_log)
        self.closed_trades.append(trade_log)
        
        # Update drawdown
        self._update_drawdown()
        
        # Remove position
        del self.positions[pos.asset]
    
    def force_close_position(self, asset: str, exit_price: float, timestamp: datetime,
                            config_hash: str = "default"):
        """Force close position (e.g., at end of backtest)."""
        if asset not in self.positions:
            return
        
        pos = self.positions[asset]
        self._close_position(pos, exit_price, timestamp, 'forced_close', 0.0, 0.0, config_hash)
    
    def _update_drawdown(self):
        """Update max drawdown."""
        self.equity_peak = max(self.equity_peak, self.balance)
        current_dd = (self.equity_peak - self.balance) / self.equity_peak
        self.max_dd = max(self.max_dd, current_dd)
    
    def _append_trade_log(self, trade_log: TradeLog):
        """Append trade log to JSONL."""
        log_path = Path('results/trade_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(trade_log.to_dict()) + '\n')
    
    def calculate_metrics(self) -> Dict:
        """Calculate final performance metrics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'ending_balance': self.balance,
                'max_drawdown': 0.0,
                'wins': 0,
                'losses': 0
            }
        
        wins = [t for t in self.closed_trades if t.pnl > 0.10]
        losses = [t for t in self.closed_trades if t.pnl < -0.10]
        breakeven = [t for t in self.closed_trades if -0.10 <= t.pnl <= 0.10]
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
        
        win_rate = len(wins) / len(self.closed_trades) * 100
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        return {
            'total_trades': len(self.closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'breakeven': len(breakeven),
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'max_drawdown': self.max_dd * 100,
            'total_return': total_return,
            'ending_balance': self.balance,
            'r_multiples': [t.r_multiple for t in self.closed_trades]
        }
    
    def print_summary(self):
        """Print performance summary."""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 70)
        print("📊 SMART EXITS PERFORMANCE SUMMARY")
        print("=" * 70)
        
        print(f"\n💰 Account:")
        print(f"  Starting: ${self.initial_balance:,.2f}")
        print(f"  Ending:   ${metrics['ending_balance']:,.2f}")
        print(f"  Return:   {metrics['total_return']:+.2f}%")
        print(f"  Max DD:   {metrics['max_drawdown']:.2f}%")
        
        if metrics['total_trades'] > 0:
            print(f"\n📈 Trades:")
            print(f"  Total:    {metrics['total_trades']}")
            print(f"  Wins:     {metrics['wins']} ({metrics['win_rate']:.1f}%)")
            print(f"  Losses:   {metrics['losses']}")
            print(f"  Breakeven: {metrics.get('breakeven', 0)}")
            
            pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "∞"
            print(f"\n💵 Performance:")
            print(f"  PF:       {pf_str}")
            
            if metrics['r_multiples']:
                avg_r = np.mean(metrics['r_multiples'])
                print(f"  Avg R:    {avg_r:+.2f}R")
        
        print("=" * 70)
