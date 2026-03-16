#!/usr/bin/env python3
"""
Structural-Only Backtester

Tests the raw structural quality of each archetype by bypassing:
  - Fusion threshold filtering (CMI dynamic threshold)
  - Crisis penalty
  - Position limits
  - Cooling periods
  - Entry spacing
  - Portfolio allocation
  - Signal deduplication

Keeps: structural checks (_check_*), hard gates, exit logic (SL/TP/trailing/time)

This answers: "If every structurally valid signal traded, what would each
archetype's raw performance be?"

Usage:
    python3 bin/backtest_structural.py --start-date 2020-01-01
    python3 bin/backtest_structural.py --start-date 2020-01-01 --archetype wick_trap,spring
    python3 bin/backtest_structural.py --start-date 2020-01-01 --gates none
    python3 bin/backtest_structural.py --start-date 2020-01-01 --compare

Author: Claude Code
Date: 2026-03-16
"""

import sys
import json
import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/bull_machine_isolated_v11_fixed.json"
DEFAULT_FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"


# ── Lightweight Position & Trade (avoid importing full backtester) ────

@dataclass
class Position:
    """Tracked position for structural backtester."""
    position_id: str
    archetype: str
    direction: str
    entry_price: float
    entry_time: pd.Timestamp
    stop_loss: float
    take_profit: float
    original_quantity: float
    current_quantity: float
    fusion_score: float
    atr_at_entry: float
    bars_held: int = 0
    executed_scale_outs: List[float] = field(default_factory=list)
    total_exits_pct: float = 0.0
    trailing_stop: Optional[float] = None
    margin_used: float = 0.0
    position_size_usd: float = 0.0
    entry_metadata: Dict[str, Any] = field(default_factory=dict)
    runner_trailing_stop: Optional[float] = None


@dataclass
class Trade:
    """Completed trade record."""
    timestamp_entry: pd.Timestamp
    timestamp_exit: pd.Timestamp
    archetype: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    duration_hours: float
    fusion_score: float
    exit_reason: str
    stop_loss: float = 0.0
    take_profit: float = 0.0


class _PosAdapter:
    """Bridge Position to ExitLogic's expected interface."""
    def __init__(self, pos: Position):
        self._pos = pos
        self.entry_price = pos.entry_price
        self.entry_time = pos.entry_time
        self.stop_loss = pos.stop_loss
        self.direction = pos.direction
        self.runner_trailing_stop = pos.runner_trailing_stop
        self.metadata = dict(pos.entry_metadata)
        self.metadata['executed_scale_outs'] = list(pos.executed_scale_outs)


# ── Structural Backtester ─────────────────────────────────────────────

class StructuralBacktester:
    """Backtest that tests pure structural signal quality.

    Every signal that passes structural check + gates (configurable) gets
    traded. No fusion threshold, no CMI, no position limits, no cooling.
    """

    def __init__(self, config: Dict, features_df: pd.DataFrame,
                 initial_cash: float = 100_000.0,
                 commission_rate: float = 0.0002,
                 slippage_bps: float = 3.0,
                 gate_mode: str = 'hard',
                 archetype_filter: Optional[List[str]] = None,
                 max_concurrent: int = 0):

        self.config = config
        self.features_df = features_df
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.gate_mode = gate_mode  # 'hard', 'soft', 'none'
        self.archetype_filter = archetype_filter
        self.max_concurrent = max_concurrent  # 0 = unlimited

        # Position sizing
        sizing_cfg = config.get('position_sizing', {})
        self.risk_per_trade = sizing_cfg.get('risk_per_trade_pct', 0.02)
        self.leverage = config.get('leverage', 1.5)
        self.max_margin_pct = sizing_cfg.get('max_margin_per_position_pct', 0.35)

        # State
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_cash]
        self.equity_timestamps: List[pd.Timestamp] = []

        # Per-archetype tracking
        self.arch_trades: Dict[str, List[Trade]] = {}
        self.arch_equity: Dict[str, List[float]] = {}

        # Stats
        self.raw_signals = 0
        self.gate_rejected = 0
        self.signals_traded = 0

        # Initialize engine (for structural checks + gates)
        self._init_engine()
        self._init_exit_logic()

        # Precompute numpy arrays for stop-loss checks
        self._highs = features_df['high'].values if 'high' in features_df.columns else None
        self._lows = features_df['low'].values if 'low' in features_df.columns else None

    def _init_engine(self):
        """Initialize the archetype engine for signal generation."""
        from engine.integrations.isolated_archetype_engine import IsolatedArchetypeEngine

        archetype_config_dir = self.config.get('archetype_config_dir', 'configs/archetypes/')

        # Ensure structural checks are enabled
        if 'structural_checks' not in self.config:
            self.config['structural_checks'] = {}
        self.config['structural_checks'].setdefault('mode_context', 'backtest')
        self.config['structural_checks'].setdefault('enabled', True)

        self.engine = IsolatedArchetypeEngine(
            archetype_config_dir=archetype_config_dir,
            portfolio_config={},  # No portfolio allocation
            enable_regime=False,  # No regime service needed
            config=self.config,
        )

        # Disable cooling periods on all archetypes
        for name, arch in self.engine.archetypes.items():
            arch._cooling_bar = -9999  # Allow immediate re-fire

        # Apply archetype filter
        if self.archetype_filter:
            filtered = {k: v for k, v in self.engine.archetypes.items()
                        if k in self.archetype_filter}
            if not filtered:
                print(f"WARNING: No archetypes match filter {self.archetype_filter}")
                print(f"Available: {list(self.engine.archetypes.keys())}")
            self.engine.archetypes = filtered

        # Override gate mode if needed
        if self.gate_mode == 'none':
            for name, arch in self.engine.archetypes.items():
                arch.config.hard_gates = []  # Remove all gates
        elif self.gate_mode == 'soft':
            for name, arch in self.engine.archetypes.items():
                arch.config.gate_mode = 'soft'

        print(f"Archetypes loaded: {list(self.engine.archetypes.keys())} ({len(self.engine.archetypes)})")

    def _init_exit_logic(self):
        """Initialize exit logic (identical to production)."""
        from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config
        exit_config = create_default_exit_config()
        if 'exit_logic' in self.config:
            exit_config.update(self.config['exit_logic'])
        self.exit_logic = ExitLogic(exit_config)

    # ── Main Loop ─────────────────────────────────────────────────────

    def run(self, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Run the structural backtest."""
        df = self.features_df
        if start_date:
            ts_start = pd.Timestamp(start_date)
            if df.index.tz is not None:
                ts_start = ts_start.tz_localize(df.index.tz)
            df = df[df.index >= ts_start]
        if end_date:
            ts_end = pd.Timestamp(end_date)
            if df.index.tz is not None:
                ts_end = ts_end.tz_localize(df.index.tz)
            df = df[df.index <= ts_end]

        if len(df) < 100:
            print(f"ERROR: Only {len(df)} bars after date filter. Need 100+.")
            return

        print(f"Running structural backtest: {len(df):,} bars, "
              f"{df.index[0].date()} to {df.index[-1].date()}")
        print(f"Gate mode: {self.gate_mode} | "
              f"Max concurrent: {'unlimited' if self.max_concurrent == 0 else self.max_concurrent}")
        print()

        t0 = time.time()
        lookback_size = 100

        for bar_idx in range(1, len(df)):
            ts = df.index[bar_idx]
            row = df.iloc[bar_idx]
            prev_row = df.iloc[bar_idx - 1]

            # Lookback for structural checks
            lb_start = max(0, bar_idx - lookback_size)
            lookback_df = df.iloc[lb_start:bar_idx]

            # 1. Update bars_held for open positions
            for pos in self.positions.values():
                pos.bars_held += 1

            # 2. Check exits for all open positions
            self._check_all_exits(row, ts, bar_idx)

            # 3. Generate signals (structural + gates, NO threshold filter)
            self._generate_and_trade_signals(row, prev_row, lookback_df, ts, bar_idx)

            # 4. Update equity
            equity = self._compute_equity(row['close'])
            self.equity_curve.append(equity)
            self.equity_timestamps.append(ts)

            # Progress logging
            if bar_idx % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  bar {bar_idx:,}/{len(df):,} | "
                      f"positions={len(self.positions)} | "
                      f"trades={len(self.trades)} | "
                      f"signals={self.raw_signals} | "
                      f"{elapsed:.0f}s")

        # Close remaining positions at last bar's close
        last_row = df.iloc[-1]
        last_ts = df.index[-1]
        for pos_id in list(self.positions.keys()):
            self._close_position(pos_id, last_row['close'], last_ts, "end_of_data", 1.0)

        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    def _generate_and_trade_signals(self, row, prev_row, lookback_df, ts, bar_idx):
        """Generate structural signals and open positions for every passing signal."""
        features = row.to_dict() if hasattr(row, 'to_dict') else dict(row)

        for name, archetype in self.engine.archetypes.items():
            # Reset cooling so it never blocks
            archetype._cooling_bar = -9999

            # Call detect() which runs structural check + gates + fusion
            signal = archetype.detect(
                features, 'neutral',  # regime doesn't matter — we bypass threshold
                current_bar_idx=bar_idx,
                prev_row=prev_row,
                lookback_df=lookback_df,
                structural_checker=self.engine.structural_checker,
            )

            if signal is None:
                continue

            self.raw_signals += 1

            # Max concurrent check (per archetype)
            if self.max_concurrent > 0:
                arch_positions = sum(1 for p in self.positions.values() if p.archetype == name)
                if arch_positions >= self.max_concurrent:
                    continue

            # Open position — NO threshold check, NO crisis penalty, NO position limit
            fusion = signal.metadata.get('fusion_score', 0.5)
            self._open_position(
                timestamp=ts,
                archetype=name,
                direction=signal.direction,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                fusion_score=fusion,
                features=row,
                bar_idx=bar_idx,
            )

    # ── Position Management ───────────────────────────────────────────

    def _open_position(self, timestamp, archetype, direction, entry_price,
                       stop_loss, take_profit, fusion_score, features, bar_idx):
        """Open a position with fixed risk sizing."""
        atr = features.get('atr_14', entry_price * 0.02)
        if pd.isna(atr) or atr <= 0:
            atr = entry_price * 0.02

        if pd.isna(stop_loss) or stop_loss <= 0:
            stop_loss = entry_price - (atr * 2.0) if direction == 'long' else entry_price + (atr * 2.0)
        if pd.isna(take_profit) or take_profit <= 0:
            take_profit = entry_price + (atr * 4.0) if direction == 'long' else entry_price - (atr * 4.0)

        stop_distance_pct = abs(entry_price - stop_loss) / entry_price
        if pd.isna(stop_distance_pct) or stop_distance_pct <= 0:
            stop_distance_pct = 0.025

        # Fixed risk sizing (same formula as production)
        risk_dollars = self.initial_cash * self.risk_per_trade
        notional = risk_dollars / stop_distance_pct
        margin = notional / self.leverage

        # Cap margin
        max_margin = self.initial_cash * self.max_margin_pct
        if margin > max_margin:
            margin = max_margin
            notional = margin * self.leverage

        # Check margin availability
        commission = notional * self.commission_rate
        slippage = notional * (self.slippage_bps / 10000.0)
        margin_cost = margin + commission + slippage

        if margin_cost > self.cash:
            return  # Skip if insufficient cash

        # Apply slippage
        if direction == 'long':
            fill_price = entry_price * (1 + self.slippage_bps / 10000.0)
        else:
            fill_price = entry_price * (1 - self.slippage_bps / 10000.0)

        quantity = notional / fill_price
        self.cash -= margin_cost

        pos_id = f"{direction}_{archetype}_{int(timestamp.timestamp())}_{bar_idx}"

        # Lookback for entry metadata
        _prev_high = fill_price
        _prev_low = fill_price
        if self._highs is not None and bar_idx > 0:
            lb_start = max(0, bar_idx - 20)
            _prev_high = float(np.nanmax(self._highs[lb_start:bar_idx]))
            _prev_low = float(np.nanmin(self._lows[lb_start:bar_idx]))

        self.positions[pos_id] = Position(
            position_id=pos_id,
            archetype=archetype,
            direction=direction,
            entry_price=fill_price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            original_quantity=quantity,
            current_quantity=quantity,
            fusion_score=fusion_score,
            atr_at_entry=atr,
            margin_used=margin,
            position_size_usd=notional,
            entry_metadata={
                'entry_prev_low': _prev_low,
                'entry_prev_high': _prev_high,
                'entry_wick_low': features.get('low', fill_price) if hasattr(features, 'get') else fill_price,
                'entry_spring_low': features.get('low', fill_price) if hasattr(features, 'get') else fill_price,
                'entry_ob_low': features.get('order_block_low', features.get('low', fill_price)) if hasattr(features, 'get') else fill_price,
                'entry_support_level': stop_loss,
                'entry_funding_z': features.get('funding_Z', 0.0) if hasattr(features, 'get') else 0.0,
                'entry_oi_delta': features.get('oi_change_4h', 0.0) if hasattr(features, 'get') else 0.0,
                'entry_volume': features.get('volume', 0.0) if hasattr(features, 'get') else 0.0,
                'entry_adx': features.get('adx_14', 0.0) if hasattr(features, 'get') else 0.0,
                'archetype': archetype,
                'executed_scale_outs': [],
            },
        )
        self.signals_traded += 1

    def _close_position(self, pos_id, exit_price, exit_timestamp, exit_reason, exit_pct=1.0):
        """Close a position (full or partial)."""
        if pos_id not in self.positions:
            return

        pos = self.positions[pos_id]
        exit_quantity = pos.original_quantity * exit_pct
        exit_quantity = min(exit_quantity, pos.current_quantity)
        if exit_quantity <= 1e-10:
            return

        # Slippage
        if pos.direction == 'long':
            fill_exit = exit_price * (1 - self.slippage_bps / 10000.0)
        else:
            fill_exit = exit_price * (1 + self.slippage_bps / 10000.0)

        # PnL
        if pos.direction == 'long':
            pnl = (fill_exit - pos.entry_price) * exit_quantity
        else:
            pnl = (pos.entry_price - fill_exit) * exit_quantity

        commission = fill_exit * exit_quantity * self.commission_rate
        pnl -= commission

        # Return margin
        exit_fraction = exit_quantity / pos.original_quantity
        margin_returned = pos.margin_used * exit_fraction
        self.cash += margin_returned + pnl

        entry_value = pos.entry_price * exit_quantity
        pnl_pct = (pnl / entry_value * 100) if entry_value > 0 else 0.0
        duration_hours = (exit_timestamp - pos.entry_time).total_seconds() / 3600.0

        trade = Trade(
            timestamp_entry=pos.entry_time,
            timestamp_exit=exit_timestamp,
            archetype=pos.archetype,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=fill_exit,
            quantity=exit_quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_hours=duration_hours,
            fusion_score=pos.fusion_score,
            exit_reason=exit_reason,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
        )
        self.trades.append(trade)

        # Per-archetype tracking
        if pos.archetype not in self.arch_trades:
            self.arch_trades[pos.archetype] = []
        self.arch_trades[pos.archetype].append(trade)

        # Update position
        pos.current_quantity -= exit_quantity
        pos.total_exits_pct += exit_pct

        if pos.current_quantity < 1e-10 or pos.total_exits_pct >= 0.99:
            if pos.current_quantity > 1e-10:
                dust_frac = pos.current_quantity / pos.original_quantity
                dust_margin = pos.margin_used * dust_frac
                if pos.direction == 'long':
                    dust_pnl = (fill_exit - pos.entry_price) * pos.current_quantity
                else:
                    dust_pnl = (pos.entry_price - fill_exit) * pos.current_quantity
                self.cash += dust_margin + dust_pnl
            del self.positions[pos_id]

    def _check_all_exits(self, row, ts, bar_idx):
        """Check exits for all open positions (identical to production)."""
        from engine.runtime.context import RuntimeContext

        regime_label = row.get('regime_label', 'neutral') if hasattr(row, 'get') else 'neutral'
        bar_context = RuntimeContext(
            ts=ts,
            row=row,
            regime_probs={regime_label: 1.0},
            regime_label=regime_label,
            adapted_params={},
            thresholds={},
        )

        for pos_id in list(self.positions.keys()):
            pos = self.positions[pos_id]

            # 1. Hard stop loss (inline, fill at stop level)
            stop_hit = False
            if pos.direction == 'long':
                effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                if row['low'] <= effective_stop:
                    stop_hit = True
                    exit_price = effective_stop
            else:
                effective_stop = pos.trailing_stop if pos.trailing_stop is not None else pos.stop_loss
                if row['high'] >= effective_stop:
                    stop_hit = True
                    exit_price = effective_stop

            if stop_hit:
                self._close_position(pos_id, exit_price, ts, "stop_loss", 1.0)
                continue

            # 2. ExitLogic (scale-outs, trailing, time exit, etc.)
            pos_adapter = _PosAdapter(pos)
            exit_signal = self.exit_logic.check_exit(
                bar=row, position=pos_adapter,
                archetype=pos.archetype, context=bar_context,
            )

            if exit_signal is not None:
                pos.executed_scale_outs = pos_adapter.metadata.get('executed_scale_outs', pos.executed_scale_outs)
                for flag_key in ('scaled_at_prev_high', 'moon_bag_taken'):
                    if pos_adapter.metadata.get(flag_key):
                        pos.entry_metadata[flag_key] = True

                if exit_signal.stop_update is not None:
                    pos.trailing_stop = exit_signal.stop_update

                if exit_signal.exit_pct > 0:
                    exit_reason = exit_signal.reason or exit_signal.exit_type
                    self._close_position(pos_id, row['close'], ts, exit_reason, exit_signal.exit_pct)
            else:
                if pos_adapter.stop_loss != pos.stop_loss:
                    pos.trailing_stop = pos_adapter.stop_loss

    def _compute_equity(self, current_price):
        """Compute current equity."""
        equity = self.cash
        for pos in self.positions.values():
            remaining_frac = pos.current_quantity / pos.original_quantity if pos.original_quantity > 0 else 0
            margin_locked = pos.margin_used * remaining_frac
            if pos.direction == 'long':
                unrealized = (current_price - pos.entry_price) * pos.current_quantity
            else:
                unrealized = (pos.entry_price - current_price) * pos.current_quantity
            equity += margin_locked + unrealized
        return equity

    # ── Reporting ─────────────────────────────────────────────────────

    def get_per_archetype_stats(self) -> Dict[str, Dict]:
        """Compute stats per archetype."""
        results = {}
        for arch_name, trades in self.arch_trades.items():
            if not trades:
                continue

            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            total_wins = sum(t.pnl for t in winners)
            total_losses = abs(sum(t.pnl for t in losers))

            pf = total_wins / total_losses if total_losses > 0 else float('inf')
            wr = len(winners) / len(trades) * 100 if trades else 0

            # Per-archetype equity curve for max DD
            equity = [self.initial_cash]
            for t in sorted(trades, key=lambda x: x.timestamp_exit):
                equity.append(equity[-1] + t.pnl)
            peak = equity[0]
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (e - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd

            results[arch_name] = {
                'trades': len(trades),
                'wins': len(winners),
                'losses': len(losers),
                'win_rate': wr,
                'profit_factor': pf,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_pnl': np.mean([t.pnl for t in trades]),
                'max_dd': max_dd,
                'avg_hold_hours': np.mean([t.duration_hours for t in trades]),
                'direction': trades[0].direction if trades else 'unknown',
            }
        return results

    def get_aggregate_stats(self) -> Dict:
        """Compute aggregate stats across all archetypes."""
        if not self.trades:
            return {'total_trades': 0, 'profit_factor': 0, 'total_pnl': 0,
                    'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        total_wins = sum(t.pnl for t in winners)
        total_losses = abs(sum(t.pnl for t in losers))

        pf = total_wins / total_losses if total_losses > 0 else float('inf')

        # Max DD from equity curve
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = float(np.min(dd))

        # Sharpe from equity curve
        if len(eq) > 10:
            returns = np.diff(eq) / eq[:-1]
            returns = returns[np.isfinite(returns)]
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(8760)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': len(winners) / len(self.trades) * 100,
            'profit_factor': pf,
            'total_pnl': sum(t.pnl for t in self.trades),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_hold_hours': np.mean([t.duration_hours for t in self.trades]),
        }

    def print_results(self, compare_stats=None):
        """Print per-archetype results table."""
        arch_stats = self.get_per_archetype_stats()
        agg = self.get_aggregate_stats()

        print(f"\n{'=' * 110}")
        print(f"STRUCTURAL BACKTEST RESULTS")
        print(f"Gate mode: {self.gate_mode} | Archetypes: {len(arch_stats)}")
        print(f"{'=' * 110}")
        print(f"\n{'Archetype':<25s} {'Dir':<6s} {'Trades':>7s} {'Wins':>6s} {'WR%':>7s} "
              f"{'PF':>7s} {'PnL($)':>10s} {'AvgPnL':>8s} {'MaxDD%':>8s} {'AvgHold':>8s}")
        print("-" * 110)

        # Sort by PnL descending
        sorted_archs = sorted(arch_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

        for name, s in sorted_archs:
            pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "inf"
            print(f"{name:<25s} {s['direction']:<6s} {s['trades']:>7d} {s['wins']:>6d} "
                  f"{s['win_rate']:>6.1f}% {pf_str:>7s} "
                  f"${s['total_pnl']:>9,.0f} ${s['avg_pnl']:>7,.0f} "
                  f"{s['max_dd']:>7.1f}% {s['avg_hold_hours']:>7.0f}h")

        print("-" * 110)
        agg_pf = f"{agg['profit_factor']:.2f}" if agg['profit_factor'] < 100 else "inf"
        print(f"{'TOTAL':<25s} {'':>6s} {agg['total_trades']:>7d} {agg['winning_trades']:>6d} "
              f"{agg['win_rate']:>6.1f}% {agg_pf:>7s} "
              f"${agg['total_pnl']:>9,.0f} {'':>8s} "
              f"{agg['max_drawdown']:>7.1f}% {agg['avg_hold_hours']:>7.0f}h")
        print(f"\nSharpe: {agg['sharpe_ratio']:.2f} | "
              f"Raw signals: {self.raw_signals} | "
              f"Traded: {self.signals_traded}")

        # Comparison with production
        if compare_stats:
            print(f"\n{'=' * 110}")
            print(f"COMPARISON: Structural vs Production")
            print(f"{'=' * 110}")
            print(f"{'Metric':<20s} {'Structural':>12s} {'Production':>12s} {'Delta':>12s}")
            print("-" * 60)

            metrics = [
                ('Trades', agg['total_trades'], compare_stats.get('total_trades', 0)),
                ('PF', agg['profit_factor'], compare_stats.get('profit_factor', 0)),
                ('PnL', agg['total_pnl'], compare_stats.get('total_pnl', 0)),
                ('Win Rate', agg['win_rate'], compare_stats.get('win_rate', 0)),
                ('MaxDD', agg['max_drawdown'], compare_stats.get('max_drawdown', 0)),
                ('Sharpe', agg['sharpe_ratio'], compare_stats.get('sharpe_ratio', 0)),
            ]

            for name, struct_val, prod_val in metrics:
                if name == 'PnL':
                    print(f"{name:<20s} ${struct_val:>11,.0f} ${prod_val:>11,.0f} ${struct_val - prod_val:>+11,.0f}")
                elif name == 'Trades':
                    print(f"{name:<20s} {struct_val:>12d} {prod_val:>12d} {struct_val - prod_val:>+12d}")
                else:
                    fmt = '.1f' if name in ('Win Rate', 'MaxDD') else '.2f'
                    print(f"{name:<20s} {struct_val:>12{fmt}} {prod_val:>12{fmt}} {struct_val - prod_val:>+12{fmt}}")

            # Key insight
            prod_pf = compare_stats.get('profit_factor', 0)
            if agg['profit_factor'] > prod_pf:
                print(f"\n  >> Fusion/CMI filtering is HURTING performance (structural PF > production PF)")
                print(f"  >> The threshold layer is killing profitable signals")
            elif agg['profit_factor'] < prod_pf:
                print(f"\n  >> Fusion/CMI filtering is HELPING performance (production PF > structural PF)")
                print(f"  >> The threshold layer correctly filters low-quality signals")
            else:
                print(f"\n  >> Fusion/CMI filtering is NEUTRAL")

        print(f"\n{'=' * 110}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Structural-Only Backtester')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--feature-store', default=DEFAULT_FEATURE_STORE)
    parser.add_argument('--start-date', default='2020-01-01')
    parser.add_argument('--end-date', default=None)
    parser.add_argument('--initial-cash', type=float, default=100_000.0)
    parser.add_argument('--commission-rate', type=float, default=0.0002)
    parser.add_argument('--slippage-bps', type=float, default=3.0)
    parser.add_argument('--archetype', type=str, default=None,
                       help='Comma-separated archetype filter (e.g., wick_trap,spring)')
    parser.add_argument('--gates', choices=['hard', 'soft', 'none'], default='hard',
                       help='Gate mode: hard=structural+gates, soft=gates as penalty, none=structural only')
    parser.add_argument('--max-concurrent', type=int, default=0,
                       help='Max concurrent positions per archetype (0=unlimited)')
    parser.add_argument('--compare', action='store_true',
                       help='Run production backtester for side-by-side comparison')
    parser.add_argument('--output-dir', default='results/structural')
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    # Parse archetype filter
    arch_filter = None
    if args.archetype:
        arch_filter = [a.strip() for a in args.archetype.split(',')]

    # Load data
    print("Loading feature store...")
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = json.load(f)
    features_df = pd.read_parquet(str(PROJECT_ROOT / args.feature_store))
    if not isinstance(features_df.index, pd.DatetimeIndex):
        features_df.index = pd.to_datetime(features_df.index)
    features_df = features_df.sort_index()
    print(f"Loaded {len(features_df):,} bars\n")

    # Run structural backtest
    bt = StructuralBacktester(
        config=config,
        features_df=features_df,
        initial_cash=args.initial_cash,
        commission_rate=args.commission_rate,
        slippage_bps=args.slippage_bps,
        gate_mode=args.gates,
        archetype_filter=arch_filter,
        max_concurrent=args.max_concurrent,
    )
    bt.run(start_date=args.start_date, end_date=args.end_date)

    # Optional: run production backtest for comparison
    compare_stats = None
    if args.compare:
        print(f"\nRunning production backtester for comparison...")
        from bin.backtest_v11_standalone import StandaloneBacktestEngine
        prod = StandaloneBacktestEngine(
            config=config, initial_cash=args.initial_cash,
            commission_rate=args.commission_rate,
            slippage_bps=args.slippage_bps,
            features_df=features_df,
        )
        prod.run(start_date=args.start_date, end_date=args.end_date)
        compare_stats = prod.get_performance_stats()

    # Print results
    bt.print_results(compare_stats=compare_stats)

    # Save trade log
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if bt.trades:
        trade_data = [{
            'entry_time': t.timestamp_entry,
            'exit_time': t.timestamp_exit,
            'archetype': t.archetype,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_pct': t.pnl_pct,
            'duration_hours': t.duration_hours,
            'fusion_score': t.fusion_score,
            'exit_reason': t.exit_reason,
        } for t in bt.trades]
        trade_df = pd.DataFrame(trade_data)
        trade_path = output_dir / f'structural_trades_{args.gates}.csv'
        trade_df.to_csv(trade_path, index=False)
        print(f"\nTrade log saved: {trade_path}")

    # Save aggregate results
    results = {
        'gate_mode': args.gates,
        'archetypes': arch_filter or 'all',
        'aggregate': bt.get_aggregate_stats(),
        'per_archetype': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
                          for k, v in bt.get_per_archetype_stats().items()},
    }
    if compare_stats:
        results['production'] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                                  for k, v in compare_stats.items()}

    with open(output_dir / f'results_{args.gates}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved: {output_dir}/")


if __name__ == '__main__':
    main()
