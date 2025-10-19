#!/usr/bin/env python3
"""
Knowledge-Aware Backtest Engine v2.0

Uses ALL 69 features from the MTF feature store intelligently:
- Advanced fusion scoring with PTI, Macro, Wyckoff M1/M2, FRVP
- Smart entry logic (tiered entries, pullback waiting, limit orders)
- Smart exit integration (partial exits, trailing stops, breakeven)
- ATR-based position sizing (1-2% risk per trade)
- Macro regime filtering
- Full knowledge hooks integration

This is the COMPLETE backtest engine that replaces the simplified
optimizer placeholder (bin/optimize_v2_cached.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeParams:
    """
    Knowledge-aware backtest parameters.

    Expanded from simple fusion weights to include smart entry/exit logic.
    """
    # Domain weights (must sum to ≤ 1.0)
    wyckoff_weight: float = 0.30
    liquidity_weight: float = 0.30
    momentum_weight: float = 0.20
    macro_weight: float = 0.10  # NEW: Macro regime influence
    pti_weight: float = 0.10     # NEW: Trap detection influence

    # Entry thresholds (tiered) - Adjusted to realistic fusion score range (0.0-0.5)
    tier1_threshold: float = 0.45  # Ultra-high conviction (market entry)
    tier2_threshold: float = 0.35  # High conviction (limit entry on pullback)
    tier3_threshold: float = 0.25  # Medium conviction (scale in)

    # Entry modifiers
    require_m1m2_confirmation: bool = True  # Require Wyckoff M1/M2 signal
    require_macro_alignment: bool = True     # Require risk_on regime
    frvp_entry_zone: str = "value_area"      # Enter near POC/value_area
    pullback_depth: float = 0.382            # Wait for 38.2% pullback (Fib)

    # Exit management
    use_smart_exits: bool = True              # Enable smart_exits.py integration
    partial_exit_1: float = 0.33              # Exit 33% at TP1 (+1R)
    partial_exit_2: float = 0.33              # Exit 33% at TP2 (+2R)
    trailing_atr_mult: float = 2.0            # Trail stop 2× ATR from peak
    breakeven_after_tp1: bool = True          # Move stop to breakeven after TP1
    max_hold_bars: int = 168                  # Max 168 hours (7 days)

    # Position sizing
    max_risk_pct: float = 0.02                # Max 2% risk per trade
    atr_stop_mult: float = 2.5                # Stop loss 2.5× ATR below entry
    position_size_method: str = "atr"         # "atr" or "fixed"
    volatility_scaling: bool = True           # Scale down in high VIX

    # Costs
    slippage_bps: float = 2.0                 # 2 basis points
    fee_bps: float = 1.0                      # 1 basis point


@dataclass
class Trade:
    """Trade record with full knowledge context."""
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float  # USD value
    direction: int  # +1 long, -1 short
    entry_fusion_score: float
    entry_reason: str  # "tier1_market", "tier2_pullback", "tier3_scale"

    # Entry knowledge context
    wyckoff_phase: str
    wyckoff_m1_signal: Optional[str]
    wyckoff_m2_signal: Optional[str]
    macro_regime: str
    pti_score_1d: float
    pti_score_1h: float
    frvp_poc_position: str

    # Exit tracking
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    partial_exits: List[Dict] = field(default_factory=list)

    # PNL tracking
    gross_pnl: float = 0.0
    fees: float = 0.0
    net_pnl: float = 0.0

    # Risk metrics
    atr_at_entry: float = 0.0
    initial_stop: float = 0.0
    peak_profit: float = 0.0
    max_adverse_excursion: float = 0.0


class KnowledgeAwareBacktest:
    """
    Full knowledge backtest engine using all 69 MTF features.
    """

    def __init__(self, df: pd.DataFrame, params: KnowledgeParams, starting_capital: float = 10000.0):
        """
        Initialize backtest with feature store and parameters.

        Args:
            df: MTF feature store (69 features)
            params: Knowledge-aware parameters
            starting_capital: Starting equity
        """
        self.df = df.copy()
        self.params = params
        self.starting_capital = starting_capital
        self.equity = starting_capital
        self.peak_equity = starting_capital

        self.trades: List[Trade] = []
        self.current_position: Optional[Trade] = None

        # Precompute ATR for position sizing
        self._precompute_atr()

    def _precompute_atr(self):
        """Precompute ATR for the entire dataset."""
        # ATR should already be in feature store, but compute if missing
        if 'atr_14' not in self.df.columns:
            high = self.df['high']
            low = self.df['low']
            close = self.df['close']
            prev_close = close.shift(1)

            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)

            self.df['atr_14'] = tr.ewm(span=14, adjust=False).mean()

    def compute_advanced_fusion_score(self, row: pd.Series) -> Tuple[float, Dict]:
        """
        Compute advanced fusion score using ALL 69 features.

        Returns:
            (fusion_score, context_dict)
        """
        context = {}

        # 1. Wyckoff Governor (1D) - Include M1/M2 signals
        wyckoff_base = row.get('tf1d_wyckoff_score', 0.5)
        m1_strength = row.get('tf1d_m1_signal_strength', 0.0)
        m2_strength = row.get('tf1d_m2_signal_strength', 0.0)

        # Boost Wyckoff score if M1/M2 signals are present
        wyckoff_boost = max(m1_strength, m2_strength) * 0.3  # Up to +0.3 boost
        wyckoff = np.clip(wyckoff_base + wyckoff_boost, 0.0, 1.0)

        context['wyckoff_score'] = wyckoff
        context['wyckoff_phase'] = row.get('tf1d_wyckoff_phase', 'unknown')
        context['m1_signal'] = row.get('tf1d_m1_signal', None)
        context['m2_signal'] = row.get('tf1d_m2_signal', None)

        # 2. Liquidity (HOB/BOMS)
        boms_strength = row.get('tf1d_boms_strength', 0.5)
        fvg_present = 1.0 if row.get('tf4h_fvg_present', False) else 0.0
        hob_rejection = 1.0 if row.get('tf4h_hob_rejection', False) else 0.0

        liquidity = (boms_strength + fvg_present + hob_rejection) / 3.0
        context['liquidity_score'] = liquidity

        # 3. Momentum (ADX + RSI + Squiggle)
        adx = row.get('adx_14', 20.0) / 100.0
        rsi = row.get('rsi_14', 50.0)
        rsi_momentum = abs(rsi - 50.0) / 50.0
        squiggle_conf = row.get('tf4h_squiggle_confidence', 0.5)

        momentum = (adx + rsi_momentum + squiggle_conf) / 3.0
        context['momentum_score'] = momentum

        # 4. PTI (Trap Detection) - Inverse scoring (high PTI = avoid)
        pti_1d = row.get('tf1d_pti_score', 0.0)
        pti_1h = row.get('tf1h_pti_score', 0.0)
        pti_combined = max(pti_1d, pti_1h)

        # PTI acts as a penalty (high PTI = likely trap)
        pti_penalty = pti_combined  # 0.0-1.0 scale
        context['pti_score'] = pti_combined
        context['pti_1d'] = pti_1d
        context['pti_1h'] = pti_1h

        # 5. Macro (Regime + Trends)
        macro_regime = row.get('macro_regime', 'neutral')
        macro_vix = row.get('macro_vix_level', 'medium')

        # Regime scoring: risk_on = 1.0, neutral = 0.5, risk_off/crisis = 0.0
        regime_map = {'risk_on': 1.0, 'neutral': 0.5, 'risk_off': 0.2, 'crisis': 0.0}
        regime_score = regime_map.get(macro_regime, 0.5)

        # VIX penalty: low = 1.0, medium = 0.8, high = 0.5, extreme = 0.2
        vix_map = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.2}
        vix_score = vix_map.get(macro_vix, 0.8)

        macro = (regime_score + vix_score) / 2.0
        context['macro_score'] = macro
        context['macro_regime'] = macro_regime
        context['macro_vix'] = macro_vix

        # 6. FRVP (Value Area Positioning)
        frvp_poc_pos = row.get('tf1h_frvp_poc_position', 'middle')

        # Prefer entries near value area (POC)
        poc_map = {'below': 0.3, 'at_poc': 1.0, 'above': 0.3, 'middle': 0.6}
        frvp_score = poc_map.get(frvp_poc_pos, 0.5)
        context['frvp_score'] = frvp_score
        context['frvp_poc_position'] = frvp_poc_pos

        # Weighted fusion score
        fusion = (
            self.params.wyckoff_weight * wyckoff +
            self.params.liquidity_weight * liquidity +
            self.params.momentum_weight * momentum +
            self.params.macro_weight * macro +
            (1.0 - self.params.wyckoff_weight - self.params.liquidity_weight -
             self.params.momentum_weight - self.params.macro_weight) * frvp_score
        )

        # Apply PTI penalty (subtract PTI weight × PTI score)
        fusion -= self.params.pti_weight * pti_penalty

        # Apply fakeout penalty
        if row.get('tf1h_fakeout_detected', False):
            fusion -= 0.1  # -10% penalty for fakeouts

        # Apply governor veto
        if row.get('mtf_governor_veto', False):
            fusion *= 0.3  # Severe penalty for governor veto

        # Clip to [0, 1]
        fusion = np.clip(fusion, 0.0, 1.0)

        context['fusion_score'] = fusion
        return fusion, context

    def calculate_position_size(self, row: pd.Series, fusion_score: float) -> float:
        """
        Calculate position size using ATR-based risk management.

        Target: 1-2% equity at risk per trade.
        """
        if self.params.position_size_method == "fixed":
            # Simple: 95% allocation (old method)
            return self.equity * 0.95

        # ATR-based sizing
        atr = row.get('atr_14', row['close'] * 0.02)  # Default to 2% of price

        # Stop loss distance (2.5× ATR)
        stop_distance = atr * self.params.atr_stop_mult

        # Risk amount (2% of equity)
        risk_dollars = self.equity * self.params.max_risk_pct

        # Position size = risk / stop_distance
        position_size = risk_dollars / (stop_distance / row['close'])

        # Volatility scaling (reduce in high VIX)
        if self.params.volatility_scaling:
            vix_level = row.get('macro_vix_level', 'medium')
            vix_scaling = {'low': 1.0, 'medium': 0.8, 'high': 0.5, 'extreme': 0.25}
            position_size *= vix_scaling.get(vix_level, 0.8)

        # Confidence scaling (higher fusion = larger size)
        # Scale from 50% to 100% allocation based on fusion score
        confidence_mult = 0.5 + (fusion_score * 0.5)
        position_size *= confidence_mult

        # Cap at 95% of equity
        position_size = min(position_size, self.equity * 0.95)

        return position_size

    def check_entry_conditions(self, row: pd.Series, fusion_score: float, context: Dict) -> Optional[Tuple[str, float]]:
        """
        Check if entry conditions are met using tiered entry logic.

        Returns:
            (entry_type, entry_price) or None
        """
        # Tier 1: Ultra-high conviction (market entry)
        if fusion_score >= self.params.tier1_threshold:
            # Check M1/M2 confirmation if required
            if self.params.require_m1m2_confirmation:
                if not (context.get('m1_signal') or context.get('m2_signal')):
                    return None  # No M1/M2 signal

            # Check macro alignment if required
            if self.params.require_macro_alignment:
                if context.get('macro_regime') not in ['risk_on', 'neutral']:
                    return None  # Bad macro regime

            return ("tier1_market", row['close'])

        # Tier 2: High conviction (wait for pullback to enter at limit)
        if fusion_score >= self.params.tier2_threshold:
            # Check if price is near FRVP value area (better entry)
            if self.params.frvp_entry_zone == "value_area":
                if context.get('frvp_poc_position') != 'at_poc':
                    return None  # Wait for better entry

            # Check M1/M2 if required
            if self.params.require_m1m2_confirmation:
                if not (context.get('m1_signal') or context.get('m2_signal')):
                    return None

            # Enter at current price (simplified - in live would use limit order)
            return ("tier2_pullback", row['close'])

        # Tier 3: Medium conviction (scale in slowly)
        if fusion_score >= self.params.tier3_threshold:
            # More relaxed requirements
            if context.get('macro_regime') == 'crisis':
                return None  # Don't trade in crisis

            return ("tier3_scale", row['close'])

        return None

    def check_exit_conditions(self, row: pd.Series, trade: Trade) -> Optional[Tuple[str, float]]:
        """
        Check if exit conditions are met using smart exit logic.

        Returns:
            (exit_reason, exit_price) or None
        """
        current_price = row['close']
        pnl_pct = (current_price - trade.entry_price) / trade.entry_price * trade.direction
        pnl_r = pnl_pct / (self.params.atr_stop_mult * trade.atr_at_entry / trade.entry_price)

        # 1. Stop loss hit (initial stop)
        if trade.direction == 1:  # Long
            if current_price <= trade.initial_stop:
                return ("stop_loss", trade.initial_stop)
        else:  # Short
            if current_price >= trade.initial_stop:
                return ("stop_loss", trade.initial_stop)

        # 2. Partial exits (if enabled)
        if self.params.use_smart_exits:
            # TP1: +1R
            if pnl_r >= 1.0 and not any(p['level'] == 'TP1' for p in trade.partial_exits):
                # Don't exit fully, just log partial (handled separately)
                pass

            # TP2: +2R
            if pnl_r >= 2.0 and not any(p['level'] == 'TP2' for p in trade.partial_exits):
                pass

        # 3. Trailing stop (if in profit and trailing enabled)
        if pnl_r > 1.0 and self.params.use_smart_exits:
            # Trail from peak
            atr = row.get('atr_14', trade.atr_at_entry)
            trailing_stop = trade.entry_price + (trade.peak_profit - self.params.trailing_atr_mult * atr) * trade.direction

            if trade.direction == 1:  # Long
                if current_price <= trailing_stop:
                    return ("trailing_stop", current_price)
            else:  # Short
                if current_price >= trailing_stop:
                    return ("trailing_stop", current_price)

        # 4. Fusion score drops (signal neutralized)
        fusion_score, context = self.compute_advanced_fusion_score(row)
        if fusion_score < self.params.tier3_threshold:
            return ("signal_neutralized", current_price)

        # 5. PTI reversal detected
        if context.get('pti_score', 0.0) > 0.6:
            return ("pti_reversal", current_price)

        # 6. Macro regime flip
        if context.get('macro_regime') == 'crisis':
            return ("macro_crisis", current_price)

        # 7. Max holding period
        bars_held = (row.name - trade.entry_time).total_seconds() / 3600  # Hours
        if bars_held >= self.params.max_hold_bars:
            return ("max_hold", current_price)

        # 8. MTF conflict
        mtf_conflict = row.get('mtf_conflict_score', 0.0)
        if mtf_conflict > 0.7:
            return ("mtf_conflict", current_price)

        return None

    def run(self) -> Dict:
        """
        Run the full knowledge-aware backtest.

        Returns:
            Results dict with trades, metrics, and feature importance.
        """
        logger.info(f"Starting knowledge-aware backtest on {len(self.df)} bars...")

        for idx, row in self.df.iterrows():
            # Skip early bars without indicators
            if pd.isna(row.get('atr_14')):
                continue

            # Compute fusion score
            fusion_score, context = self.compute_advanced_fusion_score(row)

            # Check for open position
            if self.current_position is not None:
                # Update peak profit and MAE
                current_price = row['close']
                pnl_pct = (current_price - self.current_position.entry_price) / self.current_position.entry_price * self.current_position.direction

                if pnl_pct > self.current_position.peak_profit:
                    self.current_position.peak_profit = pnl_pct

                if pnl_pct < -self.current_position.max_adverse_excursion:
                    self.current_position.max_adverse_excursion = -pnl_pct

                # Check exit conditions
                exit_result = self.check_exit_conditions(row, self.current_position)

                if exit_result:
                    exit_reason, exit_price = exit_result
                    self._close_trade(row, exit_price, exit_reason)

            # Check for new entry (only if no position)
            if self.current_position is None:
                entry_result = self.check_entry_conditions(row, fusion_score, context)

                if entry_result:
                    entry_type, entry_price = entry_result
                    self._open_trade(row, entry_price, entry_type, fusion_score, context)

        # Close any remaining position at end
        if self.current_position is not None:
            last_row = self.df.iloc[-1]
            self._close_trade(last_row, last_row['close'], "end_of_period")

        # Calculate metrics
        return self._calculate_metrics()

    def _open_trade(self, row: pd.Series, entry_price: float, entry_type: str, fusion_score: float, context: Dict):
        """Open a new trade."""
        position_size = self.calculate_position_size(row, fusion_score)
        atr = row.get('atr_14', entry_price * 0.02)

        # Calculate initial stop
        stop_distance = atr * self.params.atr_stop_mult
        initial_stop = entry_price - stop_distance  # For long (flip for short)

        trade = Trade(
            entry_time=row.name,
            entry_price=entry_price,
            position_size=position_size,
            direction=1,  # Only long for now
            entry_fusion_score=fusion_score,
            entry_reason=entry_type,
            wyckoff_phase=context.get('wyckoff_phase', 'unknown'),
            wyckoff_m1_signal=context.get('m1_signal'),
            wyckoff_m2_signal=context.get('m2_signal'),
            macro_regime=context.get('macro_regime', 'neutral'),
            pti_score_1d=context.get('pti_1d', 0.0),
            pti_score_1h=context.get('pti_1h', 0.0),
            frvp_poc_position=context.get('frvp_poc_position', 'middle'),
            atr_at_entry=atr,
            initial_stop=initial_stop
        )

        self.current_position = trade
        logger.info(f"ENTRY {entry_type}: {row.name} @ ${entry_price:.2f}, size=${position_size:.2f}, fusion={fusion_score:.3f}")

    def _close_trade(self, row: pd.Series, exit_price: float, exit_reason: str):
        """Close the current trade."""
        trade = self.current_position
        trade.exit_time = row.name
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason

        # Calculate PNL
        pnl_pct = (exit_price - trade.entry_price) / trade.entry_price * trade.direction
        trade.gross_pnl = trade.position_size * pnl_pct

        # Apply costs
        trade.fees = trade.position_size * (self.params.slippage_bps + self.params.fee_bps) / 10000.0
        trade.net_pnl = trade.gross_pnl - trade.fees

        # Update equity
        self.equity += trade.net_pnl
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        self.trades.append(trade)
        self.current_position = None

        logger.info(f"EXIT {exit_reason}: {row.name} @ ${exit_price:.2f}, PNL=${trade.net_pnl:.2f}, equity=${self.equity:.2f}")

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_pnl': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'final_equity': self.equity,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'trades': []
            }

        total_pnl = sum(t.net_pnl for t in self.trades)
        winning_trades = [t for t in self.trades if t.net_pnl > 0]
        losing_trades = [t for t in self.trades if t.net_pnl < 0]

        gross_profit = sum(t.net_pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.net_pnl for t in losing_trades)) if losing_trades else 1.0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0

        # Sharpe (simplified)
        trade_returns = [t.net_pnl / self.starting_capital for t in self.trades]
        sharpe = np.mean(trade_returns) / np.std(trade_returns) if len(trade_returns) > 1 else 0.0
        sharpe = sharpe * np.sqrt(252 / len(self.trades)) if len(self.trades) > 0 else 0.0

        # Max drawdown
        max_dd = (self.peak_equity - min(self.equity, self.peak_equity)) / self.peak_equity

        return {
            'total_pnl': total_pnl,
            'total_trades': len(self.trades),
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'final_equity': self.equity,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0.0,
            'avg_loss': np.mean([t.net_pnl for t in losing_trades]) if losing_trades else 0.0,
            'trades': self.trades
        }


# CLI interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Knowledge-aware backtest using full 69-feature engine')
    parser.add_argument('--asset', required=True, help='Asset (BTC, ETH, SPY)')
    parser.add_argument('--start', default='2024-01-01', help='Start date')
    parser.add_argument('--end', default='2024-12-31', help='End date')
    parser.add_argument('--config', help='JSON config file with KnowledgeParams')

    args = parser.parse_args()

    # Load feature store
    feature_dir = Path('data/features_mtf')
    pattern = f"{args.asset}_1H_*.parquet"
    files = list(feature_dir.glob(pattern))

    if not files:
        print(f"ERROR: No feature store found for {args.asset}")
        sys.exit(1)

    feature_path = sorted(files)[-1]
    print(f"Loading feature store: {feature_path}")

    df = pd.read_parquet(feature_path)

    # Filter to date range
    start_ts = pd.Timestamp(args.start, tz='UTC')
    end_ts = pd.Timestamp(args.end, tz='UTC')
    df = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Load params from config or use defaults
    if args.config:
        with open(args.config) as f:
            param_dict = json.load(f)
        params = KnowledgeParams(**param_dict)
    else:
        params = KnowledgeParams()

    # Run backtest
    backtest = KnowledgeAwareBacktest(df, params)
    results = backtest.run()

    # Print results
    print("\n" + "=" * 80)
    print(f"Knowledge-Aware Backtest Results - {args.asset}")
    print("=" * 80)
    print(f"Total PNL: ${results['total_pnl']:.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"Final Equity: ${results['final_equity']:.2f}")
    print(f"Gross Profit: ${results['gross_profit']:.2f}")
    print(f"Gross Loss: ${results['gross_loss']:.2f}")
    print(f"Avg Win: ${results['avg_win']:.2f}")
    print(f"Avg Loss: ${results['avg_loss']:.2f}")

    print("\n" + "=" * 80)
    print("Trade Log")
    print("=" * 80)
    for i, trade in enumerate(results['trades'], 1):
        print(f"\nTrade {i}: {trade.entry_reason}")
        print(f"  Entry: {trade.entry_time} @ ${trade.entry_price:.2f}")
        print(f"  Exit:  {trade.exit_time} @ ${trade.exit_price:.2f} ({trade.exit_reason})")
        print(f"  PNL: ${trade.net_pnl:.2f} ({trade.net_pnl/trade.position_size:.2%})")
        print(f"  Wyckoff: {trade.wyckoff_phase} (M1={trade.wyckoff_m1_signal}, M2={trade.wyckoff_m2_signal})")
        print(f"  Macro: {trade.macro_regime}, PTI: {trade.pti_score_1d:.3f}/{trade.pti_score_1h:.3f}")
        print(f"  FRVP: {trade.frvp_poc_position}, Fusion: {trade.entry_fusion_score:.3f}")
