"""
Counterfactual Trade Analysis Engine ("Quantum Trading System")

For each completed trade, computes alternative outcomes under different parameters:
- Stop loss sensitivity (0.5x to 2x of actual)
- Take profit sensitivity (0.5x to 2x of actual)
- Hold time sensitivity (0.5x to 2x of actual)
- No scale-out (hold to SL or full TP)
- Threshold sensitivity (would trade have been taken at different thresholds)

Usage:
    engine = CounterfactualEngine(feature_store_df, commission_rate=0.0002, slippage_bps=3.0)
    for trade in completed_trades:
        scenarios = engine.analyze_trade(trade)
        # scenarios is a dict of scenario_name -> CounterfactualResult
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualResult:
    """Result of a single counterfactual scenario."""
    scenario: str           # e.g., "sl_1.5x", "tp_0.75x", "hold_2.0x"
    param_changed: str      # "stop_loss", "take_profit", "hold_time", "scale_out"
    param_original: float   # Original parameter value
    param_alternative: float  # Alternative parameter value
    exit_price: float
    exit_reason: str        # "stop_loss", "take_profit", "time_exit", "end_of_data"
    pnl: float
    pnl_pct: float
    duration_hours: float
    pnl_delta: float        # vs actual trade PnL (positive = would have been better)
    pnl_pct_delta: float    # vs actual trade PnL%

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeCounterfactual:
    """Complete counterfactual analysis for one trade."""
    trade_entry_ts: str
    trade_archetype: str
    actual_pnl: float
    actual_pnl_pct: float
    actual_exit_reason: str
    scenarios: Dict[str, CounterfactualResult] = field(default_factory=dict)
    best_scenario: str = ""
    best_pnl_delta: float = 0.0
    worst_scenario: str = ""
    worst_pnl_delta: float = 0.0
    was_optimal: bool = False   # True if actual was best or near-best outcome

    def to_dict(self) -> dict:
        d = {
            'trade_entry_ts': self.trade_entry_ts,
            'trade_archetype': self.trade_archetype,
            'actual_pnl': round(self.actual_pnl, 2),
            'actual_pnl_pct': round(self.actual_pnl_pct, 4),
            'actual_exit_reason': self.actual_exit_reason,
            'best_scenario': self.best_scenario,
            'best_pnl_delta': round(self.best_pnl_delta, 2),
            'worst_scenario': self.worst_scenario,
            'worst_pnl_delta': round(self.worst_pnl_delta, 2),
            'was_optimal': self.was_optimal,
            'scenarios': {k: v.to_dict() for k, v in self.scenarios.items()},
        }
        return d


class CounterfactualEngine:
    """
    Computes alternative trade outcomes for post-hoc analysis.

    For each trade, simulates what would have happened with different:
    - Stop loss levels (tighter/wider)
    - Take profit levels (closer/farther)
    - Hold times (shorter/longer)
    - Exit strategies (no scale-out, hold full)
    """

    # Multipliers for sensitivity analysis
    SL_MULTS = [0.5, 0.75, 1.25, 1.5, 2.0]
    TP_MULTS = [0.5, 0.75, 1.25, 1.5, 2.0]
    HOLD_MULTS = [0.5, 0.75, 1.5, 2.0]

    def __init__(
        self,
        df: pd.DataFrame,
        commission_rate: float = 0.0002,
        slippage_bps: float = 3.0,
        max_lookahead_bars: int = 500,
    ):
        """
        Args:
            df: Feature store DataFrame with DatetimeIndex and OHLCV columns
            commission_rate: Commission rate per side
            slippage_bps: Slippage in basis points per side
            max_lookahead_bars: Maximum bars to look ahead from entry
        """
        self.df = df
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.max_lookahead_bars = max_lookahead_bars

    def analyze_trade(self, trade) -> TradeCounterfactual:
        """
        Compute counterfactual outcomes for a single completed trade.

        Args:
            trade: CompletedTrade object with attributes:
                   timestamp_entry, timestamp_exit, archetype, direction,
                   entry_price, exit_price, pnl, pnl_pct, stop_loss,
                   take_profit, duration_hours, exit_reason

        Returns:
            TradeCounterfactual with all scenarios computed
        """
        entry_ts = trade.timestamp_entry
        entry_price = trade.entry_price
        direction = trade.direction
        actual_sl = getattr(trade, 'stop_loss', 0.0)
        actual_tp = getattr(trade, 'take_profit', 0.0)
        actual_hold = trade.duration_hours

        # Validate we have SL/TP
        if actual_sl <= 0 or actual_tp <= 0:
            logger.debug(f"Skipping counterfactual for {trade.archetype} @ {entry_ts}: missing SL/TP")
            return TradeCounterfactual(
                trade_entry_ts=str(entry_ts),
                trade_archetype=trade.archetype,
                actual_pnl=trade.pnl,
                actual_pnl_pct=trade.pnl_pct,
                actual_exit_reason=trade.exit_reason,
            )

        # Get future bars from entry
        mask = self.df.index >= entry_ts
        future_bars = self.df.loc[mask].head(self.max_lookahead_bars)

        if len(future_bars) < 2:
            return TradeCounterfactual(
                trade_entry_ts=str(entry_ts),
                trade_archetype=trade.archetype,
                actual_pnl=trade.pnl,
                actual_pnl_pct=trade.pnl_pct,
                actual_exit_reason=trade.exit_reason,
            )

        # Stop loss distance from entry
        sl_distance = abs(entry_price - actual_sl)
        tp_distance = abs(actual_tp - entry_price)
        actual_hold_bars = max(1, int(actual_hold))  # 1H bars

        result = TradeCounterfactual(
            trade_entry_ts=str(entry_ts),
            trade_archetype=trade.archetype,
            actual_pnl=trade.pnl,
            actual_pnl_pct=trade.pnl_pct,
            actual_exit_reason=trade.exit_reason,
        )

        # --- 1. Stop Loss Sensitivity ---
        for mult in self.SL_MULTS:
            alt_sl = self._scale_stop_loss(entry_price, actual_sl, direction, mult)
            sim = self._simulate(
                future_bars, entry_price, direction,
                stop_loss=alt_sl, take_profit=actual_tp,
                max_bars=actual_hold_bars * 3,  # Allow extra time for wider SL
            )
            pnl_delta = sim['pnl'] - trade.pnl
            result.scenarios[f'sl_{mult}x'] = CounterfactualResult(
                scenario=f'sl_{mult}x',
                param_changed='stop_loss',
                param_original=actual_sl,
                param_alternative=alt_sl,
                exit_price=sim['exit_price'],
                exit_reason=sim['exit_reason'],
                pnl=sim['pnl'],
                pnl_pct=sim['pnl_pct'],
                duration_hours=sim['duration_hours'],
                pnl_delta=pnl_delta,
                pnl_pct_delta=sim['pnl_pct'] - trade.pnl_pct,
            )

        # --- 2. Take Profit Sensitivity ---
        for mult in self.TP_MULTS:
            alt_tp = self._scale_take_profit(entry_price, actual_tp, direction, mult)
            sim = self._simulate(
                future_bars, entry_price, direction,
                stop_loss=actual_sl, take_profit=alt_tp,
                max_bars=actual_hold_bars * 3,
            )
            pnl_delta = sim['pnl'] - trade.pnl
            result.scenarios[f'tp_{mult}x'] = CounterfactualResult(
                scenario=f'tp_{mult}x',
                param_changed='take_profit',
                param_original=actual_tp,
                param_alternative=alt_tp,
                exit_price=sim['exit_price'],
                exit_reason=sim['exit_reason'],
                pnl=sim['pnl'],
                pnl_pct=sim['pnl_pct'],
                duration_hours=sim['duration_hours'],
                pnl_delta=pnl_delta,
                pnl_pct_delta=sim['pnl_pct'] - trade.pnl_pct,
            )

        # --- 3. Hold Time Sensitivity ---
        for mult in self.HOLD_MULTS:
            alt_bars = max(1, int(actual_hold_bars * mult))
            sim = self._simulate(
                future_bars, entry_price, direction,
                stop_loss=actual_sl, take_profit=actual_tp,
                max_bars=alt_bars,
            )
            pnl_delta = sim['pnl'] - trade.pnl
            result.scenarios[f'hold_{mult}x'] = CounterfactualResult(
                scenario=f'hold_{mult}x',
                param_changed='hold_time',
                param_original=actual_hold,
                param_alternative=actual_hold * mult,
                exit_price=sim['exit_price'],
                exit_reason=sim['exit_reason'],
                pnl=sim['pnl'],
                pnl_pct=sim['pnl_pct'],
                duration_hours=sim['duration_hours'],
                pnl_delta=pnl_delta,
                pnl_pct_delta=sim['pnl_pct'] - trade.pnl_pct,
            )

        # --- 4. No scale-out: hold to full SL or TP ---
        sim = self._simulate(
            future_bars, entry_price, direction,
            stop_loss=actual_sl, take_profit=actual_tp,
            max_bars=actual_hold_bars * 3,
        )
        pnl_delta = sim['pnl'] - trade.pnl
        result.scenarios['no_scale_out'] = CounterfactualResult(
            scenario='no_scale_out',
            param_changed='scale_out',
            param_original=0.0,
            param_alternative=0.0,
            exit_price=sim['exit_price'],
            exit_reason=sim['exit_reason'],
            pnl=sim['pnl'],
            pnl_pct=sim['pnl_pct'],
            duration_hours=sim['duration_hours'],
            pnl_delta=pnl_delta,
            pnl_pct_delta=sim['pnl_pct'] - trade.pnl_pct,
        )

        # --- Compute summary stats ---
        if result.scenarios:
            best = max(result.scenarios.values(), key=lambda s: s.pnl_delta)
            worst = min(result.scenarios.values(), key=lambda s: s.pnl_delta)
            result.best_scenario = best.scenario
            result.best_pnl_delta = best.pnl_delta
            result.worst_scenario = worst.scenario
            result.worst_pnl_delta = worst.pnl_delta
            # "Optimal" if no scenario is more than $50 better
            result.was_optimal = best.pnl_delta < 50.0

        return result

    def analyze_all(self, trades: list) -> List[TradeCounterfactual]:
        """Analyze all trades and return counterfactual results."""
        results = []
        for i, trade in enumerate(trades):
            if i % 100 == 0 and i > 0:
                logger.info(f"Counterfactual analysis: {i}/{len(trades)} trades processed")
            try:
                result = self.analyze_trade(trade)
                results.append(result)
            except Exception as e:
                logger.warning(f"Counterfactual failed for trade {i} ({trade.archetype}): {e}")

        logger.info(f"Counterfactual analysis complete: {len(results)} trades analyzed")
        return results

    def _simulate(
        self,
        bars: pd.DataFrame,
        entry_price: float,
        direction: str,
        stop_loss: float,
        take_profit: float,
        max_bars: int = 500,
    ) -> Dict[str, Any]:
        """
        Simulate a single trade scenario on historical bars.

        Returns dict with exit_price, exit_reason, pnl, pnl_pct, duration_hours.
        """
        slippage_mult = self.slippage_bps / 10000.0

        for i, (ts, bar) in enumerate(bars.iterrows()):
            if i == 0:
                continue  # Skip entry bar

            if i >= max_bars:
                # Time exit at close
                exit_price = bar['close']
                if direction == 'long':
                    exit_price *= (1 - slippage_mult)
                else:
                    exit_price *= (1 + slippage_mult)
                return self._calc_result(entry_price, exit_price, direction, 'time_exit', i, bars.index[0])

            # Check stop loss and take profit
            if direction == 'long':
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss * (1 - slippage_mult)
                    return self._calc_result(entry_price, exit_price, direction, 'stop_loss', i, bars.index[0])
                if bar['high'] >= take_profit:
                    exit_price = take_profit * (1 - slippage_mult)
                    return self._calc_result(entry_price, exit_price, direction, 'take_profit', i, bars.index[0])
            else:
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss * (1 + slippage_mult)
                    return self._calc_result(entry_price, exit_price, direction, 'stop_loss', i, bars.index[0])
                if bar['low'] <= take_profit:
                    exit_price = take_profit * (1 + slippage_mult)
                    return self._calc_result(entry_price, exit_price, direction, 'take_profit', i, bars.index[0])

        # End of data
        last_bar = bars.iloc[-1]
        exit_price = last_bar['close']
        return self._calc_result(entry_price, exit_price, direction, 'end_of_data', len(bars) - 1, bars.index[0])

    def _calc_result(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
        exit_reason: str,
        bars_held: int,
        entry_ts: pd.Timestamp,
    ) -> Dict[str, Any]:
        """Calculate PnL for a simulated scenario."""
        if direction == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        # Subtract commission (both sides)
        pnl_pct -= self.commission_rate * 2 * 100

        # Rough PnL in dollars (assume $1000 position for normalization)
        pnl = pnl_pct / 100 * 1000.0

        return {
            'exit_price': round(exit_price, 2),
            'exit_reason': exit_reason,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 4),
            'duration_hours': bars_held,  # 1H bars = hours
        }

    def _scale_stop_loss(
        self, entry_price: float, actual_sl: float, direction: str, mult: float
    ) -> float:
        """Scale stop loss distance by multiplier."""
        distance = abs(entry_price - actual_sl)
        scaled_distance = distance * mult
        if direction == 'long':
            return entry_price - scaled_distance
        else:
            return entry_price + scaled_distance

    def _scale_take_profit(
        self, entry_price: float, actual_tp: float, direction: str, mult: float
    ) -> float:
        """Scale take profit distance by multiplier."""
        distance = abs(actual_tp - entry_price)
        scaled_distance = distance * mult
        if direction == 'long':
            return entry_price + scaled_distance
        else:
            return entry_price - scaled_distance

    @staticmethod
    def summarize(results: List[TradeCounterfactual]) -> Dict[str, Any]:
        """
        Generate aggregate summary across all trades.

        Returns insights like:
        - What % of trades were optimal?
        - Which parameter change would have helped most?
        - Average PnL improvement from best scenario
        """
        if not results:
            return {}

        total = len(results)
        optimal_count = sum(1 for r in results if r.was_optimal)

        # Aggregate by scenario type
        scenario_improvements: Dict[str, List[float]] = {}
        for r in results:
            for name, s in r.scenarios.items():
                if name not in scenario_improvements:
                    scenario_improvements[name] = []
                scenario_improvements[name].append(s.pnl_delta)

        # Average improvement per scenario
        avg_by_scenario = {
            name: round(np.mean(deltas), 2)
            for name, deltas in scenario_improvements.items()
        }

        # Best overall scenario
        best_scenario = max(avg_by_scenario.items(), key=lambda x: x[1]) if avg_by_scenario else ("none", 0.0)
        worst_scenario = min(avg_by_scenario.items(), key=lambda x: x[1]) if avg_by_scenario else ("none", 0.0)

        # Group by parameter type
        param_types: Dict[str, List[float]] = {}
        for name, deltas in scenario_improvements.items():
            param = name.split('_')[0]  # 'sl', 'tp', 'hold', 'no'
            if param not in param_types:
                param_types[param] = []
            param_types[param].extend(deltas)

        avg_by_param = {
            param: round(np.mean(deltas), 2)
            for param, deltas in param_types.items()
        }

        return {
            'total_trades_analyzed': total,
            'optimal_trades_pct': round(optimal_count / total * 100, 1) if total > 0 else 0.0,
            'best_overall_scenario': best_scenario[0],
            'best_overall_avg_improvement': best_scenario[1],
            'worst_overall_scenario': worst_scenario[0],
            'worst_overall_avg_improvement': worst_scenario[1],
            'avg_improvement_by_scenario': avg_by_scenario,
            'avg_improvement_by_param_type': avg_by_param,
            'insights': _generate_insights(avg_by_scenario, optimal_count, total),
        }


def _generate_insights(avg_by_scenario: dict, optimal_count: int, total: int) -> List[str]:
    """Generate human-readable insights from counterfactual analysis."""
    insights = []

    pct_optimal = optimal_count / total * 100 if total > 0 else 0

    if pct_optimal > 70:
        insights.append(f"{pct_optimal:.0f}% of trades were near-optimal -- exit logic is well-calibrated")
    elif pct_optimal > 40:
        insights.append(f"{pct_optimal:.0f}% of trades were near-optimal -- room for improvement")
    else:
        insights.append(f"Only {pct_optimal:.0f}% of trades were near-optimal -- exit logic needs tuning")

    # SL insights
    sl_scenarios = {k: v for k, v in avg_by_scenario.items() if k.startswith('sl_')}
    if sl_scenarios:
        best_sl = max(sl_scenarios.items(), key=lambda x: x[1])
        if best_sl[1] > 10:
            mult = best_sl[0].replace('sl_', '').replace('x', '')
            if float(mult) > 1.0:
                insights.append(f"Wider stop loss ({mult}x) would improve avg PnL by ${best_sl[1]:.0f} -- current SL may be too tight")
            else:
                insights.append(f"Tighter stop loss ({mult}x) would improve avg PnL by ${best_sl[1]:.0f} -- current SL may be too wide")

    # TP insights
    tp_scenarios = {k: v for k, v in avg_by_scenario.items() if k.startswith('tp_')}
    if tp_scenarios:
        best_tp = max(tp_scenarios.items(), key=lambda x: x[1])
        if best_tp[1] > 10:
            mult = best_tp[0].replace('tp_', '').replace('x', '')
            if float(mult) < 1.0:
                insights.append(f"Closer take profit ({mult}x) would improve avg PnL by ${best_tp[1]:.0f} -- current TP may be too ambitious")
            else:
                insights.append(f"Wider take profit ({mult}x) would improve avg PnL by ${best_tp[1]:.0f} -- capturing more upside possible")

    # Hold time insights
    hold_scenarios = {k: v for k, v in avg_by_scenario.items() if k.startswith('hold_')}
    if hold_scenarios:
        best_hold = max(hold_scenarios.items(), key=lambda x: x[1])
        if best_hold[1] > 10:
            mult = best_hold[0].replace('hold_', '').replace('x', '')
            if float(mult) > 1.0:
                insights.append(f"Longer hold time ({mult}x) would improve avg PnL by ${best_hold[1]:.0f}")
            else:
                insights.append(f"Shorter hold time ({mult}x) would improve avg PnL by ${best_hold[1]:.0f}")

    # Scale-out insight
    no_scale = avg_by_scenario.get('no_scale_out', 0)
    if no_scale > 20:
        insights.append(f"Holding to full TP (no scale-out) would improve avg PnL by ${no_scale:.0f} -- scale-outs may be premature")
    elif no_scale < -20:
        insights.append(f"Scale-outs saved avg ${abs(no_scale):.0f} per trade -- partial exits are protecting profits")

    return insights
