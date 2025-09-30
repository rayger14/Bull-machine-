"""
Transaction Cost Model for Bull Machine v1.7
Realistic modeling of fees, slippage, and spread with volatility scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

class TransactionCostModel:
    """
    Professional-grade transaction cost modeling with:
    - Volatility-scaled slippage
    - Dynamic spread widening
    - Stress scenario support
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize cost model with default or custom parameters"""
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        """Default realistic cost parameters"""
        return {
            'fee_bps': 10,           # 0.10% taker fee (Binance/Coinbase)
            'spread_bps': 2,         # 0.02% typical spread
            'slippage_k': 0.25,      # Slippage scaling factor
            'stress_multiplier': {
                'normal': 1.0,
                'elevated': 1.5,      # +50% costs
                'extreme': 2.0        # 2x costs (news/gaps)
            },
            'max_slippage_bps': 20,  # Cap at 0.20%
            'volatility_window': 20   # Bars for volatility calc
        }

    def apply_costs(self,
                   entry_px: float,
                   exit_px: float,
                   qty: float,
                   bar_volatility: float,
                   stress_mode: str = 'normal') -> Dict:
        """
        Calculate realistic transaction costs

        Args:
            entry_px: Entry price
            exit_px: Exit price
            qty: Position quantity
            bar_volatility: Recent bar volatility (std of returns)
            stress_mode: 'normal', 'elevated', or 'extreme'

        Returns:
            Dict with cost breakdown and adjusted P&L
        """
        # Get stress multiplier
        stress_mult = self.config['stress_multiplier'].get(stress_mode, 1.0)

        # Calculate components
        fee_bps = self.config['fee_bps'] * stress_mult
        spread_bps = self.config['spread_bps'] * stress_mult

        # Volatility-scaled slippage
        # slippage ≈ k * (spread + volatility) in bps
        vol_bps = bar_volatility * 10000  # Convert to bps
        slip_bps = self.config['slippage_k'] * (spread_bps + vol_bps)
        slip_bps = min(slip_bps, self.config['max_slippage_bps'])

        # Total round-trip costs
        total_bps = (fee_bps * 2) + spread_bps + slip_bps

        # Apply costs to notional value
        notional = (entry_px + exit_px) * qty * 0.5
        total_cost = notional * (total_bps / 10000)

        # Calculate gross and net P&L
        gross_pnl = qty * (exit_px - entry_px)
        net_pnl = gross_pnl - total_cost

        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_cost': total_cost,
            'cost_bps': total_bps,
            'fee_cost': notional * (fee_bps * 2 / 10000),
            'spread_cost': notional * (spread_bps / 10000),
            'slippage_cost': notional * (slip_bps / 10000),
            'stress_mode': stress_mode
        }

    def calculate_volatility(self, df: pd.DataFrame, window: Optional[int] = None) -> float:
        """
        Calculate recent volatility for slippage scaling

        Args:
            df: Price data with 'close' column
            window: Lookback window (default from config)

        Returns:
            Volatility as decimal (e.g., 0.02 for 2%)
        """
        window = window or self.config['volatility_window']

        if len(df) < window:
            window = len(df)

        returns = df['close'].pct_change().dropna()

        if len(returns) < 2:
            return 0.01  # Default 1% if insufficient data

        # Winsorize extreme values
        returns = returns.clip(lower=returns.quantile(0.01),
                              upper=returns.quantile(0.99))

        return returns.tail(window).std()

    def stress_test(self, trade: Dict, df: pd.DataFrame) -> Dict:
        """
        Run stress scenarios on a trade

        Args:
            trade: Trade dict with entry/exit prices and qty
            df: Price data around trade period

        Returns:
            Dict with normal, elevated, and extreme scenario results
        """
        vol = self.calculate_volatility(df)

        scenarios = {}
        for mode in ['normal', 'elevated', 'extreme']:
            result = self.apply_costs(
                entry_px=trade['entry_price'],
                exit_px=trade['exit_price'],
                qty=trade['quantity'],
                bar_volatility=vol,
                stress_mode=mode
            )
            scenarios[mode] = result

        return {
            'scenarios': scenarios,
            'worst_case_pnl': scenarios['extreme']['net_pnl'],
            'cost_impact_pct': {
                mode: (res['total_cost'] / abs(res['gross_pnl']) * 100
                      if res['gross_pnl'] != 0 else 0)
                for mode, res in scenarios.items()
            }
        }

    def batch_apply(self, trades: list, df: pd.DataFrame, mode: str = 'normal') -> Dict:
        """
        Apply costs to multiple trades

        Args:
            trades: List of trade dicts
            df: Price data DataFrame
            mode: Stress mode to apply

        Returns:
            Summary statistics
        """
        results = []

        for trade in trades:
            # Get volatility around trade period
            start_idx = df.index.get_loc(trade['entry_timestamp'])
            end_idx = df.index.get_loc(trade['exit_timestamp'])
            trade_df = df.iloc[max(0, start_idx-20):end_idx+1]

            vol = self.calculate_volatility(trade_df)

            cost_result = self.apply_costs(
                entry_px=trade['entry_price'],
                exit_px=trade['exit_price'],
                qty=trade.get('quantity', 1.0),
                bar_volatility=vol,
                stress_mode=mode
            )

            trade['cost_adjusted'] = cost_result
            results.append(cost_result)

        # Aggregate statistics
        total_gross = sum(r['gross_pnl'] for r in results)
        total_net = sum(r['net_pnl'] for r in results)
        total_costs = sum(r['total_cost'] for r in results)

        return {
            'total_gross_pnl': total_gross,
            'total_net_pnl': total_net,
            'total_costs': total_costs,
            'cost_drag_pct': (total_costs / abs(total_gross) * 100) if total_gross != 0 else 0,
            'avg_cost_bps': np.mean([r['cost_bps'] for r in results]),
            'trades_processed': len(trades)
        }