"""
Cost-Adjusted Performance Metrics for Bull Machine v1.7
Computes PF/DD from net PnL after transaction costs
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from engine.risk.transaction_costs import TransactionCostModel

class CostAdjustedMetrics:
    """
    Compute performance metrics from cost-adjusted trades
    Ensures PF/DD reflect realistic transaction costs
    """

    def __init__(self, cost_model: TransactionCostModel = None):
        self.cost_model = cost_model or TransactionCostModel()

    def apply_costs_to_trades(self, trades: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """
        Apply transaction costs to trade list and recompute net PnL

        Args:
            trades: List of trade dictionaries
            price_data: OHLCV data for volatility calculation

        Returns:
            Trades with cost-adjusted PnL
        """

        cost_adjusted_trades = []

        for trade in trades:
            # Calculate volatility around trade period
            entry_idx = self._find_bar_index(price_data, trade['entry_timestamp'])
            exit_idx = self._find_bar_index(price_data, trade['exit_timestamp'])

            if entry_idx is None or exit_idx is None:
                # Skip trades with missing data
                continue

            # Get volatility for cost calculation
            vol_window = price_data.iloc[max(0, entry_idx-20):exit_idx+1]
            volatility = self.cost_model.calculate_volatility(vol_window)

            # Apply transaction costs
            cost_result = self.cost_model.apply_costs(
                entry_px=trade['entry_price'],
                exit_px=trade['exit_price'],
                qty=trade.get('quantity', 1.0),
                bar_volatility=volatility,
                stress_mode=trade.get('stress_mode', 'normal')
            )

            # Update trade with cost-adjusted values
            adjusted_trade = trade.copy()
            adjusted_trade.update({
                'gross_pnl': cost_result['gross_pnl'],
                'net_pnl': cost_result['net_pnl'],
                'total_cost': cost_result['total_cost'],
                'cost_bps': cost_result['cost_bps'],
                'volatility': volatility
            })

            cost_adjusted_trades.append(adjusted_trade)

        return cost_adjusted_trades

    def _find_bar_index(self, df: pd.DataFrame, timestamp) -> int:
        """Find index of bar closest to timestamp"""
        try:
            if pd.isna(timestamp):
                return None

            timestamp = pd.to_datetime(timestamp)
            closest_idx = df.index.get_indexer([timestamp], method='nearest')[0]

            if closest_idx == -1:
                return None

            return closest_idx

        except Exception:
            return None

    def compute_performance_metrics(self, cost_adjusted_trades: List[Dict]) -> Dict:
        """
        Compute performance metrics from cost-adjusted trades

        Args:
            cost_adjusted_trades: Trades with net PnL after costs

        Returns:
            Performance metrics dictionary
        """

        if not cost_adjusted_trades:
            return self._empty_metrics()

        # Extract net PnLs
        net_pnls = [trade['net_pnl'] for trade in cost_adjusted_trades]
        gross_pnls = [trade['gross_pnl'] for trade in cost_adjusted_trades]
        total_costs = [trade['total_cost'] for trade in cost_adjusted_trades]

        # Basic statistics
        total_net_pnl = sum(net_pnls)
        total_gross_pnl = sum(gross_pnls)
        total_cost = sum(total_costs)

        # Win/Loss analysis
        winning_trades = [pnl for pnl in net_pnls if pnl > 0]
        losing_trades = [pnl for pnl in net_pnls if pnl <= 0]

        win_rate = len(winning_trades) / len(net_pnls) if net_pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        # Profit Factor (critical metric)
        profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')

        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(net_pnls)

        # Cost impact analysis
        cost_drag_pct = (total_cost / abs(total_gross_pnl) * 100) if total_gross_pnl != 0 else 0

        # Sharpe-like metrics
        if len(net_pnls) > 1:
            pnl_std = np.std(net_pnls)
            sharpe_ratio = (np.mean(net_pnls) / pnl_std) if pnl_std > 0 else 0
        else:
            sharpe_ratio = 0

        # Trade efficiency
        avg_cost_bps = np.mean([trade['cost_bps'] for trade in cost_adjusted_trades])

        return {
            # Core performance
            'total_net_pnl': total_net_pnl,
            'total_gross_pnl': total_gross_pnl,
            'total_cost': total_cost,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown / abs(total_gross_pnl) * 100 if total_gross_pnl != 0 else 0,

            # Trade statistics
            'total_trades': len(cost_adjusted_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),

            # Cost analysis
            'cost_drag_pct': cost_drag_pct,
            'avg_cost_bps': avg_cost_bps,

            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'largest_win': max(net_pnls) if net_pnls else 0,
            'largest_loss': min(net_pnls) if net_pnls else 0,

            # Quality metrics
            'expectancy': np.mean(net_pnls) if net_pnls else 0,
            'kelly_criterion': self._calculate_kelly(winning_trades, losing_trades, win_rate)
        }

    def _calculate_max_drawdown(self, pnls: List[float]) -> float:
        """Calculate maximum drawdown from PnL series"""
        if not pnls:
            return 0

        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max

        return abs(np.min(drawdowns))

    def _calculate_kelly(self, winning_trades: List[float], losing_trades: List[float], win_rate: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if not winning_trades or not losing_trades or win_rate == 0:
            return 0

        avg_win = np.mean(winning_trades)
        avg_loss = abs(np.mean(losing_trades))

        if avg_loss == 0:
            return 0

        # Kelly = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate

        kelly = (b * p - q) / b

        # Cap Kelly at reasonable levels
        return max(0, min(0.25, kelly))  # Max 25% of capital

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure"""
        return {
            'total_net_pnl': 0,
            'total_gross_pnl': 0,
            'total_cost': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'cost_drag_pct': 0,
            'avg_cost_bps': 0,
            'sharpe_ratio': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'expectancy': 0,
            'kelly_criterion': 0
        }

    def stress_test_metrics(self, trades: List[Dict], price_data: pd.DataFrame) -> Dict:
        """
        Run stress tests with different cost scenarios

        Args:
            trades: Original trade list
            price_data: Price data for volatility

        Returns:
            Metrics under different stress scenarios
        """

        scenarios = {}

        for stress_mode in ['normal', 'elevated', 'extreme']:
            # Apply stress mode to all trades
            stressed_trades = []
            for trade in trades:
                stressed_trade = trade.copy()
                stressed_trade['stress_mode'] = stress_mode
                stressed_trades.append(stressed_trade)

            # Get cost-adjusted trades for this scenario
            cost_adjusted = self.apply_costs_to_trades(stressed_trades, price_data)

            # Compute metrics
            metrics = self.compute_performance_metrics(cost_adjusted)

            scenarios[stress_mode] = metrics

        # Summary comparison
        pf_degradation = {
            'elevated': scenarios['elevated']['profit_factor'] / scenarios['normal']['profit_factor']
                       if scenarios['normal']['profit_factor'] > 0 else 0,
            'extreme': scenarios['extreme']['profit_factor'] / scenarios['normal']['profit_factor']
                      if scenarios['normal']['profit_factor'] > 0 else 0
        }

        return {
            'scenarios': scenarios,
            'pf_degradation': pf_degradation,
            'worst_case_pf': scenarios['extreme']['profit_factor'],
            'cost_sensitivity': scenarios['extreme']['cost_drag_pct'] - scenarios['normal']['cost_drag_pct']
        }


def test_cost_adjusted_metrics():
    """Test cost-adjusted metrics calculation"""

    print("📊 TESTING COST-ADJUSTED METRICS")
    print("=" * 50)

    # Create test trades
    test_trades = [
        {
            'entry_timestamp': '2025-01-01 10:00:00',
            'exit_timestamp': '2025-01-01 14:00:00',
            'entry_price': 100.0,
            'exit_price': 102.0,
            'quantity': 1.0
        },
        {
            'entry_timestamp': '2025-01-02 10:00:00',
            'exit_timestamp': '2025-01-02 18:00:00',
            'entry_price': 102.0,
            'exit_price': 101.0,
            'quantity': 1.0
        },
        {
            'entry_timestamp': '2025-01-03 09:00:00',
            'exit_timestamp': '2025-01-03 15:00:00',
            'entry_price': 101.0,
            'exit_price': 104.0,
            'quantity': 1.0
        }
    ]

    # Create mock price data
    dates = pd.date_range('2025-01-01', '2025-01-05', freq='1h')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

    price_data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(0, 0.5, len(dates)),
        'low': prices - np.random.uniform(0, 0.5, len(dates)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(dates))
    }, index=dates)

    # Test metrics calculation
    metrics_calc = CostAdjustedMetrics()

    # Apply costs to trades
    cost_adjusted_trades = metrics_calc.apply_costs_to_trades(test_trades, price_data)

    print(f"✅ Applied costs to {len(cost_adjusted_trades)} trades")

    # Compute performance metrics
    metrics = metrics_calc.compute_performance_metrics(cost_adjusted_trades)

    print(f"✅ Computed performance metrics:")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Max Drawdown: ${metrics['max_drawdown']:.2f}")
    print(f"   Win Rate: {metrics['win_rate']:.1%}")
    print(f"   Cost Drag: {metrics['cost_drag_pct']:.1f}%")

    # Stress test
    stress_results = metrics_calc.stress_test_metrics(test_trades, price_data)

    print(f"✅ Stress test completed:")
    print(f"   Normal PF: {stress_results['scenarios']['normal']['profit_factor']:.2f}")
    print(f"   Extreme PF: {stress_results['scenarios']['extreme']['profit_factor']:.2f}")
    print(f"   PF Degradation: {(1 - stress_results['pf_degradation']['extreme']) * 100:.1f}%")

    return True


if __name__ == "__main__":
    success = test_cost_adjusted_metrics()
    exit(0 if success else 1)