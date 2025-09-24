
import numpy as np
import pandas as pd
import math

def safe_cagr(start_equity: float, end_equity: float, days: float) -> float:
    """Compute CAGR safely, returning None for invalid inputs."""
    if start_equity <= 0 or end_equity <= 0 or days <= 0:
        return None
    years = days / 365.25
    if years <= 0:
        return None
    try:
        cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0
        return None if (math.isnan(cagr) or math.isinf(cagr)) else cagr
    except Exception:
        return None

def safe_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Compute Sharpe ratio safely, returning None for invalid inputs."""
    if len(returns) < 2:
        return None
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    if std_ret == 0 or math.isnan(std_ret) or math.isnan(mean_ret):
        return None
    try:
        sharpe = (mean_ret - risk_free/252) / std_ret * np.sqrt(252)
        return None if (math.isnan(sharpe) or math.isinf(sharpe)) else sharpe
    except Exception:
        return None

def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades":0,"entries":0,"win_rate":0.0,"avg_win":0.0,"avg_loss":0.0,"expectancy":0.0}

    # Count entries for diagnostic purposes
    entries = trades[trades['action'].str.contains('enter', na=False)]
    entry_count = len(entries)

    # Count exits for official metrics (stop, TP levels, manual exits)
    exit_actions = ['exit', 'stop', 'tp1', 'tp2', 'tp3', 'close_remaining']
    exits = trades[trades['action'].isin(exit_actions)]

    # Also check for actions starting with 'tp' (like 'tp1', 'tp2', etc.)
    tp_exits = trades[trades['action'].str.startswith('tp', na=False)]
    exits = pd.concat([exits, tp_exits]).drop_duplicates()

    wins = exits[exits['pnl'] > 0]
    losses = exits[exits['pnl'] <= 0]

    # NaN-safe metric computation
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0.0

    # Handle NaN in averages
    avg_win = 0.0 if math.isnan(avg_win) else avg_win
    avg_loss = 0.0 if math.isnan(avg_loss) else avg_loss

    wr = len(wins) / max(1, len(exits))
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    return {
        "trades": int(len(exits)),
        "entries": int(entry_count),
        "win_rate": float(wr),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy)
    }

def equity_metrics(equity: pd.DataFrame, risk_free: float = 0.0) -> dict:
    if equity.empty or 'equity' not in equity:
        return {"sharpe": None, "max_dd": 0.0, "cagr": None}

    # Compute returns for Sharpe
    ret = equity['equity'].pct_change().dropna()
    sharpe = safe_sharpe(ret, risk_free)

    # Compute max drawdown
    cummax = equity['equity'].cummax()
    dd_series = (cummax - equity['equity'])
    max_dd = dd_series.max()
    max_dd = 0.0 if math.isnan(max_dd) else max_dd

    # Compute CAGR safely
    cagr = None
    if len(equity) > 1:
        start_equity = equity['equity'].iloc[0]
        end_equity = equity['equity'].iloc[-1]
        days = (equity.index[-1] - equity.index[0]).days
        cagr = safe_cagr(start_equity, end_equity, days)

    return {"sharpe": sharpe, "max_dd": float(max_dd), "cagr": cagr}
