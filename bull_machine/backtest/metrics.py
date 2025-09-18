
import numpy as np
import pandas as pd

def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades":0,"win_rate":0.0,"avg_win":0.0,"avg_loss":0.0,"expectancy":0.0}
    exits = trades[trades['action'].isin(['exit','stop','tp'])]
    wins = exits[exits['pnl'] > 0]
    losses = exits[exits['pnl'] <= 0]
    avg_win = wins['pnl'].mean() if len(wins)>0 else 0.0
    avg_loss = losses['pnl'].mean() if len(losses)>0 else 0.0
    wr = len(wins) / max(1,len(exits))
    expectancy = wr*(avg_win) + (1-wr)*(avg_loss)
    return {"trades": int(len(exits)), "win_rate": float(wr), "avg_win": float(avg_win), "avg_loss": float(avg_loss), "expectancy": float(expectancy)}

def equity_metrics(equity: pd.DataFrame, risk_free: float = 0.0) -> dict:
    if equity.empty or 'equity' not in equity: return {"sharpe":0.0,"max_dd":0.0,"cagr":0.0}
    ret = equity['equity'].pct_change().dropna()
    sharpe = 0.0 if ret.std()==0 else (ret.mean()-risk_free/252)/ret.std()*np.sqrt(252)
    cummax = equity['equity'].cummax()
    dd_series = (cummax - equity['equity'])
    max_dd = dd_series.max()
    if len(equity) > 1:
        years = (equity.index[-1] - equity.index[0]).days/365.25
    else:
        years = 0.0
    cagr = 0.0 if years<=0 else (equity['equity'].iloc[-1]/max(1e-9,equity['equity'].iloc[0]))**(1/years)-1
    return {"sharpe": float(sharpe), "max_dd": float(max_dd), "cagr": float(cagr)}
