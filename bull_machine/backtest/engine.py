
from typing import List, Dict, Callable
import pandas as pd
from .metrics import trade_metrics, equity_metrics
from .report import write_report

class BacktestEngine:
    def __init__(self, cfg: dict, datafeed, broker, portfolio):
        self.cfg = cfg
        self.datafeed = datafeed
        self.broker = broker
        self.portfolio = portfolio

    def run(self, strategy_fn: Callable, symbols: List[str], tfs: List[str], out_dir: str = "out") -> dict:
        trades = []
        equity_rows = []
        lookback = self.cfg.get('engine',{}).get('lookback_bars', 250)
        for sym in symbols:
            base = self.datafeed.frames.get(sym)
            if base is None or base.empty: 
                continue
            for tf in tfs:
                df_tf = self.datafeed.resample(base, tf)
                for i in range(lookback, len(df_tf)):
                    window = df_tf.iloc[:i]
                    bar = df_tf.iloc[i-1]
                    signal = strategy_fn(sym, tf, window)
                    if not signal: continue
                    action = signal.get('action','flat')
                    if action in ('long','short'):
                        fill = self.broker.submit(ts=bar.name, symbol=sym, side=action, size=signal.get('size',1.0), price_hint=bar['close'])
                        self.portfolio.on_fill(sym, action, fill['price'], fill['size_filled'], fill['fee'])
                        trades.append({"ts":bar.name, "symbol":sym, "tf":tf, "action":f"enter_{action}", "price":float(fill['price']), "size":float(fill['size_filled']), "fee":float(fill['fee']), "pnl":0.0})
                    elif action == 'exit':
                        fill = self.broker.close(ts=bar.name, symbol=sym)
                        if fill.get('ts'):
                            self.portfolio.on_fill(sym, 'exit', fill['price'], fill['size_filled'], fill['fee'])
                            trades.append({"ts":bar.name, "symbol":sym, "tf":tf, "action":"exit", "price":float(fill['price']), "size":float(fill['size_filled']), "fee":float(fill['fee']), "pnl":0.0})
                    self.portfolio.mark(sym, float(bar['close']))
                    equity_rows.append({"ts":bar.name, "symbol":sym, "equity": self.portfolio.equity()})
        trades_df = pd.DataFrame(trades)
        eq_df = pd.DataFrame(equity_rows).set_index('ts') if equity_rows else pd.DataFrame(columns=['equity'])
        metrics = {}
        metrics.update(trade_metrics(trades_df if not trades_df.empty else pd.DataFrame(columns=['action','pnl'])))
        metrics.update(equity_metrics(eq_df if not eq_df.empty else pd.DataFrame(columns=['equity'])))
        artifacts = write_report(self.cfg.get('run_id','v1_4_demo'), self.cfg, metrics, trades_df, eq_df, out_dir)
        return {"metrics": metrics, "artifacts": artifacts}
