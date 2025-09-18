
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

                    # Check stop/TP fills before processing new signals
                    stop_tp_fills = self.broker.mark(bar.name, sym, bar['close'])
                    if stop_tp_fills:
                        for fill in stop_tp_fills:
                            self.portfolio.on_fill(sym, fill['side'], fill['price'], fill['size_filled'], fill['fee'])
                            trades.append({
                                "ts": bar.name, "symbol": sym, "tf": tf,
                                "action": fill['side'], "price": float(fill['price']),
                                "size": float(fill['size_filled']), "fee": float(fill['fee']),
                                "pnl": float(fill.get('pnl', 0.0)),
                                "reason": fill.get('reason', '')
                            })

                    if action in ('long','short'):
                        # Check exposure limits before opening position
                        risk_amount = signal.get('size', 1.0) * bar['close']  # Rough estimate
                        current_equity = self.portfolio.equity()

                        if self.portfolio.can_add(action, risk_amount, current_equity):
                            risk_plan = signal.get('risk_plan')
                            fill = self.broker.submit(
                                ts=bar.name, symbol=sym, side=action,
                                size=signal.get('size',1.0), price_hint=bar['close'],
                                risk_plan=risk_plan
                            )
                            self.portfolio.on_fill(sym, action, fill['price'], fill['size_filled'], fill['fee'])
                            trades.append({
                                "ts": bar.name, "symbol": sym, "tf": tf,
                                "action": f"enter_{action}", "price": float(fill['price']),
                                "size": float(fill['size_filled']), "fee": float(fill['fee']),
                                "pnl": 0.0,
                                "reasons": signal.get('reasons', [])
                            })
                        else:
                            # Log exposure limit rejection
                            trades.append({
                                "ts": bar.name, "symbol": sym, "tf": tf,
                                "action": "rejected", "price": float(bar['close']),
                                "size": 0.0, "fee": 0.0, "pnl": 0.0,
                                "reason": "exposure_limit"
                            })

                    elif action == 'exit':
                        fill = self.broker.close(ts=bar.name, symbol=sym, price=bar['close'])
                        if fill.get('ts'):
                            self.portfolio.on_fill(sym, 'exit', fill['price'], fill['size_filled'], fill['fee'])
                            trades.append({
                                "ts": bar.name, "symbol": sym, "tf": tf,
                                "action": "exit", "price": float(fill['price']),
                                "size": float(fill['size_filled']), "fee": float(fill['fee']),
                                "pnl": float(fill.get('pnl', 0.0))
                            })

                    self.portfolio.mark(sym, float(bar['close']))
                    equity_rows.append({"ts":bar.name, "symbol":sym, "equity": self.portfolio.equity()})
        trades_df = pd.DataFrame(trades)
        eq_df = pd.DataFrame(equity_rows).set_index('ts') if equity_rows else pd.DataFrame(columns=['equity'])
        metrics = {}
        metrics.update(trade_metrics(trades_df if not trades_df.empty else pd.DataFrame(columns=['action','pnl'])))
        metrics.update(equity_metrics(eq_df if not eq_df.empty else pd.DataFrame(columns=['equity'])))
        artifacts = write_report(self.cfg.get('run_id','v1_4_demo'), self.cfg, metrics, trades_df, eq_df, out_dir)
        return {"metrics": metrics, "artifacts": artifacts}
