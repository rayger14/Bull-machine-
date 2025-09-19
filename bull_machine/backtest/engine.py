
from typing import List, Dict, Callable, Optional
import pandas as pd
import logging
from .metrics import trade_metrics, equity_metrics
from .report import write_report

class BacktestEngine:
    def __init__(self, cfg: dict, datafeed, broker, portfolio, exit_evaluator=None):
        self.cfg = cfg
        self.datafeed = datafeed
        self.broker = broker
        self.portfolio = portfolio
        self.exit_evaluator = exit_evaluator

        # Initialize exit evaluator if config provided
        if not self.exit_evaluator and 'exit_signals' in cfg:
            try:
                from bull_machine.strategy.exits import ExitSignalEvaluator
                self.exit_evaluator = ExitSignalEvaluator(cfg['exit_signals'])
                logging.info("[ENGINE] Exit signal evaluator initialized from config")
            except ImportError as e:
                logging.warning(f"[ENGINE] Could not initialize exit evaluator: {e}")
                self.exit_evaluator = None

    def run(self, strategy_fn: Callable, symbols: List[str], tfs: List[str], out_dir: str = "out") -> dict:
        trades = []
        equity_rows = []
        lookback = self.cfg.get('engine',{}).get('lookback_bars', 250)

        # 🚀 PERFORMANCE FIX 1: Cache resamples once per (symbol, TF)
        print(f"📊 Precomputing resampled data for {len(symbols)} symbols × {len(tfs)} timeframes...")
        cached_frames = {}
        for sym in symbols:
            base = self.datafeed.frames.get(sym)
            if base is None or base.empty:
                continue
            for tf in tfs:
                key = (sym, tf)
                cached_frames[key] = self.datafeed.resample(base, tf)
                print(f"   ✅ {sym} {tf}: {len(cached_frames[key])} bars")

        # 🚀 PERFORMANCE FIX 2: Precompute rolling features vectorized
        print(f"📈 Precomputing rolling features (ATR, MA, etc.)...")
        for key, df_tf in cached_frames.items():
            # Add common indicators as columns (vectorized)
            df_tf['sma20'] = df_tf['close'].rolling(20, min_periods=1).mean()
            df_tf['sma50'] = df_tf['close'].rolling(50, min_periods=1).mean()
            df_tf['high_low'] = df_tf['high'] - df_tf['low']
            df_tf['atr14'] = df_tf['high_low'].rolling(14, min_periods=1).mean()
            print(f"   ✅ {key[0]} {key[1]}: Features added")

        print(f"🔄 Running backtest with cached data...")

        for sym in symbols:
            if sym not in [k[0] for k in cached_frames.keys()]:
                continue
            for tf in tfs:
                key = (sym, tf)
                df_tf = cached_frames.get(key)
                if df_tf is None or len(df_tf) <= lookback:
                    continue

                signal_count = 0
                for i in range(lookback, len(df_tf)):
                    # 🚀 PERFORMANCE FIX 3: Pass index instead of window copy
                    bar = df_tf.iloc[i-1]
                    signal = strategy_fn(sym, tf, df_tf, i)  # Pass (df, index) not window

                    # Debug signal processing
                    if signal is not None:
                        signal_count += 1
                        logging.debug(f"[ENGINE] Signal {signal_count} @ {bar.name}: {signal}")

                        # Validate signal format
                        if not isinstance(signal, dict):
                            logging.error(f"[ENGINE] Invalid signal type: {type(signal)}")
                            continue

                        if 'action' not in signal:
                            logging.error(f"[ENGINE] Signal missing 'action' field: {signal}")
                            continue

                    # Evaluate exit signals for existing positions before processing new entry signals
                    exit_signal = None
                    if self.exit_evaluator and sym in self.broker.positions:
                        exit_signal = self._evaluate_exit_signals(sym, cached_frames, bar)

                    # Check stop/TP fills with potential exit signal override
                    stop_tp_fills = self.broker.mark(bar.name, sym, bar['close'], exit_signal)
                    if stop_tp_fills:
                        for fill in stop_tp_fills:
                            self.portfolio.on_fill(sym, fill['side'], fill['price'], fill['size_filled'], fill['fee'])
                            trades.append({
                                "ts": bar.name, "symbol": sym, "tf": tf,
                                "action": fill['side'], "price": float(fill['price']),
                                "size": float(fill['size_filled']), "fee": float(fill['fee']),
                                "pnl": float(fill.get('pnl', 0.0)),
                                "reason": fill.get('reason', ''),
                                "confidence": fill.get('confidence', 0.0),
                                "urgency": fill.get('urgency', 0.0)
                            })

                    if not signal:
                        continue
                    action = signal.get('action','flat')

                    if action in ('long','short'):
                        # Check exposure limits before opening position
                        risk_amount = signal.get('size', 1.0) * bar['close']  # Rough estimate
                        current_equity = self.portfolio.equity()

                        if self.portfolio.can_add(action, risk_amount, current_equity):
                            risk_plan = signal.get('risk_plan')
                            logging.info(f"[ENGINE] Opening {action} position for {sym} @ {bar['close']:.4f}, size={signal.get('size',1.0)}")
                            fill = self.broker.submit(
                                ts=bar.name, symbol=sym, side=action,
                                size=signal.get('size',1.0), price_hint=bar['close'],
                                risk_plan=risk_plan
                            )
                            self.portfolio.on_fill(sym, action, fill['price'], fill['size_filled'], fill['fee'])
                            # 🚀 PERFORMANCE FIX 4: Minimal trade logging in loop
                            trades.append({
                                "ts": bar.name, "symbol": sym, "tf": tf,
                                "action": f"enter_{action}", "price": float(fill['price']),
                                "size": float(fill['size_filled']), "fee": float(fill['fee']),
                                "pnl": 0.0,
                                "reasons": len(signal.get('reasons', []))  # Just count, not full list
                            })
                            logging.info(f"[ENGINE] Trade executed: {fill}")
                        else:
                            logging.warning(f"[ENGINE] Trade rejected due to risk limits: {action} {sym} @ {bar['close']:.4f}")

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

                # Log signal statistics for this symbol/timeframe
                logging.info(f"[ENGINE] {sym} {tf}: Processed {signal_count} signals out of {len(df_tf) - lookback} bars")
        trades_df = pd.DataFrame(trades)
        eq_df = pd.DataFrame(equity_rows).set_index('ts') if equity_rows else pd.DataFrame(columns=['equity'])
        metrics = {}
        metrics.update(trade_metrics(trades_df if not trades_df.empty else pd.DataFrame(columns=['action','pnl'])))
        metrics.update(equity_metrics(eq_df if not eq_df.empty else pd.DataFrame(columns=['equity'])))
        artifacts = write_report(self.cfg.get('run_id','v1_4_demo'), self.cfg, metrics, trades_df, eq_df, out_dir)
        return {"metrics": metrics, "artifacts": artifacts}

    def _evaluate_exit_signals(self, symbol: str, cached_frames: Dict, current_bar) -> Optional[object]:
        """
        Evaluate exit signals for a symbol at current bar.

        Args:
            symbol: Trading symbol
            cached_frames: Cached multi-timeframe data
            current_bar: Current bar data

        Returns:
            ExitSignal object if exit detected, None otherwise
        """
        try:
            # Get position data from broker
            position_data = self.broker.get_position_data(symbol)
            if not position_data:
                return None

            # Prepare multi-timeframe data for exit evaluation
            mtf_data = {}
            for tf in ['1H', '4H', '1D']:  # Standard timeframes
                key = (symbol, tf)
                if key in cached_frames:
                    mtf_data[tf] = cached_frames[key]

            if not mtf_data:
                return None

            # Update position data with current market info
            position_data['entry_time'] = current_bar.name  # Approximate for now
            position_data['pnl_pct'] = self._calculate_current_pnl_pct(
                position_data, current_bar['close']
            )

            # Evaluate exit conditions
            result = self.exit_evaluator.evaluate_exits(
                symbol, position_data, mtf_data, current_bar.name
            )

            # Get recommended action
            exit_signal = self.exit_evaluator.get_action_recommendation(result)

            if exit_signal:
                logging.info(f"[EXIT] {symbol}: {exit_signal.exit_type.value} "
                           f"(confidence: {exit_signal.confidence:.2f}, "
                           f"urgency: {exit_signal.urgency:.2f})")

            return exit_signal

        except Exception as e:
            logging.error(f"[EXIT] Error evaluating exit signals for {symbol}: {e}")
            return None

    def _calculate_current_pnl_pct(self, position_data: Dict, current_price: float) -> float:
        """Calculate current PnL percentage for position."""
        try:
            entry_price = position_data.get('entry_price', current_price)
            if entry_price == 0:
                return 0.0

            if position_data.get('bias') == 'long':
                return (current_price - entry_price) / entry_price
            else:
                return (entry_price - current_price) / entry_price

        except (TypeError, ZeroDivisionError):
            return 0.0
