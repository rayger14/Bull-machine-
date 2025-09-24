
import argparse
import json
import logging
from pathlib import Path

from bull_machine.backtest.broker import PaperBroker
from bull_machine.backtest.datafeed import DataFeed
from bull_machine.backtest.engine import BacktestEngine
from bull_machine.backtest.portfolio import Portfolio

try:
    # Try optimized adapter first
    from bull_machine.backtest.strategy_adapter_optimized import strategy_from_df
    print("🚀 Using optimized strategy adapter")
except ImportError:
    # Fallback to integrated adapter
    from bull_machine.backtest.strategy_adapter_v13_integrated import strategy_from_df
    print("⚠️ Using fallback integrated adapter")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/v14_example.json')
    p.add_argument('--out', default='out/v14_run')
    p.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = p.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    cfg = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {
        "run_id": "v1_4_demo",
        "data":{"sources":{},"timeframes":["1H"]},
        "broker":{"fee_bps":2,"slippage_bps":3,"spread_bps":1},
        "portfolio":{"starting_cash":100000},
        "engine":{"lookback_bars":250,"seed":42}
    }

    feed = DataFeed(cfg['data']['sources']) if cfg['data']['sources'] else DataFeed({})

    # Data gate - fail fast if no symbols loaded
    if not feed.frames:
        raise SystemExit("[ENGINE] No dataframes loaded. Aborting.")

    print(f"[ENGINE] Symbols loaded: {list(feed.frames.keys())}")
    for s, df in feed.frames.items():
        if not df.empty:
            print(f"[ENGINE] {s}: rows={len(df)}, first={df.index[0].strftime('%Y-%m-%d %H:%M:%S')}, last={df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")

    broker = PaperBroker(**cfg.get('broker',{}))

    # Extract portfolio config with max_positions support
    portfolio_cfg = cfg.get('portfolio', {})
    starting_cash = portfolio_cfg.get('starting_cash', 100000)
    max_positions = portfolio_cfg.get('max_positions', 8)
    exposure_cap_pct = portfolio_cfg.get('exposure_cap_pct', 0.5)

    port = Portfolio(starting_cash, exposure_cap_pct, max_positions)
    eng = BacktestEngine(cfg, feed, broker, port)

    def v13_strategy(symbol: str, tf: str, df_or_bars, index_or_balance=None):
        """Adaptive strategy function that handles both old and new calling conventions"""
        balance = cfg["portfolio"]["starting_cash"]
        config_path = cfg.get("strategy",{}).get("config","config/production.json")

        if isinstance(index_or_balance, int):
            # New optimized calling convention: (symbol, tf, df_full, index, balance, config_path)
            return strategy_from_df(symbol, tf, df_or_bars, index_or_balance, balance, config_path)
        else:
            # Old calling convention: (symbol, tf, df_window, balance)
            return strategy_from_df(symbol, tf, df_or_bars, balance, config_path)

    syms = list(cfg.get('data',{}).get('sources',{}).keys())
    tfs = cfg.get('data',{}).get('timeframes',['1H'])
    res = eng.run(v13_strategy, syms, tfs, out_dir=args.out)

    # Import json_sanitize for NaN-safe output
    from bull_machine.backtest.report import json_sanitize
    clean_output = json_sanitize({"ok":True, "metrics":res['metrics'], "artifacts":res['artifacts']})
    print(json.dumps(clean_output, default=str))

if __name__ == '__main__':
    main()
