
import argparse, json
from pathlib import Path
from bull_machine.backtest.datafeed import DataFeed
from bull_machine.backtest.broker import PaperBroker
from bull_machine.backtest.portfolio import Portfolio
from bull_machine.backtest.engine import BacktestEngine
from bull_machine.backtest.strategy_adapter_v13 import strategy_from_df

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/v14_example.json')
    p.add_argument('--out', default='out/v14_run')
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {
        "run_id": "v1_4_demo",
        "data":{"sources":{},"timeframes":["1H"]},
        "broker":{"fee_bps":2,"slippage_bps":3,"spread_bps":1},
        "portfolio":{"starting_cash":100000},
        "engine":{"lookback_bars":250,"seed":42}
    }

    feed = DataFeed(cfg['data']['sources']) if cfg['data']['sources'] else DataFeed({})
    broker = PaperBroker(**cfg.get('broker',{}))
    port = Portfolio(cfg.get('portfolio',{}).get('starting_cash',100000))
    eng = BacktestEngine(cfg, feed, broker, port)

    def v13_strategy(symbol: str, tf: str, bars_df):
        return strategy_from_df(symbol, tf, bars_df, balance=cfg["portfolio"]["starting_cash"],
                                config_path=cfg.get("strategy",{}).get("config","config/production.json"))

    syms = list(cfg.get('data',{}).get('sources',{}).keys())
    tfs = cfg.get('data',{}).get('timeframes',['1H'])
    res = eng.run(v13_strategy, syms, tfs, out_dir=args.out)
    print(json.dumps({"ok":True, "metrics":res['metrics'], "artifacts":res['artifacts']}, default=str))

if __name__ == '__main__':
    main()
