import sys
import argparse
import logging
from ..config.loader import load_config
from ..io.feeders import load_csv_to_series
from ..modules.wyckoff.analyzer import analyze as analyze_wyckoff
from ..modules.liquidity.basic import analyze as analyze_liquidity
from ..signals.fusion import combine as combine_signals
from ..signals.gating import assign_ttl
from ..risk.planner import plan as plan_risk
from ..state.store import load_state, save_state

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def run_bull_machine_v1_1(csv_file: str, account_balance: float = 10000) -> dict:
    logging.info("Bull Machine v1.1 Starting...")
    logging.info(f"Processing: {csv_file}")
    try:
        config = load_config()
        logging.info(f"Config version: {config.get('version','unknown')}")
        logging.info(f"Dynamic TTL: {'Enabled' if config.get('features',{}).get('dynamic_ttl', False) else 'Disabled'}")
        state = load_state()
        series = load_csv_to_series(csv_file)
        result = {'action':'no_trade','version':'1.1'}
        if config['features'].get('wyckoff', True):
            logging.info("Running Wyckoff analysis...")
            wres = analyze_wyckoff(series, config, state)
            logging.info(f"   {wres.regime} regime, phase {wres.phase}, bias {wres.bias}")
            logging.info(f"   Confidence: phase={wres.phase_confidence:.2f}, trend={wres.trend_confidence:.2f}")
            result['wyckoff'] = wres
        else:
            logging.warning("Wyckoff analysis disabled")
            return result
        if config['features'].get('liquidity_basic', True):
            logging.info("Running Liquidity analysis...")
            lres = analyze_liquidity(series, wres.bias, config)
            logging.info(f"   Score: {lres.score:.2f}, Pressure: {lres.pressure}")
            logging.info(f"   FVGs: {len(lres.fvgs)}, OBs: {len(lres.order_blocks)}")
            result['liquidity'] = lres
        else:
            logging.warning("Liquidity analysis disabled")
            return result
        logging.info("Running Signal Fusion...")
        signal = combine_signals(wres, lres, config, state)
        if signal is None:
            logging.info("   No signal generated")
            result['reason'] = 'insufficient_confluence_or_range_suppressed'
            return result
        signal = assign_ttl(signal, series, config, wres)
        logging.info(f"   Signal: {signal.side} with confidence {signal.confidence:.2f}")
        logging.info(f"   TTL(bars): {signal.ttl_bars}")
        logging.info(f"   Reasons: {', '.join(signal.reasons)}")
        logging.info("Planning risk management...")
        plan = plan_risk(series, signal, config, account_balance)
        print(f"\n=== TRADE PLAN GENERATED ===")
        print(f"Direction: {signal.side.upper()}")
        print(f"Entry: {plan.entry:.2f}")
        print(f"Stop: {plan.stop:.2f}")
        print(f"Size: {plan.size:.4f}")
        print(f"Risk: ${abs(plan.entry - plan.stop) * plan.size:.2f}")
        if plan.tp_levels:
            print("Take Profits:")
            for tp in plan.tp_levels:
                print(f"  {tp['name']}: {tp['price']:.2f} ({tp['pct']}%) - {tp['action']}")
        state['prev_bias'] = wres.bias
        state['last_signal_ts'] = signal.ts
        save_state(state)
        return {'action':'enter_trade','signal':signal,'risk_plan':plan,'wyckoff':wres,'liquidity':lres,'version':'1.1'}
    except Exception as e:
        logging.error(f"Error: {e}")
        return {'action':'error','message':str(e)}

def main():
    setup_logging()
    p = argparse.ArgumentParser(description='Bull Machine v1.1')
    p.add_argument('--csv', required=True, help='CSV file path')
    p.add_argument('--balance', type=float, default=10000, help='Account balance')
    args = p.parse_args()
    result = run_bull_machine_v1_1(args.csv, args.balance)
    if result['action'] == 'enter_trade':
        print("\n✅ TRADE SIGNAL GENERATED")
    else:
        print(f"\n⚪ NO TRADE: {result.get('reason','unknown')}")

if __name__ == "__main__":
    main()
