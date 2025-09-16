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
from ..core.utils import detect_sweep_displacement

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

def run_bull_machine_v1_1(csv_file: str, account_balance: float = 10000, override_signals: dict = None) -> dict:
    logging.info("Bull Machine v1.1 Starting...")
    logging.info(f"Processing: {csv_file}")
    try:
        config = load_config()
        # Apply runtime overrides intelligently:
        # - if override contains 'signals', merge into config['signals']
        # - if override contains top-level keys (like 'liquidity'), merge into those keys
        # - otherwise, place simple keys into config['signals'] for backward compatibility
        if override_signals:
            for k, v in override_signals.items():
                # If the override is explicitly for signals, merge there
                if k == 'signals' and isinstance(v, dict):
                    config.setdefault('signals', {}).update(v)
                    continue
                # Known top-level sections that should accept dict merges
                top_level_sections = {'liquidity', 'wyckoff', 'risk', 'range', 'signals', 'features'}
                if isinstance(v, dict) and k in top_level_sections:
                    config.setdefault(k, {}).update(v)
                    continue
                # If the key already exists at top-level and both are dicts, merge
                if k in config and isinstance(config.get(k), dict) and isinstance(v, dict):
                    config[k].update(v)
                    continue
                # Otherwise treat it as a signals override for backward compatibility
                config.setdefault('signals', {})[k] = v
        logging.info(f"Config version: {config.get('version','unknown')}")
        logging.info(f"Dynamic TTL: {'Enabled' if config.get('features',{}).get('dynamic_ttl', False) else 'Disabled'}")
        state = load_state()
        series = load_csv_to_series(csv_file)
        # Compute sweep→displacement flag (for range allow-list)
        try:
            state['had_recent_sweep_displacement'] = detect_sweep_displacement(series)
        except Exception:
            state['had_recent_sweep_displacement'] = False
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
        # --- Debug logs: numeric values used by the fusion gate ---
        signals_cfg = config.get('signals', {})
        thr = signals_cfg.get('confidence_threshold', 0.70)
        weights = signals_cfg.get('weights', {'wyckoff': 0.60, 'liquidity': 0.40})
        wyckoff_conf = (wres.phase_confidence + wres.trend_confidence) / 2
        liq_conf = lres.score
        logging.info("Running Signal Fusion...")
        logging.info(f"   Fusion debug: wyckoff_conf={wyckoff_conf:.3f}, phase_conf={wres.phase_confidence:.3f}, trend_conf={wres.trend_confidence:.3f}")
        logging.info(f"   Fusion debug: liquidity_score={liq_conf:.3f}, liquidity_pressure={lres.pressure}")
        logging.info(f"   Fusion debug: fvgs={len(lres.fvgs)}, order_blocks={len(lres.order_blocks)}")
        logging.info(f"   Fusion debug: threshold={thr}, weights={weights}")
        if getattr(wres, 'range', None):
            rng = wres.range
            logging.info(f"   Range model: within_range={rng.get('within_range')}, penetration={rng.get('penetration'):.3f}, height={rng.get('height'):.2f}")

        signal, fuse_reason = combine_signals(wres, lres, config, state)
        if signal is None:
            # Fusion breakdown snapshot for diagnostics when requested
            if signals_cfg.get('_fusion_breakdown'):
                try:
                    w_w = signals_cfg.get('weights', {}).get('wyckoff', 0.60)
                    l_w = signals_cfg.get('weights', {}).get('liquidity', 0.40)
                    wy_conf = (wres.phase_confidence + wres.trend_confidence) / 2.0
                    combined = wy_conf * w_w + lres.score * l_w
                    logging.info(
                        f"[FUSION] wy={wy_conf:.3f}*{w_w:.2f} + liq={lres.score:.3f}*{l_w:.2f} = {combined:.3f} (threshold={signals_cfg.get('confidence_threshold', 0.72):.2f})"
                    )
                except Exception:
                    pass
            logging.info(f"   No signal generated: {fuse_reason}")
            result['reason'] = fuse_reason or 'no_signal'
            return result
        signal = assign_ttl(signal, series, config, wres)
        logging.info(f"   Signal: {signal.side} with confidence {signal.confidence:.2f}")
        logging.info(f"   TTL(bars): {signal.ttl_bars}")
        logging.info(f"   Reasons: {', '.join(signal.reasons)}")
        logging.info("Planning risk management...")
        plan = plan_risk(series, signal, config, account_balance)

        # Optionally print a compact fusion log for debugging
        if signals_cfg.get('_fusion_breakdown'):
            fusion_log = {
                "bar_idx": signal.ts,
                "symbol": getattr(series, 'symbol', 'UNKNOWN'),
                "tf": getattr(series, 'timeframe', 'UNKNOWN'),
                "wyckoff": {
                    "phase": wres.phase,
                    "phase_conf": wres.phase_confidence,
                    "trend_conf": wres.trend_confidence,
                    "regime": wres.regime
                },
                "liquidity": {
                    "score": lres.score,
                    "pressure": lres.pressure,
                    "fvgs": len(lres.fvgs),
                    "obs": len(lres.order_blocks)
                },
                "range": getattr(wres, 'range', None),
                "combined_conf": signal.confidence,
                "decision": "enter",
                "reasons": signal.reasons
            }
            print(f"[FUSION_LOG] {fusion_log}")

        print(f"\n=== TRADE PLAN GENERATED ===")
        print(f"Direction: {signal.side.upper()}")
        # entry price uses the last bar close; show the corresponding datetime (UTC)
        try:
            from datetime import datetime
            entry_ts = series.bars[-1].ts
            entry_dt = datetime.utcfromtimestamp(int(entry_ts)).isoformat() + 'Z'
        except Exception:
            entry_dt = 'unknown'
        print(f"Entry: {plan.entry:.2f}  (Entry time UTC: {entry_dt})")
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
    p.add_argument('--enter-threshold', type=float, default=None, help='Temporary override for signal entry confidence threshold')
    p.add_argument('--threshold', type=float, help='Override fusion confidence threshold, e.g. 0.72')
    p.add_argument('--weights', type=str, default=None, help='JSON string or path to JSON file to override fusion weights e.g. "{\"wyckoff\":0.6,\"liquidity\":0.4}"')
    p.add_argument('--liquidity-floor', type=float, help='Override liquidity context floor, e.g. 0.20')
    p.add_argument('--fusion-breakdown', action='store_true', help='Always print fusion math breakdown')
    args = p.parse_args()
    # allow runtime override of entry threshold
    override = {}
    if args.enter_threshold is not None:
        override['confidence_threshold'] = float(args.enter_threshold)
    if args.weights is not None:
        import json, os
        wtxt = args.weights
        parsed = None
        # if it's a file path, load it
        if os.path.exists(wtxt):
            try:
                with open(wtxt, 'r') as fh:
                    parsed = json.load(fh)
            except Exception as e:
                logging.error(f"Failed to read weights file: {e}")
        else:
            try:
                parsed = json.loads(wtxt)
            except Exception as e:
                logging.error(f"Failed to parse weights JSON: {e}")
        if isinstance(parsed, dict):
            override['weights'] = parsed
    # support shorthand --threshold
    if getattr(args, 'threshold', None) is not None:
        override['confidence_threshold'] = float(args.threshold)
    # liquidity floor override
    if getattr(args, 'liquidity_floor', None) is not None:
        override.setdefault('liquidity', {})['context_floor'] = float(args.liquidity_floor)
    # fusion breakdown flag (propagate via signals overrides)
    if getattr(args, 'fusion_breakdown', False):
        override.setdefault('signals', {})['_fusion_breakdown'] = True

    if override:
        result = run_bull_machine_v1_1(args.csv, args.balance, override_signals=override)
    else:
        result = run_bull_machine_v1_1(args.csv, args.balance)
    if result['action'] == 'enter_trade':
        print("\n✅ TRADE SIGNAL GENERATED")
    else:
        print(f"\n⚪ NO TRADE: {result.get('reason','unknown')}")

if __name__ == "__main__":
    main()
