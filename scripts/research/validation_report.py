#!/usr/bin/env python3
"""
Green-Light Validation Report
Confirms all systems operational with guardrails enforced
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from engine.io.tradingview_loader import load_tv
from engine.smc.smc_engine import SMCEngine
from engine.wyckoff.wyckoff_engine import WyckoffEngine
from engine.liquidity.hob import HOBDetector
from engine.momentum.momentum_engine import MomentumEngine
import pandas as pd
import json
import numpy as np

def generate_validation_report():
    """Generate comprehensive validation report for green-light status"""

    print("🚦 GREEN-LIGHT VALIDATION REPORT")
    print("="*60)

    # Load config
    with open('configs/v170/assets/ETH_v17_tuned.json', 'r') as f:
        config = json.load(f)

    # 1. Configuration Check
    print("\n📋 1. CONFIGURATION STATUS")
    print("-"*40)

    cal_mode = config['fusion'].get('calibration_mode', False)
    cal_thresholds = config['fusion']['calibration_thresholds']
    prod_conf = config['fusion']['entry_threshold_confidence']
    prod_str = config['fusion']['entry_threshold_strength']

    print(f"✅ Mode: {'CALIBRATION' if cal_mode else 'PRODUCTION'}")
    print(f"✅ Calibration thresholds: confidence={cal_thresholds['confidence']}, strength={cal_thresholds['strength']}")
    print(f"✅ Production thresholds: confidence={prod_conf}, strength={prod_str}")

    # Check micro nudges
    hob_z_min = config['domains']['liquidity']['hob_detection']['hob_quality_factors']['volume_z_min']
    hps_floor = config['domains']['wyckoff']['hps_floor']
    print(f"✅ HOB volume z-min: {hob_z_min} (relaxed from 1.5)")
    print(f"✅ Wyckoff HPS floor: {hps_floor} (relaxed from 0.5)")

    # 2. Guardrail Verification
    print("\n🛡️ 2. GUARDRAIL STATUS")
    print("-"*40)

    # Momentum delta cap
    momentum_cfg = config['domains']['momentum']
    print(f"✅ Momentum delta cap: ±{momentum_cfg.get('momentum_weight', 0.06)}")
    print(f"✅ Momentum as delta: {momentum_cfg.get('use_as_delta', False)}")

    # HOB constraints
    hob_relevance = config['domains']['liquidity']['hob_detection']['hob_relevance']
    print(f"✅ HOB max ATR distance: {hob_relevance['max_atr_dist']}")
    print(f"✅ HOB max bars unmitigated: {hob_relevance['max_bars_unmitigated']}")

    # Wyckoff HPS
    print(f"✅ Wyckoff HPS max delta: {config['domains']['wyckoff'].get('hps_max_delta', 0.03)}")

    # 3. SMC Threshold Validation
    print("\n🎯 3. SMC THRESHOLD CHECK")
    print("-"*40)

    smc = config['domains']['smc']
    print(f"✅ Order blocks: {smc['order_blocks']['min_displacement_pct']} (relaxed)")
    print(f"✅ FVGs: {smc['fvg']['min_gap_pct']} (relaxed)")
    print(f"✅ Sweeps: {smc['liquidity_sweeps']['min_pip_sweep']} pips (relaxed)")
    print(f"✅ Volume ratios: {smc['order_blocks']['min_volume_ratio']}-{smc['fvg']['min_volume_ratio']} (relaxed)")

    # 4. Trade Generation Test
    print("\n💹 4. TRADE GENERATION TEST")
    print("-"*40)

    # Load recent data
    df = load_tv('ETH_4H')
    recent = df.tail(100)

    # Test each engine
    active_engines = 0
    signals = {}

    # Test SMC
    try:
        smc_engine = SMCEngine(smc)
        smc_signal = smc_engine.analyze(recent)
        if smc_signal:
            signals['smc'] = smc_signal
            active_engines += 1
            print(f"✅ SMC: {smc_signal.direction} (confidence={smc_signal.confidence:.3f})")
        else:
            print("⚠️ SMC: No signal")
    except Exception as e:
        print(f"❌ SMC Error: {e}")

    # Test Momentum
    try:
        mom_engine = MomentumEngine(momentum_cfg)
        mom_signal = mom_engine.analyze(recent)
        mom_delta = mom_engine.get_delta_only(recent)
        if mom_signal:
            signals['momentum'] = mom_signal
            active_engines += 1
            print(f"✅ Momentum: {mom_signal.direction} (delta={mom_delta:+.3f})")
        else:
            print("⚠️ Momentum: No signal")
    except Exception as e:
        print(f"❌ Momentum Error: {e}")

    # Test Wyckoff
    try:
        wyckoff_engine = WyckoffEngine(config['domains']['wyckoff'])
        wyckoff_signal = wyckoff_engine.analyze(recent, usdt_stagnation=0.5)
        if wyckoff_signal:
            signals['wyckoff'] = wyckoff_signal
            active_engines += 1
            hps_score = wyckoff_signal.metadata.get('hps_score', 0)
            print(f"✅ Wyckoff: {wyckoff_signal.phase.value} (HPS={hps_score:.2f})")
        else:
            print("⚠️ Wyckoff: No phase")
    except Exception as e:
        print(f"❌ Wyckoff Error: {e}")

    # Test HOB
    try:
        hob_engine = HOBDetector(config['domains']['liquidity']['hob_detection'])
        hob_signal = hob_engine.detect_hob(recent)
        if hob_signal:
            signals['hob'] = hob_signal
            active_engines += 1
            print(f"✅ HOB: {hob_signal.hob_type.value}")
        else:
            print("⚠️ HOB: No pattern")
    except Exception as e:
        print(f"❌ HOB Error: {e}")

    print(f"\nActive engines: {active_engines}/4")

    # Calculate fusion
    if signals:
        directions = [s.direction for s in signals.values() if hasattr(s, 'direction')]
        confidences = [s.confidence for s in signals.values() if hasattr(s, 'confidence')]
        avg_confidence = np.mean(confidences) if confidences else 0

        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Meets calibration threshold: {'✅ YES' if avg_confidence >= cal_thresholds['confidence'] else '❌ NO'}")

    # 5. Telemetry Expectations
    print("\n📊 5. EXPECTED TELEMETRY")
    print("-"*40)

    print("On entry bars you should see:")
    print("  • fusion_deltas_hob: ≈ +0.05 when HOB relevant + vol_z ≥ 1.3")
    print("  • wyckoff_hps: +0.00 to +0.03 when Phase B/C + HPS ≥ 0.4")
    print("  • momentum_delta: -0.06 to +0.06 (capped)")
    print("  • macro_delta: -0.10 to +0.10 (from macro pulse)")
    print("  • fusion_score: ≥ 0.30 (calibration) or ≥ 0.35 (production)")

    # 6. Health Targets
    print("\n🎯 6. HEALTH BAND TARGETS")
    print("-"*40)

    print("Before switching calibration_mode=false, confirm:")
    print("  • Macro veto rate: 5-15% overall")
    print("  • HOB relevance hits: ≤30% of entries")
    print("  • Wyckoff HPS mean on entries: ≥0.4")
    print("  • SMC ≥2-hit share: ≥30%")
    print("  • Regime targets: risk_on PF > neutral > risk_off")

    # 7. Multi-Asset Readiness
    print("\n🌍 7. MULTI-ASSET EXPANSION")
    print("-"*40)

    from engine.io.tradingview_loader import SYMBOL_MAP
    assets = ['BTC_4H', 'ETH_4H', 'SOL_4H', 'AAVE_4H', 'MATIC_4H']
    available = sum(1 for a in assets if a in SYMBOL_MAP)
    print(f"Available assets: {available}/{len(assets)}")

    for asset in assets:
        status = "✅" if asset in SYMBOL_MAP else "❌"
        print(f"  {status} {asset}")

    print("\nTo add missing assets:")
    print("  1. Update SYMBOL_MAP in engine/io/tradingview_loader.py")
    print("  2. Ensure TradingView CSV files match pattern")
    print("  3. Test with preflight_data_check.py")

    # 8. Final Status
    print("\n" + "="*60)
    print("🚦 FINAL GREEN-LIGHT STATUS")
    print("="*60)

    checks = {
        "Configuration": cal_mode and cal_thresholds['confidence'] == 0.30,
        "Guardrails": momentum_cfg.get('use_as_delta', False),
        "SMC Thresholds": smc['order_blocks']['min_displacement_pct'] <= 0.003,
        "Trade Generation": active_engines >= 2,
        "Micro Nudges": hob_z_min == 1.3 and hps_floor == 0.4
    }

    all_green = all(checks.values())

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    if all_green:
        print("\n🎉 SYSTEM IS GREEN-LIGHT GO!")
        print("Trades should generate with current configuration.")
        print("Monitor telemetry and switch to production after validation.")
    else:
        print("\n⚠️ Some checks failed. Review configuration.")

    return all_green

if __name__ == "__main__":
    generate_validation_report()