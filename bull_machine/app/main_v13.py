"""Bull Machine v1.3 - Main Application with MTF Sync"""

import sys
import argparse
import logging
import json
import pandas as pd
from typing import Dict, Optional

from bull_machine.core.types import (
    Bar, Series, Signal, RiskPlan,
    WyckoffResult, LiquidityResult,
    BiasCtx, RangeCtx, SyncReport
)
from bull_machine.config.loader import load_config
from bull_machine.state.store import load_state, save_state
from bull_machine.io.feeders import load_csv_to_series
from bull_machine.core.timeframes import tf_to_pandas_freq
from bull_machine.core.sync import decide_mtf_entry
from bull_machine.core.utils import extract_key_levels

# Import analyzers
from bull_machine.modules.wyckoff.advanced import AdvancedWyckoffAnalyzer
from bull_machine.modules.liquidity.advanced import AdvancedLiquidityAnalyzer
from bull_machine.modules.structure.advanced import AdvancedStructureAnalyzer
from bull_machine.modules.momentum.advanced import AdvancedMomentumAnalyzer
from bull_machine.modules.volume.advanced import AdvancedVolumeAnalyzer
from bull_machine.modules.context.advanced import AdvancedContextAnalyzer

from bull_machine.fusion.fuse import FusionEngineV1_3
from bull_machine.modules.risk.advanced import AdvancedRiskManager

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def resample_to_timeframes(df: pd.DataFrame, htf: str = "1D",
                          mtf: str = "4H", ltf: str = "1H") -> Dict[str, pd.DataFrame]:
    """Resample data to multiple timeframes"""

    df = df.copy()

    # Handle timestamp column
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})

    if "timestamp" in df.columns:
        try:
            # Try parsing as Unix timestamp (seconds)
            if df["timestamp"].dtype in ['int64', 'float64']:
                df.index = pd.to_datetime(df["timestamp"], unit='s', utc=True).tz_localize(None)
            else:
                # Try parsing as datetime string
                df.index = pd.to_datetime(df["timestamp"], utc=False, errors='coerce')

            # Drop any NaT values
            df = df[df.index.notna()]
            logging.info("Using timestamp column for resampling")
        except Exception as e:
            logging.warning(f"Could not parse timestamp column: {e}")
            # Fall back to synthetic
            df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="1h", tz=None)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # No timestamp, create synthetic
        df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="1h", tz=None)
        logging.info("Using synthetic datetime index for resampling")

    result = {}

    for key, tf_str in [("htf", htf), ("mtf", mtf), ("ltf", ltf)]:
        freq = tf_to_pandas_freq(tf_str)

        try:
            resampled = df.resample(freq).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }).dropna()

            result[key] = resampled
            logging.debug(f"Resampled to {tf_str}: {len(resampled)} bars")

        except Exception as e:
            logging.error(f"Failed to resample to {tf_str}: {e}")
            result[key] = df

    return result

def build_composite_range(df: pd.DataFrame, tf: str) -> RangeCtx:
    """Build range context from DataFrame"""
    if len(df) < 20:
        return RangeCtx(tf=tf, low=0, high=0, mid=0)

    # Simple range: last 20 bars high/low
    recent_bars = df.tail(20)
    low = recent_bars['low'].min()
    high = recent_bars['high'].max()
    mid = (low + high) / 2

    return RangeCtx(
        tf=tf,
        low=low,
        high=high,
        mid=mid,
        time_in_range=0,
        compression=0.5,
        tests_high=0,
        tests_low=0,
        last_test='none',
        breakout_potential=0.5
    )

def resolve_bias(df: pd.DataFrame, tf: str, two_bar: bool = True) -> BiasCtx:
    """Resolve bias from DataFrame"""
    if len(df) < 3:
        return BiasCtx(
            tf=tf, bias="neutral", confirmed=False,
            strength=0.0, bars_confirmed=0, ma_distance=0.0,
            trend_quality=0.0, ma_slope=0.0
        )

    # Simple bias: EMA20 vs EMA50
    ma20_window = min(20, len(df))
    ma50_window = min(50, len(df))

    ma20 = df["close"].rolling(ma20_window, min_periods=1).mean()
    ma50 = df["close"].rolling(ma50_window, min_periods=1).mean()

    current_close = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2] if len(df) >= 2 else current_close

    ma20_current = ma20.iloc[-1] if len(ma20) > 0 else current_close
    ma50_current = ma50.iloc[-1] if len(ma50) > 0 else ma20_current

    # Determine bias
    if ma20_current > ma50_current and current_close > ma20_current:
        bias = "long"
        strength = 0.7
    elif ma20_current < ma50_current and current_close < ma20_current:
        bias = "short"
        strength = 0.7
    else:
        bias = "neutral"
        strength = 0.3

    # Two-bar confirmation
    confirmed = False
    bars_confirmed = 0
    if two_bar and len(df) >= 2:
        if bias == "long" and prev_close > ma20.iloc[-2]:
            confirmed = True
            bars_confirmed = 2
        elif bias == "short" and prev_close < ma20.iloc[-2]:
            confirmed = True
            bars_confirmed = 2

    return BiasCtx(
        tf=tf,
        bias=bias,
        confirmed=confirmed,
        strength=strength,
        bars_confirmed=bars_confirmed,
        ma_distance=abs(current_close - ma20_current) / ma20_current,
        trend_quality=0.6,
        ma_slope=0.01 if bias == "long" else -0.01
    )

def compute_eq_magnet(htf_range: RangeCtx, current_price: float, config: dict) -> bool:
    """Check if price is in equilibrium zone (EQ magnet)"""
    if htf_range.height == 0:
        return False

    eq_threshold = config.get('eq_threshold', 0.02)
    eq_zone = htf_range.equilibrium_zone

    return eq_zone[0] <= current_price <= eq_zone[1]

def check_nested_confluence(htf_range: RangeCtx, ltf_levels: list) -> bool:
    """Check if LTF levels nest properly within HTF zones"""
    if not ltf_levels or htf_range.height == 0:
        return False

    # Simple check: at least one LTF level in HTF premium/discount zones
    premium_zone = htf_range.premium_zone
    discount_zone = htf_range.discount_zone

    for level in ltf_levels:
        price = level['price']
        if (discount_zone[0] <= price <= discount_zone[1] or
            premium_zone[0] <= price <= premium_zone[1]):
            return True

    return False

def run_bull_machine_v1_3(csv_file: str,
                         account_balance: float = 10000,
                         config_path: Optional[str] = None,
                         mtf_enabled: bool = True) -> Dict:
    """Bull Machine v1.3 Pipeline with MTF Sync"""

    logging.info("=" * 60)
    logging.info("Bull Machine v1.3 - MTF Sync Enabled" if mtf_enabled else "Bull Machine v1.3 - MTF Sync Disabled")
    logging.info("=" * 60)

    try:
        # Load configuration
        config = load_config(config_path or "bull_machine/config/production.json")

        # Simple validation - ensure required sections exist
        config.setdefault("fusion", {})
        config.setdefault("features", {})

        # Override MTF setting
        if not mtf_enabled:
            config.setdefault("mtf", {})["enabled"] = False
        else:
            config.setdefault("mtf", {})["enabled"] = True

        # Set MTF defaults
        mtf_config = config.setdefault("mtf", {})
        mtf_config.setdefault("htf", "1D")
        mtf_config.setdefault("mtf", "4H")
        mtf_config.setdefault("ltf", "1H")
        mtf_config.setdefault("two_bar_confirm", True)
        mtf_config.setdefault("eq_magnet_gate", True)
        mtf_config.setdefault("eq_threshold", 0.02)
        mtf_config.setdefault("desync_behavior", "raise")
        mtf_config.setdefault("desync_bump", 0.10)
        mtf_config.setdefault("eq_bump", 0.05)
        mtf_config.setdefault("nested_bump", 0.03)
        mtf_config.setdefault("alignment_discount", 0.05)

        logging.info(f"MTF Sync: {'ENABLED' if mtf_config.get('enabled', False) else 'DISABLED'}")

        # Load state
        state = load_state()

        # Load primary data
        series_ltf = load_csv_to_series(csv_file)
        logging.info(f"Loaded {len(series_ltf.bars)} bars from {csv_file}")

        # Initialize results
        result = {
            "action": "no_trade",
            "version": "1.3.0",
            "mtf_enabled": mtf_config.get("enabled", False)
        }

        # MTF SYNC PROCESSING
        sync_report = None

        if mtf_config.get("enabled", False):
            logging.info("\n" + "=" * 40)
            logging.info("Running Multi-Timeframe Analysis...")
            logging.info("=" * 40)

            # Convert to DataFrame with timestamps
            df_data = []
            for b in series_ltf.bars:
                df_data.append({
                    "timestamp": b.ts,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume
                })
            df = pd.DataFrame(df_data)

            # Resample with real timestamps
            data_dict = resample_to_timeframes(
                df,
                htf=mtf_config.get("htf", "1D"),
                mtf=mtf_config.get("mtf", "4H"),
                ltf=mtf_config.get("ltf", "1H")
            )

            # Build ranges
            htf_range = build_composite_range(data_dict["htf"], mtf_config.get("htf", "1D"))

            if htf_range.height > 0:
                logging.info(f"HTF Range: {htf_range.low:.2f} - {htf_range.high:.2f} (mid: {htf_range.mid:.2f})")

            # Resolve biases
            htf_bias = resolve_bias(
                data_dict["htf"],
                mtf_config.get("htf", "1D"),
                two_bar=mtf_config.get("two_bar_confirm", True)
            )
            mtf_bias = resolve_bias(
                data_dict["mtf"],
                mtf_config.get("mtf", "4H"),
                two_bar=mtf_config.get("two_bar_confirm", True)
            )

            logging.info(f"HTF Bias: {htf_bias.bias} (confirmed: {htf_bias.confirmed}, strength: {htf_bias.strength:.2f})")
            logging.info(f"MTF Bias: {mtf_bias.bias} (confirmed: {mtf_bias.confirmed}, strength: {mtf_bias.strength:.2f})")

        # LTF ANALYSIS (using v1.2.1 modules)
        logging.info("\nAnalyzing LTF structure...")

        # Initialize analyzers
        wyckoff_analyzer = AdvancedWyckoffAnalyzer(config)
        liquidity_analyzer = AdvancedLiquidityAnalyzer(config)
        structure_analyzer = AdvancedStructureAnalyzer(config)
        momentum_analyzer = AdvancedMomentumAnalyzer(config)
        volume_analyzer = AdvancedVolumeAnalyzer(config)
        context_analyzer = AdvancedContextAnalyzer(config)

        # Run analyses
        wyckoff_result = wyckoff_analyzer.analyze(series_ltf)
        liquidity_result = liquidity_analyzer.analyze(series_ltf, wyckoff_result)
        structure_result = structure_analyzer.analyze(series_ltf, wyckoff_result)
        momentum_result = momentum_analyzer.analyze(series_ltf, wyckoff_result)
        volume_result = volume_analyzer.analyze(series_ltf, wyckoff_result)
        context_result = context_analyzer.analyze(series_ltf, wyckoff_result)

        logging.info(f"Wyckoff: {wyckoff_result.regime}/{wyckoff_result.phase}, Bias: {wyckoff_result.bias}")
        logging.info(f"Liquidity Score: {liquidity_result.score:.2f}, Pressure: {liquidity_result.pressure}")

        # MTF SYNC DECISION
        if mtf_config.get("enabled", False) and htf_range:
            ltf_bias = wyckoff_result.bias

            # Check nested confluence
            ltf_levels = extract_key_levels(liquidity_result)
            nested_ok = check_nested_confluence(htf_range, ltf_levels)

            # Check EQ magnet
            current_price = series_ltf.bars[-1].close
            eq_magnet = compute_eq_magnet(htf_range, current_price, mtf_config)

            logging.info(f"Nested Confluence: {'✓' if nested_ok else '✗'} ({len(ltf_levels)} LTF levels)")
            logging.info(f"EQ Magnet: {'ACTIVE ⚠️' if eq_magnet else 'Inactive'}")

            # Run sync decision
            sync_report = decide_mtf_entry(
                htf_bias, mtf_bias, ltf_bias,
                nested_ok, eq_magnet,
                mtf_config
            )

            logging.info(f"\nMTF Decision: {sync_report.decision.upper()}")
            logging.info(f"Alignment Score: {sync_report.alignment_score:.1%}")
            if sync_report.threshold_bump != 0:
                logging.info(f"Threshold Adjustment: {sync_report.threshold_bump:+.2f}")

            result["mtf_sync"] = sync_report.to_dict()

        # FUSION (v1.3 with all 6 layers + MTF)
        logging.info("\nSignal Fusion...")

        fusion_engine = FusionEngineV1_3(config)
        signal = fusion_engine.fuse_with_mtf(
            {
                "wyckoff": wyckoff_result,
                "liquidity": liquidity_result,
                "structure": structure_result,
                "momentum": momentum_result,
                "volume": volume_result,
                "context": context_result
            },
            sync_report
        )

        if signal is None:
            reason = 'mtf_sync_veto' if sync_report and sync_report.decision == 'veto' else 'below_threshold'
            logging.info(f"No signal generated: {reason}")
            result['reason'] = reason
            return result

        logging.info(f"Signal: {signal.side.upper()}")
        logging.info(f"Confidence: {signal.confidence:.2%}")

        # RISK PLANNING
        risk_manager = AdvancedRiskManager(config)
        plan = risk_manager.plan_trade(series_ltf, signal, account_balance)

        logging.info(f"Entry: {plan.entry:.2f}")
        logging.info(f"Stop: {plan.stop:.2f}")
        logging.info(f"Size: {plan.size:.4f}")
        logging.info(f"Risk: ${plan.risk_amount:.2f}")

        # Update state
        state["prev_bias"] = wyckoff_result.bias
        state["last_signal_ts"] = signal.ts
        if sync_report:
            state["last_htf_bias"] = sync_report.htf.bias
            state["last_mtf_bias"] = sync_report.mtf.bias
        save_state(state)

        # Return result
        result.update({
            "action": "enter_trade",
            "signal": {
                "side": signal.side,
                "confidence": signal.confidence,
                "reasons": signal.reasons[:3]
            },
            "risk_plan": {
                "entry": plan.entry,
                "stop": plan.stop,
                "size": plan.size,
                "risk_amount": plan.risk_amount
            }
        })

        logging.info("\n✅ TRADE SIGNAL GENERATED")

        return result

    except Exception as e:
        logging.error(f"Error in v1.3 pipeline: {e}")
        import traceback
        traceback.print_exc()
        return {"action": "error", "message": str(e), "version": "1.3.0"}