"""
Data-Adaptive Macro Pulse Calibration

Automatically calibrates macro thresholds based on real market data characteristics
instead of using fixed absolute values. This ensures macro pulse triggers appropriately
across different market conditions and time periods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def percentile_threshold(series: pd.Series, p: float) -> float:
    """Calculate percentile threshold from real data"""
    s = series.dropna().astype(float)
    return float(np.percentile(s, p)) if len(s) > 0 else np.nan

def rolling_volatility(series: pd.Series, window: int = 20) -> float:
    """Calculate rolling volatility (standard deviation of returns)"""
    returns = series.pct_change().dropna()
    return float(returns.rolling(window).std().mean()) if len(returns) > window else 0.01

def calibrate_macro_thresholds(series_map: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Calibrate macro thresholds based on real market data characteristics.

    Args:
        series_map: Dictionary of macro series DataFrames

    Returns:
        Dictionary of calibrated thresholds for macro pulse engine
    """
    logger.info("🔧 Calibrating macro thresholds from real market data...")

    cfg = {}

    # VIX Spike Detection (85th percentile = spike)
    vix = series_map.get("VIX_1D")
    if vix is not None and not vix.empty:
        vix_85th = percentile_threshold(vix["close"], 85)
        cfg["vix_spike"] = max(20.0, vix_85th)  # Minimum 20, but use 85th percentile
        logger.info(f"  VIX spike threshold: {cfg['vix_spike']:.2f} (85th percentile)")
    else:
        cfg["vix_spike"] = 24.0  # Fallback
        logger.warning("  VIX data not available, using fallback threshold: 24.0")

    # MOVE Spike Detection
    move = series_map.get("MOVE_1D")
    if move is not None and not move.empty:
        move_85th = percentile_threshold(move["close"], 85)
        cfg["move_spike"] = max(100.0, move_85th)
        logger.info(f"  MOVE spike threshold: {cfg['move_spike']:.2f} (85th percentile)")
    else:
        cfg["move_spike"] = 130.0  # Fallback
        logger.warning("  MOVE data not available, using fallback threshold: 130.0")

    # DXY Breakout Detection (95th percentile of 20-day rolling max)
    dxy = series_map.get("DXY_1D")
    if dxy is not None and not dxy.empty:
        cfg["dxy_breakout_len"] = 20
        cfg["dxy_breakout_quantile"] = 0.95

        # Calculate typical breakout threshold
        rolling_max = dxy["close"].rolling(20).max()
        breakout_levels = dxy["close"] / rolling_max
        typical_breakout = percentile_threshold(breakout_levels.dropna(), 95)
        cfg["dxy_breakout_threshold"] = min(0.99, typical_breakout)

        logger.info(f"  DXY breakout threshold: {cfg['dxy_breakout_threshold']:.4f} (95th percentile ratio)")
    else:
        cfg["dxy_breakout_len"] = 20
        cfg["dxy_breakout_quantile"] = 0.95
        cfg["dxy_breakout_threshold"] = 0.98
        logger.warning("  DXY data not available, using fallback breakout threshold: 0.98")

    # Yield Spike Detection (data-adaptive z-score)
    us2y = series_map.get("US2Y_1D")
    us10y = series_map.get("US10Y_1D")

    yield_volatilities = []
    if us2y is not None and not us2y.empty:
        us2_vol = rolling_volatility(us2y["close"])
        yield_volatilities.append(us2_vol)
        logger.info(f"  US2Y volatility: {us2_vol:.4f}")

    if us10y is not None and not us10y.empty:
        us10_vol = rolling_volatility(us10y["close"])
        yield_volatilities.append(us10_vol)
        logger.info(f"  US10Y volatility: {us10_vol:.4f}")

    if yield_volatilities:
        avg_yield_vol = np.mean(yield_volatilities)
        # Scale z-score threshold based on realized volatility
        # Higher vol = lower z-score needed for "spike"
        target_z_score = 2.0
        normalized_threshold = target_z_score * max(0.005, avg_yield_vol * 100)  # Convert to percentage points
        cfg["yields_spike_std"] = max(1.5, normalized_threshold)
        logger.info(f"  Yields spike std threshold: {cfg['yields_spike_std']:.2f} (adaptive z-score)")
    else:
        cfg["yields_spike_std"] = 2.0
        logger.warning("  Yield data not available, using fallback z-score: 2.0")

    # Oil+DXY Stagflation Detection (adaptive correlation)
    oil = series_map.get("WTI_1D")
    if oil is not None and dxy is not None and not oil.empty and not dxy.empty:
        # Calculate rolling correlation
        oil_returns = oil["close"].pct_change()
        dxy_returns = dxy["close"].pct_change()

        # Align series by index
        common_idx = oil_returns.index.intersection(dxy_returns.index)
        if len(common_idx) > 30:
            oil_aligned = oil_returns.loc[common_idx]
            dxy_aligned = dxy_returns.loc[common_idx]

            # Rolling 20-day correlation
            rolling_corr = oil_aligned.rolling(20).corr(dxy_aligned)

            # 85th percentile of positive correlations (stagflation regime)
            positive_corrs = rolling_corr[rolling_corr > 0]
            if len(positive_corrs) > 10:
                stagflation_threshold = percentile_threshold(positive_corrs, 85)
                cfg["oil_dxy_stagflation_veto"] = min(0.9, max(0.5, stagflation_threshold))
                logger.info(f"  Oil-DXY stagflation threshold: {cfg['oil_dxy_stagflation_veto']:.3f} (adaptive correlation)")
            else:
                cfg["oil_dxy_stagflation_veto"] = 0.7
                logger.info("  Oil-DXY insufficient positive correlations, using fallback: 0.7")
        else:
            cfg["oil_dxy_stagflation_veto"] = 0.7
            logger.warning("  Oil-DXY insufficient data overlap, using fallback: 0.7")
    else:
        cfg["oil_dxy_stagflation_veto"] = 0.85
        logger.warning("  Oil or DXY data not available, using fallback stagflation threshold: 0.85")

    # USDJPY Crisis Level (95th percentile)
    usdjpy = series_map.get("USDJPY_1D")
    if usdjpy is not None and not usdjpy.empty:
        usdjpy_95th = percentile_threshold(usdjpy["close"], 95)
        cfg["usd_jpy_break_level"] = max(140.0, usdjpy_95th)
        logger.info(f"  USDJPY crisis level: {cfg['usd_jpy_break_level']:.2f} (95th percentile)")
    else:
        cfg["usd_jpy_break_level"] = 145.0
        logger.warning("  USDJPY data not available, using fallback: 145.0")

    # HYG Credit Stress (15th percentile = stress)
    hyg = series_map.get("HYG_1D")
    if hyg is not None and not hyg.empty:
        hyg_15th = percentile_threshold(hyg["close"], 15)
        cfg["hyg_stress_level"] = hyg_15th
        cfg["hyg_break_len"] = 15
        logger.info(f"  HYG stress level: {cfg['hyg_stress_level']:.2f} (15th percentile)")
    else:
        cfg["hyg_break_len"] = 15
        cfg["hyg_stress_level"] = 75.0
        logger.warning("  HYG data not available, using fallback stress level: 75.0")

    # USDT Dominance Analysis
    usdt_d = series_map.get("USDTD_4H") or series_map.get("USDTD_1D")
    if usdt_d is not None and not usdt_d.empty:
        usdt_vol = rolling_volatility(usdt_d["close"])
        # Adaptive stagnation detection based on realized volatility
        cfg["usdt_stagnation_hours"] = 36
        cfg["usdt_range_pct"] = max(0.001, usdt_vol * 2)  # 2x rolling volatility
        logger.info(f"  USDT stagnation range: {cfg['usdt_range_pct']:.4f} (adaptive volatility)")
    else:
        cfg["usdt_stagnation_hours"] = 36
        cfg["usdt_range_pct"] = 0.002
        logger.warning("  USDT data not available, using fallback range: 0.002")

    # General adaptive settings
    cfg["time_shift_days"] = [120, 200, 240]  # DXY lag analysis windows
    cfg["wolfe_min_points"] = 4
    cfg["ethbtc_ma_length"] = 50

    # Calculate expected fire rates for validation
    fire_rate_estimates = {}

    if dxy is not None and not dxy.empty:
        # Estimate DXY breakout frequency
        rolling_max = dxy["close"].rolling(cfg["dxy_breakout_len"]).max()
        breakouts = dxy["close"] >= rolling_max * cfg["dxy_breakout_threshold"]
        fire_rate_estimates["dxy_breakout_pct"] = (breakouts.sum() / len(breakouts)) * 100

    if vix is not None and not vix.empty:
        vix_spikes = vix["close"] >= cfg["vix_spike"]
        fire_rate_estimates["vix_spike_pct"] = (vix_spikes.sum() / len(vix_spikes)) * 100

    cfg["fire_rate_estimates"] = fire_rate_estimates

    logger.info("✅ Macro threshold calibration complete")

    if fire_rate_estimates:
        logger.info("📊 Expected fire rates:")
        for metric, rate in fire_rate_estimates.items():
            logger.info(f"   {metric}: {rate:.1f}%")

    return cfg

def validate_fire_rates(fire_rates: Dict[str, float]) -> bool:
    """
    Validate that fire rates are in reasonable ranges.

    Target ranges:
    - DXY breakouts: 3-12% of bars
    - VIX spikes: 10-20% of bars
    """
    valid = True

    dxy_rate = fire_rates.get("dxy_breakout_pct", 0)
    if not (3 <= dxy_rate <= 15):
        logger.warning(f"DXY breakout rate {dxy_rate:.1f}% outside target range (3-15%)")
        valid = False

    vix_rate = fire_rates.get("vix_spike_pct", 0)
    if not (8 <= vix_rate <= 25):
        logger.warning(f"VIX spike rate {vix_rate:.1f}% outside target range (8-25%)")
        valid = False

    return valid