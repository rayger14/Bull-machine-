"""
Enhanced v1.4 Strategy Adapter with Quality Gates

This adapter uses the enhanced fusion engine with:
- Quality gates for filtering weak layers
- Alignment multipliers for confluence
- Veto/penalty matrix for risk control
- Higher threshold with margin requirement
"""

import pandas as pd
import tempfile
import os
from typing import Dict, Any, Optional
from bull_machine.app.main_v13 import (
    resample_to_timeframes, build_composite_range, resolve_bias,
    compute_eq_magnet, check_nested_confluence, load_csv_to_series
)
from bull_machine.core.sync import decide_mtf_entry
from bull_machine.core.utils import extract_key_levels
from bull_machine.modules.wyckoff.advanced import AdvancedWyckoffAnalyzer
from bull_machine.modules.liquidity.advanced import AdvancedLiquidityAnalyzer
from bull_machine.modules.structure.advanced import AdvancedStructureAnalyzer
from bull_machine.modules.momentum.advanced import AdvancedMomentumAnalyzer
from bull_machine.modules.volume.advanced import AdvancedVolumeAnalyzer
from bull_machine.modules.context.advanced import AdvancedContextAnalyzer
from bull_machine.modules.fusion.enhanced import EnhancedFusionEngineV1_4
from bull_machine.modules.risk.advanced import AdvancedRiskManager
from bull_machine.config.loader import load_config
from bull_machine.core.signal_validation import validate_signal, standardize_signal, add_risk_management, log_signal_stats
import logging
import traceback
import json

# Global cache for MTF analysis
_mtf_cache = {}

DEBUG_RAISE = False  # set True during local debugging, False in CI/prod

def _emit_adapter_error(ctx: dict, err: Exception) -> None:
    """Structured logging for adapter exceptions with context."""
    err_info = {
        "stage": "ADAPTER",
        "error": str(err),
        "type": err.__class__.__name__,
        "stack": traceback.format_exc(limit=5),
        "ctx": ctx
    }
    logging.error("[ADAPTER_EXCEPTION] %s", json.dumps(err_info, default=str))

def save_temp_csv(df_window: pd.DataFrame, symbol: str) -> str:
    """Save DataFrame window as temporary CSV."""
    fd, temp_path = tempfile.mkstemp(suffix=f'_{symbol}.csv', prefix='enhanced_v14_')
    os.close(fd)

    df_save = df_window.copy()
    if 'timestamp' not in df_save.columns and hasattr(df_save.index, 'to_series'):
        df_save['timestamp'] = df_save.index.astype(int) // 10**9

    df_save.to_csv(temp_path, index=False)
    return temp_path

def get_mtf_cache_key(symbol: str, tf: str, current_index: int) -> str:
    """Generate cache key for MTF analysis."""
    # Cache at HTF boundaries only
    htf_stride = 24 if tf == "1H" else 6 if tf == "4H" else 1
    cache_index = (current_index // htf_stride) * htf_stride
    return f"{symbol}_{tf}_{cache_index}"

def strategy_from_df(symbol: str, tf: str, df_tf: pd.DataFrame,
                     current_index: int, balance: float = 10000, config_path: str = None) -> Dict[str, Any]:
    """
    Enhanced v1.4 strategy with quality gates and smart filtering.

    This version implements:
    - Quality gates for layer filtering
    - Alignment multipliers for confluence
    - Penalty system for edge cases
    - Higher threshold (0.45) with margin requirement
    """
    global _mtf_cache

    # Minimum data requirements
    if current_index < 50 or len(df_tf) < 50:
        return {
            "action": "no_trade",
            "reason": "insufficient_data",
            "version": "enhanced_v1.4"
        }

    try:
        # Load configuration - use passed config_path or default
        default_config = "bull_machine/configs/diagnostic_v14_config.json"
        config_file = config_path or default_config
        config = load_config(config_file)

        # MTF cache management
        cache_key = get_mtf_cache_key(symbol, tf, current_index)
        mtf_data = _mtf_cache.get(cache_key)

        if not mtf_data:
            # Generate MTF data
            df_window = df_tf.iloc[:current_index].copy()

            # Save temp CSV for MTF analysis
            csv_path = save_temp_csv(df_window, symbol)
            series_ltf = load_csv_to_series(csv_path)

            # Clean up temp file
            try:
                os.unlink(csv_path)
            except:
                pass

            # MTF Analysis
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

            # Resample to multiple timeframes
            data_dict = resample_to_timeframes(df, htf="1D", mtf="4H", ltf="1H")

            # Build ranges and biases
            htf_range = build_composite_range(data_dict["htf"], "1D")
            htf_bias = resolve_bias(data_dict["htf"], "1D", two_bar=True)
            mtf_bias = resolve_bias(data_dict["mtf"], "4H", two_bar=True)

            # Cache the MTF data
            mtf_data = {
                'series_ltf': series_ltf,
                'htf_range': htf_range,
                'htf_bias': htf_bias,
                'mtf_bias': mtf_bias,
                'data_dict': data_dict
            }
            _mtf_cache[cache_key] = mtf_data

        # Extract cached data
        series_ltf = mtf_data['series_ltf']
        htf_range = mtf_data['htf_range']
        htf_bias = mtf_data['htf_bias']
        mtf_bias = mtf_data['mtf_bias']

        # Update current bar in series
        current_bar = df_tf.iloc[current_index-1]
        if hasattr(series_ltf.bars[-1], 'close'):
            series_ltf.bars[-1].close = current_bar['close']
            series_ltf.bars[-1].high = max(series_ltf.bars[-1].high, current_bar['high'])
            series_ltf.bars[-1].low = min(series_ltf.bars[-1].low, current_bar['low'])
            series_ltf.bars[-1].volume = current_bar.get('volume', series_ltf.bars[-1].volume)

        # Initialize enhanced analyzers
        wyckoff_analyzer = AdvancedWyckoffAnalyzer(config)
        liquidity_analyzer = AdvancedLiquidityAnalyzer(config)
        structure_analyzer = AdvancedStructureAnalyzer(config)
        momentum_analyzer = AdvancedMomentumAnalyzer(config)
        volume_analyzer = AdvancedVolumeAnalyzer(config)
        context_analyzer = AdvancedContextAnalyzer(config)

        # Run enhanced analyses
        wyckoff_result = wyckoff_analyzer.analyze(series_ltf)
        liquidity_result = liquidity_analyzer.analyze(series_ltf, wyckoff_result.bias)
        structure_result = structure_analyzer.analyze(series_ltf)
        momentum_result = momentum_analyzer.analyze(series_ltf)
        volume_result = volume_analyzer.analyze(series_ltf)
        context_result = context_analyzer.analyze(series_ltf)

        # MTF Sync Decision
        ltf_bias = wyckoff_result.bias
        ltf_levels = extract_key_levels(liquidity_result)
        nested_ok = check_nested_confluence(htf_range, ltf_levels)
        current_price = series_ltf.bars[-1].close
        eq_magnet = compute_eq_magnet(htf_range, current_price, config.get('mtf', {}))

        sync_report = decide_mtf_entry(
            htf_bias, mtf_bias, ltf_bias,
            nested_ok, eq_magnet,
            config.get('mtf', {})
        )

        # Enhanced Fusion with Quality Floors for calibration
        fusion_config = config.get('fusion', {})
        print(f"🔥 FUSION_CONFIG: Using enhanced fusion with quality floors: {fusion_config.get('quality_floors', 'NONE')}")
        fusion_engine = EnhancedFusionEngineV1_4(fusion_config)
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
            reason = 'quality_gates_veto'
            if sync_report and sync_report.decision == 'veto':
                reason = 'mtf_sync_veto'
            logging.debug(f"[ADAPTER] No signal: {reason}")
            return None  # Return None instead of dict for no signal

        # Enhanced Risk Management
        risk_manager = AdvancedRiskManager(config)
        plan = risk_manager.plan_trade(series_ltf, signal, balance)

        # Get current bar for entry price
        current_bar = df_tf.iloc[current_index-1]
        entry_price = current_bar['close']

        # Standardize signal format for backtest engine
        standardized_signal = standardize_signal(signal, symbol, current_bar.name)
        if not standardized_signal:
            logging.error(f"[ADAPTER] Signal standardization failed")
            return None

        # Add risk management
        standardized_signal = add_risk_management(standardized_signal, plan)
        standardized_signal["entry_price"] = entry_price

        # Convert to engine format with metadata for degradation tracking
        engine_signal = {
            "action": standardized_signal["bias"],  # "long" or "short"
            "size": standardized_signal.get("size", 1000.0),
            "risk_plan": {
                "entry": entry_price,
                "stop": standardized_signal.get("stop"),
                "tps": standardized_signal.get("tps", []),
                "size": standardized_signal.get("size", 1000.0),
                "risk_amount": standardized_signal.get("risk_amount", 100.0)
            },
            "reasons": standardized_signal["reasons"],
            "confidence": standardized_signal["score"],
            "version": "enhanced_v1.4",
            "symbol": symbol,
            "metadata": {
                "entry_fusion_score": standardized_signal["score"],
                "subscores": {
                    "wyckoff": getattr(wyckoff_result, 'confidence', 0.0) if wyckoff_result else 0.0,
                    "liquidity": getattr(liquidity_result, 'score', 0.0) if liquidity_result else 0.0,
                    "structure": structure_result.get("score", 0.0) if isinstance(structure_result, dict) else getattr(structure_result, 'score', 0.0) if structure_result else 0.0,
                    "momentum": momentum_result.get("score", 0.0) if isinstance(momentum_result, dict) else getattr(momentum_result, 'score', 0.0) if momentum_result else 0.0,
                    "volume": volume_result.get("score", 0.0) if isinstance(volume_result, dict) else getattr(volume_result, 'score', 0.0) if volume_result else 0.0,
                    "context": context_result.get("score", 0.0) if isinstance(context_result, dict) else getattr(context_result, 'score', 0.0) if context_result else 0.0
                },
                "entry_timestamp": current_bar.name,
                "entry_price": entry_price
            }
        }

        # Validate final signal
        if validate_signal(standardized_signal):
            logging.info(f"[ADAPTER] Generated signal: {engine_signal['action']} @ {engine_signal['confidence']:.3f}")
            logging.info(f"ADAPTER_EMIT side={engine_signal['action']} conf={engine_signal['confidence']:.3f} sym={symbol}")
            return engine_signal
        else:
            logging.error(f"[ADAPTER] Signal validation failed")
            return None

    except Exception as e:
        # Structured error logging with context
        _emit_adapter_error({
            "symbol": symbol,
            "tf": tf,
            "current_index": current_index,
            "balance": balance,
            "bar_close": getattr(df_tf.iloc[-1] if len(df_tf) > 0 else None, 'close', 'unknown'),
            "function": "strategy_from_df"
        }, e)

        if DEBUG_RAISE:
            raise
        return None  # DO NOT emit side="error" signals anymore

# Backward compatibility
def optimized_strategy_from_df(*args, **kwargs):
    """Backward compatibility wrapper."""
    return strategy_from_df(*args, **kwargs)