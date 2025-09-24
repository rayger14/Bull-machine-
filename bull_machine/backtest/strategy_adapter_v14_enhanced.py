"""Enhanced Strategy Adapter v1.4 with Performance Optimizations

Implements tiered logging, HTF/MTF stride caching, and boundary detection
based on user requirements for production-ready performance.
"""

import tempfile
import os
import pandas as pd
from typing import Dict, Any, Optional
from bull_machine.app.main_v13 import run_bull_machine_v1_3
from bull_machine.core.performance import (
    PerformanceConfig, MTFCache, PerformanceLogger,
    LogEntry, LogMode, create_performance_config
)

class EnhancedStrategyAdapter:
    """Performance-optimized strategy adapter with intelligent caching"""

    def __init__(self, performance_mode: str = "fast"):
        self.perf_config = create_performance_config(performance_mode)
        self.mtf_cache = MTFCache(self.perf_config)
        self.logger = PerformanceLogger(self.perf_config)
        self.bar_count = 0

    def strategy_from_df(self, symbol: str, tf: str, df_window: pd.DataFrame,
                        balance: float = 10000, config_path: str = None) -> Dict[str, Any]:
        """Enhanced strategy with stride-based caching and intelligent logging"""

        self.bar_count += 1
        current_ts = df_window.index[-1] if len(df_window) > 0 else pd.Timestamp.now()

        # Check if we should update HTF/MTF contexts
        htf_updated = False
        mtf_updated = False

        if self.mtf_cache.should_update_htf(symbol, self.bar_count):
            htf_context = self._compute_htf_context(df_window, symbol)
            self.mtf_cache.update_htf(symbol, self.bar_count, htf_context)
            htf_updated = True

        if self.mtf_cache.should_update_mtf(symbol, self.bar_count):
            mtf_context = self._compute_mtf_context(df_window, symbol)
            self.mtf_cache.update_mtf(symbol, self.bar_count, mtf_context)
            mtf_updated = True

        # Get cached contexts
        htf_context = self.mtf_cache.get_htf(symbol)
        mtf_context = self.mtf_cache.get_mtf(symbol)

        # Run core v1.3 analysis (optimized with minimal window)
        csv_path = self._save_temp_csv(df_window, symbol)

        try:
            result = run_bull_machine_v1_3(
                csv_file=csv_path,
                account_balance=balance,
                config_path=config_path,
                mtf_enabled=True
            )

            # Extract signal information for logging
            signal = result.get("signal")
            score = signal.get("confidence", 0.0) if signal else 0.0
            threshold = result.get("config", {}).get("signals", {}).get("enter_threshold", 0.35)
            decision = "ALLOW" if signal else "VETO"

            # Determine event type for logging
            event_type = "evaluation"
            if signal:
                event_type = "trade"
            elif abs(score - threshold) < self.perf_config.threshold_band:
                event_type = "boundary"
            elif htf_updated or mtf_updated:
                event_type = "mtf_update"

            # Create log entry
            log_entry = LogEntry(
                timestamp=str(current_ts),
                symbol=symbol,
                tf=tf,
                score=score,
                threshold=threshold,
                decision=decision,
                reasons=signal.get("reasons", []) if signal else ["no_signal"],
                subscores={
                    "wyckoff": result.get("modules", {}).get("wyckoff", {}).get("confidence", 0.0),
                    "liquidity": result.get("modules", {}).get("liquidity", {}).get("score", 0.0)
                },
                mtf_flags={
                    "htf_bias": htf_context.get("bias", "neutral"),
                    "mtf_bias": mtf_context.get("bias", "neutral"),
                    "htf_updated": htf_updated,
                    "mtf_updated": mtf_updated
                }
            )

            # Log based on configured mode
            self.logger.log_evaluation(log_entry, event_type)

            return result

        finally:
            self._cleanup_temp_file(csv_path)

    def _compute_htf_context(self, df_window: pd.DataFrame, symbol: str) -> Dict:
        """Compute higher timeframe context (1D analysis)"""
        if len(df_window) < 24:  # Need at least 24 hours for daily context
            return {"bias": "neutral", "confirmed": False, "strength": 0.0}

        # Simple daily trend analysis
        recent_close = df_window['close'].iloc[-1]
        daily_high = df_window['high'].tail(24).max()
        daily_low = df_window['low'].tail(24).min()
        daily_range = daily_high - daily_low

        if daily_range == 0:
            return {"bias": "neutral", "confirmed": False, "strength": 0.0}

        position = (recent_close - daily_low) / daily_range

        if position > 0.7:
            return {"bias": "long", "confirmed": True, "strength": position}
        elif position < 0.3:
            return {"bias": "short", "confirmed": True, "strength": 1.0 - position}
        else:
            return {"bias": "neutral", "confirmed": False, "strength": 0.1}

    def _compute_mtf_context(self, df_window: pd.DataFrame, symbol: str) -> Dict:
        """Compute middle timeframe context (4H analysis)"""
        if len(df_window) < 4:  # Need at least 4 hours for 4H context
            return {"bias": "neutral", "confirmed": False, "strength": 0.0}

        # Simple 4H trend analysis
        recent_close = df_window['close'].iloc[-1]
        h4_high = df_window['high'].tail(4).max()
        h4_low = df_window['low'].tail(4).min()
        h4_range = h4_high - h4_low

        if h4_range == 0:
            return {"bias": "neutral", "confirmed": False, "strength": 0.0}

        position = (recent_close - h4_low) / h4_range

        if position > 0.6:
            return {"bias": "long", "confirmed": True, "strength": position}
        elif position < 0.4:
            return {"bias": "short", "confirmed": True, "strength": 1.0 - position}
        else:
            return {"bias": "neutral", "confirmed": False, "strength": 0.1}

    def _save_temp_csv(self, df: pd.DataFrame, symbol: str) -> str:
        """Save DataFrame to temporary CSV (optimized for minimal data)"""
        # Only save last 50 bars for performance
        df_minimal = df.tail(50) if len(df) > 50 else df

        with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{symbol}.csv', delete=False) as f:
            df_minimal.to_csv(f.name)
            return f.name

    def _cleanup_temp_file(self, csv_path: str):
        """Clean up temporary CSV file"""
        try:
            if os.path.exists(csv_path):
                os.unlink(csv_path)
        except:
            pass  # Ignore cleanup errors

    def flush_logs(self):
        """Flush any pending logs"""
        self.logger.flush()

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "total_bars_processed": self.bar_count,
            "htf_cache_size": len(self.mtf_cache.htf_cache),
            "mtf_cache_size": len(self.mtf_cache.mtf_cache),
            "log_buffer_size": len(self.logger.buffer),
            "performance_mode": self.perf_config.mode.value,
            "log_mode": self.perf_config.log_mode.value
        }

# Factory function for easy integration
def create_enhanced_adapter(performance_mode: str = "fast") -> EnhancedStrategyAdapter:
    """Create enhanced strategy adapter with specified performance mode

    Args:
        performance_mode: "debug", "fast", or "prod"
    """
    return EnhancedStrategyAdapter(performance_mode)

# Backward compatibility
def optimized_strategy_from_df(symbol: str, tf: str, df_tf: pd.DataFrame, current_index: int,
                             balance: float = 10000, config_path: str = None) -> Dict[str, Any]:
    """Backward compatible optimized strategy function"""
    adapter = create_enhanced_adapter("fast")

    # Create window from current index
    window_size = min(100, current_index)
    start_idx = max(0, current_index - window_size)
    df_window = df_tf.iloc[start_idx:current_index]

    return adapter.strategy_from_df(symbol, tf, df_window, balance, config_path)