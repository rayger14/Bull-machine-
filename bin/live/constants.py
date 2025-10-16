"""
Hybrid Runner Constants

This module defines all magic numbers and configuration constants used in the
hybrid trading system. Centralizing these values makes them easier to tune
and documents their purpose.
"""

# ============================================================================
# MINIMUM DATA REQUIREMENTS
# ============================================================================

# Minimum bars required before generating signals
# These ensure sufficient data for indicator calculations
MIN_BARS_1H = 50  # Minimum 1H bars needed (supports 50-period indicators)
MIN_BARS_4H = 14  # Minimum 4H bars needed (supports 14-period RSI)
MIN_BARS_1D = 20  # Minimum 1D bars needed (supports 20-period SMA)

# Rationale:
# - 50 bars @ 1H = ~2 days of data for volume analysis
# - 14 bars @ 4H = ~2.3 days for RSI calculations
# - 20 bars @ 1D = ~20 days for trend detection


# ============================================================================
# GARBAGE COLLECTION
# ============================================================================

# Trigger garbage collection every N bars in backtest mode
# Prevents memory creep during long backtests
GC_INTERVAL_BARS = 500  # ~21 days @ 1H bars

# Rationale:
# - Backtests can run for years (30k+ bars)
# - GC every 500 bars keeps memory stable
# - Negligible performance impact (GC takes ~1-2ms)


# ============================================================================
# LOGGING & BUFFERING
# ============================================================================

# Flush buffered logs every N bars
# Balance between I/O efficiency and data safety
LOG_FLUSH_INTERVAL = 100  # ~4 days @ 1H bars

# Progress reporting interval
PROGRESS_REPORT_INTERVAL = 100  # Print progress every 100 bars

# Rationale:
# - Buffering reduces I/O overhead (10-50× fewer writes)
# - 100-bar buffer is safe (if crash, lose max 4 days of logs)
# - Frequent enough for monitoring long backtests


# ============================================================================
# ATR THROTTLE CALCULATIONS
# ============================================================================

# Minimum bars needed for ATR calculation
ATR_MIN_BARS = 100  # Need historical context for percentile calculation

# ATR period for volatility measurement
ATR_PERIOD = 14  # Standard ATR period

# ATR window for percentile calculation
ATR_PERCENTILE_WINDOW = 100  # Last 100 bars for context

# Rationale:
# - ATR needs history to establish "normal" volatility range
# - 100 bars @ 1H = ~4 days of volatility context
# - 14-period ATR is industry standard


# ============================================================================
# LOSS STREAK TRACKING
# ============================================================================

# Maximum trades to check for loss streak
MAX_TRADES_FOR_LOSS_STREAK = 10  # Last 10 trades

# Rationale:
# - Recent performance more relevant than distant history
# - 10 trades captures enough context without overreacting
# - Prevents looking back too far (months of trades)


# ============================================================================
# DOMAIN FUSION THRESHOLDS (from domain_fusion.py)
# ============================================================================

# Wyckoff confidence threshold
# Below this, treat as neutral signal
WYCKOFF_MIN_CONFIDENCE = 0.2  # Was hardcoded as 0.2

# Directional tie-breaker thresholds
# Used when domain votes are tied
FUSION_BULLISH_BIAS_THRESHOLD = 0.52  # Slight bullish bias (was 0.52)
FUSION_BEARISH_BIAS_THRESHOLD = 0.48  # Slight bearish bias (was 0.48)

# Volume surge thresholds for HOB detection
HOB_VOLUME_SURGE_THRESHOLD = 1.2  # 1.2× mean volume (was 1.2)
HOB_VOLUME_INSTITUTIONAL_THRESHOLD = 1.8  # 1.8× for institutional (was 1.8)

# Wick ratio threshold for absorption detection
HOB_WICK_RATIO_THRESHOLD = 0.3  # 30% wick ratio (was 0.3)

# MTF alignment requirements
MTF_ALIGNMENT_CONFIDENCE_THRESHOLD = 0.5  # 50% alignment required (was 0.5)
MTF_TREND_SMA_PERIOD = 20  # SMA period for trend detection (was 20)
MTF_TREND_DEVIATION = 0.01  # 1% deviation for trend classification (was 1.01/0.99)

# Rationale:
# - These values tuned through v1.8 optimization
# - Documented here for visibility and future tuning
# - Can be overridden by config if needed


# ============================================================================
# FILE PATHS (Configurable Output)
# ============================================================================

# Default output directory for results
DEFAULT_OUTPUT_DIR = "results"

# Specific log file names
FUSION_DEBUG_LOG = "fusion_debug.jsonl"
SIGNAL_BLOCKS_LOG = "signal_blocks.jsonl"
FUSION_VALIDATION_LOG = "fusion_validation.jsonl"
DECISION_LOG = "decision_log.jsonl"

# Rationale:
# - Centralized so easy to change (e.g., for Docker deployments)
# - Can be overridden via config for multi-instance setups
# - Keeps logs organized in one location
