"""Performance and Logging Configuration for Bull Machine v1.4

Implements tiered logging and performance modes based on user requirements.
"""

from typing import Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import logging

class PerformanceMode(Enum):
    DEBUG = "debug"
    FAST = "fast"
    PROD = "prod"

class LogMode(Enum):
    NONE = "none"
    TRADES = "trades"
    BOUNDARY = "boundary"
    SAMPLED = "sampled"
    FULL = "full"

@dataclass
class PerformanceConfig:
    mode: PerformanceMode = PerformanceMode.FAST
    htf_stride: int = 24  # recompute D1 every 24 × 1H bars
    mtf_stride: int = 4   # recompute 4H every 4 × 1H bars
    log_mode: LogMode = LogMode.TRADES
    sample_rate: int = 50
    threshold_band: float = 0.03
    include_reasons: bool = True
    include_subscores: bool = True
    eod_snapshots: bool = True
    retain_days: int = 30

@dataclass
class LogEntry:
    timestamp: str
    symbol: str
    tf: str
    score: float
    threshold: float
    decision: str
    reasons: list = field(default_factory=list)
    subscores: Dict[str, float] = field(default_factory=dict)
    mtf_flags: Dict[str, Any] = field(default_factory=dict)

class PerformanceLogger:
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.buffer = []
        self.buffer_size = 100

    def should_log(self, score: float, threshold: float, event_type: str = "evaluation") -> bool:
        """Determine if event should be logged based on mode"""
        if self.config.log_mode == LogMode.NONE:
            return False
        elif self.config.log_mode == LogMode.TRADES:
            return event_type in ["trade", "eod"]
        elif self.config.log_mode == LogMode.BOUNDARY:
            return (event_type in ["trade", "eod", "veto", "raise"] or
                   abs(score - threshold) < self.config.threshold_band)
        elif self.config.log_mode == LogMode.SAMPLED:
            return (event_type in ["trade", "eod", "veto", "raise"] or
                   self._should_sample())
        elif self.config.log_mode == LogMode.FULL:
            return True
        return False

    def _should_sample(self) -> bool:
        """Sample 1 out of N evaluations"""
        import random
        return random.randint(1, self.config.sample_rate) == 1

    def log_evaluation(self, entry: LogEntry, event_type: str = "evaluation"):
        """Log an evaluation if it meets criteria"""
        if self.should_log(entry.score, entry.threshold, event_type):
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                self.flush()

    def flush(self):
        """Flush buffer to storage"""
        # Implementation would write to CSV/database
        # For now, clear buffer
        self.buffer.clear()

class MTFCache:
    """Caches higher timeframe analysis with stride-based updates"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.htf_cache = {}
        self.mtf_cache = {}
        self.last_htf_update = {}
        self.last_mtf_update = {}

    def should_update_htf(self, symbol: str, current_bar: int) -> bool:
        """Check if HTF should be recomputed"""
        key = symbol
        last = self.last_htf_update.get(key, -1)
        return (current_bar - last) >= self.config.htf_stride

    def should_update_mtf(self, symbol: str, current_bar: int) -> bool:
        """Check if MTF should be recomputed"""
        key = symbol
        last = self.last_mtf_update.get(key, -1)
        return (current_bar - last) >= self.config.mtf_stride

    def update_htf(self, symbol: str, current_bar: int, context: Dict):
        """Update HTF cache"""
        self.htf_cache[symbol] = context
        self.last_htf_update[symbol] = current_bar

    def update_mtf(self, symbol: str, current_bar: int, context: Dict):
        """Update MTF cache"""
        self.mtf_cache[symbol] = context
        self.last_mtf_update[symbol] = current_bar

    def get_htf(self, symbol: str) -> Dict:
        """Get cached HTF context"""
        return self.htf_cache.get(symbol, {"bias": "neutral", "confirmed": False})

    def get_mtf(self, symbol: str) -> Dict:
        """Get cached MTF context"""
        return self.mtf_cache.get(symbol, {"bias": "neutral", "confirmed": False})

def create_performance_config(mode: str = "fast") -> PerformanceConfig:
    """Create performance config based on mode"""
    if mode == "debug":
        return PerformanceConfig(
            mode=PerformanceMode.DEBUG,
            htf_stride=4,
            mtf_stride=2,
            log_mode=LogMode.BOUNDARY,
            sample_rate=10
        )
    elif mode == "fast":
        return PerformanceConfig(
            mode=PerformanceMode.FAST,
            htf_stride=24,
            mtf_stride=4,
            log_mode=LogMode.TRADES,
            sample_rate=50
        )
    elif mode == "prod":
        return PerformanceConfig(
            mode=PerformanceMode.PROD,
            htf_stride=24,
            mtf_stride=4,
            log_mode=LogMode.BOUNDARY,
            sample_rate=100
        )
    else:
        return PerformanceConfig()