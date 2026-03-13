"""
Temporal Confluence Engine - Unified Temporal Analysis Interface

Combines all temporal analysis subsystems into a single confluence score
for each bar. Acts as the top-level interface that aggregates signals from:

    1. GannAnalyzer (gann.py) - Square of Nine price levels, time projections
    2. CycleDetector (cycles.py) - Fibonacci cycle detection via correlation
    3. TemporalEngine / TPI (tpi.py) - Time-Price Integration, cycle completion
    4. gann_cycles.py - ACF cycles, Square-9 proximity, Gann angles, LPPLS veto
    5. TemporalFusionEngine (temporal_fusion.py) - Fib/Gann/vol/emotional scoring

Design Principles:
    - Single entry point for all temporal features
    - Works both for batch feature-store computation and real-time bar scoring
    - No external dependencies beyond pandas / numpy
    - Deterministic: same inputs produce same outputs
    - Bounded outputs: all scores clipped to [0, 1]
    - Graceful degradation: missing data returns neutral scores, never errors

Output Features:
    - temporal_confluence_score : Master 0-1 score combining all temporal signals
    - gann_time_cluster        : Binary flag, 1 when near a Gann time projection
    - fib_time_cluster         : Binary flag, 1 when near a Fibonacci time zone
    - tpi_signal               : TPI signal strength [0-1]
    - cycle_phase              : Dominant cycle phase [0-1], 0=trough, 1=peak
    - temporal_reversal_zone   : Binary flag, 1 when multiple signals suggest reversal
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .gann import GannAnalyzer, GannLevel, GannTimeProject
from .cycles import CycleDetector, CycleSignal
from .tpi import TemporalEngine, TPISignal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default Fibonacci bar-count levels used to detect time clusters
_DEFAULT_FIB_LEVELS: List[int] = [13, 21, 34, 55, 89, 144, 233]

# Default Gann vibration bar-counts
_DEFAULT_GANN_VIBRATIONS: List[int] = [9, 21, 36, 45, 72, 90, 144]

# Default tolerance (bars) when matching against Fib / Gann targets
_DEFAULT_TOLERANCE_BARS: int = 3

# Component weight defaults for the master confluence score
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "gann": 0.20,
    "cycles": 0.25,
    "tpi": 0.25,
    "volatility": 0.15,
    "momentum_phase": 0.15,
}


# ---------------------------------------------------------------------------
# Helper dataclass for per-bar results
# ---------------------------------------------------------------------------

@dataclass
class TemporalBarResult:
    """Container for temporal analysis results on a single bar."""
    temporal_confluence_score: float
    gann_time_cluster: int
    fib_time_cluster: int
    tpi_signal: float
    cycle_phase: float
    temporal_reversal_zone: int
    component_scores: Dict[str, float]


# ---------------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------------

class TemporalConfluenceEngine:
    """
    Unified Temporal Confluence Engine.

    Aggregates signals from GannAnalyzer, CycleDetector, TemporalEngine (TPI),
    and internal volatility / momentum-phase heuristics to produce a single
    temporal_confluence_score per bar.

    Usage - batch (feature store)::

        engine = TemporalConfluenceEngine(config)
        df = engine.compute_temporal_features(df)
        # df now has temporal_confluence_score, gann_time_cluster, etc.

    Usage - real-time (single bar)::

        score = engine.get_confluence_score(row_dict)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the temporal confluence engine.

        Args:
            config: Optional configuration dictionary.  Supported keys:
                - fib_levels (List[int]): Fibonacci bar counts for cluster detection
                - gann_vibrations (List[int]): Gann cycle bar counts
                - tolerance_bars (int): How many bars of slack when matching targets
                - component_weights (Dict[str, float]): Weights for gann / cycles /
                  tpi / volatility / momentum_phase.  Must sum to ~1.0.
                - max_projection_days (int): Upper bound for time projections
                - min_confidence (float): Minimum confidence for TPI signals
                - vol_compression_threshold (float): ATR ratio below which compression
                - vol_expansion_threshold (float): ATR ratio above which expansion
                - reversal_zone_min_signals (int): Min signals to flag reversal zone
        """
        if config is None:
            config = {}
        self.config = config

        # Fibonacci & Gann parameters
        self.fib_levels: List[int] = config.get("fib_levels", _DEFAULT_FIB_LEVELS)
        self.gann_vibrations: List[int] = config.get("gann_vibrations", _DEFAULT_GANN_VIBRATIONS)
        self.tolerance_bars: int = config.get("tolerance_bars", _DEFAULT_TOLERANCE_BARS)

        # Component weights
        weights = config.get("component_weights", _DEFAULT_WEIGHTS)
        total = sum(weights.values())
        # Normalize so they sum to 1.0
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = _DEFAULT_WEIGHTS.copy()

        # Volatility thresholds
        self.vol_compression_threshold: float = config.get("vol_compression_threshold", 0.75)
        self.vol_expansion_threshold: float = config.get("vol_expansion_threshold", 1.25)

        # Reversal zone minimum signal count
        self.reversal_zone_min_signals: int = config.get("reversal_zone_min_signals", 3)

        # Sub-engine configs
        gann_config = config.get("gann", {"max_projection_days": config.get("max_projection_days", 30)})
        cycle_config = config.get("cycles", {})
        tpi_config = config.get("tpi", {"temporal": {
            "max_projection_days": config.get("max_projection_days", 30),
            "min_confidence": config.get("min_confidence", 0.5),
        }})

        # Instantiate sub-engines
        self.gann_analyzer = GannAnalyzer(gann_config)
        self.cycle_detector = CycleDetector(cycle_config)
        self.tpi_engine = TemporalEngine(tpi_config)

        # Internal cache for batch processing (avoids re-running sub-engines)
        self._gann_levels_cache: Optional[List[GannLevel]] = None
        self._gann_projections_cache: Optional[List[GannTimeProject]] = None
        self._cycle_signals_cache: Optional[List[CycleSignal]] = None
        self._tpi_signals_cache: Optional[List[TPISignal]] = None

        logger.info(
            f"[TemporalConfluence] Initialized | weights={self.weights} "
            f"| fib_levels={self.fib_levels} | gann_vibs={self.gann_vibrations}"
        )

    # ------------------------------------------------------------------
    # Public API - Batch Processing
    # ------------------------------------------------------------------

    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute temporal features for every bar in the dataframe.

        This is the primary batch entry point, intended for building or
        augmenting a feature store.  It adds the following columns:

            - temporal_confluence_score (float 0-1)
            - gann_time_cluster (int 0/1)
            - fib_time_cluster (int 0/1)
            - tpi_signal (float 0-1)
            - cycle_phase (float 0-1)
            - temporal_reversal_zone (int 0/1)

        Args:
            df: OHLCV DataFrame with a DatetimeIndex.  Must contain at least
                columns: open, high, low, close, volume.

        Returns:
            The same DataFrame with temporal feature columns appended.
        """
        if df.empty:
            logger.warning("[TemporalConfluence] Empty dataframe, returning as-is")
            return df

        n = len(df)
        logger.info(f"[TemporalConfluence] Computing temporal features for {n} bars")

        # Pre-compute sub-engine outputs over the full dataframe
        self._run_sub_engines(df)

        # Prepare output arrays
        confluence_scores = np.full(n, 0.5)
        gann_cluster_flags = np.zeros(n, dtype=int)
        fib_cluster_flags = np.zeros(n, dtype=int)
        tpi_signals = np.zeros(n, dtype=float)
        cycle_phases = np.full(n, 0.5)
        reversal_zone_flags = np.zeros(n, dtype=int)

        # Build per-bar features using vectorized helpers where possible,
        # falling back to row-level for components that need it.

        # --- Gann time cluster (vectorized) ---
        gann_cluster_flags = self._vectorized_gann_cluster(df)

        # --- Fibonacci time cluster (vectorized) ---
        fib_cluster_flags = self._vectorized_fib_cluster(df)

        # --- TPI signal strength (vectorized) ---
        tpi_signals = self._vectorized_tpi_signal(df)

        # --- Cycle phase (vectorized) ---
        cycle_phases = self._vectorized_cycle_phase(df)

        # --- Volatility compression/expansion score (vectorized) ---
        vol_scores = self._vectorized_volatility_score(df)

        # --- Momentum phase score (vectorized) ---
        momentum_scores = self._vectorized_momentum_phase(df)

        # --- Gann price-level proximity score (vectorized) ---
        gann_scores = self._vectorized_gann_score(df)

        # --- Cycle detection strength (vectorized) ---
        cycle_scores = self._vectorized_cycle_score(df)

        # --- Master confluence score (weighted combination) ---
        confluence_scores = (
            self.weights.get("gann", 0.20) * gann_scores
            + self.weights.get("cycles", 0.25) * cycle_scores
            + self.weights.get("tpi", 0.25) * tpi_signals
            + self.weights.get("volatility", 0.15) * vol_scores
            + self.weights.get("momentum_phase", 0.15) * momentum_scores
        )
        confluence_scores = np.clip(confluence_scores, 0.0, 1.0)

        # --- Reversal zone detection ---
        reversal_zone_flags = self._detect_reversal_zones(
            gann_cluster_flags, fib_cluster_flags, tpi_signals,
            cycle_phases, vol_scores
        )

        # Assign to dataframe
        df["temporal_confluence_score"] = confluence_scores
        df["gann_time_cluster"] = gann_cluster_flags
        df["fib_time_cluster"] = fib_cluster_flags
        df["tpi_signal"] = tpi_signals
        df["cycle_phase"] = cycle_phases
        df["temporal_reversal_zone"] = reversal_zone_flags

        logger.info(
            f"[TemporalConfluence] Features computed | "
            f"mean_confluence={np.nanmean(confluence_scores):.3f} | "
            f"gann_clusters={int(np.sum(gann_cluster_flags))} | "
            f"fib_clusters={int(np.sum(fib_cluster_flags))} | "
            f"reversal_zones={int(np.sum(reversal_zone_flags))}"
        )

        # Clear cache
        self._clear_cache()

        return df

    # ------------------------------------------------------------------
    # Public API - Single Bar Scoring
    # ------------------------------------------------------------------

    def get_confluence_score(self, row: Dict[str, Any]) -> float:
        """
        Get the temporal confluence score for a single bar (real-time use).

        Expects the row dict to contain at minimum: close, high, low, volume.
        Additional optional keys improve accuracy:
            - atr_14 or atr: Average True Range
            - rsi or rsi_14: Relative Strength Index
            - bb_width: Bollinger Band width
            - bars_since_sc, bars_since_ar, etc.: Wyckoff event distances

        Args:
            row: Dictionary of feature values for the current bar.

        Returns:
            Temporal confluence score in [0.0, 1.0].
        """
        try:
            # Component 1: Gann proximity score
            gann_score = self._single_bar_gann_score(row)

            # Component 2: Cycle score from Fibonacci alignment
            cycle_score = self._single_bar_cycle_score(row)

            # Component 3: TPI / time-price intensity
            tpi_score = self._single_bar_tpi_score(row)

            # Component 4: Volatility phase
            vol_score = self._single_bar_volatility_score(row)

            # Component 5: Momentum phase
            momentum_score = self._single_bar_momentum_score(row)

            # Weighted combination
            confluence = (
                self.weights.get("gann", 0.20) * gann_score
                + self.weights.get("cycles", 0.25) * cycle_score
                + self.weights.get("tpi", 0.25) * tpi_score
                + self.weights.get("volatility", 0.15) * vol_score
                + self.weights.get("momentum_phase", 0.15) * momentum_score
            )

            return float(np.clip(confluence, 0.0, 1.0))

        except Exception as e:
            logger.error(f"[TemporalConfluence] Error in get_confluence_score: {e}")
            return 0.5  # Neutral on error

    # ------------------------------------------------------------------
    # Sub-engine orchestration (batch)
    # ------------------------------------------------------------------

    def _run_sub_engines(self, df: pd.DataFrame) -> None:
        """Run all sub-engines once and cache their outputs."""
        try:
            self._gann_levels_cache = self.gann_analyzer.analyze_gann_levels(df)
        except Exception as e:
            logger.warning(f"[TemporalConfluence] Gann levels failed: {e}")
            self._gann_levels_cache = []

        try:
            self._gann_projections_cache = self.gann_analyzer.project_time_cycles(df)
        except Exception as e:
            logger.warning(f"[TemporalConfluence] Gann projections failed: {e}")
            self._gann_projections_cache = []

        try:
            self._cycle_signals_cache = self.cycle_detector.detect_cycles(df)
        except Exception as e:
            logger.warning(f"[TemporalConfluence] Cycle detection failed: {e}")
            self._cycle_signals_cache = []

        try:
            current_price = df["close"].iloc[-1]
            self._tpi_signals_cache = self.tpi_engine.analyze_temporal_patterns(df, current_price)
        except Exception as e:
            logger.warning(f"[TemporalConfluence] TPI analysis failed: {e}")
            self._tpi_signals_cache = []

    def _clear_cache(self) -> None:
        """Release cached sub-engine outputs."""
        self._gann_levels_cache = None
        self._gann_projections_cache = None
        self._cycle_signals_cache = None
        self._tpi_signals_cache = None

    # ------------------------------------------------------------------
    # Vectorized feature computations (batch)
    # ------------------------------------------------------------------

    def _vectorized_gann_cluster(self, df: pd.DataFrame) -> np.ndarray:
        """
        For each bar, flag 1 if the bar index (distance from any recent swing)
        aligns with a Gann vibration number within tolerance.

        Uses bars_since_* columns if available; otherwise falls back to a
        simple rolling-swing distance measure.
        """
        n = len(df)
        flags = np.zeros(n, dtype=int)

        # Prefer Wyckoff event distance columns
        event_cols = [c for c in df.columns if c.startswith("bars_since_")]
        if event_cols:
            for col in event_cols:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(9999).values
                for vib in self.gann_vibrations:
                    mask = np.abs(vals - vib) <= self.tolerance_bars
                    flags[mask] = 1
        else:
            # Fallback: use bars since last local low (rolling 20-bar)
            rolling_min_idx = df["low"].rolling(window=20, min_periods=1).apply(
                lambda x: len(x) - 1 - np.argmin(x), raw=True
            ).fillna(0).astype(int).values
            for vib in self.gann_vibrations:
                mask = np.abs(rolling_min_idx - vib) <= self.tolerance_bars
                flags[mask] = 1

        return flags

    def _vectorized_fib_cluster(self, df: pd.DataFrame) -> np.ndarray:
        """
        Flag bars that sit at a Fibonacci distance (in bars) from a recent
        swing high or low.
        """
        n = len(df)
        flags = np.zeros(n, dtype=int)

        # Try Wyckoff event distances first
        event_cols = [c for c in df.columns if c.startswith("bars_since_")]
        if event_cols:
            for col in event_cols:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(9999).values
                for fib in self.fib_levels:
                    mask = np.abs(vals - fib) <= self.tolerance_bars
                    flags[mask] = 1
        else:
            # Fallback: bars since rolling 50-bar high
            rolling_max_idx = df["high"].rolling(window=50, min_periods=1).apply(
                lambda x: len(x) - 1 - np.argmax(x), raw=True
            ).fillna(0).astype(int).values
            for fib in self.fib_levels:
                mask = np.abs(rolling_max_idx - fib) <= self.tolerance_bars
                flags[mask] = 1

        return flags

    def _vectorized_tpi_signal(self, df: pd.DataFrame) -> np.ndarray:
        """
        Produce a TPI signal strength array [0-1] for every bar.

        Combines time-price balance (price change normalised by ATR and time)
        with cycle-completion proximity from cached TPI signals.
        """
        n = len(df)
        scores = np.full(n, 0.3)  # Default mild signal

        # Price-time balance: ratio of recent move (in ATR units) to elapsed bars
        close = df["close"].values
        atr_col = "atr_14" if "atr_14" in df.columns else ("atr" if "atr" in df.columns else None)
        if atr_col is not None:
            atr = pd.to_numeric(df[atr_col], errors="coerce").fillna(0).values
        else:
            # Compute simple ATR proxy from high-low range
            atr = (df["high"] - df["low"]).rolling(window=14, min_periods=1).mean().fillna(0).values

        # 20-bar lookback price change in ATR units
        lookback = 20
        for i in range(lookback, n):
            price_change = abs(close[i] - close[i - lookback])
            if atr[i] > 0:
                move_in_atrs = price_change / atr[i]
                # Balanced move (1 ATR per ~10 bars) gets score 0.7
                expected_atrs = lookback / 10.0
                ratio = move_in_atrs / expected_atrs if expected_atrs > 0 else 1.0
                # Score peaks when ratio near 1.0 (balanced), drops for extremes
                scores[i] = float(np.clip(1.0 - abs(ratio - 1.0) * 0.5, 0.0, 1.0))

        # Boost bars that are near cached TPI cycle-completion signals
        if self._tpi_signals_cache:
            for sig in self._tpi_signals_cache:
                if sig.timestamp in df.index:
                    loc = df.index.get_loc(sig.timestamp)
                    # get_loc can return slice/array for duplicate timestamps
                    if isinstance(loc, slice):
                        idx = loc.start if loc.start is not None else 0
                    elif isinstance(loc, np.ndarray):
                        idx = int(np.flatnonzero(loc)[0])
                    else:
                        idx = int(loc)
                    # Boost the bar and its neighbors
                    for offset in range(-2, 3):
                        pos = idx + offset
                        if 0 <= pos < n:
                            boost = sig.confidence * 0.3 * (1.0 - abs(offset) / 3.0)
                            scores[pos] = min(1.0, scores[pos] + boost)

        return np.clip(scores, 0.0, 1.0)

    def _vectorized_cycle_phase(self, df: pd.DataFrame) -> np.ndarray:
        """
        Estimate the dominant cycle phase [0-1] for each bar.

        0.0 = cycle trough (potential reversal up)
        0.5 = mid-cycle
        1.0 = cycle peak (potential reversal down)

        Uses a simple detrended price oscillator normalised to [0, 1].
        """
        n = len(df)
        phases = np.full(n, 0.5)

        close = df["close"].values

        # Determine dominant cycle period from cached cycle signals
        dominant_period = 55  # Default Fibonacci period
        if self._cycle_signals_cache:
            # Pick the strongest detected cycle
            strongest = max(self._cycle_signals_cache, key=lambda s: s.strength, default=None)
            if strongest is not None:
                dominant_period = strongest.period

        # Detrended Price Oscillator (DPO)
        half = dominant_period // 2
        offset = half + 1
        if n > dominant_period + offset:
            sma = pd.Series(close).rolling(window=dominant_period, min_periods=dominant_period).mean().values
            dpo = np.full(n, 0.0)
            for i in range(offset, n):
                sma_idx = i - offset
                if sma_idx >= 0 and not np.isnan(sma[i]):
                    dpo[i] = close[i] - sma[i]

            # Normalize DPO to [0, 1] using rolling min/max
            dpo_series = pd.Series(dpo)
            roll_min = dpo_series.rolling(window=dominant_period * 2, min_periods=dominant_period).min()
            roll_max = dpo_series.rolling(window=dominant_period * 2, min_periods=dominant_period).max()
            dpo_range = (roll_max - roll_min).values
            roll_min_vals = roll_min.values

            for i in range(dominant_period * 2, n):
                if dpo_range[i] > 0:
                    phases[i] = (dpo[i] - roll_min_vals[i]) / dpo_range[i]
                else:
                    phases[i] = 0.5

        return np.clip(phases, 0.0, 1.0)

    def _vectorized_volatility_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score volatility compression/expansion phase [0-1].

        Compression (coiled spring) scores high -- good for entries.
        Expansion (climax) scores low -- late to enter.
        """
        n = len(df)
        scores = np.full(n, 0.5)

        atr_col = "atr_14" if "atr_14" in df.columns else ("atr" if "atr" in df.columns else None)
        if atr_col is not None:
            atr = pd.to_numeric(df[atr_col], errors="coerce").values
        else:
            atr = (df["high"] - df["low"]).rolling(window=14, min_periods=1).mean().values

        atr_ma = pd.Series(atr).rolling(window=20, min_periods=1).mean().values

        for i in range(20, n):
            if atr_ma[i] > 0 and not np.isnan(atr[i]):
                ratio = atr[i] / atr_ma[i]
                if ratio < self.vol_compression_threshold:
                    # Compression: score rises as ratio drops further below threshold
                    depth = (self.vol_compression_threshold - ratio) / self.vol_compression_threshold
                    scores[i] = 0.70 + 0.30 * min(depth * 2, 1.0)
                elif ratio > self.vol_expansion_threshold:
                    # Expansion: score drops as ratio exceeds threshold
                    excess = (ratio - self.vol_expansion_threshold) / self.vol_expansion_threshold
                    scores[i] = max(0.0, 0.30 - 0.30 * min(excess * 2, 1.0))
                else:
                    # Normal range: linear interpolation 0.3 to 0.7
                    norm = (ratio - self.vol_compression_threshold) / (
                        self.vol_expansion_threshold - self.vol_compression_threshold
                    )
                    scores[i] = 0.70 - 0.40 * norm  # 0.70 at compression edge, 0.30 at expansion edge

        return np.clip(scores, 0.0, 1.0)

    def _vectorized_momentum_phase(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score the current momentum phase [0-1] using RSI and rate of change.

        High score when momentum is rising from oversold (bullish setup for longs).
        Low score when momentum is extreme overbought.
        """
        n = len(df)
        scores = np.full(n, 0.5)

        # Use RSI if available
        rsi_col = "rsi" if "rsi" in df.columns else ("rsi_14" if "rsi_14" in df.columns else None)
        if rsi_col is not None:
            rsi = pd.to_numeric(df[rsi_col], errors="coerce").values
        else:
            # Simple RSI approximation from close prices
            close = df["close"].values
            rsi = np.full(n, 50.0)
            period = 14
            if n > period + 1:
                deltas = np.diff(close)
                for i in range(period, n - 1):
                    gains = np.mean(np.maximum(deltas[i - period:i], 0))
                    losses = np.mean(np.maximum(-deltas[i - period:i], 0))
                    if losses > 0:
                        rs = gains / losses
                        rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
                    else:
                        rsi[i + 1] = 100.0

        # RSI-based scoring
        # Capitulation (RSI < 25): high score 0.90 (reversal opportunity)
        # Oversold recovery (25-40): good 0.70
        # Neutral (40-60): 0.50
        # Overbought (60-75): declining 0.30
        # Euphoria (>75): low 0.10
        for i in range(n):
            r = rsi[i]
            if np.isnan(r):
                scores[i] = 0.5
            elif r < 25:
                scores[i] = 0.90
            elif r < 40:
                scores[i] = 0.50 + 0.40 * (40 - r) / 15.0  # 0.90 at 25, 0.50 at 40
            elif r < 60:
                scores[i] = 0.50
            elif r < 75:
                scores[i] = 0.50 - 0.20 * (r - 60) / 15.0  # 0.50 at 60, 0.30 at 75
            else:
                scores[i] = max(0.10, 0.30 - 0.20 * (r - 75) / 25.0)

        return np.clip(scores, 0.0, 1.0)

    def _vectorized_gann_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score based on proximity to Gann Square-of-Nine price levels.

        Uses cached GannLevel objects from the GannAnalyzer.
        """
        n = len(df)
        scores = np.full(n, 0.3)  # Default moderate score

        if not self._gann_levels_cache:
            return scores

        close = df["close"].values

        for level in self._gann_levels_cache:
            level_price = level.price
            if level_price <= 0:
                continue

            # Distance of each bar's close from the Gann level (as %)
            distances = np.abs(close - level_price) / level_price * 100.0

            # Score: 1.0 when distance=0, decays to 0 at 2% distance
            tolerance_pct = 2.0
            level_scores = np.maximum(0.0, 1.0 - distances / tolerance_pct) * level.confidence

            scores = np.maximum(scores, level_scores)

        return np.clip(scores, 0.0, 1.0)

    def _vectorized_cycle_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Score based on detected cycle signals from CycleDetector.

        Higher score when a strong cycle signal is detected near the current bar.
        """
        n = len(df)
        scores = np.full(n, 0.3)  # Default moderate score

        if not self._cycle_signals_cache:
            return scores

        for sig in self._cycle_signals_cache:
            # Each cycle signal affects bars near its timestamp
            if sig.timestamp in df.index:
                loc = df.index.get_loc(sig.timestamp)
                if isinstance(loc, slice):
                    center = loc.start if loc.start is not None else 0
                elif isinstance(loc, np.ndarray):
                    center = int(np.flatnonzero(loc)[0])
                else:
                    center = int(loc)
                spread = sig.period // 4  # Influence radius based on cycle length
                for offset in range(-spread, spread + 1):
                    pos = center + offset
                    if 0 <= pos < n:
                        decay = 1.0 - abs(offset) / (spread + 1)
                        contribution = sig.strength * sig.confidence * decay
                        scores[pos] = max(scores[pos], contribution)

        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reversal zone detection
    # ------------------------------------------------------------------

    def _detect_reversal_zones(
        self,
        gann_flags: np.ndarray,
        fib_flags: np.ndarray,
        tpi_scores: np.ndarray,
        cycle_phases: np.ndarray,
        vol_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Flag bars where multiple temporal signals converge suggesting a
        potential reversal.  A reversal zone requires at least
        ``reversal_zone_min_signals`` of these conditions:

        1. Gann time cluster active
        2. Fibonacci time cluster active
        3. TPI signal above 0.65 (strong time-price balance)
        4. Cycle phase near extremes (<0.15 trough or >0.85 peak)
        5. Volatility in compression phase (score > 0.70)
        """
        n = len(gann_flags)
        flags = np.zeros(n, dtype=int)

        cond_gann = gann_flags.astype(bool)
        cond_fib = fib_flags.astype(bool)
        cond_tpi = tpi_scores > 0.65
        cond_cycle = (cycle_phases < 0.15) | (cycle_phases > 0.85)
        cond_vol = vol_scores > 0.70

        signal_count = (
            cond_gann.astype(int)
            + cond_fib.astype(int)
            + cond_tpi.astype(int)
            + cond_cycle.astype(int)
            + cond_vol.astype(int)
        )

        flags[signal_count >= self.reversal_zone_min_signals] = 1

        return flags

    # ------------------------------------------------------------------
    # Single-bar scoring helpers (real-time)
    # ------------------------------------------------------------------

    def _single_bar_gann_score(self, row: Dict[str, Any]) -> float:
        """Score Gann proximity for a single bar."""
        close = row.get("close", 0)
        if close <= 0:
            return 0.3

        # Simple Square-of-Nine proximity (same logic as gann_cycles._square9)
        step = 9.0
        nearest_level = round(close / step) * step
        distance_pct = abs(close - nearest_level) / close * 100.0
        tolerance = 2.0
        score = max(0.0, 1.0 - distance_pct / tolerance)
        return float(np.clip(score, 0.0, 1.0))

    def _single_bar_cycle_score(self, row: Dict[str, Any]) -> float:
        """Score Fibonacci cycle alignment for a single bar."""
        # Check if we have bars_since columns
        event_keys = [k for k in row if str(k).startswith("bars_since_")]
        if not event_keys:
            return 0.3  # No event data

        hits = 0
        total_checked = 0
        for key in event_keys:
            val = row.get(key, 9999)
            if pd.notna(val) and val < 999:
                total_checked += 1
                val_int = int(val)
                for fib in self.fib_levels:
                    if abs(val_int - fib) <= self.tolerance_bars:
                        hits += 1
                        break

        if total_checked == 0:
            return 0.3

        # Score: 0 hits=0.2, 1 hit=0.50, 2+=0.75, 3+=0.90
        if hits >= 3:
            return 0.90
        elif hits >= 2:
            return 0.75
        elif hits >= 1:
            return 0.50
        else:
            return 0.20

    def _single_bar_tpi_score(self, row: Dict[str, Any]) -> float:
        """Estimate TPI signal strength for a single bar."""
        close = row.get("close", 0)
        atr = row.get("atr_14", row.get("atr", 0))
        if pd.isna(atr) or atr <= 0 or close <= 0:
            return 0.3

        # Simple price-time balance: how balanced is recent movement
        # Use available indicator as proxy
        bb_width = row.get("bb_width", None)
        if bb_width is not None and not pd.isna(bb_width):
            # Narrow BB suggests balanced/compressed state, good TPI
            if bb_width < 0.03:
                return 0.80
            elif bb_width < 0.06:
                return 0.60
            elif bb_width > 0.15:
                return 0.20
            else:
                return 0.40
        else:
            # Without BB width, use ATR relative to price
            atr_pct = atr / close * 100.0
            if atr_pct < 1.0:
                return 0.70  # Low vol, balanced
            elif atr_pct > 5.0:
                return 0.20  # High vol, unbalanced
            else:
                return 0.50  # Neutral

    def _single_bar_volatility_score(self, row: Dict[str, Any]) -> float:
        """Score volatility compression/expansion for a single bar."""
        atr = row.get("atr_14", row.get("atr", None))
        atr_ma = row.get("atr_ma_20", row.get("atr_ma", None))

        if atr is None or atr_ma is None or pd.isna(atr) or pd.isna(atr_ma) or atr_ma <= 0:
            return 0.5  # Neutral

        ratio = float(atr) / float(atr_ma)

        if ratio < self.vol_compression_threshold:
            return 0.85  # Compression
        elif ratio > self.vol_expansion_threshold:
            return 0.15  # Expansion climax
        else:
            # Linear between 0.7 (compression edge) and 0.3 (expansion edge)
            norm = (ratio - self.vol_compression_threshold) / (
                self.vol_expansion_threshold - self.vol_compression_threshold
            )
            return float(0.70 - 0.40 * norm)

    def _single_bar_momentum_score(self, row: Dict[str, Any]) -> float:
        """Score momentum phase for a single bar using RSI."""
        rsi = row.get("rsi", row.get("rsi_14", 50))
        if pd.isna(rsi):
            rsi = 50

        rsi = float(rsi)
        if rsi < 25:
            return 0.90
        elif rsi < 40:
            return 0.50 + 0.40 * (40 - rsi) / 15.0
        elif rsi < 60:
            return 0.50
        elif rsi < 75:
            return 0.50 - 0.20 * (rsi - 60) / 15.0
        else:
            return max(0.10, 0.30 - 0.20 * (rsi - 75) / 25.0)

    # ------------------------------------------------------------------
    # Utility / Introspection
    # ------------------------------------------------------------------

    def get_component_breakdown(self, row: Dict[str, Any]) -> Dict[str, float]:
        """
        Return the individual component scores for debugging/observability.

        Args:
            row: Dictionary of feature values for the current bar.

        Returns:
            Dict with keys: gann, cycles, tpi, volatility, momentum_phase,
            and the combined temporal_confluence_score.
        """
        gann = self._single_bar_gann_score(row)
        cycles = self._single_bar_cycle_score(row)
        tpi = self._single_bar_tpi_score(row)
        vol = self._single_bar_volatility_score(row)
        momentum = self._single_bar_momentum_score(row)

        confluence = (
            self.weights.get("gann", 0.20) * gann
            + self.weights.get("cycles", 0.25) * cycles
            + self.weights.get("tpi", 0.25) * tpi
            + self.weights.get("volatility", 0.15) * vol
            + self.weights.get("momentum_phase", 0.15) * momentum
        )

        return {
            "gann": gann,
            "cycles": cycles,
            "tpi": tpi,
            "volatility": vol,
            "momentum_phase": momentum,
            "temporal_confluence_score": float(np.clip(confluence, 0.0, 1.0)),
        }
