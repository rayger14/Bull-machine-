#!/usr/bin/env python3
"""
Live Feature Computer for Bull Machine Trading System — Full Soul Edition

Computes ~240+ features from raw OHLCV data for real-time signal generation
by calling the ACTUAL engine modules (Wyckoff, PTI, SMC, Temporal, FRVP, etc.)
instead of simplified proxies.

Maintains a rolling buffer of 500 1H OHLCV candles (for SMA200 + warmup).
All output feature names match the feature store column names EXACTLY so
the returned pd.Series is directly compatible with
  IsolatedArchetypeEngine.get_signals(bar=series)

Usage:
    from bin.live.live_feature_computer import LiveFeatureComputer

    computer = LiveFeatureComputer()
    computer.ingest_candles(historical_df)   # seed 500 candles

    for candle in live_feed:
        features = computer.update(candle)
        signals  = engine.get_signals(bar=features)

Author: Claude Code (System Architect)
Date: 2026-02-07, Full Soul Upgrade: 2026-02-10
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests

# Project root import path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TA-Lib with pandas fallback
# ---------------------------------------------------------------------------
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.info("TA-Lib not available; falling back to pandas rolling calculations.")

# ---------------------------------------------------------------------------
# yfinance for macro data (VIX, DXY, yields, gold)
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning(
        "MACRO DATA UNAVAILABLE: yfinance not installed. "
        "VIX_Z, DXY_Z, YIELD_CURVE, GOLD_Z will be 0. "
        "Install with: pip install yfinance"
    )

# ---------------------------------------------------------------------------
# Engine module imports (graceful fallback if deps missing on server)
# ---------------------------------------------------------------------------
try:
    from engine.wyckoff.events import detect_all_wyckoff_events, create_wyckoff_context
    WYCKOFF_AVAILABLE = True
except Exception:
    WYCKOFF_AVAILABLE = False
    logger.warning("Wyckoff engine not available; using EMA proxy fallback.")

try:
    from engine.psychology.pti import calculate_pti
    PTI_AVAILABLE = True
except Exception:
    PTI_AVAILABLE = False
    logger.warning("PTI engine not available; using default values.")

try:
    from engine.psychology.fakeout_intensity import detect_fakeout_intensity
    FAKEOUT_AVAILABLE = True
except Exception:
    FAKEOUT_AVAILABLE = False
    logger.warning("Fakeout engine not available; using default values.")

try:
    from engine.smc.smc_engine import SMCEngine
    SMC_AVAILABLE = True
except Exception:
    SMC_AVAILABLE = False
    logger.warning("SMC engine not available; using simplified BOS.")

try:
    from engine.temporal.temporal_confluence import TemporalConfluenceEngine
    TEMPORAL_AVAILABLE = True
except Exception:
    TEMPORAL_AVAILABLE = False
    logger.warning("Temporal engine not available; using default values.")

try:
    from engine.volume.frvp import calculate_frvp
    FRVP_AVAILABLE = True
except Exception:
    FRVP_AVAILABLE = False
    logger.warning("FRVP engine not available; using default values.")

try:
    from engine.structure.squiggle_pattern import detect_squiggle_123
    SQUIGGLE_AVAILABLE = True
except Exception:
    SQUIGGLE_AVAILABLE = False
    logger.warning("Squiggle engine not available; using default values.")

try:
    from engine.structure.boms_detector import detect_boms
    BOMS_AVAILABLE = True
except Exception:
    BOMS_AVAILABLE = False
    logger.warning("BOMS engine not available; using simplified values.")

try:
    from engine.context.probabilistic_regime_detector import ProbabilisticRegimeDetector
    PROB_REGIME_AVAILABLE = True
except Exception:
    PROB_REGIME_AVAILABLE = False
    logger.warning("ProbabilisticRegimeDetector not available; using SMA regime only.")

# ---------------------------------------------------------------------------
# Derivatives API (OI, L/S ratio, taker data)
# Fallback chain: Binance → OKX (free, no auth) → CoinGlass (paid)
# ---------------------------------------------------------------------------
try:
    from binance_futures_api import BinanceFuturesAPI
    HAS_BINANCE_API = True
except ImportError:
    HAS_BINANCE_API = False

try:
    from okx_derivatives_api import OKXDerivativesAPI
    HAS_OKX_API = True
except ImportError:
    HAS_OKX_API = False

try:
    from coinglass_api import CoinGlassAPI
    HAS_COINGLASS_API = True
except ImportError:
    HAS_COINGLASS_API = False


# ===========================================================================
# MacroDataFetcher — VIX, DXY, yields, gold from yfinance
# ===========================================================================

class MacroDataFetcher:
    """
    Fetches daily macro indicators (VIX, DXY, yields, gold) from yfinance.

    - 24h TTL cache (fetches at most once/day)
    - 90-day rolling z-scores
    - NEVER silently returns 0: tries all tickers, falls back to cached, warns loudly
    """

    INDICATORS = {
        'VIX':       {'tickers': ['^VIX'],              'transform': 'zscore'},
        'DXY':       {'tickers': ['DX-Y.NYB', 'UUP'],   'transform': 'zscore'},
        'YIELD_10Y': {'tickers': ['^TNX'],              'transform': 'raw'},
        'YIELD_5Y':  {'tickers': ['^FVX'],              'transform': 'raw'},  # ^FVX = 5-Year Treasury Yield Index
        'GOLD':      {'tickers': ['GC=F'],              'transform': 'zscore'},
        'OIL':       {'tickers': ['CL=F'],              'transform': 'zscore'},
    }

    TTL_SECONDS = 86400  # 24 hours

    COINGECKO_URL = 'https://api.coingecko.com/api/v3/global'
    FEAR_GREED_URL = 'https://api.alternative.me/fng/?limit=1'

    COINGECKO_ETH_URL = 'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=btc'

    def __init__(self):
        self._cache: Dict[str, float] = {}          # name → last known raw value
        self._history: Dict[str, List[float]] = {}   # name → 90-day rolling values
        self._last_fetch_time: float = 0.0
        self._ever_fetched: bool = False
        self._dominance_cache: Dict[str, float] = {}  # BTC.D / USDT.D / total_market_cap cached values
        self._last_dominance_fetch: float = 0.0
        self._fear_greed_value: Optional[float] = None  # 0-100 raw value
        self._fear_greed_label: str = ''                 # "Extreme Fear" etc.
        self._last_fg_fetch: float = 0.0
        self._eth_btc_ratio: Optional[float] = None     # ETH/BTC price ratio
        self._last_eth_fetch: float = 0.0

    def get_features(self) -> Dict[str, float]:
        """
        Return macro features dict. Fetches from yfinance if TTL expired.
        Keys: VIX_Z, DXY_Z, YIELD_CURVE, GOLD_Z, OIL_Z, BTC.D, USDT.D,
              FEAR_GREED, fear_greed_norm, FEAR_GREED_LABEL
        """
        now = time.time()
        if now - self._last_fetch_time > self.TTL_SECONDS:
            self._fetch_all()
            self._last_fetch_time = now

        out: Dict[str, float] = {}

        # Z-scored indicators
        for name, cfg in self.INDICATORS.items():
            raw = self._cache.get(name)
            if raw is None:
                out_key = f'{name}_Z' if cfg['transform'] == 'zscore' else name
                out[out_key] = 0.0
                continue

            if cfg['transform'] == 'zscore':
                history = self._history.get(name, [])
                if len(history) >= 5:
                    mean = np.mean(history)
                    std = np.std(history)
                    out[f'{name}_Z'] = float((raw - mean) / (std + 1e-10))
                else:
                    out[f'{name}_Z'] = 0.0
            else:
                out[name] = float(raw)

        # Derived: yield curve spread (10Y - 5Y; using ^FVX 5-Year as short-end proxy)
        y10 = self._cache.get('YIELD_10Y', 0.0)
        y5 = self._cache.get('YIELD_5Y', 0.0)
        out['YIELD_CURVE'] = float(y10 - y5) if (y10 and y5) else 0.0

        # BTC.D and USDT.D from CoinGecko (same 24h TTL cache)
        if now - self._last_dominance_fetch > self.TTL_SECONDS:
            self._fetch_dominance()
            self._last_dominance_fetch = now

        out['BTC.D'] = self._dominance_cache.get('BTC.D', None)
        out['USDT.D'] = self._dominance_cache.get('USDT.D', None)
        out['USDC.D'] = self._dominance_cache.get('USDC.D', None)

        if out['BTC.D'] is None:
            logger.info("MACRO: BTC.D unavailable — no cached value. Skipped.")
        if out['USDT.D'] is None:
            logger.info("MACRO: USDT.D unavailable — no cached value. Skipped.")
        if out['USDC.D'] is None:
            logger.info("MACRO: USDC.D unavailable — no cached value. Skipped.")

        # Fear & Greed Index from alternative.me (same 24h TTL)
        if now - self._last_fg_fetch > self.TTL_SECONDS:
            self._fetch_fear_greed()
            self._last_fg_fetch = now

        if self._fear_greed_value is not None:
            out['FEAR_GREED'] = self._fear_greed_value          # 0-100 raw
            out['fear_greed_norm'] = self._fear_greed_value / 100.0  # 0-1 for CMI
            out['FEAR_GREED_LABEL'] = self._fear_greed_label
        else:
            logger.warning("MACRO: Fear & Greed unavailable — no cached value.")

        # ETH/BTC ratio from CoinGecko (same 24h TTL)
        if now - self._last_eth_fetch > self.TTL_SECONDS:
            self._fetch_eth_btc()
            self._last_eth_fetch = now

        if self._eth_btc_ratio is not None:
            out['eth_btc_ratio'] = self._eth_btc_ratio

        # Total crypto market cap from CoinGecko /global (fetched with dominance)
        total_mcap = self._dominance_cache.get('total_market_cap')
        if total_mcap is not None:
            out['total_market_cap'] = total_mcap

        # Computed cross-asset ratios (BTC vs commodities)
        # These require a BTC price which the caller provides — we store raw
        # gold/oil prices from the cache so the caller can compute ratios.
        gold_raw = self._cache.get('GOLD')
        oil_raw = self._cache.get('OIL')
        if gold_raw is not None:
            out['gold_price'] = gold_raw
        if oil_raw is not None:
            out['oil_price'] = oil_raw

        return out

    def _fetch_all(self):
        """Try fetching all indicators from yfinance."""
        if not YFINANCE_AVAILABLE:
            return

        for name, cfg in self.INDICATORS.items():
            value = None
            for ticker_symbol in cfg['tickers']:
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    hist = ticker.history(period='3mo')
                    if hist is not None and len(hist) > 0 and 'Close' in hist.columns:
                        closes = hist['Close'].dropna()
                        if len(closes) > 0:
                            value = float(closes.iloc[-1])
                            # Store rolling history for z-score
                            self._history[name] = closes.values.tolist()[-90:]
                            break
                except Exception as e:
                    logger.warning(
                        f"MACRO FETCH FAILED: {name} from {ticker_symbol}: "
                        f"{type(e).__name__}({e})"
                    )
                    continue

            if value is not None:
                old = self._cache.get(name)
                self._cache[name] = value
                self._ever_fetched = True
                if old is None:
                    logger.info(f"MACRO: {name} = {value:.4f} (first fetch)")
            else:
                # All tickers failed — use cached value
                cached = self._cache.get(name)
                if cached is not None:
                    logger.warning(
                        f"MACRO FETCH FAILED: {name} — all tickers failed. "
                        f"Using cached value: {cached:.4f}"
                    )
                else:
                    logger.warning(
                        f"MACRO FETCH FAILED: {name} — all tickers failed, "
                        f"NO cached value available. Feature will be 0."
                    )

    def _fetch_dominance(self):
        """Fetch BTC, USDT, and USDC dominance + total market cap from CoinGecko /global endpoint."""
        try:
            resp = requests.get(self.COINGECKO_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('data', {})
            market_cap_pct = data.get('market_cap_percentage', {})

            btc_d = market_cap_pct.get('btc')
            usdt_d = market_cap_pct.get('usdt')
            usdc_d = market_cap_pct.get('usdc')

            if btc_d is not None:
                self._dominance_cache['BTC.D'] = float(btc_d)
                logger.info(f"MACRO: BTC.D = {btc_d:.2f}% (CoinGecko)")
            else:
                logger.warning(
                    "MACRO: CoinGecko response missing 'btc' in market_cap_percentage."
                )

            if usdt_d is not None:
                self._dominance_cache['USDT.D'] = float(usdt_d)
                logger.info(f"MACRO: USDT.D = {usdt_d:.2f}% (CoinGecko)")
            else:
                logger.warning(
                    "MACRO: CoinGecko response missing 'usdt' in market_cap_percentage."
                )

            if usdc_d is not None:
                self._dominance_cache['USDC.D'] = float(usdc_d)
                logger.info(f"MACRO: USDC.D = {usdc_d:.2f}% (CoinGecko)")
            else:
                logger.warning(
                    "MACRO: CoinGecko response missing 'usdc' in market_cap_percentage."
                )

            # Total crypto market cap (USD) from the same /global response
            total_mcap = data.get('total_market_cap', {}).get('usd')
            if total_mcap is not None:
                self._dominance_cache['total_market_cap'] = float(total_mcap)
                logger.info(f"MACRO: Total Market Cap = ${total_mcap/1e12:.2f}T (CoinGecko)")
            else:
                logger.warning("MACRO: CoinGecko response missing total_market_cap.usd.")

        except Exception as e:
            cached_btc = self._dominance_cache.get('BTC.D')
            cached_usdt = self._dominance_cache.get('USDT.D')
            cached_usdc = self._dominance_cache.get('USDC.D')
            if cached_btc is not None or cached_usdt is not None:
                logger.warning(
                    f"MACRO: CoinGecko fetch failed ({type(e).__name__}: {e}). "
                    f"Using cached BTC.D={cached_btc}, USDT.D={cached_usdt}, USDC.D={cached_usdc}"
                )
            else:
                logger.warning(
                    f"MACRO: CoinGecko fetch failed ({type(e).__name__}: {e}). "
                    f"No cached dominance values. BTC.D, USDT.D, and USDC.D will be None."
                )

    def _fetch_fear_greed(self):
        """Fetch Crypto Fear & Greed Index from alternative.me API."""
        try:
            resp = requests.get(self.FEAR_GREED_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json().get('data', [])
            if data and len(data) > 0:
                entry = data[0]
                value = int(entry.get('value', 0))
                label = entry.get('value_classification', '')
                self._fear_greed_value = float(value)
                self._fear_greed_label = label
                logger.info(
                    f"MACRO: Fear & Greed = {value} ({label}) (alternative.me)"
                )
            else:
                logger.warning("MACRO: Fear & Greed API returned empty data.")
        except Exception as e:
            if self._fear_greed_value is not None:
                logger.warning(
                    f"MACRO: Fear & Greed fetch failed ({type(e).__name__}: {e}). "
                    f"Using cached value: {self._fear_greed_value}"
                )
            else:
                logger.warning(
                    f"MACRO: Fear & Greed fetch failed ({type(e).__name__}: {e}). "
                    f"No cached value. F&G will be unavailable."
                )

    def _fetch_eth_btc(self):
        """Fetch ETH/BTC ratio from CoinGecko /simple/price endpoint."""
        try:
            resp = requests.get(self.COINGECKO_ETH_URL, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            eth_btc = data.get('ethereum', {}).get('btc')
            if eth_btc is not None:
                self._eth_btc_ratio = float(eth_btc)
                logger.info(f"MACRO: ETH/BTC = {eth_btc:.6f} (CoinGecko)")
            else:
                logger.warning("MACRO: CoinGecko ETH/BTC response missing data.")
        except Exception as e:
            if self._eth_btc_ratio is not None:
                logger.warning(
                    f"MACRO: ETH/BTC fetch failed ({type(e).__name__}: {e}). "
                    f"Using cached value: {self._eth_btc_ratio:.6f}"
                )
            else:
                logger.warning(
                    f"MACRO: ETH/BTC fetch failed ({type(e).__name__}: {e}). "
                    f"No cached value. ETH/BTC will be unavailable."
                )

    def get_historical_daily(self) -> Optional[pd.DataFrame]:
        """
        Fetch 90 days of daily macro data with timestamps for warmup backfill.

        Returns a DataFrame indexed by date with columns:
            VIX_Z, DXY_Z, GOLD_Z, OIL_Z, YIELD_CURVE,
            fear_greed_norm, BTC.D, USDT.D, USDC.D
        Daily values — caller should forward-fill to hourly.
        Also populates internal caches so get_features() works immediately.
        """
        if not YFINANCE_AVAILABLE:
            logger.warning("MACRO HISTORICAL: yfinance not available")
            return None

        frames: Dict[str, pd.Series] = {}

        # 1. Fetch yfinance indicators with timestamps
        for name, cfg in self.INDICATORS.items():
            for ticker_symbol in cfg['tickers']:
                try:
                    ticker = yf.Ticker(ticker_symbol)
                    hist = ticker.history(period='3mo')
                    if hist is not None and len(hist) > 0 and 'Close' in hist.columns:
                        closes = hist['Close'].dropna()
                        if len(closes) > 0:
                            # Update internal caches (same as _fetch_all)
                            self._cache[name] = float(closes.iloc[-1])
                            self._history[name] = closes.values.tolist()[-90:]
                            self._ever_fetched = True

                            if cfg['transform'] == 'zscore':
                                mean = closes.rolling(90, min_periods=5).mean()
                                std = closes.rolling(90, min_periods=5).std()
                                z = (closes - mean) / (std + 1e-10)
                                frames[f'{name}_Z'] = z
                            else:
                                frames[name] = closes
                            logger.info(
                                f"MACRO HISTORICAL: {name} = {len(closes)} days "
                                f"[{closes.index[0].date()} .. {closes.index[-1].date()}]"
                            )
                            break
                except Exception as e:
                    logger.warning(
                        f"MACRO HISTORICAL: {name} from {ticker_symbol} failed: "
                        f"{type(e).__name__}({e})"
                    )
                    continue

        if not frames:
            logger.warning("MACRO HISTORICAL: No yfinance data fetched")
            return None

        # Compute yield curve from raw yields (10Y - 5Y; ^FVX is 5-Year, not 2-Year)
        if 'YIELD_10Y' in frames and 'YIELD_5Y' in frames:
            frames['YIELD_CURVE'] = frames['YIELD_10Y'] - frames['YIELD_5Y']
            del frames['YIELD_10Y']
            del frames['YIELD_5Y']
        elif 'YIELD_10Y' in frames:
            del frames['YIELD_10Y']
        elif 'YIELD_5Y' in frames:
            del frames['YIELD_5Y']

        # 2. Fetch Fear & Greed historical (last 8 days from alternative.me)
        try:
            resp = requests.get(
                'https://api.alternative.me/fng/?limit=8', timeout=10
            )
            resp.raise_for_status()
            fg_data = resp.json().get('data', [])
            if fg_data:
                fg_dates = []
                fg_vals = []
                for entry in fg_data:
                    ts = int(entry.get('timestamp', 0))
                    val = int(entry.get('value', 0))
                    if ts > 0:
                        fg_dates.append(pd.Timestamp.fromtimestamp(ts, tz='UTC'))
                        fg_vals.append(val / 100.0)  # normalize to 0-1
                if fg_dates:
                    fg_series = pd.Series(fg_vals, index=fg_dates, name='fear_greed_norm')
                    fg_series = fg_series.sort_index()
                    frames['fear_greed_norm'] = fg_series
                    # Update internal cache
                    self._fear_greed_value = float(fg_vals[0]) * 100.0  # most recent
                    self._fear_greed_label = fg_data[0].get('value_classification', '')
                    self._last_fg_fetch = time.time()
                    logger.info(
                        f"MACRO HISTORICAL: Fear & Greed = {len(fg_dates)} days"
                    )
        except Exception as e:
            logger.warning(f"MACRO HISTORICAL: F&G fetch failed: {e}")

        # 3. Fetch current dominance (CoinGecko only has real-time)
        self._fetch_dominance()
        self._last_dominance_fetch = time.time()
        self._last_fetch_time = time.time()

        # Build combined DataFrame
        df = pd.DataFrame(frames)

        # Ensure timezone-naive for consistent merging
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Forward-fill so daily values cover all dates
        df = df.sort_index().ffill()

        logger.info(
            f"MACRO HISTORICAL: Combined DataFrame: {len(df)} rows x "
            f"{len(df.columns)} cols [{df.columns.tolist()}]"
        )
        return df


# ===========================================================================
# LiveFeatureComputer
# ===========================================================================

class LiveFeatureComputer:
    """
    Compute features from raw OHLCV candles for live signal generation.

    Maintains an internal rolling buffer of up to ``buffer_size`` 1H candles
    and exposes two entry points:

    * ``ingest_candles(df)``  -- seed with historical data for indicator warmup
    * ``update(candle)``      -- append one candle, return full feature Series
    """

    BUFFER_SIZE = 1000  # SMA200 + 800 bars warmup (41 days for solid 1D Wyckoff)

    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = max(buffer_size, 1000)
        # Internal DataFrame buffer (columns: open, high, low, close, volume)
        self._buf: Optional[pd.DataFrame] = None
        # Funding rate history for z-score
        self._funding_history: List[float] = []
        # Wyckoff enrichment (stored outside pd.Series — lists/dicts)
        self.last_wyckoff_event_history: list = []
        self.last_wyckoff_conviction: dict = {}

        # Engine instances (initialized once, reused per bar)
        self._smc_engine = SMCEngine({}) if SMC_AVAILABLE else None
        self._temporal_engine = TemporalConfluenceEngine() if TEMPORAL_AVAILABLE else None
        self._macro_fetcher = MacroDataFetcher()

        # Probabilistic regime detector
        self._prob_detector = None
        if PROB_REGIME_AVAILABLE:
            crisis_model = None
            model_path = PROJECT_ROOT / 'models' / 'logistic_regime_v4_no_funding_stratified.pkl'
            if model_path.exists():
                try:
                    import joblib
                    crisis_model = joblib.load(model_path)
                    logger.info(f"Regime ML model loaded from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load regime ML model: {e}. crisis_prob will be 0.")
            else:
                logger.info(f"Regime ML model not found at {model_path}. crisis_prob will be 0.")

            if crisis_model is None:
                # Mock model: returns low crisis prob
                class _MockCrisisModel:
                    def predict_proba(self, X):
                        import numpy as _np
                        return _np.array([[0.05, 0.25, 0.50, 0.20]])
                crisis_model = _MockCrisisModel()

            self._prob_detector = ProbabilisticRegimeDetector(
                crisis_model=crisis_model,
                crisis_threshold=0.15
            )
            logger.info("ProbabilisticRegimeDetector initialized (crisis_prob, risk_temperature, instability_score)")

        # Derivatives API (OI, L/S, taker data)
        # Fallback chain: Binance → OKX (free) → CoinGlass (paid)
        self.binance_api = None
        self._okx_api = None
        self._coinglass_api = None
        self._derivatives_source = "none"
        if HAS_BINANCE_API:
            self.binance_api = BinanceFuturesAPI()
            self._derivatives_source = "binance"
        if HAS_OKX_API:
            self._okx_api = OKXDerivativesAPI()
            if not self.binance_api:
                self._derivatives_source = "okx"
            logger.info("OKX API initialized (free, no auth, US-accessible)")
        if HAS_COINGLASS_API and os.environ.get("COINGLASS_API_KEY"):
            self._coinglass_api = CoinGlassAPI()
            if self._derivatives_source == "none":
                self._derivatives_source = "coinglass"
            logger.info("CoinGlass API initialized (paid fallback)")
        self._binance_cache = {}
        self._binance_last_fetch = 0
        self._binance_fetch_interval = 300  # 5 minutes between fetches
        self._binance_geo_blocked = False  # Track if Binance returned 451

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_candles(self, df: pd.DataFrame) -> None:
        """
        Seed the computer with historical candles for warmup.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            Index should be DatetimeIndex (or at least ordered).
            Only the last ``buffer_size`` rows are kept.
        """
        required = {'open', 'high', 'low', 'close', 'volume'}
        # Accept case-insensitive column names
        col_map = {}
        for col in df.columns:
            if col.lower() in required:
                col_map[col] = col.lower()
        df = df.rename(columns=col_map)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        buf = df[['open', 'high', 'low', 'close', 'volume']].copy()
        buf = buf.tail(self.buffer_size)

        # Ensure DatetimeIndex
        if not isinstance(buf.index, pd.DatetimeIndex):
            buf.index = pd.to_datetime(buf.index)

        self._buf = buf.copy()
        logger.info(
            f"Ingested {len(self._buf)} candles "
            f"({self._buf.index[0]} to {self._buf.index[-1]})"
        )

    def update(self, candle: dict) -> pd.Series:
        """
        Process one new candle and return the full feature vector.

        Parameters
        ----------
        candle : dict
            Must contain keys: open, high, low, close, volume.
            Optionally: timestamp (or datetime), funding_rate.

        Returns
        -------
        pd.Series
            Feature vector compatible with
            ``IsolatedArchetypeEngine.get_signals(bar=series)``.
            Index labels match feature store column names exactly.
        """
        # -- Append candle to buffer ------------------------------------------
        ts = candle.get('timestamp', candle.get('datetime', pd.Timestamp.now()))
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)

        new_row = pd.DataFrame(
            [{
                'open': float(candle['open']),
                'high': float(candle['high']),
                'low': float(candle['low']),
                'close': float(candle['close']),
                'volume': float(candle['volume']),
            }],
            index=[ts],
        )

        if self._buf is None:
            self._buf = new_row.copy()
        else:
            self._buf = pd.concat([self._buf, new_row])

        # Trim to buffer size
        if len(self._buf) > self.buffer_size:
            self._buf = self._buf.iloc[-self.buffer_size:]

        # Track funding rate history (if provided)
        fr = candle.get('funding_rate', None)
        if fr is not None:
            self._funding_history.append(float(fr))
            if len(self._funding_history) > 500:
                self._funding_history = self._funding_history[-500:]

        # -- Compute all features ---------------------------------------------
        features: Dict[str, Any] = {}

        # A. OHLCV
        features.update(self._ohlcv(candle, ts))

        # B. Technical indicators (RSI, ADX, ATR, SMA, EMA, BB, MACD)
        features.update(self._technical_indicators())

        # C. Volume features (z-score, ratio, 7d z)
        features.update(self._volume_features())

        # E. Wyckoff events (REAL engine: SC, BC, AR, ST, SOS, Spring, UT, LPS + multi-TF)
        features.update(self._wyckoff_features())

        # F. PTI / Fakeout (REAL engine: Psychology Trap Index + fakeout intensity)
        features.update(self._pti_fakeout_features())

        # G. SMC features (REAL engine: BOS, CHOCH, FVG, Order Blocks + 4H)
        features.update(self._smc_features())

        # H. BOMS / Liquidity (displacement, volume climax, FVG, liquidity score)
        features.update(self._boms_liquidity_features())

        # I. 4H Structure (REAL engine: Squiggle 1-2-3 + BOMS 4H/1D)
        features.update(self._structure_4h_features())

        # J. Fibonacci (swing detection + retracement + time zones)
        features.update(self._fib_features())

        # K. Coil / Squeeze (ATR ratio + BB squeeze, multi-TF)
        features.update(self._coil_features(features))

        # L. Crisis indicators (crash frequency, drawdown, aftershock)
        features.update(self._crisis_features())

        # M. Temporal confluence (REAL engine: Gann, Fib time, TPI, cycles)
        features.update(self._temporal_features(features))

        # N. FRVP (REAL engine: Volume Profile - POC, Value Area)
        features.update(self._frvp_features())

        # O. Penalty features (funding Z, bars_since_pivot, fib_in_premium)
        features.update(self._penalty_features())

        # P. Fusion scores (momentum, liquidity, structure composites)
        features.update(self._fusion_scores(features))

        # P2. Derivatives institutional flow features (OKX/Binance/CoinGlass)
        features.update(self._binance_futures_features())

        # Feed derivatives funding rate into _funding_history for Z-score computation.
        # The Z-score needs 10+ historical values to compute — this populates it
        # from the OKX/Binance/CoinGlass funding rate so funding_Z becomes non-zero
        # after 10 bars (~10 hours of warmup).
        deriv_fr = features.get('binance_funding_rate')
        if deriv_fr is not None and deriv_fr != 0.0:
            self._funding_history.append(float(deriv_fr))
            if len(self._funding_history) > 500:
                self._funding_history = self._funding_history[-500:]
            # Recompute funding_Z now that we have fresh data
            if len(self._funding_history) >= 10:
                arr = np.array(self._funding_history)
                mean_f = np.mean(arr)
                std_f = np.std(arr)
                features['funding_Z'] = float((arr[-1] - mean_f) / (std_f + 1e-10))
                features['funding_rate'] = float(arr[-1])

        # Compute oi_price_divergence if we have price context
        if 'oi_change_4h' in features and features['oi_change_4h'] != 0:
            # Use 4h price change (from close array)
            close = self._buf['close'].values.astype(float)
            if len(close) >= 4:
                price_change_4h = (close[-1] - close[-4]) / close[-4] if close[-4] > 0 else 0
                if (price_change_4h > 0.005 and features['oi_change_4h'] < -0.01) or \
                   (price_change_4h < -0.005 and features['oi_change_4h'] > 0.01):
                    features['oi_price_divergence'] = 1

        # Q. Extra features referenced by archetype_instance.py
        features.update(self._extra_archetype_features(features))

        # D. Regime detection (MOVED AFTER Q: needs rv_20d, drawdown_persistence, etc.)
        # Probabilistic: crisis_prob, risk_temperature, instability_score
        # regime_label derived from CMI components (not SMA crossovers)
        features.update(self._regime_detection(features))

        # -- Diagnostic logging ------------------------------------------------
        # Compute what the dynamic threshold WOULD be given current conditions
        _rt = features.get('risk_temperature', 0.5)
        _inst = features.get('instability_score', 0.3)
        _cp = features.get('crisis_prob', 0.02)
        # Default adaptive_fusion params (matching configs/bull_machine_isolated_v11_fixed.json)
        _base_thr = 0.18
        _temp_range = 0.38
        _instab_range = 0.15
        _projected_threshold = _base_thr + (1.0 - _rt) * _temp_range + _inst * _instab_range

        # EMA alignment state
        _p_above_50 = features.get('price_above_ema_50', 0)
        _ema_50_200 = features.get('ema_50_above_200', 0)
        if _p_above_50 and _ema_50_200:
            _ema_state = 'BULL(price>ema50>ema200)'
        elif _p_above_50:
            _ema_state = 'RECOVERY(price>ema50)'
        elif _ema_50_200:
            _ema_state = 'DISTRIBUTION(ema50>ema200)'
        else:
            _ema_state = 'BEAR(price<ema50<ema200)'

        logger.info(
            f"[FEATURES] regime={features.get('regime_label')} | "
            f"ema_state={_ema_state} | "
            f"projected_threshold={_projected_threshold:.3f} | "
            f"crisis_prob={_cp:.3f} | "
            f"risk_temp={_rt:.3f} | "
            f"instability={_inst:.3f} | "
            f"n_features={len(features)}"
        )
        logger.debug(
            f"[FEATURES_DETAIL] "
            f"wyckoff_conf={features.get('wyckoff_event_confidence', 0):.3f} | "
            f"pti={features.get('tf1h_pti_score', 0):.3f} | "
            f"smc={features.get('fusion_smc', 0):.3f} | "
            f"coil_1h={features.get('tf1h_coil_score', 0):.3f} | "
            f"adx={features.get('adx', features.get('adx_14', 0)):.1f} | "
            f"rsi={features.get('rsi_14', 0):.1f} | "
            f"chop={features.get('chop_score', 0):.3f} | "
            f"dd_persist={features.get('drawdown_persistence', 0):.3f} | "
            f"rv_20d={features.get('rv_20d', 0):.3f}"
        )

        # -- Build Series and fill NaN ----------------------------------------
        series = pd.Series(features, name=ts)
        series = self._fill_nans(series)

        return series

    # ------------------------------------------------------------------
    # Private: OHLCV
    # ------------------------------------------------------------------

    def _ohlcv(self, candle: dict, ts: pd.Timestamp) -> Dict[str, Any]:
        return {
            'open': float(candle['open']),
            'high': float(candle['high']),
            'low': float(candle['low']),
            'close': float(candle['close']),
            'volume': float(candle['volume']),
            'timestamp': ts,
        }

    # ------------------------------------------------------------------
    # Private: Technical Indicators
    # ------------------------------------------------------------------

    def _technical_indicators(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        close = self._buf['close'].values.astype(float)
        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)
        n = len(close)

        # RSI 14
        out['rsi_14'] = self._calc_rsi(close, 14)

        # ADX 14 (also store as adx and adx_14)
        adx_val = self._calc_adx(high, low, close, 14)
        out['adx'] = adx_val
        # NOTE: archetype_instance._get_momentum_score reads 'adx_14'
        out['adx_14'] = adx_val

        # ATR 14 and ATR 20
        out['atr_14'] = self._calc_atr(high, low, close, 14)
        out['atr_20'] = self._calc_atr(high, low, close, 20)

        # SMAs
        out['sma_50'] = self._sma(close, 50)
        out['sma_200'] = self._sma(close, 200)
        out['sma_20'] = self._sma(close, 20)
        out['sma_100'] = self._sma(close, 100)

        # EMAs
        out['ema_9'] = self._ema(close, 9)
        out['ema_21'] = self._ema(close, 21)
        out['ema_50'] = self._ema(close, 50)
        out['ema_200'] = self._ema(close, 200)

        # Bollinger Band width
        out['bb_width'] = self._bb_width(close, 20, 2.0)

        # MACD (12, 26, 9)
        macd_val, macd_sig, macd_h = self._calc_macd(close, 12, 26, 9)
        out['macd'] = macd_val
        out['macd_signal'] = macd_sig
        out['macd_hist'] = macd_h

        # RSI 21 (used by some features)
        out['rsi_21'] = self._calc_rsi(close, 21)

        return out

    # ------------------------------------------------------------------
    # Private: Volume Features
    # ------------------------------------------------------------------

    def _volume_features(self) -> Dict[str, float]:
        vol = self._buf['volume'].values.astype(float)
        n = len(vol)
        out: Dict[str, float] = {}

        if n >= 20:
            mean_20 = np.mean(vol[-20:])
            std_20 = np.std(vol[-20:])
            out['volume_zscore'] = (vol[-1] - mean_20) / (std_20 + 1e-10)
            out['volume_ma_20'] = mean_20
            out['volume_z'] = out['volume_zscore']
        else:
            out['volume_zscore'] = 0.0
            out['volume_ma_20'] = vol[-1] if n > 0 else 0.0
            out['volume_z'] = 0.0

        # volume_ratio (current / ma20)
        ma20 = out['volume_ma_20']
        out['volume_ratio'] = vol[-1] / ma20 if ma20 > 0 else 1.0

        return out

    # ------------------------------------------------------------------
    # Private: Regime Detection
    # ------------------------------------------------------------------

    @staticmethod
    def _regime_label_from_risk_temp(risk_temp: float, crisis_prob: float) -> str:
        """
        Derive a human-readable regime label from CMI components.

        Uses the dynamic threshold system (risk_temp, crisis_prob) which is
        the real regime intelligence, instead of the old SMA crossover labels.
        """
        if crisis_prob >= 0.5:
            return 'crisis'
        if risk_temp >= 0.6:
            return 'bull'
        if risk_temp >= 0.4:
            return 'neutral'
        return 'bear'

    def _regime_detection(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        CMI v0: Composite Market Intelligence — orthogonal to archetype fusion.

        Uses the same formulas as the standalone backtester for consistency.
        No Wyckoff/temporal/SMC here — those stay in per-archetype fusion only.

        Returns:
            crisis_prob (float): [0-1] pure stress measurement
            risk_temperature (float): [0-1] environment favorability (0=cold/bear, 1=hot/bull)
            instability_score (float): [0-1] chop/instability (0=stable, 1=choppy)
        """
        out = {}

        try:
            def _get(col, default=0.0):
                val = features.get(col, default)
                try:
                    val = float(val)
                    if val != val:  # NaN check
                        return float(default)
                    return val
                except (TypeError, ValueError):
                    return float(default)

            # --- CMI v0: risk_temperature [0-1] ---
            # Trend bias (45%)
            p_above_50 = _get('price_above_ema_50', 0)
            ema_50_200 = _get('ema_50_above_200', 0)
            if p_above_50 and ema_50_200:
                trend_align = 1.0   # Bull
            elif p_above_50:
                trend_align = 0.6   # Early recovery
            elif ema_50_200:
                trend_align = 0.4   # Distribution
            else:
                trend_align = 0.0   # Bear

            # Momentum health (25%)
            adx = _get('adx', _get('adx_14', 20.0))
            trend_strength = min(adx / 40.0, 1.0)

            # Sentiment contrarian bump (15%)
            fear_greed = _get('fear_greed_norm', 0.5)
            sentiment_score = fear_greed

            # Drawdown context
            # Default 0.5 = neutral (old 0.9 biased bearish, suppressed risk_temp)
            dd_persist = _get('drawdown_persistence', 0.5)
            dd_score = max(1.0 - dd_persist, 0.0)

            # --- Derivatives heat: institutional conviction from OI/funding/taker ---
            # NaN-safe: defaults to 0.5 (neutral) when data unavailable
            oi_4h_val = features.get('oi_change_4h')
            has_oi_data = oi_4h_val is not None and not (isinstance(oi_4h_val, float) and oi_4h_val != oi_4h_val)

            if has_oi_data:
                oi_4h = _get('oi_change_4h', 0.0)
                oi_momentum = min(max(oi_4h + 0.5, 0.0), 1.0)
                fund_rate = _get('binance_funding_rate', 0.0)
                funding_health = max(1.0 - abs(fund_rate) * 5000.0, 0.0)
                taker = _get('taker_imbalance', 0.0)
                taker_conviction = min(max(taker + 0.5, 0.0), 1.0)
                derivatives_heat = 0.40 * oi_momentum + 0.30 * funding_health + 0.30 * taker_conviction
            else:
                derivatives_heat = 0.5  # Neutral when no data

            # Weights match config: trend=0.30, strength=0.05, sentiment=0.15, dd=0.50, deriv=0.0
            # derivatives_heat DISABLED (weight=0.0) pending >2 years of OI data
            out['risk_temperature'] = (
                0.30 * trend_align + 0.05 * trend_strength +
                0.15 * sentiment_score + 0.50 * dd_score +
                0.00 * derivatives_heat
            )

            # --- CMI v0: instability [0-1] ---
            # chop_score now uses max(0, 1-ADX/50) formula (was 1-ADX/100)
            chop = _get('chop_score', 0.5)
            adx_weakness = max(1.0 - adx / 40.0, 0.0)
            wick = _get('wick_ratio', 1.0)
            wick_sc = min(wick / 5.0, 1.0)
            vol_z = _get('volume_z_7d', 0.0)
            vol_instab = min(abs(vol_z) / 2.5, 1.0)

            out['instability_score'] = (
                0.40 * chop + 0.10 * adx_weakness +
                0.25 * wick_sc + 0.25 * vol_instab
            )

            # --- CMI v0: crisis_prob [0-1] — pure stress, no Wyckoff ---
            dd = _get('drawdown_persistence', 0.0)
            crash_freq = _get('crash_frequency_7d', 0.0)
            crisis_persist = _get('crisis_persistence', 0.0)
            crisis_signals = int(dd > 0.96) + int(crash_freq >= 2) + int(crisis_persist > 0.55)
            if crisis_signals >= 2:
                base_crisis = min(0.7 + 0.1 * crisis_signals, 1.0)
            elif crisis_signals == 1:
                base_crisis = 0.10
            else:
                base_crisis = 0.02

            # Volatility shock (20%)
            rv = _get('rv_20d', 0.6)
            vol_shock = min(max(rv - 0.8, 0.0) / 0.4, 1.0)

            # Sentiment extreme (20%)
            sentiment_crisis = max(0.0, (0.20 - fear_greed) / 0.20)

            out['crisis_prob'] = 0.45 * base_crisis + 0.10 * vol_shock + 0.45 * sentiment_crisis

            # Derive regime_label from CMI components for logging/display
            out['regime_label'] = self._regime_label_from_risk_temp(
                out['risk_temperature'], out['crisis_prob']
            )

            logger.debug(
                f"[REGIME] label={out['regime_label']}, "
                f"crisis_prob={out['crisis_prob']:.3f}, "
                f"risk_temp={out['risk_temperature']:.3f}, "
                f"instability={out['instability_score']:.3f}"
            )

        except Exception as e:
            logger.warning(f"CMI v0 scoring failed: {e}. Using defaults.")
            out['crisis_prob'] = 0.0
            out['risk_temperature'] = 0.5
            out['instability_score'] = 0.3
            out['regime_label'] = 'neutral'

        return out

    # ------------------------------------------------------------------
    # Private: SMC Features (REAL engine)
    # ------------------------------------------------------------------

    def _smc_features(self) -> Dict[str, Any]:
        """
        Real Smart Money Concepts analysis using engine.smc.smc_engine.

        Detects: BOS, CHOCH, FVG, Order Blocks, Liquidity Sweeps.
        Computes composite smc_score from confluence of sub-detectors.
        Falls back to simplified BOS if engine not available.
        """
        out: Dict[str, Any] = {}

        if not SMC_AVAILABLE or self._smc_engine is None or self._buf is None or len(self._buf) < 50:
            return self._smc_features_fallback()

        try:
            # Full SMC analysis on 1H buffer
            signal = self._smc_engine.analyze_smc(self._buf)
            out['fusion_smc'] = float(signal.confluence_score)
            out['smc_strength'] = float(signal.strength)
            out['smc_confidence'] = float(signal.confidence)

            # BOS detection
            has_bull_bos = any(
                getattr(sb, 'direction', '') == 'bullish'
                for sb in (signal.structure_breaks or [])
            )
            has_bear_bos = any(
                getattr(sb, 'direction', '') == 'bearish'
                for sb in (signal.structure_breaks or [])
            )
            out['tf1h_bos_bullish'] = 1 if has_bull_bos else 0
            out['tf1h_bos_bearish'] = 1 if has_bear_bos else 0
            out['tf1h_bos_detected'] = 1 if (has_bull_bos or has_bear_bos) else 0

            # CHOCH detection
            trend_name = getattr(signal.trend_state, 'name', '') if signal.trend_state else ''
            out['tf1h_choch_detected'] = 1 if 'CHOCH' in str(trend_name).upper() else 0

            # FVG detection
            out['tf1h_fvg_present'] = 1 if signal.fair_value_gaps else 0

        except Exception as e:
            logger.warning(f"SMC 1H engine failed, using fallback: {e}")
            out = self._smc_features_fallback()

        # 4H SMC (from resampled buffer)
        try:
            if SMC_AVAILABLE and self._smc_engine and self._buf is not None:
                buf_4h = self._resample_to_tf(self._buf, '4H')
                if len(buf_4h) >= 50:
                    sig_4h = self._smc_engine.analyze_smc(buf_4h)
                    out['tf4h_bos_bullish'] = 1 if any(
                        getattr(sb, 'direction', '') == 'bullish'
                        for sb in (sig_4h.structure_breaks or [])
                    ) else 0
                    out['tf4h_bos_bearish'] = 1 if any(
                        getattr(sb, 'direction', '') == 'bearish'
                        for sb in (sig_4h.structure_breaks or [])
                    ) else 0
                    out['tf4h_fvg_present'] = 1 if sig_4h.fair_value_gaps else 0
                    out['tf4h_choch_flag'] = 1 if 'CHOCH' in str(
                        getattr(sig_4h.trend_state, 'name', '')).upper() else 0
        except Exception as e:
            logger.warning(f"SMC 4H failed: {e}")

        # Ensure 4H defaults
        out.setdefault('tf4h_bos_bullish', 0)
        out.setdefault('tf4h_bos_bearish', 0)
        out.setdefault('tf4h_fvg_present', 0)
        out.setdefault('tf4h_choch_flag', 0)

        return out

    def _smc_features_fallback(self) -> Dict[str, int]:
        """Simplified BOS fallback when SMC engine is not available."""
        out: Dict[str, int] = {}
        n = len(self._buf) if self._buf is not None else 0

        if n < 11:
            return {'tf1h_bos_bullish': 0, 'tf1h_bos_bearish': 0,
                    'tf1h_bos_detected': 0, 'tf1h_choch_detected': 0,
                    'tf1h_fvg_present': 0, 'fusion_smc': 0, 'smc_strength': 0, 'smc_confidence': 0}

        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)
        prev_high_10 = np.max(high[-11:-1])
        prev_low_10 = np.min(low[-11:-1])
        out['tf1h_bos_bullish'] = 1 if high[-1] > prev_high_10 else 0
        out['tf1h_bos_bearish'] = 1 if low[-1] < prev_low_10 else 0
        out['tf1h_bos_detected'] = max(out['tf1h_bos_bullish'], out['tf1h_bos_bearish'])
        out['tf1h_choch_detected'] = 0
        out['tf1h_fvg_present'] = self._detect_fvg_1h()
        out['fusion_smc'] = float(out['tf1h_bos_detected']) * 0.5
        out['smc_strength'] = 0.0
        out['smc_confidence'] = 0.0
        return out

    # ------------------------------------------------------------------
    # Private: BOMS / Liquidity Features
    # ------------------------------------------------------------------

    def _boms_liquidity_features(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        n = len(self._buf)
        close = self._buf['close'].values.astype(float)
        open_ = self._buf['open'].values.astype(float)
        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)
        vol = self._buf['volume'].values.astype(float)

        # Real BOMS detection from engine/structure/boms_detector.py
        if BOMS_AVAILABLE and self._buf is not None and n >= 50:
            # 4H BOMS displacement
            buf_4h = self._resample_to_tf(self._buf, '4H')
            if len(buf_4h) >= 30:
                boms_4h = detect_boms(buf_4h, timeframe='4H')
                out['tf4h_boms_displacement'] = float(boms_4h.displacement)
            else:
                out['tf4h_boms_displacement'] = 0.0

            # 1D BOMS strength (displacement / 2*ATR_14, capped at 1.0)
            buf_1d = self._resample_to_tf(self._buf, '1D')
            if len(buf_1d) >= 15:
                boms_1d = detect_boms(buf_1d, timeframe='1D')
                atr_1d = self._calc_atr(
                    buf_1d['high'].values.astype(float),
                    buf_1d['low'].values.astype(float),
                    buf_1d['close'].values.astype(float), 14)
                if atr_1d > 0 and boms_1d.displacement > 0:
                    out['tf1d_boms_strength'] = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
                else:
                    out['tf1d_boms_strength'] = 0.0
            else:
                out['tf1d_boms_strength'] = 0.0
        else:
            out['tf4h_boms_displacement'] = 0.0
            out['tf1d_boms_strength'] = 0.0

        # boms_strength (1h level, for archetype logic.py _check_G)
        out['boms_strength'] = out['tf1d_boms_strength']

        # volume_climax_last_3b: 1 if volume > 2x rolling 20-bar mean in last 3 bars
        if n >= 20:
            mean_20 = np.mean(vol[-20:])
            climax = 0
            check_bars = min(3, n)
            for i in range(1, check_bars + 1):
                if vol[-i] > 2.0 * mean_20:
                    climax = 1
                    break
            out['volume_climax_last_3b'] = climax
        else:
            out['volume_climax_last_3b'] = 0

        # tf1h_fvg_present -- 3-candle FVG detection
        out['tf1h_fvg_present'] = self._detect_fvg_1h()
        out['tf4h_fvg_present'] = self._detect_fvg_4h()

        # liquidity_score (combined proxy for archetype_instance._get_liquidity_score)
        boms_s = out['tf1d_boms_strength']
        fvg_val = 1.0 if out['tf4h_fvg_present'] else 0.0
        atr_14 = self._calc_atr(high, low, close, 14)
        disp_norm = min(out['tf4h_boms_displacement'] / (2.0 * atr_14), 1.0) if atr_14 > 0 else 0.0
        out['liquidity_score'] = (boms_s + fvg_val + disp_norm) / 3.0

        return out

    # ------------------------------------------------------------------
    # Private: Fusion Scores
    # ------------------------------------------------------------------

    def _fusion_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute simplified fusion scores.

        tf1h_fusion_score: weighted average of momentum, liquidity, structure (0-1)
        tf4h_fusion_score: same logic but 4H-equivalent lookback (periods x 4)
        """
        out: Dict[str, float] = {}

        # Momentum component
        rsi = features.get('rsi_14', 50.0)
        adx = features.get('adx', 20.0)
        rsi_momentum = abs(rsi - 50.0) / 50.0
        adx_strength = adx / 100.0
        momentum = (rsi_momentum + adx_strength) / 2.0

        # Liquidity component
        liquidity = features.get('liquidity_score', 0.5)

        # Structure component (BOS present)
        bos = float(features.get('tf1h_bos_detected', 0))

        # 1H fusion: 0.35 momentum + 0.35 liquidity + 0.30 structure
        tf1h = 0.35 * momentum + 0.35 * liquidity + 0.30 * bos
        out['tf1h_fusion_score'] = max(0.0, min(1.0, tf1h))

        # 4H fusion: use 4H BOS and slightly different weighting
        bos_4h = float(features.get('tf4h_bos_bullish', 0) or features.get('tf4h_bos_bearish', 0))
        tf4h = 0.35 * momentum + 0.35 * liquidity + 0.30 * bos_4h
        out['tf4h_fusion_score'] = max(0.0, min(1.0, tf4h))

        # tf1d_fusion_score (daily-level, use same components)
        out['tf1d_fusion_score'] = out['tf4h_fusion_score']

        # fusion_total (legacy feature name used by some components)
        out['fusion_total'] = out['tf1h_fusion_score']

        # Domain-level fusion scores for archetype_instance
        # Preserve real SMC engine score if available; fallback to simplified
        if 'fusion_smc' not in features or features.get('fusion_smc', 0) == 0:
            out['fusion_smc'] = bos * 0.5 + float(features.get('tf4h_fvg_present', 0)) * 0.5
        # else: keep the real engine's fusion_smc from _smc_features()
        out['fusion_wyckoff'] = features.get('wyckoff_score', 0.0) if 'wyckoff_score' in features else 0.0
        out['fusion_liquidity'] = liquidity
        out['fusion_momentum'] = momentum

        return out

    # ------------------------------------------------------------------
    # Private: Penalty Features
    # ------------------------------------------------------------------

    def _penalty_features(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        close = self._buf['close'].values.astype(float)
        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)
        n = len(close)

        # funding_Z -- z-score of funding rate
        if self._funding_history and len(self._funding_history) >= 10:
            arr = np.array(self._funding_history)
            mean_f = np.mean(arr)
            std_f = np.std(arr)
            out['funding_Z'] = (arr[-1] - mean_f) / (std_f + 1e-10)
            out['funding_rate'] = arr[-1]
        else:
            out['funding_Z'] = 0.0
            out['funding_rate'] = 0.0

        # bars_since_pivot -- bars since last swing high/low (window=50 centered)
        out['bars_since_pivot'] = self._bars_since_pivot(high, low, window=50)

        # fib_in_premium -- 1 if close > 0.618 retracement from recent swing low to high
        out['fib_in_premium'] = self._fib_in_premium(close, high, low)

        return out

    # ------------------------------------------------------------------
    # Private: PTI / Fakeout Features (REAL engine)
    # ------------------------------------------------------------------

    def _pti_fakeout_features(self) -> Dict[str, Any]:
        """
        Real PTI (Psychology Trap Index) and Fakeout detection.

        PTI uses 4 components: RSI divergence, volume exhaustion,
        wick traps, and failed breakouts. Weighted to detect trap setups.
        Falls back to defaults if engine not available.
        """
        out: Dict[str, Any] = {
            'tf1h_pti_score': 0.5,
            'tf1d_pti_score': 0.5,
            'tf1h_fakeout_detected': 0,
            'tf1h_fakeout_direction': 'none',
            'tf1h_fakeout_intensity': 0.0,
            'tf1h_pti_confidence': 0.0,
            'tf1h_pti_trap_type': 'none',
            'tf1h_pti_reversal_likely': 0,
            'tf1d_pti_reversal': 0,
        }

        if self._buf is None or len(self._buf) < 50:
            return out

        # Real PTI (1H)
        if PTI_AVAILABLE:
            try:
                pti_1h = calculate_pti(self._buf, timeframe='1H')
                out['tf1h_pti_score'] = float(pti_1h.pti_score)
                out['tf1h_pti_confidence'] = float(pti_1h.confidence)
                out['tf1h_pti_trap_type'] = str(pti_1h.trap_type)
                out['tf1h_pti_reversal_likely'] = 1.0 if pti_1h.reversal_likely else 0.0
            except Exception as e:
                logger.warning(f"PTI 1H failed: {e}")

        # Real PTI (1D from resampled buffer)
        if PTI_AVAILABLE:
            try:
                buf_1d = self._resample_to_tf(self._buf, '1D')
                if len(buf_1d) >= 20:
                    pti_1d = calculate_pti(buf_1d, timeframe='1D')
                    out['tf1d_pti_score'] = float(pti_1d.pti_score)
                    out['tf1d_pti_reversal'] = 1.0 if pti_1d.reversal_likely else 0.0
            except Exception as e:
                logger.warning(f"PTI 1D failed: {e}")

        # Real Fakeout detection
        if FAKEOUT_AVAILABLE:
            try:
                fakeout = detect_fakeout_intensity(self._buf, lookback=30)
                out['tf1h_fakeout_detected'] = 1.0 if fakeout.fakeout_detected else 0.0
                out['tf1h_fakeout_intensity'] = float(fakeout.intensity)
                out['tf1h_fakeout_direction'] = str(fakeout.direction)
            except Exception as e:
                logger.warning(f"Fakeout detection failed: {e}")

        return out

    # ------------------------------------------------------------------
    # Private: Wyckoff Features (REAL engine)
    # ------------------------------------------------------------------

    def _wyckoff_features(self) -> Dict[str, Any]:
        """
        Real Wyckoff event detection using engine.wyckoff.events.

        Detects: SC, BC, AR, AS, ST, SOS, SOW, Spring A/B, UT, UTAD, LPS, LPSY
        Produces: 26 event columns + composite scores + multi-TF scores.
        Falls back to EMA alignment proxy if engine not available.

        Uses hierarchical cross-timeframe detection: 1D -> 4H -> 1H.
        Higher-timeframe context modulates lower-timeframe confidence scores.
        """
        out: Dict[str, Any] = {}

        if not WYCKOFF_AVAILABLE or self._buf is None or len(self._buf) < 30:
            return self._wyckoff_features_fallback()

        try:
            # Step 1: Run hierarchical multi-TF detection (1D -> 4H)
            # This creates contexts and runs with HTF alignment
            mtf_out, htf_context_4h = self._wyckoff_multi_tf_hierarchical()
            out.update(mtf_out)

            # Step 2: Run 1H detection WITH 4H context for HTF alignment
            buf_copy = self._buf.copy()
            vol = buf_copy['volume'].values.astype(float)
            if len(vol) >= 20:
                vol_mean = pd.Series(vol).rolling(20).mean().values
                vol_std = pd.Series(vol).rolling(20).std().values
                buf_copy['volume_z'] = (vol - vol_mean) / (vol_std + 1e-10)
            # Recalibrated 1H thresholds (v2): aligned with feature store patcher
            _CFG_1H = {
                'st_lookback': 15,          # consensus 15-bar (was 30)
                'st_volume_z_max': 0.0,     # below 20-bar mean (was 0.5)
                'st_low_proximity': 0.03,   # 3% proximity (was 5%)
                'st_min_spacing': 10,       # 10-bar debounce
                'spring_b_breakdown_min': 0.002,
                'spring_b_breakdown_max': 0.015,
                'spring_b_recovery_bars': 3,
                'ut_volume_z_min': 0.3,     # relaxed for low-vol fakeouts
                'sm_st_max_count': 2,       # break ST self-loop
                'sm_spring_tolerance': 0.01,
                'sm_ut_tolerance': 0.01,
            }
            buf_copy = detect_all_wyckoff_events(buf_copy, cfg=_CFG_1H, htf_context=htf_context_4h)
            last = buf_copy.iloc[-1]

            # Extract all Wyckoff columns from the last row
            wyckoff_cols = [c for c in buf_copy.columns if c.startswith('wyckoff_')]
            for col in wyckoff_cols:
                val = last[col]
                out[col] = float(val) if isinstance(val, (int, float, np.integer, np.floating)) else val

            # Composite confidence: directional max of event confidences
            # BUG FIX: BC is a DISTRIBUTION event — must NOT be in accumulation list
            _BULLISH_EVENTS = ['sc', 'ar', 'st', 'spring_a', 'spring_b', 'sos', 'lps']
            _BEARISH_EVENTS = ['bc', 'as', 'sow', 'ut', 'utad', 'lpsy']
            bullish_confs = []
            bearish_confs = []
            for e in _BULLISH_EVENTS:
                conf_key = f'wyckoff_{e}_confidence'
                val = last.get(conf_key, 0)
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    bullish_confs.append(float(val))
            for e in _BEARISH_EVENTS:
                conf_key = f'wyckoff_{e}_confidence'
                val = last.get(conf_key, 0)
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    bearish_confs.append(float(val))
            out['wyckoff_bullish_event_confidence'] = max(bullish_confs) if bullish_confs else 0.0
            out['wyckoff_bearish_event_confidence'] = max(bearish_confs) if bearish_confs else 0.0
            # Non-directional composite (backward compat) — max of ALL events
            all_confs = bullish_confs + bearish_confs
            out['wyckoff_event_confidence'] = max(all_confs) if all_confs else 0.0
            out['wyckoff_score'] = out['wyckoff_event_confidence']

            # Event history + conviction
            self.last_wyckoff_event_history = self._wyckoff_event_history(buf_copy, max_events=20)
            self.last_wyckoff_conviction = self._wyckoff_conviction_breakdown(last)

        except Exception as e:
            logger.warning(f"Wyckoff 1H engine failed, using fallback: {e}")
            out = self._wyckoff_features_fallback()

        return out

    def _wyckoff_multi_tf_hierarchical(self):
        """Run hierarchical Wyckoff: 1D -> 4H, creating cross-TF context.

        Returns:
            Tuple of (out_dict, htf_context_4h) where htf_context_4h is a
            WyckoffHTFContext to pass to the 1H detection, or None if unavailable.
        """
        ALL_EVENTS = ['sc', 'bc', 'ar', 'as', 'st', 'sos', 'sow', 'spring_a', 'spring_b', 'lps', 'lpsy', 'ut', 'utad']
        out: Dict[str, Any] = {
            'tf4h_wyckoff_phase_score': 0.0,
            'tf4h_wyckoff_bullish_score': 0.0,
            'tf4h_wyckoff_bearish_score': 0.0,
            'tf1d_wyckoff_m1_signal': 0,
            'tf1d_wyckoff_m2_signal': 0,
            'tf1d_wyckoff_score': 0.0,
            'tf1d_wyckoff_bullish_score': 0.0,
            'tf1d_wyckoff_bearish_score': 0.0,
            'tf1d_daily_bars': 0,
        }
        htf_context_4h = None  # Will be returned for 1H to use

        if not WYCKOFF_AVAILABLE or self._buf is None:
            return out, htf_context_4h

        # Timeframe-adapted thresholds: HTF bars smooth out intraday spikes,
        # so volume_z and range_z thresholds must be softer for 4H and 1D.
        # Without this, SC/BC (volume_z > 2.5) almost never fires on daily bars.
        _CFG_4H = {
            'sc_volume_z_min': 1.8,     # 2.5 on 1H — 4H aggregates damp volume spikes
            'sc_range_z_min': 1.2,      # 1.5 on 1H
            'bc_volume_z_min': 1.8,
            'bc_range_z_min': 1.2,
            'sos_volume_z_min': 1.2,    # 1.5 on 1H
            'sow_volume_z_min': 1.2,
            'spring_a_volume_z_min': 0.3,  # 0.5 on 1H
            'ut_volume_z_min': 0.2,     # 0.3 on 1H
            # Recalibrated v2 params (scaled for 4H)
            'st_lookback': 8,           # 15 on 1H, ~2 days on 4H
            'st_volume_z_max': 0.0,
            'st_low_proximity': 0.04,   # slightly wider on 4H
            'st_min_spacing': 3,        # 10 on 1H / 4 = ~3
            'spring_b_breakdown_min': 0.002,
            'spring_b_breakdown_max': 0.02,  # slightly wider on 4H
            'spring_b_recovery_bars': 2,
            'sm_st_max_count': 2,
            'sm_spring_tolerance': 0.015,
            'sm_ut_tolerance': 0.015,
        }
        _CFG_1D = {
            'sc_volume_z_min': 1.5,     # Daily bars average 24 hourly candles
            'sc_range_z_min': 1.0,
            'bc_volume_z_min': 1.5,
            'bc_range_z_min': 1.0,
            'sos_volume_z_min': 1.0,
            'sow_volume_z_min': 1.0,
            'spring_a_volume_z_min': 0.2,
            'ut_volume_z_min': 0.15,
            'sc_range_lookback': 30,    # ~1 month on daily (vs 50 on 1H)
            'bc_range_lookback': 30,
            # Recalibrated v2 params (scaled for 1D)
            'st_lookback': 5,           # 15 on 1H → ~5 days
            'st_volume_z_max': 0.0,
            'st_low_proximity': 0.05,   # wider on daily
            'st_min_spacing': 2,        # ~2 daily bars
            'spring_b_breakdown_min': 0.003,
            'spring_b_breakdown_max': 0.025,
            'spring_b_recovery_bars': 2,
            'sm_st_max_count': 2,
            'sm_spring_tolerance': 0.02,
            'sm_ut_tolerance': 0.02,
        }

        try:
            # ---- Step 1: 1D Wyckoff (independent, no HTF context) ----
            ctx_1d = None
            buf_1d = self._resample_to_tf(self._buf, '1D')
            logger.info(f"Wyckoff 1D resample: {len(buf_1d)} daily bars from {len(self._buf)} 1H bars")
            out['tf1d_daily_bars'] = len(buf_1d)
            if len(buf_1d) >= 20:
                buf_1d_copy = buf_1d.copy()
                buf_1d_copy = detect_all_wyckoff_events(buf_1d_copy, cfg=_CFG_1D)
                ctx_1d = create_wyckoff_context(buf_1d_copy, lookback=3, timeframe="1D")

                tail_1d = buf_1d_copy.iloc[-3:] if len(buf_1d_copy) >= 3 else buf_1d_copy

                # Graded bullish/bearish scores (replace binary M1/M2)
                out['tf1d_wyckoff_bullish_score'] = ctx_1d.bullish_score
                out['tf1d_wyckoff_bearish_score'] = ctx_1d.bearish_score

                # Keep M1/M2 for backward compat but now derived from context
                out['tf1d_wyckoff_m1_signal'] = 1 if ctx_1d.bullish_score > 0.1 else 0
                out['tf1d_wyckoff_m2_signal'] = 1 if ctx_1d.bearish_score > 0.1 else 0

                # Score: max confidence across all events (backward compat)
                confs_1d = []
                for e in ALL_EVENTS:
                    col = f'wyckoff_{e}_confidence'
                    if col in tail_1d.columns:
                        max_conf = float(tail_1d[col].max()) if not tail_1d[col].isna().all() else 0.0
                        confs_1d.append(max_conf)
                out['tf1d_wyckoff_score'] = max(confs_1d) if confs_1d else 0.0

                logger.info(f"Wyckoff 1D results: bullish={out['tf1d_wyckoff_bullish_score']:.3f}, "
                            f"bearish={out['tf1d_wyckoff_bearish_score']:.3f}, "
                            f"score={out['tf1d_wyckoff_score']:.3f}")

            # ---- Step 2: 4H Wyckoff (with 1D context) ----
            buf_4h = self._resample_to_tf(self._buf, '4H')
            if len(buf_4h) >= 30:
                buf_4h_copy = buf_4h.copy()
                buf_4h_copy = detect_all_wyckoff_events(buf_4h_copy, cfg=_CFG_4H, htf_context=ctx_1d)
                htf_context_4h = create_wyckoff_context(buf_4h_copy, lookback=3, timeframe="4H")

                tail_4h = buf_4h_copy.iloc[-3:] if len(buf_4h_copy) >= 3 else buf_4h_copy

                # 4H scores
                out['tf4h_wyckoff_bullish_score'] = htf_context_4h.bullish_score
                out['tf4h_wyckoff_bearish_score'] = htf_context_4h.bearish_score

                confs_4h = []
                for e in ALL_EVENTS:
                    col = f'wyckoff_{e}_confidence'
                    if col in tail_4h.columns:
                        max_conf = float(tail_4h[col].max()) if not tail_4h[col].isna().all() else 0.0
                        confs_4h.append(max_conf)
                out['tf4h_wyckoff_phase_score'] = max(confs_4h) if confs_4h else 0.0

                logger.info(f"Wyckoff 4H results: bullish={out['tf4h_wyckoff_bullish_score']:.3f}, "
                            f"bearish={out['tf4h_wyckoff_bearish_score']:.3f}, "
                            f"score={out['tf4h_wyckoff_phase_score']:.3f}")

        except Exception as e:
            logger.warning(f"Wyckoff multi-TF hierarchical failed: {e}")

        return out, htf_context_4h

    def _wyckoff_multi_tf_legacy(self) -> Dict[str, Any]:
        """LEGACY: Run Wyckoff on 4H and 1D resampled buffers independently (no cross-TF context).

        Kept for safety/rollback. Replaced by _wyckoff_multi_tf_hierarchical().
        """
        ALL_EVENTS = ['sc', 'bc', 'ar', 'as', 'st', 'sos', 'sow', 'spring_a', 'spring_b', 'lps', 'lpsy', 'ut', 'utad']
        out: Dict[str, Any] = {
            'tf4h_wyckoff_phase_score': 0.0,
            'tf1d_wyckoff_m1_signal': 0,
            'tf1d_wyckoff_m2_signal': 0,
            'tf1d_wyckoff_score': 0.0,
            'tf1d_daily_bars': 0,
        }

        if not WYCKOFF_AVAILABLE or self._buf is None:
            return out

        try:
            # 4H Wyckoff — use last 3 bars, all 13 events
            buf_4h = self._resample_to_tf(self._buf, '4H')
            if len(buf_4h) >= 30:
                buf_4h_copy = buf_4h.copy()
                buf_4h_copy = detect_all_wyckoff_events(buf_4h_copy)
                # Use last 3 4H bars, max confidence across all 13 events
                tail_4h = buf_4h_copy.iloc[-3:] if len(buf_4h_copy) >= 3 else buf_4h_copy
                confs_4h = []
                for e in ALL_EVENTS:
                    col = f'wyckoff_{e}_confidence'
                    if col in tail_4h.columns:
                        max_conf = float(tail_4h[col].max()) if not tail_4h[col].isna().all() else 0.0
                        confs_4h.append(max_conf)
                out['tf4h_wyckoff_phase_score'] = max(confs_4h) if confs_4h else 0.0

            # 1D Wyckoff — use last 3 daily bars, all 13 events
            buf_1d = self._resample_to_tf(self._buf, '1D')
            logger.info(f"Wyckoff 1D resample: {len(buf_1d)} daily bars from {len(self._buf)} 1H bars")
            out['tf1d_daily_bars'] = len(buf_1d)
            if len(buf_1d) >= 20:
                buf_1d_copy = buf_1d.copy()
                buf_1d_copy = detect_all_wyckoff_events(buf_1d_copy)
                # Use last 3 daily bars for signal and score detection
                tail_1d = buf_1d_copy.iloc[-3:] if len(buf_1d_copy) >= 3 else buf_1d_copy
                # M1 signal: accumulation events — check if any of last 3 daily bars had the event
                m1_events = ['sc', 'ar', 'st', 'spring_a', 'spring_b']
                out['tf1d_wyckoff_m1_signal'] = 1 if any(
                    tail_1d[f'wyckoff_{e}'].any() for e in m1_events if f'wyckoff_{e}' in tail_1d.columns
                ) else 0
                # M2 signal: distribution events — check if any of last 3 daily bars had the event
                m2_events = ['ut', 'utad', 'lpsy', 'sow']
                out['tf1d_wyckoff_m2_signal'] = 1 if any(
                    tail_1d[f'wyckoff_{e}'].any() for e in m2_events if f'wyckoff_{e}' in tail_1d.columns
                ) else 0
                # Score: max confidence across last 3 daily bars across all 13 events
                confs_1d = []
                for e in ALL_EVENTS:
                    col = f'wyckoff_{e}_confidence'
                    if col in tail_1d.columns:
                        max_conf = float(tail_1d[col].max()) if not tail_1d[col].isna().all() else 0.0
                        confs_1d.append(max_conf)
                out['tf1d_wyckoff_score'] = max(confs_1d) if confs_1d else 0.0
                logger.info(f"Wyckoff 1D results: m1={out['tf1d_wyckoff_m1_signal']}, m2={out['tf1d_wyckoff_m2_signal']}, score={out['tf1d_wyckoff_score']:.3f}, daily_bars={len(buf_1d)}")

        except Exception as e:
            logger.warning(f"Wyckoff multi-TF failed: {e}")

        return out

    def _wyckoff_features_fallback(self) -> Dict[str, float]:
        """EMA alignment fallback when Wyckoff engine is not available."""
        close = self._buf['close'].values.astype(float)
        ema9 = self._ema(close, 9)
        ema21 = self._ema(close, 21)
        ema50 = self._ema(close, 50)
        ema200 = self._ema(close, 200)

        alignment = 0.0
        if ema9 > ema21: alignment += 0.333
        if ema21 > ema50: alignment += 0.333
        if ema50 > ema200: alignment += 0.334

        return {
            'wyckoff_score': alignment,
            'wyckoff_event_confidence': 0.0,
            'wyckoff_bullish_score': 0.0,
            'wyckoff_bearish_score': 0.0,
            'tf1d_wyckoff_score': alignment,
            'tf1d_wyckoff_bullish_score': 0.0,
            'tf1d_wyckoff_bearish_score': 0.0,
            'tf4h_wyckoff_phase_score': alignment,
            'tf4h_wyckoff_bullish_score': 0.0,
            'tf4h_wyckoff_bearish_score': 0.0,
            'tf1d_wyckoff_m1_signal': 0,
            'tf1d_wyckoff_m2_signal': 0,
        }

    def _wyckoff_event_history(self, buf_with_events: pd.DataFrame, max_events: int = 20) -> list:
        """Scan all bars in buffer for recent Wyckoff events. Returns list sorted newest-first."""
        EVENT_KEYS = ['sc', 'bc', 'ar', 'as', 'st', 'sos', 'sow', 'spring_a', 'spring_b', 'lps', 'lpsy', 'ut', 'utad']
        history: list = []
        for event_key in EVENT_KEYS:
            col = f'wyckoff_{event_key}'
            conf_col = f'wyckoff_{event_key}_confidence'
            if col not in buf_with_events.columns:
                continue
            mask = buf_with_events[col].astype(bool)
            event_bars = buf_with_events[mask]
            for idx, row in event_bars.iterrows():
                confidence = float(row.get(conf_col, 0) or 0)
                if confidence <= 0:
                    continue
                history.append({
                    'event': event_key,
                    'timestamp': str(idx),
                    'price': round(float(row['close']), 2),
                    'high': round(float(row['high']), 2),
                    'low': round(float(row['low']), 2),
                    'confidence': round(confidence, 3),
                    'volume_z': round(float(row.get('volume_z', 0) or 0), 2),
                })
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history[:max_events]

    def _wyckoff_conviction_breakdown(self, last_row) -> dict:
        """Break down Wyckoff conviction into per-event contributions."""
        EVENT_KEYS = ['sc', 'bc', 'ar', 'as', 'st', 'sos', 'sow', 'spring_a', 'spring_b', 'lps', 'lpsy', 'ut', 'utad']
        components: list = []
        for ek in EVENT_KEYS:
            conf = float(last_row.get(f'wyckoff_{ek}_confidence', 0) or 0)
            active = bool(last_row.get(f'wyckoff_{ek}', False))
            if active and conf > 0:
                components.append({'event': ek, 'confidence': round(conf, 3), 'weight': 1.0, 'contribution': round(conf, 3)})
        components.sort(key=lambda x: x['contribution'], reverse=True)
        total = max(c['contribution'] for c in components) if components else 0
        if not components:
            reason = "No Wyckoff events active on this bar."
        elif total >= 0.8:
            top = components[0]
            reason = f"High conviction driven by {top['event'].upper()} ({top['confidence']*100:.0f}% conf). {len(components)} event(s) reinforcing."
        elif total >= 0.6:
            top = components[0]
            reason = f"Moderate conviction. Strongest signal: {top['event'].upper()} ({top['confidence']*100:.0f}% conf)."
        else:
            reason = f"Low conviction. {len(components)} weak signal(s)."
        return {'total_score': round(total, 3), 'components': components, 'reason': reason}

    # ------------------------------------------------------------------
    # Private: Temporal Confluence Features (REAL engine)
    # ------------------------------------------------------------------

    def _temporal_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real temporal confluence using engine.temporal.temporal_confluence.

        Computes: Gann time clusters, Fibonacci time clusters, TPI signal,
        cycle phase, temporal reversal zones, and overall confluence score.
        Falls back to defaults if engine not available.
        """
        out: Dict[str, Any] = {
            'temporal_confluence_score': 0.5,
            'gann_time_cluster': 0,
            'fib_time_cluster': 0,
            'tpi_signal': 0.5,
            'cycle_phase': 0.5,
            'temporal_reversal_zone': 0,
        }

        if not TEMPORAL_AVAILABLE or self._temporal_engine is None:
            return out
        if self._buf is None or len(self._buf) < 55:
            return out

        try:
            # Single-bar API for overall score
            row = {
                'close': features.get('close', 0),
                'high': features.get('high', 0),
                'low': features.get('low', 0),
                'volume': features.get('volume', 0),
                'atr_14': features.get('atr_14', 0),
                'rsi_14': features.get('rsi_14', 50),
                'bb_width': features.get('bb_width', 0),
            }
            out['temporal_confluence_score'] = float(
                self._temporal_engine.get_confluence_score(row)
            )

            # Batch API for full 6-column output
            buf_copy = self._buf.copy()
            buf_copy = self._temporal_engine.compute_temporal_features(buf_copy)
            last = buf_copy.iloc[-1]

            out['gann_time_cluster'] = int(last.get('gann_time_cluster', 0))
            out['fib_time_cluster'] = int(last.get('fib_time_cluster', 0))
            out['tpi_signal'] = float(last.get('tpi_signal', 0.5))
            out['cycle_phase'] = float(last.get('cycle_phase', 0.5))
            out['temporal_reversal_zone'] = int(last.get('temporal_reversal_zone', 0))

        except Exception as e:
            logger.warning(f"Temporal confluence failed: {e}")

        return out

    # ------------------------------------------------------------------
    # Private: FRVP (Fixed Range Volume Profile) Features (REAL engine)
    # ------------------------------------------------------------------

    def _frvp_features(self) -> Dict[str, float]:
        """
        Real volume profile using engine.volume.frvp.

        Computes: POC, Value Area High/Low, current position relative to
        value area, distance to POC/VA edges.
        Falls back to defaults if engine not available.
        """
        out: Dict[str, float] = {
            'tf1h_frvp_poc': 0.0,
            'tf1h_frvp_va_high': 0.0,
            'tf1h_frvp_va_low': 0.0,
            'tf1h_frvp_position': 0.0,  # 0=in_va, 1=above, -1=below
            'tf1h_frvp_distance_to_poc': 0.0,
        }

        if not FRVP_AVAILABLE or self._buf is None or len(self._buf) < 20:
            return out

        try:
            frvp = calculate_frvp(self._buf, lookback=100, price_bins=50)
            out['tf1h_frvp_poc'] = float(frvp.poc)
            out['tf1h_frvp_va_high'] = float(frvp.va_high)
            out['tf1h_frvp_va_low'] = float(frvp.va_low)

            # Position encoding: above_va=1, in_va=0, below_va=-1
            pos_str = getattr(frvp, 'current_position', 'in_va')
            pos_map = {'above_va': 1.0, 'in_va': 0.0, 'below_va': -1.0}
            out['tf1h_frvp_position'] = pos_map.get(pos_str, 0.0)

            # Distance to POC
            frvp_dict = frvp.to_dict()
            out['tf1h_frvp_distance_to_poc'] = float(
                frvp_dict.get('frvp_distance_to_poc', 0)
            )

        except Exception as e:
            logger.warning(f"FRVP failed: {e}")

        return out

    # ------------------------------------------------------------------
    # Private: 4H Structure Features (Squiggle + BOMS) (REAL engine)
    # ------------------------------------------------------------------

    def _structure_4h_features(self) -> Dict[str, Any]:
        """
        Real structure detection using engine.structure modules.

        Squiggle 1-2-3: BOS → retest → continuation pattern (4H).
        BOMS: Break of Market Structure with volume confirmation (4H + 1D).
        Falls back to defaults if engines not available.
        """
        out: Dict[str, Any] = {
            'tf4h_squiggle_stage': 0,
            'tf4h_squiggle_confidence': 0.0,
            'tf4h_squiggle_direction': 0,
            'tf4h_squiggle_entry_window': 0,
            'tf4h_boms_displacement': 0.0,
            'tf4h_boms_direction': 0,
            'tf1d_boms_strength': 0.0,
            'tf1d_boms_direction': 0,
        }

        if self._buf is None or len(self._buf) < 50:
            return out

        buf_4h = self._resample_to_tf(self._buf, '4H')

        # Squiggle pattern detection (4H)
        if SQUIGGLE_AVAILABLE and len(buf_4h) >= 40:
            try:
                squiggle = detect_squiggle_123(buf_4h, timeframe='4H')
                out['tf4h_squiggle_stage'] = int(squiggle.stage)
                out['tf4h_squiggle_confidence'] = float(squiggle.confidence)
                # Direction encoding: bullish=1, bearish=-1, none=0
                dir_map = {'bullish': 1, 'bearish': -1, 'none': 0}
                out['tf4h_squiggle_direction'] = dir_map.get(squiggle.direction, 0)
                out['tf4h_squiggle_entry_window'] = 1 if squiggle.entry_window else 0
            except Exception as e:
                logger.warning(f"Squiggle 4H failed: {e}")

        # BOMS detection (4H)
        if BOMS_AVAILABLE and len(buf_4h) >= 30:
            try:
                boms_4h = detect_boms(buf_4h, timeframe='4H')
                out['tf4h_boms_displacement'] = float(boms_4h.displacement)
                dir_map = {'bullish': 1, 'bearish': -1, 'none': 0}
                out['tf4h_boms_direction'] = dir_map.get(boms_4h.direction, 0)
            except Exception as e:
                logger.warning(f"BOMS 4H failed: {e}")

        # BOMS detection (1D)
        if BOMS_AVAILABLE:
            try:
                buf_1d = self._resample_to_tf(self._buf, '1D')
                if len(buf_1d) >= 15:
                    boms_1d = detect_boms(buf_1d, timeframe='1D')
                    out['tf1d_boms_strength'] = float(boms_1d.displacement)
                    dir_map = {'bullish': 1, 'bearish': -1, 'none': 0}
                    out['tf1d_boms_direction'] = dir_map.get(boms_1d.direction, 0)
            except Exception as e:
                logger.warning(f"BOMS 1D failed: {e}")

        return out

    # ------------------------------------------------------------------
    # Private: Fibonacci Features (inline from bin/add_fib_features.py)
    # ------------------------------------------------------------------

    def _fib_features(self) -> Dict[str, Any]:
        """
        Fibonacci retracement and time zone analysis.

        Detects: nearest Fib level, OTE zone, discount/premium zones,
        Fib time zones, time-price confluence score.
        Uses centered rolling window with forward shift to avoid lookahead.
        """
        FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
        FIB_TIME_NUMBERS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        FIB_TIME_TOLERANCE = 2

        defaults = {
            'fib_retracement_level': 0.5,
            'fib_distance_pct': 0.0,
            'fib_in_ote_zone': 0,
            'fib_in_discount': 0,
            'fib_in_premium': 0,
            'fib_time_zone': 0,
            'fib_time_confluence': 0.0,
            'fib_swing_range_pct': 0.0,
        }

        if self._buf is None or len(self._buf) < 50:
            return defaults

        try:
            buf = self._buf
            highs = buf['high'].values.astype(float)
            lows = buf['low'].values.astype(float)
            close_arr = buf['close'].values.astype(float)
            n = len(buf)

            # Swing detection: centered rolling window, shifted forward to avoid lookahead
            lookback = 20
            window = 2 * lookback + 1
            if n < window:
                return defaults

            # Find swing highs and lows using rolling max/min
            high_series = pd.Series(highs, index=buf.index)
            low_series = pd.Series(lows, index=buf.index)

            swing_high_mask = (high_series == high_series.rolling(
                window, center=True).max()).shift(lookback).fillna(False)
            swing_low_mask = (low_series == low_series.rolling(
                window, center=True).min()).shift(lookback).fillna(False)

            sh_idx = swing_high_mask[swing_high_mask].index
            sl_idx = swing_low_mask[swing_low_mask].index

            if len(sh_idx) == 0 or len(sl_idx) == 0:
                return defaults

            # Last confirmed swing high and low
            last_sh = float(high_series.loc[sh_idx[-1]])
            last_sl = float(low_series.loc[sl_idx[-1]])
            current = float(close_arr[-1])
            swing_range = last_sh - last_sl

            if swing_range <= 0:
                return defaults

            # Retracement: 0 = at swing high, 1 = at swing low
            retracement = max(0.0, min(1.0, (last_sh - current) / swing_range))

            # Nearest Fibonacci level
            nearest_fib = min(FIB_LEVELS, key=lambda f: abs(retracement - f))
            distance = abs(retracement - nearest_fib)

            # Time analysis: bars since last swing point
            last_swing_time = max(sh_idx[-1], sl_idx[-1])
            last_swing_pos = buf.index.get_loc(last_swing_time)
            bars_since = n - 1 - last_swing_pos

            # Is bars_since near a Fibonacci number?
            min_time_diff = min(abs(bars_since - f) for f in FIB_TIME_NUMBERS)
            time_zone = min_time_diff <= FIB_TIME_TOLERANCE

            # Time-price confluence: geometric mean of price proximity and time proximity
            price_score = float(np.exp(-distance * 10.0))
            time_score = float(np.exp(-min_time_diff * 0.5))
            fib_time_confluence = float(np.sqrt(price_score * time_score))

            return {
                'fib_retracement_level': nearest_fib,
                'fib_distance_pct': distance,
                'fib_in_ote_zone': 1 if 0.618 <= retracement <= 0.786 else 0,
                'fib_in_discount': 1 if retracement > 0.5 else 0,
                'fib_in_premium': 1 if retracement < 0.5 else 0,
                'fib_time_zone': 1 if time_zone else 0,
                'fib_time_confluence': fib_time_confluence,
                'fib_swing_range_pct': swing_range / current if current > 0 else 0.0,
            }

        except Exception as e:
            logger.warning(f"Fibonacci features failed: {e}")
            return defaults

    # ------------------------------------------------------------------
    # Private: Coil / Squeeze Features (inline from bin/add_coil_features.py)
    # ------------------------------------------------------------------

    def _coil_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Volatility compression detection across multiple timeframes.

        Two-component score: ATR ratio (vol contraction) + BB squeeze (band narrowing).
        Higher score = tighter compression = potential breakout setup.
        """
        out: Dict[str, Any] = {}

        if self._buf is None or len(self._buf) < 50:
            return {
                'tf1h_coil_score': 0.0, 'tf1h_is_coiling': 0,
                'tf1h_coil_atr_ratio': 1.0, 'tf1h_coil_bb_squeeze': 1.0,
                'tf1h_coil_breakout': 0,
                'tf4h_coil_score': 0.0, 'tf4h_is_coiling': 0,
                'tf4h_coil_atr_ratio': 1.0, 'tf4h_coil_bb_squeeze': 1.0,
                'tf4h_coil_breakout': 0,
                'tf1d_coil_score': 0.0, 'tf1d_is_coiling': 0,
                'tf1d_coil_atr_ratio': 1.0, 'tf1d_coil_bb_squeeze': 1.0,
                'tf1d_coil_breakout': 0,
            }

        # 1H coil features
        out.update(self._compute_coil_tf(self._buf, 'tf1h', atr_period=14, lookback=20))

        # 4H coil features
        buf_4h = self._resample_to_tf(self._buf, '4H')
        if len(buf_4h) >= 30:
            out.update(self._compute_coil_tf(buf_4h, 'tf4h', atr_period=14, lookback=14))
        else:
            out.update({
                'tf4h_coil_score': 0.0, 'tf4h_is_coiling': 0,
                'tf4h_coil_atr_ratio': 1.0, 'tf4h_coil_bb_squeeze': 1.0,
                'tf4h_coil_breakout': 0,
            })

        # 1D coil features
        buf_1d = self._resample_to_tf(self._buf, '1D')
        if len(buf_1d) >= 15:
            out.update(self._compute_coil_tf(buf_1d, 'tf1d', atr_period=14, lookback=10))
        else:
            out.update({
                'tf1d_coil_score': 0.0, 'tf1d_is_coiling': 0,
                'tf1d_coil_atr_ratio': 1.0, 'tf1d_coil_bb_squeeze': 1.0,
                'tf1d_coil_breakout': 0,
            })

        return out

    # ------------------------------------------------------------------
    # Private: Binance Futures institutional flow features
    # ------------------------------------------------------------------

    def _binance_futures_features(self) -> Dict[str, Any]:
        """
        Fetch institutional flow features from derivatives APIs.
        Fallback chain: Binance → OKX (free) → CoinGlass (paid).

        Returns 8 features:
          oi_value, oi_change_4h, oi_change_24h, oi_price_divergence,
          binance_funding_rate, funding_oi_divergence,
          ls_ratio_extreme, taker_imbalance
        """
        defaults = {
            'oi_value': 0.0,
            'oi_change_4h': 0.0,
            'oi_change_24h': 0.0,
            'oi_price_divergence': 0,
            'binance_funding_rate': 0.0,
            'funding_oi_divergence': 0,
            'ls_ratio_extreme': 0.0,
            'taker_imbalance': 0.0,
        }

        if not self.binance_api and not self._okx_api and not self._coinglass_api:
            return defaults

        now = time.time()
        if now - self._binance_last_fetch < self._binance_fetch_interval and self._binance_cache:
            return self._binance_cache

        try:
            data = None

            # Try Binance first (unless previously geo-blocked)
            if self.binance_api and not self._binance_geo_blocked:
                data = self.binance_api.fetch_all_current()
                # Check if all values are 0 (geo-blocked returns 0s)
                if data and all(v == 0.0 for v in data.values()):
                    logger.info("Binance returned all zeros (likely geo-blocked), trying OKX...")
                    self._binance_geo_blocked = True
                    data = None

            # Fallback to OKX (free, no auth, US-accessible)
            if data is None and self._okx_api:
                data = self._okx_api.fetch_all_current()
                if data and any(v != 0.0 for v in data.values()):
                    self._derivatives_source = "okx"

            # Fallback to CoinGlass (paid, requires API key)
            if data is None and self._coinglass_api:
                data = self._coinglass_api.fetch_all_current()
                if data and any(v != 0.0 for v in data.values()):
                    self._derivatives_source = "coinglass"

            if not data:
                return defaults

            result = {
                'oi_value': data.get('oi_value', 0.0),
                'oi_change_4h': data.get('oi_change_4h', 0.0),
                'oi_change_24h': data.get('oi_change_24h', 0.0),
                'oi_price_divergence': 0,  # Computed below
                'binance_funding_rate': data.get('funding_rate', 0.0),
                'funding_oi_divergence': 0,  # Computed below
                'ls_ratio_extreme': 0.0,  # Computed below
                'taker_imbalance': 0.0,
            }

            # oi_price_divergence: check if price direction disagrees with OI direction
            oi_chg = result['oi_change_4h']
            # Price change would come from the OHLCV data already available
            # We'll set this to 0 here and let the caller compute with price context

            # funding_oi_divergence
            fr = result['binance_funding_rate']
            if fr > 0 and oi_chg < -0.01:
                result['funding_oi_divergence'] = -1  # bearish: longs paying but closing
            elif fr < 0 and oi_chg > 0.01:
                result['funding_oi_divergence'] = 1   # bullish: shorts paying but OI rising

            # ls_ratio_extreme (Z-score approximation from current value)
            ls = data.get('ls_ratio', 1.0)
            # Historical mean L/S ratio for BTC is ~1.0-1.2
            # Z-score approximation: (value - 1.1) / 0.3
            result['ls_ratio_extreme'] = (ls - 1.1) / 0.3 if ls else 0.0

            # taker_imbalance
            taker = data.get('taker_buy_sell_ratio', 1.0)
            # Convert ratio to imbalance: ratio=1.0 -> 0, ratio=1.5 -> positive, ratio=0.5 -> negative
            result['taker_imbalance'] = (taker - 1.0) / max(taker, 0.01) if taker else 0.0

            self._binance_cache = result
            self._binance_last_fetch = now
            return result

        except Exception as e:
            logger.warning(f"Derivatives features failed ({self._derivatives_source}): {e}")
            return defaults

    def _compute_coil_tf(self, buf: pd.DataFrame, prefix: str,
                         atr_period: int = 14, lookback: int = 20) -> Dict[str, Any]:
        """Compute coil/squeeze features for a single timeframe buffer."""
        close = buf['close'].values.astype(float)
        high = buf['high'].values.astype(float)
        low = buf['low'].values.astype(float)
        n = len(close)

        if n < atr_period + lookback:
            return {
                f'{prefix}_coil_score': 0.0, f'{prefix}_is_coiling': 0,
                f'{prefix}_coil_atr_ratio': 1.0, f'{prefix}_coil_bb_squeeze': 1.0,
                f'{prefix}_coil_breakout': 0,
            }

        # ATR series
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
        )
        atr_series = pd.Series(tr).rolling(atr_period).mean().values
        current_atr = atr_series[-1] if not np.isnan(atr_series[-1]) else 0
        atr_ma = np.nanmean(atr_series[-lookback:]) if lookback <= len(atr_series) else np.nanmean(atr_series)
        atr_ratio = current_atr / atr_ma if atr_ma > 0 else 1.0

        # BB width series
        sma_20 = pd.Series(close).rolling(20).mean().values
        std_20 = pd.Series(close).rolling(20).std().values
        bb_upper = sma_20 + 2.0 * std_20
        bb_lower = sma_20 - 2.0 * std_20
        bb_width = (bb_upper - bb_lower) / (sma_20 + 1e-10)

        current_bb = bb_width[-1] if not np.isnan(bb_width[-1]) else 0
        bb_ma = np.nanmean(bb_width[-lookback:]) if lookback <= len(bb_width) else np.nanmean(bb_width)
        bb_squeeze = current_bb / bb_ma if bb_ma > 0 else 1.0

        # Coil score: mean of ATR compression + BB compression
        atr_comp = 1.0 - min(atr_ratio, 2.0) / 2.0
        bb_comp = 1.0 - min(bb_squeeze, 2.0) / 2.0
        coil_score = max(0.0, min(1.0, (atr_comp + bb_comp) / 2.0))

        is_coiling = 1 if (atr_ratio < 0.85 and bb_squeeze < 0.85) else 0

        # Breakout: was coiling but no longer
        if n >= 2:
            prev_atr = atr_series[-2] if not np.isnan(atr_series[-2]) else current_atr
            prev_bb = bb_width[-2] if not np.isnan(bb_width[-2]) else current_bb
            prev_atr_ratio = prev_atr / atr_ma if atr_ma > 0 else 1.0
            prev_bb_squeeze = prev_bb / bb_ma if bb_ma > 0 else 1.0
            was_coiling = prev_atr_ratio < 0.85 and prev_bb_squeeze < 0.85
            breakout = 1 if (was_coiling and not is_coiling) else 0
        else:
            breakout = 0

        return {
            f'{prefix}_coil_score': coil_score,
            f'{prefix}_is_coiling': is_coiling,
            f'{prefix}_coil_atr_ratio': atr_ratio,
            f'{prefix}_coil_bb_squeeze': bb_squeeze,
            f'{prefix}_coil_breakout': breakout,
        }

    # ------------------------------------------------------------------
    # Private: Crisis Features (rolling returns / drawdown)
    # ------------------------------------------------------------------

    def _crisis_features(self) -> Dict[str, float]:
        """
        Crisis indicators: crash frequency, drawdown persistence, max drawdown.

        These feed into the regime-aware penalty system and help identify
        periods where position sizing should be reduced.
        """
        out: Dict[str, float] = {
            'crash_frequency_7d': 0.0,
            'drawdown_max_60d': 0.0,
            'drawdown_persistence': 0.0,
            'crisis_persistence': 0.0,
            'aftershock_score': 0.0,
        }

        if self._buf is None or len(self._buf) < 30:
            return out

        try:
            close = self._buf['close'].values.astype(float)
            n = len(close)
            returns = np.diff(close) / (close[:-1] + 1e-10)

            # 7-day crash frequency: count of >5% drops in last 168 bars
            lookback_7d = min(168, len(returns))
            if lookback_7d > 0:
                negative_5pct = int(np.sum(returns[-lookback_7d:] < -0.05))
                out['crash_frequency_7d'] = float(negative_5pct)

            # Max drawdown over 60 days (1440 bars) or available history
            lookback_60d = min(1440, n)
            peak = np.maximum.accumulate(close[-lookback_60d:])
            dd = (close[-lookback_60d:] - peak) / (peak + 1e-10)
            max_dd = float(np.min(dd))
            out['drawdown_max_60d'] = max_dd

            # Drawdown persistence: fraction of bars in >5% drawdown
            dd_pct_negative = float(np.mean(dd < -0.05)) if len(dd) > 0 else 0.0
            out['drawdown_persistence'] = dd_pct_negative

            # Crisis persistence: 1 if max drawdown exceeds 20%
            out['crisis_persistence'] = 1.0 if max_dd < -0.20 else 0.0

            # Aftershock score: recent recovery quality
            # Measures how quickly price recovered from recent low
            if n >= 48:
                recent_low = np.min(close[-48:])
                recovery = (close[-1] - recent_low) / (recent_low + 1e-10)
                recent_high = np.max(close[-48:])
                drop = (recent_high - recent_low) / (recent_high + 1e-10)
                # High aftershock = big drop + incomplete recovery
                if drop > 0.03:
                    out['aftershock_score'] = max(0.0, min(1.0, drop - recovery))

        except Exception as e:
            logger.warning(f"Crisis features failed: {e}")

        return out

    # ------------------------------------------------------------------
    # Private: Extra features referenced by archetype logic
    # ------------------------------------------------------------------

    def _extra_archetype_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Additional feature store columns that archetype_instance.py or
        logic.py may reference. Provide sensible defaults so KeyError
        does not occur at runtime.
        """
        close = self._buf['close'].values.astype(float)
        open_ = self._buf['open'].values.astype(float)
        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)
        vol = self._buf['volume'].values.astype(float)
        n = len(close)

        out: Dict[str, Any] = {}

        # Wick features (wick_trap archetype K reads these)
        body = abs(close[-1] - open_[-1])
        full_range = high[-1] - low[-1] + 1e-10
        lower_wick = min(close[-1], open_[-1]) - low[-1]
        upper_wick = high[-1] - max(close[-1], open_[-1])
        out['wick_ratio'] = (lower_wick + upper_wick) / full_range
        out['wick_lower_ratio'] = lower_wick / full_range
        out['lower_wick'] = lower_wick
        out['upper_wick'] = upper_wick
        out['body_size'] = body

        # ATR percentile (used by liquidity_compression _check_E)
        if n >= 100:
            atrs = []
            for i in range(14, n):
                tr_slice_high = high[i - 13:i + 1]
                tr_slice_low = low[i - 13:i + 1]
                tr_slice_close = close[i - 14:i]
                trs = np.maximum(
                    tr_slice_high - tr_slice_low,
                    np.maximum(
                        np.abs(tr_slice_high - np.concatenate([[close[i - 14]], tr_slice_close[:-1]])),
                        np.abs(tr_slice_low - np.concatenate([[close[i - 14]], tr_slice_close[:-1]]))
                    )
                )
                atrs.append(np.mean(trs))
            current_atr = features.get('atr_14', atrs[-1] if atrs else 0)
            out['atr_percentile'] = float(np.searchsorted(np.sort(atrs), current_atr) / len(atrs))
        else:
            out['atr_percentile'] = 0.5

        # ATR pct
        out['atr_pct'] = features.get('atr_14', 0) / close[-1] if close[-1] > 0 else 0.0

        # Range features
        if n >= 20:
            ranges_20 = high[-20:] - low[-20:]
            out['range_rolling_20'] = float(np.mean(ranges_20))
        else:
            out['range_rolling_20'] = float(high[-1] - low[-1])

        if n >= 50:
            ranges_50 = high[-50:] - low[-50:]
            out['range_rolling_50'] = float(np.mean(ranges_50))
        else:
            out['range_rolling_50'] = out['range_rolling_20']

        # chop_score: ADX-based choppiness measure
        # Old formula (1-ADX/100) gave median 0.74, creating permanent instability floor
        # New: ADX>=50 = strong trend (chop=0), ADX=25 = moderate (chop=0.5), ADX=0 = chop=1.0
        out['chop_score'] = max(0.0, 1.0 - features.get('adx', 20.0) / 50.0)

        # EMA alignment flags
        out['ema_21_above_50'] = 1 if features.get('ema_21', 0) > features.get('ema_50', 0) else 0
        out['ema_50_above_200'] = 1 if features.get('ema_50', 0) > features.get('ema_200', 0) else 0
        out['price_above_ema_21'] = 1 if close[-1] > features.get('ema_21', 0) else 0
        out['price_above_ema_50'] = 1 if close[-1] > features.get('ema_50', 0) else 0

        # EMA slopes
        if n >= 22:
            ema21_arr = self._ema_array(close, 21)
            out['ema_slope_21'] = (ema21_arr[-1] - ema21_arr[-2]) / (ema21_arr[-2] + 1e-10)
        else:
            out['ema_slope_21'] = 0.0

        if n >= 51:
            ema50_arr = self._ema_array(close, 50)
            out['ema_slope_50'] = (ema50_arr[-1] - ema50_arr[-2]) / (ema50_arr[-2] + 1e-10)
        else:
            out['ema_slope_50'] = 0.0

        # Volatility features
        if n >= 20:
            rets = np.diff(close[-21:]) / close[-21:-1]
            out['volatility_20'] = float(np.std(rets))
        else:
            out['volatility_20'] = 0.0

        if n >= 50:
            rets50 = np.diff(close[-51:]) / close[-51:-1]
            out['volatility_50'] = float(np.std(rets50))
        else:
            out['volatility_50'] = out['volatility_20']

        out['volatility_ratio'] = (
            out['volatility_20'] / (out['volatility_50'] + 1e-10)
        )

        # Realized vol (annualized)
        out['rv_20d'] = out['volatility_20'] * np.sqrt(365 * 24)
        out['rv_60d'] = out['volatility_50'] * np.sqrt(365 * 24)

        # Previous close (for event override detector)
        out['prev_close'] = float(close[-2]) if n >= 2 else float(close[-1])

        # CHOCH (Change of Character) -- simplified
        out['tf1h_choch_detected'] = 0
        out['tf4h_choch_flag'] = 0

        # Liquidity imbalance (simplified)
        if n >= 20:
            buy_vol = np.sum(vol[-20:][close[-20:] > open_[-20:]])
            sell_vol = np.sum(vol[-20:][close[-20:] <= open_[-20:]])
            out['liquidity_imbalance'] = buy_vol / (sell_vol + 1e-10)
        else:
            out['liquidity_imbalance'] = 1.0

        # Wick exhaustion (last 3 bars)
        if n >= 3:
            wick_sum = 0.0
            for i in range(1, 4):
                bar_range = high[-i] - low[-i] + 1e-10
                bar_lower_wick = min(close[-i], open_[-i]) - low[-i]
                bar_upper_wick = high[-i] - max(close[-i], open_[-i])
                wick_sum += (bar_lower_wick + bar_upper_wick) / bar_range
            out['wick_exhaustion_last_3b'] = wick_sum / 3.0
        else:
            out['wick_exhaustion_last_3b'] = 0.0

        # Trend strength
        out['trend_strength_score'] = features.get('wyckoff_score', 0.5)

        # Macro features (from yfinance MacroDataFetcher — NEVER silent zeros)
        out['macro_regime'] = features.get('regime_label', 'neutral')  # CMI-derived label
        macro = self._macro_fetcher.get_features()
        out.update(macro)

        # Volume 7d z-score
        if n >= 168:
            vol_7d = vol[-168:]
            out['volume_z_7d'] = (vol[-1] - np.mean(vol_7d)) / (np.std(vol_7d) + 1e-10)
        else:
            out['volume_z_7d'] = features.get('volume_zscore', 0.0)

        return out

    # ------------------------------------------------------------------
    # Private: Multi-TF Resampling Helper
    # ------------------------------------------------------------------

    def _resample_to_tf(self, buf: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Resample 1H buffer to 4H or 1D OHLCV."""
        rule = '4h' if tf in ('4H', '4h') else '1D'
        resampled = buf.resample(rule).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'
        }).dropna()
        return resampled

    # ------------------------------------------------------------------
    # Private: FVG Detection
    # ------------------------------------------------------------------

    def _detect_fvg_1h(self) -> int:
        """3-candle Fair Value Gap detection on 1H."""
        n = len(self._buf)
        if n < 3:
            return 0

        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)

        # Bullish FVG: candle[-3] high < candle[-1] low (gap up)
        if high[-3] < low[-1]:
            return 1
        # Bearish FVG: candle[-3] low > candle[-1] high (gap down)
        if low[-3] > high[-1]:
            return 1

        return 0

    def _detect_fvg_4h(self) -> int:
        """FVG detection on 4H-equivalent (use every 4th bar)."""
        n = len(self._buf)
        if n < 12:
            return 0

        high = self._buf['high'].values.astype(float)
        low = self._buf['low'].values.astype(float)

        # Sample every 4th bar for 4H-equivalent
        idx3 = -12  # 3 * 4 bars ago
        idx1 = -4   # 1 * 4 bars ago

        h3 = np.max(high[max(0, n + idx3):max(0, n + idx3 + 4)])
        l3 = np.min(low[max(0, n + idx3):max(0, n + idx3 + 4)])
        h1 = np.max(high[max(0, n + idx1):])
        l1 = np.min(low[max(0, n + idx1):])

        if h3 < l1:
            return 1
        if l3 > h1:
            return 1

        return 0

    # ------------------------------------------------------------------
    # Private: Swing / Fib Helpers
    # ------------------------------------------------------------------

    def _bars_since_pivot(self, high: np.ndarray, low: np.ndarray, window: int = 50) -> int:
        """Bars since the last swing high or swing low within window."""
        n = len(high)
        if n < 5:
            return 0

        lookback = min(window, n)
        h = high[-lookback:]
        l = low[-lookback:]

        # Find swing high: bar where high is max of 2 neighbours on each side
        last_pivot_bar = 0
        for i in range(2, lookback - 2):
            # Swing high
            if h[i] >= h[i - 1] and h[i] >= h[i - 2] and h[i] >= h[i + 1] and h[i] >= h[i + 2]:
                last_pivot_bar = max(last_pivot_bar, lookback - i)
            # Swing low
            if l[i] <= l[i - 1] and l[i] <= l[i - 2] and l[i] <= l[i + 1] and l[i] <= l[i + 2]:
                last_pivot_bar = max(last_pivot_bar, lookback - i)

        # Invert: bars_since = lookback - last_pivot_bar_from_start
        # last_pivot_bar is already distance-from-end
        return last_pivot_bar if last_pivot_bar > 0 else lookback

    def _fib_in_premium(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> int:
        """
        1 if close > 0.618 retracement from recent swing low to swing high.
        Uses last 50 bars to find swing range.
        """
        n = len(close)
        if n < 10:
            return 0

        lookback = min(50, n)
        swing_high = np.max(high[-lookback:])
        swing_low = np.min(low[-lookback:])
        fib_range = swing_high - swing_low

        if fib_range <= 0:
            return 0

        fib_618 = swing_low + 0.618 * fib_range
        return 1 if close[-1] > fib_618 else 0

    # ------------------------------------------------------------------
    # Private: NaN filling
    # ------------------------------------------------------------------

    def _fill_nans(self, series: pd.Series) -> pd.Series:
        """
        Handle NaN values:
        - Score / ratio columns: fill with 0
        - Indicator columns: forward-fill, then 0
        """
        # Score columns default to 0
        score_keys = [k for k in series.index if any(
            pat in str(k).lower() for pat in
            ['score', 'fusion', 'pti', 'fakeout', 'bos', 'fvg', 'choch',
             'climax', 'detected', 'signal', 'flag', 'premium']
        )]
        for k in score_keys:
            if pd.isna(series[k]):
                series[k] = 0.0

        # Fill remaining NaN with 0
        series = series.fillna(0.0)

        return series

    # ==================================================================
    # Indicator Calculation Helpers
    # ==================================================================

    def _calc_rsi(self, close: np.ndarray, period: int) -> float:
        """Compute RSI for the last ``period`` bars."""
        if len(close) < period + 1:
            return 50.0

        if TALIB_AVAILABLE:
            rsi = talib.RSI(close, timeperiod=period)
            val = rsi[-1]
            return float(val) if not np.isnan(val) else 50.0

        # Pandas fallback
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Compute ATR over ``period`` bars."""
        n = len(close)
        if n < period + 1:
            return float(high[-1] - low[-1]) if n > 0 else 0.0

        if TALIB_AVAILABLE:
            atr = talib.ATR(high, low, close, timeperiod=period)
            val = atr[-1]
            return float(val) if not np.isnan(val) else float(high[-1] - low[-1])

        # Pandas fallback -- true range
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        return float(np.mean(tr[-period:]))

    def _calc_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Compute ADX over ``period`` bars."""
        n = len(close)
        if n < period * 2:
            return 20.0

        if TALIB_AVAILABLE:
            adx = talib.ADX(high, low, close, timeperiod=period)
            val = adx[-1]
            return float(val) if not np.isnan(val) else 20.0

        # Pandas fallback (simplified)
        up_moves = np.diff(high)
        down_moves = -np.diff(low)
        plus_dm = np.where((up_moves > down_moves) & (up_moves > 0), up_moves, 0.0)
        minus_dm = np.where((down_moves > up_moves) & (down_moves > 0), down_moves, 0.0)

        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )

        # Simple moving averages over period
        if len(tr) < period:
            return 20.0

        atr_p = np.mean(tr[-period:])
        plus_di = 100.0 * np.mean(plus_dm[-period:]) / (atr_p + 1e-10)
        minus_di = 100.0 * np.mean(minus_dm[-period:]) / (atr_p + 1e-10)

        dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)

        # For proper ADX we need smoothed DX over ``period`` -- approximate
        if len(tr) >= period * 2:
            dx_arr = []
            for i in range(period, len(tr)):
                a = np.mean(tr[i - period:i])
                pd_ = 100.0 * np.mean(plus_dm[i - period:i]) / (a + 1e-10)
                md_ = 100.0 * np.mean(minus_dm[i - period:i]) / (a + 1e-10)
                dx_arr.append(100.0 * abs(pd_ - md_) / (pd_ + md_ + 1e-10))
            adx_val = float(np.mean(dx_arr[-period:]))
        else:
            adx_val = dx

        return min(float(adx_val), 100.0)

    def _sma(self, arr: np.ndarray, period: int) -> float:
        """Simple moving average of last ``period`` values."""
        if len(arr) < period:
            return float(np.mean(arr)) if len(arr) > 0 else 0.0
        return float(np.mean(arr[-period:]))

    def _ema(self, arr: np.ndarray, period: int) -> float:
        """Exponential moving average -- return last value."""
        if len(arr) == 0:
            return 0.0

        if TALIB_AVAILABLE:
            ema = talib.EMA(arr, timeperiod=period)
            val = ema[-1]
            return float(val) if not np.isnan(val) else float(arr[-1])

        # Pandas fallback
        alpha = 2.0 / (period + 1.0)
        ema_val = float(arr[0])
        for v in arr[1:]:
            ema_val = alpha * v + (1.0 - alpha) * ema_val
        return ema_val

    def _ema_array(self, arr: np.ndarray, period: int) -> np.ndarray:
        """Return full EMA array (for slope calculations)."""
        if len(arr) == 0:
            return np.array([0.0])

        if TALIB_AVAILABLE:
            ema = talib.EMA(arr, timeperiod=period)
            return np.nan_to_num(ema, nan=arr[-1])

        alpha = 2.0 / (period + 1.0)
        result = np.empty_like(arr, dtype=float)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
        return result

    def _bb_width(self, close: np.ndarray, period: int = 20, num_std: float = 2.0) -> float:
        """Bollinger Band width = (upper - lower) / middle."""
        if len(close) < period:
            return 0.0

        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=num_std, nbdevdn=num_std)
            u, m, l_ = upper[-1], middle[-1], lower[-1]
            if not np.isnan(m) and m > 0:
                return float((u - l_) / m)
            return 0.0

        # Pandas fallback
        window = close[-period:]
        middle = np.mean(window)
        std = np.std(window)
        upper = middle + num_std * std
        lower = middle - num_std * std

        return float((upper - lower) / middle) if middle > 0 else 0.0

    def _calc_macd(self, close: np.ndarray, fast: int, slow: int, signal: int):
        """Compute MACD, signal, histogram."""
        if len(close) < slow + signal:
            return 0.0, 0.0, 0.0

        if TALIB_AVAILABLE:
            m, s, h = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            mv = m[-1] if not np.isnan(m[-1]) else 0.0
            sv = s[-1] if not np.isnan(s[-1]) else 0.0
            hv = h[-1] if not np.isnan(h[-1]) else 0.0
            return float(mv), float(sv), float(hv)

        # Pandas fallback
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        macd_line = ema_fast - ema_slow

        # For signal line, we need the MACD line array
        fast_arr = self._ema_array(close, fast)
        slow_arr = self._ema_array(close, slow)
        macd_arr = fast_arr - slow_arr
        signal_val = self._ema(macd_arr, signal)

        hist = macd_line - signal_val
        return float(macd_line), float(signal_val), float(hist)


# ===========================================================================
# Test mode
# ===========================================================================

def _test_mode():
    """
    Load 500 candles from the feature store parquet, feed them through
    the computer one-by-one, and print the last feature vector.
    """
    import time

    project_root = Path(__file__).parent.parent.parent

    # Try to find feature store
    candidates = [
        project_root / 'data' / 'features_mtf' / 'BTC_1H_FEATURES_V12_ENHANCED.parquet',
        project_root / 'data' / 'features_mtf' / 'BTC_1H_FEATURES_FIXED_20260206.parquet',
        project_root / 'data' / 'features_mtf' / 'BTC_1H_CANONICAL_20260202.parquet',
        project_root / 'data' / 'features_mtf' / 'BTC_1H_LATEST.parquet',
    ]

    store_path = None
    for p in candidates:
        if p.exists():
            store_path = p
            break

    if store_path is None:
        print("ERROR: No feature store parquet found. Tried:")
        for p in candidates:
            print(f"  {p}")
        sys.exit(1)

    print(f"Loading feature store: {store_path}")
    df = pd.read_parquet(store_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Take 500 bars from the middle of the dataset (2023 data)
    start_idx = len(df) // 2
    subset = df.iloc[start_idx:start_idx + 500].copy()
    print(f"Using bars {start_idx} to {start_idx + 500} ({subset.index[0]} to {subset.index[-1]})")

    # Seed with first 450 candles, then update one-by-one for last 50
    computer = LiveFeatureComputer(buffer_size=500)

    seed = subset.iloc[:450]
    computer.ingest_candles(seed)
    print(f"Seeded with {len(seed)} candles")

    t0 = time.time()
    last_features = None
    for i in range(450, len(subset)):
        row = subset.iloc[i]
        candle = {
            'timestamp': row.name,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
        }
        # Add funding rate if available
        if 'funding_rate' in row and not pd.isna(row['funding_rate']):
            candle['funding_rate'] = row['funding_rate']

        last_features = computer.update(candle)

    elapsed = time.time() - t0
    print(f"\nProcessed 50 candles in {elapsed:.3f}s ({elapsed / 50 * 1000:.1f}ms per candle)")

    # Print last feature vector
    print(f"\n{'=' * 70}")
    print(f"LAST FEATURE VECTOR ({last_features.name})")
    print(f"{'=' * 70}")
    print(f"Total features: {len(last_features)}")
    print()

    # Group by category
    categories = {
        'OHLCV': ['open', 'high', 'low', 'close', 'volume'],
        'Technical': ['rsi_14', 'rsi_21', 'adx', 'adx_14', 'atr_14', 'atr_20',
                      'sma_50', 'sma_200', 'sma_20', 'sma_100',
                      'ema_9', 'ema_21', 'ema_50', 'ema_200',
                      'bb_width', 'macd', 'macd_signal', 'macd_hist'],
        'Volume': ['volume_zscore', 'volume_ma_20', 'volume_z', 'volume_ratio',
                   'volume_z_7d', 'volume_climax_last_3b'],
        'Regime': ['regime_label', 'crisis_prob', 'risk_temperature', 'instability_score'],
        'Macro': ['VIX_Z', 'DXY_Z', 'GOLD_Z', 'YIELD_CURVE', 'BTC.D', 'USDT.D', 'USDC.D'],
        'SMC/BOS': ['tf1h_bos_bullish', 'tf1h_bos_bearish', 'tf1h_bos_detected',
                    'tf1h_choch_detected', 'tf1h_fvg_present', 'fusion_smc',
                    'smc_strength', 'smc_confidence',
                    'tf4h_bos_bullish', 'tf4h_bos_bearish', 'tf4h_fvg_present'],
        'BOMS/Liquidity': ['tf1d_boms_strength', 'tf4h_boms_displacement',
                           'boms_strength', 'liquidity_score', 'liquidity_imbalance'],
        'Wyckoff': ['wyckoff_score', 'wyckoff_event_confidence',
                    'tf4h_wyckoff_phase_score', 'tf1d_wyckoff_score',
                    'tf1d_wyckoff_m1_signal', 'tf1d_wyckoff_m2_signal'],
        'PTI/Fakeout': ['tf1h_pti_score', 'tf1h_pti_confidence', 'tf1h_pti_trap_type',
                        'tf1h_pti_reversal_likely', 'tf1d_pti_score',
                        'tf1h_fakeout_detected', 'tf1h_fakeout_intensity'],
        'Temporal': ['temporal_confluence_score', 'gann_time_cluster',
                     'fib_time_cluster', 'tpi_signal', 'cycle_phase',
                     'temporal_reversal_zone'],
        'Fibonacci': ['fib_retracement_level', 'fib_distance_pct',
                      'fib_in_ote_zone', 'fib_in_discount', 'fib_in_premium',
                      'fib_time_zone', 'fib_time_confluence', 'fib_swing_range_pct'],
        'Coil/Squeeze': ['tf1h_coil_score', 'tf1h_is_coiling', 'tf1h_coil_breakout',
                         'tf4h_coil_score', 'tf4h_is_coiling', 'tf4h_coil_breakout',
                         'tf1d_coil_score', 'tf1d_is_coiling'],
        'FRVP': ['tf1h_frvp_poc', 'tf1h_frvp_va_high', 'tf1h_frvp_va_low',
                 'tf1h_frvp_position', 'tf1h_frvp_distance_to_poc'],
        'Crisis': ['crash_frequency_7d', 'drawdown_max_60d', 'drawdown_persistence',
                   'crisis_persistence', 'aftershock_score'],
        'Structure 4H': ['tf4h_squiggle_stage', 'tf4h_squiggle_confidence',
                         'tf4h_squiggle_direction', 'tf4h_squiggle_entry_window'],
        'Fusion': ['tf1h_fusion_score', 'tf4h_fusion_score', 'tf1d_fusion_score',
                   'fusion_total', 'fusion_smc', 'fusion_wyckoff',
                   'fusion_liquidity', 'fusion_momentum'],
        'Penalty': ['funding_Z', 'funding_rate', 'bars_since_pivot', 'fib_in_premium'],
        'Binance_Futures': ['oi_value', 'oi_change_4h', 'oi_change_24h',
                            'oi_price_divergence', 'binance_funding_rate',
                            'funding_oi_divergence', 'ls_ratio_extreme', 'taker_imbalance'],
    }

    for cat_name, keys in categories.items():
        print(f"  --- {cat_name} ---")
        for k in keys:
            if k in last_features.index:
                v = last_features[k]
                if isinstance(v, float):
                    print(f"    {k:35s} = {v:.6f}")
                else:
                    print(f"    {k:35s} = {v}")
            else:
                print(f"    {k:35s} = (not computed)")
        print()

    # Compare with feature store values for key features
    store_row = subset.iloc[-1]
    print(f"{'=' * 70}")
    print("COMPARISON: Live Computed vs Feature Store")
    print(f"{'=' * 70}")
    compare_keys = [
        'rsi_14', 'adx', 'atr_14', 'sma_50', 'sma_200', 'bb_width',
        'macd', 'close', 'volume', 'regime_label',
        'tf1h_fusion_score', 'wyckoff_score', 'wyckoff_event_confidence',
        'tf1h_pti_score', 'tf1h_fakeout_detected', 'fusion_smc',
        'temporal_confluence_score', 'fib_retracement_level',
        'tf1h_coil_score', 'tf4h_squiggle_confidence',
    ]
    for k in compare_keys:
        live_val = last_features.get(k, 'N/A')
        store_val = store_row.get(k, 'N/A') if k in store_row.index else 'N/A'
        if isinstance(live_val, float) and isinstance(store_val, float):
            diff = abs(live_val - store_val)
            pct = diff / (abs(store_val) + 1e-10) * 100
            print(f"  {k:30s}  Live={live_val:12.4f}  Store={store_val:12.4f}  Diff={pct:6.1f}%")
        else:
            print(f"  {k:30s}  Live={str(live_val):>12s}  Store={str(store_val):>12s}")

    print(f"\n{'=' * 70}")
    print("Test complete.")


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bull Machine Live Feature Computer')
    parser.add_argument('--test', action='store_true', help='Run test mode with feature store data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    if args.test:
        _test_mode()
    else:
        print("Live Feature Computer for Bull Machine")
        print("Usage:")
        print("  python bin/live/live_feature_computer.py --test")
        print("  python bin/live/live_feature_computer.py --test --verbose")
        print()
        print("As a library:")
        print("  from bin.live.live_feature_computer import LiveFeatureComputer")
        print("  computer = LiveFeatureComputer()")
        print("  computer.ingest_candles(df)  # seed with 500 candles")
        print("  features = computer.update(candle)  # returns pd.Series")
