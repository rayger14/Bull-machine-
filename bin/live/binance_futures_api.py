#!/usr/bin/env python3
"""
Binance Futures Public API Client

Fetches open interest, funding rates, long/short ratios, and taker buy/sell data
from Binance Futures public endpoints. No authentication required.

Endpoints used:
  - /fapi/v1/openInterest          (current OI)
  - /futures/data/openInterestHist (historical OI)
  - /fapi/v1/fundingRate           (funding rate history)
  - /futures/data/globalLongShortAccountRatio  (global L/S ratio)
  - /futures/data/topLongShortAccountRatio     (top trader L/S ratio)
  - /futures/data/takerlongshortRatio          (taker buy/sell ratio)

Rate limits:
  - Binance allows 2400 requests/min for public endpoints.
  - This client enforces a conservative 2 req/s (120 req/min) by default.
  - Historical backfills with pagination will self-throttle automatically.

US access note:
  - The /fapi/v1/* endpoints may be geo-restricted from US IPs.
  - The /futures/data/* (historical data) endpoints are generally accessible.
  - If you encounter 451 errors, use a VPN or proxy, or set the
    BinanceFuturesAPI(base_url=...) to a reachable mirror.

Usage (live):
    api = BinanceFuturesAPI()
    current = api.fetch_all_current()
    print(current['oi_change_4h'])

Usage (historical backfill):
    api = BinanceFuturesAPI()
    df = api.fetch_historical_range(start_date='2020-01-01', end_date='2024-12-31')
    df.to_parquet('data/binance_futures_features.parquet')

Usage (CLI test):
    python3 bin/live/binance_futures_api.py --test
    python3 bin/live/binance_futures_api.py --backfill --start 2023-01-01 --end 2023-03-31
"""

import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# Graceful import of dependencies ------------------------------------------------
try:
    import requests
except ImportError:
    print("ERROR: requests is not installed.")
    print("Install it with:  pip install requests")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.")
    print("Install it with:  pip install pandas")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed.")
    print("Install it with:  pip install numpy")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------
VALID_PERIODS = ("5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d")

# Number of bars that correspond to 4h and 24h for each period, used for
# computing rolling OI change columns in fetch_historical_range.
PERIOD_TO_BARS = {
    "5m":  {"4h": 48, "24h": 288},
    "15m": {"4h": 16, "24h": 96},
    "30m": {"4h": 8,  "24h": 48},
    "1h":  {"4h": 4,  "24h": 24},
    "2h":  {"4h": 2,  "24h": 12},
    "4h":  {"4h": 1,  "24h": 6},
    "6h":  {"4h": 1,  "24h": 4},
    "12h": {"4h": 1,  "24h": 2},
    "1d":  {"4h": 1,  "24h": 1},
}


# ---------------------------------------------------------------------------------
# BinanceFuturesAPI
# ---------------------------------------------------------------------------------
class BinanceFuturesAPI:
    """Binance Futures public data client. No authentication required."""

    BASE_URL = "https://fapi.binance.com"
    DATA_URL = "https://fapi.binance.com/futures/data"

    def __init__(
        self,
        base_url: Optional[str] = None,
        data_url: Optional[str] = None,
        max_retries: int = 3,
        requests_per_second: float = 2.0,
        timeout: int = 30,
    ):
        """
        Args:
            base_url: Override for /fapi/v1 endpoints (e.g. for proxy).
            data_url: Override for /futures/data endpoints.
            max_retries: Number of retries on transient errors (429, 5xx).
            requests_per_second: Max request rate (default 2/s, conservative).
            timeout: HTTP request timeout in seconds.
        """
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.data_url = (data_url or self.DATA_URL).rstrip("/")
        self.max_retries = max_retries
        self.min_interval = 1.0 / requests_per_second
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BullMachine/1.0",
        })
        self._last_request_ts: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        """Enforce minimum interval between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self.min_interval:
            sleep_for = self.min_interval - elapsed
            time.sleep(sleep_for)
        self._last_request_ts = time.monotonic()

    def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute an HTTP GET with throttling, retries, and error handling.

        Returns parsed JSON on success, or None on failure.
        """
        params = params or {}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    return resp.json()

                # Rate-limited
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning(
                        "Rate limited (429) on %s, retry %d/%d after %ds",
                        url, attempt, self.max_retries, retry_after,
                    )
                    time.sleep(retry_after)
                    continue

                # Geo-restriction
                if resp.status_code == 451:
                    logger.error(
                        "Geo-restricted (451) on %s — US IP detected. "
                        "Use a VPN or proxy for Binance Futures API.",
                        url,
                    )
                    return None

                # Server errors — retry
                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        "Server error %d on %s, retry %d/%d after %ds",
                        resp.status_code, url, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                # Client errors (4xx except 429/451) — do not retry
                logger.error(
                    "Client error %d on %s: %s",
                    resp.status_code, url, resp.text[:500],
                )
                return None

            except requests.exceptions.Timeout:
                wait = 2 ** attempt
                logger.warning(
                    "Timeout on %s, retry %d/%d after %ds",
                    url, attempt, self.max_retries, wait,
                )
                last_exc = requests.exceptions.Timeout(f"Timeout on {url}")
                time.sleep(wait)
                continue

            except requests.exceptions.ConnectionError as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Connection error on %s: %s, retry %d/%d after %ds",
                    url, exc, attempt, self.max_retries, wait,
                )
                last_exc = exc
                time.sleep(wait)
                continue

            except requests.exceptions.RequestException as exc:
                logger.error("Request failed on %s: %s", url, exc)
                return None

        logger.error(
            "All %d retries exhausted for %s (last error: %s)",
            self.max_retries, url, last_exc,
        )
        return None

    @staticmethod
    def _to_ms(dt_str: str) -> int:
        """Convert 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' to milliseconds epoch."""
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(dt_str, fmt).replace(tzinfo=timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse date string: {dt_str!r}")

    @staticmethod
    def _ms_to_timestamp(ms: int) -> pd.Timestamp:
        """Convert milliseconds epoch to pandas Timestamp (UTC)."""
        return pd.Timestamp(ms, unit="ms", tz="UTC")

    # ------------------------------------------------------------------
    # Public methods — current values
    # ------------------------------------------------------------------

    def get_open_interest(self, symbol: str = "BTCUSDT") -> float:
        """
        Get current open interest in contracts.

        Endpoint: GET /fapi/v1/openInterest
        Returns 0.0 on failure.
        """
        url = f"{self.base_url}/fapi/v1/openInterest"
        data = self._request(url, {"symbol": symbol})
        if data is None:
            return 0.0
        try:
            return float(data.get("openInterest", 0))
        except (TypeError, ValueError) as exc:
            logger.error("Failed to parse openInterest response: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Public methods — historical DataFrames
    # ------------------------------------------------------------------

    def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get historical open interest.

        Endpoint: GET /futures/data/openInterestHist
        Returns DataFrame with columns: [timestamp, sumOpenInterest, sumOpenInterestValue]
        Max 500 records per request.

        Args:
            symbol: Trading pair (default BTCUSDT).
            period: Kline period — one of 5m,15m,30m,1h,2h,4h,6h,12h,1d.
            limit: Number of records (max 500).
            start_time: Start time in milliseconds epoch (inclusive).
            end_time: End time in milliseconds epoch (inclusive).
        """
        if period not in VALID_PERIODS:
            logger.error("Invalid period %r, must be one of %s", period, VALID_PERIODS)
            return pd.DataFrame()

        url = f"{self.data_url}/openInterestHist"
        params: Dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500),
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)
            if df.empty:
                return df
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in ("sumOpenInterest", "sumOpenInterestValue"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df[["timestamp", "sumOpenInterest", "sumOpenInterestValue"]]
        except Exception as exc:
            logger.error("Failed to parse openInterestHist: %s", exc)
            return pd.DataFrame()

    def get_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get funding rate history.

        Endpoint: GET /fapi/v1/fundingRate
        Returns DataFrame with columns: [timestamp, fundingRate, fundingTime, markPrice]
        Max 1000 records per request.
        """
        url = f"{self.base_url}/fapi/v1/fundingRate"
        params: Dict[str, Any] = {
            "symbol": symbol,
            "limit": min(limit, 1000),
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)
            if df.empty:
                return df
            # Binance returns fundingTime as ms epoch
            df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
            if "markPrice" in df.columns:
                df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
            else:
                df["markPrice"] = np.nan
            df["fundingTime"] = df["timestamp"]  # keep both for compatibility
            return df[["timestamp", "fundingRate", "fundingTime", "markPrice"]]
        except Exception as exc:
            logger.error("Failed to parse fundingRate: %s", exc)
            return pd.DataFrame()

    def get_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get global long/short account ratio.

        Endpoint: GET /futures/data/globalLongShortAccountRatio
        Returns DataFrame with columns: [timestamp, longShortRatio, longAccount, shortAccount]
        Max 500 records per request.
        """
        return self._fetch_data_endpoint(
            path="/globalLongShortAccountRatio",
            symbol=symbol,
            period=period,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            columns=["timestamp", "longShortRatio", "longAccount", "shortAccount"],
            numeric_cols=["longShortRatio", "longAccount", "shortAccount"],
            label="globalLongShortAccountRatio",
        )

    def get_taker_buy_sell_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get taker buy/sell volume ratio.

        Endpoint: GET /futures/data/takerlongshortRatio
        Returns DataFrame with columns: [timestamp, buySellRatio, buyVol, sellVol]
        Max 500 records per request.
        """
        return self._fetch_data_endpoint(
            path="/takerlongshortRatio",
            symbol=symbol,
            period=period,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            columns=["timestamp", "buySellRatio", "buyVol", "sellVol"],
            numeric_cols=["buySellRatio", "buyVol", "sellVol"],
            label="takerlongshortRatio",
        )

    def get_top_trader_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get top trader long/short account ratio.

        Endpoint: GET /futures/data/topLongShortAccountRatio
        Returns DataFrame with columns: [timestamp, longShortRatio, longAccount, shortAccount]
        Max 500 records per request.
        """
        return self._fetch_data_endpoint(
            path="/topLongShortAccountRatio",
            symbol=symbol,
            period=period,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            columns=["timestamp", "longShortRatio", "longAccount", "shortAccount"],
            numeric_cols=["longShortRatio", "longAccount", "shortAccount"],
            label="topLongShortAccountRatio",
        )

    def _fetch_data_endpoint(
        self,
        path: str,
        symbol: str,
        period: str,
        limit: int,
        start_time: Optional[int],
        end_time: Optional[int],
        columns: List[str],
        numeric_cols: List[str],
        label: str,
    ) -> pd.DataFrame:
        """
        Generic fetcher for /futures/data/* endpoints that share the same
        request/response shape (symbol, period, limit, startTime, endTime
        returning timestamp + numeric columns).
        """
        if period not in VALID_PERIODS:
            logger.error("Invalid period %r, must be one of %s", period, VALID_PERIODS)
            return pd.DataFrame()

        url = f"{self.data_url}{path}"
        params: Dict[str, Any] = {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500),
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        data = self._request(url, params)
        if not data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)
            if df.empty:
                return df
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            # Return only expected columns (intersection with available)
            available = [c for c in columns if c in df.columns]
            return df[available]
        except Exception as exc:
            logger.error("Failed to parse %s: %s", label, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Composite: fetch_all_current
    # ------------------------------------------------------------------

    def fetch_all_current(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Fetch all current values from every endpoint and return a flat dict.

        Returns:
            {
                'oi_value': float,            # Current OI in USDT
                'oi_change_4h': float,        # 4h OI percentage change
                'oi_change_24h': float,       # 24h OI percentage change
                'funding_rate': float,        # Most recent funding rate
                'ls_ratio': float,            # Global long/short ratio
                'top_trader_ls_ratio': float, # Top trader L/S ratio
                'taker_buy_sell_ratio': float, # Taker buy/sell volume ratio
            }

        All values default to 0.0 on failure.
        """
        result: Dict[str, float] = {
            "oi_value": 0.0,
            "oi_change_4h": 0.0,
            "oi_change_24h": 0.0,
            "funding_rate": 0.0,
            "ls_ratio": 0.0,
            "top_trader_ls_ratio": 0.0,
            "taker_buy_sell_ratio": 0.0,
        }

        # --- Open Interest (current + recent history for change calc) ---
        try:
            oi_hist = self.get_open_interest_history(
                symbol=symbol, period="1h", limit=25,
            )
            if not oi_hist.empty and "sumOpenInterestValue" in oi_hist.columns:
                oi_vals = oi_hist["sumOpenInterestValue"].dropna()
                if len(oi_vals) > 0:
                    result["oi_value"] = float(oi_vals.iloc[-1])
                if len(oi_vals) >= 5:
                    current = oi_vals.iloc[-1]
                    oi_4h_ago = oi_vals.iloc[-5]  # 4 bars back at 1h period
                    if oi_4h_ago != 0:
                        result["oi_change_4h"] = float(
                            (current - oi_4h_ago) / oi_4h_ago
                        )
                if len(oi_vals) >= 25:
                    current = oi_vals.iloc[-1]
                    oi_24h_ago = oi_vals.iloc[-25]  # 24 bars back at 1h period
                    if oi_24h_ago != 0:
                        result["oi_change_24h"] = float(
                            (current - oi_24h_ago) / oi_24h_ago
                        )
        except Exception as exc:
            logger.warning("Failed to compute OI metrics: %s", exc)

        # --- Funding Rate ---
        try:
            fr = self.get_funding_rate(symbol=symbol, limit=1)
            if not fr.empty and "fundingRate" in fr.columns:
                val = fr["fundingRate"].iloc[-1]
                if val == val:  # NaN guard
                    result["funding_rate"] = float(val)
        except Exception as exc:
            logger.warning("Failed to fetch funding rate: %s", exc)

        # --- Global Long/Short Ratio ---
        try:
            ls = self.get_long_short_ratio(symbol=symbol, period="1h", limit=1)
            if not ls.empty and "longShortRatio" in ls.columns:
                val = ls["longShortRatio"].iloc[-1]
                if val == val:
                    result["ls_ratio"] = float(val)
        except Exception as exc:
            logger.warning("Failed to fetch L/S ratio: %s", exc)

        # --- Top Trader Long/Short Ratio ---
        try:
            top_ls = self.get_top_trader_long_short_ratio(
                symbol=symbol, period="1h", limit=1,
            )
            if not top_ls.empty and "longShortRatio" in top_ls.columns:
                val = top_ls["longShortRatio"].iloc[-1]
                if val == val:
                    result["top_trader_ls_ratio"] = float(val)
        except Exception as exc:
            logger.warning("Failed to fetch top trader L/S ratio: %s", exc)

        # --- Taker Buy/Sell Ratio ---
        try:
            taker = self.get_taker_buy_sell_ratio(
                symbol=symbol, period="1h", limit=1,
            )
            if not taker.empty and "buySellRatio" in taker.columns:
                val = taker["buySellRatio"].iloc[-1]
                if val == val:
                    result["taker_buy_sell_ratio"] = float(val)
        except Exception as exc:
            logger.warning("Failed to fetch taker buy/sell ratio: %s", exc)

        logger.info(
            "fetch_all_current: OI=$%.0f  OI_4h=%.3f%%  OI_24h=%.3f%%  "
            "FR=%.6f  LS=%.3f  TopLS=%.3f  TakerBS=%.3f",
            result["oi_value"],
            result["oi_change_4h"] * 100,
            result["oi_change_24h"] * 100,
            result["funding_rate"],
            result["ls_ratio"],
            result["top_trader_ls_ratio"],
            result["taker_buy_sell_ratio"],
        )
        return result

    # ------------------------------------------------------------------
    # Composite: fetch_historical_range (paginated backfill)
    # ------------------------------------------------------------------

    def fetch_historical_range(
        self,
        symbol: str = "BTCUSDT",
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        period: str = "1h",
        save_intermediates: bool = False,
        output_dir: str = "data/binance_futures",
    ) -> pd.DataFrame:
        """
        Fetch ALL historical data by paginating through the API, merge all
        features into a single DataFrame aligned by timestamp, and compute
        derived columns.

        Args:
            symbol: Trading pair.
            start_date: Start date 'YYYY-MM-DD'.
            end_date: End date 'YYYY-MM-DD'.
            period: Kline period for /futures/data/* endpoints.
            save_intermediates: If True, save per-endpoint CSVs to output_dir.
            output_dir: Directory for intermediate CSV files.

        Returns:
            DataFrame indexed by timestamp with columns:
              - sumOpenInterest, sumOpenInterestValue
              - fundingRate, markPrice
              - longShortRatio, longAccount, shortAccount  (global)
              - topLongShortRatio, topLongAccount, topShortAccount
              - buySellRatio, buyVol, sellVol
              - oi_change_4h, oi_change_24h          (computed)
              - oi_price_divergence                   (computed, requires markPrice)
              - funding_oi_divergence                 (computed)
              - ls_ratio_extreme                      (computed, Z-score)
              - taker_imbalance                       (computed)
        """
        start_ms = self._to_ms(start_date)
        end_ms = self._to_ms(end_date)

        if save_intermediates:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Fetching historical range: %s to %s, period=%s, symbol=%s",
            start_date, end_date, period, symbol,
        )

        # 1. Open Interest History
        logger.info("  Paginating openInterestHist...")
        oi_df = self._paginate_data_endpoint(
            fetch_fn=self.get_open_interest_history,
            symbol=symbol,
            period=period,
            start_ms=start_ms,
            end_ms=end_ms,
            max_per_page=500,
        )
        logger.info("  OI history: %d rows", len(oi_df))
        if save_intermediates and not oi_df.empty:
            oi_df.to_csv(Path(output_dir) / "oi_history.csv", index=False)

        # 2. Funding Rate (uses /fapi/v1, max 1000 per page)
        logger.info("  Paginating fundingRate...")
        fr_df = self._paginate_funding_rate(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        logger.info("  Funding rate: %d rows", len(fr_df))
        if save_intermediates and not fr_df.empty:
            fr_df.to_csv(Path(output_dir) / "funding_rate.csv", index=False)

        # 3. Global Long/Short Ratio
        logger.info("  Paginating globalLongShortAccountRatio...")
        ls_df = self._paginate_data_endpoint(
            fetch_fn=self.get_long_short_ratio,
            symbol=symbol,
            period=period,
            start_ms=start_ms,
            end_ms=end_ms,
            max_per_page=500,
        )
        logger.info("  Global L/S ratio: %d rows", len(ls_df))
        if save_intermediates and not ls_df.empty:
            ls_df.to_csv(Path(output_dir) / "long_short_ratio.csv", index=False)

        # 4. Top Trader Long/Short Ratio
        logger.info("  Paginating topLongShortAccountRatio...")
        top_ls_df = self._paginate_data_endpoint(
            fetch_fn=self.get_top_trader_long_short_ratio,
            symbol=symbol,
            period=period,
            start_ms=start_ms,
            end_ms=end_ms,
            max_per_page=500,
        )
        logger.info("  Top trader L/S ratio: %d rows", len(top_ls_df))
        if save_intermediates and not top_ls_df.empty:
            top_ls_df.to_csv(
                Path(output_dir) / "top_trader_long_short_ratio.csv", index=False,
            )

        # 5. Taker Buy/Sell Ratio
        logger.info("  Paginating takerlongshortRatio...")
        taker_df = self._paginate_data_endpoint(
            fetch_fn=self.get_taker_buy_sell_ratio,
            symbol=symbol,
            period=period,
            start_ms=start_ms,
            end_ms=end_ms,
            max_per_page=500,
        )
        logger.info("  Taker buy/sell ratio: %d rows", len(taker_df))
        if save_intermediates and not taker_df.empty:
            taker_df.to_csv(
                Path(output_dir) / "taker_buy_sell_ratio.csv", index=False,
            )

        # ------------------------------------------------------------------
        # Merge all DataFrames on timestamp
        # ------------------------------------------------------------------
        merged = self._merge_dataframes(
            oi_df=oi_df,
            fr_df=fr_df,
            ls_df=ls_df,
            top_ls_df=top_ls_df,
            taker_df=taker_df,
        )

        if merged.empty:
            logger.warning("All endpoints returned empty data — nothing to merge.")
            return merged

        # ------------------------------------------------------------------
        # Compute derived features
        # ------------------------------------------------------------------
        merged = self._compute_derived_features(merged, period)

        logger.info(
            "Historical range complete: %d rows x %d cols, %s to %s",
            len(merged),
            len(merged.columns),
            merged["timestamp"].iloc[0] if not merged.empty else "N/A",
            merged["timestamp"].iloc[-1] if not merged.empty else "N/A",
        )

        return merged

    # ------------------------------------------------------------------
    # Pagination helpers
    # ------------------------------------------------------------------

    def _paginate_data_endpoint(
        self,
        fetch_fn,
        symbol: str,
        period: str,
        start_ms: int,
        end_ms: int,
        max_per_page: int = 500,
    ) -> pd.DataFrame:
        """
        Paginate any /futures/data/* endpoint that accepts startTime/endTime
        and returns a DataFrame with a 'timestamp' column.

        Moves the start window forward based on the last returned timestamp.
        """
        all_frames: List[pd.DataFrame] = []
        current_start = start_ms
        page = 0

        while current_start < end_ms:
            page += 1
            df = fetch_fn(
                symbol=symbol,
                period=period,
                limit=max_per_page,
                start_time=current_start,
                end_time=end_ms,
            )

            if df.empty:
                logger.debug(
                    "  Page %d: empty response, ending pagination.", page,
                )
                break

            all_frames.append(df)

            # Advance start to 1ms past the last timestamp we received
            last_ts = df["timestamp"].iloc[-1]
            last_ms = int(last_ts.timestamp() * 1000)
            new_start = last_ms + 1

            if new_start <= current_start:
                # Safety: avoid infinite loop if API returns same data
                logger.warning(
                    "  Pagination stuck at %d, breaking.", current_start,
                )
                break

            current_start = new_start

            if len(df) < max_per_page:
                # Last page (fewer results than requested)
                break

            if page % 50 == 0:
                logger.info(
                    "  Page %d: fetched up to %s (%d rows so far)",
                    page, last_ts, sum(len(f) for f in all_frames),
                )

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        # Remove exact duplicates (overlapping pagination boundaries)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        ).reset_index(drop=True)
        return combined

    def _paginate_funding_rate(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """
        Paginate the /fapi/v1/fundingRate endpoint (max 1000 per page).

        Funding rate records are emitted every 8 hours, so pagination is
        coarser than the other endpoints.
        """
        all_frames: List[pd.DataFrame] = []
        current_start = start_ms
        max_per_page = 1000
        page = 0

        while current_start < end_ms:
            page += 1
            df = self.get_funding_rate(
                symbol=symbol,
                limit=max_per_page,
                start_time=current_start,
                end_time=end_ms,
            )

            if df.empty:
                break

            all_frames.append(df)

            last_ts = df["timestamp"].iloc[-1]
            last_ms = int(last_ts.timestamp() * 1000)
            new_start = last_ms + 1

            if new_start <= current_start:
                logger.warning(
                    "  Funding pagination stuck at %d, breaking.", current_start,
                )
                break

            current_start = new_start

            if len(df) < max_per_page:
                break

            if page % 20 == 0:
                logger.info(
                    "  Funding page %d: fetched up to %s (%d rows so far)",
                    page, last_ts, sum(len(f) for f in all_frames),
                )

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values(
            "timestamp"
        ).reset_index(drop=True)
        return combined

    # ------------------------------------------------------------------
    # Merge & feature computation
    # ------------------------------------------------------------------

    def _merge_dataframes(
        self,
        oi_df: pd.DataFrame,
        fr_df: pd.DataFrame,
        ls_df: pd.DataFrame,
        top_ls_df: pd.DataFrame,
        taker_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge all endpoint DataFrames on timestamp using outer join so no
        data is lost. Funding rates (8h interval) will have NaN between
        settlement times and get forward-filled.
        """
        # Start with OI as the base (usually the most granular)
        frames: List[pd.DataFrame] = []

        if not oi_df.empty:
            frames.append(oi_df.set_index("timestamp"))

        if not fr_df.empty:
            fr_subset = fr_df[["timestamp", "fundingRate", "markPrice"]].copy()
            frames.append(fr_subset.set_index("timestamp"))

        if not ls_df.empty:
            ls_renamed = ls_df.rename(columns={
                "longShortRatio": "longShortRatio",
                "longAccount": "longAccount",
                "shortAccount": "shortAccount",
            }).set_index("timestamp")
            frames.append(ls_renamed)

        if not top_ls_df.empty:
            top_renamed = top_ls_df.rename(columns={
                "longShortRatio": "topLongShortRatio",
                "longAccount": "topLongAccount",
                "shortAccount": "topShortAccount",
            }).set_index("timestamp")
            frames.append(top_renamed)

        if not taker_df.empty:
            frames.append(taker_df.set_index("timestamp"))

        if not frames:
            return pd.DataFrame()

        # Outer-join merge all frames
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.join(frame, how="outer")

        merged = merged.sort_index()

        # Forward-fill funding rate (8h granularity → fills into hourly rows)
        if "fundingRate" in merged.columns:
            merged["fundingRate"] = merged["fundingRate"].ffill()
        if "markPrice" in merged.columns:
            merged["markPrice"] = merged["markPrice"].ffill()

        # Reset index so timestamp is a column again
        merged = merged.reset_index()
        merged = merged.rename(columns={"index": "timestamp"})

        return merged

    def _compute_derived_features(
        self,
        df: pd.DataFrame,
        period: str,
    ) -> pd.DataFrame:
        """
        Compute derived columns from the raw merged data:
          - oi_change_4h, oi_change_24h
          - oi_price_divergence
          - funding_oi_divergence
          - ls_ratio_extreme
          - taker_imbalance
        """
        bars = PERIOD_TO_BARS.get(period, {"4h": 4, "24h": 24})
        bars_4h = bars["4h"]
        bars_24h = bars["24h"]

        # --- OI change ---
        if "sumOpenInterestValue" in df.columns:
            oi = df["sumOpenInterestValue"]
            df["oi_change_4h"] = oi.pct_change(periods=bars_4h)
            df["oi_change_24h"] = oi.pct_change(periods=bars_24h)
        else:
            df["oi_change_4h"] = np.nan
            df["oi_change_24h"] = np.nan

        # --- OI-Price divergence ---
        # When price moves one direction but OI moves the other, it signals
        # potential liquidation cascades or position unwinding.
        if "markPrice" in df.columns and "sumOpenInterestValue" in df.columns:
            price_change = df["markPrice"].pct_change()
            oi_change = df["sumOpenInterestValue"].pct_change()
            # Divergence = 1 when signs differ, 0 when same direction
            price_sign = np.sign(price_change)
            oi_sign = np.sign(oi_change)
            df["oi_price_divergence"] = (price_sign != oi_sign).astype(float)
            # Set NaN where either input is NaN
            mask = price_change.isna() | oi_change.isna()
            df.loc[mask, "oi_price_divergence"] = np.nan
        else:
            df["oi_price_divergence"] = np.nan

        # --- Funding-OI divergence ---
        # Signed score: +1 when funding > 0 but OI dropping (longs paying, exits)
        #               -1 when funding < 0 but OI dropping (shorts paying, exits)
        #                0 when no divergence
        if "fundingRate" in df.columns and "sumOpenInterestValue" in df.columns:
            fr = df["fundingRate"]
            oi_chg = df["sumOpenInterestValue"].pct_change()

            divergence = np.zeros(len(df), dtype=float)
            # Funding positive (longs pay shorts) + OI declining
            mask_pos = (fr > 0) & (oi_chg < 0)
            divergence[mask_pos] = 1.0
            # Funding negative (shorts pay longs) + OI declining
            mask_neg = (fr < 0) & (oi_chg < 0)
            divergence[mask_neg] = -1.0

            df["funding_oi_divergence"] = divergence
            # NaN where inputs are NaN
            nan_mask = fr.isna() | oi_chg.isna()
            df.loc[nan_mask, "funding_oi_divergence"] = np.nan
        else:
            df["funding_oi_divergence"] = np.nan

        # --- L/S ratio extreme (Z-score) ---
        if "longShortRatio" in df.columns:
            ls = df["longShortRatio"]
            ls_mean = ls.expanding(min_periods=20).mean()
            ls_std = ls.expanding(min_periods=20).std()
            # Guard division by zero
            df["ls_ratio_extreme"] = np.where(
                ls_std > 1e-10,
                (ls - ls_mean) / ls_std,
                0.0,
            )
            df.loc[ls.isna(), "ls_ratio_extreme"] = np.nan
        else:
            df["ls_ratio_extreme"] = np.nan

        # --- Taker imbalance ---
        # Normalized difference: (buy - sell) / (buy + sell), range [-1, 1]
        if "buyVol" in df.columns and "sellVol" in df.columns:
            buy = df["buyVol"]
            sell = df["sellVol"]
            total = buy + sell
            df["taker_imbalance"] = np.where(
                total > 1e-10,
                (buy - sell) / total,
                0.0,
            )
            nan_mask = buy.isna() | sell.isna()
            df.loc[nan_mask, "taker_imbalance"] = np.nan
        else:
            df["taker_imbalance"] = np.nan

        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"BinanceFuturesAPI(base_url={self.base_url!r}, "
            f"data_url={self.data_url!r})"
        )


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def _run_test(symbol: str = "BTCUSDT") -> None:
    """Quick smoke test — fetch current values from all endpoints."""
    print(f"=== BinanceFuturesAPI smoke test (symbol={symbol}) ===\n")
    api = BinanceFuturesAPI()

    # Current OI
    print("1. get_open_interest()...")
    oi = api.get_open_interest(symbol)
    print(f"   Open Interest: {oi:,.2f} contracts\n")

    # OI history (last 5)
    print("2. get_open_interest_history(limit=5)...")
    oi_hist = api.get_open_interest_history(symbol, limit=5)
    if not oi_hist.empty:
        print(oi_hist.to_string(index=False))
    else:
        print("   (empty)")
    print()

    # Funding rate (last 3)
    print("3. get_funding_rate(limit=3)...")
    fr = api.get_funding_rate(symbol, limit=3)
    if not fr.empty:
        print(fr.to_string(index=False))
    else:
        print("   (empty)")
    print()

    # Global L/S ratio (last 3)
    print("4. get_long_short_ratio(limit=3)...")
    ls = api.get_long_short_ratio(symbol, limit=3)
    if not ls.empty:
        print(ls.to_string(index=False))
    else:
        print("   (empty)")
    print()

    # Top trader L/S (last 3)
    print("5. get_top_trader_long_short_ratio(limit=3)...")
    top_ls = api.get_top_trader_long_short_ratio(symbol, limit=3)
    if not top_ls.empty:
        print(top_ls.to_string(index=False))
    else:
        print("   (empty)")
    print()

    # Taker buy/sell (last 3)
    print("6. get_taker_buy_sell_ratio(limit=3)...")
    taker = api.get_taker_buy_sell_ratio(symbol, limit=3)
    if not taker.empty:
        print(taker.to_string(index=False))
    else:
        print("   (empty)")
    print()

    # Composite current
    print("7. fetch_all_current()...")
    current = api.fetch_all_current(symbol)
    for k, v in current.items():
        print(f"   {k}: {v}")

    api.close()
    print("\n=== Test complete ===")


def _run_backfill(
    symbol: str,
    start: str,
    end: str,
    period: str,
    output: str,
) -> None:
    """Run historical backfill and save to parquet."""
    print(
        f"=== Historical backfill: {symbol} {start} to {end} "
        f"(period={period}) ===\n"
    )
    api = BinanceFuturesAPI()

    df = api.fetch_historical_range(
        symbol=symbol,
        start_date=start,
        end_date=end,
        period=period,
        save_intermediates=True,
        output_dir=str(Path(output).parent),
    )

    if df.empty:
        print("No data returned.")
    else:
        print(f"\nResult: {len(df)} rows x {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if output.endswith(".parquet"):
            df.to_parquet(output, index=False)
        else:
            df.to_csv(output, index=False)
        print(f"\nSaved to: {output}")

    api.close()
    print("\n=== Backfill complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binance Futures Public API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 bin/live/binance_futures_api.py --test\n"
            "  python3 bin/live/binance_futures_api.py --backfill "
            "--start 2023-01-01 --end 2023-12-31\n"
            "  python3 bin/live/binance_futures_api.py --backfill "
            "--start 2020-01-01 --end 2024-12-31 "
            "--output data/binance_futures_features.parquet\n"
        ),
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run smoke test against live endpoints",
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="Run historical backfill",
    )
    parser.add_argument(
        "--symbol", default="BTCUSDT",
        help="Trading pair (default: BTCUSDT)",
    )
    parser.add_argument(
        "--start", default="2020-01-01",
        help="Backfill start date YYYY-MM-DD (default: 2020-01-01)",
    )
    parser.add_argument(
        "--end", default="2024-12-31",
        help="Backfill end date YYYY-MM-DD (default: 2024-12-31)",
    )
    parser.add_argument(
        "--period", default="1h", choices=VALID_PERIODS,
        help="Kline period for historical data (default: 1h)",
    )
    parser.add_argument(
        "--output", default="data/binance_futures/binance_futures_features.parquet",
        help="Output file path (.parquet or .csv)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.test:
        _run_test(args.symbol)
    elif args.backfill:
        _run_backfill(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            period=args.period,
            output=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
