#!/usr/bin/env python3
"""
Coinbase Advanced Trade adapter for Bull Machine live trading.

Uses coinbase-advanced-py SDK for BTC perpetual futures (BTC-PERP-INTX).
Drop-in replacement for BinanceAdapter with identical interface.

Authentication priority:
    1. Environment variables: COINBASE_API_KEY, COINBASE_API_SECRET
    2. Key file: ~/.coinbase/cdp_api_key.json
    3. No auth (public endpoints only -- suitable for paper trading)

Usage:
    # As a module
    from bin.live.coinbase_client import CoinbaseAdapter
    adapter = CoinbaseAdapter()
    df = adapter.fetch_ohlcv_1h(limit=100)

    # CLI test mode
    python3 bin/live/coinbase_client.py --test
"""

import sys
import os
import json
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Graceful import of dependencies
# ---------------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.")
    print("Install it with:  pip install pandas")
    sys.exit(1)

# Primary SDK: coinbase-advanced-py
_HAS_COINBASE_SDK = False
try:
    from coinbase.rest import RESTClient as CoinbaseRESTClient
    _HAS_COINBASE_SDK = True
except ImportError:
    CoinbaseRESTClient = None  # type: ignore[assignment,misc]

# Fallback: ccxt
_HAS_CCXT = False
try:
    import ccxt
    _HAS_CCXT = True
except ImportError:
    ccxt = None  # type: ignore[assignment]

if not _HAS_COINBASE_SDK and not _HAS_CCXT:
    print("ERROR: Neither coinbase-advanced-py nor ccxt is installed.")
    print("Install one of:")
    print("  pip install coinbase-advanced-py")
    print("  pip install ccxt")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SYMBOL = "BTC/USDT:USDT"           # CCXT format (fallback)
COINBASE_PRODUCT_ID = "BTC-PERP-INTX"      # Coinbase perpetual futures
COINBASE_SPOT_PRODUCT = "BTC-USD"           # Spot fallback if perps unavailable
DEFAULT_TIMEFRAME = "1h"
GRANULARITY_ONE_HOUR = "ONE_HOUR"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]                   # seconds -- exponential backoff
CANDLE_DURATION_MS = 3600 * 1000            # 1 hour in milliseconds
CANDLE_DURATION_S = 3600                    # 1 hour in seconds
MAX_CANDLES_PER_REQUEST = 300               # Coinbase API limit per request
KEY_FILE_PATH = Path.home() / ".coinbase" / "cdp_api_key.json"


class CoinbaseAdapter:
    """
    Read-only Coinbase Advanced Trade adapter.

    Provides market data access (OHLCV, funding rates, ticker) with
    automatic retry logic.  Does NOT place orders.

    Drop-in replacement for BinanceAdapter -- identical public interface.
    """

    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        """
        Initialize the Coinbase adapter.

        Args:
            symbol: Trading pair in CCXT unified format (used only for
                    ccxt fallback).  Default: 'BTC/USDT:USDT'.
        """
        self.symbol = symbol
        self.product_id = COINBASE_PRODUCT_ID
        self._using_sdk = False
        self._using_ccxt = False
        self._authenticated = False

        if _HAS_COINBASE_SDK:
            self._init_coinbase_sdk()
        elif _HAS_CCXT:
            self._init_ccxt_fallback()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_real_credential(value: str) -> bool:
        """Return False for placeholder/template values."""
        if not value:
            return False
        placeholders = {"YOUR_ORG_ID", "YOUR_KEY_ID", "YOUR_KEY_HERE", "CHANGE_ME"}
        return not any(p in value for p in placeholders)

    def _init_coinbase_sdk(self):
        """Initialize using the official coinbase-advanced-py SDK."""
        api_key = os.environ.get("COINBASE_API_KEY", "")
        api_secret = os.environ.get("COINBASE_API_SECRET", "")

        try:
            if self._is_real_credential(api_key) and self._is_real_credential(api_secret):
                # Option 1: Environment variables
                self.client = CoinbaseRESTClient(
                    api_key=api_key,
                    api_secret=api_secret,
                )
                self._authenticated = True
                logger.info(
                    "CoinbaseAdapter initialized (SDK, authenticated via env vars): "
                    "product=%s",
                    self.product_id,
                )
            elif KEY_FILE_PATH.exists():
                # Option 2: Key file
                self.client = CoinbaseRESTClient(
                    key_file=str(KEY_FILE_PATH),
                )
                self._authenticated = True
                logger.info(
                    "CoinbaseAdapter initialized (SDK, authenticated via key file %s): "
                    "product=%s",
                    KEY_FILE_PATH,
                    self.product_id,
                )
            else:
                # Option 3: No auth -- public endpoints only
                self.client = CoinbaseRESTClient()
                self._authenticated = False
                logger.info(
                    "CoinbaseAdapter initialized (SDK, unauthenticated -- public only): "
                    "product=%s",
                    self.product_id,
                )

            self._using_sdk = True

        except Exception as exc:
            logger.warning(
                "Failed to initialize coinbase-advanced-py SDK: %s  "
                "-- falling back to ccxt",
                exc,
            )
            if _HAS_CCXT:
                self._init_ccxt_fallback()
            else:
                raise RuntimeError(
                    f"Coinbase SDK initialization failed and ccxt not available: {exc}"
                ) from exc

    def _init_ccxt_fallback(self):
        """Initialize using ccxt as a fallback."""
        logger.info("Initializing CoinbaseAdapter with ccxt fallback")

        api_key = os.environ.get("COINBASE_API_KEY")
        api_secret = os.environ.get("COINBASE_API_SECRET")

        config = {
            "enableRateLimit": True,
        }

        if api_key and api_secret:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            self._authenticated = True

        self.exchange = ccxt.coinbase(config)
        self._using_ccxt = True
        logger.info(
            "CoinbaseAdapter initialized (ccxt fallback, auth=%s): symbol=%s",
            self._authenticated,
            self.symbol,
        )

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------
    def _retry(self, func, description: str):
        """
        Execute *func* with exponential-backoff retries.

        Args:
            func: Callable (no args) to execute.
            description: Human-readable label for log messages.

        Returns:
            The return value of *func* on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return func()
            except Exception as exc:
                last_exc = exc
                # Determine if retryable
                retryable = self._is_retryable(exc)
                if not retryable:
                    logger.error("%s non-retryable error: %s", description, exc)
                    raise

                delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                logger.warning(
                    "%s failed (attempt %d/%d): %s  -- retrying in %ds",
                    description,
                    attempt + 1,
                    MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)

        # All retries exhausted
        logger.error("%s failed after %d retries", description, MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Determine whether an exception is retryable."""
        # Network / timeout errors are retryable
        exc_name = type(exc).__name__.lower()
        retryable_patterns = [
            "timeout", "connection", "network", "unavailable",
            "ratelimit", "rate_limit", "toomanyrequests",
            "502", "503", "504",
        ]
        exc_str = str(exc).lower()
        for pattern in retryable_patterns:
            if pattern in exc_name or pattern in exc_str:
                return True

        # ccxt-specific retryable errors
        if _HAS_CCXT:
            if isinstance(exc, (ccxt.NetworkError, ccxt.ExchangeNotAvailable)):
                return True

        return False

    # ------------------------------------------------------------------
    # Public methods  (identical interface to BinanceAdapter)
    # ------------------------------------------------------------------
    def fetch_ohlcv_1h(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch 1-hour OHLCV candles from Coinbase.

        Only *completed* candles are returned (the current in-progress
        candle is excluded).

        Args:
            limit: Maximum number of candles to fetch (default 500).

        Returns:
            pd.DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume.
        """
        if self._using_sdk:
            return self._fetch_ohlcv_sdk(limit)
        else:
            return self._fetch_ohlcv_ccxt(limit)

    def _fetch_ohlcv_sdk(self, limit: int) -> pd.DataFrame:
        """Fetch OHLCV using coinbase-advanced-py SDK with pagination."""
        # We request one extra candle so we can drop the in-progress one
        total_needed = limit + 1
        all_candles = []

        # Calculate time window: work backwards from now
        now_s = int(time.time())

        # Paginate: Coinbase allows max ~300 candles per request
        remaining = total_needed
        end_s = now_s

        while remaining > 0:
            batch_size = min(remaining, MAX_CANDLES_PER_REQUEST)
            # start = end - (batch_size * 3600) with a small buffer
            start_s = end_s - (batch_size * CANDLE_DURATION_S)

            def _fetch(s=str(start_s), e=str(end_s)):
                if self._authenticated:
                    return self.client.get_candles(
                        product_id=self.product_id,
                        start=s,
                        end=e,
                        granularity=GRANULARITY_ONE_HOUR,
                    )
                else:
                    # Public endpoint (no auth required)
                    return self.client.get_public_candles(
                        product_id=self.product_id,
                        start=s,
                        end=e,
                        granularity=GRANULARITY_ONE_HOUR,
                    )

            try:
                response = self._retry(_fetch, f"fetch_ohlcv_sdk (batch ending {end_s})")
            except Exception as exc:
                logger.warning(
                    "Failed to fetch candles for product %s, trying spot %s: %s",
                    self.product_id,
                    COINBASE_SPOT_PRODUCT,
                    exc,
                )
                # Fallback to spot product if perps unavailable
                try:
                    def _fetch_spot(s=str(start_s), e=str(end_s)):
                        _method = (
                            self.client.get_candles
                            if self._authenticated
                            else self.client.get_public_candles
                        )
                        return _method(
                            product_id=COINBASE_SPOT_PRODUCT,
                            start=s,
                            end=e,
                            granularity=GRANULARITY_ONE_HOUR,
                        )
                    response = self._retry(
                        _fetch_spot,
                        f"fetch_ohlcv_sdk spot fallback (batch ending {end_s})",
                    )
                    if not all_candles:
                        logger.info(
                            "Using spot product %s as perps fallback",
                            COINBASE_SPOT_PRODUCT,
                        )
                except Exception as exc2:
                    logger.error("Spot fallback also failed: %s", exc2)
                    break

            # Parse response -- SDK returns object with 'candles' attribute
            candles = self._extract_candles(response)
            if not candles:
                logger.debug("No candles in batch ending at %d", end_s)
                break

            all_candles.extend(candles)
            remaining -= len(candles)

            # Move end_s backwards for next batch
            # Find the earliest candle start in this batch
            earliest_start = min(int(c.get("start", c.get("timestamp", 0))) for c in candles)
            end_s = earliest_start

            # Safety: if we got fewer candles than expected, no more data
            if len(candles) < batch_size:
                break

        if not all_candles:
            logger.warning("fetch_ohlcv_1h returned no data")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Build DataFrame from candle data
        df = self._candles_to_dataframe(all_candles)

        # Remove duplicates (safety)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # Drop the last candle if it is still in progress
        now_ms = int(time.time() * 1000)
        if len(df) > 0:
            last_open_ms = int(df.index[-1].timestamp() * 1000)
            if last_open_ms + CANDLE_DURATION_MS > now_ms:
                df = df.iloc[:-1]

        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit)

        logger.info(
            "fetch_ohlcv_1h: %d completed candles  [%s .. %s]",
            len(df),
            df.index[0] if len(df) else "N/A",
            df.index[-1] if len(df) else "N/A",
        )
        return df

    @staticmethod
    def _extract_candles(response) -> list:
        """
        Extract candle list from SDK response.

        The SDK may return the data in different formats depending on
        version and endpoint.  This handles the known variants.
        """
        # Case 1: response has a 'candles' attribute (object)
        if hasattr(response, "candles"):
            candles_raw = response.candles
            if isinstance(candles_raw, list):
                # Each item may be a dict or an object with attributes
                result = []
                for c in candles_raw:
                    if isinstance(c, dict):
                        result.append(c)
                    else:
                        # Object with attributes
                        result.append({
                            "start": getattr(c, "start", "0"),
                            "open": getattr(c, "open", "0"),
                            "high": getattr(c, "high", "0"),
                            "low": getattr(c, "low", "0"),
                            "close": getattr(c, "close", "0"),
                            "volume": getattr(c, "volume", "0"),
                        })
                return result

        # Case 2: response is a dict with 'candles' key
        if isinstance(response, dict):
            candles_raw = response.get("candles", [])
            if isinstance(candles_raw, list):
                return candles_raw

        # Case 3: response itself is a list
        if isinstance(response, list):
            return response

        logger.warning("Unexpected candle response format: %s", type(response))
        return []

    @staticmethod
    def _candles_to_dataframe(candles: list) -> pd.DataFrame:
        """
        Convert a list of candle dicts to a pandas DataFrame.

        Coinbase candle fields:
            start: UNIX timestamp (seconds, as string)
            open, high, low, close: price strings
            volume: volume string
        """
        rows = []
        for c in candles:
            try:
                ts = int(c.get("start", c.get("timestamp", 0)))
                rows.append({
                    "timestamp": ts,
                    "open": float(c.get("open", 0)),
                    "high": float(c.get("high", 0)),
                    "low": float(c.get("low", 0)),
                    "close": float(c.get("close", 0)),
                    "volume": float(c.get("volume", 0)),
                })
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping malformed candle: %s (%s)", c, exc)
                continue

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")
        df = df.sort_index()

        # --- Data Sanity Checks ---
        n_before = len(df)

        # Reject zero-volume candles
        zero_vol = df["volume"] <= 0
        if zero_vol.any():
            logger.warning("SANITY: Dropping %d zero-volume candles", zero_vol.sum())
            df = df[~zero_vol]

        # Reject invalid OHLC ordering (high must be >= low, prices must be positive)
        bad_ohlc = (df["high"] < df["low"]) | (df["close"] <= 0) | (df["open"] <= 0)
        if bad_ohlc.any():
            logger.warning("SANITY: Dropping %d candles with invalid OHLC", bad_ohlc.sum())
            df = df[~bad_ohlc]

        # Reject >15% single-bar price moves (flash crash / data error)
        if len(df) >= 2:
            pct_change = df["close"].pct_change().abs()
            flash = pct_change > 0.15
            if flash.any():
                logger.warning("SANITY: Dropping %d candles with >15%% price move", flash.sum())
                df = df[~flash]

        if len(df) < n_before:
            logger.info("SANITY: Kept %d of %d candles after validation", len(df), n_before)

        return df

    def _fetch_ohlcv_ccxt(self, limit: int) -> pd.DataFrame:
        """Fetch OHLCV using ccxt as fallback."""
        # Request one extra candle so we can drop the in-progress one
        fetch_limit = min(limit + 1, 1500)

        def _fetch():
            return self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=DEFAULT_TIMEFRAME,
                limit=fetch_limit,
            )

        raw = self._retry(_fetch, "fetch_ohlcv_1h (ccxt)")

        if not raw:
            logger.warning("fetch_ohlcv_1h (ccxt) returned no data")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        # Build DataFrame
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df.sort_index()

        # Remove duplicates (safety)
        df = df[~df.index.duplicated(keep="last")]

        # Drop the last candle if it is still in progress
        now_ms = int(time.time() * 1000)
        if len(df) > 0:
            last_open_ms = int(df.index[-1].timestamp() * 1000)
            if last_open_ms + CANDLE_DURATION_MS > now_ms:
                df = df.iloc[:-1]

        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit)

        logger.info(
            "fetch_ohlcv_1h (ccxt): %d completed candles  [%s .. %s]",
            len(df),
            df.index[0] if len(df) else "N/A",
            df.index[-1] if len(df) else "N/A",
        )
        return df

    def fetch_funding_rate(self) -> float:
        """
        Fetch the latest funding rate for the perpetual contract.

        NOTE: The coinbase-advanced-py SDK does NOT currently expose a
        funding rate endpoint.  This returns 0.0 as a placeholder.
        A separate coinbase_funding.py module will provide real funding
        rates via the Coinbase INTX API.

        Returns:
            float: The most recent funding rate (e.g. 0.0001 = 0.01%).
        """
        if self._using_ccxt:
            return self._fetch_funding_ccxt()

        # SDK path: funding rate not available in coinbase-advanced-py
        logger.warning(
            "fetch_funding_rate: coinbase-advanced-py SDK does not expose "
            "funding rates for %s.  Returning 0.0.  "
            "Use coinbase_funding.py for real funding data via INTX API.",
            self.product_id,
        )
        return 0.0

    def _fetch_funding_ccxt(self) -> float:
        """Fetch funding rate via ccxt (if supported)."""
        try:
            def _fetch():
                return self.exchange.fetch_funding_rate(self.symbol)

            data = self._retry(_fetch, "fetch_funding_rate (ccxt)")
            funding_rate = float(data.get("fundingRate", 0.0) or 0.0)
            logger.info("fetch_funding_rate (ccxt): %.6f", funding_rate)
            return funding_rate
        except Exception as exc:
            logger.warning(
                "fetch_funding_rate (ccxt) failed: %s  -- returning 0.0",
                exc,
            )
            return 0.0

    def fetch_ticker(self) -> dict:
        """
        Fetch current ticker data.

        Returns:
            dict with keys: price, bid, ask, spread.
        """
        if self._using_sdk:
            return self._fetch_ticker_sdk()
        else:
            return self._fetch_ticker_ccxt()

    def _fetch_ticker_sdk(self) -> dict:
        """Fetch ticker using coinbase-advanced-py SDK."""
        if self._authenticated:
            def _fetch():
                return self.client.get_best_bid_ask(
                    product_ids=[self.product_id],
                )
        else:
            # Public endpoint: get_public_product_book returns order book
            def _fetch():
                return self.client.get_public_product_book(
                    product_id=self.product_id,
                    limit=1,
                )

        try:
            response = self._retry(_fetch, "fetch_ticker_sdk")
        except Exception as exc:
            logger.warning(
                "fetch_ticker for %s failed, trying spot %s: %s",
                self.product_id,
                COINBASE_SPOT_PRODUCT,
                exc,
            )
            # Fallback to spot
            if self._authenticated:
                def _fetch_spot():
                    return self.client.get_best_bid_ask(
                        product_ids=[COINBASE_SPOT_PRODUCT],
                    )
            else:
                def _fetch_spot():
                    return self.client.get_public_product_book(
                        product_id=COINBASE_SPOT_PRODUCT,
                        limit=1,
                    )
            response = self._retry(_fetch_spot, "fetch_ticker_sdk spot fallback")

        # Parse response
        bid, ask, last_price = 0.0, 0.0, 0.0

        try:
            bids, asks = None, None

            # Try to extract from dict for convenience
            resp_dict = (
                response.to_dict()
                if hasattr(response, "to_dict")
                else response if isinstance(response, dict)
                else {}
            )

            # get_public_product_book: has "pricebook" (singular) + "mid_market"
            if "mid_market" in resp_dict or "pricebook" in resp_dict:
                last_price = float(resp_dict.get("mid_market", 0) or 0)
                book = resp_dict.get("pricebook", {})
                bids = book.get("bids", [])
                asks = book.get("asks", [])
            else:
                # get_best_bid_ask: has "pricebooks" (plural, list)
                pricebooks = resp_dict.get("pricebooks", [])
                if pricebooks and len(pricebooks) > 0:
                    book = pricebooks[0]
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])

            def _extract_price(entries):
                if entries and len(entries) > 0:
                    entry = entries[0]
                    if isinstance(entry, dict):
                        return float(entry.get("price", 0))
                    elif hasattr(entry, "price"):
                        return float(entry.price)
                return 0.0

            bid = _extract_price(bids)
            ask = _extract_price(asks)
            if last_price == 0.0:
                last_price = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)

        except Exception as exc:
            logger.error("Error parsing ticker response: %s", exc)

        spread = ask - bid if (bid > 0 and ask > 0) else 0.0

        ticker = {
            "price": last_price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
        }
        logger.info(
            "fetch_ticker: price=%.2f  bid=%.2f  ask=%.2f  spread=%.2f",
            last_price,
            bid,
            ask,
            spread,
        )
        return ticker

    def _fetch_ticker_ccxt(self) -> dict:
        """Fetch ticker using ccxt as fallback."""
        def _fetch():
            return self.exchange.fetch_ticker(self.symbol)

        data = self._retry(_fetch, "fetch_ticker (ccxt)")

        bid = float(data.get("bid") or 0.0)
        ask = float(data.get("ask") or 0.0)
        last_price = float(data.get("last") or 0.0)
        spread = ask - bid if (bid > 0 and ask > 0) else 0.0

        ticker = {
            "price": last_price,
            "bid": bid,
            "ask": ask,
            "spread": spread,
        }
        logger.info(
            "fetch_ticker (ccxt): price=%.2f  bid=%.2f  ask=%.2f  spread=%.2f",
            last_price,
            bid,
            ask,
            spread,
        )
        return ticker

    def is_candle_closed(self) -> bool:
        """
        Check whether the current hour's candle has closed.

        Returns True when the current minute > 0 (i.e., we have moved
        past the hour boundary and the previous candle is complete).
        Useful for scheduling feature computation right after candle close.

        Returns:
            bool: True if the latest hourly candle is complete.
        """
        now = datetime.now(timezone.utc)
        closed = now.minute > 0 or now.second > 5
        logger.debug("is_candle_closed: minute=%d -> %s", now.minute, closed)
        return closed


# -----------------------------------------------------------------------
# CLI test mode
# -----------------------------------------------------------------------
def _run_test():
    """Run a quick connectivity and data sanity test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("Bull Machine -- Coinbase Adapter Test")
    print("=" * 60)

    adapter = CoinbaseAdapter()

    # Report backend in use
    if adapter._using_sdk:
        auth_status = "authenticated" if adapter._authenticated else "unauthenticated"
        print(f"Backend: coinbase-advanced-py SDK ({auth_status})")
    elif adapter._using_ccxt:
        auth_status = "authenticated" if adapter._authenticated else "unauthenticated"
        print(f"Backend: ccxt fallback ({auth_status})")
    print(f"Product: {adapter.product_id}")

    # 1. Fetch 5 completed candles
    print("\n--- OHLCV (5 completed 1H candles) ---")
    try:
        df = adapter.fetch_ohlcv_1h(limit=5)
        if df.empty:
            print("WARNING: No candles returned!")
        else:
            print(df.to_string())
            print(f"\nShape: {df.shape}")
            print(f"Index dtype: {df.index.dtype}")
            print(f"Columns: {list(df.columns)}")
    except Exception as exc:
        print(f"Failed to fetch OHLCV: {exc}")

    # 2. Funding rate
    print("\n--- Funding Rate ---")
    try:
        rate = adapter.fetch_funding_rate()
        print(f"Funding rate: {rate:.6f}  ({rate * 100:.4f}%)")
    except Exception as exc:
        print(f"Failed to fetch funding rate: {exc}")

    # 3. Ticker
    print("\n--- Ticker ---")
    try:
        ticker = adapter.fetch_ticker()
        for k, v in ticker.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"Failed to fetch ticker: {exc}")

    # 4. Candle closed check
    print("\n--- Candle Closed ---")
    closed = adapter.is_candle_closed()
    now = datetime.now(timezone.utc)
    print(f"UTC now: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Candle closed: {closed}")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coinbase Advanced Trade adapter for Bull Machine (read-only)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run connectivity test: fetch 5 candles, funding rate, and ticker.",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Trading pair in CCXT format for fallback (default: {DEFAULT_SYMBOL}).",
    )
    parser.add_argument(
        "--product",
        default=COINBASE_PRODUCT_ID,
        help=f"Coinbase product ID (default: {COINBASE_PRODUCT_ID}).",
    )
    args = parser.parse_args()

    if args.test:
        _run_test()
    else:
        parser.print_help()
