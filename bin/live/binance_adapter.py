#!/usr/bin/env python3
"""
Binance Futures REST adapter for Bull Machine live trading (read-only).

Uses CCXT binanceusdm for USD-M Futures public endpoints.
No authentication required -- read-only market data only, NO order placement.

Usage:
    # As a module
    from bin.live.binance_adapter import BinanceAdapter
    adapter = BinanceAdapter()
    df = adapter.fetch_ohlcv_1h(limit=100)

    # CLI test mode
    python3 bin/live/binance_adapter.py --test
"""

import sys
import logging
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Graceful import of dependencies
try:
    import ccxt
except ImportError:
    print("ERROR: ccxt is not installed.")
    print("Install it with:  pip install ccxt")
    print("Or:               pip install ccxt[async]")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is not installed.")
    print("Install it with:  pip install pandas")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "1h"
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # seconds -- exponential backoff
CANDLE_DURATION_MS = 3600 * 1000  # 1 hour in milliseconds


class BinanceAdapter:
    """
    Read-only Binance USD-M Futures adapter using CCXT.

    Provides market data access (OHLCV, funding rates, ticker) with
    automatic retry logic and rate limiting.  Does NOT place orders.
    """

    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        """
        Initialize the Binance adapter.

        Args:
            symbol: Trading pair in CCXT unified format.
                    Default: 'BTC/USDT:USDT' (BTCUSDT perpetual).
        """
        self.symbol = symbol
        self.exchange = ccxt.binanceusdm({
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
            },
        })
        logger.info(
            "BinanceAdapter initialized: symbol=%s, exchange=%s",
            self.symbol,
            self.exchange.id,
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
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                last_exc = exc
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
            except ccxt.ExchangeError as exc:
                # Non-retryable exchange error
                logger.error("%s exchange error: %s", description, exc)
                raise
        # All retries exhausted
        logger.error("%s failed after %d retries", description, MAX_RETRIES)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def fetch_ohlcv_1h(self, limit: int = 500) -> pd.DataFrame:
        """
        Fetch 1-hour OHLCV candles from Binance Futures.

        Only *completed* candles are returned (the current in-progress
        candle is excluded).

        Args:
            limit: Maximum number of candles to fetch (max 1500 per
                   Binance API; default 500).

        Returns:
            pd.DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, volume.
        """
        # Request one extra candle so we can drop the in-progress one
        fetch_limit = min(limit + 1, 1500)

        def _fetch():
            return self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=DEFAULT_TIMEFRAME,
                limit=fetch_limit,
            )

        raw = self._retry(_fetch, "fetch_ohlcv_1h")

        if not raw:
            logger.warning("fetch_ohlcv_1h returned no data")
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

        # Drop the last candle if it is still in progress.
        # A candle is in-progress if its open time + 1h > now.
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

    def fetch_funding_rate(self) -> float:
        """
        Fetch the latest funding rate for the perpetual contract.

        Returns:
            float: The most recent funding rate (e.g. 0.0001 = 0.01%).
        """
        def _fetch():
            # CCXT unified method for funding rate
            rates = self.exchange.fetch_funding_rate(self.symbol)
            return rates

        data = self._retry(_fetch, "fetch_funding_rate")

        funding_rate = float(data.get("fundingRate", 0.0) or 0.0)
        logger.info("fetch_funding_rate: %.6f", funding_rate)
        return funding_rate

    def fetch_ticker(self) -> dict:
        """
        Fetch current ticker data.

        Returns:
            dict with keys: price, bid, ask, spread.
        """
        def _fetch():
            return self.exchange.fetch_ticker(self.symbol)

        data = self._retry(_fetch, "fetch_ticker")

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
            "fetch_ticker: price=%.2f  bid=%.2f  ask=%.2f  spread=%.2f",
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
    print("Bull Machine -- Binance Adapter Test")
    print("=" * 60)

    adapter = BinanceAdapter()

    # 1. Fetch 5 completed candles
    print("\n--- OHLCV (5 completed 1H candles) ---")
    df = adapter.fetch_ohlcv_1h(limit=5)
    if df.empty:
        print("WARNING: No candles returned!")
    else:
        print(df.to_string())
        print(f"\nShape: {df.shape}")
        print(f"Index dtype: {df.index.dtype}")
        print(f"Columns: {list(df.columns)}")

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
        description="Binance Futures adapter for Bull Machine (read-only)."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run connectivity test: fetch 5 candles, funding rate, and ticker.",
    )
    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
        help=f"Trading pair in CCXT format (default: {DEFAULT_SYMBOL}).",
    )
    args = parser.parse_args()

    if args.test:
        _run_test()
    else:
        parser.print_help()
