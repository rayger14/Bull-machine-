#!/usr/bin/env python3
"""
Coinbase INTX (International Exchange) funding rate client for Bull Machine.

Fetches perpetual funding rates from the Coinbase International Exchange API.
This is a SEPARATE module from the main Coinbase adapter because funding rates
use the INTX API, not the Advanced Trade API.

Funding mechanics at Coinbase INTX:
  - Calculated hourly (average premium/discount measured every 3 minutes)
  - Smoothing: (Premium * 0.75) + (Previous_Funding_Rate * 0.25)
  - Settlement: twice daily (midday and end-of-day)

Endpoints:
  - Primary:  GET https://api.international.coinbase.com/api/v1/instruments/{instrument}/funding
  - Fallback: GET https://api.exchange.fairx.net/rest/funding-rate

Usage:
    # As a module
    from bin.live.coinbase_funding import CoinbaseFundingClient
    client = CoinbaseFundingClient()
    rate = client.get_current_funding_rate()

    # CLI test mode
    python3 bin/live/coinbase_funding.py --test
"""

import sys
import os
import logging
import time
import hmac
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Graceful import of dependencies
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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INSTRUMENT = "BTC-PERP"
DEFAULT_FUNDING_RATE = 0.0001  # 0.01% -- sensible fallback when API unavailable

INTX_BASE_URL = "https://api.international.coinbase.com"
INTX_FUNDING_PATH = "/api/v1/instruments/{instrument}/funding"

FAIRX_BASE_URL = "https://api.exchange.fairx.net"
FAIRX_FUNDING_PATH = "/rest/funding-rate"

MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # seconds -- exponential backoff

# Coinbase INTX funding is calculated hourly
FUNDING_PERIODS_PER_DAY = 24
FUNDING_SETTLEMENTS_PER_DAY = 2

# Request timeout in seconds
REQUEST_TIMEOUT = 15


class CoinbaseFundingClient:
    """
    Coinbase International Exchange (INTX) funding rate client.

    Fetches BTC-PERP funding rates from the Coinbase INTX API.  Falls back
    to the FairX derivatives endpoint if the primary is unavailable.  Can
    operate without authentication for public endpoints, or with HMAC-SHA256
    auth when credentials are provided.
    """

    def __init__(self, api_key=None, api_secret=None, passphrase=None):
        """
        Initialize the INTX funding rate client.

        Args:
            api_key:    Coinbase INTX API key.  Falls back to
                        COINBASE_INTX_API_KEY env var.
            api_secret: Coinbase INTX API secret.  Falls back to
                        COINBASE_INTX_API_SECRET env var.
            passphrase: Coinbase INTX passphrase.  Falls back to
                        COINBASE_INTX_PASSPHRASE env var.
        """
        self.api_key = api_key or os.environ.get("COINBASE_INTX_API_KEY")
        self.api_secret = api_secret or os.environ.get("COINBASE_INTX_API_SECRET")
        self.passphrase = passphrase or os.environ.get("COINBASE_INTX_PASSPHRASE")

        self._has_auth = bool(self.api_key and self.api_secret and self.passphrase)
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

        if self._has_auth:
            logger.info(
                "CoinbaseFundingClient initialized WITH authentication "
                "(key=%.8s...)",
                self.api_key,
            )
        else:
            logger.info(
                "CoinbaseFundingClient initialized WITHOUT authentication "
                "(public endpoints only)"
            )

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------
    def _sign_request(self, method: str, path: str, body: str = "") -> dict:
        """
        Generate HMAC-SHA256 authentication headers for INTX API.

        Args:
            method: HTTP method (GET, POST, etc.).
            path:   Request path (e.g. /api/v1/instruments/BTC-PERP/funding).
            body:   Request body string (empty for GET requests).

        Returns:
            dict of authentication headers.
        """
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase,
            "CB-ACCESS-SIGN": signature,
        }

    # ------------------------------------------------------------------
    # Retry wrapper
    # ------------------------------------------------------------------
    def _retry(self, func, description: str):
        """
        Execute *func* with exponential-backoff retries.

        Args:
            func:        Callable (no args) to execute.
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
            except (requests.ConnectionError, requests.Timeout) as exc:
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
            except requests.HTTPError as exc:
                # Non-retryable HTTP error (4xx)
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    logger.error("%s client error: %s", description, exc)
                    raise
                # Server error (5xx) -- retry
                last_exc = exc
                delay = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else RETRY_DELAYS[-1]
                logger.warning(
                    "%s server error (attempt %d/%d): %s  -- retrying in %ds",
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

    # ------------------------------------------------------------------
    # Internal fetch methods
    # ------------------------------------------------------------------
    def _fetch_intx_funding(self, instrument: str, limit: int = 25, offset: int = 0) -> list:
        """
        Fetch funding rates from the Coinbase INTX API.

        Args:
            instrument: Instrument ID (e.g. 'BTC-PERP').
            limit:      Number of results (max 100).
            offset:     Pagination offset.

        Returns:
            List of funding rate records from the API.

        Raises:
            requests.HTTPError: On non-retryable API errors.
        """
        path = INTX_FUNDING_PATH.format(instrument=instrument)
        url = INTX_BASE_URL + path

        params = {
            "result_limit": min(limit, 100),
            "result_offset": offset,
        }

        headers = {}
        if self._has_auth:
            # Build query string for signature
            query_string = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            sign_path = path + "?" + query_string
            headers = self._sign_request("GET", sign_path)

        def _fetch():
            resp = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        data = self._retry(_fetch, f"fetch_intx_funding({instrument})")

        # The response may be a single dict or a list of dicts
        if isinstance(data, dict):
            # Wrap single result or extract from nested key
            if "results" in data:
                return data["results"]
            return [data]
        return data if isinstance(data, list) else [data]

    def _fetch_fairx_funding(self, instrument: str) -> list:
        """
        Fetch funding rates from the FairX / Coinbase Derivatives API
        (fallback endpoint).

        Args:
            instrument: Instrument ID (e.g. 'BTC-PERP').

        Returns:
            List of funding rate records from the API.
        """
        url = FAIRX_BASE_URL + FAIRX_FUNDING_PATH
        params = {"instrument": instrument}

        headers = {}
        if self._has_auth:
            sign_path = FAIRX_FUNDING_PATH + "?" + f"instrument={instrument}"
            headers = self._sign_request("GET", sign_path)

        def _fetch():
            resp = self._session.get(
                url,
                params=params,
                headers=headers,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()

        data = self._retry(_fetch, f"fetch_fairx_funding({instrument})")

        if isinstance(data, dict):
            if "results" in data:
                return data["results"]
            return [data]
        return data if isinstance(data, list) else [data]

    def _parse_funding_record(self, record: dict) -> dict:
        """
        Normalize a raw funding rate record into a standard format.

        Args:
            record: Raw dict from the API response.

        Returns:
            Normalized dict with funding_rate, mark_price, event_time,
            and annualized_rate.
        """
        # Parse funding rate -- may be string or float
        raw_rate = record.get("funding_rate", record.get("fundingRate", 0))
        try:
            funding_rate = float(raw_rate)
        except (ValueError, TypeError):
            funding_rate = DEFAULT_FUNDING_RATE
            logger.warning(
                "Could not parse funding_rate=%r, using default %.6f",
                raw_rate,
                DEFAULT_FUNDING_RATE,
            )

        # Parse mark price
        raw_price = record.get("mark_price", record.get("markPrice", 0))
        try:
            mark_price = float(raw_price)
        except (ValueError, TypeError):
            mark_price = 0.0

        # Parse event time
        raw_time = record.get("event_time", record.get("eventTime", None))
        if raw_time:
            try:
                # Handle ISO 8601 format: "2023-03-16T23:59:53.000Z"
                if isinstance(raw_time, str):
                    event_time = datetime.fromisoformat(
                        raw_time.replace("Z", "+00:00")
                    )
                else:
                    # Unix timestamp in seconds or milliseconds
                    ts = float(raw_time)
                    if ts > 1e12:
                        ts = ts / 1000.0
                    event_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, TypeError):
                event_time = datetime.now(timezone.utc)
        else:
            event_time = datetime.now(timezone.utc)

        # Annualized rate: hourly rate * 24 hours * 365 days
        annualized_rate = funding_rate * FUNDING_PERIODS_PER_DAY * 365

        return {
            "funding_rate": funding_rate,
            "mark_price": mark_price,
            "event_time": event_time,
            "annualized_rate": annualized_rate,
        }

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def get_current_funding_rate(self, instrument: str = DEFAULT_INSTRUMENT) -> dict:
        """
        Get the most recent funding rate.

        Tries the INTX API first, falls back to FairX, and finally
        returns a sensible default if both fail.

        Args:
            instrument: Instrument ID (default: 'BTC-PERP').

        Returns:
            dict with keys:
                funding_rate   (float): e.g., 0.000154
                mark_price     (float): e.g., 97000.63
                event_time  (datetime): UTC timestamp
                annualized_rate(float): funding_rate * 24 * 365
        """
        # Try INTX primary endpoint
        try:
            records = self._fetch_intx_funding(instrument, limit=1)
            if records:
                parsed = self._parse_funding_record(records[0])
                logger.info(
                    "get_current_funding_rate: rate=%.6f  mark=%.2f  "
                    "annualized=%.4f  source=INTX",
                    parsed["funding_rate"],
                    parsed["mark_price"],
                    parsed["annualized_rate"],
                )
                return parsed
        except Exception as exc:
            logger.warning(
                "INTX funding endpoint failed: %s -- trying FairX fallback",
                exc,
            )

        # Try FairX fallback endpoint
        try:
            records = self._fetch_fairx_funding(instrument)
            if records:
                parsed = self._parse_funding_record(records[0])
                logger.info(
                    "get_current_funding_rate: rate=%.6f  mark=%.2f  "
                    "annualized=%.4f  source=FairX",
                    parsed["funding_rate"],
                    parsed["mark_price"],
                    parsed["annualized_rate"],
                )
                return parsed
        except Exception as exc:
            logger.warning(
                "FairX funding endpoint also failed: %s -- using default",
                exc,
            )

        # All endpoints failed -- return sensible default
        default = {
            "funding_rate": DEFAULT_FUNDING_RATE,
            "mark_price": 0.0,
            "event_time": datetime.now(timezone.utc),
            "annualized_rate": DEFAULT_FUNDING_RATE * FUNDING_PERIODS_PER_DAY * 365,
        }
        logger.warning(
            "get_current_funding_rate: using DEFAULT rate=%.6f  "
            "(all endpoints unavailable)",
            DEFAULT_FUNDING_RATE,
        )
        return default

    def get_funding_history(
        self,
        instrument: str = DEFAULT_INSTRUMENT,
        limit: int = 100,
    ) -> pd.DataFrame:
        """
        Get historical funding rates.

        Paginates through the INTX API to collect up to *limit* records.
        Falls back to FairX if INTX is unavailable.

        Args:
            instrument: Instrument ID (default: 'BTC-PERP').
            limit:      Maximum number of records to fetch (default 100).

        Returns:
            pd.DataFrame with columns:
                event_time      (datetime, UTC)
                funding_rate    (float)
                mark_price      (float)
                annualized_rate (float)
            Sorted by event_time ascending.  Empty DataFrame on failure.
        """
        all_records = []
        empty_df = pd.DataFrame(
            columns=["event_time", "funding_rate", "mark_price", "annualized_rate"]
        )

        # Try INTX with pagination
        try:
            remaining = limit
            offset = 0
            page_size = min(100, limit)

            while remaining > 0:
                batch_size = min(page_size, remaining)
                records = self._fetch_intx_funding(
                    instrument, limit=batch_size, offset=offset
                )
                if not records:
                    break
                all_records.extend(records)
                remaining -= len(records)
                offset += len(records)
                # Stop if we got fewer than requested (end of data)
                if len(records) < batch_size:
                    break

            logger.info(
                "get_funding_history: fetched %d records from INTX",
                len(all_records),
            )
        except Exception as exc:
            logger.warning(
                "INTX funding history failed: %s -- trying FairX", exc
            )
            # Try FairX fallback (may not support pagination)
            try:
                all_records = self._fetch_fairx_funding(instrument)
                logger.info(
                    "get_funding_history: fetched %d records from FairX",
                    len(all_records),
                )
            except Exception as exc2:
                logger.error(
                    "get_funding_history: all endpoints failed: %s", exc2
                )
                return empty_df

        if not all_records:
            logger.warning("get_funding_history: no records returned")
            return empty_df

        # Parse all records
        parsed = [self._parse_funding_record(r) for r in all_records]
        df = pd.DataFrame(parsed)

        # Sort by event_time ascending
        df = df.sort_values("event_time").reset_index(drop=True)

        # Trim to requested limit
        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        logger.info(
            "get_funding_history: %d records  [%s .. %s]",
            len(df),
            df["event_time"].iloc[0] if len(df) else "N/A",
            df["event_time"].iloc[-1] if len(df) else "N/A",
        )
        return df

    def estimate_funding_cost(
        self,
        position_size_usd: float,
        hold_hours: float,
        leverage: float = 1.0,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> dict:
        """
        Estimate funding cost for a hypothetical position.

        Uses the current funding rate to project costs over the hold period.
        Coinbase INTX calculates funding hourly, so the number of funding
        periods equals hold_hours.

        Args:
            position_size_usd: Position size in USD.
            hold_hours:        Expected holding period in hours.
            leverage:          Leverage multiplier (default 1.0 = no leverage).
            instrument:        Instrument ID (default: 'BTC-PERP').

        Returns:
            dict with keys:
                total_cost_usd       (float): Estimated total funding cost.
                cost_pct             (float): Cost as % of position size.
                funding_periods      (int):   Number of hourly funding periods.
                avg_rate_per_period  (float): Average funding rate per period.
        """
        current = self.get_current_funding_rate(instrument)
        rate = current["funding_rate"]

        # Funding is applied each hour on the notional value
        notional = position_size_usd * leverage
        funding_periods = int(hold_hours)
        total_cost = abs(rate) * notional * funding_periods
        cost_pct = (total_cost / position_size_usd * 100) if position_size_usd > 0 else 0.0

        result = {
            "total_cost_usd": round(total_cost, 4),
            "cost_pct": round(cost_pct, 6),
            "funding_periods": funding_periods,
            "avg_rate_per_period": rate,
        }

        logger.info(
            "estimate_funding_cost: pos=$%.2f  hold=%dh  lev=%.1fx  "
            "cost=$%.4f (%.4f%%)",
            position_size_usd,
            funding_periods,
            leverage,
            total_cost,
            cost_pct,
        )
        return result


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
    print("Bull Machine -- Coinbase INTX Funding Rate Test")
    print("=" * 60)

    client = CoinbaseFundingClient()

    # 1. Current funding rate
    print("\n--- Current Funding Rate ---")
    try:
        rate_info = client.get_current_funding_rate()
        print(f"  Funding rate:    {rate_info['funding_rate']:.6f}  "
              f"({rate_info['funding_rate'] * 100:.4f}%)")
        print(f"  Annualized:      {rate_info['annualized_rate']:.4f}  "
              f"({rate_info['annualized_rate'] * 100:.2f}%)")
        print(f"  Mark price:      ${rate_info['mark_price']:,.2f}")
        print(f"  Event time (UTC): {rate_info['event_time']}")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # 2. Funding history
    print("\n--- Funding History (last 10 records) ---")
    try:
        df = client.get_funding_history(limit=10)
        if df.empty:
            print("  No historical records returned.")
        else:
            # Format for display
            display_df = df.copy()
            display_df["event_time"] = display_df["event_time"].apply(
                lambda x: x.strftime("%Y-%m-%d %H:%M") if hasattr(x, "strftime") else str(x)
            )
            display_df["funding_rate"] = display_df["funding_rate"].apply(
                lambda x: f"{x:.6f}"
            )
            display_df["annualized_rate"] = display_df["annualized_rate"].apply(
                lambda x: f"{x:.4f}"
            )
            display_df["mark_price"] = display_df["mark_price"].apply(
                lambda x: f"${x:,.2f}"
            )
            print(display_df.to_string(index=False))
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # 3. Funding cost estimation
    print("\n--- Funding Cost Estimate ---")
    print("  Scenario: $10,000 position, 24h hold, 2x leverage")
    try:
        cost = client.estimate_funding_cost(
            position_size_usd=10000.0,
            hold_hours=24.0,
            leverage=2.0,
        )
        print(f"  Funding periods:     {cost['funding_periods']}")
        print(f"  Avg rate/period:     {cost['avg_rate_per_period']:.6f}")
        print(f"  Total cost:          ${cost['total_cost_usd']:.4f}")
        print(f"  Cost (% of capital): {cost['cost_pct']:.4f}%")
    except Exception as exc:
        print(f"  FAILED: {exc}")

    # 4. Auth status
    print("\n--- Authentication Status ---")
    if client._has_auth:
        print(f"  Authenticated: YES (key={client.api_key[:8]}...)")
    else:
        print("  Authenticated: NO (using public endpoints)")
        print("  Set env vars to enable auth:")
        print("    COINBASE_INTX_API_KEY")
        print("    COINBASE_INTX_API_SECRET")
        print("    COINBASE_INTX_PASSPHRASE")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Coinbase INTX funding rate client for Bull Machine."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run connectivity test: fetch current rate, history, and cost estimate.",
    )
    parser.add_argument(
        "--instrument",
        default=DEFAULT_INSTRUMENT,
        help=f"Instrument ID (default: {DEFAULT_INSTRUMENT}).",
    )
    parser.add_argument(
        "--history",
        type=int,
        metavar="N",
        help="Fetch N historical funding rate records and print them.",
    )
    parser.add_argument(
        "--estimate",
        type=float,
        nargs=3,
        metavar=("SIZE_USD", "HOLD_HOURS", "LEVERAGE"),
        help="Estimate funding cost: SIZE_USD HOLD_HOURS LEVERAGE.",
    )
    args = parser.parse_args()

    if args.test:
        _run_test()
    elif args.history:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        client = CoinbaseFundingClient()
        df = client.get_funding_history(
            instrument=args.instrument, limit=args.history
        )
        if df.empty:
            print("No funding rate history returned.")
        else:
            print(df.to_string(index=False))
    elif args.estimate:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        size_usd, hold_hours, leverage = args.estimate
        client = CoinbaseFundingClient()
        cost = client.estimate_funding_cost(
            position_size_usd=size_usd,
            hold_hours=hold_hours,
            leverage=leverage,
            instrument=args.instrument,
        )
        print(f"Position:        ${size_usd:,.2f}")
        print(f"Hold:            {int(hold_hours)}h")
        print(f"Leverage:        {leverage:.1f}x")
        print(f"Funding periods: {cost['funding_periods']}")
        print(f"Avg rate/period: {cost['avg_rate_per_period']:.6f}")
        print(f"Total cost:      ${cost['total_cost_usd']:.4f}")
        print(f"Cost (%):        {cost['cost_pct']:.4f}%")
    else:
        parser.print_help()
