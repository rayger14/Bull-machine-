#!/usr/bin/env python3
"""
CoinGlass API Client — US-accessible alternative to Binance Futures API.

Drop-in replacement for BinanceFuturesAPI.fetch_all_current().
CoinGlass aggregates derivatives data from Binance, OKX, Bybit, etc.

Requirements:
    - CoinGlass API key ($29/month Hobbyist tier, 30 req/min)
    - Set env var: COINGLASS_API_KEY=your_key_here

Endpoints used (V4):
    - /api/futures/openInterest/ohlc-aggregated-history  (aggregated OI)
    - /api/futures/fundingRate/oi-weight-ohlc-history     (OI-weighted funding)
    - /api/futures/global-long-short-account-ratio/history (L/S ratio)
    - /api/futures/aggregated-taker-buy-sell-volume/history (taker volume)

Usage:
    api = CoinGlassAPI()  # reads COINGLASS_API_KEY from env
    data = api.fetch_all_current()
    print(data['oi_change_4h'])

CLI test:
    python3 bin/live/coinglass_api.py --test
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Dict, Any

try:
    import requests
except ImportError:
    print("ERROR: requests is not installed. pip install requests")
    sys.exit(1)

logger = logging.getLogger(__name__)

BASE_URL = "https://open-api-v4.coinglass.com"


class CoinGlassAPI:
    """CoinGlass derivatives data client. Requires API key."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "No CoinGlass API key found. Set COINGLASS_API_KEY env var. "
                "All requests will fail with 401."
            )
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "CG-API-KEY": self.api_key,
        })
        self._last_request_ts: float = 0.0
        # 30 req/min on Hobbyist = 1 req per 2 seconds
        self._min_interval = 2.0

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute GET request with retries and error handling."""
        url = f"{BASE_URL}{path}"
        params = params or {}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("code") == "0" or body.get("success"):
                        return body.get("data", [])
                    # CoinGlass error in response body
                    logger.warning(
                        "CoinGlass API error on %s: code=%s msg=%s",
                        path, body.get("code"), body.get("msg"),
                    )
                    return None

                if resp.status_code == 401:
                    logger.error("CoinGlass 401 Unauthorized — check COINGLASS_API_KEY")
                    return None

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    logger.warning(
                        "CoinGlass rate limited (429) on %s, retry %d/%d after %ds",
                        path, attempt, self.max_retries, retry_after,
                    )
                    time.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        "CoinGlass server error %d on %s, retry %d/%d after %ds",
                        resp.status_code, path, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue

                logger.error(
                    "CoinGlass error %d on %s: %s",
                    resp.status_code, path, resp.text[:300],
                )
                return None

            except requests.exceptions.Timeout:
                wait = 2 ** attempt
                logger.warning("CoinGlass timeout on %s, retry %d/%d", path, attempt, self.max_retries)
                last_exc = requests.exceptions.Timeout(f"Timeout on {path}")
                time.sleep(wait)

            except requests.exceptions.ConnectionError as exc:
                wait = 2 ** attempt
                logger.warning("CoinGlass connection error: %s, retry %d/%d", exc, attempt, self.max_retries)
                last_exc = exc
                time.sleep(wait)

            except requests.exceptions.RequestException as exc:
                logger.error("CoinGlass request failed: %s", exc)
                return None

        logger.error("All %d retries exhausted for %s (last: %s)", self.max_retries, path, last_exc)
        return None

    def fetch_all_current(self, symbol: str = "BTC") -> Dict[str, float]:
        """
        Fetch current institutional data from CoinGlass.

        Returns same dict shape as BinanceFuturesAPI.fetch_all_current():
            {
                'oi_value': float,
                'oi_change_4h': float,
                'oi_change_24h': float,
                'funding_rate': float,
                'ls_ratio': float,
                'top_trader_ls_ratio': float,
                'taker_buy_sell_ratio': float,
            }
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

        # --- Open Interest (aggregated across exchanges) ---
        try:
            oi_data = self._request(
                "/api/futures/openInterest/ohlc-aggregated-history",
                {"symbol": symbol, "interval": "1h", "limit": 25},
            )
            if oi_data and len(oi_data) > 0:
                # Data comes as list of {t, o, h, l, c} where c = close OI value
                oi_vals = []
                for row in oi_data:
                    try:
                        oi_vals.append(float(row.get("c", 0)))
                    except (TypeError, ValueError):
                        pass

                if oi_vals:
                    result["oi_value"] = oi_vals[-1]

                if len(oi_vals) >= 5:
                    current = oi_vals[-1]
                    ago_4h = oi_vals[-5]
                    if ago_4h != 0:
                        result["oi_change_4h"] = (current - ago_4h) / ago_4h

                if len(oi_vals) >= 25:
                    current = oi_vals[-1]
                    ago_24h = oi_vals[-25]
                    if ago_24h != 0:
                        result["oi_change_24h"] = (current - ago_24h) / ago_24h
        except Exception as exc:
            logger.warning("CoinGlass OI fetch failed: %s", exc)

        # --- Funding Rate (OI-weighted across exchanges) ---
        try:
            fr_data = self._request(
                "/api/futures/fundingRate/oi-weight-ohlc-history",
                {"symbol": symbol, "interval": "1h", "limit": 1},
            )
            if fr_data and len(fr_data) > 0:
                try:
                    result["funding_rate"] = float(fr_data[-1].get("c", 0))
                except (TypeError, ValueError):
                    pass
        except Exception as exc:
            logger.warning("CoinGlass funding rate fetch failed: %s", exc)

        # --- Global Long/Short Account Ratio ---
        try:
            ls_data = self._request(
                "/api/futures/global-long-short-account-ratio/history",
                {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1h", "limit": 1},
            )
            if ls_data and len(ls_data) > 0:
                row = ls_data[-1]
                try:
                    long_qty = float(row.get("longAccount", row.get("long_quantity", 0)))
                    short_qty = float(row.get("shortAccount", row.get("short_quantity", 0)))
                    if short_qty > 0:
                        result["ls_ratio"] = long_qty / short_qty
                    elif "longShortRatio" in row:
                        result["ls_ratio"] = float(row["longShortRatio"])
                except (TypeError, ValueError):
                    pass
        except Exception as exc:
            logger.warning("CoinGlass L/S ratio fetch failed: %s", exc)

        # --- Top Trader Long/Short Ratio ---
        try:
            top_data = self._request(
                "/api/futures/top-long-short-account-ratio/history",
                {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "1h", "limit": 1},
            )
            if top_data and len(top_data) > 0:
                row = top_data[-1]
                try:
                    long_qty = float(row.get("longAccount", row.get("long_quantity", 0)))
                    short_qty = float(row.get("shortAccount", row.get("short_quantity", 0)))
                    if short_qty > 0:
                        result["top_trader_ls_ratio"] = long_qty / short_qty
                    elif "longShortRatio" in row:
                        result["top_trader_ls_ratio"] = float(row["longShortRatio"])
                except (TypeError, ValueError):
                    pass
        except Exception as exc:
            logger.warning("CoinGlass top trader L/S fetch failed: %s", exc)

        # --- Taker Buy/Sell Volume ---
        try:
            taker_data = self._request(
                "/api/futures/aggregated-taker-buy-sell-volume/history",
                {"symbol": symbol, "interval": "1h", "limit": 1},
            )
            if taker_data and len(taker_data) > 0:
                row = taker_data[-1]
                try:
                    buy_vol = float(row.get("buyVolume", row.get("long_volume_usd", 0)))
                    sell_vol = float(row.get("sellVolume", row.get("short_volume_usd", 0)))
                    total = buy_vol + sell_vol
                    if total > 0:
                        result["taker_buy_sell_ratio"] = buy_vol / sell_vol if sell_vol > 0 else 2.0
                except (TypeError, ValueError):
                    pass
        except Exception as exc:
            logger.warning("CoinGlass taker volume fetch failed: %s", exc)

        logger.info(
            "CoinGlass fetch_all_current: OI=$%.0f  OI_4h=%.3f%%  OI_24h=%.3f%%  "
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

    def close(self) -> None:
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __repr__(self) -> str:
        masked = self.api_key[:4] + "..." if len(self.api_key) > 4 else "(empty)"
        return f"CoinGlassAPI(key={masked})"


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="CoinGlass API Client")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--symbol", default="BTC", help="Coin symbol (default: BTC)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.test:
        print(f"=== CoinGlass API smoke test (symbol={args.symbol}) ===\n")
        api = CoinGlassAPI()
        print(f"Client: {api}")
        print()

        data = api.fetch_all_current(args.symbol)
        print("Results:")
        for k, v in data.items():
            if isinstance(v, float) and abs(v) > 1000:
                print(f"  {k}: {v:,.0f}")
            else:
                print(f"  {k}: {v}")

        api.close()
        print("\n=== Test complete ===")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
