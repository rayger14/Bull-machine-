#!/usr/bin/env python3
"""
OKX Public Derivatives API Client — free, no auth, US-accessible.

Drop-in replacement for BinanceFuturesAPI.fetch_all_current().
Uses OKX public endpoints which are NOT geo-blocked from US IPs.

Endpoints used:
  - /api/v5/public/open-interest         (current OI)
  - /api/v5/rubik/stat/contracts/open-interest-volume  (OI history for change calc)
  - /api/v5/public/funding-rate           (current funding rate)
  - /api/v5/rubik/stat/contracts/long-short-account-ratio  (L/S ratio)
  - /api/v5/rubik/stat/taker-volume       (taker buy/sell volume)

Rate limits: ~3 req/s per IP (conservative: we do 5 calls per hour)

Usage:
    api = OKXDerivativesAPI()
    data = api.fetch_all_current()
    print(data['oi_change_4h'])

CLI test:
    python3 bin/live/okx_derivatives_api.py --test
"""

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


def select_completed_taker_bucket(taker_data, now):
    """Pick the last COMPLETED 1H taker-volume bucket (SENSOR REPAIR 2026-08-24).

    OKX returns buckets newest-first as [ts_ms, sell_vol, buy_vol]. The runner
    fetches ~1 minute past the hour, so the head bucket is usually the hour
    that just STARTED (~60s of volume) — reading it made live taker_imbalance
    white noise (autocorr 0.007; r=0.017 vs real CME hourly flow). The
    validated store feature is a full-hour aggregate, so we must select the
    bucket for the last completed hour by TIMESTAMP, never by position.

    Returns (sell_vol, buy_vol) floats, or None if no completed bucket found.
    """
    import pandas as _pd
    now = _pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize('UTC')
    completed = (now.floor('h') - _pd.Timedelta(hours=1)).timestamp() * 1000
    for row in taker_data or []:
        try:
            ts = float(row[0])
            if abs(ts - completed) < 1:
                return float(row[1]), float(row[2])
        except (IndexError, TypeError, ValueError):
            continue
    return None


def taker_imbalance_from_vols(buy, sell):
    """Store-matching imbalance: (buy-sell)/(buy+sell) == (r-1)/(r+1), in [-1, 1].

    The old live formula (r-1)/max(r,0.01) had a different scale (std 0.27 vs
    the store's 0.075) on top of the snapshot bug; this restores the exact
    definition the seller-flow boost was validated on
    (derivatives_data_backfill_method.md).
    """
    tot = buy + sell
    return (buy - sell) / tot if tot > 0 else 0.0

BASE_URL = "https://www.okx.com"


class OKXDerivativesAPI:
    """OKX public derivatives data client. No authentication required."""

    def __init__(
        self,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "BullMachine/1.0",
        })
        self._last_request_ts: float = 0.0
        self._min_interval = 0.5  # 2 req/s conservative

    def _throttle(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_ts = time.monotonic()

    def _request(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute GET request with retries."""
        url = f"{BASE_URL}{path}"
        params = params or {}
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)

                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("code") == "0":
                        return body.get("data", [])
                    logger.warning(
                        "OKX API error on %s: code=%s msg=%s",
                        path, body.get("code"), body.get("msg"),
                    )
                    return None

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 5))
                    logger.warning("OKX rate limited, retry %d/%d after %ds", attempt, self.max_retries, retry_after)
                    time.sleep(retry_after)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning("OKX server error %d, retry %d/%d", resp.status_code, attempt, self.max_retries)
                    time.sleep(wait)
                    continue

                logger.error("OKX error %d on %s: %s", resp.status_code, path, resp.text[:300])
                return None

            except requests.exceptions.Timeout:
                last_exc = requests.exceptions.Timeout(f"Timeout on {path}")
                logger.warning("OKX timeout on %s, retry %d/%d", path, attempt, self.max_retries)
                time.sleep(2 ** attempt)

            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                logger.warning("OKX connection error: %s, retry %d/%d", exc, attempt, self.max_retries)
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as exc:
                logger.error("OKX request failed: %s", exc)
                return None

        logger.error("All %d retries exhausted for %s (last: %s)", self.max_retries, path, last_exc)
        return None

    def fetch_all_current(self, symbol: str = "BTCUSDT") -> Dict[str, float]:
        """
        Fetch current institutional data from OKX public API.

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

        # OKX uses BTC-USDT-SWAP for perp, BTC for rubik stats
        inst_id = "BTC-USDT-SWAP"
        ccy = "BTC"

        # --- 1. Open Interest (current) ---
        try:
            oi_data = self._request(
                "/api/v5/public/open-interest",
                {"instType": "SWAP", "instId": inst_id},
            )
            if oi_data and len(oi_data) > 0:
                # oiUsd is OI in USD, oi is in contracts, oiCcy is in coin
                oi_usd = float(oi_data[0].get("oiUsd", 0) or 0)
                result["oi_value"] = oi_usd
        except Exception as exc:
            logger.warning("OKX OI current failed: %s", exc)

        # --- 2. OI History (for 4h/24h change) ---
        try:
            oi_hist = self._request(
                "/api/v5/rubik/stat/contracts/open-interest-volume",
                {"ccy": ccy, "period": "1H"},
            )
            if oi_hist and len(oi_hist) > 0:
                # Response: list of [timestamp, oi, volume] — newest first
                # Reverse to oldest-first for change calculation
                oi_vals = []
                for row in oi_hist:
                    try:
                        oi_vals.append(float(row[1]))  # index 1 = OI
                    except (IndexError, TypeError, ValueError):
                        pass

                # OKX returns newest first, reverse to oldest-first
                oi_vals.reverse()

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

                # Update oi_value from history if current endpoint failed
                if result["oi_value"] == 0.0 and oi_vals:
                    result["oi_value"] = oi_vals[-1]
        except Exception as exc:
            logger.warning("OKX OI history failed: %s", exc)

        # --- 3. Funding Rate (current) ---
        try:
            fr_data = self._request(
                "/api/v5/public/funding-rate",
                {"instId": inst_id},
            )
            if fr_data and len(fr_data) > 0:
                fr_val = float(fr_data[0].get("fundingRate", 0) or 0)
                result["funding_rate"] = fr_val
        except Exception as exc:
            logger.warning("OKX funding rate failed: %s", exc)

        # --- 4. Long/Short Account Ratio ---
        try:
            ls_data = self._request(
                "/api/v5/rubik/stat/contracts/long-short-account-ratio",
                {"ccy": ccy, "period": "1H"},
            )
            if ls_data and len(ls_data) > 0:
                # Response: list of [timestamp, ratio] — newest first
                try:
                    result["ls_ratio"] = float(ls_data[0][1])
                except (IndexError, TypeError, ValueError):
                    pass
        except Exception as exc:
            logger.warning("OKX L/S ratio failed: %s", exc)

        # --- 5. Taker Buy/Sell Volume ---
        try:
            taker_data = self._request(
                "/api/v5/rubik/stat/taker-volume",
                {"ccy": ccy, "instType": "CONTRACTS", "period": "1H"},
            )
            if taker_data and len(taker_data) > 0:
                # Response: list of [timestamp, sell_vol, buy_vol] — newest first.
                # SENSOR REPAIR 2026-08-24: use the last COMPLETED hour (by
                # timestamp), never the just-started head bucket. See
                # select_completed_taker_bucket docstring.
                import pandas as _pd
                bucket = select_completed_taker_bucket(taker_data, _pd.Timestamp.utcnow())
                if bucket is not None:
                    sell_vol, buy_vol = bucket
                    result["taker_sell_vol_1h"] = sell_vol
                    result["taker_buy_vol_1h"] = buy_vol
                    if sell_vol > 0:
                        result["taker_buy_sell_ratio"] = buy_vol / sell_vol
        except Exception as exc:
            logger.warning("OKX taker volume failed: %s", exc)

        logger.info(
            "OKX fetch_all_current: OI=$%.0f  OI_4h=%.3f%%  OI_24h=%.3f%%  "
            "FR=%.6f  LS=%.3f  TakerBS=%.3f",
            result["oi_value"],
            result["oi_change_4h"] * 100,
            result["oi_change_24h"] * 100,
            result["funding_rate"],
            result["ls_ratio"],
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
        return "OKXDerivativesAPI(public, no-auth)"


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="OKX Derivatives API Client (free, public)")
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.test:
        print("=== OKX Derivatives API smoke test ===\n")
        api = OKXDerivativesAPI()

        data = api.fetch_all_current()
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
