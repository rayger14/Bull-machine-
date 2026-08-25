"""CME institutional Open Interest live feed (Databento) — SHADOW features.

Why (docs/knowledge/cme_oi_regime_study_2026_08_21.md): oi_divergence re-fed
with CME daily OI flipped from −$5,334/WR 37% to +$6,804/WR 59% (2021-2024,
zero tuning) — the archetype's premise was right, its perp-snapshot witness
was garbage. The live path still reads that witness. This module fetches the
once-a-day CME settlement OI (GLBX.MDP3 `statistics` schema, stat_type 9)
and derives CAUSAL daily features under a `cme_` prefix.

SHADOW CONTRACT: nothing in the trading path consumes `cme_*` features.
Promotion = a later, deliberate config change rewiring oi_divergence's gates
after live shadow validation. The whale-conflict penalty keeps its perp 4h
witness permanently (daily-for-4h swap REJECTED: semantics mismatch, −$24.7K).

Causality: CME OI for day D is a settlement figure published after the
session; we only ever expose the change up to D-1 (yesterday's completed
settlement), matching the offline study exactly.

Cost: one small statistics request per UTC day (~$0.001 against the
Databento account). API key read from DATABENTO_API_KEY or ~/.databento_key
— never from the repo.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HIST_URL = "https://hist.databento.com/v0/timeseries.get_range"
_STAT_OPEN_INTEREST = 9  # Databento statistics stat_type for open interest


def load_api_key() -> Optional[str]:
    """DATABENTO_API_KEY env var, else ~/.databento_key file, else None."""
    key = os.environ.get("DATABENTO_API_KEY", "").strip()
    if key:
        return key
    p = Path.home() / ".databento_key"
    if p.exists():
        key = p.read_text().strip()
        return key or None
    return None


def fetch_daily_oi(api_key: str, lookback_days: int = 10,
                   symbol: str = "BTC.c.0",
                   timeout: int = 60) -> pd.Series:
    """Fetch the last `lookback_days` of daily CME OI settlements.

    Returns a Series of OI (contracts) indexed by UTC-normalized settlement
    day, sorted, deduped (last wins). Raises on network/HTTP errors — the
    caller degrades gracefully.
    """
    end = pd.Timestamp.utcnow().floor("D")
    start = end - pd.Timedelta(days=lookback_days)
    params = urllib.parse.urlencode({
        "dataset": "GLBX.MDP3",
        "symbols": symbol,
        "stype_in": "continuous",
        "schema": "statistics",
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoding": "json",
        "pretty_ts": "true",
        "map_symbols": "true",
    })
    req = urllib.request.Request(f"{_HIST_URL}?{params}")
    auth = (api_key + ":").encode()
    import base64
    req.add_header("Authorization", "Authorization: Basic".split()[0] and
                   "Basic " + base64.b64encode(auth).decode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        lines = r.read().decode().strip().splitlines()
    rows: List[Dict] = []
    for line in lines:
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        rec = o.get("record", o)
        hd = rec.get("hd", {})
        if int(rec.get("stat_type", -1)) != _STAT_OPEN_INTEREST:
            continue
        ts = rec.get("ts_ref") or hd.get("ts_event") or rec.get("ts_event")
        qty = rec.get("quantity")
        if ts is None or qty is None:
            continue
        rows.append({"ts": ts, "oi": float(qty)})
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["day"] = pd.to_datetime(df["ts"], utc=True).dt.normalize()
    s = df.groupby("day")["oi"].last().sort_index()
    return s


def compute_cme_oi_features(oi: pd.Series, price: pd.Series,
                            now: pd.Timestamp) -> Dict[str, float]:
    """Derive the causal shadow features from daily OI + daily close series.

    Uses only settlements up to YESTERDAY relative to `now` (t-1 causal),
    mirroring the offline study. Returns NaNs when history is insufficient —
    consumers must be nan-safe (they are: nothing consumes these yet).

    Features:
        cme_oi_value        : yesterday's settlement OI (contracts)
        cme_oi_change_24h   : yesterday's settlement vs the one before
        cme_oi_price_divergence : -sign(doi)*sign(dpx)*|doi| (daily analog)
        cme_oi_age_days     : age of the newest usable settlement (staleness)
    """
    out = {
        "cme_oi_value": np.nan,
        "cme_oi_change_24h": np.nan,
        "cme_oi_price_divergence": np.nan,
        "cme_oi_age_days": np.nan,
    }
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    cutoff = now.normalize() - pd.Timedelta(days=1)
    oi = oi.dropna()
    if len(oi) == 0:            # empty series has no DatetimeIndex to compare
        return out
    if oi.index.tz is None:
        oi.index = oi.index.tz_localize("UTC")
    oi = oi[oi.index <= cutoff]
    if len(oi) < 2:
        return out
    out["cme_oi_value"] = float(oi.iloc[-1])
    out["cme_oi_age_days"] = float((now.normalize() - oi.index[-1]).days)
    prev = float(oi.iloc[-2])
    doi = (float(oi.iloc[-1]) / prev - 1.0) if prev > 0 else np.nan
    out["cme_oi_change_24h"] = doi
    if price is not None and len(price.dropna()) >= 2 and doi == doi:
        px = price.dropna()
        px = px[px.index <= cutoff]
        if len(px) >= 2 and float(px.iloc[-2]) > 0:
            dpx = float(px.iloc[-1]) / float(px.iloc[-2]) - 1.0
            out["cme_oi_price_divergence"] = float(
                -np.sign(doi) * np.sign(dpx) * abs(doi))
    return out


class CMEOIFeed:
    """Once-per-UTC-day fetcher with graceful degradation.

    On failure keeps the previous day's series (features go stale, the
    cme_oi_age_days field says by how much). Never raises into the runner.
    """

    _AUTO = "__auto__"

    def __init__(self, api_key: Optional[str] = _AUTO):
        # api_key=None means explicitly no key (inert feed); the default
        # sentinel resolves from env/keyfile via load_api_key().
        self._key = load_api_key() if api_key == self._AUTO else api_key
        self._oi: pd.Series = pd.Series(dtype=float)
        self._last_fetch_day: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._key is not None

    def refresh_if_needed(self, now: pd.Timestamp) -> None:
        if not self.available:
            return
        day = pd.Timestamp(now).strftime("%Y-%m-%d")
        if self._last_fetch_day == day:
            return
        try:
            s = fetch_daily_oi(self._key)
            if len(s) > 0:
                self._oi = s
                self._last_fetch_day = day
                logger.info("[CME_OI] refreshed: %d settlements through %s",
                            len(s), s.index[-1].date())
            else:
                logger.warning("[CME_OI] empty response — keeping previous series")
        except Exception as exc:  # noqa: BLE001 — never break the bar loop
            logger.warning("[CME_OI] refresh failed (%s) — keeping previous series", exc)

    def features(self, now: pd.Timestamp,
                 daily_price: Optional[pd.Series] = None) -> Dict[str, float]:
        return compute_cme_oi_features(self._oi, daily_price if daily_price
                                       is not None else pd.Series(dtype=float), now)
