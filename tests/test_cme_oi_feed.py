"""CME OI live feed — causality and degradation tests (no network)."""
import numpy as np
import pandas as pd

from bin.live.cme_oi_feed import compute_cme_oi_features, CMEOIFeed


def _series(vals, start="2026-08-18"):
    idx = pd.date_range(start, periods=len(vals), freq="1D", tz="UTC")
    return pd.Series(vals, index=idx, dtype=float)


NOW = pd.Timestamp("2026-08-24 15:30", tz="UTC")


def test_causal_uses_only_completed_settlements():
    """Today's (partial/just-published) settlement must never be used."""
    oi = _series([100, 110, 120, 130, 140, 150, 160])  # last = 2026-08-24 (today)
    f = compute_cme_oi_features(oi, pd.Series(dtype=float), NOW)
    assert f["cme_oi_value"] == 150.0  # yesterday (08-23), not today's 160
    assert abs(f["cme_oi_change_24h"] - (150 / 140 - 1)) < 1e-12


def test_staleness_reported():
    oi = _series([100, 110], start="2026-08-18")  # newest = 08-19, 5 days old
    f = compute_cme_oi_features(oi, pd.Series(dtype=float), NOW)
    assert f["cme_oi_age_days"] == 5.0
    assert f["cme_oi_value"] == 110.0


def test_insufficient_history_is_nan():
    f = compute_cme_oi_features(_series([100])[:1], pd.Series(dtype=float), NOW)
    assert np.isnan(f["cme_oi_value"]) and np.isnan(f["cme_oi_change_24h"])


def test_divergence_sign_matches_study_definition():
    # price up, OI down -> hollow move -> positive divergence (study formula)
    oi = _series([100, 110, 105, 100, 95])           # falling into 08-22
    px = _series([50000, 50500, 51000, 51500, 52000])
    now = pd.Timestamp("2026-08-23 12:00", tz="UTC")
    f = compute_cme_oi_features(oi, px, now)
    assert f["cme_oi_price_divergence"] > 0
    # price up, OI up -> backed move -> negative divergence
    oi2 = _series([100, 105, 110, 115, 120])
    f2 = compute_cme_oi_features(oi2, px, now)
    assert f2["cme_oi_price_divergence"] < 0


def test_feed_without_key_is_inert():
    feed = CMEOIFeed(api_key=None)
    assert not feed.available
    feed.refresh_if_needed(NOW)  # must not raise, must not fetch
    f = feed.features(NOW)
    assert np.isnan(f["cme_oi_value"])
