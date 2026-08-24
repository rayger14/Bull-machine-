"""
Seller-flow sensor repair (2026-08-24): live taker_imbalance must be computed
from the last COMPLETED 1H taker-volume bucket with the store-matching formula.

Bug (cme_seller_flow_replication_2026_08_20.md): the runner fetches ~1min past
the hour and read taker_data[0] — the just-STARTED bucket (~60s of volume) —
then applied (r-1)/max(r,0.01). Result: autocorr 0.007 (white noise), r=0.017
vs real CME hourly flow. The validated store feature is a full-hour aggregate
with imbalance = (buy-sell)/(buy+sell) == (r-1)/(r+1).
"""
import pandas as pd
from bin.live.okx_derivatives_api import select_completed_taker_bucket, taker_imbalance_from_vols


def _ms(ts): return str(int(pd.Timestamp(ts, tz='UTC').timestamp()*1000))


def test_selects_last_completed_hour_not_in_progress():
    now = pd.Timestamp('2026-08-24 15:01:40', tz='UTC')
    buckets = [
        [_ms('2026-08-24 15:00'), '10', '12'],   # in-progress hour — must NOT be used
        [_ms('2026-08-24 14:00'), '500', '400'], # last completed hour — the answer
        [_ms('2026-08-24 13:00'), '300', '300'],
    ]
    sell, buy = select_completed_taker_bucket(buckets, now)
    assert (sell, buy) == (500.0, 400.0)


def test_selects_head_when_head_is_completed():
    now = pd.Timestamp('2026-08-24 15:01:40', tz='UTC')
    buckets = [
        [_ms('2026-08-24 14:00'), '500', '400'],  # head already the completed hour
        [_ms('2026-08-24 13:00'), '300', '300'],
    ]
    sell, buy = select_completed_taker_bucket(buckets, now)
    assert (sell, buy) == (500.0, 400.0)


def test_returns_none_when_no_completed_bucket():
    now = pd.Timestamp('2026-08-24 15:01:40', tz='UTC')
    buckets = [[_ms('2026-08-24 15:00'), '10', '12']]
    assert select_completed_taker_bucket(buckets, now) is None


def test_formula_matches_store_definition():
    # store: imbalance = (r-1)/(r+1) = (buy-sell)/(buy+sell), bounded [-1, 1]
    assert abs(taker_imbalance_from_vols(buy=150.0, sell=100.0) - 0.2) < 1e-9
    assert abs(taker_imbalance_from_vols(buy=100.0, sell=150.0) + 0.2) < 1e-9
    assert taker_imbalance_from_vols(buy=0.0, sell=0.0) == 0.0
    assert taker_imbalance_from_vols(buy=100.0, sell=0.0) == 1.0
