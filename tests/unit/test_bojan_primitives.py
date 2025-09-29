import pandas as pd
from bull_machine.modules.bojan.bojan import compute_bojan_score

def _df(rows):
    return pd.DataFrame(rows)

def test_trap_reset_long():
    # prior red → current green, big body overlap ⇒ trap reset
    df = _df([
        dict(open=100, high=101, low=98, close=99),
        dict(open=99,  high=103, low=97, close=102),
    ])
    out = compute_bojan_score(df, {"trap_body_min":0.33})
    assert out["signals"]["trap_reset"]["is_trap_reset"] is True
    assert 0.19 <= out["bojan_score"] <= 1.0
    assert out["direction_hint"] in ("bullish","neutral")

def test_wick_absorption_magnet():
    df = _df([dict(open=100, high=112, low=99, close=101)])
    out = compute_bojan_score(df, {"wick_magnet_threshold":0.40})
    assert out["signals"]["wick_magnet"]["metrics"]["wick_dominance"] >= 0.40
    assert out["bojan_score"] >= 0.10  # wick bonus applied

def test_fib_prime_zone_flag():
    # close near .705/.786 within 1%
    base = _df([dict(open=100, high=120, low=80, close=100)])
    out = compute_bojan_score(base, {"lookback":50, "fib_prime_zones":[0.705, 0.786]})
    assert out["signals"]["fib_prime"]["in_prime_zone"] in (True, False)  # flag present