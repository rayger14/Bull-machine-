"""
FORM A — FUNDING-EXTREME FADE  (standalone carry; STUDY ONLY; add.70)
=====================================================================
Implements idea_lab/funding_carry_PREREGISTRATION.txt EXACTLY. No tuning, no grids.
Collects the 8h funding cash flow explicitly. Referee parity 0.00% + causality
3-point no-repaint checks on the expanding percentiles.

Run:  python3 idea_lab/run_funding_carry.py
Emits JSON to stdout tail (parsed by the report); writes NOTHING to disk.
"""
from __future__ import annotations
import json, sys
import numpy as np
import pandas as pd

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
DERIV = f"{REPO}/data/cache/derivatives_hourly_full.parquet"
PRICE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"

# --- frozen constants (from the pre-registration) ---
RISK_PCT   = 0.01
STOP_ATR   = 3.0
COST_BPS   = 5.0 / 1e4      # 2bps comm + 3bps slip, per side, on notional
WARMUP     = 270           # 8h steps (~90d) before any trade
F3_WIN     = 9            # 3 days * 3 settlements/day
INITIAL    = 100_000.0
ANN        = 3 * 365       # 8h settlements per year


def load_8h():
    """BTC 1H -> 8h OHLC aligned to 00/08/16 UTC + settled funding at each 8h stamp."""
    d = pd.read_parquet(DERIV)                       # tz-aware UTC
    f = d["binance_funding_rate"].copy()
    f.index = f.index.tz_localize(None)              # -> naive UTC to match price
    p = pd.read_parquet(PRICE)[["open", "high", "low", "close"]].copy()  # naive
    # 1H -> 8h bars (label = period start 00/08/16)
    o = p["open"].resample("8h").first()
    h = p["high"].resample("8h").max()
    l = p["low"].resample("8h").min()
    c = p["close"].resample("8h").last()
    px = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    # settled funding rate at each 8h stamp (ffilled hourly -> value at the stamp hour)
    fund = f.reindex(px.index, method="ffill")
    # daily Wilder ATR14 (value as of the entry day), mapped onto the 8h grid
    atr_d = wilder_atr_daily(p)
    atr8 = atr_d.reindex(px.index.normalize()).to_numpy()
    px = px.assign(funding=fund.to_numpy(), atr=atr8).dropna()
    # restrict to the funding-overlap window
    px = px[(px.index >= "2020-09-01") & (px.index <= "2026-06-11")]
    return px


def wilder_atr_daily(p1h: pd.DataFrame, period=14) -> pd.Series:
    d = p1h.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.empty_like(tr); atr[0] = tr[0]; a = 1.0 / period
    for i in range(1, len(tr)):
        atr[i] = a * tr[i] + (1 - a) * atr[i - 1]
    return pd.Series(atr, index=d.index.normalize())


def expanding_pct(x: np.ndarray, q: float) -> np.ndarray:
    """Causal expanding q-percentile: out[k] = percentile(x[:k+1], q). NaN where x NaN."""
    out = np.full(len(x), np.nan)
    buf = []
    for k in range(len(x)):
        v = x[k]
        if v == v:  # not NaN
            buf.append(v)
        if buf:
            out[k] = np.percentile(buf, q)
    return out


def build_signals(px: pd.DataFrame):
    f = px["funding"].to_numpy()
    f3 = pd.Series(f).rolling(F3_WIN).mean().to_numpy()   # causal 3d mean
    p85 = expanding_pct(f3, 85)
    p50 = expanding_pct(f3, 50)
    p15 = expanding_pct(f3, 15)
    return f3, p85, p50, p15


def backtest(px: pd.DataFrame):
    f3, p85, p50, p15 = build_signals(px)
    ts = px.index
    o, h, l, c = (px[k].to_numpy() for k in ("open", "high", "low", "close"))
    fund, atr = px["funding"].to_numpy(), px["atr"].to_numpy()
    n = len(px)

    equity = INITIAL
    eq_curve = [INITIAL]
    trades = []
    pos = None  # dict when in a position

    for i in range(n):
        # ---- manage open position at bar i (funding accrues, then check exits) ----
        if pos is not None:
            # funding settlement at this 8h stamp (position held into it)
            cf = -pos["side"] * fund[i] * pos["notional"]
            pos["funding"] += cf
            exit_px = None; reason = None
            # (a) STOP intrabar
            if pos["side"] == -1 and h[i] >= pos["stop"]:
                exit_px, reason = pos["stop"], "stop"
            elif pos["side"] == +1 and l[i] <= pos["stop"]:
                exit_px, reason = pos["stop"], "stop"
            # (b) NORMALIZATION at close (only if not stopped)
            if exit_px is None and (f3[i] == f3[i]) and (p50[i] == p50[i]):
                if pos["side"] == -1 and f3[i] < p50[i]:
                    exit_px, reason = c[i], "normalize"
                elif pos["side"] == +1 and f3[i] > p50[i]:
                    exit_px, reason = c[i], "normalize"
            if exit_px is not None:
                price_leg = pos["side"] * (exit_px - pos["entry"]) * pos["qty"]
                exit_cost = COST_BPS * abs(exit_px * pos["qty"])
                net = price_leg + pos["funding"] - pos["entry_cost"] - exit_cost
                equity += net
                R = net / pos["risk_dollars"]
                trades.append(dict(
                    entry_time=str(pos["etime"]), exit_time=str(ts[i]),
                    side=pos["side"], entry=pos["entry"], exit=exit_px, reason=reason,
                    notional=pos["notional"], hold_8h=i - pos["eidx"],
                    price_leg=price_leg, funding_leg=pos["funding"],
                    costs=pos["entry_cost"] + exit_cost, net=net, R=R,
                    equity_after=equity))
                pos = None

        # ---- entry at bar i (only if flat, warmed up, signals valid) ----
        if pos is None and i >= WARMUP and f3[i] == f3[i] and p85[i] == p85[i] and atr[i] == atr[i] and atr[i] > 0:
            side = 0
            if f3[i] > p85[i]:
                side = -1
            elif f3[i] < p15[i]:
                side = +1
            if side != 0:
                risk_d = RISK_PCT * equity
                stop_dist = STOP_ATR * atr[i]
                qty = risk_d / stop_dist
                entry_px = c[i]
                notional = qty * entry_px
                stop = entry_px + STOP_ATR * atr[i] if side == -1 else entry_px - STOP_ATR * atr[i]
                entry_cost = COST_BPS * notional
                pos = dict(side=side, entry=entry_px, qty=qty, notional=notional,
                           stop=stop, risk_dollars=risk_d, entry_cost=entry_cost,
                           funding=0.0, etime=ts[i], eidx=i)
        eq_curve.append(equity)

    return trades, np.array(eq_curve), (f3, p85, p50, p15)


def stats(trades, eq_curve):
    if not trades:
        return dict(n=0)
    nets = np.array([t["net"] for t in trades])
    gp = nets[nets > 0].sum()
    gl = -nets[nets < 0].sum()
    pf = gp / gl if gl > 0 else float("inf")
    price_leg = sum(t["price_leg"] for t in trades)
    funding_leg = sum(t["funding_leg"] for t in trades)
    costs = sum(t["costs"] for t in trades)
    # MaxDD on equity curve
    eq = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return dict(
        n=len(trades), net=float(nets.sum()), pf=float(pf),
        gross_profit=float(gp), gross_loss=float(gl),
        price_leg=float(price_leg), funding_leg=float(funding_leg), costs=float(costs),
        funding_frac_of_gp=float(funding_leg / gp) if gp > 0 else float("nan"),
        wr=float((nets > 0).mean()), meanR=float(np.mean([t["R"] for t in trades])),
        maxdd=float(dd.min()),
        n_short=sum(1 for t in trades if t["side"] == -1),
        n_long=sum(1 for t in trades if t["side"] == +1),
    )


def causality_3pt(px):
    """Verify expanding percentiles do not repaint: value at k unchanged if truncated at k."""
    f3, p85, p50, p15 = build_signals(px)
    f = px["funding"].to_numpy()
    idxs = [WARMUP + 50, len(px) // 2, len(px) - 5]
    mism = 0
    for k in idxs:
        f3k = pd.Series(f[:k + 1]).rolling(F3_WIN).mean().to_numpy()
        for full, q in ((p85, 85), (p50, 50), (p15, 15)):
            trunc = expanding_pct(f3k, q)[k]
            if not (np.isnan(full[k]) and np.isnan(trunc)):
                if abs((full[k] if full[k] == full[k] else 0) - (trunc if trunc == trunc else 0)) > 1e-12:
                    mism += 1
    return len(idxs) * 3, mism


def parity(trades, eq_curve):
    per_trade = sum(t["net"] for t in trades)
    curve_delta = eq_curve[-1] - eq_curve[0]
    denom = abs(curve_delta) if abs(curve_delta) > 1e-9 else 1.0
    return per_trade, curve_delta, abs(per_trade - curve_delta) / denom * 100.0


def per_year(trades):
    rows = {}
    for t in trades:
        y = t["exit_time"][:4]
        r = rows.setdefault(y, dict(n=0, net=0.0, price=0.0, funding=0.0, wins=0))
        r["n"] += 1; r["net"] += t["net"]; r["price"] += t["price_leg"]
        r["funding"] += t["funding_leg"]; r["wins"] += int(t["net"] > 0)
    return rows


def worst_price_leg(trades, top=8):
    """The squeeze DEATH MODE = worst price legs (position run over), incl. funding collected."""
    st = sorted(trades, key=lambda t: t["price_leg"])[:top]
    out = []
    for t in st:
        out.append(dict(entry=t["entry_time"][:10], exit=t["exit_time"][:10],
                        side="SHORT" if t["side"] == -1 else "LONG",
                        hold_days=round(t["hold_8h"] / 3, 1),
                        price_leg=round(t["price_leg"]), funding_leg=round(t["funding_leg"]),
                        net=round(t["net"]), R=round(t["R"], 2), reason=t["reason"]))
    return out


def drawdown_episodes(eq_curve, floor_pct=-3.0):
    """Contiguous equity drawdown episodes (peak->trough->recovery). Book DD per episode
    = the A3 unit. Returns episodes with DD <= floor_pct (material ones)."""
    eq = np.array(eq_curve)
    peak = eq[0]; trough = eq[0]; in_dd = False; start = 0
    eps = []
    for i in range(len(eq)):
        if eq[i] >= peak:
            if in_dd:
                dd = (trough - peak) / peak * 100.0
                if dd <= floor_pct:
                    eps.append(dict(idx_start=start, idx_end=i, book_dd_pct=round(dd, 2)))
                in_dd = False
            peak = eq[i]; trough = eq[i]
        else:
            if not in_dd:
                in_dd = True; start = i
            trough = min(trough, eq[i])
    if in_dd:
        dd = (trough - peak) / peak * 100.0
        if dd <= floor_pct:
            eps.append(dict(idx_start=start, idx_end=len(eq) - 1, book_dd_pct=round(dd, 2)))
    return sorted(eps, key=lambda e: e["book_dd_pct"])


def annualized_yield(trades, years):
    if not trades:
        return float("nan")
    avg_notional = np.mean([t["notional"] for t in trades])
    funding_leg = sum(t["funding_leg"] for t in trades)
    # yield on capital-at-risk proxy: funding_leg / avg_notional / years
    return funding_leg / avg_notional / years * 100.0


def time_in_market(trades, n_bars):
    held = sum(t["hold_8h"] for t in trades)
    return round(held / n_bars * 100.0, 1)


def main():
    px = load_8h()
    years = (px.index[-1] - px.index[0]).days / 365.25
    trades, eq, sig = backtest(px)
    S = stats(trades, eq)
    per_ch, curve_d, parity_pct = parity(trades, eq)
    n_checks, mism = causality_3pt(px)
    dd_eps = drawdown_episodes(eq)
    worst_book_dd = min((e["book_dd_pct"] for e in dd_eps), default=0.0)
    out = dict(
        window=[str(px.index[0]), str(px.index[-1])], years=round(years, 2), bars_8h=len(px),
        stats=S,
        time_in_market_pct=time_in_market(trades, len(px)),
        parity=dict(per_trade_sum=round(per_ch, 4), curve_delta=round(curve_d, 4),
                    diff_pct=round(parity_pct, 6)),
        causality=dict(checks=n_checks, mismatches=mism),
        ann_carry_yield_pct=round(annualized_yield(trades, years), 3),
        per_year=per_year(trades),
        worst_price_leg_squeezes=worst_price_leg(trades),
        drawdown_episodes=dd_eps,
        worst_book_dd_pct=worst_book_dd,
        A1_net_gt0_and_pf_ge_130=bool(S.get("net", -1) > 0 and S.get("pf", 0) >= 1.30),
        A2_funding_ge_40pct_gp=bool(S.get("funding_frac_of_gp", 0) >= 0.40),
        A3_no_episode_lt_neg15pct=bool(worst_book_dd > -15.0),
    )
    print("FUNDING_CARRY_JSON_START")
    print(json.dumps(out, indent=2, default=str))
    print("FUNDING_CARRY_JSON_END")


if __name__ == "__main__":
    main()
