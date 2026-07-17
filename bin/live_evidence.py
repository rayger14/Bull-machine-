#!/usr/bin/env python3
"""Live Evidence Engine — turn collected paper-trading metadata into decisions.

Reads the three metadata streams the live runner already produces
(trades.json, signal_log.json, phantom_trades.json) and prints:

  1. SCORECARD  — per-archetype live WR/PF with Wilson confidence bounds,
                  compared against the offline-validated holdout expectation.
  2. COUNTERFACTUAL — enforced-threshold subset (threshold_margin >= 0) vs
                  bypass-only extras: what would the validated config have
                  done live? Plus phantom-trade summary (skipped signals).
  3. WATCH-ITEM LEDGER — LC early-weakness time-cut (pre-registered trigger:
                  deploy at >=15 cut events with <=1 baseline-winner).
                  Paths reconstructed from Binance 1h klines.
  4. EXECUTION COST — simulated fees/slippage paid, maker-first recovery.

Read-only analytics. No strategy changes, no production writes.

Usage:
    python3 bin/live_evidence.py [--dir results/coinbase_paper] [--no-network]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

# Offline-validated holdout expectations (2025-01-01 -> 2026-06-10)
# wick_trap/LC on V14L; order_block_retest validated 2026-07-17 on V15
# (post BOS/CHoCH/index repairs): holdout PF 2.08 n=31, CPCV 15/15 med 1.39.
VALIDATED = {
    "wick_trap": {"pf": 1.43, "note": "CPCV 15/15, holdout n=70"},
    "liquidity_compression": {"pf": 1.14, "note": "co-move 1.14/1.14"},
    "order_block_retest": {"pf": 2.08, "note": "V15 CPCV 15/15, holdout n=31 — small n, expect regression toward CPCV median 1.39"},
}
COMMISSION_RATE = 0.0002   # per side, taker (matches backtest & paper sim)
SLIPPAGE_BPS = 3
MAKER_RATE = 0.0000        # BTC-PERP-INTX maker rebate tier ~0; conservative 0
TIMECUT_MFE_R = 0.25       # pre-registered LC watch-item cell: x0.25_h24
TIMECUT_HOURS = 24
TRIGGER_N = 15
TRIGGER_MAX_WINNERS = 1


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def profit_factor(pnls: list[float]) -> float:
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    if gl == 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def load_positions(trades_path: Path) -> list[dict]:
    """Group scale-out fills into positions by position_id."""
    fills = json.loads(trades_path.read_text())
    by_pos: dict[str, list[dict]] = defaultdict(list)
    for f in fills:
        by_pos[f["position_id"]].append(f)
    positions = []
    for pid, fs in by_pos.items():
        fs.sort(key=lambda f: f["timestamp_exit"])
        first = fs[0]
        positions.append({
            "position_id": pid,
            "archetype": first["archetype"],
            "direction": first["direction"],
            "entry_ts": first["timestamp_entry"],
            "exit_ts": fs[-1]["timestamp_exit"],
            "entry_price": first["entry_price"],
            "stop_loss": first.get("stop_loss"),
            "pnl": sum(f["pnl_usd"] for f in fs),
            "fusion_score": first.get("fusion_score"),
            "threshold_margin": first.get("threshold_margin"),
            "size_usd": first.get("position_size_usd") or 0.0,
            "qty": sum(abs(f.get("quantity") or 0.0) for f in fs),
            "n_fills": len(fs),
            "exit_reasons": [f.get("exit_reason") for f in fs],
        })
    positions.sort(key=lambda p: p["entry_ts"])
    return positions


def scorecard(positions: list[dict]) -> None:
    print("=" * 78)
    print("1. LIVE SCORECARD — per archetype vs validated expectation")
    print("=" * 78)
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for p in positions:
        by_arch[p["archetype"]].append(p)
    print(f"  {'archetype':<24}{'n':>4}{'WR':>6}{'WR 95% CI':>14}{'PF':>7}"
          f"{'PnL$':>10}  expectation")
    for arch, ps in sorted(by_arch.items(), key=lambda kv: -sum(p['pnl'] for p in kv[1])):
        pnls = [p["pnl"] for p in ps]
        wins = sum(1 for x in pnls if x > 0)
        lo, hi = wilson(wins, len(ps))
        pf = profit_factor(pnls)
        exp = VALIDATED.get(arch)
        if exp:
            verdict = f"holdout PF {exp['pf']:.2f} ({exp['note']})"
            if len(ps) < 15:
                verdict += " — n too small to judge"
            elif pf >= exp["pf"] * 0.75:
                verdict += " — ON TRACK"
            else:
                verdict += " — UNDERPERFORMING"
        else:
            verdict = "data-collection only (not validated)"
        pf_s = f"{pf:.2f}" if math.isfinite(pf) else "inf"
        print(f"  {arch:<24}{len(ps):>4}{wins/len(ps):>6.0%}"
              f"{f'[{lo:.0%},{hi:.0%}]':>14}{pf_s:>7}{sum(pnls):>10.0f}  {verdict}")
    pnls = [p["pnl"] for p in positions]
    print(f"\n  TOTAL: {len(positions)} positions, PF {profit_factor(pnls):.2f}, "
          f"PnL ${sum(pnls):,.0f}")


def counterfactual(positions: list[dict], signal_log_path: Path,
                   phantom_path: Path) -> None:
    print("\n" + "=" * 78)
    print("2. COUNTERFACTUAL — what would the VALIDATED config have done live?")
    print("=" * 78)
    enforced = [p for p in positions
                if p["threshold_margin"] is not None and p["threshold_margin"] >= 0]
    bypass = [p for p in positions
              if p["threshold_margin"] is None or p["threshold_margin"] < 0]
    for label, ps in [("threshold-cleared (validated config takes these)", enforced),
                      ("bypass-only extras (data collection)", bypass)]:
        if not ps:
            print(f"  {label}: none")
            continue
        pnls = [p["pnl"] for p in ps]
        wins = sum(1 for x in pnls if x > 0)
        pf = profit_factor(pnls)
        pf_s = f"{pf:.2f}" if math.isfinite(pf) else "inf"
        print(f"  {label}:")
        print(f"      n={len(ps)}  WR {wins/len(ps):.0%}  PF {pf_s}  "
              f"PnL ${sum(pnls):,.0f}")
        by_arch: dict[str, list[float]] = defaultdict(list)
        for p in ps:
            by_arch[p["archetype"]].append(p["pnl"])
        top = sorted(by_arch.items(), key=lambda kv: -abs(sum(kv[1])))[:4]
        for arch, ap in top:
            print(f"        {arch:<24} n={len(ap):<3} PnL ${sum(ap):>8,.0f}")

    sl = json.loads(signal_log_path.read_text())
    rejected = [s for s in sl if s.get("status") == "rejected"]
    stages = defaultdict(int)
    for s in rejected:
        stages[s.get("rejection_stage") or s.get("rejection_reason") or "?"] += 1
    print(f"\n  signal log (last {len(sl)}): {len(rejected)} rejected — "
          + ", ".join(f"{k}={v}" for k, v in sorted(stages.items(), key=lambda kv: -kv[1])))

    ph = json.loads(phantom_path.read_text())
    print(f"  phantoms (skipped/blocked signals replayed): "
          f"{ph.get('completed_phantom_trades')} completed, "
          f"WR {ph.get('phantom_win_rate')}%, PnL ${ph.get('phantom_pnl')}")
    by_arch_ph = ph.get("by_archetype") or {}
    if by_arch_ph:
        worst = sorted(by_arch_ph.items(),
                       key=lambda kv: kv[1].get("pnl", 0) if isinstance(kv[1], dict) else 0)
        for arch, st in worst[:3]:
            if isinstance(st, dict):
                print(f"      phantom {arch:<22} n={st.get('count', st.get('n', '?'))} "
                      f"PnL ${st.get('pnl', 0):,.0f}")


KLINE_HOSTS = ["api.binance.com", "api.binance.us"]  # .com geo-blocks (451) from US


def fetch_klines(start_ms: int, end_ms: int) -> dict[int, tuple[float, float]]:
    """Binance 1h klines -> {open_time_ms: (high, low)}. Public endpoint, paginated."""
    out: dict[int, tuple[float, float]] = {}
    last_err: Exception | None = None
    for host in KLINE_HOSTS:
        try:
            cur = start_ms
            while cur < end_ms:
                url = (f"https://{host}/api/v3/klines?symbol=BTCUSDT&interval=1h"
                       f"&startTime={cur}&endTime={end_ms}&limit=1000")
                with urllib.request.urlopen(url, timeout=30) as r:
                    rows = json.loads(r.read())
                if not rows:
                    break
                for k in rows:
                    out[int(k[0])] = (float(k[2]), float(k[3]))
                cur = int(rows[-1][0]) + 3_600_000
                if len(rows) < 1000:
                    break
            return out
        except Exception as e:
            last_err = e
            out.clear()
    raise last_err or RuntimeError("no kline host reachable")


def timecut_ledger(positions: list[dict], no_network: bool) -> None:
    print("\n" + "=" * 78)
    print(f"3. WATCH-ITEM LEDGER — LC time-cut (MFE < {TIMECUT_MFE_R}R by "
          f"{TIMECUT_HOURS}h)")
    print(f"   pre-registered deploy trigger: >= {TRIGGER_N} cut events with "
          f"<= {TRIGGER_MAX_WINNERS} baseline-winner")
    print("=" * 78)
    lc = [p for p in positions if p["archetype"] == "liquidity_compression"
          and p.get("stop_loss") and p["direction"] == "long"]
    if not lc:
        print("  no LC positions yet")
        return
    if no_network:
        print(f"  {len(lc)} LC positions on file — skipped (--no-network)")
        return
    import datetime as dt

    def to_ms(ts: str) -> int:
        return int(dt.datetime.fromisoformat(ts).timestamp() * 1000)

    start = min(to_ms(p["entry_ts"]) for p in lc)
    end = max(to_ms(p["exit_ts"]) for p in lc) + 3_600_000
    try:
        kl = fetch_klines(start, end)
    except Exception as e:  # network failure -> report and move on
        print(f"  kline fetch failed ({e}) — rerun when network is available")
        return
    cut_events, cut_winners, insufficient = [], 0, 0
    for p in lc:
        R = abs(p["entry_price"] - p["stop_loss"])
        if R <= 0:
            continue
        t0 = to_ms(p["entry_ts"]) - to_ms(p["entry_ts"]) % 3_600_000
        t_exit = to_ms(p["exit_ts"])
        horizon = min(t0 + TIMECUT_HOURS * 3_600_000, t_exit)
        mfe = 0.0
        bars = 0
        t = t0
        while t < horizon:
            if t in kl:
                mfe = max(mfe, (kl[t][0] - p["entry_price"]) / R)
                bars += 1
            t += 3_600_000
        held_h = (t_exit - t0) / 3_600_000
        if held_h < TIMECUT_HOURS:
            insufficient += 1   # exited before the horizon: cut can't fire
            continue
        if bars >= TIMECUT_HOURS // 2 and mfe < TIMECUT_MFE_R:
            cut_events.append(p)
            if p["pnl"] > 0:
                cut_winners += 1
    print(f"  LC long positions: {len(lc)}  |  exited before {TIMECUT_HOURS}h: "
          f"{insufficient}")
    print(f"  CUT EVENTS: {len(cut_events)} / {TRIGGER_N} needed  |  "
          f"baseline-winners among them: {cut_winners} (max {TRIGGER_MAX_WINNERS})")
    if cut_events:
        saved = -sum(p["pnl"] for p in cut_events)
        print(f"  PnL of cut-event positions: ${sum(p['pnl'] for p in cut_events):,.0f} "
              f"(a live time-cut would have avoided ~${saved:,.0f} of that, "
              f"minus the ~-{TIMECUT_MFE_R}R cut cost)")
    status = ("TRIGGERED — run the deployment adjudication"
              if len(cut_events) >= TRIGGER_N and cut_winners <= TRIGGER_MAX_WINNERS
              else "accumulating")
    print(f"  status: {status}")


def maker_shadow_report(ledger_path: Path) -> None:
    if not ledger_path.exists():
        print("  maker-shadow ledger: not started yet (deploys with the runner)")
        return
    recs = [json.loads(l) for l in ledger_path.read_text().splitlines() if l.strip()]
    if not recs:
        print("  maker-shadow ledger: empty (no entries resolved yet)")
        return
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_arch[r["archetype"]].append(r)
    print(f"\n  MAKER-SHADOW LEDGER ({len(recs)} resolved entries):")
    print(f"  {'archetype':<24}{'n':>4}{'fill rate':>10}{'avg chase on miss':>19}")
    for arch, rs in sorted(by_arch.items(), key=lambda kv: -len(kv[1])):
        fills = [r for r in rs if r["filled"]]
        misses = [r for r in rs if not r["filled"]]
        chase = (sum(r["chase_bps"] for r in misses) / len(misses)) if misses else 0.0
        print(f"  {arch:<24}{len(rs):>4}{len(fills)/len(rs):>10.0%}"
              f"{chase:>17.0f}bp")
    fills = sum(1 for r in recs if r["filled"])
    print(f"  overall fill rate: {fills/len(recs):.0%} — flip an archetype to "
          f"maker-first only if its fill rate is high AND misses chase small")


def execution_costs(positions: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("4. EXECUTION COSTS — simulated fees/slippage on paper fills")
    print("=" * 78)
    notional = sum(p["qty"] * p["entry_price"] for p in positions) * 2  # in+out
    taker_fees = notional * COMMISSION_RATE
    slip = notional * SLIPPAGE_BPS / 10_000
    maker_fees = notional * MAKER_RATE
    total_pnl = sum(p["pnl"] for p in positions)
    print(f"  round-trip notional traded: ${notional:,.0f}")
    print(f"  taker fees paid (sim {COMMISSION_RATE*1e4:.0f}bp/side): ${taker_fees:,.0f}")
    print(f"  slippage modeled ({SLIPPAGE_BPS}bp): ${slip:,.0f}")
    print(f"  net PnL: ${total_pnl:,.0f} -> costs are "
          f"{abs((taker_fees+slip)/total_pnl)*100:.0f}% of |net PnL|"
          if total_pnl else f"  net PnL: $0")
    print(f"  maker-first entries would recover up to "
          f"${(taker_fees - maker_fees)/2 + slip/2:,.0f} "
          f"(half of fees+slip; entries only, exits stay taker)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/coinbase_paper")
    ap.add_argument("--no-network", action="store_true",
                    help="skip the kline-based watch-item ledger")
    args = ap.parse_args()
    d = Path(args.dir)
    trades, sl, ph = d / "trades.json", d / "signal_log.json", d / "phantom_trades.json"
    for f in (trades, sl, ph):
        if not f.exists():
            sys.exit(f"missing {f}")
    positions = load_positions(trades)
    first, last = positions[0]["entry_ts"][:10], positions[-1]["entry_ts"][:10]
    print(f"LIVE EVIDENCE REPORT — {len(positions)} positions, {first} -> {last}\n")
    scorecard(positions)
    counterfactual(positions, sl, ph)
    timecut_ledger(positions, args.no_network)
    execution_costs(positions)
    maker_shadow_report(d / "maker_shadow.jsonl")


if __name__ == "__main__":
    main()
