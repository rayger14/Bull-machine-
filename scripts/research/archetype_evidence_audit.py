#!/usr/bin/env python3
"""Read-only live evidence accounting. This does not backtest or select a strategy.

Exit-level numbers reproduce the dashboard. Completed-position statistics require
an ID and a terminal exit; partial and unidentified records remain separate.
Weekly cluster intervals describe this selected historical sample, not future P&L.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import random
import statistics


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"Timezone missing from timestamp: {value}")
    return dt.astimezone(timezone.utc)


def metrics(pnls):
    gp = sum(p for p in pnls if p > 0)
    gl = -sum(p for p in pnls if p < 0)
    wins = sum(p > 0 for p in pnls)
    losses = sum(p < 0 for p in pnls)
    return dict(n=len(pnls), wins=wins, losses=losses, pnl=round(sum(pnls), 2),
                win_rate_pct=round(100 * wins / len(pnls), 2) if pnls else None,
                profit_factor=round(gp / gl, 4) if gl else None,
                gross_profit=round(gp, 2), gross_loss=round(gl, 2),
                average_win=round(gp / wins, 2) if wins else None,
                average_loss=round(-gl / losses, 2) if losses else None)


def weekly_interval(positions):
    """90% percentile CI of mean position dollars; resample entry-week clusters."""
    clusters = defaultdict(list)
    for p in positions:
        dt = timestamp(p["entry"])
        monday = (dt - timedelta(days=dt.weekday())).date().isoformat()
        clusters[monday].append(p["pnl"])
    weeks = list(clusters.values())
    if len(weeks) < 8:
        return dict(entry_weeks=len(weeks), mean_pnl_90pct_interval=None)
    rng = random.Random(1729)
    means = []
    for _ in range(2000):
        sample = [v for week in rng.choices(weeks, k=len(weeks)) for v in week]
        means.append(sum(sample) / len(sample))
    means.sort()
    return dict(entry_weeks=len(weeks),
                mean_pnl_90pct_interval=[round(means[99], 2), round(means[1899], 2)])


def group_positions(trades, open_ids):
    by_id = defaultdict(list)
    unknown = []
    for t in trades:
        if t.get("position_id"):
            by_id[t["position_id"]].append(t)
        else:
            unknown.append(t)
    positions = []
    for pid, fills in by_id.items():
        identity = {(f["archetype"], f["direction"], timestamp(f["timestamp_entry"]))
                    for f in fills}
        if len(identity) != 1:
            raise ValueError(f"Conflicting position identity: {pid}")
        fills = sorted(fills, key=lambda f: timestamp(f["timestamp_exit"]))
        first, last = fills[0], fills[-1]
        reason = last.get("exit_reason", "").lower()
        # Do not claim closure from an arbitrary nonempty reason. The caller may
        # be using a different schema or an incomplete capture.
        terminal = (reason in {"stop_loss", "take_profit", "time_exit", "max_hold"}
                    or reason.startswith(("time exit", "s1 target", "invalidation",
                                          "trailing stop", "trailing_stop")))
        state = "open" if pid in open_ids else "completed" if terminal else "unresolved"
        qty = sum(abs(float(f.get("quantity") or 0)) for f in fills)
        entry = float(first.get("entry_price") or 0)
        stop = float(first.get("stop_loss") or 0)
        risk = qty * abs(entry - stop) if entry > 0 and stop > 0 else None
        positions.append(dict(id=pid, archetype=first["archetype"],
                              direction=first["direction"], entry=first["timestamp_entry"],
                              exit=last["timestamp_exit"], pnl=sum(f["pnl_usd"] for f in fills),
                              state=state, exit_rows=len(fills),
                              reconstructed_initial_risk=risk,
                              exited_notional=qty * entry, last_exit_reason=reason))
    return positions, unknown


def summarize(trades, positions, unknown):
    closed = [p for p in positions if p["state"] == "completed"]
    opened = [p for p in positions if p["state"] == "open"]
    unresolved = [p for p in positions if p["state"] == "unresolved"]
    risks = [p["reconstructed_initial_risk"] for p in closed
             if p["reconstructed_initial_risk"] is not None]
    pnls = [t["pnl_usd"] for t in trades]
    return dict(exit_rows=len(trades), realized_pnl=round(sum(pnls), 2),
                exits=metrics(pnls), identified_positions=len(positions),
                completed=metrics([p["pnl"] for p in closed]),
                open_positions_with_exits=len(opened),
                open_realized_pnl=round(sum(p["pnl"] for p in opened), 2),
                unresolved_positions=len(unresolved),
                unresolved_realized_pnl=round(sum(p["pnl"] for p in unresolved), 2),
                unidentified_exit_rows=len(unknown),
                unidentified_realized_pnl=round(sum(t["pnl_usd"] for t in unknown), 2),
                average_reconstructed_initial_risk=round(statistics.mean(risks), 2) if risks else None,
                reconstructed_risk_n=len(risks),
                uncertainty=weekly_interval(closed))


def audit(trades, status, roster, cutoff):
    normalized = []
    for t in trades:
        t = dict(t)
        pnl = float(t["pnl_usd"] if "pnl_usd" in t else t["pnl"])
        if not math.isfinite(pnl):
            raise ValueError("Nonfinite P&L")
        if timestamp(t["timestamp_exit"]) < timestamp(t["timestamp_entry"]):
            raise ValueError("Exit precedes entry")
        t["pnl_usd"] = pnl
        normalized.append(t)
    hb = status.get("heartbeat", {})
    opened = hb.get("open_position_details") or []
    open_ids = {p["id"] for p in opened if p.get("id")}
    unexpected = sorted({t["archetype"] for t in normalized} - set(roster))
    names = list(roster) + unexpected
    report = dict(roster=list(roster), unexpected_archetypes=unexpected,
                  entry_cutoff=timestamp(cutoff).isoformat(),
                  performance_snapshot=status.get("performance", {}),
                  open_ids=sorted(open_ids),
                  limitations=[
                      "Live paper fills; not evidence of real-money execution quality.",
                      "P&L is reported dollars, with changing notional/configuration; not a constant-risk return.",
                      "No unrealized P&L or separate funding is added to exit totals.",
                      "Missing IDs are unidentified exits, never invented independent trades.",
                      "A terminal reason establishes apparent closure; no entry-fill ledger was supplied.",
                      "Bootstrap resamples observed entry weeks; regime nonstationarity and selection remain.",
                      "This is descriptive, not WFO/CPCV or a new independent holdout.",
                  ])
    for label, subset in [("all_history", normalized),
                          ("post_cutoff", [t for t in normalized
                                           if timestamp(t["timestamp_entry"]) >= timestamp(cutoff)])]:
        positions, unknown = group_positions(subset, open_ids)
        monthly = defaultdict(list)
        for t in subset:
            month = timestamp(t["timestamp_exit"]).strftime("%Y-%m")
            monthly[month].append(t["pnl_usd"])
        rows = {}
        for name in names:
            rows[name] = summarize([t for t in subset if t["archetype"] == name],
                                   [p for p in positions if p["archetype"] == name],
                                   [t for t in unknown if t["archetype"] == name])
        report[label] = dict(totals=summarize(subset, positions, unknown),
                             archetypes=rows, positions=positions,
                             monthly_exit_pnl={k: metrics(v) for k, v in sorted(monthly.items())})
    report["coverage"] = dict(
        first_entry=min((t["timestamp_entry"] for t in normalized), key=timestamp, default=None),
        last_exit=max((t["timestamp_exit"] for t in normalized), key=timestamp, default=None))
    report["reconciliation"] = dict(
        all_history_pnl=report["all_history"]["totals"]["realized_pnl"],
        filtered_pnl=report["post_cutoff"]["totals"]["realized_pnl"],
        status_filtered_pnl=status.get("performance", {}).get("total_pnl"),
        performance_open_count=status.get("performance", {}).get("open_positions"),
        heartbeat_open_detail_count=len(opened))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--archetype-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-03-09T00:00:00Z")
    args = parser.parse_args()
    import yaml
    configs = {}
    for path in sorted(args.archetype_dir.glob("*.yaml")):
        if "example" in path.stem:
            continue
        cfg = yaml.safe_load(path.read_text())
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        name = cfg.get("name")
        if not name:
            continue
        if name in configs:
            raise ValueError(f"Duplicate archetype name: {name}")
        configs[name] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             direction=cfg.get("direction"), gate_mode=cfg.get("gate_mode"))
    if not configs:
        raise ValueError("No archetype configurations found")
    result = audit(json.loads(args.trades.read_text()), json.loads(args.status.read_text()),
                   list(configs), args.cutoff)
    result["provenance"] = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [args.trades, args.status]}, configs=configs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scorecard.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    columns = ["view", "archetype", "exit_rows", "realized_pnl", "completed_positions",
               "completed_pnl", "position_win_rate_pct", "position_pf", "unidentified_exits",
               "open_with_exits", "unresolved_positions", "entry_weeks", "mean_pnl_ci_low", "mean_pnl_ci_high"]
    with (args.out / "scorecard.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for view in ["all_history", "post_cutoff"]:
            for name, row in result[view]["archetypes"].items():
                ci = row["uncertainty"]["mean_pnl_90pct_interval"] or [None, None]
                writer.writerow(dict(view=view, archetype=name, exit_rows=row["exit_rows"],
                    realized_pnl=row["realized_pnl"], completed_positions=row["completed"]["n"],
                    completed_pnl=row["completed"]["pnl"], position_win_rate_pct=row["completed"]["win_rate_pct"],
                    position_pf=row["completed"]["profit_factor"], unidentified_exits=row["unidentified_exit_rows"],
                    open_with_exits=row["open_positions_with_exits"], unresolved_positions=row["unresolved_positions"],
                    entry_weeks=row["uncertainty"]["entry_weeks"], mean_pnl_ci_low=ci[0], mean_pnl_ci_high=ci[1]))
    print(json.dumps(dict(roster_count=len(configs), coverage=result["coverage"],
                          reconciliation=result["reconciliation"],
                          post_cutoff=result["post_cutoff"]["totals"]), indent=2))


if __name__ == "__main__":
    main()
