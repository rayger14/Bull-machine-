#!/usr/bin/env python3
"""Nightly backtest/live feature-parity check.

Compares the LIVE feature stream (results/coinbase_paper/live_features/*.jsonl,
written by coinbase_runner._log_feature_row) against the V14 backtest store, the
exact monitor the parity study prescribed and the project lacked when the
liquidity_score scale mismatch silently locked out two top archetypes.

Three layers (industry_study_backtest_live_parity_2026_06_11.md):
  1. DISTRIBUTION drift per feature — PSI + Jensen-Shannon distance
       PSI  <0.10 ok | 0.10-0.25 WARN | >=0.25 FAIL
       JS   >=0.10 WARN | >=0.20 FAIL   (JS chosen over KS: KS is tail-insensitive)
  2. RANGE / scale assertion — live [min,max] vs baseline [q01,q99]
       live max < baseline q05  OR  live min > baseline q95  => FAIL (scale shift,
       the liquidity_score class of bug that distribution metrics can smooth over)
  3. THRESHOLD COVERAGE — for each archetype gate, does the live feature range
       actually SPAN the gate value? A gate the live data can never satisfy is a
       structural lockout (exactly the wick_trap 0.72 failure).

The lockout check is sample-independent and runs always; drift/range wait for
~30 days of live data (--min-bars) and run only on FAST per-bar features.

Exit code: 0 OK / 1 WARN / 2 FAIL (or always 0 with --exit-zero, for the timer
where 'failed' should mean the check crashed). Writes parity_report.{json,md}.

Usage:
  python3 scripts/rebuild/parity_check.py [--pull] [--baseline-days 90] [--exit-zero]
    --pull           scp the live_features dir from the server first (local runs)
    --baseline-days  drift baseline = last N days of V14 (regime-comparable)
    --exit-zero      always exit 0 on a successful run (for the systemd timer)
"""
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
V14 = REPO / "data/features_mtf/BTC_1H_FEATURES_V14_FULL_LIVE_PATH.parquet"
LIVE_DIR = REPO / "results/coinbase_paper/live_features"
ARCH_DIR = REPO / "configs/archetypes"
REPORT_J = REPO / "results/rebuild/parity_report.json"
REPORT_M = REPO / "results/rebuild/parity_report.md"
SERVER = "ubuntu@165.1.79.19"
SSH_KEY = "~/.ssh/oracle_bullmachine"

# Gate-threshold -> feature mappings (min-gates: live feature must be able to exceed value)
GATE_FEATURE_MAP = {
    "liquidity_threshold": "liquidity_score",
    "adx_threshold": "adx",
    "rsi_threshold": "rsi_14",
    "volume_z_min": "volume_zscore",
    "vol_z": "volume_zscore",
    "atr_percentile_min": "atr_percentile",
}

# Drift/range monitoring applies ONLY to FAST per-bar engine features — those
# recomputed fresh every hour, whose hourly distribution is meaningful.
# EXCLUDED on purpose:
#  - raw price levels (close/ema/sma): non-stationary, always "drift" over years
#  - slow MACRO/derivative series (DXY_Z, VIX_Z, funding_Z, fear_greed,
#    oi_change_4h, ls_ratio_extreme): daily-or-slower values forward-filled to
#    hourly, so their hourly "distribution" is autocorrelation-dominated and a
#    short live window sits at one point in their slow cycle — drift/scale on them
#    is noise, not signal. (Macro is better watched via the heartbeat/macro_outlook.)
MONITOR = [
    "rsi_14", "adx", "atr_percentile", "volume_zscore", "liquidity_score",
    "wyckoff_score", "fusion_smc", "chop_score", "instability", "taker_imbalance",
]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using baseline deciles as bins."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 20:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    eps = 1e-6
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def js_distance(expected: np.ndarray, actual: np.ndarray, bins: int = 30) -> float:
    """Jensen-Shannon distance (sqrt of JS divergence), 0..1."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 20:
        return float("nan")
    lo, hi = np.min(expected), np.max(expected)
    if hi <= lo:
        return float("nan")
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(expected, edges)[0].astype(float)
    q = np.histogram(actual, edges)[0].astype(float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-12
    def kl(x, y): return np.sum(np.where(x > 0, x * np.log((x + eps) / (y + eps)), 0))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(jsd, 0.0)) / np.sqrt(np.log(2)))


def load_live() -> pd.DataFrame:
    files = sorted(glob.glob(str(LIVE_DIR / "*.jsonl")))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_json(f, lines=True) for f in files if Path(f).stat().st_size > 0]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    return df


def pull_from_server():
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["scp", "-i", SSH_KEY.replace("~", str(Path.home())), "-q",
           f"{SERVER}:Bull-machine-/results/coinbase_paper/live_features/*.jsonl",
           str(LIVE_DIR) + "/"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(f"pull: {'ok' if r.returncode == 0 else 'FAILED ' + r.stderr[-200:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--min-bars", type=int, default=720,
                    help="min live bars (~30d) before drift/range metrics run; below "
                         "this only the sample-independent lockout check runs")
    ap.add_argument("--baseline-days", type=int, default=90,
                    help="compare live drift against the most recent N days of V14 "
                         "(regime-comparable) rather than all history; 0 = full store")
    ap.add_argument("--exit-zero", action="store_true",
                    help="always exit 0 on a successful run (verdict still in report/log); "
                         "for the systemd timer, where 'failed' should mean the check crashed")
    args = ap.parse_args()

    if args.pull:
        pull_from_server()

    live = load_live()
    # Memory-safe load: only the columns we actually compare (VM has 1GB RAM —
    # loading all 295 cols of V14 would peak ~400MB; this keeps it under ~15MB).
    import pyarrow.parquet as _pq
    avail = set(_pq.ParquetFile(V14).schema.names)
    need = [c for c in MONITOR if c in avail]
    base_full = pd.read_parquet(V14, columns=need) if need else pd.read_parquet(V14)
    if getattr(base_full.index, "tz", None) is None:
        base_full.index = base_full.index.tz_localize("UTC")
    # Drift/range baseline = recent regime-comparable window (default last 90d).
    # Comparing a few live days against all 8 years conflates pipeline divergence
    # (what we want) with current-regime-differs-from-history (expected, not a bug).
    # base_full is still used for the sample-independent lockout check below.
    if args.baseline_days and len(base_full):
        cutoff = base_full.index.max() - pd.Timedelta(days=args.baseline_days)
        base = base_full.loc[base_full.index >= cutoff]
    else:
        base = base_full

    n_live = len(live)
    report = {"n_live_bars": int(n_live), "min_bars": args.min_bars,
              "drift": {}, "range": {}, "coverage": [], "status": "OK"}
    worst = 0  # 0 ok, 1 warn, 2 fail

    # --- Layer 1+2: drift + range — only once enough live data accumulates ---
    # The lockout check (Layer 3) is sample-independent and runs always, below.
    drift_ready = n_live >= args.min_bars
    _errctx = np.errstate(all="ignore")
    _errctx.__enter__()
    for feat in (MONITOR if drift_ready else []):
        if feat not in live.columns or feat not in base.columns:
            continue
        lv = pd.to_numeric(live[feat], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(base[feat], errors="coerce").to_numpy(dtype=float)
        bv = bv[~np.isnan(bv)]
        if len(bv) < 100:
            continue

        # Range / scale (works even at small n — this is the scale-bug catcher)
        lmin, lmax = np.nanmin(lv), np.nanmax(lv)
        bq = np.quantile(bv, [0.01, 0.05, 0.50, 0.95, 0.99])
        rstat = "ok"
        if lmax < bq[1]:
            rstat = "FAIL"; worst = max(worst, 2)
        elif lmin > bq[3]:
            rstat = "FAIL"; worst = max(worst, 2)
        report["range"][feat] = {
            "live_min": round(float(lmin), 4), "live_max": round(float(lmax), 4),
            "base_q01": round(float(bq[0]), 4), "base_q50": round(float(bq[2]), 4),
            "base_q99": round(float(bq[4]), 4), "status": rstat}

        # Distribution drift
        p = psi(bv, lv)
        j = js_distance(bv, lv)
        dstat = "ok"
        if (p == p and p >= 0.25) or (j == j and j >= 0.20):
            dstat = "FAIL"; worst = max(worst, 2)
        elif (p == p and p >= 0.10) or (j == j and j >= 0.10):
            dstat = "WARN"; worst = max(worst, 1)
        report["drift"][feat] = {
            "psi": None if p != p else round(p, 4),
            "js": None if j != j else round(j, 4), "status": dstat}

    # --- Layer 3: threshold coverage — SAMPLE-INDEPENDENT structural lockout ---
    # A gate is a true lockout only if the feature can't reach it in the HONEST
    # FULL STORE (V14, 8 years of live-path values) — not merely if a small live
    # sample hasn't hit it yet. base_full max is the best estimate of the live
    # ceiling; live frac is reported as info. This is the liquidity_score 0.72
    # (V14 max 0.675) class, immune to small-n false positives like a quiet week
    # never sampling high atr_percentile.
    for yml in sorted(ARCH_DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(yml.read_text())
        except Exception:
            continue
        if not cfg or "thresholds" not in cfg:
            continue
        arch = cfg.get("name", yml.stem)
        for tkey, feat in GATE_FEATURE_MAP.items():
            if tkey not in cfg["thresholds"] or feat not in base_full.columns:
                continue
            val = cfg["thresholds"][tkey]
            fullv = pd.to_numeric(base_full[feat], errors="coerce").to_numpy(dtype=float)
            full_max = float(np.nanmax(fullv))
            lv = pd.to_numeric(live[feat], errors="coerce").to_numpy(dtype=float) if feat in live.columns else np.array([])
            lv = lv[~np.isnan(lv)]
            live_frac = float(np.mean(lv >= val)) if len(lv) else None
            if full_max < val:
                status = "FAIL_LOCKOUT"
                worst = max(worst, 2)
            else:
                status = "ok"
            report["coverage"].append({
                "archetype": arch, "gate": tkey, "feature": feat,
                "threshold": val, "store_max": round(full_max, 4),
                "live_frac_passing": None if live_frac is None else round(live_frac, 4),
                "status": status})

    report["status"] = {0: "OK", 1: "WARN", 2: "FAIL"}[worst]
    if not drift_ready:
        report["note"] = (f"{n_live}/{args.min_bars} live bars — lockout check ran "
                          "(sample-independent); drift/range accumulating until ~30d of data")
    report["baseline_days"] = args.baseline_days
    _write(report)
    _print(report)
    return 0 if args.exit_zero else worst


def _write(report):
    REPORT_J.parent.mkdir(parents=True, exist_ok=True)
    REPORT_J.write_text(json.dumps(report, indent=2))
    lines = [f"# Parity Report — {report['status']} ({report['n_live_bars']} live bars)", ""]
    if report.get("note"):
        lines += [f"_{report['note']}_", ""]
    fails = [f"- **{k}**: {v}" for k, v in report.get("range", {}).items() if v["status"] == "FAIL"]
    fails += [f"- **{k}** drift: {v}" for k, v in report.get("drift", {}).items() if v["status"] == "FAIL"]
    fails += [f"- **{c['archetype']}.{c['gate']}** lockout: store_max {c['store_max']} < {c['threshold']} ({c['feature']})"
              for c in report.get("coverage", []) if c["status"] == "FAIL_LOCKOUT"]
    if fails:
        lines += ["## FAILURES", *fails, ""]
    warns = [f"- **{k}** drift: PSI={v['psi']} JS={v['js']}" for k, v in report.get("drift", {}).items() if v["status"] == "WARN"]
    if warns:
        lines += ["## WARNINGS", *warns, ""]
    REPORT_M.write_text("\n".join(lines))


def _print(report):
    print(f"\n=== PARITY: {report['status']} ({report['n_live_bars']} live bars) ===")
    if report.get("note"):
        print(report["note"])
    for c in report.get("coverage", []):
        if c["status"] == "FAIL_LOCKOUT":
            print(f"  LOCKOUT {c['archetype']}.{c['gate']}: store_max {c['store_max']} < {c['threshold']} ({c['feature']})")
    for k, v in report.get("range", {}).items():
        if v["status"] == "FAIL":
            print(f"  SCALE {k}: live[{v['live_min']},{v['live_max']}] vs base q01/q99 [{v['base_q01']},{v['base_q99']}]")
    for k, v in report.get("drift", {}).items():
        if v["status"] in ("WARN", "FAIL"):
            print(f"  DRIFT {k}: PSI={v['psi']} JS={v['js']} [{v['status']}]")
    print(f"report -> {REPORT_J}")


if __name__ == "__main__":
    sys.exit(main())
