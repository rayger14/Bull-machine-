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

Exit code: 0 all-clear, 1 WARN, 2 FAIL. Writes results/rebuild/parity_report.json + .md.

Usage:
  python3 scripts/rebuild/parity_check.py [--pull] [--min-bars 100]
    --pull   scp the live_features dir from the server first
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

# Features worth drift-monitoring: STATIONARY decision inputs only.
# Raw price levels (close, ema, sma, atr-absolute) are excluded — they are
# non-stationary (BTC rises over years) so they always "drift" vs a multi-year
# baseline, which is meaningless for parity. Only bounded/z-scored/ratio features
# whose distribution should be regime-comparable belong here.
MONITOR = [
    "rsi_14", "adx", "atr_percentile", "volume_zscore", "liquidity_score",
    "wyckoff_score", "fusion_smc", "chop_score", "instability",
    "oi_change_4h", "taker_imbalance", "ls_ratio_extreme", "funding_Z",
    "fear_greed", "VIX_Z", "DXY_Z",
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
    ap.add_argument("--min-bars", type=int, default=100,
                    help="min live bars before distribution metrics are trusted")
    args = ap.parse_args()

    if args.pull:
        pull_from_server()

    live = load_live()
    # Memory-safe load: only the columns we actually compare (VM has 1GB RAM —
    # loading all 295 cols of V14 would peak ~400MB; this keeps it under ~15MB).
    import pyarrow.parquet as _pq
    avail = set(_pq.ParquetFile(V14).schema.names)
    need = [c for c in MONITOR if c in avail]
    base = pd.read_parquet(V14, columns=need) if need else pd.read_parquet(V14)
    if getattr(base.index, "tz", None) is None:
        base.index = base.index.tz_localize("UTC")

    n_live = len(live)
    # Sample floors: a feature's max/spread is only meaningful with enough bars.
    RANGE_MIN, COVERAGE_MIN = 30, 50
    report = {"n_live_bars": int(n_live), "min_bars": args.min_bars,
              "drift": {}, "range": {}, "coverage": [], "status": "OK"}
    worst = 0  # 0 ok, 1 warn, 2 fail

    if n_live == 0:
        report["status"] = "NO_DATA"
        _write(report)
        print("No live feature logs yet — nothing to check.")
        return 0
    if n_live < RANGE_MIN:
        report["status"] = "ACCUMULATING"
        report["note"] = f"only {n_live} live bars (<{RANGE_MIN}); accumulating — no checks run yet"
        _write(report)
        print(f"Accumulating: {n_live} live bars (need {RANGE_MIN} before checks are meaningful).")
        return 0

    # --- Layer 1+2: drift + range, per monitored feature ---
    insufficient = n_live < args.min_bars
    _errctx = np.errstate(all="ignore")
    _errctx.__enter__()
    for feat in MONITOR:
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

        # Distribution drift (only trustworthy at n>=min_bars)
        if not insufficient:
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

    # --- Layer 3: threshold coverage (needs enough bars to observe each feature's max) ---
    coverage_ready = n_live >= COVERAGE_MIN
    for yml in sorted(ARCH_DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(yml.read_text())
        except Exception:
            continue
        if not cfg or "thresholds" not in cfg:
            continue
        arch = cfg.get("name", yml.stem)
        for tkey, feat in GATE_FEATURE_MAP.items():
            if tkey not in cfg["thresholds"] or feat not in live.columns:
                continue
            val = cfg["thresholds"][tkey]
            lv = pd.to_numeric(live[feat], errors="coerce").to_numpy(dtype=float)
            lv = lv[~np.isnan(lv)]
            if len(lv) == 0:
                continue
            reachable = bool(np.nanmax(lv) >= val)
            frac = float(np.mean(lv >= val))
            if not coverage_ready:
                status = "insufficient"
            elif reachable:
                status = "ok"
            else:
                status = "FAIL_LOCKOUT"
                worst = max(worst, 2)
            report["coverage"].append({
                "archetype": arch, "gate": tkey, "feature": feat,
                "threshold": val, "live_max": round(float(np.nanmax(lv)), 4),
                "frac_passing": round(frac, 4), "status": status})

    report["status"] = {0: "OK", 1: "WARN", 2: "FAIL"}[worst]
    if insufficient:
        report["note"] = (f"only {n_live} live bars (<{args.min_bars}); drift metrics "
                          "skipped, range + coverage still checked")
    _write(report)
    _print(report)
    return worst


def _write(report):
    REPORT_J.parent.mkdir(parents=True, exist_ok=True)
    REPORT_J.write_text(json.dumps(report, indent=2))
    lines = [f"# Parity Report — {report['status']} ({report['n_live_bars']} live bars)", ""]
    if report.get("note"):
        lines += [f"_{report['note']}_", ""]
    fails = [f"- **{k}**: {v}" for k, v in report.get("range", {}).items() if v["status"] == "FAIL"]
    fails += [f"- **{k}** drift: {v}" for k, v in report.get("drift", {}).items() if v["status"] == "FAIL"]
    fails += [f"- **{c['archetype']}.{c['gate']}** lockout: live_max {c['live_max']} < {c['threshold']} ({c['feature']})"
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
            print(f"  LOCKOUT {c['archetype']}.{c['gate']}: live_max {c['live_max']} < {c['threshold']} ({c['feature']})")
    for k, v in report.get("range", {}).items():
        if v["status"] == "FAIL":
            print(f"  SCALE {k}: live[{v['live_min']},{v['live_max']}] vs base q01/q99 [{v['base_q01']},{v['base_q99']}]")
    for k, v in report.get("drift", {}).items():
        if v["status"] in ("WARN", "FAIL"):
            print(f"  DRIFT {k}: PSI={v['psi']} JS={v['js']} [{v['status']}]")
    print(f"report -> {REPORT_J}")


if __name__ == "__main__":
    sys.exit(main())
