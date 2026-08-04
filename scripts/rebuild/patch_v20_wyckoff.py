#!/usr/bin/env python3
"""V20 wyckoff patch — recompute Wyckoff event columns with repaired code.

Why: 2026-08-04 audit found (a) detect_spring_type_b read future closes in
batch runs (look-ahead; fixed to the spring_a candidate/confirm pattern),
(b) the state machine discarded nearly all springs/UTs because SC's
triple-extreme gate almost never validates a structure (fixed with the
sanctioned SOS/SOW-style no-context 0.5x-confidence fallback), and
(c) wyckoff_phase_abc conflates accumulation/distribution direction (new
directional columns: wyckoff_phase_dir, wyckoff_context).

Replays bin/live/live_feature_computer._wyckoff_features() bar-by-bar over a
bounded 1,000-bar buffer — the same path live uses — so store == live parity
holds by construction. Only the wyckoff family is touched; V18's other
columns are carried through unchanged.

Usage:
  python3 scripts/rebuild/patch_v20_wyckoff.py --chunk 1 6      # worker
  python3 scripts/rebuild/patch_v20_wyckoff.py --limit 300      # smoke
  python3 scripts/rebuild/patch_v20_wyckoff.py --stitch         # splice V20
Chunks checkpoint every 2,000 bars and are resumable.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V18_ROTATION.parquet"
OUT_DIR = REPO / "results/rebuild/v20_wyckoff"
V20 = REPO / "data/features_mtf/BTC_1H_FEATURES_V20_WYCKOFF.parquet"
WARMUP = 1000
CHECKPOINT_EVERY = 2000
# String columns worth keeping in the store (all other object-dtype
# diagnostics are dropped at stitch, as in the v15 patch)
KEEP_STR_COLS = ["wyckoff_phase_abc", "wyckoff_phase_dir", "wyckoff_context"]

logging.basicConfig(level=logging.WARNING)  # silence per-bar wyckoff INFO spam
log = logging.getLogger("v20patch")
log.setLevel(logging.INFO)


def load_ohlcv() -> pd.DataFrame:
    df = pd.read_parquet(STORE, columns=["open", "high", "low", "close", "volume"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def compute_rows(ohlcv: pd.DataFrame, start: int, end: int,
                 out_path: Path) -> pd.DataFrame:
    """Replay wyckoff features for ohlcv.iloc[start:end], resumable."""
    from bin.live.live_feature_computer import LiveFeatureComputer

    done = None
    if out_path.exists():
        done = pd.read_parquet(out_path)
        n_done = len(done)
        if n_done >= end - start:
            log.info("%s already complete (%d rows)", out_path.name, n_done)
            return done
        start = start + n_done
        log.info("resuming %s at offset %d", out_path.name, n_done)

    lfc = LiveFeatureComputer(buffer_size=WARMUP)
    warm_lo = max(0, start - WARMUP)
    if start > warm_lo:  # chunk 1 starts at row 0: no warmup exists (cold start)
        lfc.ingest_candles(ohlcv.iloc[warm_lo:start])

    rows, t0 = [], time.time()
    idx = ohlcv.index
    for i in range(start, end):
        ts = idx[i]
        bar = ohlcv.iloc[i]
        new_row = pd.DataFrame([{c: float(bar[c]) for c in
                                 ("open", "high", "low", "close", "volume")}],
                               index=[ts])
        lfc._buf = new_row.copy() if lfc._buf is None else pd.concat([lfc._buf, new_row])
        if len(lfc._buf) > lfc.buffer_size:
            lfc._buf = lfc._buf.iloc[-lfc.buffer_size:]

        feats: dict = {"__ts": ts}
        feats.update(lfc._wyckoff_features())
        rows.append(feats)

        n = i - start + 1
        if n % CHECKPOINT_EVERY == 0 or i == end - 1:
            chunk = pd.DataFrame(rows).set_index("__ts")
            merged = pd.concat([done, chunk]) if done is not None else chunk
            out_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(out_path)
            done, rows = merged, []
            rate = n / (time.time() - t0)
            eta_h = (end - 1 - i) / rate / 3600 if rate > 0 else float("inf")
            log.info("%s: %d/%d bars (%.1f bars/s, eta %.1fh)",
                     out_path.name, n, end - start, rate, eta_h)
    return done


def stitch(ohlcv_index: pd.Index) -> None:
    chunks = sorted(OUT_DIR.glob("chunk_*.parquet"))
    if not chunks:
        sys.exit("no chunks found")
    patch = pd.concat([pd.read_parquet(c) for c in chunks])
    patch = patch[~patch.index.duplicated(keep="last")].sort_index()
    # Event booleans come back as object dtype (bool + None on cold-start rows).
    # The v15 patch dropped ALL object columns, which silently discarded the
    # event bools and left the store carrying V14-era values — coerce them to
    # float 0/1 instead. Keep the directional strings; drop other diagnostics.
    for c in patch.columns:
        if patch[c].dtype == object and c not in KEEP_STR_COLS:
            coerced = pd.to_numeric(
                patch[c].map(lambda v: float(v) if isinstance(v, (bool, int, float))
                             and v == v else 0.0),
                errors="coerce").fillna(0.0)
            n_true = int((coerced != 0).sum())
            sample = patch[c].dropna()
            if len(sample) and all(isinstance(v, (bool, int, float))
                                   for v in sample.head(200)):
                patch[c] = coerced
                log.info("coerced object col %s to float (%d nonzero)", c, n_true)
    drop = [c for c in patch.columns
            if patch[c].dtype == object and c not in KEEP_STR_COLS]
    if drop:
        log.info("dropping non-numeric diagnostics: %s", drop)
    patch = patch.drop(columns=drop, errors="ignore")
    missing = ohlcv_index.difference(patch.index)
    if len(missing):
        sys.exit(f"patch incomplete: {len(missing)} store rows missing "
                 f"(first: {missing[0]})")
    patch = patch.reindex(ohlcv_index)

    store = pd.read_parquet(STORE)
    if store.index.tz is not None:
        store.index = store.index.tz_localize(None)
    replaced = [c for c in patch.columns if c in store.columns]
    added = [c for c in patch.columns if c not in store.columns]
    for c in patch.columns:
        store[c] = patch[c].values
    store.to_parquet(V20)
    print(f"V20 written: {V20}")
    print(f"  rows {len(store)}, replaced {len(replaced)} cols, added {len(added)} cols")
    print(f"  added: {sorted(added)}")
    key = ["wyckoff_spring_a", "wyckoff_spring_b", "wyckoff_ut", "wyckoff_utad",
           "wyckoff_sc", "wyckoff_bullish_score", "wyckoff_bearish_score"]
    print("  event counts / nonzero rates (patched):")
    for c in key:
        if c in store.columns:
            s = store[c].fillna(0)
            print(f"    {c:26s} nonzero={int((s != 0).sum()):6d} "
                  f"({float((s != 0).mean()):6.2%})")
    if "wyckoff_phase_dir" in store.columns:
        print("  phase_dir distribution:")
        print(store["wyckoff_phase_dir"].value_counts().to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", nargs=2, type=int, metavar=("I", "N"),
                    help="1-based worker index and total chunk count")
    ap.add_argument("--limit", type=int, help="smoke test: last N store bars")
    ap.add_argument("--stitch", action="store_true")
    args = ap.parse_args()

    ohlcv = load_ohlcv()
    if args.stitch:
        stitch(ohlcv.index)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(ohlcv)
    if args.limit:
        out = OUT_DIR / "smoke.parquet"
        out.unlink(missing_ok=True)
        df = compute_rows(ohlcv, n - args.limit, n, out)
        for c in ["wyckoff_spring_a", "wyckoff_spring_b", "wyckoff_ut",
                  "wyckoff_bullish_score", "tf1d_wyckoff_bullish_score"]:
            if c in df.columns:
                s = df[c].fillna(0)
                print(f"{c:28s} nonzero={float((s != 0).mean()):6.2%}  "
                      f"max={float(s.max()):.3f}")
        print(f"columns produced: {len(df.columns)}")
        return
    if not args.chunk:
        sys.exit("need --chunk I N, --limit, or --stitch")
    i, total = args.chunk
    per = (n + total - 1) // total
    start, end = (i - 1) * per, min(i * per, n)
    log.info("chunk %d/%d: rows %d..%d (%s -> %s)", i, total, start, end - 1,
             ohlcv.index[start], ohlcv.index[end - 1])
    compute_rows(ohlcv, start, end, OUT_DIR / f"chunk_{i:02d}.parquet")


if __name__ == "__main__":
    main()
