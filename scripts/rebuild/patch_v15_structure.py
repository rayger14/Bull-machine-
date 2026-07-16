#!/usr/bin/env python3
"""V15 structure patch — recompute SMC/BOMS/Wyckoff columns with repaired code.

Why: the V14 store was built while (a) the BOS emitter read a nonexistent
attribute (flags 0 on all 73,829 rows), (b) CHoCH was clobbered to 0 by
_extra_archetype_features, and (c) the graded Wyckoff score columns
(wyckoff_bullish_score, tf1d_wyckoff_bullish_score, ...) didn't exist yet —
so fusion's wyckoff domain backtests on zero-fallbacks while live computes
real values. All three families derive from a bounded 1,000-bar OHLCV buffer,
so they can be recomputed standalone and spliced into V14 → V15. No macro,
derivatives, or regime state is touched.

OHLCV source: the V14L store's own open/high/low/close/volume columns
(same raw klines the original replay consumed → guaranteed index alignment).
First 1,000 bars (Jan–Feb 2018) have a cold-start buffer, same as V14 itself.

Usage:
  python3 scripts/rebuild/patch_v15_structure.py --chunk 1 8      # worker
  python3 scripts/rebuild/patch_v15_structure.py --limit 300      # smoke
  python3 scripts/rebuild/patch_v15_structure.py --stitch         # splice V15
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

STORE = REPO / "data/features_mtf/BTC_1H_FEATURES_V14L_LEVELS.parquet"
OUT_DIR = REPO / "results/rebuild/v15_structure"
V15 = REPO / "data/features_mtf/BTC_1H_FEATURES_V15_STRUCTURE.parquet"
WARMUP = 1000
CHECKPOINT_EVERY = 2000

logging.basicConfig(level=logging.WARNING)  # silence per-bar wyckoff INFO spam
log = logging.getLogger("v15patch")
log.setLevel(logging.INFO)


def load_ohlcv() -> pd.DataFrame:
    df = pd.read_parquet(STORE, columns=["open", "high", "low", "close", "volume"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def compute_rows(ohlcv: pd.DataFrame, start: int, end: int,
                 out_path: Path) -> pd.DataFrame:
    """Replay structure features for ohlcv.iloc[start:end], resumable."""
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
        feats.update(lfc._smc_features())
        feats.update(lfc._boms_liquidity_features())
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
    # drop non-numeric diagnostics the store shouldn't carry
    patch = patch.drop(columns=[c for c in patch.columns
                                if patch[c].dtype == object], errors="ignore")
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
    store.to_parquet(V15)
    print(f"V15 written: {V15}")
    print(f"  rows {len(store)}, replaced {len(replaced)} cols, added {len(added)} cols")
    print(f"  added: {sorted(added)}")
    key = ["tf1h_bos_bullish", "tf1h_bos_bearish", "tf1h_choch_detected",
           "tf4h_bos_bullish", "tf4h_choch_flag", "boms_strength",
           "tf1d_wyckoff_score", "wyckoff_bullish_score"]
    print("  nonzero rates (patched):")
    for c in key:
        if c in store.columns:
            print(f"    {c:26s} {float((store[c].fillna(0) != 0).mean()):6.2%}")


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
        for c in ["tf1h_bos_bullish", "tf1h_bos_bearish", "tf1h_choch_detected",
                  "tf4h_bos_bullish", "tf4h_choch_flag", "boms_strength",
                  "tf1d_wyckoff_score", "wyckoff_bullish_score",
                  "tf1d_wyckoff_bullish_score"]:
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
