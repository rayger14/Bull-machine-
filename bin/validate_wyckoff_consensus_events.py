#!/usr/bin/env python3
"""
Validate 14 additional BTC Wyckoff consensus events against feature store detections.

For each known historical Wyckoff event, searches within a +/- 48 bar (hour) window
and 5% price range for matching detections in the feature store.
"""

import pandas as pd
import numpy as np

# ─── Load feature store ───────────────────────────────────────────────────────
FEATURE_STORE = "data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet"
df = pd.read_parquet(FEATURE_STORE)
print(f"Feature store loaded: {df.shape[0]:,} bars x {df.shape[1]} cols")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print()

# ─── Define event-to-column mapping ──────────────────────────────────────────
# Each event type maps to:
#   - bool_cols: list of boolean detection columns to check
#   - conf_cols: list of corresponding confidence columns
EVENT_COL_MAP = {
    "AR": {
        "bool_cols": ["wyckoff_ar"],
        "conf_cols": ["wyckoff_ar_confidence"],
    },
    "ST": {
        "bool_cols": ["wyckoff_st"],
        "conf_cols": ["wyckoff_st_confidence"],
    },
    "PSY": {
        # PSY (Preliminary Supply) is closest to BC (Buying Climax) in our detector
        "bool_cols": ["wyckoff_bc"],
        "conf_cols": ["wyckoff_bc_confidence"],
    },
    "UTAD": {
        "bool_cols": ["wyckoff_utad", "wyckoff_ut"],
        "conf_cols": ["wyckoff_utad_confidence", "wyckoff_ut_confidence"],
    },
    "LPSY": {
        "bool_cols": ["wyckoff_lpsy"],
        "conf_cols": ["wyckoff_lpsy_confidence"],
    },
    "SOS": {
        "bool_cols": ["wyckoff_sos"],
        "conf_cols": ["wyckoff_sos_confidence"],
    },
    "LPS": {
        "bool_cols": ["wyckoff_lps"],
        "conf_cols": ["wyckoff_lps_confidence"],
    },
    "Spring": {
        "bool_cols": ["wyckoff_spring_a", "wyckoff_spring_b"],
        "conf_cols": ["wyckoff_spring_a_confidence", "wyckoff_spring_b_confidence"],
    },
    "BC": {
        "bool_cols": ["wyckoff_bc"],
        "conf_cols": ["wyckoff_bc_confidence"],
    },
}

# ─── Define the 14 consensus events ─────────────────────────────────────────
# Each event: (num, event_type, center_date, approx_price, cycle_context, notes)
EVENTS = [
    (1,  "AR",     "2019-02-08",  4200,  "2018-2019 Accumulation", ""),
    (2,  "ST",     "2019-02-06",  3400,  "2018-2019 Accumulation", ""),
    (3,  "PSY",    "2021-02-21",  58000, "2021 Distribution", ""),
    (4,  "ST",     "2021-09-07",  52000, "2021 Distribution Phase B", ""),
    (5,  "UTAD",   "2021-11-10",  69000, "2021 Distribution Phase C", "overlaps original #9 BC"),
    (6,  "LPSY",   "2022-01-15",  47000, "2021-2022 Distribution Phase D", ""),
    (7,  "AR",     "2022-07-20",  24000, "2022 Accumulation", ""),
    (8,  "ST",     "2022-09-20",  18500, "2022 Accumulation Phase B", ""),
    (9,  "SOS",    "2023-03-15",  28000, "2022-2023 Accumulation Phase D", "Creek Jump"),
    (10, "LPS",    "2023-09-15",  25000, "2022-2023 Accumulation Phase D", ""),
    (11, "Spring", "2024-08-05",  49000, "2024 Reaccumulation", ""),
    (12, "BC",     "2024-03-14",  73660, "2024 Distribution/Reacc.", ""),
    (13, "AR",     "2024-04-15",  60000, "2024 Distribution/Reacc.", ""),
    (14, "LPS",    "2024-10-15",  59000, "2024 Reaccumulation", ""),
]

WINDOW_BARS = 48  # +/- 48 hours
PRICE_TOLERANCE = 0.05  # 5%

# ─── Validation loop ─────────────────────────────────────────────────────────
results = []

for num, etype, date_str, approx_price, context, notes in EVENTS:
    center = pd.Timestamp(date_str, tz="UTC")
    start = center - pd.Timedelta(hours=WINDOW_BARS)
    end = center + pd.Timedelta(hours=WINDOW_BARS)

    # Slice window
    window = df.loc[start:end].copy()

    if len(window) == 0:
        results.append({
            "num": num,
            "event": etype,
            "date": date_str,
            "price": approx_price,
            "context": context,
            "result": "NO DATA",
            "best_conf": 0.0,
            "offset_hrs": None,
            "detected_col": "",
            "detected_price": None,
            "notes": notes,
        })
        continue

    # Price filter: only bars within 5% of expected price
    price_low = approx_price * (1 - PRICE_TOLERANCE)
    price_high = approx_price * (1 + PRICE_TOLERANCE)
    price_mask = (window["close"] >= price_low) & (window["close"] <= price_high)

    mapping = EVENT_COL_MAP[etype]
    bool_cols = mapping["bool_cols"]
    conf_cols = mapping["conf_cols"]

    best_conf = 0.0
    best_offset = None
    best_col = ""
    best_price = None
    hit = False

    for bcol, ccol in zip(bool_cols, conf_cols):
        # Check boolean detection within window AND price range
        detected = window[bcol] & price_mask
        if detected.any():
            hit = True
            # Find bar with highest confidence among detected
            candidates = window.loc[detected, ccol]
            if len(candidates) > 0:
                best_idx = candidates.idxmax()
                conf_val = candidates.loc[best_idx]
                if conf_val > best_conf:
                    best_conf = conf_val
                    best_offset = int((best_idx - center).total_seconds() / 3600)
                    best_col = bcol
                    best_price = window.loc[best_idx, "close"]

    # If no hit with price filter, also check without price filter (report separately)
    hit_no_price = False
    best_conf_np = 0.0
    best_offset_np = None
    best_col_np = ""
    best_price_np = None

    if not hit:
        for bcol, ccol in zip(bool_cols, conf_cols):
            detected = window[bcol]
            if detected.any():
                hit_no_price = True
                candidates = window.loc[detected, ccol]
                if len(candidates) > 0:
                    best_idx = candidates.idxmax()
                    conf_val = candidates.loc[best_idx]
                    if conf_val > best_conf_np:
                        best_conf_np = conf_val
                        best_offset_np = int((best_idx - center).total_seconds() / 3600)
                        best_col_np = bcol
                        best_price_np = window.loc[best_idx, "close"]

    if hit:
        result_str = "HIT"
    elif hit_no_price:
        result_str = "PRICE_MISS"
    else:
        result_str = "MISS"

    results.append({
        "num": num,
        "event": etype,
        "date": date_str,
        "price": approx_price,
        "context": context,
        "result": result_str,
        "best_conf": best_conf if hit else best_conf_np,
        "offset_hrs": best_offset if hit else best_offset_np,
        "detected_col": best_col if hit else best_col_np,
        "detected_price": best_price if hit else best_price_np,
        "notes": notes,
    })

# ─── Summary table ────────────────────────────────────────────────────────────
print("=" * 130)
print("WYCKOFF CONSENSUS EVENT VALIDATION (14 events, +/-48h window, 5% price tolerance)")
print("=" * 130)
print(f"{'#':<3} {'Event':<8} {'Date':<12} {'Exp Price':>10} {'Result':<12} {'Conf':>6} {'Offset':>8} {'Det Price':>10} {'Column':<25} {'Context'}")
print("-" * 130)

hits = 0
total = 0
hits_ex5 = 0
total_ex5 = 0

for r in results:
    total += 1
    is_hit = r["result"] == "HIT"
    if is_hit:
        hits += 1

    if r["num"] != 5:
        total_ex5 += 1
        if is_hit:
            hits_ex5 += 1

    offset_str = f"{r['offset_hrs']:+d}h" if r["offset_hrs"] is not None else "N/A"
    conf_str = f"{r['best_conf']:.3f}" if r["best_conf"] > 0 else "0.000"
    det_price_str = f"${r['detected_price']:,.0f}" if r["detected_price"] is not None else "N/A"
    exp_price_str = f"${r['price']:,}"

    marker = ""
    if r["result"] == "HIT":
        marker = "[OK]"
    elif r["result"] == "PRICE_MISS":
        marker = "[PRICE]"
    else:
        marker = "[X]"

    ctx = r["context"]
    if r["notes"]:
        ctx += f" ({r['notes']})"

    print(f"{r['num']:<3} {r['event']:<8} {r['date']:<12} {exp_price_str:>10} {marker + ' ' + r['result']:<12} {conf_str:>6} {offset_str:>8} {det_price_str:>10} {r['detected_col']:<25} {ctx}")

print("-" * 130)
print()
print(f"Overall hit rate:          {hits}/{total} ({hits/total*100:.1f}%)")
print(f"Hit rate (excl. #5 dup):   {hits_ex5}/{total_ex5} ({hits_ex5/total_ex5*100:.1f}%)")
print()

# ─── Detailed breakdown for misses ──────────────────────────────────────────
misses = [r for r in results if r["result"] != "HIT"]
if misses:
    print("=" * 130)
    print("DETAILED MISS ANALYSIS")
    print("=" * 130)
    for r in misses:
        num = r["num"]
        etype = r["event"]
        center = pd.Timestamp(r["date"], tz="UTC")
        start = center - pd.Timedelta(hours=WINDOW_BARS)
        end = center + pd.Timedelta(hours=WINDOW_BARS)
        window = df.loc[start:end]

        mapping = EVENT_COL_MAP[etype]
        bool_cols = mapping["bool_cols"]
        conf_cols = mapping["conf_cols"]

        print(f"\n--- Event #{num}: {etype} @ {r['date']} (~${r['price']:,}) ---")
        print(f"    Window: {start} to {end} ({len(window)} bars)")
        if len(window) > 0:
            print(f"    Price range in window: ${window['close'].min():,.0f} - ${window['close'].max():,.0f}")
            print(f"    Expected price range:  ${r['price'] * 0.95:,.0f} - ${r['price'] * 1.05:,.0f}")

        for bcol, ccol in zip(bool_cols, conf_cols):
            det_count = window[bcol].sum()
            print(f"    {bcol}: {det_count} detections in window")
            if det_count > 0:
                det_rows = window[window[bcol]]
                for idx, row in det_rows.iterrows():
                    print(f"      -> {idx} | price=${row['close']:,.0f} | conf={row[ccol]:.3f}")

        if r["result"] == "PRICE_MISS":
            print(f"    STATUS: Detected but price outside 5% tolerance")
        else:
            # Check wider window for any detection
            wide_start = center - pd.Timedelta(hours=168)  # 1 week
            wide_end = center + pd.Timedelta(hours=168)
            wide_window = df.loc[wide_start:wide_end]
            for bcol, ccol in zip(bool_cols, conf_cols):
                det_count = wide_window[bcol].sum()
                if det_count > 0:
                    print(f"    WIDER (+/-7d): {bcol} has {det_count} detections")
                    det_rows = wide_window[wide_window[bcol]].nlargest(3, ccol)
                    for idx, row in det_rows.iterrows():
                        offset = int((idx - center).total_seconds() / 3600)
                        print(f"      -> {idx} | price=${row['close']:,.0f} | conf={row[ccol]:.3f} | offset={offset:+d}h")
                else:
                    print(f"    WIDER (+/-7d): {bcol} has 0 detections")

print()
print("=" * 130)
print("VALIDATION COMPLETE")
print("=" * 130)
