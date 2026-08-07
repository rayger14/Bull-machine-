"""
STRUCTURAL-RANGE OBJECT  (STUDY ONLY — sensor-fix prototype)

The keystone sensor fix flagged by wyckoff_audit addenda 31-33: springs, POC,
Bojan and premium/discount all MISFIRE because they key off ROLLING windows
(rolling(20).min, rolling-60d POC) instead of a DRAWN structural range. This
module replaces rolling extremes with a persistent, causal, DRAWN box that holds
for weeks and redraws only on a body-close break.

Anchors: the already-confirmed swing_low_50 / swing_high_50 pivots in the V22
store (PIT-correct, 10-bar pivot lag). We do NOT invent a new pivot detector.

------------------------------------------------------------------------------
PRE-REGISTERED CONSTRUCTION RULES (fixed before validation; NOT tunables to sweep)
------------------------------------------------------------------------------
1. ACTIVE RANGE state = (range_low, range_high, formed_at_bar).
   Initialize from the first pair of confirmed swing_low_50 / swing_high_50 that
   (a) BRACKET price  (range_low < close < range_high)  and
   (b) are >= MIN_WIDTH_ATR = 1.5 ATR apart  (noise filter).

2. PERSISTENCE (the whole point): range_low and range_high are FIXED LEVELS that
   hold bar-to-bar. They do NOT slide with the rolling swing columns. They change
   ONLY on the two events below (plus the optional floor-tighten in rule 4).

3. REDRAW / BREAK  (body close, not wick):
     close > range_high  -> emit break_up,   mark broken, enter 'forming'
     close < range_low   -> emit break_down, mark broken, enter 'forming'
   In 'forming' we re-establish a new active range from the next confirmed swing
   pivots that bracket price at >= MIN_WIDTH_ATR (same logic as init).
   A WICK beyond a boundary that CLOSES back inside is NOT a break — it's a SWEEP:
     low  < range_low  and close >= range_low  -> struct_sweep_low  = 1 (stay active)
     high > range_high and close <= range_high -> struct_sweep_high = 1 (stay active)
   Break is checked BEFORE sweep; they are mutually exclusive by close position.

4. OPTIONAL IN-RANGE TIGHTENING (pre-registered YES for range_low only):
   While active, if a NEW confirmed swing low prints ABOVE the current range_low
   and price is respecting the range (close strictly inside), RAISE range_low to
   that new swing low (higher-low = accumulation tightening). We NEVER lower
   range_low and NEVER lower range_high (the ceiling stays the target).
   Guard (design decision, documented): only tighten if the raised floor leaves
   width >= TIGHTEN_MIN_WIDTH_ATR = 0.75 ATR, so the box cannot collapse to noise.

5. CAUSALITY: only confirmed (already-lagged) pivots and CLOSED bars are used.
   No future data anywhere.

------------------------------------------------------------------------------
EMITTED COLUMNS (per 1H bar)
------------------------------------------------------------------------------
  struct_range_high      fixed drawn ceiling of the active range (NaN if forming)
  struct_range_low       fixed drawn floor  of the active range (NaN if forming)
  struct_range_age_bars  bars since formed_at (0 at formation; NaN if forming)
  struct_range_pos       (close-low)/(high-low): 0=at low(discount) 1=at high(premium)
  struct_range_state     {forming, active, broken_up, broken_down}
  struct_sweep_low       0/1  wick pierced drawn low, closed back inside
  struct_sweep_high      0/1  wick pierced drawn high, closed back inside
  struct_range_width_atr (high-low)/atr_14  at the current bar (NaN if forming)

DESIGN CONSTANTS (honest caveats — these are choices, not swept):
  MIN_WIDTH_ATR         = 1.5
  TIGHTEN_MIN_WIDTH_ATR = 0.75
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_WIDTH_ATR = 1.5
TIGHTEN_MIN_WIDTH_ATR = 0.75

# state labels
FORMING = "forming"
ACTIVE = "active"
BROKEN_UP = "broken_up"
BROKEN_DOWN = "broken_down"


def build_structural_range(
    df: pd.DataFrame,
    break_buffer_atr: float = 0.0,
    break_confirm_bars: int = 1,
) -> pd.DataFrame:
    """Causal forward pass. Returns a DataFrame (same index) with struct_* columns.

    Required input columns: open, high, low, close, swing_low_50, swing_high_50, atr_14.

    PRE-REGISTERED defaults reproduce the registered spec exactly:
      break_buffer_atr = 0.0   (any body-close beyond the boundary = a break)
      break_confirm_bars = 1   (a single close beyond is enough)

    The two params are DIAGNOSTIC KNOBS (not part of the pre-registered design) used
    only in validate step B to characterize the acceptance needed for weeks-long,
    human-like ranges — see the validation report. Non-default values require:
      close > range_high + buffer   for `break_confirm_bars` consecutive closes
    (symmetric on the low side) before a break is emitted; sub-threshold pokes that
    close back inside remain SWEEPS.
    """
    req = ["open", "high", "low", "close", "swing_low_50", "swing_high_50", "atr_14"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    swl = df["swing_low_50"].to_numpy(dtype=float)
    swh = df["swing_high_50"].to_numpy(dtype=float)
    atr = df["atr_14"].to_numpy(dtype=float)

    # detect where the confirmed swing pivots CHANGE value (a fresh confirmation)
    swl_changed = np.zeros(n, dtype=bool)
    swh_changed = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(swl[i]) and (np.isnan(swl[i - 1]) or swl[i] != swl[i - 1]):
            swl_changed[i] = True
        if not np.isnan(swh[i]) and (np.isnan(swh[i - 1]) or swh[i] != swh[i - 1]):
            swh_changed[i] = True

    # outputs
    o_high = np.full(n, np.nan)
    o_low = np.full(n, np.nan)
    o_age = np.full(n, np.nan)
    o_pos = np.full(n, np.nan)
    o_state = np.array([FORMING] * n, dtype=object)
    o_sweep_lo = np.zeros(n, dtype=np.int8)
    o_sweep_hi = np.zeros(n, dtype=np.int8)
    o_width = np.full(n, np.nan)

    # active-range state
    r_low = np.nan
    r_high = np.nan
    formed_at = -1
    active = False
    up_streak = 0    # consecutive closes beyond range_high+buffer (break acceptance)
    dn_streak = 0    # consecutive closes below range_low-buffer

    def can_form(i):
        """Try to establish a range from the currently-known confirmed pivots."""
        lo, hi = swl[i], swh[i]
        if np.isnan(lo) or np.isnan(hi) or np.isnan(atr[i]) or atr[i] <= 0:
            return None
        if not (lo < close[i] < hi):          # must bracket price
            return None
        if (hi - lo) / atr[i] < MIN_WIDTH_ATR:  # width filter
            return None
        return lo, hi

    for i in range(n):
        if not active:
            # FORMING (or pre-init): attempt to re-establish
            formed = can_form(i)
            if formed is not None:
                r_low, r_high = formed
                formed_at = i
                active = True
                up_streak = 0
                dn_streak = 0
                o_state[i] = ACTIVE
                o_high[i] = r_high
                o_low[i] = r_low
                o_age[i] = 0
                o_width[i] = (r_high - r_low) / atr[i]
                denom = r_high - r_low
                o_pos[i] = (close[i] - r_low) / denom if denom > 0 else np.nan
            else:
                o_state[i] = FORMING
            continue

        # ---- ACTIVE ----
        buf = break_buffer_atr * atr[i] if atr[i] > 0 else 0.0
        # 1) BREAK check (body close beyond boundary+buffer, with acceptance streak)
        up_streak = up_streak + 1 if close[i] > r_high + buf else 0
        dn_streak = dn_streak + 1 if close[i] < r_low - buf else 0
        if up_streak >= break_confirm_bars:
            o_state[i] = BROKEN_UP
            o_high[i] = r_high
            o_low[i] = r_low
            o_age[i] = i - formed_at
            o_width[i] = (r_high - r_low) / atr[i] if atr[i] > 0 else np.nan
            denom = r_high - r_low
            o_pos[i] = (close[i] - r_low) / denom if denom > 0 else np.nan
            active = False
            r_low = r_high = np.nan
            formed_at = -1
            continue
        if dn_streak >= break_confirm_bars:
            o_state[i] = BROKEN_DOWN
            o_high[i] = r_high
            o_low[i] = r_low
            o_age[i] = i - formed_at
            o_width[i] = (r_high - r_low) / atr[i] if atr[i] > 0 else np.nan
            denom = r_high - r_low
            o_pos[i] = (close[i] - r_low) / denom if denom > 0 else np.nan
            active = False
            r_low = r_high = np.nan
            formed_at = -1
            continue

        # 2) SWEEP check (wick beyond, close back inside w/o triggering a break)
        if low[i] < r_low and close[i] >= r_low - buf:
            o_sweep_lo[i] = 1
        if high[i] > r_high and close[i] <= r_high + buf:
            o_sweep_hi[i] = 1

        # 3) OPTIONAL floor-tighten (range_low only, higher-low accumulation)
        if swl_changed[i] and not np.isnan(atr[i]) and atr[i] > 0:
            new_lo = swl[i]
            respecting = (r_low < close[i] < r_high)
            higher_low = new_lo > r_low and new_lo < close[i]
            if respecting and higher_low:
                if (r_high - new_lo) / atr[i] >= TIGHTEN_MIN_WIDTH_ATR:
                    r_low = new_lo  # RAISE the floor (identity/formed_at preserved)

        # emit active-bar state
        o_state[i] = ACTIVE
        o_high[i] = r_high
        o_low[i] = r_low
        o_age[i] = i - formed_at
        o_width[i] = (r_high - r_low) / atr[i] if atr[i] > 0 else np.nan
        denom = r_high - r_low
        o_pos[i] = (close[i] - r_low) / denom if denom > 0 else np.nan

    out = pd.DataFrame(
        {
            "struct_range_high": o_high,
            "struct_range_low": o_low,
            "struct_range_age_bars": o_age,
            "struct_range_pos": o_pos,
            "struct_range_state": o_state,
            "struct_sweep_low": o_sweep_lo,
            "struct_sweep_high": o_sweep_hi,
            "struct_range_width_atr": o_width,
        },
        index=df.index,
    )
    return out


# ---------------------------------------------------------------------------
# STRUCTURAL POC — max-volume 0.25%-bin WITHIN the active [range_low, range_high],
# computed CAUSALLY over the range's own bars (cumulative from formed_at..i).
# Jumps on redraw; ~pins within a range. Returns a Series (NaN when forming).
# ---------------------------------------------------------------------------
def build_structural_poc(df: pd.DataFrame, sr: pd.DataFrame, bin_pct: float = 0.0025) -> pd.Series:
    close = df["close"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)
    rlow = sr["struct_range_low"].to_numpy(dtype=float)
    rhigh = sr["struct_range_high"].to_numpy(dtype=float)
    state = sr["struct_range_state"].to_numpy(dtype=object)
    n = len(df)
    poc = np.full(n, np.nan)

    # running accumulation of volume-at-price within the CURRENT active range
    # reset whenever a new range forms (formed_at boundary detected via age==0)
    age = sr["struct_range_age_bars"].to_numpy(dtype=float)
    bins = {}  # price-bin-index -> accumulated volume
    lo_ref = np.nan  # bin anchor = range_low at formation
    for i in range(n):
        st = state[i]
        if st == ACTIVE:
            if age[i] == 0 or np.isnan(lo_ref):
                bins = {}
                lo_ref = rlow[i]
            width = rhigh[i] - rlow[i]
            if width <= 0 or np.isnan(width):
                continue
            step = lo_ref * bin_pct if lo_ref > 0 else width * bin_pct
            if step <= 0:
                continue
            # bin THIS bar's close (only bars whose close sits inside the range)
            if rlow[i] <= close[i] <= rhigh[i]:
                b = int((close[i] - lo_ref) / step)
                bins[b] = bins.get(b, 0.0) + vol[i]
            if bins:
                best_b = max(bins, key=bins.get)
                poc[i] = lo_ref + (best_b + 0.5) * step
        else:
            lo_ref = np.nan
            bins = {}
    return pd.Series(poc, index=df.index, name="struct_poc")


if __name__ == "__main__":
    import sys
    store = "/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
    df = pd.read_parquet(store)
    sr = build_structural_range(df)
    print(sr["struct_range_state"].value_counts())
    active = sr["struct_range_state"] == ACTIVE
    print("active bars:", int(active.sum()), f"({active.mean()*100:.1f}%)")
    chg = (sr["struct_range_low"] != sr["struct_range_low"].shift(1)) & active
    print("struct_range_low change frac (of all bars):", f"{chg.mean()*100:.2f}%")
