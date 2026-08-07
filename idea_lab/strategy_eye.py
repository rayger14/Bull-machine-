"""
INTEGRATED ALL-SEEING-EYE / WYCKOFF-CAMPAIGN STRATEGY  (LONG-ONLY, pre-registered)

Wired as ONE strategy — the integrated method the production engine can never
express because fusion/dedup/margin-competition decompose it into competing
fragments. Spec (pre-registered, exactly as-run):

CONTEXT GATE (HTF permission):
    only look for entries when wyckoff_phase_dir in {C_accum, D_accum}
    (confirmed accumulation / early markup). Flat otherwise.

ENTRY (trigger inside the gate) + RETURN-TO-ZONE acceptance:
    a spring/LPS confirmation fires when, on a GATED bar,
        wyckoff_spring_a==1  OR  wyckoff_spring_b==1  OR  tf1h_pti_trap_type=='spring'
    Record H_conf = confirmation bar high, and the confirmation-region low.
    Enter on the FIRST later bar (within K=24 bars) that CLOSES back above H_conf
    (WI acceptance, not the raw event bar). Entry at that bar's close.
    ONE position per accumulation episode (episode = a contiguous run inside the
    accum gate; once a trade is taken, no re-entry until phase leaves the gate).

STOP (WI model-invalidation, tight, under the LPS — NOT the deep range low):
    stop = confirmation-region low  -  0.25 * ATR14(at entry)
    region low = min(low) over [confirmation_bar .. entry_bar]

EXIT — two variants sharing IDENTICAL entries+stops:
  (WI banked-and-derisked geometry)
    TP1 = 40% at nearest resistance = swing_high_50(at entry) if it is >= 0.5R
          above entry, else entry + 1R.
    On TP1 fill: move stop to entry + 0.1R (conservative derisk under recent HL).
    runner = 60% held to TT = measured move:
          projected = swing_high_50 + (swing_high_50 - swing_low_50)   [range projected up]
          TT = max(entry + 2R, projected)     (falls back to 2R when swings absent)
    max_hold = 168h.
  (NAIVE ladder baseline)
    same entry + same stop, plain 1R/2R/3R equal-thirds scale-out ladder,
    FIXED stop (no move), max_hold = 168h. Isolates whether WI exit geometry
    matters here vs a simple ladder.

Feature provenance: wyckoff_phase_dir is the M2-shadow feature (validated,
addendum 2/4). Store: BTC_1H_FEATURES_V22_CTX.parquet (causal).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtester import EntryPlan

K_RTZ = 24                 # return-to-zone acceptance window (bars)
GATE = {"C_accum", "D_accum"}
MAX_HOLD = 168


def _precompute(df: pd.DataFrame):
    """Vectorized causal arrays + the confirmation/RTZ episode plan.

    Returns a dict: idx-> pending entry parameters, and a 'gate'/'conf' arrays.
    We resolve, for every bar, whether it is a valid RTZ ENTRY bar and, if so,
    the confirmation-region low and H_conf that produced it. Episode-dedup and
    one-position-at-a-time are enforced at run time by the backtester (it resumes
    scanning after each exit) PLUS an explicit episode flag here.
    """
    phase = df["wyckoff_phase_dir"].to_numpy(dtype=object)
    gate = np.array([p in GATE for p in phase])
    sa = np.nan_to_num(df["wyckoff_spring_a"].to_numpy(float)) == 1
    sb = np.nan_to_num(df["wyckoff_spring_b"].to_numpy(float)) == 1
    pti = df["tf1h_pti_trap_type"].to_numpy(dtype=object)
    pti_spring = np.array([p == "spring" for p in pti])
    conf = gate & (sa | sb | pti_spring)         # confirmation must be inside gate
    return {"gate": gate, "conf": conf}


class EyeStrategy:
    """Callable entry_fn wrapper carrying the two exit variants + episode state.

    variant: 'wi'    -> WI banked-and-derisked geometry
             'naive' -> plain 1R/2R/3R ladder, fixed stop
    """

    def __init__(self, df: pd.DataFrame, variant: str = "wi"):
        self.variant = variant
        pre = _precompute(df)
        self.gate = pre["gate"]
        self.conf = pre["conf"]
        self.highs = df["high"].to_numpy(float)
        self.lows = df["low"].to_numpy(float)
        self.closes = df["close"].to_numpy(float)
        self.atr = df["atr_14"].to_numpy(float)
        self.sh50 = df["swing_high_50"].to_numpy(float)
        self.sl50 = df["swing_low_50"].to_numpy(float)
        self.n = len(df)
        # episode-dedup: track whether the current accum episode already traded.
        # An episode ends when the gate turns False; a new one starts on re-entry.
        self._episode_traded = False
        self._last_gate = False
        # remember which confirmation bars have been "consumed" so a single
        # confirmation cannot spawn two entries.
        self._consumed_until = -1
        self.entries_log = []   # for the spring-overlap / orthogonality check

    def _resolve_entry(self, i: int):
        """Is bar i a valid RTZ ENTRY? If so return (region_low, H_conf, conf_i)."""
        # bar i is an entry if some confirmation bar c in [i-K, i-1] (gated, and
        # not yet consumed) had H_conf and close[i] > H_conf, with i being the
        # FIRST such acceptance after c.
        lo_j = max(0, i - K_RTZ)
        for c in range(i - 1, lo_j - 1, -1):
            if not self.conf[c]:
                continue
            if c <= self._consumed_until:
                continue
            H_conf = self.highs[c]
            # first acceptance: no bar between c+1..i-1 already closed above H_conf
            already = False
            for m in range(c + 1, i):
                if self.closes[m] > H_conf:
                    already = True
                    break
            if already:
                continue
            if self.closes[i] > H_conf:
                region_low = np.nanmin(self.lows[c:i + 1])
                return region_low, H_conf, c
        return None

    def __call__(self, df: pd.DataFrame, i: int):
        # maintain episode state on the gate transition
        g = self.gate[i]
        if not g:
            self._episode_traded = False      # gate closed -> episode over
        self._last_gate = g

        if not g or self._episode_traded:
            return None
        res = self._resolve_entry(i)
        if res is None:
            return None
        region_low, H_conf, conf_i = res

        entry_raw = self.closes[i]
        atr = self.atr[i]
        if not np.isfinite(atr) or not np.isfinite(region_low):
            return None
        stop = region_low - 0.25 * atr
        # entry fill is slightly above close (slippage) but for plan geometry we
        # anchor R on the RAW entry close vs stop (the backtester re-slips fills;
        # geometry defined on model prices is standard).
        R = entry_raw - stop
        if R <= 0:
            return None

        self._episode_traded = True
        self._consumed_until = conf_i          # consume this confirmation

        sh = self.sh50[i]
        sl = self.sl50[i]

        # record for overlap/orthogonality analysis
        pti_here = df["tf1h_pti_trap_type"].iloc[i]
        self.entries_log.append({
            "entry_time": df.index[i], "conf_time": df.index[conf_i],
            "phase": df["wyckoff_phase_dir"].iloc[i],
            "pti_trap_at_entry": pti_here,
            "conf_spring_a": bool(df["wyckoff_spring_a"].iloc[conf_i] == 1),
            "conf_spring_b": bool(df["wyckoff_spring_b"].iloc[conf_i] == 1),
            "conf_pti_spring": bool(df["tf1h_pti_trap_type"].iloc[conf_i] == "spring"),
            "R_price": R,
        })

        if self.variant == "naive":
            targets = [(entry_raw + 1 * R, 1 / 3.0),
                       (entry_raw + 2 * R, 1 / 3.0),
                       (entry_raw + 3 * R, 1 / 3.0)]
            return EntryPlan(direction="long", stop=stop, targets=targets,
                             move_stop_to_after_first_tp=None, runner_target=None,
                             max_hold_bars=MAX_HOLD,
                             meta={"variant": "naive", "conf_i": int(conf_i)})

        # ---- WI banked-and-derisked geometry ----
        # TP1 = nearest resistance if it clears 0.5R above entry, else entry+1R
        if np.isfinite(sh) and sh >= entry_raw + 0.5 * R:
            tp1 = sh
        else:
            tp1 = entry_raw + 1 * R
        # runner TT = measured move (range projected up), floored at entry+2R
        if np.isfinite(sh) and np.isfinite(sl) and sh > sl:
            projected = sh + (sh - sl)
        else:
            projected = -np.inf
        tt = max(entry_raw + 2 * R, projected)
        move_stop = entry_raw + 0.1 * R
        return EntryPlan(direction="long", stop=stop,
                         targets=[(tp1, 0.40)],
                         move_stop_to_after_first_tp=move_stop,
                         runner_target=tt,
                         max_hold_bars=MAX_HOLD,
                         meta={"variant": "wi", "conf_i": int(conf_i),
                               "tp1": tp1, "tt": tt})
