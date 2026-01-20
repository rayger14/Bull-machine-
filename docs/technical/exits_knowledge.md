# Exit System – Knowledge Pack for Claude

## 1) Ground Rules (Non-Negotiables)

- **Exits are signals, not side effects.** They must emit structured `ExitSignal` objects with: `type`, `confidence`, `urgency`, `reason`, `params_used`, and `ctx` (bar index, tf, symbol).
- **Precedence:** `CHOCH_AGAINST` > `MTF_DESYNC` > `MOMENTUM_FADE` > `TIME_STOP`. On ties, pick highest confidence; if equal, earliest `first_seen_idx`.
- **No silent fallbacks.** If a required param is missing, raise and log. Never "guess".
- **Non-repainting:** All swing/pivot logic excludes current forming bar.
- **Partial exits** change position size only; they do not reset `opened_at_*` aging fields.

## 2) Data Requirements Per Exit

Every evaluator receives:
- `bar_df` (current TF)
- `htf_df`/`ltf_df` (if needed)
- `pos_snapshot` (immutable: `opened_at_ts`, `opened_at_idx`, `bars_held`, `entry_price`, `avg_price`, `size`, `side`)
- `sync_report` (for MTF)
- `feature_cache` (ATR, MA, RSI, body/ATR)

**Aging is bar-count based** (`bars_held`), not time deltas.

## 3) Exit Types (Definitions & Parameters)

### A) CHoCH-Against
**Idea:** Structure flips against the current position with confirmation.

**Trigger (long):** Close < last confirmed swing low (built with `swing_lookback`), confirmed by `bars_confirm` bars. Vice-versa for shorts.

**Params:**
- `bars_confirm` (int): 1-3 (use exactly this key)
- `swing_lookback` (int): 3-7
- `min_break_strength` (float): 0.02–0.10 (break distance vs ATR)

**Output:** `EXIT_FULL` with `confidence = clamp01(break_strength/0.15)`, `urgency = confidence`.

### B) Momentum Fade
**Idea:** Impulse is decaying; protect profits / reduce risk.

**Signal metric:** `fade_score = max(body_to_ATR_drop, RSI_divergence_score, velocity_drop)`

**Params (required):**
- `drop_pct` (float 0–1): e.g., 0.15, 0.20, 0.25 (use this key; don't shadow with `rsi_divergence_threshold`)
- `lookback` (int): 5–10 bars
- `partial_size` (float): 0.25–0.50 (fraction to exit)

**Threshold:** trigger if `fade_score >= drop_pct`.

**Output:** `EXIT_PARTIAL` with `size_ratio=partial_size`, `confidence = fade_score`, `urgency = confidence`.

### C) Time Stop
**Idea:** If position overstays expected horizon, clean up.

**Params (required; per TF):**
- `max_bars_1h`, `max_bars_4h`, `max_bars_1d` (ints). No `bars_max` alias.
- Optional: `grace_bars` (int) for hysteresis (e.g., 2).

**Trigger:** `bars_held >= max_bars_for(position.tf)`.

**Output:** `EXIT_FULL`, `confidence = min(1.0, bars_held/max_bars)`, `urgency = confidence`.

### D) MTF Desync (optional but recommended)
**Idea:** Hard veto if HTF bias flips against position with high confidence.

**Params:** `min_confidence` (0.6–0.8).

**Trigger:** `sync_report.decision == "veto"` and suggested bias contradicts position and `sync_report.confidence >= min_confidence`.

**Output:** `EXIT_FULL`, `confidence = sync_report.confidence`, `urgency = confidence`.

## 4) Config Schema (Single Source of Truth)

```json
{
  "exits": {
    "enabled": true,
    "enabled_exits": ["choch_against", "momentum_fade", "time_stop", "mtf_desync"],
    "choch_against": {
      "bars_confirm": 2,
      "swing_lookback": 5,
      "min_break_strength": 0.05
    },
    "momentum_fade": {
      "drop_pct": 0.20,
      "lookback": 7,
      "partial_size": 0.33
    },
    "time_stop": {
      "max_bars_1h": 24,
      "max_bars_4h": 42,
      "max_bars_1d": 10,
      "grace_bars": 2
    },
    "mtf_desync": {
      "min_confidence": 0.70
    },
    "precedence": ["choch_against", "mtf_desync", "momentum_fade", "time_stop"],
    "emit_exit_debug": true
  }
}
```

## 5) What to Log (Telemetry for Calibration)

- **On init:** `EXIT_EVAL_INIT exits=[...] effective_cfg={...}`
- **On each fire:** `EXIT_FIRE type=momentum_fade side=long conf=0.74 urg=0.74 size=0.33 bars_held=11 params={"drop_pct":0.2,"lookback":7}`
- **On each scan (debug level):** `EXIT_SCAN counts={"choch":X,"mom":Y,"time":Z,"none":N}`
- **On application:** broker emits `EXIT_APPLIED` with realized R, PnL, duration.

## 6) Unit Tests Claude Should Include

- `test_choch_uses_bars_confirm_parameter` – fails if `confirmation_bars` is referenced.
- `test_momentum_uses_drop_pct_parameter` – ensure threshold uses `drop_pct`.
- `test_time_stop_uses_max_bars_per_timeframe` – no `bars_max` fallback.
- `test_scale_in_preserves_opened_at` – aging fields immutable.
- `test_exit_precedence` – CHoCH beats Momentum when both fire.
- `test_non_repainting_pivots` – current bar excluded from swing formation.
- `test_partial_exit_does_not_reset_aging` – `bars_held` continues.

## 7) Edge Cases to Handle

- **Low-ATR regime** (avoid div by zero; clamp).
- **Gaps or missing bars** (aging still advances by bar count, not timestamp).
- **Multiple exits in one bar:** apply precedence; do at most one action.
- **Flip logic:** Only allowed if both exit_full + immediate opposite entry are signaled by strategy rules (avoid accidental churn).

## 8) Calibration Guidance (What Claude Should Bake In)

- Provide a tiny sweep harness for exits (3×3×3) that varies `bars_confirm`, `drop_pct`, and `max_bars_1h`.
- Persist `exit_counts.json` and `exit_examples.jsonl` (first 50 fires with `params_used`) per run.
- Add quality gates OFF/ON toggles so exits can be evaluated independently from entry selectivity.

## 9) Broker Contract (So Exits Actually "Do" Something)

- **EXIT_FULL:** close remaining size at market (slippage model) and mark trade complete.
- **EXIT_PARTIAL:** reduce size by `size_ratio`, keep `opened_at_*` unchanged.
- **Stop tightening** (if you later add it): update stop to `max(current_stop, new_stop)` for longs (`min` for shorts).

---

*This knowledge pack ensures exits are implemented deterministically, testably, and MTF-aware without re-introducing bugs like parameter shadowing, aging resets, or silent fallbacks.*