# Feature Store 4H Wyckoff Rebuild

**Date**: 2026-05-01
**Branch**: `feat/feature-store-4h-wyckoff-rebuild`
**Status**: Script + tests delivered. **Production parquet rebuild was NOT executed on this machine** because `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet` is gitignored (`*.parquet` in `.gitignore`) and is not present in any local checkout, worktree, or accessible volume on this machine. The script must be run on the host that has the parquet (most likely the deploy server `165.1.79.19` or the user's primary workstation).

## Problem

`tf4h_wyckoff_bullish_score` and `tf4h_wyckoff_bearish_score` (and `tf4h_wyckoff_phase_score`) in the parquet are broken (frozen / 0 / 0.5). The live engine recomputes them correctly via `engine/wyckoff/`, so the discrepancy means the 1.25× sizing boost from commit `5059285` and the `distribution_at_resistance` gate from commit `fb02a66` have **no effect in backtest** but **do** fire live.

## Approach

Reuse the existing `detect_all_wyckoff_events` pipeline in `engine/wyckoff/events.py` (no edits to engine code) and replicate, in batch mode, what the live engine does at every 1H bar:

1. Resample 1H OHLCV → 4H using `closed='left', label='left'` (pandas default — matches `bin/live/live_feature_computer.py::_resample_to_tf`).
2. Run `detect_all_wyckoff_events` on the full 4H series with `_CFG_4H` taken verbatim from `bin/live/live_feature_computer.py:1602`.
3. Per-4H-bar rolling-max over the 7 accumulation event confidences (`sc, ar, st, spring_a, spring_b, sos, lps`) and the 6 distribution event confidences (`bc, as, sow, ut, utad, lpsy`), using a 250-bar lookback (matches the live engine's `lookback=len(buf_4h_copy)` when the 1000-1H buffer is full ≈ 250 4H bars ≈ 41 days).
4. Forward-fill onto the 1H index using a **close-time mapping**: a 4H bar labeled `T` covers `[T, T+4h)` and only becomes readable at `T+4h`. So the 1H bar at time `t` reads from the most recent 4H bar with `T+4h <= t`. Implementation: shift the 4H label by `+4h`, reindex on the union with the 1H index, sort, ffill, then reindex on the 1H index. Initial 1H bars before the first 4H close get 0.
5. **Update only** the three `tf4h_wyckoff_*_score` columns in the parquet, leaving all other ~298 columns bit-for-bit identical (verified by SHA1 of every non-update column before and after, plus a full reload).
6. Atomic write via tempfile + `os.replace`.

### Why a rolling 250-bar max (and not a per-bar score)?

Inspection of `bin/live/live_feature_computer.py:1694` shows the live engine computes 4H scores via `create_wyckoff_context(buf_4h_copy, lookback=len(buf_4h_copy))`. `buf_4h_copy` is bounded by the buffer cap (`buffer_size=1000` 1H bars → ~250 4H bars). So the live score at any given 1H bar is the maximum event confidence over the trailing ~250 4H bars (~41 days). Anything shorter would diverge from live; anything longer would exceed the live buffer's actual lookback.

### Look-ahead bias safeguards

This is the single most critical correctness property: a 1H bar at `t` must NEVER see a 4H bar that closes after `t`. We enforce this with three layers:

1. **Construction**: shift the 4H label to its close time before the forward-fill, so any 4H bar still open at hour `t` is mathematically excluded from the available scores at `t`.
2. **Unit tests** (`tests/test_rebuild_4h_wyckoff_features.py`):
   - `test_no_look_ahead_close_time_mapping`: hand-built case proves the 1H value at hour `t` is never in advance of the 4H close.
   - `test_rolling_lookback_matches_truncated_pipeline`: builds the full pipeline once, then for sample hours rebuilds the entire pipeline from a TRUNCATED 1H input ending at `t` and confirms the score at `t` is identical. Catches any subtle leakage in the rolling max.
3. **Runtime check**: `--look-ahead-samples N` (default 8) randomly picks N 1H hours after the burn-in, runs the same truncated-rebuild check inside the script, and aborts with non-zero exit code if any sample disagrees. The script does NOT write the parquet on look-ahead mismatch.

## Validation on Synthetic Data (this machine)

Production parquet is unavailable locally so the script was validated on a synthetic 6,000-bar 1H feature store with 80 dummy columns + frozen-0.5 update columns:

| Property | Result |
|----------|--------|
| Shape preserved | OK (6000, 88) → (6000, 88) |
| Non-update columns bit-for-bit | OK (85/85 SHA1 match before+after, plus reload check) |
| `tf4h_wyckoff_bullish_score` std | 0.288 (was 0.000) |
| `tf4h_wyckoff_bearish_score` std | 0.391 (was 0.000) |
| `tf4h_wyckoff_phase_score` std | 0.382 (was 0.000) |
| `% bars non-zero` (bullish/bearish/phase) | 56% / 55% / 74% |
| Look-ahead bias (5 samples) | PASS (0 mismatches) |
| Top bearish bars cluster near synthetic top? | YES — 5 of top-5 within hours of `index = N//3` (the planted distribution top) |
| Atomic write + reload | OK |
| Dry-run preserves mtime | OK |

Tests:
```
$ python3 -m pytest tests/test_rebuild_4h_wyckoff_features.py -v
======================= 5 passed, 12 warnings in 10.41s ========================
```

## Pending: Run on the production parquet

The actual parquet update must run on the host that has `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet` (61,306 × ~301). Recommended invocation:

```bash
# 1) Sanity preview (no write)
python3 scripts/rebuild_4h_wyckoff_features.py --dry-run

# 2) Actual rebuild
python3 scripts/rebuild_4h_wyckoff_features.py

# 3) Backtest delta (optional but recommended)
python3 bin/backtest_v11_standalone.py \
    --start-date 2020-01-01 \
    --commission-rate 0.0002 \
    --slippage-bps 3
```

The script logs:
- Pre/post distribution stats (mean, std, % non-zero, n_unique) for each updated column.
- Top-5 highest `tf4h_wyckoff_bearish_score` and `tf4h_wyckoff_bullish_score` bars (sanity check — bearish should cluster near Apr 2021 ($64k) / Nov 2021 ($69k); bullish near Mar 2020 ($3.8k) / Jun 2022 ($17.5k) / Nov 2022 ($15.5k)).
- Look-ahead spot-check results.
- Bit-for-bit confirmation of all non-update columns.

If the look-ahead check or the byte-equality check fails, the script aborts with a non-zero exit code and **does not** write the parquet.

## Columns updated (and any new columns added)

- Updated (overwritten): `tf4h_wyckoff_bullish_score`, `tf4h_wyckoff_bearish_score`, `tf4h_wyckoff_phase_score`.
- New: none (all three columns already exist in the parquet schema, just with broken values).

## Why the parquet is gitignored

`*.parquet` is in `.gitignore`. The deliverable is therefore the script + tests + this report; the parquet update lives only on the host machine that runs the script.

## Files

- `scripts/rebuild_4h_wyckoff_features.py` — the rebuild script (idempotent; ADD/UPDATE columns only; atomic write).
- `tests/test_rebuild_4h_wyckoff_features.py` — five tests, including a synthetic-truncation look-ahead check and a CLI end-to-end test.
- `docs/knowledge/feature_store_4h_wyckoff_rebuild.md` — this report.

## Constants encoded in the script (all sourced from existing live code)

- `CFG_4H` — verbatim copy of `_CFG_4H` from `bin/live/live_feature_computer.py:1602`.
- `ACCUM_EVENTS` / `DISTRIB_EVENTS` — verbatim from `engine/wyckoff/events.py:1192`.
- `ROLLING_LOOKBACK_4H_BARS = 250` — derived from `LiveFeatureComputer.buffer_size = 1000` 1H bars / 4 = 250 4H bars.

No new tunables introduced.
