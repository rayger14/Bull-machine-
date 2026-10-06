# LC practice lab — source alignment preflight

September 29, 2026. Source-only diagnostic; **not an agent assessment or backtest**.

The user approved continuing toward the existing 12-case practice experiment.
The installed architectural workflow still requires a written implementation
plan review. The controller asked whether to waive the remaining spec/plan
approval pauses and implement directly, retaining one final software review
and a separate bounded market-call launch authorization. No reply to that
specific waiver question had arrived at this checkpoint. Do not mistake this
note for permission to dispatch paid market roles.

## What was checked

A read-only Python command used the existing `build_structure_packet`, pandas
and the local parquet archive. It completed with exit code 0.

- All 20 original source-request raw SHA256 hashes matched `evidence_lock.json`.
- All 20 source requests rebuilt through the existing structure packet builder.
- The archive raw SHA256 matched the original lock:
  `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
- Selected the first 12 packets sorted by `(decision_time, case_id)`, matching
  the existing design's January 20–May 3 roster.
- For each case, loaded only archive rows from its earliest supplied candle
  opening through, but excluding, its decision time.
- Checked unique, ordered, minute-aligned archive timestamps.
- For every supplied candle, required its constituent minute index to equal
  the complete expected left-closed/right-open UTC interval.
- Recomputed open/maximum high/minimum low/final close/summed volume directly
  from those minutes. OHLC absolute tolerance was `1e-8`; volume tolerance was
  `max(1e-6, abs(source_volume) * 1e-9)`.

Each case had 110 supplied candles: 10 daily, 18 four-hour, 24 hourly,
16 fifteen-minute, 12 five-minute and 30 one-minute candles.
**All 1,320 comparisons passed, with zero mismatches or coverage failures.**
These are overlapping multi-timeframe observations, not 1,320 independent setups.

PyArrow printed sandbox CPU-cache/NEON `sysctlbyname` permission warnings;
the parquet reads and all comparisons completed successfully.

## Boundaries and remaining work

The diagnostic did not read post-decision rows for outcome scoring, run any
model assessment, change frozen sources, authenticate exchange receipts,
rerun the parent detector, or establish a trading edge. Hashing the entire
archive bound its bytes but did not expose its future rows to an assessor.
This remains exposed practice/development data, not a pristine holdout.

No product code or implementation plan was created in this step. The missing
runner/capture/entry-resolution/report integration remains as specified in
[the existing design](../superpowers/specs/2026-09-29-lc-practice-lab-design.md).
Do not restart the already-completed packet/proposal/pre-entry/outcome modules.

Nothing is running. No market calls, live changes, commit, push or PR occurred.
Private sources and prices remain local at:

- `results/lc_consolidated_2026_09_15/judgment_v1/evidence/`
- `data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`

Next: resolve the requested workflow waiver (or complete the mandated plan
review), implement the existing design, verify its integrated path, then
present the prepared manifest and call-limited spending terms before launch.
