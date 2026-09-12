# September 8 minute-layer inventory

The user requested minute data inside larger-timeframe structure, including the
known September 8 LC winner as a separate diagnostic. Targeted read-only search
of the current project and its parent checkout, including ignored files, found
**no minute source extending to September 8**. This is absence from discovered
sources, not proof that no other archive exists on the computer.

- Existing Binance archive: `/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/833eefef-c5b5-45bb-af8d-fd9afb9e129c/scratchpad/btc_1m_2021_2026.parquet`.
  Existing validated inventory: 2,979,360 contiguous minutes, January 1, 2021
  through August 31, 2026 23:59 UTC. Source receipts remain uncertified.
- Additional local CME futures file: `/Users/rayghandchi/Bull Machine/Bull-machine-/data/databento/btc_fut_1m_2021_2026.parquet`.
  Independent timestamp/symbol inspection found BTC.c.0 through August 18
  23:54 UTC (1,314,629 rows) and MBT.c.0 through August 18 23:57 UTC
  (1,318,697 rows). No within-symbol duplicate timestamps. There are many
  intervals longer than a minute, including session/weekend effects; this was
  not a missing-expected-session-bar audit. These are different instruments and
  must not be silently spliced into Binance evidence.
- The derived minute parent-economics fixture covers June 1–20 only.

The saved September 8 audit snapshots remain in
`results/research_validation_2026_09_10/sep8_case_audit/`: `trades.json`,
`status.json`, `signal-log.json`, `candle-history.json`, `case_summary.json`.
They came from separate dashboard GET requests, not an atomic capture. The
candle snapshot contains 200 hourly rows with three exact duplicate pairs;
hourly bars cannot reconstruct minute ordering.

The known LC source label is September 8 at 13:00 UTC. That hourly candle closes
at 14:00 UTC, before any additional unknown receipt/processing delay. Its low
must not be counted as post-decision movement. Current code and rounded historical
narrative telemetry do not prove exact consumed gate values. See the existing
[case audit](sep8_liquidity_compression_case_2026_09_10.md).

No source was downloaded, rewritten or filled. The known-success case remains
excluded from blinded performance comparisons; its minute layer is unverified.
