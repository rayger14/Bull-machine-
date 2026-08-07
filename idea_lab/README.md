# idea_lab — standalone single-strategy idea-screener (STUDY ONLY)

A clean, one-position-at-a-time backtester that makes engine artifacts
(fusion / dedup / margin crowd-out / 16-archetype competition) STRUCTURALLY
IMPOSSIBLE. Endorsed by wyckoff_audit addendum 26 as a fast idea-screener; the
production engine stays source-of-truth for live.

- `backtester.py`  — core engine + `run_selftest()` (0.00% parity vs bt_audit referee).
  Correct cost accounting: commission both sides once in PnL, slippage once in the
  fill price (deliberately does NOT replicate the addendum-26 double-count bug).
  Stop-first intrabar pessimism (stop on wick @ level; target needs a close).
  Fixed-risk sizing, non-binding wallet.
- `strategy_eye.py` — the integrated all-seeing-eye / Wyckoff-campaign strategy
  (accum-gate + spring/LPS confirmation + return-to-zone acceptance), with WI
  banked-and-derisked exit geometry AND a naive 1R/2R/3R ladder baseline.
- `run_eras.py` — 3-era validation (TRAIN 2018-2022, OOS-A 2023-2024,
  OOS-B 2025..2026-06) + spring-overlap + verdict.
- `run_output.txt` — captured run.

Run: `python3 run_eras.py`  (needs data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet)

## Result (as-run)
Self-test PASS (0.00%). Integrated strategy FAILS the pre-registered 3/3 bar:
TRAIN PF 0.72 (fail), OOS-A PF 1.56 (pass), OOS-B PF 0.58 (fail) — only the
bull era passes. WI exit geometry does NOT beat the naive ladder (loses 2/3,
ties 1/3). 69% of entries are orthogonal to the production spring archetype.
Clean lab => the failure is signal/regime, not engine contamination.
STUDY ONLY — nothing shipped.
