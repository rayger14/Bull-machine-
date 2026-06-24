# Industry Study: How Production Systems Detect & Prevent Backtest-Live Divergence

**Date**: 2026-06-11
**Method**: Deep-research workflow — 5 search angles, 22 sources fetched, 101 claims extracted, 25 adversarially verified by 3-vote panels (23 confirmed 3-0, 2 refuted 0-3).
**Trigger**: [[liquidity_score_root_cause_2026_06_10]] — our liquidity_score scale mismatch is a textbook instance of what the industry calls **training-serving skew**.

## The three-layer industry playbook (all claims verified 3-0)

### Layer 1 — Prevention by architecture: single code path
- NautilusTrader and QuantConnect/Lean run the IDENTICAL strategy/feature code through one
  event-driven engine in both backtest and live ("deploy from research to production with no
  code changes"). Backtests stream data in fast-forward; live streams it in real time; the
  only difference is the clock/data-source abstraction.
- Point-in-time correctness: all events sequenced on close-timestamps (`ts_init`), making
  lookahead structurally impossible.
- **Critical caveat (also verified)**: even perfect code parity does NOT eliminate divergence —
  QC documents backtest tick slices of 1ms vs live slices up to 70ms causing residual
  equity divergence; Freqtrade documents intra-candle evaluation-order artifacts. So
  architecture is necessary, not sufficient: you still need Layers 2-3.

### Layer 2 — Detection by feature logging + distribution comparison (ML feature-store practice)
- Canonical pattern (Google Vertex AI, Rules of ML #29): log the exact features the live
  system computes (D_serve) and compare per-feature against the historical/training baseline
  (D_train).
- **Metric choices**: Jensen-Shannon divergence or Wasserstein distance for continuous
  features. AVOID plain Kolmogorov-Smirnov as primary metric — it is insensitive to tail
  changes (verifier reproduced with scipy: q99 truncation undetected at n=1000). Our q90-scale
  failure was gross enough for KS, but subtler top-tail skew is exactly where KS fails.
- **Thresholds (verified industry defaults)**:
  - PSI bands: < 0.1 fine | 0.1–0.25 warn/attention | ≥ 0.25 investigate/fail
  - Vertex AI default per-feature alert: distance score ≥ 0.3 (JS for numeric)
  - NannyML default alert: JS > 0.1
  - PLUS hard range assertions (live min/max/q01/q99 vs historical) — these catch scale bugs
    that distribution metrics can smooth over. (Range assertion alone would have caught our
    0.675-vs-0.72 bug.)
  - Caveat: PSI bands are credit-scoring rules of thumb (Siddiqi 2006), sample-size sensitive,
    not statistically derived. Calibrate on own history.

### Layer 3 — Detection by shadow reconciliation
- Freqtrade: dry-run before live is MANDATORY ("backtesting will never replace dry-run");
  ships `lookahead-analysis` that re-runs sliced backtests and diffs dataframes
  column-by-column to empirically detect bias. Their war-story lesson, verbatim relevant to us:
  **"the bias in the strategy is usually THE driving factor for too-good-to-be-true profits"** —
  fixing it usually destroys the apparent edge.
- QuantConnect: runs a parallel out-of-sample backtest against EVERY live deployment and
  overlays equity curves (returns correlation + DTW distance) — continuous reconciliation.
- Generic pattern (Microsoft shadow-testing / Twitter Diffy): mirror identical inputs to both
  versions, withhold the candidate's outputs, auto-diff. Applies directly to running two
  feature computers side by side.
- Freqtrade's assumption-stress pattern: re-run backtests under harsher live-like assumptions
  (`--timeframe-detail`) as the LAST step of development; big deltas = divergence red flag.

## Ranked adoption plan for Bull Machine (leverage ÷ effort)

1. **Feature logging (prerequisite, ~free)** — live runner persists every computed feature
   row to parquet. (Already planned as the live feature-row logger; this study confirms it's
   the foundation of everything.)
2. **Golden-sample parity test in CI** — push a fixed window of raw historical bars through
   the LIVE feature code, diff column-by-column vs the feature store. Deterministic; would
   have caught liquidity_score before any statistics. (We effectively did this manually with
   the bridge test — codify it as `tests/test_feature_parity.py`.)
3. **Threshold-coverage unit tests** — for EVERY gate/threshold in every archetype YAML,
   assert the live-computed feature's observed range actually spans the threshold; else flag
   the archetype as structurally untradeable. (Would have flagged wick_trap and
   trap_within_trend instantly.)
4. **Nightly distribution check** — per-feature PSI/JS, live logs vs same-regime historical
   baseline: warn at PSI ≥ 0.1 or JS ≥ 0.1; fail at PSI ≥ 0.25 or distance ≥ 0.3; plus
   min/max/q01/q99 schema assertions.
5. **Signal-for-signal shadow reconciliation** — nightly re-run of the backtester over the
   exact dates the paper trader ran; diff emitted signals one-to-one (QC's pattern at signal
   level, which is stricter than their equity level).
6. **Converge to a single code path** — one feature module imported by BOTH the store builder
   and the live runner, clock/data-source abstraction as the only difference. Highest payoff,
   highest cost; the only item that PREVENTS rather than detects. Our `scripts/rebuild/`
   replay pipeline is already a big step: it builds history through the live path.

## Verified-refuted claims (do not cite)
- "Serving-side feature logs are strictly NECESSARY for skew detection" — refuted 0-3
  (best practice, not the only mechanism).
- A blog's overly narrow definition of training-serving skew — refuted 0-3.

## Open questions (unverified by this study)
- Quantile/rank-based thresholds vs fixed cutoffs in trading literature: no claims survived
  verification. Our planned re-quantiling of 0.72 → live-q90 is CONSISTENT with the verified
  tail-skew findings but not directly evidenced by external literature.
- Quantitative halt thresholds for signal-level reconciliation (QC's DTW/correlation bounds).
- Third-party quant postmortems of feature-scale divergence: none survived verification —
  the war-story evidence base beyond framework docs is thin.

## Key sources (all verified current June 2026)
NautilusTrader docs/repo · QuantConnect Lean docs + live reconciliation docs ·
Freqtrade backtesting + lookahead-analysis docs · Google Vertex AI skew monitoring ·
Google Rules of ML #29 · NannyML drift-metric comparison · Microsoft shadow-testing playbook ·
Arize/Fiddler PSI references (Siddiqi 2006 lineage)
