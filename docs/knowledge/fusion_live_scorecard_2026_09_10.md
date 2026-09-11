# Live fusion scorecard: source inventory and descriptive experiment

## Status and purpose

The user asked whether fusion and its thresholds were correctly designed, not simply whether to turn the old cutoff back on. The approved first experiment separates stored fusion, stored entry threshold and logged margin, using explicit-ID grouped recorded exits. The subsequent challenger ladder compares native threshold enforcement, simple static cutoff and narrowly defined parent/context rules; none is promoted here.

The reusable implementation passed task review and reproduces the independent raw-reference measurements below. The [registered contract](../superpowers/specs/2026-09-10-fusion-live-scorecard-design.md) fixes grouping, exclusions, metrics and limits before additional outcome rankings. Final integration review is pending. No live or production changes.

## Frozen sources

Separate GET downloads from http://165.1.79.19:8081 are non-atomic paper/shadow snapshots, saved privately under `results/research_validation_2026_09_10/fusion_scorecard/`:

| File | SHA256 |
|---|---|
| trades_20260910.json | 3c6ad8619dd5d5ea22f382d346c8f58ff02071307973913a5a403fe845bcbf12 |
| status_20260910.json | 56343b9e1fc091cd622f988798101dda7f04785e2f7b5041d8168cf83eb6b03b |
| signal_log_20260910.json | 662dbcdd64e1cf7f4af0ad4af8b0f11d32663e14fee7eb88741fa4feb2913f38 |

Status server time2026-09-10T23:50:30.522662UTC; heartbeat updated23:01:56.103217UTC. The trades and signal snapshots are byte-identical to the earlier September8-case downloads; no newly completed exits appeared between those snapshots.

535 rows are exit legs, not 535 trades. Entries span 2026-02-15 20:00 UTC through 2026-09-09 20:00 UTC; exits through September 10 at 12:00 UTC. 467 rows carry explicit position IDs, forming 230 groups; 68 older rows lack IDs. Explicit-group entry range starts March 3 at 08:00 UTC. Group multiplicities: 110 single-leg, 36 two-leg, 51 three-leg, 33 four-leg. Three current open IDs have no overlapping exit groups and no scale-outs yet.

The bounded project/sibling/project-associated temporary-directory search found no historical completed-position receipt ledger with original quantities. Current open inventory does have original/current quantities. Consequently, absent-from-open exit groups are not certified complete positions. Risk from displayed stop and summed recorded exit quantity is a diagnostic proxy, not fully net initial-risk R. No starting equity or average actual per-trade risk is reconstructed; the report is not an account-equity backtest.

Older saved snapshots (532,529,and292exit rows) and334 historical ENTRY-action CSV rows were found, but no richer completed receipt history. They are overlapping evidence, not additional independent outcomes; do not concatenate them to inflate sample size. Current snapshots have no historical source/config commit hashes.

## Threshold-stage evidence

Every explicit group's entry/score/threshold metadata is invariant across its recorded exits.165of230groups have a factor-display dynamic threshold differing from `threshold_at_entry` (359of535exit rows overall). Preserve both fields.

Local source explains two different uses: per-archetype threshold is assigned to `s._threshold_at_entry` at `bin/live/v11_shadow_runner.py:1161` and persisted on positions; factor attribution reads `last_dynamic_threshold` at`:1691`. For the last exit record, score0.2595 minus entry threshold0.2807 equals margin−0.0212, while factor-display threshold is0.3707. Arithmetic consistency does not prove historical decision-stage authenticity or good calibration. No remote-version attestation is supplied by the present checkout.

The rolling200signal rows spanJuly8 21:00UTC throughSeptember10 13:00UTC:118allocated and82rejected, across14archetypes. They are already downstream of inner gates, cooldown and dedup—not a full opportunity population.173of200top-level signal thresholds do not reconcile with score/margin under the registered rounding tolerance. Local source initializes a global threshold field while overwriting margin from the per-archetype comparison.46bypass notes encode a per-archetype threshold;33differ from the top-level threshold. Narrative text is descriptive telemetry, not a substitute for authenticated numeric decision fields.

Inventory-only synthesized joins of allocated signal tuples suggest112match exit IDs,3match current opens and3match neither. The scorecard deliberately does not use these inferred joins or reconstruct unobserved outcomes. This incomplete capture is another reason not to call a filtered result the engine's threshold-on backtest.

## Accounting limits

Current source calculates recorded exit PnL using fill prices and exit commission. Entry commission/cash movements and per-position funding are not reconstructed in these rows. No ad-hoc commission is subtracted from historical groups using today's configuration. Observed dollar subtotals and stop-risk-proxy ratios are separately named; neither is fully net account return.

The registered report provides raw coverage and conflicts, all17archetype counts, stored-margin cohorts, dollar subtotals, proxy ratios and descriptive rank associations. It does not infer a universal useful threshold from pooled raw archetype scores, calculate trade-row Sharpe/drawdown, optimize weights, or attach PnL to parent annotations lacking an economic contract.

## External libraries

The [library assessment](backtesting_library_assessment_2026_09_10.md) recommends no new dependency for this descriptive unit. NautilusTrader is a candidate for a separately pinned quote/latency/gap-fill benchmark; hftbacktest needs suitable depth/event data for a queue study. The repository's existing Nautilus-named strategy imports its own EventEngine, not the external package.

## Verified descriptive measurements

After the protocol was frozen, root and a separate read-only quant agent independently calculated the following literal reference from the raw snapshot. The reviewed reusable implementation reproduces the counts, subtotals, PF and all six rank correlations. These are grouped recorded exit subtotals, not verified closed-position, fully net or threshold-on strategy results.

| Logged margin cohort | Groups | Recorded exit subtotal | Recorded-subtotal PF | Win fraction | Mean displayed-stop risk proxy | Mean PnL / risk proxy |
|---|---:|---:|---:|---:|---:|---:|
| Nonnegative | 87 | +$11,270.02 | 1.2475 | 41.38% | $1,032.26 | +0.10321 |
| Negative | 143 | −$11,897.42 | 0.8364 | 43.36% | $929.56 | −0.06277 |
| All explicit-ID groups | 230 | −$627.40 | 0.9947 | 42.61% | $968.41 | +0.000012 |

Win fractions count groups after all recorded legs are summed, not individual scale-outs. The risk ratio is explicitly a proxy: displayed stop distance times observed exit quantity is not authenticated initial risk or full netR. No account starting equity, net account return or average actual risk is implied.

| Stored input | Spearman vs recorded dollars | Spearman vs risk-proxy ratio |
|---|---:|---:|
| Fusion | 0.11489 | 0.09821 |
| Entry threshold | 0.01154 | 0.06068 |
| Logged margin | 0.07733 | 0.02094 |

These weak pooled rank associations do not establish predictive calibration or causality. They neither support a universal score inversion nor prove the existing cutoff is correctly calibrated. Historical source versions and decision stages are not authenticated; all results mix selected observations and changing environments.

### Concentration and composition

All explicit-ID groups by entry month: March 20 / −$9,190.54; April 28 / +$2,078.62; May 42 / −$10,777.14; June 22 / −$3,569.29; July 54 / −$3,920.95; August 51 / +$28,312.61; September 13 / −$3,560.71. These are calendar groupings, not assumed implementation epochs.

August contributes 40 of 87 nonnegative-margin groups and +$23,141.93 of their recorded subtotal; outside August that cohort totals −$11,871.91. August's 11 negative-margin groups also total +$5,170.68. Excluding August is a concentration sensitivity diagnostic chosen after seeing this table, not a pre-registered alternative strategy or independent holdout.

LC alone has 37 explicit-ID groups with +$11,168.69 recorded subtotal. 12 nonnegative-margin LC groups contribute +$2,512.81; 25 negative-margin LC groups contribute +$8,655.88. Both LC cohorts are sparse. This shows what a static retained-group accounting would discard, not the PnL of a native threshold-enforced LC strategy. Five August LC groups account for +$8,198.95; the selected September 8 winner is not independent validation.

The earlier July reports and this broader snapshot cover different samples. A negative historical correlation does not justify declaring fusion permanently inverted; a later positive subtotal does not justify switching it on. The next tests must separate archetype, period, actual decision-stage score, and portfolio selection effects.

## Implementation verification and reproducibility

Source: `scripts/research/fusion_live_scorecard.py`; tests: `tests/research/test_fusion_live_scorecard.py`. Initial implementation `98ca934`; raw-schema boundary fix `2a0827b`. The first actual-snapshot check exposed a mismatch between invented synthetic input keys and the API's `timestamp_entry`, `timestamp_exit`, `stop_loss`. A full raw-shaped fixture and no-alias regression reproduced it before the boundary-only fix. Scoped re-review approved the fix; the actual snapshot then retained all 230 explicit-ID groups with zero quarantines and accounted for all 535 rows.

Focused tests: **39 passed**. Root's fresh full research suite after the fix: **310 passed**, one existing urllib3/LibreSSL warning, 12.64 seconds. Production `engine/`, `bin/`, and `configs/` diff from phase base `4cfaded` is empty. All 17 archetype keys are present, including zero observed groups for volume_fade_chop and whipsaw; no definition was changed.

Private generated artifact: `results/research_validation_2026_09_10/fusion_scorecard/scorecard_20260910.json`, SHA256 `d066ce45ef5d2d2b32752e0ad76da1cdbc5f2e7b14b75cc900c54583afc13da2`. Policy source SHA256 `69475e36e0a97c44e3d5b1a75c7128d2bcdbe72fa904feba54b5416121a5f80f`. It embeds raw source hashes, snapshot clocks, all groups/exclusions/open inventory, all archetype summaries, signals, and check results. Raw source hashes matched; copied-input repeatability and input nonmutation passed; the independent reference counts/subtotals/PF/ranks matched. JSON serialization rejects nonfinite values.

All 230 group source versions remain unknown; completion and overall certification remain false. No displayed-threshold substitutions, signal-to-position outcome joins, learned cutoffs, inferential intervals, forward-test results, or native policy backtests are included.

## Next registered comparisons

1. Recorded score/threshold/margin description on the frozen snapshot.
2. Source-stage and candidate-history reconstruction for native collection versus native outer-cutoff enforcement.
3. A simple static cutoff trained only on earlier qualified evidence, evaluated later with all attempted variants recorded.
4. Setup-specific context: parent lifecycle/location, room to the next opposing structure, then one direction-and-horizon-specific macro interaction. Preserve separate hourly/minute source, ATR, entry and exit contracts; no universal timeframe-agreement gate.

Existing live/historical data have already informed research. Later chronological splits are retrospective validation, not pristine holdout. New forward evidence is required after candidate freeze. Entry, exit, costs and risk must be fixed before economic testing; no favorable-result early stopping or performance claim from the present scorecard.
