# Candidate rules: evidence, sequence, and parent structure

Research decisions delegated by Ray were reviewed by the quant team on September 10, 2026. These are **candidate contracts, not validated strategies**. All 17 native archetypes remain independent references. No production gate, threshold, risk setting, or exit rule is changed.

## Order of work

1. Separate missing evidence from observed values.
2. Test one explicit prior-compression prerequisite for hourly LC.
3. Define a causal, fixed parent-range ledger shared by hourly and minute candidates.
4. Compare each intervention independently before testing combinations.

Hourly and minute BTC receive equal consideration; equal consideration does not mean forcing them through identical indicators or clocks. The common model is parent structure, child setup, observable trigger, invalidation, and management. A new combined archetype is conditional on those parts surviving testing individually.

## H1 — observed evidence required

Two separately identified interventions: `oi_observed_evidence_v1` and `lc_observed_evidence_v1`.

OI requires attributable, finite, as-of-available inputs for both OI horizons and the inputs used for taker imbalance. LC requires finite raw volume Z, RSI, BB width and chop, including prerequisites of derived gates. Observed zero is valid evidence; fallback zero is not. Preserve existing numerical comparisons, including OI's inclusive `<= 0`, so this does not also become a sign-threshold experiment.

The validity check is unconditional permission before fusion/cooldown, not another soft score that a collection bypass can overcome. Freshness must be specified by a frozen feed contract, not selected from outcomes. Unknown provenance/freshness is unavailable. Missing evidence rejects a new entry; it does not introduce a new liquidation rule for existing positions.

Source basis: LFC missing-feed defaults (`bin/live/live_feature_computer.py:2346`), OI skip/soft gates (`configs/champion/archetypes_v14rq/oi_divergence.yaml:62`), LC raw/derived missing-value witnesses (`tests/research/test_trader_intent_witnesses.py`). This is an engineering hypothesis, not attributed trader doctrine.

First output: paired decision counts for observed/defaulted/missing/invalid/late inputs, plus rejected opportunities. In the OHLCV-only replay, rejecting OI demonstrates contract compliance, not economic improvement. A full candidate replay must separately expose changes to cooldown, dedup and other archetypes' opportunities; filtering a completed trade list is insufficient.

## H2 — prior compression for hourly LC

`lc_prior_compression_v1` adds exactly one prerequisite: the immediately preceding **completed hourly** feature row has finite `bb_width <= 0.06`. Missing prior history rejects. Retain all current thresholds, both RSI-extreme branches, current-hour trigger, direction, stops, and management.

This is the smallest operational test of the YAML's “after compression” description. One prior hour is a project hypothesis, not a sourced trader duration or optimized setting. Do not add a longer search window, directional RSI restriction, parent geometry or altered exit in the same arm.

Source basis: current `_check_E` uses the terminal vector rather than history (`engine/archetypes/logic.py:662`); current LC gates check present BB width (`configs/champion/archetypes_v14rq/liquidity_compression.yaml:21`). The existing paired-history witness shows why a separate sequence rule is needed.

Compare native LC to the candidate with unchanged causal prehistory. Report predicate/gate candidates, cooldown events, dedup winners/losers, entries, and full-book displacement. Report RSI-low/high strata without choosing the winner afterward. Minute input must aggregate to completed hours for this archetype, not generate extra LC decisions.

## H3 — shared parent structure, separately tested children

This stage is conditional on a versioned **fixed**, causally observable parent-range constructor and ledger. Each record needs range lineage ID and immutable version ID, bounds, anchor IDs/timestamps, confirmation/availability time, and explicit body-close invalidation time/rule. Preserve pre/post-bar bounds and the version against which a sweep was evaluated. Freeze the parent assigned before the child's first sweep: its state `available_at` must precede the sweep minute's open, not merely share an earlier-looking hour-open label. Later pivots, redraws or revised annotations cannot authorize earlier events. Source floor tightening creates a new version; source reformation with reused anchors creates a new lifecycle and records those anchor IDs rather than silently requiring fresh pivots.

- `lc_fixed_parent_reclaim_v1`: parent known before the event; current low breaches its lower boundary and current close returns inside the same range. Compare parent-only and H2-only before their conjunction. Keep RSI choices unchanged in this arm.
- `minute_child_sweep_parent_location_v1`: preserve the causal minute equal-low selector. Require its child level inside the frozen parent's lower half and reclaim close inside that range. Reject absent, unconfirmed or already-invalidated parents. Preserve the child stop, next-eligible-open reference entry, four-hour hold/lockout and cost convention for the diagnostic comparator.

Exact parent anchoring and the lower-half rule are research hypotheses. A completed-hour parent is one proposed scale, not a teacher-certified choice. Do not replace a fixed range with rolling extrema and call it faithful. Do not impose a universal HTF direction veto: the local provenance ledger describes HTF permission as a sizing dial in the cited teaching, not necessarily exclusion.

Source/invention boundary: `docs/knowledge/wyckoff_audit.md:905` distinguishes sourced fixed objects and wick/body-close semantics from invented magnitudes and notes a thin Moneytaur corpus. A wider search recovered both referenced modules under `/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/`: `structural_range.py` and `htf_pivots.py`. Their comments explicitly label pivot lookbacks and width constants as project hypotheses. Recovery is not verification: causal input validation, timestamp semantics, and immutable event-bound parent versions still need review before reuse.

No new parent-triggered exit, trailing policy or scale-in is included. First annotate fixed baseline events to measure the direct gate effect; then replay candidate selection/lockout to measure displacement. Those are different questions.

## Required evidence before judging a candidate

- Freeze input/config/model identities, source code, exact rule, clocks, and all attempted variants before viewing its outcomes.
- Test missing/defaulted versus observed zero, boundary equality, future append, parent redraw, incomplete higher-timeframe buckets, and restart behavior.
- Keep entry-rule changes separate from fill/accounting changes. Native close-price and gap-stop outcomes are source diagnostics, not executable performance.
- Previously examined 2018–2026 hourly and 2021–2026 minute data remain reused research, not pristine holdout. Use qualified coverage only, carry prehistory, purge overlapping outcome windows where splitting, and retain unresolved positions as censored.
- Report coverage and independent-position counts, costs, exposure, initial-risk-normalized outcomes, drawdown, and target/full-book effects. A rejection-count improvement is not a profit finding.
- Reserve genuinely new forward evidence after the design freeze. Do not select a threshold or combine arms because it looks best on reused data.

Immediate quant approval covers H1 evidence witnesses and H2's single prior-hour prerequisite. Historical parent-gate testing awaits the frozen range constructor. No candidate has been declared safe, profitable, or ready for capital.
