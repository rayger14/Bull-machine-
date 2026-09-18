# Hourly cutoff experiment: source-stage readiness

## Finding

The existing native research pipeline is a guarded current-code diagnostic, not a configurable historical-live backtest. Independent quant inspection found that changing native `bypass_threshold` does not isolate fusion: it also changes position limits, entry spacing and allocation. This matters before comparing dollars or attributing any difference to threshold quality.

No production source/config or live state was modified. Hourly and minute BTC remain separate, equally important research tracks. The minute parent-context sleeve experiment can proceed under its existing frozen economics; hourly LC must retain its own selection and management contract rather than inherit minute stops/time exits.

## Decision order and confounding

The runner-owned engine performs archetype structure, cooldown, gates/inner thresholds and emitted-score dedup before returning selected signals. The runner then applies crisis-adjusted fusion and an exact per-archetype outer threshold in Step3b. Its stored display threshold can remain global and its diagnostic score/margin are rounded; those are not sufficient to reconstruct the exact decision boundary.

At `bin/live/v11_shadow_runner.py:1161`, exact per-archetype threshold/margin metadata are assigned to the signal. The below-threshold bypass branch can additionally enforce hard gates. At `:1269` onward, non-bypass activates position limits and at `:1290` same-direction spacing. At `:1357`, collection bypass constructs fixed-size intents while the non-bypass path invokes the allocator. Thus a true/false bypass comparison is a legitimate **bundled native-policy comparison**, not an outer-cutoff-only intervention.

The distinction does not mean book states should remain identical after a properly isolated gate change. They should diverge causally when different candidates enter. It means the non-gate rules and parameterization must be held fixed if the question is the marginal effect of that gate.

## Missing research interfaces

`scripts/research/virtual_book_replay.py` constructs the runner from hardcoded `CHAMPION`, and its manifest reflects that fixed source. `NativePipelineProcessor`, `run_pipeline_replay` and `native_pipeline_report.exercise` currently carry economics but no immutable variant-config interface. Mutating runner state after constructing its manifest would make experiment identity inaccurate.

Before an isolated hourly comparison, design and test a research-only intervention boundary with:

1. Immutable, hashed variant identity propagated through construction, reports and checkpoint contracts; reject cross-arm checkpoints.
2. An exact copied Step3b trace before further mutation: pre-outer-cutoff selected-signal fusion score (already processed by the inner engine), crisis-adjusted score, actual per-archetype threshold, margin, gate status and disposition. Do not infer that trace from rounded dashboard fields or call the selected-signal value an untouched detector score.
3. An intervention that rejects only the designated sub-threshold candidates while preserving the remaining collection allocation/spacing/limit semantics. Keep a bundled bypass-toggle comparison separately named if later studied.
4. Full original prehistory replay in each arm, including positions before the emitted reporting window, with separate prefix/restart checks and complete exits/open inventory. Filtering historical exits is not this experiment.
5. Separate executable fill/accounting design: the current adapter explicitly preserves completed-hour-close prices, hour-open labels and native gap-stop assumptions. Source-parity results cannot be labeled obtainable market returns.

An agent suggested an additional runner flag/config seam as one implementation route. That would touch active source and is **not approved or implemented here**. The next design must first establish research-only isolation and its witnesses; production changes would require separate authority.

## Relation to the all-context objective

The immediate goal is not another scalar that obscures decision stages. It is observable evidence for context, setup, entry, invalidation and book consequences. An exact stage trace lets a later quant comparison answer which constraint changed an outcome without conflating technical quality, crisis penalties, threshold choice and allocation changes.

This is a readiness diagnosis, not an economic result or a reason to rank hourly below minute. The completed minute experiment must remain an explicitly bounded unfunded reference comparison with its own source and management assumptions.

## Follow-up: candidate fixture-only seam, September 10

A parallel read-only review during the broader minute-coverage audit identified a narrower **candidate mechanism**, without editing production: temporarily substitute a source-pinned truth-value proxy for the research runner's `bypass_threshold`. It would return false only when native `process_bar` reads the below-cutoff branch at line 1171, and true at the downstream collection reads (1270, 1283, 1291, 1358). Thus the native rejection path can be exercised while preserving skipped position limits/spacing and the fixed `0.02` collection intents. Root inspected those branch sites and confirmed they are distinct. This does not imply subsequent book states should remain identical after admission differs.

The agent reported two in-memory synthetic checks (one below cutoff, one above) supporting feasibility, not general isolation or economic validity. **No implementation is approved or retained yet.** A line-sensitive proxy is brittle: any eventual design must pin the runner hash/code object, locate/validate the exact branch structurally, fail on unknown reads or source drift, scope restoration even on exceptions, and remain research-only. Current runner SHA256: `10722d523ef931bc6172b8b67efb69d1dec3f956d4926552aec1f85a36f659c8`.

The smallest proposed fixture contract exposes exactly two immutable named arms, `collection_current` and `outer_cutoff_only_current_collection_downstream`, rather than arbitrary config/threshold mutation. Propagate identity through `NativeSignalBook`, `NativePipelineProcessor`, `run_pipeline_replay` and report `exercise`, binding arm/intervention/source hashes into manifests and processor contracts. Existing `replay_clock.py:184` already places processor identity in the checkpoint prefix hash; its mismatch rejection can enforce cross-arm isolation.

The current observer copies selected signals before runner mutation (`engine_signal_replay.py:38–91`), but the book does not retain exact post-Step3b information for every rejection. Future tracing must capture the native selected score and original crisis-adjusted score, exact per-archetype threshold/margin, gate status, comparator reachability and final disposition. **Do not label `threshold + margin` an exact capture of adjusted fusion:** floating-point cancellation can alter it. Capture the original native local directly or prove bit-for-bit equality; rounded display reconstruction is also insufficient. The reviewer confirmed this correction and proposed a separately source-pinned, single-thread scoped line observer immediately before the comparator (currently line 1168), with existing-tracer refusal and unconditional restoration. The bypass proxy alone cannot capture above-threshold signals because they never evaluate its branch. Earlier regime-blocked signals need `reached_outer_comparator=false`, not invented scores. This tracing approach also remains an unimplemented design candidate.

Before any historical experiment, require below/equal/above boundary cases, gates passing/failing, source-drift/unknown-callsite failures, exact rejected-signal traces, immutable/cross-arm checkpoint rejection, and witnesses that surviving signals preserve downstream collection semantics under full positions, consecutive same-direction entries and allocator spying. Keep all 17 evaluations per completed hour and same-arm full-prehistory/prefix/restart/state/exit checks. Existing source-faithful fixtures cover a 20-hour synthetic witness and June 10–20's 240 hours (168-hour prehistory, emission June 17, checks at 168/239); these are reused and retain nonexecutable native fill assumptions. This follow-up is a concrete next design input, not a new hourly strategy result.
