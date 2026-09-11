# Hourly cutoff experiment: source-stage readiness

## Finding

The existing native research pipeline is a guarded current-code diagnostic, not a configurable historical-live backtest. Independent quant inspection found that changing native `bypass_threshold` does not isolate fusion: it also changes position limits, entry spacing and allocation. This matters before comparing dollars or attributing any difference to threshold quality.

No production source/config or live state was modified. Hourly and minute BTC remain separate, equally important research tracks. The minute parent-context sleeve experiment can proceed under its existing frozen economics; hourly LC must retain its own selection and management contract rather than inherit minute stops/time exits.

## Decision order and confounding

The runner-owned engine performs archetype structure, cooldown, gates/inner thresholds and emitted-score dedup before returning selected signals. The runner then applies crisis-adjusted fusion and an exact per-archetype outer threshold in Step3b. Its stored display threshold can remain global and its diagnostic score/margin are rounded; those are not sufficient to reconstruct the exact decision boundary.

At `bin/live/v11_shadow_runner.py:1161`, exact per-archetype threshold/margin metadata are assigned to the signal. The below-threshold bypass branch can additionally enforce hard gates. At `:1269` onward, non-bypass activates position limits and at `:1287` same-direction spacing. At `:1357`, collection bypass constructs fixed-size intents while the non-bypass path invokes the allocator. Thus a true/false bypass comparison is a legitimate **bundled native-policy comparison**, not an outer-cutoff-only intervention.

The distinction does not mean book states should remain identical after a properly isolated gate change. They should diverge causally when different candidates enter. It means the non-gate rules and parameterization must be held fixed if the question is the marginal effect of that gate.

## Missing research interfaces

`scripts/research/virtual_book_replay.py` constructs the runner from hardcoded `CHAMPION`, and its manifest reflects that fixed source. `NativePipelineProcessor`, `run_pipeline_replay` and `native_pipeline_report.exercise` currently carry economics but no immutable variant-config interface. Mutating runner state after constructing its manifest would make experiment identity inaccurate.

Before an isolated hourly comparison, design and test a research-only intervention boundary with:

1. Immutable, hashed variant identity propagated through construction, reports and checkpoint contracts; reject cross-arm checkpoints.
2. An exact copied Step3b trace before further mutation: raw detector score, crisis-adjusted score, actual per-archetype threshold, margin, gate status and disposition. Do not infer that trace from rounded dashboard fields.
3. An intervention that rejects only the designated sub-threshold candidates while preserving the remaining collection allocation/spacing/limit semantics. Keep a bundled bypass-toggle comparison separately named if later studied.
4. Full original prehistory replay in each arm, including positions before the emitted reporting window, with separate prefix/restart checks and complete exits/open inventory. Filtering historical exits is not this experiment.
5. Separate executable fill/accounting design: the current adapter explicitly preserves completed-hour-close prices, hour-open labels and native gap-stop assumptions. Source-parity results cannot be labeled obtainable market returns.

An agent suggested an additional runner flag/config seam as one implementation route. That would touch active source and is **not approved or implemented here**. The next design must first establish research-only isolation and its witnesses; production changes would require separate authority.

## Relation to the all-context objective

The immediate goal is not another scalar that obscures decision stages. It is observable evidence for context, setup, entry, invalidation and book consequences. An exact stage trace lets a later quant comparison answer which constraint changed an outcome without conflating technical quality, crisis penalties, threshold choice and allocation changes.

This is a readiness diagnosis, not an economic result or a reason to rank hourly below minute. The completed minute experiment must remain an explicitly bounded unfunded reference comparison with its own source and management assumptions.
