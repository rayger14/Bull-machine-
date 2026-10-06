# Contextual support-reaction research suite

## Intent, scope and authority

The user approved building the missing tests and setup layer. This implements
one offline long support-reaction hypothesis, not a native LC repair, complete
Wyckoff accumulation detector, all17 rewrite, or live upgrade. Structure chooses
the applicable setup, volume behavior supports/challenges, and a causal minute
child supplies timing. No scalar fusion vote or requirement that every indicator
be positive. Every decision must identify its range, recovery, support, minute
anchors, invalidation, available room, and uncertainty.

Use the existing quant checkout, Python/pandas/pytest and local archive. Preserve
all frozen thesis/LC files, results, native code, live/config/fusion and unrelated
changes. No installs, paid API assessments, commits, push or PR. Root implements
inline, with a delegated quant design review and one final independent software
review, following the user's standing delegation of offline technical decisions.

Deliverables: executable versioned contract; contrasting behavior and causality
tests; transparent fixed-execution adapter; chronological comparison harness;
outcome-hidden benchmark exporter and annotation validator; reproducible bounded
engineering check. A benchmark export is NOT completed independent annotation.
No strategy promotion or claim of economic edge is a deliverable of this build.

## Source fidelity and immutable population

Read `research_notes/Thesis entry rule source fidelity/source_audit.md` and
`research_notes/Trader origins and archetype fidelity/original_trader_sources.md`.
Modern supporting source: https://www.wyckoffanalytics.com/wyckoff-method/ .
Known locations, relative volume/spread, tests, and timeframe-specific evidence
are supported concepts. Every constructor, threshold and clock below is a
PROJECT hypothesis. None is attributed as an authenticated trader formula.

The natural development population is ALL183 raw frozen spring episodes in
`results/thesis_census_2026_10_03/source_v1/source.json`, SHA256
`b27440a9247ea93b77458f08fc13009ab56679a43bbfc66aba3a86d43039730b`.
It is not the17 support survivors or3 old intents. The census covers origin
closes [2024-01-01,2026-08-24)UTC. Reuse original raw IDs and parent lineages;
revised rule identities are separate. Shared price windows are dependent cases.
This known/exposed calendar is development data, never a pristine holdout.

The minute archive and parent source remain pinned by the frozen thesis spec.
Verify existing source receipt/bindings before and after a natural run. The new
experiment has its own policy, code/data hashes and output directory. Never
relabel a modified packet as an unchanged old strategy decision.

## Price structure, sequence and evidence clocks

Use only complete source candles at each decision. Bind L/H from the original
4h parent strictly before the spring open. The range is `pivot_range`; its
phase is `unclassified`, reason `no_qualified_phase_sequence`. Daily context
is annotation, not a blanket bullish permission gate. Local recovery and parent
breakout are recorded separately. A completed4h close above H is a breakout
observation, not proof of held/retested acceptance. No accepted-breakout claim.

B is an in-range support reaction, with one support and one proposal per origin:

1. Recovery: first complete1h candle starting at/after T0 and closing strictly
   above the frozen spring high K, available by T0+48h inclusive. Name it
   `spring_high_recovery`, not parent breakout. No mandatory pre-recovery test.
2. Support: first later complete1h candle starting at/after recovery availability,
   with spring_low < low <= K and close > K, available by T0+72h inclusive.
   Earlier nonqualifying touches do not permanently cancel the sequence.
3. Child: minute candles start at/after support availability. A high pivot has
   two strictly lower highs on each side; a low pivot two strictly higher lows.
   All five constituents must be complete and post-support. Availability is the
   second right-hand candle close, never the pivot center time. Equal highs/lows
   do not qualify. Freeze the first confirmed high, then the first later-center
   confirmed low strictly above the support candle low.
4. Trigger: first subsequent completed minute close strictly above the locked
   child high; its bar must start at/after low-pivot availability. No same-bar
   low confirmation and breakout. It must be strictly before support+60min and
   before T0+7d. This clock is one nested hourly interval, not a fitted extension
   of the old15min window. No rearming or alternative pivot search after failure.
5. At that decision, require L < close < H, close > original_stop, and
   H-close >= 2*(close-original_stop). A failure is a known nonentry, not an
   unknown outcome or an invitation to search for another trigger.

Original stop = spring_low -0.1*frozen4h ATR14. The parent thesis invalidates on
a later complete4h close below L. Pre-launch lifecycle clarification: before
support, a completed1h candle with low<=original_stop cancels this original
spring setup at its hourly close; it does not invalidate the whole parent range.
This is an explicit project rule, not a fictional active stop order before entry.
After support, the tighter minute child low guard applies. A minute low <= support low cancels this child,
not the entire parent range. Required post-origin1h/4h or child minute price gaps
make the affected path unknown from availability. Stop/invalidation/unknown at
the same clock beat a new trigger. Later missing/invalid evidence never erases
an earlier completed decision. Prefix/cutoff records distinguish pending from
known expired; they do not convert unobserved paths to no-trade zeros.

## Contextual volume and range evidence

At recovery/support availability, independently measure against the immediately
preceding20 expected hourly intervals (never skip a gap to reach farther back):
volume / median(volume), (high-low) / median(high-low), and
close location (close-low)/(high-low). Never mix hourly and minute baselines.
Record all constituent IDs and availability, raw values and ratios. Missing,
negative/nonfinite volume or nonpositive baselines/current spread means unknown
evidence, not zero/neutral. Missing volume alone does not invalidate A/B prices.

Recovery demand:
- supportive: close>open, relative volume>=1, relative spread>=1,
  close location>=2/3;
- adverse: relative volume>1 and close location<=1/3;
- otherwise neutral.

Support reaction supply:
- supportive: relative volume<1, relative spread<1, close location>=1/2;
- adverse: close<open, relative volume>1, relative spread>1,
  close location<=1/3;
- otherwise neutral.

These are named mechanical observations, not proof of institutional intent.
They apply to this recovery/reaction location; they are not universal rules that
heavy volume is good/bad or that low volume is always bullish. Phase remains
unclassified even if both records are supportive.

C uses B's exact first trigger: both evidence records must be known, neither
adverse, and at least one supportive. Neutral+supportive is allowed; neutral+
neutral is no-entry; adverse is a known challenge veto; missing is unknown C.
C never waits for a more favorable second trigger or substitutes another support.
Its raw episode IDs, trigger IDs and decision times are a subset of B's;
policy-specific intent IDs may differ. Occupied fills need not be a subset.
Both intents expire at min(support+60min,T0+7d), exclusive; a timely trigger
can therefore expire before a delayed fill.

## Three isolated policies and fixed execution

A = unchanged old thesis entry, B = revised structure, C = B plus the above
evidence rule. Retain every raw episode in every arm. A-to-B measures the whole
structural policy change, including timing; B-to-C isolates the evidence rule
only in capacity-free paired attribution. Separate occupied books expose the
capacity effect. Do not count repeated scenario fills as independent samples.

Reuse `thesis_execution.replay_book` as a FIXED runtime through an explicit new
adapter. The adapter verifies the new record and marshals the old runtime's
transport schema; it records both new policy seal and legacy runtime policy seal.
It must not claim the marshalled entry was produced by the old compiler. A uses
the original packet unchanged. B/C transport keeps original stop/deadline and
structural invalidation, replaces only the entry directive and known closure,
and removes unused management reviews/Fib. Never write transport over source.

Common original spring stop, $100 intended risk, $50,000 notional cap, fixed2R
target, T0+7d deadline. Primary90s delay,6bp each-side fee,8bp adverse8h funding;
stress180s delay,12bp fee,8bp funding. These are assumptions, not real venue fill
or observed funding evidence. Same-minute stop-first, adverse gap fills, unknown
execution path remains null. No adaptive management or fitted thresholds.

Before B/C admission, recheck at the first runtime-eligible opening:
L < opening < H and H-opening >=2*(opening-stop). Inspect only that opening,
not its future high/low/close. Also cancel when child support low is touched
between trigger and admission. Preserve legacy parent-invalidation/stop guards.
Use completed minute lows before admission and the admission opening only.
Keep the pending intent/reservation until the actual guard cancellation clock;
never remove it retrospectively. Marshal a timestamped guard through the legacy
terminal/unknown transport channel, tagged separately from parent invalidation.
Tests must include another episode arriving during that pending interval.
These are explicit B/C admission rules, not silently applied to A. Return the
reason and decision/admission evidence separately. No room claim at fill without
this check. Missing admission evidence is unknown, not a profitable skipped loss.

## Semantic and engineering acceptance

The test matrix must include local recovery without parent breakout, pivot-box
phase uncertainty, support after earlier failed touch, causal high/low/trigger,
equal pivots, wrong stream/parent/identity, insufficient room, exact clocks,
support failure, parent invalidation, evidence role alternatives, absent volume,
missing prices, future mutations, prefix/rebuild equality, and runtime admission
gap/latency behavior. Hand-check expected timestamps/values independently of the
new implementation. Do not derive expected labels from the classifier being tested.

Natural semantic roster: three lowest raw-ID hashes in each fixed block
[2024-01-01,2024-09-01),[2024-09-01,2025-05-01),[2025-05-01,2026-01-01),
[2026-01-01,2026-08-24), twelve total. Never select by PnL, acceptance or appearance.
Export only source available at the first terminal B source decision; otherwise
the observed child expiry, or applicable recovery/support expiry. Include the
prior20h normalization context and post-origin hourly/4h data plus the child
minute window. Strip old/new policy verdicts, future tails, PnL and developer
history. Cutoff choice uses structural events, not subsequent economic outcomes.
Synthetic contrast fixtures accompany the suite, not the natural roster.

Annotation schema: packet hash/cutoff, source-cited location, phase certainty,
recovery, support, minute anchors, invalidation, room, action and unknown/disputed
status. Reject future/nonexistent citations and unqualified accumulation claims.
Independent reviewers see packets and rubric only. Export/validator completion
does not mean agreement was measured; annotations are a separate evidence stage.

## Chronological economic harness and bounded verification

Use the tested `event_walkforward.split_events` with origin time and maximum
T0+7d label horizon, not realized exits. Fixed blocks above: first block is
development context; next three are chronological evaluation blocks. Purge
boundary-overlap labels, expose exclusions, never randomly split nested cases.
No training or tuning in v1; these are chronological fixed-policy reports, not
proof that exposed history became out-of-sample. Group dependency/overlap counts
must be visible. Keep full raw denominator outside each split's explicit exclusions.

Report counts, known subtotal vs complete net, cost/assumption sensitivity,
winners preserved/missed and losers avoided on matched complete outcomes;
capacity-free sums are not portfolio returns/drawdown. Fewer than50fills or12
distinct origin months is insufficient under the existing research floor; the
floor is not proof of adequate statistical power. No positive edge verdict in v1.

First launch is source/benchmark export and a deterministic synthetic engineering
comparison, not full natural-history economics. One process,600s wall guard,
256MiB aggregate output cap, exclusive new directory, no automatic retry. Save
bindings before launch and verify after; failure record on exceptions. Natural
economics requires a separate source/semantic review receipt bound to the exact
policy, code and source. Full economic runner may be built/tested but not launched
without that review. Runtime, tests, what ran and what remains must be in PROJECT.
