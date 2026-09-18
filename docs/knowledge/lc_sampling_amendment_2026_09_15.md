# LC judgment sampling amendment

User approved this amendment after the source census and before new campaign
agent assessments or economic scoring. Policy: `lc_unassessed_chronological_v1`.

## Rule

Keep the fixed January–July2026 interval and original prior-exposure registry.
Sort native LC candidates chronologically, exclude previously assessed identities
or matched decision times, then take the first30 remaining candidates (or all
remaining candidates when fewer). Do not select by returns, context quality,
model answers or packet readiness. Do not replace difficult or failed cases.

This replaces the original earliest consecutive30/longest contiguous-run rule.
Continuity is not required to evaluate blinded setup judgments. Excluded cases
still interrupt the detector history: never describe the filtered roster as a
continuous live opportunity stream. This amendment followed inspection of counts
and prior-exposure gaps; it was not the original preregistered sampling rule.
Unassessed cases are not necessarily untouched holdout data: Q1 is exposed
development history. Disclose that limitation in the final report.

## Census and consequences

The completed31-month census has142 native LC candidates. The fixed2026 interval
has36;16 match prior-exposure exclusions, leaving20 for the revised roster,
January20 through July26,2026. The previous longest consecutive block had5.
At most20 specialist calls and20 conditional critic calls are permitted for this
roster, within the user's original30-setup/60-call ceiling. No extra calls are
authorized merely because the ceiling has unused capacity.

Use identical filtered opportunities for matched A/B/C comparisons, independent
books and unchanged execution rules. Report these as filtered-sample results,
not full-history engine performance. Broad A/B census results remain separate.
Keep missing/failed assessments in the frozen denominator. This sample is
descriptive evidence, not proof of profitability or statistical independence.

## Implementation boundary and next step

### Preparation progress

The new `JudgmentCampaign` source-only preparation adapter and its tests now
exist in `scripts/research/lc_judgment_campaign.py`. The actual20-case roster
is frozen in `results/lc_consolidated_2026_09_15/judgment_v1/judgment_prepare.json`.
All20 causal evidence packets, source requests, published specialist wrappers
and transport envelopes were generated using the existing evidence builders;
`judgment_v1/evidence/evidence_lock.json` freezes their inputs and outputs.
Reopening the evidence lock verified20 cases. These private artifacts remain
local; GitHub alone does not contain the data or frozen requests.

Root tests:7 sampling/preparation tests passed;54 selector/context/published-job
tests passed (the selector's3 tests overlap these counts). No market roles or
outcome scoring. The legacy source controller remains unchanged. Preparation
does not yet provide a working paid-role orchestration path.

Launch is additionally budget-blocked: latest observed local usage record at
2026-09-16T06:29:05.640Z reports1589.73461 credits, below the existing1900
market-start floor. Recheck current balance before launch; do not silently
lower the floor, shrink the roster or reuse calls. Independent readiness review
accepted the causal preparation. The matching roster, source preparation and
evidence locks are cross-bound in `judgment_v1/preparation_binding.json`, SHA256
`147b4788a99ad7a1844954a4864a7419cac9a35a7dc3300c2c02130d42cf8726`.
Next: close its launch blockers and obtain sufficient verified
budget or an explicit budget revision before the20-specialist/conditional-critic
comparison. Previously frozen evidence must be verified again before use.

### Remaining launch engineering (not an archetype result)

- Add a versioned20-case runtime/capture runner using the existing January19
  pilot lifecycle. Frozen old `CampaignController.prepare()` still selects the
  old block; do not use it for the amended roster.
- Keep one ledger/runtime identity per live controller process. Existing
  `_ledger()` constructs a new `CampaignLedger` every time, so default UUIDs
  change even without restart and invalidate measured S4/S5 timing. A genuine
  restart must still invalidate timing rather than invent continuity.
- Handle the interrupted-dispatch window between ledger reservation and
  controller wrapper persistence, without retrying an uncertain invocation.
- Freeze/validate runtime launch instructions and conditional critic creation,
  then perform a no-model end-to-end transport/timing check. Only then can a
  sufficient-budget launch proceed. No further historical census needed.

`scripts/research/lc_judgment_sampling.py` implements the amended selection rule.
The frozen `run_v2` source controller and its pinned dependencies are unchanged;
its existing `prepare` method still uses the historical consecutive-block rule.
Do not call that preparation path expecting the amended cohort.

Next: bind this selector in a versioned assessment preparation path, preserving
and validating all31 existing source receipts and exposure-registry hashes;
freeze the20-case roster, causal multi-timeframe packets and exact requests;
then run the existing readiness/budget gate before any market roles. No source
replay is required solely to change sampling. All terminal decisions must still
be locked before outcome scoring. No live engine or fusion settings change.
