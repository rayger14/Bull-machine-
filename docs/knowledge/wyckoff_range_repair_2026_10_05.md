# Wyckoff range and retest repair

## Result and limits

October 5, 2026. The bounded repair is implemented locally: the initial reaction
can develop before its range boundary is locked, post-escape tests reference the
broken level, and the existing phase-C sizing increase requires current supporting
evidence. Independent software review accepted the corrected implementation for
offline regression. This is not a complete Wyckoff classifier or deployment approval.

The unchanged twelve-case regression now identifies the intended initial ranges,
but **all four positive examples still miss complete recognition**. Three expose
a limitation of the approved contract: it mistakes the first quiet pullback candle
for a completed test, then rejects the continuing pullback. The fourth still lacks
qualified raw strength. No thresholds were adjusted to rescue these exposed cases.

Across the 48 recorded checkpoints, the new guard removes 18 conditional long
size increases from 1.25 to 1.0. Event flags and directional scores are unchanged
at all recorded native timeframes/checkpoints in this particular regression.
That does not make the repair economically neutral: its sizing behavior changes,
and its new anchored events can affect scores on other inputs. No actual orders,
portfolio allocations, returns or trading edge were measured.

## Contract and source basis

The primary educational reference distinguishes initial reaction boundaries,
subsequent tests, strength and pullbacks to former resistance; relative volume
and spread are meaningful evidence. It does not establish the project's exact
percentage tolerance, time window, turn-confirmation rule or sizing multiplier.
[Wyckoff Analytics methodology](https://www.wyckoffanalytics.com/wyckoff-method/)
supports the conceptual roles, not universal numeric rules. Existing archived
source work remains in `reports/Wyckoff recognition edge cases.md`.

The delegated quant approved these conservative operational definitions before
implementation:

- First AR/AS starts a reaction leg. Subsequent highs/lows update its extreme.
  A later close through the opposite extreme of that extreme's candle locks the
  initial bounds. An extending candle cannot confirm its own turn; equal extremes
  retain the original anchor. The existing reaction horizon bounds confirmation.
- Locked bounds mean `reaction_complete`, not confirmed accumulation/distribution.
  A quiet, narrower test near the climax-side boundary followed by a held recovery
  supplies `tested` evidence. Actual traded volume and candle spread are compared;
  existing volume-z rules are not silently redefined.
- Validated SOS/SOW inside the range remains local evidence. An escape is a close
  beyond previously locked bounds, with a separate role for price-only crossing
  and crossing supported by recent same-parent strength/weakness.
- Post-escape LPS/LPSY references an immutable broken boundary and strength candle.
  It requires a later retreat near that boundary, lower volume and spread, and a
  further close through the test candle's recovery extreme while its adverse
  extreme holds. The existing 3% proximity convention is retained.
- Known escape episodes suppress legacy generic LPS/LPSY, confidence and the
  resulting D transition until that anchored test confirms. Failed/expired tests
  cannot fall back to local labels in the same episode. Inside-range legacy
  observations remain explicitly `local_unverified`, not certified entry setups.
- Candidate anchors/deadlines cannot move. Missing chronology, gaps, boundary
  loss, failed hold and parent replacement remove authority. Delayed spring/UT
  justification checks the original candidate and every intervening candle.
- A phase-C boost requires a confirmed spring test tied to the current parent,
  locked bounds and available observation. Phase strings, domain scores and
  context-only M2 labels alone cannot authorize it. Unsupported no-spring roles
  fail closed. Full M2 remains off; the 1.25 multiplier/config are unchanged.

`engine/wyckoff/range_evidence.py` holds this evidence lifecycle, integrated into
`engine/wyckoff/events.py`. The adapter exports scalar JSON plus parent identity
and preserves original raw flags/confidences for validation replay. Confirmation
never backfills the candidate row. New anchored retest confidence inherits the
associated strength confidence; it is not a calibrated probability.

`bin/live/v11_shadow_runner.py` forwards fresh evidence and applies the narrow
guard to Boost 6. Missing features overwrite stale metadata. Other sizing boosts
were not audited or changed. Only local source changed; the running live engine
was not deployed, restarted or reconfigured.

## Unchanged exam findings

| Case | Improvement | Remaining gap |
|---|---|---|
| W01 spring accumulation | AR grows to 110 at row 586 and locks at 589; spring/SOS associate with the intended range; qualified escape at 654 | First retest candidate 657 fails at 658; later multi-bar pullback is not recognized as a new completed test |
| W02 no spring accumulation | Same corrected range; geometric escape is explicitly not qualified strength | Raw SOS still absent, so no anchored LPS authority; no supported no-spring sizing role |
| W03 upthrust distribution | AS reaches 100 at row 586 and locks at 589; upthrust/SOW associate with the intended range; qualified escape at 654 | Mirrored first-candle retest cancellation at 658; LPSY absent |
| W04 no upthrust distribution | Corrected range and SOW association; qualified escape at 654 | Same multi-bar LPSY limitation |

W01's candidate at 657 has low 111.35, undercut by 111.05 at 658. Price continues toward
support 110, reaches 110.2 at 660, then recovers. W03/W04 mirror this: candidate high
98.65 becomes 98.95 at 658, then 99.8 near resistance 100 at 660. The implementation
honors the approved single-candle contract; that contract does not represent the
whole developing pullback. Do not fix this by enlarging tolerances, restarting
deadlines indefinitely or changing the frozen labels until these examples pass.

W01's spring at 644 receives a quieter test at 648 and confirmed recovery at 649. Its phase
justification is active at 649–651, then cancelled by strength/phase advance and
later parent escape. Those three rows lie between recorded sizing checkpoints;
all recorded sizing probes are 1.0, but that is not evidence the guard always rejects.
Fresh engineering tests verify the same actual sizing branch allows 1.25 when
current evidence supports it, and disallows it when stale, missing or failed.

W05–W12 introduce no new frozen-grader prohibited claim. They are not eight full
semantic passes. W07/W08 retain score-parent lineage gaps. W11 now exposes the
locked range and absence of an escape explicitly; its frozen grader's static
`unrepresented_distinctions` list was not rewritten to certify the new schema.
W12 still lacks an actual nested parent/child structure model. Initial locked
ranges do not expand into newly established larger ranges in this repair.

There is also a status limitation: `tested` currently means active test authority
and reverts after expiry. The retained test record preserves its historical
confirmation, but the top-level label mixes historical testing with current
authority. A later contract should separate those concepts; neither historical
testing nor a surviving phase label grants indefinite sizing authority.

## Verification and review

`tests/test_wyckoff_range_anchors.py` contains 77 fresh engineering tests, not 77
trades. The initial 47-test batch failed before implementation. Additional red
tests reproduced missing justification, intermediate delayed-origin breaches,
stale runner phase metadata and clock gaps before legacy retest transitions.
These failures were corrected before acceptance.

Two fresh mirrored raw-OHLCV examples compute their own event flags and volume
z-scores, exercise developing/locked/tested ranges, qualified escape and later
anchored retest, and compare seven exact prefixes each. Reviewer-supplied default
spring-detector witnesses preserve valid delayed recovery but reject a breached
original extreme, with five exact-prefix comparisons per variant. The controller
independently reproduced and retained both witnesses.

The independent review's final range/exam selection passed 100 tests with 97 warnings.
It additionally verified ten actual-detector prefix comparisons. The later
observer-only trace addition passed its targeted test and separate review;
`deepcopy` prevents future mutations from changing earlier records.

Final controller selection is the command in the preceding climax repair report
with `tests/test_wyckoff_range_anchors.py` added: **317 passed, 445 warnings,
18.99 seconds, exit 0**. Warnings remain existing pandas downcasting/deprecation
and urllib3/LibreSSL warnings; this is not a warning-free repository.

The old two-module selection remains 24 passed / 8 failed:

- `tests/test_wyckoff_events.py`: `test_sc_basic_detection`, `test_bc_basic_detection`,
  `test_ar_after_sc`, `test_st_basic_detection`, `test_spring_a_fake_breakdown`.
- `tests/test_archetype_instance.py`: `test_validation_invalid_direction`,
  `test_fusion_calculation_wyckoff_heavy`, `test_fusion_calculation_liquidity_heavy`.

Fresh bare `python3 -m pytest` exits 3 with five collection errors, stopping at
missing `configs/baseline_wyckoff_test.json` and `tests/test_integration_fixes.py:46`
collection-time `SystemExit` (10.00s). Those unrelated tests were not weakened.
`git diff --check` passes. There is no full-suite or merge-readiness certification.

## Reproducibility

One separately sealed, unchanged-input regression completed in 78.6263 seconds and
wrote 56,475,835 bytes before its receipt, within 600 seconds / 100 MiB. It ran 48 decision
prefixes and 24 witness reruns. All 233 current source bindings, 14 artifact hashes and
12 equal prefix/future-tail witness triples were verified. This exposed synthetic
exam is development regression evidence, not a new holdout or market backtest.

Base directory: `results/wyckoff_recognition_2026_10_05/`.

- Inputs remain `input_v2/packet.json` and `review_v1/approval.json`.
- New seal: `range_repair_seal_v1.json`.
- New output: `exam_range_repair_v1/`.
- Old `exam_v1` and `exam_climax_repair_v1` receipts and all 14 artifacts in each
  were reverified unchanged. Packet, annotations and grading ledger are unchanged.
- Among old bound files, only events, the local runner and research harness changed;
  the new range-evidence module adds one binding. Harness changes forward evidence
  to the actual sizing branch and record per-row structural evidence. Its explicit
  AST fingerprint was updated for the new guard; old seals cannot certify new code.

```text
range_evidence.py SHA256
0570ceb076a58ca0096b335d9c8c979f3de033a1f912925a3ca2e17486eb671a
events.py SHA256
aac9b761564bf065c9fdd1849adda83a43505e928f86a07eb921a1af4a1adaf8
v11_shadow_runner.py SHA256
3d10e3e5e8a4498739eff95c1c8b0f01e5538bc591d26795ecfcf311fcbb28f5
research harness SHA256
87fe8076e7d52597142d08145150b0656deb1f8aaa303713ee65f2f603983d5b
new 77-case test module SHA256
c77091310171c73b9722b4815dddd92ee6fdb4b3cc1259b5b75a0f897773b75f
seal SHA256
455bf65d3197f106237110638e7bc99f5c245d8bba0f7521f7b88f5d09043b11
receipt SHA256
d160fbf7b7083c1984196f79432dcbdd481edd89d50fe3223486e50f1fbb33c0
```

Existing research branch/dirty work are preserved. No installs, external paid
assessment campaign, fine-tuning, live orders/config changes, deployment,
commit/push/PR occurred. Session quant and software agents use ordinary session
usage. Work and artifacts remain local/uncommitted; the synthetic run uses existing
packages and the local regime model, not private market archives. Older market
studies still require their separately documented local data.

## Next deliverable

Freeze one independently source-reviewed **pullback-episode contract and fresh
mirrored contrast packet**. Distinguish approach, evolving unconfirmed pullback,
confirmed test and subsequent failure. Specify evidence aggregation, causal turn
confirmation, immutable episode deadline and genuine boundary invalidation.
Include gradual pullbacks, one-bar tests, adverse expansion, boundary loss, expiry
and failure after confirmation. Separate historical testing from active authority.

Only after those definitions are fixed should the episode logic replace the
single-candle approximation. Do not tune it on the twelve exposed examples. W02's
gradual-strength/no-spring gap remains a separate subsequent decision. Entry,
minute timing, management, costs and untouched chronological economic validation
come after recognition acceptance; this repair authorizes none of them by itself.
