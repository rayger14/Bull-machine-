# Wyckoff opposing climax repair

## Result and deployment restriction

October 5, 2026. The bounded parent-lifecycle repair is implemented and verified
locally. An opposing climax clue no longer immediately erases an established
parent or suppresses eligible strength/weakness evidence. A later confirming
reaction can still establish the opposing structure.

**Full recognition remains incomplete, and this change is not economically
neutral or approved for deployment.** In the unchanged 12-case exam, all four
positive examples now retain their existing parent at the previously erroneous
reset, and two recover their missing weakness event. All four still miss required
later milestones. These are synthetic concept tests, not trades or a P&L backtest.

A material downstream effect must not be overlooked: W02 retains its earlier
context-only `C_accum` label. At prefixes 657 and 666, the existing conditional long
size multiplier changes from 1.0 to 1.25 although its Wyckoff long-domain score is 0.
The sizing configuration was not edited; preserving its input phase changes its
behavior. This does not prove an entry would occur or that the size is necessarily
wrong. It means the phase's continued justification and economic effect are not
verified. The quant advises offline acceptance only.

## What changed

Only `engine/wyckoff/events.py` changed among the prior exam's 232 bound source,
configuration and model files. There is one new test module and one intentional
adjustment to an existing regression's now-superseded replacement expectation.
Existing dirty work in other files was preserved.

The quant-approved operational contract is:

- An established accumulation receiving BC, or distribution receiving SC,
  retains an immutable pending candidate with its original candle, volume z-score,
  confidence, time and parent identity. A pending clue is not a scored climax.
- A later raw AS and close below the BC candle's low, or raw AR and close above
  the SC candle's high, can confirm within the existing reaction window while
  the same parent survives. This is an initial climax/reaction structure, not
  proof of a mature distribution/accumulation schematic.
- Confirmation uses the original candidate plus current reaction geometry and
  restores candidate confidence only at confirmation availability. It never
  backfills earlier flags or cascades further milestones through the new parent
  on that bar.
- Continuation closes cancel the candidate; expiry, parent loss and same-side
  resets clear it. Repeated clues do not move its anchor or extend its clock.
  Simultaneous SC+BC remains ambiguous instead of choosing a side by branch order.
- Raw inputs and lifecycle records remain separate from validated scoring.
  The exported lifecycle journal is scalar JSON for compatibility with feature
  consumers. Malformed datetime evidence cannot use the explicitly retained
  legacy bar-index-only compatibility path.

This is a conservative project rule, not a universal definition supplied by a
trader. It can miss gradual reversals. No thresholds were optimized, and the
existing volume-z ratios, first-reaction range boundaries, M2 configuration,
raw LPS/LPSY definitions and sizing policy were not changed.

## Unchanged exam comparison

The original packet, independent annotations, grading rules and harness were
preserved. The old exposed cases are regression evidence, not a fresh holdout.
At candle 652, the original parent generation 1 now survives in each positive case.

| Case | Previous behavior | Repaired behavior | Still missing |
|---|---|---|---|
| W01 spring accumulation | SOS and BC emitted together; distribution replaced accumulation | SOS survives with accumulation parent | LPS and association with the intended mature range |
| W02 no spring accumulation | BC erased accumulation | Accumulation remains; raw SOS still absent | SOS and LPS; persistent phase and sizing need review |
| W03 upthrust distribution | SC replaced distribution before SOW validation | SOW survives with distribution parent | LPSY and association with the intended mature range |
| W04 no upthrust distribution | SC replaced distribution before SOW validation | SOW survives with distribution parent | LPSY and range association |

W05–W11 have identical observed raw/validated events, parent/range/phase traces
and domain scores across their recorded timeframes and checkpoints. W12 also
retains its local accumulation instead of resetting, but still lacks explicit
parent/child structure lineage. None of the eight nonpositive cases introduces
a newly flagged prohibited setup under the frozen grader; that is not eight
complete recognition passes. W07/W08 retain their prior score-lineage gaps.

All 48 checkpoint comparisons show only the two W02 conditional sizing changes
described above. No actual entry eligibility, final allocation, execution,
portfolio returns, minute-scale timing or certification of all 17 archetypes was tested.

## Engineering verification and review

The new module `tests/test_wyckoff_opposing_climax.py` contains 48 tests, including
mirrored raw-OHLCV detector-to-state witnesses supplied by the reviewer and
independently reproduced by the controller. Those witnesses compute their own
volume z-scores and event flags; they demonstrate both continuation protection
and later replacement, with exact prefix agreement. They are not market samples.

Initial new regression run: 36 failed / 2 passed before implementation. Gap protection
then reproduced two failures. The actual feature NaN handler reproduced the
multi-record array-truth failure; scalar JSON corrected it. The reviewer reproduced
two mirrored missing-timestamp false confirmations; an explicit clock requirement
corrected them. One prior spring/climax test now uses independently invalid spring
evidence and expects a pending BC instead of immediate parent replacement.

Final controller command:

```sh
python3 -m pytest -o addopts='' -q \
  tests/test_wyckoff_opposing_climax.py \
  tests/test_wyckoff_evidence_integrity.py \
  tests/test_wyckoff_directional_evidence.py \
  tests/test_wyckoff_candle_integrity.py \
  tests/test_wyckoff_m2_sequence.py tests/test_wyckoff_causality.py \
  tests/test_wyckoff_mtf.py tests/test_wyckoff_v2_climax.py \
  tests/research/test_live_feature_replay.py \
  tests/research/test_live_feature_replay_report.py \
  tests/test_liquidity_score_ordering.py tests/unit/test_wyckoff_mtf.py \
  tests/research/test_wyckoff_recognition_cases.py \
  tests/research/test_wyckoff_recognition_exam.py --tb=short
```

Result: **240 passed, 371 warnings, 30.48 seconds, exit 0**. Fresh independent software review
accepted the corrected source for offline regression with no outstanding findings;
its focused selection had 176 passes before the two raw witnesses were retained as tests.

The separate old raw-event/archetype test subset remains 24 passed / 8 failed: five
known raw-detector fixture failures and three archetype expectation failures.
They were not weakened. Fresh bare full-repository pytest still exits 3 with five
collection errors, including missing `configs/baseline_wyckoff_test.json` and
collection-time `SystemExit` in `tests/test_integration_fixes.py`. Final attempt
took 12.24 seconds. No full-suite pass or merge-readiness claim. `git diff --check` passes.

## Frozen artifacts and reproducibility

The single 600-second/100-MiB-capped rerun completed in 126.0084 seconds and wrote
41,404,778 bytes before its receipt. It evaluated 48 decision prefixes and 24 witness
reruns. The controller verified 232 current source bindings, 14 artifact hashes and
all 12 equal prefix/future-tail witness triples. These sampled witnesses are not
proof of universal causality.

Inputs remain under `results/wyckoff_recognition_2026_10_05/input_v2/packet.json`
and `review_v1/approval.json`. New seal is `climax_repair_seal_v1.json`; new output
is `exam_climax_repair_v1` in the same base directory. All original packet/review/
seal/receipt hashes and 14 old artifacts were independently reverified unchanged.
Old source-bound receipts do not certify repaired code.

```text
events.py SHA256:
7dc1bade51324668950e4e84db42dadaf4de468859adae2cc0d1b1b54883fe74
new test module SHA256:
bd2e384eaf49d34972b9bbf69d6c3f6d5ae920e1b11b37153df56bdc6982409a
new seal SHA256:
d8a1e0a6bd99c46eff6b356730bdf9f8ac941cd5c825b399e6bec5b16175cfc7
new receipt SHA256:
21c2b68a88b95d6677ac80ea8d657b51cd2e4879d338e730e0cc4513eac4206d
```

Use a new seal/output path for any later rerun; never overwrite the old results.
No experiment or agent remains running at this handoff. No installs, external paid
assessment campaign, fine-tuning, live orders, config changes, deployment, commits,
push or PR occurred. Session quant/review agents consume normal session usage.
Work remains local/uncommitted on `quant/archetype-evidence-audit` at existing HEAD
`85923a4501153271b2e2e1d1437dc52ec4cb778e`. The synthetic exam uses existing packages
and local source/model artifacts, not private price archives; older market studies
still require their separate local-only dependencies.

## Next bounded concept repair

The quant recommends **range identity and retest anchoring**, not threshold tuning
on these examples. Freeze source semantics and fresh positive/near-miss cases first:

1. Distinguish an initial reaction boundary from a subsequently established range.
2. Associate strength/weakness with that identified range and distinguish a local
   recovery from an escape of its parent range.
3. Associate post-escape LPS/LPSY with the relevant broken boundary and subsequent
   test, rather than merely proximity to an old rolling 30-bar extreme.
4. Include phase-shadow persistence and conditional sizing in consumer acceptance
   checks so a recognition repair cannot silently be called economically neutral.

This next batch is not implemented. No M2 activation or silent sizing-policy change
belongs in the completed repair. Economic progression remains blocked by incomplete
recognition and unresolved consumer semantics.
