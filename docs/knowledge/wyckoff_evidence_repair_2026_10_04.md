# Wyckoff evidence-integrity repair — October 4, 2026

## Outcome

The first bounded repair from the M1/M2 audit is implemented and independently
reviewed in the local research checkout. **158 focused tests pass**, including
96 new regression cases. This corrects evidence handling; it does not establish
accurate recognition of complete Wyckoff schematics or a profitable strategy.

Branch `quant/archetype-evidence-audit`; starting and current HEAD
`85923a4501153271b2e2e1d1437dc52ec4cb778e`. Changes remain uncommitted/unpushed.
Quant advice resolved interpretation questions; controller implemented test-first;
one fresh independent software reviewer found no corrective change required.

The [original audit](wyckoff_m1_m2_code_audit_2026_10_04.md) remains historical,
unchanged. The [design](../superpowers/specs/2026-10-04-wyckoff-evidence-integrity-design.md)
and [plan](../superpowers/plans/2026-10-04-wyckoff-evidence-integrity.md) define the
bounded contract. Completion is recorded here and in the local progress ledger,
not inferred from unchecked implementation-plan bullets.

## What changed

| Defect | Repair | Evidence |
|---|---|---|
| Delayed spring/upthrust validated the confirmation candle instead of the original sweep | Immutable candidate price, prior boundary, timestamps and parent-generation snapshot; confirmation must retain that parent | Actual raw A/B/UT recognizers, lost/replaced/expired parents, clock gaps and prefix invariance |
| A rejected spring could still advance phase C | Phase change requires an accepted spring in accumulation context | Rejected spring leaves phase unchanged; valid independent events still work |
| Opposite-direction or unavailable evidence could enter generic fallback scoring | Directional key presence blocks generic/binary escape; per-timeframe status/source gating | Long/short, explicit zero, NaN, proxy and mixed-availability cases; same-direction weights pinned |
| Partial or gapped higher-timeframe candles could appear confirmed | Wyckoff-only preparation requires completed, aligned, unique, valid constituents and a contiguous history suffix | Literal aggregates; missing/duplicate/corrupt/open bars; stale latest bin; daily source seam |
| Repeated warmup/poll could append the same hour and advance histories again | First identical warmup-tail poll computes without appending; subsequent identical polls return a copied cached vector; conflicts/older polls reject | Actual offline feature-computer buffer, funding history and return-value isolation checks |

Modified source: `engine/wyckoff/events.py`,
`engine/archetypes/archetype_instance.py`, `bin/live/live_feature_computer.py`.
New source: `engine/wyckoff/candle_integrity.py`.
New tests: `tests/test_wyckoff_evidence_integrity.py` (26),
`tests/test_wyckoff_directional_evidence.py` (38),
`tests/test_wyckoff_candle_integrity.py` (32).

## Contract details and limits

- Parent identity is a sequence-local generation, including the no-parent epoch.
  Replacing a range with identical levels is still a different generation.
  A candidate born without a parent cannot acquire one retrospectively.
- Legacy direct `process_bar` callers may deliberately omit metadata; the real
  batch adapter always supplies a mapping and rejects absent/invalid provenance.
  Provenance appears on the confirmation row only, never backfilled onto the
  candidate. Retrying batch validation overwrites, rather than duplicates, columns.
- Legacy generic-only scoring remains a guarded compatibility path. Explicit
  directional schema, including all-zero or malformed values, cannot escape into
  generic scores. Unavailable Wyckoff contribution is zero, not a whole-trade veto
  and not proof of neutral market structure. Existing weights are not retuned.
- `wyckoff_evidence_status`, reason, source, last-input-close and available-at
  are exposed per timeframe, with `tf4h_`/`tf1d_` prefixes. Hourly, 4H and daily
  detector failures are isolated. Missing provenance survives numeric NaN filling
  as null. EMA alignment has its own `wyckoff_ema_alignment_proxy`; it is not
  presented as a detected Wyckoff score.
- Native daily observations have distinct provenance. They may supply older
  history before the first fully observable UTC day of the original hourly
  coverage. They cannot patch an internal invalid/missing hourly-derived day.
  A disconnected seam breaks history. Native daily validation does not assert
  that its 24 underlying hourly constituents were independently inspected.
- The explicit preparation cutoff is causal. In the live adapter it is derived
  from the supplied hourly start plus one hour: the caller promises a completed
  hour. This is **not an independent wall-clock freshness check**.
- Generic resampling for other feature families and the runner's decision-dedup
  guard are unchanged. Daily ingestion now preserves invalid duplicate/alignment
  evidence for validation instead of silently normalizing/deduplicating it.
- Duplicate feature results are frozen for the already processed candle; later
  external-feed updates do not recompute that candle. Warmup timestamps must be
  unique and ordered. This does not implement transaction rollback after an
  unrelated mid-computation exception.

## Verification

Final controller command:

```sh
python3 -m pytest -o addopts='' -q \
  tests/test_wyckoff_evidence_integrity.py \
  tests/test_wyckoff_directional_evidence.py \
  tests/test_wyckoff_candle_integrity.py \
  tests/test_wyckoff_m2_sequence.py tests/test_wyckoff_causality.py \
  tests/test_wyckoff_mtf.py tests/test_wyckoff_v2_climax.py \
  tests/research/test_live_feature_replay.py \
  tests/research/test_live_feature_replay_report.py \
  tests/test_liquidity_score_ordering.py tests/unit/test_wyckoff_mtf.py --tb=short
```

Result: **158 passed, 290 warnings, 11.22 seconds, exit 0**. Warnings include
existing pandas deprecations and local LibreSSL compatibility, plus numeric-fill
deprecation exercised by the new null-provenance test. No claim of warning-free
or repository-wide certification.

Red-to-green evidence: delayed-event witnesses initially23failed/1passed;
directional witnesses31failed/7passed; missing candle preparer22failed; actual
live-feature integration10failed/22passed. A revalidation duplicate-column
regression was also reproduced and fixed. The independent reviewer separately
ran all96new cases (96passed/36warnings/3.64s), checked a malformed native-daily
source and disconnected source seam under `deny_network`, and approved local
acceptance. No reviewer edits or nested reviews.

Expanded baseline rerun: **53 passed, 8 failed, 291 warnings, 3.35s**, matching
the pre-change53pass/8fail baseline. Unchanged failures:

- `tests/test_archetype_instance.py`: invalid-direction validation expectation;
  Wyckoff-heavy fusion range expectation; liquidity-heavy fusion range expectation.
- `tests/test_wyckoff_events.py`: basic SC, basic BC, AR-after-SC, basic ST and
  spring-A fake-breakdown fixtures.

These contain stale/inconsistent expectations/fixtures noted in the audit; they
were not weakened to hide failures. The raw-event fixtures still need separate
source-adjudicated replacement or correction, not threshold changes for green.

Fresh full-repository `python3 -m pytest --tb=short`: **exit3**, five collection
errors reported,9.70s; missing `configs/baseline_wyckoff_test.json` triggers
`SystemExit` in `tests/test_integration_fixes.py`. No full-suite pass or merge
readiness claim. `git diff --check` passes. No changes to selected configs,
`coinbase_runner.py`, existing research scripts/tests or their frozen specs.

## Preservation and provenance

Rehashed all112 file bindings and10 economic artifacts in the October3
support-reaction economic receipt: all match. Existing receipt hashes also match:

- Economics: `e6ab5ec785e0c1d1c9ca4e099a5d476f80b40b1a866a78a7f85d6a79a4598a8a`
- Source: `5c261f850fc125d7523a92c9e319594ef259d4cd3d4cc607236cb10fac28d71b`
- Semantic: `d998fae5d939a897d1ded8c214205c402c75d1cd20657a4ba4aa7dd6d5885b6e`

This is preservation of those experiments, not a claim that old receipts certify
the newly changed engine. Do not silently rebind old experiments to new source.

Current SHA256:

```text
50c2bc04569f32b7fa62454d063f57785ce93334f8872ec556d205e944836fb7  engine/wyckoff/events.py
3c67abc571d9df4454cb87f658f396381f7b2db2648f2e8ff9502ddf3abd3487  engine/wyckoff/candle_integrity.py
7c3820aaf1d53997a92bc305ec99000791d680292b48342a8f6bd313f2e1d0db  engine/archetypes/archetype_instance.py
1cd1463285d3c1155aeb3a842a8904ca57f5d9af17bb4d048def4345dc74a41c  bin/live/live_feature_computer.py
```

## What remains and the next deliverable

Full M2 stays off; hourly context-only M2 stays at its existing setting. V2 is not
promoted. Volume z-score ratio semantics, UT/UTAD diversity double counting,
phase interpretation, historical max carry-forward and true nested parent/child
lineage remain outside this batch. Compatibility M1/M2 daily flags still mean
bullish/bearish scores, not complete schematic recognition. None of the17 native
archetypes is newly certified; Fibonacci/Gann and adaptive management are not
tested here. These fixes do not attribute past live losses or imply a future edge.

Quant-advised next deliverable, **not yet built**: one frozen12-case raw-candle
semantic acceptance matrix, ending at current consumer decisions rather than
returns. Independently adjudicate source-supported expectations before executing
detectors. Include four positive structures (spring/no-spring accumulation and
upthrust/no-upthrust distribution), four near-misses (missing established range,
missing recovery, failed test, invalidated parent) and four ambiguous structures
(climax versus continuation, unresolved range, local versus range-escaping
strength/weakness, conflicting timeframe evidence).

Each case needs valid OHLCV, parent boundaries, candidate/confirmation times,
prefix-available evidence, permissible phase claims/uncertainty, and a trace from
raw recognition through sequencing/features to directional scoring and configured
phase sizing. Keep settings frozen. Separate source-supported structure, raw
recognition and configured suppression: M2-disabled behavior can pass config
fidelity but cannot count as full-M2 recognition. Name every missed positive and
unsupported distinction; no fixture replacement or threshold search to improve
the score. Material false confirmation blocks P&L progression. Missing M2 support
requires a separately specified implementation decision.

No study, deployment, live orders, config tuning, installation, paid external
model experiment, commit, push or PR occurred. Quant/review agents used session
usage. Existing data archives and prior study outputs remain local dependencies;
this synthetic repair suite requires the existing Python environment, not new
market downloads. No repair jobs remain running. Another CLI reads PROJECT first.
