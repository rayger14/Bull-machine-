# Archetype-isolated research replay — September 13, 2026

## Implemented boundary

User clarified that archetypes must not interfere with each other's tests.
The approved bounded extension now adds `replay_isolated_sleeves` and
`replay_reviewed_isolated_sleeves` to
[`conditional_occupancy.py`](../../scripts/research/conditional_occupancy.py).
The original single-sleeve APIs and frozen past results are unchanged.

Each declared `(archetype, track, variant)` receives a fresh long-only research
book: its own capacity, pending resolutions, uncertainty and accounting. Shared
price data is read-only. Busy skips operate only inside that book. There is no
shared capital pool, combined policy PnL, or combined portfolio interpretation.
The underlying books remain unfunded, with starting equity unspecified.

Silos are declared explicitly, including empty ones. Rows repeat their owning
archetype, track and variant; mismatches and duplicate silo declarations fail
instead of silently moving candidates. Candidate IDs are unique within a silo
and may repeat across independent experiments. Output order is deterministic.
Unavailable plans and incomplete data retain the existing null-result rules.

Reviewed packets must contain `research_identity` with archetype, track,
variant and candidate ID **before** menu compilation and review. The existing
whole-packet binding then prevents reusing an old assessment under a new silo
label. The wrapper does not retrofit identities into old locked packets.
Identity binding is not proof of actual independent roles or teaching fidelity.

### Calling shape

For `replay_isolated_sleeves`, each silo is a dictionary containing
`archetype`, `track`, `variant`, and `candidates`. Each candidate has the previous
candidate fields plus `archetype` and `variant` (`track` already existed).
For `replay_reviewed_isolated_sleeves`, replace `candidates` with `requests`;
each old reviewed-request object also carries the three silo identity fields.
Both functions take read-only minute OHLC bars and an explicit `as_of` clock.

The wrapper supports named archetypes without treating every name as a verified
strategy. It does not translate all seventeen native plans, add short execution,
generate candidates, or implement native cooldown state. Caller-side source
collection and any entry-dependent cooldown logic require separate verification.
Different cost/notional conventions are allowed between independent books;
matched comparisons must freeze equal assumptions separately.

## Candidate-selection audit

Independent quant source tracing, checked against the local files, found:

- The current hourly LC research selector already ignores final native
  `selected` status. It uses LC structural pass and explicit current/prior
  numeric criteria: local
  `results/agent_layered_entry_2026_09_11/prepare_sources.py:75`.
- The September 12 preparer calls that selector at
  `results/evidence_id_pilot_2026_09_12/prepare_sources.py:388`; it records
  `selected` only as descriptive `native_emitted` metadata. No repair to that
  frozen selector was needed or made.
- The native engine applies relative-score filtering and deduplication after
  detection in
  [`isolated_archetype_engine.py`](../../engine/integrations/isolated_archetype_engine.py).
  Its final signal list is therefore not the complete per-archetype population.
- The existing observer in
  [`engine_signal_replay.py`](../../scripts/research/engine_signal_replay.py)
  captures each `detect()` return as `native_signal` before that cross-archetype
  filtering, alongside structure/gates/fusion and before/after cooldown state.
  Those diagnostics are the source hook for a new pre-winner export.

Native pre-winner signals still reflect their own native gates and cooldowns.
The broader LC structural/numeric research cohort is a different candidate
definition; do not interchange them or infer causal isolation of every upstream
feature/detector merely from the new book tests.

## Verification

Twenty-one new behavioral tests first failed because the explicit silo API was
absent, then passed with the implementation. Together with the original
occupancy tests, **46 focused tests pass**. Full research verification:
**681 passed**, one existing urllib3/LibreSSL warning, in 12.29 seconds.

The independent quant reviewer approved the implementation and independently
reran all 46 focused tests, finding no correctness blocker. Source follow-up
found no post-winner cooldown update in `get_signals()`; native detection arms
its own cooldown before dedup, while signal-only replay neither allocates nor
calls entry-driven `arm_cooldown`. No shared feature mutation was found in the
inspected detection path. This is source inspection, not an all-seventeen
historical non-interference experiment.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research
```

Tests cover cross-archetype non-interference; timeframe and variant isolation;
same-silo busy skips; local uncertainty, unavailable plans, size and costs;
empty declarations; input/output aliasing; bad routing; actual reviewed-plan
gates and rejected cross-silo relabeling. Choices/reviews are synthetic fixtures,
not new market assessments.

| Artifact | SHA256 |
|---|---|
| Updated occupancy module | `6d86a3f74355ebcfcff0c352c504486f2e490f72f2505bee0982e2b50efe681b` |
| New silo tests | `86a45f6d05c9bc5db91a7a7c8b2dc12855eb6120f309c6566bb263d45bb26c2d` |

## Next concrete work

Produce a new source-only full-candidate artifact without overwriting the old
pilot. Reconcile the existing April LC/minute cohort against all 245 retained
IDs/classifications, and persist full causal evidence rather than only three
selected records. For all-archetype expansion, retain every timestamp's
per-archetype diagnostic and name the candidate definition; never start from
only the final cross-archetype winners. The 245 IDs are not all seventeen
archetype populations.

Then freeze matched code-immediate, code-confirmation and agent comparisons
inside each silo, with decisions/reviews locked before future outcomes. Keep
hourly and minute tracks equally important and separate. Previously exposed
April cases remain development evidence, not a fresh holdout.

No new historical performance run, model market calls, tuning, dependency,
production/live/fusion/archetype changes, push or PR occurred in this unit.
Full walk-forward/CPCV evaluation and demonstrated incremental agent value
remain outstanding.
