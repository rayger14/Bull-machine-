# LC structural-target outcome replay

September 29, 2026. Offline software extension; no historical strategy result.

## What finished

The user's approval to proceed with outcome checking advances the completed
proposal/pre-entry contract to a **single-case structural bracket scorer**:

- `scripts/research/lc_structure_outcome.py`
- `tests/research/test_lc_structure_outcome.py`

The old fixed-2R scorer, all frozen experiments, all17 archetypes, live engine,
configuration and fusion remain unchanged. The bounded extension follows the
previously explained replay design; it does not create another campaign controller.
Graphify's existing lessons identify this recent LC pipeline as incomplete graph
coverage; actual source files were used rather than rebuilding the graph.

The API is:

```python
score_structure_outcome(source_request, packet, raw_response, policy, execution, future)
```

The first five arguments are the existing structure-contract/pre-entry inputs.
`future` contains exactly `instrument`, `data_stream_id`, and `minutes`. Its
minute records start at the proposed fill, with timezone-aware `open_time` and
OHLCV. A resolved open needs only `open_time`/`open`. A caller with a full minute
archive must slice from that fill first; the function never sorts or fills gaps.
No future data is added to the assessor packet.

## Frozen mechanics of this version

1. Revalidate original source, packet, proposal and external policy. Re-run
   pre-entry timing, confirmation, expiry, cancellation and room checks. Never
   trust a caller-supplied passing grade or silently choose a later entry.
2. Use the pre-entry checker's rounded structural stop and destination, **not
   entry plus2R**. Both LC theses retain their labels; no pooled strategy inference.
3. Use a holding deadline of **fill time + policy.horizon_minutes**. This differs
   explicitly from the old conditional scorer's decision-relative deadline.
   Any future like-for-like structural comparator must use the same new semantics.
4. Consume uninterrupted one-minute observations only until resolution. Missing
   data before resolution is unknown, not zero. Corruption or missing history
   after a resolved exit cannot erase that known outcome.
5. Open gaps are known before intrabar extremes: stop gaps exit at the worse open;
   target gaps fill at the target without positive price improvement. If neither
   open barrier fires, the deadline exits at its open. Never read deadline HLC.
6. Otherwise first barrier touch exits. A candle touching both is stop-first and
   explicitly flagged ambiguous. An intrabar exact exit timestamp remains null;
   report its bar open and when the completed candle reveals the outcome.
7. Size is the existing **fractional theoretical quantity upper bound**, constrained
   by modeled risk including costs, notional and equity/leverage. It is not a
   lot-rounded executable size. Flat frozen round-trip cost is charged once on
   entry notional; no unprovided funding/slippage assumptions are invented.
8. Report gross/net PnL, modeled costs, modeled loss at the stop and netR. Stop
   gaps may exceed the stated risk budget; that budget is not a guaranteed loss cap.

**Deliberate unsupported boundary:** structural invalidation must equal the
effective protective stop for this first exit adapter. Otherwise return
`not_scored/separate_postentry_invalidation` with null PnL. This also covers a
raw invalidation that no longer equals the stop after tick rounding. The earlier
proposal contract permits different levels, but has not defined how to manage
them after entry. Silently ignoring either level would test an invented policy.

All outputs retain `execution_authorized: false`. Nonentry, rejected, cancelled,
invalid or unsupported proposals remain **unscored/null** in this single-case
adapter. A later book-accounting protocol must separately distinguish legitimate
zero-exposure decisions from failures; this module awards no avoided-loss credit.
Source reconstruction faults raise a controller error rather than blaming an
assessor or converting tampering to a profitable rejection.

Bindings cover the packet, policy, raw response, eligible execution fixture,
consumed future observations and versioned replay semantics. These are integrity
bindings, **not proof that a proposal was committed before prices were revealed**.
An eventual campaign still needs its immutable lock and code/data provenance.
On failed-data results, `consumed_future_sha256` covers validated fields up to
failure, not rejected values. Two different corrupt rows can share that prefix
hash. Retain/hash raw archive and failure evidence separately in a campaign;
do not use this partial hash as complete provenance for null cases.

## Verified synthetic outcomes

These prices are made up, not BTC trades or new agent judgments. Entry105,
stop99, destination110, costs12bps, risk budget100; theoretical quantity
100/6.126. Three-minute fixture horizon is only for concise public tests.

| Future path | Resolution | Net modeled PnL |
|---|---|---:|
| Target touched first | Target110 | +79.56 |
| Stop touched first | Stop99 | -100.00 |
| Both touched in the same candle | Stop99; ambiguity flagged | -100.00 |
| Neither touched; deadline open106 | Timed exit106 | +14.27 |

Both downside-rebound and upside-expansion labels pass the same mechanical
positive controls. Wait proposals also pass with their later eligible fill and
own structural target. This demonstrates arithmetic/contract behavior, not that
an agent knows which real setups will win.

## Verification and review

- TDD:38 tests first failed because the new outcome API did not exist.
- Initial outcome tests:38 passed in2.40s; after review-inspired boundary tests,
 48 passed in2.97s. No runtime algorithm change was needed after review.
- Initial related regressions:315 passed in51.15s. Expanded final command below:
  **325 passed in51.02s** (48 new outcome tests plus277 related regressions).
- Bare repository `python3 -m pytest`: exit3,5 collection errors in9.38s;
  fatal traceback is the pre-existing `tests/test_integration_fixes.py:27`
  missing `configs/baseline_wyckoff_test.json`, followed by `sys.exit(1)` at46.
  The aborted run does not enumerate all five errors. No whole-repository-green
  claim; the unrelated collection problem was not changed.
- The entire research suite was not rerun for this extension. Earlier broad
  research counts belong to the previous checkpoint, not this change.
- One independent read-only software review accepted the bounded adapter with
  no blocking finding; independently38 tests passed in2.48s. Its one P3 audit
  observation was correct: different invalid HLCV inputs can have the same
  accepted-prefix hash. Root reproduced both null results and documented the
  exact binding scope in the API and this checkpoint. No full rejected-input
  fingerprint is implemented or claimed. Ten additional tests cover missing
  policy/malformed response, huge horizon, tick-separated invalidation, zero
  costs, deadline barrier precedence and consumed-versus-unconsumed hash changes.
- Root agrees with the review's declared exclusions: profitability/interpretation,
  source authentication/freeze-time proof, executable fills/lot sizes/funding/
  portfolio modeling, separate invalidation/campaign integration, and frozen
  reference internals beyond this integration. These remain explicit limitations,
  not deferred correctness fixes. No trading assessor or critic was called.

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q \
  tests/research/test_lc_structure_outcome.py \
  tests/research/test_lc_structure_packet.py \
  tests/research/test_lc_structure_proposal.py \
  tests/research/test_lc_structure_preentry.py \
  tests/research/test_lc_context_assessment.py \
  tests/research/test_lc_published_assessment.py \
  tests/research/test_lc_context_facts.py \
  tests/research/test_conditional_entry.py \
  tests/research/test_lc_single_assessment.py \
  tests/research/test_entry_case_outcome.py
```

## What remains / concrete next deliverable

The scorer answers **what a specified hypothetical trade would have done**, not
whether the agent makes better proposals than code. Next is one frozen economic
comparison protocol and its thin integration: candidate roster/exposure status,
causal archive mapping, exact policy, measured/assumed timing, a code-only
structural comparator using identical execution mechanics, subtype-isolated
books and explicit invalid/nonentry accounting. Lock real proposals before
revealing their outcomes, then report winners retained, losses avoided, rejected
winners, participation, costs and uncertainty. No automatic paid launch or
parameter sweep is authorized by this software checkpoint.

All142 previously scored cases remain exposed development data. Old agent
responses selecting the fixed menu cannot be silently translated into new
structural judgments. New outcome-hidden proposals and a prospective or genuinely
unexposed evaluation remain necessary to assess incremental agent value.

At the initial replay handoff these files were uncommitted at HEAD c68fd1e.
The subsequent [practice-lab design checkpoint](../superpowers/specs/2026-09-29-lc-practice-lab-design.md)
includes the scorer/tests/report and handoff in its local checkpoint; verify git
log/status for the commit. Fresh pre-checkpoint focused regressions325 passed
in51.71s. No work remains running; unrelated untracked work was preserved.

Public tests need no local market data, network or new libraries. A real study
still depends on the private minute archive and source manifests listed in
PROJECT.md; GitHub alone is insufficient. No real historical outcome, production
change, push or PR was made in this extension.
