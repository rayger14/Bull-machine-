# Developing post escape pullback design

October 5, 2026. Draft for review, not implemented or a frozen experiment.
This is Chunk 1 of the [teaching rulebook](../../knowledge/teaching_rulebook/README.md).
The user approved moving from source review into chunked, written rules. This
written contract and its explicit conventions still require review before the
implementation plan and code changes.

## Intent and scope

Recognize a pullback that develops over several closed candles after a qualified
escape of an already identified parent range. Distinguish its developing retreat,
confirmed turn and subsequent failure. Do not mistake the first quiet candle for
the completed pullback or extend deadlines until a desired label appears.

The output is structural evidence, not an order, a Bojan pattern, confirmed
accumulation/distribution or calibrated probability. Interior M2 tests are a
different valid role, outside this first repair's positive criteria. The existing
full M2 switch stays off. Existing spring/upthrust phase justification, detector
thresholds, strength qualification, range construction and sizing configuration
are outside scope. Retained-score lineage and W02-style gradual strength are
separate chunks, not hidden additions to this repair.

## Teaching and operational choices

[SRC-WI-05](https://x.com/Wyckoff_Insider/status/2104619692862103782) distinguishes
interior LPS from a post-breakout former-resistance test. The mirrored distribution
role is supported by [SRC-WI-06](https://x.com/Wyckoff_Insider/status/2104231291373699342),
without imposing every narrated event as mandatory. [SRC-ZI-01](https://x.com/IamZeroIka/status/1962520664436621597)
and [SRC-ZI-03](https://x.com/IamZeroIka/status/1962520672661291044) distinguish
developing response from completed confirmation and failure. None supplies the
mean-volume formula, exact turn clock, deadline or proximity below.

Three approaches were considered. Enlarging the first-candle tolerance is small
but leaves the wrong episode representation. A full multi-timeframe/M2/zone rewrite
could represent more setups but expands the review surface. The chosen first
candidate is a single post-escape episode attached to existing parent/bound IDs,
with explicitly labeled conventions and mirrored tests. Its purpose is meaning
and causality, not fitting the exposed recognition examples.

## Inputs and fixed references

Use completed, contiguous OHLCV with timezone-aware candle start and available-at
timestamps, following the existing candle-integrity contract. Record the venue,
instrument and timeframe through the containing stream. Missing or nonpositive
traded volume cannot count as quiet supply/demand evidence.

An eligible parent has previously locked bounds and the current existing validator's
qualified escape. Preserve parent ID, bound ID, escape index/availability, broken
boundary and the associated strength/weakness candle and confidence. Eligibility
must have been known before the retreat. A price-only escape is explicitly
unqualified; an inside-range LPS is explicitly another role.

Direction normalization mirrors the geometry: for long use price directly; for
short use its negative with high/low swapped. The favorable side is then above
the normalized boundary in both directions. Never mirror raw traded volume to a
negative value. Unknown context does not default to a trade side.

## Proposed episode contract

The following are project conventions proposed for review, not exact trader rules.

1. **Start:** on the first later candle whose normalized close falls below the
   preceding close while the qualified escape remains active. Start tracking even
   if the broken boundary is not yet near. This prevents selecting only the final
   quiet candle and discarding earlier adverse evidence.
2. **Identity:** bind one episode to parent, bound and escape. Freeze start index,
   start availability and deadline. A new adverse extreme updates the developing
   leg, not episode identity or elapsed time.
3. **Deadline:** inherit the existing escape horizon `H = sm_ar_max_bars` (default
   15), measured from escape, not from each candidate. A bar at distance H is
   eligible; a bar beyond H expires the episode before confirmation. Record the
   equivalent absolute deadline and require the same fixed candle duration.
4. **Running extreme:** store the lowest normalized low since start and that
   candle's high, index and availability. Strictly lower lows replace it while
   developing. Equal lows retain the earlier reference. A candle establishing a
   new extreme cannot also confirm a recovery from itself.
5. **Location:** require that running extreme is within the existing 3% proximity
   of the immutable broken boundary: `abs(extreme - boundary) <= abs(boundary)*0.03`.
   This compatibility convention is not a source-authenticated or optimized
   tolerance. Do not increase it to rescue old examples.
6. **Volume and spread:** compare the arithmetic mean traded volume and mean
   high-low spread from episode start through the bar immediately before the
   proposed confirmation with the frozen strength/weakness candle. Both means
   must be strictly smaller for the first candidate's `supportive` classification;
   otherwise classify `challenging`. Any missing required observation makes the
   episode terminal `unavailable` as specified below. Record count, sums,
   reference and ratios. This operationalizes
   lower average activity, not a claim of monotonically diminishing supply or
   observed institutional absorption. Do not drop inconvenient intervening bars.
7. **Turn:** on a later candle, if no new adverse extreme occurs and normalized
   close exceeds the stored extreme candle's high, recognize the causal turn.
   Confirm only if location and supportive complete volume/spread evidence are
   already satisfied. The confirmation candle is excluded from those means but
   must still have valid positive traded volume and pass clock, geometry and
   boundary-integrity checks. Freeze confirmation
   at its available-at timestamp; never backfill an earlier candle.
8. **Boundary loss:** a normalized close below the broken boundary ends current
   authority before any turn can confirm. Equality holds for this candidate;
   intrabar wicks are recorded and may extend the unconfirmed extreme. This close
   basis is an explicit project convention, not a recovered universal invalidation.
9. **After confirmation:** freeze the confirmed extreme. A later low below it,
   boundary loss, parent/bound replacement, unavailable clock/data or expiry revokes
   authority. Preserve historical confirmed-at and reason for revocation. An
   unchanged historical confirmation is not current entry permission.
10. **No resurrection:** terminal episodes cannot resume or restart their deadline.
    A newly qualified escape can create a new identity, but cannot overwrite the
    old episode or confirm its retest on the same bar. Repeated crossings are not
    permission to reuse expired strength or silently renew old authority.

An early turn with challenging evidence remains unconfirmed and records the
failed condition; the episode can continue within its original deadline. Every
later decision must use all intervening episode observations, not a newly selected
quiet subwindow. This is a single candidate proposed for pre-run freezing, not a
preregistered experiment yet or a threshold search.

### Missing evidence transitions

These are explicit fail-closed project conventions. `failed`, `expired` and
`unavailable` are terminal for the episode; `confirmed` is not terminal because
later observations can revoke its authority.

| Missing or invalid evidence | Required transition |
|---|---|
| Frozen strength/weakness reference has absent, nonfinite or nonpositive volume, or invalid price/clock | At the first attempted retreat, record a terminal `unavailable_reference` episode; do not substitute another reference or confirm a test. |
| Any contributing retreat/intervening candle lacks valid positive traded volume | Terminal `unavailable_episode_volume`; do not drop it, fill it with zero, or resume once later volume is present. |
| Proposed confirmation candle lacks valid positive traded volume | Terminal `unavailable_confirmation_volume`; exclusion from the averages does not exempt it from input integrity. |
| Invalid OHLC, missing/noncontiguous/invalid availability clock, or unavailable required identity | Terminal `unavailable_input_or_clock` before evaluating confirmation; preserve the existing parent's separate integrity handling. |
| Required observation becomes unavailable after confirmation | Revoke current authority with the appropriate unavailable reason; keep historical confirmation unchanged. |

Observation may continue for diagnostics after a terminal state, but no later
bar or backfill can revive that identity or alter earlier snapshots. A fresh
qualified escape with complete required references may start a new episode.
Data unavailability does not assert that the market thesis failed; preserve that
distinction from an observed boundary loss. The emitted primary state is
`unavailable`; the reason values above explain which evidence was missing.

## State and output requirements

Expose `approaching`, `developing`, `confirmed`, `failed`, `expired` and `unavailable`
states with explicit reasons. Approaching becomes developing once the running
extreme first satisfies location. Historical confirmation is an independent
record; current authority is a separate boolean derived from complete current
state, never from the phase label alone.

The episode record includes parent/bound/escape IDs; role `post_escape`; direction;
start, extreme, confirmation and deadline clocks; immutable boundary/reference;
volume/spread aggregates; last observed-at; historical confirmation; active status;
and terminal reason. Snapshot data must be detached so later updates cannot mutate
previously emitted records. Keep local observations and historical tests visible
without granting them post-escape authority.

Integration is confined to post-escape retest handling in
`engine/wyckoff/range_evidence.py` and its `engine/wyckoff/events.py` consumer.
Do not change the shared `_advance_test` behavior for range/phase tests as a side
effect. Keep existing scalar/phase outputs compatible; any additive episode
payload and terminal reason mapping must be specified in the subsequent plan.
The current phase-C guard remains unchanged and receives no new boost permission.
Trace raw candle → evidence → event → feature → score/sizing observation, but do
not claim every retained-score consumer is repaired in this chunk.

## Fresh recognition matrix

These are twelve proposed case specifications, not generated or frozen raw
candle files. A separate source/contract review and pre-run label freeze precede
implementation results. Use new synthetic OHLCV with recomputed detectors, not
copied feature flags, old row numbers or old exposed geometry.

| Cases | Role and contrast | Required observation |
|---|---|---|
| P01 long / P02 short | Genuine gradual test: several retreating extremes, intact broken level, supportive aggregate, later turn, then failed hold | No premature event; confirm only at the turn; revoke later authority without erasing history |
| P03 long / P04 short | Genuine one-candle quiet test with later recovery; then observations extend past the original deadline | No same-candle confirmation; valid later event; fixed expiry cannot move |
| P05 long / P06 short | Misleading final quiet candle after an adverse episode whose complete volume/spread aggregate is challenging | Final candle alone cannot certify the test; preserve challenge reason |
| P07 long / P08 short | Misleading recovery after a completed boundary-loss close | Old episode remains failed; any new escape has a different identity and requires its own later test |
| P09 long / P10 short | Ambiguous to this contract: an interior LPS/LPSY-like test with no parent escape | Preserve interior/out-of-scope role, no post-escape event; not a claim that the broader model is invalid |
| P11 long / P12 short | Ambiguous strength: price crosses the parent boundary but no qualified same-parent strength/weakness is available | Expose price-only escape and missing qualification; plausible pullback does not invent it |

For each family include clock-gap/missing-volume perturbations, equal-extreme and
deadline-equality checks, parent-replacement variants and prefix/future-tail
comparisons where meaningful. These are engineering variations, not additional
independent market observations. Compare actual production-path fields, not only
a helper's self-consistent assertions. Test raw-input and witness-prefix availability
before output hashes are sealed. The old exposed 12 remain unchanged regression.

Explicit missing-data assertions cover the frozen reference, a contributing
retreat candle, the proposed confirmation candle and a post-confirmation candle.
Each must deny/revoke authority as specified, and a later complete candle must
not restore it. Include an invalid/gapped-clock variant and verify that diagnostic
observation and historical confirmation remain distinguishable from authority.

## Acceptance and limits

Acceptance requires all reviewed positive milestones at the expected first
available candle, correct withheld authority on lookalikes/unavailable inputs,
explicit alternative roles on ambiguous cases, exact long/short mirroring and
unchanged prior outputs under appended future tails. Retain failures rather than
adjust labels, tolerances, horizons or source definitions after seeing results.

The same run must show unchanged out-of-scope range/phase-test behavior and full-M2
configuration. Report score/sizing effects separately, even if no orders are run.
The acceptance receipt records code/config/data hashes and review disposition.
Twelve synthetic cases supply bounded recognition coverage, not market edge,
trade frequency, minute execution, WFO/CPCV or deployment approval.

## Next review and implementation boundary

Review the source-versus-convention separation, aggregate choice, turn definition,
deadline equality, close-based boundary failure and post-confirmation revocation.
After the written contract is accepted, write the detailed implementation plan,
construct and independently review/freeze the fresh packet, implement with failing
tests first, then run the new recognition suite and existing focused regressions.
Do not start code, paid assessments, full M2, live configuration or economic tests
from this draft alone. Exact Bojan execution and complete Fib/Gann construction
remain separate source gaps, not silently implemented features.
