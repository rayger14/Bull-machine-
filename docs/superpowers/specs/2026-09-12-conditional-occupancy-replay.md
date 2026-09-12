# Conditional readiness and occupancy — frozen research contract

User approved the next multi-candidate replay step. Delegated quant review
approved actual-exit release with a causal position scanner. This is a bounded
extension of the existing conditional resolver and reviewed-choice adapter,
not production execution, a funded portfolio or a validated agent strategy.

## One sleeve, complete candidate accounting

Run hourly and minute tracks separately, with one fixed-notional long position
per sleeve. Every supplied candidate has a unique ID, track, original decision
time and either a frozen research plan or an explicit unavailable reason.
Reject mixed tracks, duplicate IDs, inconsistent decision clocks and invalid
economics. All executable plans in a sleeve share notional and cost assumptions.
The caller remains responsible for complete source enumeration and prior locks;
this helper cannot prove it received every opportunity or real role receipts.

Pending intentions do NOT reserve capacity. Resolve each immutable intent using
the existing causal resolver at the requested as-of cutoff. Sort entries by
(eligible entry time, original decision time, candidate ID), independent of input
order. If a position still occupies the slot, mark skipped_busy permanently;
do not retry, move the fill, or retroactively filter already executed trades.

## Actual-exit release

Advance the active position using only observations available by the next event
clock. Opening stop gaps exit at the open; favorable target gaps fill at target.
Both release before another entry at that open. Intrabar stop/target touches are
known at candle completion and release at the following minute open; both-touch
uses stop-first. A newly admitted trade cannot use its entry candle's future
extremes to release for another candidate at that same open.

The original absolute deadline exits at its OPEN and releases before an entry
at that open. Stop, actual-entry-derived 2R target and costs remain fixed.
Record bar-open exit label, observation/release time and open/intrabar phase
separately. A stop may release earlier than the legacy minute fixed-deadline
lockout; that legacy policy is different and is not silently reused.

The old full-horizon scorer must NOT decide admission or release: a later
missing tail can invalidate its full-horizon diagnostic after an earlier exit
was already causally known. The new scanner stops consuming prices at exit.
No full-horizon MFE/MAE or equity-curve claims are needed in this unit.

## Uncertainty and reporting

Missing observations while a candidate is pending can conceal an entry, even
when the known slot is flat. Schedule an uncertainty event at its first missing
observation; uncertainty takes precedence over entries at that same timestamp.
Stop the trustworthy admission prefix there and label later entries
admission_indeterminate. Do the same for missing data required to manage an
active position. This conservative first version does not try to recover a
known book state after a gap. Preserve earlier known admissions and exits.

Unavailable assessments/plans are explicit fail-closed nonorders, not successful
rejections. Their presence prevents a complete-policy PnL claim. Pending/open,
invalid, unknown and indeterminate records are not assigned zero profit. Report
known closed-trade PnL only as a subtotal; full policy net PnL is null unless the
entire supplied sleeve is resolved without these exclusions. Starting equity
unspecified; no funding, impact, margin, inference costs or live certification.

## Tests and next boundary

Test reordered waits, same-time ties, permanent busy skips, open-gap versus
intrabar release, deadline reuse, entry-bar ambiguity, missing pending-intent
data, active-position gaps, and early exit despite a missing later diagnostic
tail. Test prefix invariance, malformed inputs, real reviewed-choice integration
and parity with the old scorer on fully observed single-position paths.

No new market-assessment calls or historical optimization are part of this
implementation. After independent review and regression checks, the next step
is source-complete candidate manifests and frozen matched chronological windows,
not another isolated known-case profitability claim. Preserve all seventeen
archetypes, production, old artifacts, and unrelated graph directories.
