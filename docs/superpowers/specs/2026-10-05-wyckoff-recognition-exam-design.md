# Wyckoff recognition exam design

## Purpose

Test whether the selected native Wyckoff path recognizes source-supported market
relationships, not merely whether pretagged events advance a state machine.
The user approved the12-case exam, requested original trader X edge cases, and
delegated design questions to a quant. The quant approved this scope on October5.
Written specification and plan review remain with that delegated reviewer.

Use the existing quant branch and preserve all prior dirty work. New research
modules, tests, fixtures, source notes and reports only. No engine/config edits,
M2 activation, V2 promotion, thresholds/weights tuning, deployment, economic study,
paid external assessments, installs, commit/push/PR or rewriting prior artifacts.

## Chosen approach

Use actual source functions with observable intermediate stages. Pretagged
sequencer tests cannot measure recognition; a whole exchange/portfolio replay
introduces unrelated entry gates and operational state. A separate source-bound
harness is the smallest useful middle option. It must not become a parallel
implementation of the trading strategy.

## Source and fixture contract

Two focused researchers refresh original X leads for Wyckoff Insider/Moneytaur
and Crypto Chase/Zero Ika/Bojan. Keep precise URL, retrieval date, readable text
versus inspected image versus indexed excerpt, narrow supported teaching, and
limitations. Saved notes are leads; inaccessible original content remains so.
Primary Wyckoff teaching may supplement but never impersonate an X source.
Predictions, retrospective labels and winning outcomes are not ground truth.

Exactly12 deterministic, valid UTC-start-stamped hourly OHLCV sequences:
four positives (spring/no-spring accumulation, upthrust/no-upthrust distribution),
four near-misses (missing range, missing recovery, failed test, invalidated parent),
four ambiguous cases (climax versus continuation, developing range, local rebound
versus range escape, conflicting timeframes). Adequate fixed historical warmup
allows hourly/4H/daily processing. Values are teaching-derived synthetic data,
not reconstructed real trades or precise chart prices.

Each case declares parent boundaries, candidate/confirmation/invalidation and
decision checkpoints, source IDs, qualitative volume/spread facts and permissible
uncertainty. A source-only independent quant reviews the actual fixture and
expectations BEFORE detector execution. The reviewer does not read engine outputs,
project performance history or detector code. Revise inadequate cases only before
freeze. Freeze exact inputs, annotations, checkpoints, code/config/source hashes
and approval together. No substitutions or threshold adjustment after results.

## Execution

Use the existing offline actual-source LiveFeatureProcessor import boundary and
deny_network. At each predeclared checkpoint, give only that hourly prefix to
LiveFeatureComputer._wyckoff_features. Observe raw event flags before state
validation and resulting events/phases/context; wrappers must call unchanged
original functions. Record actual per-TF configs and input closure/source status.
Capture parent state/geometry through a non-mutating wrapper if necessary.
No injected flags, supplied volume z-scores or synthetic context objects.

Feed those produced features into the actual ArchetypeInstance directional
scorer for long and short. Execute the unique phase-boost branch from
V11ShadowRunner source via a tightly checked AST extraction: verify source hash,
enclosing class/method and exact unique branch shape; compile unchanged statements
with explicit inert logger/config/intent inputs. Probe both directions and
qualifying/nonqualifying phases. Report conditional multiplier/capex changes only,
not entry eligibility, allocation, final risk or orders. No exchange runner starts.

Use declared prefix reruns and future-tail perturbations to check causal sampled
checkpoints. This is sampled-prefix verification, not proof of every code path.
Run sequentially with a600second wall-clock limit and100MiB artifact ceiling.
Fail closed on source drift, unsupported schema, invalid fixture, unexpected
network, missing approval, ambiguous AST extraction or output overwrite.

## Verdicts and acceptance

Keep three layers separate: source-supported interpretation; engine recognition
and configured availability; consumer response. Semantic claims concern market
relationships, not current thresholds or phase mappings. Inadequate annotation is
unresolved, not a passed case. A no-spring positive stays positive when full M2
is off; suppression/unsupported representation does not count as recognition.

For negative/ambiguous cases inspect false confirmed identity and associated
phase-sizing effects, while distinguishing provisional evidence from a confirmed
setup. Positive cases inspect missed milestone recognition and unsupported
distinctions. Record full traces, each discrepancy and its scope. No aggregate
pass percentage may hide missed positives or recognition failures behind disabled
configuration compatibility. Material false confirmation or unsupported sizing
attribution blocks economic progression. Zero semantic failures is not an edge.

## Deliverables and verification

Source notes and concise synthesis; locked12-case input/expectation manifest;
additive reusable runner with engineering regression tests; one bounded offline
exam result with case-by-case recognition/consumer ledger; quant and independent
software review; PROJECT/MEMORY continuity updates. Tests prove harness integrity,
not that semantic results must be green. Report existing full-repo blockers.
Keep complete Wyckoff/Fibonacci/Gann vision and adaptive trade management outside
this recognition exam. No automatic repairs or P&L launch after it.
