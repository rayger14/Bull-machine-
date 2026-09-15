# LC context discrimination: separate research contract

## Decision and scope

September 15, 2026. User approved proceeding from the reviewed LC pattern table
and delegated bounded quant judgment. Independent quant design review recommends
this three-arm comparison, with a facts-only implementation slice first.
This is a new hypothesis, not a relaxation or replacement of the frozen
`lc_nested_child_rejection_v1` experiment. No production changes or live orders.

Research question: does an agent interpreting nested structure choose between
immediate entry, existing minute confirmation and abstention more usefully than
either fixed code policy? No predictive benefit is assumed.

Chosen approach: preserve the existing single confirmation level and expose
context as facts. Rejected alternatives for this version are (a) fitting another
fusion/room cutoff, and (b) adding separate subtype-specific reclaim/retest levels.
Both add degrees of freedom before the agent's discrimination is demonstrated.
The five-minute-high control is generic; context conditioning belongs to the
agent's selection, not to an undisclosed new deterministic structural filter.

## Exact comparison

| Arm | Entry decision | Purpose |
|---|---|---|
| A | Native LC immediate | Unchanged candidate baseline |
| B | Always select `wait_5m_high` | Generic confirmation control |
| C | Agent selects `enter`, `wait_5m_high`, `reject`, or null | Contextual judgment |

All arms use native **pre-winner** LC long emissions, not final dedup selection.
No RSI subtype is removed. No intact-parent, daily/4H agreement, same-hour reclaim
or minimum ceiling-distance filter is added. Integrity/available operands are
checked separately from adverse or absent structural states. A supported entry
must identify its larger context, opposing evidence, horizon and invalidation;
the reviewer checks factual support, not whether it likes the trade. Verified
absence of an algorithmic parent is not permission to invent one. Any alternative
structural interpretation must cite the supplied completed candles and their
clocks, not an unconfirmed pivot or unavailable feed.

Fixed economics for every executable alternative: $50,000 notional; 12bps
round-trip assumed costs ($60); 90seconds processing rounded up to a minute arm;
zero routing; 15minute exclusive entry expiry; stop=candidate close−2.7×ATR14;
target=actual entry+2×(actual entry−stop); deadline=decision+1440minutes. Original
stop cancellation remains active from decision while waiting and during latency.
No stop, target, expiry, sizing or exit-management tuning.

`wait_5m_high` freezes the latest aligned completed five-minute candle's high at
decision. Only a **fully closed post-arm one-minute candle** with close strictly
above that high confirms; fill is the next minute open, strictly before expiry.
No full-five-minute confirmation, retest, alternative anchor or intrabar inferred
path is implied. A is also subject to the common processing delay, not a fill at
the signal's indicative close. Existing resolver and isolated occupancy remain
the accounting authority.

## Facts and judgments

`scripts/research/lc_context_facts.py::describe_lc_context(packet)` consumes a
sealed packet built from caller-verified reconstruction. It does not independently
authenticate prices, revalidate a full ledger, grant entry permission or read
outcomes. It checks its consumed operands and binds the original packet hash.

- Hourly close relation: below prior low, above prior high, otherwise inside or
  on a boundary. Low/high sweeps use strict inequalities and are independent.
  Lower reclaim means current low < prior low < current close; upper rejection
  means current high > prior high > current close. These flags need not be mutually
  exclusive or imply that the close is inside the previous range. RSI stays a
  separate supplied feature, not the geometry classifier.
- Each 4H/N3 and daily/N3 view retains strictly-before binding, original bound,
  updates through decision, known absence versus unavailable evidence, directional
  break and prior-hour containment. Bound lifecycle is **not** current market
  state: a range absent before setup can form by decision. Both are reported.
- Signed range position and signed distance to the specified ceiling use
  `(close−low)/(high−low)` and `(high−close)/(close−frozen_stop)`. No clipping,
  fitted threshold or assertion of unobstructed room. Invalid operands yield null.
- Latest two completed aligned 1m and 5m candles provide higher-low/higher-close,
  candle body and close-location observations. They do not score later confirmation.
  The full existing 1m/5m/15m/1h/4h/daily packet remains available to future roles.

Continuation, exhaustion, which horizon controls, whether a boundary obstructs
the chosen trade, and whether waiting is appropriate are judgments, not facts
deduced from a volume spike or a high score. Facts-only output has no plan ID or
overall trading-readiness score. Old aggregate `conditions.status` is not reused
as source validity.

## Source population and exposure

Reuse the existing Q1 source reconstruction; no source rerun or date extension.
Candidate window [2026-01-01,2026-04-01) UTC; observation tail ends April2 without
admitting April candidates. Full code books preserve all18 candidates and
continuous occupancy. The four-case agent comparison is a separately labelled
matched subset, not a full-Q1 agent portfolio.

Existing files under `results/lc_persistent_master_2026_09_14/preflight_phase_a_final/`:

| File | SHA256 |
|---|---|
| `cohort.json` | `7dc5befe89da52bd57d8c8e8dc37bce92debb8b7219a83d66c7f5523f6c77639` |
| `selection.json` | `f600ab65d768762bda14f5cb5d7d129ce39031b2c5a9336b44de8916cf32ee10` |
| `cases.json` | `25e86e044b87d199be2aa8cf6d37043cfcba699018d63083d9afe286b25ac78e` |

Fixed pilot IDs: `hourly-lc:` followed by these exact UTC clocks:
2026-01-02T04:00:00+00:00; 2026-01-04T01:00:00+00:00;
2026-01-05T01:00:00+00:00; 2026-01-16T16:00:00+00:00.
These are the existing first-four ordering, not selected by context or outcome.
No replacements for abstention, invalid evidence, failed review or loss.

Before roles, append a separate exposure ledger to the new experiment. Preserve
the original selection ledger; reconcile earlier case registries by exact ID or
clock when identity is unavailable. Already-known March examples belong to
development evidence if their exact candidate IDs/clocks match, not merely dates.
Unassessed is not synonymous with untouched. Q1 is retrospectively exposed
development history, **not** pristine holdout, walk-forward or CPCV validation.
A later untouched chronological evaluation needs its own preregistered freeze;
four cases cannot establish profitability or statistical significance.

## Memory, role contract and immutable jobs

New role namespace: `lc_context_discrimination_v1`. New artifacts must not
overwrite v1 requests, memory snapshots, grades, locks or outcomes.
Reuse the five already-reviewed policy-neutral records from the old snapshot:
`2186fae6060c59e8555b4025eeb873f082f3a9e3cc497651abbbfbc0cf5cb611`,
`3ac824655a3fe6484b628daa9aa7031b328ebc24bcd3d672464e24649db84fd3`,
`457aa2ecf80062eaeca2dbef2c2e75fa3c6fce97f43e7588146606cbeabda8dd`,
`51dd5fc4bd5b8cb53e999aa9ade4d6d31e02f32afeec26ba6e8d70309f8d186f`,
`c56966b1bfa9a340dd607b3d1a7829b3139ea50ea3cc028a3cd952b65f8d5f64`.
Exclude old hypothesis
`6f84d4de456e046fb38e2762d9381e9e02a2093c09abd4858704afaf70d5c8d3`
and master-brief record
`d30b1cf051576cdd67cc4555a4eeaf4fd53d05d8e4e855b3f54a7252540700cd`
from the new snapshot without modifying or rejecting them in the original experiment.
They explicitly mandate the old reclaim contract. A new outcome-free brief
must be versioned, independently reviewed and frozen; no case-outcome lessons.

New adapter must replace old outer **and nested** instructions, remove legacy
mandatory conditions from market-role context, rebuild catalogs/hashes, and
restrict the menu to the three plans above. It cannot call the old grader or
pretend its aggregate structural fail is now pass. Preserve current exact LC
response/critic schema, citations and support/oppose/uncertain mapping, but bind
them to the new policy. Source uncertainty or ungrounded interpretation is not a
profitable rejection. Code-verified facts and selected economics remain immutable.

The existing `ResearchJob` imports v1 validation/grading directly; it is not a
drop-in generic runner. A new job adapter must recompute the **new** grade from
exact captured raw responses. Never monkeypatch its imports or reuse a v1 job
directory. Reuse transaction/hash-chain semantics with regression coverage.
At most four fresh specialists/four critics, one per fixed case, no retries or
answer repair. The outcome-free brief may be controller-authored and independently
reviewed to conserve calls; authorship must be recorded, not labelled an Astra run.

Freeze source, exposure, code, policy, memory, instructions and exact packets
before roles. Lock raw specialist outputs before critics and all grades before
any new outcomes. Enforce packet-only tool access for market roles; no project
handoff, graph, known-case reports or external source browsing. Model/transport
metadata must describe actual observed calls, not fabricated model certification.

## Scoring and stopping

Run independent chronological books per arm and archetype, never filter completed
trades and add them up as a substitute for occupancy replay. Report full code
cohort separately from matched-four A/B/C. A reviewed explicit reject is zero
exposure; null, missing/invalid transport, incomplete review and material errors
are unknown output. If unknown could affect occupancy, aggregate agent PnL stays
null. Isolated valid-case contrasts may be reported separately, never renamed a
complete agent book or avoided-loss benefit.

Report entries, expiries, stop cancellations, rejections, unavailable cases, busy
skips, costs and net PnL; C−A and C−B on the same valid population; missed winners
and avoided losses only for valid comparable alternatives. Gross target hits and
profitable deadline exits remain distinct. No inferential performance claims from
four cases and no forced acceptance if all reviewed choices reject.

## Implementation checkpoint and next acceptance gate

This turn implements only the facts module and literal tests, plus a source-only
four-packet diagnostic. The new request/job adapters, new brief/snapshot,
permanent role freeze and economic comparison are **not implemented or run**.
Next deliverable: tested separate request/grader/job integration, independent
review, then frozen four-specialist/four-critic comparison using this contract.
Required integration tests include positive choices remaining legal when old v1
fails, unknown not earning rejection credit, altered context/menu/hash refusal,
restart/transport/grade binding, and unchanged old v1 tests.
