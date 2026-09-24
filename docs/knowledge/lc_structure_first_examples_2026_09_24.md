# LC structure-first worked examples

September 24, 2026. Companion to the [decision-contract design](../superpowers/specs/2026-09-24-lc-structure-first-design.md).
**Human-readable development illustrations, not new agent answers, trade advice,
backtest results, or desired labels for a test set.** Historical outcomes are
already exposed. Do not give this document to outcome-hidden assessors.

## 1. What a justified entry proposal would look like — synthetic

These prices and events are invented solely to explain the contract. Before the
setup, a confirmed parent range spans 90–110. A previously established child
support is 100. Price probes to 96, reclaims 100, then a completed minute retest
holds above 100 and closes at 101. All events are available by the decision cutoff.
The frozen illustrative catalog contains 96, 100 and 110 with their provenance;
the toy fixture declares no intervening catalogued opposition.

An `enter_proposal` can say: “This is a rebound inside the parent range, following
a probe and reclaim of pre-existing child support. The observed retest—not the
mere wick—is the confirmation. A break of the probe low at 96 invalidates my
rebound thesis; 110 is the observed parent ceiling and proposed destination.
The alternative is a temporary bounce before another breakdown.”

At an **illustrative, not guaranteed**, fill of 101, risk to 96 is 5 and distance
to 110 is 9: gross reward/risk is 1.8, not an automatically assigned 2R target.
Code must still check fill-time geometry, costs and the external policy. If the
eligible fill is 109, remaining distance is 1 against risk of 13; the old
indicative calculation cannot justify entry. If the policy is absent, no entry
is executable. This is a schema/logic example, not evidence of an edge.

## 2. January 31, 15:00 UTC — a defensible rejection, not a universal veto

Saved facts: hourly close 81,452.5; setup low 80,754.3. The 4H floor at 81,832
was available at 09:00, before the 14:00 setup hour, and is broken by the decision.
The last completed 5m high is 81,539. Both 1m and 5m last-two-bar comparisons show
higher lows and higher closes. The strict pre-setup daily bound is absent.

A defensible explanation is: “There is a local rebound, but a break of 81,539
alone would still leave price under the failed 81,832 floor. I do not yet have
enough evidence to propose a continuation through that supply.” This acknowledges
the bullish observation rather than pretending it is absent. A distinct bounded
rebound proposal could be considered only with a justified invalidation,
destination and viable room; this example does not prove one exists.

Historical agent choice: reject. That choice is not correct *because* the old
trade later lost. It must stand on the decision-time argument.

## 3. March 8, 23:00 UTC — rebound possibility inside conflicting structure

Saved facts: daily parent 62,979.5–69,999 was available March 6 at 13:00 and remains
intact. The tightened 4H floor 66,508 was available March 8 at 21:00, before the
22:00 setup; it breaks by the decision. Setup low is 65,569.2, hourly close
66,198.4 and last completed 5m high 66,214.

A **conditional hypothesis**, not a new assessment: “The larger decline remains
opposing evidence, but the child response could support a bounded rebound.
A post-arm completed minute close over 66,214 would confirm local recovery,
not recovery of the 4H floor. Treat 66,508 as an intervening obstacle or first
destination; justify the stop and net room before proposing any entry.”

At the indicative close, distance to that obstacle is only 309.6 while distance
back to the setup low is 629.2, about 0.49 gross R. This does not impose a new
universal 0.49R cutoff. It exposes the trade-off: a tighter, causally justified
child stop or a separately evidenced continuation thesis would be necessary to
describe a materially different trade. Inventing a tighter stop to improve the
ratio is prohibited. Waiting above 66,508 also changes the thesis and entry;
it is not interchangeable with the original local-confirmation proposal.

Historical agent choice: reject, despite acknowledging local recovery. The old
fixed trade later won. Neither fact forces the new design to accept this case.
The improvement sought is a testable argument and geometry, not hindsight rescue.

## 4. February 25, 02:00 UTC — a conditional wait can be coherent

Saved facts: expansion hour 64,106–66,283.1, closing 65,860.4 above the previous
hour's high. Last completed 5m high is 65,953.2; its low is 65,811.7. Final minute
closes lower. Strict pre-setup daily and 4H bounds are absent; structures formed
at the decision cannot be claimed to have preceded the setup.

Historical agent choice: wait for confirmation above 65,953.2. That is a coherent
way to distinguish the observed expansion from the still-unconfirmed resumption
after a pullback. Under the new contract it is **not sufficient by itself**:
the agent must also select a supported invalidation and destination and confront
nearby opposition, including the observed 66,283.1 setup high. This example does
not certify the full structural proposal or label the raw 5m low confirmed support.

## 5. July 26, 23:00 UTC — local confirmation is not cleared overhead room

Saved facts: hourly close 65,384.7, setup high 65,480 and last 5m high 65,400.2.
Pre-existing intact daily and 4H ceilings are 65,589.7 and 65,780. The old fixed
stop is 64,887.98937511034. At the indicative close, those ceilings are about
0.41R and 0.80R away under that old stop; the local setup high is nearer still.

Historical agent choice: wait, while explicitly acknowledging those obstacles.
The design should require: “Does the proposed trade terminate before resistance,
or require breaking it? What evidence supports the latter?” A close over
65,400.2 does not establish acceptance above 65,480 or 65,589.7. One may reject
the present proposal or describe a different conditional thesis; one may not
delete these obstacles or declare every near-ceiling trade a loser.

## Source bindings and reproducibility

Private root: `results/lc_single_bounded_2026_09_20/run_v1/`.
Manifest case indices are 001, 006, 003 and 017 respectively. For each case,
`manifest.json` supplies `source_path`; the examples use that source request's
`context`, `source_packet.current.source_candle` and `plan`. Original choices are
in `cases/NNN/role_response.txt`. These artifacts are local-only, not guaranteed
available from a GitHub checkout. Timestamps above are UTC.

Exact source-request SHA-256 values (checked against manifest file bindings):

| Case index | SHA-256 |
|---|---|
| 001 | `aa73627427be37b64f985d327894031dd96f62326d71cc3f9a24e4672938c3f1` |
| 003 | `57dc038882c6646b55e0f548f6f502c1ddad5d9686925edc0c6442b60b39b61b` |
| 006 | `6b8d6f172b0eae4e275fd1b6f80380fef4889e1d867b4bf5eb65783fcbccb296` |
| 017 | `28a2786b740dddc554db435d9356cc6996ad57c1809f28ccc0e294ae515a6074` |

Parent lifecycle is the saved detector's reconstructed fact, not proof that a
teacher or trader would draw exactly that range. No new outcomes were computed.
For existing economics, see the [bounded report](lc_bounded_single_results_2026_09_21.md)
and [missed-rebound diagnosis](lc_missed_rebounds_2026_09_22.md).
