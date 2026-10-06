# LC agent practice — actual results, September 30, 2026

**The experiment ran. It did not demonstrate incremental agent value.**
All 12 fresh assessments were captured and locked before price replay. The agent
book lost $100 after modeled costs, identical to the same-delay mechanical book.
The mechanical rule at its separate 90-second assumption made $178.80. These are
independent hypothetical case sums, not actual money or portfolio returns.

## Spending and completion

- Exactly 12 fresh Astra/high-requested assessor runs; no market critics,
  retries, replacements, follow-up assessments or extra study.
- 12/12 original responses passed the structure contract: 9 rejects and
  3 wait proposals. Semantic interpretations remain unreviewed.
- All 12 full-read receipts match the exact frozen request bytes/hash. No invalid
  transport, interruptions, timeouts or missing outcomes. Role-reported receipts
  are not provider attestation or an OS-level isolation guarantee.
- Reserve-to-capture latency: minimum 129.979s, median 161.021s, maximum 209.067s.
  First reservation 09:55:48 UTC; all-case lock confirmed 10:28:37 UTC.
- Requests total 1,860,603 ASCII bytes; responses total 80,862 UTF-8 bytes.
  These are not billed-token measurements. Exact credits, balance and model
  snapshot are unavailable. Twelve assessor runs are not necessarily twelve
  raw API/model turns: each role can perform internal tool/model steps.
- No further model calls authorized by this result. All owner/report processes
  have exited; no live trading, config, fusion, archetype or frozen code changes.

## Economic comparison

Frozen paper policy: $100 modeled risk including costs per independent case,
12bps flat round-trip cost, maximum $50,000 notional, decision+15-minute entry
expiry, fill+24-hour bracket horizon. No funding, impact or venue lot rounding.
USDT/USD assumed 1:1. Both subtype books remain independent; no overlapping
filled pairs occurred in this particular run.

| Arm | Filled trades | Other cases | Net modeled sum |
|---|---:|---|---:|
| Agent, measured availability | 1 | 9 rejects; 1 expiry; 1 cancellation | −$100.00 |
| Mechanical, same measured availability | 1 | 9 cancellations; 2 no-setups | −$100.00 |
| Mechanical, fixed 90 seconds | 2 | 8 cancellations; 2 no-setups | +$178.80 |
| Stay flat | 0 | 12 nonentries | $0.00 |

Matched coverage is 12/12; agent-minus-matched-rule delta is $0.00. There are
no unknowns hidden in those totals. The different $50,000 fixed-notional legacy
LC reference made $5,003.88, but its stop/2R target/sizing/deadline differ. Its
dollars are **not** a matched estimate of added agent value.

| Source subtype | Cases | Agent | Matched rule | Difference |
|---|---:|---:|---:|---:|
| Downside rebound | 9 | $0.00 | −$100.00 | +$100.00 |
| Upside expansion | 3 | −$100.00 | $0.00 | −$100.00 |

The rebound gain is an avoided comparator loss, not earned trading profit.
The agent rejected the March 7 trade that lost in the matched rule. Its separate
February 25 expansion trade lost $100 where the matched rule did not enter.
Those effects offset. Eleven matched baseline cases had no entry, so this is
mostly a participation test; it provides very little information about preserving
matched-rule winners. Three expansion observations are not a statistical sample.

## Every original decision and outcome

All decisions/times are UTC in 2026. No answer was rewritten after capture.

| Setup | Original decision | Agent replay | Agent net | Matched rule net | 90s rule net |
|---|---|---|---:|---:|---:|
| Jan 20 06:00 | Reject | No entry | $0 | $0 | $0 |
| Jan 25 09:00 | Reject | No entry | $0 | $0 | $0 |
| Jan 29 16:00 | Wait, rebound | Confirmation did not arrive before expiry | $0 | $0 | $0 |
| Jan 31 15:00 | Reject | No entry | $0 | $0 | $0 |
| Feb 23 02:00 | Reject | No entry | $0 | $0 | $0 |
| Feb 25 02:00 | Wait, expansion | Filled, then stopped | −$100 | $0 | $0 |
| Feb 28 07:00 | Reject | No entry | $0 | $0 | +$278.80 |
| Mar 7 20:00 | Reject | No entry | $0 | −$100 | −$100 |
| Mar 8 23:00 | Wait, rebound | Cancelled: insufficient room | $0 | $0 | $0 |
| Mar 22 22:00 | Reject | No entry | $0 | $0 | $0 |
| May 2 22:00 | Reject | No entry | $0 | $0 | $0 |
| May 3 23:00 | Reject | No entry | $0 | $0 | $0 |

## What the actual cases show

1. **A coherent bigger-picture thesis did not ensure a good executable trade.**
   On February 25 the agent identified local expansion inside a bearish larger
   context, waited above 65,953.2, and chose 65,744.6 as stop and 66,574.5 as
   destination. The modeled fill was 66,189.4 at 02:07. Stop was touched in the
   03:16 minute; $84.85 price loss plus $15.15 modeled costs equals −$100.
   Price later touched the proposed destination in the 13:49 minute, after
   falling as low as 64,720.4 following entry. That later move does not rescue
   the stopped trade. Entry, stop and thesis horizon need to work together;
   this does not justify widening the stop after seeing the answer.
2. **Response time can consume the entry opportunity.** On February 28, the
   90s mechanical rule filled at 07:03 and eventually made $278.80. With the
   measured 167.519s availability, the same rule's eligible confirmation came
   later and the entry-cap check cancelled it. The agent itself rejected the
   setup on sequence/context grounds. Both its rejection and operational delay
   deserve examination; do not claim the matched comparison captured this winner.
3. **A valid proposal can have no executable room left.** On March 8, the agent
   proposed a rebound toward the broken 4h floor at 66,508 with stop 66,002 and
   trigger 66,214. At the actual eligible price, the frozen net-reward/risk rule
   cancelled entry. A setup being plausible at the decision close is not enough.
4. **The agent was not merely applying a blanket bearish-parent veto.** Its
   two rebound proposals explicitly recognized bearish/broken parent context
   and still proposed conditional local rebounds. That demonstrates the intended
   kind of contextual interpretation in its stated rationale, not reasoning
   correctness or predictive edge.

These are post-outcome observations from the fixed run, not new optimized rules.
The next inexpensive step is to inspect these saved cases for trigger latency,
stop/timeframe consistency and available room before paying for another batch.
Do not retrofit the original answers, loosen gates to manufacture fills or expand
the sample until it becomes positive. Any changed method needs its own protocol
and new evaluation. This exposed practice batch cannot certify profitability;
raw OI/funding/fusion/Fibonacci, adaptive exits and the other archetypes were not
tested here. No automatic live promotion.

## Artifacts and verification

Local run: `results/lc_practice_2026_09_29/run_v1/`.

- `report.html`: 12 case cards, 12 embedded seven-panel charts, original
  rationales/citations, measured timing, outcomes and subtype scorecards.
- `case_results.json`: complete immutable replay records and accounting.
- `summary.md`: compact per-case comparison.
- `cases/*/assessor_response.json`, `delivery_receipt.json`, `capture.json`,
  `terminal.json`: original answers, receipts, exact capture bytes and terminal
  records. Requests, source bindings and all research-code hashes unchanged.

Manifest SHA256:
`1d51c49b68732f821160a2fd22a39bf9f2559826f47d62e5210fb5e75e53a86c`.
Terminal lock SHA256:
`81a3faaba692a3b97e4e4296e83343c99dcf5740be7bea26d754ae577df75f42`.
Result SHA256:
`3b012a50a3a646b3be60e8620f8c8f29fa20c00123aae444f6e909c1930a4c1e`.

Root verified all 12 original response bytes against captures, delivery receipts,
unique role identities, source/code/terminal/result locks, all five arm totals
and matched delta. Separately scanned raw minute-archive prices for all four
filled agent/mechanical arm-case records and confirmed earliest bracket exits,
fill opens, quantity/notional and 12bps PnL arithmetic. No ambiguous stop/target
bar among these four. All 12 cards/charts and Markdown entries present; real
February 25 chart visually inspected. Report generation and status exited zero;
final status pending0/inflight0/lockedtrue. Existing Arrow CPU-discovery warnings
did not prevent reads. No new source edits or software-review loop was needed.

This root accounting check is not a fresh independent quant/model assessment.
The prior 368-test result belongs to the implementation checkpoint, not a new
test-suite run today. Exact model snapshot and billed usage remain unknown.
Source data and run artifacts are local-only; code/docs remain uncommitted on
`quant/archetype-evidence-audit`, HEAD85923a4. No push or PR.
