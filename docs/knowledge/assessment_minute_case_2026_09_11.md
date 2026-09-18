# One outcome-hidden minute structural assessment

## Result

One context-free assessor invocation evaluated the **earliest chronological candidate** in the frozen June 10–20, 2026 population of 76 minute events. Fixed parent hypothesis: 4H/N3, not chosen by profit. Sweep opened June 10 at00:36 UTC, reclaim bar at00:37 and decision at00:38. Inputs are derived historical event/ledger records, not authenticated live-feed receipts or raw-tick reconstruction.

The assessor received numeric levels, timestamps, instrument/stream identities, a pre-sweep active ledger snapshot and the complete (empty) intervening hourly transition list. Reference answers and existing H3 evaluator output were withheld; no PnL, exits, fusion or trade outcome was supplied. Input and independent reference were frozen before the call and reproduced unchanged afterward.

| Relationship | Evidence | Result |
|---|---|---|
| Larger structure precedes setup | Parent available June9 17:00, sweep June10 00:36 | pass |
| Same instrument and stream | Matching child/parent identities | pass |
| Sequence and decision clock | Sweep00:36, reclaim00:37, decision/availability00:38 | pass |
| Bound lineage survives under hourly ledger rules | Same active version/lineage at00:00; no hourly update through00:38 | pass |
| Child level in parent lower half | 61,150.20 <=61,519.00 <=62,664.85 | pass |
| Reclaim strictly inside parent | 61,150.20 <61,551.60 <64,179.50 | pass |
| Smaller level swept and reclaimed | 61,428.60 <61,519.00 <61,551.60 | pass |
| Larger floor breached? Diagnostic, not required | 61,428.60 is ABOVE61,150.20 | false |
| Source trust / execution readiness | Receipts and execution plan not supplied | both unknown |

Agent structural permission: **true**, matching `evaluate_h3_permission` PASS. Overall assessment: **unresolved**, because structural permission does not establish trusted inputs or an executable, managed trade.

All ten rule answers, citations, midpoint62,664.85 and sweep-to-decision duration2minutes matched the separately frozen reference. Root manually checked the prose. The agent kept child-level sweeps separate from parent-floor breaches; the prompt explicitly required this distinction, so it is not a spontaneous-discovery claim.

## Limits and implementation status

The empty transition interval proves only continuity under the supplied hourly update contract, not intraminute survival or authentic receipt timing. Parent versions and selected child events are supplied derived evidence; the agent did not reconstruct the complete pivot detector, prove first-sweep selection from raw bars, or independently validate all earlier lineage transitions. The deterministic H3 evaluator remains the stronger mechanical authority.

This is one already-researched case, not a pristine holdout, reliability estimate, measured latency test or evidence that an agent adds value over code. No trade was simulated; account capital, position risk and performance are not applicable. The parent high was not treated as a justified target. Missing execution evidence in this packet does not mean no execution logic exists elsewhere.

No reusable code, archetype, threshold, production setting or live state changed. Only a throwaway local extraction/comparison probe and tracked report were added. No full-suite rerun for this documentation-only tracked change; the previous implementation recorded382 passing research tests. No critic or second assessor; exact internal model snapshot/token accounting unavailable.

Private artifacts under `results/assessment_minute_case_2026_09_11/`: frozen `probe.py`, `input.json`, `reference.json`, `manifest.json`, saved `response.json`, one-case `validate.py`, `validation.json`.

- Source SHA256: `9f9587253250b56d547d2a5af17a2d02bc1a8035e86d5f962eae0b7e0f01ff20`.
- Input SHA256: `368d567c38af3c44d9831e1cad08bb7c371cf5c6cff8bf6b7abc610443c79984`.
- Saved response SHA256: `eb76b6a9ed94a9ca81f1842a7c6fa5922482beb2ce20e75b89b4a368c1ff8849`.

## Next smallest useful test

Use a clearly labeled synthetic negative control derived from this packet: move parent availability to exactly the first-sweep boundary (strict-before must fail), without changing prices. Check that structural permission flips while unrelated geometry remains unchanged. This tests causal-rule sensitivity cheaply; it is not a second historical observation or a new economic experiment. Keep the original real packet immutable. Work remains local, not automatically pushed to PR83.
