# One outcome-hidden hourly LC assessment

## Result

One context-free assessor invocation evaluated the sole selected LC candidate in the already-audited 240-hour historical replay: source hour June 14, 2026 at 21:00 UTC; decision at 22:00 UTC. This is a real-data **historical replay candidate**, not a verified live execution. Selection was fixed by its uniqueness in that window, not by profit. The period is previously researched and not pristine holdout data.

Unlike the preceding synthetic handshake, the assessor received actual numeric features and explicit rules, **without precomputed pass/fail answers**, fusion, native gate labels, position accounting, exits or eventual outcome. The reference answers and extraction manifest were saved before invoking it. Only the allowlisted input packet was supplied; no conversation history or further file/tool access was permitted after its initial read.

| Check | Supplied value | Reference and agent |
|---|---|---|
| Current volume z-score >=3 | 4.225450 | true |
| RSI >65 or <35, strict | 71.426069 | true |
| Current BB width <=0.06 | 0.022398 | true |
| Chop <=0.50 | 0.434101 | true |
| Immediate preceding completed-hour width <=0.06 | 0.017620, available at21:00 | true |
| Long price ordering | stop64,377.5113 < entry65,280.67 < target67,020.0868 | true |
| Trusted field-level receipt/formula provenance | Not supplied | unresolved |
| Identifiable prior larger structure | Not supplied | unresolved |
| Complete management beyond stop/target | Not supplied | unresolved |

Agent final assessment: **unresolved**. All nine rule responses and their citations matched the independent one-case checker. Nominal per-unit risk903.158716, reward1,739.416786 and reward/risk1.925926 matched within the registered numerical comparison tolerance. This is signal-price arithmetic, not position sizing, fees, structural room, obtainable fills or a backtest. Starting equity and average funded-position risk are not applicable; no positions were simulated.

The agent noted that RSI rose from45.748767 to71.426069 while the signal was long: the direction-neutral RSI gate passes, but does not itself justify the direction. It did not silently substitute an oversold-only rule. Root manually reviewed that explanation; prose semantics are not automatically verified. The prompt explicitly requested directional analysis, so this is not a claim of spontaneous discovery.

## Independent checks and limitations

Existing `evaluate_policy` reproduced H2 prior-width PASS and LC observed-evidence UNKNOWN on the supplied fields. Root checked the numeric thresholds against the inspected champion YAML and the strict two-sided RSI predicate. Those comparisons do not reconstruct every historical production gate override or prove historical live enforcement. Missing structure/management in this packet means unavailable evidence, not proof neither exists elsewhere in the engine.

This is still manual orchestration and a throwaway research probe, not a new reusable raw-feature assessment adapter. The earlier witness-handshake module is unchanged. Ordinary code computes the same nine answers; **no incremental agent value or profitable edge is demonstrated**. One case cannot measure reliability, model memorization, latency, prompt sensitivity or repeatability. Exact internal model snapshot/token usage were not exposed. No critic or second assessor was used.

## Private reproduction artifacts

`results/assessment_real_case_2026_09_11/`: `probe.py`, `input.json`, `reference.json`, `manifest.json`, `response.json`, `validate.py`, `validation.json`. Extraction and reference reproduced unchanged; response comparison passed. No reusable code or production configuration changed, so the full suite was not rerun for this document-only tracked change; the previous pilot recorded382 passing research tests.

- Historical source SHA256: `418ab56dd2a774343fb98a81739e72976ffc267ee73600756f01d1935d855167`.
- Input SHA256: `93962343ad4d424ee8e974111d5c5e2f2fda7b9181febee9d4de280c67b1893d`.
- Saved response SHA256: `d3e79eab4bdeb56c8d8c9616e9b27ea6e1e9d3cb49d0c0597d75a2029d41a563`.

## Next small step

One outcome-hidden minute candidate with actual parent/child timestamps and explicit location rules, selected by a fixed chronological rule rather than outcome. Test whether the assessor can derive sequence/availability judgments from raw fields; retain deterministic answers and no trade authority. Do not expand to17archetypes, tune RSI direction, or deploy from this result. Work remains local; no PR update in this step.
