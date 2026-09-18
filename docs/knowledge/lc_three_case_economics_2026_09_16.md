# Three locked LC decisions — exploratory economic results

User authorized revealing these examples now, overriding the earlier full-batch reveal restriction for these cases only. No new model calls. Original decisions, menus and campaign locks remain unchanged. January 20/25 are now outcome-exposed, not an untouched validation sample. January 19 was already exposed development.

## Main result

Independent hypothetical $50,000 long trades, 12bps round-trip costs ($60), 90-second processing rounded to the next minute open, original stop, 2R target based on actual simulated entry, 15-minute entry expiry, original 24-hour deadline:

| Decision UTC | Immediate entry | Mechanical wait | Agent choice | Agent net PnL |
|---|---:|---:|---|---:|
| Jan19 01:00 | +$5.37 | -$103.29 | Wait | -$103.29 |
| Jan20 06:00 | -$663.80 | $0, trigger expired | Reject | $0 |
| Jan25 09:00 | -$508.02 | -$495.51 | Reject | $0 |
| Sum of these three nonoverlapping examples | -$1,166.46 | -$598.80 | — | -$103.29 |

Agent decisions improved this illustrative sum by $1,063.17 versus immediate entry and $495.51 versus mechanical waiting. This is loss reduction on three examples, not a profitable strategy or statistically established edge. Reject-all would have made $0 and beaten all three arms here; there is no demonstrated ability to accept winners.

- **Jan19:** confirmation occurred and the wait plan filled at 01:12 at $92,756.40. Neither bracket barrier exited the trade; the deadline open on Jan20 at 01:00 was $92,676.10. Net loss $103.29. Immediate entry at 01:02 at $92,555.10 was $108.65 better. Thus a critic-passed, plausible conditional-entry judgment was not economically better on this example.
- **Jan20:** immediate entry at 06:02 at $91,556.80 hit its $90,451.15444737271 stop at 14:47. Rejecting avoided that loss. However, the simple confirmation rule never triggered before 06:15, so no incremental benefit over mechanical waiting was established here.
- **Jan25:** immediate entry at 09:02 at $88,344.40 and confirmation entry at 09:10 at $88,322.10 both hit the $87,552.79540635632 stop at 16:20. Here rejection avoided losses that both baseline policies incurred.

## Sensitivities and limits

All registered combinations of 12/24bps and 90/300 seconds were scored. At 24bps, each filled trade costs an extra $60; skips still have zero exposure. January19 immediate becomes a loss, while the qualitative ranking of total results remains agent, mechanical wait, immediate. These are sensitivity calculations, not new tuned strategies.

Measured complete decision delays were rounded up to461seconds for Jan20 and515seconds for Jan25. Additional equal-delay alternative scenarios were also computed: immediate lost $664.45 and $482.43 respectively at12bps; wait still expired Jan20 and lost $495.51 Jan25; agent remained rejected. January19 has no verified measured timing in this comparison, so no measured-delay result was invented. All fixed-delay results remain hypothetical, not demonstrated live latency.

OHLC simulations assume the specified minute open fill, conservative stop-first resolution if both barriers touch, gap-through-stop handling, and no real order-book execution. No ambiguous stop/target bar occurred in these results. Funding, market impact and account-level capital/margin constraints are not modeled. The 12/24bps charges are assumptions, not measured execution costs.

## Verification and reproducibility

- Read only the three authorized decision-to-deadline outcome windows; each has exactly1,441 consecutive minute rows including the deadline open.
- Recovered source archive SHA256 matches the original pilot pin: `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
- Existing `score_conditional` results were cross-checked against existing `replay_sleeve` for all48 case/scenario/arm rows: matching net PnL, entry time and exit reason for filled trades, and matching resolved zero exposure for nonentries. These are two existing accounting paths, not an independent external audit.
- Existing conditional-entry, case-outcome and occupancy regression suites passed.
- Exact command, tool output, full results and input hashes saved locally under `results/lc_exploratory_outcomes_2026_09_16/{execution_receipt,results}.json` (ignored research artifacts). First calculation attempt failed only at final archive-hash serialization because this Python lacks `hashlib.file_digest`; streaming SHA256 was used successfully. No decision was rerun.
- Formal20-case terminal/reveal gates remain closed. This separate user-authorized diagnostic does not falsely mark the other18 cases completed.

## Next step

The result supports testing whether context can improve filtering, not tuning a gate from these examples. Use the next chronological unseen cases with the same evidence/menu and a prospectively recorded single-specialist policy if approved. Let code validate schema, citation references, arithmetic and legal plans; those checks do not certify semantic judgment. Reserve independent critics for a fixed audit sample rather than every case. Do not claim the existing two-role controller already implements this single-role mode. Keep these revealed outcomes out of future assessor prompts and explicitly preserve exposure accounting. Include mechanical-wait and reject-all comparisons so avoiding every trade cannot masquerade as an edge.
