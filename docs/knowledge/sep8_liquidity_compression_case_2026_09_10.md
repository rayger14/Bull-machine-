# September 8 liquidity compression: case evidence, not a fitted strategy

## Result and scope

Fresh read-only dashboard snapshots identify the requested position as `long_liquidity_compression_1788872400`, entry source label September 8 at 13:00 UTC / 06:00 PDT. Its three recorded exits total **+$721.73**. The immediately preceding wick-trap position eventually totals **−$1,462.65**, not the +$125.94 partial exit visible in the screenshot. Both are absent from the later heartbeat's open positions; their exit quantities reconcile to 100% in the recorded scale-out proportions.

These are native shadow/paper ledger records, not broker-verified execution or evidence of a profitable strategy. User selected the winner after seeing its outcome; the neighbour is an illustrative comparison, not a controlled experiment. Starting account equity and full net account return are not reconstructed here.

## Acquisition and reproducibility

GET-only sources: `/api/trades`, `/api/status`, `/api/signal-log`, `/api/candle-history` on `http://165.1.79.19:8081`. Status server time: `2026-09-10T21:03:44.917460+00:00`; heartbeat update: `2026-09-10T21:01:57.011151+00:00`. Downloads are separate, not an atomic server transaction. Snapshot has 535 exit rows; do not call these 535 independent trades.

Ignored, private artifacts: `results/research_validation_2026_09_10/sep8_case_audit/` (four original JSON snapshots plus `case_summary.json`). Summary SHA-256: `ecb982d5dffbb5296ff54ed811fcdb243a6dd23cd85afed8d8998c4df75a2123`.

| Snapshot | SHA-256 |
|---|---|
| trades | `3c6ad8619dd5d5ea22f382d346c8f58ff02071307973913a5a403fe845bcbf12` |
| status | `a395ba1788222553410684c6e2563d651820eec40b0f656fae129811541e9ed2` |
| signal-log | `662dbcdd64e1cf7f4af0ad4af8b0f11d32663e14fee7eb88741fa4feb2913f38` |
| candle-history | `fdaa3de355591f2e09eeadc78ea49b769a0c8abcaba2a0639eb14ce672e328b4` |

Candle snapshot has 200 rows, including three exactly identical duplicate pairs at September 2 17:00, September 4 05:00 and September 9 02:00 UTC. Raw files are preserved. A validated analytical view may explicitly deduplicate exact copies; they must not silently pass the replay grid contract.

## What differs between the neighbouring entries

Gate values below are rounded narrative telemetry, not verified raw consumed features.

| Evidence | Wick trap | Liquidity compression |
|---|---:|---:|
| September 8 source hour (UTC) | 12:00 | 13:00 |
| Signal close price | 78,427.90 | 77,944.00 |
| Ledger entry price | 78,451.42837 | 77,967.38320 |
| Initial stop | 76,801.76 | 76,933.20 |
| RSI | 38.4 | 31.2 |
| Logged volume Z | 0.52 | 3.28 |
| BB width | 0.0181 | 0.0203 |
| ADX | 32.7 | 34.0 |
| FVG flag | 1 | 0 |
| Fusion / entry threshold | 0.2805 / 0.3081 | 0.2651 / 0.3319 |
| Sum exit notional | $82,031.24 | $52,500.00 |
| Initial price-to-stop risk × total exit quantity | $1,724.94 | $696.38 |
| Recorded exit PnL | −$1,462.65 | +$721.73 |
| Recorded exit PnL / initial stop risk | −0.848 | +1.036 |
| Last-exit source-label duration | 48 hours | 20 hours |

Both signals explicitly say their fusion threshold was **bypassed for data collection**. The winner's fusion score is lower than the loser's. This one pair does not justify removing or reversing the fusion gate; it demonstrates that a winner is not evidence the gate passed or that its calibration is correct.

The LC source candle fell from 78,427.9 to 77,944.0, with high 78,528.9, low 77,602.0 and volume 7,488.25. It follows a lower-volume prior hour (2,895.86). This supports the descriptive statement “later, lower entry after a stronger selloff/volume event.” It does **not** prove absorption, a completed spring, a parent-range reclaim or the cause of profitability.

The previous hour's rounded BB width 0.0181 is consistent with the separate prior-compression hypothesis (≤0.06). It is not an observed-evidence H1 pass or a full-history counterfactual result.

## Management matters independently of the entry label

LC exits, all times source labels:

| Exit label UTC | Portion | Recorded reason | Recorded PnL |
|---|---:|---|---:|
| Sep 8 14:00 | 10% | Scale-out at 0.5R | +$36.01 |
| Sep 9 04:00 | 30% | Scale-out at 1.0R | +$233.00 |
| Sep 9 09:00 | 60% | stop_loss | +$452.72 |

The last stop-labelled exit is above entry, consistent with a moved stop protecting profit. Rows still display the original stop, so the exact stop ratchet/trigger cannot be recovered from these rows alone. Do not label the profitable final exit a loss, infer exact intrabar ordering, or infer actual exit fills from the reason text.

Wick trap exited 10% for +$125.94 at September 9 08:00 and the remaining 90% for −$1,588.59 at September 10 12:00. A partial winner became a completed loser. Comparing position-level outcome requires grouping all exits and checking open inventory.

The risk ratios above are **not fully net account R**: exit-row accounting does not reconstruct all entry commission, funding and other cash movements. Entry prices are numerically consistent with a 3 bp markup over each signal close; that arithmetic does not independently verify the server's historical slippage configuration or market execution.

## Timing and historical-version cautions

- Entry timestamps are source candle-open labels. The LC signal uses the 13:00 candle's close, which is not knowable until 14:00 UTC, plus any actual receipt/processing delay. The current host's delay policy does not establish the historical September 8 receipt time.
- Do not count the 13:00 candle's low as post-entry adverse excursion. No verified September 8 minute path is available in the recovered local minute dataset (it ends August 31).
- LC narrative logs RSI 31.2 but `rsi_extreme_65=0`; current local narrative code computes the flag using RSI >65 or <35. Current local runner also documents a September 8 volume-field telemetry change. These discrepancies prevent treating today's source as proof of the exact historical consumed gate values.
- Narrative BOS/BOMS are zero and do not prove a market-structure confirmation. A descriptive headline is not an event ledger.

## What this case says about the all-seeing eye

Keep five questions separate and attach an answer or explicit unknown to every candidate:

1. **Evidence:** Were the inputs required by this setup actually observed, fresh and consumed under the expected formula? Depending on the archetype these may include price, OI, taker flow or derived indicators; this is not a universal OI/taker requirement. Missing required input is not neutral confirmation.
2. **Parent structure:** Which confirmed higher-timeframe range and fixed boundary existed before the setup? A parent must have an identity, confirmation time and version.
3. **Sequence and location:** Did the setup-specific ordered sequence occur at a meaningful location? For a compression/sweep/reclaim candidate, that means prior compression, then sweep, then reclaim/acceptance. It is not the required sequence for every archetype. Simultaneous indicator agreement cannot substitute for event order.
4. **Entry economics:** At the first available decision, what entry was executable, what invalidates it, and is there room to the objective after costs? Evaluate hourly and minute triggers under the same parent context.
5. **Management and portfolio:** What happened to the whole position, how did stops/scale-outs evolve, and what correlated exposure already existed? Preserve all 17 native archetypes while comparing their independent contributions.

This is a shared context and decision record, not a super-score that averages every input. Different archetypes can correctly want different evidence at different phases. A spring's selling pressure during its sweep is not automatically a veto just because a continuation setup would want buying confirmation at entry.

## Next experiment, without fitting to this winner

First finish independently reviewed observed-evidence/prior-compression audits and the recovered-source causal parent ledger. Then bind an explicitly defined child event to a parent version strictly preceding its first sweep. Use fresh native-history replay for candidate enforcement so cooldown, deduplication and displaced trades are recomputed; post-filtering winners is not sufficient. Compare baseline, evidence-only, prior-compression-only and parent-context arms separately before combinations. Register the entry/exit/cost contracts and chronological holdouts before measuring improvement.

No parameter has been tuned on this case, no archetype has been promoted, and no live settings have changed.
