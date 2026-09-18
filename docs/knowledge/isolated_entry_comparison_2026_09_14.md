# Isolated code-versus-agent entry comparison — September 14, 2026

## Scope and source recovery

This executes the [frozen comparison](../superpowers/specs/2026-09-14-isolated-entry-comparison.md),
not all seventeen archetypes or a full native-engine management backtest. April
2026 is exposed development data, not an untouched holdout. Hourly liquidity
compression and the separate minute equal-low sweep/reclaim research family
have independent `(archetype, track, variant)` books.

The original temporary Binance minute archive had disappeared. Recreating its
original January 2021–August 2026 monthly concatenation recovered exactly
2,979,360 rows and the original parquet SHA256:
`5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Permanent local storage is
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`;
the original temporary path now resolves to it through a symlink. The different
local Databento file was not substituted. Source downloads and checksums are
recorded in the local recovery receipt.

The unchanged source engine processed 1,440 hourly updates with March warm-up.
The regenerated `prepared_sources.json` is byte-identical to the original pilot
artifact (SHA256
`077e4ee17ec25b7ad39ff6d07c2ba0980ea3e9e2b16d41f115d99ebd6eb91854`).
All 245 full candidate envelopes are now retained: three hourly and 242 minute.
The existing ledger, classifications, counts and three original selected
payloads reconcile exactly. Previously unretained full envelopes are newly
regenerated evidence under unchanged inputs, not compared with nonexistent old
full payloads.

The capture driver's final live-object digest failed on pandas timestamp/string
and tuple/list representation differences. A separately hashed private finalizer
reloaded both sides through the same persisted JSON boundary and reconciled them;
no selector, price, gate or original artifact was changed. The resulting
`source/full_population.json` SHA256 is
`99637b0372a2548a78a9a3f68b9b97531285d6f9998ae086f69ef64296c7f6e7`.
The local capture driver is not a standalone clean-clone reproduction workflow.

## Experiment conditions

Four cases were selected before assessment: the first two previously unassessed
candidates per track, without filtering on structural permission, native signal,
plan validity or future outcome. Three previously assessed candidates were
excluded and 238 remained outside the fixed model budget. The code references
also cover the full 245-candidate cohort, reported separately from the four-case
agent comparison.

Each fresh assessor receives completed 1m/5m/15m/1h/4h/daily evidence, the
master-authored specialist rulecard, evidence IDs and a sealed finite entry menu.
Each independent reviewer receives the exact bound assessment. Requests specify
`gpt-5.6-sol`, high; actual snapshot and billed tokens are not observable here.
No retries, answer repair or replacement cases are allowed. Role receipts prove
captured delivery, not model attention or enforced filesystem isolation.

Every arm uses $50,000 fixed notional, $60 assumed round-trip cost, 90-second
assumed processing delay, zero routing delay, a 15-minute exclusive entry expiry,
the original fixed stop, an actual-entry-derived 2R target, and an absolute
24-hour hourly / four-hour minute deadline. Books allow one active position;
pending waits do not reserve capacity. These are unfunded research brackets,
not production-management parity or certified fills. Funding, impact and
inference expense are excluded, and model latency is assumed rather than measured.

## Results

### Actual model delivery and review

Exactly four fresh assessors and four fresh reviewers completed, with 107 valid
captured chunks, all twenty fixed factual echoes correct and no deterministic
assessment/review schema errors. All four semantic reviews nevertheless failed.
Every validated agent plan and full-policy PnL therefore remains **unavailable**,
not zero, a rejection, or an avoided loss. The replay's zero known closed-trade
subtotal is not valid agent performance.

Independent source-only quant review confirmed the following interpretation
before reveal; no frozen response or grade was repaired:

| Case | Raw choice | Review failure |
|---|---|---|
| C01 hourly, Apr 8 00:00 UTC | Wait above latest 15m high | Claim-local citations, including teaching; status describes unresolved H1 rather than support for the whole sentence |
| C02 hourly, Apr 29 19:00 UTC | Wait above latest 1m high | Missing separate LC-H2 prior-compression assessment; missing teaching citation for interpretation |
| C03 minute, Apr 1 01:36 UTC | Wait above latest 1m high | Claim-local event, child-level and parent-bound citations; whole-sentence status mismatch |
| C04 minute, Apr 1 09:11 UTC | Reject | Correctly described absent parent, but labelled that supported description `contradicted` |

Most failures concern evidence attribution and response semantics, not wrong
prices or an illegal entry menu. C02's omitted required condition is the clearest
substantive completeness failure. Thus zero validated responses does not mean
all four market interpretations were factually wrong; it does mean this frozen
agent/reviewer interface is not yet dependable.

### Full-cohort code references — not full-cohort agent performance

All money figures below use the same $50,000-per-position research convention;
they are not returns on a specified account. Wins/losses count closed trades only.

| Track / arm | Candidates | Trades | Wins / losses | Mean initial price risk | Gross PnL | Assumed costs | Net PnL |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hourly immediate cohort | 3 | 3 | 1 / 2 | $874.88 | −$483.37 | $180 | −$663.37 |
| Hourly fixed 5m-high wait | 3 | 2 | 1 / 1 | $1,151.70 | −$163.12 | $120 | −$283.12 |
| Hourly native pre-winner | 3 | 3 | 1 / 2 | $874.88 | −$483.37 | $180 | −$663.37 |
| Minute immediate cohort | 242 | 178 | 67 / 111 | $134.41 | +$1,513.45 | $10,680 | −$9,166.55 |
| Minute fixed 5m-high wait | 242 | 85 | 33 / 52 | $193.70 | +$892.85 | $5,100 | −$4,207.15 |

Hourly immediate and native pre-winner happen to coincide in this three-case
cohort; that does not establish general equivalence. The hourly wait expired
once. Minute immediate had 61 busy skips and three pre-entry cancellations;
minute wait had 25 busy skips, 90 expiries and 42 cancellations. All supplied
code books resolved without indeterminate admissions.

The minute baseline's modest positive gross result is overwhelmed by the fixed
12bps round-trip assumption. Waiting cuts trade count from 178 to 85, but net
loss per entered trade changes only from $51.50 to $49.50. Lower aggregate losses
are therefore not evidence of a profitable timing rule. These findings apply
to the stated costs, candidates and brackets, not a measurement of actual live
venue fees or the engine's native management.

### Same four cases — raw choice diagnostic only

The following raw-choice column deliberately bypasses failed semantic review,
as preregistered. It is **not validated agent PnL**. No rejected or expired choice
is counted as a winning trade, and no validated avoided-loss credit is awarded.

| Case | Immediate reference net | Fixed 5m-high wait net | Raw agent choice and hypothetical outcome |
|---|---:|---:|---|
| C01 hourly | −$670.36 | −$771.87 | 15m-high wait expired; no trade, $0 |
| C02 hourly | +$475.83 | +$488.75 | 1m-high wait entered 19:05 UTC; deadline exit, +$488.75 |
| C03 minute | −$263.10 | $0, expired | 1m-high wait entered 01:45 UTC; stopped, −$253.20 |
| C04 minute | −$167.16 | $0, expired | Raw reject; no trade, $0 |

The hourly selected immediate book totals −$194.54 (one win, one loss), while
the fixed wait totals −$283.12 (one win, one loss). Its raw-agent diagnostic is
+$488.75 from one trade with $919.07 initial risk and one expiry. The minute
selected immediate book totals −$430.27 (two losses); fixed wait makes no trades.
Its raw-agent diagnostic is −$253.20 from one stopped trade with $193.20 initial
risk and one rejection. These tracks are not combined into a portfolio total.

This is mixed diagnostic evidence: the raw hourly choices look better on these
two cases, while the raw minute choice underperforms the simple fixed-wait
reference. Four exposed development cases cannot establish incremental edge,
calibrated confidence, profitability or superiority to deterministic code.

## What this changes, and the next bounded step

We now have an actual isolated economic comparison and a specific problem to
address, rather than only passing software tests. The current minute entry
family does not pay its assumed trading costs; adding confirmation alone did
not solve it. The agent can distinguish missing larger structure and choose
different confirmation levels, but its required evidence reporting fails too
often to evaluate a validated policy here.

For a **new protocol version**, separate mechanically derived facts and explicit
named-condition results from the agent's market interpretation. Represent
condition state (`pass/fail/unknown`) separately from whether a sentence is
supported, and bind reusable evidence groups without asking the model to repeat
the same citations in every sentence. Keep substantive missing-condition,
chronology, invented-evidence and illegal-plan failures blocking. This is a
proposed interface correction, not implemented or applied to these grades.

Then freeze one chronological follow-on comparison, equal hourly/minute model
budgets and unchanged independent books, with all decisions locked before price
reveal. Select the period using the prior-exposure registry and actual source
coverage; do not call an already inspected period an untouched holdout. Include
the fixed baselines, net-cost economics and latency assumptions. Set an explicit
assessment-validity stop rule before new model calls. Wider walk-forward or
CPCV evaluation comes after a usable policy and adequate independent event
coverage—not by claiming this four-case development sample is either method.
No new date window, threshold or winning archetype is selected by this report.

## Locked provenance

| Artifact | SHA256 |
|---|---|
| `precall_lock.json` | `1f0abcb24b03cd1328bf8a8d851221afdbb8ee9816640c8eb31ccf97d4d5acb5` |
| `grades_before_outcomes.json` | `9baffe8983e5520e5c0ff82a314b98e9302c0f99e917608d62dc68e00a408fb7` |
| `all_preoutcome_lock.json` | `51e18b90806d0d9f6cde25d6e0e45db6be0987d5348cc4e7f00ce628a1dfbcfc` |
| `role_transport_lock.json` | `8c6a0ee12530befdef32ae6edfb5f54d66854e4c3bee6b5e491f90043cf55379` |
| `results.json` | `a9a9f7d4ba69e170fc013cc0de709bb4a4d066c70df9af3969f7cc50b852563c` |

Independent post-reveal quant review verified all eight runtime receipts and
source/precall/role/pre-outcome/transport locks, reconciled individual position
arithmetic and aggregate counts, and repeated the registered replay with the
same result hash. No accounting or integrity discrepancy was found.

## Local reproduction boundary

Detailed source, packets, captured role returns, immutable responses, hash locks,
private harness and economic replay live in ignored
`results/isolated_entry_comparison_2026_09_14/`. They depend on earlier ignored
pilot helpers as well as the recovered archive. This report alone on GitHub is
not the full experiment dataset. Original pilot artifacts remain unchanged.

The final local verification command is:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research results/isolated_entry_comparison_2026_09_14/test_experiment.py results/isolated_entry_comparison_2026_09_14/test_rebuild_entry_population.py
```

It passed 694 tests with one existing urllib3/LibreSSL warning. This verifies
software behavior, not profitability. No production configuration, live orders,
fusion policy or archetype definitions were changed by this experiment.
