# Operating the offline LC practice lab

**September30 update: this prepared run has completed.** All12 assessments,
terminal lock and actual replay/report are finished. See
[results](lc_practice_results_2026_09_30.md). Do not authorize, reserve or relaunch
any case in this run. Safe `status`/`report` reopening verifies existing records
without new market calls. The preparation/launch instructions below document the
original workflow, not permission for a new campaign or changed manifest.

This is the structure-first practice workflow, not the old fixed-menu campaign.
The real preset selects the frozen first12 of the existing20 saved source cases:
January20–May3,2026;9 downside-rebound /3 upside-expansion candidates. They are
exposed practice/development cases, not pristine holdout observations.

## Prepared real run — September 29

`results/lc_practice_2026_09_29/run_v1` is prepared, not assessed or scored.
All 12 sources verified; 9 rebound / 3 expansion. Manifest SHA256:
`1d51c49b68732f821160a2fd22a39bf9f2559826f47d62e5210fb5e75e53a86c`.
Requests are 151,836–156,227 canonical ASCII bytes each, 1,860,603 total,
before a host wrapper. These are byte counts, not billed-token estimates.
No billing/balance telemetry is exposed by the current tools; actual credits
and an enforceable credit cap are unknown. Authorize at most 12 Astra/high
assessment calls, one per case, no critics/retries, one in flight, 600 seconds
per attempt, only after explicit user launch approval tied to this manifest.
Implementation approval has not authorized any of these calls. At preparation
checkpoint: 12 pending, zero terminals/reservations, authorization false,
outcomes not unlocked. Recheck `status` before continuing.

Do not rerun preparation or edit `scripts/research/*.py` while using this frozen
run: all research modules are hash-bound. Documentation edits are not in that
code binding. Any genuinely necessary code change needs a separate preserved
run namespace and renewed launch disclosure, not resealing this manifest.

## What is automated

`scripts/research/lc_practice.py` connects source alignment, fixed paper policy,
agent-response validation/capture, chronological entry resolution, minute exit
replay, independent-case accounting and an offline HTML/JSON/Markdown report.
It does not dispatch models or place orders. A host controller delivers the
prepared request to each fresh outcome-hidden role and returns the raw response.

Source-only preparation is free of market-model calls and future-price scoring:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m scripts.research.lc_practice prepare --run results/lc_practice_2026_09_29/run_v1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m scripts.research.lc_practice status --run results/lc_practice_2026_09_29/run_v1
```

The real preset pins the original archive SHA and chronological roster. A
changed frozen input or code dependency is an error, not permission to reseal
an existing run. Preserve partial/failed runs; never overwrite them to obtain
better results. Public synthetic tests use temporary directories and need no
private archives or credits.

## Authorization and host bridge

Before real dispatch, show the user the actual manifest hash, complete request
sizes, available usage telemetry, and the bound: at most12 new assessor calls,
one per eligible case, Astra/high, no critics/retries/replacements, one in flight,
600seconds per attempt including delivery. A call ceiling is not a credit cap;
actual billing and exact model snapshot may be unavailable. Record explicit
authorization tied to this manifest. Implementation approval is not launch approval.

Start ONE persistent owner process and keep it alive through all reservations
and captures. With a PTY, pass each command as one JSON line and consume its reply:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m scripts.research.lc_practice owner --run results/lc_practice_2026_09_29/run_v1
```

Owner commands:

- `status`: no additional fields.
- `authorize`: `user_instruction` (actual user's approval) and `manifest_sha256`.
- `reserve`: `case_id`; returns the prepared request path, raw-byte hash and
  requested model/effort. `include_request_hex:true` also returns the payload.
- `attach`: `case_id`, `agent_id` from the host's actual fresh role dispatch.
- `capture`: `case_id`, `response_path`, `delivered_path`, `metadata`.
- `fail`: `case_id`, nonempty `reason` for an attempted dispatch that failed.
- `not_run`: `case_id`, nonempty `reason` for abandoning an unstarted case.
- `lock`: requires every roster case to have a terminal result.

Capture metadata has exactly these keys:

```json
{"agent_id":"actual host role ID","requested_model":"gpt-6-astra","requested_effort":"high","observed_model":null,"observed_snapshot":null,"delivery_complete":true}
```

Use null for genuinely unobserved model identity/snapshot, not a guessed value.
Contradictory observed model, wrong role, truncated/mismatched request, invalid
JSON or timeout remain explicit failures. Host-recorded metadata is not independent
provider attestation. Keep delivery evidence honest: `delivered_path` must contain
the exact request bytes actually made available to the assessor, not a shortened
summary claimed to be the full packet. Store the exact original UTF-8 response
without fence stripping, hash repair or after-the-fact decision rewriting.

The host must use a fresh, packet-only role for each case: no inherited controller
history, PROJECT.md, MEMORY.md, old answers, reports, other cases, price archives
or web access. The only allowed input is that case's prepared request, whose
curriculum/brief contains the outcome-free teaching material. If the host cannot
deliver it completely, fail that attempt without shortening evidence or retrying.
This restricted instruction boundary is not an OS-level sandbox guarantee.

Host dispatch and cancellation are external responsibilities. If an attempt
reaches600seconds, interrupt that role and record the failure; never accept a
late answer as if it arrived earlier. The owner additionally checks measured
latency at capture. Restarting the owner makes unfinished reservations interrupted
and null, never redispatchable. Continue only untouched cases. Each reply is
durable before moving on. No automatic strategy change follows any decision.

## Reveal, report and interpretation

Only after all terminal records are locked:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m scripts.research.lc_practice score --run results/lc_practice_2026_09_29/run_v1
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m scripts.research.lc_practice report --run results/lc_practice_2026_09_29/run_v1
```

Outputs: `case_results.json`, `report.html`, `summary.md`. The HTML embeds its
charts, needs no server/CDN, and is OWNER-ONLY because it contains future prices.
Rerendering invokes no model and cannot replace an unequal existing result.

Primary comparison uses the agent's measured availability for both the agent and
the mechanical structural rule, with the same90second minimum, risk100 including
modeled costs,12bps,15minute entry expiry and fill-relative24hour holding limit.
The mechanical90second operational control is separate. The legacy immediate
reference has different sizing/stop/2R target/deadline; its dollars are not a
matched test of incremental agent value. USDT/USD is assumed1:1; funding, market
impact, exchange lot rounding and actual execution are not modeled.

Cases do not compete for capital or suppress overlapping cases. Report sums are
independent hypothetical results, not a portfolio equity curve. Deliberate valid
rejects and verified no-entry outcomes have zero exposure. Missing data, invalid
answers, unsupported plans and transport failures are null. A rejected loss is
an avoided comparator counterfactual, not earned money. Full totals are null if
required observations are unknown; known subtotals state their denominator.

This first lab does not include raw funding/OI/fusion/Fibonacci inputs, adaptive
management, all17 archetypes, or a trained persistent trading model. It evaluates
one bounded interpretation layer. Passing schemas proves neither true reasoning
nor profitability. No automatic tuning, next campaign or live promotion.
