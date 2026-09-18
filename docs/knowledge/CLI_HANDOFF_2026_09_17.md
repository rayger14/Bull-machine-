# LC research: CLI restart checkpoint — September 17, 2026

This is developer/controller context, **not** an outcome-hidden assessor packet.
This document supersedes older next-action instructions in PROJECT/MEMORY.

## Goal and current verdict

Test whether an agent that understands trader teachings and structure within
structure improves BTC liquidity-compression (LC) entries over the engine alone.
Use daily/4H structure, hourly LC setup and minute confirmation. Keep archetypes
and comparison books independent. No profitable edge or live readiness has been
established. All 17 archetypes remain preserved; the immediate study is LC-first.

User authorized checkpoint commit, push and PR publication. Continue on
`quant/archetype-evidence-audit`; existing PR is
https://github.com/rayger14/Bull-machine-/pull/83. Do not merge/deploy automatically.

## Completed versus unfinished

- Completed: 31-month source census, 142 LC candidates; January–July 2026 has
  36 candidates, 16 prior-exposure exclusions and a frozen 20-case roster.
- Completed: source-bound packets and two-role immutable capture/grade lifecycle.
  Two cases (Jan20/25) received specialists plus critics: both rejected, both
  reviews found no material errors. Four actual role calls, no retries.
- Completed: authorized exploratory economics for Jan19/20/25. Under fixed
  $50k notional, 12bps and assumed 90s processing, immediate totals -$1,166.46,
  mechanical wait -$598.80 and agent -$103.29. Reject-all is $0 and beats these
  books. This tiny exposed sample does not establish an edge. Actual two-role
  paths took 7.68/8.58 minutes; actual credit usage is unavailable.
- Completed: September live-feature diagnosis. High-RSI/distribution also
  appears in large winners; a blanket veto is not supported. Dual-bearish
  oversold examples are too few. Historical minute availability and sizing
  attribution remain unresolved.
- Completed: `lc_setup_preflight.py` labels candidate downside-rebound versus
  upside-expansion theses and validates available context. All 20 frozen packet
  hashes matched: 13 downside/7 upside. Remaining 18 are 11 downside/7 upside.
  Evidence-ready is not trade-ready or authenticated live evidence.
- **Unfinished:** cheaper single-assessor contract/capture, the new controlled
  economic comparison, and integration into a live shadow-only service.

## Exact next implementation milestone

1. Read the context-preflight plan and report linked below; write a small new
   versioned single-assessor contract. Keep the old pinned two-role controller
   and runtime intact. Never synthesize critic approval. Distinguish `unreviewed`
   from `critic_reviewed` and choose any audit subset before calls/outcomes.
2. One specialist chooses from the fixed enter/wait/reject menu using validated
   predecision evidence. Code checks schema, citations, arithmetic, timestamps
   and legal plans. Missing evidence is not a valid reject or a $0 result.
3. Test capture/restart/failure handling offline first. Freeze the exact sample,
   budget, menu, audit policy, costs and latency treatment before new paid calls.
   Do not start paid work merely because old documents say to run all 20.
4. Compare native-immediate, mechanical-confirmation, agent-choice and reject-all
   in independent books with common sizing, stops, costs and horizons; account
   for actual decision delay. Lock decisions before revealing new outcomes.
5. Report winners preserved, losers avoided, missed opportunities, invalid
   assessments, net economics and limitations. Shadow-only integration comes
   after validated research plumbing; enabling live entry gates requires a
   separate decision, stronger evidence and authorization.

Jan19 is development-exposed; Jan20/25 are now outcome-exposed. Do not reassess
them or label the other 18 pristine holdouts: their status is only
`no_new_reveal_recorded`. Formal full-20 terminal/reveal gates remain closed.
Do not rerun the census, alter frozen dependencies or tune rules to these outcomes.

## Reading order

1. `docs/knowledge/lc_context_preflight_2026_09_16.md`
2. `docs/superpowers/plans/2026-09-16-lc-context-preflight.md`
3. `docs/knowledge/lc_three_case_economics_2026_09_16.md`
4. `docs/knowledge/lc_september_distinguishing_factors_2026_09_16.md`
5. `docs/knowledge/lc_assessment_runbook_2026_09_16.md` (historical two-role mode;
   not authority to launch or a completed single-role implementation).

## GitHub versus local-only continuity

GitHub contains research code, tests, plans and reports, not every dataset or
experiment artifact. A new CLI on **this computer and this checkout** can use
the ignored local files. A fresh clone elsewhere needs a separately authorized
private transfer plus hash/path verification before source-dependent work.

Local project root:
`/Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-`

Important ignored dependencies (paths relative to that root):

- `data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`
  SHA256 `5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`;
  coverage January2021–August2026, not September.
- `results/lc_consolidated_2026_09_15/run_v2/`: frozen source census.
- `results/lc_consolidated_2026_09_15/judgment_v1/`: source packets, requests,
  manifests and runtimes, including checkpoint captures.
- `results/lc_context_preflight_2026_09_16/`: actual 20-case preflight.
- `results/lc_exploratory_outcomes_2026_09_16/`: authorized exploratory scoring.
- `results/lc_september_comparison_2026_09_16/`: private live-feature evidence.
- `results/lc_jan19_validity_2026_09_15/run_v1/`: frozen earlier validity pilot.

Manifests bind hashes and may contain absolute paths. Do not silently regenerate
or rebase frozen artifacts. Follow their transitive source dependencies and
report missing files. Do not upload private data, keys, raw captures or credentials.
Unrelated graphify outputs and `scripts/research/rebuild_entry_population.py`
are intentionally excluded from this checkpoint.

## Verification and safe restart

Check branch/status/log first. Safe research regression command:

```sh
python3 -m pytest -o addopts= -q tests/research --maxfail=1
```

Publication verification: **1116 passed, 1 existing urllib3/LibreSSL warning in
1033.89 seconds (17m13s)** on September17. `git diff --cached --check` passed.
Execution-controller integration tests dominate runtime; allow about 18 minutes
for this full research command on this machine (duration varies).
Do not run the entire legacy suite blindly: `tests/test_batch_parity.py` invokes
a live-runner script and moves a trade log. CI alone does not certify this study.
Publication-time results are recorded in the PR checkpoint update.

Suggested instruction to a new CLI:

> Read AGENTS.md, PROJECT.md and docs/knowledge/CLI_HANDOFF_2026_09_17.md. Verify
> the existing research branch and local artifacts. Resume the separately
> versioned single-assessor milestone with offline tests first; preserve old
> experiments. Do not deploy, rerun paid roles or reveal new outcomes automatically.
