# Thesis management lab engineering results

## What finished

The new offline lab is implemented and its January engineering replay is complete.
It keeps a range-based thesis separate from entry, follows ordered structural
events, and compares fixed with adaptive management. It does not establish a
profitable strategy or implement the entire trader curriculum.

This is a new established-range spring/test/strength/last-support hypothesis,
not a modification to the parked LC or R3 studies. All 17 native archetypes,
production exits, live orders, config and fusion are unchanged. No paid market
assessment calls, installs, downloads, commits, push or PR were performed. Quant
and software reviewers consumed ordinary session usage. Everything is local and
uncommitted on `quant/archetype-evidence-audit`, HEAD `85923a4`.

Read the [frozen design](../superpowers/specs/2026-10-02-thesis-management-lab-design.md)
and [implementation plan](../superpowers/plans/2026-10-02-thesis-management-lab.md).
The new modules are `thesis_contract.py`, `thesis_sequence.py`,
`thesis_execution.py`, `thesis_source.py`, `thesis_study.py` and
`run_thesis_study.py` under `scripts/research/`.

## What the lab can represent

One shared, causal market history supplies completed daily, four-hour, hourly and
minute evidence. A known four-hour range is bound before the spring candle opens;
its identity and levels never change retroactively. Daily context annotates
location rather than universally vetoing entries. The sequence requires a later
test, strength, first valid support retest and subsequent minute confirmation.
Failed entry progression does not erase the larger thesis record.

Adaptive management can reduce a quarter of original quantity at the range roof,
another quarter at the defined Fib extension, ratchet stops after confirmed hourly
pivots, and exit on invalidation, no progress or the common deadline. Old stops
remain active during replacement latency. Fees follow actual fill notional and
funding follows remaining quantity. Each action retains its evidence and clocks.

Fib price anchors, Fib review times and elapsed-time Gann-style clocks have
explicit project formulas. They are not authenticated hidden-Fibonacci or Gann
formulas from a trader. Pivot ranges are not proof of complete SC/AR/ST Wyckoff
accumulation. Macro/order-flow judgment, distribution/shorts, all-archetype
reasoning and a market-assessing master model are not implemented here.

## January source findings

The engineering origin window was January 2024 UTC, with December 2 warmup and
February 8 exclusive source tail. The pinned archive supplied 97,920 continuous
minutes. A quant reviewer independently reconstructed all 186 eligible four-hour
closes: 103 were not qualifying springs, 50 had no active prior parent, 31 belonged
to already-consumed lineages, and two originated raw episodes.

The January 3 16:00 episode tested at 19:00 and showed strength January 4 16:00,
but its first support touch failed January 5 02:00. The January 19 20:00 episode
never produced a valid test within 24 hours; its parent thesis was later
invalidated January 22 20:00. Neither generated a thesis-entry intent. No rule was
relaxed to obtain a trade.

## Hypothetical engineering outcomes

Each occupied book has its own capacity. All use the same raw episodes, $100
intended stop risk, $50,000 notional cap, 90-second processing delay rounded to
minute opens, 6 bps fee per side and assumed adverse funding of 8 bps per eight
hours. This funding is stress, not observed historical funding. OHLC market fills
are an all-or-none simulation, not authenticated exchange execution.

| Entry | Management | Fills | Net dollars |
| --- | --- | ---: | ---: |
| Simple spring | Fixed | 2 | -66.34 |
| Simple spring | Adaptive | 2 | -46.75 |
| Full sequence | Fixed | 0 | 0.00 |
| Full sequence | Adaptive | 0 | 0.00 |

The zeroes mean no entries, not proven successful selection. Two opportunities
are far too few to measure edge. The independent overlapping attribution runs
preserved identical entry prices, quantities and original stops within each
entry family; those shadow runs are not additional independent opportunities.

Adaptive net is $19.59 less negative, but gross trading performance is $12.41
worse. Approximately $31.99 less assumed funding, plus a negligible fee difference,
explains the net difference. This is reduced exposure under a funding stress,
not demonstrated forecasting skill.

On January 3, fixed management made $52.47 after costs while the adaptive trail
lost $28.10. On January 19, the fixed stop lost $118.81 including funding while
the adaptive no-progress exit lost $18.65. The tradeoff is visible: one later
winner was cut, and one larger loss was reduced. Neither package earns promotion.

Historical activity covered four effective stop changes, two Gann-clock reviews
and one no-progress exit. No partial reduction, Fib-driven action or thesis-entry
fill occurred. Those mechanisms are covered by synthetic software tests, not
historically demonstrated by this pilot. This is not the full economic study,
walk-forward validation, or an untouched holdout.

## Verification and review

The fresh focused run passed 167 tests in 4.01 seconds: 53 new tests and 114
existing research dependencies. The whole repository was attempted once and
stopped on the same three existing collection errors: missing
`engine.strategies.archetypes.bull.wick_trap_moneytaur` and two archived imports of
`FusionEngine`. These unrelated modules were not repaired.

Quant and software reviewers checked the written contract before implementation.
A fresh whole-implementation software review found an unfinished-watcher-to-zero
defect and a related expiry/unknown precedence defect. Four counterexamples failed
before fixes and passed afterward. The final run includes those regressions.
The quant reviewer then cleared the saved source before engineering economics.

The controller's separate audit, importing no thesis modules, reconstructed
2,108 candles, 231 confirmed pivots and both parent/ATR bindings. It reconciled
90 cashflow records across eight repeated-scenario positions, including actual
fees, remaining-quantity funding and intended-risk limits. Eight positions repeat
two opportunities; they do not enlarge the statistical sample.

Real source restart/future-append testing through January 15 reproduced the
entire prior packet and all 1,515 past catalog events exactly. Real execution
restart at January 3 21:01, with a stop replacement still pending, reproduced
the entire saved adaptive book exactly. Neither witness is another strategy
variant. Nine protected historical file hashes match the pre-edit values; all 71
bindings in the frozen LC source receipt also match. Production paths have no
tracked diff.

Source runtime was 1.178 seconds; engineering runtime was 10.996 seconds, both
below the 600-second per-stage limit. Artifacts occupy about 1.5 MiB, below the
256 MiB bound. Nonfatal PyArrow sandbox CPU-probe warnings occurred. No process
remains running after verification.

## Artifacts and continuity

Local artifacts are under `results/thesis_management_2026_10_02/`:
`source_v1/`, `source_review_v1.json`, `source_audit_v1.json`,
`source_prefix_witness_v1.json`, `engineering_v1/`, `economic_audit_v1.json`, and
`execution_restart_witness_v1.json`. Preserve them; do not overwrite or rerun to
search for better results.

Key file SHA256 values:

```text
source.json       fcf43d779641185f45b33d56c1872360997c9c1b32202976ced7e9c9ee39fa64
comparison.json   2f059c9e37e4c286171921262899aefc6a792a4292d51b0eafb1319fd4b08f3b
economic receipt  426b4b563d5fc08dd9e2d033640dd3e8bfa439eb22ade641f21eedd94fdf1878
economic audit    7c576f28a133d6b8b645cef3cdc188332271e2ab41200c6f79f820b535a0db71
```

The local minute archive and saved continuous parent ledger are dependencies;
GitHub alone does not supply them. This pilot consumes saved parent artifacts
rather than rerunning the external recovered helpers. Those helpers remain
necessary to regenerate the parent ledger itself. Developer continuity is in
PROJECT/MEMORY, not in outcome-hidden market-role packets.

## Next action and delegated choices

The next justified study is one separately reviewed **full-calendar source-only
census**, with origins `[2024-01-01, 2026-08-24)` and the already specified tail.
Keep the predicates unchanged and report sequence attrition, thesis-intent counts
by month, observable management clocks/destinations, coverage and restart checks.
Separate run-calendar metadata from hypothesis identity and preserve January's
bound files. Do not launch a larger economic campaign or tune this revealed sample.

That census determines whether a separately frozen economic test has enough
opportunities. Before that test, lock cost/delay stresses, the three component
ablations, paired estimands, uncertainty and stopping rules. Insufficient counts
remain insufficient evidence. A profitable baseline is not a prerequisite to
testing, but losses or sparse counts are not permission to loosen the hypothesis.

Delegated implementation choices: work inline on the user-selected existing
checkout to contain cost; retain local ledgers and make no commits without
authority; generate a scoped uncommitted review diff because a commit-range diff
would be empty. Their costs are potential local refactoring/review repackaging
and local-only portability until a separately authorized publication. No material
software review finding remains deferred in this milestone.
