# Thesis entry rule source fidelity

**Bottom line:** the frozen range → spring → test → strength → support → minute-entry policy is implemented faithfully, but it is only partly faithful to the teachings. No implementation bug or specification mismatch was found. The teachings support starting from known location and timeframe, waiting for confirmation, and defining invalidation and destination; they do not supply this exact sequence, geometry, or clock. The policy is a **project hypothesis**, not a trader-authenticated formula or complete Wyckoff model. Its three signals are too sparse for the planned study and say nothing about profitability. The justified next move is one bounded specification rewrite, not code changes or threshold tuning.

## The compiler enforces the frozen contract

The code applies the milestones literally: the hourly test reaches the parent floor; “strength” closes above the **spring candle high**; the first later touch fails permanently if it closes at or below that level; and entry requires a minute close above the support candle’s high before the 15-minute deadline. The 24/48/72-hour gates match these branches ([specification](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/docs/superpowers/specs/2026-10-02-thesis-management-lab-design.md:89>); [compiler](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/scripts/research/thesis_sequence.py:100>)). Nothing justifies a code repair.

This is fidelity to the project contract, not an authenticated recipe. The specification calls its formulas project hypotheses and says the parent range does not detect the events that establish a Wyckoff accumulation range ([scope](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/docs/superpowers/specs/2026-10-02-thesis-management-lab-design.md:16>)).

## Teaching support stops at the concepts

The reviewed material supports distinct timeframe roles, horizon-appropriate confirmation, body-versus-wick acceptance, and planned invalidation/destinations ([source ledger](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/research_notes/Trader origins and archetype fidelity/original_trader_sources.md:17>)). It does **not** verify the N3 range constructor, first-spring-only admission, one-touch cancellation, 24/48/72-hour deadlines, ATR offset, seven-day cap, or 15-minute trigger.

Two labels especially overstate what the rules detect. “Strength” only clears the spring high, not the range high. “Minute confirmation” compares a minute close with an **hourly** candle high; it does not identify a minute-native swing, range, or change of character. The frozen sequence also omits relative volume, spread contraction/expansion, and phase evidence. Those omissions matter because the reviewed Wyckoff educational material ties tests, signs of strength, and last points of support to both price behavior and volume/spread behavior ([Wyckoff Analytics](https://www.wyckoffanalytics.com/wyckoff-method/)).

This conclusion is bounded to reviewed material. The October 1 signed-in ledger remains valid, but the later spot check retrieved neither Moneytaur’s text (403) nor the Crypto Chase transcript; no exhaustive account archive is claimed ([audit limitation](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/research_notes/Thesis entry rule source fidelity/source_audit.md:30>)).

## The census shows scarcity, not bad trades

The rules produced **183 raw episodes, 17 support rows, and 3 entry signals across 32 origin months**. Only three months had a signal, so this cannot meet the floor of 50 fills across 12 months; signals are only an upper bound on fills ([census](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/docs/knowledge/thesis_census_results_2026_10_03.md:3>)).

All eligible minute candles were present. Fourteen of 17 rows correctly expired; two rows shared one support window, leaving 16 distinct windows ([trace](</Users/rayghandchi/Bull Machine/Bull-machine-/Bull-machine-/docs/knowledge/thesis_support_trace_2026_10_03.md:7>)). Expiry does **not** show those hypothetical trades lost, and two near misses do not justify relaxing the gate after seeing the data.

## Write one bounded replacement specification

The next justified action is to specify one long **support-reaction playbook inside identified higher-timeframe location**. Parent context should determine which setup applies; genuine minute child structure should supply the trigger; volume and price spread (candle range, not bid–ask spread) should corroborate or challenge the reading; and invalidation should terminate it. These are different evidence roles, not a demand that every indicator agree. Define unknown evidence explicitly rather than treating absence as confirmation. Give the child swing or zone its own identity and clock, and state whether entry occurs on its break or return. Mark every unsupported constructor, exception, stop, and expiry as a new project choice.

This audit provides no basis to launch implementation, a full backtest, changes to all 17 archetypes or native LC, or any claim that the proposed structure is profitable. Preserve the frozen census as translation evidence; review the new specification and validation plan before any code or experiment.
