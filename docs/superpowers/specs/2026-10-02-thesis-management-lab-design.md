# Thesis and adaptive management research lab

## Purpose and limits

Build an offline, causal research environment that keeps a market thesis alive
across entry attempts and manages an actual simulated position as new evidence
arrives. The first playbook is **established range, spring, test, strength and
last support**. It is a new research hypothesis, not an LC repair, not all 17
archetypes, and not a certified complete Wyckoff accumulation detector.

The user approved building with delegated quant guidance. Reviewers check this
written contract and implementation plan before code. Work stays on the existing
quant branch. No live/config/fusion changes, paid market roles, installs, downloads,
commits, push or PR. Preserve every frozen LC/R3 result and module.

## Source fidelity

The [original teaching ledger](../../../research_notes/Trader%20origins%20and%20archetype%20fidelity/original_trader_sources.md)
supports known locations, ordered confirmation, originating-timeframe acceptance,
planned invalidation, level-to-level management, and indicators as supporting
evidence. It does not supply executable hidden Fib anchors or Gann conversions.
All constants/formulas below are declared PROJECT hypotheses. No profitability
or trader-authenticated formula is assumed.

The recovered N3 parent ledger identifies pivot ranges, not SC/AR/ST events.
Native Wyckoff no-context fallbacks and native scalar scores are not substituted
for a complete sequence. Native derivatives/regime models are not required here;
macro, funding observations and order flow remain unavailable, never invented.
Adverse funding below is an execution stress assumption, not observed funding.

## Fixed source and bounded launch

Use BTC and the local minute archive
`data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet`, SHA256
`5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035`.
Use the saved continuous parent ledger
`results/archetype_study_2026_10_01/census_v1/parent_ledgers.json`, SHA256
`e189c3f1dd4c284613994a1aae4cc0b396f6dc882482d97180f57e1c049c1966`.
Parent stream is `btc_1m_2021_2026_saved_5b8a4533f70b8ccd`.
Parent construction seeds December 2, 2023; aggregate/ATR seed matches it.

First engineering census: origins January 1 through January 31, 2024 UTC,
exclusive February 1; outcomes end February 8 exclusive. First deliverable is
source-only, with all raw episodes, event citations and entry intents. Source
review must clear integrity and counts without outcomes before engineering replay.
Then run exactly four primary books in one primary scenario for accounting and
behavior qualification, not edge selection. Maximum 600 seconds and 256 MiB output
per stage, one process, no automatic retries. No new market-model calls.

Future full-development calendar, not authorized by this implementation milestone:
origins January 1, 2024 through August 24, 2026 exclusive; source tail through
September 1. This history has already been exposed to research and is not a fresh
holdout. A full campaign needs saved-source clearance and an explicit reviewed
launch receipt. No tuning or sample extension in response to observed results.

## Shared evidence and raw episode population

Aggregate complete UTC 1h, 4h and 1d half-open candles from minutes. Candle value
is available at its close. Any missing constituent marks that candle unknown;
no forward filling. Reject invalid geometry, duplicate or unordered timestamps.
ATR14 on 4h is the arithmetic mean of 14 true ranges, using prior close;
require all 14 complete ranges. This explicit research estimator is not native ATR.
TR=max(high-low, abs(high-previous_close), abs(low-previous_close)); the first
bar without a previous complete close has no TR. ATR is available at 4h close.

At each 4h candle open bind the active 4H_N3 parent strictly before the open using
`parent_asof(..., strict=True)`. Keep the original version/lineage and L/H forever.
The first candle per lineage satisfying `low < L < close < H` originates an episode
at its close T0. Start consuming lineages only at the frozen census start; replay
warmup for indicators/context but do not silently count pre-period opportunities.
Every episode remains in the denominator even if stop/entry/sequence is unavailable.

Daily context uses the last complete daily candle and strictly prior active daily
parent when present: above/inside/below/boundary plus levels and IDs. Absent parent
is known absence; missing coverage is unknown. It annotates direction and nearby
destinations, never universally vetoes an entry. Parent hourly lifecycle and actual
4h close acceptance are distinct. IDs use causal values/identities, never archive
totals, end date, future bars or write time.

## Thesis and entry sequence

Long only, one entry maximum per episode, no pyramiding or re-entry. Initial stop
is spring low minus 0.1 times frozen ATR14_4H. The common thesis deadline is T0+7d.
A subsequent complete 4h close below frozen L invalidates the thesis. Missing
required source makes the thesis unknown. Either is terminal for future entry.
Stop touch while a proposal is pending cancels it; entry expiry does not erase
the thesis/history or prevent its later structural events being recorded.

Simple entry decision is T0, expires strictly before T0+15m. Thesis entry requires:

1. First later 1h test by T0+24h: spring low < low <= L and close > L.
2. First later 1h strength close by T0+48h: close > frozen spring high.
3. First later 1h candle touching strength threshold (spring high) by T0+72h:
   low <= threshold. If its close <= threshold, cancel the sequence permanently;
   otherwise it is last support. Bars must occur after the previous milestone.
4. A subsequent complete minute close above last-support high strictly before
   last-support availability+15m creates entry intent. Intents expire at that same
   deadline, not 15 minutes after confirmation. No same-candle chain advancement.

The phase record survives a failed entry sequence until thesis failure/deadline.
Milestone expiries prohibit later sequence advancement but are not fabricated
Wyckoff invalidation. Require observation start >= prior milestone availability.
Thesis decisions reference only events available at their decision clock.

## Defined price and time hypotheses

Fib price: A=spring low, B=first strength candle high. Anchors become available
at strength close, IDs immutable. Level(r)=A+r*(B-A) for .382/.618/1/1.618.
Retracements annotate location; 1.618 has a management role below. This is not a
claim to implement Moneytaur hidden Fib construction.

Fib time: origin=strength availability, delta=strength minus spring availability.
Review clocks are origin+{1,1.618,2.618}*delta, rounded up to the first complete
hourly close. Gann-style clock: T0+{24,48,72,96,120,144,168} hours UTC. It is an
elapsed-time cycle hypothesis, not square-of-nine/astronomy or authenticated Gann.
One activation per review, simultaneous clocks deduplicated with both citations.

## Adaptive policy

Protective stop is always active. Fixed management uses original stop, a standing
2-price-R target, and common seven-day deadline. Adaptive management has no fixed
2R target and uses the same initial risk and deadline:

- Exit remainder on thesis invalidation, unknown required structure, or deadline.
  Unknown structure yields a null economic result, not an assumed sell fill.
- At the first complete hourly close >= frozen range H, reduce 25% of original
  quantity after latency. Once Fib anchors exist, a close >= Fib1.618 reduces a
  further 25% of original quantity. If both occur together, one 50% reduction
  cites both destinations. Never retroactively fill at the trigger level.
  Only hourly events available strictly after fill may trigger reductions;
  pre-entry hits do not consume destination allocations.
- Confirm hourly pivot lows with two strictly higher lows on each side. On
  confirmation, propose pivot low minus frozen 0.1 ATR buffer. Only ratchet up,
  only to a level strictly below decision close, never increase size/reset R.
  The pivot center must open at/after T0; pre-episode pivots cannot trail a trade.
  Confirmation must be strictly after fill, but the center may precede fill.
  All five constituent candles must be complete; equal lows are not pivots.
- At each Fib-time/Gann review after entry, exit remainder if no new ordered
  milestone occurred since the prior review (initial boundary is entry time)
  and latest complete hourly close <= entry. Otherwise hold and advance the
  review boundary. Same-time milestone counts as progress. Pre-entry reviews do
  not start the position's no-progress clock.
  Progress kinds are test, strength, last_support and trigger only. A fully
  confirmed thesis entry has no later sequence milestones in v1; its reviews
  therefore test price progress. Review availability must be strictly after fill.

These rules fit evidence into roles rather than requiring all indicators to agree.
Destination states are untriggered/scheduled/executed/cancelled; repeated qualifying
closes cannot schedule the same original-quantity allocation twice. Unknown evidence
carries its first availability clock; it cannot retroactively erase earlier fills.
Three diagnostic ablations are supported: remove Fib-price reduction, remove
Fib-time review, remove Gann-style review. No automatic winner selection.

## Execution and accounting

Minute-OHLC all-or-none market fills only. Partial position reductions are modeled
orders, not claims about exchange partial-fill liquidity. Processing delay is
90 seconds, rounded up to the first minute open at/after ready time. Fee is 6 bps
per actual fill notional in primary (12 bps round trip near unchanged price).
Slippage is not independently authenticated; stress is 12 bps per side and
180 seconds delay. No bar's own close can fill its own open.
Standing stop/target orders have no new observation delay. All evidence-generated
full exits, reductions and replacements are ready at decision+delay, rounded up
to a minute open. The known maximum deadline is pre-scheduled and executes exactly
at T0+7d, with stop/funding precedence; it needs no newly generated evidence.

Intended stop risk $100; notional cap $50,000. For entry P, original stop S and
per-side fee f, q=min(100/((P-S)+f*(P+S)),50000/P). Gap losses may exceed $100.
Standing stops fill at stop or worse opening gap; fixed targets at target, no
favorable gap improvement. If stop and target touch same minute, stop wins.
Stops cannot be loosened; a pending replacement does not disable the old stop.
Funding stress charges remaining quantity times entry price times 8 bps at
00/08/16 UTC. Existing positions pay before same-clock exit; new entries do not.
Fees occur per actual fill, reductions use original quantity fractions, cumulative
quantity never exceeds original. Mark PnL includes realized and remaining unrealized
PnL less fees/funding; keep original R denominator unchanged.

At each minute open: funding on quantity held entering that clock; process the
completed prior minute against old orders (any discovered hit is attributed to
this clock); old-stop opening gap; consume newly available structure to cancel
unfilled intents and schedule delayed actions; due full exits; due partial
reductions; acknowledge due stop replacement; new admissions. An old stop touching a gap
beats its replacement. Newly effective stop applies only after its ACK/open and
may trigger against the current open if already marketable, never the prior bar.
Full exit cancels lower-priority changes. Full exits rank invalidation, deadline,
no-progress. A delayed full exit keeps stop/capacity live until execution.

Missing minute while occupied makes economics and future occupancy unknown for
that book; never release capacity on an assumed outcome. Known no-entry is $0.
Missing optional context does not poison a valid simple path. Structural unknown
poisons adaptive or unfilled thesis-dependent paths, not fixed simple positions.
Dependency matrix: simple/fixed needs execution minutes only after initial risk;
thesis/fixed also needs sequence 1h/4h evidence until fill, then minutes only;
simple/adaptive adds post-fill 1h/4h structure; thesis/adaptive needs both.
All unfilled intents cancel on a known 4h invalidation or required structural
unknown at/before fill. Initial unknown risk remains unknown for every arm.

## Interfaces and restart

All public records are finite JSON. `thesis_contract.py` owns protocol, IDs,
aware clocks, seals and validation. `thesis_source.py` builds a shared candle
catalog and episode packets. Packet fields: id, stream_id, parent, origin,
original_stop, atr4h, deadline, daily_context, events, entry_intents,
source_status, execution_authorized=false. Events contain id, kind, timeframe,
start/end/available_at, status, payload and input_ids; required citations resolve.

`thesis_sequence.py` consumes complete events into ordered milestones/entry
intents and Fib/clock definitions, without reading later outcome bars. Prefix
events/intents must be invariant under future append. `thesis_execution.py`
replays positions/books, with a JSON checkpoint containing bindings, cursor,
pending entry/exit/stop orders, remaining quantity, cashflows, consumed event IDs,
review boundary and unknown status. Same prefix reconstruction is permitted for
source restart; execution restart must continue reducer state without double fees.
Reject foreign source/policy/arm or modified seals.

`thesis_study.py` and `run_thesis_study.py` orchestrate source/engineering replay
and report common denominators, occupancy and cashflows. Source artifact hashes
and protocol hashes bind scoring; source-only receipt states economic books absent
and economic outcomes not computed. Its management tail contains future prices,
so it must not claim that all outcomes are hidden from a development reviewer.

## Comparisons and finish line

Primary cells: simple/fixed, thesis/fixed, simple/adaptive, thesis/adaptive.
First capacity-free attribution: for each entry policy clone exact fills/quantity/
stop into both managers. These overlapping shadow positions are not a portfolio.
Then four occupied books, capacity one each, independent state. Watchers reserve
nothing; accepted pending entry reserves until fill/cancel/expiry. Admission order
is decision time then stable episode ID. Never compare only mutually filled trades.

Report all raw IDs, known no-entry/unknown/busy, fills, net PnL, fees, funding,
exposure, drawdown, reason counts, partials/trails and per-episode paired deltas.
Distinguish destination triggers from executed reductions, scheduled from effective
trails, clock reviews/holds/exits and cancelled delayed orders.
Engineering January results certify software behavior only. Full-development
advancement would require >=50 thesis fills across >=12 origin months, positive
absolute net in both stresses, nonconcentration and uncertainty analysis with
calendar-block pairing. Failing those means unsupported/insufficient, not tuning.
No historical result certifies live readiness. Walk-forward applies if fitting
any parameters later; no fitted model or parameter search exists in this build.

Implementation finish: deterministic tests, causality and restart witnesses,
source-only January pilot independently reviewed, four primary engineering books
and identical-entry comparisons reconciled, final fresh review, precise handoff.
No claim that macro/order flow, full accumulation/distribution, all 17 archetypes,
trader-exact hidden Fib/Gann or an intelligent market agent have been implemented.
