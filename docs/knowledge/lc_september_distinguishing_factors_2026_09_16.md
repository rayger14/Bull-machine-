# LC September winner/loser comparison

## Accomplished

Compared the September10 winner with September11/15 losers using current dashboard records and actual server feature logs. One independent research agent reviewed the initial evidence and supplied a short addendum after the entry features were recovered. This was outcome-aware diagnosis, not a blind trade assessment or new strategy backtest. No live engine, risk, fusion or entry configuration was changed.

## Distinguishing observations

| Logged entry feature | Sep10 winner | Sep11 loser | Sep15 loser |
|---|---:|---:|---:|
| Recorded exit subtotal | +$735.92 | -$1,115.15 | -$1,014.24 |
| Recorded exit notional sum | $52,500 | $52,500 | $65,625 |
| RSI | 26.17 | 73.49 | 29.16 |
| Hourly Wyckoff context | accumulation | distribution | accumulation |
| 4H net Wyckoff side | bullish | bullish | bearish |
| Daily net Wyckoff side | bullish | bullish | bearish |
| 20-hour range position | 0.109 | 0.967 | 0.150 |
| Prior12-hour return | -1.64% | +2.95% | -1.67% |
| Volume z-score | 4.08 | 3.53 | 4.04 |
| BB width | 0.0190 | 0.0282 | 0.0413 |
| Chop | 0.334 | 0.313 | 0.456 |

These are logged algorithm outputs, not independently verified Wyckoff phases or reconstructed causal parent bounds. Net side means comparing the logged bullish/bearish scores, not requiring every indicator to agree. Raw sides can both contain evidence even when arbitration zeros one side.

**Sep11:** hourly closes rose77,002→78,000.6→79,223 across two bars; the candidate is near the top of its20-hour range and tagged distribution. A bullish larger backdrop did not prevent a bad late-expansion entry. At its logged entry hour the older Sep10 position was scaling out near the new entry price. The working question is whether continuation had adequate room and a valid new entry sequence—not whether all high-RSI LC candidates should be prohibited.

**Sep15 versus Sep10:** both have low RSI, exceptional volume, a low range location, and hourly accumulation labels. Sep15 differs in bearish4H AND daily net scores, wider bands and greater chop. This is a local rebound thesis against adverse larger context. However, its candle recovered about51% of its range from the low versus about19% for the winner. A stronger hourly wick/reclaim requirement alone would favor the loser.

**Fusion:** Sep11's losing entry passed its threshold, while both Sep8/10 winners failed theirs. These cases do not support simply restoring the old fusion cutoff or inverting it.

## Broader falsification check

Before calculating additional contrasts, saved two named hypotheses without fitting any new numerical cutoff. Retrieved all24 explicit-ID LC exit groups with entry labels from June onward and attempted matching server feature records.21 had exactly one selected feature row; three earlyJune entries had none. All21 matched recorded entry price exactly as feature close×1.0003, supporting the timestamp association. This is not an authenticated ID-bound feature receipt or proof of complete positions.

- **High-RSI plus hourly distribution:** six matched groups, three positive and three negative, recorded subtotal **+$5,270.73**. August17/18/19 are winners, including the large August18/19 gains. A blanket distribution/overbought veto would remove substantial winners.
- **Low-RSI with both4H and daily bearish dominance:** only two matched groups, August11 and September15; both negative, subtotal **-$1,492.70**. This is a candidate risk-context interaction, not enough observations to fit or certify a gate.
- **Both daily side scores zero:**11 matched groups, seven positive, subtotal+$8,769.92. These cannot automatically be treated as bearish or trustworthy neutral context. Source code initializes scores tozero and also uses arbitration; historical availability, defaults and version changes need separate treatment.

All group results remain recorded-exit subtotals with incomplete closure/entry-cost provenance and changing sizing. They are NOT the PnL of an applied filter, because allocator, occupancy and displaced trades were not replayed. This is historical discovery; no untouched test claim.

## Independent agent's take

The agent agreed that Sep11 represents a different directional event from the two downside cases, and that Sep15 weakens simplistic oversold/volume/wick rules. With the recovered features, it identified local accumulation versus bearish larger context as the useful hypothesis. It advised research-only context annotation, explicit missing/default states, and separate testing by subtype. It did not endorse a production gate. The six-case distribution counterexample above was calculated by the lead afterward, not claimed as independently reviewed.

## What remains unresolved

- Current performance metadata says adapter_source=coinbase; local Coinbase adapter can fall back from perps to spot, so the label alone does not establish the historical instrument. Do not substitute Binance minute candles. The recovered Binance minute archive ends August31.
- Actual September minute sequence and independently reconstructed4H/daily parent lifecycles are not yet recovered. Current dashboard hourly history has200 rows/199 unique timestamps, starting Sep8 18:00; it is too short to authenticate all larger structure. Full server hourly features are available, but they do not create missing1m observations.
- Feature timestamps are candle-open labels processed after the candle closes, per local runner flow. They are not execution receipt timestamps. Causal tests must add bar-close availability and actual/assumed processing delay.
- Allthree selected entries have fib_swing_range_pct=0, retracement=.5 and zero time-confluence fields. This is not verified Fibonacci anchor/time evidence.
- Sep15 size is25% above the earlier trades. Local sizing applies allocation, caps and optionally a post-cap tape dial, but exact historical modifiers/config receipts are absent from these selected records. Do not attribute the increase to conviction or the dial without those receipts. Larger size explains part of dollar loss, not the failed direction.
- Server checkout HEAD at inspection:293280881d94b8ddc185f6e9b76a292669ce06f9. This is current metadata, not a deployed historical code attestation.

## Proposed engine application — design for approval, not deployed

First add a **shadow-only LC context record**, without changing native eligibility, sizing or orders:

1. Separate downside-exhaustion, upside-expansion and unresolved subtype using existing entry facts; preserve original candidate IDs and rule version.
2. Record hourly context,4H/daily raw and net sides, range position, recent return and supporting level facts. Store missing/default/availability information explicitly. Do not turn a zero score into a structural verdict.
3. Separate signal candle label, evidence available_at, assessment time and hypothetical executable time. Attach actual source/instrument identifiers, anchor timestamps and sizing modifiers.
4. Let one specialist choose from fixed immediate/wait/reject plans using validated facts. Code checks schema, citations, clocks and arithmetic. A separately fixed audit sample gets a critic; code alone does not certify semantic judgments.

Then test two context interactions in separate chronological LC books: downside rebound with/without higher-timeframe conflict, and upside expansion with/without a defined new-entry sequence. Keep stops/costs/sizing comparable, count missed winners/expired entries, compare simple-code confirmation and reject-all, and freeze evaluation dates before looking at outcomes. Do not implement a universal RSI, distribution, BB-width or higher-timeframe-agreement veto from this sample. This research has identified a candidate design, not validated a profitable change.

## Private artifacts

`results/lc_september_comparison_2026_09_16/`:

- `snapshot.json`: scoped GET snapshots, retrieved timestamp, four LC signals/eight exit legs and200 hourly candles; separate non-atomic requests. Trade factor-attribution display fields deliberately omitted from this scoped copy.
- `entry_features.json`: exact three feature rows and remote September JSONL SHA256 at read.
- `hypotheses_before_broader_check.md`: two fixed descriptive comparisons before viewing broader matching results.
- `broader_descriptive_join.json`:24 groups, selected entry fields, match counts and source-file hashes for June–September logs.

Initial attempts to print overly broad API snapshots exceeded output budgets and were not used as parsed evidence. The bounded snapshots above parsed successfully. All SSH operations were read-only; no service restart, deployment, order or remote file mutation.
