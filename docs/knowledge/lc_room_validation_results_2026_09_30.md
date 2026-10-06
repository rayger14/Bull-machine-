# Known-room LC rule: mechanical replication

September 30, 2026. Completed: frozen discovery comparison and full-August
additional-month replication, with independent verification. Recommendation:
do not enable the known-room rule as a mandatory LC gate. Keep it as recorded
context/research evidence; leave the baseline and live engine unchanged.

## Discovery result: a trade-off, not an outright improvement

The frozen gate ran in its own book, not merely a filtered trade-summary table.
All four unchanged baseline ledgers reproduce exactly. Primary12bps/90seconds:

| Measure | Unchanged upside LC | Known-room gate |
|---|---:|---:|
| Supplied upside candidates |68|68|
| Filled trades |68|14|
| Wins / losses |28 /40|9 /5|
| Net simulated dollars |$9,384.99|$4,367.77|
| Mark-to-market maximum drawdown |$6,737.06|$2,215.94|
| Dollar profit factor |1.373|2.282|
| Mean cost-inclusive R per fill |0.1483|0.4412|
| Mean R per supplied candidate |0.1483|0.0908|
| Net after removing top3 winners |$1,106.37|$212.41|

The gate avoids35 losers but misses19 winners, preserving9 winners and5 losses.
Drawdown falls about67%, while net profit falls about53%. At14 trades over31
calendar months, frequency is about0.45/month. This is a sparse candidate for
lower exposure, not evidence of consistent income or a dominant strategy.
The filtered book's yearly dollar contributions are2024+$2,602.55 (2trades),
2025−$547.57 (5),2026throughJuly+$2,312.79 (7). Its descriptive monthly-bootstrap
95% mean-R interval is−0.3323 to+1.0463, versus baseline−0.0838 to+0.4031;
both include zero, and neither accounts for strategy selection. Twelve of31
calendar months contain a filtered trade. Smaller drawdown partly reflects much
less exposure and does not itself establish better entries or a tradable edge.

| Fixed sensitivity | Baseline net | Gate net | Gate minus baseline |
|---|---:|---:|---:|
|12bps /90s|$9,384.99|$4,367.77|−$5,017.23|
|24bps /90s|$5,304.99|$3,527.77|−$1,777.23|
|12bps /300s|$8,480.22|$5,081.12|−$3,399.09|
|24bps /300s|$4,400.22|$4,241.12|−$159.09|

Both remain aggregate-positive in these four modeled scenarios. The gate has
less dollar profit in every scenario. Per-supplied-candidate R is lower in the
first three, but higher in the24bps/300s case (0.0794 versus0.0540); do not hide
that difference between fixed-notional dollar and risk-normalized accounting.
No optimization or reinvestment of freed capital was performed.

Do not rescue the rule with new thresholds after these results. Discovery-check result SHA256:
`1846f6fcd225a8760dd3e112833b23b6566861b0b337295e730c59ce40fb008f`.

## August result: the gate rejected a winner

The complete month contains four native long LC candidates: August10/11 rebound
setups and August17/19 upside-expansion setups. The two rebound cases remain in
the source census but are excluded from this upside-only book by the frozen
subtype definition, not by their outcomes. Both upside24-hour windows are complete.

| Upside decision, UTC | Frozen room classification | Gate decision | Baseline net,12bps/90s |
|---|---|---|---:|
|August17 16:00|at least2R|enter|+$461.63|
|August19 13:00|below2R|abstain|+$678.51|

August17 fills at16:02 for64,154.60 and exits at the next day's16:00 deadline
for64,823.90. August19 fills at13:02 for64,807.00 and reaches its2R target of
65,764.2183 in the14:55 minute. These are hypothetical fixed-policy fills,
not actual account trades. Each includes$60 modeled costs in the primary case.

| Fixed sensitivity | Baseline,2wins | Gate,1win | Missed winner |
|---|---:|---:|---:|
|12bps /90s|+$1,140.15|+$461.63|+$678.51|
|24bps /90s|+$1,020.15|+$401.63|+$618.51|
|12bps /300s|+$1,111.98|+$485.27|+$626.72|
|24bps /300s|+$991.98|+$425.27|+$566.72|

The gate avoids no August loser and loses the larger winner in every scenario.
Primary maximum drawdown is the same$409.25 in both books; higher-delay drawdown
is the same$409.44. There are no unknown context categories or unresolved outcomes
in either upside cohort. Known absence remains distinct from unknown evidence;
the discovery book still retains its20 no-reference abstentions.

This is only two additional upside cases in an already-exposed month. It is
neither a dependable-edge certification nor proof that structural room never
helps. It supplies no support for enabling this particular mandatory gate.
Both August cases have broken-up4H parents and intact daily parents; the result
also must not be repurposed into a newly optimized intact-parent or5m rule.

## Question and decision standard

Does filtering upside liquidity compression for at least two source-stop
distances to the nearest mapped overhead level improve the whole policy, after
including winners it rejects? This is exactly one frozen research variant under
the [protocol](lc_room_validation_protocol_2026_09_30.md), not another agent study
or a threshold search. A higher accepted-trade average alone is insufficient.

The old January2024–July2026 diagnostic provides the discovery context:14 of68
upside candidates had known room,34 had a closer mapped level,20 had no mapped
overhead. No reference does not mean unlimited room. These are development data.

## Exposure audit

August2026 is the next archive month, selected before new outcome scoring. It is
already exposed: September11's teaching-transfer study reconstructed it and
scored selected examples; the earlier fresh-data battery also covered it. The
saved August file retained only two hourly LC examples through an old selector,
not the native census. The new adapter reconstructs all native long LC before
winner/H2 selection, using unchanged source code and a30-day cold start.

This month is additional-calendar retrospective replication, not a holdout.
The permanent same-stream minute archive ends August31 23:59UTC. A related CME
archive ends earlier and is a different instrument/data stream, not a substitute.
No certified unexamined same-stream post-July saved interval was established.

## What changed

- `lc_august_source.py`: separate August census adapter, preserving frozen old
  month allowlists. Reuses saved4H/daily N3 parent ledgers only after exact hourly
  input hash and guarded code/source checks; retains all native LC directions
  eligible for the long study, regardless of competing archetype selection.
- `lc_august_receipt.py`: versioned recovery for a timestamp-formatting bug in
  the first attempt's example comparison; compares aware clocks, retains exact
  non-clock checks, and saves a hash-bound unverified capture before parity.
- `lc_room_validation.py`: accepts only the existing `at_least_2r` label. Distinct
  `below_2r`, `no_reference`, `unknown` abstentions; unavailable original plans
  remain unavailable. Baseline/gated books have independent position capacity.
- `run_lc_room_validation.py`: source-fact and decision freeze before economic
  scoring, exact consumed-path checks, old baseline parity, independent minute
  outcome checks, immutable result/verification publication.

No engine archetypes, live configuration, fusion, targets, scale-outs or orders
changed. No model-assessed trades, new dependencies, commits, pushes or PRs.

## Economic contract and interpretation

Both books use the same isolated upside LC population: completed hourly close
above prior hourly high,2.7ATR source stop, actual-fill2R target,15-minute entry
expiry,24-hour decision-relative deadline,$50,000 notional. Primary90-second
processing and12bps roundtrip costs; repeat the same old24bps/300-second cases.
One active position per book; rejecting a setup cannot occupy it. Conservative
stop-first ambiguous bars and adverse opening gaps remain unchanged.

Report both per-fill and per-supplied-candidate cost-inclusive R. Cash abstentions
contribute zero but stay in the denominator. Missing evidence is recorded
separately; missing economic outcomes invalidate complete-policy totals rather
than becoming zero. Drawdown is simulated mark-to-market dollars, not a funded
account return. Broken parent boundaries remain historical reference levels,
not guaranteed resistance; source-close geometry is not actual-fill room.

The direct independent case scorer requires the complete horizon even when the
book exits earlier. Missing later data can therefore block publication despite
a known early exit; do not remove that case or fabricate full coverage.

## Verification

425 focused tests passed in37.79seconds, including31 new source/gate/binding tests.
One read-only code review found no Important/Critical issues; it did not certify
economic results or independently rerun that suite. Bare repository pytest again
exits3 at `tests/test_integration_fixes.py`, because the existing
`configs/baseline_wyckoff_test.json` is absent. No repository-wide pass is claimed.

The first source launch verified270 file bindings and matched the old August hourly
input hash `7d9733945412cda11021e00e332b88b055f00c5dd219e473b6e4b18bc29f34d9`.
The versioned source launch verifies272bindings. Independent fresh construction
reproduces both entire saved parent ledgers exactly, including all1464transitions
per timeframe (`run_v2/parent_reconstruction.json`). Discovery verification
checks544 direct minute outcomes/nonentries, all four original baseline ledgers
and147bindings before/after, stored under `discovery_check/verification.json`.
The final combined run verifies417bindings before and after scoring, all68 old
plus2 August complete minute windows, and560 direct outcomes/nonentries across
the four scenarios and two books. Both complete cohort outputs match separate
repeated calculations exactly. An additional independent raw-price scan checks
all8 August baseline brackets across costs/delays, including pending-stop checks,
entry clocks, first exit, risk and fees. See verification.json and
final_verification.json. No source replay, model roles or outcome process remains
running. Native replay retains three existing malformed example-config load
messages and its macro/derivative fallback blockers; no full-live-parity claim.

Final input lock:
`f5f214a54058787fe33b40967a6145df90b605a1d153d89d8b80a4f118ca6fdb`.
Final result SHA256:
`be84f9e518c95e6ae6620572c7dab9028e4db097dc9748fe8913f5a1f4e1ea6e`.

The first native replay completed but failed its final example-parity comparison:
the earlier examples use a space between date/time while the canonical census
uses ISO `T`. The check incorrectly compared these equivalent timestamps as
text. No source or economic result was published. Original adapter/launch bytes
remain preserved in run_v1, with failure.json. A separate versioned recovery
passes six RED→GREEN tests and a focused re-review; run_v2 reconstructed once more,
preserved its raw capture, and passed exact parity on both old examples. The rule,
population definition and economic contract did not change.

## Local reproduction and continuity

Run from the existing research branch/repository root, after reading `PROJECT.md`:

```sh
python3 -m scripts.research.lc_august_receipt
python3 -m scripts.research.run_lc_room_validation prepare
python3 -m scripts.research.run_lc_room_validation score
```

Do not start a duplicate source capture while the original process runs. A
completed source receipt supports verified reuse without repeating the engine.
Outputs are immutable under `results/lc_room_validation_2026_09_30/run_v2`.
Rule/exposure locks are in its parent directory. Rule lock:
`1e9de337221c5e5d7e67268fa102de1ceb155da1a7d3598b944ecd884a4dc029`.

The minute archive, prior source/parent receipts, ignored private preparer and
new results are local-only dependencies. Bindings contain absolute paths; copying
files to another path is not automatic certified reproduction. New code/docs and
earlier work remain uncommitted on `quant/archetype-evidence-audit`, HEAD85923a4.

## Next boundary

This bounded comparison is finished. Do not promote the room gate or add gates
to recover the missed winners. Keep room classifications as diagnostics. The
next priority is a fixed benchmark of unchanged upside LC on genuinely new
chronological evidence, with the same conservative costs/delays and known source
limitations; no new paid trading agents. A prospective study must record source
decisions before outcomes and keep both research books separate from live orders.
Freeze its calendar and data contract before any new download/scoring; September
live examples were already discussed, so downloading newer data does not by
itself make a whole month pristine. No collector deployment is implicit.

The archive recovery receipt identifies Binance USD-M BTCUSDT minute candles;
any new archive must preserve that source identity and verify overlap/continuity
without appending to or overwriting this hash-bound archive. No new download or
prospective recorder was implemented in this turn. Neither baseline nor filter
is certified for deployment; separate evidence and user authority are required.
