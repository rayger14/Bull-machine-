# LC fixed-rule extension —142 cases, January2024–July2026

## Decision

Completed the approved larger code-only comparison and missed-rebound diagnosis.
**Neither full-LC policy is profitable across this reconstructed history.**
The confirmation rule reduces the primary loss and drawdown but is not a universal
fix. Do not deploy it across LC or start another paid agent-assessment campaign.

The descriptive upside-expansion subgroup is the more promising next research
target. Its immediate-entry contribution is positive in each calendar year in
the primary scenario and positive in aggregate under each of the four cost/delay
scenarios. This is a hypothesis-prioritization result, not a
validated subtype filter or a certified new archetype. Downside rebounds need a
separate thesis; their favorable2026 results do not generalize to2024–2025.

## Scope and fixed rules

[Protocol](lc_mechanical_protocol_2026_09_22.md) frozen before broad scoring.
Reused all142 native long candidates from31 verified monthly source receipts:
51 in2024,55 in2025,36 in January–July2026.71 downside,68 upside,3 unresolved.
No missing pre-entry plans or outcome windows; no candidate removal/resampling.
Zero new market assessments. One software reviewer approved the adapter before
reveal; that review is separate model usage, not a claim of zero billed credits.

Two independent single-position hourly-LC books, continuous occupancy through
the supplied roster. Original source-close minus2.7 ATR14 stop; $50k notional;
actual-fill-based2R target;15minute entry expiry;24hour original deadline.
Confirmation means a fully closed post-arm1m candle above the predecision last
completed5m high. Minute-open fills and existing adverse-gap/stop-first handling.
No optimization, compounding, funding, impact or native live scale-outs.

The source **detector** was reconstructed separately each month with30days of
warm-up; continuous book replay does not make its detector state continuous.
The source lacks historical live receipt, macro and derivatives evidence;
recorded blockers include defaulted derivatives and regime-model fallbacks.
This is not the full runner's selected/allocated account history. It is not
pristine holdout, CPCV, walk-forward optimization, or proof of a live edge.

## Primary result:12bps round-trip,90second processing

| Metric | Immediate | Mechanical confirmation |
|---|---:|---:|
| Candidates |142|142|
| Filled and closed trades |141|89|
| Winners / losers |54 /87|37 /52|
| Net dollars |−7,592.67|−1,266.58|
| Dollar marked-to-market drawdown |18,899.58|14,145.44|
| Average initial dollar risk per fill |910.52|989.55|
| Expired / cancelled |0 /1|51 /2|
| Busy skips / unresolved records |0 /0|0 /0|

Confirmation improves net dollars by$6,326.09 and reduces primary drawdown by
$4,754.15, but taking no trades ($0) still beats both totals. Different entry
prices also change risk under fixed notional; the difference is not pure
selection alpha. No claim of better equal-risk or funded-account performance.

### All four fixed scenarios

| Round-trip cost /processing delay | Immediate net | Confirmation net |
|---|---:|---:|
|12bps /90s|−7,592.67|−1,266.58|
|24bps /90s|−16,052.67|−6,606.58|
|12bps /300s|−8,828.40|−8,506.76|
|24bps /300s|−17,288.40|−13,306.76|

At300s confirmation admits80 rather than89 trades. Its primary advantage mostly
disappears at12bps. Secondary MTM curves were not calculated; do not infer their
drawdowns from the primary scenario.

### Calendar-year contributions, primary scenario

| Decision year | Candidates | Immediate net | Confirmation net |
|---|---:|---:|---:|
|2024|51|−3,114.11|−425.65|
|2025|55|−11,227.08|−8,705.15|
|2026 Jan–Jul|36|+6,748.52|+7,864.23|

The preceding18-case screen came entirely from2026. This larger test shows why
its positive control result could not establish multi-period profitability.

### Calendar-quarter contributions, primary scenario

| Decision quarter | Immediate net | Confirmation net |
|---|---:|---:|
|2024Q1|+1,799.71|+1,442.27|
|2024Q2|−6,790.12|−2,356.72|
|2024Q3|−4,312.19|−3,317.19|
|2024Q4|+6,188.48|+3,805.99|
|2025Q1|−6,494.85|−6,567.16|
|2025Q2|−1,121.68|+325.41|
|2025Q3|+2,780.16|+2,901.88|
|2025Q4|−6,390.71|−5,365.27|
|2026Q1|+10,157.47|+9,846.49|
|2026Q2|−2,382.27|−861.61|
|2026Q3 (July only)|−1,026.69|−1,120.66|

Groups partition the existing continuous ledger by decision time. They do not
restart occupancy at quarter boundaries or represent separate backtests.

## The important subtype distinction

| Predecision subtype | Candidates | Immediate net | Confirmation net |
|---|---:|---:|---:|
|Downside rebound|71|−15,376.13|−8,252.10|
|Upside expansion|68|+9,384.99|+7,669.09|
|Unresolved|3|−1,601.54|−683.56|

The subtype definition was fixed before this broad scoring, not chosen from
returns: close above previous hourly high means upside; below previous low or a
prior-low sweep/reclaim means downside, with upside precedence. It describes
price geometry, not confirmed exhaustion or continuation.

Upside immediate contributions:2024+$3,398.33 (26fills),2025+$2,027.96 (26),
2026+$3,958.70 (16). At24bps/90s the combined upside contribution is+$5,304.99;
at12bps/300s+$8,480.22; at24bps/300s+$4,400.22. Conversely, upside confirmation
turns negative at300s:−$490.13 /−$3,130.13 at12/24bps. Blanket waiting can cost
continuation opportunities even when it helps some rebounds.

Concentration remains material: removing the three largest primary upside
immediate winners leaves only+$1,106.37; removing the three largest upside
confirmation winners leaves−$747.50. This is a post-hoc fragility diagnostic,
not a new trading policy, significance test or justification to exclude trades.
Yearly positivity in the primary scenario does not hold in every stress case:
2025 upside immediate is−$95.49 at24bps/300s.

Subtype figures are descriptive contributions, not a newly replayed isolated
subtype-only policy. Zero busy skips limits that concern here, but does not
remove selection bias, source limitations or the need for separate validation.

## What happened to the agent-trader idea?

See [source-backed missed-rebound diagnosis](lc_missed_rebounds_2026_09_22.md).
The agent already had a no-universal-HTF-veto instruction and explicitly noticed
useful local rebound evidence. Three missed winners hit the original2R target.
That argues against simply adding more warnings or blaming all misses on exits.
However, the broader71-case downside loss shows that making the agent accept
more rebounds is not itself the solution. The task is discrimination, not
acceptance rate. Do not convert the exposed ten-case pattern into new thresholds.

## Completed verification and artifacts

-105 root focused tests passed in7.62s;11 new adapter tests were first observed
  failing for the missing implementation and then passed. Reviewer separately
  ran42 focused tests and found no critical/important issues before reveal.
  Post-scoring review found no accounting defects; two wording ambiguities were
  corrected (year/scenario positivity and an obsolete next-step instruction).
-18 old subtype labels and36 old menu plans match the new construction within
 1e-9 numerical tolerance (including old floating-point cost representation).
-All1,136 case/arm/scenario resolutions matched the existing direct conditional
  scorer within1e-7 dollars. This shares primitives with replay, not external
  execution certification. Input hashes verified before and after scoring.
-142 outcome windows each contain1,441 minutes including the deadline open;
 204,621 unique rows (one shared endpoint). All records resolved.
-Private artifacts: `results/lc_mechanical_extension_2026_09_22/run_v1/` contains
 `cases.json`, `input_lock.json`, `result.json`, `verification.json`.
-Input lock: `edc44eaadc11cba0059bbc650538925f9024d3d498f0c11373a7100675d1e550`.
-Result SHA256: `f6e6013651eb31750553de2f543b32b23789bbcecf47935b4a35f067b15f37b9`.
-New adapter: `scripts/research/lc_mechanical_extension.py`; pinned previous
  research modules, live engine/configs and order state remain unchanged.

### Reproduce using the local frozen inputs

Run from the repository root. Requires the hash-pinned private archive and source
files; GitHub alone does not supply them. This repeats an exposed backtest, not
a fresh experiment. It can take several minutes, particularly the MTM curves.

```python
import json
from pathlib import Path
import pandas as pd
from scripts.research.lc_campaign import _sha_file
from scripts.research.lc_campaign_source import ARCHIVE
from scripts.research.lc_mechanical_extension import score_extension

root = Path('results/lc_mechanical_extension_2026_09_22/run_v1')
lock = json.loads((root/'input_lock.json').read_text())
for path, expected in lock['files'].items():
    assert _sha_file(path) == expected, path
cases = json.loads((root/'cases.json').read_text())
windows = [(pd.Timestamp(c['decision_time']),
            pd.Timestamp(c['decision_time']) + pd.Timedelta(days=1)) for c in cases]
bars = pd.read_parquet(ARCHIVE, columns=['open','high','low','close'], filters=[
    [('ts','>=',a.to_pydatetime()), ('ts','<=',b.to_pydatetime())] for a,b in windows])
assert score_extension(cases, bars) == json.loads((root/'result.json').read_text())
```

## Stop / next action

The approved cycle is complete. No broad-LC promotion, no blanket confirmation
gate, no extra paid assessor batch. Preserve all142 cases as now outcome-exposed.

Recommended next **separately approved** research task: specify an upside-LC
specialist thesis and test a frozen isolated upside policy on genuinely different
or prospective data, with causal structure/room and realistic timing. Reserve
downside LC for a separate exhaustion-versus-liquidation investigation. Do not
manufacture a holdout by splitting the142 after inspecting these results, or
discard the failing years and call2026 validation.

Prior single-assessor work is locally committed at `1d4132d`. This extension and
the diagnosis are being checkpointed locally; no push/PR update, no private data
transfer, and no live change occurred in this cycle. Actual model billing unknown.
