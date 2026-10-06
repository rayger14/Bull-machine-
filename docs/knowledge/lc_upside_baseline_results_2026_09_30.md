# Upside LC mechanical baseline scorecard

September 30, 2026. **The isolated baseline is positive in this exposed history,
but a dependable edge remains unproven.** The immediate policy is the better
starting benchmark than a blanket confirmation requirement. This completes the
first deliverable of the user-approved mechanical-first research plan, not its
new-variant study, walk-forward validation or live deployment.

No market assessments were launched. One narrow read-only software review checked
the new scorecard helper; it was not a trading judgment or profitability review.
All 17 archetypes, live settings and previous experiments remain unchanged.

## What was actually tested

Selected all 68 upside-expansion candidates from the frozen 142-case January 2024
through July 2026 source census **before** replaying independent single-position
books. The other 71 downside and 3 unresolved cases were not competing for book
capacity. Selection used the existing predecision label: the setup close exceeded
the preceding hourly high. This label describes price geometry, not proven
continuation or an optimized new archetype.

The baseline retains the old research assumptions: $50,000 notional per trade,
source-close minus 2.7 ATR14 stop, a target 2 times the actual entry-to-stop distance,
15-minute entry expiry, decision-relative 24-hour deadline, and minute-open fills.
The existing confirmation comparator waits for a completed post-arm minute close
above the final predecision five-minute high. It is a diagnostic comparator,
not one of the two new structural variants still to be specified.

These are unfunded hypothetical books after flat modeled costs, excluding funding,
impact, venue lot rounding and native live scale-outs. Dollars below are not
account returns. Native source stops match all 68 research stops. Native source
targets use 5.2 ATR against a 2.7 ATR stop (about 1.93R at source close), whereas
this replay uses actual-fill-based 2R. **This is a simplified mechanical research
baseline, not an exact reproduction of the full live engine.**

## Primary scorecard

Round-trip cost 12bps of entry notional; processing assumption 90 seconds.
The latter rounds to the next minute boundary, not an observed live latency.

| Metric | Immediate baseline | Existing confirmation comparator |
|---|---:|---:|
| Candidates | 68 | 68 |
| Filled and closed | 68 | 49 |
| Expired before entry | 0 | 19 |
| Wins / losses | 28 / 40 | 22 / 27 |
| Win rate | 41.18% | 44.90% |
| Net modeled PnL | $9,384.99 | $7,669.09 |
| Dollar profit factor | 1.373 | 1.406 |
| Average initial stop-distance dollar risk | $845.29 | $929.75 |
| Net R per filled trade | 0.1483 | 0.1476 |
| Descriptive 95% interval for mean net R | −0.0838 to +0.4031 | −0.1007 to +0.4295 |
| Dollar marked-to-market drawdown | $6,737.06 | $4,586.48 |
| Net after excluding the three biggest winners | $1,106.37 | −$747.50 |

No busy skips, unknown outcomes or ambiguous stop/target bars in these primary
books. R normalizes each net result by its initial stop-distance dollar risk
**plus** modeled round-trip costs. It is not an independently simulated funded
equal-risk account or a guarantee against losses beyond 1R through gaps.

The interval uses 5,000 fixed-seed calendar-month cluster resamples across all 31
months, including empty months; 28 months contain baseline fills. It preserves
trades within each month, but assumes exchangeable month blocks and does not model
all cross-month dependence. Crucially, it does **not** correct for choosing upside
LC after examining prior results. Both intervals include zero, even before that
selection issue. The appropriate conclusion is insufficient evidence for promotion,
not proof of either no edge or a profitable edge.

## Costs and delay

| Round-trip cost / processing | Immediate net | Confirmation net |
|---|---:|---:|
| 12bps / 90 seconds | $9,384.99 | $7,669.09 |
| 24bps / 90 seconds | $5,304.99 | $4,729.09 |
| 12bps / 300 seconds | $8,480.22 | −$490.13 |
| 24bps / 300 seconds | $4,400.22 | −$3,130.13 |

Immediate entry remains positive in aggregate in all four existing scenarios.
That does not establish positivity in every year, parameter robustness, or
realistic funded execution. Confirmation admits only 44 trades at 300 seconds
and turns negative in fixed-notional dollars. At 12bps/300 seconds its normalized
R is slightly positive despite negative dollars: differences in trade risk matter,
so dollar totals and equal-risk interpretation must not be conflated.

## Consistency and concentration

Baseline annual contributions are +$3,398.33 in 2024, +$2,027.96 in 2025 and
+$3,958.70 in January–July 2026. However, six of eleven calendar quarters are
negative, including the July-only final quarter. In 2026, Q1 made $6,450.09 from
five winners, while Q2 lost $2,282.92 from two winners and seven losers. A positive
year masks substantial changes within the year.

The top three baseline winners account for $8,278.63, about 88% of net profit.
Excluding them is a fragility diagnostic, not a proposed trading rule: profitable
trend strategies can legitimately depend on large winners. Nevertheless, at
24bps/300 seconds, removing the top three changes the total to −$3,373.86. This
sample cannot support a claim of consistent profitability.

## Structure coverage for the next diagnostic

A separate source-only check reused the existing strict-before-setup parent
selector and its confirmed N3 pivot-range definition. It did not fit a gate or
associate these categories with outcomes.

| Parent state at the setup under this detector | 4-hour | Daily |
|---|---:|---:|
| Pre-existing bound remained intact | 25 | 41 |
| Bound lineage broke during setup updates | 25 | 4 |
| No qualifying pre-existing bound | 18 | 23 |
| Unknown view | 0 | 0 |

An absent qualifying range is not proof that the market had no structure. A broken
range is not automatically an invalid long: an upside break can be part of the
thesis. These counts show why the next test must distinguish state, direction,
location and available room rather than add a blanket parent-pass veto. They are
reconstructed bar-close facts, not authenticated historical exchange receipts.

## Verification and remaining limitations

- Verified all 120 original locked file bindings and the old result hash before
  work and again after replay. Rebuilt all 142 causal plans and subtype labels
  exactly from the source records and their preceding two completed hours.
- All 68 outcome windows contain the required 1,441 minutes; 97,987 unique rows.
  All 544 scenario/arm/case rows exactly match their prior full-book rows after
  isolation. The existing direct conditional scorer also agrees on all 544 entry
  resolutions and PnLs within $0.0000001. This cross-check shares some primitives;
  it is not independent market-execution certification.
- 12 new tests were observed failing for the absent helper, then passed. Final
  focused regression: **130 passed in 10.78 seconds**. One independent read-only
  reviewer found no issues, ran the 12 targeted tests and additional synthetic
  scenario/calendar/bootstrap checks. Source verification and real economics
  were checked by the controller, not that reviewer.
- Bare repository pytest was attempted and exited during collection at unchanged
  `tests/test_integration_fixes.py`, which requires missing
  `configs/baseline_wyckoff_test.json`; it is not a repository-wide pass.
  A collection-only diagnostic excluding that aborting file exposed ten additional
  import failures: `tests/archetypes/test_bull_archetypes_mvp.py`,
  `tests/archive/versions/v170/test_macro_pulse.py`,
  `tests/archive/versions/v170/test_v17_integration.py`,
  `tests/integration/test_macro_backtest.py`, `tests/integration/test_wiring_gates.py`,
  `tests/test_macro_backtest.py`, `tests/test_multi_position.py`,
  `tests/test_multi_position_full.py`, `tests/v170/test_macro_pulse.py`, and
  `tests/v170/test_v17_integration.py`. That diagnostic collected 2,329 tests but
  did not execute them; it exited 2. These unrelated files were not changed or fixed.
- The detector was reconstructed independently each month with 30 days of warm-up.
  Native eligibility is conditional on that census, not a new continuous live-state
  detector. Full runner selection, allocation and management remain unreconciled.
  Historical derivatives/macro receipts are absent; replay blockers explicitly
  record defaulted derivatives and regime-model fallbacks. Older source limit text
  saying inputs were not defaulted does not override those recorded blockers.
- All 142 original cases, including these 68, remain exposed development material.
  No matched non-signal event study, new filter, walk-forward/CPCV study or fresh
  holdout was performed. No new libraries, live changes, commit, push or PR.

## Decision and next action

**Keep upside LC as the research candidate; do not promote it.** Keep the existing
immediate policy as the benchmark. Do not impose the current confirmation rule
universally and do not revive paid market assessors from this result.

Next in the approved plan: describe pre-existing 4-hour/daily state, setup location,
room and local execution sequence for winners and losers, with explicit missing
evidence. Define a comparable non-signal control before an event-study comparison.
This is discovery, not certification. Then freeze at most the two proposed
variants: one location/room condition and that condition plus a minute trigger.
Do not tune exits, fusion, Fibonacci and many thresholds simultaneously.

Final validation requires demonstrably unexamined or prospective data. Current
local inventory does not establish an untouched validation cohort. The study
must ultimately end promising, inconclusive or unsupported under the predeclared
criteria, not expand until something looks profitable.

## Artifacts and reproduction

Local-only run: `results/lc_upside_baseline_2026_09_30/run_v1/` contains
`preflight.json`, `result.json`, `scorecard.json` and `context_coverage.json`.
The source archive and original source receipts remain local-only dependencies.
The new helper/tests/protocol/report and handoff remain uncommitted on
`quant/archetype-evidence-audit`, HEAD `85923a4`. Nothing remains running.

Preflight SHA256: `3200bf7cd43217d2f58348952816398ae3f9830b1753c2512daabe8b03b60b64`.
Result SHA256: `a15da5b8f7e096164e387e41a30af87ecd63de1751fae4a9acbae1ec3161faab`.

Reproduce the exact exposed baseline from repository root; this launches no model:

```python
from pathlib import Path
import pandas as pd
from scripts.research.lc_campaign import _sha_file
from scripts.research.lc_judgment_runner import _load, _verify_digest
from scripts.research.lc_upside_scorecard import score_upside, upside_cases

old = Path('results/lc_mechanical_extension_2026_09_22/run_v1')
run = Path('results/lc_upside_baseline_2026_09_30/run_v1')
lock = _verify_digest(old / 'input_lock.json')
preflight = _verify_digest(run / 'preflight.json')
for mapping in (lock['files'], preflight['files']):
    for path, expected in mapping.items():
        assert _sha_file(path) == expected, path
cases = _load(old / 'cases.json')
archive = next(p for p in lock['files'] if p.endswith('btc_1m_2021_2026.parquet'))
bars = pd.read_parquet(archive, columns=['open', 'high', 'low', 'close'], filters=[
    [('ts', '>=', pd.Timestamp(c['decision_time']).to_pydatetime()),
     ('ts', '<=', (pd.Timestamp(c['decision_time']) + pd.Timedelta(days=1)).to_pydatetime())]
    for c in upside_cases(cases)])
saved = _load(run / 'result.json')
assert _sha_file(run / 'result.json') == _load(run / 'scorecard.json')['result_sha256']
assert score_upside(cases, bars) == {k: v for k, v in saved.items() if k != 'verification'}
```
