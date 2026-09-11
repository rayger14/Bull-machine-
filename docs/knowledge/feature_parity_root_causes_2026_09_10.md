# Feature parity root causes — September 10, 2026

## Outcome

Read-only follow-up to the [validation build](research_validation_results_2026_09_10.md). The differences are not explained by market regime alone: overlapping timestamps expose stale dependent features, version drift, inconsistent feature definitions and NaN-sensitive predicates. No production repair, strategy tuning or new P&L backtest was performed.

The proposed next implementation is specified in [the research replay contract](../superpowers/specs/2026-09-10-research-replay-contract-design.md). It deliberately separates faithful reproduction from corrected candidate behavior.

## Matched evidence, not unmatched distributions

Inputs: V23 extended store and June/July archived live JSONL from the original checkout. The archives contain 533 rows and 529 unique timestamps. Exclude **all seven rows belonging to three duplicated timestamps**, rather than select an arbitrary winner. The remaining **526** timestamps all match the store, June 18 23:00 through July 10 23:00 UTC.

Duplicate timestamps: June 30 22:00, July 1 00:00, July 10 22:00 UTC. This is consistent with the earlier report's four duplicate occurrences beyond the first; the two counts use different definitions.

| Same-timestamp measurement | Result |
|---|---:|
| Close price, median absolute difference | $6.52 |
| Close correlation | 0.99999 |
| RSI correlation | 0.99985 |
| Volume correlation | 0.78571 |
| Liquidity-score correlation | 0.78372 |
| Taker-imbalance correlation | 0.02033 |
| Four-hour OI-change correlation | 0.12881 |

These correlations are descriptive and serially dependent, not independent significance tests. Feed/instrument and software versions are not identical. Similar prices do not establish interchangeable volumes or derivatives. Archives lack enough per-row source/version metadata to attribute every difference conclusively.

## 1. V23 dependent features were not refreshed after derivatives overlay

Source: `bin/build_v23_parity_store.py:11–16,34–54` explicitly disables derivatives during replay and describes later witness splicing. No funding is passed into `fc.update`; the live feature computer returns `funding_Z=0` when its funding history has fewer than ten observations (`bin/live/live_feature_computer.py:1520`). Later adding raw funding cannot retroactively populate that state.

The exact current `_liquidity_score_from` function, extracted from source without constructing a live client, gives stronger numerical evidence:

- On **all 74,436 store rows**, saved `liquidity_score` matches the formula with `oi_change_4h=0`.
- Only **22,023** rows match the formula using their saved OI input; **52,413** do not.
- On the 526 matched rows, none match the current full-input formula; median absolute error is 0.013578.

Comparison tolerance: `rtol=1e-8`, `atol=1e-10`. The zero-OI match plus builder source supports the inference that scores were computed before the derivatives overlay. It does not identify every command used to assemble this particular parquet.

Repair implication: replay historical observations before dependent calculations, or explicitly recompute the dependency closure afterward. A raw-column splice is not sufficient. Funding history is stateful and cannot be repaired with an arbitrary cross-sectional z-score.

## 2. Archived live liquidity belongs to an older formula version

All **526/526** archived scores match the old formula:

`0.125 + 0.35 * clip(volume_zscore, 0, 2.5) / 2.5 + 0.20 * tf1h_fvg_present`

The constant 0.125 comes from default ATR percentile 0.5 multiplied by 0.25; OI was unavailable at that pipeline stage. None match the current full-input formula. Commit `d3db500` dated August 28 moved a recomputation after ATR/OI become available. The June/July archives predate that commit.

Therefore these archives can be an old-version behavior witness, but cannot certify current live feature parity. The earlier broad difference is real, but calling it a current-server defect would overstate the evidence.

## 3. Same name, different derivatives definitions

The source paths do not implement an identical `funding_oi_divergence`:

| Definition | Live feature computer | Historical backfill |
|---|---|---|
| Funding threshold | Positive/negative relative to zero | Greater than +0.0001 / less than −0.0001 |
| OI horizon | Four hours | Twenty-four hours |
| Long/short z-score | Last 168 sampled readings; population std; no clip | 168 hourly values; sample std; clipped to ±5 |
| OI/price divergence | Thresholded event flag in live update | Signed continuous magnitude |

Sources: `bin/live/live_feature_computer.py:887–897,2404–2450`; `scripts/data/backfill_binance_vision_derivatives.py:180–219`.

Controlled source execution confirms live `funding_oi_divergence=-1` for funding +0.00005 and both OI changes −2%; the historical predicate is zero for those inputs. This disagreement exists even when the OI horizons happen to have the same value. It cannot be fixed by moving timestamps alone.

The live adapter can return OKX/CoinGlass observations under the key `binance_funding_rate`. Source identity must therefore be separate metadata, not inferred from a legacy column name. June/July taker and long/short differences also predate documented July/August repairs. Do not use those old snapshots to claim the repaired source still has the same defect today.

## 4. Missing FVG values can become affirmative evidence

`engine/archetypes/archetype_instance.py:52` ends `derived:any_fvg` with `bool(f.get('fvg_present', False))`. Python treats floating NaN as true.

With both timeframe flags zero: absent `fvg_present` fails; `None` fails; NaN passes. On the full store:

- Current predicate passes **34,352/74,436** rows.
- The same predicate with missing keys omitted passes **24,988**.
- **9,364 passes depend on NaN's truthiness.**

For the 526 matched timestamps, this predicate passes all store rows versus 172 archived live rows. It appears in the current failed-continuation and FVG-continuation gate sets. Current champion definitions applied to all matched inputs produce 3,721 runtime boolean differences across 60 individual gate profiles; these are repeated, correlated predicates, not trade counts or a live rejection rate.

The backtester passes row Series into `get_signals`; the isolated engine uses `bar.to_dict()` and passes values into `detect` without a general NaN normalization (`bin/backtest_v11_standalone.py:620–641`; `engine/integrations/isolated_archetype_engine.py:709–735`). Structural and cooldown checks still precede the gate (`archetype_instance.py:798–823`). Thus the NaN path is reachable as a gate input, but actual trade impact has not been replayed.

Do not silently normalize the reference replay. First reproduce this behavior; label any finite-value correction as a separate candidate policy and measure its downstream effect.

## 5. Replay clock and restart defects need regression coverage

The current V23 builder uses `body=src.loc[start:end]` and `warm=src.loc[:start].tail(warmup)`. For an existing start timestamp, it seeds the first body bar and then appends it again. A direct timestamp-selection probe reproduces the duplicate.

On resume, it seeds candles near `resume_from`, then iterates the original body from its start, updating the computer on earlier timestamps before reaching the checkpoint. A direct probe reproduces a non-monotonic buffer. This does not prove the saved store was built with a resumed run, nor quantify any seam contamination; it disqualifies automatic reuse of this builder as a proven restart-invariant oracle.

The existing older `scripts/rebuild/replay_segment.py` selects daily macro history through the current normalized date. A daily closing observation may not yet be published at an earlier hour that day. This is a source-level availability hazard, not proof of leakage in the V23 file. Daily dates alone cannot certify point-in-time availability.

## 6. Structure within structure: developing is not completed

Actual `_resample_to_tf` includes the current partial 4H/day bucket. In a source-executed fixture, the same 4H label contains volume 1 after one closed hourly bar, 2 after two, and 4 after four. This can be causal in sequential live use. It becomes future leakage if research joins the final 4H candle into its earlier hours.

The daily patch explicitly splices strictly prior completed days with the current partial day (`scripts/rebuild/patch_v16_daily.py:87–89`). Preserve that distinction in any new minute/hourly context contract. Store both bar-open time and available-at time; do not treat an open-time label as an entry-time permission.

## Reproduction and scope

Saved local diagnostic: `results/research_validation_2026_09_10/paired_feature_probe.json`, with original input/config/source hashes, duplicate exclusions, 60 gate profiles, formula checks and limitations. It is a spike artifact, not a new production library or CI certificate.

Key checks can be reproduced with existing utilities:

```python
import pandas as pd
from engine.archetypes.archetype_instance import DERIVED_FEATURES
from scripts.research.gate_observability import gate_observation

gate = {'feature': 'derived:any_fvg', 'op': 'bool_true'}
inputs = {'tf1h_fvg_present': 0, 'tf4h_fvg_present': 0}
assert not gate_observation(gate, inputs)['runtime_passed']
assert gate_observation(gate, dict(inputs, fvg_present=float('nan')))['runtime_passed']

store = pd.read_parquet('../data/features_mtf/BTC_1H_FEATURES_V23_PARITY_2018_2026.parquet',
                       columns=['tf1h_fvg_present','tf4h_fvg_present','fvg_present'])
rows = store.to_dict('records')
raw = sum(DERIVED_FEATURES['any_fvg'](row) for row in rows)
omitted = sum(DERIVED_FEATURES['any_fvg']({k:v for k,v in row.items() if pd.notna(v)})
              for row in rows)
assert (raw, omitted, raw - omitted) == (34352, 24988, 9364)
```

No historical strategy was promoted or rejected solely on these new feature probes. No claim of current-server parity, full end-to-end replay parity, repaired strategy profitability, or unused 2025–2026 holdout is made.
