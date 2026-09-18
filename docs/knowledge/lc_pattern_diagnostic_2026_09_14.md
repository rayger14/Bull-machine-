# LC pattern diagnostic: what should be tuned?

September 14,2026. Read-only diagnosis requested by the user, not a new
backtest, optimized strategy or completion of the persistent-master experiment.
No new Q1 outcome reveal, market-role calls or production changes.

## Finding: setup subtype before another cutoff

Native LC and the new nested-child rejection hypothesis describe different
populations. The all-fail result does not establish that native LC loses or
that its filter is profitable. Independent gate review found no implementation
mismatch explaining these actual failures.

Root recomputed source counts from the unchanged three monthly source JSONs
under `results/lc_persistent_master_2026_09_14/source/`. Independent agent
`lc_gate_pattern_audit` recomputed all18 causal parent views, matched recorded
conditions and checked native logic/config manifest hashes.

| Native source anatomy | Count | Candidate close | Parent state |
|---|---:|---|---|
| RSI below35, bearish candle | 13 | Below previous-hour low | 8 broken down,3 intact,2 absent |
| RSI above65, bullish candle | 5 | Above previous-hour high | 2 broken up,2 intact,1 absent |

All18 have volume-climax flag1 and volume-z3.0162–4.2598. `absorption_flag` is
missing, not observed false; no conclusion about actual absorption follows.
Previous-hour BB width0.005411–0.036337 passes0.06 for all18. There are **zero
same-hour sweeps followed by reclaim of the previous-hour low**.

Native [identity](../../engine/archetypes/logic.py#L662) accepts climax/absorption
or high-volume two-sided RSI extremes; [champion YAML](../../configs/champion/archetypes_v14rq/liquidity_compression.yaml)
adds gates and specifies long direction. The new [reference](../superpowers/specs/2026-09-14-lc-persistent-master.md)
adds same-hour sweep/reclaim and an intact enclosing4H/N3 parent. Its one-hour
child is explicitly a project proxy, not universal trader doctrine.

### Evidence failures are structural states, not missing price data

| Parent / geometry state | Count |
|---|---:|
| Intact parent, nesting passes, rejection fails | 5 |
| Broken parent, nesting passes, rejection fails | 7 |
| Broken parent, nesting fails, rejection fails | 3 |
| Absent active parent, nesting/rejection unknown | 3 |

The13 `evidence` failures mean10 known lineage breaks (8 down,2 up) and3
valid `forming` states following earlier downside breaks. Those absent-parent
decisions are Feb25 02:00, Feb28 07:00 and Mar22 22:00 UTC. Of15 bound-parent
cases,11 sweep but close below the child low;4 never sweep, including2 that
close above the parent ceiling. [evaluate_reference](../../scripts/research/lc_master_assessment.py#L221)
combines source validity and parent lifecycle under the evidence label.

A broken-up range may belong to continuation; a broken-down range may require
later reversal evidence. These are research distinctions, not permission to
call every break bullish or bypass missing-data checks. All valid responses
under the current [grading rule](../../scripts/research/lc_master_assessment.py#L351)
must reject this sample. Consequently it can test explanations/compliance, not
agent entry discrimination. Any economic benefit versus native would belong to
the deterministic gate, not discretionary agent selection.

## Existing live winner/loser evidence

The already-revealed September10 scorecard has37 explicit-ID LC groups from89
recorded exit legs:20 positive,17 negative, subtotal+$11,168.69. Fourteen older
LC exit rows lack IDs and remain excluded. These are **recorded-exit subtotals,
not certified closed-position full-net returns**. Original quantities/complete
receipts, entry commissions, funding allocation and historical source versions
are not reconstructed. Overlapping older snapshots are not additional samples.

Root rehashed the scorecard and independently recomputed counts, signs, monthly
totals and these contrasts; `lc_win_loss_data_audit` inspected raw metadata and
join availability separately.

| Descriptive contrast | Positive groups | Negative groups |
|---|---:|---:|
| Count | 20 | 17 |
| Median fusion | 0.32435 | 0.39800 |
| Median threshold margin | -0.10205 | -0.01130 |
| Single-exit groups | 0 | 13 |

Five August groups contribute+$8,198.95, about73% of the overall subtotal;
one of those five loses. This is concentration, not an August trading rule.
Changing size/config and market conditions confound raw-dollar comparisons.
Lower fusion among positives does not justify inverting its cutoff. All20
positives have multiple exits, but exit count/duration occur after entry and
cannot be used as entry predictors. See the [scorecard report](fusion_live_scorecard_2026_09_10.md).

The37 groups retain fusion/threshold, ATR, regime, risk-temperature, instability
and crisis metadata. Factor attribution is a derived weight decomposition,
not preserved observed domain scores; it cannot prove micro/macro alignment.
The rolling signal log contains13 allocated LC rows with no position IDs and12
potential tuple-inferred exit matches. This truncated post-gate/cooldown/dedup
sample is not a verified feature/outcome join or an opportunity population.
The auditor did not find the expected local `results/coinbase_paper/trade_outcomes.csv`.

We can describe score/outcome contrasts now, but not claim a reliable live
winner/loser comparison of nested structure and minute confirmation. Historical
annotations must use matched instruments, sources and causal clocks and remain
labelled reconstructions, not original live observations. Prospective stable-ID
feature snapshots and completion receipts would improve this; none were deployed.

## Recommended next study — not implemented

Earlier [volume-band research](within_gate_quality_study_2026_08_30.md) reports
raising the BTC LC volume floor to3.0; that value is already in champion YAML.
Those historical results were not newly reproduced here, and newer evidence
limits supersede old certification language. The18 native emissions already
passed the floor: they cannot tell us whether admitting below-floor events helps.
Testing an existing gate requires a population before that gate; testing a new
context layer can use native candidates. Do not confuse the two.

The focused proposal is an LC-only **subtype × parent lifecycle × minute
confirmation** table, followed by a small separately frozen comparison:

1. Keep one row per candidate/position with explicit source, clock, coverage,
   rule version and accounting status. Keep observed live subtotals separate
   from fixed-contract reconstructed entry outcomes.
2. Label upside expansion versus downside exhaustion, intact/broken/re-forming
   parent, known opposing levels and available room. A true multi-bar child
   range is a separate definition from the present one-hour proxy.
3. Test an identified1m/5m reclaim or breakout retest within fixed expiry for
   the appropriate hourly subtype and4H/daily context. Later confirmation is
   a new decision, never knowledge retroactively available at the original signal.
4. Compare native, simple subtype-aware code, and same-evidence agent judgment
   with identical costs/risk and independently replayed occupancy. Code owns
   trustworthy facts; the agent interprets competing valid structural theses.
   Do not force acceptance, universal timeframe agreement or safety overrides.
5. Freeze a short trial list and exact rules before economic comparisons.
   Separate discovery from later chronological testing, purge overlapping
   outcome intervals and record repeated searches. Use walk-forward validation
   when coverage permits; CPCV cannot cure sparse data or previously mined history.

Report counts, unknowns, net expectancy under declared costs, loss tails, missed
winners and displaced opportunities—not win rate alone. A retrospective pattern
is a hypothesis until later frozen tests and forward evidence support it.
Repeated-search risk: Bailey et al., [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

Preserve the current frozen study and its zero-admission finding. A different
judgment policy needs separate approval/versioning; no replacement rule, fitted
cutoff or new economic result was created here. Planned eight market-role calls
must not be represented as an informative entry-selection experiment when only
reject is legal.

## Verification / handoff

Monthly source hashes remain January6ba371d2..., February8c703b38...,
March4de2cdbf..., matching `source_verification.json`. Scorecard SHA256 remains
9c2aed6b00afc4eb52459e9ad5138f944c699293794dc5cbd2e8d098901cb903.
Root freshly ran private harness plus research tests:849 passed, one existing
urllib3/LibreSSL warning,18.93s. This verifies software, not trading performance.
Task4 PhaseA runner/preflight exists; independent whole-harness review, permanent
freeze, actual roles and outcomes remain pending. No Q1 outcome reveal, live
config/orders, push or PR. This report and handoff changes are local only.
