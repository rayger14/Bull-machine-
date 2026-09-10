# Fusion: original intent, current authority, and research revival

Date: 2026-09-10. Research-only; no production changes or live-setting changes.

## Answer

The user's recollection is supported by the original code: Fusion attempted to ask whether a technical setup made sense in its surrounding market context, including macro confirmation or veto. That purpose remains useful. The current scalar is not that original all-context assessor: it is primarily an archetype-specific mixture of technical domain scores, with a separate CMI score-and-threshold modulation layer.

Fusion was **not removed**. The outer threshold was bypassed for collection, while inner score floors, cross-archetype ranking, and some score-conditioned gate enforcement survived. It is therefore premature both to restore the old cutoff and to assume the present book is an unbiased collection of every archetype opportunity.

Recommended direction: recover contextual reasoning as explicit, timestamped evidence and setup-specific conditions, not a bigger weighted average. Maintain the existing bypass setting and all 17 archetypes. Score/data integrity, event clocks, and selection effects must be resolved before claiming a better trading rule.

## Evidence scope

Two independent read-only agents audited history and active mathematics. Root inspected key source branches and independently reran the NaN/default, gate-boundary, and emergency-size probes below. Source/config conclusions concern this local checkout, with production files unchanged from `eb135f5`; service files identify the intended deployed path but are not a new remote deployment attestation. Historical documents are distinguished from experiments rerun today. Graphify located legacy components; direct source/git inspection established the active path.

## What the idea originally meant—and what changed

| Period / commit | What happened | What it establishes |
|---|---|---|
| 2025-09-30, `a5bad48` | `engine/fusion.py` combines technical domains with Macro Pulse, macro delta and hard veto; returns explainable factors | Original context-confirmation intent is real. Tiny prototype samples are not validation |
| 2025-10-08, `914f959` | “True Fusion” uses four technical domains plus MTF penalty; macro is outside the weighted average | More real domain engines did not establish predictive calibration; documented 721-bar month produced zero signals |
| 2026-02-25, `edb91b7` / sibling `9d266b1` | Isolated per-archetype fusion and separate CMI threshold architecture | Technical pattern score and regime context were deliberately separated; exact live cutover date is not separately verified |
| 2026-03-06, `aaa2826` then `c659478` | Enforcement and nine disables were reversed 17 minutes later to restore collection | There was no single final “fusion off” migration |
| April–May, `5cbf55f`, `8b22cbb`, later `274f8a4` | Structural bypass semantics and gate enforcement were repaired; per-archetype exemptions followed | Bypass, structure and score gates are distinct mechanisms |
| May–July | Historical and live diagnostics did not support “higher fusion means better trade” | Reason not to restore an uncalibrated cutoff, not proof that all context is useless |
| August | Conjunctive structural/context architecture and a meta-label prototype were studied | The complex learned prototype failed generalization; do not repeat it under a new name |

Sources: `engine/fusion.py:65`, `docs/archive/V1.8.1_TRUE_FUSION_COMPLETE.md:71`, `docs/knowledge/founding_knowledge_archaeology_2026_07_17.md:3`, `engine/context/regime_service.py:28`, and the named git commits. The old global FusionEngine and dormant knowledge hooks are not the present Coinbase entry path.

### Why threshold enforcement lost support

- `MEMORY.md:182` reports Pearson −0.082 and Spearman −0.122. The exact generating artifact, row unit and source version were not recovered as a tracked reproducible experiment. These numbers remain a historical diagnostic, not a fresh result.
- July 2's **position-level** report found 5 threshold-cleared positions, all losers (−$5,886), versus 25 bypass-only positions with all 9 winners. July 15 reported threshold-cleared PF 0.27 on 20 positions versus PF 0.76 on 108 bypass-only positions. See `live_trade_forensic_2026_07_02.md:61` and `live_evidence_engine_2026_07_15.md:25`.
- These reports overlap, span changing implementations and a long-biased BTC decline/bounce, and are descriptive cohorts. Filtering completed trades does **not** reproduce a native threshold-on run with different accepted-position history, entry spacing, position limits, allocator sizing, capital use and exits. Cooldown and dedup occur upstream of this outer threshold; toggling that bypass alone does not change their mechanisms. Both July 15 groups lost money. Negative observed correlation is not a license to invert the score.
- The September 8 LC position likewise won despite its outer threshold: 0.2651 versus 0.3319, recorded exit PnL +$721.73. The preceding wick position scored 0.2805 versus 0.3081 and ultimately lost $1,462.65; the screenshot showed only its earlier profitable partial. One pair is a case study, not calibration. See `sep8_liquidity_compression_case_2026_09_10.md`.

### Why some macro additions stayed off

- Derivatives heat remained at zero after failed historical acceptance; the preserved report for 10% showed +$1,221 OOS against −$25,954 train. Full source report is branch-only (`9a9ce98`); current memory records it at `MEMORY.md:205`.
- The alternative macro/sentiment crisis composite recognized crisis windows but worsened the tested trading results: −12.9% PnL and worse DD/Sharpe, with fewer trades (`cmi_crisis_prob_rebuild_2026_06_02.md:65`). Better crisis classification is not automatically better entry permission for a reversal setup.
- Some funding/DXY relationships reversed across train and holdout, and macro feature distributions/coverage differed between stored and live inputs (`resurrection_verdicts_2026_07_18.md:227`; `composite_boost_wfo_2026_06_03.md:23`). These undermine a global weighting claim, not the possibility of a narrower useful interaction.

## Current path and mathematics

`deploy/coinbase-paper.service:14` passes champion config to Coinbase runner; `coinbase_runner.py:179` creates V11ShadowRunner; `v11_shadow_runner.py:186` builds IsolatedArchetypeEngine with the configured champion directory. The local loader has 17 enabled archetypes, not merely the graduated subset. Two neutral-direction definitions cannot emit through the current neutral rejection branch (`archetype_instance.py:877`); enabled is not synonymous with executable.

For archetype a, before its gate penalty:

```text
F_a = clip[0,1](
  (wW_a*W + wL_a*L + wM_a*M + wS_a*S
   - 0.10*fakeout - 0.10*max(PTI_1D, PTI_1H))
  * whale_conflict_multiplier * premium_multiplier * stale_multiplier
)
```

Weights differ by archetype. W is direction-aware multi-timeframe Wyckoff; L is a precomputed score or volume/ATR/FVG/absolute-OI proxy; M combines ADX, absolute RSI distance from 50, and squiggle confidence; S combines directional BOS, CHOCH and FVG. M measures extremity/strength, not necessarily favorable directional momentum. Missing components and differing mixtures make raw cross-archetype values non-equivalent to calibrated probabilities (`archetype_instance.py:245–302,384–556`).

Soft gates then multiply F by their pass fraction. In the default native fusion mode an **inner YAML threshold** remains active (`:822–850`). The isolated engine keeps the highest emitted detector fusion score per direction—after soft-gate penalty and inner threshold, before runner crisis adjustment (`isolated_archetype_engine.py:585–633`). Cooldown can arm before dedup, even for candidates later discarded (`archetype_instance.py:901–908`).

The runner applies:

```text
adjusted_F = F * (1 - 0.50 * crisis_prob)
outer_threshold_a = base_a + (1-risk_temperature)*0.38 + instability*0.15
```

This **outer** cutoff is bypassed in collection mode, not the whole pipeline (`v11_shadow_runner.py:1121–1228`, `champion_paper.json:86–125`). Above-cutoff signals are admitted directly; below-cutoff signals in bypass must pass an additional gate-enforcement branch unless the archetype opts out. Thus fusion still affects admission even in bypass, as well as dedup. CMI can also remain causal: crisis scaling can move a failed-soft-gate signal from direct admission into the below-threshold gate-block branch. For gate-passed signals, the outer cutoff itself is bypassed and logged.

Macro is not a direct term in the technical F formula. Current champion selects `crisis_prob_source="original"`; DXY/VIX enter the optional substitute branch, not that selected CMI calculation (`champion_paper.json:90`; `v11_shadow_runner.py:512–607`). Yields/yield-curve enter neither of those CMI branches. Fear/greed sentiment does enter current CMI. Broad macro outlook/logging/factor-attribution display should not be mistaken for an active directional entry model. This conclusion is scoped to the inspected path, not every dormant macro module.

The intended orthogonality is incomplete at input level: ADX appears in M, risk temperature and instability; volume and other fields also recur. That is overlapping influence, not automatically a bug, but it must be measured explicitly rather than sold as independent confirmations.

## Reproduced implementation hazards

These are deterministic local synthetic probes, **not measured prevalence in actual live trades**. No production fix was made.

| Input / boundary | Observed result | Interpretation |
|---|---|---|
| Empty features, equal domain weights | L=0.125, M=0.06667, F=0.0479167 | Default ATR percentile supplies nonzero liquidity; it may exceed some minimum-liquidity floors without observed evidence |
| NaN ADX, RSI or squiggle | M=1.0; F=0.28125 | NaN combined with Python min/max clipping promotes invalid momentum to maximum |
| NaN daily PTI | F=1.0 | Invalid penalty propagates into the same clipping issue |
| NaN CHOCH flag | S=1/3 | NaN is truthy in this boolean branch |
| Precomputed liquidity_score=2 | L=2, F=0.516667 | Domain input is returned without bounding; final score clipping does not restore intended relative weights |
| Synthetic failed gates, bypass on, score .10 vs .50 with threshold .30 | .10 rejected; .50 admitted | Score-conditioned enforcement, not unconditional structural validity; acceptance is still monotone in this score-only example |
| Normal vs crisis synthetic paper-book case | Both open $2,625 notional / $1,750 margin | The emergency “50% sizing cap” changes score, not allocated size in the current bypass path |

Source roots: `archetype_instance.py:263,301,468,509,541`; runner boundary `v11_shadow_runner.py:1168–1203`; emergency mutation `:1230–1239`; bypass allocation fixed intent 0.02 at `:1357–1361`; size calculation `:1729–1753`.

In the crisis fixture, tracked pre-emergency score=0.475 while position score=0.2375; stored threshold margin remains based on 0.475. Threshold-margin metadata and final stored fusion can therefore represent different stages. This must be reconciled before new outcome calibration. Fixture notional is not a claim about the actual server's position size.

Root reproduction used `side_effect_guard` for domain construction and the existing guarded `VirtualBookFixture` for source book execution. No denied network/write attempts occurred. Boundary extraction ran the unmodified source loop with source SHA256 `10722d523ef931bc6172b8b67efb69d1dec3f956d4926552aec1f85a36f659c8`. The tests/probe fixtures do not establish how often upstream real features produce these inputs.

### Actual saved-input check: do not blame hypothetical NaNs for measured losses

Root also inspected all **74,436 V23 rows** (2018-03-01 through 2026-08-30) and all **240 emitted June native-replay feature rows**. The eight inspected fields were ADX14, RSI14, 4H squiggle, daily/hourly PTI, 1H CHOCH, liquidity score and ATR percentile. None was absent, null/NaN or infinite in either inspected set; liquidity/ATR-percentile values stayed inside [0,1]. Thus these sets do **not** show the synthetic NaN or oversized-liquidity failure being triggered. Actual live receipt prevalence remains unmeasured.

They do expose a different, directly observed mismatch: daily PTI is **0.0 on every V23 row**, but **0.5 throughout the 240-hour current-native replay**. `_pti_fakeout_features` initializes it to 0.5 and only computes daily PTI after at least 20 resampled daily bars (`bin/live/live_feature_computer.py:1551–1583`); the ten-day replay cannot meet that history requirement. Since hourly PTI in that replay is at most 0.5, the daily default supplies a 0.05 raw fusion subtraction on every bar before multipliers. This is warmup/default evidence, not a claim about a fully warmed server or a measured trade-level PnL effect. V23 squiggle also spans negative values (minimum −0.6802); its signed semantics need preservation, not automatic clamping merely because its name says confidence.

Input identities: V23 SHA256 `0e8604d0dacf7435e42e9759fe64aca15bfa1e3d66f14c1d21c142f41d1ab61f`; private replay `results/research_validation_2026_09_10/native_pipeline/hourly_240h_all_emitted.json`, SHA256 `418ab56dd2a774343fb98a81739e72976ffc267ee73600756f01d1935d855167`. These are stored inputs/current-native OHLCV-only replay, not independently authenticated historical live consumption.

## Revival design: preserve the purpose, separate the jobs

This is a research proposal, not a validated new archetype:

1. **Evidence validity:** which fields were observed, fresh, from the expected source, available and actually consumed at the decision? Missing/defaulted is not neutral agreement. The H1 witness work is a start; historical consumed-field provenance is still incomplete.
2. **Parent context:** what fixed daily/4H structure contains the hourly/minute setup, where is the child within it, and is that lifecycle still valid? The H3 sidecar preserves immutable pre-sweep binding. Macro/flow context needs the same publication/arrival clocks and explicit horizon.
3. **Archetype sequence:** did this particular setup complete its necessary sequence—e.g., valid level, sweep, reclaim—rather than collect enough unrelated points? Essential predicates cannot be bought back by unrelated strength. Not every contextual preference must be a hard veto.
4. **Conditional quality:** only within valid setups, test whether additional information improves expected **net R and its downside distribution**, not just win rate or a globally compared raw score. Favor a simple baseline before any flexible learned model. Keep direction, holding horizon and archetype distinctions.
5. **Execution and portfolio risk:** apply any approved size change to actual intent/quantity, account for correlated overlapping positions, spread/slippage/funding and censored trades. Check the filled/book outcome, not merely a smaller displayed score.

An hourly bearish backdrop can coexist with a minute rebound opportunity; a universal “all timeframes agree” veto is not established by the trader corpus or this audit. The key question is whether the child trade's location, target, invalidation and intended horizon are coherent within its parent state.

The July founding report and August addendum already contain versions of this idea (`founding_knowledge_archaeology_2026_07_17.md`, `wyckoff_audit.md:765`). They are useful prior work, not proof. In particular, the first GBT meta-label prototype reported train-resubstitution AUC .93 versus purged-CV .52 and no useful regime discrimination (`wyckoff_audit.md:776`). Feature importance from that failed model does not validate the architecture, and win probability alone is not an optimal size rule when payoff/loss magnitudes differ.

## Frozen research sequence—no immediate fusion reactivation

1. Finish the H3 evaluator and temporal/identity tests. Historical event annotations follow only with same-stream frozen witnesses; report all four parent variants, not a winner.
2. Characterize score input failures on available causal candidate records. Preserve original raw components, gate penalties, pre/post-crisis scores, threshold source, missingness and dedup/cooldown state. Quantify prevalence before attributing historical losses to synthetic hazards.
3. Propose a separately reviewed finite-input/score-stage contract; test missing, NaN, infinity, out-of-range and observed zero. Any production repair requires its own explicit approval. Do not silently tune weights while repairing evidence handling.
4. Freeze independent comparisons: existing collection behavior; old outer cutoff as an offline control; structural/context permission alone; later a narrowly specified macro interaction. Do not combine arms because one looks good on reused history.
5. Measure direct fixed-event effects separately from full native cooldown/dedup/book displacement. Group exits into positions; compare constant-risk diagnostics and actual sizing separately. Reused 2018–2026 hourly / 2021–2026 minute data are not pristine holdout.
6. Only after parity and stable cost-aware evidence, consider a simple calibrated secondary quality layer. Register all trials, use time-respecting splits with overlapping outcome-window purging, retain unresolved positions as censored, and reserve new forward evidence after the freeze.

Reporting every attempted variant matters: repeated selection inflates apparent backtest quality, as described by Bailey and López de Prado's [Deflated Sharpe Ratio paper](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). That statistical correction cannot repair wrong clocks, missing sensor provenance, or an invalid execution model.

No claim is made here that fusion inversion is universal, macro cannot help, a learned score will work, or any archetype is safe for capital.
