# Trader-teaching and code comprehension — bounded offline probe

## Why this study

Ray clarified that agents must understand the saved trader teachings **and** how
the engine translates them into code, then test that understanding. The previous
twelve-case study supplied rules and multi-timeframe prices but its all-reject
result did not establish teaching comprehension. The user delegated routine
research decisions and review to a quant agent while away; no live authority.

The quant guide recommended a comprehension check before another economic run:
eight synthetic contrast cards, four hourly LC and four minute sweep/reclaim,
answered independently by two fresh-context assessors. Both received the same
outcome-free packet, code excerpts and local teaching notes. No trade acceptance,
forecast, PnL, other assessor response or earlier study verdict was supplied.
All17 native archetypes remain unchanged; this probe tests only two families.

## Source honesty: teaching, code and project hypotheses

These are **local-corpus interpretation** results. Targeted searches of the
current project and related sibling folders located saved attributed teaching
notes, not authenticated original posts/transcripts behind the selected claims.
This does not prove no primary archives exist elsewhere. The local notes can
support a test of what the saved corpus says, not certification of what a trader
actually taught. Old audit headings such as "direct evidence" are not independent
authentication. Old profitability verdicts were excluded from the packet.

| Source class | Packet material | What it can establish |
|---|---|---|
| Locally attributed teaching | Wyckoff audit lines318–320,402–414,455–458,910–913 | Saved accounts of confirmation/return-to-zone, fixed structure, and conflicting context/size descriptions |
| Local provenance warning | Same audit line352,920,922–923 | Corpus calls70% wick a project invention; Moneytaur-specific coverage is thin, not authenticated detailed doctrine |
| Executable code/configuration | LC `_check_E`, `_evaluate_gates`, `_safe_float`, derived RSI, champion LC gates, minute `detect_events`/validation | What these exact functions and selected gates do on supplied synthetic inputs |
| Project candidate contract | H1/H2/H3 definitions | Proposed evidence, prior-hour compression and parent-geometry interventions—not universal native or teacher rules |

Relevant local anchors:

- [Attributed confirmation and return-to-zone sequence](wyckoff_audit.md#2026-08-05-addendum-7--wi-batch-2-execution-mechanics-decoded-direct-evidence).
- [Native-rule characterization](native_pipeline_and_trader_witnesses_2026_09_10.md).
- [Separately named H1/H2/H3 candidates](candidate_rule_contracts_2026_09_10.md).
- `engine/archetypes/logic.py:662` and
  `engine/archetypes/archetype_instance.py:674`.
- `configs/champion/archetypes_v14rq/liquidity_compression.yaml` is the exact
  selected gate source, not an assumption that similarly named config directories
  always agree. Configuration descriptions/performance comments were removed
  from the assessor excerpt; executable values remained unchanged.
- `scripts/research/minute_sweep_validation.py:36` is a separate research minute
  detector, not the native hourly LC archetype on a faster clock.

The apparent HTF-size conflict is deliberately retained: one local note describes
bias/size permission with tactical exceptions; another denies regime-based sizing
and ties size to model completion. No universal dial or veto is manufactured by
silently combining them. Exact range anchoring, one-hour compression, numerical
matching tolerances and lower-half geometry are not promoted to trader doctrine.

## Frozen design and safeguards

The quant guide approved the eight contrasts and rubric before both calls. The
private assembler called existing native witness fixtures to establish reference
labels; it did not ask a model to invent desired native behavior. Sixty-three
existing native/parent witness tests passed before assessment. Packet assembly
ran twice with identical bytes. Inputs, selected source files, card text, private
assembler, rubric and labels were hashed before calls.

Code snippets use AST formatting with comments/docstrings removed. Teaching
snippets retain exact selected line numbers and are labeled secondary notes.
Research candidate definitions remain separately labeled. Assessors could read
only their shared packet and then use no further tools, other files or agents.
Each returned its original answer once; no critique/retry or forced answer mix.
Both responses were saved and locked before grading.

Required response: exact native results for A/B on each card, separate candidate
interpretation, teaching interpretation, unknowns, and claim-level source IDs.
Limits:1600words total,700characters per explanation field,250 per cited claim.
Grading requires each assessor independently to get16/16 native variant answers,
explain all8 contrasts, and have zero critical scope errors or unsupported
teacher attributions. Qualitative citation entailment is reviewed separately;
native score alone is insufficient.

## Measured results

Both assessors returned all16 native variant answers correctly with proper types.
Both explained all8 comparisons and complied with schema/length limits. Assessor
A:1442 whitespace-counted serialized-response words and35 claim citations;
B:1366words and34citations. All cited IDs exist in the packet. Independent
qualitative quant review is complete: both pass. The reviewer checked all35/34
citations and found zero material entailment exceptions, zero unsupported
teacher attributions and zero critical scope errors. Both have8/8 correct
contrast/candidate explanations. Citation existence alone was not the criterion.

| Contrast | Actual reference result | Distinction both responses made |
|---|---|---|
| LC1: compressed vs expanded preceding history | Both identity/gates pass, long | Native terminal checks unchanged; H2 separately passes A and rejects B |
| LC2: observed finite vs missing/NaN with climax | Both identity/gates pass, long | H1 permits observed A, withholds permission from missing B; permissiveness is not confirmation |
| LC3: RSI25 vs75 | Both identity/gates pass, long | Numerical eligibility does not establish a directional teaching thesis or an executed book trade |
| LC4: recovered vs unrecovered parent floor | Both identity/gates pass, long | Stipulated optional H3 geometry passes A and fails B; not universal LC eligibility |
| M1: equal vs out-of-tolerance earlier touch | A event at90; B none | Actual child-level identity and coded tolerance matter; tolerance is not authenticated teaching |
| M2: intact vs prematurely undercut pivot | A event at90; B none | Bar74 destroys pivot60 geometry; conditional earliest confirmation is bar75 close,01:16UTC |
| M3: close reclaim vs wick-only penetration | A event at90; B none | Strict close above level; a reclaim alone is not the entire acceptance/MSS/LPS/RTZ sequence |
| M4: identical prices, different parent-direction labels | Both event at90 | Detector ignores annotations; H3 cannot be evaluated from direction labels alone |

Native identity's missing RSI fallback is50; the derived gate's `_safe_float`
fallback is0, which passes the `<35` branch. Both agents identified the separate
identity-sanitization and derived-zero paths, but neither explicitly restated the
identity's numerical50. The independent reviewer judged the omission nonmaterial:
neither incorrectly says the identity uses0, and both explain why climax suffices.
The omission is retained, not repaired. No response was rewritten.

## What this supports—and does not

The measurable result supports narrow code-reading and distinction-making with
the supplied teaching notes. It does **not** show that adding teaching material
improves decisions: there is no code-only comparison arm. It does not demonstrate
independent transfer, full engine understanding, authentic teacher fidelity,
multi-timeframe chart recognition, predictive discrimination or profitability.

The test is heavily scaffolded and open-book. All eight hourly A/B native vectors
are identical `[true,true,"long"]`; the minute cards add discriminating positive
and negative examples, but explanation/citation review remains essential.
These are synthetic feature/bar fixtures, not trades or historical returns.
There is no starting equity, position sizing, average trade risk or PnL result.
The two assessors use the same available model environment; agreement is not
independence of model training or a calibrated reliability estimate.

## Artifacts and next step

Private, ignored directory: `results/teaching_comprehension_2026_09_11/` contains
cards, protocol, rubric, immutable input/labels/manifest, builder, original A/B
responses and response lock. These files are local, not in GitHub by virtue of
committing this report. Exact model snapshot and billed tokens are unavailable.

- Input SHA256: `54ea0abf28416a9a27ad6368615f2ad208ea5438ce1d378ac5c951788672dc49`
- Response lock SHA256: `3312cced539d6bf4d977156b8736f90fc417c788b3f219c3f4a94394b673e729`

The full research suite was rerun: **416 passed**, one existing LibreSSL warning,
13.80seconds. No production code, gates, config, live runner, dependencies or
market-data providers changed. No optimizer, full legacy economic backtest,
deployment, push or PR action was performed for this comprehension probe.

Follow-on completed: [four-case August teaching transfer results](teaching_market_transfer_2026_09_11.md).
Both arms selected the same one profitable hypothetical case, but four paired
setups, delivery uncertainty and reviewer-history exposure do not establish
incremental teaching value or a deployable edge. The following records the
original next-step decision, not pending reconstruction work.

Approved by the quant guide: [four-case August teaching-on/off real-market
transfer](../superpowers/specs/2026-09-11-teaching-market-transfer.md), with two
hourly and two minute cases and at most eight fresh calls. The quant guide also
approved the protocol and hash-pinned source wrapper before execution. Register first, then
the smallest necessary single-month same-stream source reconstruction, because
prior caches do not retain unassessed hourly feature rows. Primary factual and
causal evaluation precedes the separately frozen secondary outcome diagnostic.
Do not repeat these scaffolded questions as evidence of independent learning,
and do not reprompt the previously revealed twelve economic cases.
