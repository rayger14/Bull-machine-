# Evidence-backed archetype agents: advisory design

September 11, 2026. Independent quant design advice incorporated. **Not implemented, not a validated strategy, and not authorization for live integration.**

## Answer to the user's intent

Yes: agents could help connect trusted observations, pre-existing larger structure, an ordered local setup and an executable plan. The test is whether that reasoning improves decisions beyond deterministic rules supplied with the same evidence—not whether its explanation sounds like an expert trader. Keep all 17 archetypes distinct and hourly/minute BTC equally important.

The proposed replacement for a universal fusion cutoff is a traceable trade thesis. Hard prerequisites remain machine-checked; softer interpretations remain explicit, contestable hypotheses. No missing observation becomes neutral confirmation and no agent confidence score is treated as a calibrated probability of profit.

## Recent research and its limits

[TradingAgents](https://arxiv.org/abs/2412.20138) demonstrates a specialist analyst/research/trader/risk architecture. That supports considering role separation, but does not validate our BTC data, minute execution or profitability.

The August 31, 2026 [Agentic Quantitative Trading survey](https://arxiv.org/abs/2608.31041) covers workflows from signals to execution and risk, highlighting the gap between promising model capabilities and reliable trading systems. This is motivation to evaluate the complete workflow, not permission to assume an edge.

[Agentic Trading: When LLM Agents Meet Financial Markets](https://arxiv.org/abs/2605.19337) documents evaluation and reproducibility gaps in its reviewed studies. [Look-Ahead-Bench](https://arxiv.org/abs/2601.13770) raises another important issue: model weights themselves can contain future knowledge. Timestamp-clean retrieved data alone cannot establish a hindsight-free historical LLM test. These are primary research sources, not independent certification of any advertised system.

## Proposed responsibilities

| User's question | Deterministic evidence and authority | Bounded agent contribution |
|---|---|---|
| Can I trust these inputs? | Source, units, formula, receipt/availability clocks, completeness, freshness and default flags | Explain conflicts and abstain when required evidence is unresolved |
| Did the larger structure exist before the setup? | Frozen parent version, lineage and strict pre-setup binding | Interpret permitted context and opposing structural evidence |
| Did the sequence occur in the right location? | Ordered event IDs, causal timestamps and archetype-specific location predicates | Assess fidelity to the trader's thesis and articulate counterexamples |
| Where are entry, invalidation and available room? | Registered plan, executable entry convention, structural levels, cost/risk arithmetic and expiry | Explain whether that plan fits the evidence; reject unsupported assumptions |
| How will the position be managed? | Frozen management state machine and portfolio limits | Explain the plan; later experiments may test bounded reassessment with fresh evidence |

One shared evidence layer should serve specialist archetype assessments. Seventeen separately invented market histories would create inconsistency, not broader vision. Role separation does not require five model calls per trade.

### Shared snapshot

Each immutable snapshot carries instrument/timeframe, decision cutoff, evidence IDs, values, observed/defaulted/missing status, source references, formula/units, event time, available time, receipt time, consumption time, expiry, dependencies and code/config/source hashes. Hashes prove identity, not authentic historical receipt. Unknown provenance stays unknown.

Agents use only allowlisted read-only evidence. They cannot silently fetch future/revised information, change source versions, fill missing fields or override the clock. News/tool text is untrusted data, never authority to alter instructions or access secrets.

### Archetype rulecards

Each versioned rulecard states required inputs, larger-context requirements, lifecycle, ordered sequence, location, entry/cancellation, invalidation, objective/available room, management and portfolio constraints. Separate hard predicates from contextual preferences. Preserve `unknown`, `not_evaluable`, `not_applicable` and `native_nonemission` as different states.

Do not impose liquidity compression's rules on all 17 archetypes. A minute reversal can be valid inside a larger downtrend if its own objective, room and invalidation justify it. Neutral definitions that cannot currently emit must not acquire invented execution semantics through an agent.

### Assessor, critic and validator

The assessor returns a structured record: snapshot ID; archetype/rulecard versions; model/prompt/tool hashes; generated/decision/expiry times; each predicate's status and evidence IDs; bound parent version/lineage; ordered event references; supporting and opposing claims; unresolved issues; registered plan ID or null; and `supported`, `contradicted`, `unresolved` or `not_applicable` assessment.

An independent critic first examines the evidence before seeing the proposer narrative, then challenges unsupported claims. This reduces anchoring but does not make correlated models statistically independent. A deterministic validator checks references, clocks, arithmetic, allowed plans, hard gates, expiry and risk constraints. Agent agreement cannot overrule a failed hard prerequisite or risk limit.

Initially these are annotations only: no admission, sizing, stop, exit or order authority. Management remains fixed across comparisons. A later management experiment must specify its own states, permissible actions and fresh decision snapshots; it cannot retrospectively rewrite the entry thesis.

For minute execution, slower context assessment can produce an expiring conditional plan ahead of the setup; a fast deterministic trigger acts only if its conditions still hold. Do not place open-ended agent debate on the entry path. Any inference delay and deadline misses must be measured and included in later economic tests.

## What exists versus what is proposed

Existing research components include observed-evidence/context policies in `scripts/research/archetype_context_policy.py`, strict historical parent bindings in `causal_parent_ledger.py`, and fixed-event H3 annotations in `parent_context_policy.py`. These are narrow, explicitly uncertified research interfaces—not the complete agent layer.

The latest coverage work produced 3,764 annotations across 941 minute candidates, not certification of all 17 archetypes. Hourly source-faithful replay exists, but exact outer-stage isolation and executable historical fills remain unresolved. Full rulecards, agent snapshots/manifests, semantic critics, agent evaluation and promotion infrastructure described here are proposals.

## Smallest useful pilot

First freeze an annotation-only evaluation for hourly liquidity compression and the separate minute equal-low sweep/reclaim pattern. Give both equal standing, but distinct clocks, source limitations and execution contracts. Preserve the full native roster and minute candidate denominator; the other rulecards remain visibly unimplemented, not deleted. Include losses, nonselections and invalid examples—not just successful trades.

Measure evidence fidelity, correct causal references, abstention, repeatability and latency before testing trade selection. Compare:

1. Native baseline.
2. Deterministic enriched-evidence/rulecard baseline.
3. One reasoner using the same evidence.
4. Reasoner plus critic.
5. A larger single-agent reasoning budget matched to the multi-agent inference budget.

Report incremental inference cost against deterministic processing; do not assign fictional LLM charges to the deterministic baseline. Pin model/prompt versions and archive requests/responses. Test missing/stale/defaulted inputs, wrong units, future/revised data, exact availability boundaries, lineage breaks, duplicate/reordered evidence, prompt injection, outages and expiration. Report p50/p95/p99 latency and deadline failures.

Only a separately frozen later intervention can test economics through complete stateful replay, including cooldown, dedup, displaced opportunities, full exits, costs, gaps and execution delay. Use chronological separation and account for overlapping trades and common parent lineages. Register every model/prompt/rule trial. No shared future-informed memory across folds or online learning during the evaluation.

Masking dates/tickers or rescaling prices can diagnose memorization but cannot prove its absence. Genuinely forward shadow observations after model/prompt/rule freeze are needed before any promotion claim. No new external data uploads or paid inference pipeline are authorized by this advisory.

## New-archetype discovery is a separate workbench

Agents may propose falsifiable rulecards from a frozen discovery corpus, stating supporting evidence, counterexamples, required inputs, entry/invalidation/management and how the hypothesis differs from the existing 17. Then build a deterministic detector, freeze the proposal and trial registry, evaluate chronologically with costs and concentration, and collect new forward paper evidence before separately authorized promotion.

Novelty, causal detectability, economic edge and operational reliability are four different requirements. Mining winning trades or clustering revised features does not satisfy them. No discovered archetype self-promotes into the live engine.

## Current economics constrain the story

The [four-month experiment](monthly_parent_economics_2026_09_11.md) did not establish an edge for the frozen minute pattern with parent filters and fixed management. Agents did not participate in those outcomes. Better explanations cannot retroactively turn that test positive.

Next bounded work: diagnose existing entered-trade losses by stop/time exit, entry-bar stops, adverse gaps, gross expectancy versus modeled costs and parent concentration; separately freeze the annotation-only pilot. Assessing every raw candidate's predictive value requires a separately specified event study. Do not optimize N, restore fusion thresholds, change management or enable live autonomy from these results.
