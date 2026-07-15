# Industry Study: How Pros Improve Live Strategies (Post-Entry Alpha)

**Date**: 2026-07-14. **Method**: deep-research workflow, 104 agents, adversarially verified
(3-0 votes; 2 claims refuted and excluded). **Trigger**: user question — if entries can't predict
outcomes, what after entry can, and how do firms like Renaissance improve live strategies?

## The verified professional playbook (ranked for a solo operator with 2 validated long-only strategies)

1. **META-LABELING (the #1 pro move)**: never retune the primary strategy — layer a secondary
   sizing model on top that estimates P(this signal wins) and sizes the bet accordingly (size=0
   subsumes filtering). Crucially, feed it features the PRIMARY doesn't use — for us that means
   POST-ENTRY behavior + regime state. Peer-reviewed (López de Prado AFML; Joubert JFDS 2022;
   Meyer/Barziy/Joubert 2023 w/ open-source code, six sizing algorithms benchmarked).
2. **MAE/MFE-BASED TRADE MANAGEMENT** (Sweeney 1996): winners and losers show statistically
   distinct post-entry excursion signatures — codify cut-loser/stop/profit rules from your OWN
   excursion distributions (100+ trades). **Directly validates our path-conditional study**: our
   curves (LC: P(win | MFE<0.25R @12h) = 0%) are exactly this methodology.
3. **BREADTH OVER DEPTH** (Grinold IR = IC·√N, with the Ding/Martin correction that IC
   *consistency* bounds IR): at N=2 strategies, adding a third UNCORRELATED edge is the
   highest-marginal-value move available — worth more than any tuning of the existing two.
4. **TRANSFER-COEFFICIENT REALITY** (IR = TC·IC·√BR): long-only constraints mean only TC² of
   realized P&L variation reflects signal skill — **short live windows are mostly noise**, which
   quantifies why judging strategies on recent live P&L is invalid (our n=1 wick_trap loss).
5. **EXECUTION ALPHA IS FIRST-ORDER**: maker-vs-taker on perps ≈ 2bp/side swing; queue position
   ≈ another 0.7bp. For our ~500-trade/yr scale: real money, the quietest available edge.
6. **WHAT THEY NEVER DO**: retune entries on recent live losses (Bailey/López de Prado: uncounted
   trials make any backtest "worthless"; the trial counter "cannot be turned back") — formal
   endorsement of our frozen-entries discipline.

## Honest gaps (didn't survive verification)
- Live-degradation detection specifics (CUSUM/SPRT parameters, sample sizes): nothing survived —
  only the negative result (short windows ≈ noise) and trial-counting discipline.
- No direct evidence of named firms' internal practice — published literature only.
- Grinold math is cross-sectional equity portfolio theory; mapping to 2 single-asset strategies
  is analogy. Meta-labeling evidence is backtest/synthetic, not audited live records.

## Implications for Bull Machine (in order)
1. The path-conditional/time-cut study now running = playbook #2, already in flight.
2. If time-cut passes → it's also the seed of playbook #1 (post-entry features are exactly what a
   meta-labeling layer would consume; the sizing-not-filtering framing matches Rule 8).
3. Playbook #3 says the next PROJECT after this is a third uncorrelated edge (the short side or a
   non-flush long thesis) — not more work on the existing two.
4. Playbook #5: check our paper engine's fee assumption (taker 2bp) vs a maker-first live policy —
   cheap study, real compounding.

## Cross-references
[[path_conditional_verdict_2026_07_14]] (our empirical companion) ·
[[industry_study_backtest_live_parity_2026_06_11]] · [[trailing_sweep_verdict_2026_07_13]]
