# Industry study: TRUE vs FALSE stop-hunt flushes (2026-07-28)

Deep-research (100 agents, 3-vote adversarial verification) on how the
founding-knowledge traders + orderflow literature qualify a flush AT the
flush bar. Full cited result: workflow wf_b0e54474-318.

## The verified consensus
A buyable flush is qualified by EFFORT-VS-RESULT MISMATCH + RECLAIM, not by
the flush itself. Ranked encodable signal stack:
1. **Reclaim close** (top signal): bar dips BELOW the swept level but CLOSES
   back above it, long lower wick. The reclaim — not the breach, not the
   volume — qualifies. (= our Bojan wick-majority boost, staged)
2. **Absorption signature**: high volume z + NEGATIVE delta/taker imbalance
   (sellers aggressing) + minimal penetration + close near range high =
   passive buyers absorbing panic. "Buyers aggressing while price falls" =
   someone distributing into them = the FALSE flush. (= our taker-flow
   boost, running — INDEPENDENTLY CONFIRMS the 2026-07-28 data discovery,
   direction and mechanism. Also the July-27 loser's exact anatomy:
   ti=+0.06 buyer-aggressed flush.)
   Caveat verified: raw delta alone is directionless/continuation-leaning —
   only the MISMATCH (aggression without price progress) predicts.
3. **Symmetric falsification**: close below the flush low converts reversal
   → continuation (former support = resistance).
4. Recovery quality: declining volume + narrowing ranges + higher low on
   the retest.
5. Hard disqualifier: retest makes a DEEPER low on HIGHER volume.
6. Bullish CVD divergence (price LL without CVD LL) — needs CVD built
   (archaeology item; order_flow_delta columns are dead in store).
7. Orthogonalized delta (residual after regressing out concurrent return) —
   the academically-validated form (JFM 2026), but only proven at
   daily/weekly horizons.

## Verification honesty
- NO published study validates any of this at the 1H flush-bar level.
- EVERY specific numeric threshold in blog sources failed adversarial
  verification — thresholds must be fit in-house (ours are pre-registered
  a-priori definitions, not fitted).
- Some verification votes ran while the safety classifier was degraded;
  core claims had 3-0 votes with broad cross-corroboration.

## Program mapping
Boost 1 (taker_imbalance<=0, running) = signal #2. Boost 2 (wick_lower_ratio
>=0.5, staged behind #1) = signal #1. Signals #3-#5 are post-entry/exit-
shaped (0-for-5 territory — only via pre-registered watch-items). #6 needs
CVD feature build (V17 candidate with effort_result_ratio). #7 = future
meta-labeling input.
