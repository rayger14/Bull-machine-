# LC subtype and evidence preflight — first implementation milestone

## Completed

Added `scripts/research/lc_setup_preflight.py`, a versioned, outcome-free adapter around the existing sealed context extractor. It labels candidate theses from trusted hourly geometry, preserves source-bound daily/4H and1m/5m context, retains unknowns and exposure flags, and never grants execution or assessment dispatch authority. All existing pinned code and packets are unchanged.

Definitions are descriptive, not fitted entry rules:

- `downside_rebound_candidate`: close below the previous hourly low, or a sweep followed by reclaim of that low (unless close is above the previous hourly high).
- `upside_expansion_candidate`: close above the previous hourly high.
- `unresolved`: other or untrusted geometry.

RSI cannot override geometry. A downside label does not prove selling exhaustion; an upside label does not prove continuation. Broken or absent parents remain information for judgment, not automatic rejection. All original candidate eligibility remains unchanged.

## Real frozen roster check

20 original packet file hashes match their existing evidence lock, and packet integrity/context checks passed. No archive re-reconstruction, outcomes or paid assessment calls were run.

| Scope | Downside candidates | Upside candidates | Total |
|---|---:|---:|---:|
| Original frozen roster | 13 | 7 | 20 |
| Already assessed and outcome-revealed Jan20/25 | 2 | 0 | 2 |
| Still awaiting decisions | 11 | 7 | 18 |

All20 have known hourly and1m/5m operands, known parent evidence states and verified native-long status within the caller-verified reconstruction. `evidence_ready` means sufficient supplied evidence to assess, not profitable, live-authenticated, or authorized to dispatch. Known parent absence is not missing data. Remaining18 are labeled `no_new_reveal_recorded`, not untouched holdout. Previous source research and sampling exclusions remain relevant.

Local report: `results/lc_context_preflight_2026_09_16/report.json`, with source packet file hashes, adapter hash, chronological cases, subtype, exposure, evidence reasons and compact context. Complete context can be reproduced from the bound original packets. An initial full-context print was truncated; only the subsequent complete compact JSON was parsed and saved.

## Verification

12 new tests failed before implementation, then passed. Final new preflight plus existing context and assessment suites: **77 passed in1.75seconds**. Tests cover geometry versus RSI, input nonmutation, untrusted/future minute observations, known-absent parent state, duplicate IDs, exposure accounting and seal tampering. `git diff --check` passed. No new dependencies or production changes.

## Four-requirement status

1. **LC subtype distinction:** implemented as research annotation, not a production gate.
2. **Validated assessor context:** existing context is now bound to subtype and checked for the20 historical packets. Single-assessor request/execution integration remains pending. September live1m source/instrument/availability remains unresolved; don't substitute the Binance archive endingAug31.
3. **Shadow decisions without changing orders:** descriptive adapter has no execution authority. Historical single-role decision capture and running live shadow service are not yet implemented.
4. **Preserve winners/avoid losers:** comparison design recorded; no new policy backtest result or promotion claim this turn.

## Next executable milestone

Implement a new single-specialist research contract using existing published request validation and fixed immediate/wait/reject menu, with explicit `unreviewed` versus `critic_reviewed` provenance. Preserve all original two-role contracts; never manufacture critic approval. Fix audit selection, chronological sample and spend ceiling before calls. Then compare separate native immediate, mechanical-wait, agent and reject-all books with unchanged costs/stops/sizing and measured delays. There are18 already-prepared candidate packets across both subtypes; do not rerun the source census or select a winner by its outcome to populate the test.

The full four-stage rollout is NOT finished. This milestone removes subtype/coverage uncertainty and supplies tested inputs for the cheaper execution path. Branch remains `quant/archetype-evidence-audit`; changes are local, not committed or pushed.
