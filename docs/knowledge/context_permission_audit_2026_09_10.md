# Independent evidence and prior-compression audit

## Delivered scope

`scripts/research/archetype_context_policy.py` implements three independent audit-only policies:

- `oi_observed_evidence_v1`: OI horizons plus taker output and its raw branch inputs.
- `lc_observed_evidence_v1`: LC numeric feature inputs plus ADX dependency for chop.
- `lc_prior_compression_v1`: immediately previous completed hour has finite BB width ≤0.06.

They do not change an archetype's numeric gates or invoke candidates, cooldown, selection, deduplication, position sizing or execution. `annotate_pipeline` returns a separate sidecar. These are permission diagnostics, not a backtest of enforcing a policy.

H1 requires explicit field/instrument/value/digest/consumed-path identity, expected source/units/formula contracts, observed/computed origin, observation/dependency hash references and event/available/received/expiry/consumed timestamps. Availability uses the later of publication and receipt; `valid_until` is exclusive, matching the replay clock. The actual hourly feature-availability time must be supplied, and the row must have closed by consumption. Taker volume and ratio branches bind their raw values and formula, including nonnegative volumes, positive finite denominator and genuine balanced-flow zero. Defaulted zero is not observed zero.

Even complete metadata is **attested evidence, not verified acquisition**. Opaque hashes do not prove the provider sent a record or that native code consumed it. The present native feature adapter does not emit the necessary field-level consumption chain; an observation ID appended after replay cannot substitute for that chain. Historical missing evidence therefore stays UNKNOWN. No invented TTL or historical formula mapping is used.

H2 is deliberately separate, numeric-only and not proof of genuine compression. Finite warmup defaults can satisfy its numeric comparison while H1 remains unknown. Combining the policies, or enforcing either natively, is a separate experiment.

## Tests and independent review

20 focused tests pass. Initial absent implementation produced RED; additional failing witnesses preceded fixes for missing consumption target, overflowing taker denominator, arbitrary formula labels, future feature hours, expiry-boundary disagreement and EXIT rows mislabelled as entries. Independent reviewer reproduced four issues, then approved their fixes after six targeted checks. Source/formula expectations are deep-copied into the manifest and affect contract identity. Tests cover zero semantics, invalid/nonfinite/late/stale inputs, raw taker branches, LC ADX metadata, exact previous-hour boundaries, minute carry, full-history restart, prefix stability and native nonmutation.

## Historical diagnostic: all 240 supplied hours

Input: OHLCV only, V23 window June 10 00:00 to June 20 00:00 UTC, 2026, 240 hourly bars. Native paper-book parameters: $100,000 initial cash, 4 bp commission, 5 bp slippage. **These are diagnostic accounting parameters, not executable-return assumptions. Average initial risk per trade was not computed; no risk-adjusted performance claim.** No external provider observations or attested consumption contracts were supplied.

Reran the guarded native pipeline with all history emitted, not just the earlier final 72 hours. Both 120-hour and 239-hour cutoffs pass prefix rows, restarted rows, full final state and contract equality checks. Sidecar separately preserves its 120-row prefix and fresh full-history rebuild; native input digest is unchanged after annotation.

Cross-run check against the earlier 72-emitted-row artifact found identical overlapping feature/signal/book values except native book contract IDs; final state is numerically equal. Exact type-tagged hashes differ because this Python invocation supplied integer `100000`/`5` where the earlier CLI supplied floats `100000.0`/`5.0`. Source/config/input hashes otherwise match. This is an explicit constructor-type identity difference, not an observed trading-behaviour change; do not claim cross-run byte-for-byte contract equality.

| Independent policy | Pass | Reject | Unknown | Native selected candidates | Selected without policy permission |
|---|---:|---:|---:|---:|---:|
| OI observed evidence | 0 | 0 | 240 | 8 | 8 |
| LC observed evidence | 0 | 0 | 240 | 1 | 1 |
| LC prior-hour BB width | 239 | 1 | 0 | 1 | 0 |

The sole H2 rejection is the initial row with no supplied previous hour. The narrow window does not establish useful H2 discrimination: all subsequent numeric comparisons pass. It does not justify optimizing the threshold on this window. UNKNOWN is not proof all inputs were bad; it means this diagnostic cannot certify their consumption provenance.

OI had eight native candidates/selected signals, but one entry and seven later book rejections. LC had one candidate, one entry and four recorded exit events. Do not turn candidate counts or scale-outs into independent trades. LC's source-labelled entry is June 14 21:00; OI's is June 18 16:00. Native decision closes are one hour later; neither is verified market execution time.

No “avoided loss” calculation is made. Enforcing a new permission before candidate detection can change cooldown, displaced trades and positions, requiring fresh native replay—not a post hoc filter of this ledger.

## Private artifacts

Under `results/research_validation_2026_09_10/native_pipeline/`:

- `hourly_240h_all_emitted.json`, SHA-256 `418ab56dd2a774343fb98a81739e72976ffc267ee73600756f01d1935d855167`.
- `hourly_240h_context_audit.json`, SHA-256 `cd135fb9b0e0b3a17d6127014ca5b25b92ddc7a4b3bb38b1202dfa0a0ab17834`.
- Policy contract `a357aeac116b8d1abe9a0d095c8289f1cf87d5f13834bf32bbf60d3f494bdabd`.

Artifacts remain ignored/private. No native production code, configuration, live settings or source data changed.

## Next dependency

The [parent ledger specification](../superpowers/specs/2026-09-10-causal-parent-ledger-design.md) creates causal, frozen parent identities before attaching child setups. Native observed-consumption instrumentation remains necessary before H1 can become a trustworthy enforced gate. Keep these concerns separate from profitability validation.
