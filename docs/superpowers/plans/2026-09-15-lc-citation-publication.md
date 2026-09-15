# Prospective LC citation publication correction

User requested actionable continuation; completed four-case run exposed a dual-catalog interface defect. Independent quant audit recommends one explicit accepted catalog before further paid roles. This bounded correction implements publication/citation validation only, not a new market run or retroactive grading.

## Scope

- Create `scripts/research/lc_citation_contract_v2.py` and `tests/research/test_lc_citation_contract_v2.py` only. Preserve every existing source/runner/curriculum/request/grade/outcome file and frozen hash. No model calls, outcome reads, runner/job rewrite or live change.
- Input is an existing valid sealed `lc_context_discrimination_v1` request. Output is a separately versioned **prospective publication contract**, not a request accepted by the old grader or job. Disclose this distinction in metadata/docstrings.
- Publish one authoritative `citation_catalog`: merge packet-owned groups with precise compiler `E####` evidence entries, converting every locator to the publication's own root. Reject collisions, missing targets, unsupported paths or mutable/unbound content.
- The publication contains a deterministic view of the source request, without nested `evidence_catalog`/`group_catalog` copies that appear to define competing ID authorities. Original hashes may remain only as clearly named source bindings, not falsely presented as seals of redacted data. Bind the exact original request digest and the new publication/catalog digests.
- Declare exact accepted IDs/source of `evidence_ids` explicitly. Every advertised ID must be accepted by citation validation and resolve to its own published data. Citation existence alone is not claim entailment, trade approval, new decision grading or execution permission.
- Suggested interfaces: `publish_citation_contract(request)`, `validate_citation_contract(request, publication)`, `validate_citation_ids(publication, ids)`. Keep names clear and implementation small; equivalent better-named functions allowed if report documents them.

## Steps and acceptance

1. Literal RED tests with public context-request fixtures: every fine/group ID resolves and is accepted; no competing catalogs in the complete publication; explicit schema authority; unchanged original request; invalid IDs/duplicates/tampering/source mismatch rejected.
2. Implement the smallest immutable publication/citation validator. Do not copy or rewrite the full trading grader/job runner. If that is necessary, stop and report instead.
3. Focused GREEN, self-review and read-only publication checks on the four already-frozen requests (source-only, no model calls and no grading their answers). Report exact new file hashes and limits. Do not create a replacement run namespace or new brief.
4. Independent scoped review; root checks original run still verifies, reruns tests and records the next integration boundary. Public report must say this is the next contract component, not a tested v2 trading strategy or completed job integration.

## Completion checkpoint

All four bounded steps completed September 15. Sixteen new literal tests pass;
all 798 advertised IDs in the four source-only publication checks resolve and
validate. Independent scoped review approved with no material findings. Root
final verification passed 966 distinct tests across two runs (950 + 16), and the
original run still verifies its completed locked status. No market roles were
invoked for this correction and no old answers were repaired or regraded.

Next work is a separate specialist/critic/grader/job integration, followed by a
new preregistered chronological comparison. This component alone cannot run or
grade a trading assessment and does not establish agent advantage.
