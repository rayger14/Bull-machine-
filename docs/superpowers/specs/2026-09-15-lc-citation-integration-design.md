# LC single-catalog assessment integration

## Purpose and authority

The user approved the next deliverable: connect the reviewed citation publisher
through specialist assessment, independent review and saved research jobs.
Routine quant/design decisions are delegated to the controller and quant reviewer.
This is an offline interface correction, not a new trading hypothesis.

## Chosen design

Create a separately versioned integration in new files. Editing the old grader
would invalidate frozen source hashes; merely widening accepted IDs in a prompt
would leave incompatible schemas and job validation. The new integration instead
reuses the publisher and immutable storage primitives while owning its role
contracts and validation. Keep original source requests controller-side.

### Model-facing request

`build_published_request(source_request)` accepts a valid sealed context-v1
request and calls `publish_citation_contract`. Derive a detached role request
with version `lc_context_citations_v2`, case ID, `published_request` evidence
view, one `citation_catalog`, explicitly labelled original-request/source-
publication bindings, one instruction and one response schema, and a new seal.
The source publication is not falsely labelled as the new request's seal.

The evidence view excludes embedded `instruction`, `response_schema` and
`review_schema` fields in addition to the publisher's existing redactions.
Only the new root instruction/schema are authoritative. Catalog locators remain
rooted at `published_request`; every advertised ID resolves in this exact view.
The role receives neither the original request nor the controller job bundle.
Do not pass the transformed request to the strict original publication validator.

`validate_published_request(source_request, request)` must rebuild the original
publication and expected role request, check exact canonical equality and all
locators. Re-sealing changed context, instructions, menu, catalog or source
bindings cannot bypass validation. Preserve source evidence and fixed economics.

### Specialist and critic

The choice has exact keys: `case_id`, `request_sha256`, `interpretation`,
`plan_id`, `supporting`, `opposing`, `unknowns`, `structural_invalidation`.
`request_sha256` copies the visible `request['seal']`, the digest of the request
body excluding that seal, binding all published evidence, memory, brief, menu
and instructions. The instruction/schema explicitly identify this copy operation;
the model must not compute an unavailable whole-object hash. Items retain
`text` and `evidence_ids`; text is
nonempty and at most 1200 characters, lists at most 8, IDs 1..8 unique exact
catalog keys. Both fine and group IDs are accepted by all consuming paths.

Keep the existing readiness and interpretation mapping: support selects enter
or wait_5m_high; oppose selects reject; uncertain selects null. Unknown required
source evidence permits only uncertain/null. Known adverse structure remains a
judgment, not a new cutoff. No additional confidence/probability fields.

`grade_published_choice(source_request, request, choice)` validates the exact
request and choice. Invalid IDs must produce explicit citation errors distinct
from semantic review findings. Resolving an ID does not prove the claim true.
Reuse existing pure parsing/readiness/shape helpers when their semantics match;
do not modify old modules or remap an answer to trick the old grader.

`build_published_review_request(source_request, request, choice)` requires a
valid choice, preserves its exact raw string and binds the exact request and
answer. It exposes that role request, not controller source data. Critic schema
retains case_id, reviewed_sha256, complete, material_errors and notes. Categories
remain factual, chronology, missing_required and unsupported_plan. Every material
finding uses the same catalog. Discretionary disagreement belongs in nonblocking
notes; missing/incomplete/material-error review blocks a research plan.

`grade_published_review(source_request, review_request, response)` checks the
entire review-request binding and response contract. `gate_published_choice`
returns only the chosen unchanged menu plan after valid complete review, or null
for invalid, missing review, material errors or uncertain choice. It always
returns execution_authorized=false and transport_authenticated=false. This is
research selection, not live authority or independently proven semantic truth.

### Saved jobs and delivery

`PublishedContextResearchJob(directory)` subclasses the unchanged `ResearchJob`
storage transactions. `prepare(source_request)` saves a separately versioned
controller-only bundle with source and derived request. `role_request(role)`
returns only the specialist request or the exact review packet after a valid
captured specialist answer; invalid specialists cannot be routed to a critic.
Unknown roles fail. Returned objects are detached.

Every reopen validates source/role request and re-derives any stored grade from
exact raw captures and declared transport. A self-consistent but wrong grade
hash must fail. Reveal records must bind that grade; pre-grade reveal is refused.
Different role namespaces cannot share an existing directory. Changed answers,
catalogs, request bytes, forged grades and incorrect-stage writes fail closed;
equal retries are idempotent. Keep the base capture provenance contract explicit:
it is caller-declared, not authenticated model identity or actual delivery.

The existing ordered storage requires both role slots before locking a grade.
`skip_review()` may fill the reviewer slot only after recomputing an invalid
specialist contract or failed declared specialist transport. Store a deterministic
`kind='review_not_run'` event with reason `invalid_assessment` or
`invalid_transport`, empty raw text, its real UTF-8 hash and explicitly
controller-not-invoked/transport-false provenance with capture_sha256=null because
no delivery occurred. This is a storage event, not a model response. Validate the
exact event on restart. A valid delivered specialist
cannot skip review; missing reviewer stage cannot lock a grade.

Skipped-review grades preserve the actual specialist failure, set
`critic_status='not_invoked'`, and have null plans. Actual critic captures use
`critic_status='captured'`, which does not imply semantic approval. Never report
a skipped critic as a failed critic invocation or score a null plan as a skip.

Independent quant design review approved this approach, including the explicit
skip event, exact raw binding and restart grade recomputation. The controller
adopted these recommendations under the user's delegated-review authority.

## Fixed boundaries

- Preserve all existing source files, frozen requests, answers, grades, outcomes and hashes.
- Same enter/wait_5m_high/reject/null menu, 90-second processing, 0 routing, 15-minute exclusive expiry, $50,000 notional, 12bps costs, close−2.7ATR stop, actual-entry 2R target and 1440-minute deadline.
- No new structural/fusion/RSI/room filter, learned threshold, exit optimization or archetype change.
- No model calls, price/outcome reads, source reruns, new dependencies, live orders, config changes, push or PR in this integration task.
- Keep the existing quant research branch and local data; do not create another worktree.
- Literal and source-only checks establish software integration, not agent performance, semantic accuracy, WFO/CPCV or profitability.

## Acceptance and next boundary

Tests must exercise real publication -> assessment -> review -> saved job ->
reopen -> grade -> reveal ordering, using literal synthetic answers only.
All advertised IDs must work for specialist items and critic findings. Test
unknown/duplicate/malformed IDs, raw whitespace binding, source/request/review
tampering even after recomputed seals, all legal menu choices, unknown readiness,
failed/missing review, declared transport failure and cross-namespace rejection.

After independent review, run a source-only publication check on existing
requests without reading or repairing old answers. Preserve the completed run's
verification. The next market deliverable is a separately preregistered new
chronological sample with explicit exposure and role limits, not a retry of the
four revealed assessments.
