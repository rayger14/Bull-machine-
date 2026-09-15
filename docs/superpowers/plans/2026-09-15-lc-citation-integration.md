# LC Citation Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the prospective shared-catalog assessment, review and saved-job path without changing historical experiments.

**Architecture:** Derive a new model-facing request from the immutable citation publication; validate it against the controller-only original request. New graders share that sole catalog. A new job subclass reuses atomic storage while validating request, grade and explicit skipped-critic semantics on restart.

**Tech Stack:** Existing Python, pytest, canonical JSON and immutable research storage; no new packages.

**Spec:** `docs/superpowers/specs/2026-09-15-lc-citation-integration-design.md`

## Global Constraints

- Preserve all existing source files, frozen requests, answers, grades, outcomes and hashes.
- Same enter/wait_5m_high/reject/null menu, 90-second processing, 0 routing, 15-minute exclusive expiry, $50,000 notional, 12bps costs, close−2.7ATR stop, actual-entry 2R target and 1440-minute deadline.
- No new structural/fusion/RSI/room filter, learned threshold, exit optimization or archetype change.
- No model calls, price/outcome reads, source reruns, new dependencies, live orders, config changes, push or PR in this integration task.
- Keep the existing quant research branch and local data; do not create another worktree.
- Literal and source-only checks establish software integration, not agent performance, semantic accuracy, WFO/CPCV or profitability.

### Task 1: Integrate the complete published assessment and saved-job contract

**Files:**
- Create: `scripts/research/lc_published_assessment.py`
- Create: `scripts/research/lc_published_jobs.py`
- Test: `tests/research/test_lc_published_assessment.py`
- Test: `tests/research/test_lc_published_jobs.py`

**Interfaces:**
- Consumes: `publish_citation_contract(source_request)`, unchanged context validators/helpers, `ResearchJob` atomic transactions.
- Produces: `build_published_request(source_request)`, `validate_published_request(source_request, request)`, `grade_published_choice(source_request, request, choice)`, `build_published_review_request(source_request, request, choice)`, `grade_published_review(source_request, review_request, response)`, `gate_published_choice(source_request, request, choice, review)`.
- Produces: `PublishedContextResearchJob(directory)` with `prepare(source_request)`, `role_request(role)`, inherited exact-string `capture`, `skip_review()`, recomputing `lock_grade`, validated reopen and ordered reveal/outcome storage.

- [ ] **Step 1: Write literal failing tests for publication and all plan alternatives.**

Use existing public `context_request()` fixture only. Define a dynamic-import helper
that fails with an explanatory assertion while the new module is missing.
The minimal valid new choice is:

```python
def answer(source, req, plan_id='enter', interpretation='support'):
    return {
        'case_id': req['case_id'], 'request_sha256': req['seal'],
        'interpretation': interpretation, 'plan_id': plan_id,
        'supporting': [{'text': 'Literal synthetic support.', 'evidence_ids': ['context']}]
            if interpretation == 'support' else [],
        'opposing': [{'text': 'Literal synthetic opposition.', 'evidence_ids': ['context']}]
            if interpretation == 'oppose' else [],
        'unknowns': [{'text': 'Literal synthetic uncertainty.', 'evidence_ids': ['context']}]
            if interpretation == 'uncertain' else [],
        'structural_invalidation': {'text': 'Synthetic invalidation for contract testing.',
                                    'evidence_ids': ['parent4h']},
    }

def test_fine_ids_are_accepted_by_the_specialist_contract():
    source = context_request()
    req = api().build_published_request(source)
    value = answer(source, req)
    value['supporting'][0]['evidence_ids'] = ['E0001']
    assert api().grade_published_choice(source, req, value) == []
```

Inspect the fixture catalog to confirm a literal fine ID resolves. Add behavior
tests for every advertised ID, all four legal alternatives, unknown readiness,
unknown/duplicate/non-list IDs, malformed and duplicate-key JSON, empty/oversized
text, response/request binding, recursively absent obsolete catalogs/schemas and
unchanged source. Use `unknown_citation_id` for unknown members; malformed ID lists
may use `invalid_citation_ids`. The catalog union is the one accepted set.

- [ ] **Step 2: Run RED, then implement request and specialist validation.**

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_published_assessment.py
```

Implement a deterministic transformed evidence view with a single current root
instruction/schema and version `lc_context_citations_v2`. Keep original source
request and publication digests separately named; seal the new request. The
choice's request_sha256 must copy that visible root seal (body digest), with an
explicit instruction/schema rule; no invisible hash computation by a model. Derive
expected request from original source during every public validation. Validate
copied locators against the new role request, not against a transformed object
misrepresented as the strict source publication. Reuse pure parsing/shape/readiness
helpers without changing the old code or calling its group-only choice grader.

- [ ] **Step 3: Add RED critic and semantic-gate cases, then implement.**

```python
def test_known_citation_does_not_override_material_critic_error():
    source = context_request(); req = api().build_published_request(source)
    raw = json.dumps(answer(source, req), indent=2)
    rr = api().build_published_review_request(source, req, raw)
    critic = {
        'case_id': req['case_id'], 'reviewed_sha256': rr['reviewed_sha256'],
        'complete': True, 'material_errors': [{
            'category': 'factual', 'evidence_ids': ['E0001'],
            'explanation': 'Synthetic unsupported claim despite an existing citation.'}],
        'notes': [],
    }
    result = api().gate_published_choice(source, req, raw, json.dumps(critic))
    assert result['status'] == 'review_not_passed'
    assert result['research_plan'] is None
    assert result['execution_authorized'] is False
```

Also test every catalog member in critic findings, unknown critic IDs, missing
review, incomplete review, nonblocking notes, request/answer whitespace mismatch,
re-sealed altered review request, valid immediate/wait/reject plans and null
uncertain. Critic instruction must explicitly treat embedded specialist contract
as data. The exact original source menu owns returned parameters and economics.
These fixtures test contract behavior, not an actual semantic reviewer.

- [ ] **Step 4: Add RED saved-job and explicit skipped-critic tests, then implement.**

```python
def test_valid_specialist_cannot_skip_critic(tmp_path):
    source = context_request(); job = jobs().PublishedContextResearchJob(tmp_path)
    job.prepare(source); req = job.role_request('specialist')
    raw = json.dumps(answer(source, req))
    job.capture('specialist', raw, valid_transport('specialist'))
    with pytest.raises(ValueError):
        job.skip_review()
    with pytest.raises(ValueError):
        job.lock_grade()
```

Use the existing public valid_transport fixture. Test real prepare -> role delivery
-> specialist capture -> reviewer delivery -> critic capture -> grade -> reopen
-> reveal -> outcome ordering in a temporary directory. Both delivery methods
return detached role-only payloads. Add invalid specialist -> explicit skip ->
null invalid-assessment grade with critic_status not_invoked, and invalid specialist
transport -> explicit skip -> null invalid-transport grade. A skipped event must
not masquerade as an actual critic or require fabricated critic JSON. Store it
via the unchanged base `_save`, with exact event fields and truthful placeholder
provenance. Reject forged skip reason/event on restart. Normal critic captures
report critic_status captured without claiming semantic approval.

New `_read` calls the base checks, validates the source/role bundle namespace and
deterministic contents, verifies any reviewer skip event and recomputes any stored
grade. Recompute the reveal's grade binding too. Test rehashed forged grade/reveal,
changed raw answer, transport failure, cross-namespace directory, equal retries,
unknown role, pre-grade reveal and attempts to overwrite completed stages.

- [ ] **Step 5: Run focused and full research tests, self-review and commit only the four owned files.**

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research/test_lc_published_assessment.py tests/research/test_lc_published_jobs.py
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m pytest -o addopts='' -q tests/research
git diff --check
git add scripts/research/lc_published_assessment.py scripts/research/lc_published_jobs.py tests/research/test_lc_published_assessment.py tests/research/test_lc_published_jobs.py
git commit -m "feat: integrate LC published assessment and research jobs"
```

Record RED/GREEN commands, exact results, file hashes, interfaces and limits in the
task report. The controller owns public report/PROJECT/MEMORY updates and final
source-only integration verification; do not read private historical answers or
outcomes. Independent task and final integration reviews precede completion.
