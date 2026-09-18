"""Literal public tests for the prospective LC citation publication."""
from copy import deepcopy
import importlib
import re

import pytest

from scripts.research.assessment_evidence_guard import resolve_evidence
from scripts.research.conditional_assessment import digest
from tests.research.test_lc_context_assessment import context_request


VERSION = "lc_citation_publication_v2"
SCHEMA = {
    "field": "evidence_ids",
    "authority_path": ["citation_catalog"],
    "accepted_set": "exact_keys",
    "cardinality": {
        "minimum": 1,
        "maximum": 8,
        "unique": True,
        "item_type": "string",
    },
    "sources": {
        "group": "validated_request.source_packet.group_catalog",
        "fine": "compiler.build_catalog(validated_request.source_packet)",
    },
    "locator_root": "published_request",
}
LIMITS = {
    "citation_membership_only": True,
    "semantic_entailment_validated": False,
    "trade_approval": False,
    "decision_grading": False,
    "runner_integration": False,
    "execution_authorized": False,
}


def api():
    name = "scripts.research.lc_citation_contract_v2"
    assert importlib.util.find_spec(name), "prospective citation publisher missing"
    return importlib.import_module(name)


def nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


def reseal_publication(publication):
    publication["published_request_sha256"] = digest(publication["published_request"])
    publication["citation_catalog_sha256"] = digest(publication["citation_catalog"])
    body = deepcopy(publication)
    body.pop("publication_sha256", None)
    publication["publication_sha256"] = digest(body)
    return publication


def test_literal_publication_unifies_every_group_and_fine_id_without_mutation():
    request = context_request()
    before = deepcopy(request)

    publication = api().publish_citation_contract(request)

    assert request == before
    assert api().publish_citation_contract(request) == publication
    assert publication["version"] == VERSION
    assert publication["kind"] == "prospective_citation_publication"
    assert publication["case_id"] == request["case_id"]
    assert publication["accepted_id_schema"] == SCHEMA
    assert publication["limits"] == LIMITS
    assert publication["source_binding"] == {
        "original_request_version": request["version"],
        "original_request_sha256": digest(request),
        "original_declared_seal": request["seal"],
    }

    groups = request["source_packet"]["group_catalog"]
    fine = request["source_packet"]["evidence_catalog"]
    assert set(publication["citation_catalog"]) == set(groups) | set(fine)
    assert set(groups).isdisjoint(fine)
    for evidence_id, source_path in groups.items():
        entry = publication["citation_catalog"][evidence_id]
        assert entry == {"kind": "group", "locator": ["published_request", *source_path]}
        expected = resolve_evidence(request, source_path)
        if evidence_id == "context":
            expected.pop("source_packet_sha256")
        assert resolve_evidence(publication, entry["locator"]) == expected
    for evidence_id, source_path in fine.items():
        entry = publication["citation_catalog"][evidence_id]
        assert entry == {
            "kind": "fine",
            "locator": ["published_request", "source_packet", *source_path],
        }
        assert resolve_evidence(publication, entry["locator"]) == resolve_evidence(
            request["source_packet"], source_path
        )
    for evidence_id in publication["citation_catalog"]:
        assert api().validate_citation_ids(publication, [evidence_id])[0]["evidence_id"] == evidence_id

    api().validate_citation_contract(request, publication)


def test_complete_publication_has_one_catalog_and_no_stale_integrity_claims_in_view():
    publication = api().publish_citation_contract(context_request())

    keys = list(nested_keys(publication))
    assert keys.count("citation_catalog") == 1
    assert "group_catalog" not in keys
    assert "evidence_catalog" not in keys
    view_keys = set(nested_keys(publication["published_request"]))
    assert "seal" not in view_keys
    assert not any(key.endswith("_sha256") for key in view_keys)
    assert publication["published_request_sha256"] == digest(publication["published_request"])
    assert publication["citation_catalog_sha256"] == digest(publication["citation_catalog"])
    body = deepcopy(publication); body.pop("publication_sha256")
    assert publication["publication_sha256"] == digest(body)


def test_exact_catalog_keys_are_the_only_accepted_citation_ids_and_resolve():
    publication = api().publish_citation_contract(context_request())
    fine_id = next(key for key in publication["citation_catalog"] if re.fullmatch(r"E\d{4}", key))

    resolved = api().validate_citation_ids(publication, ["current", fine_id])

    assert [item["evidence_id"] for item in resolved] == ["current", fine_id]
    for item in resolved:
        entry = publication["citation_catalog"][item["evidence_id"]]
        assert item["kind"] == entry["kind"]
        assert item["locator"] == entry["locator"]
        assert item["resolved_value"] == resolve_evidence(publication, entry["locator"])


@pytest.mark.parametrize("ids", [
    [], ["missing"], ["current", "current"], "current", [True],
    ["current"] * 9,
])
def test_invalid_unknown_duplicate_or_malformed_citation_ids_are_rejected(ids):
    publication = api().publish_citation_contract(context_request())
    with pytest.raises(ValueError, match="evidence_ids"):
        api().validate_citation_ids(publication, ids)


@pytest.mark.parametrize("change", ["view", "catalog", "catalog_digest", "source_binding"])
def test_tampering_is_rejected_even_when_outer_publication_is_resealed(change):
    request = context_request()
    publication = api().publish_citation_contract(request)
    if change == "view":
        publication["published_request"]["case_id"] = "other"
    if change == "catalog":
        publication["citation_catalog"]["current"]["locator"] = ["published_request", "context"]
    if change == "catalog_digest":
        publication["citation_catalog_sha256"] = "0" * 64
    if change == "source_binding":
        publication["source_binding"]["original_request_sha256"] = "0" * 64
    if change == "catalog_digest":
        body = deepcopy(publication); body.pop("publication_sha256")
        publication["publication_sha256"] = digest(body)
    else:
        reseal_publication(publication)

    with pytest.raises(ValueError):
        api().validate_citation_contract(request, publication)


def test_publication_is_bound_to_the_exact_original_request():
    first = context_request()
    second = context_request(prior_bb=.05)
    publication = api().publish_citation_contract(first)

    with pytest.raises(ValueError, match="source request"):
        api().validate_citation_contract(second, publication)


def test_compiler_id_collision_and_unresolvable_compiler_path_fail_closed(monkeypatch):
    request = context_request()
    module = api()
    monkeypatch.setattr(module, "build_catalog", lambda packet: {"current": ["evidence"]})
    with pytest.raises(ValueError, match="collision"):
        module.publish_citation_contract(request)

    monkeypatch.setattr(module, "build_catalog", lambda packet: {"E0001": ["missing"]})
    with pytest.raises(ValueError, match="resolve"):
        module.publish_citation_contract(request)


def test_structurally_invalid_publication_and_reserved_group_id_are_rejected():
    publication = api().publish_citation_contract(context_request())
    publication["citation_catalog"]["E9999"] = {
        "kind": "group",
        "locator": ["published_request", "context"],
    }
    reseal_publication(publication)

    with pytest.raises(ValueError):
        api().validate_citation_ids(publication, ["E9999"])
