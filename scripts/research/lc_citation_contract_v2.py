"""Prospective citation publication for sealed LC context requests.

This is a separately versioned publication/citation component.  It is not an
``lc_context_discrimination_v1`` grader or job input, does not approve a trade,
and does not establish citation entailment or runner integration.
"""
from copy import deepcopy
import re

from scripts.research.assessment_evidence_guard import _canonical, resolve_evidence
from scripts.research.conditional_assessment import digest
from scripts.research.evidence_id_assessment import build_catalog
from scripts.research.lc_context_assessment import POLICY, validate_context_request


VERSION = "lc_citation_publication_v2"
KIND = "prospective_citation_publication"
_CATALOG_KEYS = {"group_catalog", "evidence_catalog"}
_FINE_ID = re.compile(r"E\d{4}")
_PUBLICATION_FIELDS = {
    "version", "kind", "case_id", "source_binding", "published_request",
    "published_request_sha256", "citation_catalog", "citation_catalog_sha256",
    "accepted_id_schema", "limits", "publication_sha256",
}
_SOURCE_BINDING_FIELDS = {
    "original_request_version", "original_request_sha256", "original_declared_seal",
}
_ENTRY_FIELDS = {"kind", "locator"}
_ACCEPTED_ID_SCHEMA = {
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
_LIMITS = {
    "citation_membership_only": True,
    "semantic_entailment_validated": False,
    "trade_approval": False,
    "decision_grading": False,
    "runner_integration": False,
    "execution_authorized": False,
}


def _hex_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _published_view(value):
    """Detach source data while omitting stale catalogs and integrity labels."""
    if isinstance(value, dict):
        return {
            key: _published_view(item)
            for key, item in value.items()
            if key not in _CATALOG_KEYS and key != "seal" and not key.endswith("_sha256")
        }
    if isinstance(value, list):
        return [_published_view(item) for item in value]
    return deepcopy(value)


def _contains_forbidden_view_key(value):
    if isinstance(value, dict):
        return any(
            key in _CATALOG_KEYS or key == "seal" or key.endswith("_sha256")
            or _contains_forbidden_view_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_view_key(item) for item in value)
    return False


def _publication_body(request):
    packet = request["source_packet"]
    groups = packet["group_catalog"]
    fine = build_catalog(packet)
    if set(groups) & set(fine):
        raise ValueError("group and fine citation ID collision")
    if any(not isinstance(name, str) or not name or _FINE_ID.fullmatch(name)
           for name in groups):
        raise ValueError("group citation ID collides with reserved fine-ID schema")

    view = _published_view(request)
    catalog = {}
    for evidence_id, path in groups.items():
        catalog[evidence_id] = {
            "kind": "group",
            "locator": ["published_request", *deepcopy(path)],
        }
    for evidence_id, path in fine.items():
        if not isinstance(evidence_id, str) or not _FINE_ID.fullmatch(evidence_id):
            raise ValueError("compiler returned an unsupported fine citation ID")
        catalog[evidence_id] = {
            "kind": "fine",
            "locator": ["published_request", "source_packet", *deepcopy(path)],
        }

    body = {
        "version": VERSION,
        "kind": KIND,
        "case_id": request["case_id"],
        "source_binding": {
            "original_request_version": request["version"],
            "original_request_sha256": digest(request),
            "original_declared_seal": request["seal"],
        },
        "published_request": view,
        "published_request_sha256": digest(view),
        "citation_catalog": catalog,
        "citation_catalog_sha256": digest(catalog),
        "accepted_id_schema": deepcopy(_ACCEPTED_ID_SCHEMA),
        "limits": deepcopy(_LIMITS),
    }
    for entry in catalog.values():
        try:
            resolved = resolve_evidence(body, entry["locator"])
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ValueError("citation locator does not resolve in publication") from exc
        if resolved is None:
            raise ValueError("citation locator resolves to unsupported null content")
    return body


def _validate_publication_structure(publication):
    try:
        _canonical(publication)
        if not isinstance(publication, dict) or set(publication) != _PUBLICATION_FIELDS:
            raise ValueError("invalid citation publication shape")
        if (publication["version"] != VERSION or publication["kind"] != KIND
                or not isinstance(publication["case_id"], str)
                or not publication["case_id"].strip()):
            raise ValueError("invalid citation publication identity")
        binding = publication["source_binding"]
        if (not isinstance(binding, dict) or set(binding) != _SOURCE_BINDING_FIELDS
                or binding["original_request_version"] != POLICY
                or not _hex_digest(binding["original_request_sha256"])
                or not _hex_digest(binding["original_declared_seal"])):
            raise ValueError("invalid source request binding")
        view = publication["published_request"]
        if not isinstance(view, dict) or _contains_forbidden_view_key(view):
            raise ValueError("published request contains a competing catalog or stale integrity label")
        if publication["published_request_sha256"] != digest(view):
            raise ValueError("published request digest mismatch")
        if _canonical(publication["accepted_id_schema"]) != _canonical(_ACCEPTED_ID_SCHEMA):
            raise ValueError("accepted evidence_ids schema mismatch")
        if _canonical(publication["limits"]) != _canonical(_LIMITS):
            raise ValueError("citation publication limits mismatch")

        catalog = publication["citation_catalog"]
        if not isinstance(catalog, dict) or not catalog:
            raise ValueError("citation catalog must be nonempty")
        for evidence_id, entry in catalog.items():
            if type(evidence_id) is not str or not evidence_id:
                raise ValueError("invalid citation ID")
            if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
                raise ValueError("invalid citation catalog entry")
            kind, locator = entry["kind"], entry["locator"]
            if (kind == "fine" and not _FINE_ID.fullmatch(evidence_id)):
                raise ValueError("invalid fine citation ID")
            if kind == "group" and _FINE_ID.fullmatch(evidence_id):
                raise ValueError("group citation ID uses the reserved fine-ID schema")
            if kind not in ("group", "fine"):
                raise ValueError("invalid citation kind")
            if (not isinstance(locator, list) or not locator
                    or locator[0] != "published_request"):
                raise ValueError("citation locator is not rooted in the publication")
            if resolve_evidence(publication, locator) is None:
                raise ValueError("citation locator resolves to unsupported null content")
        if publication["citation_catalog_sha256"] != digest(catalog):
            raise ValueError("citation catalog digest mismatch")
        body = deepcopy(publication)
        actual_digest = body.pop("publication_sha256")
        if not _hex_digest(actual_digest) or actual_digest != digest(body):
            raise ValueError("citation publication digest mismatch")
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError("invalid citation publication") from exc


def publish_citation_contract(request):
    """Return a tamper-evident citation view without mutating the v1 request."""
    validate_context_request(request)
    body = _publication_body(request)
    publication = deepcopy(body)
    publication["publication_sha256"] = digest(body)
    _validate_publication_structure(publication)
    return publication


def validate_citation_contract(request, publication):
    """Validate exact source binding and deterministic prospective publication."""
    validate_context_request(request)
    _validate_publication_structure(publication)
    if publication["source_binding"]["original_request_sha256"] != digest(request):
        raise ValueError("publication is bound to a different source request")
    expected = publish_citation_contract(request)
    if _canonical(publication) != _canonical(expected):
        raise ValueError("citation publication differs from its source request")


def validate_citation_ids(publication, ids):
    """Validate and resolve evidence IDs; membership is not semantic approval."""
    _validate_publication_structure(publication)
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 8
            or any(type(evidence_id) is not str for evidence_id in ids)
            or len(ids) != len(set(ids))
            or any(evidence_id not in publication["citation_catalog"] for evidence_id in ids)):
        raise ValueError("evidence_ids must be 1..8 unique exact citation_catalog keys")
    resolved = []
    for evidence_id in ids:
        entry = publication["citation_catalog"][evidence_id]
        resolved.append({
            "evidence_id": evidence_id,
            "kind": entry["kind"],
            "locator": deepcopy(entry["locator"]),
            "resolved_value": resolve_evidence(publication, entry["locator"]),
        })
    return resolved
