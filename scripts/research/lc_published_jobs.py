"""Immutable offline jobs for the published LC specialist/critic contract."""
from copy import deepcopy
import hashlib

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.conditional_assessment import digest
from scripts.research.lc_published_assessment import (
    build_published_request,
    build_published_review_request,
    gate_published_choice,
    grade_published_choice,
    validate_published_request,
)
from scripts.research.master_research_jobs import ResearchJob


VERSION = "lc_context_citations_job_v2"
_BUNDLE_KEYS = {
    "version", "source_request", "source_request_sha256", "role_request",
    "role_request_sha256",
}
_CAPTURE_KEYS = {"raw_response", "raw_sha256", "provenance"}
_SKIP_KEYS = _CAPTURE_KEYS | {"kind", "reason"}
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SKIP_PROVENANCE = {
    "role": "reviewer",
    "agent_id": "controller-not-invoked",
    "requested_model": "not-invoked",
    "actual_model": None,
    "capture_sha256": None,
    "transport_valid": False,
}


def _bundle(source_request):
    role_request = build_published_request(source_request)
    return {
        "version": VERSION,
        "source_request": deepcopy(source_request),
        "source_request_sha256": digest(source_request),
        "role_request": role_request,
        "role_request_sha256": digest(role_request),
    }


def _validate_bundle(bundle):
    if not isinstance(bundle, dict) or set(bundle) != _BUNDLE_KEYS:
        raise ValueError("invalid published job bundle shape")
    if bundle["version"] != VERSION:
        raise ValueError("invalid published job namespace")
    source = bundle["source_request"]
    request = bundle["role_request"]
    if (bundle["source_request_sha256"] != digest(source)
            or bundle["role_request_sha256"] != digest(request)):
        raise ValueError("invalid published job bundle binding")
    validate_published_request(source, request)
    if _canonical(bundle) != _canonical(_bundle(source)):
        raise ValueError("published job bundle differs from source")


def _skip_reason(bundle, specialist):
    if specialist["provenance"]["transport_valid"] is not True:
        return "invalid_transport"
    errors = grade_published_choice(
        bundle["source_request"], bundle["role_request"], specialist["raw_response"])
    return "invalid_assessment" if errors else None


def _skip_event(reason):
    return {
        "kind": "review_not_run",
        "reason": reason,
        "raw_response": "",
        "raw_sha256": _EMPTY_SHA256,
        "provenance": deepcopy(_SKIP_PROVENANCE),
    }


def _is_skip(payload):
    return isinstance(payload, dict) and payload.get("kind") == "review_not_run"


def _skip_grade(bundle, specialist, reason):
    errors = grade_published_choice(
        bundle["source_request"], bundle["role_request"], specialist["raw_response"])
    if reason == "invalid_transport":
        errors = list(dict.fromkeys(errors + ["unverified_declared_transport"]))
    return {
        "status": reason,
        "research_plan": None,
        "execution_authorized": False,
        "transport_authenticated": False,
        "critic_status": "not_invoked",
        "errors": errors,
    }


def _expected_grade(bundle, specialist, reviewer):
    reason = _skip_reason(bundle, specialist)
    if _is_skip(reviewer):
        if reason is None:
            raise ValueError("valid specialist cannot have a skipped review")
        return _skip_grade(bundle, specialist, reason)
    if reason is not None:
        raise ValueError("invalid specialist cannot have a critic capture")
    expected = gate_published_choice(
        bundle["source_request"], bundle["role_request"],
        specialist["raw_response"], reviewer["raw_response"],
    )
    if reviewer["provenance"]["transport_valid"] is not True:
        expected = dict(
            expected,
            status="invalid_transport",
            research_plan=None,
            errors=list(dict.fromkeys(
                expected["errors"] + ["unverified_declared_transport"])),
        )
    expected["critic_status"] = "captured"
    return expected


class PublishedContextResearchJob(ResearchJob):
    """Restartable source/controller bundle with detached role delivery."""

    def _read(self):
        stages = super()._read()
        if "request" not in stages:
            return stages
        try:
            bundle = stages["request"]["payload"]
            _validate_bundle(bundle)
        except ValueError as exc:
            raise ValueError("invalid published request artifact") from exc

        specialist = stages.get("specialist", {}).get("payload")
        reviewer = stages.get("reviewer", {}).get("payload")
        if specialist is not None and set(specialist) != _CAPTURE_KEYS:
            raise ValueError("invalid published specialist capture")
        if reviewer is not None:
            if _is_skip(reviewer):
                reason = _skip_reason(bundle, specialist)
                if (set(reviewer) != _SKIP_KEYS or reason is None
                        or _canonical(reviewer) != _canonical(_skip_event(reason))):
                    raise ValueError("invalid published reviewer skip event")
            else:
                if set(reviewer) != _CAPTURE_KEYS:
                    raise ValueError("invalid published reviewer capture")
                if _skip_reason(bundle, specialist) is not None:
                    raise ValueError("invalid specialist was routed to critic")

        if "grade" in stages:
            try:
                expected = _expected_grade(bundle, specialist, reviewer)
            except ValueError as exc:
                raise ValueError("invalid published grade prerequisites") from exc
            if _canonical(stages["grade"]["payload"]) != _canonical(expected):
                raise ValueError("invalid published grade artifact")
        if "reveal" in stages:
            expected_reveal = {
                "grade_sha256": stages["grade"]["sha256"],
                "authorized": True,
            }
            if _canonical(stages["reveal"]["payload"]) != _canonical(expected_reveal):
                raise ValueError("invalid published reveal artifact")
        return stages

    def prepare(self, source_request):
        self._read()
        return self._save("request", _bundle(source_request))

    def role_request(self, role):
        stages = self._read()
        if role not in ("specialist", "reviewer"):
            raise ValueError("unknown published role")
        if "request" not in stages:
            raise ValueError("published request must be prepared before role delivery")
        bundle = stages["request"]["payload"]
        if role == "specialist":
            return deepcopy(bundle["role_request"])
        if "specialist" not in stages:
            raise ValueError("specialist capture required before reviewer delivery")
        specialist = stages["specialist"]["payload"]
        if _skip_reason(bundle, specialist) is not None:
            raise ValueError("invalid specialist cannot be routed to critic")
        return build_published_review_request(
            bundle["source_request"], bundle["role_request"], specialist["raw_response"])

    def capture(self, role, raw_response, provenance):
        """Reject ineligible reviewer writes before entering the atomic store."""
        if role == "reviewer":
            stages = self._read()
            if "specialist" not in stages:
                raise ValueError("specialist capture required before reviewer capture")
            bundle = stages["request"]["payload"]
            specialist = stages["specialist"]["payload"]
            if _skip_reason(bundle, specialist) is not None:
                raise ValueError("invalid specialist cannot be routed to critic")
        return super().capture(role, raw_response, provenance)

    def skip_review(self):
        stages = self._read()
        if "specialist" not in stages:
            raise ValueError("specialist capture required before review skip")
        if "reviewer" in stages:
            reviewer = stages["reviewer"]["payload"]
            if _is_skip(reviewer):
                return deepcopy(reviewer)
            raise ValueError("captured review cannot be replaced by a skip")
        bundle = stages["request"]["payload"]
        reason = _skip_reason(bundle, stages["specialist"]["payload"])
        if reason is None:
            raise ValueError("valid specialist requires critic review")
        return self._save("reviewer", _skip_event(reason))

    def lock_grade(self, grade=None):
        stages = self._read()
        if "reviewer" not in stages:
            raise ValueError("review capture or explicit skip required")
        expected = _expected_grade(
            stages["request"]["payload"],
            stages["specialist"]["payload"],
            stages["reviewer"]["payload"],
        )
        if grade is not None and _canonical(grade) != _canonical(expected):
            raise ValueError("grade differs from pinned published captures/transport")
        return self._save("grade", deepcopy(expected))
