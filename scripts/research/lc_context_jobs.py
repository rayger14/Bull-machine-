"""Immutable research jobs pinned to the LC context discrimination namespace."""
from copy import deepcopy

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_context_assessment import (
    gate_context_choice, validate_context_request,
)
from scripts.research.master_research_jobs import ResearchJob


class ContextResearchJob(ResearchJob):
    """Reuse storage transactions while recomputing only the context policy."""

    def _read(self):
        stages = super()._read()
        if 'request' in stages:
            try:
                validate_context_request(stages['request']['payload'])
            except ValueError as exc:
                raise ValueError('invalid context request artifact') from exc
        return stages

    def prepare(self, request):
        self._read()
        validate_context_request(request)
        return self._save('request', request)

    def lock_grade(self, grade=None):
        """Recompute the context grade from exact immutable raw captures."""
        stages = self._read()
        if 'reviewer' not in stages:
            raise ValueError('both captures required')
        request = stages['request']['payload']
        specialist = stages['specialist']['payload']
        reviewer = stages['reviewer']['payload']
        expected = gate_context_choice(
            request, specialist['raw_response'], reviewer['raw_response'])
        captures = (specialist, reviewer)
        if not all(capture['provenance']['transport_valid'] is True for capture in captures):
            expected = dict(expected, status='invalid_transport', research_plan=None,
                            errors=list(dict.fromkeys(expected['errors']
                                  + ['unverified_declared_transport'])))
        if grade is not None and _canonical(grade) != _canonical(expected):
            raise ValueError('grade differs from pinned captures/transport')
        return self._save('grade', deepcopy(expected))
