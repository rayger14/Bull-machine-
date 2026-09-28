"""Synthetic offline inputs only; no market archive or agent calls."""
from scripts.research.conditional_assessment import digest
from scripts.research.lc_context_assessment import build_context_request
from tests.research.test_lc_context_assessment import empty_snapshot, context_brief, SETTINGS
from tests.research.test_lc_master_assessment import packet


def structure_source(mutator=None):
    value = packet()
    if mutator:
        mutator(value)
    value.pop('seal', None)
    value['seal'] = digest(value)
    return build_context_request(value, empty_snapshot(), context_brief(), SETTINGS)
