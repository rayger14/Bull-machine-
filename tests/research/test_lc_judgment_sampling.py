import pytest

from scripts.research.lc_judgment_sampling import select_unassessed


def test_interior_prior_assessment_does_not_discard_later_eligible_cases():
    assert select_unassessed(['a', 'b', 'c', 'd', 'e'], {'c'}, 3) == ['a', 'b', 'd']


def test_returns_all_available_cases_without_padding_or_reordering():
    assert select_unassessed(['z', 'a', 'b'], {'a'}) == ['z', 'b']
    assert select_unassessed(['z'], {'z'}) == []


def test_duplicate_ids_and_excessive_budget_fail():
    with pytest.raises(ValueError):
        select_unassessed(['a', 'a'], set())
    with pytest.raises(ValueError):
        select_unassessed(['a'], set(), 31)
