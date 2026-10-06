"""Breaks caught: invalid prices/clocks and caller mutation of frozen policy."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.research.study_contract import finite_number, protocol, stable_id, utc_minute


@pytest.mark.parametrize('value', [True, False, np.bool_(True), None, '1', math.nan, math.inf])
def test_prices_cannot_be_boolean_text_or_nonfinite(value):
    with pytest.raises(ValueError):
        finite_number(value, positive=True)


@pytest.mark.parametrize('value', [0, -1])
def test_positive_price_rejects_zero_and_negative(value):
    with pytest.raises(ValueError):
        finite_number(value, positive=True)


def test_numeric_and_zero_cost_values_remain_usable():
    assert finite_number(np.float64(2), positive=True) == 2.0
    assert finite_number(0) == 0.0


@pytest.mark.parametrize('value', ['2024-01-01', '2024-01-01T00:00:01Z', None, True, 0, 'NaT'])
def test_naive_unaligned_or_numeric_clocks_are_rejected(value):
    with pytest.raises(ValueError):
        utc_minute(value)


def test_offset_clock_is_normalized_without_moving_instant():
    assert utc_minute('2023-12-31T16:00:00-08:00') == pd.Timestamp('2024-01-01T00:00:00Z')


def test_returned_policy_cannot_mutate_later_runs():
    first = protocol()
    first['active_hypotheses'].append('R2')
    first['cost_bps'].append(0)
    assert protocol()['active_hypotheses'] == ['R1', 'R3']
    assert protocol()['cost_bps'] == [12, 24]


def test_semantic_identity_is_order_independent_and_rejects_nonfinite():
    assert stable_id('op', {'a': 1, 'b': 'x'}) == stable_id('op', {'b': 'x', 'a': 1})
    assert stable_id('op', {'a': 1}) != stable_id('op', {'a': 2})
    with pytest.raises(ValueError):
        stable_id('op', {'value': math.nan})
    with pytest.raises(ValueError):
        stable_id('', {'a': 1})
