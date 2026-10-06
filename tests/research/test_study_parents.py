from copy import deepcopy

import pandas as pd
import pytest

from scripts.research import causal_parent_ledger as frozen
from scripts.research.study_parents import MAX_STUDY_HOURS, build_study_parents
from tests.research.test_causal_parent_ledger import (ATR_CONTRACT, EXPECTED_HASHES, LOCAL_SOURCE_AVAILABLE,
                                                     SOURCE_PATHS, hourly_bars, local_sources)


def build(bars, paths, hashes):
    return build_study_parents(bars, instrument='BTC-USD', data_stream_id='continuous-fixture',
                              atr_contract=ATR_CONTRACT, source_paths=paths, expected_hashes=hashes)


def test_portable_continuous_input_exceeds_old_cap_without_mutating_it(tmp_path):
    paths, hashes = local_sources(tmp_path)
    bars = hourly_bars(3001)
    result = build(bars, paths, hashes)
    assert len(result['transitions']) == 3001
    assert frozen.MAX_INPUT_HOURS == 2048
    assert result['manifest']['study_adapter']['schema'] == 'continuous-study-parent-v1'
    assert result['manifest']['study_adapter']['restart_method'] == 'rebuild_same_full_prefix'
    assert result['coverage']['last_processed_close'] == str(bars.index[-1] + pd.Timedelta('1h'))


def test_portable_parity_partial_anchor_and_future_append(tmp_path):
    paths, hashes = local_sources(tmp_path)
    bars = hourly_bars(39, start='2026-01-01T01:00Z')
    actual = build(bars, paths, hashes)
    expected = frozen.build_parent_ledger(bars, instrument='BTC-USD', data_stream_id='continuous-fixture',
                                          anchor_timeframe='4H', pivot_n=3, atr_contract=ATR_CONTRACT,
                                          source_paths=paths, expected_hashes=hashes)
    normalized = deepcopy(actual)
    normalized['manifest'].pop('study_adapter')
    assert normalized == expected
    prefix = build(bars.iloc[:23], paths, hashes)
    assert prefix['transitions'] == actual['transitions'][:23]


@pytest.mark.skipif(not LOCAL_SOURCE_AVAILABLE, reason='explicit recovered local helpers unavailable')
def test_real_helpers_semantic_parity_and_multiple_lineages_across_old_cap():
    bars = hourly_bars(3001)
    extended = build(bars, SOURCE_PATHS, EXPECTED_HASHES)
    prefix = build(bars.iloc[:1800], SOURCE_PATHS, EXPECTED_HASHES)
    frozen_prefix = frozen.build_parent_ledger(bars.iloc[:1800], instrument='BTC-USD', data_stream_id='continuous-fixture',
                                                anchor_timeframe='4H', pivot_n=3, atr_contract=ATR_CONTRACT,
                                                source_paths=SOURCE_PATHS, expected_hashes=EXPECTED_HASHES)
    normalized = deepcopy(prefix)
    normalized['manifest'].pop('study_adapter')
    assert normalized == frozen_prefix
    cutoff = bars.index[1799] + pd.Timedelta('1h')
    for name in ('pivots', 'versions', 'transitions'):
        assert prefix[name] == [r for r in extended[name] if pd.Timestamp(r['available_at']) <= cutoff]
    assert len({v['lineage_id'] for v in extended['versions']}) > 1
    assert extended == build(bars, SOURCE_PATHS, EXPECTED_HASHES)


def test_extended_cap_gap_and_changed_helper_fail_closed(tmp_path):
    paths, hashes = local_sources(tmp_path)
    with pytest.raises(ValueError, match='study cap'):
        build(hourly_bars(MAX_STUDY_HOURS + 1), paths, hashes)
    with pytest.raises(ValueError):
        build(hourly_bars(30).drop(hourly_bars(30).index[4]), paths, hashes)
    hashes['htf'] = '0' * 64
    with pytest.raises(ValueError, match='hash'):
        build(hourly_bars(30), paths, hashes)
