"""Wyckoff Accumulation Schematic #2 (M2) sequencer path.

M2 = accumulation WITHOUT a spring (Bogomazov/Pruden Schematic #2, and the
pattern Wyckoff_Insider trades most): PS -> SC -> AR -> ST -> [ST in Phase B,
undercut allowed] -> LPS in Phase C (higher low ABOVE support) -> SOS ->
BU/LPS -> markup. The pre-M2 sequencer only accepted an LPS *after* an SOS,
making the M2 signature moment unrepresentable.
"""
import pytest

from engine.wyckoff.events import WyckoffStateMachine, WyckoffState, WyckoffContext


def _row(low, close, high=None, volume_z=0.0):
    high = high if high is not None else close + 1
    return {'open': close, 'high': high, 'low': low, 'close': close,
            'volume_z': volume_z}


def _drive_phase_ab(sm):
    """SC -> AR -> ST: establish the accumulation range (sc_low=100, ar_high=110)."""
    v, _ = sm.process_bar(0, _row(low=100, close=102, volume_z=3.0), {'sc': True})
    assert v['sc'] and sm.state == WyckoffState.ACCUM_SC
    v, _ = sm.process_bar(3, _row(low=105, close=109, high=110), {'ar': True})
    assert v['ar'] and sm.state == WyckoffState.ACCUM_AR
    v, _ = sm.process_bar(6, _row(low=101, close=103, volume_z=0.5), {'st': True})
    assert v['st'] and sm.state == WyckoffState.ACCUM_ST


def test_m2_full_sequence_lps_before_sos():
    """SC->AR->ST -> quiet undercut (ST-B) -> phase-C LPS -> SOS -> BU/LPS."""
    sm = WyckoffStateMachine({})
    _drive_phase_ab(sm)

    # ST in Phase B: quiet undercut of SC low must NOT invalidate the structure
    sm.process_bar(10, _row(low=99.0, close=101.5, volume_z=-0.5), {})
    assert sm.context == WyckoffContext.ACCUMULATION, \
        "quiet undercut (low < SC low, weak volume) must not reset the structure"

    # Phase C LPS: higher low holding ABOVE support, BEFORE any SOS
    v, mods = sm.process_bar(14, _row(low=103.0, close=105.0, volume_z=-0.3),
                             {'lps': True})
    assert v['lps'], "M2 path: LPS before SOS must validate (higher low above support)"
    assert sm.state == WyckoffState.ACCUM_LPS_C
    assert sm.get_phase() == 'C'
    assert sm.get_phase_dir() == 'C_accum'

    # SOS reachable FROM the phase-C LPS
    v, _ = sm.process_bar(18, _row(low=108, close=112, high=113, volume_z=2.0),
                          {'sos': True})
    assert v['sos'], "SOS must be reachable from the phase-C LPS state"
    assert sm.state == WyckoffState.ACCUM_SOS

    # BU/LPS: post-SOS LPS still works (legacy path)
    v, _ = sm.process_bar(20, _row(low=109, close=111, volume_z=-0.2), {'lps': True})
    assert v['lps'] and sm.state == WyckoffState.ACCUM_LPS


def test_m2_lps_below_support_rejected():
    """An 'LPS' whose low breaks the SC low is NOT a phase-C LPS (that would be
    spring territory) — the M2 path must reject it."""
    sm = WyckoffStateMachine({})
    _drive_phase_ab(sm)
    v, _ = sm.process_bar(14, _row(low=99.5, close=104.0), {'lps': True})
    assert not v['lps'], "low below SC support is not an M2 phase-C LPS"


def test_m2_path_config_gated_off():
    """sm_m2_path=False reproduces legacy behavior (LPS only after SOS)."""
    sm = WyckoffStateMachine({'sm_m2_path': False})
    _drive_phase_ab(sm)
    v, _ = sm.process_bar(14, _row(low=103.0, close=105.0), {'lps': True})
    assert not v['lps'], "legacy mode: pre-SOS LPS must stay rejected"
    assert sm.state == WyckoffState.ACCUM_ST


def test_m2_distribution_mirror_lpsy_before_sow():
    """Mirror: BC->AS->UT context, LPSY (lower high BELOW resistance) before SOW.
    Phase-B work (a UT) must precede the LPSY — never straight from AS/AR."""
    sm = WyckoffStateMachine({})
    v, _ = sm.process_bar(0, _row(low=98, close=99.5, high=110, volume_z=3.0),
                          {'bc': True})
    assert v['bc'] and sm.state == WyckoffState.DISTRIB_BC
    v, _ = sm.process_bar(3, _row(low=100, close=101, high=104), {'as': True})
    assert v['as'] and sm.state == WyckoffState.DISTRIB_AR

    # Straight from AR, an LPSY must be REJECTED (no phase-B work yet)
    v, _ = sm.process_bar(5, _row(low=102, close=104, high=107), {'lpsy': True})
    assert not v['lpsy'], "LPSY straight from AR must be rejected (needs UT/ST first)"

    # Phase-B work: UT (fake poke near resistance)
    v, _ = sm.process_bar(6, _row(low=105, close=107, high=109.5, volume_z=0.8),
                          {'ut': True})
    assert v['ut'] and sm.state == WyckoffState.DISTRIB_UT

    # Phase C LPSY: lower high holding BELOW resistance, before any SOW
    v, _ = sm.process_bar(8, _row(low=102, close=104, high=107, volume_z=-0.3),
                          {'lpsy': True})
    assert v['lpsy'], "M2 mirror: LPSY before SOW must validate"
    assert sm.state == WyckoffState.DISTRIB_LPSY_C
    assert sm.get_phase() == 'C'
    assert sm.get_phase_dir() == 'C_distrib'

    v, _ = sm.process_bar(12, _row(low=95, close=96, high=99, volume_z=2.0),
                          {'sow': True})
    assert v['sow'], "SOW must be reachable from the phase-C LPSY state"
