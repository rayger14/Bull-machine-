#!/usr/bin/env python3
"""
Wyckoff Event Detection System - Institutional Grade Implementation

Detects all 18 classic Wyckoff structural events for accumulation and distribution
cycles. Integrates with existing PTI, volume_z, and liquidity_score features.

Events Implemented:
    Phase A (Selling Climax/Buying Climax):
        - SC (Selling Climax): Extreme volume spike at lows, capitulation
        - BC (Buying Climax): Extreme volume spike at highs, euphoria peak
        - AR (Automatic Rally): Relief bounce after SC on declining volume
        - AS (Automatic Reaction): Relief drop after BC on declining volume
        - ST (Secondary Test): Retest of SC lows on lower volume
        - ST_BC (Secondary Test of BC): Retest of BC highs on lower volume

    Phase B (Building Cause):
        - SOS (Sign of Strength): First decisive move up with volume
        - SOW (Sign of Weakness): First decisive move down with volume
        - LPS (Last Point of Support): Final test before markup begins
        - LPSY (Last Point of Supply): Final rally before markdown begins
        - Spring_A (Type A Spring): Fake breakdown below range to trap sellers
        - Spring_B (Type B Spring): Shallow spring with quick recovery

    Phase C (Testing):
        - UT (Upthrust): Fake breakout above range to trap buyers
        - UTAD (Upthrust After Distribution): Final trap before major decline
        - Shakeout: Violent SC-like action mid-range to test strength

    Phase D/E (Trend):
        - Markup_Continuation: Validated uptrend with structure
        - Markdown_Continuation: Validated downtrend with structure
        - Terminal_Shakeout: Final trap at trend exhaustion

Design Philosophy:
    1. Backward Compatible: All new features default to False/0.0
    2. Vectorized: Pure pandas/numpy, no loops
    3. Confluence-Based: Uses volume_z, liquidity_score, range_position
    4. Observable: Returns confidence scores for each detection
    5. PTI-Integrated: Links psychological traps to Wyckoff events

Author: Bull Machine v2.0 - Institutional Grade Wyckoff System
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
from engine.wyckoff.range_evidence import RangeEvidence

logger = logging.getLogger(__name__)


class WyckoffPhaseABC(Enum):
    """Wyckoff sub-phases (finer granularity than existing accumulation/distribution)"""
    PHASE_A = "A"  # Preliminary support/resistance
    PHASE_B = "B"  # Building cause/effect
    PHASE_C = "C"  # Testing phase (springs/upthrusts)
    PHASE_D = "D"  # Trend beginning
    PHASE_E = "E"  # Trend continuation
    NEUTRAL = "neutral"


class WyckoffContext(Enum):
    """High-level market context for state machine"""
    NONE = "none"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"


class WyckoffState(Enum):
    """Detailed sub-state within Wyckoff context"""
    NONE = "none"
    # Accumulation states
    ACCUM_SC = "accum_sc"
    ACCUM_AR = "accum_ar"
    ACCUM_ST = "accum_st"
    ACCUM_SPRING = "accum_spring"
    ACCUM_LPS_C = "accum_lps_c"   # M2 (Schematic #2): phase-C LPS BEFORE SOS
    ACCUM_SOS = "accum_sos"
    ACCUM_LPS = "accum_lps"
    # Distribution states
    DISTRIB_BC = "distrib_bc"
    DISTRIB_AR = "distrib_ar"
    DISTRIB_ST = "distrib_st"
    DISTRIB_UT = "distrib_ut"
    DISTRIB_LPSY_C = "distrib_lpsy_c"  # M2 mirror: phase-C LPSY BEFORE SOW
    DISTRIB_SOW = "distrib_sow"
    DISTRIB_LPSY = "distrib_lpsy"


@dataclass
class RangeReference:
    """Tracks key price/volume levels for an active Wyckoff structure"""
    context: WyckoffContext = WyckoffContext.NONE
    # Accumulation range
    sc_low: float = 0.0
    sc_volume: float = 0.0
    sc_bar_idx: int = -1
    ar_high: float = 0.0
    ar_bar_idx: int = -1
    # Distribution range
    bc_high: float = 0.0
    bc_volume: float = 0.0
    bc_bar_idx: int = -1
    as_low: float = 0.0
    as_bar_idx: int = -1
    # Metadata
    bars_since_start: int = 0
    invalidated: bool = False
    st_count: int = 0  # Track number of STs validated (max 2 per structure)


@dataclass(frozen=True)
class DelayedEventEvidence:
    """Candidate identity observed causally, available only at confirmation close."""
    event_type: str
    candidate_index: int
    confirmation_index: int
    candidate_extreme: float
    prior_swept_boundary: float
    candidate_parent_id: int
    candidate_parent_context: str
    candidate_parent_status: str
    candidate_timestamp: pd.Timestamp
    confirmation_timestamp: pd.Timestamp
    available_at: pd.Timestamp


@dataclass(frozen=True)
class OpposingClimaxCandidate:
    """An opposing clue, not authority to erase an established parent."""
    event_type: str
    candidate_index: int
    candidate_parent_id: int
    candidate_parent_context: str
    open: float
    high: float
    low: float
    close: float
    volume_z: float
    confidence: float
    candidate_timestamp: Optional[pd.Timestamp]
    candidate_available_at: Optional[pd.Timestamp]
    clock_required: bool
    volume: Optional[float] = None


def _delayed_event_parameters(event_type: str, cfg: dict) -> tuple:
    family = 'ut' if event_type == 'utad' else event_type
    delay = cfg.get(f'{family}_recovery_bars', 3)
    lookback = cfg.get(f'{family}_lookback', 20)
    if (not isinstance(delay, (int, np.integer)) or isinstance(delay, bool) or delay < 0
            or not isinstance(lookback, (int, np.integer)) or isinstance(lookback, bool)
            or lookback < 1):
        raise ValueError('Delayed Wyckoff offsets must be nonnegative integer bars; lookback must be positive')
    # Preserve Spring B's existing same-bar convention.
    if family == 'spring_b' and delay <= 1:
        delay = 0
    return delay, lookback, 'low' if family.startswith('spring') else 'high'


class WyckoffStateMachine:
    """
    Sequential validator for Wyckoff events.

    Processes bars one at a time. Uses raw detector outputs as candidates,
    then validates them against sequential rules and relative volume comparisons.

    Key rules:
    - AR requires prior SC within sm_ar_max_bars (default 15)
    - ST requires SC+AR, and ST_volume < SC_volume * sm_st_volume_ratio (default 0.7)
    - Spring requires established range (SC_low and AR_high defined)
    - SOS requires prior accumulation context (at least SC+AR)
    - LPS requires prior SOS
    - Symmetric rules for distribution
    - Invalidation: close below SC_low on volume >= SC_volume * ratio -> reset
    - Max structure age: sm_max_structure_bars (default 1000)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.state = WyckoffState.NONE
        self.context = WyckoffContext.NONE
        self.range_ref = RangeReference()
        self.bars_in_structure = 0
        self._parent_generation = 0  # Monotonic even through no-context intervals.
        self.pending_opposing_climax = None
        self.climax_evidence = []  # Current-bar lifecycle records; never scored.
        self.range_evidence = RangeEvidence(cfg)
        self.structural_evidence = self.range_evidence.snapshot()
        self.anchored_confidence = {}
        # sm_m2_context_only shadow: phase-C reading without state/event change
        self.m2_shadow_c = False

        # Config with sm_ prefix
        self.max_structure_bars = cfg.get('sm_max_structure_bars', 1000)
        self.ar_max_bars = cfg.get('sm_ar_max_bars', 15)
        self.st_volume_ratio = cfg.get('sm_st_volume_ratio', 0.7)
        self.invalidation_vol_ratio = cfg.get('sm_invalidation_vol_ratio', 0.8)
        self.spring_max_break_pct = cfg.get('sm_spring_max_break_pct', 0.05)

    def reset(self):
        """Reset state machine to NONE"""
        self._parent_generation += 1
        self.state = WyckoffState.NONE
        self.context = WyckoffContext.NONE
        self.range_ref = RangeReference()
        self.bars_in_structure = 0
        self.m2_shadow_c = False
        self.pending_opposing_climax = None
        self.climax_evidence = []
        self.range_evidence = RangeEvidence(self.cfg, self._parent_generation)
        self.structural_evidence = self.range_evidence.snapshot()

    def parent_snapshot(self) -> dict:
        """Immutable values describing the parent before a candidate is processed."""
        established = (
            self.context == WyckoffContext.ACCUMULATION and self.range_ref.ar_high > 0
        ) or (
            self.context == WyckoffContext.DISTRIBUTION and self.range_ref.as_low > 0
        )
        status = 'none' if self.context == WyckoffContext.NONE else (
            'established' if established else 'unestablished')
        return dict(id=self._parent_generation, context=self.context.value, status=status)

    def _climax_candidate(self, key, bar_idx, row, parent):
        return OpposingClimaxCandidate(
            key, bar_idx, parent['id'], parent['context'],
            float(row['open']), float(row['high']), float(row['low']),
            float(row['close']), float(row.get('volume_z', 0)),
            float(row.get(f'{key}_confidence', 0)),
            row.get('timestamp'), row.get('available_at'),
            bool(row.get('clock_required', False) or 'timestamp' in row or 'available_at' in row),
            float(row['volume']) if row.get('volume') is not None and np.isfinite(row['volume']) else None)

    def _record_climax(self, status, bar_idx, row, candidate=None, **extra):
        record = dict(vars(candidate)) if candidate is not None else {}
        record.update(status=status, observed_index=bar_idx,
                      observed_timestamp=row.get('timestamp'),
                      available_at=row.get('available_at'), **extra)
        self.climax_evidence.append({
            k: v.isoformat() if isinstance(v, pd.Timestamp) else v
            for k, v in record.items()})

    def _discard_opposing_climax(self, status, bar_idx, row):
        if self.pending_opposing_climax is not None:
            self._record_climax(status, bar_idx, row, self.pending_opposing_climax)
            self.pending_opposing_climax = None

    @staticmethod
    def _climax_time_contiguous(candidate, age, row):
        start, available = candidate.candidate_timestamp, candidate.candidate_available_at
        if not candidate.clock_required and start is None and available is None:
            return True  # Legacy pretagged bar-index callers have no clock evidence.
        times = (start, available, row.get('timestamp'), row.get('available_at'))
        if any(not isinstance(t, pd.Timestamp) or pd.isna(t) or t.tz is None for t in times):
            return False
        interval = available - start
        return (interval > pd.Timedelta(0)
                and times[2] == start + age * interval
                and times[3] == times[2] + interval)

    def _arbitrate_climaxes(self, bar_idx, row, raw, entry_parent, validated):
        """Mutate only the local raw copy. True means atomic replacement committed.

        This conservative operational contract requires a later AR/AS AND a close
        across the stored candle's reaction-side extreme. It does not claim to
        recognize every gradual reversal or a complete Wyckoff schematic.
        """
        if raw.get('sc', False) and raw.get('bc', False):
            self._discard_opposing_climax('cancelled_ambiguity', bar_idx, row)
            self._record_climax('ambiguous', bar_idx, row,
                                parent_id=entry_parent['id'], event_type='sc+bc')
            raw['sc'] = raw['bc'] = False
            return False

        opposing = 'bc' if entry_parent['context'] == 'accumulation' else 'sc'
        if entry_parent['status'] != 'established':
            return False  # Preserve provisional/no-parent initialization.
        if entry_parent['id'] != self._parent_generation:
            # Invalidation/age has priority. Do not evade it through NONE fallback.
            if raw.get(opposing, False):
                self._record_climax('rejected_parent_loss', bar_idx, row,
                                    self._climax_candidate(opposing, bar_idx, row, entry_parent))
                raw[opposing] = False
            return False

        same_side = 'sc' if opposing == 'bc' else 'bc'
        if raw.get(same_side, False):
            self._discard_opposing_climax('cancelled_parent_replacement', bar_idx, row)
            return False  # Existing same-side reset remains authoritative.

        candidate = self.pending_opposing_climax
        if candidate is not None:
            age = bar_idx - candidate.candidate_index
            if candidate.candidate_parent_id != self._parent_generation:
                self._discard_opposing_climax('cancelled_parent_loss', bar_idx, row)
            elif age > self.ar_max_bars:
                self._discard_opposing_climax('expired', bar_idx, row)
            elif not self._climax_time_contiguous(candidate, age, row):
                self._discard_opposing_climax('cancelled_time_discontinuity', bar_idx, row)
            elif age > 0:
                continuation = (row['close'] > candidate.high if opposing == 'bc'
                                else row['close'] < candidate.low)
                reaction = 'as' if opposing == 'bc' else 'ar'
                crossed = (row['close'] < candidate.low if opposing == 'bc'
                           else row['close'] > candidate.high)
                if continuation:
                    self._discard_opposing_climax('cancelled_continuation', bar_idx, row)
                elif raw.get(reaction, False) and crossed:
                    self._parent_generation += 1
                    if opposing == 'bc':
                        self.context = WyckoffContext.DISTRIBUTION
                        self.state = WyckoffState.DISTRIB_AR
                        self.range_ref = RangeReference(
                            context=self.context, bc_high=candidate.high,
                            bc_volume=candidate.volume_z, bc_bar_idx=candidate.candidate_index,
                            as_low=row['low'], as_bar_idx=bar_idx)
                    else:
                        self.context = WyckoffContext.ACCUMULATION
                        self.state = WyckoffState.ACCUM_AR
                        self.range_ref = RangeReference(
                            context=self.context, sc_low=candidate.low,
                            sc_volume=candidate.volume_z, sc_bar_idx=candidate.candidate_index,
                            ar_high=row['high'], ar_bar_idx=bar_idx)
                    self.bars_in_structure = age
                    self.m2_shadow_c = False
                    self.pending_opposing_climax = None
                    validated[opposing] = validated[reaction] = True
                    self._record_climax('confirmed', bar_idx, row, candidate,
                                        replacement_parent_id=self._parent_generation)
                    return True  # No same-bar follow-on transitions in a new parent.

        if raw.get(opposing, False):
            raw[opposing] = False
            if self.pending_opposing_climax is None:
                self.pending_opposing_climax = self._climax_candidate(
                    opposing, bar_idx, row, entry_parent)
        if self.pending_opposing_climax is not None:
            self._record_climax('pending', bar_idx, row, self.pending_opposing_climax)
        return False

    def _candidate_extreme(self, key, bar_idx, row, event_metadata):
        """None rejects evidence; omission alone permits legacy pretagged callers."""
        if event_metadata is None:
            return row['low' if key.startswith('spring') else 'high']
        if not isinstance(event_metadata, dict):
            return None
        item = event_metadata.get(key)
        if not isinstance(item, DelayedEventEvidence) or item.event_type != key:
            return None
        try:
            delay, _, _ = _delayed_event_parameters(key, self.cfg)
            if (not isinstance(item.candidate_index, (int, np.integer))
                    or isinstance(item.candidate_index, bool)
                    or item.candidate_index < 0
                    or item.confirmation_index != bar_idx
                    or bar_idx - item.candidate_index != delay):
                return None
            extreme, boundary = float(item.candidate_extreme), float(item.prior_swept_boundary)
            if not np.isfinite([extreme, boundary]).all() or min(extreme, boundary) <= 0:
                return None
            if (key.startswith('spring') and extreme >= boundary) or (
                    not key.startswith('spring') and extreme <= boundary):
                return None
            times = (item.candidate_timestamp, item.confirmation_timestamp, item.available_at)
            if any(not isinstance(t, pd.Timestamp) or pd.isna(t) or t.tz is None for t in times):
                return None
            interval = item.available_at - item.confirmation_timestamp
            if interval <= pd.Timedelta(0) or (
                    item.confirmation_timestamp - item.candidate_timestamp != delay * interval):
                return None
            if 'timeframe' in self.cfg and interval != pd.Timedelta(str(self.cfg['timeframe']).lower()):
                return None
        except (ValueError, TypeError, OverflowError):
            return None
        now = self.parent_snapshot()
        if (item.candidate_parent_id != now['id']
                or item.candidate_parent_context != now['context']
                or item.candidate_parent_status != now['status']):
            return None
        desired = 'accumulation' if key.startswith('spring') else 'distribution'
        if now['status'] == 'established' and now['context'] == desired:
            return extreme
        if now['status'] == 'none' and now['context'] == 'none':
            return extreme
        return None

    def process_bar(self, bar_idx: int, row: dict, raw_events: dict,
                    event_metadata: Optional[Dict[str, DelayedEventEvidence]] = None) -> tuple:
        """
        Process one bar. Returns (validated_bools, confidence_modifiers).

        Args:
            bar_idx: Integer index of this bar
            row: Dict with open, high, low, close, volume_z
            raw_events: Dict of raw detector bools
            event_metadata: Candidate provenance. Only deliberate omission uses
                legacy current-row geometry; the batch adapter always supplies it.

        Returns:
            Tuple of (validated dict, modifiers dict)
        """
        entry_parent = self.parent_snapshot()
        self.anchored_confidence = {}
        pending_at_entry = self.pending_opposing_climax
        self.climax_evidence = []
        raw_events = dict(raw_events)  # Keep detector evidence immutable to callers.
        self.bars_in_structure += 1
        validated = {k: False for k in raw_events}
        modifiers = {}

        # Check structure age - reset if too old
        if self.bars_in_structure > self.max_structure_bars:
            self.reset()
            if pending_at_entry is not None:
                self._record_climax('cancelled_parent_expiry', bar_idx, row, pending_at_entry)

        # Check invalidation
        self._check_invalidation(row)
        if self.range_ref.invalidated:
            self.reset()
            if pending_at_entry is not None:
                self._record_climax('cancelled_parent_invalidation', bar_idx, row, pending_at_entry)

        if self._arbitrate_climaxes(bar_idx, row, raw_events, entry_parent, validated):
            self._update_range_evidence(bar_idx, row, validated, raw_events, event_metadata)
            return validated, modifiers

        # --- ACCUMULATION PATH ---

        # SC: provisional/same-side initializer after opposing-climax arbitration.
        if raw_events.get('sc', False):
            self._parent_generation += 1
            validated['sc'] = True
            self.context = WyckoffContext.ACCUMULATION
            self.state = WyckoffState.ACCUM_SC
            self.range_ref = RangeReference(
                context=WyckoffContext.ACCUMULATION,
                sc_low=row['low'],
                sc_volume=row.get('volume_z', 0),
                sc_bar_idx=bar_idx,
            )
            self.bars_in_structure = 0
            self.m2_shadow_c = False  # new structure: clear stale shadow phase

        # AR: Requires prior SC within lookback
        if raw_events.get('ar', False) and self.context == WyckoffContext.ACCUMULATION:
            bars_since_sc = bar_idx - self.range_ref.sc_bar_idx
            if 0 < bars_since_sc <= self.ar_max_bars and self.state == WyckoffState.ACCUM_SC:
                validated['ar'] = True
                self.state = WyckoffState.ACCUM_AR
                self.range_ref.ar_high = row['high']
                self.range_ref.ar_bar_idx = bar_idx

        # ST: Requires SC+AR, volume < SC_volume * ratio, max 2 per structure
        st_max_count = self.cfg.get('sm_st_max_count', 2)
        if raw_events.get('st', False) and self.context == WyckoffContext.ACCUMULATION:
            if self.state in (WyckoffState.ACCUM_AR, WyckoffState.ACCUM_ST):
                if self.range_ref.st_count < st_max_count:
                    bar_vol = row.get('volume_z', 0)
                    if bar_vol < self.range_ref.sc_volume * self.st_volume_ratio:
                        validated['st'] = True
                        self.state = WyckoffState.ACCUM_ST
                        self.range_ref.st_count += 1

        # Spring: Requires established range (SC+AR), price near or below SC_low
        spring_tolerance = self.cfg.get('sm_spring_tolerance', 0.01)  # 1% tolerance for shallow springs
        for spring_key in ('spring_a', 'spring_b'):
            if not raw_events.get(spring_key, False):
                continue
            candidate_low = self._candidate_extreme(spring_key, bar_idx, row, event_metadata)
            # Only an authorized BC initializer remains in the local raw copy.
            # A pending BC must not veto valid evidence in the surviving parent.
            if candidate_low is None or (event_metadata is not None
                    and raw_events.get('bc', False) and not raw_events.get('sc', False)):
                continue
            if self.context == WyckoffContext.ACCUMULATION and self.range_ref.ar_high > 0:
                # Spring A: must break below SC_low (classic)
                # Spring B: within tolerance of SC_low (shallow spring)
                sc_low_with_tolerance = self.range_ref.sc_low * (1 + spring_tolerance)
                if candidate_low < sc_low_with_tolerance:
                    break_pct = (self.range_ref.sc_low - candidate_low) / (self.range_ref.sc_low + 1e-9)
                    if break_pct < self.spring_max_break_pct:
                        if spring_key == 'spring_b' or candidate_low < self.range_ref.sc_low:
                            validated[spring_key] = True
            elif (self.context == WyckoffContext.NONE
                  and self.cfg.get('sm_no_context_fallback', True)):
                # No validated SC->AR structure exists — SC's triple-extreme gate
                # makes that the norm, which starved springs to near-zero. Mirror
                # the SOS/SOW no-context fallback: keep the raw detector event at
                # half confidence instead of discarding it. State is NOT advanced
                # (no structure to advance).
                validated[spring_key] = True
                modifiers[spring_key] = 0.5

        if self.context == WyckoffContext.ACCUMULATION and (
                validated.get('spring_a', False) or validated.get('spring_b', False)):
            self.state = WyckoffState.ACCUM_SPRING

        # SOS: Requires accumulation context, or fires with reduced confidence when no context
        if raw_events.get('sos', False):
            if self.context == WyckoffContext.ACCUMULATION:
                if self.state in (WyckoffState.ACCUM_AR, WyckoffState.ACCUM_ST,
                                 WyckoffState.ACCUM_SPRING, WyckoffState.ACCUM_LPS_C):
                    validated['sos'] = True
                    self.state = WyckoffState.ACCUM_SOS
            elif self.context == WyckoffContext.NONE:
                validated['sos'] = True
                modifiers['sos'] = 0.5  # 50% confidence penalty for no-context SOS

        # LPS: after SOS = BU/LPS (Schematic #1 & #2 phase D). M2 path
        # (Schematic #2, default ON): a phase-C LPS BEFORE any SOS is valid
        # when the range is established and the bar's low HOLDS ABOVE support
        # (higher low = the M2 signature; a low below SC_low is spring
        # territory, not an M2 LPS).
        if (raw_events.get('lps', False) and self.context == WyckoffContext.ACCUMULATION
                and not self.range_evidence.blocks_local_retest(self._parent_generation, row, bar_idx)):
            if self.state == WyckoffState.ACCUM_SOS:
                validated['lps'] = True
                self.state = WyckoffState.ACCUM_LPS
            elif (self.range_ref.ar_high > 0
                  and self.state in (WyckoffState.ACCUM_ST,
                                     WyckoffState.ACCUM_SPRING)
                  and row['low'] > self.range_ref.sc_low):
                # Schematic #2 structure constraints (NOT tuned numbers):
                # phase-B work must precede the LPS (an ST or spring has
                # occurred — never straight from AR), and ONE phase-C LPS per
                # structure (no LPS_C self-loop; without this, every quiet dip
                # in a range re-fires: 16 -> 2,727 events, a 170x explosion).
                if self.cfg.get('sm_m2_path', False):
                    validated['lps'] = True
                    self.state = WyckoffState.ACCUM_LPS_C
                elif self.cfg.get('sm_m2_context_only', False):
                    # SHADOW phase-C: only wyckoff_phase_dir/_phase_abc
                    # readings change — no event emission, no state change, so
                    # event columns / scores / fusion inputs stay bit-identical
                    # to the M2-off build. (Fusion reshuffle was the full
                    # path's failure mode: train +13K / hold -5.2K. The phase
                    # READING co-moved as an entry qualifier in both eras.)
                    self.m2_shadow_c = True

        # --- DISTRIBUTION PATH ---

        # BC: provisional/same-side initializer after opposing-climax arbitration.
        if raw_events.get('bc', False):
            if not validated.get('sc', False):
                self._parent_generation += 1
                validated['bc'] = True
                self.context = WyckoffContext.DISTRIBUTION
                self.state = WyckoffState.DISTRIB_BC
                self.range_ref = RangeReference(
                    context=WyckoffContext.DISTRIBUTION,
                    bc_high=row['high'],
                    bc_volume=row.get('volume_z', 0),
                    bc_bar_idx=bar_idx,
                )
                self.bars_in_structure = 0
                self.m2_shadow_c = False  # new structure: clear stale shadow

        # AS: Requires prior BC within lookback
        if raw_events.get('as', False) and self.context == WyckoffContext.DISTRIBUTION:
            bars_since_bc = bar_idx - self.range_ref.bc_bar_idx
            if 0 < bars_since_bc <= self.ar_max_bars and self.state == WyckoffState.DISTRIB_BC:
                validated['as'] = True
                self.state = WyckoffState.DISTRIB_AR
                self.range_ref.as_low = row['low']
                self.range_ref.as_bar_idx = bar_idx

        # UT/UTAD: Requires distribution context, within tolerance of bc_high
        ut_tolerance = self.cfg.get('sm_ut_tolerance', 0.01)  # 1% — "approaches or slightly exceeds"
        for ut_key in ('ut', 'utad'):
            if not raw_events.get(ut_key, False):
                continue
            candidate_high = self._candidate_extreme(ut_key, bar_idx, row, event_metadata)
            if candidate_high is None:
                continue
            if self.context == WyckoffContext.DISTRIBUTION and self.range_ref.as_low > 0:
                bc_high_threshold = self.range_ref.bc_high * (1 - ut_tolerance)
                if candidate_high > bc_high_threshold:
                    # Allow UT from more states (not just after AS)
                    if self.state in (WyckoffState.DISTRIB_AR, WyckoffState.DISTRIB_ST,
                                     WyckoffState.DISTRIB_SOW, WyckoffState.DISTRIB_UT):
                        validated[ut_key] = True
                        self.state = WyckoffState.DISTRIB_UT
            elif (self.context == WyckoffContext.NONE
                  and self.cfg.get('sm_no_context_fallback', True)):
                # Mirror of the spring no-context fallback (see accumulation path).
                validated[ut_key] = True
                modifiers[ut_key] = 0.5

        # SOW: Requires distribution context, or fires with reduced confidence when no context
        if raw_events.get('sow', False):
            if self.context == WyckoffContext.DISTRIBUTION:
                if self.state in (WyckoffState.DISTRIB_AR, WyckoffState.DISTRIB_ST,
                                 WyckoffState.DISTRIB_UT,
                                 WyckoffState.DISTRIB_LPSY_C):
                    validated['sow'] = True
                    self.state = WyckoffState.DISTRIB_SOW
            elif self.context == WyckoffContext.NONE:
                validated['sow'] = True
                modifiers['sow'] = 0.5

        # LPSY: after SOW (legacy) or M2 mirror — phase-C LPSY BEFORE any SOW,
        # lower high HOLDING BELOW resistance.
        if (raw_events.get('lpsy', False) and self.context == WyckoffContext.DISTRIBUTION
                and not self.range_evidence.blocks_local_retest(self._parent_generation, row, bar_idx)):
            if self.state == WyckoffState.DISTRIB_SOW:
                validated['lpsy'] = True
                self.state = WyckoffState.DISTRIB_LPSY
            elif (self.range_ref.as_low > 0
                  and self.state in (WyckoffState.DISTRIB_ST,
                                     WyckoffState.DISTRIB_UT)
                  and row['high'] < self.range_ref.bc_high):
                # Mirror of the accumulation M2 constraints: phase-B work (ST
                # or UT) must precede; one phase-C LPSY per structure.
                if self.cfg.get('sm_m2_path', False):
                    validated['lpsy'] = True
                    self.state = WyckoffState.DISTRIB_LPSY_C
                elif self.cfg.get('sm_m2_context_only', False):
                    self.m2_shadow_c = True  # shadow phase-C (see accum path)

        self._update_range_evidence(bar_idx, row, validated, raw_events, event_metadata)
        return validated, modifiers

    def _update_range_evidence(self, bar_idx, row, validated, raw_events, metadata):
        """Augment the existing validator; do not rewrite raw detector thresholds."""
        tracker = self.range_evidence
        initial = None
        if tracker.parent_id != self._parent_generation or tracker.context != self.context.value:
            tracker = self.range_evidence = RangeEvidence(self.cfg, self._parent_generation, self.context.value)
            for record in self.climax_evidence:
                if record['status'] == 'confirmed':
                    original = dict(record, timestamp=pd.Timestamp(record['candidate_timestamp']),
                                    available_at=pd.Timestamp(record['candidate_available_at']))
                    initial = (record['candidate_index'], original)
        if self.context != WyckoffContext.NONE:
            tracker.consume(bar_idx, row, validated, raw_events, metadata, self.get_phase_dir(), initial)
            if tracker.reaction is not None:
                if self.context == WyckoffContext.ACCUMULATION:
                    self.range_ref.ar_high = tracker.reaction.high
                    self.range_ref.ar_bar_idx = tracker.reaction.index
                else:
                    self.range_ref.as_low = -tracker.reaction.high
                    self.range_ref.as_bar_idx = tracker.reaction.index
            if tracker.emission is not None:
                key = tracker.emission
                validated[key] = True
                self.anchored_confidence[key] = tracker.retest['confidence']
                self.state = WyckoffState.ACCUM_LPS if key == 'lps' else WyckoffState.DISTRIB_LPSY
        self.structural_evidence = tracker.snapshot()

    def _check_invalidation(self, row: dict):
        """Check if current structure should be invalidated"""
        if self.context == WyckoffContext.ACCUMULATION and self.range_ref.sc_low > 0:
            if row['close'] < self.range_ref.sc_low:
                bar_vol = row.get('volume_z', 0)
                if bar_vol >= self.range_ref.sc_volume * self.invalidation_vol_ratio:
                    self.range_ref.invalidated = True
        elif self.context == WyckoffContext.DISTRIBUTION and self.range_ref.bc_high > 0:
            if row['close'] > self.range_ref.bc_high:
                bar_vol = row.get('volume_z', 0)
                if bar_vol >= self.range_ref.bc_volume * self.invalidation_vol_ratio:
                    self.range_ref.invalidated = True

    def get_phase(self) -> str:
        """Return current Wyckoff phase letter based on state"""
        # Context-only M2 shadow: report phase C while the underlying state is
        # still the phase-B state (no state/event mutation — see M2 branches)
        if self.m2_shadow_c and self.state in (
                WyckoffState.ACCUM_ST, WyckoffState.ACCUM_SPRING,
                WyckoffState.DISTRIB_ST, WyckoffState.DISTRIB_UT):
            return 'C'
        phase_map = {
            WyckoffState.NONE: 'neutral',
            WyckoffState.ACCUM_SC: 'A', WyckoffState.ACCUM_AR: 'A',
            WyckoffState.ACCUM_ST: 'A', WyckoffState.ACCUM_SPRING: 'C',
            WyckoffState.ACCUM_LPS_C: 'C',
            WyckoffState.ACCUM_SOS: 'B', WyckoffState.ACCUM_LPS: 'D',
            WyckoffState.DISTRIB_BC: 'A', WyckoffState.DISTRIB_AR: 'A',
            WyckoffState.DISTRIB_ST: 'A', WyckoffState.DISTRIB_UT: 'C',
            WyckoffState.DISTRIB_LPSY_C: 'C',
            WyckoffState.DISTRIB_SOW: 'B', WyckoffState.DISTRIB_LPSY: 'D',
        }
        return phase_map.get(self.state, 'neutral')

    def get_context_str(self) -> str:
        """Return context string"""
        return self.context.value

    def get_phase_dir(self) -> str:
        """Directional phase label: phase letter + structure direction.

        The plain letter conflates accumulation and distribution — e.g. both
        ACCUM_SPRING and DISTRIB_UT map to 'C' in get_phase(). Consumers that
        need direction should read this (or pair phase with wyckoff_context).
        """
        phase = self.get_phase()
        if phase == 'neutral':
            return 'neutral'
        if self.context == WyckoffContext.ACCUMULATION:
            return f"{phase}_accum"
        if self.context == WyckoffContext.DISTRIBUTION:
            return f"{phase}_distrib"
        return phase


@dataclass
class WyckoffEvent:
    """Single Wyckoff event detection result"""
    event_type: str
    confidence: float
    bar_index: int
    metadata: Dict


@dataclass
class WyckoffHTFContext:
    """Context from higher-timeframe Wyckoff detection to inform lower-TF analysis.

    Created by create_wyckoff_context() after running detect_all_wyckoff_events()
    on a higher timeframe. Passed to detect_all_wyckoff_events() on a lower timeframe
    to modulate confidence scores based on HTF alignment.

    Example flow:
        df_1d = detect_all_wyckoff_events(df_1d)
        ctx_1d = create_wyckoff_context(df_1d, timeframe="1D")
        df_4h = detect_all_wyckoff_events(df_4h, htf_context=ctx_1d)
        ctx_4h = create_wyckoff_context(df_4h, timeframe="4H")
        df_1h = detect_all_wyckoff_events(df_1h, htf_context=ctx_4h)
    """
    phase: str              # "accumulation", "distribution", "transition", "none"
    bullish_score: float    # 0-1, graded strength of accumulation signals
    bearish_score: float    # 0-1, graded strength of distribution signals
    dominant_direction: str  # "bullish", "bearish", "neutral"
    recent_events: Dict[str, float]  # {event_name: max_confidence} from recent bars
    timeframe: str          # "1D", "4H", etc. for logging
    # Raw pre-arbitration sides (2026-08-20): net-dominance forces the losing
    # side to exactly 0.0, so downstream displays can't tell "no bullish
    # evidence" from "bullish evidence netted away". These carry the raw maxes
    # for display/logging; net bullish_score/bearish_score stay authoritative
    # for decisions. Defaults keep older constructors valid.
    raw_bullish_score: float = 0.0
    raw_bearish_score: float = 0.0


def _rolling_z_score(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling z-score for a series.

    Args:
        series: Input series (e.g., volume, range)
        window: Rolling window size

    Returns:
        Z-score series
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    z_score = (series - rolling_mean) / (rolling_std + 1e-9)
    return z_score.fillna(0.0)


def _range_position(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Calculate position in recent range (0-1).

    Args:
        df: OHLCV dataframe
        lookback: Number of bars for range calculation

    Returns:
        Series with range position (0 = at lows, 1 = at highs)
    """
    rolling_high = df['high'].rolling(lookback).max()
    rolling_low = df['low'].rolling(lookback).min()
    range_size = rolling_high - rolling_low

    position = (df['close'] - rolling_low) / (range_size + 1e-9)
    return position.clip(0, 1).fillna(0.5)


def _wick_quality(df: pd.DataFrame, direction: str = 'lower') -> pd.Series:
    """
    Calculate wick size relative to body and range.

    Args:
        df: OHLCV dataframe
        direction: 'lower' for lower wicks, 'upper' for upper wicks

    Returns:
        Series with wick quality score (0-1)
    """
    body = (df['close'] - df['open']).abs()
    total_range = df['high'] - df['low']

    if direction == 'lower':
        wick = df[['open', 'close']].min(axis=1) - df['low']
    else:
        wick = df['high'] - df[['open', 'close']].max(axis=1)

    # Wick quality: larger wick relative to body = higher score
    wick_ratio = wick / (body + 1e-9)
    range_ratio = wick / (total_range + 1e-9)

    quality = (wick_ratio.clip(0, 5) / 5.0) * 0.6 + range_ratio * 0.4
    return quality.clip(0, 1).fillna(0.0)


# ============================================================================
# PHASE A: SELLING CLIMAX / BUYING CLIMAX EVENTS
# ============================================================================

def detect_selling_climax(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Selling Climax (SC) - Capitulation event at market bottom.

    Characteristics (Institutional Moneytaur Pattern):
        - Extreme volume spike (volume_z > 2.5)
        - Price at 20-bar lows (range_position < 0.2)
        - Large range bar (range_z > 1.5)
        - Often coincides with liquidity_score < -0.5 (stop hunts)
        - Lower wick > 60% of total range (absorption)

    Args:
        df: OHLCV dataframe with technical indicators
        cfg: Configuration dict with thresholds

    Returns:
        Tuple of (detected: pd.Series[bool], confidence: pd.Series[float])
    """
    # Get or compute required features
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    if 'range_position' not in df.columns:
        df['range_position'] = _range_position(df, lookback=20)

    # Compute range z-score
    bar_range = df['high'] - df['low']
    range_z = _rolling_z_score(bar_range, window=20)

    # Wick quality (absorption)
    lower_wick_quality = _wick_quality(df, direction='lower')

    # Get config thresholds
    volume_z_min = cfg.get('sc_volume_z_min', 2.5)
    range_pos_max = cfg.get('sc_range_pos_max', 0.2)
    range_z_min = cfg.get('sc_range_z_min', 1.5)
    sc_range_lookback = cfg.get('sc_range_lookback', 50)

    # Use extended lookback for structural context (20-bar misses post-crash bounces)
    long_range_pos = _range_position(df, lookback=sc_range_lookback)

    # Detection criteria: 3 hard gates (wick is now confidence modifier only)
    extreme_volume = df['volume_z'] > volume_z_min
    at_lows = long_range_pos < range_pos_max
    wide_range = range_z > range_z_min

    # Confluence: volume + position + range (wick removed as gate)
    detected = extreme_volume & at_lows & wide_range

    # Confidence score: wick now 30% of confidence (was 15% gate)
    confidence = (
        (df['volume_z'] / 5.0).clip(0, 1) * 0.30 +
        (1 - long_range_pos) * 0.20 +
        (range_z / 3.0).clip(0, 1) * 0.20 +
        lower_wick_quality.clip(0, 1) * 0.30
    )
    confidence = confidence.fillna(0.0)

    # Only non-zero confidence where detected
    confidence = confidence * detected

    return detected, confidence


def detect_buying_climax(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Buying Climax (BC) - Euphoria peak at market top.

    Characteristics:
        - Extreme volume spike (volume_z > 2.5)
        - Price at 20-bar highs (range_position > 0.8)
        - Large range bar (range_z > 1.5)
        - Upper wick > 60% of total range (rejection)
        - Often coincides with RSI > 75

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    if 'range_position' not in df.columns:
        df['range_position'] = _range_position(df, lookback=20)

    bar_range = df['high'] - df['low']
    range_z = _rolling_z_score(bar_range, window=20)
    upper_wick_quality = _wick_quality(df, direction='upper')

    # Config
    volume_z_min = cfg.get('bc_volume_z_min', 2.5)
    range_pos_min = cfg.get('bc_range_pos_min', 0.8)
    range_z_min = cfg.get('bc_range_z_min', 1.5)
    bc_range_lookback = cfg.get('bc_range_lookback', 50)

    # Use extended lookback for structural context
    long_range_pos = _range_position(df, lookback=bc_range_lookback)

    # Detection: 3 hard gates (wick removed as gate)
    extreme_volume = df['volume_z'] > volume_z_min
    at_highs = long_range_pos > range_pos_min
    wide_range = range_z > range_z_min

    detected = extreme_volume & at_highs & wide_range

    # Close conviction: euphoric tops close near high (no upper wick)
    # max(upper_wick, close_conviction) captures both rejection wicks AND euphoric closes
    close_conviction = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).clip(0, 1)
    wick_or_conviction = pd.concat([upper_wick_quality, close_conviction], axis=1).max(axis=1)

    # Confidence: wick/conviction now 30% modifier (was 15% gate)
    confidence = (
        (df['volume_z'] / 5.0).clip(0, 1) * 0.30 +
        long_range_pos * 0.20 +
        (range_z / 3.0).clip(0, 1) * 0.20 +
        wick_or_conviction * 0.30
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_climax_v2(df: pd.DataFrame, cfg: dict,
                     side: str = "buying") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    v2 SHADOW climax detector — disambiguates climax from breakout/breakdown.

    Background (wyckoff_audit.md + live audit 2026-08-20): the v1 BC/SC
    detectors share 3 hard gates (volume_z, long range_position, range_z) with
    NO reversal requirement, so an SOS breakout bar and a Buying Climax are
    indistinguishable — BC's forward returns match SOS's (+1.2%/72h), i.e. it
    fires on rallies. The wick gate can't simply be restored: real euphoric
    tops (Apr/Nov 2021) close AT their highs and were only caught after the
    wick gate was demoted (see "6 Fixes Applied", fix #2). The only honest
    disambiguator for a strong-close candidate is WHAT HAPPENS NEXT:
        reversal within k bars  -> climax   (confirmed, stamped late)
        continuation            -> breakout (neither column fires)

    Emits three causal series:
        detected   — immediate fire: v1 gates AND rejection evidence on the
                     bar itself (wick quality >= wick_min OR close in the
                     opposite third of the bar's range).
        confidence — v1-style confidence, only on immediate fires.
        confirmed  — a gate-passing candidate (with or without rejection)
                     whose reversal completed: for buying, a close BELOW the
                     candidate bar's low within confirm_bars; for selling, a
                     close ABOVE the candidate bar's high. Stamped at the
                     CONFIRMATION bar — on truncated data the flag simply
                     hasn't appeared yet, so earlier bars never repaint.

    SHADOW CONTRACT: these columns are data-collection only. They are NOT in
    _ACCUM_EVENTS/_DISTRIB_EVENTS, so they cannot touch directional scores,
    the state machine, phases, or any live consumer.
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    bar_range = df['high'] - df['low']
    range_z = _rolling_z_score(bar_range, window=20)
    close_loc = ((df['close'] - df['low']) / (bar_range + 1e-9)).clip(0, 1)

    volume_z_min = cfg.get('climax_v2_volume_z_min', 2.5)
    range_z_min = cfg.get('climax_v2_range_z_min', 1.5)
    wick_min = cfg.get('climax_v2_wick_min', 0.5)
    confirm_bars = cfg.get('climax_v2_confirm_bars', 5)
    range_lookback = cfg.get('climax_v2_range_lookback', 50)

    extreme_volume = df['volume_z'] > volume_z_min
    wide_range = range_z > range_z_min

    # Position of the EXTREME reached, not the close. v1's _range_position is
    # close-based, so a genuine rejection top (which closes low) fails the
    # "at_highs" gate — the structural reason v1 BC only catches strong-close
    # bars (breakouts). A climax is defined by where price REACHED.
    rolling_high = df['high'].rolling(range_lookback).max()
    rolling_low = df['low'].rolling(range_lookback).min()
    range_size = rolling_high - rolling_low + 1e-9

    if side == "buying":
        extreme_reach = ((df['high'] - rolling_low) / range_size).clip(0, 1)
        at_extreme = extreme_reach > cfg.get('bc_range_pos_min', 0.8)
        wick_quality = _wick_quality(df, direction='upper')
        # rejection NOW: real upper wick or a close in the lower third
        rejection = (wick_quality >= wick_min) | (close_loc <= 0.35)
        extreme_pos_conf = extreme_reach
    else:
        extreme_reach = ((df['low'] - rolling_low) / range_size).clip(0, 1)
        at_extreme = extreme_reach < cfg.get('sc_range_pos_max', 0.2)
        wick_quality = _wick_quality(df, direction='lower')
        # absorption NOW: real lower wick or a close in the upper third
        rejection = (wick_quality >= wick_min) | (close_loc >= 0.65)
        extreme_pos_conf = 1.0 - extreme_reach

    candidate = (extreme_volume & at_extreme & wide_range).fillna(False)
    detected = (candidate & rejection).fillna(False)

    confidence = (
        (df['volume_z'] / 5.0).clip(0, 1) * 0.30 +
        extreme_pos_conf.clip(0, 1) * 0.20 +
        (range_z / 3.0).clip(0, 1) * 0.20 +
        wick_quality.clip(0, 1) * 0.30
    )
    confidence = confidence.fillna(0.0) * detected

    # Delayed confirmation — causal by construction: we only ever stamp a bar
    # AFTER the candidate, using closes up to that bar. Truncating the frame
    # removes future stamps without altering past ones (no repaint).
    confirmed = pd.Series(False, index=df.index)
    closes = df['close'].values
    cand_idx = np.flatnonzero(candidate.values)
    n = len(df)
    for i in cand_idx:
        ref_low = df['low'].values[i]
        ref_high = df['high'].values[i]
        for j in range(i + 1, min(i + 1 + confirm_bars, n)):
            if side == "buying" and closes[j] < ref_low:
                confirmed.iloc[j] = True
                break
            if side == "selling" and closes[j] > ref_high:
                confirmed.iloc[j] = True
                break

    return detected.astype(bool), confidence, confirmed


def detect_automatic_rally(df: pd.DataFrame, cfg: dict,
                          sc_events: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Automatic Rally (AR) - Relief bounce after Selling Climax.

    Characteristics (Zeroika Pattern):
        - Must occur within 5-10 bars after SC
        - Volume declining (volume_z < 1.0)
        - Retraces 40-70% of SC range
        - Closes in upper 60% of bar (buying pressure)
        - No new lows made

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict
        sc_events: Optional SC detection results for context

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback_max = cfg.get('ar_lookback_max', 10)
    volume_z_max = cfg.get('ar_volume_z_max', 1.0)
    retrace_min = cfg.get('ar_retrace_min', 0.40)
    retrace_max = cfg.get('ar_retrace_max', 0.70)
    close_position_min = cfg.get('ar_close_position_min', 0.6)

    # Calculate retrace from recent low
    rolling_low = df['low'].rolling(lookback_max).min()
    rolling_high = df['high'].rolling(lookback_max).max()
    recent_range = rolling_high - rolling_low
    retrace_pct = (df['close'] - rolling_low) / (recent_range + 1e-9)

    # Bar close position (buying pressure indicator)
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Detection criteria
    declining_volume = df['volume_z'] < volume_z_max
    proper_retrace = (retrace_pct > retrace_min) & (retrace_pct < retrace_max)
    strong_close = close_position > close_position_min
    no_new_lows = df['low'] > df['low'].shift(1)  # Not making new lows

    detected = declining_volume & proper_retrace & strong_close & no_new_lows

    # Confidence
    retrace_quality = 1 - abs(retrace_pct - 0.55) / 0.55  # Peak at 55% retrace
    confidence = (
        (1 - df['volume_z'] / 3.0).clip(0, 1) * 0.30 +
        retrace_quality.clip(0, 1) * 0.35 +
        close_position * 0.25 +
        no_new_lows.astype(float) * 0.10
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_automatic_reaction(df: pd.DataFrame, cfg: dict,
                              bc_events: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Automatic Reaction (AS) - Relief drop after Buying Climax.

    Mirror of AR for distribution phase.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict
        bc_events: Optional BC detection results

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback_max = cfg.get('as_lookback_max', 10)
    volume_z_max = cfg.get('as_volume_z_max', 1.0)
    retrace_min = cfg.get('as_retrace_min', 0.40)
    retrace_max = cfg.get('as_retrace_max', 0.70)
    close_position_max = cfg.get('as_close_position_max', 0.4)

    # Calculate retrace from recent high
    rolling_high = df['high'].rolling(lookback_max).max()
    rolling_low = df['low'].rolling(lookback_max).min()
    recent_range = rolling_high - rolling_low
    retrace_pct = (rolling_high - df['close']) / (recent_range + 1e-9)

    # Bar close position (selling pressure indicator)
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Detection
    declining_volume = df['volume_z'] < volume_z_max
    proper_retrace = (retrace_pct > retrace_min) & (retrace_pct < retrace_max)
    weak_close = close_position < close_position_max
    no_new_highs = df['high'] < df['high'].shift(1)

    detected = declining_volume & proper_retrace & weak_close & no_new_highs

    # Confidence
    retrace_quality = 1 - abs(retrace_pct - 0.55) / 0.55
    confidence = (
        (1 - df['volume_z'] / 3.0).clip(0, 1) * 0.30 +
        retrace_quality.clip(0, 1) * 0.35 +
        (1 - close_position) * 0.25 +
        no_new_highs.astype(float) * 0.10
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_secondary_test(df: pd.DataFrame, cfg: dict,
                          sc_events: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Secondary Test (ST) - Retest of SC lows on lower volume.

    Characteristics:
        - Price returns to SC low area (within 5%)
        - Volume significantly lower than SC (volume_z < 0.5)
        - Price holds above SC low (no new low)
        - Occurs 10-30 bars after SC

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict
        sc_events: Optional SC detection for reference

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config — recalibrated v2 (consensus: PyQuantLab VSA, loopofM, QuantVue)
    lookback = cfg.get('st_lookback', 15)            # was 30, consensus 15-bar
    low_proximity = cfg.get('st_low_proximity', 0.03)  # was 0.05 (5%), now 3%
    volume_z_max = cfg.get('st_volume_z_max', 0.0)   # was 0.5, now below 20-bar mean
    min_spacing = cfg.get('st_min_spacing', 10)       # 10-bar debounce (loopofM)

    # Find recent lows
    rolling_low = df['low'].rolling(lookback).min()
    distance_from_low = (df['low'] - rolling_low) / (rolling_low + 1e-9)

    # Detection
    near_low = distance_from_low < low_proximity
    lower_volume = df['volume_z'] < volume_z_max
    holds_above = df['low'] > rolling_low  # No new low

    detected = near_low & lower_volume & holds_above

    # Apply debounce: suppress detections within min_spacing bars of last ST
    if min_spacing > 0 and detected.any():
        mask = detected.values.copy()
        last_fire = -min_spacing - 1  # Allow first detection
        for i in range(len(mask)):
            if mask[i]:
                if (i - last_fire) < min_spacing:
                    mask[i] = False
                else:
                    last_fire = i
        detected = pd.Series(mask, index=df.index)

    # Confidence (higher if closer to low with much lower volume)
    proximity_score = (1 - distance_from_low / low_proximity).clip(0, 1)
    volume_score = (1 - df['volume_z'] / 2.0).clip(0, 1)

    confidence = (
        proximity_score * 0.40 +
        volume_score * 0.40 +
        holds_above.astype(float) * 0.20
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


# ============================================================================
# PHASE B: SIGN OF STRENGTH / WEAKNESS
# ============================================================================

def detect_sign_of_strength(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Sign of Strength (SOS) - First decisive move up with volume.

    Characteristics:
        - Breaks above recent range (> 20-bar high)
        - Strong volume (volume_z > 1.5)
        - Close in top 70% of bar
        - Follows period of consolidation
        - Often coincides with BOS (break of structure)

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback = cfg.get('sos_lookback', 20)
    volume_z_min = cfg.get('sos_volume_z_min', 1.5)
    close_position_min = cfg.get('sos_close_position_min', 0.7)
    breakout_margin = cfg.get('sos_breakout_margin', 0.01)  # 1% above high

    # Calculate breakout
    rolling_high = df['high'].rolling(lookback).max().shift(1)
    breaks_high = df['close'] > rolling_high * (1 + breakout_margin)

    # Bar characteristics
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Volume
    strong_volume = df['volume_z'] > volume_z_min

    # Detection
    detected = breaks_high & strong_volume & (close_position > close_position_min)

    # Confidence
    breakout_strength = ((df['close'] - rolling_high) / (rolling_high + 1e-9)).clip(0, 0.1) / 0.1
    confidence = (
        breakout_strength * 0.30 +
        (df['volume_z'] / 4.0).clip(0, 1) * 0.35 +
        close_position * 0.25 +
        0.10  # Base confidence for meeting criteria
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_sign_of_weakness(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Sign of Weakness (SOW) - First decisive move down with volume.

    Mirror of SOS for distribution.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('sow_lookback', 20)
    volume_z_min = cfg.get('sow_volume_z_min', 1.5)
    close_position_max = cfg.get('sow_close_position_max', 0.3)
    breakdown_margin = cfg.get('sow_breakdown_margin', 0.01)

    # Calculate breakdown
    rolling_low = df['low'].rolling(lookback).min().shift(1)
    breaks_low = df['close'] < rolling_low * (1 - breakdown_margin)

    # Bar characteristics
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    # Volume
    strong_volume = df['volume_z'] > volume_z_min

    # Detection
    detected = breaks_low & strong_volume & (close_position < close_position_max)

    # Confidence
    breakdown_strength = ((rolling_low - df['close']) / (rolling_low + 1e-9)).clip(0, 0.1) / 0.1
    confidence = (
        breakdown_strength * 0.30 +
        (df['volume_z'] / 4.0).clip(0, 1) * 0.35 +
        (1 - close_position) * 0.25 +
        0.10
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


# ============================================================================
# PHASE C: SPRINGS AND UPTHRUSTS
# ============================================================================

def detect_spring_type_a(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Spring Type A - Deep fake breakdown below range to trap sellers.

    Characteristics (Classic Wyckoff):
        - Breaks below 20-bar low convincingly (> 2%)
        - Reverses quickly (within 1-3 bars)
        - Closes back inside range
        - Volume spike on breakdown, then decline on recovery
        - Often coincides with liquidity sweep (liquidity_score spike)

    Confirmation-based approach (no lookahead bias):
        Instead of looking FORWARD from the candidate bar to see if recovery
        happens, we look BACKWARD from the current bar to see if a candidate
        existed recovery_bars ago AND the current close confirms recovery.
        The event fires on the CONFIRMATION bar, not the candidate bar.
        Confidence values come from the candidate bar (shifted) to preserve
        the original scoring semantics.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    # Config
    lookback = cfg.get('spring_a_lookback', 20)
    breakdown_margin = cfg.get('spring_a_breakdown_margin', 0.015)  # 1.5% below low (was 2%)
    recovery_bars = cfg.get('spring_a_recovery_bars', 3)
    volume_z_min = cfg.get('spring_a_volume_z_min', 0.5)  # Configurable (was hardcoded 1.0)

    # Calculate range
    rolling_low = df['low'].rolling(lookback).min().shift(1)
    rolling_high = df['high'].rolling(lookback).max().shift(1)

    # Breakdown detection (candidate spring: all conditions EXCEPT recovery)
    breaks_low = df['low'] < rolling_low * (1 - breakdown_margin)

    # Volume characteristics on the candidate bar (relaxed from 1.0)
    volume_spike = df['volume_z'] > volume_z_min

    # Candidate spring = breakdown + volume spike (no future data needed)
    candidate_spring = breaks_low & volume_spike

    # Confirmation-based recovery: check if a candidate existed recovery_bars
    # ago AND the current bar's close is back above that candidate's rolling_low.
    # This fires on the confirmation bar with zero lookahead bias.
    confirmed_spring = (
        candidate_spring.shift(recovery_bars).fillna(False) &
        (df['close'] > rolling_low.shift(recovery_bars))
    )

    detected = confirmed_spring

    # Confidence from the CANDIDATE bar (shifted), not the confirmation bar,
    # to preserve the original scoring semantics based on breakdown depth/volume.
    breakdown_depth = ((rolling_low - df['low']) / (rolling_low + 1e-9)).clip(0, 0.05) / 0.05
    candidate_confidence = (
        breakdown_depth * 0.40 +
        (df['volume_z'] / 3.0).clip(0, 1) * 0.35 +
        0.25  # Base for meeting criteria
    )
    candidate_confidence = candidate_confidence.fillna(0.0)

    # Shift confidence to align with confirmation bar
    confidence = candidate_confidence.shift(recovery_bars).fillna(0.0) * detected

    return detected, confidence


def detect_spring_type_b(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Spring Type B - Shallow spring with quick recovery.

    Characteristics:
        - Slightly below range low (0.5-1%)
        - Immediate reversal (same or next bar)
        - Lower volume than Type A
        - Close above mid-range quickly

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('spring_b_lookback', 20)
    breakdown_min = cfg.get('spring_b_breakdown_min', 0.002)   # was 0.005, wider shallow band
    breakdown_max = cfg.get('spring_b_breakdown_max', 0.015)   # was 0.01, more realistic wicks
    recovery_bars = cfg.get('spring_b_recovery_bars', 3)       # 3-bar confirmation (QuantVue)

    rolling_low = df['low'].rolling(lookback).min().shift(1)
    rolling_high = df['high'].rolling(lookback).max().shift(1)
    range_lower_quartile = rolling_low + (rolling_high - rolling_low) * 0.25

    # Shallow breakdown
    breakdown_pct = (rolling_low - df['low']) / (rolling_low + 1e-9)
    shallow_break = (breakdown_pct > breakdown_min) & (breakdown_pct < breakdown_max)

    # Moderate volume (relaxed lower bound) — evaluated on the candidate bar
    moderate_volume = (df['volume_z'] > -0.5) & (df['volume_z'] < 2.0)

    # Candidate spring: shallow breakdown + moderate volume (no future data)
    candidate = shallow_break & moderate_volume

    # Confirmation-based recovery (same pattern as detect_spring_type_a /
    # detect_upthrust): a candidate existed recovery_bars ago AND the current
    # close is back above that candidate's lower quartile. Fires on the
    # CONFIRMATION bar with zero lookahead — a recovery that did not HOLD
    # through the confirmation bar does not fire.
    if recovery_bars > 1:
        detected = (
            candidate.shift(recovery_bars).fillna(False).astype(bool) &
            (df['close'] > range_lower_quartile.shift(recovery_bars))
        )
        conf_shift = recovery_bars
    else:
        detected = candidate & (df['close'] > range_lower_quartile)
        conf_shift = 0

    # Confidence: breakdown depth + volume from the CANDIDATE bar (shifted to
    # the confirmation bar); recovery strength from the CURRENT close measured
    # against the candidate's range.
    quart_c = range_lower_quartile.shift(conf_shift)
    high_c = rolling_high.shift(conf_shift)
    confidence = (
        (breakdown_pct.shift(conf_shift) / breakdown_max).clip(0, 1) * 0.35 +
        ((df['close'] - quart_c) / (high_c - quart_c + 1e-9)).clip(0, 1) * 0.35 +
        (df['volume_z'].shift(conf_shift).clip(-1, 3) / 4.0 + 0.25).clip(0, 1) * 0.30
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_upthrust(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Upthrust (UT) - Fake breakout above range to trap buyers.

    Characteristics:
        - Breaks above range high
        - Reverses quickly (fails to hold)
        - Closes back inside range
        - Volume spike on breakout
        - Often precedes distribution

    Confirmation-based approach (no lookahead bias):
        Instead of looking FORWARD from the candidate bar to see if the
        breakout fails, we look BACKWARD from the current bar to see if a
        candidate existed recovery_bars ago AND the current close confirms
        failure (price retreated below the range high). The event fires on
        the CONFIRMATION bar. Confidence values come from the candidate bar
        (shifted) to preserve the original scoring semantics.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('ut_lookback', 20)
    breakout_margin = cfg.get('ut_breakout_margin', 0.015)  # 1.5% breakout (was 2%)
    recovery_bars = cfg.get('ut_recovery_bars', 3)
    volume_z_min = cfg.get('ut_volume_z_min', 0.3)  # Relaxed: low-vol fakeouts are valid UTs

    rolling_high = df['high'].rolling(lookback).max().shift(1)
    rolling_low = df['low'].rolling(lookback).min().shift(1)

    # Breakout above range (candidate upthrust: all conditions EXCEPT reversal)
    breaks_high = df['high'] > rolling_high * (1 + breakout_margin)

    # Volume spike on the candidate bar (relaxed from 1.0)
    volume_spike = df['volume_z'] > volume_z_min

    # Candidate upthrust = breakout + volume spike (no future data needed)
    candidate_ut = breaks_high & volume_spike

    # Confirmation-based reversal: check if a candidate existed recovery_bars
    # ago AND the current bar's close is back below that candidate's rolling_high.
    # This fires on the confirmation bar with zero lookahead bias.
    confirmed_ut = (
        candidate_ut.shift(recovery_bars).fillna(False) &
        (df['close'] < rolling_high.shift(recovery_bars))
    )

    detected = confirmed_ut

    # Confidence from the CANDIDATE bar (shifted), not the confirmation bar,
    # to preserve the original scoring semantics based on breakout size/volume.
    breakout_size = ((df['high'] - rolling_high) / (rolling_high + 1e-9)).clip(0, 0.05) / 0.05
    candidate_confidence = (
        breakout_size * 0.40 +
        (df['volume_z'] / 3.0).clip(0, 1) * 0.35 +
        0.25
    )
    candidate_confidence = candidate_confidence.fillna(0.0)

    # Shift confidence to align with confirmation bar
    confidence = candidate_confidence.shift(recovery_bars).fillna(0.0) * detected

    return detected, confidence


def detect_upthrust_after_distribution(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect UTAD - Final upthrust at end of distribution before markdown.

    More aggressive than regular UT, often marks the absolute top.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    # Similar to UT but with additional context checks
    ut_detected, ut_confidence = detect_upthrust(df, cfg)

    # RSI as confidence modifier (not hard gate — euphoric tops may not have RSI>70)
    if 'rsi_14' in df.columns:
        rsi_val = df['rsi_14'].fillna(50)
        # RSI 60-80 adds 0-0.2 bonus; below 60 gets no bonus but isn't blocked
        rsi_bonus = ((rsi_val - 60) / 100).clip(0, 0.2)
        ut_confidence = (ut_confidence + rsi_bonus * ut_detected).clip(upper=1.0)

    return ut_detected, ut_confidence


# ============================================================================
# PHASE D/E: LAST POINT OF SUPPORT/SUPPLY
# ============================================================================

def detect_last_point_of_support(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Last Point of Support (LPS) - Final test before markup begins.

    Characteristics:
        - Price pulls back to support area
        - Volume very low (volume_z < 0.0)
        - Holds above ST/Spring low
        - Occurs after SOS
        - Close in upper 60% of bar

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('lps_lookback', 30)
    volume_z_max = cfg.get('lps_volume_z_max', 0.0)
    close_position_min = cfg.get('lps_close_position_min', 0.6)

    # Find support (recent low after which price rose)
    rolling_low = df['low'].rolling(lookback).min()

    # Price at support area (within 3% of low)
    at_support = (df['low'] - rolling_low) / (rolling_low + 1e-9) < 0.03

    # Very low volume
    low_volume = df['volume_z'] < volume_z_max

    # Strong close
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)
    strong_close = close_position > close_position_min

    detected = at_support & low_volume & strong_close

    # Confidence
    confidence = (
        (1 - df['volume_z'] / 2.0).clip(0, 1) * 0.40 +
        close_position * 0.35 +
        at_support.astype(float) * 0.25
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


def detect_last_point_of_supply(df: pd.DataFrame, cfg: dict) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Last Point of Supply (LPSY) - Final rally before markdown.

    Mirror of LPS for distribution phase.

    Args:
        df: OHLCV dataframe
        cfg: Configuration dict

    Returns:
        Tuple of (detected, confidence)
    """
    if 'volume_z' not in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    lookback = cfg.get('lpsy_lookback', 30)
    volume_z_max = cfg.get('lpsy_volume_z_max', 0.0)
    close_position_max = cfg.get('lpsy_close_position_max', 0.4)

    # Find resistance
    rolling_high = df['high'].rolling(lookback).max()

    # Price at resistance (within 3%)
    at_resistance = (rolling_high - df['high']) / (rolling_high + 1e-9) < 0.03

    # Very low volume
    low_volume = df['volume_z'] < volume_z_max

    # Weak close
    bar_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)
    weak_close = close_position < close_position_max

    detected = at_resistance & low_volume & weak_close

    # Confidence
    confidence = (
        (1 - df['volume_z'] / 2.0).clip(0, 1) * 0.40 +
        (1 - close_position) * 0.35 +
        at_resistance.astype(float) * 0.25
    )
    confidence = confidence.fillna(0.0) * detected

    return detected, confidence


# ============================================================================
# HIGHER-TIMEFRAME CONTEXT CREATION & MODULATION
# ============================================================================

# Events aligned with accumulation (bullish Wyckoff structure)
_ACCUM_EVENTS = ['sc', 'ar', 'st', 'spring_a', 'spring_b', 'sos', 'lps']
# Events aligned with distribution (bearish Wyckoff structure)
_DISTRIB_EVENTS = ['bc', 'as', 'sow', 'ut', 'utad', 'lpsy']


def create_wyckoff_context(df: pd.DataFrame, lookback: int = 3,
                           timeframe: str = "unknown",
                           mutual_exclusion: bool = True) -> WyckoffHTFContext:
    """
    Create a WyckoffHTFContext from a DataFrame that has been through detect_all_wyckoff_events().

    Extracts graded bullish/bearish scores from the last `lookback` bars, determines
    the dominant phase and direction. This context is then passed to a lower-timeframe
    detect_all_wyckoff_events() call to modulate confidence scores.

    Args:
        df: DataFrame with wyckoff_*_confidence columns (output of detect_all_wyckoff_events)
        lookback: Number of recent bars to scan for events (default: 3)
        timeframe: Label for logging ("1D", "4H", etc.)
        mutual_exclusion: If True (default), apply net-dominance arbitration so
            bullish/bearish behave as a direction signal (loser = max(0, loser_raw
            - winner_raw)). If False, both scores are returned as raw maxes
            (LEGACY behavior — known bug, kept for opt-out / rollback).

    Returns:
        WyckoffHTFContext with graded scores and phase determination
    """
    tail = df.iloc[-lookback:] if len(df) >= lookback else df

    # Graded bullish score: max confidence across recent accumulation events
    bullish_confs = []
    for e in _ACCUM_EVENTS:
        col = f'wyckoff_{e}_confidence'
        if col in tail.columns:
            vals = tail[col].dropna()
            if len(vals) > 0:
                bullish_confs.append(float(vals.max()))
    raw_bullish_score = max(bullish_confs) if bullish_confs else 0.0

    # Graded bearish score: max confidence across recent distribution events
    bearish_confs = []
    for e in _DISTRIB_EVENTS:
        col = f'wyckoff_{e}_confidence'
        if col in tail.columns:
            vals = tail[col].dropna()
            if len(vals) > 0:
                bearish_confs.append(float(vals.max()))
    raw_bearish_score = max(bearish_confs) if bearish_confs else 0.0

    # --- Direction arbitration (fix for 2026-06-02 mutual-exclusion bug) ---
    # Without this, raw_bullish and raw_bearish are independent maxes over
    # disjoint event families, so both can be > 0.8 simultaneously (28/94 live
    # trades in trade_outcomes_live_jun2.csv had this). Downstream archetype
    # gates treat them as direction signals — that's broken without arbitration.
    #
    # Strategy: subtract loser from winner ("net dominance"). This guarantees:
    #   - true ties (margin=0) collapse BOTH to 0 (regime is ambiguous → no signal)
    #   - decisive dominance (margin >= winner) leaves only the winner
    #   - the invariant "not both > 0.5 when there's any directional edge" holds
    if mutual_exclusion:
        if raw_bullish_score >= raw_bearish_score:
            bullish_score = max(0.0, raw_bullish_score - raw_bearish_score)
            bearish_score = 0.0
        else:
            bearish_score = max(0.0, raw_bearish_score - raw_bullish_score)
            bullish_score = 0.0

        # Regression guard: with net dominance, by construction at most ONE side
        # can be > 0. This warning will never fire under the current formula but
        # is left as belt-and-suspenders for future tweaks.
        if bullish_score > 0.5 and bearish_score > 0.5:
            logger.warning(
                f"WyckoffHTFContext({timeframe}): mutual_exclusion failed to separate scores "
                f"(bull={bullish_score:.3f}, bear={bearish_score:.3f}, raw_bull={raw_bullish_score:.3f}, "
                f"raw_bear={raw_bearish_score:.3f}) — should be impossible with net-dominance formula."
            )
    else:
        # Legacy / opt-out: raw maxes (known bug, kept for rollback)
        bullish_score = raw_bullish_score
        bearish_score = raw_bearish_score

    # Determine phase from score comparison
    if bullish_score > 0.3 and bullish_score > bearish_score * 1.5:
        phase = "accumulation"
    elif bearish_score > 0.3 and bearish_score > bullish_score * 1.5:
        phase = "distribution"
    elif bullish_score > 0 or bearish_score > 0:
        phase = "transition"
    else:
        phase = "none"

    # Direction (needs clearer edge than phase)
    if bullish_score > bearish_score + 0.1:
        direction = "bullish"
    elif bearish_score > bullish_score + 0.1:
        direction = "bearish"
    else:
        direction = "neutral"

    # Recent events dict (non-zero confidences only)
    recent: Dict[str, float] = {}
    for e in _ACCUM_EVENTS + _DISTRIB_EVENTS:
        col = f'wyckoff_{e}_confidence'
        if col in tail.columns:
            vals = tail[col].dropna()
            if len(vals) > 0:
                max_c = float(vals.max())
                if max_c > 0:
                    recent[e] = max_c

    ctx = WyckoffHTFContext(
        phase=phase,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        dominant_direction=direction,
        recent_events=recent,
        timeframe=timeframe,
        raw_bullish_score=raw_bullish_score,
        raw_bearish_score=raw_bearish_score,
    )
    logger.info(f"WyckoffHTFContext({timeframe}): phase={phase}, "
                f"bullish={bullish_score:.3f}, bearish={bearish_score:.3f}, "
                f"direction={direction}, events={len(recent)}")
    return ctx


def _delayed_evidence(df, cfg, key, confirmation, parents):
    """Recover the raw detector's candidate without observing beyond confirmation."""
    delay, lookback, price_col = _delayed_event_parameters(key, cfg)
    candidate = confirmation - delay
    if candidate < lookback or not isinstance(df.index, pd.DatetimeIndex):
        return None
    times = df.index[max(0, candidate - 1):confirmation + 1]
    times = times.tz_localize('UTC') if times.tz is None else times.tz_convert('UTC')
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        return None
    try:
        interval = pd.Timedelta(str(cfg['timeframe']).lower()) if 'timeframe' in cfg else times[1] - times[0]
    except (ValueError, TypeError, IndexError):
        return None
    if interval <= pd.Timedelta(0) or not (times[1:] - times[:-1] == interval).all():
        return None
    prior = df[price_col].iloc[candidate - lookback:candidate]
    boundary = prior.min() if price_col == 'low' else prior.max()
    snap = parents[candidate]
    return DelayedEventEvidence(
        event_type=key, candidate_index=candidate, confirmation_index=confirmation,
        candidate_extreme=df[price_col].iloc[candidate], prior_swept_boundary=boundary,
        candidate_parent_id=snap['id'], candidate_parent_context=snap['context'],
        candidate_parent_status=snap['status'], candidate_timestamp=times[1],
        confirmation_timestamp=times[-1], available_at=times[-1] + interval)


def _apply_state_machine_validation(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run state machine validation over all bars.
    Modifies event bool columns in-place: sets False for events that fail validation.
    Replaces phase_abc with state-machine-derived values.
    """
    sm = WyckoffStateMachine(cfg)

    event_keys = ['sc', 'bc', 'ar', 'as', 'st', 'sos', 'sow',
                  'spring_a', 'spring_b', 'ut', 'utad', 'lps', 'lpsy']

    n = len(df)
    validated_arrays = {k: np.zeros(n, dtype=bool) for k in event_keys}
    phases = ['neutral'] * n
    phase_dirs = ['neutral'] * n
    contexts = ['none'] * n
    confidence_mod_arrays = {k: np.ones(n, dtype=float) for k in event_keys}
    delayed_keys = ('spring_a', 'spring_b', 'ut', 'utad')
    parents = []
    provenance = {f'wyckoff_{k}_{field}': [None] * n
                  for k in delayed_keys for field in (
                      'candidate_index', 'candidate_extreme', 'prior_swept_boundary',
                      'candidate_parent_id', 'candidate_parent_context', 'candidate_parent_status',
                      'candidate_timestamp', 'confirmation_timestamp', 'available_at', 'evidence_status')}
    provenance['wyckoff_opposing_climax_evidence'] = [None] * n
    provenance['wyckoff_structure_evidence'] = [None] * n
    provenance['wyckoff_parent_id'] = [None] * n
    # Keep raw clues distinct from delayed confirmed events. This also makes a
    # validation-only replay use the original climax inputs, not its own output.
    climax_inputs = {}
    climax_confidence_overrides = {k: np.full(n, np.nan) for k in ('sc', 'bc', 'lps', 'lpsy')}
    for k in event_keys:
        for suffix, default in (('', False), ('_confidence', 0.0)):
            saved = f'wyckoff_{k}_raw{suffix}'
            source = saved if saved in df.columns else f'wyckoff_{k}{suffix}'
            dtype = float if suffix else bool
            values = (df[source].to_numpy(dtype=dtype, copy=True) if source in df.columns
                      else np.full(n, default, dtype=dtype))
            climax_inputs[k + suffix] = values
            provenance[saved] = values

    # Pre-extract arrays for performance
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volume_zs = df['volume_z'].values if 'volume_z' in df.columns else np.zeros(n)
    volumes = df['volume'].values if 'volume' in df.columns else np.full(n, np.nan)

    for i in range(n):
        parents.append(sm.parent_snapshot())
        row = {
            'open': opens[i],
            'high': highs[i],
            'low': lows[i],
            'close': closes[i],
            'volume_z': volume_zs[i],
            'volume': volumes[i],
            'sc_confidence': climax_inputs['sc_confidence'][i],
            'bc_confidence': climax_inputs['bc_confidence'][i],
            'sos_confidence': climax_inputs['sos_confidence'][i],
            'sow_confidence': climax_inputs['sow_confidence'][i],
            'clock_required': isinstance(df.index, pd.DatetimeIndex),
        }
        if isinstance(df.index, pd.DatetimeIndex) and not pd.isna(df.index[i]):
            timestamp = df.index[i]
            timestamp = timestamp.tz_localize('UTC') if timestamp.tz is None else timestamp.tz_convert('UTC')
            row['timestamp'] = timestamp
            try:
                interval = (pd.Timedelta(str(cfg['timeframe']).lower()) if 'timeframe' in cfg
                            else df.index[i] - df.index[i - 1] if i > 0 else pd.NaT)
                if pd.notna(interval) and interval > pd.Timedelta(0):
                    row['available_at'] = timestamp + interval
            except (ValueError, TypeError):
                pass  # Never invent a close timestamp when interval is unknown.

        raw_events = {}
        for k in event_keys:
            col = f'wyckoff_{k}'
            raw_events[k] = (bool(climax_inputs[k][i]) if k in climax_inputs
                             else bool(df[col].iat[i]) if col in df.columns else False)

        metadata = {}
        for k in delayed_keys:
            if raw_events.get(k, False):
                item = _delayed_evidence(df, cfg, k, i, parents)
                if item is not None:
                    metadata[k] = item
                    for field in ('candidate_index', 'candidate_extreme', 'prior_swept_boundary',
                                  'candidate_parent_id', 'candidate_parent_context', 'candidate_parent_status',
                                  'candidate_timestamp', 'confirmation_timestamp', 'available_at'):
                        value = getattr(item, field)
                        provenance[f'wyckoff_{k}_{field}'][i] = value.isoformat() if isinstance(value, pd.Timestamp) else value

        validated, conf_modifiers = sm.process_bar(i, row, raw_events, event_metadata=metadata)
        # The live feature adapter expects scalar cells (including its NaN
        # handler). JSON preserves multiple same-bar lifecycle records safely.
        provenance['wyckoff_opposing_climax_evidence'][i] = json.dumps(
            sm.climax_evidence, sort_keys=True, separators=(',', ':'), allow_nan=False)
        provenance['wyckoff_structure_evidence'][i] = json.dumps(
            sm.structural_evidence, sort_keys=True, separators=(',', ':'), allow_nan=False)
        provenance['wyckoff_parent_id'][i] = sm.parent_snapshot()['id']
        for key, confidence in sm.anchored_confidence.items():
            climax_confidence_overrides[key][i] = confidence
        for record in sm.climax_evidence:
            if record['status'] == 'confirmed':
                climax_confidence_overrides[record['event_type']][i] = record['confidence']
        for k in delayed_keys:
            if raw_events.get(k, False):
                item = metadata.get(k)
                provenance[f'wyckoff_{k}_evidence_status'][i] = (
                    'unavailable' if item is None else 'rejected' if not validated[k] else
                    'structured' if item.candidate_parent_status == 'established' else 'unstructured')

        for k in event_keys:
            validated_arrays[k][i] = validated[k]

        for k, mod in conf_modifiers.items():
            confidence_mod_arrays[k][i] = mod

        phases[i] = sm.get_phase()
        phase_dirs[i] = sm.get_phase_dir()
        contexts[i] = sm.get_context_str()

    # Write back validated events and zero out confidence for invalidated events
    for k in event_keys:
        col = f'wyckoff_{k}'
        conf_col = f'{col}_confidence'
        if col in df.columns and conf_col in df.columns:
            df[conf_col] = climax_inputs[k + '_confidence']
            # Where raw detected but SM rejected -> zero confidence
            detected = climax_inputs[k]
            rejected = detected.astype(bool) & ~validated_arrays[k]
            df.loc[rejected, conf_col] = 0.0
            df[col] = validated_arrays[k]
            if k in climax_confidence_overrides:
                override = climax_confidence_overrides[k]
                mask = np.isfinite(override)
                df.loc[mask, conf_col] = override[mask]

    # Apply confidence modifiers (e.g., 0.5x for no-context SOS/SOW)
    for k in event_keys:
        conf_col = f'wyckoff_{k}_confidence'
        if conf_col in df.columns:
            mod_mask = confidence_mod_arrays[k] < 1.0
            if mod_mask.any():
                df.loc[mod_mask, conf_col] = df.loc[mod_mask, conf_col] * confidence_mod_arrays[k][mod_mask]

    # Override phase with state-machine-derived phase
    df['wyckoff_phase_abc'] = phases
    df['wyckoff_phase_dir'] = phase_dirs
    df['wyckoff_context'] = contexts

    sm_counts = {k: validated_arrays[k].sum() for k in event_keys}
    logger.info(f"State machine validation: {sum(v for v in sm_counts.values())} events survived")
    for k, v in sm_counts.items():
        if v > 0:
            logger.info(f"  SM {k}: {v} validated")

    # Object dtype keeps absent provenance absent, rather than fabricated zeros.
    return pd.concat([df.drop(columns=list(provenance), errors='ignore'),
                      pd.DataFrame(provenance, index=df.index, dtype=object)], axis=1)


def _apply_htf_modulation(df: pd.DataFrame,
                          htf_context: WyckoffHTFContext) -> pd.DataFrame:
    """
    Apply confidence modulation based on higher-timeframe Wyckoff context.

    Events that ALIGN with the HTF direction get a confidence boost (+20%).
    Events that CONFLICT with the HTF direction get a confidence penalty (-30%).
    Modulation strength scales with HTF confidence (weak HTF signal = mild modulation).

    Args:
        df: DataFrame with wyckoff_*_confidence columns
        htf_context: Context from higher timeframe

    Returns:
        DataFrame with modulated confidence columns
    """
    if htf_context.phase == "none":
        return df

    # Determine which events to boost/penalize based on HTF direction
    if htf_context.dominant_direction == "bullish":
        boost_events = _ACCUM_EVENTS
        penalize_events = _DISTRIB_EVENTS
    elif htf_context.dominant_direction == "bearish":
        boost_events = _DISTRIB_EVENTS
        penalize_events = _ACCUM_EVENTS
    else:
        return df  # neutral = no modulation

    # Base modulation factors
    ALIGNED_BOOST = 1.20     # +20% confidence when aligned with HTF
    CONFLICT_PENALTY = 0.70  # -30% confidence when conflicting with HTF

    # Scale modulation by HTF strength (stronger HTF signal = stronger modulation)
    htf_strength = max(htf_context.bullish_score, htf_context.bearish_score)
    actual_boost = 1.0 + (ALIGNED_BOOST - 1.0) * htf_strength
    actual_penalty = 1.0 + (CONFLICT_PENALTY - 1.0) * htf_strength

    modulated_count = 0
    for e in boost_events:
        conf_col = f'wyckoff_{e}_confidence'
        if conf_col in df.columns:
            mask = df[conf_col] > 0
            if mask.any():
                df.loc[mask, conf_col] = (df.loc[mask, conf_col] * actual_boost).clip(0, 1)
                modulated_count += mask.sum()

    for e in penalize_events:
        conf_col = f'wyckoff_{e}_confidence'
        if conf_col in df.columns:
            mask = df[conf_col] > 0
            if mask.any():
                df.loc[mask, conf_col] = (df.loc[mask, conf_col] * actual_penalty).clip(0, 1)
                modulated_count += mask.sum()

    logger.info(f"HTF modulation applied ({htf_context.timeframe}→LTF): "
                f"boost={actual_boost:.3f}, penalty={actual_penalty:.3f}, "
                f"modulated {modulated_count} confidence values")

    return df


def _compute_directional_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute graded directional Wyckoff scores (bullish/bearish) from event confidences.

    Uses diversity-weighted scoring: score = mean_of_top2_confidences * diversity_bonus.
    A single event type (e.g., ST alone) caps at ~0.33. Multiple concurrent events
    (SC + AR + ST = genuine accumulation) can reach 1.0.

    This prevents high-frequency events like ST from dominating the score.

    Output columns:
        wyckoff_bullish_score: 0-1, diversity-weighted accumulation strength
        wyckoff_bearish_score: 0-1, diversity-weighted distribution strength
    """
    # Bullish score (accumulation events)
    accum_conf_cols = [f'wyckoff_{e}_confidence' for e in _ACCUM_EVENTS
                       if f'wyckoff_{e}_confidence' in df.columns]
    if accum_conf_cols:
        accum_df = df[accum_conf_cols].fillna(0.0)
        # Count how many distinct event types fire (confidence > 0)
        n_firing = (accum_df > 0).sum(axis=1)
        # Diversity bonus: 1 event = 0.33, 2 events = 0.67, 3+ events = 1.0
        diversity = (n_firing / 3.0).clip(upper=1.0)
        # Mean of top-2 confidences (robust to single noisy event)
        top2_mean = accum_df.apply(lambda row: row.nlargest(2).mean() if row.max() > 0 else 0.0, axis=1)
        df['wyckoff_bullish_score'] = (top2_mean * diversity).clip(0.0, 1.0)
    else:
        df['wyckoff_bullish_score'] = 0.0

    # Bearish score (distribution events)
    distrib_conf_cols = [f'wyckoff_{e}_confidence' for e in _DISTRIB_EVENTS
                         if f'wyckoff_{e}_confidence' in df.columns]
    if distrib_conf_cols:
        distrib_df = df[distrib_conf_cols].fillna(0.0)
        n_firing = (distrib_df > 0).sum(axis=1)
        diversity = (n_firing / 3.0).clip(upper=1.0)
        top2_mean = distrib_df.apply(lambda row: row.nlargest(2).mean() if row.max() > 0 else 0.0, axis=1)
        df['wyckoff_bearish_score'] = (top2_mean * diversity).clip(0.0, 1.0)
    else:
        df['wyckoff_bearish_score'] = 0.0

    return df


# ============================================================================
# COMPOSITE EVENT DETECTOR
# ============================================================================

def detect_all_wyckoff_events(df: pd.DataFrame, cfg: Optional[dict] = None,
                              htf_context: Optional[WyckoffHTFContext] = None) -> pd.DataFrame:
    """
    Detect all Wyckoff events and add columns to dataframe.

    This is the main entry point for Wyckoff event detection. It computes
    all 13 events and adds boolean + confidence columns to the dataframe.

    Hierarchical multi-timeframe usage:
        1. Run on 1D data (no context) → create_wyckoff_context(df_1d, timeframe="1D")
        2. Run on 4H data with 1D context → create_wyckoff_context(df_4h, timeframe="4H")
        3. Run on 1H data with 4H context → final detection with full HTF alignment

    When htf_context is provided, confidence scores are modulated:
        - Events aligned with HTF direction: boosted up to +20%
        - Events conflicting with HTF direction: penalized up to -30%
        - Modulation scales with HTF signal strength

    Args:
        df: OHLCV dataframe (must have volume, and ideally volume_z, liquidity_score)
        cfg: Optional configuration dict with event-specific thresholds
        htf_context: Optional higher-timeframe Wyckoff context for confidence modulation

    Returns:
        DataFrame with added Wyckoff event columns:
            - wyckoff_sc, wyckoff_sc_confidence (and 12 more event pairs)
            - wyckoff_bullish_score (graded 0-1, max accumulation confidence)
            - wyckoff_bearish_score (graded 0-1, max distribution confidence)
            - wyckoff_phase_abc (categorical: A/B/C/D/E/neutral)
            - wyckoff_sequence_position (int: 1-10 within phase)
    """
    cfg = cfg or {}

    # Ensure required base features exist
    if 'volume_z' not in df.columns and 'volume' in df.columns:
        df['volume_z'] = _rolling_z_score(df['volume'], window=20)

    htf_label = f" (HTF context: {htf_context.timeframe} {htf_context.dominant_direction})" if htf_context else ""
    logger.info(f"Detecting Wyckoff events on {len(df)} bars{htf_label}")

    # Phase A events
    df['wyckoff_sc'], df['wyckoff_sc_confidence'] = detect_selling_climax(df, cfg)
    df['wyckoff_bc'], df['wyckoff_bc_confidence'] = detect_buying_climax(df, cfg)
    df['wyckoff_ar'], df['wyckoff_ar_confidence'] = detect_automatic_rally(df, cfg, df['wyckoff_sc'])
    df['wyckoff_as'], df['wyckoff_as_confidence'] = detect_automatic_reaction(df, cfg, df['wyckoff_bc'])
    df['wyckoff_st'], df['wyckoff_st_confidence'] = detect_secondary_test(df, cfg, df['wyckoff_sc'])

    # Phase B events
    df['wyckoff_sos'], df['wyckoff_sos_confidence'] = detect_sign_of_strength(df, cfg)
    df['wyckoff_sow'], df['wyckoff_sow_confidence'] = detect_sign_of_weakness(df, cfg)

    # Phase C events
    df['wyckoff_spring_a'], df['wyckoff_spring_a_confidence'] = detect_spring_type_a(df, cfg)
    df['wyckoff_spring_b'], df['wyckoff_spring_b_confidence'] = detect_spring_type_b(df, cfg)
    df['wyckoff_ut'], df['wyckoff_ut_confidence'] = detect_upthrust(df, cfg)
    df['wyckoff_utad'], df['wyckoff_utad_confidence'] = detect_upthrust_after_distribution(df, cfg)

    # Phase D events
    df['wyckoff_lps'], df['wyckoff_lps_confidence'] = detect_last_point_of_support(df, cfg)
    df['wyckoff_lpsy'], df['wyckoff_lpsy_confidence'] = detect_last_point_of_supply(df, cfg)

    # A fresh detector run supersedes saved inputs from any earlier validation.
    # Validation-only replay uses these originals, including rejected retests.
    for k in _ACCUM_EVENTS + _DISTRIB_EVENTS:
        df[f'wyckoff_{k}_raw'] = df[f'wyckoff_{k}']
        df[f'wyckoff_{k}_raw_confidence'] = df[f'wyckoff_{k}_confidence']

    # ------------------------------------------------------------------
    # v2 SHADOW columns (2026-08-20) — data collection ONLY. Deliberately
    # NOT in _ACCUM_EVENTS/_DISTRIB_EVENTS: they cannot touch directional
    # scores, the state machine, phases, or any live consumer. Promotion to
    # decision inputs requires a passing validation study first.
    # ------------------------------------------------------------------
    if cfg.get('shadow_v2_enabled', True):
        (df['wyckoff_bc_v2'], df['wyckoff_bc_v2_confidence'],
         df['wyckoff_bc_v2_confirmed']) = detect_climax_v2(df, cfg, side='buying')
        (df['wyckoff_sc_v2'], df['wyckoff_sc_v2_confidence'],
         df['wyckoff_sc_v2_confirmed']) = detect_climax_v2(df, cfg, side='selling')

    # Determine phase and sequence (before SM so phase exists)
    df['wyckoff_phase_abc'] = _determine_wyckoff_phase(df)
    df['wyckoff_sequence_position'] = _compute_sequence_position(df)

    # State machine validation (sequential context filter)
    # Must run BEFORE HTF modulation so SM filters raw detections first
    if cfg.get('state_machine_enabled', True):
        df = _apply_state_machine_validation(df, cfg)
        # Recompute sequence position based on SM-derived phases
        df['wyckoff_sequence_position'] = _compute_sequence_position(df)
    else:
        # Directional phase/context only exist via the state machine; keep the
        # columns present (neutral) so consumers see a stable schema.
        df['wyckoff_phase_dir'] = 'neutral'
        df['wyckoff_context'] = 'none'

    # Anchored spring shadow (wyckoff_audit add.31: springs fire at fresh
    # 20-bar lows mid-trend, not at structural range lows — "the highest-
    # leverage fix: anchor to the 50-bar swing low; test, do NOT wire").
    # Computed AFTER state-machine validation so it is a strict subset of the
    # final wyckoff_spring_b column (the one every consumer sees). spring_b
    # stamps at the confirmation bar; its candidate bar sits recovery_bars
    # earlier, so all anchor inputs are shifted to align.
    if cfg.get('shadow_v2_enabled', True):
        recovery_bars = cfg.get('spring_b_recovery_bars', 3)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        prior_low50 = df['low'].shift(1).rolling(50).min()
        cand_low = df['low'].shift(recovery_bars)
        anchor_tol = cfg.get('spring_anchor_atr_tol', 1.0)
        anchored_ok = (
            (cand_low - prior_low50.shift(recovery_bars)).abs()
            <= anchor_tol * atr14.shift(recovery_bars)
        ).fillna(False)
        df['wyckoff_spring_b_anchored'] = (
            df['wyckoff_spring_b'].astype(bool) & anchored_ok
        )

    # Apply higher-timeframe confidence modulation (on SM-validated events only)
    if htf_context is not None:
        df = _apply_htf_modulation(df, htf_context)

    # Compute directional scores (after SM + HTF modulation)
    df = _compute_directional_scores(df)

    # Honest current-state gauge (2026-08-20 live audit): the legacy display
    # scores are rolling/window MAXIMA (24h carry, 42d/90d buffer max) that can
    # never fall while an old event sits in the buffer. wyckoff_event_now is
    # the per-bar max SM-validated event confidence — exactly 0 on bars where
    # no event is active. Display/logging only; no live consumer reads it.
    _v1_conf_cols = [f'wyckoff_{e}_confidence' for e in _ACCUM_EVENTS + _DISTRIB_EVENTS
                     if f'wyckoff_{e}_confidence' in df.columns]
    df['wyckoff_event_now'] = (
        df[_v1_conf_cols].fillna(0.0).max(axis=1) if _v1_conf_cols else 0.0
    )

    # Log summary
    event_counts = {
        'SC': df['wyckoff_sc'].sum(),
        'BC': df['wyckoff_bc'].sum(),
        'AR': df['wyckoff_ar'].sum(),
        'AS': df['wyckoff_as'].sum(),
        'ST': df['wyckoff_st'].sum(),
        'SOS': df['wyckoff_sos'].sum(),
        'SOW': df['wyckoff_sow'].sum(),
        'Spring_A': df['wyckoff_spring_a'].sum(),
        'Spring_B': df['wyckoff_spring_b'].sum(),
        'UT': df['wyckoff_ut'].sum(),
        'UTAD': df['wyckoff_utad'].sum(),
        'LPS': df['wyckoff_lps'].sum(),
        'LPSY': df['wyckoff_lpsy'].sum(),
    }

    logger.info("Wyckoff event detection complete:")
    for event, count in event_counts.items():
        if count > 0:
            logger.info(f"  {event}: {count} events detected")

    return df


def _determine_wyckoff_phase(df: pd.DataFrame) -> pd.Series:
    """
    Determine current Wyckoff phase based on recent events.

    Phase A: SC/BC/AR/AS/ST detected recently
    Phase B: SOS/SOW detected, no recent Spring/UT
    Phase C: Spring/UT detected recently
    Phase D: LPS/LPSY detected, trend beginning
    Phase E: Sustained trend continuation

    Args:
        df: DataFrame with event columns

    Returns:
        Series with phase labels
    """
    phase = pd.Series('neutral', index=df.index)

    # Phase A: Require SC/BC confidence > 0.3 in last 20 bars (not just any event > 0)
    # ST alone (the old over-firer) no longer dominates — need real climax events
    phase_a_conf = (
        df['wyckoff_sc_confidence'].rolling(20).max().fillna(0) +
        df['wyckoff_bc_confidence'].rolling(20).max().fillna(0)
    )
    # Also count AR/AS as supporting (but not sufficient alone)
    phase_a_support = (
        df['wyckoff_ar'].rolling(20).sum().fillna(0) +
        df['wyckoff_as'].rolling(20).sum().fillna(0) +
        df['wyckoff_st'].rolling(20).sum().fillna(0)
    )
    # Phase A requires a real climax (SC or BC with confidence > 0.3) OR
    # multiple supporting events (AR/AS/ST) indicating genuine Phase A structure
    phase.loc[(phase_a_conf > 0.3) | (phase_a_support >= 2)] = 'A'

    # Phase B: SOS/SOW with confidence > 0.3 in last 30 bars
    phase_b_conf = (
        df['wyckoff_sos_confidence'].rolling(30).max().fillna(0) +
        df['wyckoff_sow_confidence'].rolling(30).max().fillna(0)
    )
    phase.loc[(phase_b_conf > 0.3) & (phase == 'neutral')] = 'B'

    # Phase C: Springs/upthrusts override Phase B (these are rare, sensitivity is correct)
    phase_c_events = (
        df['wyckoff_spring_a'].rolling(10).sum() +
        df['wyckoff_spring_b'].rolling(10).sum() +
        df['wyckoff_ut'].rolling(10).sum() +
        df['wyckoff_utad'].rolling(10).sum()
    )
    phase.loc[phase_c_events > 0] = 'C'

    # Phase D indicators
    phase_d_events = (
        df['wyckoff_lps'].rolling(15).sum() +
        df['wyckoff_lpsy'].rolling(15).sum()
    )
    phase.loc[(phase_d_events > 0) & (phase != 'C')] = 'D'

    return phase


def _compute_sequence_position(df: pd.DataFrame) -> pd.Series:
    """
    Compute position in Wyckoff sequence (1-10).

    Maps events to sequence numbers:
    1-3: Phase A (SC/BC → AR/AS → ST)
    4-6: Phase B (SOS/SOW → Building → Testing)
    7-8: Phase C (Spring/UT → Confirmation)
    9-10: Phase D/E (LPS/LPSY → Trend)

    Args:
        df: DataFrame with phase column

    Returns:
        Series with sequence position
    """
    position = pd.Series(0, index=df.index)

    # Simplified mapping based on phase
    position.loc[df['wyckoff_phase_abc'] == 'A'] = 2
    position.loc[df['wyckoff_phase_abc'] == 'B'] = 5
    position.loc[df['wyckoff_phase_abc'] == 'C'] = 7
    position.loc[df['wyckoff_phase_abc'] == 'D'] = 9
    position.loc[df['wyckoff_phase_abc'] == 'E'] = 10

    return position


# ============================================================================
# PTI INTEGRATION
# ============================================================================

def integrate_wyckoff_with_pti(df: pd.DataFrame, pti_scores: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Integrate Wyckoff events with PTI (Psychology Trap Index).

    Key integrations:
        - Springs/UTAD often coincide with high PTI (retail trapped)
        - SC/BC events validate PTI bearish/bullish traps
        - LPS/LPSY can indicate PTI trap resolution

    Args:
        df: DataFrame with Wyckoff event columns
        pti_scores: Optional PTI scores (if not in df)

    Returns:
        DataFrame with integration columns:
            - wyckoff_pti_confluence (bool)
            - wyckoff_pti_score (float)
    """
    if pti_scores is not None:
        df['pti_score'] = pti_scores

    if 'pti_score' not in df.columns:
        logger.warning("PTI scores not available, skipping integration")
        df['wyckoff_pti_confluence'] = False
        df['wyckoff_pti_score'] = 0.0
        return df

    # High PTI + Spring/UT = strong trap confirmation
    trap_events = df['wyckoff_spring_a'] | df['wyckoff_spring_b'] | df['wyckoff_ut'] | df['wyckoff_utad']
    high_pti = df['pti_score'] > 0.6

    df['wyckoff_pti_confluence'] = trap_events & high_pti

    # Composite score (weighted)
    df['wyckoff_pti_score'] = (
        df['wyckoff_spring_a_confidence'] * 0.25 +
        df['wyckoff_spring_b_confidence'] * 0.20 +
        df['wyckoff_ut_confidence'] * 0.25 +
        df['wyckoff_utad_confidence'] * 0.30
    ) * df['pti_score']

    logger.info(f"PTI-Wyckoff confluence: {df['wyckoff_pti_confluence'].sum()} events")

    return df
