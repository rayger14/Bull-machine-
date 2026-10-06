"""Causal range and level-test evidence for the existing Wyckoff validator.

An operational recognition contract, not a complete schematic classifier or
economic certification. Raw price/volume units never replace legacy z-scores.
Direction-normalized geometry makes the accumulation/distribution rules mirrors.
"""
from dataclasses import dataclass
import json
import math

import pandas as pd


@dataclass(frozen=True)
class Candle:
    index: int
    high: float
    low: float
    close: float
    volume: object
    timestamp: pd.Timestamp
    available_at: pd.Timestamp

    @classmethod
    def read(cls, index, row, direction):
        try:
            o, h, l, c = (float(row[k]) for k in ('open','high','low','close'))
            t, a = row.get('timestamp'), row.get('available_at')
            if (not all(math.isfinite(x) and x > 0 for x in (o,h,l,c))
                    or not l <= min(o,c) <= max(o,c) <= h
                    or any(not isinstance(x,pd.Timestamp) or pd.isna(x) or x.tz is None for x in (t,a))
                    or a <= t):
                return None
            v = row.get('volume')
            v = float(v) if v is not None else None
            if v is not None and (not math.isfinite(v) or v <= 0): v = None
            return cls(index,h if direction == 1 else -l,l if direction == 1 else -h,
                       direction*c,v,t,a)
        except (ValueError,TypeError,KeyError,OverflowError):
            return None

    @property
    def spread(self):
        return self.high-self.low

    def record(self, direction):
        return dict(index=self.index, high=direction*(self.high if direction == 1 else self.low),
                    low=direction*(self.low if direction == 1 else self.high),
                    close=direction*self.close, volume=self.volume, spread=self.spread,
                    timestamp=self.timestamp.isoformat(), available_at=self.available_at.isoformat())


class RangeEvidence:
    def __init__(self, cfg, parent_id=0, context='none'):
        self.cfg, self.parent_id, self.context = cfg, parent_id, context
        self.direction = 1 if context == 'accumulation' else -1
        self.horizon = cfg.get('sm_ar_max_bars',15)
        self.climax = self.reaction = self.previous = None
        self.reaction_start = None
        self.bounds = dict(status='unavailable',bound_id=None)
        self.strength = self.escape = self.retest = None
        self.spring = None
        self.phase_test = dict(status='unsupported',role=None)
        self.range_test = None
        self.history = {}
        self.local_role = None
        self.unavailable = False
        self.emission = None
        self.observed_at = None

    def blocks_local_retest(self, parent_id, row, index):
        if parent_id != self.parent_id:
            return False
        clock_required = row.get('clock_required') or 'timestamp' in row or 'available_at' in row
        if self.unavailable and clock_required:
            return True
        if self.previous is not None:
            c = Candle.read(index,row,self.direction)
            if c is None or not self._contiguous(c):
                return True
        # Once an escape episode occurred, failed/expired evidence cannot fall
        # back to an unanchored local label in the same parent.
        if self.escape is not None:
            return True
        return (self.bounds.get('bound_id') is not None
                and self.direction*row['close'] > self.reaction.high)

    def _contiguous(self, c):
        return (c.index == self.previous.index+1 and c.timestamp == self.previous.available_at
                and c.available_at-c.timestamp == self.previous.available_at-self.previous.timestamp)

    def _near(self, price, level, tolerance):
        return abs(price-level) <= abs(level)*tolerance

    @staticmethod
    def _quieter(candle, reference):
        return (candle.volume is not None and reference.volume is not None
                and candle.volume < reference.volume and candle.spread < reference.spread)

    def _new_test(self, c, role, level, reference):
        return dict(status='pending',role=role,parent_id=self.parent_id,
                    bound_id=self.bounds['bound_id'],level=self.direction*level,
                    origin_index=reference.index,candidate_index=c.index,
                    candidate=c.record(self.direction),reference=reference.record(self.direction),
                    confirmed_index=None,confirmed_at=None,
                    expires_at=(c.available_at+self.horizon*(c.available_at-c.timestamp)).isoformat())

    def _advance_test(self, test, c):
        if test is None or test['status'] not in ('pending','confirmed'):
            return
        candidate = test['candidate']
        adverse = self.direction*candidate['low' if self.direction == 1 else 'high']
        recovery = self.direction*candidate['high' if self.direction == 1 else 'low']
        if c.index-test['candidate_index'] > self.horizon:
            test['status'] = 'expired'
        elif c.low < adverse:
            test['status'] = 'cancelled_failed_hold'
        elif c.index > test['candidate_index'] and c.close > recovery and test['status'] == 'pending':
            test.update(status='confirmed',confirmed_index=c.index,confirmed_at=c.available_at.isoformat())

    def _cancel_authority(self, reason):
        for test in (self.range_test,self.retest,self.phase_test):
            if test is not None and test['status'] in ('pending','confirmed'):
                test['status'] = reason
        if self.escape is not None:
            self.escape['status'] = reason
        self.spring = None

    def _reaction_step(self, c, events):
        key = 'ar' if self.direction == 1 else 'as'
        if self.reaction is None and events.get(key,False):
            self.reaction = c
            self.reaction_start = c.index
            self.bounds.update(status='developing',extreme_index=c.index,
                               extreme_at=c.timestamp.isoformat())
        elif self.reaction is not None and self.bounds['status'] == 'developing':
            if c.index-self.reaction_start > self.horizon:
                self.bounds['status'] = 'expired'
            elif c.high > self.reaction.high:
                self.reaction = c
                self.bounds.update(extreme_index=c.index,extreme_at=c.timestamp.isoformat())
            elif c.index > self.reaction.index and c.close < self.reaction.low:
                lower,upper = ((self.climax.low,self.reaction.high) if self.direction == 1
                               else (-self.reaction.high,-self.climax.low))
                if lower < upper:
                    self.bounds.update(status='reaction_complete',lower=lower,upper=upper,
                                       bound_id=f'{self.parent_id}:{c.index}',locked_index=c.index,
                                       locked_at=c.available_at.isoformat())

    def _range_test_step(self, c):
        self._advance_test(self.range_test,c)
        if self.range_test is not None and self.range_test['status'] == 'confirmed':
            self.bounds['status'] = 'tested'
        elif self.bounds['status'] == 'tested':
            self.bounds['status'] = 'reaction_complete'
        # One immutable pending test. A failed test can be followed by a new,
        # separately identified test; it cannot confirm on its failure bar.
        if self.range_test is None and c.low >= self.climax.low and self._near(
                c.low,self.climax.low,self.cfg.get('st_low_proximity',.03)) and self._quieter(c,self.climax):
            self.range_test = self._new_test(c,'climax_boundary_test',self.climax.low,self.climax)
        elif self.range_test is not None and self.range_test['status'].startswith(('cancelled','expired')):
            self.bounds['test'] = dict(self.range_test)
            self.range_test = None
        if self.range_test is not None:
            self.bounds['test'] = dict(self.range_test)

    def _escape_step(self, c, events, row):
        key = 'sos' if self.direction == 1 else 'sow'
        if events.get(key,False):
            confidence = row.get(key+'_confidence',0.)
            confidence = float(confidence) if confidence is not None else 0.
            self.strength = dict(role='parent_escape' if c.close > self.reaction.high else 'local',
                                 parent_id=self.parent_id,bound_id=self.bounds['bound_id'],
                                 candle=c.record(self.direction),confidence=confidence if math.isfinite(confidence) else 0.)
        crosses = (self.previous is not None and self.previous.close <= self.reaction.high
                   and c.close > self.reaction.high)
        if crosses:
            strength = self.strength
            qualified = (strength is not None
                         and 0 <= c.index-strength['candle']['index'] <= self.horizon)
            self.escape = dict(status='qualified' if qualified else 'price_escape_without_qualified_strength',
                               parent_id=self.parent_id,bound_id=self.bounds['bound_id'],
                               escape_index=c.index,available_at=c.available_at.isoformat(),
                               level=self.direction*self.reaction.high,
                               strength_index=strength['candle']['index'] if qualified else None,
                               strength=dict(strength) if qualified else None)
            self.retest = None
            self.phase_test['status'] = 'cancelled_parent_escape'
        if self.escape is None:
            return
        if c.close < self.reaction.high:
            self.escape['status'] = 'cancelled_boundary_loss'
            if self.retest is not None:
                self.retest['status'] = 'cancelled_boundary_loss'
        elif c.index-self.escape['escape_index'] > self.horizon:
            self.escape['status'] = 'expired'
            if self.retest is not None:
                self.retest['status'] = 'expired'
        if self.escape['status'] != 'qualified':
            return
        self._advance_test(self.retest,c)
        if self.retest is not None:
            if self.retest['status'] == 'confirmed' and self.retest['confirmed_index'] == c.index:
                self.emission = 'lps' if self.direction == 1 else 'lpsy'
            return  # Failed tests need a fresh escape, not a moving deadline.
        reference = self.escape['strength']['candle']
        strength = Candle.read(reference['index'],dict(reference,open=reference['close'],
                              timestamp=pd.Timestamp(reference['timestamp']),
                              available_at=pd.Timestamp(reference['available_at'])),self.direction)
        proximity = .03  # Same fixed 3% convention as the existing LPS/LPSY detectors.
        if (c.index > self.escape['escape_index'] and self.previous is not None
                and c.close < self.previous.close and c.close >= self.reaction.high
                and self._near(c.low,self.reaction.high,proximity)
                and strength is not None and self._quieter(c,strength)):
            self.retest = self._new_test(c,'post_escape',self.reaction.high,strength)
            self.retest['escape_index'] = self.escape['escape_index']
            self.retest['confidence'] = max(0.,min(1.,self.escape['strength']['confidence']))

    def _phase_step(self, c, events, metadata, phase):
        desired = 'C_accum' if self.direction == 1 else 'C_distrib'
        if phase != desired or self.escape is not None:
            if self.phase_test['status'] in ('pending','confirmed'):
                self.phase_test['status'] = 'cancelled_phase_advance'
            self.spring = None
            return
        for key in (('spring_a','spring_b') if self.direction == 1 else ('ut','utad')):
            if not events.get(key,False): continue
            evidence = metadata.get(key) if metadata is not None else None
            origin = self.history.get(evidence.candidate_index) if evidence is not None else c
            intact = (origin is not None and all(
                j in self.history and self.history[j].low >= origin.low
                for j in range(origin.index,c.index+1)))
            if (intact and origin.index > self.bounds['locked_index'] and origin.volume is not None):
                self.spring = origin
                self.phase_test = dict(status='awaiting_test',role='spring_test' if self.direction == 1 else 'upthrust_test',
                                       origin_index=origin.index,origin_confirmed_index=c.index)
        self._advance_test(self.phase_test,c)
        if self.spring is None or self.phase_test['status'] != 'awaiting_test':
            return
        if c.index-self.spring.index > self.horizon or c.low < self.spring.low:
            self.phase_test['status'] = 'expired' if c.index-self.spring.index > self.horizon else 'cancelled_failed_hold'
        elif (c.index > self.phase_test['origin_confirmed_index'] and c.low > self.spring.low
              and c.low >= self.climax.low
              and self._near(c.low,self.climax.low,self.cfg.get('st_low_proximity',.03))
              and self._quieter(c,self.spring)):
            self.phase_test = self._new_test(c,self.phase_test['role'],self.climax.low,self.spring)

    def consume(self, i, row, events, raw, metadata, phase, initial=None):
        self.emission = None
        self.local_role = 'local_unverified' if raw.get('lps' if self.direction == 1 else 'lpsy',False) else None
        c = Candle.read(i,row,self.direction)
        self.observed_at = c.available_at.isoformat() if c is not None else None
        if c is None or self.unavailable:
            self.unavailable = True
            self.bounds['status'] = 'unavailable'
            self._cancel_authority('cancelled_time_or_input_unavailable')
            return
        if self.climax is None:
            self.climax = Candle.read(initial[0],initial[1],self.direction) if initial else c
            if self.climax is None:
                self.unavailable = True
                return
            self.bounds['status'] = 'awaiting_reaction'
        elif self.previous is not None and not self._contiguous(c):
            self.unavailable = True
            self.bounds['status'] = 'unavailable'
            self._cancel_authority('cancelled_time_discontinuity')
            return
        self.history[i] = c
        # No subsequent lock can retroactively qualify this bar's strength/test.
        locked_before = self.bounds.get('bound_id') is not None
        self._reaction_step(c,events)
        if locked_before:
            self._range_test_step(c)
            self._escape_step(c,events,row)
            self._phase_step(c,events,metadata,phase)
        self.previous = c
        keep = max(self.horizon, *(self.cfg.get(f'{k}_recovery_bars',3) for k in ('spring_a','spring_b','ut')))+2
        self.history = {j:bar for j,bar in self.history.items() if j >= i-keep}

    def snapshot(self):
        return dict(schema='wyckoff-structure-v1',parent_id=self.parent_id,context=self.context,
                    observed_at=self.observed_at,range=self.bounds,strength=self.strength,
                    escape=self.escape,retest=self.retest,local_retest_role=self.local_role,
                    phase_justification=self.phase_test)


def phase_c_sizing_eligible(metadata):
    """Fail closed: a legacy phase string alone is never sizing authority."""
    if (metadata.get('wyckoff_phase_dir') != 'C_accum'
            or metadata.get('wyckoff_evidence_status') != 'available'):
        return False
    try:
        evidence = json.loads(metadata['wyckoff_structure_evidence'])
        bounds, test = evidence['range'], evidence['phase_justification']
        now = pd.Timestamp(metadata['wyckoff_available_at'])
        observed = pd.Timestamp(evidence['observed_at'])
        confirmed, expiry = pd.Timestamp(test['confirmed_at']), pd.Timestamp(test['expires_at'])
        times = (now,observed,confirmed,expiry)
        return bool(
            all(pd.notna(t) and t.tz is not None for t in times)
            and now == observed and confirmed <= now <= expiry
            and evidence['schema'] == 'wyckoff-structure-v1'
            and evidence['context'] == 'accumulation'
            and evidence['parent_id'] == metadata['wyckoff_parent_id'] == test['parent_id']
            and bounds['bound_id'] is not None and bounds['bound_id'] == test['bound_id']
            and bounds['status'] in ('reaction_complete','tested')
            and test['status'] == 'confirmed' and test['role'] == 'spring_test'
            and evidence['escape'] is None)
    except (KeyError,TypeError,ValueError,AttributeError,OverflowError):
        return False
