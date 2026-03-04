import { Activity, ShieldAlert, ShieldCheck, Gauge, Layers, Timer, Zap, Settings2 } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import { timeSince } from '../../utils/format';
import type { Heartbeat } from '../../api/types';

interface EngineHealthProps {
  hb: Heartbeat;
}

// ── Helpers ──────────────────────────────────────────────────────────

function thresholdBreakdown(hb: Heartbeat): {
  base: number;
  bearPenalty: number;
  instabPenalty: number;
  total: number;
} {
  const tc = hb.cmi_breakdown?.threshold_config;
  const riskTemp = hb.risk_temp ?? 0.5;
  const instability = hb.instability ?? 0;

  const base = tc?.base_threshold ?? hb.engine_health?.per_archetype_base_threshold
    ? 0.18  // default if per-arch in use
    : 0.18;
  const tempRange = tc?.temp_range ?? hb.engine_health?.temp_range ?? 0.38;
  const instabRange = tc?.instab_range ?? hb.engine_health?.instab_range ?? 0.15;

  const bearPenalty = (1 - riskTemp) * tempRange;
  const instabPenalty = instability * instabRange;
  const total = hb.threshold ?? (base + bearPenalty + instabPenalty);

  return { base, bearPenalty, instabPenalty, total };
}

function lastSignalInfo(hb: Heartbeat): { time: string; archetype: string } | null {
  // Try last_signal_narrative first
  const narrative = hb.last_signal_narrative;
  if (narrative) {
    // The narrative itself doesn't always have timestamp, check last_bar_signals
    const lastSignals = hb.last_bar_signals;
    if (lastSignals && lastSignals.length > 0) {
      const last = lastSignals[lastSignals.length - 1];
      return {
        time: last.timestamp ?? hb.timestamp ?? '',
        archetype: last.archetype ?? narrative.headline ?? 'unknown',
      };
    }
    return {
      time: hb.last_signal_time ?? hb.timestamp ?? '',
      archetype: narrative.headline ?? 'unknown',
    };
  }

  // Fall back to last_bar_signals
  const lastSignals = hb.last_bar_signals;
  if (lastSignals && lastSignals.length > 0) {
    const last = lastSignals[lastSignals.length - 1];
    return {
      time: last.timestamp ?? hb.timestamp ?? '',
      archetype: last.archetype ?? 'unknown',
    };
  }

  return null;
}

// ── Component ────────────────────────────────────────────────────────

export default function EngineHealth({ hb }: EngineHealthProps) {
  const eh = hb.engine_health;
  const bypassOn = eh?.bypass_threshold ?? false;
  const breakdown = thresholdBreakdown(hb);
  const disabledCount = eh?.disabled_archetypes?.length ?? 0;
  const calibratedCount = eh?.calibrated_archetypes?.length ?? 7;
  const baseMaxPos = eh?.base_max_positions ?? 3;
  const currentPos = hb.positions ?? 0;
  const leverage = hb.leverage ?? 1.0;
  const entrySpacing = eh?.entry_spacing_bars ?? 2;
  const gateMode = eh?.gate_mode ?? 'soft';
  const lastSig = lastSignalInfo(hb);
  const crisisCoeff = eh?.crisis_coefficient ?? 0.50;
  const emergencyThreshold = eh?.emergency_crisis_threshold ?? 0.7;
  const crisisProb = hb.crisis_prob ?? 0;

  // Emergency sizing active?
  const emergencyActive = crisisProb > emergencyThreshold;

  return (
    <GlassCard className="border-slate-500/10">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-500 uppercase tracking-wider">
            Engine Health
          </span>
        </div>
        {emergencyActive && (
          <Badge variant="red">EMERGENCY SIZING</Badge>
        )}
      </div>

      {/* Main grid: 2 rows of indicators */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {/* Bypass Mode */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            {bypassOn ? (
              <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            ) : (
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            )}
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Bypass</span>
          </div>
          <div className={`text-sm font-bold ${bypassOn ? 'text-rose-400' : 'text-emerald-400'}`}>
            {bypassOn ? 'ON' : 'OFF'}
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">
            {bypassOn ? 'No threshold filtering' : 'Threshold active'}
          </div>
        </div>

        {/* Active Threshold */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Gauge className="w-3.5 h-3.5 text-violet-400" />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Threshold</span>
          </div>
          <div className="text-sm font-bold font-mono text-slate-200">
            {breakdown.total.toFixed(3)}
          </div>
          <div className="text-[10px] text-slate-700 font-mono mt-0.5">
            {breakdown.base.toFixed(2)} + {breakdown.bearPenalty.toFixed(2)} + {breakdown.instabPenalty.toFixed(2)}
          </div>
        </div>

        {/* Positions */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Layers className="w-3.5 h-3.5 text-cyan-400" />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Positions</span>
          </div>
          <div className="text-sm font-bold font-mono text-slate-200">
            {currentPos} / {baseMaxPos}
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">
            {leverage}x leverage
          </div>
        </div>

        {/* Archetypes */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Settings2 className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Archetypes</span>
          </div>
          <div className="text-sm font-bold font-mono text-slate-200">
            {calibratedCount} active
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">
            {disabledCount > 0
              ? `${disabledCount} disabled`
              : 'None disabled'}
          </div>
        </div>
      </div>

      {/* Bottom row: secondary indicators */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-3">
        {/* Last Signal */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Zap className="w-3.5 h-3.5 text-yellow-400" />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Last Signal</span>
          </div>
          {lastSig ? (
            <>
              <div className="text-xs font-medium text-slate-300 truncate">
                {lastSig.archetype.replace(/_/g, ' ')}
              </div>
              <div className="text-[10px] text-slate-600 mt-0.5">
                {timeSince(lastSig.time)}
              </div>
            </>
          ) : (
            <div className="text-xs text-slate-600 italic">No signals yet</div>
          )}
        </div>

        {/* Gate Mode */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Gate Mode</span>
          </div>
          <div className="text-sm font-bold text-slate-200 capitalize">
            {gateMode}
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">
            Per-archetype gates
          </div>
        </div>

        {/* Entry Spacing */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Timer className="w-3.5 h-3.5 text-teal-400" />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Entry Spacing</span>
          </div>
          <div className="text-sm font-bold font-mono text-slate-200">
            {entrySpacing} bars
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">
            Same-direction min gap
          </div>
        </div>

        {/* Crisis Coefficient */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.06] p-3">
          <div className="flex items-center gap-1.5 mb-1.5">
            <ShieldAlert className={`w-3.5 h-3.5 ${emergencyActive ? 'text-rose-400' : 'text-slate-500'}`} />
            <span className="text-[10px] text-slate-600 uppercase tracking-wider">Crisis Guard</span>
          </div>
          <div className={`text-sm font-bold font-mono ${emergencyActive ? 'text-rose-400' : 'text-slate-200'}`}>
            {(crisisCoeff * 100).toFixed(0)}% coeff
          </div>
          <div className="text-[10px] text-slate-700 mt-0.5">
            {emergencyActive
              ? `Emergency: ${((eh?.emergency_size_multiplier ?? 0.5) * 100).toFixed(0)}% sizing`
              : `Triggers at ${(emergencyThreshold * 100).toFixed(0)}% crisis`}
          </div>
        </div>
      </div>
    </GlassCard>
  );
}
