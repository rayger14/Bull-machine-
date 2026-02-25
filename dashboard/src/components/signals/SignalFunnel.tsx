import { useMemo } from 'react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { Signal } from '../../api/types';

interface SignalFunnelProps {
  signals: Signal[];
}

export default function SignalFunnel({ signals }: SignalFunnelProps) {
  const stats = useMemo(() => {
    if (!signals.length) return null;
    const rejected = signals.filter(s => s.status === 'rejected');
    const passed = signals.filter(s => s.status === 'passed');
    const allocated = signals.filter(s => s.status === 'allocated');

    const byStage: Record<string, number> = {};
    rejected.forEach(s => {
      const stage = s.rejection_stage || 'unknown';
      byStage[stage] = (byStage[stage] || 0) + 1;
    });
    const stages = Object.entries(byStage)
      .map(([stage, count]) => ({ stage, count, pct: (count / signals.length * 100) }))
      .sort((a, b) => b.count - a.count);

    const byArch: Record<string, { total: number; rejected: number; allocated: number }> = {};
    signals.forEach(s => {
      const a = s.archetype || 'unknown';
      if (!byArch[a]) byArch[a] = { total: 0, rejected: 0, allocated: 0 };
      byArch[a].total++;
      if (s.status === 'rejected') byArch[a].rejected++;
      if (s.status === 'allocated') byArch[a].allocated++;
    });
    const archs = Object.entries(byArch)
      .map(([name, a]) => ({ name, ...a, convRate: (a.allocated / a.total * 100) }))
      .sort((a, b) => b.total - a.total);

    return {
      total: signals.length,
      rejected: rejected.length,
      passed: passed.length,
      allocated: allocated.length,
      rejectionRate: (rejected.length / signals.length * 100),
      conversionRate: (allocated.length / signals.length * 100),
      stages,
      archs,
    };
  }, [signals]);

  if (!stats) return null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Funnel summary + rejection bars */}
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Signal Funnel</div>
        <div className="flex gap-4 mb-4 text-xs">
          <div className="text-center">
            <div className="text-lg font-bold text-slate-200">{stats.total}</div>
            <div className="text-slate-600">Generated</div>
          </div>
          <div className="text-center text-slate-600 self-center">&rarr;</div>
          <div className="text-center">
            <div className="text-lg font-bold text-amber-400">{stats.passed}</div>
            <div className="text-slate-600">Passed</div>
          </div>
          <div className="text-center text-slate-600 self-center">&rarr;</div>
          <div className="text-center">
            <div className="text-lg font-bold text-emerald-400">{stats.allocated}</div>
            <div className="text-slate-600">Allocated</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-bold text-rose-400">{stats.rejected}</div>
            <div className="text-slate-600">Rejected</div>
          </div>
        </div>
        <div className="text-[10px] text-slate-600 uppercase mb-2">Rejection Breakdown</div>
        <div className="space-y-1.5">
          {stats.stages.map((s) => (
            <div key={s.stage} className="flex items-center gap-2 text-xs">
              <span className="text-slate-400 w-32 truncate">{s.stage.replace(/_/g, ' ')}</span>
              <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-rose-500/60 rounded-full transition-all"
                  style={{ width: `${Math.min(s.pct * 2, 100)}%` }}
                />
              </div>
              <span className="text-slate-500 w-16 text-right font-mono">{s.count} ({s.pct.toFixed(1)}%)</span>
            </div>
          ))}
        </div>
      </GlassCard>

      {/* Conversion by archetype */}
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Conversion by Archetype</div>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-600 border-b border-white/[0.05]">
              <th className="text-left py-1.5 font-medium">Archetype</th>
              <th className="text-right py-1.5 font-medium">Total</th>
              <th className="text-right py-1.5 font-medium">Allocated</th>
              <th className="text-right py-1.5 font-medium">Conv %</th>
            </tr>
          </thead>
          <tbody>
            {stats.archs.map((a) => (
              <tr key={a.name} className="border-b border-white/[0.03] hover:bg-white/[0.02]">
                <td className="py-1.5">
                  <Badge variant="cyan">{a.name.replace(/_/g, ' ')}</Badge>
                </td>
                <td className="py-1.5 text-right font-mono text-slate-400">{a.total}</td>
                <td className="py-1.5 text-right font-mono text-emerald-400">{a.allocated}</td>
                <td className="py-1.5 text-right font-mono text-slate-300">{a.convRate.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </GlassCard>
    </div>
  );
}
