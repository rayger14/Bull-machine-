import { useState } from 'react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import FilterPills from '../ui/FilterPills';
import { timeSince } from '../../utils/format';
import type { Signal } from '../../api/types';

interface SignalExplorerProps {
  signals: Signal[];
}

const filterOptions = [
  { value: 'all', label: 'All' },
  { value: 'allocated', label: 'Allocated' },
  { value: 'passed', label: 'Passed' },
  { value: 'rejected', label: 'Rejected' },
];

export default function SignalExplorer({ signals }: SignalExplorerProps) {
  const [filter, setFilter] = useState('all');
  const [expandedId, setExpandedId] = useState<number | null>(null);

  const filtered = filter === 'all' ? signals : signals.filter(s => s.status === filter);

  const statusVariant = (s?: string): 'green' | 'cyan' | 'red' | 'neutral' => {
    if (s === 'allocated') return 'green';
    if (s === 'passed') return 'cyan';
    if (s === 'rejected') return 'red';
    return 'neutral';
  };

  return (
    <GlassCard>
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Signal Log ({filtered.length})</span>
        <FilterPills options={filterOptions} selected={filter} onChange={setFilter} />
      </div>
      <div className="max-h-[500px] overflow-y-auto space-y-1">
        {filtered.slice(0, 100).map((sig, i) => (
          <div key={i}>
            <button
              className="w-full flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-white/[0.03] transition-colors text-left"
              onClick={() => setExpandedId(expandedId === i ? null : i)}
            >
              <Badge variant={statusVariant(sig.status)}>{sig.status}</Badge>
              <Badge variant="cyan">{sig.archetype?.replace(/_/g, ' ')}</Badge>
              <span className="text-xs text-slate-500 font-mono ml-auto">
                F:{sig.fusion_score?.toFixed(3)} T:{sig.threshold?.toFixed(3)}
              </span>
              <span className="text-[10px] text-slate-600">{timeSince(sig.timestamp)}</span>
            </button>
            {expandedId === i && (
              <div className="px-3 pb-3 text-xs space-y-2 bg-white/[0.01] rounded-b-lg">
                {sig.rejection_stage && (
                  <div className="text-rose-400">
                    Rejected at: <span className="font-mono">{sig.rejection_stage}</span>
                    {sig.rejection_reason && ` \u2014 ${sig.rejection_reason}`}
                  </div>
                )}
                {sig.regime && (
                  <div className="text-slate-400">Regime: <span className="text-slate-300">{sig.regime}</span></div>
                )}
                {sig.gate_values && Object.keys(sig.gate_values).length > 0 && (
                  <div>
                    <div className="text-slate-600 text-[10px] uppercase mb-1">Gate Values</div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                      {Object.entries(sig.gate_values).map(([k, v]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-slate-500">{k.replace(/_/g, ' ')}</span>
                          <span className="text-slate-300 font-mono">{String(v)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {sig.narrative?.confluence_factors && (
                  <div className="flex flex-wrap gap-1">
                    {sig.narrative.confluence_factors.map((f, j) => (
                      <Badge key={j} variant="cyan">{f}</Badge>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {filtered.length === 0 && (
          <div className="text-center text-slate-600 py-8">No signals</div>
        )}
      </div>
    </GlassCard>
  );
}
