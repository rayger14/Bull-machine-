import { useState } from 'react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import ProgressBar from '../ui/ProgressBar';
import { ARCHETYPES } from '../../data/archetypes';

export default function ArchetypeCards() {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Archetypes (17)</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {Object.entries(ARCHETYPES).map(([key, arch]) => (
          <GlassCard
            key={key}
            padding="p-4"
            className="cursor-pointer"
            onClick={() => setExpanded(expanded === key ? null : key)}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-slate-200">{arch.name}</span>
                {arch.calibrated && <Badge variant="green">Calibrated</Badge>}
                {!arch.calibrated && <Badge variant="orange">Uncalibrated</Badge>}
                {!arch.proven && arch.pf && parseFloat(arch.pf) < 1 && <Badge variant="red">Disabled</Badge>}
              </div>
              <div className="flex items-center gap-2">
                {arch.pf && (
                  <span className={`text-sm font-mono font-bold ${parseFloat(arch.pf) >= 1 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    PF {arch.pf}
                  </span>
                )}
                <Badge variant={arch.dir === 'long' ? 'green' : arch.dir === 'short' ? 'red' : 'yellow'}>
                  {arch.dir}
                </Badge>
              </div>
            </div>
            <div className="text-xs text-slate-400 mb-2">{arch.desc}</div>
            {arch.trades && <div className="text-[10px] text-slate-600 mb-2">{arch.trades}</div>}

            {/* Fusion weights bar */}
            <ProgressBar
              segments={[
                { label: 'Wyckoff', value: arch.weights.wyckoff, color: '#8b5cf6' },
                { label: 'Liquidity', value: arch.weights.liquidity, color: '#3b82f6' },
                { label: 'Momentum', value: arch.weights.momentum, color: '#fb923c' },
                { label: 'SMC', value: arch.weights.smc, color: '#34d399' },
              ]}
              height="h-2"
            />

            {/* Expandable details */}
            {expanded === key && (
              <div className="mt-3 pt-3 border-t border-white/[0.05] space-y-3">
                {!arch.calibrated && (
                  <div className="p-2 bg-orange-500/[0.06] border border-orange-500/20 rounded-lg">
                    <div className="text-[10px] text-orange-400 font-medium mb-0.5">Observation Mode</div>
                    <div className="text-[10px] text-orange-300/70">
                      This archetype uses the default 0.18 threshold and has not been calibrated with per-archetype optimization.
                      Signals from this archetype are for observation only — their performance has not been validated out-of-sample.
                    </div>
                  </div>
                )}
                {arch.explanation && (
                  <div>
                    <div className="text-[10px] text-cyan-500/60 uppercase mb-1">What It Detects</div>
                    <div className="text-xs text-slate-300">{arch.explanation}</div>
                  </div>
                )}
                {arch.whyItWorks && (
                  <div>
                    <div className="text-[10px] text-violet-400/60 uppercase mb-1">Why It Works</div>
                    <div className="text-xs text-slate-400 italic">{arch.whyItWorks}</div>
                  </div>
                )}
                {arch.gates && arch.gates.length > 0 && (
                  <div>
                    <div className="text-[10px] text-slate-600 uppercase mb-1">Gate Conditions</div>
                    <div className="space-y-0.5">
                      {arch.gates.map((g, i) => (
                        <div key={i} className="text-xs text-slate-400 font-mono">
                          <span className="text-cyan-500/60 mr-1">&#9679;</span> {g}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </GlassCard>
        ))}
      </div>
    </div>
  );
}
