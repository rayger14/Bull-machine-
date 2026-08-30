import GlassCard from '../components/ui/GlassCard';
import Badge from '../components/ui/Badge';
import ArchetypeCards from '../components/strategy/ArchetypeCards';
import { BOOK_COMPARISON, PARITY_METHOD, PARITY_DATE } from '../data/parityResults';

export default function StrategyPage() {
  return (
    <div className="space-y-4">
      {/* V23 parity-store research verdicts */}
      <GlassCard>
        <div className="flex items-center gap-2 mb-3">
          <div className="text-xs text-slate-500 uppercase tracking-wider">Honest Research Baseline</div>
          <Badge variant="cyan">V23 parity store · {PARITY_DATE}</Badge>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead><tr className="text-slate-600 text-left">
              <th className="py-1 pr-3">Book configuration</th><th className="text-right pr-3">PnL (5yr)</th>
              <th className="text-right pr-3">PF</th><th className="text-right pr-3">MaxDD</th><th className="text-right pr-3">Sharpe</th>
            </tr></thead>
            <tbody>
              {BOOK_COMPARISON.map((r) => (
                <tr key={r.label} className="border-t border-white/[0.04]">
                  <td className="py-1.5 pr-3 text-slate-300">{r.label}
                    <div className="text-[10px] text-slate-600">{r.note}</div></td>
                  <td className="text-right pr-3 font-mono text-emerald-400">${'{'}(r.pnl/1000).toFixed(0){'}'}K</td>
                  <td className="text-right pr-3 font-mono">{Number.isFinite(r.pf) ? r.pf.toFixed(2) : '—'}</td>
                  <td className="text-right pr-3 font-mono text-rose-400">{Number.isFinite(r.maxDD) ? r.maxDD.toFixed(1)+'%' : '—'}</td>
                  <td className="text-right pr-3 font-mono">{Number.isFinite(r.sharpe) ? r.sharpe.toFixed(2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[10px] text-slate-600 mt-2 leading-relaxed">{PARITY_METHOD}</p>
      </GlassCard>

      {/* System overview */}
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">System Architecture</div>
        <div className="text-sm text-slate-300 leading-relaxed space-y-3">
          <p>
            Bull Machine is a systematic BTC trading engine built on <Badge variant="cyan">archetype fusion scoring</Badge> —
            17 structural patterns (archetypes) each compute a fusion score from 4 weighted factors:
            Wyckoff accumulation/distribution, liquidity dynamics, momentum indicators, and Smart Money Concepts (SMC).
          </p>
          <p>
            Each archetype has hard <Badge variant="violet">gate conditions</Badge> that must pass before a signal is generated.
            If gates pass, the fusion score must exceed a <Badge variant="violet">dynamic threshold</Badge> set by the
            CMI (Contextual Market Intelligence) regime system.
          </p>
          <p>
            The CMI system is <strong className="text-slate-200">orthogonal</strong> to archetype fusion — it uses only macro/trend data
            (risk temperature, instability, crisis probability) to modulate the threshold, avoiding double-counting with the
            Wyckoff/SMC/temporal features already in archetype scoring.
          </p>
        </div>
      </GlassCard>

      {/* CMI Threshold */}
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">CMI v0 — Dynamic Threshold</div>
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-4 font-mono text-sm text-cyan-400 mb-3">
          threshold = base + (1 - risk_temp) × temp_range + instability × instab_range
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-slate-500 font-medium mb-2">Risk Temperature (45% trend + 25% ADX + 15% F&G + 15% DD)</div>
            <div className="space-y-1 text-slate-400">
              <div>0.0 → Deep Bear: threshold ~0.66 (very selective)</div>
              <div>0.5 → Neutral: threshold ~0.42</div>
              <div>1.0 → Strong Bull: threshold ~0.18 (permissive)</div>
            </div>
          </div>
          <div>
            <div className="text-slate-500 font-medium mb-2">Instability (35% chop + 25% ADX weakness + 20% wick + 20% volume)</div>
            <div className="space-y-1 text-slate-400">
              <div>0.0 → Stable: +0.00 to threshold</div>
              <div>0.5 → Moderate: +0.10 to threshold</div>
              <div>1.0 → Extreme: +0.20 to threshold</div>
            </div>
          </div>
          <div>
            <div className="text-slate-500 font-medium mb-2">Crisis Probability (60% stress + 20% vol shock + 20% sentiment)</div>
            <div className="space-y-1 text-slate-400">
              <div>Penalizes fusion score directly</div>
              <div>&gt; 0.7 → 50% emergency sizing cap</div>
            </div>
          </div>
          <div>
            <div className="text-slate-500 font-medium mb-2">Config (v17 Production)</div>
            <div className="space-y-1 text-slate-400">
              <div>base_threshold = 0.18</div>
              <div>temp_range = 0.38</div>
              <div>instab_range = 0.15</div>
              <div>crisis_coeff = 0.50</div>
            </div>
          </div>
        </div>
      </GlassCard>

      {/* Risk Management */}
      <GlassCard>
        <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">Risk Management</div>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs text-slate-400">
          <div>
            <div className="text-slate-500 font-medium mb-2">Position Sizing</div>
            <div>Max 3 concurrent positions (stress-scaled)</div>
            <div>Isolated margin per position</div>
            <div>1.5x leverage (1/8 Kelly)</div>
          </div>
          <div>
            <div className="text-slate-500 font-medium mb-2">Smart Exits V2</div>
            <div>Composite invalidation (4/5 features)</div>
            <div>Distress half-exit (50% underwater)</div>
            <div>Chop-aware trailing stops</div>
            <div>R-multiple scale-out targets</div>
          </div>
          <div>
            <div className="text-slate-500 font-medium mb-2">Cost Model</div>
            <div>Commission: 2 bps (Coinbase)</div>
            <div>Slippage: 3 bps</div>
            <div>Funding costs tracked</div>
          </div>
        </div>
      </GlassCard>

      {/* Archetype cards */}
      <ArchetypeCards />
    </div>
  );
}
