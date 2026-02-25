import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { SignalNarrative, OracleData } from '../../api/types';

interface LastSignalNarrativeProps {
  narrative: SignalNarrative | null | undefined;
  oracle?: OracleData;
}

export default function LastSignalNarrative({ narrative, oracle }: LastSignalNarrativeProps) {
  if (!narrative) return null;

  const regimeVariant = (r?: string) => {
    if (r === 'bull') return 'green';
    if (r === 'bear' || r === 'crisis') return 'red';
    return 'yellow';
  };

  // Determine if signal was accepted or rejected
  const fusionScore = narrative.fusion_score ?? 0;
  const threshold = narrative.threshold ?? 0;
  const isAccepted = fusionScore >= threshold && fusionScore > 0;

  // Derive factor strengths from confluence_factors and risk_factors
  const factors: Array<{ name: string; strength: number }> = [];
  if (narrative.confluence_factors) {
    for (const f of narrative.confluence_factors) {
      factors.push({ name: f, strength: 0.8 });
    }
  }
  if (narrative.risk_factors) {
    for (const f of narrative.risk_factors) {
      factors.push({ name: f, strength: 0.2 });
    }
  }

  return (
    <GlassCard>
      {/* Decision badge */}
      <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold mb-3 ${
        isAccepted
          ? 'bg-green-500/20 text-green-400 border border-green-500/30'
          : 'bg-red-500/20 text-red-400 border border-red-500/30'
      }`}>
        {isAccepted ? 'SIGNAL ACCEPTED' : 'SIGNAL REJECTED'}
      </div>

      <div className="flex items-center gap-2 mb-3">
        <span className="text-xs text-slate-500 uppercase tracking-wider">Last Signal</span>
        {narrative.regime && <Badge variant={regimeVariant(narrative.regime)}>{narrative.regime.replace('_', ' ')}</Badge>}
      </div>
      {narrative.headline && (
        <div className="text-sm font-semibold text-slate-200 mb-2">{narrative.headline}</div>
      )}
      {narrative.text && (
        <div className="text-xs text-slate-400 mb-3 leading-relaxed">{narrative.text}</div>
      )}
      <div className="flex gap-4 text-xs mb-3">
        <div>
          <span className="text-slate-600">Fusion:</span>{' '}
          <span className="text-cyan-400 font-mono">{narrative.fusion_score?.toFixed(3) ?? '--'}</span>
        </div>
        <div>
          <span className="text-slate-600">Threshold:</span>{' '}
          <span className="text-violet-400 font-mono">{narrative.threshold?.toFixed(3) ?? '--'}</span>
        </div>
        {narrative.fusion_score != null && narrative.threshold != null && (
          <div>
            <span className="text-slate-600">Margin:</span>{' '}
            <span className={`font-mono ${fusionScore >= threshold ? 'text-emerald-400' : 'text-rose-400'}`}>
              {(fusionScore - threshold) >= 0 ? '+' : ''}{(fusionScore - threshold).toFixed(3)}
            </span>
          </div>
        )}
      </div>

      {/* Factor strength breakdown */}
      {factors.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {factors.map((factor, i) => (
            <span key={i} className={`text-xs px-2 py-0.5 rounded ${
              factor.strength > 0.6 ? 'bg-green-500/20 text-green-300' :
              factor.strength > 0.3 ? 'bg-yellow-500/20 text-yellow-300' :
              'bg-gray-500/20 text-gray-400'
            }`}>
              {factor.name}
            </span>
          ))}
        </div>
      )}

      {/* Oracle context — engine posture */}
      {oracle?.engine_status?.posture_description && (
        <div className="text-[10px] text-slate-600 mt-2 italic">
          Engine: {oracle.engine_status.posture_description}
        </div>
      )}
    </GlassCard>
  );
}
