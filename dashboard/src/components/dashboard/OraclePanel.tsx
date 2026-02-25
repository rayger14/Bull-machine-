import { useState } from 'react';
import { ChevronDown, AlertTriangle, Zap, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { OracleData } from '../../api/types';

interface OraclePanelProps {
  oracle: OracleData | undefined;
}

const POSTURE_CONFIG: Record<string, { color: string; bg: string; border: string; badge: 'green' | 'yellow' | 'orange' | 'red' }> = {
  RISK_ON:   { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', badge: 'green' },
  CAUTIOUS:  { color: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   badge: 'yellow' },
  DEFENSIVE: { color: 'text-orange-400',  bg: 'bg-orange-500/10',  border: 'border-orange-500/30',  badge: 'orange' },
  CRISIS:    { color: 'text-rose-400',    bg: 'bg-rose-500/10',    border: 'border-rose-500/30',    badge: 'red' },
};

const BIAS_ICON = {
  bullish: <TrendingUp className="w-4 h-4 text-emerald-400" />,
  bearish: <TrendingDown className="w-4 h-4 text-rose-400" />,
  neutral: <Minus className="w-4 h-4 text-slate-400" />,
};

function confidenceBarColor(confidence: number): string {
  if (confidence >= 0.7) return 'bg-emerald-500';
  if (confidence >= 0.5) return 'bg-cyan-500';
  if (confidence >= 0.3) return 'bg-amber-500';
  return 'bg-slate-500';
}

function riskStatusColor(status: string): string {
  if (status === 'active') return 'text-rose-400';
  if (status === 'watching') return 'text-amber-400';
  return 'text-slate-500';
}

function riskDotColor(status: string): string {
  if (status === 'active') return 'bg-rose-400';
  if (status === 'watching') return 'bg-amber-400';
  return 'bg-slate-500';
}

export default function OraclePanel({ oracle }: OraclePanelProps) {
  const [expanded, setExpanded] = useState(false);

  // Placeholder state when oracle data is not yet available
  if (!oracle) {
    return (
      <GlassCard accent className="space-y-3">
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-slate-500 animate-pulse" />
          <span className="text-sm text-slate-500 italic">Oracle synthesizing...</span>
        </div>
      </GlassCard>
    );
  }

  const posture = POSTURE_CONFIG[oracle.posture] ?? POSTURE_CONFIG.CAUTIOUS;
  const confidence = oracle.confidence ?? 0;
  const biasIcon = BIAS_ICON[oracle.bias] ?? BIAS_ICON.neutral;

  return (
    <GlassCard accent className="space-y-4">
      {/* Top row: posture badge + confidence bar */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3">
          <Badge variant={posture.badge}>{oracle.posture.replace('_', ' ')}</Badge>
          {biasIcon}
          <span className="text-xs text-slate-500 capitalize">{oracle.bias} bias</span>
        </div>
        <div className="flex items-center gap-2 flex-1 min-w-[200px] max-w-[350px]">
          <span className="text-[10px] text-slate-600 uppercase tracking-wider whitespace-nowrap">Oracle Confidence</span>
          <div className="flex-1 h-2.5 bg-white/[0.04] rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ${confidenceBarColor(confidence)}`}
              style={{ width: `${Math.min(confidence * 100, 100)}%` }}
            />
          </div>
          <span className="text-xs font-mono text-slate-300 w-10 text-right">
            {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* One-liner */}
      {oracle.one_liner && (
        <p className="text-base text-slate-200 leading-relaxed font-medium">
          &ldquo;{oracle.one_liner}&rdquo;
        </p>
      )}

      {/* Expand/collapse toggle */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-cyan-400/70 hover:text-cyan-300 transition-colors"
      >
        <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
        {expanded ? 'Collapse' : 'Full Analysis'}
      </button>

      {/* Expanded section */}
      {expanded && (
        <div className="space-y-4 pt-1">
          {/* Thesis */}
          {oracle.thesis && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1.5">Thesis</div>
              <p className="text-xs text-slate-300 leading-relaxed">{oracle.thesis}</p>
            </div>
          )}

          {/* Aligned & Conflicting Factors */}
          {(oracle.aligned_factors?.length > 0 || oracle.conflicting_factors?.length > 0) && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-1.5">
              {oracle.aligned_factors?.map((factor, i) => (
                <div key={`a-${i}`} className="flex items-start gap-2 text-xs">
                  <span className="text-emerald-400 mt-0.5 shrink-0">&#10003;</span>
                  <span className="text-slate-300">{factor}</span>
                </div>
              ))}
              {oracle.conflicting_factors?.map((factor, i) => (
                <div key={`c-${i}`} className="flex items-start gap-2 text-xs">
                  <AlertTriangle className="w-3 h-3 text-amber-400 mt-0.5 shrink-0" />
                  <span className="text-slate-400">{factor}</span>
                </div>
              ))}
            </div>
          )}

          {/* Risks */}
          {oracle.risks?.length > 0 && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Risks</div>
              <div className="space-y-1.5">
                {oracle.risks.map((risk, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${riskDotColor(risk.status)}`} />
                    <span className={`font-medium ${riskStatusColor(risk.status)}`}>
                      {risk.name}
                    </span>
                    <span className="text-slate-600 font-mono">
                      ({(risk.probability * 100).toFixed(0)}%)
                    </span>
                    <span className="text-slate-500">&mdash; {risk.impact}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Catalysts */}
          {oracle.catalysts?.length > 0 && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">Catalysts</div>
              <div className="space-y-1.5">
                {oracle.catalysts.map((catalyst, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <Zap className="w-3 h-3 text-blue-400 mt-0.5 shrink-0" />
                    <span className="text-slate-300">{catalyst}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Outlook */}
          {oracle.outlook && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-2">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Outlook</div>
              {(['short_term', 'medium_term', 'long_term'] as const).map((tf) => {
                const o = oracle.outlook?.[tf];
                if (!o) return null;
                return (
                  <div key={tf} className="text-xs">
                    <span className="font-medium text-slate-300 uppercase">{o.label ?? tf.replace('_', ' ')}</span>
                    <span className="text-slate-600 font-mono ml-1.5">
                      ({(o.confidence * 100).toFixed(0)}%)
                    </span>
                    <span className="text-slate-400 ml-1.5">&mdash; {o.summary}</span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Engine Status */}
          {oracle.engine_status && (
            <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-1 text-xs text-slate-500">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Engine Status</div>
              {oracle.engine_status.posture_description && (
                <p className="text-slate-400">{oracle.engine_status.posture_description}</p>
              )}
              {oracle.engine_status.threshold_context && (
                <p>{oracle.engine_status.threshold_context}</p>
              )}
              {oracle.engine_status.recent_performance && (
                <p>{oracle.engine_status.recent_performance}</p>
              )}
            </div>
          )}
        </div>
      )}
    </GlassCard>
  );
}
