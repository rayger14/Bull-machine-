import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { CMIComparison as CMIComparisonData, CMIBreakdown, FusionBucket } from '../../api/types';

interface CMIComparisonProps {
  comparison?: CMIComparisonData | null;
  breakdown?: CMIBreakdown;
  fusionBuckets?: Record<string, FusionBucket>;
}

function weightBar(label: string, handTuned: number, optimized: number | undefined) {
  const maxVal = Math.max(handTuned, optimized ?? 0, 0.01);
  const htPct = (handTuned / maxVal) * 100;
  const optPct = optimized != null ? (optimized / maxVal) * 100 : 0;
  const diff = optimized != null ? ((optimized - handTuned) / handTuned * 100) : 0;

  return (
    <div className="flex items-center gap-2 text-[11px] mb-1.5">
      <span className="w-24 text-slate-500 text-right truncate">{label}</span>
      <div className="flex-1 flex flex-col gap-0.5">
        <div className="flex items-center gap-1">
          <div className="h-2.5 rounded-sm bg-blue-500/40" style={{ width: `${htPct}%`, minWidth: 2 }} />
          <span className="text-slate-400 font-mono w-8 text-right">{(handTuned * 100).toFixed(0)}%</span>
        </div>
        {optimized != null && (
          <div className="flex items-center gap-1">
            <div className="h-2.5 rounded-sm bg-emerald-500/40" style={{ width: `${optPct}%`, minWidth: 2 }} />
            <span className="text-emerald-400 font-mono w-8 text-right">{(optimized * 100).toFixed(0)}%</span>
            {Math.abs(diff) > 5 && (
              <span className={`text-[9px] ${diff > 0 ? 'text-emerald-500' : 'text-amber-500'}`}>
                {diff > 0 ? '+' : ''}{diff.toFixed(0)}%
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function thresholdGauge(label: string, value: number, color: string) {
  const pct = Math.min(value * 100, 100);
  return (
    <div className="text-center">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-sm font-bold font-mono ${color}`}>{value.toFixed(3)}</div>
      <div className="w-full h-1.5 bg-white/[0.04] rounded-full mt-1">
        <div className={`h-full rounded-full ${color === 'text-blue-400' ? 'bg-blue-500/50' : 'bg-emerald-500/50'}`}
          style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function fusionBucketChart(buckets: Record<string, FusionBucket>) {
  const keys = Object.keys(buckets).sort((a, b) => parseFloat(a) - parseFloat(b));
  if (keys.length === 0) return null;

  const maxTotal = Math.max(...keys.map(k => buckets[k].wins + buckets[k].losses), 1);

  return (
    <div className="mt-4">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">
        Fusion Score Clusters (Win Rate by Bucket)
      </div>
      <div className="flex items-end gap-0.5 h-20">
        {keys.map(k => {
          const b = buckets[k];
          const total = b.wins + b.losses;
          if (total === 0) return null;
          const wr = b.wins / total;
          const height = (total / maxTotal) * 100;
          const greenH = wr * 100;

          return (
            <div key={k} className="flex-1 flex flex-col items-center gap-0.5" title={`Fusion ${k}: ${b.wins}W/${b.losses}L (${(wr * 100).toFixed(0)}% WR) PnL: $${(b.total_pnl ?? 0).toFixed(0)}`}>
              <div className="w-full relative rounded-t-sm overflow-hidden" style={{ height: `${height}%`, minHeight: 4 }}>
                <div className="absolute bottom-0 w-full bg-emerald-500/50" style={{ height: `${greenH}%` }} />
                <div className="absolute top-0 w-full bg-red-500/30" style={{ height: `${100 - greenH}%` }} />
              </div>
              <span className="text-[8px] text-slate-600 font-mono">{parseFloat(k).toFixed(1)}</span>
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-[8px] text-slate-600 mt-1">
        <span>Low fusion</span>
        <span>High fusion</span>
      </div>
      <div className="flex items-center gap-3 mt-1 text-[9px] text-slate-500">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-emerald-500/50" /> Wins</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-sm bg-red-500/30" /> Losses</span>
      </div>
    </div>
  );
}

export default function CMIComparison({ comparison, breakdown, fusionBuckets }: CMIComparisonProps) {
  const hasOptimized = comparison != null;
  const weights = breakdown;

  // Always show current weights even without optimized
  if (!weights) {
    return (
      <GlassCard>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs text-slate-500 uppercase tracking-wider">CMI Weight Analysis</span>
          <Badge variant="neutral">waiting</Badge>
        </div>
        <p className="text-xs text-slate-600 italic">
          Run bin/optimize_cmi_weights.py to generate data-driven weights for comparison.
        </p>
      </GlassCard>
    );
  }

  const rtW = weights.risk_temp_weights ?? {};
  const instW = weights.instability_weights ?? {};
  const crW = weights.crisis_weights ?? {};

  return (
    <GlassCard>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 uppercase tracking-wider">CMI Weight Comparison</span>
          {hasOptimized ? (
            comparison.agreement
              ? <Badge variant="green">agree</Badge>
              : <Badge variant="yellow">diverge ({comparison.delta > 0 ? '+' : ''}{comparison.delta.toFixed(3)})</Badge>
          ) : (
            <Badge variant="neutral">baseline only</Badge>
          )}
        </div>
        {hasOptimized && (
          <div className="flex items-center gap-1 text-[9px]">
            <span className="w-2 h-2 rounded-sm bg-blue-500/40" /> Hand-tuned
            <span className="w-2 h-2 rounded-sm bg-emerald-500/40 ml-2" /> Optimized
          </div>
        )}
      </div>

      {/* Threshold Comparison */}
      {hasOptimized && (
        <div className="grid grid-cols-2 gap-3 mb-4 bg-white/[0.02] rounded-lg p-3 border border-white/[0.04]">
          {thresholdGauge('Hand-Tuned Threshold', comparison.hand_tuned.threshold, 'text-blue-400')}
          {thresholdGauge('Optimized Threshold', comparison.optimized.threshold, 'text-emerald-400')}
        </div>
      )}

      {/* Weight Bars by Group */}
      <div className="space-y-3">
        {/* Risk Temperature */}
        <div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Risk Temperature Weights</div>
          {weightBar('Trend Align', rtW['trend_align'] ?? 0.45, undefined)}
          {weightBar('ADX Strength', rtW['trend_strength'] ?? 0.25, undefined)}
          {weightBar('Sentiment', rtW['sentiment'] ?? 0.15, undefined)}
          {weightBar('DD Recovery', rtW['dd_score'] ?? 0.15, undefined)}
        </div>

        {/* Instability */}
        <div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Instability Weights</div>
          {weightBar('Chop Score', instW['chop'] ?? 0.35, undefined)}
          {weightBar('ADX Weakness', instW['adx_weakness'] ?? 0.25, undefined)}
          {weightBar('Wick Score', instW['wick_score'] ?? 0.20, undefined)}
          {weightBar('Vol Instab', instW['vol_instab'] ?? instW['vol_instability'] ?? 0.20, undefined)}
        </div>

        {/* Crisis */}
        <div>
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Crisis Probability Weights</div>
          {weightBar('Base Crisis', crW['base_crisis'] ?? 0.60, undefined)}
          {weightBar('Vol Shock', crW['vol_shock'] ?? 0.20, undefined)}
          {weightBar('Sent Crisis', crW['sentiment_crisis'] ?? 0.20, undefined)}
        </div>
      </div>

      {/* Fusion Bucket Chart */}
      {fusionBuckets && fusionBucketChart(fusionBuckets)}

      {/* Guidance */}
      {!hasOptimized && (
        <div className="mt-3 text-[10px] text-slate-600 bg-white/[0.02] rounded p-2 border border-white/[0.04]">
          <strong className="text-slate-500">Next step:</strong> Run{' '}
          <code className="text-amber-400/80">python3 bin/optimize_cmi_weights.py</code>{' '}
          after collecting enough trade data. IC analysis + logistic regression will derive
          data-driven weights to compare against these hand-tuned values.
        </div>
      )}
    </GlassCard>
  );
}
