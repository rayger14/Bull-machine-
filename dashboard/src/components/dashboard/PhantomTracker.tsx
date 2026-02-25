import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import { fmtUsd, fmtPct } from '../../utils/format';
import type { PhantomTracker as PhantomTrackerData } from '../../api/types';

interface PhantomTrackerProps {
  phantom?: PhantomTrackerData;
  threshold?: number;
  winRate?: number; // Real win rate for comparison
}

function statBox(label: string, value: string, sub?: string, color?: string) {
  return (
    <div className="bg-white/[0.02] rounded-lg border border-white/[0.04] p-3 text-center">
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-sm font-bold font-mono ${color ?? 'text-slate-300'}`}>{value}</div>
      {sub && <div className="text-[10px] text-slate-600 mt-0.5">{sub}</div>}
    </div>
  );
}

export default function PhantomTracker({ phantom, threshold, winRate }: PhantomTrackerProps) {
  // Placeholder when no data
  if (!phantom || (phantom.total_phantom_signals ?? 0) === 0) {
    return (
      <GlassCard>
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs text-slate-500 uppercase tracking-wider">
            Phantom Trade Tracker
          </span>
          <Badge variant="neutral">collecting</Badge>
        </div>
        <p className="text-xs text-slate-600 italic">
          Phantom tracker collecting data... Rejected signals are being tracked to measure filter effectiveness.
        </p>
      </GlassCard>
    );
  }

  const totalSignals = phantom.total_phantom_signals ?? 0;
  const completed = phantom.completed_phantom_trades ?? 0;
  const phantomWR = phantom.phantom_win_rate ?? 0;
  const phantomPnl = phantom.phantom_pnl ?? 0;
  const realPnl = phantom.real_pnl ?? 0;
  const phantomAvg = phantom.phantom_avg_pnl ?? 0;
  const activeCount = phantom.active_phantom_positions ?? 0;

  // Fusion bucket chart data
  const buckets = phantom.fusion_buckets ?? {};
  const bucketKeys = Object.keys(buckets).sort((a, b) => parseFloat(a) - parseFloat(b));
  const maxBucketCount = Math.max(
    1,
    ...bucketKeys.map((k) => (buckets[k].wins ?? 0) + (buckets[k].losses ?? 0))
  );

  // Per-archetype breakdown
  const archBreakdown = phantom.by_archetype ?? {};
  const archKeys = Object.keys(archBreakdown).sort((a, b) => {
    const pnlA = archBreakdown[a].pnl ?? 0;
    const pnlB = archBreakdown[b].pnl ?? 0;
    return pnlB - pnlA;
  });

  // Recent trades (last 10)
  const recentTrades = (phantom.trades ?? []).slice(-10).reverse();

  // Active phantom positions
  const activePositions = phantom.active ?? [];

  return (
    <GlassCard>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-500 uppercase tracking-wider">
            Phantom Trade Tracker
          </span>
          <Badge variant="violet">{totalSignals} tracked</Badge>
        </div>
        {phantom.insight && (
          <span className="text-[10px] text-slate-500 max-w-xs truncate" title={phantom.insight}>
            {phantom.insight}
          </span>
        )}
      </div>

      {/* A) Summary Stats Row */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 mb-4">
        {statBox('Phantom Signals', totalSignals.toLocaleString())}
        {statBox('Completed', completed.toLocaleString(), `${activeCount} active`)}
        {statBox(
          'Phantom WR',
          fmtPct(phantomWR, 1),
          winRate != null ? `real: ${fmtPct(winRate, 1)}` : undefined,
          phantomWR > (winRate ?? 0) ? 'text-amber-400' : 'text-slate-300'
        )}
        {statBox(
          'Phantom PnL',
          fmtUsd(phantomPnl),
          undefined,
          phantomPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
        )}
        {statBox(
          'Real PnL',
          fmtUsd(realPnl),
          undefined,
          realPnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
        )}
        {statBox(
          'Avg Phantom PnL',
          fmtUsd(phantomAvg),
          undefined,
          phantomAvg >= 0 ? 'text-emerald-400' : 'text-rose-400'
        )}
      </div>

      {/* Insight block (full width if present) */}
      {phantom.insight && (
        <div className="bg-violet-500/5 border border-violet-500/10 rounded-lg px-3 py-2 mb-4">
          <span className="text-[10px] text-violet-400 uppercase tracking-wider">Insight</span>
          <p className="text-xs text-slate-300 mt-1">{phantom.insight}</p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* B) Fusion Score Distribution */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-3">
            Fusion Score Distribution
          </div>
          <div className="space-y-1">
            {bucketKeys.map((key) => {
              const b = buckets[key];
              const wins = b.wins ?? 0;
              const losses = b.losses ?? 0;
              const total = wins + losses;
              if (total === 0) return null;
              const wr = total > 0 ? ((wins / total) * 100).toFixed(0) : '0';
              const winPct = (wins / maxBucketCount) * 100;
              const lossPct = (losses / maxBucketCount) * 100;
              const bucketVal = parseFloat(key);
              const isThresholdBucket =
                threshold != null &&
                bucketVal <= threshold &&
                bucketVal + 0.1 > threshold;

              return (
                <div key={key} className="flex items-center gap-2 text-[10px] relative">
                  <span className="w-6 text-right text-slate-500 font-mono shrink-0">
                    {parseFloat(key).toFixed(1)}
                  </span>
                  <div className="flex-1 flex h-4 rounded overflow-hidden bg-white/[0.03] relative">
                    <div
                      className="bg-emerald-500/40 h-full"
                      style={{ width: `${winPct}%` }}
                    />
                    <div
                      className="bg-rose-500/40 h-full"
                      style={{ width: `${lossPct}%` }}
                    />
                    {isThresholdBucket && (
                      <div
                        className="absolute top-0 bottom-0 w-px border-l border-dashed border-violet-400"
                        style={{
                          left: `${((threshold - bucketVal) / 0.1) * 100}%`,
                        }}
                        title={`Threshold: ${threshold.toFixed(2)}`}
                      />
                    )}
                  </div>
                  <span className="w-14 text-right text-slate-400 font-mono shrink-0">
                    {wins}W/{losses}L
                  </span>
                  <span className="w-10 text-right text-slate-300 font-mono shrink-0">
                    {wr}%
                  </span>
                </div>
              );
            })}
          </div>
          {threshold != null && (
            <div className="flex items-center gap-1 mt-2 text-[10px] text-violet-400">
              <span className="inline-block w-3 border-t border-dashed border-violet-400" />
              <span>Dynamic threshold: {threshold.toFixed(2)}</span>
            </div>
          )}
        </div>

        {/* C) Per-Archetype Breakdown */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-3">
            Per-Archetype Breakdown
          </div>
          {archKeys.length === 0 ? (
            <p className="text-[10px] text-slate-600 italic">No archetype data yet</p>
          ) : (
            <table className="w-full text-[10px]">
              <thead>
                <tr className="text-slate-500 uppercase">
                  <th className="text-left pb-1 font-medium">Archetype</th>
                  <th className="text-right pb-1 font-medium">W</th>
                  <th className="text-right pb-1 font-medium">L</th>
                  <th className="text-right pb-1 font-medium">WR</th>
                  <th className="text-right pb-1 font-medium">PnL</th>
                </tr>
              </thead>
              <tbody>
                {archKeys.map((arch) => {
                  const d = archBreakdown[arch];
                  const wins = d.wins ?? 0;
                  const losses = d.losses ?? 0;
                  const total = wins + losses;
                  const wr = total > 0 ? (wins / total) * 100 : 0;
                  const pnl = d.pnl ?? 0;
                  return (
                    <tr key={arch} className="border-t border-white/[0.03]">
                      <td className="py-1 text-slate-300">
                        {arch.replace(/_/g, ' ')}
                      </td>
                      <td className="py-1 text-right text-emerald-400 font-mono">{wins}</td>
                      <td className="py-1 text-right text-rose-400 font-mono">{losses}</td>
                      <td className="py-1 text-right text-slate-300 font-mono">
                        {wr.toFixed(0)}%
                      </td>
                      <td
                        className={`py-1 text-right font-mono ${
                          pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {fmtUsd(pnl)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* D) Recent Phantom Trades */}
      {recentTrades.length > 0 && (
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3 mt-4">
          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-3">
            Recent Phantom Trades (last {recentTrades.length})
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px]">
              <thead>
                <tr className="text-slate-500 uppercase">
                  <th className="text-left pb-1 font-medium">Archetype</th>
                  <th className="text-right pb-1 font-medium">Entry</th>
                  <th className="text-right pb-1 font-medium">Exit</th>
                  <th className="text-right pb-1 font-medium">PnL</th>
                  <th className="text-right pb-1 font-medium">PnL%</th>
                  <th className="text-right pb-1 font-medium">Fusion</th>
                  <th className="text-left pb-1 font-medium pl-2">Rejection</th>
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((t, i) => {
                  const pnl = t.pnl ?? 0;
                  const isWin = pnl >= 0;
                  return (
                    <tr
                      key={i}
                      className={`border-t border-white/[0.03] ${
                        isWin ? 'bg-emerald-500/[0.02]' : 'bg-rose-500/[0.02]'
                      }`}
                    >
                      <td className="py-1 text-slate-300">
                        {t.archetype?.replace(/_/g, ' ') ?? '--'}
                      </td>
                      <td className="py-1 text-right text-slate-400 font-mono">
                        {t.entry_price != null ? `$${t.entry_price.toFixed(0)}` : '--'}
                      </td>
                      <td className="py-1 text-right text-slate-400 font-mono">
                        {t.exit_price != null ? `$${t.exit_price.toFixed(0)}` : '--'}
                      </td>
                      <td
                        className={`py-1 text-right font-mono font-bold ${
                          isWin ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {fmtUsd(pnl)}
                      </td>
                      <td
                        className={`py-1 text-right font-mono ${
                          isWin ? 'text-emerald-400' : 'text-rose-400'
                        }`}
                      >
                        {t.pnl_pct != null ? fmtPct(t.pnl_pct, 1) : '--'}
                      </td>
                      <td className="py-1 text-right text-cyan-400 font-mono">
                        {t.fusion_score != null ? t.fusion_score.toFixed(2) : '--'}
                      </td>
                      <td className="py-1 pl-2 text-slate-500 max-w-[120px] truncate" title={t.rejection_reason}>
                        {t.rejection_reason ?? t.rejection_stage ?? '--'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* E) Active Phantom Positions */}
      {activePositions.length > 0 && (
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3 mt-4">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">
              Active Phantom Positions
            </span>
            <Badge variant="yellow">{activePositions.length}</Badge>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {activePositions.map((p, i) => (
              <div
                key={i}
                className="bg-white/[0.02] rounded-lg border border-white/[0.04] p-2"
              >
                <div className="flex items-center justify-between mb-1">
                  <Badge variant="cyan">{p.archetype?.replace(/_/g, ' ') ?? 'unknown'}</Badge>
                  <span className="text-[10px] text-slate-500">{p.direction ?? 'long'}</span>
                </div>
                <div className="grid grid-cols-2 gap-1 text-[10px]">
                  <div>
                    <span className="text-slate-600">Entry:</span>{' '}
                    <span className="text-slate-300 font-mono">
                      {p.entry_price != null ? `$${p.entry_price.toFixed(0)}` : '--'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-600">Fusion:</span>{' '}
                    <span className="text-cyan-400 font-mono">
                      {p.fusion_score != null ? p.fusion_score.toFixed(2) : '--'}
                    </span>
                  </div>
                  {p.stop_loss != null && (
                    <div>
                      <span className="text-slate-600">SL:</span>{' '}
                      <span className="text-rose-400 font-mono">${p.stop_loss.toFixed(0)}</span>
                    </div>
                  )}
                  {p.take_profit != null && (
                    <div>
                      <span className="text-slate-600">TP:</span>{' '}
                      <span className="text-emerald-400 font-mono">
                        ${p.take_profit.toFixed(0)}
                      </span>
                    </div>
                  )}
                </div>
                {p.rejection_reason && (
                  <div className="mt-1 text-[10px] text-slate-500 italic truncate" title={p.rejection_reason}>
                    {p.rejection_reason}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </GlassCard>
  );
}
