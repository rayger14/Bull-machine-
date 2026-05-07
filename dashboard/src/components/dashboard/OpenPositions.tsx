import { useState } from 'react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import { fmtUsd, fmtPct } from '../../utils/format';
import type { OpenPosition } from '../../api/types';

interface OpenPositionsProps {
  positions: OpenPosition[] | null | undefined;
}

/* ── Helpers ───────────────────────────────────────────────── */

function directionBadgeVariant(dir?: string): 'green' | 'red' | 'neutral' {
  if (!dir) return 'neutral';
  const d = dir.toLowerCase();
  if (d === 'long' || d === 'buy') return 'green';
  if (d === 'short' || d === 'sell') return 'red';
  return 'neutral';
}

function regimeBadge(regime?: string): { label: string; variant: 'green' | 'red' | 'blue' | 'violet' | 'neutral' } {
  if (!regime) return { label: 'Unknown', variant: 'neutral' };
  const r = regime.toLowerCase();
  if (r === 'bull') return { label: 'Bull', variant: 'green' };
  if (r === 'bear') return { label: 'Bear', variant: 'red' };
  if (r === 'neutral') return { label: 'Neutral', variant: 'blue' };
  if (r === 'crisis') return { label: 'Crisis', variant: 'violet' };
  return { label: regime, variant: 'neutral' };
}

function formatBarsHeld(bars?: number): string {
  if (bars == null) return '--';
  if (bars >= 48) return `${Math.round(bars / 24)}d`;
  return `${bars}h`;
}

function formatPrice(v?: number): string {
  if (v == null) return '--';
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 2 });
}

/* ── Mini gauge (horizontal bar 0-1) ─────────────────────── */

function MiniGauge({ label, value, color }: { label: string; value?: number; color: string }) {
  const v = value ?? 0;
  const pct = Math.min(Math.max(v * 100, 0), 100);
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] text-slate-500 w-24 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[10px] text-slate-400 font-mono w-10 text-right">{v.toFixed(3)}</span>
    </div>
  );
}

/* ── Fusion score bar with threshold marker ──────────────── */

function FusionBar({ score, threshold }: { score?: number; threshold?: number }) {
  const s = score ?? 0;
  const pct = Math.min(Math.max(s * 100, 0), 100);
  const threshPct = threshold != null ? Math.min(Math.max(threshold * 100, 0), 100) : null;

  return (
    <div className="mt-1.5">
      <div className="flex items-center justify-between text-[10px] mb-1">
        <span className="text-slate-500">Fusion Score</span>
        <span className="text-slate-400 font-mono">{s.toFixed(3)}</span>
      </div>
      <div className="relative h-2 bg-white/[0.04] rounded-full overflow-visible">
        <div
          className="h-full rounded-full bg-gradient-to-r from-cyan-500/60 to-cyan-400"
          style={{ width: `${pct}%` }}
        />
        {threshPct != null && (
          <div
            className="absolute top-[-3px] w-0.5 h-[14px] bg-amber-400 rounded-full"
            style={{ left: `${threshPct}%` }}
            title={`Threshold: ${threshold?.toFixed(3)}`}
          />
        )}
      </div>
      {threshold != null && (
        <div className="text-[9px] text-slate-600 mt-0.5 font-mono">
          Threshold: {threshold.toFixed(3)}
        </div>
      )}
    </div>
  );
}

/* ── Scale-out progress bar ──────────────────────────────── */

function ScaleOutProgress({ totalExitsPct, scaleOuts }: { totalExitsPct?: number; scaleOuts?: number[] }) {
  if (totalExitsPct == null && (!scaleOuts || scaleOuts.length === 0)) return null;
  const exitedPct = (totalExitsPct ?? 0) * 100;

  return (
    <div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Scale-Out Progress</div>
      <div className="flex items-center gap-2 mb-1.5">
        <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-500/60 to-emerald-400"
            style={{ width: `${Math.min(exitedPct, 100)}%` }}
          />
        </div>
        <span className="text-[10px] text-slate-400 font-mono w-12 text-right">{exitedPct.toFixed(0)}% exited</span>
      </div>
      {scaleOuts && scaleOuts.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {scaleOuts.map((level, i) => (
            <span key={i} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {(level * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Detail cell ─────────────────────────────────────────── */

function DetailCell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <div className="text-[10px] text-slate-600">{label}</div>
      <div className={`text-xs font-mono ${color ?? 'text-slate-300'}`}>{value}</div>
    </div>
  );
}

/* ── Main Component ──────────────────────────────────────── */

export default function OpenPositions({ positions }: OpenPositionsProps) {
  const [expandedId, setExpandedId] = useState<number | null>(null);

  if (!positions || positions.length === 0) return null;

  const toggle = (idx: number) => setExpandedId(prev => (prev === idx ? null : idx));

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-3">
        Open Positions ({positions.length})
      </div>

      <div className="space-y-2">
        {positions.map((p, i) => {
          const isExpanded = expandedId === i;
          const pnlPositive = (p.unrealized_pnl ?? 0) >= 0;
          const dirVariant = directionBadgeVariant(p.direction);

          return (
            <div key={p.id ?? i} className="rounded-xl border border-white/[0.04] overflow-hidden">
              {/* ── Collapsed card (always visible) ── */}
              <div
                className="flex items-center justify-between p-3 bg-white/[0.02] cursor-pointer hover:bg-white/[0.04] transition-colors duration-200"
                onClick={() => toggle(i)}
              >
                {/* Left: archetype + direction */}
                <div className="flex items-center gap-2 min-w-0">
                  <Badge variant={dirVariant}>
                    {p.archetype?.replace(/_/g, ' ') ?? 'unknown'}
                  </Badge>
                  {p.direction && (
                    <span className={`text-[10px] uppercase font-semibold tracking-wide ${dirVariant === 'green' ? 'text-emerald-400' : dirVariant === 'red' ? 'text-rose-400' : 'text-slate-400'}`}>
                      {p.direction}
                    </span>
                  )}
                </div>

                {/* Center: entry -> current price */}
                <div className="flex items-center gap-1.5 text-xs font-mono text-slate-400">
                  <span>{formatPrice(p.entry_price)}</span>
                  <svg className="w-3 h-3 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>
                  <span className={pnlPositive ? 'text-emerald-400' : 'text-rose-400'}>{formatPrice(p.current_price)}</span>
                </div>

                {/* Right: P&L */}
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-bold font-mono ${pnlPositive ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {fmtUsd(p.unrealized_pnl)} ({fmtPct(p.unrealized_pnl_pct)})
                  </span>
                  <svg
                    className={`w-4 h-4 text-slate-600 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>

              {/* ── Expanded detail panel ── */}
              <div
                className={`transition-all duration-300 ease-in-out overflow-hidden ${isExpanded ? 'max-h-[800px] opacity-100' : 'max-h-0 opacity-0'}`}
              >
                <div className="p-4 pt-2 space-y-4 border-t border-white/[0.04]">

                  {/* Section 1: Trade Summary */}
                  <div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Trade Summary</div>
                    <div className="grid grid-cols-4 gap-x-4 gap-y-2">
                      <DetailCell label="Entry Price" value={formatPrice(p.entry_price)} />
                      <DetailCell label="Current Price" value={formatPrice(p.current_price)} />
                      <DetailCell label="Stop Loss" value={formatPrice(p.stop_loss)} color="text-rose-400" />
                      <DetailCell label="Take Profit" value={formatPrice(p.take_profit)} color="text-emerald-400" />

                      <DetailCell
                        label="SL Distance"
                        value={p.sl_distance_pct != null ? `${p.sl_distance_pct.toFixed(2)}%` : '--'}
                        color="text-rose-400"
                      />
                      <DetailCell
                        label="TP Distance"
                        value={p.tp_distance_pct != null ? `+${p.tp_distance_pct.toFixed(2)}%` : '--'}
                        color="text-emerald-400"
                      />
                      <DetailCell
                        label="Risk:Reward"
                        value={p.risk_reward != null ? p.risk_reward.toFixed(2) : '--'}
                        color="text-cyan-400"
                      />
                      <DetailCell
                        label="Leverage"
                        value={p.leverage != null ? `${p.leverage.toFixed(1)}x` : '--'}
                        color="text-amber-400"
                      />

                      <DetailCell
                        label="Notional Value"
                        value={fmtUsd(p.position_size_usd, 0)}
                        color="text-slate-300"
                      />
                      <DetailCell
                        label="Margin (Cash Used)"
                        value={p.position_size_usd != null && p.leverage != null && p.leverage > 0
                          ? fmtUsd(p.position_size_usd / p.leverage, 0)
                          : fmtUsd(p.margin_used, 0)}
                        color="text-amber-400"
                      />
                      <DetailCell
                        label="Risk Amount"
                        value={p.position_size_usd != null && p.sl_distance_pct != null
                          ? fmtUsd(p.position_size_usd * (p.sl_distance_pct / 100), 0)
                          : '--'}
                        color="text-rose-400"
                      />
                      <DetailCell
                        label="Original Qty"
                        value={p.original_quantity != null ? p.original_quantity.toFixed(6) : '--'}
                      />
                      <DetailCell
                        label="Current Qty"
                        value={p.current_quantity != null ? p.current_quantity.toFixed(6) : '--'}
                      />
                      <DetailCell label="Bars Held" value={formatBarsHeld(p.bars_held)} />
                      <DetailCell
                        label="Entry Time"
                        value={p.entry_time ? new Date(p.entry_time).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false }) : '--'}
                      />
                    </div>
                    {/* Trailing stop */}
                    <div className="mt-2">
                      <DetailCell
                        label="Trailing Stop"
                        value={p.trailing_stop != null ? formatPrice(p.trailing_stop) : 'None'}
                        color={p.trailing_stop != null ? 'text-amber-400' : 'text-slate-600'}
                      />
                    </div>
                  </div>

                  {/* Section 2: Entry Logic */}
                  {(p.narrative || p.fusion_score != null) && (
                    <div>
                      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Entry Logic</div>

                      {p.narrative?.headline && (
                        <p className="text-xs text-slate-300 mb-2">{p.narrative.headline}</p>
                      )}

                      {p.narrative?.confluence_factors && p.narrative.confluence_factors.length > 0 && (
                        <div className="mb-1.5">
                          {p.narrative.confluence_factors.map((f, j) => (
                            <div key={j} className="flex items-start gap-1.5 text-[11px] text-emerald-400 mb-0.5">
                              <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                              <span>{f}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {p.narrative?.risk_factors && p.narrative.risk_factors.length > 0 && (
                        <div className="mb-2">
                          {p.narrative.risk_factors.map((f, j) => (
                            <div key={j} className="flex items-start gap-1.5 text-[11px] text-amber-400 mb-0.5">
                              <span className="mt-0.5 w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                              <span>{f}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      <FusionBar score={p.fusion_score} threshold={p.threshold_at_entry} />

                      {/* Domain score breakdown */}
                      {p.narrative?.domain_scores && (
                        <div className="mt-3 space-y-1">
                          <div className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Domain Scores</div>
                          <MiniGauge label="Wyckoff" value={p.narrative.domain_scores.wyckoff} color="bg-violet-400" />
                          <MiniGauge label="Liquidity" value={p.narrative.domain_scores.liquidity} color="bg-cyan-400" />
                          <MiniGauge label="Momentum" value={p.narrative.domain_scores.momentum} color="bg-amber-400" />
                          <MiniGauge label="SMC" value={p.narrative.domain_scores.smc} color="bg-emerald-400" />
                        </div>
                      )}

                      {/* Hard Gate Values — what features actually triggered the entry */}
                      {p.narrative?.gate_values && Object.keys(p.narrative.gate_values).length > 0 && (
                        <div className="mt-3">
                          <div className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Hard Gates at Entry</div>
                          <div className="grid grid-cols-3 gap-x-3 gap-y-1">
                            {Object.entries(p.narrative.gate_values).map(([k, v]) => (
                              <div key={k} className="flex justify-between text-[10px] font-mono">
                                <span className="text-slate-500">{k}</span>
                                <span className="text-slate-300">{typeof v === 'number' ? v.toFixed(v < 1 && v > -1 ? 3 : 2) : String(v)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Section 3: Market Conditions at Entry */}
                  {(p.regime_at_entry || p.risk_temp_at_entry != null || p.threshold_at_entry != null) && (
                    <div>
                      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Market Conditions at Entry</div>

                      <div className="flex items-center gap-2 mb-2.5">
                        {p.regime_at_entry && (() => {
                          const rb = regimeBadge(p.regime_at_entry);
                          return <Badge variant={rb.variant}>{rb.label}</Badge>;
                        })()}
                      </div>

                      <div className="space-y-1.5">
                        <MiniGauge label="Risk Temp" value={p.risk_temp_at_entry} color="bg-amber-400" />
                        <MiniGauge label="Instability" value={p.instability_at_entry} color="bg-orange-400" />
                        <MiniGauge label="Crisis Prob" value={p.crisis_prob_at_entry} color="bg-rose-400" />
                      </div>

                      {p.threshold_at_entry != null && (
                        <div className="mt-2 flex flex-wrap items-center gap-3 text-[10px] font-mono">
                          <span className="text-slate-500">
                            Threshold: <span className="text-slate-300">{p.threshold_at_entry.toFixed(3)}</span>
                          </span>
                          {p.threshold_margin != null && (
                            <span className="text-slate-500">
                              Margin:{' '}
                              <span className={p.threshold_margin >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                                {p.threshold_margin >= 0 ? '+' : ''}{p.threshold_margin.toFixed(3)}
                              </span>
                            </span>
                          )}
                          {p.would_have_passed != null && (
                            <span className="text-slate-500">
                              Threshold:{' '}
                              <span className={p.would_have_passed ? 'text-emerald-400' : 'text-amber-400'}>
                                {p.would_have_passed ? 'Passed' : 'Bypassed'}
                              </span>
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Section 4: Scale-Out Progress */}
                  <ScaleOutProgress totalExitsPct={p.total_exits_pct} scaleOuts={p.executed_scale_outs} />

                  {/* Factor Attribution */}
                  {p.factor_attribution && Object.keys(p.factor_attribution).length > 0 && (
                    <div>
                      <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Factor Attribution</div>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(p.factor_attribution).map(([k, v]) => (
                          <span
                            key={k}
                            className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-white/[0.03] border border-white/[0.06] text-slate-400"
                          >
                            {k}: <span className={v >= 0 ? 'text-emerald-400' : 'text-rose-400'}>{v >= 0 ? '+' : ''}{(v * 100).toFixed(1)}%</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}
