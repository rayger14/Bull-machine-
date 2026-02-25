import { Newspaper } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { Heartbeat, OracleData } from '../../api/types';

interface MarketBriefingProps {
  hb: Heartbeat;
  oracle?: OracleData;
}

// ── Intelligence synthesis helpers ────────────────────────────────────

function buildHeadline(hb: Heartbeat): string {
  const regime = hb.regime ?? 'neutral';
  const fg = hb.macro?.fear_greed;
  const fgLabel = hb.macro?.fear_greed_label;
  const wyPhase = hb.wyckoff?.phase ?? 'neutral';
  const threshold = hb.threshold ?? 0.18;
  const crisis = hb.crisis_prob;

  const regimeDesc =
    regime === 'bull'
      ? 'bullish'
      : regime === 'bear'
        ? 'bearish'
        : 'neutral';

  // Wyckoff phase mapping
  const wyckoffDesc: Record<string, string> = {
    A: 'accumulation-phase',
    B: 'markup-phase',
    C: 'distribution-phase',
    D: 'markdown-phase',
    E: 'late-cycle',
  };
  const wyckoffStr = wyckoffDesc[wyPhase] ?? '';

  // Threshold context
  let thresholdCtx: string;
  if (threshold >= 0.55) {
    thresholdCtx = `highly selective (threshold ${threshold.toFixed(2)})`;
  } else if (threshold >= 0.35) {
    thresholdCtx = `moderately selective (threshold ${threshold.toFixed(2)})`;
  } else {
    thresholdCtx = `permissive (threshold ${threshold.toFixed(2)})`;
  }

  // Fear & Greed context
  let fgCtx = '';
  if (fg != null) {
    if (fg < 20) fgCtx = ` with extreme fear (F&G: ${fg})`;
    else if (fg < 40) fgCtx = ` with elevated fear (F&G: ${fg})`;
    else if (fg > 80) fgCtx = ` with extreme greed (F&G: ${fg})`;
    else if (fg > 60) fgCtx = ` with elevated greed (F&G: ${fg})`;
    else fgCtx = ` with ${fgLabel ?? 'moderate'} sentiment (F&G: ${fg})`;
  }

  // Crisis override
  if (crisis != null && crisis > 0.7) {
    return `CRISIS ALERT: Bearish regime with crisis probability at ${(crisis * 100).toFixed(0)}%. The engine is ${thresholdCtx} with emergency sizing active${fgCtx}. Only the strongest setups will deploy capital.`;
  }

  // Build sentence
  const parts: string[] = [];
  if (wyckoffStr) {
    parts.push(`BTC is in a ${regimeDesc} regime with ${wyckoffStr} Wyckoff signals.`);
  } else {
    parts.push(`BTC is in a ${regimeDesc} regime.`);
  }
  parts.push(`The engine is ${thresholdCtx}${fgCtx}.`);

  if (regime === 'bear' && threshold > 0.5) {
    parts.push('Only the strongest signals will pass.');
  } else if (regime === 'neutral') {
    parts.push('Markets are range-bound with no dominant directional signal.');
  }

  return parts.join(' ');
}

function buildImmediateOutlook(hb: Heartbeat): string[] {
  const bullets: string[] = [];
  const threshold = hb.threshold ?? 0.18;
  const positions = hb.positions ?? 0;

  // 1w macro narrative
  const weekOutlook = hb.macro_outlook?.['1w'];
  if (weekOutlook?.narrative) {
    bullets.push(weekOutlook.narrative);
  }

  // Wyckoff events
  const wyEvents = hb.wyckoff?.events;
  if (wyEvents) {
    const accumEvents = ['SC', 'AR', 'ST', 'Spring', 'LPS', 'SOS'];
    const distribEvents = ['BC', 'UT', 'UTAD', 'SOW', 'LPSY'];
    let accumCount = 0;
    let distribCount = 0;
    for (const [key, ev] of Object.entries(wyEvents)) {
      if (ev?.active) {
        if (accumEvents.includes(key)) accumCount++;
        if (distribEvents.includes(key)) distribCount++;
      }
    }
    if (accumCount > 0 && distribCount === 0) {
      bullets.push(`${accumCount} accumulation event(s) active -- Wyckoff favors upside.`);
    } else if (distribCount > 0 && accumCount === 0) {
      bullets.push(`${distribCount} distribution event(s) active -- Wyckoff favors downside.`);
    } else if (accumCount > 0 && distribCount > 0) {
      bullets.push(`Mixed Wyckoff signals: ${accumCount} accumulation and ${distribCount} distribution event(s) active.`);
    }
  }

  // Stress scenarios
  const stress = hb.active_stress_scenarios?.filter(s => s.active) ?? [];
  if (stress.length > 0) {
    bullets.push(`${stress.length} active stress scenario(s): ${stress.map(s => s.name).join(', ')}.`);
  }

  // Threshold context
  if (threshold > 0.5) {
    bullets.push('Engine is highly selective -- only the strongest signals will deploy capital.');
  } else if (threshold < 0.3) {
    bullets.push('Engine is permissive -- moderate-confidence signals can deploy capital.');
  } else {
    bullets.push('Engine selectivity is moderate -- balanced risk posture.');
  }

  // Positions
  if (positions > 0) {
    bullets.push(`${positions} open position(s) being managed.`);
  } else {
    bullets.push('No open positions. Watching for next high-confidence setup.');
  }

  return bullets;
}

function buildWatchList(hb: Heartbeat): string[] {
  const items: string[] = [];
  const fg = hb.macro?.fear_greed;

  // Cointegration opportunities
  const pairs = hb.cointegration?.pairs ?? [];
  const nearOpps = pairs.filter(
    p => p.cointegrated && Math.abs(p.current_zscore ?? p.z_score ?? 0) > 1.5,
  );
  if (nearOpps.length > 0) {
    items.push(`${nearOpps.length} cointegration pair(s) approaching reversion threshold.`);
  }

  // Macro extremes
  if (hb.macro?.vix_z != null && Math.abs(hb.macro.vix_z) > 1.5) {
    items.push(
      `VIX at ${hb.macro.vix_z > 0 ? 'elevated' : 'suppressed'} levels (${hb.macro.vix_z.toFixed(1)} sigma).`,
    );
  }
  if (hb.macro?.dxy_z != null && Math.abs(hb.macro.dxy_z) > 1.5) {
    items.push(
      `Dollar strength ${hb.macro.dxy_z > 0 ? 'elevated' : 'weak'} (${hb.macro.dxy_z.toFixed(1)} sigma).`,
    );
  }
  if (hb.macro?.gold_z != null && Math.abs(hb.macro.gold_z) > 1.5) {
    items.push(
      `Gold at ${hb.macro.gold_z > 0 ? 'elevated' : 'depressed'} levels (${hb.macro.gold_z.toFixed(1)} sigma).`,
    );
  }

  // Fear & Greed extremes
  if (fg != null && fg < 20) {
    items.push(`Extreme fear (F&G: ${fg}) -- contrarian buy signal historically.`);
  }
  if (fg != null && fg > 80) {
    items.push(`Extreme greed (F&G: ${fg}) -- markets may be overextended.`);
  }

  // Crisis probability
  if (hb.crisis_prob != null && hb.crisis_prob > 0.5) {
    items.push(
      `Crisis probability elevated at ${(hb.crisis_prob * 100).toFixed(0)}% -- emergency sizing may activate above 70%.`,
    );
  }

  // Capital flows
  const edges = hb.capital_flows?.edges;
  if (edges) {
    const activeEdges = Object.values(edges).filter(e => e?.active);
    const drains = activeEdges.filter(e => e?.direction === 'drain');
    if (drains.length > 0) {
      items.push(
        `${drains.length} capital drain flow(s) active -- institutional money is leaving risk assets.`,
      );
    }
    const inflows = activeEdges.filter(e => e?.direction === 'inflow');
    if (inflows.length > 0) {
      items.push(
        `${inflows.length} capital inflow(s) active -- money rotating into risk assets.`,
      );
    }
  }

  // Fallback
  if (items.length === 0) {
    items.push('No unusual conditions detected. Markets are in equilibrium.');
  }

  return items;
}

// ── Formatting helpers ────────────────────────────────────────────────

function formatTimePST(ts?: string): string {
  if (!ts) return '';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleString('en-US', {
      timeZone: 'America/Los_Angeles',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    }) + ' PST';
  } catch {
    return ts;
  }
}

function formatPrice(price?: number): string {
  if (price == null || isNaN(price)) return '--';
  return price.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });
}

function fgBadgeVariant(fg?: number): 'green' | 'red' | 'yellow' | 'orange' | 'cyan' | 'neutral' {
  if (fg == null) return 'neutral';
  if (fg < 20) return 'red';
  if (fg < 40) return 'orange';
  if (fg > 80) return 'green';
  if (fg > 60) return 'cyan';
  return 'yellow';
}

// ── Component ─────────────────────────────────────────────────────────

export default function MarketBriefing({ hb, oracle }: MarketBriefingProps) {
  const regime = hb.regime ?? 'neutral';
  const threshold = hb.threshold ?? 0.18;
  const fg = hb.macro?.fear_greed;
  const fgLabel = hb.macro?.fear_greed_label;
  const positions = hb.positions ?? 0;
  const btcPrice = hb.btc_price;

  const headline = oracle?.thesis || buildHeadline(hb);
  const outlook = buildImmediateOutlook(hb);
  const watchItems = buildWatchList(hb);

  // Stress count
  const stressCount = hb.active_stress_scenarios?.filter(s => s.active)?.length ?? 0;

  // Active capital flow edges
  const activeFlowCount = hb.capital_flows?.edges
    ? Object.values(hb.capital_flows.edges).filter(e => e?.active).length
    : 0;

  // Regime color
  const regimeColor =
    regime === 'bull'
      ? 'text-emerald-400'
      : regime === 'bear' || regime === 'crisis'
        ? 'text-rose-400'
        : 'text-amber-400';

  const regimeLabel =
    regime === 'bull' ? 'BULL' : regime === 'bear' ? 'BEAR' : regime === 'crisis' ? 'CRISIS' : 'NEUTRAL';

  return (
    <GlassCard className="border-cyan-500/10">
      {/* Header Row */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Newspaper className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-500 uppercase tracking-wider">
            Market Briefing
          </span>
        </div>
        <span className="text-[10px] text-slate-700 font-mono">
          {formatTimePST(hb.timestamp ?? hb.updated_at)}
        </span>
      </div>

      {/* Top Section: Regime Card + Headline */}
      <div className="flex gap-4 mb-4">
        {/* Regime Card (left) */}
        <div className="flex-shrink-0 w-28 bg-white/[0.03] rounded-xl border border-white/[0.06] p-3 text-center">
          <div className="text-[10px] text-slate-600 uppercase mb-1">Regime</div>
          <div className={`text-lg font-bold ${regimeColor}`}>{regimeLabel}</div>
          <div className="text-base font-mono text-slate-300 mt-1">
            ${formatPrice(btcPrice)}
          </div>
          <div className="text-[10px] text-slate-600 mt-1">
            Threshold:{' '}
            <span className="font-mono text-slate-400">{threshold.toFixed(2)}</span>
          </div>
        </div>

        {/* Headline (right, takes remaining space) */}
        <div className="flex-1">
          <p className="text-sm text-slate-300 leading-relaxed">{headline}</p>
          {/* Sentiment badges row */}
          <div className="flex flex-wrap gap-2 mt-2">
            {fg != null && (
              <Badge variant={fgBadgeVariant(fg)}>
                F&G: {fg} {fgLabel ? `(${fgLabel})` : ''}
              </Badge>
            )}
            {stressCount > 0 && (
              <Badge variant="red">
                {stressCount} Stress Scenario{stressCount > 1 ? 's' : ''}
              </Badge>
            )}
            {positions > 0 && (
              <Badge variant="cyan">
                {positions} Position{positions > 1 ? 's' : ''} Open
              </Badge>
            )}
            {activeFlowCount > 0 && (
              <Badge variant="violet">
                {activeFlowCount} Active Flow{activeFlowCount > 1 ? 's' : ''}
              </Badge>
            )}
          </div>
          {/* Oracle-sourced factor bullets */}
          {oracle && (
            <div className="mt-3 space-y-1">
              {oracle.aligned_factors?.slice(0, 3).map((f, i) => (
                <div key={`af-${i}`} className="flex items-start gap-2 text-xs">
                  <span className="text-green-400 mt-0.5">●</span>
                  <span className="text-gray-300">{f}</span>
                </div>
              ))}
              {oracle.conflicting_factors?.slice(0, 2).map((f, i) => (
                <div key={`cf-${i}`} className="flex items-start gap-2 text-xs">
                  <span className="text-amber-400 mt-0.5">●</span>
                  <span className="text-gray-300">{f}</span>
                </div>
              ))}
              {oracle.risks?.filter(r => r.status === 'active').slice(0, 2).map((r, i) => (
                <div key={`ri-${i}`} className="flex items-start gap-2 text-xs">
                  <span className="text-red-400 mt-0.5">●</span>
                  <span className="text-gray-300">{r.name}: {r.impact}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Bottom Section: Two columns */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Immediate Outlook */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
          <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">
            Immediate Outlook{' '}
            <span className="normal-case text-slate-700">(hours to days)</span>
          </div>
          <ul className="space-y-1.5">
            {outlook.map((bullet, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                <span className="mt-1.5 w-1 h-1 rounded-full bg-cyan-500/60 flex-shrink-0" />
                <span>{bullet}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* What to Watch */}
        <div className="bg-white/[0.02] rounded-xl border border-white/[0.04] p-3">
          <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">
            What to Watch
          </div>
          <ul className="space-y-1.5">
            {watchItems.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                <span className="mt-1.5 w-1 h-1 rounded-full bg-amber-500/60 flex-shrink-0" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </GlassCard>
  );
}
