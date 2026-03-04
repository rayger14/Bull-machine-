import { useState } from 'react';
import { Info, TrendingUp, TrendingDown, Activity, Eye, ChevronDown } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import Gauge from '../ui/Gauge';
import type { WyckoffData, WyckoffMarketContext, OracleData } from '../../api/types';

interface WyckoffCycleProps {
  wyckoff: WyckoffData | undefined;
  oracle?: OracleData;
}

// ── Event metadata derived from engine/wyckoff/events.py ──────────────

interface EventMeta {
  short: string;
  label: string;
  explanation: string;
  engineUsage: string;
  type: 'accum' | 'distrib';
  /** Detection criteria from engine/wyckoff/events.py — shown when event is active */
  criteria: DetectionCriterion[];
}

interface DetectionCriterion {
  label: string;
  /** Which market_context key to check */
  contextKey?: keyof WyckoffMarketContext;
  /** How to evaluate: 'above' means value > threshold is met */
  direction: 'above' | 'below';
  /** Threshold value */
  threshold: number;
  /** Display format */
  format: (v: number) => string;
  /** Weight in confidence formula */
  weight: number;
}

const EVENT_META: Record<string, EventMeta> = {
  // ── Accumulation sequence ──
  sc: {
    short: 'SC',
    label: 'Selling Climax',
    type: 'accum',
    explanation:
      'Panic selling at extreme volume. The market capitulates — weak hands dump positions and smart money begins absorbing supply. Identified by volume Z-score > 2.5, price at range lows, and wide-range bar with strong lower wick (>60% absorption).',
    engineUsage:
      'Triggers Phase A detection. SC confidence feeds wyckoff_event_confidence (35% volume weight, 25% range position, 25% range Z, 15% wick quality). Signals potential accumulation beginning.',
    criteria: [
      { label: 'Volume Spike', contextKey: 'volume_z', direction: 'above', threshold: 2.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.35 },
      { label: 'Range Size', contextKey: 'range_z', direction: 'above', threshold: 1.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.25 },
      { label: 'Lower Wick', contextKey: 'lower_wick_pct', direction: 'above', threshold: 0.6, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.15 },
      { label: 'At Range Lows', contextKey: 'close_position', direction: 'below', threshold: 0.2, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  ar: {
    short: 'AR',
    label: 'Automatic Rally',
    type: 'accum',
    explanation:
      'The relief bounce after a Selling Climax. Short sellers cover and bargain hunters enter, creating a sharp move up on declining volume. Must retrace 40-70% of the SC range within 10 bars.',
    engineUsage:
      'Validates Phase A — confirms supply was genuinely exhausted at SC. AR + SC together establish the trading range boundaries that define the accumulation zone.',
    criteria: [
      { label: 'Volume Declining', contextKey: 'volume_z', direction: 'below', threshold: 1.0, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.30 },
      { label: 'Strong Close', contextKey: 'close_position', direction: 'above', threshold: 0.6, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  st: {
    short: 'ST',
    label: 'Secondary Test',
    type: 'accum',
    explanation:
      'The market retests the SC low on significantly lower volume. Price approaches the low but doesn\'t make a new low — proving that selling pressure is drying up. Smart money is absorbing remaining supply.',
    engineUsage:
      'Confirms Phase A. Engine requires volume_z < 0.5 (much lower than SC) and no new low. Higher ST confidence = stronger evidence that accumulation is genuine.',
    criteria: [
      { label: 'Low Volume', contextKey: 'volume_z', direction: 'below', threshold: 0.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.40 },
      { label: 'Near Support', contextKey: 'close_position', direction: 'below', threshold: 0.4, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.40 },
    ],
  },
  spring_a: {
    short: 'Sp-A',
    label: 'Spring (Deep)',
    type: 'accum',
    explanation:
      'A deliberate fake breakdown — price drops 1.5%+ below the trading range to trigger stop losses and shake out remaining weak holders. Strong lower wick (>50%) shows immediate buying absorption. The classic "bear trap."',
    engineUsage:
      'Phase C event with high signal value. Spring_A is one of the strongest entry signals — if confirmed by PTI > 0.6, it triggers wyckoff_pti_confluence (25% weight in PTI score). Optimized from 0.5% to 1.5% for crypto volatility.',
    criteria: [
      { label: 'Lower Wick', contextKey: 'lower_wick_pct', direction: 'above', threshold: 0.5, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.30 },
      { label: 'Volume Spike', contextKey: 'volume_z', direction: 'above', threshold: 0.8, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.25 },
    ],
  },
  spring_b: {
    short: 'Sp-B',
    label: 'Spring (Shallow)',
    type: 'accum',
    explanation:
      'A milder version of Spring A — price dips 0.2-2% below range with >30% lower wick and quick recovery. Less dramatic but still indicates smart money defending support. Bonus confidence if price closes above range midpoint.',
    engineUsage:
      'Phase C event (20% weight in PTI score). Less reliable than Spring A alone but gains significance with volume confirmation and when combined with other accumulation evidence.',
    criteria: [
      { label: 'Lower Wick', contextKey: 'lower_wick_pct', direction: 'above', threshold: 0.3, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.30 },
      { label: 'Recovery Close', contextKey: 'close_position', direction: 'above', threshold: 0.5, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  sos: {
    short: 'SOS',
    label: 'Sign of Strength',
    type: 'accum',
    explanation:
      'The first decisive move upward on strong volume — price breaks the 20-bar high with volume Z > 1.5 and a strong close. This is the market\'s first clear signal that demand now exceeds supply.',
    engineUsage:
      'Phase B event. SOS feeds directly into archetype fusion scoring at 15-35% weight (varies by archetype). A high-confidence SOS significantly boosts trap_within_trend and liquidity_vacuum archetype scores.',
    criteria: [
      { label: 'Volume Surge', contextKey: 'volume_z', direction: 'above', threshold: 1.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.35 },
      { label: 'Strong Close', contextKey: 'close_position', direction: 'above', threshold: 0.7, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  lps: {
    short: 'LPS',
    label: 'Last Point of Support',
    type: 'accum',
    explanation:
      'The final pullback before markup begins — price retests support on very low volume (Z < 0) with a strong close (>60% of range). This is the "last chance to buy cheap" before institutional buying drives prices up.',
    engineUsage:
      'Phase D event — the highest-confidence long entry in Wyckoff theory. LPS + SOS together signal that accumulation is complete and markup is imminent. Engine scores LPS at support levels with strong close quality.',
    criteria: [
      { label: 'Very Low Volume', contextKey: 'volume_z', direction: 'below', threshold: 0.0, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.40 },
      { label: 'Strong Close', contextKey: 'close_position', direction: 'above', threshold: 0.6, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.35 },
    ],
  },
  st_bc: {
    short: 'ST-BC',
    label: 'Secondary Test (Distribution)',
    type: 'distrib',
    explanation:
      'The market retests the BC high on significantly lower volume. Price approaches the high but doesn\'t make a new high — proving that buying pressure is fading. Smart money is distributing into remaining demand.',
    engineUsage:
      'Confirms distribution Phase A. Engine requires volume_z < 0.5 (much lower than BC) and no new high. Higher ST-BC confidence = stronger evidence that distribution is genuine.',
    criteria: [
      { label: 'Low Volume', contextKey: 'volume_z', direction: 'below', threshold: 0.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.40 },
      { label: 'Near Resistance', contextKey: 'close_position', direction: 'above', threshold: 0.6, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.40 },
    ],
  },
  // ── Distribution sequence ──
  bc: {
    short: 'BC',
    label: 'Buying Climax',
    type: 'distrib',
    explanation:
      'Euphoric buying at extreme volume. The market peaks — retail FOMO drives prices to unsustainable levels while smart money begins distributing. Mirror of SC but at market tops.',
    engineUsage:
      'Triggers Phase A (distribution). BC confidence feeds distribution detection. The engine uses BC to identify potential market tops and begin tracking for distribution events.',
    criteria: [
      { label: 'Volume Spike', contextKey: 'volume_z', direction: 'above', threshold: 2.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.35 },
      { label: 'Upper Wick', contextKey: 'upper_wick_pct', direction: 'above', threshold: 0.6, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.15 },
      { label: 'At Range Highs', contextKey: 'close_position', direction: 'above', threshold: 0.8, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  as: {
    short: 'AR/AS',
    label: 'Automatic Reaction',
    type: 'distrib',
    explanation:
      'The relief drop after a Buying Climax. Profit-takers sell and the initial wave of distribution creates a sharp move down on declining volume. Establishes the lower bound of the distribution range.',
    engineUsage:
      'Validates distribution Phase A — confirms demand was exhausted at BC. AS + BC establish the distribution range. Used to identify 1D M2 (distribution) signals.',
    criteria: [
      { label: 'Volume Declining', contextKey: 'volume_z', direction: 'below', threshold: 1.0, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.30 },
      { label: 'Weak Close', contextKey: 'close_position', direction: 'below', threshold: 0.4, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  sow: {
    short: 'SOW',
    label: 'Sign of Weakness',
    type: 'distrib',
    explanation:
      'The first decisive move downward on strong volume — price breaks the 20-bar low with volume Z > 1.5 and a weak close. Mirror of SOS. Supply now visibly exceeds demand.',
    engineUsage:
      'Phase B distribution event. SOW feeds tf1d_m2_signal and contributes to bearish regime detection. When SOW fires on daily timeframe, the engine increases dynamic threshold (more selective).',
    criteria: [
      { label: 'Volume Surge', contextKey: 'volume_z', direction: 'above', threshold: 1.5, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.35 },
      { label: 'Weak Close', contextKey: 'close_position', direction: 'below', threshold: 0.3, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.25 },
    ],
  },
  ut: {
    short: 'UT',
    label: 'Upthrust',
    type: 'distrib',
    explanation:
      'A fake breakout above the distribution range that reverses within 3 bars on a volume spike. Designed to trap late buyers and trigger breakout stops. The classic "bull trap."',
    engineUsage:
      'Phase C distribution event (25% weight in PTI score). UT with high PTI > 0.6 triggers wyckoff_pti_confluence. Engine uses this as a strong warning against entering longs.',
    criteria: [
      { label: 'Volume Spike', contextKey: 'volume_z', direction: 'above', threshold: 1.0, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.35 },
      { label: 'Upper Wick', contextKey: 'upper_wick_pct', direction: 'above', threshold: 0.3, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.30 },
    ],
  },
  utad: {
    short: 'UTAD',
    label: 'Upthrust After Distribution',
    type: 'distrib',
    explanation:
      'The final and most dangerous bull trap — an UT validated by extreme RSI (>70). This is smart money\'s last distribution play before major decline. Often the highest price before a significant markdown phase.',
    engineUsage:
      'Strongest Phase C distribution signal (30% weight in PTI score). UTAD fires on 1D timeframe = engine sets tf1d_m2_signal = 1, significantly raising the dynamic threshold to block new longs.',
    criteria: [
      { label: 'Volume Spike', contextKey: 'volume_z', direction: 'above', threshold: 1.0, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.35 },
      { label: 'Overbought RSI', contextKey: 'rsi_14', direction: 'above', threshold: 70, format: (v) => `${v.toFixed(0)}`, weight: 0.30 },
    ],
  },
  lpsy: {
    short: 'LPSY',
    label: 'Last Point of Supply',
    type: 'distrib',
    explanation:
      'The final weak rally before markdown — price attempts to rise but volume is low and the close is weak (<40% of range). Mirror of LPS. This is the "last chance to sell" before institutional selling drives prices down.',
    engineUsage:
      'Phase D distribution event. LPSY confirms distribution is complete and markdown is imminent. Engine uses this alongside SOW to maximize threshold penalty in bearish regimes.',
    criteria: [
      { label: 'Very Low Volume', contextKey: 'volume_z', direction: 'below', threshold: 0.0, format: (v) => `Z=${v.toFixed(1)}`, weight: 0.40 },
      { label: 'Weak Close', contextKey: 'close_position', direction: 'below', threshold: 0.4, format: (v) => `${(v * 100).toFixed(0)}%`, weight: 0.35 },
    ],
  },
};

// ── Phase groups ──────────────────────────────────────────────────────

interface PhaseGroup {
  label: string;
  sublabel: string;
  color: string;
  borderColor: string;
  events: string[];
  description: string;
}

const PHASE_GROUPS: PhaseGroup[] = [
  {
    label: 'Phase A — Stopping Action',
    sublabel: 'Accumulation',
    color: 'text-emerald-400',
    borderColor: 'border-emerald-500/20',
    events: ['sc', 'ar', 'st'],
    description:
      'Capitulation and initial demand absorption. SC marks the panic low, AR the relief bounce, ST confirms supply is drying up.',
  },
  {
    label: 'Phase B/C — Building Cause',
    sublabel: 'Accumulation',
    color: 'text-cyan-400',
    borderColor: 'border-cyan-500/20',
    events: ['sos', 'spring_a', 'spring_b'],
    description:
      'Smart money accumulates within the range. Springs shake out remaining weak holders. SOS proves demand exceeds supply.',
  },
  {
    label: 'Phase D — Markup Begins',
    sublabel: 'Accumulation',
    color: 'text-violet-400',
    borderColor: 'border-violet-500/20',
    events: ['lps'],
    description:
      'Final test before the uptrend. LPS on low volume = last entry before institutional buying drives price up.',
  },
  {
    label: 'Phase A — Stopping Action',
    sublabel: 'Distribution',
    color: 'text-amber-400',
    borderColor: 'border-amber-500/20',
    events: ['bc', 'as', 'st_bc'],
    description:
      'Euphoria peaks and initial supply hits market. BC marks the panic top, AS the relief drop, ST-BC confirms buying pressure is fading.',
  },
  {
    label: 'Phase B/C — Building Cause',
    sublabel: 'Distribution',
    color: 'text-orange-400',
    borderColor: 'border-orange-500/20',
    events: ['sow', 'ut', 'utad'],
    description:
      'Smart money distributes within the range. Upthrusts trap late buyers. SOW proves supply exceeds demand.',
  },
  {
    label: 'Phase D — Markdown Begins',
    sublabel: 'Distribution',
    color: 'text-rose-400',
    borderColor: 'border-rose-500/20',
    events: ['lpsy'],
    description:
      'Final weak rally before the downtrend. LPSY on low volume = last chance to exit before institutional selling drives price down.',
  },
];

// ── Phase info for the phase indicator ────────────────────────────────

const PHASE_INFO: Record<string, { label: string; description: string; color: string }> = {
  A: {
    label: 'Phase A — Stopping Action',
    description: 'The market is stopping its prior trend. Climax events (SC or BC) have fired, marking potential exhaustion. Automatic reactions and secondary tests are establishing the trading range boundaries.',
    color: 'text-emerald-400',
  },
  B: {
    label: 'Phase B — Building Cause',
    description: 'Smart money is building a position within the trading range. Signs of Strength (SOS) or Weakness (SOW) are beginning to emerge. The longer Phase B lasts, the larger the eventual move.',
    color: 'text-cyan-400',
  },
  C: {
    label: 'Phase C — Testing',
    description: 'The critical test phase. Springs/Upthrusts create deliberate false breakouts to trap traders on the wrong side. This is where the engine looks for high-confidence entry signals (Spring + PTI confluence).',
    color: 'text-violet-400',
  },
  D: {
    label: 'Phase D — Trend Emerging',
    description: 'The new trend is beginning. LPS/LPSY confirms the last holdouts have been absorbed. Price should move decisively in the new direction with increasing volume.',
    color: 'text-amber-400',
  },
  E: {
    label: 'Phase E — Trend Continuation',
    description: 'The trend is fully established. Pullbacks are shallow and on low volume. The engine adjusts dynamic thresholds: permissive in trending markets (lower bar for entries), restrictive against the trend.',
    color: 'text-blue-400',
  },
  neutral: {
    label: 'No Active Phase',
    description: 'No Wyckoff events detected in recent bars. The market may be in a trend without identifiable accumulation/distribution structure, or between cycles.',
    color: 'text-slate-500',
  },
};

// ── Narrative generation ─────────────────────────────────────────────

function buildNarrative(
  events: Record<string, { active?: boolean; confidence?: number }>,
  _phase: string,
  _seqPos: number,
  ctx?: WyckoffMarketContext,
): string[] {
  const active = Object.entries(events)
    .filter(([, e]) => e?.active)
    .map(([k, e]) => ({ key: k, conf: e.confidence ?? 0, meta: EVENT_META[k] }))
    .sort((a, b) => b.conf - a.conf);

  if (active.length === 0) {
    return ['No Wyckoff events are currently active. The market is between structural patterns — price action is not forming recognizable accumulation or distribution sequences on the 1H timeframe.'];
  }

  const lines: string[] = [];
  const accumKeys = new Set(['sc', 'ar', 'st', 'sos', 'spring_a', 'spring_b', 'lps']);
  const distribKeys = new Set(['bc', 'as', 'st_bc', 'sow', 'ut', 'utad', 'lpsy']);

  const accumActive = active.filter((a) => accumKeys.has(a.key));
  const distribActive = active.filter((a) => distribKeys.has(a.key));
  const bias = accumActive.length > distribActive.length ? 'accumulation' : distribActive.length > accumActive.length ? 'distribution' : 'mixed';

  // Lead with the overall reading
  if (bias === 'accumulation') {
    const avgConf = accumActive.reduce((s, a) => s + a.conf, 0) / accumActive.length;
    lines.push(
      `The Wyckoff structure is showing ${accumActive.length === 1 ? 'an' : ''} **accumulation** signal${accumActive.length > 1 ? 's' : ''} with ${avgConf >= 0.8 ? 'high' : avgConf >= 0.6 ? 'moderate' : 'early'} confidence. Smart money appears to be absorbing supply at these levels.`,
    );
  } else if (bias === 'distribution') {
    const avgConf = distribActive.reduce((s, a) => s + a.conf, 0) / distribActive.length;
    lines.push(
      `The Wyckoff structure is showing **distribution** signal${distribActive.length > 1 ? 's' : ''} with ${avgConf >= 0.8 ? 'high' : avgConf >= 0.6 ? 'moderate' : 'early'} confidence. Institutional selling may be underway beneath the surface.`,
    );
  } else {
    lines.push('Mixed signals — both accumulation and distribution events are active, suggesting the market is in a transitional or choppy phase.');
  }

  // Explain each active event in context
  for (const a of active) {
    const pct = `${(a.conf * 100).toFixed(0)}%`;
    switch (a.key) {
      case 'sc':
        lines.push(`**Selling Climax** (${pct}) — Capitulation-level volume detected. Panic sellers are dumping while smart money absorbs. This marks the potential low of the current cycle.`);
        break;
      case 'bc':
        lines.push(`**Buying Climax** (${pct}) — Euphoria-level buying detected. This often marks the peak before smart money begins distributing.`);
        break;
      case 'ar':
        lines.push(`**Automatic Rally** (${pct}) — Relief bounce on declining volume validates the prior climax. The trading range is being established.`);
        break;
      case 'as':
        lines.push(`**Automatic Reaction** (${pct}) — Pullback after the buying climax. The distribution range is being established.`);
        break;
      case 'st':
        lines.push(`**Secondary Test** (${pct}) — Price is retesting support on dry volume. The fact that volume is NOT returning means sellers are exhausted — this confirms the accumulation base.`);
        break;
      case 'sos':
        lines.push(`**Sign of Strength** (${pct}) — First decisive breakout on strong volume. Demand now exceeds supply. This is early confirmation that accumulation is working.`);
        break;
      case 'sow':
        lines.push(`**Sign of Weakness** (${pct}) — Breakdown on strong volume confirms supply exceeds demand. The distribution structure is progressing.`);
        break;
      case 'spring_a':
        lines.push(`**Deep Spring** (${pct}) — A deliberate shakeout below support with immediate recovery. This is a high-value entry signal — stop losses have been triggered and absorbed by smart money.`);
        break;
      case 'spring_b':
        lines.push(`**Shallow Spring** (${pct}) — A mild dip below the range with quick recovery. Smart money is defending support, though the signal is less dramatic than a deep spring.`);
        break;
      case 'lps':
        lines.push(`**Last Point of Support** (${pct}) — Price is retesting support on minimal volume with a strong close. This is the highest-confidence long entry in Wyckoff theory — the final test before markup begins.`);
        break;
      case 'lpsy':
        lines.push(`**Last Point of Supply** (${pct}) — A weak rally attempt on low volume. This is the final distribution signal before markdown — expect lower prices.`);
        break;
      case 'st_bc':
        lines.push(`**Secondary Test (Distribution)** (${pct}) — Price retests the BC high on dry volume. The fact that buying volume is NOT returning means buyers are exhausted — this confirms the distribution base.`);
        break;
      case 'ut':
        lines.push(`**Upthrust** (${pct}) — False breakout above resistance that reversed quickly. This is a classic bull trap designed to catch breakout traders.`);
        break;
      case 'utad':
        lines.push(`**Upthrust After Distribution** (${pct}) — The most dangerous bull trap — final push to new highs with RSI extreme before the markdown phase. Stay cautious.`);
        break;
    }
  }

  // Combined pattern interpretation
  const activeKeys = new Set(active.map((a) => a.key));
  if (activeKeys.has('st') && activeKeys.has('lps')) {
    lines.push('**Combined reading:** ST + LPS together is a powerful accumulation confluence. The secondary test confirmed sellers are exhausted, and the last point of support indicates markup is imminent. The engine treats this as a high-probability long zone.');
  } else if (activeKeys.has('sc') && activeKeys.has('ar')) {
    lines.push('**Combined reading:** SC + AR establishes the accumulation trading range. The climax has been reached and the initial recovery validates exhaustion. Watch for ST to confirm.');
  } else if (activeKeys.has('sos') && activeKeys.has('lps')) {
    lines.push('**Combined reading:** SOS + LPS is a textbook Phase D signal. Demand has proven itself (SOS) and the pullback to support (LPS) held on dry volume. Markup should follow.');
  } else if (activeKeys.has('bc') && activeKeys.has('as')) {
    lines.push('**Combined reading:** BC + AS establishes the distribution range. The euphoric peak has been followed by initial selling. Watch for SOW to confirm.');
  } else if (activeKeys.has('sow') && activeKeys.has('lpsy')) {
    lines.push('**Combined reading:** SOW + LPSY is a bearish confluence. Supply overwhelmed demand (SOW) and the weak rally attempt (LPSY) failed. Markdown is imminent.');
  } else if (activeKeys.has('ut') && activeKeys.has('sow')) {
    lines.push('**Combined reading:** UT + SOW confirms the distribution structure. The false breakout trapped buyers, and the subsequent breakdown on volume confirmed supply dominance.');
  }

  // Engine impact
  if (bias === 'accumulation' && active.some((a) => a.conf >= 0.8)) {
    lines.push('**Engine impact:** These accumulation signals boost wyckoff_score (15-35% of fusion weight, varies by archetype), making the engine more willing to enter long positions when other archetype criteria align.');
  } else if (bias === 'distribution' && active.some((a) => a.conf >= 0.8)) {
    lines.push('**Engine impact:** These distribution signals raise the dynamic threshold, making the engine MORE selective about new entries. The CMI regime detector interprets this as increased instability.');
  }

  // Volume context
  if (ctx?.volume_z != null) {
    const vz = ctx.volume_z;
    if (vz > 2) {
      lines.push(`Current volume is ${vz.toFixed(1)}x above normal — consistent with climax-type events.`);
    } else if (vz < -0.5) {
      lines.push(`Current volume is ${Math.abs(vz).toFixed(1)}x below normal — consistent with test/support events (drying supply).`);
    }
  }

  return lines;
}

// ── Helpers ───────────────────────────────────────────────────────────

const confidenceColor = (c: number): string => {
  if (c >= 0.8) return 'text-emerald-400';
  if (c >= 0.5) return 'text-amber-400';
  return 'text-slate-500';
};

// ── Detection Evidence component ─────────────────────────────────────

function DetectionEvidence({
  eventKey,
  confidence,
  ctx,
}: {
  eventKey: string;
  confidence: number;
  ctx?: WyckoffMarketContext;
}) {
  const meta = EVENT_META[eventKey];
  if (!meta || !ctx) return null;

  return (
    <div className="mt-2 space-y-2">
      <div className="flex items-center gap-1.5 text-[10px] text-slate-500 uppercase tracking-wider">
        <Eye className="w-3 h-3" />
        Detection Evidence
      </div>
      {/* Overall confidence bar */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-slate-600 w-16 shrink-0">Confidence</span>
        <div className="flex-1 h-2.5 bg-white/[0.04] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              confidence >= 0.8 ? 'bg-emerald-500' : confidence >= 0.6 ? 'bg-cyan-500' : 'bg-amber-500'
            }`}
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
        <span className={`text-xs font-mono w-10 text-right ${confidenceColor(confidence)}`}>
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>
      {/* Per-criterion bars */}
      {meta.criteria.map((crit) => {
        const raw = crit.contextKey ? ctx[crit.contextKey] : undefined;
        const met =
          raw != null
            ? crit.direction === 'above'
              ? raw >= crit.threshold
              : raw <= crit.threshold
            : false;
        const barPct =
          raw != null
            ? crit.direction === 'above'
              ? Math.min(1, Math.max(0, raw / (crit.threshold * 2)))
              : Math.min(1, Math.max(0, 1 - raw / (crit.threshold * 2)))
            : 0;

        return (
          <div key={crit.label} className="flex items-center gap-2">
            <span className="text-[10px] text-slate-600 w-16 shrink-0 truncate" title={crit.label}>
              {crit.label}
            </span>
            <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden relative">
              {/* Threshold marker */}
              <div
                className="absolute top-0 bottom-0 w-px bg-white/20"
                style={{ left: `50%` }}
                title={`Threshold: ${crit.threshold}`}
              />
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  met ? 'bg-emerald-500/80' : 'bg-rose-500/50'
                }`}
                style={{ width: `${barPct * 100}%` }}
              />
            </div>
            <span className={`text-[10px] font-mono w-14 text-right ${met ? 'text-emerald-400' : 'text-rose-400/60'}`}>
              {raw != null ? crit.format(raw) : '--'}
            </span>
            <span className={`text-[9px] w-4 ${met ? 'text-emerald-500' : 'text-rose-500/40'}`}>
              {met ? '\u2713' : '\u2717'}
            </span>
          </div>
        );
      })}
      <div className="text-[9px] text-slate-700 italic">
        Thresholds from engine/wyckoff/events.py. Bar center = threshold. Green = criterion met.
      </div>
    </div>
  );
}

// ── EventChip component ──────────────────────────────────────────────

function EventChip({
  eventKey,
  events,
  expanded,
  onToggle,
  ctx,
  eventNarratives,
}: {
  eventKey: string;
  events: Record<string, { active?: boolean; confidence?: number }>;
  expanded: string;
  onToggle: (k: string) => void;
  ctx?: WyckoffMarketContext;
  eventNarratives?: Record<string, string>;
}) {
  const meta = EVENT_META[eventKey];
  const evt = events?.[eventKey];
  const isActive = evt?.active ?? false;
  const confidence = evt?.confidence ?? 0;
  const isExpanded = expanded === eventKey;

  return (
    <div>
      <button
        onClick={() => onToggle(eventKey)}
        className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-lg border text-xs font-medium transition-all duration-200 ${
          isActive
            ? 'bg-cyan-500/[0.08] border-cyan-500/30 text-cyan-300 shadow-[0_0_8px_rgba(6,182,212,0.12)]'
            : isExpanded
              ? 'bg-white/[0.04] border-white/[0.12] text-slate-400'
              : 'bg-white/[0.02] border-white/[0.06] text-slate-600 hover:border-white/[0.12] hover:text-slate-400'
        }`}
      >
        <span>{meta ? `${meta.label} (${meta.short})` : eventKey.toUpperCase()}</span>
        {isActive && (
          <span className={`font-mono text-[10px] ${confidenceColor(confidence)}`}>
            {(confidence * 100).toFixed(0)}%
          </span>
        )}
        <Info className="w-2.5 h-2.5 opacity-40" />
      </button>
      {isExpanded && meta && (
        <div className="mt-2 ml-1 mb-1 p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl text-xs space-y-2">
          <div className="font-medium text-slate-300">{meta.label}</div>
          <p className="text-slate-400 leading-relaxed">{meta.explanation}</p>
          {/* Detection evidence for active events */}
          {isActive && (
            <DetectionEvidence eventKey={eventKey} confidence={confidence} ctx={ctx} />
          )}
          {/* Structural narrative — what the engine saw */}
          {isActive && eventNarratives?.[eventKey] && (
            <div className="border-t border-white/[0.04] pt-2 mt-2">
              <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">What the engine saw</div>
              <p className="text-[11px] text-slate-400 leading-relaxed">{eventNarratives[eventKey]}</p>
            </div>
          )}
          <div className="border-t border-white/[0.04] pt-2">
            <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">How the engine uses this</div>
            <p className="text-slate-500 leading-relaxed">{meta.engineUsage}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Simple markdown-like bold rendering ──────────────────────────────

function RichText({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return (
    <span>
      {parts.map((part, i) =>
        part.startsWith('**') && part.endsWith('**') ? (
          <span key={i} className="font-semibold text-slate-300">
            {part.slice(2, -2)}
          </span>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </span>
  );
}

// ── Main Component ───────────────────────────────────────────────────

function formatPrice(price: number): string {
  if (price >= 1000) return price.toLocaleString('en-US', { maximumFractionDigits: 0 });
  return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
}

function interpretMTF(bullish: number, bearish: number): string {
  if (bullish > 0.3) return 'accumulating';
  if (bearish > 0.3) return 'distributing';
  return 'quiet';
}

export default function WyckoffCycle({ wyckoff, oracle }: WyckoffCycleProps) {
  const [expandedEvent, setExpandedEvent] = useState('');
  const [showPhaseInfo, setShowPhaseInfo] = useState(false);
  const [showNarrative, setShowNarrative] = useState(true);
  const [showWyckoffInfo, setShowWyckoffInfo] = useState(false);
  const [showEventHistory, setShowEventHistory] = useState(false);
  const [showMethodology, setShowMethodology] = useState(false);

  if (!wyckoff) return null;

  const events = wyckoff.events ?? {};
  const ctx = wyckoff.market_context;
  const score1h = wyckoff.score ?? 0;
  const score4h = wyckoff.tf4h_phase_score ?? 0;
  const score1d = wyckoff.tf1d_score ?? 0;
  const phase = wyckoff.phase ?? 'neutral';

  const activeCount = Object.values(events).filter((e) => e?.active).length;
  const activeAccum = ['sc', 'ar', 'st', 'sos', 'spring_a', 'spring_b', 'lps']
    .filter((k) => events[k]?.active).length;
  const activeDist = ['bc', 'as', 'st_bc', 'sow', 'ut', 'utad', 'lpsy']
    .filter((k) => events[k]?.active).length;

  const phaseInfo = PHASE_INFO[phase] ?? PHASE_INFO.neutral;
  const narrative = buildNarrative(events, phase, 0, ctx);

  const toggleEvent = (k: string) => setExpandedEvent(expandedEvent === k ? '' : k);

  const biasIcon =
    activeAccum > activeDist ? (
      <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
    ) : activeDist > activeAccum ? (
      <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
    ) : (
      <Activity className="w-3.5 h-3.5 text-slate-500" />
    );

  return (
    <GlassCard>
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs text-slate-500 uppercase tracking-wider">
          Wyckoff Cycle Analysis
        </span>
        {activeCount > 0 && (
          <Badge variant={activeAccum > activeDist ? 'green' : 'red'}>
            {activeCount} Active
          </Badge>
        )}
        {wyckoff.tf1d_m1_signal === 1 && <Badge variant="green">M1 Accum</Badge>}
        {wyckoff.tf1d_m2_signal === 1 && <Badge variant="red">M2 Distrib</Badge>}
      </div>

      {/* HTF Structural Bias Summary — most important at a glance */}
      {(() => {
        const bull4h = wyckoff.bullish_4h ?? 0;
        const bear4h = wyckoff.bearish_4h ?? 0;
        const bull1d = wyckoff.bullish_1d ?? 0;
        const bear1d = wyckoff.bearish_1d ?? 0;
        const htfBull = bull4h * 0.4 + bull1d * 0.6;
        const htfBear = bear4h * 0.4 + bear1d * 0.6;
        const netBias = htfBull - htfBear;
        const biasLabel = netBias > 0.15 ? 'Accumulation' : netBias < -0.15 ? 'Distribution' : 'Neutral';
        const biasColor = netBias > 0.15 ? 'text-emerald-400' : netBias < -0.15 ? 'text-rose-400' : 'text-slate-400';
        const biasStrength = Math.abs(netBias);
        const strengthLabel = biasStrength > 0.4 ? 'Strong' : biasStrength > 0.2 ? 'Moderate' : biasStrength > 0.05 ? 'Weak' : '';

        return (htfBull > 0 || htfBear > 0) ? (
          <div className="mb-4 p-3 bg-white/[0.03] rounded-lg border border-white/[0.06]">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-slate-600 uppercase tracking-wider">Structural Bias (4H + 1D weighted)</span>
              <span className={`text-sm font-semibold ${biasColor}`}>
                {strengthLabel} {biasLabel}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1 h-2 bg-white/[0.04] rounded-full overflow-hidden relative">
                <div
                  className="absolute top-0 left-0 h-full bg-emerald-500/60 rounded-l-full transition-all"
                  style={{ width: `${Math.min(htfBull * 100, 50)}%` }}
                />
                <div
                  className="absolute top-0 right-0 h-full bg-rose-500/60 rounded-r-full transition-all"
                  style={{ width: `${Math.min(htfBear * 100, 50)}%` }}
                />
              </div>
            </div>
            <div className="flex justify-between mt-1 text-[9px]">
              <span className="text-emerald-600">Bull {(htfBull * 100).toFixed(0)}%</span>
              <span className="text-slate-700">1D: {bull1d > 0.1 ? 'accum' : bear1d > 0.1 ? 'distrib' : 'quiet'} | 4H: {bull4h > 0.1 ? 'accum' : bear4h > 0.1 ? 'distrib' : 'quiet'}</span>
              <span className="text-rose-600">Bear {(htfBear * 100).toFixed(0)}%</span>
            </div>
          </div>
        ) : null;
      })()}

      {/* Oracle Market Structure Narrative */}
      {oracle?.market_structure?.summary && (
        <div className="mb-4 p-3 bg-gray-800/50 rounded-lg">
          <p className="text-sm text-gray-200">{oracle.market_structure.summary}</p>
          {oracle.market_structure.key_levels && (
            <div className="flex gap-4 mt-2 text-xs flex-wrap">
              <span className="text-green-400">Support: ${formatPrice(oracle.market_structure.key_levels.support)}</span>
              <span className="text-red-400">Resistance: ${formatPrice(oracle.market_structure.key_levels.resistance)}</span>
              <span className="text-gray-400">Invalidation: ${formatPrice(oracle.market_structure.key_levels.invalidation)}</span>
            </div>
          )}
          {oracle.market_structure.next_expected && (
            <p className="text-xs text-blue-400 mt-1">Next: {oracle.market_structure.next_expected}</p>
          )}
        </div>
      )}

      {/* What is Wyckoff? Educational Section */}
      <div className="mb-4">
        <button
          onClick={() => setShowWyckoffInfo(!showWyckoffInfo)}
          className="flex items-center gap-2 mb-2 text-xs text-slate-500 uppercase tracking-wider hover:text-slate-400 transition-colors w-full"
        >
          <Info className="w-3.5 h-3.5 text-slate-500" />
          <span>What is Wyckoff Analysis?</span>
          <span className="text-[9px] text-slate-700 normal-case">
            {showWyckoffInfo ? '(collapse)' : '(expand)'}
          </span>
        </button>
        {showWyckoffInfo && (
          <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-3">
            <p className="text-xs text-slate-400 leading-relaxed">
              <RichText text="**Wyckoff Method** is a 100-year-old framework for reading institutional money flow through price and volume. It identifies four market phases — Accumulation (smart money buying), Markup (price trending up), Distribution (smart money selling), and Markdown (price trending down)." />
            </p>
            <p className="text-xs text-slate-400 leading-relaxed">
              <RichText text="**Role in the engine:** Wyckoff carries **15-35% weight** (varies by archetype) in archetype fusion scoring — the largest single factor. The engine detects 14 specific events across 5 phases on 3 timeframes (1H, 4H, 1D). High-confidence Wyckoff events significantly boost archetype scores for trap_within_trend and liquidity_vacuum." />
            </p>
            <div>
              <p className="text-xs text-slate-400 leading-relaxed mb-2">
                <RichText text="**Phases and what they mean:**" />
              </p>
              <div className="space-y-2 pl-2">
                <p className="text-xs text-slate-400 leading-relaxed">
                  <RichText text="**Phase A (Stopping Action):** The prior trend is exhausted. Climax events (SC or BC) mark extreme volume capitulation. Near-term: expect a range to form. Long-term: the foundation for a new trend is being laid, but it's too early to trade aggressively." />
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  <RichText text="**Phase B (Building Cause):** Smart money quietly accumulates (or distributes). This is the LONGEST phase — weeks to months of sideways action. Near-term: choppy, range-bound trading. Long-term: the longer Phase B lasts, the LARGER the eventual move (Wyckoff's 'cause and effect' principle)." />
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  <RichText text="**Phase C (Testing):** The critical test — springs/upthrusts create deliberate false breakouts to trap traders on the wrong side. Near-term: volatile fakeouts, sharp reversals. Long-term: this is where the engine finds its highest-confidence entry signals." />
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  <RichText text="**Phase D (Trend Emerging):** LPS/LPSY confirms the last holdouts absorbed. Near-term: price moves decisively in the new direction. Long-term: new trend is confirmed, pullbacks should be bought (or rallies sold)." />
                </p>
                <p className="text-xs text-slate-400 leading-relaxed">
                  <RichText text="**Phase E (Trend Continuation):** Full trend underway with increasing participation. Near-term: momentum-driven moves. Long-term: ride the trend until distribution/accumulation signals appear again." />
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Phase Indicator */}
      <div
        className="mb-4 p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl cursor-pointer hover:bg-white/[0.04] transition-all"
        onClick={() => setShowPhaseInfo(!showPhaseInfo)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className={`text-sm font-medium ${phaseInfo.color}`}>{phaseInfo.label}</span>
            <Info className="w-3 h-3 text-slate-600" />
          </div>
        </div>
        {showPhaseInfo && (
          <p className="text-xs text-slate-400 mt-2 leading-relaxed">{phaseInfo.description}</p>
        )}
      </div>

      {/* Cycle Timeline */}
      {wyckoff.cycle_start && (
        <div className="mb-4 p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-2">
          <div className="text-[10px] text-slate-600 uppercase tracking-wider">Cycle Timeline</div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
            <div>
              <span className="text-slate-600">Started: </span>
              <span className="text-slate-400 font-mono">
                {new Date(wyckoff.cycle_start).toLocaleDateString('en-US', { timeZone: 'America/Los_Angeles' })}{' '}
                {new Date(wyckoff.cycle_start).toLocaleTimeString('en-US', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            {wyckoff.cycle_duration_hours != null && (
              <div>
                <span className="text-slate-600">Duration: </span>
                <span className="text-slate-300 font-mono">
                  {wyckoff.cycle_duration_hours >= 24
                    ? `${(wyckoff.cycle_duration_hours / 24).toFixed(1)}d`
                    : `${wyckoff.cycle_duration_hours.toFixed(0)}h`}
                </span>
              </div>
            )}
          </div>
          {/* Phase transition history */}
          {wyckoff.phase_transitions && wyckoff.phase_transitions.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-1">
              {wyckoff.phase_transitions.map((t, i) => (
                <span key={i} className="text-[9px] text-slate-600 bg-white/[0.03] px-1.5 py-0.5 rounded">
                  {t.from_phase}→{t.to_phase} at ${t.price?.toLocaleString()} ({new Date(t.timestamp ?? '').toLocaleDateString('en-US', { timeZone: 'America/Los_Angeles', month: 'short', day: 'numeric' })})
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Market Reading Narrative */}
      <div className="mb-4">
        <button
          onClick={() => setShowNarrative(!showNarrative)}
          className="flex items-center gap-2 mb-2 text-xs text-slate-500 uppercase tracking-wider hover:text-slate-400 transition-colors w-full"
        >
          {biasIcon}
          <span>Market Reading</span>
          <span className="text-[9px] text-slate-700 normal-case">
            {showNarrative ? '(collapse)' : '(expand)'}
          </span>
        </button>
        {showNarrative && (
          <div className="p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-2">
            {narrative.map((line, i) => (
              <p key={i} className="text-xs text-slate-400 leading-relaxed">
                <RichText text={line} />
              </p>
            ))}
          </div>
        )}
      </div>

      {/* Event Confidence Breakdown */}
      {wyckoff.conviction && wyckoff.conviction.components && wyckoff.conviction.components.length > 0 && (
        <div className="mb-4 p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <div className="text-[10px] text-slate-600 uppercase tracking-wider">Event Confidence Breakdown</div>
            <span className={`text-xs font-mono ${
              (wyckoff.conviction.total_score ?? 0) >= 0.8 ? 'text-emerald-400'
                : (wyckoff.conviction.total_score ?? 0) >= 0.6 ? 'text-amber-400' : 'text-slate-500'
            }`}>
              {((wyckoff.conviction.total_score ?? 0) * 100).toFixed(0)}%
            </span>
          </div>
          <p className="text-[11px] text-slate-400">{wyckoff.conviction.reason}</p>
          <div className="space-y-1.5">
            {wyckoff.conviction.components.map((c) => (
              <div key={c.event} className="flex items-center gap-2 text-[10px]">
                <span className="text-slate-500 w-14 shrink-0 font-mono uppercase">{c.event}</span>
                <div className="flex-1 h-1.5 bg-white/[0.04] rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${(c.confidence ?? 0) >= 0.8 ? 'bg-emerald-500' : (c.confidence ?? 0) >= 0.6 ? 'bg-cyan-500' : 'bg-slate-500'}`}
                    style={{ width: `${Math.min((c.confidence ?? 0) * 100, 100)}%` }}
                  />
                </div>
                <span className="text-slate-600 w-12 text-right font-mono">{((c.confidence ?? 0) * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Event Groups */}
      <div className="space-y-4 mb-5">
        {PHASE_GROUPS.map((group, gi) => {
          const groupActive = group.events.some((k) => events[k]?.active);
          return (
            <div key={gi}>
              <div className="flex items-center gap-2 mb-1.5">
                <span className={`text-[10px] uppercase tracking-wider ${group.color}`}>
                  {group.label}
                </span>
                <span className="text-[9px] text-slate-700">({group.sublabel})</span>
                {groupActive && <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />}
              </div>
              <p className="text-[10px] text-slate-600 mb-2 leading-relaxed">{group.description}</p>
              <div className="flex flex-wrap gap-1.5">
                {group.events.map((key) => (
                  <EventChip
                    key={key}
                    eventKey={key}
                    events={events}
                    expanded={expandedEvent}
                    onToggle={toggleEvent}
                    ctx={ctx}
                    eventNarratives={wyckoff.event_narratives}
                  />
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Event History Timeline */}
      {wyckoff.event_history && wyckoff.event_history.length > 0 && (
        <div className="mb-4">
          <button
            onClick={() => setShowEventHistory(!showEventHistory)}
            className="flex items-center gap-2 mb-2 text-[10px] text-slate-600 uppercase tracking-wider hover:text-slate-400 transition-colors w-full"
          >
            <Activity className="w-3 h-3" />
            <span>Recent Event History ({wyckoff.event_history.length})</span>
            <ChevronDown className={`w-3 h-3 ml-auto transition-transform ${showEventHistory ? 'rotate-180' : ''}`} />
          </button>
          {showEventHistory && (
            <div className="space-y-1 max-h-64 overflow-y-auto">
              {wyckoff.event_history.map((eh, i) => {
                const isAccum = ['sc', 'ar', 'st', 'spring_a', 'spring_b', 'sos', 'lps'].includes(eh.event ?? '') && eh.event !== 'st_bc';
                return (
                  <div key={i} className="flex items-center gap-2 text-[10px] px-2 py-1.5 bg-white/[0.02] rounded-lg border border-white/[0.04]">
                    <span className={`font-mono w-12 shrink-0 uppercase ${isAccum ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {(eh.event ?? '').replace('_', ' ')}
                    </span>
                    <span className="text-slate-400 font-mono w-20 shrink-0">
                      ${(eh.price ?? 0).toLocaleString()}
                    </span>
                    <span className={`font-mono w-10 shrink-0 ${
                      (eh.confidence ?? 0) >= 0.8 ? 'text-emerald-400' : (eh.confidence ?? 0) >= 0.6 ? 'text-amber-400' : 'text-slate-500'
                    }`}>
                      {((eh.confidence ?? 0) * 100).toFixed(0)}%
                    </span>
                    <span className="text-slate-700 text-[9px] ml-auto">
                      {eh.timestamp ? new Date(eh.timestamp).toLocaleDateString('en-US', { timeZone: 'America/Los_Angeles', month: 'short', day: 'numeric' }) : ''}
                      {' '}
                      {eh.timestamp ? new Date(eh.timestamp).toLocaleTimeString('en-US', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit' }) : ''}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Multi-TF Scores */}
      <div className="border-t border-white/[0.06] pt-4">
        <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">
          Multi-Timeframe Peak Event Confidence
        </div>
        <p className="text-[10px] text-slate-700 mb-3">
          Wyckoff is primarily a higher-timeframe structural tool — 1D and 4H scores matter most for identifying accumulation/distribution phases. 1H events are tactical entry triggers that fire within the HTF context. A 1H score of 0 is normal between events.
        </p>
        <div className="grid grid-cols-3 gap-3">
          <div>
            <Gauge
              value={score1d}
              label="1D Macro"
              colorStops={[
                'bg-rose-500',
                'bg-amber-400',
                'bg-emerald-400',
                'bg-emerald-500',
                'bg-cyan-500',
              ]}
            />
            {wyckoff.tf1d_bars != null && (
              <div className="text-[9px] text-slate-700 text-center mt-1">
                {wyckoff.tf1d_bars}/20 daily bars
              </div>
            )}
          </div>
          <Gauge
            value={score4h}
            label="4H Structure"
            colorStops={[
              'bg-rose-500',
              'bg-amber-400',
              'bg-emerald-400',
              'bg-emerald-500',
              'bg-cyan-500',
            ]}
          />
          <Gauge
            value={score1h}
            label="1H Trigger"
            colorStops={[
              'bg-rose-500',
              'bg-amber-400',
              'bg-emerald-400',
              'bg-emerald-500',
              'bg-cyan-500',
            ]}
          />
        </div>
      </div>

      {/* MTF Directional Scores (graded 0-1, diversity-weighted) */}
      {(wyckoff.bullish_1h || wyckoff.bearish_1h || wyckoff.bullish_4h || wyckoff.bearish_4h || wyckoff.bullish_1d || wyckoff.bearish_1d) ? (
        <div className="border-t border-white/[0.06] pt-4 mt-3">
          <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">
            MTF Directional Scores (Bullish / Bearish)
          </div>
          <p className="text-[10px] text-slate-700 mb-3">
            Diversity-weighted scores from SM-validated events. Unlike M1/M2 binary, these are graded 0-1. A single event caps at ~0.33; multiple concurrent events (genuine accumulation/distribution) can reach 1.0.
          </p>
          <div className="grid grid-cols-3 gap-3">
            {[
              { label: '1D', bull: wyckoff.bullish_1d ?? 0, bear: wyckoff.bearish_1d ?? 0 },
              { label: '4H', bull: wyckoff.bullish_4h ?? 0, bear: wyckoff.bearish_4h ?? 0 },
              { label: '1H', bull: wyckoff.bullish_1h ?? 0, bear: wyckoff.bearish_1h ?? 0 },
            ].map(({ label, bull, bear }) => (
              <div key={label} className="p-2 bg-white/[0.02] rounded-lg border border-white/[0.04]">
                <div className="text-[9px] text-slate-600 text-center mb-1">{label}</div>
                <div className="flex justify-between items-center gap-1">
                  <div className="text-center flex-1">
                    <div className={`text-xs font-mono ${bull > 0.1 ? 'text-emerald-400' : 'text-slate-700'}`}>
                      {(bull * 100).toFixed(1)}%
                    </div>
                    <div className="text-[8px] text-emerald-600">Bull</div>
                  </div>
                  <div className="w-px h-4 bg-white/[0.06]" />
                  <div className="text-center flex-1">
                    <div className={`text-xs font-mono ${bear > 0.1 ? 'text-rose-400' : 'text-slate-700'}`}>
                      {(bear * 100).toFixed(1)}%
                    </div>
                    <div className="text-[8px] text-rose-600">Bear</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-400 mt-2">
            1H: {interpretMTF(wyckoff.bullish_1h ?? 0, wyckoff.bearish_1h ?? 0)} · 4H: {interpretMTF(wyckoff.bullish_4h ?? 0, wyckoff.bearish_4h ?? 0)} · 1D: {interpretMTF(wyckoff.bullish_1d ?? 0, wyckoff.bearish_1d ?? 0)}
          </p>
        </div>
      ) : null}

      {/* Overall Confidence */}
      {wyckoff.event_confidence != null && wyckoff.event_confidence > 0 && (
        <div className="mt-3 flex items-center gap-2 text-xs">
          <span className="text-slate-600">Peak Event Confidence:</span>
          <Badge variant={wyckoff.event_confidence >= 0.8 ? 'green' : wyckoff.event_confidence >= 0.5 ? 'cyan' : 'yellow'}>
            {(wyckoff.event_confidence * 100).toFixed(1)}%
          </Badge>
        </div>
      )}

      {/* Methodology Transparency */}
      {wyckoff.methodology && (
        <div className="mt-3 border-t border-white/[0.06] pt-3">
          <button
            onClick={() => setShowMethodology(!showMethodology)}
            className="flex items-center gap-1 text-[9px] text-slate-700 hover:text-slate-500 transition-colors"
          >
            <Info className="w-2.5 h-2.5" />
            <span>How this detection works — methodology &amp; limitations</span>
          </button>
          {showMethodology && (
            <div className="mt-2 p-3 bg-white/[0.02] rounded-lg border border-white/[0.04] space-y-2">
              <p className="text-[10px] text-slate-500 leading-relaxed">{wyckoff.methodology.description}</p>
              {wyckoff.methodology.limitations && wyckoff.methodology.limitations.length > 0 && (
                <div>
                  <div className="text-[9px] text-slate-700 uppercase tracking-wider mb-1">Known Limitations</div>
                  <ul className="space-y-0.5">
                    {wyckoff.methodology.limitations.map((lim, i) => (
                      <li key={i} className="text-[9px] text-slate-600 flex items-start gap-1.5">
                        <span className="text-slate-700 mt-0.5 shrink-0">--</span>
                        <span>{lim}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </GlassCard>
  );
}
