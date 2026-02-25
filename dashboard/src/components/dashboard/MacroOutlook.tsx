import { useState } from 'react';
import { ChevronDown, TrendingUp, TrendingDown } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import { useAppStore } from '../../stores/useAppStore';
import type { MacroOutlookTimeframe, OracleData } from '../../api/types';

interface MacroOutlookProps {
  outlook: Record<string, MacroOutlookTimeframe> | null | undefined;
  oracle?: OracleData | null;
}

const timeframes = [
  { key: '1w', label: '1 Week', oracleKey: 'short_term' as const },
  { key: '1m', label: '1 Month', oracleKey: 'medium_term' as const },
  { key: '6m', label: '6 Months', oracleKey: 'long_term' as const },
  { key: '1y', label: '1 Year', oracleKey: 'long_term' as const },
];

const regimeVariant = (label?: string): 'green' | 'red' | 'yellow' => {
  const lower = (label ?? '').toLowerCase();
  if (lower.includes('bullish') || lower.includes('bull')) return 'green';
  if (lower.includes('bearish') || lower.includes('bear')) return 'red';
  return 'yellow';
};

/** Convert raw score to a confidence percentage and direction label */
function scoreToConfidence(score: number): { pct: number; direction: 'Bullish' | 'Bearish' | 'Neutral'; color: string } {
  const pct = Math.round(Math.abs(score) * 100);
  if (score > 0.05) return { pct: Math.max(pct, 51), direction: 'Bullish', color: 'text-emerald-400' };
  if (score < -0.05) return { pct: Math.max(pct, 51), direction: 'Bearish', color: 'text-rose-400' };
  return { pct, direction: 'Neutral', color: 'text-amber-400' };
}

/** Determine if a timeframe is bullish based on its score */
function isBullish(score?: number): boolean {
  return (score ?? 0) > 0.05;
}

function isBearish(score?: number): boolean {
  return (score ?? 0) < -0.05;
}

/** Build the "Bull vs Bear" summary sentence */
function buildSummary(
  outlook: Record<string, MacroOutlookTimeframe>,
  oracle?: OracleData | null,
): string {
  let bullCount = 0;
  let bearCount = 0;
  let strongestBullFactor = '';
  let strongestBearFactor = '';
  let bestBullScore = 0;
  let bestBearScore = 0;

  for (const { key } of timeframes) {
    const tf = outlook[key];
    if (!tf) continue;
    const score = tf.score ?? 0;
    if (isBullish(score)) {
      bullCount++;
      if (score > bestBullScore) {
        bestBullScore = score;
        strongestBullFactor = tf.key_movers?.[0]?.replace(/_/g, ' ') ?? '';
      }
    } else if (isBearish(score)) {
      bearCount++;
      if (Math.abs(score) > bestBearScore) {
        bestBearScore = Math.abs(score);
        strongestBearFactor = tf.key_movers?.[0]?.replace(/_/g, ' ') ?? '';
      }
    }
  }

  // Use oracle aligned/conflicting factors as key driver if available
  const oracleDriver = oracle?.aligned_factors?.[0]?.replace(/_/g, ' ') ?? '';

  if (bullCount > bearCount) {
    const driver = oracleDriver || strongestBullFactor;
    const driverSuffix = driver ? ` -- ${driver} is the key driver` : '';
    return `Bull case leads ${bullCount}-${bearCount}${driverSuffix}`;
  } else if (bearCount > bullCount) {
    const driver = oracleDriver || strongestBearFactor;
    const driverSuffix = driver ? ` -- ${driver} is the key driver` : '';
    return `Bear case leads ${bearCount}-${bullCount}${driverSuffix}`;
  } else {
    return `Mixed signals -- ${bullCount} bull, ${bearCount} bear`;
  }
}

export default function MacroOutlook({ outlook, oracle }: MacroOutlookProps) {
  const { showOutlookDetail, setShowOutlookDetail } = useAppStore();
  const [showDetails, setShowDetails] = useState<Record<string, boolean>>({});

  if (!outlook) return null;

  const summary = buildSummary(outlook, oracle);
  const summaryColor = summary.startsWith('Bull')
    ? 'text-emerald-400'
    : summary.startsWith('Bear')
      ? 'text-rose-400'
      : 'text-amber-400';

  return (
    <GlassCard>
      <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">
        Macro Outlook
      </div>

      {/* Bull vs Bear summary sentence */}
      <p className={`text-sm font-medium ${summaryColor} mb-3`}>
        {summary}
      </p>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {timeframes.map(({ key, label, oracleKey }) => {
          const tf = outlook[key];
          if (!tf) return null;
          const expanded = showOutlookDetail === key;
          const score = tf.score ?? 0;
          const { pct, direction, color } = scoreToConfidence(score);
          const oracleTf = oracle?.outlook?.[oracleKey];
          const detailsOpen = showDetails[key] ?? false;

          return (
            <div
              key={key}
              className={`rounded-xl border p-3 cursor-pointer transition-all duration-200 ${
                expanded
                  ? 'bg-white/[0.04] border-white/[0.10] col-span-2 lg:col-span-4'
                  : 'bg-white/[0.02] border-white/[0.04] hover:bg-white/[0.04]'
              }`}
              onClick={() => setShowOutlookDetail(expanded ? '' : key)}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-[10px] text-slate-600 uppercase">{label}</span>
                <ChevronDown
                  className={`w-3 h-3 text-slate-600 transition-transform duration-200 ${
                    expanded ? 'rotate-180' : ''
                  }`}
                />
              </div>

              {/* Confidence percentage + direction (replaces raw score) */}
              <div className="flex items-center gap-2">
                <Badge variant={regimeVariant(tf.label)}>
                  {tf.label ?? '--'}
                </Badge>
                <span className={`text-sm font-bold ${color}`}>
                  {pct}% {direction}
                </span>
              </div>

              {/* Oracle outlook summary */}
              {oracleTf?.summary && (
                <p className="text-xs text-gray-300 mt-1.5 leading-relaxed">
                  {oracleTf.summary}
                </p>
              )}

              {/* Expanded Detail */}
              {expanded && (
                <div className="mt-4 space-y-4">
                  {/* Narrative */}
                  {tf.narrative && (
                    <div className="text-xs italic text-slate-400 leading-relaxed">
                      {tf.narrative}
                    </div>
                  )}

                  {/* Bull / Bear Cases */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {/* Bull Case */}
                    {tf.bull_case && (
                      <div className="bg-emerald-500/[0.04] border border-emerald-500/10 rounded-lg p-3">
                        <div className="flex items-center gap-1.5 mb-2">
                          <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                          <span className="text-[10px] text-emerald-400 uppercase font-medium">
                            Bull Case
                          </span>
                          <span className="text-xs font-mono text-emerald-400 ml-auto">
                            {Math.round((tf.bull_case.score ?? 0) * 100)}%
                          </span>
                        </div>
                        <div className="space-y-0.5">
                          {tf.bull_case.factors?.map((f, i) => (
                            <div key={i} className="text-[10px] text-emerald-300/70">
                              {f}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Bear Case */}
                    {tf.bear_case && (
                      <div className="bg-rose-500/[0.04] border border-rose-500/10 rounded-lg p-3">
                        <div className="flex items-center gap-1.5 mb-2">
                          <TrendingDown className="w-3.5 h-3.5 text-rose-400" />
                          <span className="text-[10px] text-rose-400 uppercase font-medium">
                            Bear Case
                          </span>
                          <span className="text-xs font-mono text-rose-400 ml-auto">
                            {Math.round(Math.abs(tf.bear_case.score ?? 0) * 100)}%
                          </span>
                        </div>
                        <div className="space-y-0.5">
                          {tf.bear_case.factors?.map((f, i) => (
                            <div key={i} className="text-[10px] text-rose-300/70">
                              {f}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Key Movers */}
                  {tf.key_movers && tf.key_movers.length > 0 && (
                    <div>
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1.5">
                        Key Movers
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {tf.key_movers.map((mover) => (
                          <Badge key={mover} variant="cyan">
                            {mover.replace(/_/g, ' ')}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Trader Signals */}
                  {tf.trader_signals && (
                    <TraderSignals signals={tf.trader_signals} />
                  )}

                  {/* Factor Contributions Table (collapsible) */}
                  {tf.factors && tf.factors.length > 0 && (
                    <div>
                      <button
                        className="flex items-center gap-1 text-[10px] text-slate-500 uppercase tracking-wider mb-1.5 hover:text-slate-400 transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowDetails((prev) => ({ ...prev, [key]: !prev[key] }));
                        }}
                      >
                        <span>Details</span>
                        <ChevronDown
                          className={`w-3 h-3 transition-transform duration-200 ${
                            detailsOpen ? 'rotate-180' : ''
                          }`}
                        />
                      </button>
                      {detailsOpen && (
                        <div className="space-y-1">
                          {tf.factors
                            .slice()
                            .sort((a, b) => Math.abs(b?.contribution ?? 0) - Math.abs(a?.contribution ?? 0))
                            .map((factor) => {
                              const contribution = factor?.contribution ?? 0;
                              const absPct = Math.min(Math.abs(contribution) * 500, 100);
                              return (
                                <div key={factor?.name} className="flex items-center gap-2 text-[10px]">
                                  <span className="text-slate-500 w-24 truncate">
                                    {(factor?.name ?? '').replace(/_/g, ' ')}
                                  </span>
                                  <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden relative">
                                    <div
                                      className={`absolute inset-y-0 rounded-full transition-all duration-300 ${
                                        contribution >= 0
                                          ? 'bg-emerald-500/60 left-1/2'
                                          : 'bg-rose-500/60 right-1/2'
                                      }`}
                                      style={{
                                        width: `${absPct / 2}%`,
                                        ...(contribution < 0 ? { right: '50%', left: 'auto' } : { left: '50%' }),
                                      }}
                                    />
                                    <div className="absolute inset-y-0 left-1/2 w-px bg-slate-600" />
                                  </div>
                                  <span
                                    className={`w-14 text-right font-mono ${
                                      contribution > 0 ? 'text-emerald-400' : contribution < 0 ? 'text-rose-400' : 'text-slate-500'
                                    }`}
                                  >
                                    {contribution >= 0 ? '+' : ''}{contribution.toFixed(4)}
                                  </span>
                                </div>
                              );
                            })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </GlassCard>
  );
}

function TraderSignals({ signals }: { signals: { wyckoff?: string[]; moneytaur?: string[]; zeroika?: string[] } }) {
  const sections = [
    { key: 'wyckoff', label: 'Wyckoff', items: signals?.wyckoff, variant: 'blue' as const },
    { key: 'moneytaur', label: 'Moneytaur', items: signals?.moneytaur, variant: 'orange' as const },
    { key: 'zeroika', label: 'Zeroika', items: signals?.zeroika, variant: 'violet' as const },
  ].filter((s) => s.items && s.items.length > 0);

  if (sections.length === 0) return null;

  return (
    <div>
      <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1.5">
        Trader Signals
      </div>
      <div className="space-y-2">
        {sections.map(({ key, label, items, variant }) => (
          <div key={key}>
            <div className="flex items-center gap-1.5 mb-1">
              <Badge variant={variant}>{label}</Badge>
            </div>
            <div className="space-y-0.5 pl-2 border-l border-white/[0.06]">
              {items?.map((signal, i) => (
                <div key={i} className="text-[10px] text-slate-400">
                  {signal}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
