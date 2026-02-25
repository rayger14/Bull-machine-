import { useState } from 'react';
import { AlertTriangle, Info, ShieldCheck } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import { useAppStore } from '../../stores/useAppStore';
import type { StressScenario, OracleData } from '../../api/types';

interface StressScenariosProps {
  scenarios: StressScenario[] | null | undefined;
  oracle?: OracleData | null;
}

// ── Educational content per scenario (from stress_simulator.py) ──────

const SCENARIO_INFO: Record<string, { explanation: string; engineImpact: string; threshold: string }> = {
  'VIX Spike': {
    explanation:
      'The CBOE Volatility Index (VIX) measures expected 30-day volatility in S&P 500 options. When VIX spikes above 2 standard deviations from its mean, equity markets are in panic mode. Historically, crypto sells off as institutions de-risk across all asset classes -- BTC correlation with equities jumps to 0.6+ during VIX events.',
    engineImpact:
      'VIX_Z feeds the CMI crisis_probability component (20% weight via vol_shock). When active, the dynamic threshold rises by ~0.1-0.2, blocking marginal signals. Emergency cap triggers if crisis_prob > 0.7.',
    threshold: 'VIX_Z > 2.0 (top 2.3% of history)',
  },
  'Dollar Surge': {
    explanation:
      'The US Dollar Index (DXY) measures the dollar against a basket of major currencies. A surging dollar (DXY_Z > 2\u03C3) means capital is flowing into the world\'s reserve currency -- a classic risk-off signal. BTC is denominated in USD, so dollar strength mechanically pushes BTC price down. Historically, BTC drops 3-5% in the 72h following a dollar surge.',
    engineImpact:
      'DXY_Z feeds risk_temperature (trend alignment component). High DXY_Z increases instability and raises the dynamic threshold, making the engine more selective about new entries.',
    threshold: 'DXY_Z > 2.0 (top 2.3% of history)',
  },
  'Oil Shock': {
    explanation:
      'Oil price shocks (WTI Z > 2\u03C3) signal global supply disruptions or geopolitical instability. Rising oil drives inflation expectations, pushes the Fed toward hawkish policy, and increases input costs across the economy. Crypto historically underperforms during oil shocks as liquidity tightens.',
    engineImpact:
      'OIL_Z is a macro indicator feeding the macro environment assessment. Extreme oil values contribute to overall instability scoring, which widens the dynamic threshold spread.',
    threshold: 'WTI_Z > 2.0 (top 2.3% of history)',
  },
  'Extreme Fear': {
    explanation:
      'The Crypto Fear & Greed Index drops below 15 (out of 100), indicating extreme market fear. This combines volatility, momentum, social media sentiment, BTC dominance, and Google Trends. At extreme fear, most retail traders have capitulated -- but historically, extreme fear is a contrarian buy signal (avg +4.2% over 168h).',
    engineImpact:
      'fear_greed_norm (0-1 scale) feeds risk_temperature at 15% weight. Extreme fear pushes risk_temp toward 0, which via the dynamic threshold formula raises the bar to ~0.63+ (very selective). The engine avoids catching falling knives but watches for Wyckoff accumulation events.',
    threshold: 'fear_greed_norm < 0.15 (Extreme Fear)',
  },
  'Correlation Storm': {
    explanation:
      'Multiple stress indicators fire simultaneously -- VIX elevated, dollar surging, and fear extreme. This is a systemic risk event where correlations across all asset classes spike toward 1.0. "There is no diversification in a crisis." These events are rare (2-4% of history) but produce the largest drawdowns.',
    engineImpact:
      'Multiple CMI components fire simultaneously: crisis_probability spikes, instability maxes out, risk_temperature drops to near zero. The dynamic threshold can reach 0.8+ (practically blocking all signals). Emergency cap halves position sizing. This is the engine\'s most defensive posture.',
    threshold: 'VIX_Z > 1.5 AND DXY_Z > 1.5 AND fear_greed_norm < 0.25',
  },
  'Bond Stress': {
    explanation:
      'Bond market stress (MOVE Index spiking + yield curve inversions) signals that the most sophisticated institutional investors are pricing in recession or systemic banking risk. Bond stress preceded every major crypto crash since 2020. The yield curve inversion has predicted the last 7 recessions.',
    engineImpact:
      'YIELD_CURVE feeds the macro environment. Inverted yield curve (negative) is a long-term bearish signal that increases the base instability score. MOVE Index (bond volatility) feeds stress_signals in crisis_probability at 60% weight.',
    threshold: 'MOVE_Z > 2.0 AND Yield Curve < -0.5',
  },
};

// ── Engine response text per scenario type ──────
const ENGINE_RESPONSES: Record<string, string> = {
  'VIX Spike': 'Threshold raised, sizing reduced 50%',
  'Dollar Surge': 'Regime shift to defensive, higher entry bar',
  'Oil Shock': 'Inflation hedge positioning, wider stops',
  'Extreme Fear': 'Emergency position cap, no new entries',
  'Correlation Storm': 'Diversification benefits reduced',
  'Bond Stress': 'Long-term risk elevated, shorter holds preferred',
};

export default function StressScenarios({ scenarios, oracle: _oracle }: StressScenariosProps) {
  const { expandedStress, setExpandedStress } = useAppStore();
  const [infoExpanded, setInfoExpanded] = useState('');

  if (!scenarios || scenarios.length === 0) return null;

  const active = scenarios.filter(s => s.active);
  const severityVariant = (s?: string) => {
    if (s === 'red' || s === 'high' || s === 'critical') return 'red';
    if (s === 'orange' || s === 'medium') return 'orange';
    if (s === 'yellow') return 'yellow';
    return 'green';
  };

  // Sort: active scenarios first (by probability descending), then inactive (by probability descending)
  const sorted = [...scenarios].sort((a, b) => {
    if (a.active && !b.active) return -1;
    if (!a.active && b.active) return 1;
    return ((b as { probability?: number }).probability ?? 0) - ((a as { probability?: number }).probability ?? 0);
  });

  return (
    <GlassCard>
      <div className="flex items-center gap-2 mb-2">
        <AlertTriangle className="w-4 h-4 text-amber-400" />
        <span className="text-xs text-slate-500 uppercase tracking-wider">Stress Scenarios</span>
        {active.length > 0 ? (
          <Badge variant="red">{active.length} Active</Badge>
        ) : (
          <Badge variant="green">All Clear</Badge>
        )}
      </div>
      <p className="text-[10px] text-slate-600 mb-3 leading-relaxed">
        The engine monitors 6 macro stress scenarios in real-time. When triggered, the CMI raises the dynamic threshold
        to filter out marginal signals. Scenarios activate when indicators exceed 2 standard deviations from their historical mean.
      </p>

      <div className="space-y-2">
        {sorted.map((sc) => {
          const name = sc.name ?? '';
          const isExpanded = expandedStress === name;
          const isInfoOpen = infoExpanded === name;
          const info = SCENARIO_INFO[name];
          const currentVals = sc.current_values;
          const engineResponse = ENGINE_RESPONSES[name];

          return (
            <div
              key={name}
              className={
                sc.active
                  ? 'border border-red-500/50 bg-red-500/10 p-3 rounded-lg'
                  : 'border border-gray-700/30 bg-gray-800/30 p-2 rounded-lg opacity-60'
              }
            >
              {/* Header row */}
              <div
                className="flex items-center justify-between cursor-pointer"
                onClick={() => setExpandedStress(isExpanded ? '' : name)}
              >
                <div className="flex items-center gap-2">
                  {sc.active ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                  ) : (
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-500/50" />
                  )}
                  <span className={sc.active ? 'text-sm text-slate-200 font-medium' : 'text-xs text-slate-500'}>{name}</span>
                  {sc.active && <Badge variant={severityVariant(sc.severity)}>{sc.severity}</Badge>}
                  {!sc.active && <Badge variant="green">clear</Badge>}
                </div>
                <div className="flex items-center gap-2">
                  {/* Show key metric inline */}
                  {currentVals && Object.keys(currentVals).length > 0 && (
                    <span className="text-[10px] font-mono text-slate-600">
                      {Object.entries(currentVals).map(([k, v]) => `${k}: ${v.toFixed(2)}`).join(' | ')}
                    </span>
                  )}
                  {info && (
                    <button
                      onClick={(e) => { e.stopPropagation(); setInfoExpanded(isInfoOpen ? '' : name); }}
                      className="p-0.5 hover:bg-white/[0.05] rounded transition-colors"
                    >
                      <Info className="w-3.5 h-3.5 text-slate-600 hover:text-slate-400" />
                    </button>
                  )}
                </div>
              </div>

              {/* Active scenario: engine response guidance */}
              {sc.active && engineResponse && (
                <div className="mt-2 text-xs text-amber-300/90 bg-amber-500/[0.06] border border-amber-500/15 rounded-md px-2.5 py-1.5">
                  <span className="font-medium text-amber-400">Engine Response:</span>{' '}
                  {engineResponse}
                </div>
              )}

              {/* Educational info panel */}
              {isInfoOpen && info && (
                <div className="mt-3 p-3 bg-white/[0.02] border border-white/[0.06] rounded-lg text-xs space-y-2">
                  <p className="text-slate-400 leading-relaxed">{info.explanation}</p>
                  <div className="border-t border-white/[0.04] pt-2">
                    <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-1">Engine impact</div>
                    <p className="text-slate-500 leading-relaxed">{info.engineImpact}</p>
                  </div>
                  <div className="text-[10px] text-slate-600">
                    Trigger: <span className="text-slate-400 font-mono">{info.threshold}</span>
                  </div>
                </div>
              )}

              {/* Expanded data */}
              {isExpanded && (
                <div className="mt-3 text-xs space-y-2">
                  {sc.description && <div className="text-slate-400">{sc.description}</div>}
                  {sc.recommendation && <div className="text-amber-400/80">{sc.recommendation}</div>}

                  {/* Current trigger values */}
                  {currentVals && Object.keys(currentVals).length > 0 && (
                    <div className="flex flex-wrap gap-3 mt-1">
                      {Object.entries(currentVals).map(([k, v]) => {
                        const absV = Math.abs(v);
                        const isHot = absV > 2.0 || (k === 'fear_greed_norm' && v < 0.15);
                        return (
                          <div key={k} className={`rounded-lg px-2 py-1 ${isHot ? 'bg-rose-500/[0.06] border border-rose-500/20' : 'bg-white/[0.03]'}`}>
                            <span className="text-slate-600">{k}: </span>
                            <span className={`font-mono ${isHot ? 'text-rose-400' : 'text-slate-300'}`}>{v.toFixed(4)}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}

                  {sc.historical && (
                    <div className="mt-2 space-y-2">
                      <div className="text-[10px] text-slate-600 uppercase tracking-wider">Historical Analysis (2018-2024)</div>
                      <div className="grid grid-cols-4 gap-2">
                        <div><span className="text-slate-600">24h:</span> <span className="text-slate-300 font-mono">{sc.historical.avg_return_24h?.toFixed(2)}%</span></div>
                        <div><span className="text-slate-600">72h:</span> <span className="text-slate-300 font-mono">{sc.historical.avg_return_72h?.toFixed(2)}%</span></div>
                        <div><span className="text-slate-600">168h:</span> <span className="text-slate-300 font-mono">{sc.historical.avg_return_168h?.toFixed(2)}%</span></div>
                        <div><span className="text-slate-600">Worst 24h:</span> <span className="text-rose-400 font-mono">{sc.historical.worst_return_24h?.toFixed(2)}%</span></div>
                      </div>
                      <div className="grid grid-cols-4 gap-2">
                        <div><span className="text-slate-600">MaxDD:</span> <span className="text-rose-400 font-mono">{sc.historical.max_drawdown_pct?.toFixed(2)}%</span></div>
                        <div><span className="text-slate-600">Episodes:</span> <span className="text-slate-300 font-mono">{sc.historical.num_episodes ?? '--'}</span></div>
                        <div><span className="text-slate-600">Bars:</span> <span className="text-slate-300 font-mono">{sc.historical.occurrences?.toLocaleString() ?? '--'}</span></div>
                        <div><span className="text-slate-600">Avg Dur:</span> <span className="text-slate-300 font-mono">{sc.historical.avg_duration_hours?.toFixed(0)}h</span></div>
                      </div>
                      {sc.historical.estimated_win_rate != null && (
                        <div className="text-slate-500">
                          Est. win rate: <span className="text-slate-300 font-mono">{sc.historical.estimated_win_rate.toFixed(1)}%</span>
                          {sc.historical.pct_of_history != null && (
                            <span className="ml-3">History coverage: <span className="font-mono">{sc.historical.pct_of_history.toFixed(1)}%</span></span>
                          )}
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
