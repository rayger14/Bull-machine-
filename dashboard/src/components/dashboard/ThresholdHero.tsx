import React, { useState } from 'react';
import { Info } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Gauge from '../ui/Gauge';
import CMIDetailPanel from './CMIDetailPanel';
import { useAppStore } from '../../stores/useAppStore';
import type { Heartbeat, OracleData } from '../../api/types';

interface ThresholdHeroProps {
  hb: Heartbeat;
  oracle?: OracleData;
}

function trafficLightDot(value: number, invert: boolean): React.ReactElement {
  let color: string;
  if (invert) {
    // For Trend Health (risk_temp): high = good = green
    if (value >= 0.6) color = 'bg-emerald-400';
    else if (value >= 0.3) color = 'bg-amber-400';
    else color = 'bg-rose-400';
  } else {
    // For Market Chop and Crash Risk: high = bad = red
    if (value >= 0.6) color = 'bg-rose-400';
    else if (value >= 0.3) color = 'bg-amber-400';
    else color = 'bg-emerald-400';
  }
  return <div className={`w-2.5 h-2.5 rounded-full ${color} shrink-0`} />;
}

export default function ThresholdHero({ hb, oracle }: ThresholdHeroProps) {
  const [showHowItWorks, setShowHowItWorks] = useState(false);
  const { showRiskDetail, showInstabDetail, showCrisisDetail, toggleRiskDetail, toggleInstabDetail, toggleCrisisDetail } = useAppStore();

  const riskTemp = hb.risk_temp ?? 0;
  const instability = hb.instability ?? 0;
  const crisisProb = hb.crisis_prob ?? 0;
  const threshold = hb.threshold ?? 0;

  const riskDesc = (v: number) => {
    if (v < 0.2) return 'Deep Bear -- threshold near max';
    if (v < 0.4) return 'Bear -- high selectivity';
    if (v < 0.6) return 'Neutral -- moderate threshold';
    if (v < 0.8) return 'Bull -- threshold lowering';
    return 'Strong Bull -- near base threshold';
  };

  const instabDesc = (v: number) => {
    if (v < 0.2) return 'Very Stable';
    if (v < 0.4) return 'Stable';
    if (v < 0.6) return 'Moderate Chop';
    if (v < 0.8) return 'Choppy';
    return 'Extreme Chop';
  };

  const crisisDesc = (v: number) => {
    if (v < 0.1) return 'Normal -- minimal penalty';
    if (v < 0.3) return 'Elevated -- ~4-12% penalty';
    if (v < 0.5) return 'High -- ~12-20% penalty';
    if (v < 0.7) return 'Severe -- ~20-28% penalty';
    return 'CRITICAL -- 50% sizing cap';
  };

  const cmi = hb.cmi_breakdown;
  const tc = cmi?.threshold_config;

  return (
    <GlassCard accent className="space-y-4">
      {/* Dynamic Threshold Value */}
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs text-slate-500 uppercase tracking-wider">Dynamic Threshold</div>
          <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-violet-400 bg-clip-text text-transparent">
            {threshold.toFixed(3)}
          </div>
        </div>
        <div className="text-right text-[10px] text-slate-600 font-mono leading-relaxed">
          {tc ? (
            <>
              {tc.base_threshold?.toFixed(2)} + (1 - {riskTemp.toFixed(2)}) * {tc.temp_range?.toFixed(2)}<br/>
              + {instability.toFixed(2)} * {tc.instab_range?.toFixed(2)}
            </>
          ) : (
            <>
              base + (1 - risk_temp) * temp_range<br/>
              + instability * instab_range
            </>
          )}
        </div>
      </div>

      {/* How This Works - collapsible educational section */}
      <div>
        <button
          onClick={() => setShowHowItWorks(!showHowItWorks)}
          className="flex items-center gap-1.5 text-[10px] text-slate-600 hover:text-slate-400 transition-colors"
        >
          <Info className="w-3 h-3" />
          {showHowItWorks ? 'Hide explanation' : 'How this works'}
        </button>
        {showHowItWorks && (
          <div className="mt-2 p-3 bg-white/[0.02] border border-white/[0.06] rounded-xl text-xs text-slate-400 space-y-2 leading-relaxed">
            <p>
              The <span className="font-semibold text-slate-300">Dynamic Threshold</span> is the minimum fusion score a signal must achieve to be accepted. It adapts in real-time based on three independent market condition measures:
            </p>
            <p>
              <span className="font-semibold text-slate-300">Risk Temperature</span> (updates hourly, reflects macro regime over weeks) — Measures overall trend health via EMA alignment (45%), trend strength/ADX (25%), Fear &amp; Greed sentiment (15%), and drawdown recovery (15%). When risk_temp is high (&gt;0.7), the threshold drops, allowing more trades. When low (&lt;0.3, bear market), the threshold rises sharply — Variant A adds +48% bear penalty.
            </p>
            <p>
              <span className="font-semibold text-slate-300">Instability</span> (updates hourly, reflects local noise over hours-days) — Measures market choppiness via Chop Index (35%), ADX weakness (25%), wick rejection intensity (20%), and volume extremes (20%). High instability = choppy conditions where trades get stopped out, so the threshold rises to filter marginal signals.
            </p>
            <p>
              <span className="font-semibold text-slate-300">Crisis Probability</span> (updates hourly, can spike within 1 bar) — Detects acute stress: drawdown severity (60%), realized volatility shocks (20%), and extreme fear sentiment (20%). Above 0.70, an emergency cap halves position sizing. This is the system's circuit breaker.
            </p>
            <p>
              The threshold ranges from ~0.18 (strong bull, clean trend) to ~0.75 (crisis). This 4.5x variation surgically filters adverse-regime trades — most losing trades pass with thin margins over the threshold, so raising the bear penalty kills marginal entries without affecting high-conviction bull trades.
            </p>
          </div>
        )}
      </div>

      {/* Oracle posture description */}
      {oracle?.engine_status?.posture_description && (
        <p className="text-sm text-gray-300 mb-4 italic">
          {oracle.engine_status.posture_description}
        </p>
      )}

      {/* 3 Gauges */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <div className="flex items-center justify-center gap-1.5 mb-1">
            {trafficLightDot(riskTemp, true)}
          </div>
          <Gauge
            value={riskTemp}
            label="Trend Health"
            description={riskDesc(riskTemp)}
            onClick={toggleRiskDetail}
            colorStops={['bg-rose-500', 'bg-orange-400', 'bg-amber-400', 'bg-emerald-400', 'bg-emerald-500']}
          />
          <div className="text-[9px] text-slate-700 text-center mt-0.5">Macro regime | 2-4 week cycles</div>
          {showRiskDetail && cmi?.risk_temp_weights && (
            <CMIDetailPanel
              title="Trend Health"
              value={riskTemp}
              weights={cmi.risk_temp_weights}
              components={cmi.risk_temp_components}
              rawFeatures={cmi.raw_features}
            />
          )}
        </div>
        <div>
          <div className="flex items-center justify-center gap-1.5 mb-1">
            {trafficLightDot(instability, false)}
          </div>
          <Gauge
            value={instability}
            label="Market Chop"
            description={instabDesc(instability)}
            onClick={toggleInstabDetail}
          />
          <div className="text-[9px] text-slate-700 text-center mt-0.5">Local noise | hours to days</div>
          {showInstabDetail && cmi?.instability_weights && (
            <CMIDetailPanel
              title="Market Chop"
              value={instability}
              weights={cmi.instability_weights}
              components={cmi.instability_components}
              rawFeatures={cmi.raw_features}
            />
          )}
        </div>
        <div>
          <div className="flex items-center justify-center gap-1.5 mb-1">
            {trafficLightDot(crisisProb, false)}
          </div>
          <Gauge
            value={crisisProb}
            label="Crash Risk"
            description={crisisDesc(crisisProb)}
            onClick={toggleCrisisDetail}
            colorStops={['bg-emerald-500', 'bg-emerald-400', 'bg-amber-400', 'bg-orange-400', 'bg-rose-500']}
          />
          <div className="text-[9px] text-slate-700 text-center mt-0.5">Acute stress | can spike in 1 bar</div>
          {showCrisisDetail && cmi?.crisis_weights && (
            <CMIDetailPanel
              title="Crash Risk"
              value={crisisProb}
              weights={cmi.crisis_weights}
              components={cmi.crisis_components}
              rawFeatures={cmi.raw_features}
            />
          )}
        </div>
      </div>
    </GlassCard>
  );
}
