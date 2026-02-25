import { useState } from 'react';
import { ArrowRight, ArrowDown, TrendingUp, ChevronDown } from 'lucide-react';
import GlassCard from '../ui/GlassCard';
import Badge from '../ui/Badge';
import type { CapitalFlowsData, OracleData } from '../../api/types';

interface CapitalFlowsProps {
  flows: CapitalFlowsData | undefined;
  oracle?: OracleData | null;
}

/* ------------------------------------------------------------------ */
/*  Educational content for each asset-class node                      */
/* ------------------------------------------------------------------ */
const NODE_EDUCATION: Record<string, { name: string; description: string; cryptoRelation: string; watchFor: string }> = {
  crypto: {
    name: 'Bitcoin / Crypto',
    description: 'The primary risk-on digital asset. BTC leads the crypto market and is increasingly correlated with tech equities.',
    cryptoRelation: 'This IS the asset we trade. All other flows ultimately affect BTC through liquidity, sentiment, and institutional allocation.',
    watchFor: 'Stablecoin dominance rising (fear), BTC dominance falling (alt rotation), and macro liquidity tightening.',
  },
  dollar: {
    name: 'US Dollar (DXY)',
    description: 'The US Dollar Index measures the dollar against a basket of 6 major currencies. The world\'s reserve currency and global liquidity benchmark.',
    cryptoRelation: 'BTC and USD are inversely correlated. A strong dollar drains liquidity from risk assets including crypto. When DXY weakens, capital flows into alternative stores of value like BTC and gold.',
    watchFor: 'DXY breaking below 0\u03C3 signals a liquidity tailwind for BTC. Fed rate decisions and Treasury issuance are the primary catalysts. Watch for divergences between DXY and BTC.',
  },
  gold: {
    name: 'Gold (XAU)',
    description: 'The traditional safe haven and inflation hedge. Gold competes with BTC for the "store of value" narrative.',
    cryptoRelation: 'Gold and BTC share the "hard money" thesis but diverge during acute stress. When gold surges on geopolitical fear, BTC initially sells off (liquidity crunch) before following gold higher. Long-term, they\'re positively correlated.',
    watchFor: 'Gold breaking to new highs while BTC lags creates a convergence trade opportunity. Gold strength + dollar weakness = optimal BTC environment.',
  },
  equities: {
    name: 'Equities (VIX)',
    description: 'The VIX measures implied volatility of S&P 500 options \u2014 essentially equity market fear. Low VIX = complacency, high VIX = panic.',
    cryptoRelation: 'BTC increasingly trades as a risk asset correlated with Nasdaq. VIX spikes above 2\u03C3 trigger institutional de-risking across ALL portfolios including crypto. Low VIX environments are the sweet spot for BTC accumulation.',
    watchFor: 'VIX above 25 = headwind, above 35 = potential capitulation buy. VIX below 15 with declining DXY = ideal crypto conditions.',
  },
  bonds: {
    name: 'Bonds (Yield Curve)',
    description: 'The yield curve (10Y - 2Y Treasury spread) signals economic expectations. Inversion (negative spread) historically precedes recessions.',
    cryptoRelation: 'Yield curve inversion creates credit tightening that eventually hits crypto through reduced institutional risk appetite. However, the initial inversion often coincides with BTC rallies \u2014 the damage comes 6-18 months later when the curve re-steepens.',
    watchFor: 'Yield curve deeply inverted (< -0.5%) = stress building. Re-steepening from inverted = recession approaching, historically the worst period for risk assets. Positive curve + rate cuts = most bullish macro backdrop for BTC.',
  },
  oil: {
    name: 'Oil (WTI Crude)',
    description: 'West Texas Intermediate crude oil \u2014 a proxy for inflation expectations and global economic activity.',
    cryptoRelation: 'Oil connects crypto to the energy/inflation cycle. Rising oil drives inflation expectations, which drives Fed hawkishness, which tightens liquidity, which hurts BTC. The stagflation scenario (high oil + strong dollar) is the worst macro environment for crypto.',
    watchFor: 'Oil above 1\u03C3 combined with strong dollar = stagflation risk, very bearish for crypto. Oil falling with weakening dollar = Goldilocks for risk assets.',
  },
  stablecoins: {
    name: 'Stablecoins (USDT + USDC Dominance)',
    description: 'The combined market share of USDT and USDC in total crypto market cap. Rising dominance means traders are moving to the sidelines.',
    cryptoRelation: 'Stablecoin dominance is a direct fear gauge for crypto. When traders sell BTC for USDT/USDC, dominance rises. Falling dominance means stablecoin holders are buying back into BTC \u2014 bullish for price.',
    watchFor: 'Dominance above 8% = elevated fear, capital is on the sidelines. Below 6% = full risk-on deployment. Rapid rises signal panic selling.',
  },
  altcoins: {
    name: 'Altcoins (BTC Dominance)',
    description: 'BTC dominance measures Bitcoin\'s share of total crypto market cap. It reflects capital rotation between BTC and altcoins.',
    cryptoRelation: 'Rising BTC dominance (>55%) = risk-off within crypto, capital fleeing to the "safe" asset. Falling dominance (<50%) = alt season, speculative capital rotating into higher-beta altcoins. The engine only trades BTC, but dominance shifts signal regime changes.',
    watchFor: 'BTC dominance falling below 50% = capital rotating to alts (risk-on crypto). Rising above 55% = flight to quality within crypto. Extreme readings (>60% or <40%) often reverse.',
  },
};

/* ------------------------------------------------------------------ */
/*  Educational content for each flow edge                             */
/* ------------------------------------------------------------------ */
const EDGE_EDUCATION: Record<string, { mechanism: string; historicalPattern: string; engineUse: string }> = {
  liquidity_drain: {
    mechanism: 'A strong US dollar (DXY > 1\u03C3) means global dollar liquidity is tightening. Since crypto is denominated in USD and funded by dollar-based leverage, a rising dollar directly reduces crypto buying power and forces deleveraging.',
    historicalPattern: 'In every major DXY rally (2014-2015, 2018, 2022), BTC experienced 50-80% drawdowns. The correlation is strongest during sustained dollar strength lasting 3+ months.',
    engineUse: 'The engine penalizes this through risk_temperature \u2014 when DXY is elevated, the dynamic threshold rises, requiring higher-confidence signals to enter. This is a core component of CMI macro intelligence.',
  },
  liquidity_flow: {
    mechanism: 'A weakening dollar (DXY < -1\u03C3) means global liquidity is expanding. Cheaper dollars make risk assets more attractive, and foreign capital can buy BTC at a relative discount. This is the "everything rally" catalyst.',
    historicalPattern: 'The 2020-2021 bull market coincided with the weakest dollar since 2018. BTC rallied 10x during the sustained DXY decline. Dollar weakness + QE = the most powerful crypto tailwind.',
    engineUse: 'Weak DXY lowers risk_temperature through favorable trend alignment, making the engine more permissive. Combined with low VIX, this creates the most favorable conditions for trade deployment.',
  },
  stagflation: {
    mechanism: 'When oil spikes (>1\u03C3) AND the dollar strengthens (>0.5\u03C3) simultaneously, it signals stagflation \u2014 rising costs with tightening monetary conditions. This double squeeze is the worst macro scenario for risk assets.',
    historicalPattern: 'Stagflation episodes (1970s, 2022) are the most destructive for speculative assets. BTC has limited history but showed 70%+ drawdowns during the 2022 oil spike + dollar rally combination.',
    engineUse: 'The engine treats this as a crisis-level signal. Both oil_z and dxy_z feed into crisis_prob, which can trigger the 50% emergency sizing cap and dramatically raise thresholds.',
  },
  flight_to_safety: {
    mechanism: 'When equity market fear spikes (VIX > 1\u03C3), institutional capital rotates from equities into gold as a traditional safe haven. This represents risk-off behavior across all asset classes.',
    historicalPattern: 'Gold typically outperforms during VIX spikes, gaining 3-8% during acute stress events. BTC initially correlates with equities (sells off) but can decouple and follow gold higher in the recovery phase.',
    engineUse: 'Flight to safety behavior raises instability scores. The engine tracks gold strength as a secondary confirmation of stress, particularly when combined with dollar strength.',
  },
  bond_stress: {
    mechanism: 'Yield curve inversion (10Y - 2Y < -0.3%) signals that bond markets expect recession. Short-term rates exceeding long-term rates means the market believes the Fed has overtightened.',
    historicalPattern: 'Every US recession since 1970 was preceded by yield curve inversion (12-18 month lead time). Crypto didn\'t exist for most, but the 2019 inversion preceded the 2020 crash, and the 2022 inversion preceded the crypto winter.',
    engineUse: 'Bond stress contributes to crisis_prob scoring. Deep inversion increases the dynamic threshold, making the engine more selective. The engine doesn\'t try to time the recession \u2014 it reduces exposure during the warning period.',
  },
  credit_contagion: {
    mechanism: 'When yield curve inversion deepens (<-0.5%) AND VIX spikes (>1.5\u03C3), it signals systemic financial stress. Bond and equity markets are BOTH flashing danger \u2014 this is the "everything is breaking" scenario.',
    historicalPattern: 'Credit contagion events (2008, March 2020) cause 30-50%+ crypto drawdowns within days. These are the moments when correlated selling overwhelms all fundamental analysis. Cash is king.',
    engineUse: 'This triggers the highest crisis_prob readings. The emergency cap kicks in (50% sizing), position limits drop to 1, and the dynamic threshold reaches maximum restrictiveness (~0.75). The engine essentially goes to cash.',
  },
  risk_appetite: {
    mechanism: 'When VIX drops below its mean (< 0\u03C3), it signals complacency and risk-seeking behavior. Low volatility compresses option premiums, which frees up capital for speculative assets including crypto.',
    historicalPattern: 'Sustained low VIX periods (VIX < 15) historically coincide with crypto bull runs. The 2017, 2020-2021, and 2024 rallies all occurred during extended low-volatility equity regimes.',
    engineUse: 'Low VIX lowers risk_temperature, making the engine more permissive with entries. This is one of the most reliable macro tailwinds \u2014 the engine deploys more capital when equity markets are calm.',
  },
  crypto_rotation: {
    mechanism: 'When BTC dominance drops below 50%, speculative capital is flowing from BTC into altcoins. This represents peak risk appetite within crypto \u2014 traders are reaching for higher beta and leverage.',
    historicalPattern: 'BTC dominance below 50% has historically coincided with alt season peaks, often preceding major corrections. The "alt season to BTC crash" pipeline is well-documented: peak speculation \u2192 leverage buildup \u2192 cascade liquidation.',
    engineUse: 'The engine doesn\'t trade altcoins, but falling BTC dominance raises instability scores slightly. It signals that the crypto market is becoming more speculative, which increases the probability of sharp reversals.',
  },
  stablecoin_flight: {
    mechanism: 'When stablecoin dominance exceeds 8%, it means a disproportionate amount of crypto market cap is parked in stablecoins. Traders have already sold into safety \u2014 dry powder is sitting on the sidelines.',
    historicalPattern: 'High stablecoin dominance is a contrarian indicator at extremes. Above 10% has historically marked bottoms (maximum fear = maximum dry powder for recovery). Between 6-8% is neutral.',
    engineUse: 'Elevated stablecoin dominance increases risk_temperature through the drawdown component. The engine becomes more selective but also recognizes that extreme readings (>10%) may signal capitulation \u2014 a potential accumulation phase.',
  },
  fear_exit: {
    mechanism: 'When Fear & Greed drops below 15 (extreme fear), retail and institutional capital is leaving crypto entirely \u2014 not just rotating to stablecoins but withdrawing from the ecosystem. This is full capitulation.',
    historicalPattern: 'F&G below 15 has occurred during every major crypto bottom (March 2020: F&G=8, June 2022: F&G=6, Jan 2023: F&G=12). These readings are rare and historically mark the best risk-reward entry points for long-term holders.',
    engineUse: 'Extreme fear feeds directly into crisis_prob through the sentiment_extreme component (20% weight). The engine raises thresholds dramatically but also recognizes that extreme fear readings often coincide with Wyckoff Selling Climax events \u2014 potential accumulation signals.',
  },
};

/* ------------------------------------------------------------------ */
/*  Color mapping for node accent borders                              */
/* ------------------------------------------------------------------ */
const nodeAccentColor: Record<string, string> = {
  crypto: 'border-amber-500/30',
  dollar: 'border-emerald-500/30',
  gold: 'border-yellow-500/30',
  equities: 'border-violet-500/30',
  bonds: 'border-blue-500/30',
  oil: 'border-rose-500/30',
  stablecoins: 'border-cyan-500/30',
  altcoins: 'border-orange-500/30',
};

const nodeAccentBg: Record<string, string> = {
  crypto: 'bg-amber-500/5',
  dollar: 'bg-emerald-500/5',
  gold: 'bg-yellow-500/5',
  equities: 'bg-violet-500/5',
  bonds: 'bg-blue-500/5',
  oil: 'bg-rose-500/5',
  stablecoins: 'bg-cyan-500/5',
  altcoins: 'bg-orange-500/5',
};

const nodeAccentText: Record<string, string> = {
  crypto: 'text-amber-400',
  dollar: 'text-emerald-400',
  gold: 'text-yellow-400',
  equities: 'text-violet-400',
  bonds: 'text-blue-400',
  oil: 'text-rose-400',
  stablecoins: 'text-cyan-400',
  altcoins: 'text-orange-400',
};

/* ------------------------------------------------------------------ */
/*  Helpers (unchanged from original)                                  */
/* ------------------------------------------------------------------ */
const stateVariant = (state?: string): 'orange' | 'violet' | 'red' | 'green' | 'cyan' | 'neutral' => {
  if (state === 'hot') return 'orange';
  if (state === 'breakout') return 'violet';
  if (state === 'spike' || state === 'inverted') return 'red';
  if (state === 'flight') return 'orange';
  if (state === 'calm' || state === 'cool' || state === 'steepening') return 'green';
  if (state === 'rotation' || state === 'suppressed' || state === 'breakdown') return 'cyan';
  return 'neutral';
};

const statusColor = (status?: string): string => {
  if (status === 'bullish') return 'text-emerald-400';
  if (status === 'bearish') return 'text-rose-400';
  return 'text-slate-300';
};

/** Generate a one-line live interpretation of what this node's value means for BTC */
function liveInterpretation(key: string, node: { value?: string; state?: string; status?: string }): string {
  const state = node?.state ?? 'neutral';
  const status = node?.status ?? 'neutral';

  switch (key) {
    case 'dollar':
      if (status === 'bearish') return 'Strong dollar is draining global liquidity \u2014 headwind for BTC. Engine raises threshold.';
      if (status === 'bullish') return 'Weak dollar expanding global liquidity \u2014 tailwind for BTC. Engine lowers threshold.';
      return 'Dollar is range-bound \u2014 no macro signal for BTC.';
    case 'oil':
      if (state === 'hot') return 'Oil is elevated \u2014 inflation fears rising, Fed likely to stay hawkish. Bearish for BTC via tighter liquidity.';
      if (state === 'cool') return 'Oil is depressed \u2014 inflation cooling, dovish Fed expected. Bullish for risk assets including BTC.';
      return 'Oil is in normal range \u2014 no inflation signal for BTC.';
    case 'gold':
      if (state === 'flight') return 'Gold surging on safe-haven demand \u2014 institutions are de-risking. BTC typically sells off first, then follows gold higher.';
      return 'Gold is stable \u2014 no flight-to-safety signal active.';
    case 'equities':
      if (state === 'spike') return 'VIX spiking \u2014 equity panic triggers institutional de-risking across all assets including crypto.';
      if (state === 'calm') return 'VIX suppressed \u2014 risk appetite is high, favorable for BTC entries. Engine becomes more permissive.';
      return 'VIX in neutral range \u2014 equity markets not signaling stress.';
    case 'bonds':
      if (state === 'inverted') return 'Yield curve inverted \u2014 recession signal. Credit tightening ahead, engine raises crisis probability.';
      if (state === 'steepening') return 'Yield curve healthy \u2014 economic optimism supports risk-on assets like BTC.';
      return 'Yield curve is flat \u2014 no strong macro signal.';
    case 'stablecoins':
      if (state === 'breakout') return 'High stablecoin dominance \u2014 capital is on the sidelines. Contrarian signal: dry powder available for recovery.';
      if (state === 'breakdown') return 'Low stablecoin dominance \u2014 full risk-on, capital is deployed. Late-cycle signal.';
      return 'Stablecoin dominance is normal \u2014 balanced positioning.';
    case 'altcoins':
      if (state === 'rotation') return 'Capital rotating from BTC to alts \u2014 speculative risk appetite is high. Late-cycle warning.';
      if (state === 'suppressed') return 'BTC dominance rising \u2014 flight to quality within crypto. Risk-off within the ecosystem.';
      return 'BTC/alt balance is normal \u2014 no rotation signal.';
    case 'crypto':
      return `Live BTC price. All other flows converge here through the engine\u2019s CMI threshold.`;
    default:
      return '';
  }
}

const directionVariant = (direction?: string): 'cyan' | 'red' => {
  if (direction === 'flow') return 'cyan';
  return 'red';
};

const nodeIcons: Record<string, string> = {
  crypto: 'BTC',
  dollar: 'USD',
  gold: 'Au',
  equities: 'EQ',
  bonds: 'BD',
  oil: 'OIL',
  stablecoins: 'SC',
  altcoins: 'ALT',
};

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
/** Classify an edge as bullish or bearish for BTC based on direction and destination */
function classifyEdgeForBtc(edge: { direction?: string; to?: string; from?: string }): 'bullish' | 'bearish' | 'neutral' {
  // "flow" direction toward crypto = bullish; "drain" from crypto = bearish
  if (edge.direction === 'flow' && edge.to === 'crypto') return 'bullish';
  if (edge.direction === 'drain' && edge.from === 'crypto') return 'bearish';
  // Risk appetite / liquidity flow = bullish; flight to safety / stress = bearish
  if (edge.direction === 'flow') return 'bullish';
  if (edge.direction === 'drain') return 'bearish';
  return 'neutral';
}

export default function CapitalFlows({ flows, oracle: _oracle }: CapitalFlowsProps) {
  const [expandedNode, setExpandedNode] = useState<string>('');
  const [expandedEdge, setExpandedEdge] = useState<string>('');

  if (!flows) return null;

  const nodes = flows.nodes;
  const edges = flows.edges;

  if (!nodes || Object.keys(nodes).length === 0) return null;

  const activeEdges = edges
    ? Object.entries(edges).filter(([, e]) => e?.active)
    : [];

  // Classify active flows for the summary
  const bullishFlows = activeEdges.filter(([, e]) => classifyEdgeForBtc(e ?? {}) === 'bullish').length;
  const bearishFlows = activeEdges.filter(([, e]) => classifyEdgeForBtc(e ?? {}) === 'bearish').length;

  const toggleNode = (key: string) => {
    setExpandedNode((prev) => (prev === key ? '' : key));
  };

  const toggleEdge = (key: string) => {
    setExpandedEdge((prev) => (prev === key ? '' : key));
  };

  return (
    <GlassCard>
      <div className="flex items-center gap-2 mb-2">
        <TrendingUp className="w-4 h-4 text-cyan-400" />
        <span className="text-xs text-slate-500 uppercase tracking-wider">
          Capital Flows -- Cycle of Money
        </span>
        {activeEdges.length > 0 && (
          <Badge variant="cyan">{activeEdges.length} Active</Badge>
        )}
      </div>

      {/* Flow summary sentence */}
      {activeEdges.length > 0 && (
        <p className="text-sm text-gray-200 mb-3">
          {bullishFlows > bearishFlows
            ? `Capital flowing INTO crypto -- ${bullishFlows} bullish flow${bullishFlows !== 1 ? 's' : ''}, ${bearishFlows} bearish`
            : bearishFlows > bullishFlows
              ? `Capital flowing OUT of crypto -- ${bearishFlows} bearish flow${bearishFlows !== 1 ? 's' : ''}, ${bullishFlows} bullish`
              : `Capital flows mixed -- ${bullishFlows} bullish, ${bearishFlows} bearish`
          }
        </p>
      )}

      <p className="text-[10px] text-slate-600 mb-4">
        How money moves between asset classes in real-time (daily/weekly macro signals).
        Active flows show where institutional capital is rotating right now.
        <span className="text-slate-500 ml-1">Click any node or flow to learn more.</span>
      </p>

      {/* Node Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-2">
        {Object.entries(nodes).map(([key, node]) => (
          <div
            key={key}
            role="button"
            tabIndex={0}
            onClick={() => toggleNode(key)}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleNode(key); }}
            className={`rounded-xl border p-3 flex flex-col gap-1 cursor-pointer transition-all duration-300 select-none ${
              expandedNode === key
                ? `${nodeAccentBg[key] ?? 'bg-white/[0.04]'} ${nodeAccentColor[key] ?? 'border-white/[0.12]'} shadow-lg`
                : 'bg-white/[0.02] border-white/[0.06] hover:bg-white/[0.04] hover:border-white/[0.10]'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-slate-600 uppercase">
                {nodeIcons[key] ?? key.slice(0, 3).toUpperCase()}
              </span>
              <div className="flex items-center gap-1">
                <Badge variant={stateVariant(node?.state)}>
                  {node?.state ?? 'neutral'}
                </Badge>
                <ChevronDown
                  className={`w-3 h-3 text-slate-500 transition-transform duration-300 ${
                    expandedNode === key ? 'rotate-180' : ''
                  }`}
                />
              </div>
            </div>
            <div className="text-xs text-slate-400 truncate">
              {node?.label ?? key}
            </div>
            <div className={`text-base font-bold font-mono ${statusColor(node?.status)}`}>
              {node?.value ?? '--'}
            </div>
            {liveInterpretation(key, node ?? {}) && (
              <p className="text-[9px] leading-tight text-slate-500 mt-1 line-clamp-2">
                {liveInterpretation(key, node ?? {})}
              </p>
            )}
          </div>
        ))}
      </div>

      {/* Expanded Node Education Panel */}
      {expandedNode && NODE_EDUCATION[expandedNode] && (
        <div
          className={`mb-4 rounded-2xl border backdrop-blur-xl p-4 transition-all duration-300 animate-in fade-in slide-in-from-top-2 ${
            nodeAccentColor[expandedNode] ?? 'border-white/[0.10]'
          } ${nodeAccentBg[expandedNode] ?? 'bg-white/[0.03]'}`}
        >
          <div className="flex items-center gap-2 mb-3">
            <span className={`text-sm font-bold ${nodeAccentText[expandedNode] ?? 'text-slate-300'}`}>
              {NODE_EDUCATION[expandedNode].name}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); setExpandedNode(''); }}
              className="ml-auto text-slate-500 hover:text-slate-300 text-xs"
            >
              close
            </button>
          </div>

          <div className="space-y-3">
            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                What is it
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {NODE_EDUCATION[expandedNode].description}
              </p>
            </div>

            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                Relationship to BTC
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {NODE_EDUCATION[expandedNode].cryptoRelation}
              </p>
            </div>

            <div>
              <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                What to Watch
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {NODE_EDUCATION[expandedNode].watchFor}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Active Flows */}
      {activeEdges.length > 0 && (
        <div>
          <div className="text-[10px] text-slate-600 uppercase tracking-wider mb-2">
            Active Flows
          </div>
          <div className="space-y-2">
            {activeEdges.map(([key, edge]) => {
              const fromNode = nodes[edge?.from ?? ''];
              const toNode = nodes[edge?.to ?? ''];
              const strength = edge?.strength ?? 0;
              const isExpanded = expandedEdge === key;
              const edgeEdu = EDGE_EDUCATION[key];
              const isDrain = edge?.direction === 'drain';
              const btcClassification = classifyEdgeForBtc(edge ?? {});

              return (
                <div key={key}>
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => toggleEdge(key)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleEdge(key); }}
                    className={`rounded-xl border p-3 cursor-pointer transition-all duration-300 select-none ${
                      isExpanded
                        ? isDrain
                          ? 'bg-rose-500/[0.04] border-rose-500/20 shadow-lg'
                          : 'bg-cyan-500/[0.04] border-cyan-500/20 shadow-lg'
                        : isDrain
                          ? 'bg-white/[0.02] border-rose-500/15 hover:bg-rose-500/[0.03] hover:border-rose-500/25'
                          : 'bg-white/[0.02] border-emerald-500/15 hover:bg-emerald-500/[0.03] hover:border-emerald-500/25'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`text-xs font-medium ${isDrain ? 'text-rose-300' : 'text-emerald-300'}`}>
                        {fromNode?.label ?? edge?.from ?? '?'}
                      </span>
                      {isDrain ? (
                        <ArrowDown className="w-3.5 h-3.5 text-rose-400 flex-shrink-0" />
                      ) : (
                        <ArrowRight className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                      )}
                      <span className={`text-xs font-medium ${isDrain ? 'text-rose-300' : 'text-emerald-300'}`}>
                        {toNode?.label ?? edge?.to ?? '?'}
                      </span>
                      <Badge variant={btcClassification === 'bullish' ? 'green' : btcClassification === 'bearish' ? 'red' : directionVariant(edge?.direction)}>
                        {btcClassification === 'bullish' ? 'bullish' : btcClassification === 'bearish' ? 'bearish' : (edge?.direction ?? 'flow')}
                      </Badge>
                      <ChevronDown
                        className={`w-3 h-3 text-slate-500 ml-auto transition-transform duration-300 ${
                          isExpanded ? 'rotate-180' : ''
                        }`}
                      />
                    </div>

                    {/* Label */}
                    <div className="text-[10px] text-slate-500 mb-2">
                      {edge?.label}
                    </div>

                    {/* Strength Bar */}
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-slate-600 w-14 flex-shrink-0">
                        Strength
                      </span>
                      <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isDrain
                              ? 'bg-gradient-to-r from-rose-500 to-rose-400'
                              : 'bg-gradient-to-r from-cyan-500 to-cyan-400'
                          }`}
                          style={{ width: `${Math.min(strength * 100, 100)}%` }}
                        />
                      </div>
                      <span className="text-[10px] font-mono text-slate-400 w-8 text-right">
                        {(strength * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>

                  {/* Expanded Edge Education Panel */}
                  {isExpanded && edgeEdu && (
                    <div
                      className={`mt-1 rounded-2xl border backdrop-blur-xl p-4 transition-all duration-300 animate-in fade-in slide-in-from-top-2 ${
                        isDrain
                          ? 'border-rose-500/20 bg-rose-500/[0.03]'
                          : 'border-cyan-500/20 bg-cyan-500/[0.03]'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-3">
                        <span className={`text-xs font-bold ${isDrain ? 'text-rose-400' : 'text-cyan-400'}`}>
                          {key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </span>
                        <button
                          onClick={(e) => { e.stopPropagation(); setExpandedEdge(''); }}
                          className="ml-auto text-slate-500 hover:text-slate-300 text-xs"
                        >
                          close
                        </button>
                      </div>

                      <div className="space-y-3">
                        <div>
                          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                            Mechanism
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {edgeEdu.mechanism}
                          </p>
                        </div>

                        <div>
                          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                            Historical Pattern
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {edgeEdu.historicalPattern}
                          </p>
                        </div>

                        <div>
                          <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">
                            How the Engine Uses This
                          </div>
                          <p className="text-xs text-slate-300 leading-relaxed">
                            {edgeEdu.engineUse}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {activeEdges.length === 0 && (
        <div className="text-center py-4 text-xs text-slate-600">
          No active capital flows detected. Markets are in equilibrium.
        </div>
      )}
    </GlassCard>
  );
}
