#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Optimization Results Analyzer
Comprehensive analysis of Stage A/B results with quality gates
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from metric_gates import MetricGateValidator, QualityReport
from datetime import datetime

def load_results(file_path: str = "reports/opt/results.jsonl") -> List[Dict]:
    """Load all optimization results"""
    results = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if result.get("status") == "ok":
                        results.append(result)
                except:
                    continue
    except FileNotFoundError:
        print(f"Results file not found: {file_path}")
        return []

    return results

def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Comprehensive analysis of optimization results"""

    if not results:
        return {"error": "No results to analyze"}

    # Separate by strategy
    stage_a_results = [r for r in results if r.get("strategy") != "stage_b"]
    stage_b_results = [r for r in results if r.get("strategy") == "stage_b"]

    # Filter for results with actual trades
    trading_results = [r for r in results if r.get("trades", 0) > 0]

    print(f"📊 Total results: {len(results)}")
    print(f"📈 Stage A results: {len(stage_a_results)}")
    print(f"📈 Stage B results: {len(stage_b_results)}")
    print(f"💰 Results with trades: {len(trading_results)}")

    if not trading_results:
        return {
            "summary": "No results with trades found",
            "total_results": len(results),
            "stage_a_count": len(stage_a_results),
            "stage_b_count": len(stage_b_results)
        }

    # Analyze trading results
    analysis = {}

    # Performance metrics
    pnl_values = [r.get("pnl_pct", 0) for r in trading_results]
    wr_values = [r.get("wr", 0) for r in trading_results]
    pf_values = [r.get("pf", 0) for r in trading_results]
    sharpe_values = [r.get("sharpe", 0) for r in trading_results]
    dd_values = [r.get("dd_pct", 0) for r in trading_results]
    trade_counts = [r.get("trades", 0) for r in trading_results]

    analysis["performance_summary"] = {
        "total_configs_with_trades": len(trading_results),
        "avg_pnl_pct": np.mean(pnl_values),
        "median_pnl_pct": np.median(pnl_values),
        "avg_win_rate": np.mean(wr_values),
        "avg_profit_factor": np.mean(pf_values),
        "avg_sharpe": np.mean(sharpe_values),
        "avg_max_dd": np.mean(dd_values),
        "avg_trades": np.mean(trade_counts)
    }

    # Top performers
    # Calculate utility score for ranking
    def calculate_utility(result):
        trades = result.get("trades", 0)
        wr = result.get("wr", 0)
        pnl = result.get("pnl_pct", 0)
        pf = result.get("pf", 0)
        dd = result.get("dd_pct", 0)
        sharpe = result.get("sharpe", 0)

        if trades < 5:
            return 0

        # Normalize metrics
        n_pnl = max(0, min(1, pnl / 30))
        n_pf = max(0, min(1, (pf - 1) / 2))
        n_wr = wr
        n_sharpe = max(0, min(1, sharpe / 3))

        # Penalties
        dd_penalty = min(1, dd / 25) ** 2
        freq_penalty = max(0, (10 - trades) / 10) if trades < 10 else 0

        utility = (1.0 * n_pnl + 0.8 * n_pf + 0.6 * n_wr + 0.4 * n_sharpe -
                  1.2 * dd_penalty - 0.6 * freq_penalty)

        return max(0, utility)

    # Sort by utility score
    scored_results = [(calculate_utility(r), r) for r in trading_results]
    scored_results.sort(reverse=True)

    analysis["top_performers"] = []
    for i, (score, result) in enumerate(scored_results[:10]):
        analysis["top_performers"].append({
            "rank": i + 1,
            "utility_score": score,
            "asset": result.get("asset"),
            "cfg": result.get("cfg"),
            "fold": result.get("fold"),
            "trades": result.get("trades"),
            "win_rate": result.get("wr"),
            "pnl_pct": result.get("pnl_pct"),
            "profit_factor": result.get("pf"),
            "sharpe_ratio": result.get("sharpe"),
            "max_dd_pct": result.get("dd_pct"),
            "strategy": result.get("strategy", "stage_a")
        })

    # Parameter analysis
    analysis["parameter_insights"] = analyze_parameters(trading_results)

    # Risk analysis
    analysis["risk_analysis"] = {
        "max_drawdown_seen": max(dd_values) if dd_values else 0,
        "configs_with_high_dd": len([dd for dd in dd_values if dd > 20]),
        "configs_with_low_trades": len([t for t in trade_counts if t < 5]),
        "negative_pnl_configs": len([pnl for pnl in pnl_values if pnl < 0])
    }

    return analysis

def analyze_parameters(results: List[Dict]) -> Dict[str, Any]:
    """Analyze which parameters lead to better performance"""

    # Extract parameter values from config strings
    param_performance = {
        "entry_threshold": {},
        "risk_pct": {},
        "cooldown_days": {},
        "sl_atr_multiplier": {},
        "tp_atr_multiplier": {}
    }

    for result in results:
        cfg = result.get("cfg", "")
        pnl = result.get("pnl_pct", 0)
        utility = calculate_utility_simple(result)

        # Parse parameters from config string
        params = parse_config_string(cfg)

        for param_name, param_value in params.items():
            if param_name in param_performance:
                if param_value not in param_performance[param_name]:
                    param_performance[param_name][param_value] = {
                        "count": 0, "total_pnl": 0, "total_utility": 0
                    }

                param_performance[param_name][param_value]["count"] += 1
                param_performance[param_name][param_value]["total_pnl"] += pnl
                param_performance[param_name][param_value]["total_utility"] += utility

    # Calculate averages
    insights = {}
    for param_name, param_data in param_performance.items():
        insights[param_name] = {}
        for param_value, stats in param_data.items():
            if stats["count"] > 0:
                insights[param_name][param_value] = {
                    "avg_pnl": stats["total_pnl"] / stats["count"],
                    "avg_utility": stats["total_utility"] / stats["count"],
                    "sample_count": stats["count"]
                }

    return insights

def parse_config_string(cfg: str) -> Dict[str, float]:
    """Parse configuration string into parameter values"""
    params = {}
    parts = cfg.split("_")

    for part in parts:
        if part.startswith("thresh"):
            params["entry_threshold"] = float(part[6:])
        elif part.startswith("cd") and part[2:].isdigit():
            params["cooldown_days"] = int(part[2:])
        elif part.startswith("r") and len(part) > 1:
            try:
                params["risk_pct"] = float(part[1:])
            except:
                pass
        elif part.startswith("sl"):
            try:
                params["sl_atr_multiplier"] = float(part[2:])
            except:
                pass
        elif part.startswith("tp"):
            try:
                params["tp_atr_multiplier"] = float(part[2:])
            except:
                pass

    return params

def calculate_utility_simple(result: Dict) -> float:
    """Simplified utility calculation"""
    trades = result.get("trades", 0)
    pnl = result.get("pnl_pct", 0)
    wr = result.get("wr", 0)
    pf = result.get("pf", 0)

    if trades < 3:
        return 0

    return pnl * wr * min(pf, 3)  # Cap PF at 3 for utility

def generate_recommendations(analysis: Dict) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    recommendations = []

    if not analysis.get("top_performers"):
        recommendations.append("❌ No profitable configurations found. Consider:")
        recommendations.append("   - Lowering entry thresholds")
        recommendations.append("   - Reducing minimum domain requirements")
        recommendations.append("   - Adjusting risk management parameters")
        return recommendations

    top_config = analysis["top_performers"][0]

    recommendations.append(f"🎯 Best configuration: {top_config['cfg']}")
    # Fix win rate formatting - metrics already stored as percentage
    wr = top_config['win_rate']
    wr_formatted = f"{wr:.1f}%" if wr > 1 else f"{wr*100:.1f}%"
    recommendations.append(f"   → {top_config['trades']} trades, {wr_formatted} WR, {top_config['pnl_pct']:+.1f}% PnL")

    # Parameter insights
    if "parameter_insights" in analysis:
        insights = analysis["parameter_insights"]

        # Find best entry threshold
        if "entry_threshold" in insights:
            best_thresh = max(insights["entry_threshold"].items(),
                            key=lambda x: x[1]["avg_utility"] if x[1]["sample_count"] >= 2 else 0)
            recommendations.append(f"📊 Optimal entry threshold: {best_thresh[0]} (avg utility: {best_thresh[1]['avg_utility']:.3f})")

        # Find best risk level
        if "risk_pct" in insights:
            best_risk = max(insights["risk_pct"].items(),
                          key=lambda x: x[1]["avg_utility"] if x[1]["sample_count"] >= 2 else 0)
            recommendations.append(f"💰 Optimal risk per trade: {best_risk[0]*100:.1f}% (avg utility: {best_risk[1]['avg_utility']:.3f})")

    # Risk warnings
    risk_analysis = analysis.get("risk_analysis", {})
    if risk_analysis.get("max_drawdown_seen", 0) > 25:
        recommendations.append(f"⚠️  Maximum drawdown seen: {risk_analysis['max_drawdown_seen']:.1f}% - Consider tighter risk management")

    if risk_analysis.get("negative_pnl_configs", 0) > len(analysis.get("top_performers", [])) // 2:
        recommendations.append("⚠️  Many configurations showing losses - Consider parameter range adjustment")

    return recommendations

def main():
    """Main analysis entry point"""
    print("🔬 Bull Machine Optimization Results Analysis")
    print("=" * 60)

    # Load and analyze results
    results = load_results()
    analysis = analyze_results(results)

    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return

    # Print summary
    print("\n📊 PERFORMANCE SUMMARY")
    perf = analysis.get("performance_summary", {})
    print(f"   Configs with trades: {perf.get('total_configs_with_trades', 0)}")
    print(f"   Average PnL: {perf.get('avg_pnl_pct', 0):+.2f}%")
    # Fix win rate formatting - metrics already stored as percentage
    avg_wr = perf.get('avg_win_rate', 0)
    avg_wr_formatted = f"{avg_wr:.1f}%" if avg_wr > 1 else f"{avg_wr*100:.1f}%"
    print(f"   Average Win Rate: {avg_wr_formatted}")
    print(f"   Average Profit Factor: {perf.get('avg_profit_factor', 0):.2f}")
    print(f"   Average Sharpe Ratio: {perf.get('avg_sharpe', 0):.2f}")

    # Print top performers
    print("\n🏆 TOP PERFORMERS")
    for performer in analysis.get("top_performers", [])[:5]:
        print(f"   {performer['rank']}. {performer['asset']} | Utility: {performer['utility_score']:.3f}")
        # Fix win rate formatting - metrics already stored as percentage
        wr = performer['win_rate']
        wr_formatted = f"{wr:.1f}%" if wr > 1 else f"{wr*100:.1f}%"
        print(f"      {performer['trades']}T, {wr_formatted} WR, {performer['pnl_pct']:+.1f}% PnL, {performer['profit_factor']:.2f} PF")
        print(f"      Config: {performer['cfg'][:60]}...")
        print()

    # Generate and print recommendations
    print("💡 RECOMMENDATIONS")
    recommendations = generate_recommendations(analysis)
    for rec in recommendations:
        print(f"   {rec}")

    # Save detailed analysis
    output_file = f"reports/opt/analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n📁 Detailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main()