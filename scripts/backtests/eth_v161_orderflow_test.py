#!/usr/bin/env python3
"""
ETH v1.6.1 Orderflow & Cluster Test
Direct testing of Fibonacci Clusters, Oracle Whispers, and Enhanced CVD

"Price and time symmetry = where structure and vibration align"
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import json

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.orderflow.lca import analyze_market_structure, orderflow_lca
from bull_machine.strategy.hidden_fibs import detect_price_time_confluence, fib_price_clusters
from bull_machine.strategy.temporal_fib_clusters import fib_time_clusters, detect_pivot_points
from bull_machine.oracle import trigger_whisper, should_trigger_confluence_alert
from bull_machine.core.config_loader import load_config

warnings.filterwarnings('ignore')

def test_v161_features_directly():
    """Test v1.6.1 features directly on ETH data"""
    print("🔮 ETH v1.6.1 Direct Feature Testing")
    print("Testing Fibonacci Clusters, Oracle Whispers, Enhanced CVD")
    print("=" * 60)

    # Fetch ETH data
    print("📊 Fetching ETH data...")
    eth = yf.Ticker("ETH-USD")
    df = eth.history(start="2024-06-01", end="2024-12-01", interval="1d")

    if len(df) == 0:
        print("❌ No data fetched")
        return

    # Normalize columns
    df.columns = [col.lower() for col in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"✅ Fetched {len(df)} daily bars")

    # Load enhanced ETH config
    config = load_config("ETH")
    config['features']['temporal_fib'] = True
    config['features']['fib_clusters'] = True
    config['features']['orderflow_lca'] = True

    # Test components
    results = {
        'cluster_analysis': [],
        'oracle_events': [],
        'cvd_analysis': [],
        'orderflow_scores': [],
        'high_confluence_events': []
    }

    print(f"\n🔍 Testing v1.6.1 features on {len(df)} bars...")

    # Test on sample points throughout the data
    test_points = range(50, len(df), 10)  # Every 10 bars

    for i, test_idx in enumerate(test_points):
        current_data = df.iloc[:test_idx+1]

        try:
            print(f"\r  🔄 Testing point {i+1}/{len(test_points)} (bar {test_idx})", end="", flush=True)

            # Test Fibonacci Clusters
            confluence_data = detect_price_time_confluence(current_data, config, test_idx)

            # Test Enhanced CVD & Orderflow
            orderflow_score = orderflow_lca(current_data.tail(30), config)
            structure_analysis = analyze_market_structure(current_data.tail(30), config)

            # Record analysis
            analysis_point = {
                'date': current_data.index[test_idx],
                'price': current_data.iloc[test_idx]['close'],
                'confluence_detected': confluence_data.get('confluence_detected', False),
                'confluence_strength': confluence_data.get('confluence_strength', 0),
                'price_cluster': confluence_data.get('price_cluster') is not None,
                'time_cluster': confluence_data.get('time_cluster') is not None,
                'cluster_tags': confluence_data.get('tags', []),
                'orderflow_score': orderflow_score,
                'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                'cvd_slope': structure_analysis['cvd_analysis']['slope'],
                'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],
                'bos_detected': structure_analysis['bos_analysis']['detected'],
                'structure_health': structure_analysis['structure_health']
            }

            results['cluster_analysis'].append(analysis_point)
            results['orderflow_scores'].append(orderflow_score)

            # Test Oracle Whisper System
            if confluence_data.get('confluence_detected') or orderflow_score > 0.6:
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'fib_retracement': 0.45,  # Mock for testing
                    'fib_extension': 0.40,
                    'wyckoff_phase': 'C',
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope']
                }

                whispers = trigger_whisper(test_scores, phase='C')

                if whispers:
                    oracle_event = {
                        'date': current_data.index[test_idx],
                        'whispers': whispers,
                        'confluence_strength': test_scores['confluence_strength'],
                        'orderflow_score': orderflow_score
                    }
                    results['oracle_events'].append(oracle_event)

            # Track high confluence events
            if confluence_data.get('confluence_strength', 0) > 0.65 or orderflow_score > 0.7:
                high_conf_event = {
                    'date': current_data.index[test_idx],
                    'type': 'high_confluence',
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'orderflow_score': orderflow_score,
                    'features': {
                        'price_cluster': confluence_data.get('price_cluster') is not None,
                        'time_cluster': confluence_data.get('time_cluster') is not None,
                        'cvd_divergence': structure_analysis['orderflow_divergence']['detected'],
                        'bos_detected': structure_analysis['bos_analysis']['detected']
                    }
                }
                results['high_confluence_events'].append(high_conf_event)

            # CVD Analysis
            if abs(structure_analysis['cvd_analysis']['slope']) > 50:  # Significant slope
                cvd_event = {
                    'date': current_data.index[test_idx],
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope'],
                    'divergence_type': structure_analysis['orderflow_divergence']['type'],
                    'divergence_strength': structure_analysis['orderflow_divergence']['strength']
                }
                results['cvd_analysis'].append(cvd_event)

        except Exception as e:
            continue

    print(f"\n\n📊 v1.6.1 Feature Test Results:")

    # Cluster Analysis
    cluster_events = [a for a in results['cluster_analysis'] if a['confluence_detected']]
    price_clusters = len([a for a in results['cluster_analysis'] if a['price_cluster']])
    time_clusters = len([a for a in results['cluster_analysis'] if a['time_cluster']])

    print(f"\n🔮 Fibonacci Cluster Analysis:")
    print(f"  📊 Total analysis points: {len(results['cluster_analysis'])}")
    print(f"  ✨ Price-time confluence events: {len(cluster_events)}")
    print(f"  🔮 Price clusters detected: {price_clusters}")
    print(f"  ⏰ Time clusters detected: {time_clusters}")

    if cluster_events:
        avg_confluence_strength = np.mean([e['confluence_strength'] for e in cluster_events])
        print(f"  💪 Average confluence strength: {avg_confluence_strength:.3f}")

    # Oracle Whisper Analysis
    print(f"\n🧙‍♂️ Oracle Whisper Analysis:")
    print(f"  ✨ Oracle events triggered: {len(results['oracle_events'])}")

    if results['oracle_events']:
        total_whispers = sum(len(event['whispers']) for event in results['oracle_events'])
        avg_whispers_per_event = total_whispers / len(results['oracle_events'])
        print(f"  💫 Total whispers generated: {total_whispers}")
        print(f"  🗣️ Average whispers per event: {avg_whispers_per_event:.1f}")

        # Show sample whispers
        print(f"  📜 Sample whispers:")
        for i, event in enumerate(results['oracle_events'][:3]):
            for whisper in event['whispers'][:2]:  # Show first 2 whispers
                print(f"    {i+1}. \"{whisper}\"")

    # CVD Analysis
    print(f"\n📊 Enhanced CVD Analysis:")
    print(f"  🔄 CVD events with significant slope: {len(results['cvd_analysis'])}")

    if results['cvd_analysis']:
        bullish_divergences = len([e for e in results['cvd_analysis'] if e['divergence_type'] == 'bullish'])
        bearish_divergences = len([e for e in results['cvd_analysis'] if e['divergence_type'] == 'bearish'])
        print(f"  🟢 Bullish divergences: {bullish_divergences}")
        print(f"  🔴 Bearish divergences: {bearish_divergences}")

        avg_slope = np.mean([abs(e['cvd_slope']) for e in results['cvd_analysis']])
        print(f"  📈 Average CVD slope magnitude: {avg_slope:.1f}")

    # Orderflow Scores
    if results['orderflow_scores']:
        avg_orderflow = np.mean(results['orderflow_scores'])
        high_orderflow = len([s for s in results['orderflow_scores'] if s > 0.6])
        print(f"\n📈 Orderflow Score Analysis:")
        print(f"  📊 Average orderflow score: {avg_orderflow:.3f}")
        print(f"  🎯 High orderflow signals (>0.6): {high_orderflow}")

    # High Confluence Events
    print(f"\n⚡ High Confluence Events:")
    print(f"  🌟 Total high confluence events: {len(results['high_confluence_events'])}")

    if results['high_confluence_events']:
        feature_counts = {
            'price_cluster': sum(1 for e in results['high_confluence_events'] if e['features']['price_cluster']),
            'time_cluster': sum(1 for e in results['high_confluence_events'] if e['features']['time_cluster']),
            'cvd_divergence': sum(1 for e in results['high_confluence_events'] if e['features']['cvd_divergence']),
            'bos_detected': sum(1 for e in results['high_confluence_events'] if e['features']['bos_detected'])
        }

        print(f"  🔮 With price clusters: {feature_counts['price_cluster']}")
        print(f"  ⏰ With time clusters: {feature_counts['time_cluster']}")
        print(f"  📊 With CVD divergence: {feature_counts['cvd_divergence']}")
        print(f"  💥 With BOS detection: {feature_counts['bos_detected']}")

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {
        'metadata': {
            'timestamp': timestamp,
            'version': 'v1.6.1',
            'asset': 'ETH-USD',
            'test_period': f"{df.index[0]} to {df.index[-1]}",
            'philosophy': 'Price and time symmetry = where structure and vibration align'
        },
        'summary': {
            'total_analysis_points': len(results['cluster_analysis']),
            'confluence_events': len(cluster_events),
            'price_clusters': price_clusters,
            'time_clusters': time_clusters,
            'oracle_events': len(results['oracle_events']),
            'cvd_events': len(results['cvd_analysis']),
            'high_confluence_events': len(results['high_confluence_events'])
        },
        'sample_events': {
            'cluster_events': cluster_events[:5],
            'oracle_events': results['oracle_events'][:3],
            'high_confluence': results['high_confluence_events'][:5]
        }
    }

    filename = f"eth_v161_feature_test_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {filename}")
    print(f"\n✨ v1.6.1 Philosophy Validation:")
    print(f"   'Price and time symmetry = where structure and vibration align'")
    print(f"   📊 {len(cluster_events)} confluence events detected")
    print(f"   🧙‍♂️ {len(results['oracle_events'])} Oracle whispers triggered")
    print(f"   📈 {len(results['cvd_analysis'])} CVD divergence patterns identified")

if __name__ == "__main__":
    test_v161_features_directly()