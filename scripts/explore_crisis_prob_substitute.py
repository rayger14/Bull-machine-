"""Explore design of substitute crisis_prob using features the engine already has.

Validation criteria:
  - LUNA week (2022-05-09 to 2022-05-15): crisis_prob > 0.5 sustained
  - 2024 trending: crisis_prob low (<0.3)
  - Live W3 (May28-Jun02 2026): higher than W1 (Apr20-May17)

Design:
    crisis_prob_substitute = clamp(
        w_macro * macro_stress +
        w_sentiment * sentiment_stress +
        w_structural * structural_stress +
        w_vol * vol_shock,
        0, 1)

    macro_stress = sigmoid(a1*DXY_Z + a2*VIX_Z - 1.0)  # USD up + vol up = stress
    sentiment_stress = clamp((30 - fear_greed)/30, 0, 1)  # fear≤30 → stress
    structural_stress = sigmoid(-b1*ema_slope_50 + b2*(1-tf4h_wyckoff_bullish_score) - 0.5)
                                        # downtrend + bull breakdown
    vol_shock = clamp((rv_20d - 0.8)/0.4, 0, 1)
"""

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_substitute(row,
                       w_macro=0.30, w_sentiment=0.30, w_structural=0.25, w_vol=0.15,
                       a1=0.60, a2=0.45,
                       b1=8.0, b2=2.0,
                       fear_threshold=30.0,
                       rv_floor=0.75, rv_range=0.5):
    """Compute substitute crisis_prob.

    Inputs (engine feature names, with fallbacks for CSV):
      - DXY_Z (or dxy_z)
      - VIX_Z (or vix_z)
      - fear_greed
      - ema_slope_50
      - tf4h_wyckoff_bullish_score (or wyckoff_4h_bull) — using bull breakdown rather than buggy bear
      - rv_20d (or atr_14 as fallback)
    """
    def g(key, *aliases, default=0.0):
        for k in (key, *aliases):
            v = row.get(k, None) if hasattr(row, 'get') else None
            if v is not None and not (isinstance(v, float) and v != v):
                return float(v)
        return float(default)

    dxy = g('DXY_Z', 'dxy_z', default=0.0)
    vix = g('VIX_Z', 'vix_z', default=0.0)
    fg = g('fear_greed', default=50.0)
    if fg > 1.0:
        fg_score = fg
    else:
        fg_score = fg * 100.0  # if it's fear_greed_norm scale to 0-100

    ema_slope = g('ema_slope_50', default=0.0)
    w_bull = g('tf4h_wyckoff_bullish_score', 'wyckoff_4h_bull', default=0.5)
    rv = g('rv_20d', default=0.6)

    macro_stress = sigmoid(a1 * dxy + a2 * vix - 1.0)
    sentiment_stress = np.clip((fear_threshold - fg_score) / fear_threshold, 0.0, 1.0)
    structural_stress = sigmoid(-b1 * ema_slope + b2 * (1.0 - w_bull) - 0.5)
    vol_shock = np.clip((rv - rv_floor) / rv_range, 0.0, 1.0)

    crisis_prob = (
        w_macro * macro_stress +
        w_sentiment * sentiment_stress +
        w_structural * structural_stress +
        w_vol * vol_shock
    )
    return float(np.clip(crisis_prob, 0.0, 1.0)), {
        'macro_stress': float(macro_stress),
        'sentiment_stress': float(sentiment_stress),
        'structural_stress': float(structural_stress),
        'vol_shock': float(vol_shock)
    }


def load_live_trades():
    df = pd.read_csv('/tmp/trade_outcomes_live_jun2.csv')
    old = list(df.columns)
    new_cols = ['timestamp_entry'] + old[2:] + ['_extra']
    new_cols = new_cols[:len(old)]
    df.columns = new_cols
    df['timestamp_entry'] = pd.to_datetime(df['timestamp_entry'])
    return df


def evaluate_on_live(weights=None):
    df = load_live_trades()
    if weights is None:
        weights = {}
    crisis_list = []
    parts_list = []
    for _, row in df.iterrows():
        cp, parts = compute_substitute(row, **weights)
        crisis_list.append(cp)
        parts_list.append(parts)
    df['crisis_sub'] = crisis_list
    for k in ['macro_stress', 'sentiment_stress', 'structural_stress', 'vol_shock']:
        df[k] = [p[k] for p in parts_list]

    w1 = df[(df.timestamp_entry >= '2026-04-20') & (df.timestamp_entry < '2026-05-17')]
    w2 = df[(df.timestamp_entry >= '2026-05-17') & (df.timestamp_entry < '2026-05-28')]
    w3 = df[(df.timestamp_entry >= '2026-05-28') & (df.timestamp_entry <= '2026-06-02')]

    print(f'\n== LIVE TRADES ({len(df)} trades, weights={weights}) ==')
    print(f'  All     mean={df.crisis_sub.mean():.3f}  std={df.crisis_sub.std():.3f}  '
          f'p10={df.crisis_sub.quantile(0.1):.3f}  p90={df.crisis_sub.quantile(0.9):.3f}')
    print(f'  W1 n={len(w1):3d}  crisis_sub={w1.crisis_sub.mean():.3f}  '
          f'macro={w1.macro_stress.mean():.3f} sent={w1.sentiment_stress.mean():.3f} '
          f'struct={w1.structural_stress.mean():.3f} vol={w1.vol_shock.mean():.3f}')
    print(f'  W2 n={len(w2):3d}  crisis_sub={w2.crisis_sub.mean():.3f}  '
          f'macro={w2.macro_stress.mean():.3f} sent={w2.sentiment_stress.mean():.3f} '
          f'struct={w2.structural_stress.mean():.3f} vol={w2.vol_shock.mean():.3f}')
    print(f'  W3 n={len(w3):3d}  crisis_sub={w3.crisis_sub.mean():.3f}  '
          f'macro={w3.macro_stress.mean():.3f} sent={w3.sentiment_stress.mean():.3f} '
          f'struct={w3.structural_stress.mean():.3f} vol={w3.vol_shock.mean():.3f}')
    return df


def evaluate_on_history(weights=None):
    df = pd.read_parquet('data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet')
    if weights is None:
        weights = {}
    # Sample a subset to keep it fast: take eod 4H sampling
    sub = df.iloc[::4].copy()

    crisis = []
    for _, row in sub.iterrows():
        cp, _ = compute_substitute(row, **weights)
        crisis.append(cp)
    sub['crisis_sub'] = crisis

    luna = sub.loc['2022-05-09':'2022-05-15']
    ftx = sub.loc['2022-11-07':'2022-11-15']
    covid = sub.loc['2020-03-09':'2020-03-20']
    bull_2024 = sub.loc['2024-03-01':'2024-04-15']
    bull_2021 = sub.loc['2021-04-01':'2021-05-01']
    summer_2024 = sub.loc['2024-07-15':'2024-08-15']

    print(f'\n== HISTORICAL ({len(sub)} 4H bars sampled, weights={weights}) ==')
    print(f'  Overall    mean={sub.crisis_sub.mean():.3f}  std={sub.crisis_sub.std():.3f}')
    print(f'  >0.5 rate={(sub.crisis_sub > 0.5).mean():.3%}')
    print(f'  LUNA week (2022-05-09→15)   n={len(luna):3d}  '
          f'mean={luna.crisis_sub.mean():.3f}  max={luna.crisis_sub.max():.3f}  '
          f'%>0.5={(luna.crisis_sub > 0.5).mean():.1%}')
    print(f'  FTX week  (2022-11-07→15)   n={len(ftx):3d}  '
          f'mean={ftx.crisis_sub.mean():.3f}  max={ftx.crisis_sub.max():.3f}  '
          f'%>0.5={(ftx.crisis_sub > 0.5).mean():.1%}')
    print(f'  COVID    (2020-03-09→20)   n={len(covid):3d}  '
          f'mean={covid.crisis_sub.mean():.3f}  max={covid.crisis_sub.max():.3f}  '
          f'%>0.5={(covid.crisis_sub > 0.5).mean():.1%}')
    print(f'  Bull 2021 Apr             n={len(bull_2021):3d}  '
          f'mean={bull_2021.crisis_sub.mean():.3f}  max={bull_2021.crisis_sub.max():.3f}')
    print(f'  Bull 2024 Mar-Apr         n={len(bull_2024):3d}  '
          f'mean={bull_2024.crisis_sub.mean():.3f}  max={bull_2024.crisis_sub.max():.3f}')
    print(f'  Summer 2024 (chop)        n={len(summer_2024):3d}  '
          f'mean={summer_2024.crisis_sub.mean():.3f}  max={summer_2024.crisis_sub.max():.3f}')
    return sub


def main():
    # Default weights
    evaluate_on_live()
    evaluate_on_history()
    print()

    # Try alternate weights — more sensitive to macro
    print('=' * 70)
    print('Alt config 2: higher macro weight, lower volatility weight')
    print('=' * 70)
    alt = dict(w_macro=0.35, w_sentiment=0.30, w_structural=0.25, w_vol=0.10,
               a1=0.80, a2=0.35, fear_threshold=35.0)
    evaluate_on_live(alt)
    evaluate_on_history(alt)

    # Alt config 3: emphasize sentiment + EMA slope (catches FTX better)
    # Use absolute(DXY_Z) symmetric since both directions can signal stress
    # via decoupled USD volatility
    print('=' * 70)
    print('Alt config 3: balanced macro + sentiment + ema slope')
    print('=' * 70)
    alt = dict(w_macro=0.25, w_sentiment=0.35, w_structural=0.30, w_vol=0.10,
               a1=0.60, a2=0.40, b1=15.0, b2=1.0, fear_threshold=35.0)
    evaluate_on_live(alt)
    evaluate_on_history(alt)

    print('=' * 70)
    print('Alt config 4: tighter, more selective')
    print('=' * 70)
    alt = dict(w_macro=0.25, w_sentiment=0.40, w_structural=0.25, w_vol=0.10,
               a1=0.50, a2=0.50, b1=20.0, b2=0.5, fear_threshold=30.0,
               rv_floor=0.70, rv_range=0.6)
    evaluate_on_live(alt)
    evaluate_on_history(alt)


if __name__ == '__main__':
    main()
