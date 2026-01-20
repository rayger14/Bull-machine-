#!/usr/bin/env python3
"""
HMM Regime Detection Training Script
=====================================

Trains a 4-state Gaussian Hidden Markov Model to detect market regimes:
- Bull market
- Bear market
- Neutral/ranging market
- Crisis/high volatility market

The model learns from multi-timeframe features including price action,
volume, liquidity, funding rates, and macro indicators.

Usage:
    python bin/train_hmm_regime.py

Outputs:
    - models/hmm_regime_v1.pkl: Trained HMM model
    - data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_hmm.parquet: Data with regime labels
"""

import pandas as pd
import numpy as np
from hmmlearn import hmm
import pickle
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load feature data and prepare for HMM training.

    Args:
        file_path: Path to parquet file with features

    Returns:
        Tuple of (original_dataframe, feature_matrix, feature_names)
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load data
    df = pd.read_parquet(file_path)
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Feature mapping (requested name -> actual column name)
    feature_mapping = {
        'rsi_14': 'rsi_14',
        'volume_z': 'volume_zscore',  # Corrected name
        'liquidity_score': 'liquidity_score',
        'funding_Z': 'funding_Z',
        'dxy_z': 'DXY_Z',  # Corrected case
        'vix_z': 'VIX_Z',  # Corrected case
        'btc_dominance': 'BTC.D',  # Actual column name
        'total_mcap': 'TOTAL',  # Actual column name
        'total2_mcap': 'TOTAL2'  # Actual column name
    }

    # Get actual column names
    feature_columns = [feature_mapping[key] for key in [
        'rsi_14', 'volume_z', 'liquidity_score', 'funding_Z',
        'dxy_z', 'vix_z', 'btc_dominance', 'total_mcap', 'total2_mcap'
    ]]

    print(f"\nSelected features:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"  {i}. {feat}")

    # Extract features
    X = df[feature_columns].copy()

    # Handle missing values
    print("\n" + "=" * 80)
    print("HANDLING MISSING VALUES")
    print("=" * 80)

    missing_before = X.isnull().sum()
    print("\nMissing values per feature (before):")
    for feat in feature_columns:
        missing_count = missing_before[feat]
        missing_pct = (missing_count / len(X)) * 100
        if missing_count > 0:
            print(f"  {feat}: {missing_count:,} ({missing_pct:.2f}%)")

    # Fill missing values with 0
    X = X.fillna(0)

    missing_after = X.isnull().sum().sum()
    print(f"\nTotal missing values after fillna(0): {missing_after}")

    # Convert to numpy array
    X_array = X.values

    print(f"\nFeature matrix shape: {X_array.shape}")
    print(f"Feature statistics:")
    print(pd.DataFrame(X_array, columns=feature_columns).describe())

    return df, X_array, feature_columns


def train_hmm_model(X: np.ndarray, n_states: int = 4) -> hmm.GaussianHMM:
    """
    Train Gaussian HMM model.

    Args:
        X: Feature matrix (n_samples, n_features)
        n_states: Number of hidden states (default: 4)

    Returns:
        Trained GaussianHMM model
    """
    print("\n" + "=" * 80)
    print("TRAINING HMM MODEL")
    print("=" * 80)

    print(f"\nModel configuration:")
    print(f"  States: {n_states}")
    print(f"  Covariance type: diagonal")
    print(f"  Max iterations: 1000")
    print(f"  Random state: 42")

    # Initialize model
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
        verbose=False
    )

    # Train model
    print("\nTraining HMM...")
    model.fit(X)

    print(f"✓ Training complete!")
    print(f"  Converged: {model.monitor_.converged}")
    print(f"  Final log-likelihood: {model.monitor_.history[-1]:.2f}")
    print(f"  Iterations: {len(model.monitor_.history)}")

    return model


def predict_regimes(model: hmm.GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Predict regime for each observation.

    Args:
        model: Trained HMM model
        X: Feature matrix

    Returns:
        Array of regime labels (0 to n_states-1)
    """
    print("\n" + "=" * 80)
    print("PREDICTING REGIMES")
    print("=" * 80)

    regimes = model.predict(X)

    print(f"\nRegime distribution:")
    unique, counts = np.unique(regimes, return_counts=True)
    for state, count in zip(unique, counts):
        pct = (count / len(regimes)) * 100
        print(f"  Regime {state}: {count:,} bars ({pct:.2f}%)")

    return regimes


def interpret_regimes(model: hmm.GaussianHMM, feature_names: list[str]) -> pd.DataFrame:
    """
    Analyze and interpret regime characteristics.

    Args:
        model: Trained HMM model
        feature_names: List of feature names

    Returns:
        DataFrame with regime statistics
    """
    print("\n" + "=" * 80)
    print("REGIME INTERPRETATION")
    print("=" * 80)

    # Get means for each regime
    means = model.means_

    # Create DataFrame for analysis
    regime_stats = pd.DataFrame(means, columns=feature_names)
    regime_stats.index.name = 'regime'

    print("\nRegime means (normalized features):")
    print(regime_stats.to_string())

    print("\n" + "-" * 80)
    print("REGIME CHARACTERISTICS")
    print("-" * 80)

    # Analyze each regime
    interpretations = []

    for state in range(len(means)):
        print(f"\n{'Regime ' + str(state):-^80}")

        characteristics = []

        # RSI
        rsi = regime_stats.loc[state, 'rsi_14']
        if rsi > 60:
            characteristics.append("Overbought (RSI > 60)")
        elif rsi < 40:
            characteristics.append("Oversold (RSI < 40)")
        else:
            characteristics.append("Neutral RSI")

        # Volume
        vol_z = regime_stats.loc[state, 'volume_zscore']
        if vol_z > 1:
            characteristics.append("High volume")
        elif vol_z < -0.5:
            characteristics.append("Low volume")
        else:
            characteristics.append("Normal volume")

        # Liquidity
        liq = regime_stats.loc[state, 'liquidity_score']
        if liq > 0.5:
            characteristics.append("High liquidity")
        elif liq < -0.5:
            characteristics.append("Low liquidity")

        # Funding rate
        funding = regime_stats.loc[state, 'funding_Z']
        if funding > 1:
            characteristics.append("Extreme positive funding (overheated)")
        elif funding < -1:
            characteristics.append("Extreme negative funding (fear)")

        # VIX
        vix = regime_stats.loc[state, 'VIX_Z']
        if vix > 1:
            characteristics.append("High market fear (VIX)")
        elif vix < -1:
            characteristics.append("Low market fear (VIX)")

        # DXY
        dxy = regime_stats.loc[state, 'DXY_Z']
        if dxy > 1:
            characteristics.append("Strong dollar (bearish for crypto)")
        elif dxy < -1:
            characteristics.append("Weak dollar (bullish for crypto)")

        # BTC Dominance
        btc_dom = regime_stats.loc[state, 'BTC.D']
        if btc_dom > 50:
            characteristics.append("BTC dominance high (altcoin weakness)")
        elif btc_dom < 40:
            characteristics.append("BTC dominance low (altcoin season)")

        # Print characteristics
        for char in characteristics:
            print(f"  • {char}")

        # Suggest regime type
        print(f"\n  Suggested interpretation:")
        if rsi > 55 and funding > 0 and vix < 0:
            print(f"    → Likely BULL MARKET (high RSI, positive funding, low fear)")
            interpretations.append("BULL")
        elif rsi < 45 and vix > 0.5:
            print(f"    → Likely BEAR/CRISIS MARKET (low RSI, high fear)")
            interpretations.append("BEAR/CRISIS")
        elif vix > 1.5:
            print(f"    → Likely CRISIS MARKET (extreme fear)")
            interpretations.append("CRISIS")
        elif abs(funding) < 0.5 and abs(vol_z) < 0.5:
            print(f"    → Likely NEUTRAL/RANGING MARKET (balanced conditions)")
            interpretations.append("NEUTRAL")
        else:
            print(f"    → MIXED SIGNALS (manual review recommended)")
            interpretations.append("MIXED")

    return regime_stats


def save_model(model: hmm.GaussianHMM, output_path: str):
    """
    Save trained model to disk.

    Args:
        model: Trained HMM model
        output_path: Path to save model pickle file
    """
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n✓ Model saved to: {output_path}")

    # Print file size
    file_size = Path(output_path).stat().st_size
    print(f"  File size: {file_size / 1024:.2f} KB")


def save_augmented_data(df: pd.DataFrame, regimes: np.ndarray, output_path: str):
    """
    Save original data with regime labels added.

    Args:
        df: Original dataframe
        regimes: Array of regime predictions
        output_path: Path to save augmented parquet file
    """
    print("\n" + "=" * 80)
    print("SAVING AUGMENTED DATA")
    print("=" * 80)

    # Add regime column
    df_augmented = df.copy()
    df_augmented['hmm_regime'] = regimes

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df_augmented.to_parquet(output_path)

    print(f"\n✓ Augmented data saved to: {output_path}")

    # Print file size
    file_size = Path(output_path).stat().st_size
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"  Rows: {len(df_augmented):,}")
    print(f"  Columns: {len(df_augmented.columns)}")
    print(f"\n  New column added: 'hmm_regime'")


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print(" HMM REGIME DETECTION TRAINING ".center(80, "="))
    print("=" * 80)

    # Configuration
    data_path = "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet"
    model_path = "models/hmm_regime_v1.pkl"
    output_path = "data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_hmm.parquet"
    n_states = 4

    # Step 1: Load and prepare data
    df, X, feature_names = load_and_prepare_data(data_path)

    # Step 2: Train HMM model
    model = train_hmm_model(X, n_states=n_states)

    # Step 3: Predict regimes
    regimes = predict_regimes(model, X)

    # Step 4: Interpret regimes
    regime_stats = interpret_regimes(model, feature_names)

    # Step 5: Save model
    save_model(model, model_path)

    # Step 6: Save augmented data
    save_augmented_data(df, regimes, output_path)

    print("\n" + "=" * 80)
    print(" TRAINING COMPLETE ".center(80, "="))
    print("=" * 80)

    print(f"\nOutputs:")
    print(f"  1. Trained model: {model_path}")
    print(f"  2. Augmented data: {output_path}")

    print(f"\nNext steps:")
    print(f"  1. Review regime interpretations above")
    print(f"  2. Validate regimes against known market periods")
    print(f"  3. Use 'hmm_regime' column in your trading strategies")
    print(f"  4. Consider regime-specific parameter tuning")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
