#!/usr/bin/env python3
"""
Train Regime Classifier for Bull Machine v1.9

Uses Gaussian Mixture Model to cluster macro features into regimes.
Labels are derived from VIX bands initially, then refined by GMM clustering.

Usage:
    python3 bin/train/train_regime_classifier.py --data data/macro/macro_history.parquet --output models/regime_classifier_gmm.pkl
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

from engine.context.regime_classifier import RegimeClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_macro_data(data_path: str, feature_order: list) -> pd.DataFrame:
    """
    Load macro data from parquet or CSV

    Args:
        data_path: Path to macro data file
        feature_order: List of expected features

    Returns:
        DataFrame with macro features
    """
    logger.info(f"Loading macro data from {data_path}")

    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, parse_dates=['timestamp'])

    logger.info(f"Loaded {len(df)} rows")

    # Check for required features
    missing = [f for f in feature_order if f not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")

    return df


def label_by_vix_bands(df: pd.DataFrame, vix_bands: dict) -> pd.Series:
    """
    Create initial labels using VIX bands

    Args:
        df: DataFrame with VIX column
        vix_bands: Dict of regime -> [min_vix, max_vix]

    Returns:
        Series of regime labels
    """
    labels = pd.Series(["neutral"] * len(df), index=df.index)

    vix = df['VIX'].fillna(20.0)  # Default to neutral

    for regime, (vix_min, vix_max) in vix_bands.items():
        mask = (vix >= vix_min) & (vix < vix_max)
        labels[mask] = regime

    logger.info(f"VIX-based label distribution:")
    for regime in ["risk_on", "neutral", "risk_off", "crisis"]:
        count = (labels == regime).sum()
        pct = count / len(labels) * 100
        logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

    return labels


def train_gmm_classifier(
    X: np.ndarray,
    n_components: int = 4,
    random_state: int = 42
) -> GaussianMixture:
    """
    Train Gaussian Mixture Model

    Args:
        X: Feature matrix (scaled)
        n_components: Number of clusters (regimes)
        random_state: Random seed

    Returns:
        Trained GaussianMixture model
    """
    logger.info(f"Training GMM with {n_components} components...")

    model = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        max_iter=200,
        n_init=10,
        random_state=random_state
    )

    model.fit(X)

    # Evaluate clustering quality
    labels = model.predict(X)
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)

    logger.info(f"GMM trained successfully")
    logger.info(f"  Converged: {model.converged_}")
    logger.info(f"  Iterations: {model.n_iter_}")
    logger.info(f"  BIC: {model.bic(X):.2f}")
    logger.info(f"  AIC: {model.aic(X):.2f}")
    logger.info(f"  Silhouette Score: {silhouette:.3f} (higher is better, range [-1, 1])")
    logger.info(f"  Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")

    return model


def map_clusters_to_regimes(
    model: GaussianMixture,
    X: np.ndarray,
    feature_order: list,
    vix_idx: int
) -> dict:
    """
    Map GMM cluster IDs to regime labels based on cluster means

    Args:
        model: Trained GMM
        X: Feature matrix
        feature_order: List of feature names
        vix_idx: Index of VIX in feature_order

    Returns:
        Dict mapping cluster_id -> regime_label
    """
    logger.info("Mapping clusters to regimes based on VIX means...")

    # Get cluster means
    means = model.means_  # Shape: (n_components, n_features)

    # Get VIX values for each cluster
    cluster_vix = means[:, vix_idx]

    # Sort clusters by VIX (ascending)
    sorted_indices = np.argsort(cluster_vix)

    # Map to regimes: lowest VIX = risk_on, highest = crisis
    regime_names = ["risk_on", "neutral", "risk_off", "crisis"]
    label_map = {}

    for i, cluster_id in enumerate(sorted_indices):
        if i < len(regime_names):
            label_map[int(cluster_id)] = regime_names[i]
        else:
            # If more clusters than regimes, map extras to neutral
            label_map[int(cluster_id)] = "neutral"

    logger.info("Cluster -> Regime mapping:")
    for cluster_id in sorted(label_map.keys()):
        vix_mean = cluster_vix[cluster_id]
        regime = label_map[cluster_id]
        logger.info(f"  Cluster {cluster_id} (VIX={vix_mean:.1f}) → {regime}")

    return label_map


def validate_classifier(
    classifier: RegimeClassifier,
    X_test: np.ndarray,
    feature_order: list,
    vix_labels: pd.Series
) -> dict:
    """
    Validate classifier on test set

    Args:
        classifier: Trained RegimeClassifier
        X_test: Test feature matrix (unscaled)
        feature_order: List of feature names
        vix_labels: True labels from VIX bands

    Returns:
        Dict of validation metrics
    """
    logger.info("Validating classifier on test set...")

    predictions = []
    confidences = []

    for i in range(len(X_test)):
        macro_row = {feat: X_test[i, j] for j, feat in enumerate(feature_order)}
        result = classifier.classify(macro_row)
        predictions.append(result['regime'])
        confidences.append(result['proba'][result['regime']])

    predictions = pd.Series(predictions, index=vix_labels.index)

    # Calculate agreement with VIX labels
    agreement = (predictions == vix_labels).sum() / len(vix_labels) * 100

    logger.info(f"Validation Results:")
    logger.info(f"  Agreement with VIX labels: {agreement:.1f}%")
    logger.info(f"  Mean confidence: {np.mean(confidences):.2f}")

    # Distribution
    logger.info("Predicted regime distribution:")
    for regime in ["risk_on", "neutral", "risk_off", "crisis"]:
        count = (predictions == regime).sum()
        pct = count / len(predictions) * 100
        logger.info(f"  {regime:12s}: {count:6d} ({pct:5.1f}%)")

    return {
        'agreement_pct': agreement,
        'mean_confidence': np.mean(confidences),
        'predictions': predictions.value_counts().to_dict()
    }


def main():
    parser = argparse.ArgumentParser(description="Train regime classifier")
    parser.add_argument('--data', required=True, help="Path to macro data (parquet or CSV)")
    parser.add_argument('--output', default="models/regime_classifier_gmm.pkl", help="Output model path")
    parser.add_argument('--policy', default="configs/v19/regime_policy.json", help="Path to regime policy config")
    parser.add_argument('--n-components', type=int, default=4, help="Number of GMM components (default: 4)")
    parser.add_argument('--test-size', type=float, default=0.2, help="Test set fraction (default: 0.2)")
    parser.add_argument('--random-seed', type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("="*70)
    print("🔬 Bull Machine v1.9 - Regime Classifier Training")
    print("="*70)

    # Load policy config for feature order and VIX bands
    with open(args.policy) as f:
        policy_cfg = json.load(f)

    feature_order = policy_cfg['feature_order']
    vix_bands = policy_cfg['training']['vix_bands']

    logger.info(f"Features: {feature_order}")
    logger.info(f"VIX bands: {vix_bands}")

    # Load macro data
    df = load_macro_data(args.data, feature_order)

    # Extract features
    X = df[feature_order].values

    # Handle NaN values
    n_nan_before = np.isnan(X).sum()
    if n_nan_before > 0:
        logger.warning(f"Found {n_nan_before} NaN values, imputing with column means")
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = col_means[i]

    # Create initial labels from VIX bands
    vix_labels = label_by_vix_bands(df, vix_bands)

    # Train/test split (time-series aware - use last 20% as test)
    split_idx = int(len(X) * (1 - args.test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    labels_train, labels_test = vix_labels.iloc[:split_idx], vix_labels.iloc[split_idx:]

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train GMM
    model = train_gmm_classifier(X_train_scaled, n_components=args.n_components, random_state=args.random_seed)

    # Map clusters to regimes
    vix_idx = feature_order.index('VIX')
    label_map = map_clusters_to_regimes(model, X_train_scaled, feature_order, vix_idx)

    # Create classifier
    classifier = RegimeClassifier(model=model, label_map=label_map, feature_order=feature_order)

    # Validate
    validation_metrics = validate_classifier(classifier, X_test, feature_order, labels_test)

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_obj = {
        'model': model,
        'label_map': label_map,
        'feature_order': feature_order,
        'scaler': scaler,
        'validation_metrics': validation_metrics,
        'training_config': {
            'n_components': args.n_components,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'random_seed': args.random_seed
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_obj, f)

    logger.info(f"✅ Model saved to {output_path}")

    # Feature importance
    importance = classifier.get_feature_importance()
    logger.info("\n📈 Feature Importance (variance across clusters):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  {feat:12s}: {imp:.4f}")

    print("\n" + "="*70)
    print("✅ Training complete!")
    print(f"   Agreement with VIX labels: {validation_metrics['agreement_pct']:.1f}%")
    print(f"   Mean confidence: {validation_metrics['mean_confidence']:.2f}")
    print(f"   Model saved: {output_path}")
    print("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
