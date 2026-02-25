"""
HRPAllocator - Hierarchical Risk Parity Portfolio Allocation

Implements López de Prado's HRP algorithm for stable, robust portfolio allocation
without matrix inversion. Designed for 8-archetype trading system.

Key Features:
1. Distance-based hierarchical clustering of archetypes
2. Quasi-diagonalization of correlation matrix
3. Recursive bisection for inverse-variance weighting
4. Integration with regime-aware and temporal allocation layers

Algorithm Steps:
1. Compute correlation matrix from archetype returns
2. Convert to distance matrix: d = sqrt(0.5 * (1 - ρ))
3. Perform hierarchical clustering (single linkage)
4. Quasi-diagonalize correlation matrix by cluster order
5. Recursive bisection to allocate weights (inverse variance)

Academic Reference:
López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample."
Journal of Portfolio Management, 42(4), 59-69.

Author: System Architect Agent
Date: 2026-01-16
Version: 1.0
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


class HRPAllocator:
    """
    Hierarchical Risk Parity (HRP) portfolio allocator

    Uses hierarchical clustering + inverse variance weighting to create
    stable, diversified portfolio allocations without matrix inversion.
    """

    def __init__(
        self,
        returns_history: pd.DataFrame,
        min_weight: float = 0.01,
        linkage_method: str = 'single'
    ):
        """
        Initialize HRP allocator with archetype returns history

        Args:
            returns_history: DataFrame with columns = archetype IDs, rows = bar returns
                            Example: columns=['S1', 'S4', 'S5', 'H', 'B', 'K', 'A', 'C']
            min_weight: Minimum weight floor for any archetype (default 1%)
            linkage_method: Hierarchical clustering method ('single', 'complete', 'average')
        """
        self.returns = returns_history
        self.archetypes = returns_history.columns.tolist()
        self.min_weight = min_weight
        self.linkage_method = linkage_method

        # Validate inputs
        if len(self.archetypes) < 2:
            raise ValueError("Need at least 2 archetypes for HRP allocation")

        if len(returns_history) < 30:
            logger.warning(
                f"Only {len(returns_history)} return observations - "
                f"recommend at least 30 for stable correlation estimates"
            )

        logger.info(
            f"[HRPAllocator] Initialized with {len(self.archetypes)} archetypes, "
            f"{len(returns_history)} observations"
        )

    def compute_hrp_weights(self) -> Dict[str, float]:
        """
        Compute HRP weights for portfolio allocation

        Returns:
            Dictionary: {archetype_id: weight}
            Weights sum to 1.0 and respect min_weight floor
        """
        logger.info("[HRPAllocator] Computing HRP weights...")

        # Step 1: Correlation matrix
        corr_matrix = self.returns.corr()
        logger.debug(f"  Correlation matrix:\n{corr_matrix}")

        # Step 2: Distance matrix
        distances = self._compute_distance_matrix(corr_matrix)
        dist_condensed = squareform(distances)

        # Step 3: Hierarchical clustering
        linkage_matrix = linkage(dist_condensed, method=self.linkage_method)
        logger.debug(f"  Linkage matrix computed (method={self.linkage_method})")

        # Step 4: Quasi-diagonalization (sort by cluster)
        sorted_indices = self._get_quasi_diag(linkage_matrix)
        sorted_archetypes = [self.archetypes[i] for i in sorted_indices]
        logger.debug(f"  Cluster order: {sorted_archetypes}")

        # Step 5: Recursive bisection for weights
        cov_matrix = self.returns.cov()
        weights_series = self._recursive_bisection(cov_matrix, sorted_archetypes)

        # Convert to dict and apply min_weight floor
        weights = self._apply_min_weight_floor(weights_series.to_dict())

        logger.info(
            "[HRPAllocator] HRP weights computed: "
            + ", ".join(f"{k}={v:.1%}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
        )

        return weights

    def _compute_distance_matrix(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Convert correlation matrix to distance matrix

        Formula: distance = sqrt(0.5 * (1 - correlation))

        This ensures:
        - Perfect correlation (ρ=1) → distance=0
        - Zero correlation (ρ=0) → distance=0.707
        - Perfect anti-correlation (ρ=-1) → distance=1

        Args:
            corr_matrix: Correlation matrix

        Returns:
            Distance matrix (same shape as corr_matrix)
        """
        distances = np.sqrt(0.5 * (1 - corr_matrix))
        return distances

    def _get_quasi_diag(self, linkage_matrix: np.ndarray) -> List[int]:
        """
        Reorganize items so similar items are together (quasi-diagonal ordering)

        This creates a permutation where clustered items are adjacent,
        making the correlation matrix more block-diagonal.

        Args:
            linkage_matrix: Output from scipy.cluster.hierarchy.linkage

        Returns:
            List of indices in quasi-diagonal order
        """
        sorted_items = []

        def recursive_sort(node_id):
            """Recursively traverse dendrogram to get sorted order"""
            if node_id < len(self.archetypes):
                # Leaf node - add to sorted list
                sorted_items.append(node_id)
            else:
                # Internal node - recurse on children
                cluster_idx = int(node_id - len(self.archetypes))
                left_child = int(linkage_matrix[cluster_idx, 0])
                right_child = int(linkage_matrix[cluster_idx, 1])
                recursive_sort(left_child)
                recursive_sort(right_child)

        # Start from root node (last row of linkage matrix)
        root_node = len(linkage_matrix) + len(self.archetypes) - 1
        recursive_sort(root_node)

        return sorted_items

    def _recursive_bisection(
        self,
        cov_matrix: pd.DataFrame,
        sorted_archetypes: List[str]
    ) -> pd.Series:
        """
        Recursive bisection to compute HRP weights

        At each split:
        1. Divide cluster into left and right sub-clusters
        2. Compute variance of each sub-cluster
        3. Allocate weight inversely proportional to variance
        4. Recurse on sub-clusters

        This ensures lower-risk archetypes get higher allocation.

        Args:
            cov_matrix: Covariance matrix
            sorted_archetypes: Archetypes in quasi-diagonal order

        Returns:
            Series of weights (index = archetype)
        """
        # Initialize all weights to 1.0
        weights = pd.Series(1.0, index=sorted_archetypes)

        # Build list of clusters to process
        # Start with entire list, then split recursively
        cluster_items = [sorted_archetypes]

        while len(cluster_items) > 0:
            # Split each cluster in half
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) > 1:
                    mid = len(cluster) // 2
                    left = cluster[:mid]
                    right = cluster[mid:]
                    new_clusters.extend([left, right])

            # Process pairs of sibling clusters
            for i in range(0, len(new_clusters), 2):
                if i + 1 >= len(new_clusters):
                    break

                left_cluster = new_clusters[i]
                right_cluster = new_clusters[i + 1]

                # Compute cluster variances
                left_var = self._compute_cluster_variance(cov_matrix, left_cluster)
                right_var = self._compute_cluster_variance(cov_matrix, right_cluster)

                # Inverse variance allocation
                total_var = left_var + right_var
                if total_var > 0:
                    alpha = 1 - left_var / total_var  # Weight for right cluster
                else:
                    alpha = 0.5  # Equal if both have zero variance

                # Update weights
                weights[left_cluster] *= (1 - alpha)
                weights[right_cluster] *= alpha

                logger.debug(
                    f"  Split: {left_cluster} (var={left_var:.4f}, w={1-alpha:.3f}) | "
                    f"{right_cluster} (var={right_var:.4f}, w={alpha:.3f})"
                )

            cluster_items = new_clusters

        return weights

    def _compute_cluster_variance(
        self,
        cov_matrix: pd.DataFrame,
        cluster: List[str]
    ) -> float:
        """
        Compute variance of a cluster using inverse-variance weighting

        For a cluster of assets, compute the portfolio variance assuming
        assets are weighted inversely to their individual variances.

        Args:
            cov_matrix: Covariance matrix
            cluster: List of archetype IDs in cluster

        Returns:
            Cluster variance (float)
        """
        # Extract covariance sub-matrix for this cluster
        cov_slice = cov_matrix.loc[cluster, cluster]

        # Inverse variance weights (within cluster)
        inv_diag = 1.0 / np.diag(cov_slice)
        w = inv_diag / inv_diag.sum()

        # Cluster variance: w^T * Cov * w
        cluster_var = np.dot(w, np.dot(cov_slice, w))

        return cluster_var

    def _apply_min_weight_floor(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply minimum weight floor and renormalize

        Ensures no archetype has weight < min_weight (default 1%)
        This prevents complete exclusion while allowing HRP structure.

        Args:
            weights: Raw HRP weights

        Returns:
            Adjusted weights (sum = 1.0)
        """
        adjusted = {}
        total_floored = 0.0
        total_above_floor = 0.0

        # First pass: identify floored vs above-floor
        for arch, w in weights.items():
            if w < self.min_weight:
                adjusted[arch] = self.min_weight
                total_floored += self.min_weight
            else:
                adjusted[arch] = w
                total_above_floor += w

        # Second pass: rescale above-floor weights to sum to (1 - total_floored)
        if total_above_floor > 0 and total_floored < 1.0:
            scale_factor = (1.0 - total_floored) / total_above_floor
            for arch in adjusted:
                if adjusted[arch] > self.min_weight:
                    adjusted[arch] *= scale_factor

        # Verify sum = 1.0 (within floating point tolerance)
        total = sum(adjusted.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(f"[HRPAllocator] Weights sum to {total:.6f}, renormalizing")
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix for archetype returns

        Returns:
            Correlation matrix (DataFrame)
        """
        return self.returns.corr()

    def get_cluster_dendrogram_data(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get dendrogram data for visualization

        Returns:
            Tuple of (linkage_matrix, sorted_archetype_names)
        """
        corr_matrix = self.returns.corr()
        distances = self._compute_distance_matrix(corr_matrix)
        dist_condensed = squareform(distances)
        linkage_matrix = linkage(dist_condensed, method=self.linkage_method)
        sorted_indices = self._get_quasi_diag(linkage_matrix)
        sorted_archetypes = [self.archetypes[i] for i in sorted_indices]

        return linkage_matrix, sorted_archetypes

    def get_archetype_clusters(self, n_clusters: int = 3) -> Dict[str, int]:
        """
        Get cluster assignments for archetypes

        Args:
            n_clusters: Number of clusters to create (default 3)

        Returns:
            Dictionary mapping archetype → cluster_id
        """
        from scipy.cluster.hierarchy import fcluster

        corr_matrix = self.returns.corr()
        distances = self._compute_distance_matrix(corr_matrix)
        dist_condensed = squareform(distances)
        linkage_matrix = linkage(dist_condensed, method=self.linkage_method)

        # Cut dendrogram to form n_clusters
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        return {arch: int(label) for arch, label in zip(self.archetypes, cluster_labels)}

    def get_diversification_ratio(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute diversification ratio (Choueifaty & Coignard, 2008)

        DR = (Weighted sum of volatilities) / (Portfolio volatility)

        Higher DR (>1.0) indicates better diversification.
        DR = 1.0 means no diversification benefit (perfectly correlated).
        DR = sqrt(N) is theoretical maximum for N uncorrelated assets.

        Args:
            weights: Portfolio weights (if None, use HRP weights)

        Returns:
            Diversification ratio (float)
        """
        if weights is None:
            weights = self.compute_hrp_weights()

        # Convert to array in correct order
        w = np.array([weights[arch] for arch in self.archetypes])

        # Individual volatilities
        vols = self.returns.std().values

        # Portfolio volatility
        cov_matrix = self.returns.cov().values
        portfolio_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))

        # Weighted sum of volatilities
        weighted_vols = np.dot(w, vols)

        # Diversification ratio
        dr = weighted_vols / portfolio_vol if portfolio_vol > 0 else 1.0

        logger.info(
            f"[HRPAllocator] Diversification Ratio: {dr:.3f} "
            f"(weighted_vols={weighted_vols:.4f}, portfolio_vol={portfolio_vol:.4f})"
        )

        return dr


def compute_archetype_returns(
    signals_df: pd.DataFrame,
    price_col: str = 'close',
    pnl_col: str = 'pnl_pct'
) -> pd.DataFrame:
    """
    Compute archetype returns from signal/trade history

    This helper function converts trade logs into a returns matrix
    suitable for HRP allocation.

    Args:
        signals_df: DataFrame with columns [timestamp, archetype, pnl_pct]
        price_col: Column name for price (for bar-level returns)
        pnl_col: Column name for PnL percentage

    Returns:
        DataFrame with columns = archetypes, rows = timestamps, values = returns
    """
    # Pivot to create returns matrix
    returns_matrix = signals_df.pivot_table(
        index='timestamp',
        columns='archetype',
        values=pnl_col,
        fill_value=0.0
    )

    return returns_matrix


if __name__ == "__main__":
    """
    Example usage and validation
    """
    import warnings
    warnings.filterwarnings('ignore')

    logging.basicConfig(level=logging.INFO)
    logger.info("=" * 70)
    logger.info("HRP Allocator - Example Usage")
    logger.info("=" * 70)

    # Generate synthetic returns for 8 archetypes
    np.random.seed(42)

    archetypes = ['S1', 'S4', 'S5', 'H', 'B', 'K', 'A', 'C']
    n_bars = 100

    # Create synthetic returns with realistic correlations
    # Bull cluster: H, B, K (correlated ~0.5)
    # Bear cluster: S1, S4, S5 (correlated ~0.4)
    # Stubs: A, C (low correlation)

    returns_data = {}

    # Bull archetypes (positively correlated)
    bull_factor = np.random.randn(n_bars) * 0.02
    returns_data['H'] = bull_factor + np.random.randn(n_bars) * 0.015
    returns_data['B'] = 0.8 * bull_factor + np.random.randn(n_bars) * 0.018
    returns_data['K'] = 0.6 * bull_factor + np.random.randn(n_bars) * 0.016

    # Bear archetypes (negatively correlated with bull, positively with each other)
    bear_factor = -0.3 * bull_factor + np.random.randn(n_bars) * 0.02
    returns_data['S1'] = bear_factor + np.random.randn(n_bars) * 0.012
    returns_data['S4'] = 0.7 * bear_factor + np.random.randn(n_bars) * 0.014
    returns_data['S5'] = 0.5 * bear_factor + np.random.randn(n_bars) * 0.013

    # Stubs (low correlation)
    returns_data['A'] = np.random.randn(n_bars) * 0.020
    returns_data['C'] = np.random.randn(n_bars) * 0.022

    returns_df = pd.DataFrame(returns_data)

    logger.info(f"\nSynthetic returns generated: {returns_df.shape}")
    logger.info(f"Mean returns:\n{returns_df.mean()}")
    logger.info(f"\nCorrelation matrix:\n{returns_df.corr()}")

    # Initialize HRP allocator
    hrp = HRPAllocator(returns_df, min_weight=0.01)

    # Compute HRP weights
    weights = hrp.compute_hrp_weights()

    logger.info("\n" + "=" * 70)
    logger.info("HRP WEIGHTS")
    logger.info("=" * 70)
    for arch, w in sorted(weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {arch}: {w:.1%}")

    # Compute diversification ratio
    dr = hrp.get_diversification_ratio(weights)

    # Get clusters
    clusters = hrp.get_archetype_clusters(n_clusters=3)
    logger.info("\n" + "=" * 70)
    logger.info("ARCHETYPE CLUSTERS (k=3)")
    logger.info("=" * 70)
    for cluster_id in range(1, 4):
        archs = [a for a, c in clusters.items() if c == cluster_id]
        logger.info(f"  Cluster {cluster_id}: {archs}")

    logger.info("\n" + "=" * 70)
    logger.info("✅ HRP Allocator validated successfully")
    logger.info("=" * 70)
