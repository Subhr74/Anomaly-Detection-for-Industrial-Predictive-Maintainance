"""
PredictX AI - Unsupervised Anomaly Detection Module
Implements K-Means, DBSCAN, and Agglomerative Hierarchical Clustering
for cold-start operational state modeling, drift tracking, and noise detection.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import Dict, Any, List, Optional
import time


class UnsupervisedAnomalyDetector:
    """
    Multi-model unsupervised anomaly detection engine.
    - K-Means: Centroid-distance based anomaly scoring with dynamic thresholding.
    - DBSCAN: Density-reachability noise detection (cluster -1).
    - Agglomerative Clustering: Hierarchical tree clustering for operational regimes.
    """
    def __init__(self, n_clusters: int = 4, eps: float = 1.4, min_samples: int = 5):
        self.n_clusters = n_clusters
        self.eps = eps
        self.min_samples = min_samples

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.agglomerative = AgglomerativeClustering(n_clusters=self.n_clusters)
        
        self.kmeans_threshold_: float = 3.0
        self.kmeans_fitted: bool = False
        self.dbscan_fitted: bool = False
        self.reference_data_scaled: Optional[np.ndarray] = None
        self.reference_labels_agg: Optional[np.ndarray] = None
        self.linkage_matrix: Optional[np.ndarray] = None
        self.operational_regimes_: List[str] = [
            "Regime 0: Low-Load Spindle Idling / Warmup",
            "Regime 1: Nominal High-Speed Machining",
            "Regime 2: Heavy Cutting / High-Torque Milling",
            "Regime 3: Transitional Dynamic Shift"
        ]

    def fit(self, X_scaled: np.ndarray) -> "UnsupervisedAnomalyDetector":
        """
        Fit K-Means, DBSCAN, and Agglomerative models on scaled baseline data.
        Determines dynamic Euclidean anomaly threshold at 95th percentile.
        """
        self.reference_data_scaled = X_scaled.copy()
        
        # 1. Fit K-Means
        self.kmeans.fit(X_scaled)
        # Calculate distances from each training point to its assigned cluster centroid
        centroids = self.kmeans.cluster_centers_
        assigned_centroids = centroids[self.kmeans.labels_]
        train_distances = np.linalg.norm(X_scaled - assigned_centroids, axis=1)
        # Dynamic 97.5th percentile cut-off for anomaly threshold
        self.kmeans_threshold_ = float(np.percentile(train_distances, 97.5) * 1.25)
        self.kmeans_fitted = True

        # 2. Fit DBSCAN on baseline sample (subsample if large to maintain speed)
        n_samples = min(len(X_scaled), 600)
        subsample = X_scaled[:n_samples]
        self.dbscan.fit(subsample)
        self.dbscan_fitted = True

        # 3. Fit Agglomerative Clustering & Linkage Matrix for Dendrogram
        dendro_sample = X_scaled[:min(len(X_scaled), 60)]
        self.reference_labels_agg = self.agglomerative.fit_predict(dendro_sample)
        self.linkage_matrix = linkage(dendro_sample, method='ward')

        return self

    def predict_kmeans(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Scores sample using K-Means centroid distance.
        Points beyond dynamic threshold are flagged as anomalous.
        """
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)
            
        distances = self.kmeans.transform(x_scaled)[0]
        nearest_cluster = int(np.argmin(distances))
        min_distance = float(distances[nearest_cluster])
        
        is_anomaly = bool(min_distance > self.kmeans_threshold_)
        # Normalized anomaly score [0.0 - 1.0]
        anomaly_score = float(min(1.0, min_distance / (self.kmeans_threshold_ * 1.8)))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model": "K-Means Clustering",
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "nearest_cluster": nearest_cluster,
            "cluster_regime": self.operational_regimes_[nearest_cluster % len(self.operational_regimes_)],
            "euclidean_distance": round(min_distance, 3),
            "threshold": round(self.kmeans_threshold_, 3),
            "latency_ms": round(latency_ms, 3)
        }

    def predict_dbscan(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Determines if point fits within the dense manifold or is isolated noise.
        Computes minimum distance to core samples.
        """
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        core_samples = self.dbscan.components_
        if len(core_samples) == 0:
            core_samples = self.reference_data_scaled[:100]

        # Distance to all core samples in DBSCAN manifold
        dists = np.linalg.norm(core_samples - x_scaled, axis=1)
        min_core_dist = float(np.min(dists))

        # A point is classified as noise (-1) if its distance to nearest core sample exceeds eps
        is_noise = min_core_dist > self.eps
        anomaly_score = float(min(1.0, min_core_dist / (self.eps * 1.7)))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model": "DBSCAN Density Clustering",
            "is_anomaly": is_noise,
            "cluster_id": -1 if is_noise else 0,
            "anomaly_score": round(anomaly_score, 4),
            "min_core_distance": round(min_core_dist, 3),
            "eps_threshold": round(self.eps, 3),
            "latency_ms": round(latency_ms, 3)
        }

    def get_dendrogram_data(self) -> Dict[str, Any]:
        """
        Serializes hierarchical linkage structure for frontend dendrogram rendering.
        """
        if self.linkage_matrix is None:
            return {"nodes": [], "regimes": self.operational_regimes_}

        # Structure top 15 merges of the dendrogram
        merges = []
        for i, row in enumerate(self.linkage_matrix[-15:]):
            merges.append({
                "step": int(i + 1),
                "cluster_a": int(row[0]),
                "cluster_b": int(row[1]),
                "distance": round(float(row[2]), 3),
                "sample_count": int(row[3])
            })
            
        return {
            "top_merges": merges,
            "regimes": self.operational_regimes_,
            "total_regimes": self.n_clusters
        }

