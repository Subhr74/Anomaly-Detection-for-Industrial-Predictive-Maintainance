"""
PredictX AI - Sensor Data Preprocessor & Feature Engineering Module
Standardizes multivariate industrial telemetry, calculates Mahalanobis distance matrices,
and computes dimensionality reduction (PCA) for cluster visualization.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, List

# Sensor telemetry feature names
FEATURE_NAMES = [
    "vibration_rms",       # mm/s: Overall RMS vibration velocity (ISO 10816)
    "bearing_temp",        # °C: Spindle bearing race temperature
    "acoustic_emission",   # dB: Ultrasonic stress wave emissions (spalling/cracks)
    "hydraulic_pressure",  # bar: Hydraulic fluid lubrication & clamp pressure
    "spindle_rpm",         # RPM: Rotational speed of spindle shaft
    "load_torque"          # Nm: Drive load torque / mechanical resistance
]

FEATURE_UNITS = {
    "vibration_rms": "mm/s",
    "bearing_temp": "°C",
    "acoustic_emission": "dB",
    "hydraulic_pressure": "bar",
    "spindle_rpm": "RPM",
    "load_torque": "Nm"
}

FEATURE_THRESHOLDS = {
    "vibration_rms": {"normal_max": 2.8, "warning": 4.5, "critical": 7.1},
    "bearing_temp": {"normal_max": 65.0, "warning": 75.0, "critical": 88.0},
    "acoustic_emission": {"normal_max": 52.0, "warning": 68.0, "critical": 80.0},
    "hydraulic_pressure": {"normal_min": 140.0, "normal_max": 160.0, "warning_min": 125.0, "critical_min": 110.0},
    "spindle_rpm": {"normal_min": 2900.0, "normal_max": 3100.0, "warning_dev": 200.0},
    "load_torque": {"normal_max": 58.0, "warning": 72.0, "critical": 88.0}
}


class TelemetryPreprocessor:
    """
    Handles normalization, covariance estimation for Mahalanobis anomaly detection,
    and 2D/3D PCA projection for feature space visualization.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.mean_vector: np.ndarray = np.zeros(len(FEATURE_NAMES))
        self.inv_cov_matrix: np.ndarray = np.eye(len(FEATURE_NAMES))
        self.is_fitted: bool = False
        self.feature_names = FEATURE_NAMES

    def fit(self, X: np.ndarray) -> "TelemetryPreprocessor":
        """
        Fit standard scaler, PCA projector, and inverse covariance matrix
        on nominal baseline historical telemetry.
        """
        X = np.asarray(X, dtype=float)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.pca_2d.fit(X_scaled)
        
        # Calculate mean and regularized inverse covariance on scaled data
        self.mean_vector = np.mean(X_scaled, axis=0)
        cov_matrix = np.cov(X_scaled, rowvar=False)
        # Regularization (shrinkage) to prevent singular matrices
        reg_cov = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-4
        self.inv_cov_matrix = np.linalg.pinv(reg_cov)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale telemetry features using fitted StandardScaler."""
        if not self.is_fitted:
            raise RuntimeError("TelemetryPreprocessor must be fitted before transform.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def project_pca(self, X: np.ndarray) -> np.ndarray:
        """Project scaled features to 2D PCA coordinates."""
        if not self.is_fitted:
            raise RuntimeError("TelemetryPreprocessor must be fitted before PCA projection.")
        X_scaled = self.transform(X)
        return self.pca_2d.transform(X_scaled)

    def compute_mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Computes Mahalanobis distance D_M(x) from the baseline normal operational center:
        D_M(x) = sqrt((x - mu)^T * Sigma^{-1} * (x - mu))
        """
        X_scaled = self.transform(X)
        diff = X_scaled - self.mean_vector
        # Vectorized Mahalanobis distance
        left = np.dot(diff, self.inv_cov_matrix)
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def get_pca_explained_variance(self) -> Dict[str, Any]:
        """Returns variance explained by PC1 and PC2."""
        if not self.is_fitted:
            return {"pc1": 0.0, "pc2": 0.0, "total": 0.0}
        ev = self.pca_2d.explained_variance_ratio_
        return {
            "pc1": float(round(ev[0] * 100, 2)),
            "pc2": float(round(ev[1] * 100, 2)),
            "total": float(round(np.sum(ev) * 100, 2)),
            "components": self.pca_2d.components_.tolist()
        }

