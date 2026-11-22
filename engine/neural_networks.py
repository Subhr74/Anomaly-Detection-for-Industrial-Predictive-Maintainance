"""
PredictX AI - Neural Network Models (Deep Telemetry Analysis)
1. Multilayer Perceptron (ANN) for Remaining Useful Life (RUL) regression.
2. Radial Basis Function Neural Network (RBFNN) for localized radial kernel anomaly classification.
"""

import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans
from typing import Dict, Any, List, Tuple
import time

FAULT_LABELS = {
    0: "Normal Operation",
    1: "Bearing Inner Race Fault",
    2: "Shaft Misalignment",
    3: "Hydraulic Cavitation",
    4: "Spindle Overheating / Thermal Runaway"
}

FAULT_SEVERITY = {
    0: "NORMAL",
    1: "CRITICAL",
    2: "WARNING",
    3: "WARNING",
    4: "CRITICAL"
}


class RadialBasisFunctionNetwork:
    """
    Exact Radial Basis Function Neural Network (RBFNN).
    - Hidden Layer: Gaussian Radial Kernels phi_i(x) = exp(- gamma * ||x - c_i||^2)
    - Centers (c_i): Optimized via K-Means clustering of the operational input space.
    - Width / Spread: Dynamically estimated based on inter-centroid distances.
    - Output Layer: Analytical regularized Ridge pseudo-inverse solution.
    Provides localized receptive fields for non-linear boundary learning and novelty detection.
    """
    def __init__(self, n_centers: int = 15, gamma: float = 0.5):
        self.n_centers = n_centers
        self.gamma = gamma
        self.centers: np.ndarray = np.empty((0, 0))
        self.weights: np.ndarray = np.empty((0, 0))
        self.sigma: float = 1.0
        self.is_fitted: bool = False
        self.n_classes: int = 5

    def _calculate_activations(self, X: np.ndarray) -> np.ndarray:
        """Computes Gaussian kernel activations across all centers."""
        # X: (N, D), centers: (K, D) -> diffs: (N, K, D)
        diff = X[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        # Squared Euclidean distances: (N, K)
        sq_dists = np.sum(diff ** 2, axis=2)
        # Gaussian radial activation
        phi = np.exp(-self.gamma * sq_dists / (2.0 * (self.sigma ** 2)))
        # Add bias column (all ones)
        phi_with_bias = np.hstack([phi, np.ones((X.shape[0], 1))])
        return phi_with_bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RadialBasisFunctionNetwork":
        """
        Fit RBFNN:
        1. Cluster X to establish radial centers c_i.
        2. Compute dynamic width sigma.
        3. Solve weights W using Tikhonov regularized least squares.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes = len(np.unique(y)) if len(np.unique(y)) > 0 else 5

        # Step 1: Select centers using K-Means
        actual_centers = min(self.n_centers, len(X))
        kmeans = KMeans(n_clusters=actual_centers, random_state=42, n_init=5)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # Step 2: Set spread parameter sigma = mean distance between centers
        if len(self.centers) > 1:
            center_diff = self.centers[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
            center_dists = np.sqrt(np.sum(center_diff ** 2, axis=2))
            np.fill_diagonal(center_dists, np.nan)
            self.sigma = float(np.nanmean(center_dists) * 0.8)
            if self.sigma < 1e-3:
                self.sigma = 1.0
        else:
            self.sigma = 1.0

        # Step 3: One-hot encode targets
        Y_onehot = np.zeros((len(y), self.n_classes))
        for i, val in enumerate(y):
            if 0 <= val < self.n_classes:
                Y_onehot[i, val] = 1.0

        # Step 4: Analytical solution with Ridge regularization
        Phi = self._calculate_activations(X)
        reg = 1e-3 * np.eye(Phi.shape[1])
        # W = (Phi^T Phi + lambda I)^{-1} Phi^T Y
        self.weights = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y_onehot)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes class probabilities using Softmax on linear projections."""
        if not self.is_fitted:
            raise RuntimeError("RBFNN must be fitted before predict.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        Phi = self._calculate_activations(X)
        raw_outputs = Phi @ self.weights
        
        # Softmax normalization
        exp_out = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True))
        probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_kernel_activations(self, x: np.ndarray) -> List[float]:
        """Returns localized kernel activations for inspection."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        Phi = self._calculate_activations(x)
        # Exclude bias column
        return [round(float(v), 4) for v in Phi[0][:-1]]


class TelemetryNeuralSuite:
    """
    Combined Neural Suite:
    1. MLP Regressor for Remaining Useful Life (RUL in hours)
    2. RBFNN for localized non-linear boundary fault classification
    3. MLP Classifier for deep multi-sensor representation
    """
    def __init__(self):
        # MLP for RUL Regression: 6 inputs -> 64 -> 32 -> 1 output
        self.rul_mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=350,
            random_state=42,
            learning_rate_init=0.005
        )
        
        # RBF Neural Network
        self.rbf_nn = RadialBasisFunctionNetwork(n_centers=16, gamma=0.45)

        # MLP Multi-class Classifier
        self.mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            max_iter=300,
            random_state=42
        )
        self.is_fitted: bool = False

    def fit(self, X_scaled: np.ndarray, y_class: np.ndarray, y_rul: np.ndarray) -> "TelemetryNeuralSuite":
        """Train RUL MLP Regressor, RBFNN, and MLP Classifier."""
        self.rul_mlp.fit(X_scaled, y_rul)
        self.rbf_nn.fit(X_scaled, y_class)
        self.mlp_classifier.fit(X_scaled, y_class)
        self.is_fitted = True
        return self

    def predict_rul(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """Predicts Remaining Useful Life in hours."""
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        predicted_rul = float(self.rul_mlp.predict(x_scaled)[0])
        # Floor at 0 hours, cap at 1200 hours
        predicted_rul = max(0.0, min(1200.0, round(predicted_rul, 1)))

        # Compute confidence interval based on wear level
        uncertainty = round(predicted_rul * 0.08 + 5.0, 1)
        rul_lower = max(0.0, round(predicted_rul - uncertainty, 1))
        rul_upper = round(predicted_rul + uncertainty, 1)

        # Health status percentage
        health_index = round((predicted_rul / 1000.0) * 100.0, 1)
        health_index = min(100.0, max(0.0, health_index))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_rul_hours": predicted_rul,
            "confidence_interval": [rul_lower, rul_upper],
            "health_index_pct": health_index,
            "recommended_action_window_hrs": max(4.0, round(predicted_rul * 0.4, 1)),
            "latency_ms": round(latency_ms, 3)
        }

    def predict_rbfnn(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """Inference with Radial Basis Function Neural Network."""
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        probs = self.rbf_nn.predict_proba(x_scaled)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        activations = self.rbf_nn.get_kernel_activations(x_scaled)
        
        # Max kernel activation represents proximity to nearest training radial basin
        max_kernel_act = max(activations) if activations else 1.0
        # If max kernel activation is extremely low, point is outside normal kernel bounds
        is_radial_novelty = max_kernel_act < 0.08

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model": "Radial Basis Function NN (RBFNN)",
            "predicted_fault_id": pred_class,
            "fault_name": FAULT_LABELS[pred_class],
            "severity": FAULT_SEVERITY[pred_class],
            "confidence": round(confidence, 4),
            "class_probabilities": {FAULT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "max_kernel_activation": round(max_kernel_act, 4),
            "is_radial_novelty": is_radial_novelty,
            "active_centers_count": int(np.sum(np.array(activations) > 0.2)),
            "latency_ms": round(latency_ms, 3)
        }

    def predict_ann(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """Inference with Multilayer Perceptron Classifier."""
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        probs = self.mlp_classifier.predict_proba(x_scaled)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model": "Multilayer Perceptron (ANN)",
            "predicted_fault_id": pred_class,
            "fault_name": FAULT_LABELS[pred_class],
            "severity": FAULT_SEVERITY[pred_class],
            "confidence": round(confidence, 4),
            "class_probabilities": {FAULT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "latency_ms": round(latency_ms, 3)
        }
