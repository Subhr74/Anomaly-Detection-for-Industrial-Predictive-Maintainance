"""
PredictX AI - Supervised Fault Classification Module
Implements Decision Tree, Random Forest, and k-Nearest Neighbors (k-NN)
with white-box tree path feature attribution and historical failure analogue retrieval.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from typing import Dict, Any, List, Tuple, Optional
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


class SupervisedFaultClassifier:
    """
    Supervised industrial failure mode classifier suite:
    - Decision Tree: White-box human-interpretable conditional rules.
    - Random Forest: High-accuracy ensemble with feature importance attribution.
    - k-NN: Instance-based nearest neighbor analogue lookup in historical records.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.dt_classifier = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42)
        self.rf_classifier = RandomForestClassifier(n_estimators=60, max_depth=7, random_state=42, n_jobs=1)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        self.nn_searcher = NearestNeighbors(n_neighbors=5, metric='minkowski', p=2)

        self.historical_records: List[Dict[str, Any]] = []
        self.is_fitted: bool = False

    def fit(self, X_scaled: np.ndarray, y: np.ndarray, X_raw: Optional[np.ndarray] = None) -> "SupervisedFaultClassifier":
        """Fit DT, RF, and k-NN on scaled training telemetry."""
        self.dt_classifier.fit(X_scaled, y)
        self.rf_classifier.fit(X_scaled, y)
        self.knn_classifier.fit(X_scaled, y)

        # Store historical fleet breakdown incidents
        n_samples = len(y)
        hist_limit = min(n_samples, 500)
        self.historical_records = []
        for i in range(hist_limit):
            raw_vals = X_raw[i] if X_raw is not None else X_scaled[i]
            label = int(y[i])
            self.historical_records.append({
                "record_id": f"HIST-FLT-202{label % 5}-{1000 + i}",
                "fault_code": label,
                "fault_name": FAULT_LABELS[label],
                "severity": FAULT_SEVERITY[label],
                "telemetry": {self.feature_names[j]: round(float(raw_vals[j]), 2) for j in range(len(self.feature_names))},
                "resolution": self._get_historical_resolution(label)
            })

        # Fit nearest neighbors searcher on the exact subset stored in historical_records
        self.nn_searcher.fit(X_scaled[:hist_limit])

        self.is_fitted = True
        return self

    def _get_historical_resolution(self, label: int) -> str:
        resolutions = {
            0: "Standard preventative routine inspection; all clearance tolerances verified nominal.",
            1: "Emergency shutdown executed; ceramic hybrid spindle bearings replaced; inner raceway showed micro-spalling.",
            2: "Laser alignment recalibration performed; flexible shaft coupling replaced; angular deviation corrected to <0.02 mm.",
            3: "Hydraulic reservoir suction filter purged and back-flushed; anti-foaming hydraulic fluid replenished; pump cavitation silenced.",
            4: "Coolant chiller manifold unclogged; heat exchanger flushed; spindle jacket flow rate restored from 1.2 to 4.8 L/min."
        }
        return resolutions.get(label, "Operational parameters recalibrated to nominal baseline.")

    def predict_decision_tree(self, x_scaled: np.ndarray, x_raw: np.ndarray) -> Dict[str, Any]:
        """
        Inference with Decision Tree + White-Box Decision Path Rule Extraction.
        """
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)
        if x_raw.ndim == 1:
            x_raw = x_raw.reshape(1, -1)

        pred_class = int(self.dt_classifier.predict(x_scaled)[0])
        probs = self.dt_classifier.predict_proba(x_scaled)[0]
        confidence = float(np.max(probs))

        # Extract white-box decision path
        node_indicator = self.dt_classifier.decision_path(x_scaled)
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

        tree = self.dt_classifier.tree_
        path_rules = []
        feature_importance_map = {}

        for node_id in node_index:
            # If not a leaf node
            if tree.children_left[node_id] != tree.children_right[node_id]:
                feature_idx = tree.feature[node_id]
                feature_name = self.feature_names[feature_idx]
                threshold_scaled = tree.threshold[node_id]
                sample_val = x_scaled[0, feature_idx]

                comparison = "<=" if sample_val <= threshold_scaled else ">"
                rule_desc = f"{feature_name} {comparison} threshold (scaled val: {sample_val:.2f})"
                path_rules.append(rule_desc)
                feature_importance_map[feature_name] = feature_importance_map.get(feature_name, 0) + 1

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Feature importances from tree
        importances = {
            self.feature_names[i]: round(float(imp), 4)
            for i, imp in enumerate(self.dt_classifier.feature_importances_)
        }

        return {
            "model": "Decision Tree (White-Box)",
            "predicted_fault_id": pred_class,
            "fault_name": FAULT_LABELS[pred_class],
            "severity": FAULT_SEVERITY[pred_class],
            "confidence": round(confidence, 4),
            "class_probabilities": {FAULT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "decision_path": path_rules,
            "feature_importances": importances,
            "latency_ms": round(latency_ms, 3)
        }

    def predict_random_forest(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """Inference with Random Forest ensemble."""
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        pred_class = int(self.rf_classifier.predict(x_scaled)[0])
        probs = self.rf_classifier.predict_proba(x_scaled)[0]
        confidence = float(np.max(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        importances = {
            self.feature_names[i]: round(float(imp), 4)
            for i, imp in enumerate(self.rf_classifier.feature_importances_)
        }

        return {
            "model": "Random Forest Ensemble",
            "predicted_fault_id": pred_class,
            "fault_name": FAULT_LABELS[pred_class],
            "severity": FAULT_SEVERITY[pred_class],
            "confidence": round(confidence, 4),
            "class_probabilities": {FAULT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "feature_importances": importances,
            "latency_ms": round(latency_ms, 3)
        }

    def predict_knn(self, x_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Inference with k-NN + Historical Incident Analogue Retrieval.
        Finds Top-3 closest past maintenance breakdowns in feature space.
        """
        t0 = time.perf_counter()
        if x_scaled.ndim == 1:
            x_scaled = x_scaled.reshape(1, -1)

        pred_class = int(self.knn_classifier.predict(x_scaled)[0])
        probs = self.knn_classifier.predict_proba(x_scaled)[0]
        confidence = float(np.max(probs))

        # Retrieve nearest neighbor analogues
        distances, indices = self.nn_searcher.kneighbors(x_scaled, n_neighbors=3)
        nearest_analogues = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.historical_records):
                record = self.historical_records[idx].copy()
                similarity = max(0.0, 100.0 - float(dist) * 18.0)
                record["similarity_pct"] = round(similarity, 1)
                record["euclidean_distance"] = round(float(dist), 3)
                nearest_analogues.append(record)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "model": "k-Nearest Neighbors (k-NN)",
            "predicted_fault_id": pred_class,
            "fault_name": FAULT_LABELS[pred_class],
            "severity": FAULT_SEVERITY[pred_class],
            "confidence": round(confidence, 4),
            "class_probabilities": {FAULT_LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)},
            "nearest_historical_analogues": nearest_analogues,
            "latency_ms": round(latency_ms, 3)
        }
