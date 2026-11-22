"""
PredictX AI - Comprehensive Automated Test Suite
Validates all ML models, simulator physics, and inference pathways.
"""

import sys
import numpy as np

def run_test_suite():
    print("[TEST] 1. Importing engine modules...")
    from engine.model_manager import ModelManager
    from simulator.telemetry_stream import IndustrialTelemetrySimulator

    print("[TEST] 2. Initializing ModelManager & training baseline models...")
    manager = ModelManager()
    manager.initialize_and_train(n_samples=1000)
    assert manager.is_ready, "ModelManager failed to set is_ready"
    print("       -> ModelManager successfully trained.")

    print("[TEST] 3. Verifying Benchmark Matrix...")
    assert len(manager.benchmark_results) >= 5, "Fewer than 5 models in benchmark results"
    for bm in manager.benchmark_results:
        print(f"       -> {bm['model_name']:<32} | Acc: {bm['accuracy']}% | F1: {bm['f1_score']}% | Latency: {bm['inference_latency_us']} us")
        assert bm['accuracy'] > 85.0, f"Accuracy too low for {bm['model_name']}"

    print("[TEST] 4. Testing Telemetry Simulator physics...")
    sim = IndustrialTelemetrySimulator()
    snap = sim.step()
    assert "telemetry" in snap
    assert "vibration_rms" in snap["telemetry"]
    print(f"       -> Snapshot: Vibration={snap['telemetry']['vibration_rms']} mm/s, Temp={snap['telemetry']['bearing_temp']} C")

    print("[TEST] 5. Testing Multi-Model Inference on Healthy Telemetry...")
    healthy_telemetry = {
        "vibration_rms": 1.65,
        "bearing_temp": 51.5,
        "acoustic_emission": 42.0,
        "hydraulic_pressure": 150.0,
        "spindle_rpm": 3000.0,
        "load_torque": 48.0
    }
    inf = manager.run_inference(healthy_telemetry)
    print(f"       -> Consensus Diagnosis: {inf['consensus']['fault_name']} (Severity: {inf['overall_status']})")
    assert inf["consensus"]["consensus_fault_id"] == 0, "Healthy telemetry was not classified as Normal Operation"
    assert inf["rul"]["predicted_rul_hours"] > 300.0, "Healthy RUL should be high"

    print("[TEST] 6. Testing Bearing Fault Injection...")
    bearing_telemetry = {
        "vibration_rms": 6.8,
        "bearing_temp": 65.0,
        "acoustic_emission": 82.0,
        "hydraulic_pressure": 148.0,
        "spindle_rpm": 2980.0,
        "load_torque": 55.0
    }
    inf_bearing = manager.run_inference(bearing_telemetry)
    print(f"       -> Bearing Fault Diagnosis: {inf_bearing['consensus']['fault_name']} (Severity: {inf_bearing['overall_status']})")
    assert inf_bearing["consensus"]["consensus_fault_id"] == 1, "Bearing fault was not correctly classified"
    assert inf_bearing["overall_status"] == "CRITICAL", "Bearing fault should have CRITICAL severity"
    assert len(inf_bearing["models"]["decision_tree"]["decision_path"]) > 0, "Decision tree path empty"

    print("[TEST] 7. Testing Cavitation Fault...")
    cavit_telemetry = {
        "vibration_rms": 3.4,
        "bearing_temp": 56.0,
        "acoustic_emission": 68.0,
        "hydraulic_pressure": 105.0,  # Cavitation drop
        "spindle_rpm": 2980.0,
        "load_torque": 60.0
    }
    inf_cavit = manager.run_inference(cavit_telemetry)
    print(f"       -> Cavitation Diagnosis: {inf_cavit['consensus']['fault_name']}")
    assert inf_cavit["consensus"]["consensus_fault_id"] == 3, "Cavitation was not correctly classified"

    print("[TEST] 8. Testing k-NN Historical Failure Analogues...")
    knn_analogues = inf_bearing["models"]["knn"]["nearest_historical_analogues"]
    assert len(knn_analogues) > 0, "k-NN historical analogues empty"
    print(f"       -> Top Match: {knn_analogues[0]['record_id']} - {knn_analogues[0]['fault_name']} ({knn_analogues[0]['similarity_pct']}%)")

    print("[TEST] 9. Testing 2D PCA Space Projection...")
    assert len(manager.pca_scatter_points) > 0, "PCA scatter points empty"
    print(f"       -> PCA Current Point: {inf_bearing['pca_current_point']}")

    print("\n" + "=" * 60)
    print("  ALL 9 PREDICTX AI TEST SUITES PASSED FLAWLESSLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_test_suite()

