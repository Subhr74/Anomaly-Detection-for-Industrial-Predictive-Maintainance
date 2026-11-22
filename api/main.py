"""
PredictX AI - FastAPI Industrial Telemetry & Model Inference Server
Provides REST endpoints and WebSocket channels for real-time telemetry streaming,
fault injection, stress-test sandbox, and prescriptive maintenance reporting.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from engine.model_manager import ModelManager
from simulator.telemetry_stream import IndustrialTelemetrySimulator, FEATURE_NAMES

app = FastAPI(
    title="PredictX AI - Industrial Predictive Maintenance & Anomaly Detection API",
    version="1.0.0",
    description="Multi-paradigm ML engine for real-time industrial machinery condition monitoring."
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrators
model_manager = ModelManager()
simulator = IndustrialTelemetrySimulator(machine_id="CNC-SPINDLE-VF4-04")

# Data models
class TelemetryPayload(BaseModel):
    vibration_rms: float = Field(..., example=1.65, description="RMS vibration velocity (mm/s)")
    bearing_temp: float = Field(..., example=51.5, description="Spindle bearing race temp (°C)")
    acoustic_emission: float = Field(..., example=42.0, description="Ultrasonic stress wave emissions (dB)")
    hydraulic_pressure: float = Field(..., example=150.0, description="Hydraulic fluid pressure (bar)")
    spindle_rpm: float = Field(..., example=3000.0, description="Spindle shaft speed (RPM)")
    load_torque: float = Field(..., example=48.0, description="Drive load torque (Nm)")
    selected_model: Optional[str] = Field("ensemble", description="Model view filter")

class FaultInjectionPayload(BaseModel):
    fault_mode: int = Field(..., ge=0, le=4, description="0: Normal, 1: Bearing, 2: Misalignment, 3: Cavitation, 4: Overheating")
    intensity: float = Field(1.0, ge=0.1, le=1.0, description="Fault severity intensity multiplier")

class SliderOverridePayload(BaseModel):
    feature: str = Field(..., description="Sensor name to override")
    value: Optional[float] = Field(None, description="Value to set, or null to clear override")


@app.on_event("startup")
async def startup_event():
    """Initializes and pre-trains ML models on startup."""
    print("[API] Initializing ML Engine models...")
    model_manager.initialize_and_train(n_samples=1000)
    print("[API] System ready. Telemetry simulator running.")


@app.get("/api/health")
async def health_check():
    """Returns engine health, machine metadata, and model readiness status."""
    return {
        "status": "OPERATIONAL",
        "engine_ready": model_manager.is_ready,
        "machine_id": simulator.machine_id,
        "active_fault_mode": simulator.active_fault_mode,
        "operating_hours": round(simulator.operating_hours, 1),
        "timestamp": time.time()
    }


@app.get("/api/models/benchmarks")
async def get_benchmarks():
    """Returns benchmark comparison matrix across all models."""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still initializing")
    return {
        "models": model_manager.benchmark_results,
        "timestamp": time.time()
    }


@app.get("/api/models/dendrogram")
async def get_dendrogram():
    """Returns hierarchical agglomerative linkage and regime cluster definitions."""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still initializing")
    return model_manager.unsupervised.get_dendrogram_data()


@app.get("/api/models/pca-space")
async def get_pca_space():
    """Returns 2D PCA projected baseline scatter points for operational clusters."""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still initializing")
    
    exp_var = model_manager.preprocessor.get_pca_explained_variance()
    return {
        "scatter_points": model_manager.pca_scatter_points,
        "explained_variance": exp_var
    }


@app.get("/api/telemetry/current")
async def get_current_telemetry():
    """Advances simulator by 1 step and returns sensor values with multi-model inference."""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still initializing")

    snapshot = simulator.step()
    inference = model_manager.run_inference(snapshot["telemetry"])
    return {
        "telemetry": snapshot["telemetry"],
        "operating_hours": snapshot["operating_hours"],
        "injected_fault_mode": snapshot["injected_fault_mode"],
        "inference": inference
    }


@app.post("/api/inference/predict")
async def predict_custom_telemetry(payload: TelemetryPayload):
    """
    Evaluates arbitrary telemetry values (e.g. from the stress-test sandbox sliders)
    through the complete multi-paradigm ML engine.
    """
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still initializing")

    sensor_dict = {
        "vibration_rms": payload.vibration_rms,
        "bearing_temp": payload.bearing_temp,
        "acoustic_emission": payload.acoustic_emission,
        "hydraulic_pressure": payload.hydraulic_pressure,
        "spindle_rpm": payload.spindle_rpm,
        "load_torque": payload.load_torque
    }
    inference = model_manager.run_inference(sensor_dict, active_model=payload.selected_model or "ensemble")
    return {
        "input_telemetry": sensor_dict,
        "inference": inference
    }


@app.post("/api/simulator/inject-fault")
async def inject_fault(payload: FaultInjectionPayload):
    """Injects a specific failure mode into the simulator."""
    simulator.set_fault_mode(payload.fault_mode, payload.intensity)
    return {
        "status": "FAULT_INJECTED",
        "fault_mode": payload.fault_mode,
        "intensity": payload.intensity
    }


@app.post("/api/simulator/clear-fault")
async def clear_fault():
    """Resets the simulator to healthy baseline."""
    simulator.clear_fault()
    return {"status": "FAULT_CLEARED", "message": "Telemetry restored to nominal baseline."}


@app.post("/api/simulator/set-slider")
async def set_slider_override(payload: SliderOverridePayload):
    """Overrides a single sensor parameter from the sandbox slider."""
    if payload.feature not in FEATURE_NAMES:
        raise HTTPException(status_code=400, detail=f"Invalid feature name: {payload.feature}")
    simulator.set_manual_override(payload.feature, payload.value)
    return {"status": "OVERRIDE_UPDATED", "feature": payload.feature, "value": payload.value}


@app.get("/api/maintenance/report")
async def generate_maintenance_report():
    """Generates an ISO 13374 / ISO 17359 condition monitoring diagnostic report."""
    if not model_manager.is_ready:
        raise HTTPException(status_code=503, detail="Models still initializing")

    snapshot = simulator.step()
    inference = model_manager.run_inference(snapshot["telemetry"])
    
    report = {
        "report_id": f"ISO-CBM-REP-{int(time.time())}",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "machine_asset": {
            "asset_id": simulator.machine_id,
            "type": "CNC High-Speed Spindle (Cartridge Bearing Assembly)",
            "operating_hours": round(simulator.operating_hours, 1),
            "criticality_index": "Class A (Production Critical)"
        },
        "condition_summary": {
            "overall_status": inference["overall_status"],
            "diagnosed_fault": inference["consensus"]["fault_name"],
            "consensus_agreement": f"{inference['consensus']['agreement_pct']}%",
            "estimated_rul_hours": inference["rul"]["predicted_rul_hours"],
            "rul_confidence_bounds": inference["rul"]["confidence_interval"],
            "machine_health_score": f"{inference['rul']['health_index_pct']}%"
        },
        "sensor_telemetry_snapshot": snapshot["telemetry"],
        "anomaly_indicators": {
            "mahalanobis_distance": inference["mahalanobis_distance"],
            "mahalanobis_threshold": inference["mahalanobis_threshold"],
            "kmeans_anomaly": inference["models"]["kmeans"]["is_anomaly"],
            "dbscan_noise": inference["models"]["dbscan"]["is_anomaly"],
            "threshold_violations": inference["threshold_alerts"]
        },
        "white_box_attribution": {
            "decision_path": inference["models"]["decision_tree"]["decision_path"],
            "feature_importances": inference["models"]["random_forest"]["feature_importances"]
        },
        "prescriptive_work_order": inference["prescription"]
    }
    return report


@app.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    High-speed WebSocket streaming channel emitting real-time telemetry
    and inference payloads directly to connected frontend clients.
    """
    await websocket.accept()
    try:
        while True:
            # Advance simulator step
            snapshot = simulator.step()
            inference = model_manager.run_inference(snapshot["telemetry"])
            
            payload = {
                "timestamp": snapshot["timestamp"],
                "step_count": snapshot["step_count"],
                "operating_hours": snapshot["operating_hours"],
                "injected_fault_mode": snapshot["injected_fault_mode"],
                "telemetry": snapshot["telemetry"],
                "inference": inference
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.5)  # 500ms streaming cadence (2 Hz)
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as e:
        print(f"[WebSocket Error] {e}")


# Serve static frontend files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_index():
    """Serves the main single page web application."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "PredictX AI API is active. Static frontend not found."}
