"""
PredictX AI - Application Entry Point
Starts Uvicorn ASGI server hosting the FastAPI backend and single-page web dashboard.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import sys
import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("  PREDICTX AI - INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM")
    print("  Multi-Paradigm ML Telemetry Intelligence Engine")
    print("=" * 70)
    print("  Starting FastAPI server on http://127.0.0.1:8000")
    print("  Dashboard UI: http://127.0.0.1:8000")
    print("  API Docs:     http://127.0.0.1:8000/docs")
    print("=" * 70)

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
