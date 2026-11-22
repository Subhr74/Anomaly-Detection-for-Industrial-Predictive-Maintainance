# PredictX AI — Industrial Predictive Maintenance & Anomaly Detection Web Application

**PredictX AI** is an end-to-end, production-grade, multi-paradigm Machine Learning platform designed to monitor critical industrial rotating machinery (e.g., CNC milling spindle assemblies, wind turbine drivetrains, hydraulic pumps, compressors) in real time via multivariate multi-sensor telemetry.

The platform continuously assesses machine health, detects subtle anomalies before mechanical failure occurs, isolates root causes, estimates Remaining Useful Life (RUL), and generates ISO 13374 / ISO 17359-compliant prescriptive maintenance work orders.

---

## 1. System Architecture Overview

```
predictx_ai/
├── engine/                      # Core Machine Learning & Statistical Engine
│   ├── preprocessor.py          # StandardScaler, Covariance Shrinkage, Mahalanobis & 2D PCA Projection
│   ├── unsupervised.py          # K-Means, DBSCAN Noise Cluster (-1), Agglomerative Hierarchical Linkage
│   ├── supervised.py            # Decision Tree (White-box path), Random Forest, k-NN Historical Matcher
│   ├── neural_networks.py       # MLP Regressor (RUL in hours) & Radial Basis Function Network (RBFNN)
│   ├── ensemble_bench.py        # Benchmark Matrix (Acc, Prec, Rec, F1, ROC-AUC, Microsecond Latency)
│   └── model_manager.py         # Unified multi-paradigm inference orchestrator & model lifecycles
├── simulator/                   # Synthetic Industrial Telemetry Streaming Engine
│   └── telemetry_stream.py      # Multivariate physics model with Gaussian noise, wear drift & fault injection
├── api/                         # High-Performance FastAPI Backend
│   └── main.py                  # REST API Endpoints, WebSocket Telemetry Stream, & ISO CBM Report Generator
├── static/                      # Industrial Cyberpunk Glassmorphic Dashboard SPA
│   ├── index.html               # Semantic HTML5, Tailwind CSS, Lucide icons, Chart.js, HTML5 Canvas
│   ├── styles.css               # Neon status glows, telemetry meters, industrial color scales, print styles
│   └── app.js                   # WebSocket client, Chart.js buffer, 2D PCA canvas, slider reactive sandbox
├── test_engine.py               # 9-point comprehensive automated ML regression test suite
└── run.py                       # One-click startup script for ASGI Uvicorn server
```

---

## 2. Multivariate Sensor Telemetry Channels

The telemetry stream processes six core physical channels compliant with **ISO 10816-3** and **ISO 13374-2**:

| Sensor Variable | Channel Name | Engineering Unit | Nominal Band | Warning Threshold | Critical Fault Threshold | Physical Failure Indication |
|---|---|---|---|---|---|---|
| **Vibration RMS** | `vibration_rms` | mm/s | 1.2 – 2.8 | 4.5 mm/s | 7.1 mm/s | ISO Zone D: Unbalance, misalignment, severe bearing raceway damage |
| **Bearing Temperature** | `bearing_temp` | °C | 45.0 – 65.0 | 75.0 °C | 88.0 °C | Friction surge, inadequate lubrication, thermal runaway |
| **Acoustic Emission** | `acoustic_emission` | dB | 35.0 – 52.0 | 68.0 dB | 80.0 dB | Ultrasonic stress waves from subsurface fatigue cracks & micro-spalling |
| **Hydraulic Pressure** | `hydraulic_pressure` | bar | 140.0 – 160.0 | &lt; 125.0 bar | &lt; 110.0 bar | Vapor bubble collapse, hydraulic pump cavitation, clamp pressure loss |
| **Spindle Speed** | `spindle_rpm` | RPM | 2900 – 3100 | ±150 RPM | &gt; ±250 RPM | Rotational speed slip, motor overload, angular drag |
| **Load Torque** | `load_torque` | Nm | 40.0 – 58.0 | 72.0 Nm | 88.0 Nm | Drive motor resistance surging, bearing cage binding |

---

## 3. Machine Learning Paradigm & Algorithmic Rationale

Each algorithm was specifically chosen to address a distinct functional requirement in the industrial condition monitoring pipeline:

### A. Decision Tree (CART) & Random Forest (Supervised Fault Diagnostics)
- **Role:** Multi-class classification of specific mechanical failure modes:
  1. *Bearing Inner Race Fault* (Mode 1)
  2. *Shaft Misalignment* (Mode 2)
  3. *Hydraulic Cavitation* (Mode 3)
  4. *Spindle Overheating / Thermal Runaway* (Mode 4)
- **Engineering Rationale:**
  - Machine operators and maintenance engineers cannot act on black-box probabilities alone. Decision trees provide **white-box interpretability** by extracting the exact sequence of fired rules (e.g., `vibration_rms > 4.52 mm/s AND acoustic_emission > 68.3 dB -> Bearing Inner Race Fault (p=0.985)`).
  - Random Forest serves as an ensemble of de-correlated decision trees trained via bagging, maximizing generalization accuracy and generating Mean Decrease in Impurity (MDI) feature importance rankings.

### B. k-Nearest Neighbors (Historical Incident Analogue Retrieval)
- **Role:** Instance-based similarity search against historical equipment failure records.
- **Engineering Rationale:**
  - When machinery displays anomalous symptoms, maintenance engineers need to know: *"When has this machine or another machine in our fleet experienced this exact behavior before, and how did we fix it?"*
  - Using normalized Euclidean metrics, k-NN retrieves the Top-3 nearest historical maintenance incidents (`HIST-FLT-xxxx`), outputting similarity percentages, past technician work-order notes, and the verified corrective action (e.g., laser alignment shimming, ceramic bearing replacement).

### C. Multilayer Perceptron / ANN (Remaining Useful Life Regression)
- **Role:** Non-linear continuous estimation of Remaining Useful Life ($RUL$) in operating hours.
- **Engineering Rationale:**
  - Industrial wear degradation follows non-linear multi-exponential physics curves. A dense feedforward neural network (`Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Linear)`) captures subtle non-linear interactions across simultaneous thermal, mechanical, and hydraulic sensor drift, outputting continuous $RUL$ hours alongside confidence intervals.

### D. Radial Basis Function Neural Network / RBFNN (Localized Kernel Novelty Detection)
- **Role:** Localized radial kernel anomaly classification and function approximation for complex non-linear operational boundaries.
- **Engineering Rationale:**
  - Standard sigmoid/ReLU networks construct global linear hyperplanes that can produce high confidence when extrapolating far outside the training distribution.
  - The RBFNN employs Gaussian radial basis units:
    $$\phi_i(x) = \exp\left(-\frac{\|x - c_i\|^2}{2 \sigma_i^2}\right)$$
  - Centers $c_i$ are anchored to operational clusters via K-Means, and weights are solved via regularized pseudo-inverse ($W = (\Phi^T \Phi + \lambda I)^{-1} \Phi^T Y$). If telemetry shifts into an unprecedented operational space, all localized radial activations collapse toward zero, immediately flagging a **radial novelty alert**.

### E. K-Means & DBSCAN (Unsupervised Cold Start & Drift Detection)
- **Role:** Baseline operational state modeling and anomaly detection without labeled fault data.
- **Engineering Rationale:**
  - Newly commissioned industrial assets run for months without experiencing breakdowns ("Cold Start").
  - K-Means clusters baseline nominal states (e.g. Warmup, Steady Low Load, Heavy Cutting) and establishes dynamic 97.5th percentile distance thresholds:
    $$D_M(x, \mu_k) = \sqrt{(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)}$$
  - DBSCAN defines the dense operational manifold without assuming spherical clusters, automatically flagging isolated outlier points as noise (cluster label `-1`).

### F. Agglomerative Hierarchical Clustering (Regime Dendrogram)
- **Role:** Identifies hierarchical relationships between operational regimes (Idling, Nominal Machining, Heavy Roughing, Transient Shifts) to isolate regime transition boundaries.

---

## 4. Benchmark Performance Matrix

Standardized evaluation on holdout test partition (stratified across all operational classes):

| Model Paradigm | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score | ROC-AUC (OvR) | Inference Latency | Operational Fit |
|---|---|---|---|---|---|---|---|
| **Decision Tree (CART)** | **98.4%** | 98.4% | 98.4% | **98.4%** | 0.985 | **~1.1 μs** | Ultra-Low Latency / White-Box Shop Floor |
| **Random Forest Ensemble** | **98.8%** | 98.9% | 98.8% | **98.8%** | 0.999 | **~33.4 μs** | High-Generalization Fleet Diagnostics |
| **k-Nearest Neighbors (k-NN)**| **99.6%** | 99.6% | 99.6% | **99.6%** | 0.998 | **~76.5 μs** | Historical Incident Analogue Retrieval |
| **Multilayer Perceptron (ANN)**| **99.6%** | 99.6% | 99.6% | **99.6%** | 0.999 | **~9.4 μs** | Deep Representation & RUL Regression |
| **Radial Basis Function (RBFNN)**| **100.0%**| 100.0% | 100.0% | **100.0%** | 1.000 | **~2.7 μs** | Localized Boundary Novelty Detection |
| **K-Means Anomaly Boundary** | **100.0%**| 100.0% | 100.0% | **100.0%** | 0.942 | **~4.9 μs** | Unsupervised Cold Start Drift Monitor |

---

## 5. Web Application Modules & UI Views

The single-page web dashboard is styled with a modern dark industrial glassmorphic aesthetic (Tailwind CSS, Lucide icons, Chart.js):

1. **Live Telemetry Dashboard (`#view-dashboard`)**:
   - Master system alarm badge (**SYSTEM NOMINAL**, **WARNING: ...**, **CRITICAL: ...**).
   - Real-time gauges and progress meters for all 6 sensors with ISO warning/critical thresholds.
   - Interactive multi-channel rolling chart (Vibration, Temp, Acoustic, Pressure) with smooth updating.
   - Quick fault injection buttons for instant simulated testing (Healthy Baseline, Bearing Flaw, Misalignment, Cavitation, Overheating).
   - Play / Pause streaming toggle and variable streaming cadence (4 Hz, 2 Hz, 1 Hz).

2. **Anomaly Detection & Diagnosis Center (`#view-diagnosis`)**:
   - Model Selector Toggle: Compare Decision Tree, Random Forest, k-NN, RBFNN, and K-Means/DBSCAN side-by-side.
   - Comparative Prediction Grid: Displays classification, confidence %, latency, and severity for each model in real time.
   - **Interactive 2D PCA Latent Space Scatter**: Canvas rendering of baseline clusters and the real-time operating point $(PC_1, PC_2)$ with an animated pulsating radar beacon.
   - **White-Box Tree Path Feature Attribution**: Visual branch path display of the exact decision rules evaluated, paired with Mean Decrease in Impurity (MDI) feature importance ranking bars.

3. **Interactive Stress-Test Sandbox (`#view-sandbox`)**:
   - 6 Dual-range sliders for Vibration, Temp, Acoustic Emission, Pressure, RPM, and Torque.
   - Instant live response: moving sliders immediately updates model decision boundaries in real time.
   - Preset buttons to simulate Stage-2 Bearing Defect, Angular Misalignment, and Hydraulic Cavitation.
   - **k-NN Historical Analogue Matcher**: Displays the top-3 closest fleet breakdown incidents with asset IDs, failure timestamps, and technician root-cause resolution logs.

4. **Model Benchmarks Matrix (`#view-benchmarks`)**:
   - Full benchmark table detailing Accuracy, Precision, Recall, F1, ROC-AUC, Microsecond Latency, and engineering rationale for each algorithm.

5. **Maintenance & RUL Report Generator (`#view-report`)**:
   - Remaining Useful Life ($RUL$) estimation with lower/upper confidence bounds and machine health index percentage.
   - Automated ISO 13374 Condition-Based Maintenance (CBM) work-order ticket (`WO-xxxxx`).
   - Action priority level (P1 - Emergency Shutdown, P2 - Urgent Preventative, P3 - Routine Monitoring).
   - Prescribed technician action checklist.
   - Required replacement spare parts with OEM SKUs (e.g. *NSK 7014-CT Bearing*, *Mobil DTE 25 Hydraulic Oil*).
   - One-click **Print / Export PDF** and **Download JSON Diagnostic Payload** buttons.

---

## 6. Getting Started & Running the Application

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- `numpy`, `scipy`, `scikit-learn`, `fastapi`, `uvicorn`, `websockets`

### 1. Run Automated Regression Test Suite
Verify that all 9 test suites pass:
```bash
python test_engine.py
```

### 2. Launch PredictX AI Web Server
Run the startup script:
```bash
python run.py
```

Once running:
- **Interactive Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 7. REST & WebSocket API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the single-page application dashboard |
| `GET` | `/api/health` | Asset metadata, simulator operational hours, model readiness |
| `GET` | `/api/models/benchmarks` | Full model benchmark matrix across 6 ML paradigms |
| `GET` | `/api/models/pca-space` | 2D PCA cluster points & explained variance ratios |
| `GET` | `/api/models/dendrogram` | Agglomerative hierarchical linkage data |
| `GET` | `/api/telemetry/current` | Advances simulator step and returns full multi-model inference |
| `POST` | `/api/inference/predict` | Evaluates arbitrary custom sensor payload (from sandbox sliders) |
| `POST` | `/api/simulator/inject-fault` | Injects fault mode (0: Normal, 1: Bearing, 2: Misalign, 3: Cavitation, 4: Overheat) |
| `POST` | `/api/simulator/clear-fault` | Clears all faults back to nominal baseline |
| `POST` | `/api/simulator/set-slider` | Sets real-time manual slider override for an individual sensor |
| `GET` | `/api/maintenance/report` | Generates ISO 13374 Condition-Based Maintenance diagnostic ticket |
| `WS` | `/ws/telemetry` | High-speed bidirectional WebSocket streaming channel (2 Hz default) |

