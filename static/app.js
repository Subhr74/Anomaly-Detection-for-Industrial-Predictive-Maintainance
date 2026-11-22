/**
 * PredictX AI - Frontend Application Orchestrator
 * Real-time WebSocket streaming, Chart.js telemetry charts, 2D PCA canvas,
 * multi-model diagnosis comparator, and stress-test sandbox.
 */

// Global State
const state = {
  isStreaming: true,
  streamIntervalMs: 500,
  activeTab: 'dashboard',
  activeModelView: 'ensemble',
  latestPayload: null,
  pcaData: null,
  benchmarks: null,
  chart: null,
  chartHistory: {
    labels: [],
    vibration: [],
    temperature: [],
    acoustic: [],
    pressure: []
  },
  maxHistoryPoints: 30,
  ws: null,
  pollTimer: null
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  initChart();
  initPCA();
  fetchBenchmarks();
  fetchPCAData();
  startStreaming();
});

// Tab Switching
function switchTab(tabId) {
  state.activeTab = tabId;
  const tabs = ['dashboard', 'diagnosis', 'sandbox', 'benchmarks', 'report'];
  
  tabs.forEach(id => {
    const el = document.getElementById(`view-${id}`);
    const btn = document.getElementById(`tab-${id}`);
    if (el && btn) {
      if (id === tabId) {
        el.classList.remove('hidden');
        btn.classList.add('border-cyan-400', 'text-cyan-400');
        btn.classList.remove('border-transparent', 'text-slate-400');
      } else {
        el.classList.add('hidden');
        btn.classList.remove('border-cyan-400', 'text-cyan-400');
        btn.classList.add('border-transparent', 'text-slate-400');
      }
    }
  });

  if (tabId === 'diagnosis' && state.pcaData) {
    drawPCACanvas();
  }
  if (tabId === 'benchmarks' && !state.benchmarks) {
    fetchBenchmarks();
  }
  if (window.lucide) {
    lucide.createIcons();
  }
}

// Chart.js Telemetry Initialization
function initChart() {
  const ctx = document.getElementById('chart-telemetry');
  if (!ctx) return;

  state.chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: state.chartHistory.labels,
      datasets: [
        {
          label: 'Vibration (mm/s)',
          borderColor: '#22d3ee',
          backgroundColor: 'rgba(34, 211, 238, 0.08)',
          borderWidth: 2,
          pointRadius: 0,
          data: state.chartHistory.vibration,
          tension: 0.35,
          yAxisID: 'y'
        },
        {
          label: 'Bearing Temp (°C)',
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251, 191, 36, 0.05)',
          borderWidth: 2,
          pointRadius: 0,
          data: state.chartHistory.temperature,
          tension: 0.35,
          yAxisID: 'y'
        },
        {
          label: 'Acoustic Emission (dB)',
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129, 140, 248, 0.05)',
          borderWidth: 2,
          pointRadius: 0,
          data: state.chartHistory.acoustic,
          tension: 0.35,
          yAxisID: 'y'
        },
        {
          label: 'Hydraulic Press. (bar/2)',
          borderColor: '#60a5fa',
          backgroundColor: 'rgba(96, 165, 250, 0.05)',
          borderWidth: 2,
          pointRadius: 0,
          data: state.chartHistory.pressure,
          tension: 0.35,
          yAxisID: 'y'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#0f172a',
          borderColor: '#334155',
          borderWidth: 1,
          titleFont: { family: 'Plus Jakarta Sans', size: 11 },
          bodyFont: { family: 'JetBrains Mono', size: 10 }
        }
      },
      scales: {
        x: {
          display: true,
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 }, maxTicksLimit: 8 }
        },
        y: {
          display: true,
          grid: { color: 'rgba(255, 255, 255, 0.06)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 9 } }
        }
      }
    }
  });
}

// Streaming Lifecycle (WebSocket with REST Polling fallback)
function startStreaming() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  try {
    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
      setConnectionStatus(true, 'Live WebSocket (2 Hz)');
    };

    state.ws.onmessage = (event) => {
      if (!state.isStreaming) return;
      try {
        const data = JSON.parse(event.data);
        handleTelemetryUpdate(data);
      } catch (err) {
        console.error('Error parsing telemetry payload', err);
      }
    };

    state.ws.onerror = () => {
      fallbackToRestPolling();
    };

    state.ws.onclose = () => {
      fallbackToRestPolling();
    };
  } catch (e) {
    fallbackToRestPolling();
  }
}

function fallbackToRestPolling() {
  if (state.pollTimer) return;
  setConnectionStatus(true, 'REST Polling');
  state.pollTimer = setInterval(async () => {
    if (!state.isStreaming) return;
    try {
      const res = await fetch('/api/telemetry/current');
      if (res.ok) {
        const data = await res.json();
        handleTelemetryUpdate(data);
      }
    } catch (e) {
      setConnectionStatus(false, 'Disconnected');
    }
  }, state.streamIntervalMs);
}

function setConnectionStatus(connected, label) {
  const dot = document.getElementById('conn-dot');
  const txt = document.getElementById('conn-text');
  if (dot && txt) {
    dot.className = connected ? 'w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse' : 'w-2.5 h-2.5 rounded-full bg-red-500';
    txt.textContent = label;
  }
}

function toggleStream() {
  state.isStreaming = !state.isStreaming;
  const btn = document.getElementById('btn-toggle-stream');
  const text = document.getElementById('text-stream');
  const icon = document.getElementById('icon-stream');

  if (state.isStreaming) {
    text.textContent = 'Pause Stream';
    icon.setAttribute('data-lucide', 'pause');
    btn.classList.remove('bg-cyan-950', 'text-cyan-400');
  } else {
    text.textContent = 'Resume Stream';
    icon.setAttribute('data-lucide', 'play');
    btn.classList.add('bg-cyan-950', 'text-cyan-400');
  }
  if (window.lucide) lucide.createIcons();
}

function changeStreamSpeed(ms) {
  state.streamIntervalMs = parseInt(ms, 10);
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    fallbackToRestPolling();
  }
}

// Telemetry Snapshot Processor
function handleTelemetryUpdate(payload) {
  state.latestPayload = payload;
  const tel = payload.telemetry;
  const inf = payload.inference;

  // 1. Update Header
  const hrsEl = document.getElementById('header-hours');
  if (hrsEl && payload.operating_hours) {
    hrsEl.textContent = `${payload.operating_hours.toFixed(1)}h`;
  }

  // Master Alarm Badge
  updateMasterAlarm(inf.overall_status, inf.consensus.fault_name);

  // 2. Update Gauges & Meters
  updateSensorCard('vibration', tel.vibration_rms, 10.0, 'mm/s');
  updateSensorCard('temp', tel.bearing_temp, 110.0, '°C');
  updateSensorCard('acoustic', tel.acoustic_emission, 95.0, 'dB');
  updateSensorCard('pressure', tel.hydraulic_pressure, 180.0, 'bar');
  updateSensorCard('rpm', tel.spindle_rpm, 4000.0, 'RPM');
  updateSensorCard('torque', tel.load_torque, 100.0, 'Nm');

  // 3. Update Chart
  if (state.chart) {
    const timeLabel = new Date().toLocaleTimeString().split(' ')[0];
    state.chartHistory.labels.push(timeLabel);
    state.chartHistory.vibration.push(tel.vibration_rms);
    state.chartHistory.temperature.push(tel.bearing_temp);
    state.chartHistory.acoustic.push(tel.acoustic_emission);
    state.chartHistory.pressure.push(tel.hydraulic_pressure / 2.0); // Scaled for multi-channel viewing

    if (state.chartHistory.labels.length > state.maxHistoryPoints) {
      state.chartHistory.labels.shift();
      state.chartHistory.vibration.shift();
      state.chartHistory.temperature.shift();
      state.chartHistory.acoustic.shift();
      state.chartHistory.pressure.shift();
    }
    state.chart.update('none');
  }

  // 4. Update Live Summary Strip
  updateDiagnosisStrip(inf);

  // 5. Update Models Comparison Grid & Feature Attribution
  updateModelsGrid(inf);

  // 6. Update 2D PCA canvas with current point
  if (state.activeTab === 'diagnosis') {
    drawPCACanvas();
  }

  // 7. Update Maintenance Report View
  updateMaintenanceReport(inf, payload);
}

function updateSensorCard(name, val, maxVal, unit) {
  const valEl = document.getElementById(`val-${name}`);
  const meterEl = document.getElementById(`meter-${name}`);
  if (valEl) {
    valEl.textContent = (name === 'rpm') ? Math.round(val) : val.toFixed(1);
  }
  if (meterEl) {
    const pct = Math.min(100, Math.max(5, (val / maxVal) * 100));
    meterEl.style.width = `${pct}%`;
    
    // Color coding based on severity
    if (name === 'vibration') {
      meterEl.className = val >= 7.1 ? 'meter-fill bg-red-500' : (val >= 4.5 ? 'meter-fill bg-amber-500' : 'meter-fill bg-emerald-500');
    } else if (name === 'temp') {
      meterEl.className = val >= 88.0 ? 'meter-fill bg-red-500' : (val >= 75.0 ? 'meter-fill bg-amber-500' : 'meter-fill bg-emerald-500');
    } else if (name === 'acoustic') {
      meterEl.className = val >= 80.0 ? 'meter-fill bg-red-500' : (val >= 68.0 ? 'meter-fill bg-amber-500' : 'meter-fill bg-emerald-500');
    } else if (name === 'pressure') {
      meterEl.className = val <= 110.0 ? 'meter-fill bg-red-500' : (val <= 125.0 ? 'meter-fill bg-amber-500' : 'meter-fill bg-emerald-500');
    }
  }
}

function updateMasterAlarm(status, faultName) {
  const badge = document.getElementById('master-status-badge');
  const text = document.getElementById('master-status-text');
  if (!badge || !text) return;

  if (status === 'CRITICAL') {
    badge.className = 'px-3.5 py-1.5 rounded-full text-xs font-bold tracking-wider status-critical flex items-center space-x-2 transition-all';
    text.textContent = `CRITICAL: ${faultName.toUpperCase()}`;
  } else if (status === 'WARNING') {
    badge.className = 'px-3.5 py-1.5 rounded-full text-xs font-bold tracking-wider status-warning flex items-center space-x-2 transition-all';
    text.textContent = `WARNING: ${faultName.toUpperCase()}`;
  } else {
    badge.className = 'px-3.5 py-1.5 rounded-full text-xs font-bold tracking-wider status-normal flex items-center space-x-2 transition-all';
    text.textContent = 'SYSTEM NOMINAL';
  }
}

function updateDiagnosisStrip(inf) {
  const diagLabel = document.getElementById('strip-diag-label');
  const diagSub = document.getElementById('strip-diag-sub');
  const rulVal = document.getElementById('strip-rul-val');
  const healthPct = document.getElementById('strip-health-pct');
  const actionWin = document.getElementById('strip-action-win');
  const mahalVal = document.getElementById('strip-mahal-val');
  const clusterRegime = document.getElementById('strip-cluster-regime');

  if (diagLabel) diagLabel.textContent = inf.consensus.fault_name;
  if (diagSub) diagSub.textContent = `Consensus Agreement: ${inf.consensus.agreement_pct}% | Severity: ${inf.consensus.severity}`;
  if (rulVal) rulVal.textContent = `${inf.rul.predicted_rul_hours.toFixed(1)} Hours`;
  if (healthPct) {
    healthPct.textContent = `(Health ${inf.rul.health_index_pct.toFixed(1)}%)`;
    healthPct.className = inf.rul.health_index_pct < 25 ? 'text-xs text-red-400 font-semibold font-mono' : (inf.rul.health_index_pct < 60 ? 'text-xs text-amber-400 font-semibold font-mono' : 'text-xs text-emerald-400 font-semibold font-mono');
  }
  if (actionWin) actionWin.textContent = `Action Window: ${inf.prescription.action_window}`;
  if (mahalVal) {
    mahalVal.textContent = inf.mahalanobis_distance.toFixed(2);
    mahalVal.className = inf.mahalanobis_distance > 3.8 ? 'text-sm font-bold text-red-400 font-mono' : 'text-sm font-bold text-white font-mono';
  }
  if (clusterRegime) {
    clusterRegime.textContent = inf.models.kmeans.cluster_regime || 'Regime: Nominal High-Speed Machining';
  }
}

function updateModelsGrid(inf) {
  const m = inf.models;

  // Decision Tree
  setCardContent('dt', m.decision_tree.fault_name, (m.decision_tree.confidence * 100).toFixed(1) + '%', m.decision_tree.latency_ms + ' ms', m.decision_tree.severity);
  // Random Forest
  setCardContent('rf', m.random_forest.fault_name, (m.random_forest.confidence * 100).toFixed(1) + '%', m.random_forest.latency_ms + ' ms', m.random_forest.severity);
  // k-NN
  setCardContent('knn', m.knn.fault_name, (m.knn.confidence * 100).toFixed(1) + '%', m.knn.latency_ms + ' ms', m.knn.severity);
  // RBFNN
  const rbfAct = document.getElementById('rbf-kernel-act');
  if (rbfAct) rbfAct.textContent = m.rbfnn.max_kernel_activation.toFixed(3);
  setCardContent('rbf', m.rbfnn.fault_name, null, m.rbfnn.latency_ms + ' ms', m.rbfnn.severity);

  // Decision Tree Path
  const treeContainer = document.getElementById('tree-path-container');
  if (treeContainer && m.decision_tree.decision_path) {
    treeContainer.innerHTML = m.decision_tree.decision_path.map(r => `<div>→ ${r}</div>`).join('') +
      `<div class="${m.decision_tree.severity === 'CRITICAL' ? 'text-red-400 font-bold' : (m.decision_tree.severity === 'WARNING' ? 'text-amber-400 font-bold' : 'text-emerald-400 font-bold')}">` +
      `✔ Classified: ${m.decision_tree.fault_name} (Confidence: ${(m.decision_tree.confidence * 100).toFixed(1)}%)</div>`;
  }

  // Feature Importances
  const fiContainer = document.getElementById('feature-importances-bars');
  if (fiContainer && m.random_forest.feature_importances) {
    const entries = Object.entries(m.random_forest.feature_importances).sort((a, b) => b[1] - a[1]);
    fiContainer.innerHTML = entries.map(([feat, imp]) => {
      const pct = (imp * 100).toFixed(1);
      return `
        <div class="space-y-1">
          <div class="flex justify-between text-[11px]">
            <span class="text-slate-300">${feat}</span>
            <span class="font-mono text-cyan-400">${pct}%</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill bg-cyan-500" style="width: ${pct}%;"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  // k-NN Historical Failure Analogues
  const knnList = document.getElementById('knn-analogues-list');
  if (knnList && m.knn.nearest_historical_analogues) {
    knnList.innerHTML = m.knn.nearest_historical_analogues.map(record => `
      <div class="p-3.5 rounded bg-slate-900/80 border border-slate-800 space-y-2">
        <div class="flex justify-between items-center text-xs">
          <span class="font-mono font-bold text-cyan-400">${record.record_id}</span>
          <span class="px-2 py-0.5 rounded text-[10px] font-bold ${record.severity === 'CRITICAL' ? 'bg-red-950 text-red-400 border border-red-800' : (record.severity === 'WARNING' ? 'bg-amber-950 text-amber-400 border border-amber-800' : 'bg-emerald-950 text-emerald-400 border border-emerald-800')}">${record.severity}</span>
        </div>
        <div class="text-xs font-semibold text-white">${record.fault_name}</div>
        <div class="text-[11px] text-slate-400">Similarity Match: <strong class="text-cyan-300 font-mono">${record.similarity_pct}%</strong> (Dist: ${record.euclidean_distance})</div>
        <p class="text-[11px] text-slate-400 border-t border-slate-800/60 pt-1.5 italic">${record.resolution}</p>
      </div>
    `).join('');
  }
}

function setCardContent(prefix, label, conf, lat, sev) {
  const lblEl = document.getElementById(`${prefix}-pred-label`);
  const confEl = document.getElementById(`${prefix}-conf`);
  const latEl = document.getElementById(`${prefix}-latency`);
  const sevEl = document.getElementById(`${prefix}-severity`);
  const meterEl = document.getElementById(`${prefix}-conf-meter`);

  if (lblEl) lblEl.textContent = label;
  if (confEl && conf) confEl.textContent = conf;
  if (latEl) latEl.textContent = lat;
  if (sevEl) {
    sevEl.textContent = sev;
    sevEl.className = sev === 'CRITICAL' ? 'font-bold text-red-400' : (sev === 'WARNING' ? 'font-bold text-amber-400' : 'font-bold text-emerald-400');
  }
  if (meterEl && conf) {
    meterEl.style.width = conf;
  }
}

// 2D PCA Canvas Rendering
function initPCA() {
  const canvas = document.getElementById('pca-canvas');
  if (!canvas) return;
  // Handle DPI scaling
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
}

async function fetchPCAData() {
  try {
    const res = await fetch('/api/models/pca-space');
    if (res.ok) {
      state.pcaData = await res.json();
      const badge = document.getElementById('pca-var-badge');
      if (badge && state.pcaData.explained_variance) {
        badge.textContent = `Explained Var: ${state.pcaData.explained_variance.total}% (PC1: ${state.pcaData.explained_variance.pc1}%, PC2: ${state.pcaData.explained_variance.pc2}%)`;
      }
      drawPCACanvas();
    }
  } catch (e) {
    console.error('Failed to load PCA data', e);
  }
}

function drawPCACanvas() {
  const canvas = document.getElementById('pca-canvas');
  if (!canvas || !state.pcaData) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;

  ctx.clearRect(0, 0, width, height);

  // Coordinate mapper: Map PCA [-4, 4] to canvas dimensions
  const scaleX = (x) => ((x + 4.0) / 8.0) * width;
  const scaleY = (y) => height - (((y + 4.0) / 8.0) * height);

  // Draw grid axes
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, scaleY(0));
  ctx.lineTo(width, scaleY(0));
  ctx.moveTo(scaleX(0), 0);
  ctx.lineTo(scaleX(0), height);
  ctx.stroke();

  // Draw baseline cluster scatter points
  const points = state.pcaData.scatter_points || [];
  points.forEach(pt => {
    const px = scaleX(pt.x);
    const py = scaleY(pt.y);

    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);

    if (pt.fault_code === 0) {
      ctx.fillStyle = 'rgba(16, 185, 129, 0.45)'; // Green (Normal)
    } else if (pt.fault_code === 1) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.6)';   // Red (Bearing)
    } else if (pt.fault_code === 2) {
      ctx.fillStyle = 'rgba(245, 158, 11, 0.6)';  // Amber (Misalign)
    } else if (pt.fault_code === 3) {
      ctx.fillStyle = 'rgba(168, 85, 247, 0.6)';  // Purple (Cavitation)
    } else {
      ctx.fillStyle = 'rgba(249, 115, 22, 0.6)';  // Orange (Overheat)
    }
    ctx.fill();
  });

  // Draw Current Operating Telemetry Point
  if (state.latestPayload && state.latestPayload.inference) {
    const curPt = state.latestPayload.inference.pca_current_point;
    if (curPt) {
      const cx = scaleX(curPt.x);
      const cy = scaleY(curPt.y);

      // Pulsing outer halo
      ctx.beginPath();
      ctx.arc(cx, cy, 14, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(6, 182, 212, 0.25)';
      ctx.fill();
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Inner solid point
      ctx.beginPath();
      ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      // Coordinates text
      ctx.font = '9px "JetBrains Mono"';
      ctx.fillStyle = '#22d3ee';
      ctx.fillText(`(${curPt.x}, ${curPt.y})`, cx + 8, cy - 8);
    }
  }
}

// Model View Switcher
function setActiveModelView(modelKey) {
  state.activeModelView = modelKey;
  document.querySelectorAll('.model-view-btn').forEach(btn => {
    btn.className = 'model-view-btn px-3 py-2 rounded text-xs font-semibold bg-slate-900 border border-slate-700 text-slate-300 hover:border-slate-500';
  });
  const activeBtn = document.getElementById(`mbtn-${modelKey}`);
  if (activeBtn) {
    activeBtn.className = 'model-view-btn px-3 py-2 rounded text-xs font-semibold bg-cyan-950 border border-cyan-500 text-cyan-300';
  }
}

// Quick Fault Injection from Toolbar
async function triggerFaultInjection(mode) {
  try {
    const res = await fetch('/api/simulator/inject-fault', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fault_mode: mode, intensity: 1.0 })
    });
    if (res.ok) {
      console.log(`Fault mode ${mode} injected successfully.`);
    }
  } catch (e) {
    console.error('Failed to inject fault', e);
  }
}

// Stress-Test Sandbox Sliders
async function onSandboxSliderChange(feature, value, labelId, unit) {
  const label = document.getElementById(labelId);
  if (label) label.textContent = `${value}${unit}`;

  try {
    // Send override to simulator
    await fetch('/api/simulator/set-slider', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ feature: feature, value: parseFloat(value) })
    });
  } catch (e) {
    console.error('Slider update failed', e);
  }
}

async function resetSandbox() {
  await fetch('/api/simulator/clear-fault', { method: 'POST' });
  // Reset slider positions
  setSliderVal('vib', 1.65, '1.65 mm/s');
  setSliderVal('temp', 51.5, '51.5 °C');
  setSliderVal('acoust', 42.0, '42.0 dB');
  setSliderVal('press', 150.0, '150.0 bar');
  setSliderVal('rpm', 3000, '3000 RPM');
  setSliderVal('torque', 48.0, '48.0 Nm');
}

function setSliderVal(idSuffix, val, text) {
  const s = document.getElementById(`sb-slider-${idSuffix}`);
  const v = document.getElementById(`sb-val-${idSuffix}`);
  if (s) s.value = val;
  if (v) v.textContent = text;
}

function injectSandboxPreset(type) {
  if (type === 'bearing') {
    setSliderVal('vib', 6.2, '6.2 mm/s');
    setSliderVal('acoust', 78.0, '78.0 dB');
    setSliderVal('temp', 64.0, '64.0 °C');
    onSandboxSliderChange('vibration_rms', 6.2, 'sb-val-vib', ' mm/s');
    onSandboxSliderChange('acoustic_emission', 78.0, 'sb-val-acoust', ' dB');
    onSandboxSliderChange('bearing_temp', 64.0, 'sb-val-temp', ' °C');
  } else if (type === 'misalign') {
    setSliderVal('vib', 4.9, '4.9 mm/s');
    setSliderVal('torque', 75.0, '75.0 Nm');
    setSliderVal('temp', 68.0, '68.0 °C');
    onSandboxSliderChange('vibration_rms', 4.9, 'sb-val-vib', ' mm/s');
    onSandboxSliderChange('load_torque', 75.0, 'sb-val-torque', ' Nm');
    onSandboxSliderChange('bearing_temp', 68.0, 'sb-val-temp', ' °C');
  } else if (type === 'cavitation') {
    setSliderVal('press', 108.0, '108.0 bar');
    setSliderVal('acoust', 69.0, '69.0 dB');
    onSandboxSliderChange('hydraulic_pressure', 108.0, 'sb-val-press', ' bar');
    onSandboxSliderChange('acoustic_emission', 69.0, 'sb-val-acoust', ' dB');
  }
}

// Model Benchmarks Matrix Fetcher
async function fetchBenchmarks() {
  try {
    const res = await fetch('/api/models/benchmarks');
    if (res.ok) {
      const data = await res.json();
      state.benchmarks = data.models;
      renderBenchmarkTable(data.models);
    }
  } catch (e) {
    console.error('Failed to fetch benchmarks', e);
  }
}

function renderBenchmarkTable(models) {
  const tbody = document.getElementById('benchmark-tbody');
  if (!tbody || !models) return;

  tbody.innerHTML = models.map(m => `
    <tr class="hover:bg-slate-800/40 transition">
      <td class="py-3 px-4 font-bold text-white font-sans">${m.model_name}</td>
      <td class="py-3 px-3 text-cyan-400">${m.accuracy}%</td>
      <td class="py-3 px-3">${m.precision}%</td>
      <td class="py-3 px-3">${m.recall}%</td>
      <td class="py-3 px-3 text-emerald-400">${m.f1_score}%</td>
      <td class="py-3 px-3 text-indigo-400">${m.roc_auc}</td>
      <td class="py-3 px-3 text-amber-400">${m.inference_latency_us} μs</td>
      <td class="py-3 px-4 text-slate-400 font-sans text-[11px]">${m.primary_rationale}</td>
    </tr>
  `).join('');
}

// Maintenance Report Generation
function updateMaintenanceReport(inf, payload) {
  const p = inf.prescription;
  const rul = inf.rul;

  const ticketEl = document.getElementById('rep-ticket-id');
  const rulEl = document.getElementById('rep-rul-hours');
  const boundsEl = document.getElementById('rep-rul-bounds');
  const priorityEl = document.getElementById('rep-priority');
  const actionWinEl = document.getElementById('rep-action-window');
  const estDowntimeEl = document.getElementById('rep-est-downtime');
  const diagEl = document.getElementById('rep-diagnosis-text');
  const actionsList = document.getElementById('rep-actions-list');
  const partsList = document.getElementById('rep-parts-list');

  if (ticketEl) ticketEl.textContent = p.ticket_id;
  if (rulEl) rulEl.textContent = `${rul.predicted_rul_hours.toFixed(1)} Hours`;
  if (boundsEl) boundsEl.textContent = `Confidence: [${rul.confidence_interval[0]}h - ${rul.confidence_interval[1]}h] (Health ${rul.health_index_pct}%)`;
  if (priorityEl) {
    priorityEl.textContent = p.priority;
    priorityEl.className = p.priority.startsWith('P1') ? 'text-xl font-bold text-red-400' : (p.priority.startsWith('P2') ? 'text-xl font-bold text-amber-400' : 'text-xl font-bold text-emerald-400');
  }
  if (actionWinEl) actionWinEl.textContent = `Action Window: ${p.action_window}`;
  if (estDowntimeEl) estDowntimeEl.textContent = `Est. Repair Downtime: ${p.estimated_downtime}`;
  if (diagEl) diagEl.textContent = p.diagnosis || `Diagnosed fault: ${p.diagnosed_fault} with ${p.confidence_level} model confidence.`;

  if (actionsList && p.recommended_actions) {
    actionsList.innerHTML = p.recommended_actions.map(act => `
      <li class="flex items-start space-x-2">
        <span class="text-cyan-400 font-bold">✔</span>
        <span>${act}</span>
      </li>
    `).join('');
  }

  if (partsList && p.required_replacement_parts) {
    partsList.innerHTML = p.required_replacement_parts.map(part => `
      <li class="flex items-center space-x-2 bg-slate-900/90 p-2 rounded border border-slate-800">
        <span class="text-yellow-400 font-bold">📦</span>
        <span>${part}</span>
      </li>
    `).join('');
  }
}

async function downloadJsonReport() {
  try {
    const res = await fetch('/api/maintenance/report');
    if (res.ok) {
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `PredictX-Diagnostic-${data.report_id}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  } catch (e) {
    alert('Failed to export diagnostic report');
  }
}

