const POLL_STATE_MS   = 1000;
const POLL_HISTORY_MS = 3000;

// Class label order must match preprocessing_config.json / class_names.json
const ML_CLASSES = ['Normal', 'Dry Run', 'Overheating', 'Abnormal Operating Condition'];
const ML_CLASS_SHORT = ['Normal', 'Dry Run', 'Overheating', 'Abnormal'];

const elements = {
  connectionState: document.querySelector('#connection-state'),
  connectionLabel: document.querySelector('#connection-label'),
  healthPanel:     document.querySelector('#health-panel'),
  healthScore:     document.querySelector('#health-score'),
  healthLabel:     document.querySelector('#health-label'),
  healthBar:       document.querySelector('#health-bar'),
  pumpStatus:      document.querySelector('#pump-status'),
  lastUpdated:     document.querySelector('#last-updated'),
  feedAge:         document.querySelector('#feed-age'),
  temperature:     document.querySelector('#temperature'),
  current:         document.querySelector('#current'),
  vibration:       document.querySelector('#vibration'),
  faultCard:       document.querySelector('#fault-card'),
  faultLabel:      document.querySelector('#fault-label'),
  faultDetail:     document.querySelector('#fault-detail'),
  eventsList:      document.querySelector('#events-list'),
  // ML prediction panel
  mlPanel:         document.querySelector('#ml-panel'),
  mlConditionCard: document.querySelector('#ml-condition-card'),
  mlCondition:     document.querySelector('#ml-condition'),
  mlClassBadge:    document.querySelector('#ml-class-badge'),
  mlInferBadge:    document.querySelector('#ml-infer-badge'),
  mlConfidence:    document.querySelector('#ml-confidence'),
  mlConfBar:       document.querySelector('#ml-conf-bar'),
  // Sensor blackout modal elements
  blackoutOverlay: document.querySelector('#sensor-blackout-overlay'),
  disconnectedSensorName: document.querySelector('#disconnected-sensor-name'),
  btnRetry:        document.querySelector('#btn-retry-sensors'),
};

const charts = {};
let latestReading = null;

function formatTime(ts) {
  if (!ts) return 'Awaiting first message';
  return new Date(Number(ts) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function setText(el, val) { if (el) el.textContent = val; }

// ── Sensor Disconnection & Blackout Management ──────────────────────────────
function checkSensorHealth(state) {
  if (!elements.blackoutOverlay) return;
  if (!state || !state.latest || !state.online) {
    // If entire device is offline, connection status banner handles it, don't falsely blame a single sensor
    elements.blackoutOverlay.classList.add('hidden');
    return;
  }

  const r = state.latest;
  let faultSensor = null;

  // 1. Explicit sensor fault sent by ESP32 firmware
  if (r.sensor_fault && r.disconnected_sensor) {
    faultSensor = r.disconnected_sensor;
  } else if (r.disconnected_sensor && r.disconnected_sensor.trim() !== '') {
    faultSensor = r.disconnected_sensor;
  }
  // 2. Direct sensor reading checks (covers older & new firmware alike)
  else if (r.temperature <= -40.0 || r.temperature === -127.0 || r.temperature == null) {
    faultSensor = 'DS18B20 Temperature Sensor';
  } else if (r.vibration_rms == null || isNaN(r.vibration_rms)) {
    faultSensor = 'MPU-6050 Accelerometer';
  } else if (r.current == null || isNaN(r.current)) {
    faultSensor = 'ACS712 Current Sensor';
  }

  if (faultSensor) {
    // BLACKOUT: Make everything black and show modal
    setText(elements.disconnectedSensorName, faultSensor);
    elements.blackoutOverlay.classList.remove('hidden');
  } else {
    // All sensors connected and operating normally
    elements.blackoutOverlay.classList.add('hidden');
  }
}

// Global retry button handler
window.retrySensorCheck = async function() {
  if (elements.btnRetry) elements.btnRetry.classList.add('spinning');
  try {
    await fetchState();
    await fetchHistory();
  } finally {
    setTimeout(() => {
      if (elements.btnRetry) elements.btnRetry.classList.remove('spinning');
    }, 450);
  }
};

// ── Connection / health panel ───────────────────────────────────────────────
function updateConnection(state) {
  checkSensorHealth(state);

  const online = state.online && state.latest;
  elements.connectionState.classList.toggle('online',  Boolean(online));
  elements.connectionState.classList.toggle('offline', !online && Boolean(state.latest));
  setText(elements.connectionLabel, online ? 'ESP32 ONLINE' : state.latest ? 'ESP32 OFFLINE' : 'WAITING FOR DEVICE');

  if (!state.latest) {
    setText(elements.healthScore, '--');
    setText(elements.healthLabel, 'NO TELEMETRY');
    elements.healthBar.style.width = '0%';
    setText(elements.pumpStatus,  'UNKNOWN');
    setText(elements.feedAge,     '--');
    setText(elements.lastUpdated, 'Awaiting first message');
    setText(elements.faultLabel,  'NO COMMUNICATION');
    setText(elements.faultDetail, 'The twin has not received a current device state.');
    elements.faultCard.className = 'fault-card';
    return;
  }

  const r = state.latest;
  setText(elements.feedAge,     `${state.seconds_since_update}s ago`);
  setText(elements.lastUpdated, formatTime(r.timestamp));

  if (!online) {
    setText(elements.healthScore, '--');
    setText(elements.healthLabel, 'FEED OFFLINE');
    elements.healthBar.style.width = '0%';
    setText(elements.pumpStatus, 'STALE');
    setText(elements.faultLabel,  'NO COMMUNICATION');
    setText(elements.faultDetail, `Last device message was ${state.seconds_since_update}s ago.`);
    elements.faultCard.className = 'fault-card';
    return;
  }

  let condition = r.predicted_condition || 'Normal';
  const temp = Number(r.temperature);
  if ((condition === 'Overheating' || r.predicted_class === 2) && temp <= 37.0) {
    condition = 'Normal';
  }

  let fault = 'NORMAL';
  let isTrip = false;
  if (r.pump_status === 'OFF' || r.fault_label === 'OVERLOAD') {
    fault = 'OVERLOAD';
    isTrip = true;
  } else {
    fault = condition.toUpperCase();
  }

  setText(elements.healthScore, Math.round(r.health_score));
  setText(elements.healthLabel, fault);
  elements.healthBar.style.width = `${Math.max(0, Math.min(100, r.health_score))}%`;
  setText(elements.pumpStatus,  r.pump_status);
  setText(elements.faultLabel,  fault);

  if (isTrip) {
    setText(elements.faultDetail, `Hardware safety relay tripped! Current (${Number(r.current).toFixed(2)}A) exceeded safety limit.`);
    elements.faultCard.className = 'fault-card critical';
  } else {
    const isNormal = (fault === 'NORMAL');
    setText(elements.faultDetail, isNormal
      ? 'System operating within safe electrical and mechanical limits.'
      : `Operating pattern detected: ${condition}.`);
    elements.faultCard.className = `fault-card ${isNormal ? 'normal' : 'critical'}`;
  }
}

// ── Live measurements ───────────────────────────────────────────────────────
function updateMeasurements(r) {
  if (!r) return;
  setText(elements.temperature, Number(r.temperature).toFixed(1));
  setText(elements.current,     Number(r.current).toFixed(2));
  setText(elements.vibration,   Number(r.vibration_rms).toFixed(2));
}

// ── ML prediction panel ─────────────────────────────────────────────────────
const ML_CONDITION_CLASSES = {
  'Normal':                        'ml-normal',
  'Dry Run':                       'ml-dryrun',
  'Overheating':                   'ml-overheat',
  'Abnormal Operating Condition':  'ml-abnormal',
};

function updateMLPanel(r) {
  // r is state.latest
  const hasML = r && r.predicted_condition != null;

  if (!hasML) {
    // No ML prediction in this payload (legacy firmware or no reading yet)
    setText(elements.mlCondition,  'NO TELEMETRY');
    setText(elements.mlClassBadge, 'CLASS --');
    setText(elements.mlInferBadge, '--');
    setText(elements.mlConfidence, '--%');
    if (elements.mlConfBar) elements.mlConfBar.style.width = '0%';
    elements.mlConditionCard.className = 'ml-condition-card';
    // Reset class bars
    for (let i = 0; i < 4; i++) {
      const item = document.querySelector(`#cls-${i}`);
      if (!item) continue;
      item.querySelector('.ml-cls-bar').style.width = '0%';
      setText(item.querySelector('.ml-cls-pct'), '--');
      item.classList.remove('ml-cls-active');
    }
    return;
  }

  let condition = r.predicted_condition;
  let cls       = r.predicted_class != null ? r.predicted_class : 0;
  const temp    = Number(r.temperature);

  // Physical guardrail: only show Overheating if temperature > 37.0 degC, otherwise show Normal
  if ((condition === 'Overheating' || cls === 2) && temp <= 37.0) {
    condition = 'Normal';
    cls = 0;
  }

  const conf      = r.confidence != null ? r.confidence : 0;
  const confPct   = (conf * 100).toFixed(1);
  const loc       = r.inference_location || 'ESP32';

  setText(elements.mlCondition,  condition);
  setText(elements.mlClassBadge, `CLASS ${cls}`);
  setText(elements.mlInferBadge, loc);
  setText(elements.mlConfidence, `${confPct}%`);
  if (elements.mlConfBar) elements.mlConfBar.style.width = `${confPct}%`;

  // Condition card colour
  elements.mlConditionCard.className = 'ml-condition-card ' + (ML_CONDITION_CLASSES[condition] || '');

  // Class probability bars — the ESP32 only sends the winning class and confidence.
  // We show 100% on the predicted class bar and 0 on others (upgrade later if ESP32 sends all 4).
  for (let i = 0; i < 4; i++) {
    const item = document.querySelector(`#cls-${i}`);
    if (!item) continue;
    const isWinner = i === cls;
    const barPct   = isWinner ? confPct : 0;
    item.querySelector('.ml-cls-bar').style.width = `${barPct}%`;
    setText(item.querySelector('.ml-cls-pct'), isWinner ? `${confPct}%` : '0%');
    item.classList.toggle('ml-cls-active', isWinner);
  }
}

// ── Charts ──────────────────────────────────────────────────────────────────
function createChart(id, color, label, unit) {
  return new Chart(document.querySelector(`#${id}`), {
    type: 'line',
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, backgroundColor: `${color}18`, borderWidth: 2, pointRadius: 0, fill: true, tension: .35 }] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false },
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Time', color: '#536567', font: { family: 'DM Sans', size: 11, weight: '600' } },
          ticks: { color: '#829092', font: { family: 'DM Sans', size: 10 }, maxTicksLimit: 6, maxRotation: 0 },
          grid: { color: '#f0f3f1' },
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: `${label} (${unit})`, color: '#536567', font: { family: 'DM Sans', size: 11, weight: '600' } },
          grid: { color: '#e8eeea' },
          ticks: { color: '#829092', font: { family: 'DM Sans', size: 10 }, maxTicksLimit: 4 },
        },
      },
    },
  });
}

function updateCharts(history) {
  const labels = history.map((e) => formatTime(e.timestamp));
  const series = {
    temperature: history.map((e) => e.temperature),
    current:     history.map((e) => e.current),
    vibration:   history.map((e) => e.vibration_rms),
  };
  Object.entries(charts).forEach(([name, chart]) => {
    chart.data.labels = labels;
    chart.data.datasets[0].data = series[name];
    chart.update();
  });
}

// ── Events list ─────────────────────────────────────────────────────────────
function updateEvents(events) {
  if (!events.length && latestReading) {
    elements.eventsList.innerHTML = `<div class="event-row"><span class="event-time">${formatTime(latestReading.timestamp)}</span><span class="event-fault">TELEMETRY</span><span class="event-health">${Math.round(latestReading.health_score)} / 100</span><span class="event-action">DATA_RECEIVED</span></div>`;
    return;
  }
  if (!events.length) { elements.eventsList.innerHTML = '<div class="empty-state">Waiting for telemetry.</div>'; return; }
  elements.eventsList.innerHTML = events.map((e) =>
    `<div class="event-row"><span class="event-time">${formatTime(e.timestamp)}</span><span class="event-fault">${e.fault_label}</span><span class="event-health">${Math.round(e.health_score)} / 100</span><span class="event-action">${e.action}</span></div>`
  ).join('');
}

// ── API polling ─────────────────────────────────────────────────────────────
async function fetchState() {
  try {
    const res = await fetch('/api/state', { cache: 'no-store' });
    if (!res.ok) throw new Error(`State request failed: ${res.status}`);
    const state = await res.json();
    latestReading = state.latest;
    updateConnection(state);
    updateMeasurements(state.latest);
    updateMLPanel(state.latest);
  } catch (err) {
    updateConnection({ online: false, latest: null });
    setText(elements.connectionLabel, 'BACKEND UNREACHABLE');
  }
}

async function fetchHistory() {
  try {
    const [histRes, evtRes] = await Promise.all([
      fetch(`/api/history?limit=60&_=${Date.now()}`, { cache: 'no-store' }),
      fetch(`/api/alerts?limit=20&_=${Date.now()}`,  { cache: 'no-store' }),
    ]);
    if (!histRes.ok || !evtRes.ok) throw new Error('History request failed');
    updateCharts(await histRes.json());
    updateEvents(await evtRes.json());
  } catch (err) {
    console.warn('History refresh failed', err);
  }
}

// ── Init ────────────────────────────────────────────────────────────────────
charts.temperature = createChart('temperature-chart', '#e86f4f', 'Temperature', '°C');
charts.current     = createChart('current-chart',     '#087f78', 'Current draw', 'A');
charts.vibration   = createChart('vibration-chart',   '#7c9e4a', 'Vibration RMS', 'g');

fetchState();
fetchHistory();
setInterval(fetchState,   POLL_STATE_MS);
setInterval(fetchHistory, POLL_HISTORY_MS);
