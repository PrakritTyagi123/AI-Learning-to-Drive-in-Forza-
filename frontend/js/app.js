/* ═══════════════════════════════════════════════
   ForzaTek AI — Dashboard JavaScript
   WebSocket client + all rendering logic
   ═══════════════════════════════════════════════ */

const WS_URL = `ws://${location.hostname || 'localhost'}:8765`;
const CAR_CLASSES = ["D", "C", "B", "A", "S1", "S2", "X"];
const CLASS_COLORS = { D: "#3b82f6", C: "#22c55e", B: "#f59e0b", A: "#f97316", S1: "#ef4444", S2: "#a855f7", X: "#ec4899" };
const DRIVETRAIN = ["FWD", "RWD", "AWD"];

let ws = null;
let wsConnected = false;
let lastData = null;
let trackPoints = [];
let inputTraceData = { throttle: [], brake: [] };
let reconnectTimer = null;
let latencyMs = 0;

// ═══════════════════════════════════════════════
//  WebSocket Connection
// ═══════════════════════════════════════════════

function connectWS() {
    if (ws && ws.readyState <= 1) return;

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        wsConnected = true;
        updateUDPStatus(true);
        console.log("[WS] Connected to", WS_URL);
    };

    ws.onmessage = (evt) => {
        const t0 = performance.now();
        try {
            const payload = JSON.parse(evt.data);
            if (payload.type === "telemetry") {
                lastData = payload.data;
                updateDashboard(payload.data, payload.laps || [], payload.frame, payload.captureFps);
            }
        } catch (e) {
            console.error("[WS] Parse error:", e);
        }
        latencyMs = (performance.now() - t0).toFixed(1);
    };

    ws.onclose = () => {
        wsConnected = false;
        updateUDPStatus(false);
        console.log("[WS] Disconnected — reconnecting in 2s...");
        reconnectTimer = setTimeout(connectWS, 2000);
    };

    ws.onerror = () => {
        ws.close();
    };
}

function updateUDPStatus(connected) {
    const dot = document.getElementById("udp-dot");
    const status = document.getElementById("udp-status");
    if (connected) {
        dot.className = "status-dot green pulse";
        status.textContent = "CONNECTED";
        status.style.color = "#22c55e";
    } else {
        dot.className = "status-dot";
        status.textContent = "WAITING";
        status.style.color = "#666";
    }
}

// ═══════════════════════════════════════════════
//  Main Dashboard Update
// ═══════════════════════════════════════════════

function updateDashboard(d, laps, frame, captureFps) {
    // ─── Speed ───
    setText("speed-value", Math.round(d.speed));

    // ─── Gear gauge ───
    drawGearGauge(d.gear, d.rpm, d.maxRpm || 8000);

    // ─── Input bars ───
    setBar("bar-throttle", d.throttle);
    setBar("bar-brake", d.brake);
    setBar("bar-clutch", d.clutch);

    // ─── G-Force ───
    drawGForcePlot(d.gForceX || 0, d.gForceY || 0);
    setText("gforce-total", (d.gForceTotal || 0).toFixed(2));

    // ─── Lap performance ───
    setText("current-lap", d.currentLap || "--:--.---");
    const delta = d.currentLapRaw && d.bestLapRaw && d.bestLapRaw > 0 ? d.currentLapRaw - d.bestLapRaw : 0;
    const deltaEl = document.getElementById("lap-delta");
    deltaEl.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`;
    deltaEl.style.color = delta < 0 ? "#22c55e" : "#ff3b3b";

    setText("best-lap", d.bestLap && d.bestLap !== "--:--.---" ? d.bestLap : "--:--.---");

    // ─── Lap table ───
    if (laps && laps.length > 0) {
        const tbody = document.getElementById("lap-table-body");
        tbody.innerHTML = laps.map(l => `
            <tr>
                <td style="color:#666;font-weight:700">#${String(l.num).padStart(2, "0")}</td>
                <td style="text-align:center;color:#aaa">${l.time}</td>
                <td style="text-align:right;font-weight:700;color:${l.split.startsWith("-") ? "#22c55e" : "#ff3b3b"}">${l.split}</td>
            </tr>
        `).join("");
    }

    // ─── Suspension ───
    setSusp("susp-fl", d.suspFL);
    setSusp("susp-fr", d.suspFR);
    setSusp("susp-rl", d.suspRL);
    setSusp("susp-rr", d.suspRR);

    // ─── Car info ───
    const cls = CAR_CLASSES[d.carClass] || "?";
    const clsEl = document.getElementById("car-class");
    clsEl.textContent = cls;
    clsEl.style.color = CLASS_COLORS[cls] || "#888";
    clsEl.style.textShadow = `0 0 12px ${CLASS_COLORS[cls] || "#888"}40`;
    setText("car-pi", d.carPI || "---");
    setText("car-drivetrain", DRIVETRAIN[d.drivetrainType] || "---");
    setText("car-cylinders", d.numCylinders ? `V${d.numCylinders}` : "---");

    // ─── Power / Torque ───
    setText("power-val", Math.round(d.power || 0));
    setText("torque-val", Math.round(d.torque || 0));
    setBar("bar-power", (d.power || 0) / 500);
    setBar("bar-torque", (d.torque || 0) / 600);

    // ─── Wheel speeds ───
    renderWheelSpeeds(d);

    // ─── Input trace ───
    inputTraceData.throttle.push(d.throttle || 0);
    inputTraceData.brake.push(d.brake || 0);
    if (inputTraceData.throttle.length > 200) { inputTraceData.throttle.shift(); inputTraceData.brake.shift(); }

    // ─── Tire traction ───
    setTireCell("tire-fl", d.tireSlipFL, d.tireAngleFL);
    setTireCell("tire-fr", d.tireSlipFR, d.tireAngleFR);
    setTireCell("tire-rl", d.tireSlipRL, d.tireAngleRL);
    setTireCell("tire-rr", d.tireSlipRR, d.tireAngleRR);

    // ─── Tire temps ───
    renderTireTemps(d);

    // ─── Steering ───
    setText("steer-val", `${(d.steeringAngle || 0).toFixed(1)}°`);
    const steerPct = ((d.steeringAngle || 0) / 45 * 50) + 50;
    document.getElementById("steer-thumb").style.left = `calc(${steerPct}% - 7px)`;

    // ─── Vehicle dynamics ───
    renderDynamics(d);

    // ─── Track map ───
    if (d.posX !== undefined && d.posZ !== undefined && d.isRaceOn) {
        trackPoints.push({ x: d.posX, z: d.posZ });
        if (trackPoints.length > 5000) trackPoints = trackPoints.slice(-5000);
        drawTrackMap();
    }
    setText("map-points", trackPoints.length);

    // ─── Viewport / screen capture ───
    if (frame) {
        const img = document.getElementById("viewport-img");
        img.src = "data:image/jpeg;base64," + frame;
        img.style.display = "block";
        document.getElementById("viewport-placeholder").style.display = "none";
    }

    // ─── Footer ───
    setText("footer-lap", d.lapNumber || 0);
    setText("footer-gforce", `${(d.gForceTotal || 0).toFixed(2)}G`);
    setText("footer-class", `${cls} ${d.carPI || ""}`);
    setText("footer-fps", captureFps ? Math.round(captureFps) : "--");
    setText("footer-latency", `${latencyMs}MS`);

    // ─── Game detection ───
    if (d.isRaceOn) {
        setText("game-name", d.isHorizon ? "FH5 / FORZA HORIZON 5" : "FM / FORZA MOTORSPORT");
    }
}

// ═══════════════════════════════════════════════
//  Canvas: G-Force Plot
// ═══════════════════════════════════════════════

function drawGForcePlot(gx, gy) {
    const canvas = document.getElementById("gforce-plot");
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#080808";
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = "#151515";
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(cx, 8); ctx.lineTo(cx, h - 8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(8, cy); ctx.lineTo(w - 8, cy); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 30, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, 60, 0, Math.PI * 2); ctx.stroke();

    // Dot
    const dx = cx + gx * 60;
    const dy = cy - gy * 60;
    ctx.fillStyle = "#ff3b3b";
    ctx.globalAlpha = 0.9;
    ctx.beginPath(); ctx.arc(dx, dy, 5, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 0.2;
    ctx.strokeStyle = "#ff3b3b";
    ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.arc(dx, dy, 14, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════════════
//  Canvas: Gear Gauge
// ═══════════════════════════════════════════════

function drawGearGauge(gear, rpm, maxRpm) {
    const canvas = document.getElementById("gear-gauge");
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    const cx = w / 2, cy = h / 2;
    const r = 56;

    ctx.clearRect(0, 0, w, h);

    const pct = rpm / maxRpm;
    const startA = -135 * Math.PI / 180;
    const sweepA = 270 * Math.PI / 180;

    // Background arc
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.arc(cx, cy, r, startA, startA + sweepA); ctx.stroke();

    // Value arc
    ctx.strokeStyle = pct > 0.85 ? "#ff3b3b" : pct > 0.65 ? "#f59e0b" : "#333";
    ctx.beginPath(); ctx.arc(cx, cy, r, startA, startA + pct * sweepA); ctx.stroke();

    // Gear text
    const gearLabel = gear === 0 ? "R" : gear === 11 ? "N" : gear;
    ctx.fillStyle = "#e0e0e0";
    ctx.font = "900 34px 'Orbitron', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(gearLabel, cx, cy - 6);

    ctx.fillStyle = "#444";
    ctx.font = "700 7px 'JetBrains Mono', monospace";
    ctx.letterSpacing = "3px";
    ctx.fillText("G E A R", cx, cy + 12);

    ctx.fillStyle = "#ff3b3b";
    ctx.font = "700 15px 'Orbitron', sans-serif";
    ctx.fillText(rpm.toLocaleString(), cx, cy + 30);

    ctx.fillStyle = "#ff3b3b50";
    ctx.font = "700 7px 'JetBrains Mono', monospace";
    ctx.fillText("R P M", cx, cy + 42);
}

// ═══════════════════════════════════════════════
//  Canvas: Input Trace
// ═══════════════════════════════════════════════

function drawInputTrace() {
    const canvas = document.getElementById("input-trace");
    if (!canvas) return;
    canvas.width = canvas.clientWidth;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    const drawLine = (data, color) => {
        if (data.length < 2) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.8;
        data.forEach((v, i) => {
            const x = (i / 200) * w;
            const y = h - v * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.globalAlpha = 0.06;
        ctx.lineTo((data.length - 1) / 200 * w, h);
        ctx.lineTo(0, h);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1;
    };

    drawLine(inputTraceData.throttle, "#22c55e");
    drawLine(inputTraceData.brake, "#ff3b3b");
}

// ═══════════════════════════════════════════════
//  Canvas: Track Map
// ═══════════════════════════════════════════════

function drawTrackMap() {
    const canvas = document.getElementById("track-map");
    if (!canvas || trackPoints.length < 2) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Find bounds
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    trackPoints.forEach(p => {
        minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
        minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
    });

    const rangeX = maxX - minX || 1;
    const rangeZ = maxZ - minZ || 1;
    const scale = Math.min((w - 20) / rangeX, (h - 20) / rangeZ);
    const offX = (w - rangeX * scale) / 2;
    const offZ = (h - rangeZ * scale) / 2;

    ctx.strokeStyle = "#ff3b3b";
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    trackPoints.forEach((p, i) => {
        const x = offX + (p.x - minX) * scale;
        const y = offZ + (p.z - minZ) * scale;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Current position dot
    const last = trackPoints[trackPoints.length - 1];
    const lx = offX + (last.x - minX) * scale;
    const lz = offZ + (last.z - minZ) * scale;
    ctx.fillStyle = "#22c55e";
    ctx.beginPath(); ctx.arc(lx, lz, 3, 0, Math.PI * 2); ctx.fill();
}

// ═══════════════════════════════════════════════
//  Rendering Helpers
// ═══════════════════════════════════════════════

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function setBar(id, pct) {
    const el = document.getElementById(id);
    if (el) el.style.width = `${Math.min(pct * 100, 100)}%`;
}

function setSusp(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    const pct = Math.min(Math.max(val || 0, 0), 1) * 100;
    el.style.height = `${pct}%`;
    el.style.background = val > 0.8 ? "#ff3b3b" : val > 0.5 ? "#f59e0b" : "#3b82f6";
}

function slipColor(s) { return s > 0.5 ? "#ff3b3b" : s > 0.2 ? "#f59e0b" : "#22c55e"; }
function tempColor(t) { return t > 100 ? "#ff3b3b" : t > 85 ? "#f59e0b" : t > 60 ? "#22c55e" : "#3b82f6"; }
function wsColor(ws, carSpd) { const d = Math.abs(ws - carSpd) / (carSpd || 1); return d > 0.05 ? "#ff3b3b" : d > 0.02 ? "#f59e0b" : "#22c55e"; }

function setTireCell(id, slip, angle) {
    const el = document.getElementById(id);
    if (!el) return;
    const s = slip || 0, a = angle || 0;
    const c = slipColor(s);
    el.style.borderColor = c + "20";
    el.querySelector(".tire-slip").textContent = s.toFixed(2);
    el.querySelector(".tire-slip").style.color = c;
    el.querySelector(".tire-slip").style.textShadow = `0 0 10px ${c}30`;
    el.querySelector(".tire-angle").textContent = `${a.toFixed(1)}°`;
}

function renderTireTemps(d) {
    const container = document.getElementById("tire-temps");
    if (!container) return;
    const wheels = [
        { label: "FL", temp: d.tireTempFL },
        { label: "FR", temp: d.tireTempFR },
        { label: "RL", temp: d.tireTempRL },
        { label: "RR", temp: d.tireTempRR },
    ];
    container.innerHTML = wheels.map(w => {
        const t = w.temp || 0;
        const c = tempColor(t);
        const pct = Math.min(t / 120 * 100, 100);
        return `<div class="temp-row">
            <span class="temp-label">${w.label}</span>
            <div class="temp-bar"><div class="temp-fill" style="width:${pct}%;background:${c};box-shadow:0 0 4px ${c}40"></div></div>
            <span class="temp-val" style="color:${c}">${Math.round(t)}°</span>
        </div>`;
    }).join("");
}

function renderWheelSpeeds(d) {
    const container = document.getElementById("wheel-speeds");
    if (!container) return;
    const spd = d.speed || 1;
    const max = Math.max(d.wheelSpeedFL || 0, d.wheelSpeedFR || 0, d.wheelSpeedRL || 0, d.wheelSpeedRR || 0, spd) * 1.1;
    const wheels = [
        { label: "FL", speed: d.wheelSpeedFL },
        { label: "FR", speed: d.wheelSpeedFR },
        { label: "RL", speed: d.wheelSpeedRL },
        { label: "RR", speed: d.wheelSpeedRR },
    ];
    container.innerHTML = wheels.map(w => {
        const s = w.speed || 0;
        const c = wsColor(s, spd);
        return `<div class="ws-row">
            <span class="ws-label">${w.label}</span>
            <div class="ws-bar"><div class="ws-fill" style="width:${(s / max * 100)}%;background:${c}"></div></div>
            <span class="ws-val" style="color:${c}">${Math.round(s)}</span>
        </div>`;
    }).join("") + `<div style="border-top:1px solid #151515;margin-top:4px;padding-top:4px;display:flex;justify-content:space-between;font-size:8px;color:#333">
        <span>VS CAR SPEED</span><span style="color:#666">${Math.round(spd)} MPH</span>
    </div>`;
}

function renderDynamics(d) {
    const container = document.getElementById("dynamics-grid");
    if (!container) return;
    const items = [
        { l: "PITCH", v: `${(d.pitch || 0).toFixed(1)}°` },
        { l: "DISTANCE", v: `${(d.distanceTraveled || 0).toFixed(1)}`, u: "MI" },
        { l: "YAW", v: `${(d.yaw || 0).toFixed(1)}°` },
        { l: "POSITION", v: d.position || 0 },
        { l: "ROLL", v: `${(d.roll || 0).toFixed(1)}°` },
        { l: "DRIVING LINE", v: d.normalizedDrivingLine || 0 },
    ];
    container.innerHTML = items.map(i => `<div class="dyn-row">
        <span class="dyn-label">${i.l}</span>
        <span class="dyn-val">${i.v}${i.u ? `<span style="font-size:7px;color:#444;margin-left:2px">${i.u}</span>` : ""}</span>
    </div>`).join("");
}

// ═══════════════════════════════════════════════
//  Neural Link Signal Bars
// ═══════════════════════════════════════════════

function initNeuralBars() {
    const container = document.getElementById("neural-bars");
    if (!container) return;
    let html = "";
    for (let i = 0; i < 12; i++) {
        const active = i < 8;
        html += `<div class="neural-bar" style="background:${active ? "#ff3b3b" : "#111"};opacity:${active ? 0.5 + i * 0.06 : 0.2};${active ? "box-shadow:0 0 4px #ff3b3b30" : ""}"></div>`;
    }
    container.innerHTML = html;
}

// ═══════════════════════════════════════════════
//  Animation Loop
// ═══════════════════════════════════════════════

function animationLoop() {
    drawInputTrace();
    requestAnimationFrame(animationLoop);
}

// ═══════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
    initNeuralBars();
    connectWS();
    animationLoop();

    // If no data arrives, keep canvases visible with placeholder draws
    drawGForcePlot(0, 0);
    drawGearGauge(0, 0, 8000);

    console.log("[ForzaTek] Dashboard initialized — waiting for WebSocket data...");
});
