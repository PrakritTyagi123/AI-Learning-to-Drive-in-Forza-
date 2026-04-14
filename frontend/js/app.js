/* ═══════════════════════════════════════════════
   ForzaTek AI — Dashboard JavaScript (Phase 1 + 2)
   WebSocket client + telemetry rendering + AI overlay rendering
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
let latencyMs = 0;

// ─── Phase 2: Overlay state ───
let overlayToggles = { road: true, obstacles: true, center: true, drivingLine: true, stats: true };
let lastOverlayData = null;


// ═══════════════════════════════════════════════
//  WebSocket Connection
// ═══════════════════════════════════════════════

function connectWS() {
    if (ws && ws.readyState <= 1) return;
    ws = new WebSocket(WS_URL);

    ws.onopen = () => { wsConnected = true; updateUDPStatus(true); console.log("[WS] Connected"); };

    ws.onmessage = (evt) => {
        const t0 = performance.now();
        try {
            const payload = JSON.parse(evt.data);
            if (payload.type === "telemetry") {
                lastData = payload.data;
                updateDashboard(payload.data, payload.laps || [], payload.frame, payload.captureFps);
                // Phase 2: Handle overlay data
                if (payload.overlays) {
                    lastOverlayData = payload.overlays;
                    drawOverlays(payload.overlays);
                    updateAIStats(payload.overlays, payload.aiStats);
                }
            }
        } catch (e) { console.error("[WS] Parse error:", e); }
        latencyMs = (performance.now() - t0).toFixed(1);
    };

    ws.onclose = () => { wsConnected = false; updateUDPStatus(false); setTimeout(connectWS, 2000); };
    ws.onerror = () => { ws.close(); };
}

function updateUDPStatus(connected) {
    const dot = document.getElementById("udp-dot");
    const status = document.getElementById("udp-status");
    if (connected) { dot.className = "status-dot green pulse"; status.textContent = "CONNECTED"; status.style.color = "#22c55e"; }
    else { dot.className = "status-dot"; status.textContent = "WAITING"; status.style.color = "#666"; }
}


// ═══════════════════════════════════════════════
//  Phase 2: Overlay Rendering on Viewport Canvas
// ═══════════════════════════════════════════════

function drawOverlays(ov) {
    const canvas = document.getElementById("overlay-canvas");
    const viewport = document.getElementById("viewport");
    if (!canvas || !viewport) return;

    // Match canvas pixel dimensions to the viewport element size
    const rect = viewport.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // We need to scale from frame coordinates (960x540) to canvas coordinates
    const frameW = ov.stats?.frameSize?.[0] || 960;
    const frameH = ov.stats?.frameSize?.[1] || 540;
    const scaleX = canvas.width / frameW;
    const scaleY = canvas.height / frameH;

    // ─── Road boundaries ───
    if (overlayToggles.road && ov.roadBounds) {
        drawRoadBoundary(ctx, ov.roadBounds.left, scaleX, scaleY);
        drawRoadBoundary(ctx, ov.roadBounds.right, scaleX, scaleY);
    }

    // ─── Road center line ───
    if (overlayToggles.center && ov.roadCenter) {
        const cx = ov.roadCenter.x * scaleX;
        const cy = ov.roadCenter.y * scaleY;
        ctx.save();
        ctx.strokeStyle = "#a855f7";
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.arc(cx, cy, 8 * scaleX, 0, Math.PI * 2);
        ctx.stroke();
        // Cross-hair at center
        ctx.beginPath();
        ctx.moveTo(cx - 4 * scaleX, cy); ctx.lineTo(cx + 4 * scaleX, cy);
        ctx.moveTo(cx, cy - 4 * scaleY); ctx.lineTo(cx, cy + 4 * scaleY);
        ctx.stroke();
        ctx.restore();
    }

    // ─── Obstacle bounding boxes ───
    if (overlayToggles.obstacles && ov.obstacles) {
        ov.obstacles.forEach(ob => drawObstacleBox(ctx, ob, scaleX, scaleY));
    }

    // ─── Forza driving line arrows (blue/yellow/red) ───
    if (overlayToggles.drivingLine && ov.drivingLine && ov.drivingLine.length > 0) {
        drawDrivingLine(ctx, ov.drivingLine, scaleX, scaleY);
    }

    // ─── Show/hide HUD ───
    const hud = document.getElementById("overlay-hud");
    if (hud) {
        hud.style.display = (ov.obstacles || ov.roadBounds) ? "flex" : "none";
        const tagObs = document.getElementById("tag-obstacles");
        if (tagObs) tagObs.textContent = `OBSTACLES: ${ov.obstacles?.length || 0}`;
        const statsHud = document.getElementById("overlay-stats-hud");
        if (statsHud && ov.stats) {
            statsHud.textContent = `YOLO: ${ov.stats.yoloMs}ms | SEG: ${ov.stats.segMs}ms | TOTAL: ${ov.stats.totalMs}ms`;
            statsHud.style.display = overlayToggles.stats ? "block" : "none";
        }
    }

    // Update AI status badge
    const badge = document.getElementById("ai-status-badge");
    if (badge) {
        badge.textContent = "AI_DETECTING";
        badge.style.color = "#22c55e";
        badge.style.borderColor = "#22c55e40";
    }
}

function drawRoadBoundary(ctx, points, sx, sy) {
    if (!points || points.length < 2) return;
    ctx.save();

    // Glow effect — wider transparent stroke behind
    ctx.strokeStyle = "#22c55e";
    ctx.lineWidth = 8 * sx;
    ctx.globalAlpha = 0.08;
    ctx.beginPath();
    points.forEach((p, i) => { i === 0 ? ctx.moveTo(p[0] * sx, p[1] * sy) : ctx.lineTo(p[0] * sx, p[1] * sy); });
    ctx.stroke();

    // Main dashed boundary line
    ctx.lineWidth = 2.5 * sx;
    ctx.globalAlpha = 0.75;
    ctx.setLineDash([8 * sx, 4 * sx]);
    ctx.beginPath();
    points.forEach((p, i) => { i === 0 ? ctx.moveTo(p[0] * sx, p[1] * sy) : ctx.lineTo(p[0] * sx, p[1] * sy); });
    ctx.stroke();

    ctx.restore();
}

function drawObstacleBox(ctx, ob, sx, sy) {
    const x = ob.x * sx, y = ob.y * sy, w = ob.w * sx, h = ob.h * sy;
    const color = ob.threat === "high" ? "#ff3b3b" : ob.threat === "med" ? "#f59e0b" : "#3b82f6";

    ctx.save();

    // Box outline
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5 * sx;
    ctx.globalAlpha = 0.8;
    ctx.strokeRect(x, y, w, h);

    // Corner emphasis marks (short lines at each corner)
    const cornerLen = Math.min(w, h) * 0.25;
    ctx.lineWidth = 2.5 * sx;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
    // Top-right
    ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen);
    // Bottom-left
    ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h);
    // Bottom-right
    ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen);
    ctx.stroke();

    // Label above box
    const label = `${ob.label} ${ob.dist}m`;
    const fontSize = Math.max(9, 11 * sx);
    ctx.font = `700 ${fontSize}px 'JetBrains Mono', monospace`;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.fillText(label, x, y - 4 * sy);

    // Confidence bar below label
    if (ob.conf) {
        const barW = w * ob.conf;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(x, y - 2 * sy, barW, 2 * sy);
    }

    ctx.restore();
}

function drawDrivingLine(ctx, points, sx, sy) {
    if (points.length < 2) return;

    const colorMap = {
        blue: { stroke: "#3b9eff", fill: "#3b9eff", glow: "#3b9eff" },
        yellow: { stroke: "#fbbf24", fill: "#fbbf24", glow: "#fbbf24" },
        red: { stroke: "#ff4444", fill: "#ff4444", glow: "#ff4444" },
    };

    ctx.save();

    // Draw a smooth path through all driving line points
    // Group consecutive same-color points into segments
    let segments = [];
    let current = { color: points[0].color, pts: [points[0]] };

    for (let i = 1; i < points.length; i++) {
        if (points[i].color === current.color) {
            current.pts.push(points[i]);
        } else {
            segments.push(current);
            current = { color: points[i].color, pts: [points[i]] };
        }
    }
    segments.push(current);

    // Draw each color segment
    for (const seg of segments) {
        const c = colorMap[seg.color] || colorMap.blue;

        if (seg.pts.length >= 2) {
            // Draw connected path with glow
            ctx.strokeStyle = c.glow;
            ctx.lineWidth = 10 * sx;
            ctx.globalAlpha = 0.12;
            ctx.beginPath();
            seg.pts.forEach((p, i) => {
                const px = p.x * sx, py = p.y * sy;
                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            });
            ctx.stroke();

            // Main line
            ctx.strokeStyle = c.stroke;
            ctx.lineWidth = 3 * sx;
            ctx.globalAlpha = 0.7;
            ctx.setLineDash([6 * sx, 3 * sx]);
            ctx.beginPath();
            seg.pts.forEach((p, i) => {
                const px = p.x * sx, py = p.y * sy;
                i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
            });
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Draw dots at each detected arrow position
        for (const p of seg.pts) {
            const px = p.x * sx, py = p.y * sy;
            // Filled dot
            ctx.fillStyle = c.fill;
            ctx.globalAlpha = 0.8;
            ctx.beginPath();
            ctx.arc(px, py, 3.5 * sx, 0, Math.PI * 2);
            ctx.fill();
            // Outer ring
            ctx.strokeStyle = c.stroke;
            ctx.lineWidth = 1 * sx;
            ctx.globalAlpha = 0.4;
            ctx.beginPath();
            ctx.arc(px, py, 6 * sx, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    ctx.restore();
}

function updateAIStats(ov, aiStats) {
    if (ov && ov.stats) {
        setText("ai-yolo-ms", ov.stats.yoloMs || "--");
        setText("ai-seg-ms", ov.stats.segMs || "--");
        setText("footer-infer", `${ov.stats.totalMs || "--"}ms`);
    }
    if (aiStats) {
        const status = document.getElementById("ai-status");
        if (status && aiStats.modelsLoaded) {
            status.textContent = "DETECTING";
            status.style.color = "#22c55e";
        }
        setText("footer-ai", aiStats.modelsLoaded ? "DETECTING" : "LOADING...");
        document.getElementById("footer-ai").style.color = aiStats.modelsLoaded ? "#22c55e" : "#f59e0b";
    }
}


// ═══════════════════════════════════════════════
//  Dashboard Update (Phase 1 — unchanged)
// ═══════════════════════════════════════════════

function updateDashboard(d, laps, frame, captureFps) {
    setText("speed-value", Math.round(d.speed));
    drawGearGauge(d.gear, d.rpm, d.maxRpm || 8000);
    setBar("bar-throttle", d.throttle); setBar("bar-brake", d.brake); setBar("bar-clutch", d.clutch);
    drawGForcePlot(d.gForceX || 0, d.gForceY || 0);
    setText("gforce-total", (d.gForceTotal || 0).toFixed(2));
    setText("current-lap", d.currentLap || "--:--.---");
    const delta = d.currentLapRaw && d.bestLapRaw && d.bestLapRaw > 0 ? d.currentLapRaw - d.bestLapRaw : 0;
    const deltaEl = document.getElementById("lap-delta");
    deltaEl.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(3)}`;
    deltaEl.style.color = delta < 0 ? "#22c55e" : "#ff3b3b";
    setText("best-lap", d.bestLap && d.bestLap !== "--:--.---" ? d.bestLap : "--:--.---");

    if (laps && laps.length > 0) {
        document.getElementById("lap-table-body").innerHTML = laps.map(l =>
            `<tr><td style="color:#666;font-weight:700">#${String(l.num).padStart(2,"0")}</td><td style="text-align:center;color:#aaa">${l.time}</td><td style="text-align:right;font-weight:700;color:${l.split.startsWith("-") ? "#22c55e" : "#ff3b3b"}">${l.split}</td></tr>`
        ).join("");
    }

    setSusp("susp-fl", d.suspFL); setSusp("susp-fr", d.suspFR); setSusp("susp-rl", d.suspRL); setSusp("susp-rr", d.suspRR);

    const cls = CAR_CLASSES[d.carClass] || "?";
    const clsEl = document.getElementById("car-class");
    clsEl.textContent = cls; clsEl.style.color = CLASS_COLORS[cls] || "#888";
    setText("car-pi", d.carPI || "---");
    setText("car-drivetrain", DRIVETRAIN[d.drivetrainType] || "---");
    setText("car-cylinders", d.numCylinders ? `V${d.numCylinders}` : "---");

    setText("power-val", Math.round(d.power || 0)); setText("torque-val", Math.round(d.torque || 0));
    setBar("bar-power", (d.power || 0) / 500); setBar("bar-torque", (d.torque || 0) / 600);

    renderWheelSpeeds(d);
    inputTraceData.throttle.push(d.throttle || 0); inputTraceData.brake.push(d.brake || 0);
    if (inputTraceData.throttle.length > 200) { inputTraceData.throttle.shift(); inputTraceData.brake.shift(); }

    setTireCell("tire-fl", d.tireSlipFL, d.tireAngleFL); setTireCell("tire-fr", d.tireSlipFR, d.tireAngleFR);
    setTireCell("tire-rl", d.tireSlipRL, d.tireAngleRL); setTireCell("tire-rr", d.tireSlipRR, d.tireAngleRR);
    renderTireTemps(d);

    setText("steer-val", `${(d.steeringAngle || 0).toFixed(1)}°`);
    document.getElementById("steer-thumb").style.left = `calc(${((d.steeringAngle || 0) / 45 * 50) + 50}% - 7px)`;

    renderDynamics(d);

    if (d.posX !== undefined && d.posZ !== undefined && d.isRaceOn) {
        trackPoints.push({ x: d.posX, z: d.posZ });
        if (trackPoints.length > 5000) trackPoints = trackPoints.slice(-5000);
        drawTrackMap();
    }
    setText("map-points", trackPoints.length);

    if (frame) {
        const img = document.getElementById("viewport-img");
        img.src = "data:image/jpeg;base64," + frame;
        img.style.display = "block";
        document.getElementById("viewport-placeholder").style.display = "none";
    }

    setText("footer-lap", d.lapNumber || 0);
    setText("footer-gforce", `${(d.gForceTotal || 0).toFixed(2)}G`);
    setText("footer-class", `${cls} ${d.carPI || ""}`);
    setText("footer-fps", captureFps ? Math.round(captureFps) : "--");
    setText("footer-latency", `${latencyMs}MS`);
    if (d.isRaceOn) setText("game-name", d.isHorizon ? "FH5 / FORZA HORIZON 5" : "FM / FORZA MOTORSPORT");
}


// ═══════════════════════════════════════════════
//  Canvas Renderers (Phase 1 — unchanged)
// ═══════════════════════════════════════════════

function drawGForcePlot(gx, gy) {
    const c = document.getElementById("gforce-plot"), ctx = c.getContext("2d"), w = c.width, h = c.height, cx = w/2, cy = h/2;
    ctx.clearRect(0,0,w,h); ctx.fillStyle="#080808"; ctx.fillRect(0,0,w,h);
    ctx.strokeStyle="#151515"; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(cx,8); ctx.lineTo(cx,h-8); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(8,cy); ctx.lineTo(w-8,cy); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx,cy,30,0,Math.PI*2); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx,cy,60,0,Math.PI*2); ctx.stroke();
    const dx=cx+gx*60, dy=cy-gy*60;
    ctx.fillStyle="#ff3b3b"; ctx.globalAlpha=0.9; ctx.beginPath(); ctx.arc(dx,dy,5,0,Math.PI*2); ctx.fill();
    ctx.globalAlpha=0.2; ctx.strokeStyle="#ff3b3b"; ctx.beginPath(); ctx.arc(dx,dy,14,0,Math.PI*2); ctx.stroke(); ctx.globalAlpha=1;
}

function drawGearGauge(gear, rpm, maxRpm) {
    const c = document.getElementById("gear-gauge"), ctx = c.getContext("2d"), w = c.width, h = c.height, cx = w/2, cy = h/2, r = 56;
    ctx.clearRect(0,0,w,h);
    const pct = rpm/maxRpm, sA = -135*Math.PI/180, sw = 270*Math.PI/180;
    ctx.strokeStyle="#1a1a1a"; ctx.lineWidth=5; ctx.lineCap="round"; ctx.beginPath(); ctx.arc(cx,cy,r,sA,sA+sw); ctx.stroke();
    ctx.strokeStyle = pct>0.85?"#ff3b3b":pct>0.65?"#f59e0b":"#333"; ctx.beginPath(); ctx.arc(cx,cy,r,sA,sA+pct*sw); ctx.stroke();
    const gl = gear===0?"R":gear===11?"N":gear;
    ctx.fillStyle="#e0e0e0"; ctx.font="900 34px 'Orbitron',sans-serif"; ctx.textAlign="center"; ctx.textBaseline="middle"; ctx.fillText(gl,cx,cy-6);
    ctx.fillStyle="#444"; ctx.font="700 7px 'JetBrains Mono',monospace"; ctx.fillText("G E A R",cx,cy+12);
    ctx.fillStyle="#ff3b3b"; ctx.font="700 15px 'Orbitron',sans-serif"; ctx.fillText(rpm.toLocaleString(),cx,cy+30);
    ctx.fillStyle="#ff3b3b50"; ctx.font="700 7px 'JetBrains Mono',monospace"; ctx.fillText("R P M",cx,cy+42);
}

function drawInputTrace() {
    const c = document.getElementById("input-trace"); if (!c) return;
    c.width = c.clientWidth;
    const ctx = c.getContext("2d"), w = c.width, h = c.height;
    ctx.clearRect(0,0,w,h);
    const line = (data,color) => {
        if (data.length<2) return;
        ctx.beginPath(); ctx.strokeStyle=color; ctx.lineWidth=1.5; ctx.globalAlpha=0.8;
        data.forEach((v,i) => { const x=(i/200)*w, y=h-v*h; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
        ctx.stroke(); ctx.globalAlpha=0.06; ctx.lineTo((data.length-1)/200*w,h); ctx.lineTo(0,h); ctx.fillStyle=color; ctx.fill(); ctx.globalAlpha=1;
    };
    line(inputTraceData.throttle,"#22c55e"); line(inputTraceData.brake,"#ff3b3b");
}

function drawTrackMap() {
    const c = document.getElementById("track-map"); if (!c || trackPoints.length<2) return;
    const ctx = c.getContext("2d"), w = c.width, h = c.height; ctx.clearRect(0,0,w,h);
    let mnX=Infinity,mxX=-Infinity,mnZ=Infinity,mxZ=-Infinity;
    trackPoints.forEach(p => { mnX=Math.min(mnX,p.x); mxX=Math.max(mxX,p.x); mnZ=Math.min(mnZ,p.z); mxZ=Math.max(mxZ,p.z); });
    const rX=mxX-mnX||1, rZ=mxZ-mnZ||1, sc=Math.min((w-20)/rX,(h-20)/rZ), oX=(w-rX*sc)/2, oZ=(h-rZ*sc)/2;
    ctx.strokeStyle="#ff3b3b"; ctx.lineWidth=1.5; ctx.globalAlpha=0.5; ctx.beginPath();
    trackPoints.forEach((p,i) => { const x=oX+(p.x-mnX)*sc, y=oZ+(p.z-mnZ)*sc; i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); });
    ctx.stroke(); ctx.globalAlpha=1;
    const last=trackPoints[trackPoints.length-1], lx=oX+(last.x-mnX)*sc, lz=oZ+(last.z-mnZ)*sc;
    ctx.fillStyle="#22c55e"; ctx.beginPath(); ctx.arc(lx,lz,3,0,Math.PI*2); ctx.fill();
}


// ═══════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════

function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setBar(id, pct) { const el = document.getElementById(id); if (el) el.style.width = `${Math.min(pct*100,100)}%`; }
function setSusp(id, val) { const el = document.getElementById(id); if (!el) return; const p = Math.min(Math.max(val||0,0),1)*100; el.style.height=`${p}%`; el.style.background = val>0.8?"#ff3b3b":val>0.5?"#f59e0b":"#3b82f6"; }
function slipColor(s) { return s>0.5?"#ff3b3b":s>0.2?"#f59e0b":"#22c55e"; }
function tempColor(t) { return t>100?"#ff3b3b":t>85?"#f59e0b":t>60?"#22c55e":"#3b82f6"; }

function setTireCell(id, slip, angle) {
    const el = document.getElementById(id); if (!el) return;
    const s=slip||0, a=angle||0, c=slipColor(s);
    el.style.borderColor=c+"20";
    el.querySelector(".tire-slip").textContent=s.toFixed(2); el.querySelector(".tire-slip").style.color=c; el.querySelector(".tire-slip").style.textShadow=`0 0 10px ${c}30`;
    el.querySelector(".tire-angle").textContent=`${a.toFixed(1)}°`;
}

function renderTireTemps(d) {
    const c = document.getElementById("tire-temps"); if (!c) return;
    c.innerHTML = [{l:"FL",t:d.tireTempFL},{l:"FR",t:d.tireTempFR},{l:"RL",t:d.tireTempRL},{l:"RR",t:d.tireTempRR}].map(w => {
        const t=w.t||0, co=tempColor(t), p=Math.min(t/120*100,100);
        return `<div class="temp-row"><span class="temp-label">${w.l}</span><div class="temp-bar"><div class="temp-fill" style="width:${p}%;background:${co};box-shadow:0 0 4px ${co}40"></div></div><span class="temp-val" style="color:${co}">${Math.round(t)}°</span></div>`;
    }).join("");
}

function renderWheelSpeeds(d) {
    const c = document.getElementById("wheel-speeds"); if (!c) return;
    const spd = d.speed||1, mx = Math.max(d.wheelSpeedFL||0,d.wheelSpeedFR||0,d.wheelSpeedRL||0,d.wheelSpeedRR||0,spd)*1.1;
    const wsC = (ws) => { const diff=Math.abs(ws-spd)/(spd||1); return diff>0.05?"#ff3b3b":diff>0.02?"#f59e0b":"#22c55e"; };
    c.innerHTML = [{l:"FL",s:d.wheelSpeedFL},{l:"FR",s:d.wheelSpeedFR},{l:"RL",s:d.wheelSpeedRL},{l:"RR",s:d.wheelSpeedRR}].map(w => {
        const s=w.s||0, co=wsC(s);
        return `<div class="ws-row"><span class="ws-label">${w.l}</span><div class="ws-bar"><div class="ws-fill" style="width:${s/mx*100}%;background:${co}"></div></div><span class="ws-val" style="color:${co}">${Math.round(s)}</span></div>`;
    }).join("") + `<div style="border-top:1px solid #151515;margin-top:4px;padding-top:4px;display:flex;justify-content:space-between;font-size:8px;color:#333"><span>VS CAR SPEED</span><span style="color:#666">${Math.round(spd)} MPH</span></div>`;
}

function renderDynamics(d) {
    const c = document.getElementById("dynamics-grid"); if (!c) return;
    c.innerHTML = [
        {l:"PITCH",v:`${(d.pitch||0).toFixed(1)}°`},{l:"DISTANCE",v:`${(d.distanceTraveled||0).toFixed(1)}`,u:"MI"},
        {l:"YAW",v:`${(d.yaw||0).toFixed(1)}°`},{l:"POSITION",v:d.position||0},
        {l:"ROLL",v:`${(d.roll||0).toFixed(1)}°`},{l:"DRIVING LINE",v:d.normalizedDrivingLine||0},
    ].map(i => `<div class="dyn-row"><span class="dyn-label">${i.l}</span><span class="dyn-val">${i.v}${i.u?`<span style="font-size:7px;color:#444;margin-left:2px">${i.u}</span>`:""}</span></div>`).join("");
}

function initNeuralBars() {
    const c = document.getElementById("neural-bars"); if (!c) return;
    let h = "";
    for (let i=0;i<12;i++) { const a=i<8; h+=`<div class="neural-bar" style="background:${a?"#ff3b3b":"#111"};opacity:${a?0.5+i*0.06:0.2};${a?"box-shadow:0 0 4px #ff3b3b30":""}"></div>`; }
    c.innerHTML = h;
}


// ═══════════════════════════════════════════════
//  Phase 2: Overlay Toggle Handlers
// ═══════════════════════════════════════════════

function initOverlayToggles() {
    const ids = { "toggle-road": "road", "toggle-obstacles": "obstacles", "toggle-center": "center", "toggle-driving-line": "drivingLine", "toggle-stats": "stats" };
    Object.entries(ids).forEach(([elId, key]) => {
        const el = document.getElementById(elId);
        if (el) {
            el.addEventListener("change", (e) => {
                overlayToggles[key] = e.target.checked;
                // Re-draw with last data if available
                if (lastOverlayData) drawOverlays(lastOverlayData);
            });
        }
    });
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
    initOverlayToggles();    // NEW Phase 2
    connectWS();
    animationLoop();
    drawGForcePlot(0, 0);
    drawGearGauge(0, 0, 8000);
    console.log("[ForzaTek] Dashboard v2.0 initialized — Phase 1 + Phase 2 AI Vision");
});
