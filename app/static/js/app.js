const API = "/api/v1";
const $ = (id) => document.getElementById(id);

const el = {
  status: $("status"), feed: $("feed"), placeholder: $("placeholder"), loading: $("loading"),
  screen: $("screen"), sourceLabel: $("sourceLabel"), fps: $("fps"),
  imageInput: $("imageInput"), videoInput: $("videoInput"), webcamBtn: $("webcamBtn"),
  urlForm: $("urlForm"), urlInput: $("urlInput"), stopBtn: $("stopBtn"),
  persons: $("mPersons"), objects: $("mObjects"), fire: $("mFire"), smoke: $("mSmoke"),
  density: $("mDensity"), ms: $("mMs"), counts: $("counts"), alert: $("alert"),
  people: $("people"), scene: $("scene"), sceneSetting: $("sceneSetting"), liveDot: $("liveDot"),
};

let currentStream = null;
let statsTimer = null;

// ------------------------------------------------------------------ helpers
async function api(path, options = {}) {
  const res = await fetch(API + path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : `HTTP ${res.status}`);
  }
  return res.status === 204 ? null : res.json();
}

function setBusy(busy) { el.loading.hidden = !busy; }

function showFeed(src, label) {
  el.feed.src = src;
  el.feed.hidden = false;
  el.placeholder.hidden = true;
  el.sourceLabel.textContent = label;
}

function notify(message) {
  el.alert.textContent = message;
  el.alert.hidden = !message;
}

const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function genderText(p) {
  if (p.gender === "male" || p.gender === "female") return `${p.gender} · ${Math.min(99, Math.round(p.gender_confidence * 100))}% (estimate)`;
  if (p.gender === "uncertain") return "uncertain";
  return "no clear face yet";
}

function renderPeople(people) {
  if (!people.length) { el.people.innerHTML = `<p class="muted">No people in view</p>`; return; }
  el.people.innerHTML = people.map((p, i) => {
    const doing = [...p.activities, p.posture, p.motion, ...p.gestures, p.facing].filter((x) => x && x !== "unknown");
    const rows = [
      ["Human", `${Math.round(p.confidence * 100)}% confidence`],
      ["Gender", genderText(p)],
      ["Age", p.age != null ? `about ${p.age} (estimate)` : "—"],
      ["Mood", p.expression ? `${p.expression} ${Math.round(p.expression_confidence * 100)}%` : "—"],
      ["Doing", doing.join(", ") || "—"],
      ["Wearing", [p.clothing_color && `${p.clothing_color} top`, ...p.accessories].filter(Boolean).join(" + ") || "—"],
      ["Holding", p.holding.join(", ") || "—"],
    ];
    return `<div class="person">
      <div class="head"><span>${p.track_id ? "#" + p.track_id : "PERSON " + (i + 1)}</span><span>${Math.round(p.confidence * 100)}%</span></div>
      <div class="desc">${esc(p.description)}</div>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
    </div>`;
  }).join("");
}

function renderScene(scene) {
  if (!scene) return;
  el.sceneSetting.textContent = scene.setting.toUpperCase();
  el.scene.innerHTML = scene.regions.length
    ? scene.regions.map((r) => `<li><span><i class="bar" style="width:${Math.max(4, Math.round(r.fraction * 120))}px"></i>${esc(r.label)}</span><b>${Math.round(r.fraction * 100)}%</b></li>`).join("")
    : `<li class="muted">Nothing recognised</li>`;
}

function renderSummary(s) {
  if (!s) return;
  el.persons.textContent = s.persons;
  el.objects.textContent = s.objects;
  $("mTypes").textContent = Object.keys(s.counts).length;
  $("mFaces").textContent = s.people.filter((p) => p.gender && p.gender !== "uncertain").length;
  renderPeople(s.people);
  renderScene(s.scene);
  el.fire.textContent = s.fire;
  el.smoke.textContent = s.smoke;
  el.density.textContent = s.density_level;
  el.ms.textContent = Math.round(s.total_ms);

  const entries = Object.entries(s.counts);
  el.counts.innerHTML = entries.length
    ? entries.map(([k, v]) => `<li><span>${esc(k)}</span><b>${v}</b></li>`).join("")
    : `<li class="muted">No detections</li>`;

  const hazards = [];
  if (s.fire) hazards.push(`${s.fire} fire region(s)`);
  if (s.smoke) hazards.push(`${s.smoke} smoke region(s)`);
  if (s.density_level === "HIGH") hazards.push("high crowd density");
  if (s.no_signal) {
    notify(cam.running
      ? "⚠ NO SIGNAL: the browser camera picture is blank (black, or the grey \"camera off\" icon). That comes from the camera itself, not this app. " +
        "Press the camera key (F9 / Fn+F9 on Dell) so the camera is ON, and slide open the privacy shutter above the screen."
      : "⚠ NO SIGNAL: the camera is sending black frames. Open the privacy shutter / press the camera key (F9 on many Dell laptops), and close other apps using the camera.");
    return;
  }
  notify(hazards.length ? "⚠ ALERT: " + hazards.join(" · ") : "");
}

// ------------------------------------------------------------------ streams
async function stopStream() {
  if (cam.running) await stopBrowserCamera();
  clearInterval(statsTimer);
  statsTimer = null;
  el.fps.textContent = "";
  el.stopBtn.hidden = true;
  el.liveDot.hidden = true;
  if (currentStream) {
    const id = currentStream.stream_id;
    currentStream = null;
    el.feed.src = "";
    await api(`/streams/${id}`, { method: "DELETE" }).catch(() => {});
  }
}

function startStream(created, label) {
  currentStream = created;
  showFeed(`${created.video_url}?t=${Date.now()}`, label);
  el.stopBtn.hidden = false;
  el.liveDot.hidden = false;
  statsTimer = setInterval(async () => {
    try {
      const st = await api(`/streams/${created.stream_id}/stats`);
      if (st.error) {               // e.g. camera taken by another app
        await stopStream();
        notify("⚠ " + st.error);
        return;
      }
      renderSummary(st.summary);
      el.fps.textContent = `${st.fps.toFixed(1)} FPS · ${st.frames} FRAMES`;
    } catch { /* stream gone */ }
  }, 500);
}

// ------------------------------------------------------------ browser camera
// The browser opens the webcam (works with laptop cameras OpenCV can't read) and sends
// frames over a WebSocket; one frame in flight at a time, so it never lags behind.
const cam = { stream: null, ws: null, video: null, canvas: null, tiny: null, running: false,
              darkSince: null, url: null, brightness: 255 };

function cameraErrorText(e) {
  if (e.name === "NotAllowedError") return "Camera permission was blocked. Click the camera icon in the address bar and allow it.";
  if (e.name === "NotFoundError") return "No camera found by the browser.";
  if (e.name === "NotReadableError") return "The camera is in use by another app (Teams, Zoom, Camera, another tab) or blocked by Windows.";
  return `Camera error: ${e.name || ""} ${e.message || e}`;
}

async function listCameras() {
  const sel = $("cameraSelect");
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const current = sel.value;
  sel.innerHTML = devices.map((d, i) =>
    `<option value="${esc(d.deviceId)}">${esc(d.label || "Camera " + (i + 1))}</option>`).join("");
  if (current && devices.some((d) => d.deviceId === current)) sel.value = current;
  sel.hidden = devices.length < 2;
}

function frameBrightness() {
  const ctx = cam.tiny.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(cam.video, 0, 0, 32, 18);
  const px = ctx.getImageData(0, 0, 32, 18).data;
  let sum = 0;
  for (let i = 0; i < px.length; i += 4) sum += (px[i] + px[i + 1] + px[i + 2]) / 3;
  return sum / (px.length / 4);
}

function sendFrame() {
  const v = cam.video;
  if (!cam.running) return;
  if (!v.videoWidth) { setTimeout(sendFrame, 50); return; }
  const scale = Math.min(1, 960 / v.videoWidth);
  const w = Math.round(v.videoWidth * scale), h = Math.round(v.videoHeight * scale);
  cam.canvas.width = w;
  cam.canvas.height = h;
  cam.canvas.getContext("2d").drawImage(v, 0, 0, w, h);
  cam.brightness = frameBrightness();
  cam.canvas.toBlob((blob) => {
    if (blob && cam.running && cam.ws?.readyState === WebSocket.OPEN) cam.ws.send(blob);
  }, "image/jpeg", 0.8);
}

function checkDark() {
  if (cam.brightness >= 10) { cam.darkSince = null; return; }
  cam.darkSince ??= Date.now();
  if (Date.now() - cam.darkSince > 3000) {
    notify("⚠ The BROWSER also gets a black picture from this camera, so this is not a software problem: " +
           "open the privacy shutter / press the camera key (F9 or Fn+F9 on Dell), or pick another camera in the list.");
  }
}

async function stopBrowserCamera() {
  const wasRunning = cam.running;
  cam.running = false;
  cam.ws?.close();
  cam.stream?.getTracks().forEach((t) => t.stop());
  if (cam.url) URL.revokeObjectURL(cam.url);
  Object.assign(cam, { ws: null, stream: null, url: null, darkSince: null });
  if (wasRunning) {
    el.stopBtn.hidden = true;
    el.liveDot.hidden = true;
    el.fps.textContent = "";
  }
}

async function startBrowserCamera() {
  await stopStream();
  await stopBrowserCamera();
  if (!navigator.mediaDevices?.getUserMedia) {
    notify("This browser can't open cameras here. Use http://127.0.0.1:8000 or https.");
    return;
  }
  setBusy(true);
  notify("");
  const deviceId = $("cameraSelect").value;
  try {
    cam.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { ...(deviceId ? { deviceId: { exact: deviceId } } : {}), width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    setBusy(false);
    notify("⚠ " + cameraErrorText(e));
    return;
  }
  await listCameras().catch(() => {});
  const label = cam.stream.getVideoTracks()[0]?.label || "camera";
  cam.video = Object.assign(document.createElement("video"), { muted: true, playsInline: true, srcObject: cam.stream });
  await cam.video.play();
  cam.canvas = document.createElement("canvas");
  cam.tiny = Object.assign(document.createElement("canvas"), { width: 32, height: 18 });

  const proto = location.protocol === "https:" ? "wss" : "ws";
  cam.ws = new WebSocket(`${proto}://${location.host}${API}/live`);
  cam.ws.binaryType = "blob";
  cam.ws.onopen = () => {
    setBusy(false);
    cam.running = true;
    el.stopBtn.hidden = false;
    el.liveDot.hidden = false;
    el.sourceLabel.textContent = `BROWSER CAMERA · ${label}`;
    sendFrame();
  };
  cam.ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") {            // annotated frame
      const old = cam.url;
      cam.url = URL.createObjectURL(ev.data);
      showFeed(cam.url, `BROWSER CAMERA · ${label}`);
      if (old) URL.revokeObjectURL(old);
      return;
    }
    const msg = JSON.parse(ev.data);              // stats (or an error) -> send next frame
    if (msg.error) notify("⚠ " + msg.error);
    else {
      renderSummary(msg.summary);
      el.fps.textContent = `${msg.fps.toFixed(1)} FPS · LIVE`;
      checkDark();
    }
    sendFrame();
  };
  cam.ws.onclose = (ev) => {
    if (cam.running) notify(`Live connection closed${ev.reason ? ": " + ev.reason : ""}.`);
    setBusy(false);
    stopBrowserCamera();
  };
}

// ------------------------------------------------------------------ actions
async function analyseImage(file) {
  await stopStream();
  setBusy(true);
  try {
    const form = new FormData();
    form.append("file", file);
    const mode = $("maxRecall").checked ? "max_recall" : "balanced";
    const res = await api(`/detect?include_image=true&mode=${mode}`, { method: "POST", body: form });
    showFeed(res.annotated_image, `IMAGE · ${file.name} · ${res.width}×${res.height}`);
    renderSummary(res.summary);
  } catch (e) {
    notify(e.message);
  } finally {
    setBusy(false);
  }
}

async function uploadVideo(file) {
  await stopStream();
  setBusy(true);
  try {
    const form = new FormData();
    form.append("file", file);
    const created = await api("/streams/upload", { method: "POST", body: form });
    startStream(created, `VIDEO · ${file.name}`);
  } catch (e) {
    notify(e.message);
  } finally {
    setBusy(false);
  }
}

async function openSource(source, label) {
  await stopStream();
  setBusy(true);
  try {
    const created = await api("/streams/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source }),
    });
    startStream(created, label);
  } catch (e) {
    notify(e.message);
  } finally {
    setBusy(false);
  }
}

function handleFile(file) {
  if (!file) return;
  if (file.type.startsWith("image/")) analyseImage(file);
  else if (file.type.startsWith("video/")) uploadVideo(file);
  else notify("Unsupported file type");
}

// ------------------------------------------------------------------ wiring
el.imageInput.addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });
el.videoInput.addEventListener("change", (e) => { handleFile(e.target.files[0]); e.target.value = ""; });
el.webcamBtn.addEventListener("click", startBrowserCamera);
$("cameraSelect").addEventListener("change", () => { if (cam.running) startBrowserCamera(); });
el.urlForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const v = el.urlInput.value.trim();
  if (v) openSource(v, `SOURCE · ${v.replace(/\/\/[^@]*@/, "//***@")}`);
});
el.stopBtn.addEventListener("click", () => { stopStream(); stopBrowserCamera(); });

["dragenter", "dragover"].forEach((t) => el.screen.addEventListener(t, (e) => {
  e.preventDefault(); el.screen.classList.add("drag");
}));
["dragleave", "drop"].forEach((t) => el.screen.addEventListener(t, (e) => {
  e.preventDefault(); el.screen.classList.remove("drag");
}));
el.screen.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

api("/health")
  .then((h) => {
    el.status.textContent = `${h.model} · OpenCV ${h.opencv_version}`;
    if (h.hazard_mode !== "disabled") document.querySelectorAll(".hazard").forEach((n) => { n.hidden = false; });
    el.status.className = "status ok";
  })
  .catch(() => { el.status.textContent = "API offline"; el.status.className = "status err"; });
