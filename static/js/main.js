import * as THREE from "three";
import * as GS from "@mkkellogg/gaussian-splats-3d";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
let SCENES = [];

/* ------------------------------------------------------------------ video carousel */
function initCarousel() {
  const root = $("#video-carousel");
  const track = $(".carousel-track", root);
  const items = $$(".carousel-item", root);
  const videos = items.map((it) => $("video", it));
  let cur = 0;
  $(".carousel-dots", root).innerHTML = items.map((_, i) => `<button aria-label="video ${i + 1}"></button>`).join("");
  const dots = $$(".carousel-dots button", root);

  function go(i) {
    cur = (i + items.length) % items.length;
    track.style.transform = `translateX(${-100 * cur}%)`;
    dots.forEach((d, k) => d.classList.toggle("on", k === cur));
    videos.forEach((v, k) => {
      if (k === cur) { v.preload = "auto"; v.play().catch(() => {}); } else { v.pause(); }
    });
  }
  $(".carousel-nav.prev", root).addEventListener("click", () => go(cur - 1));
  $(".carousel-nav.next", root).addEventListener("click", () => go(cur + 1));
  dots.forEach((d, k) => d.addEventListener("click", () => go(k)));
  // the shown video plays once it is on screen, and pauses when scrolled away
  new IntersectionObserver((es) => es.forEach((e) => {
    if (e.isIntersecting) videos[cur].play().catch(() => {}); else videos[cur].pause();
  }), {threshold: 0.25}).observe(root);
  go(0);
}

/* ------------------------------------------------------------------ splat viewer */
function webgl2() {
  try { return !!document.createElement("canvas").getContext("webgl2"); } catch (e) { return false; }
}

async function initExplore() {
  const box = $("#explore .splat-box");
  const root = $(".viewer-canvas", box);
  const status = $(".viewer-status", box);
  const info = $(".viewer-info", box);
  let viewer = null, scene = null, state = "ours", token = 0;

  if (!webgl2()) {
    status.textContent = "This browser has no WebGL 2; the videos above show the same scenes.";
    return;
  }

  function applyState() {
    if (!viewer || viewer.getSceneCount() < 2) return;
    viewer.getSplatScene(0).visible = state === "pred";
    viewer.getSplatScene(1).visible = state === "ours";
    viewer.forceRenderNextFrame();
    $$(".seg", box).forEach((b) => b.classList.toggle("active", b.dataset.s === state));
  }

  function resetView() {
    if (!viewer || !scene) return;
    const c = viewer.controls;
    // lift the limits, go back to the first input camera, then keep the view within reach of the input cameras
    c.minAzimuthAngle = -Infinity; c.maxAzimuthAngle = Infinity; c.minPolarAngle = 0; c.maxPolarAngle = Math.PI;
    c.minDistance = 0; c.maxDistance = Infinity;
    viewer.camera.position.set(0, 0, 0);
    viewer.camera.up.set(0, -1, 0);
    c.target.set(0, 0, scene.look_depth);
    viewer.camera.lookAt(0, 0, scene.look_depth);
    c.update();
    // per-scene limits (degrees about the look point, from the spread of the input cameras): an orbit by azimuth a
    // and elevation e moves the controls' angles to (theta0 - a, phi0 + e)
    const o = scene.orbit, r = Math.PI / 180;
    const az = c.getAzimuthalAngle(), pol = c.getPolarAngle();
    c.minAzimuthAngle = az - o.az[1] * r; c.maxAzimuthAngle = az - o.az[0] * r;
    c.minPolarAngle = pol + o.el[0] * r; c.maxPolarAngle = pol + o.el[1] * r;
    c.minDistance = o.dist[0] * scene.look_depth; c.maxDistance = o.dist[1] * scene.look_depth;
    c.enablePan = false;
    c.update();
    viewer.forceRenderNextFrame();
  }

  async function show(i) {
    const my = ++token;
    scene = SCENES[i];
    $$(".splat-picker .scene-card").forEach((c, k) => c.classList.toggle("active", k === i));
    $("#explore .input-strip").innerHTML = Array.from({length: scene.n_inputs}, (_, k) =>
      `<figure class="thumb sq"><img src="static/thumbs/${scene.id}_${k}.jpg" alt="input view ${k + 1}"><figcaption>view ${k + 1}</figcaption></figure>`).join("");
    info.innerHTML = `<div class="vi-title">${scene.title}</div><div class="vi-meta">${scene.dataset} · ${scene.sub} · ${scene.n_inputs} views with cameras · ${scene.n_gaussians.toLocaleString()} Gaussians</div>`;
    status.hidden = false;
    status.textContent = "Loading the Gaussians…";
    if (viewer) { try { await viewer.dispose(); } catch (e) { /* already gone */ } viewer = null; }
    if (my !== token) return;
    root.innerHTML = "";
    // match the horizontal field of view of the (square) input views, so no empty margin shows at the sides
    const aspect = root.clientWidth / Math.max(1, root.clientHeight);
    const t = Math.tan(scene.fov_y * Math.PI / 360);
    const fovY = aspect > 1 ? 2 * Math.atan(t / aspect) * 180 / Math.PI : scene.fov_y;
    const camera = new THREE.PerspectiveCamera(fovY, aspect, 0.02, 500);
    camera.up.set(0, -1, 0);
    viewer = new GS.Viewer({
      rootElement: root, camera, cameraUp: [0, -1, 0],
      initialCameraPosition: [0, 0, 0], initialCameraLookAt: [0, 0, scene.look_depth],
      sharedMemoryForWorkers: false, selfDrivenMode: true, useBuiltInControls: true,
      sphericalHarmonicsDegree: 0, enableOptionalEffects: true, dynamicScene: false,
      logLevel: GS.LogLevel ? GS.LogLevel.None : 0,
    });
    // The loader sizes its buffer from Content-Length, which a server that gzips .splat files (GitHub Pages does)
    // reports for the compressed body. Download the files here, where the browser decompresses them, and hand the
    // loader blob URLs of the exact size.
    let urls = [];
    try {
      urls = await Promise.all(["pred", "ours"].map(async (s) => {
        const r = await fetch(`static/splats/${scene.id}_${s}.splat`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return URL.createObjectURL(await r.blob());
      }));
      if (my !== token) { urls.forEach((u) => URL.revokeObjectURL(u)); viewer.dispose(); return; }
      await viewer.addSplatScenes(urls.map((u) => ({path: u, format: GS.SceneFormat.Splat, splatAlphaRemovalThreshold: 1})), false);
    } catch (e) {
      status.textContent = "Could not load the Gaussians: " + e;
      return;
    } finally {
      urls.forEach((u) => URL.revokeObjectURL(u));
    }
    if (my !== token) { viewer.dispose(); return; }
    viewer.start();
    applyState();
    resetView();
    status.hidden = true;
  }

  $(".splat-picker").innerHTML = SCENES.map((s, i) => `
    <button class="scene-card" data-i="${i}">
      <div class="mosaic">${[0, 2, 5].map((k) => `<img src="static/thumbs/${s.id}_${k}.jpg" alt="">`).join("")}</div>
      <div class="sc-title">${s.title}</div><div class="sc-sub">${s.dataset} · ${s.sub}</div>
    </button>`).join("");
  $$(".splat-picker .scene-card").forEach((c) => c.addEventListener("click", () => show(+c.dataset.i)));
  $$(".seg", box).forEach((b) => b.addEventListener("click", () => { state = b.dataset.s; applyState(); }));
  $(".btn-reset", box).addEventListener("click", resetView);
  $(".btn-full", box).addEventListener("click", () => {
    if (document.fullscreenElement) document.exitFullscreen(); else box.requestFullscreen();
  });
  await show(0);
}

/* ------------------------------------------------------------------ steps */
function initSteps() {
  const names = (s) => s.steps.map((n) => n === "prediction" ? "Predicted" : n.replace("appearance ", "App. ").replace("shape ", "Shape "));
  let cur = 0;
  const range = $(".steps-range"), img = $(".steps-img"), gt = $(".steps-gt"), cap = $(".steps-cap");

  function draw() {
    const s = SCENES[cur], k = +range.value, nm = names(s);
    img.src = `static/steps/${s.id}_${k}.jpg`;
    cap.innerHTML = `${k === 0 ? "As predicted, before the steps" : (k <= 3 ? `After appearance step ${k}` : `After shape step ${k - 3}`)} · <b>${s.target_psnr[k].toFixed(2)} dB</b>`;
    $(".steps-ticks").innerHTML = nm.map((n, i) => `<span data-k="${i}" class="${i === k ? "on" : ""}">${n}<i>${s.target_psnr[i].toFixed(1)}</i></span>`).join("");
    $$(".steps-ticks span").forEach((t) => t.addEventListener("click", () => { range.value = t.dataset.k; draw(); }));
  }

  function pick(i) {
    cur = i;
    $$(".steps-chips .chip").forEach((c, k) => c.classList.toggle("active", k === i));
    const s = SCENES[i];
    gt.src = `static/steps/${s.id}_gt.jpg`;
    $(".steps-note").textContent = `${s.title} (${s.dataset}, ${s.sub}): target view ${s.target_view + 1} of ${s.n_targets}, the target whose PSNR rises the most from the prediction to the output. Pictures are rendered at 448 × 448 pixels.`;
    draw();
  }

  $(".steps-chips").innerHTML = SCENES.map((s, i) => `<button class="chip" data-i="${i}">${s.title}</button>`).join("");
  $$(".steps-chips .chip").forEach((c) => c.addEventListener("click", () => pick(+c.dataset.i)));
  range.addEventListener("input", draw);
  pick(0);
}

/* ------------------------------------------------------------------ tables and figures */
async function initTables() {
  $("#table1").innerHTML = await (await fetch("static/data/table1.html")).text();
  $("#table2").innerHTML = await (await fetch("static/data/table2.html")).text();
}

/* ------------------------------------------------------------------ method comparison */
async function initCompare() {
  const root = $("#compare");
  const DATA = await (await fetch("static/data/compare.json")).json();
  const panes = $$(".cmp-pane", root).map((el) => ({el, img: $(".cmp-img img", el), badge: $(".cmp-badge", el), method: null}));
  const view = {s: 1, x: 0, y: 0};
  let set = null, scene = null, target = 0, V = 42;

  const base = () => `static/compare/${set.id}/${scene.id}`;
  const psnrOf = (m) => (set.views ? scene.psnr[V][m] : scene.psnr[m]);
  const file = (m) => (m === "gt" ? `${base()}/gt_${target}.jpg` : `${base()}/${m}${set.views ? `_v${V}` : ""}_${target}.jpg`);

  function applyView() {
    panes.forEach((p) => { p.img.style.transform = `translate(${view.x}px, ${view.y}px) scale(${view.s})`; });
  }
  function resetView() { view.s = 1; view.x = 0; view.y = 0; applyView(); }

  function drawPane(p) {
    $$(".mbtn", p.el).forEach((b) => b.classList.toggle("active", b.dataset.m === p.method));
    p.img.src = file(p.method);
    const label = set.methods.find((m) => m.key === p.method).label;
    p.badge.innerHTML = p.method === "gt" ? "Ground truth"
      : `${label} · ${p.method === "ours" ? `<b>${psnrOf(p.method)[target].toFixed(2)} dB</b>` : `${psnrOf(p.method)[target].toFixed(2)} dB`}`;
  }
  function drawAll() { panes.forEach(drawPane); }

  function pickTarget(t) {
    target = t;
    $$(".cmp-targets .thumb-btn", root).forEach((b, k) => b.classList.toggle("active", k === t));
    drawAll();
  }

  function pickV(v) {
    V = v;
    $$(".cmp-views .chip", root).forEach((b) => b.classList.toggle("active", +b.dataset.v === v));
    $(".cmp-inputs", root).innerHTML = `<div class="cmp-row-label">${v} input views</div><img src="${base()}/inputs_v${v}.jpg" alt="input views">`;
    drawAll();
  }

  function pickScene(i) {
    scene = set.scenes[i];
    $$(".cmp-scenes .thumb-btn", root).forEach((b, k) => b.classList.toggle("active", k === i));
    panes[0].method = scene.left;
    panes[1].method = "ours";
    $(".cmp-targets", root).innerHTML = `<div class="cmp-row-label">Target views (ground truth); the model never sees them</div>` +
      `<div class="thumb-row">${Array.from({length: scene.n_targets}, (_, t) =>
        `<button class="thumb-btn small" title="target view ${t + 1}"><img src="static/compare/${set.id}/${scene.id}/gt_${t}.jpg" alt="target ${t + 1}" loading="lazy"></button>`).join("")}</div>`;
    $$(".cmp-targets .thumb-btn", root).forEach((b, k) => b.addEventListener("click", () => pickTarget(k)));
    if (set.views) {
      pickV(V);
    } else {
      $(".cmp-inputs", root).innerHTML = `<div class="cmp-row-label">${scene.n_inputs} input views</div><img src="${base()}/inputs.jpg" alt="input views">`;
    }
    resetView();
    pickTarget(scene.target);
  }

  function pickSet(id) {
    set = DATA.settings.find((s) => s.id === id);
    $$(".cmp-tabs .chip", root).forEach((b) => b.classList.toggle("active", b.dataset.s === id));
    $(".cmp-caption", root).textContent = set.caption;
    panes.forEach((p) => {
      $(".method-buttons", p.el).innerHTML = set.methods.map((m) =>
        `<button class="mbtn${m.key === "ours" || m.key === "ours_ft" ? " ours" : ""}${m.key === "gt" ? " gt" : ""}" data-m="${m.key}">${m.label}</button>`).join("");
      $$(".mbtn", p.el).forEach((b) => b.addEventListener("click", () => { p.method = b.dataset.m; drawPane(p); }));
    });
    $(".cmp-views", root).innerHTML = set.views ? set.views.map((v) =>
      `<button class="chip small" data-v="${v}">V = ${v}</button>`).join("") : "";
    $$(".cmp-views .chip", root).forEach((b) => b.addEventListener("click", () => pickV(+b.dataset.v)));
    $(".cmp-scenes", root).innerHTML = set.scenes.map((sc, i) =>
      `<button class="thumb-btn" title="scene ${i + 1}"><img src="static/compare/${set.id}/${sc.id}/gt_${sc.target}.jpg" alt="scene ${i + 1}" loading="lazy"></button>`).join("");
    $$(".cmp-scenes .thumb-btn", root).forEach((b, k) => b.addEventListener("click", () => pickScene(k)));
    pickScene(0);
  }

  // one zoom and pan for both images: scroll to zoom about the cursor, drag to pan, double-click to reset
  $$(".cmp-img", root).forEach((box) => {
    box.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = box.getBoundingClientRect();
      const cx = e.clientX - r.left, cy = e.clientY - r.top;
      const s = Math.min(8, Math.max(1, view.s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      view.x = cx - (cx - view.x) * (s / view.s);
      view.y = cy - (cy - view.y) * (s / view.s);
      view.s = s;
      clamp(r.width);
      applyView();
    }, {passive: false});
    let drag = null;
    box.addEventListener("pointerdown", (e) => { drag = {x: e.clientX - view.x, y: e.clientY - view.y}; box.setPointerCapture(e.pointerId); });
    box.addEventListener("pointermove", (e) => {
      if (!drag) return;
      view.x = e.clientX - drag.x; view.y = e.clientY - drag.y;
      clamp(box.getBoundingClientRect().width);
      applyView();
    });
    box.addEventListener("pointerup", () => { drag = null; });
    box.addEventListener("dblclick", resetView);
  });
  function clamp(w) {
    view.x = Math.min(0, Math.max(w - w * view.s, view.x));
    view.y = Math.min(0, Math.max(w - w * view.s, view.y));
  }
  $(".btn-cmp-reset", root).addEventListener("click", resetView);

  $(".cmp-tabs", root).innerHTML = DATA.settings.map((s) => `<button class="chip" data-s="${s.id}">${s.label}</button>`).join("");
  $$(".cmp-tabs .chip", root).forEach((b) => b.addEventListener("click", () => pickSet(b.dataset.s)));
  pickSet(DATA.settings[0].id);
}

function initCopy() {
  $("#copy-bibtex").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#bibtex-text").textContent); $("#copy-bibtex").textContent = "Copied"; }
    catch (e) { $("#copy-bibtex").textContent = "Select and copy"; }
    setTimeout(() => { $("#copy-bibtex").textContent = "Copy"; }, 1600);
  });
}

function lazyInit(id, fn) {
  const el = document.getElementById(id);
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, {rootMargin: "300px"});
  io.observe(el);
}

(async () => {
  SCENES = await (await fetch("static/data/scenes.json")).json();
  initCarousel();
  initSteps();
  initCopy();
  initTables();
  lazyInit("explore", initExplore);
  lazyInit("compare", initCompare);
})();
