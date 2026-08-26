/* M3 实验台 app —— 真实神经元连接 + 游戏式实验（数据全部来自 Brian2 确定性仿真）
 * 依赖：runs.js（window.M3_DATA，由 tools/gen_m3_lab.py 生成）
 */
"use strict";

const DATA = window.M3_DATA;
if (!DATA) { throw new Error("runs.js 未加载：请先在同一目录放置 runs.js"); }

const D = DATA, M = D.morph;
const ROLES = M.roles;                                  // ["PLM","AVM","DA","VB"]
const DT = D.meta.dtMs;                                 // 0.1 ms/采样
const T_TOTAL = D.meta.tTotalMs;                        // 150 ms
const TOUCH = M.touch;                                  // {role, site, i0, start, dur}
const THRESH = D.meta.dThreshold;                       // 0.3

const ROLE_INFO = {
  PLM: "后部触觉感觉神经元",
  AVM: "中间神经元（传导 + 分流）",
  DA: "后退运动神经元",
  VB: "前进运动神经元（张力维持基线）",
};
const ROLE_SITE = {
  PLM: "触刺激注入 dend2#1（树突端）",
  AVM: "突触后：PLM node3 → soma",
  DA: "突触后：AVM node3 → soma",
  VB: "突触后：AVM node3 → soma · 张力 14µA/cm²",
};

/* ---------------- SVG 辅助 ---------------- */
const SVGNS = "http://www.w3.org/2000/svg";
const W = 1160, H = 680;
const svgRoot = document.getElementById("circuit");
function svgEl(name, attrs, parent) {
  const el = document.createElementNS(SVGNS, name);
  if (attrs) for (const k in attrs) el.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(el);
  return el;
}
function textEl(x, y, str, attrs, parent) {
  const t = svgEl("text", Object.assign({ x, y, "text-anchor": "middle" }, attrs || {}), parent);
  t.appendChild(document.createTextNode(str));
  return t;
}

/* 画布布局（µm → px）：树突在左（负 x），轴突在右（正 x），node3 在最右 */
const X0 = 70, X_UM_MIN = -125, X_UM_MAX = 640, SCALE = 0.60;
const ROW_Y = { PLM: 115, AVM: 250, DA: 390, VB: 530 };
const MUSCLE_X = 692, MUSCLE_W = 26;
function umX(um) { return X0 + (um - X_UM_MIN) * SCALE; }
function compR(c) {
  if (c.t === "soma") return 11;
  if (c.t === "node") return 5.2;
  if (c.t === "ais") return 4.4;
  if (c.t === "dend") return 3.6;
  return 2.8; // myelin
}
function anchor(role, site) {
  const c = M.comps[role].find(x => x.s === site);
  return { x: umX(c.x), y: ROW_Y[role] };
}

/* 膜电位 → 颜色（-80 深蓝 → -40 青 → -10 琥珀 → +40 红） */
const V_STOPS = [
  [-80, [38, 40, 96]], [-62, [58, 92, 168]], [-40, [46, 158, 158]],
  [-10, [240, 190, 70]], [40, [222, 54, 54]],
];
function vColor(v) {
  let a = V_STOPS[0], b = V_STOPS[V_STOPS.length - 1];
  for (let i = 0; i < V_STOPS.length - 1; i++) {
    if (v >= V_STOPS[i][0] && v <= V_STOPS[i + 1][0]) { a = V_STOPS[i]; b = V_STOPS[i + 1]; break; }
  }
  const p = (b[0] - a[0]) === 0 ? 0 : (v - a[0]) / (b[0] - a[0]);
  const mix = (ca, cb) => Math.round(ca + (cb - ca) * p);
  return `rgb(${mix(a[1][0], b[1][0])},${mix(a[1][1], b[1][1])},${mix(a[1][2], b[1][2])})`;
}

/* ---------------- 状态与播放 ---------------- */
const state = {
  runId: "i1", t: 45.0, playing: false, speedMs: 1.0,
  intensity: 1.0, durMs: 5.0, ablation: false,
  raf: null, lastTs: 0,
  game: { active: false, touchAt: 0 },
};
const compEls = {};      // role -> { "seg#i": circle }
const flashEls = {};     // role -> node3 发光环
const linkEls = [];      // {path, dot, pre, delay, len, cut}
let muscleBack, muscleFwd;
let wormG, dirText, dMeterText;
let touchFlash;

function currentRun() { return D.runs[state.runId]; }
function runIdFor(intensity, durMs, ablation) {
  if (ablation) return "abl";
  if (durMs >= 20) return "i8_long";
  if (intensity === 0) return "ctrl";
  if (intensity === 0.5) return "i05";
  if (intensity === 1) return "i1";
  if (intensity === 2) return "i2";
  if (intensity === 4) return "i4";
  if (intensity === 8) return "i8";
  return "i1";
}

/* ---------------- 构建静态图 ---------------- */
function buildCircuit() {
  svgRoot.innerHTML = "";
  svgRoot.setAttribute("viewBox", `0 0 ${W} ${H}`);

  // 箭头标记
  const defs = svgEl("defs", {}, svgRoot);
  [["arrowO", "#ff7f0e"], ["arrowR", "#d62728"], ["arrowG", "#2e7d32"], ["arrowK", "#78909c"]]
    .forEach(([id, c]) => {
      const m = svgEl("marker", { id, viewBox: "0 0 10 10", refX: 8, refY: 5,
        markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
      svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: c }, m);
    });

  textEl(W / 2, 26, "M3 触觉反射弧 · 真实神经元连接回放（Brian2 确定性仿真数据）",
    { "font-size": 17, "font-weight": "bold", fill: "#1c2733" }, svgRoot);

  // 神经元：按真实 18 隔室形态绘制，颜色 = 该隔室实时 V(t)
  ROLES.forEach(role => {
    const g = svgEl("g", { class: "neuron" }, svgRoot);
    compEls[role] = {};
    M.comps[role].forEach(c => {
      const circle = svgEl("circle", {
        cx: umX(c.x), cy: ROW_Y[role], r: compR(c), class: "comp " + c.t,
        stroke: c.t === "node" ? "#7b1fa2" : "none", "stroke-width": 1,
      }, g);
      compEls[role][c.s + "#" + c.i] = circle;
    });
    const soma = M.comps[role].find(c => c.s === "soma");
    textEl(umX(soma.x), ROW_Y[role] - 28, role,
      { "font-size": 15, "font-weight": "bold", fill: "#263238" }, g);
    textEl(umX(soma.x), ROW_Y[role] + 27, ROLE_INFO[role],
      { "font-size": 9.5, fill: "#607d8b" }, g);
    textEl(umX(soma.x), ROW_Y[role] + 40, ROLE_SITE[role],
      { "font-size": 8.5, fill: "#90a4ae" }, g);
    const n3 = M.comps[role].find(c => c.s === "node3");
    flashEls[role] = svgEl("circle", {
      cx: umX(n3.x), cy: ROW_Y[role], r: compR(n3) + 5, fill: "none",
      stroke: "#ffd700", "stroke-width": 3, opacity: 0,
    }, g);
    // 轴突方向小箭头（soma → node3）
    svgEl("path", { d: `M ${umX(soma.x) + 8},${ROW_Y[role] - 14} L ${umX(soma.x) + 26},${ROW_Y[role] - 14}`,
      stroke: "#b0bec5", "stroke-width": 1.2, "marker-end": "url(#arrowK)" }, g);
  });

  // 触刺激指示（PLM 树突端 dend2#1）
  const tip = M.comps.PLM.find(c => c.s === "dend2" && c.i === 1);
  touchFlash = svgEl("circle", { cx: umX(tip.x), cy: ROW_Y.PLM, r: 14, fill: "#ff7f0e", opacity: 0 }, svgRoot);
  textEl(umX(tip.x), ROW_Y.PLM - 26, "触刺激 I0", { "font-size": 9.5, fill: "#e65100" }, svgRoot);
  svgEl("line", { x1: umX(tip.x) - 22, y1: ROW_Y.PLM, x2: umX(tip.x) - 3, y2: ROW_Y.PLM,
    stroke: "#ff7f0e", "stroke-width": 2, "marker-end": "url(#arrowO)" }, svgRoot);

  // 张力注记（VB soma）
  const vbs = M.comps.VB.find(c => c.s === "soma");
  svgEl("circle", { cx: umX(vbs.x) - 28, cy: ROW_Y.VB, r: 7, fill: "#90a4ae", opacity: 0.45 }, svgRoot);
  textEl(umX(vbs.x) - 28, ROW_Y.VB - 17, "张力", { "font-size": 8, fill: "#546e7a" }, svgRoot);

  // 突触连接（node3 → soma，与 CSV 一致；消融时 AVM→VB 虚线断开）
  M.links.forEach(lk => {
    const pre = lk.from, to = lk.to;
    let a, b, color;
    if (to === "MUSCLE_B") { a = anchor(pre, "node3"); b = { x: MUSCLE_X - MUSCLE_W / 2, y: ROW_Y.DA }; }
    else if (to === "MUSCLE_F") { a = anchor(pre, "node3"); b = { x: MUSCLE_X - MUSCLE_W / 2, y: ROW_Y.VB }; }
    else { a = anchor(pre, "node3"); b = anchor(to, "soma"); }
    const mx = (a.x + b.x) / 2 + 52, my = (a.y + b.y) / 2;
    const d = `M ${a.x},${a.y} Q ${mx},${my} ${b.x},${b.y}`;
    color = lk.type === "gaba" ? "#d62728" : lk.type === "ampa" ? "#2e7d32" : "#78909c";
    const path = svgEl("path", { d, fill: "none", stroke: color,
      "stroke-width": lk.type === "muscle" ? 3 : 2.2,
      "marker-end": "url(#arrow" + (lk.type === "gaba" ? "R" : lk.type === "ampa" ? "G" : "K") + ")" }, svgRoot);
    const dot = svgEl("circle", { r: 4, fill: color, opacity: 0 }, svgRoot);
    textEl((a.x + b.x) / 2, (a.y + b.y) / 2 - 9,
      lk.type === "muscle" ? `w=${lk.w}` : (lk.type === "gaba" ? "GABA 15nS" : "AMPA 5nS"),
      { "font-size": 8.5, fill: color, "font-weight": "bold" }, svgRoot);
    linkEls.push({ path, dot, pre, delay: lk.delay || 0.5, len: path.getTotalLength(),
      cut: lk.type === "gaba" });
  });

  // 肌肉条（后退/前进）
  function muscle(x, yTop, yBot, color, label) {
    svgEl("rect", { x: x - MUSCLE_W / 2, y: yTop, width: MUSCLE_W, height: yBot - yTop,
      fill: "#eceff1", stroke: "#b0bec5" }, svgRoot);
    const fill = svgEl("rect", { x: x - MUSCLE_W / 2, y: yBot, width: MUSCLE_W, height: 0,
      fill: color, opacity: 0.85 }, svgRoot);
    textEl(x, yTop - 10, label, { "font-size": 9.5, fill: color, "font-weight": "bold" }, svgRoot);
    textEl(x + MUSCLE_W + 26, (yTop + yBot) / 2 + 3, "0.00",
      { "font-size": 9, fill: color, "text-anchor": "start" }, svgRoot);
    return { fill, yTop, yBot };
  }
  muscleBack = muscle(MUSCLE_X, ROW_Y.DA - 26, ROW_Y.DA + 30, "#2ca02c", "后退收缩 C_back");
  muscleFwd = muscle(MUSCLE_X, ROW_Y.VB - 26, ROW_Y.VB + 30, "#8c564b", "前进收缩 C_fwd");

  // 线虫身体（底部，随净方向 D 移动）
  wormG = svgEl("g", {}, svgRoot);
  svgEl("ellipse", { cx: 0, cy: 0, rx: 60, ry: 19, fill: "#ffe0b2", stroke: "#e65100",
    "stroke-width": 2 }, wormG);
  svgEl("circle", { cx: 55, cy: 0, r: 12, fill: "#ffcc80", stroke: "#e65100",
    "stroke-width": 2 }, wormG);
  svgEl("circle", { cx: 52, cy: -4, r: 1.6, fill: "#4e342e" }, wormG);
  svgEl("circle", { cx: 57, cy: -5, r: 1.6, fill: "#4e342e" }, wormG);
  dirText = textEl(0, 42, "", { "font-size": 12, "font-weight": "bold", fill: "#37474f" }, wormG);
  dMeterText = textEl(300, 662, "", { "font-size": 11, fill: "#455a64" }, svgRoot);

  // 图例
  const leg = [
    ["#2e7d32", "AMPA 谷氨酸兴奋（PLM→AVM→DA）"],
    ["#d62728", "GABA 抑制（AVM→VB）· 切除后虚线断开"],
    ["#7b1fa2", "郎飞结 node1/2/3（300 mS/cm² Na，node3=突触前）"],
    ["#eceff1", "髓鞘 myelin（绝缘，被动）"],
  ];
  leg.forEach(([c, s], i) => {
    svgEl("circle", { cx: 34, cy: 636 + i * 14, r: 5, fill: c, stroke: "#cfd8dc" }, svgRoot);
    textEl(46, 640 + i * 14, s, { "font-size": 9, fill: "#546e7a", "text-anchor": "start" }, svgRoot);
  });
}

/* ---------------- 每帧渲染 ---------------- */
function render() {
  const run = currentRun();
  const i = Math.max(0, Math.min(run.t.length - 1, Math.round(state.t / DT)));
  const t = state.t;

  ROLES.forEach(role => {
    const vMap = run.v[role];
    for (const key in compEls[role]) {
      const arr = vMap[key];
      compEls[role][key].setAttribute("fill", vColor(arr ? arr[i] : -65));
    }
    const sp = run.spikes[role];
    flashEls[role].setAttribute("opacity", sp.some(s => s >= t - 1.6 && s <= t) ? 0.9 : 0);
  });

  // 突触发射点沿路径游走
  linkEls.forEach(lk => {
    const sp = run.spikes[lk.pre];
    let p = -1;
    for (const s of sp) { if (s >= t - lk.delay && s <= t) { p = (t - s) / lk.delay; break; } }
    if (p >= 0) {
      const pt = lk.path.getPointAtLength(p * lk.len);
      lk.dot.setAttribute("cx", pt.x); lk.dot.setAttribute("cy", pt.y);
      lk.dot.setAttribute("opacity", 0.95);
    } else lk.dot.setAttribute("opacity", 0);
    const cut = state.ablation && lk.cut;
    lk.path.setAttribute("stroke-dasharray", cut ? "5,4" : "");
    lk.path.setAttribute("opacity", cut ? 0.35 : 1);
  });

  // 触刺激闪烁
  const inTouch = t >= TOUCH.start && t <= TOUCH.start + run.durMs && run.intensity > 0;
  touchFlash.setAttribute("opacity", inTouch ? 0.9 : 0.0);

  // 肌肉
  const hb = run.cBack[i] * (muscleBack.yBot - muscleBack.yTop);
  muscleBack.fill.setAttribute("height", hb);
  muscleBack.fill.setAttribute("y", muscleBack.yBot - hb);
  const hf = run.cFwd[i] * (muscleFwd.yBot - muscleFwd.yTop);
  muscleFwd.fill.setAttribute("height", hf);
  muscleFwd.fill.setAttribute("y", muscleFwd.yBot - hf);

  // 线虫 + D
  const Dv = run.cBack[i] - run.cFwd[i];
  const wx = 300 + Math.max(-120, Math.min(120, Dv * 260));
  wormG.setAttribute("transform", `translate(${wx}, 626)`);
  if (Math.abs(Dv) < 0.03) {
    dirText.textContent = "静止";
    dirText.setAttribute("fill", "#78909c");
  } else {
    dirText.textContent = (Dv > 0 ? "← 后退" : "前进 →") + `  (D=${Dv.toFixed(2)})`;
    dirText.setAttribute("fill", Dv > 0 ? "#d62728" : "#2e7d32");
  }
  dMeterText.textContent =
    `t=${t.toFixed(1)}ms   D = C_back − C_fwd = ${Dv.toFixed(3)}   （D_peak 判据 > ${THRESH}）`;

  // 面板联动
  el("scrub").value = t;
  updateReadout(run, i, t);
}

/* ---------------- 读数面板 ---------------- */
const el = id => document.getElementById(id);
function updateReadout(run, i, t) {
  const s = run.stats;
  el("rd-t").textContent = t.toFixed(1);
  ROLES.forEach(role => {
    const sp = run.spikes[role].filter(x => x <= t);
    el("rd-" + role).textContent = sp.length ? sp.map(x => x.toFixed(1)).join(", ") : "—";
  });
  el("rd-lat").textContent = s.latencyMs === null ? "—" : s.latencyMs.toFixed(2) + " ms";
  el("rd-dpeak").textContent = s.dPeak.toFixed(3);
  el("rd-cback").textContent = run.cBack[i].toFixed(3);
  el("rd-cfwd").textContent = run.cFwd[i].toFixed(3);
  const v = el("rd-verdict");
  v.textContent = s.verdict;
  v.className = "verdict " + (s.verdict === "后退" ? "ok" : s.verdict === "静默（无反应）" ? "idle" : "warn");
  el("rd-run").textContent = run.label + (run.ablation ? "（AVM→VB 已切除）" : "");
}

/* ---------------- 播放循环 ---------------- */
function tick(ts) {
  if (!state.playing) return;
  if (!state.lastTs) state.lastTs = ts;
  const dt = (ts - state.lastTs) / 1000;
  state.lastTs = ts;
  state.t += dt * 1000 * (state.speedMs / 16.67);   // 1× ≈ 1ms 仿真 / 帧
  if (state.t >= T_TOTAL - 0.05) { state.t = T_TOTAL - 0.05; state.playing = false; el("btn-play").textContent = "▶ 播放"; }
  render();
  state.raf = requestAnimationFrame(tick);
}
function play() {
  if (state.playing) return;
  state.playing = true; state.lastTs = 0;
  el("btn-play").textContent = "⏸ 暂停";
  state.raf = requestAnimationFrame(tick);
}
function pause() {
  state.playing = false;
  el("btn-play").textContent = "▶ 播放";
  if (state.raf) cancelAnimationFrame(state.raf);
}
function setRun(id, opts, autoplay) {
  pause();
  const run = D.runs[id];
  state.runId = id;
  if (opts) { state.intensity = run.intensity; state.durMs = run.durMs; state.ablation = run.ablation; }
  state.t = TOUCH.start - 4;
  syncControls();
  render();
  appendLog(run);
  if (autoplay) play();   // 点刺激/手术 → 自动从头播放，立即看到反应
}

/* 实验记录表 */
function appendLog(run) {
  const s = run.stats;
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${run.label}</td><td>${s.latencyMs === null ? "—" : s.latencyMs.toFixed(2)}</td>
    <td>${s.dPeak.toFixed(3)}</td><td>${s.cBackPeak.toFixed(3)}</td>
    <td>${s.vbWinCount}</td><td>${s.verdict}</td>`;
  document.querySelectorAll("#log tbody tr").forEach(x => (x.className = ""));
  tr.className = "cur";
  el("log").querySelector("tbody").appendChild(tr);
}

/* ---------------- 控制绑定 ---------------- */
function syncControls() {
  el("sl-intensity").value = state.intensity;
  el("chk-ablation").checked = state.ablation;
  document.querySelectorAll(".dur-btn").forEach(b => b.classList.toggle("on", parseFloat(b.dataset.dur) === state.durMs));
  document.querySelectorAll(".stim-btn").forEach(b => b.classList.toggle("on", parseFloat(b.dataset.v) === state.intensity && state.durMs === 5 && !state.ablation));
}
function onIntensity() {
  const v = parseFloat(el("sl-intensity").value);
  state.intensity = v; state.durMs = 5;
  const wasPlaying = state.playing;   // 拖动滑条时保留播放状态，不强制打断
  setRun(runIdFor(v, 5, state.ablation), null, wasPlaying);
}
function onDur(dur) {
  state.durMs = dur;
  setRun(runIdFor(state.intensity, dur, state.ablation), null, true);
}
function onAblation() {
  state.ablation = el("chk-ablation").checked;
  setRun(runIdFor(state.intensity, state.durMs, state.ablation), null, true);
}

/* 反应速度挑战（娱乐向：对比你的手速与线虫 9.6ms 潜伏期） */
function startGame() {
  setRun("i1", {});
  state.t = TOUCH.start - 5;
  el("btn-react").classList.remove("hidden");
  el("btn-react").disabled = false;
  el("game-msg").textContent = "触刺激已发出 —— 线虫 9.6ms 内就会缩！快按【我缩了！】";
  state.game.active = true;
  state.game.touchAt = performance.now();
  play();
}
function reactNow() {
  if (!state.game.active) return;
  const human = performance.now() - state.game.touchAt;
  const worm = currentRun().stats.latencyMs;
  el("game-msg").textContent = worm === null
    ? "本组无反应（静默）"
    : `线虫 ${worm.toFixed(1)}ms vs 你 ${human.toFixed(0)}ms —— ${human < worm * 1.5 ? "你赢了？！不可能" : "线虫完胜（你按了两次按钮的时间，够它缩好几回）"}`;
  state.game.active = false;
  el("btn-react").disabled = true;
}

/* ---------------- 初始化 ---------------- */
function init() {
  buildCircuit();
  el("btn-play").addEventListener("click", () => (state.playing ? pause() : play()));
  el("btn-reset").addEventListener("click", () => { state.t = 0; render(); });
  el("btn-clear").addEventListener("click", () => { el("log").querySelector("tbody").innerHTML = ""; });
  el("sl-intensity").addEventListener("input", onIntensity);
  el("chk-ablation").addEventListener("change", onAblation);
  document.querySelectorAll(".dur-btn").forEach(b => b.addEventListener("click", () => onDur(parseFloat(b.dataset.dur))));
  document.querySelectorAll(".stim-btn").forEach(b => b.addEventListener("click", () => {
    const v = parseFloat(b.dataset.v);
    state.intensity = v; state.durMs = 5; state.ablation = false;
    setRun(runIdFor(v, 5, false), null, true);   // 自动播放
  }));
  el("speed").addEventListener("change", () => { state.speedMs = parseFloat(el("speed").value); });
  el("scrub").addEventListener("input", () => { state.t = parseFloat(el("scrub").value); render(); });
  el("btn-game").addEventListener("click", startGame);
  el("btn-react").addEventListener("click", reactNow);

  setRun("i1", {});
  state.t = TOUCH.start - 4;
  render();
  play();
}
document.addEventListener("DOMContentLoaded", init);
