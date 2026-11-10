/* 영상 캡처 (2026-11-10, 사용자 요청) — 영상을 불러와 몇 초마다 한 장 뽑고, 넣을 것만 골라 데이터(unlabeled/<카메라>/)에 넣는다.
 *
 * 라벨링·이미지 보기와 같은 페이지의 세 번째 보기다(app.js setView — body.view-capture). 서버 쪽은 src/autolabel/capture.py.
 * - 영상은 서버에 올린다(사본) — 브라우저는 고른 파일의 경로를 주지 않고, 브라우저가 못 푸는 코덱도 있어서다. 서버가
 *   tools/extract_frames.py 와 같은 방식(OpenCV)으로 뽑는다. 넣거나 닫으면 사본·남은 프레임을 지운다(원본 영상은 그대로).
 * - 뽑는 동안 0.7초마다 새 프레임만 받아(since=) 칸을 늘린다. 다시 뽑으면 서버의 run 이 바뀌어 처음부터 받는다.
 * - 넣기/빼기는 이 화면이 들고 있다가 0.6초 뒤 서버에 남긴다(새로 고쳐도 이어진다). 넣을 때는 고른 번호를 그대로 보낸다.
 * - 키는 app.js keydown 이 captureKey 로만 넘긴다(입력 칸·대화상자는 거기서 먼저 걸렀다) — 숨은 라벨링에 먹지 않는다.
 */

const C = {
  sessions: [],       // 불러온 영상 [{sid, name, state, n, keep, …}] (/api/capture)
  cameras: [],        // 등록된 카메라 [{key, kind, place, mount, split}] (cameras.yaml)
  cameraError: null,  // 등록 파일을 못 읽은 이유
  unlabeled: "",      // 넣을 폴더 (화면 표시용 상대경로)
  limits: { min_every: 0.2, max_every: 3600, max_frames: 5000 },
  cur: null,          // 지금 영상 (프레임은 C.frames 에 따로)
  frames: [],         // [{i, t, c, s, k}] — t 영상 안 초, c 앞 장과 바뀐 픽셀 %, s 픽셀까지 같은 앞 장 번호, k 넣기
  cells: [],          // 그린 칸 (frames 와 같은 자리)
  focus: -1,          // 지금 칸 (방향키·Enter)
  anchor: -1,         // Shift+클릭 범위의 시작 칸
  poll: null,
  polling: false,
  seq: 0,             // 영상 여는 순번 — 늦게 온 옛 응답을 버린다
  keepTimer: null,
  keepDirty: false,
  startTimer: null,
  startPending: null, // 아직 서버에 안 남긴 시작 시각 {sid, start}
  hintHtml: "",       // 시작 시각 안내 — 바뀔 때만 다시 쓴다(뽑는 동안 0.7초마다 그리면 누르던 버튼이 사라진다)
  xhr: null,          // 올리는 중인 요청
  committing: false,
  shown: false,       // 이 보기가 떠 있나
  pv: -1,             // 크게 보는 프레임 (-1 = 닫힘)
  size: 220,
};
const CAP_RUNNING = new Set(["queued", "extracting"]);
const CAP_STATE = {
  uploading: "올리는 중", ready: "뽑기 전", queued: "기다리는 중", extracting: "뽑는 중", done: "다 뽑음",
  stopped: "멈춤", error: "실패", committing: "넣는 중",
};
const CAP_CUR_KEY = `capture:cur:${BASE}`;
const CAP_SIZE_KEY = "captureSize";
const CAP_EVERY_KEY = "captureEvery";
const CAP_VIDEO_EXT = /\.(mp4|mkv|avi|mov|ts|m2ts|mts|flv|wmv|webm|mpe?g|m4v|3gp)$/i;

/* ───────────────────────── 꼴 ───────────────────────── */

function capClock(sec) {
  if (sec == null || !isFinite(sec)) return "?";
  sec = Math.max(0, sec);
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60);
  const p = (x) => String(x).padStart(2, "0");
  return h ? `${h}:${p(m)}:${p(s)}` : `${m}:${p(s)}`;
}

/** 칸에 쓰는 영상 안 시각 — 0.5초 간격처럼 초 아래가 있으면 한 자리 붙인다. */
function capOffset(sec) {
  const frac = sec - Math.floor(sec);
  return capClock(sec) + (frac >= 0.05 ? `.${Math.floor(frac * 10)}` : "");
}

function capBytes(n) {
  if (n == null) return "?";
  return n >= 1e9 ? `${(n / 1e9).toFixed(2)}GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}MB` : `${Math.max(1, Math.round(n / 1e3))}KB`;
}

function capPct(x) {
  return x < 1 ? `${x.toFixed(2)}%` : x < 10 ? `${x.toFixed(1)}%` : `${Math.round(x)}%`;
}

/** '90' · '1:30' · '1:02:03.5' → 초. 비면 null, 못 읽으면 NaN. */
function capParseClock(text) {
  const s = String(text || "").trim();
  if (!s) return null;
  if (!/^\d+(:\d{1,2}){0,2}(\.\d+)?$/.test(s)) return NaN;
  return s.split(":").reduce((acc, part) => acc * 60 + Number(part), 0);
}

/** 촬영 시작 시각 'YYYY-MM-DD HH:MM:SS(.fff)' · 'YYYYMMDD_HHMMSS' → {base(ms, UTC 로 셈), us(초 아래 µs)}. 못 읽으면 null.
 *  서버 capture.parse_start 와 같은 꼴만 받는다 — 여기서는 이름 미리 보기용이고, 확인은 서버가 넣을 때 한다. */
function capParseStart(text) {
  const s = String(text || "").trim();
  let m = s.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$/);
  if (!m) {
    m = s.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
    if (m) m[7] = undefined;
  }
  if (!m) return null;
  const [y, mo, d, h, mi, se] = m.slice(1, 7).map(Number);
  const base = Date.UTC(y, mo - 1, d, h, mi, se);
  const dt = new Date(base);
  if (dt.getUTCFullYear() !== y || dt.getUTCMonth() !== mo - 1 || dt.getUTCDate() !== d
      || dt.getUTCHours() !== h || dt.getUTCMinutes() !== mi || dt.getUTCSeconds() !== se) return null;   // 13월·25시 등
  const us = m[7] ? Number(m[7].padEnd(6, "0")) : 0;
  return { base, us };
}

/** 시작 + 영상 안 시각 → Date(UTC 로 센 현지 시각). 서버처럼 µs 까지 더하고 밀리초 아래는 버린다. */
function capAt(st, sec) {
  return new Date(st.base + Math.floor((st.us + Math.round(sec * 1e6)) / 1000));
}

/** capture.frame_name 과 같은 이름 `<카메라>_<YYYYMMDD_HHMMSS>_<밀리초>.jpg` 과 날짜 폴더. */
function capName(cam, d) {
  const p = (x, n = 2) => String(x).padStart(n, "0");
  const day = `${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}`;
  return { day, name: `${cam}_${day}_${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}_${p(d.getUTCMilliseconds(), 3)}.jpg` };
}

function capWallClock(d) {
  const p = (x) => String(x).padStart(2, "0");
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`;
}

function capKept() {
  let n = 0;
  for (const f of C.frames) if (f.k) n++;
  return n;
}

function capRunning() {
  return !!C.cur && CAP_RUNNING.has(C.cur.state);
}

/* ───────────────────────── 보기 전환 ───────────────────────── */

/** 영상 캡처로 올 때(setView). 목록을 새로 받고, 보던 영상(없으면 지난번 것)을 다시 연다 — 숨어 있는 동안에도 서버는 뽑는다. */
async function captureShow() {
  C.shown = true;
  try {
    await capLoadList();
  } catch (err) {
    toast(`영상 목록을 받지 못했습니다: ${err.message}`, true);
    capRenderAll();
    return;
  }
  let sid = C.cur ? C.cur.sid : null;
  if (!sid) { try { sid = localStorage.getItem(CAP_CUR_KEY); } catch { /* 없으면 그만 */ } }
  if (sid && C.sessions.some((s) => s.sid === sid)) await capOpen(sid);
  else { capReset(); capRenderAll(); }
}

/** 다른 보기로 갈 때(setView). 서버는 계속 뽑는다 — 돌아오면 captureShow 가 따라잡는다. */
function captureHide() {
  C.shown = false;
  capStopPoll();
  capFlushKeep();
  capFlushStart();
  capClosePreview();
}

/* ───────────────────────── 서버 ───────────────────────── */

async function capLoadList() {
  const d = await api("/api/capture");
  C.sessions = d.sessions || [];
  C.cameras = d.cameras || [];
  C.cameraError = d.camera_error || null;
  C.unlabeled = d.unlabeled || "";
  if (d.limits) C.limits = d.limits;
  capRenderSessions();
  capRenderCameras();
}

/** 영상 하나를 연다 — 프레임을 처음부터 받는다. */
async function capOpen(sid) {
  const seq = ++C.seq;
  capFlushKeep();
  capFlushStart();
  let d;
  try {
    d = await api(`/api/capture/${sid}?since=0`);
  } catch (err) {
    if (seq !== C.seq) return;
    if (err.status === 404) { capGone(sid, err.message); return; }
    toast(err.message, true);
    return;
  }
  if (seq !== C.seq) return;
  capStopPoll();
  capClosePreview();
  C.frames = d.frames || [];
  delete d.frames;
  C.cur = d;
  C.focus = -1;
  C.anchor = -1;
  try { localStorage.setItem(CAP_CUR_KEY, sid); } catch { /* 그만 */ }
  // 입력 칸: 이 영상을 뽑은 설정이 있으면 그것, 없으면 지난번 간격(구간은 비운다)
  const st = d.settings;
  let every = st ? st.every : null;
  if (every == null) { try { every = Number(localStorage.getItem(CAP_EVERY_KEY)) || 5; } catch { every = 5; } }
  $("#cap-every").value = String(every);
  $("#cap-from").value = st && st.t_from ? capClock(st.t_from) : "";
  $("#cap-to").value = st && st.t_to != null ? capClock(st.t_to) : "";
  $("#cap-start").value = d.start || "";
  capRenderAll();
  if (capRunning()) capStartPoll();
}

/** 그 영상이 서버에서 없어졌다(다른 탭에서 넣었거나 닫았다). */
function capGone(sid, msg) {
  if (C.cur && C.cur.sid === sid) {
    capReset();
    toast(msg || "이 영상은 이미 닫혔습니다", true);
  }
  capLoadList().catch(() => {}).finally(capRenderAll);
}

/** 지금 영상을 비운다(넣기·닫기 뒤). */
function capReset() {
  capStopPoll();
  capClosePreview();
  clearTimeout(C.keepTimer);
  clearTimeout(C.startTimer);
  C.keepDirty = false;
  C.cur = null;
  C.frames = [];
  C.cells = [];
  C.focus = -1;
  C.anchor = -1;
  $("#cap-start").value = "";
  try { localStorage.removeItem(CAP_CUR_KEY); } catch { /* 그만 */ }
}

function capStartPoll() {
  capStopPoll();
  C.poll = setInterval(capTick, 700);
}

function capStopPoll() {
  clearInterval(C.poll);
  C.poll = null;
}

/** 뽑는 동안: 새 프레임만 받아 칸을 늘린다. */
async function capTick() {
  if (C.polling || !C.cur) return;
  C.polling = true;
  const sid = C.cur.sid, seq = C.seq, before = C.frames.length;
  try {
    const d = await api(`/api/capture/${sid}?since=${before}`);
    if (seq !== C.seq || !C.cur || C.cur.sid !== sid || C.frames.length !== before) return;
    if (d.run !== C.cur.run) { await capOpen(sid); return; }     // 다른 탭에서 다시 뽑았다
    C.frames.push(...(d.frames || []));
    // 카메라·시작 시각은 받지 않는다 — 고치는 중인 값을 옛 값으로 덮지 않게
    for (const k of ["state", "pos", "error", "note", "n", "settings", "video"]) C.cur[k] = d[k];
    capAddCells(before);
    capRenderExtract();
    capRenderCount();
    capRenderCommit();
    if (!CAP_RUNNING.has(d.state)) {
      capStopPoll();
      const n = C.frames.length;
      if (d.state === "done") toast(`다 뽑았습니다 — ${n}장. 넣을 것만 남기고 '넣기' 를 누르세요`);
      else if (d.state === "error") toast(`뽑지 못했습니다: ${d.error}`, true);
      capLoadList().catch(() => {});
    }
  } catch (err) {
    if (err.status === 404) { capStopPoll(); capGone(sid, err.message); }
    // 그 밖의 오류는 다음 tick 에 다시 해 본다
  } finally {
    C.polling = false;
  }
}

/* ───────────────────────── 올리기 ───────────────────────── */

function capUpload(file) {
  if (!file) return;
  if (C.xhr) { toast("다른 영상을 올리는 중입니다 — 끝난 뒤 다시 하세요", true); return; }
  if (!/^video\//.test(file.type) && !CAP_VIDEO_EXT.test(file.name)
      && !confirm(`'${file.name}' 은 영상 파일로 보이지 않습니다. 그래도 올릴까요?`)) return;
  const q = new URLSearchParams({ name: file.name, size: String(file.size) });
  if (file.lastModified > 0) q.set("mtime", String(file.lastModified));
  const xhr = new XMLHttpRequest();
  C.xhr = xhr;
  const bar = $("#cap-upload-bar"), text = $("#cap-upload-text");
  $("#cap-upload").classList.remove("hidden");
  $("#cap-result").classList.add("hidden");
  bar.style.width = "0%";
  text.textContent = `${file.name} 올리는 중…`;
  const done = () => {
    C.xhr = null;
    $("#cap-upload").classList.add("hidden");
    capRenderExtract();
  };
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    bar.style.width = `${(100 * e.loaded / e.total).toFixed(1)}%`;
    text.textContent = `${file.name} 올리는 중 ${capBytes(e.loaded)} / ${capBytes(e.total)}`;
  };
  xhr.upload.onload = () => { text.textContent = `${file.name} — 영상을 확인하는 중…`; };
  xhr.onload = async () => {
    done();
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch { /* 본문 없음 */ }
    if (xhr.status < 200 || xhr.status >= 300) {
      toast(errorText(data.detail) || `올리지 못했습니다 (HTTP ${xhr.status})`, true);
      return;
    }
    try {
      await capLoadList();
    } catch { /* 목록은 다음에 */ }
    if (VIEW === "capture") {
      await capOpen(data.sid);
    } else {                         // 다른 보기로 가 있었다 — 돌아오면 이 영상이 열리게
      capFlushKeep();
      capFlushStart();
      capReset();
      try { localStorage.setItem(CAP_CUR_KEY, data.sid); } catch { /* 그만 */ }
    }
    toast(`올렸습니다: ${file.name} — 간격을 정하고 '캡처 시작' 을 누르세요`);
  };
  xhr.onerror = () => { done(); toast("올리지 못했습니다 — 서버와 연결이 끊겼습니다", true); };
  xhr.onabort = () => { done(); toast("올리기를 취소했습니다"); };
  xhr.open("POST", url(`/api/capture/upload?${q}`));
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.send(file);
  capRenderExtract();
}

/* ───────────────────────── 뽑기 ───────────────────────── */

/** 지금 입력으로 몇 장이 나오나 (영상 길이를 알 때만). */
function capEstimate() {
  const every = Number($("#cap-every").value);
  const from = capParseClock($("#cap-from").value), to = capParseClock($("#cap-to").value);
  const dur = C.cur && C.cur.video ? C.cur.video.duration : null;
  const bad = [];
  if (!(every >= C.limits.min_every && every <= C.limits.max_every)) bad.push(`간격은 ${C.limits.min_every}~${C.limits.max_every}초`);
  if (Number.isNaN(from) || Number.isNaN(to)) bad.push("구간은 90 · 1:30 · 1:02:03 꼴로");
  else if (to != null && to <= (from || 0)) bad.push("구간의 끝이 시작보다 뒤여야 합니다");
  else if (dur && (from || 0) >= dur) bad.push(`시작이 영상 길이(${capClock(dur)})를 넘습니다`);
  $("#cap-every").classList.toggle("bad", bad.length > 0 && !(every >= C.limits.min_every && every <= C.limits.max_every));
  $("#cap-from").classList.toggle("bad", Number.isNaN(from));
  $("#cap-to").classList.toggle("bad", Number.isNaN(to));
  if (bad.length) return { ok: false, text: bad.join(" · ") };
  if (!dur) return { ok: true, every, from: from || 0, to, text: "영상 길이를 몰라 몇 장이 나올지 셀 수 없습니다 — 끝까지 읽어 봅니다" };
  const end = to != null ? Math.min(to, dur) : dur;
  const n = Math.floor((end - (from || 0)) / every + 1e-9) + 1;
  const over = n > C.limits.max_frames;
  return {
    ok: !over, every, from: from || 0, to, n,
    text: over ? `약 ${n}장 — 한 번에 ${C.limits.max_frames}장까지입니다. 간격을 늘리거나 구간을 줄이세요`
      : `약 ${n}장 (${capClock(from || 0)} ~ ${capClock(end)})`,
  };
}

async function capRun() {
  if (!C.cur || capRunning()) return;
  const est = capEstimate();
  if (!est.ok) { toast(est.text, true); return; }
  if (C.frames.length && !confirm(`다시 뽑으면 지금 뽑은 ${C.frames.length}장과 넣기/빼기 고른 것이 지워집니다. 계속할까요?`)) return;
  const sid = C.cur.sid, seq = ++C.seq;
  try {
    const d = await api(`/api/capture/${sid}/extract`, {
      method: "POST", body: { every: est.every, t_from: est.from, t_to: est.to },
    });
    if (seq !== C.seq) return;
    try { localStorage.setItem(CAP_EVERY_KEY, String(est.every)); } catch { /* 그만 */ }
    capClosePreview();
    clearTimeout(C.keepTimer);
    C.keepDirty = false;
    C.frames = d.frames || [];
    delete d.frames;
    d.start = C.cur.start;          // 고치는 중인 시작 시각·카메라는 화면 값을 그대로
    d.camera = C.cur.camera;
    C.cur = d;
    C.focus = -1;
    C.anchor = -1;
    capRenderAll();
    capStartPoll();
  } catch (err) {
    toast(err.message, true);
  }
}

async function capStop() {
  if (!C.cur) return;
  try {
    const d = await api(`/api/capture/${C.cur.sid}/stop`, { method: "POST" });
    C.cur.state = d.state;
    capRenderExtract();
  } catch (err) {
    toast(err.message, true);
  }
}

/* ───────────────────────── 고르기 ───────────────────────── */

function capSetKeep(idxs, val) {
  let changed = false;
  for (const i of idxs) {
    const f = C.frames[i];
    if (!f || f.k === val) continue;
    f.k = val;
    changed = true;
    capSyncCell(i);
  }
  if (!changed) return;
  capRenderCount();
  capRenderCommit();
  C.keepDirty = true;
  clearTimeout(C.keepTimer);
  C.keepTimer = setTimeout(capFlushKeep, 600);
}

/** 고른 것을 서버에 남긴다 — 다시 열면 이어진다. 실패해도 넣을 때 고른 번호를 그대로 보내므로 알리기만 한다. */
function capFlushKeep() {
  clearTimeout(C.keepTimer);
  if (!C.keepDirty || !C.cur) return;
  C.keepDirty = false;
  const drop = [];
  for (const f of C.frames) if (!f.k) drop.push(f.i);
  api(`/api/capture/${C.cur.sid}/state`, { method: "PUT", body: { drop, upto: C.frames.length } })
    .catch((err) => { if (err.status !== 404) console.warn("넣기/빼기를 남기지 못했습니다", err); });
}

/** 고친 시작 시각을 기다리지 않고 바로 남긴다(다른 영상을 열거나 보기를 떠날 때). */
function capFlushStart() {
  clearTimeout(C.startTimer);
  C.startTimer = null;
  const p = C.startPending;
  C.startPending = null;
  if (p) api(`/api/capture/${p.sid}/state`, { method: "PUT", body: { start: p.start } }).catch(() => {});
}

function capRange(a, b) {
  const [lo, hi] = a < b ? [a, b] : [b, a];
  const out = [];
  for (let i = lo; i <= hi; i++) out.push(i);
  return out;
}

/* ───────────────────────── 넣기·닫기 ───────────────────────── */

async function capCommit() {
  if (!C.cur || C.committing) return;
  const keep = C.frames.filter((f) => f.k).map((f) => f.i);
  const cam = $("#cap-camera").value, start = $("#cap-start").value.trim();
  if (!keep.length) { toast("넣을 프레임을 고르지 않았습니다", true); return; }
  if (!cam) { toast("카메라를 고르세요", true); return; }
  if (!capParseStart(start)) { toast("촬영 시작 시각을 읽을 수 없습니다 — 예: 2026-11-10 14:30:00", true); return; }
  const drop = C.frames.length - keep.length;
  const msg = `${keep.length}장을 ${C.unlabeled}/${cam}/ 에 넣고 스캔합니다(DB 에 미라벨로 등록).\n`
    + `${drop ? `안 고른 ${drop}장과 ` : ""}올린 영상 사본은 지웁니다 — 원본 영상 파일은 그대로입니다.\n\n계속할까요?`;
  if (!confirm(msg)) return;
  const sid = C.cur.sid, name = C.cur.name;
  C.committing = true;
  capRenderCommit();
  try {
    const r = await api(`/api/capture/${sid}/commit`, { method: "POST", body: { camera: cam, start, keep } });
    capShowResult(r, name);
    capReset();
    capRenderAll();
    capLoadList().catch(() => {});
    // 라벨링 쪽 숫자·목록 (미라벨이 늘었다)
    try { await loadState(); await loadImageList(); } catch { /* 라벨링으로 가면 다시 받는다 */ }
  } catch (err) {
    toast(err.message, true);
    if (err.status === 404) capGone(sid, err.message);
  } finally {
    C.committing = false;
    capRenderCommit();
  }
}

/** 넣은 결과를 왼쪽 '넣기' 칸에 남긴다(다음 영상을 열 때까지). */
function capShowResult(r, name) {
  const lines = [];
  const reg = r.registered;
  lines.push(`<b>${esc(name)}</b> — ${esc(r.camera)} 에 <b>${r.moved}</b>장을 넣었습니다`
    + (reg != null ? ` (DB 등록 ${reg}장)` : "") + ".");
  if (r.exists) lines.push(`같은 이름이 이미 있어 건너뜀 ${r.exists}장 — 같은 구간을 전에 넣었습니다.`);
  if (r.missing) lines.push(`프레임 파일이 없어 건너뜀 ${r.missing}장.`);
  if (r.duplicates) lines.push(`내용이 이미 있는 이미지와 같아 등록하지 않음 ${r.duplicates}장(멈춘 화면 등).`);
  if (r.unreadable) lines.push(`읽지 못해 등록하지 않음 ${r.unreadable}장.`);
  if (r.other_new) lines.push(`폴더에 있던 다른 새 이미지 ${r.other_new}장도 함께 등록됐습니다.`);
  if (r.scan_error) lines.push(`<span class="err-text">파일은 옮겼지만 스캔이 멈췄습니다: ${esc(r.scan_error)}<br>고친 뒤 데이터 → 폴더 스캔.</span>`);
  if (r.left) lines.push(esc(r.left));
  if (r.moved && !r.scan_error) {
    lines.push('다음: 라벨링에서 <b>미라벨</b>로 보고, 모델 → 자동 라벨링. '
      + '<button class="linkbtn" id="cap-go-label">라벨링에서 보기</button>');
  }
  const box = $("#cap-result");
  box.innerHTML = lines.join("<br>");
  box.classList.toggle("err", !!r.scan_error || !r.moved);
  box.classList.remove("hidden");
  const go = $("#cap-go-label");
  if (go) {
    go.onclick = async () => {
      $("#filter-status").value = "unlabeled";
      await setView("label");
      try { await reloadList(); } catch (err) { toast(err.message, true); }
    };
  }
  toast(r.moved ? `${r.moved}장을 넣었습니다` : "넣은 것이 없습니다", !r.moved || !!r.scan_error);
}

async function capDiscard() {
  if (!C.cur) return;
  const n = C.frames.length;
  if (!confirm(`이 영상(${C.cur.name})을 닫을까요?\n`
    + `${n ? `뽑은 ${n}장과 ` : ""}올린 사본을 지웁니다 — 원본 영상 파일은 그대로입니다.`)) return;
  const sid = C.cur.sid;
  try {
    const r = await api(`/api/capture/${sid}`, { method: "DELETE" });
    if (r.left) toast(r.left, true);
  } catch (err) {
    if (err.status !== 404) { toast(err.message, true); return; }
  }
  if (C.cur && C.cur.sid === sid) capReset();
  $("#cap-result").classList.add("hidden");
  await capLoadList().catch(() => {});
  capRenderAll();
}

/* ───────────────────────── 그리기 ───────────────────────── */

function capRenderAll() {
  capRenderVideo();
  capRenderExtract();
  capRenderCommit();
  capRenderSessions();
  capRenderGrid();
  capRenderCount();
}

function capRenderVideo() {
  const box = $("#cap-video");
  const c = C.cur;
  if (!c) {
    box.innerHTML = '<div class="empty">영상 파일을 고르거나 이 화면에 끌어다 놓으세요.<br>'
      + 'mp4·mkv·avi 등 OpenCV(FFmpeg)가 읽는 형식이면 됩니다.</div>';
    return;
  }
  const v = c.video || {};
  const meta = [
    v.width && v.height ? `${v.width}×${v.height}` : null,
    v.fps ? `${Number(v.fps).toFixed(2).replace(/\.?0+$/, "")}fps` : null,
    v.duration ? capClock(v.duration) : "길이 모름",
    capBytes(v.size),
  ].filter(Boolean).join(" · ");
  box.innerHTML = `<div class="nm">${esc(c.name)}</div><div class="meta">${esc(meta)}</div>`;
}

function capRenderExtract() {
  const c = C.cur, running = capRunning();
  const est = c ? capEstimate() : null;
  const estEl = $("#cap-estimate");
  estEl.textContent = est ? est.text : "";
  estEl.classList.toggle("warn", !!est && !est.ok);
  const run = $("#cap-run");
  run.disabled = !c || running || !est || !est.ok || ["uploading", "committing"].includes(c.state) || C.committing;
  run.textContent = c && C.frames.length && !running ? "다시 캡처" : "캡처 시작";
  $("#cap-stop").classList.toggle("hidden", !running);
  const prog = $("#cap-progress");
  if (!c || c.state === "ready" || c.state === "uploading") { prog.classList.add("hidden"); return; }
  prog.classList.remove("hidden");
  const st = c.settings || {};
  const dur = c.video ? c.video.duration : null;
  const from = st.t_from || 0, end = st.t_to != null ? Math.min(st.t_to, dur || st.t_to) : dur;
  let frac = null;
  if (end && end > from) frac = Math.max(0, Math.min(1, ((c.pos || 0) - from) / (end - from)));
  if (c.state === "done") frac = 1;
  $("#cap-progress-bar").style.width = frac == null ? "0%" : `${(frac * 100).toFixed(1)}%`;
  const n = C.frames.length;
  const pos = `${capClock(c.pos || 0)}${end ? ` / ${capClock(end)}` : ""}`;
  const text = {
    queued: "기다리는 중 — 다른 영상을 먼저 뽑고 있습니다",
    extracting: `뽑는 중 ${pos} · ${n}장`,
    done: `다 뽑았습니다 — ${n}장 (${st.every}초마다)`,
    stopped: `멈췄습니다 — ${n}장 (${capClock(c.pos || 0)}까지)`,
    error: `실패: ${c.error || ""}`,
    committing: "넣는 중…",
  }[c.state] || CAP_STATE[c.state] || c.state;
  const el = $("#cap-progress-text");
  el.textContent = text + (c.note ? ` — ${c.note}` : "");
  el.classList.toggle("warn", c.state === "error" || !!c.note);
}

function capRenderCameras() {
  const sel = $("#cap-camera");
  const cur = C.cur ? C.cur.camera || "" : "";
  const opts = ['<option value="">카메라 고르기…</option>'];
  for (const cam of C.cameras) {
    const tail = [cam.kind, cam.split ? `${cam.split} 고정` : null].filter(Boolean).join(" · ");
    opts.push(`<option value="${esc(cam.key)}" title="${esc(cam.place || "")}">${esc(cam.key)} (${esc(tail)})</option>`);
  }
  // 등록 파일에서 빠진 카메라를 골라 둔 영상 — 목록에 남겨 두고(넣을 때 서버가 막는다) 표시만 한다
  if (cur && !C.cameras.some((c) => c.key === cur)) opts.push(`<option value="${esc(cur)}">${esc(cur)} (등록 안 됨)</option>`);
  sel.innerHTML = opts.join("");
  sel.value = cur;
}

function capRenderCommit() {
  const c = C.cur;
  const sel = $("#cap-camera"), startEl = $("#cap-start");
  if (c && sel.value !== (c.camera || "")) capRenderCameras();
  sel.disabled = !c;
  startEl.disabled = !c;
  const hint = $("#cap-start-hint"), dest = $("#cap-dest");
  const cam = sel.value;
  const st = capParseStart(startEl.value);
  const kept = capKept();
  startEl.classList.toggle("bad", !!c && !!startEl.value.trim() && !st);
  // 시작 시각 안내
  let h = "";
  if (C.cameraError) h += `<span class="warn">카메라 등록 파일을 읽지 못했습니다: ${esc(C.cameraError)}</span><br>`;
  else if (c && !C.cameras.length) h += '<span class="warn">등록된 카메라가 없습니다 — cameras.yaml 에 먼저 등록하세요(docs/provenance.md).</span><br>';
  if (c) {
    const typed = startEl.value.trim();
    if (!typed) {
      h += c.start_guess ? "" : "영상 이름에 시각이 없습니다 — 첫 프레임의 현지 시각을 적으세요.";
    } else if (!st) {
      h += "읽을 수 없습니다 — 예: 2026-11-10 14:30:00";
    } else if (typed === c.start_guess) {
      h += "영상 이름에서 읽었습니다.";
    }
    if (c.start_mtime && typed !== c.start_mtime) {
      h += `${h && !h.endsWith("<br>") ? " " : ""}<button class="linkbtn" id="cap-use-mtime" title="파일 수정 시각에서 영상 길이를 뺀 값입니다. OBS 녹화처럼 녹화가 끝날 때 닫힌 파일에서만 맞고, 복사·내려받은 파일이면 틀립니다">수정 시각 − 길이: ${esc(c.start_mtime)} 로 채우기</button>`;
    }
    if (c.start_guess && typed !== c.start_guess) {
      h += `${h && !h.endsWith("<br>") ? " " : ""}<button class="linkbtn" id="cap-use-guess">영상 이름의 시각 ${esc(c.start_guess)} 로</button>`;
    }
  }
  if (h !== C.hintHtml) {
    C.hintHtml = h;
    hint.innerHTML = h;
    const m = $("#cap-use-mtime");
    if (m) m.onclick = () => capSetStart(C.cur && C.cur.start_mtime);
    const g = $("#cap-use-guess");
    if (g) g.onclick = () => capSetStart(C.cur && C.cur.start_guess);
  }
  // 넣을 곳 미리 보기
  if (c && cam && st && kept) {
    const ks = C.frames.filter((f) => f.k);
    const a = capName(cam, capAt(st, ks[0].t)), b = capName(cam, capAt(st, ks[ks.length - 1].t));
    dest.textContent = `→ ${C.unlabeled}/${cam}/${a.day}/\n   ${a.name}` + (ks.length > 1 ? `\n ~ ${b.day === a.day ? "" : `${b.day}/`}${b.name}` : "");
  } else {
    dest.textContent = "";
  }
  const btn = $("#cap-commit");
  btn.textContent = C.committing ? "넣는 중…" : `고른 ${kept}장 넣기`;
  btn.disabled = !c || C.committing || capRunning() || !kept || !cam || !st || ["uploading", "committing"].includes(c.state);
  btn.title = !c ? "" : capRunning() ? "뽑는 중에는 넣을 수 없습니다 — 끝나거나 멈춘 뒤"
    : !kept ? "넣을 프레임을 고르세요" : !cam ? "카메라를 고르세요" : !st ? "촬영 시작 시각을 적으세요"
      : "고른 프레임을 unlabeled/<카메라>/ 로 옮기고 스캔합니다";
  $("#cap-discard").disabled = !c || C.committing || ["uploading", "committing"].includes(c ? c.state : "");
}

function capSetStart(v) {
  $("#cap-start").value = v || "";
  capStartChanged();
}

/** 시작 시각을 고쳤다 — 화면을 다시 그리고, 0.6초 뒤 서버에 남긴다. */
function capStartChanged() {
  if (!C.cur) return;
  C.cur.start = $("#cap-start").value.trim();
  capRenderCommit();
  if (C.pv >= 0) capRenderPreview();
  clearTimeout(C.startTimer);
  C.startPending = { sid: C.cur.sid, start: C.cur.start };
  C.startTimer = setTimeout(capFlushStart, 600);
}

function capRenderSessions() {
  const ul = $("#cap-sessions");
  $("#cap-sess-n").textContent = C.sessions.length ? ` ${C.sessions.length}` : "";
  if (!C.sessions.length) {
    ul.innerHTML = '<li class="empty">아직 없습니다</li>';
    return;
  }
  ul.innerHTML = C.sessions.map((s) => {
    const n = s.n ? `${s.n}장 · 넣기 ${s.keep}` : "";
    const meta = [CAP_STATE[s.state] || s.state, n, s.duration ? capClock(s.duration) : null].filter(Boolean).join(" · ");
    const on = C.cur && C.cur.sid === s.sid ? " on" : "";
    return `<li class="${on}" data-sid="${esc(s.sid)}" title="${esc(s.name)}"><span class="nm">${esc(s.name)}</span>`
      + `<span class="meta">${esc(meta)}</span></li>`;
  }).join("");
}

function capRenderCount() {
  const el = $("#cap-count");
  if (!C.cur) { el.textContent = ""; return; }
  // 왼쪽 '불러온 영상' 의 지금 영상 줄도 맞춘다(상태·장수 — 목록은 뽑기가 끝날 때만 새로 받는다)
  const row = C.sessions.find((s) => s.sid === C.cur.sid);
  if (row && (row.state !== C.cur.state || row.n !== C.frames.length || row.keep !== capKept())) {
    Object.assign(row, { state: C.cur.state, n: C.frames.length, keep: capKept() });
    capRenderSessions();
  }
  const n = C.frames.length, k = capKept();
  const same = C.frames.filter((f) => f.s != null).length;
  el.innerHTML = n ? `넣기 <b>${k}</b> / ${n}장` + (same ? ` · 같은 화면 ${same}장` : "") + (capRunning() ? " · 뽑는 중…" : "")
    : capRunning() ? "뽑는 중…" : "";
}

function capRenderGrid() {
  const grid = $("#cap-grid");
  grid.innerHTML = "";
  C.cells = [];
  const c = C.cur;
  if (c && c.video && c.video.width && c.video.height) {
    grid.style.setProperty("--c-ratio", String(Math.max(0.3, Math.min(1.5, c.video.height / c.video.width))));
  }
  if (!c) {
    grid.innerHTML = '<div class="cap-empty">왼쪽 위 <b>영상 불러오기</b> 로 영상 파일을 고르거나, 이 화면에 끌어다 놓으세요.<br>'
      + '몇 초마다 한 장 뽑아 여기에 보여 주고, 넣을 것만 골라 <b>넣기</b> 로 데이터(미라벨)에 넣습니다.</div>';
    return;
  }
  if (!C.frames.length) {
    grid.innerHTML = `<div class="cap-empty">${capRunning() ? "뽑는 중… 뽑히는 대로 여기에 나옵니다"
      : "왼쪽에서 간격을 정하고 <b>캡처 시작</b> 을 누르세요"}</div>`;
    return;
  }
  capAddCells(0);
}

/** frames[from..] 의 칸을 덧붙인다(뽑는 중에는 새 것만). */
function capAddCells(from) {
  const grid = $("#cap-grid");
  if (from === 0 || !C.cells.length) {
    grid.innerHTML = "";
    C.cells = [];
    from = 0;
  }
  if (!C.frames.length) { capRenderGrid(); return; }
  const frag = document.createDocumentFragment();
  for (let i = from; i < C.frames.length; i++) {
    const el = capCell(C.frames[i]);
    C.cells[i] = el;
    frag.appendChild(el);
  }
  grid.appendChild(frag);
}

function capCell(f) {
  const el = document.createElement("div");
  el.className = "cap-cell";
  el.dataset.i = String(f.i);
  el.innerHTML = '<span class="cx">✕</span><div class="ci"><img loading="lazy" decoding="async" alt=""></div>'
    + '<div class="cm"><span class="no"></span><span class="tm"></span><span class="ch"></span></div>';
  const img = el.querySelector("img");
  img.onerror = () => { img.remove(); el.querySelector(".ci").textContent = "열 수 없음"; };
  img.src = url(`/api/capture/${C.cur.sid}/thumb/${f.i}?r=${C.cur.run}`);
  el.querySelector(".no").textContent = `#${f.i + 1}`;
  el.querySelector(".tm").textContent = capOffset(f.t);
  const ch = el.querySelector(".ch");
  if (f.s != null) {
    ch.textContent = `= #${f.s + 1}`;
    ch.classList.add("same");
    ch.title = `#${f.s + 1} 과 픽셀까지 같은 화면(멈춘 영상)이라 처음부터 뺐습니다 — 넣어도 스캔의 중복 검사가 막습니다`;
  } else if (f.c != null) {
    ch.textContent = capPct(f.c);
    ch.title = "앞 장과 밝기가 크게(20 넘게) 바뀐 픽셀 비율";
  }
  el.classList.toggle("off", !f.k);
  el.classList.toggle("focus", f.i === C.focus);
  return el;
}

function capSyncCell(i) {
  const el = C.cells[i];
  if (el) el.classList.toggle("off", !C.frames[i].k);
  if (C.pv === i) capRenderPreview();
}

function capSetFocus(i, scroll = true) {
  if (C.focus >= 0 && C.cells[C.focus]) C.cells[C.focus].classList.remove("focus");
  C.focus = i;
  if (i < 0 || !C.cells[i]) return;
  C.cells[i].classList.add("focus");
  if (scroll) C.cells[i].scrollIntoView({ block: "nearest" });
}

function capGridCols() {
  const grid = $("#cap-grid");
  const cs = getComputedStyle(grid);
  const inner = grid.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const gap = parseFloat(cs.columnGap) || 8;
  return Math.max(1, Math.floor((inner + gap) / (C.size + gap)));
}

function capMoveFocus(d) {
  if (!C.frames.length) return;
  const i = C.focus < 0 ? 0 : Math.max(0, Math.min(C.frames.length - 1, C.focus + d));
  capSetFocus(i);
}

/* ───────────────────────── 크게 보기 ───────────────────────── */

function capPreview(i) {
  if (!C.frames.length || !C.cur) return;
  C.pv = Math.max(0, Math.min(C.frames.length - 1, i));
  capSetFocus(C.pv);
  $("#cap-preview").classList.remove("hidden");
  capRenderPreview();
}

function capRenderPreview() {
  if (C.pv < 0 || !C.cur) return;
  const f = C.frames[C.pv];
  if (!f) { capClosePreview(); return; }
  const src = (j) => url(`/api/capture/${C.cur.sid}/frame/${j}?r=${C.cur.run}`);
  const img = $("#cap-pv-img");
  const want = src(f.i);
  if (img.dataset.src !== want) { img.src = want; img.dataset.src = want; }
  if (C.frames[C.pv + 1]) new Image().src = src(C.pv + 1);     // 다음 장을 미리 받아 둔다
  const st = capParseStart($("#cap-start").value);
  const parts = [`#${f.i + 1} / ${C.frames.length}`, `영상 ${capOffset(f.t)}`];
  if (st) parts.push(capWallClock(capAt(st, f.t)));
  if (f.s != null) parts.push(`#${f.s + 1} 과 같은 화면`);
  else if (f.c != null) parts.push(`바뀐 픽셀 ${capPct(f.c)}`);
  $("#cap-pv-title").innerHTML = `${esc(parts.join(" · "))} · `
    + (f.k ? '<span class="on">넣기</span>' : '<span class="off">빼기</span>');
  $("#cap-pv-toggle").textContent = f.k ? "빼기로 (Space)" : "넣기로 (Space)";
  $("#cap-preview").classList.toggle("off", !f.k);
  $("#cap-pv-prev").disabled = C.pv <= 0;
  $("#cap-pv-next").disabled = C.pv >= C.frames.length - 1;
}

function capClosePreview() {
  if (C.pv < 0) return;
  C.pv = -1;
  $("#cap-preview").classList.add("hidden");
}

/* ───────────────────────── 키 ───────────────────────── */

/** 영상 캡처의 단축키 (app.js keydown 이 넘긴다 — 입력 칸·대화상자는 거기서 먼저 걸렀다). */
function captureKey(e) {
  if (C.pv >= 0) {
    switch (e.key) {
      case "Escape": case "Enter": e.preventDefault(); capClosePreview(); break;
      case "ArrowLeft": case "ArrowUp": e.preventDefault(); capPreview(C.pv - 1); break;
      case "ArrowRight": case "ArrowDown": e.preventDefault(); capPreview(C.pv + 1); break;
      case " ": e.preventDefault(); if (!e.repeat) capSetKeep([C.pv], !C.frames[C.pv].k); break;
      default: break;
    }
    return;
  }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  switch (e.key) {
    case "ArrowRight": e.preventDefault(); capMoveFocus(1); break;
    case "ArrowLeft": e.preventDefault(); capMoveFocus(-1); break;
    case "ArrowDown": e.preventDefault(); capMoveFocus(capGridCols()); break;
    case "ArrowUp": e.preventDefault(); capMoveFocus(-capGridCols()); break;
    case "Home": e.preventDefault(); capMoveFocus(-C.frames.length); break;
    case "End": e.preventDefault(); capMoveFocus(C.frames.length); break;
    case " ":
      e.preventDefault();
      if (!e.repeat && C.focus >= 0) { capSetKeep([C.focus], !C.frames[C.focus].k); C.anchor = C.focus; }
      break;
    case "Enter": e.preventDefault(); if (C.focus >= 0) capPreview(C.focus); break;
    default: break;
  }
}

/* ───────────────────────── 배선 ───────────────────────── */

$("#cap-open").onclick = (e) => { e.currentTarget.blur(); $("#cap-file").click(); };
$("#cap-file").addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  e.target.value = "";          // 같은 파일을 다시 골라도 change 가 나게
  if (f) capUpload(f);
});
$("#cap-upload-cancel").onclick = () => { if (C.xhr) C.xhr.abort(); };
window.addEventListener("beforeunload", (e) => {
  if (C.xhr) { e.preventDefault(); e.returnValue = ""; }     // 올리는 중에 닫으면 처음부터 다시 올려야 한다
});

// 끌어다 놓기 — 영상 캡처 화면 어디에 놓아도 된다. 다른 곳에 놓친 파일을 브라우저가 열지 않게 창 전체에서 막는다
const capEl = $("#capture");
capEl.addEventListener("dragover", (e) => {
  if (VIEW !== "capture" || !e.dataTransfer || ![...e.dataTransfer.types].includes("Files")) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = "copy";
  capEl.classList.add("drag");
});
capEl.addEventListener("dragleave", (e) => {
  if (!capEl.contains(e.relatedTarget)) capEl.classList.remove("drag");
});
capEl.addEventListener("drop", (e) => {
  if (VIEW !== "capture") return;
  e.preventDefault();
  capEl.classList.remove("drag");
  const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (f) capUpload(f);
});
window.addEventListener("dragover", (e) => { if (VIEW === "capture") e.preventDefault(); });
window.addEventListener("drop", (e) => { if (VIEW === "capture") e.preventDefault(); });

for (const id of ["#cap-every", "#cap-from", "#cap-to"]) {
  $(id).addEventListener("input", () => capRenderExtract());
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) { e.target.blur(); capRun(); } });
}
$("#cap-run").onclick = (e) => { e.currentTarget.blur(); capRun(); };
$("#cap-stop").onclick = (e) => { e.currentTarget.blur(); capStop(); };

$("#cap-camera").addEventListener("change", (e) => {
  e.target.blur();
  if (!C.cur) return;
  C.cur.camera = e.target.value || null;
  capRenderCommit();
  api(`/api/capture/${C.cur.sid}/state`, { method: "PUT", body: { camera: e.target.value } }).catch(() => {});
});
$("#cap-start").addEventListener("input", capStartChanged);
$("#cap-start").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.isComposing) e.target.blur(); });
$("#cap-commit").onclick = (e) => { e.currentTarget.blur(); capCommit(); };
$("#cap-discard").onclick = (e) => { e.currentTarget.blur(); capDiscard(); };

$("#cap-sessions").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-sid]");
  if (!li || (C.cur && C.cur.sid === li.dataset.sid)) return;
  $("#cap-result").classList.add("hidden");
  capOpen(li.dataset.sid);
});

$("#cap-grid").addEventListener("click", (e) => {
  const el = e.target.closest(".cap-cell");
  if (!el) return;
  const i = Number(el.dataset.i);
  if (!C.frames[i]) return;
  if (e.shiftKey && C.anchor >= 0 && C.frames[C.anchor]) {
    capSetKeep(capRange(C.anchor, i), C.frames[C.anchor].k);     // 앞서 누른 칸과 같은 쪽으로
  } else {
    capSetKeep([i], !C.frames[i].k);
    C.anchor = i;
  }
  capSetFocus(i, false);
});
$("#cap-grid").addEventListener("dblclick", (e) => {
  const el = e.target.closest(".cap-cell");
  if (!el) return;
  const i = Number(el.dataset.i);
  // 더블클릭의 두 클릭이 넣기/빼기를 두 번 바꿔 제자리다
  capPreview(i);
});

$("#cap-all").onclick = (e) => { e.currentTarget.blur(); capSetKeep(C.frames.map((f) => f.i), true); };
$("#cap-none").onclick = (e) => { e.currentTarget.blur(); capSetKeep(C.frames.map((f) => f.i), false); };
$("#cap-sim-drop").onclick = (e) => {
  e.currentTarget.blur();
  const thr = Number($("#cap-sim").value);
  if (!(thr >= 0)) { toast("비율을 숫자로 적으세요", true); return; }
  const idx = C.frames.filter((f) => f.k && (f.s != null || (f.c != null && f.c < thr))).map((f) => f.i);
  capSetKeep(idx, false);
  toast(idx.length ? `${idx.length}장을 빼기로 했습니다 (앞 장과 바뀐 픽셀 ${thr}% 미만·같은 화면)` : "뺄 장이 없습니다");
};

$("#cap-pv-prev").onclick = (e) => { e.currentTarget.blur(); capPreview(C.pv - 1); };
$("#cap-pv-next").onclick = (e) => { e.currentTarget.blur(); capPreview(C.pv + 1); };
$("#cap-pv-toggle").onclick = (e) => { e.currentTarget.blur(); if (C.pv >= 0) capSetKeep([C.pv], !C.frames[C.pv].k); };
$("#cap-pv-close").onclick = (e) => { e.currentTarget.blur(); capClosePreview(); };
$("#cap-preview").addEventListener("click", (e) => { if (e.target.classList.contains("cap-pv-img")) capClosePreview(); });

function capApplySize(px) {
  C.size = px;
  $("#cap-grid").style.setProperty("--g-size", `${px}px`);
  $("#cap-size").value = String(px);
}
$("#cap-size").addEventListener("input", (e) => {
  capApplySize(Number(e.target.value));
  try { localStorage.setItem(CAP_SIZE_KEY, e.target.value); } catch { /* 그만 */ }
});
$("#cap-size").addEventListener("change", (e) => e.target.blur());
try {
  const px = Number(localStorage.getItem(CAP_SIZE_KEY));
  if (px >= 140 && px <= 380) capApplySize(px);
} catch { /* 기본 크기 */ }
try {
  const ev = Number(localStorage.getItem(CAP_EVERY_KEY));
  if (ev > 0) $("#cap-every").value = String(ev);
} catch { /* 기본 5초 */ }
capRenderAll();
