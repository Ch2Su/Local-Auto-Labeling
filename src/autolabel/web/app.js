/* 로컬 자동 라벨링 - 검수 UI
 *
 * 박스는 항상 YOLO 정규화 좌표(cx, cy, w, h / 0~1)로 들고 있고,
 * 화면에 그릴 때만 이미지 픽셀 -> 스크린 좌표로 변환한다.
 */

const $ = (sel) => document.querySelector(sel);

/* 프로젝트 탭 (2026-09-21). 서버가 프로젝트마다 앱을 따로 만들어 /p/<이름>/ 에 붙이므로,
 * 이 페이지의 API 는 전부 '지금 주소' 기준이다. BASE 는 끝의 파일명만 떼어낸 것이라
 * /p/visible/ 면 '/p/visible/', `--config` 로 하나만 띄웠으면 '/' 가 된다. */
const BASE = location.pathname.replace(/[^/]*$/, "");
const url = (path) => (path.startsWith("/") ? BASE + path.slice(1) : path);

const S = {
  classes: [],
  classBoxes: [],     // 클래스별 박스 수 [{total, approved, pending, ...}]
  confidentAt: 0.9,
  images: [],         // 펼쳐진 묶음들의 이미지를 이어붙인 것 (탐색 대상)
  chunks: [],         // [{start, end, open, images}]
  total: 0,           // 현재 필터의 전체 이미지 수
  cur: null,          // 현재 이미지 메타
  boxes: [],          // [{cls, cx, cy, w, h, conf, origin}]
  sel: -1,            // 선택된 박스 인덱스
  curClass: 0,        // 새 박스에 붙일 클래스
  img: null,          // HTMLImageElement
  view: { scale: 1, ox: 0, oy: 0 },
  mode: "box",        // "box" = 새 박스 / "pan" = 화면 끌기 / "sam" = SAM 클릭
  ctrl: false,        // Ctrl 을 누르고 있는 동안 두 모드가 뒤바뀐다
  sam: null,          // SAM 사용 가능 여부 {enabled, model, error}
  samBusy: false,     // SAM 요청 중 (중복 클릭 방지)
  undo: [],
  dirty: false,
  rev: 0,             // 수정할 때마다 1 씩. 저장하는 사이에 또 고쳤는지 가린다
  saving: null,       // 진행 중인 저장 요청(Promise). 저장은 한 번에 하나만 돈다
  jobPoll: null,
  ids: null,          // 'id 로 열기' 로 지정한 목록 (필터가 "ids" 일 때 탐색 대상)
  filterClasses: new Set(),   // 클래스 필터: 고른 것을 **전부** 가진 이미지만 (교집합)
  filterEmpty: false,         // 박스가 하나도 없는 이미지만. 클래스 선택과 함께 못 쓴다
  filterCamera: "",           // 카메라(출처) 필터: "" 전체 / 카메라키 / "__none__" 출처 없음
  filterDomain: "",           // 도메인 필터: "" 전체 / cctv·noncctv·drone / "__none__" 미분류
  filterSplit: "",            // 분할 필터: "" 전체 / train·val·test / "__none__" 미배정
  domainBusy: false,          // 도메인 지정이 도는 중 — 그동안의 Q·W·E(꾹 눌렀을 때 반복 입력)는 버린다
  tags: [],                   // [{id, name, n}] 이름순 (/api/tags)
  tagName: {},                // id -> 이름
  curTag: null,               // T 로 붙일 태그 id (오른쪽 태그 목록에서 줄을 눌러 고른다)
  filterTag: "",              // 태그 필터: "" 전체 / 태그 id / "__none__" 태그 없음
  tagBusy: false,             // T 가 도는 중 — 꾹 눌렀을 때 반복 입력이 겹치지 않게
  listBusy: false,            // '새로 받기' 가 도는 중 — 그동안의 Q·W·E·T 는 버린다(다음 줄을 엉뚱하게 고른다)
  loadedCount: 0,     // 이 이미지를 열었을 때의 박스 수. 저장 뒤 이보다 적으면 표시한다
  histOpen: false,    // 오른쪽 '변경 내역' 이 펼쳐져 있나
  history: [],        // 이 이미지의 변경 내역 (최근 것부터)
  ghost: null,        // 캔버스에 흰 점선으로 겹쳐 그릴 박스들 (변경 내역 미리 보기)
  predOn: false,      // 예측 비교(P) 켜짐
  pred: null,         // 지금 이미지의 예측 {weights, device, ms, model_classes, preds}
  predLoading: false,
  findings: [],       // 라벨과 예측이 어긋나는 곳 (computeFindings)
  predIgnore: new Set(),  // 이 이미지에서 '무시' 한 항목의 키
  findHot: -1,        // 목록에서 마우스를 올린 항목 (캔버스에서 굵게)
};

// 지금 보기: "label" 라벨링 / "gallery" 이미지 보기(gallery.js). setView 로만 바꾼다 (2026-11-04)
let VIEW = "label";

const canvas = $("#canvas");
const ctx = canvas.getContext("2d");

/* ───────────────────────────── 유틸 ───────────────────────────── */

function classColor(i) {
  return `hsl(${(i * 137.508) % 360} 75% 58%)`;
}

function toast(msg, isErr = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("err", isErr);
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), isErr ? 6000 : 2500);
}

async function api(path, opts = {}) {
  const res = await fetch(url(path), {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  return data;
}


/* ───────────────────── 좌표 변환 (이미지px <-> 스크린px) ───────────────────── */

function toScreen(ix, iy) {
  return [ix * S.view.scale + S.view.ox, iy * S.view.scale + S.view.oy];
}
function toImage(sx, sy) {
  return [(sx - S.view.ox) / S.view.scale, (sy - S.view.oy) / S.view.scale];
}

/** 정규화 박스를 이미지 픽셀 사각형으로. */
function boxRect(b) {
  const W = S.cur.width, H = S.cur.height;
  return [(b.cx - b.w / 2) * W, (b.cy - b.h / 2) * H, b.w * W, b.h * H];
}

/** 이미지 픽셀 사각형을 이미지 경계 안으로 자른다. 비면 w 나 h 가 0 이 된다. */
function clipRect(x, y, w, h) {
  const W = S.cur.width, H = S.cur.height;
  const x1 = Math.max(0, Math.min(W, x));
  const y1 = Math.max(0, Math.min(H, y));
  const x2 = Math.max(0, Math.min(W, x + w));
  const y2 = Math.max(0, Math.min(H, y + h));
  return [x1, y1, Math.max(0, x2 - x1), Math.max(0, y2 - y1)];
}

/**
 * 이미지 픽셀 사각형을 정규화 박스 필드로. 이미지 밖은 잘라낸다.
 *
 * 서버의 `Box.clamped()` 와 **같은 방식(xyxy 로 펴서 자르기)** 이어야 한다.
 * 예전에는 cx/cy/w/h 를 각각 clamp01 했는데, 그건 클리핑이 아니다.
 * x -500~-300 (이미지 밖) 을 넣으면 cx=clamp01(-0.4)=0, w=0.2 가 되어
 * '이미지 안에 걸친 박스'로 둔갑하고, 서버가 그걸 0~0.1 로 잘라
 * **가장자리에 달라붙은 유령 박스**를 저장했다. 원래 값이 그대로 갔다면
 * `clamped()` 가 w=0 을 만들고 `is_valid()` 가 버렸을 것이다.
 */
function rectToBox(x, y, w, h) {
  const W = S.cur.width, H = S.cur.height;
  const [rx, ry, rw, rh] = clipRect(x, y, w, h);
  return { cx: (rx + rw / 2) / W, cy: (ry + rh / 2) / H, w: rw / W, h: rh / H };
}

/** 이동용. 크기를 유지한 채 이미지 안으로 밀어 넣는다 (잘라서 줄이지 않는다). */
function rectToBoxInside(x, y, w, h) {
  const W = S.cur.width, H = S.cur.height;
  return rectToBox(
    Math.max(0, Math.min(W - Math.min(w, W), x)),
    Math.max(0, Math.min(H - Math.min(h, H), y)),
    w, h,
  );
}

function fitView() {
  if (!S.cur) return;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (cw < 2 || ch < 2) {
    // 이미지 onload가 레이아웃보다 먼저 끝나면 캔버스 크기가 아직 0이다.
    // ResizeObserver가 크기를 잡아준 뒤 다시 맞춘다.
    S.pendingFit = true;
    return;
  }
  S.pendingFit = false;
  const scale = Math.min(cw / S.cur.width, ch / S.cur.height) * 0.98;
  S.view.scale = scale;
  S.view.ox = (cw - S.cur.width * scale) / 2;
  S.view.oy = (ch - S.cur.height * scale) / 2;
  draw();
}

/* ───────────────────────────── 렌더링 ───────────────────────────── */

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * dpr);
  canvas.height = Math.round(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (S.cur && (S.pendingFit || !S.view.scale)) fitView();
  else draw();
}

const HANDLE = 8;

function handlePositions(b) {
  const [x, y, w, h] = boxRect(b);
  const pts = [
    ["nw", x, y], ["n", x + w / 2, y], ["ne", x + w, y],
    ["e", x + w, y + h / 2], ["se", x + w, y + h],
    ["s", x + w / 2, y + h], ["sw", x, y + h], ["w", x, y + h / 2],
  ];
  return pts.map(([k, ix, iy]) => [k, ...toScreen(ix, iy)]);
}

function draw() {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  ctx.clearRect(0, 0, cw, ch);
  if (!S.cur || !S.img) return;

  const [ox, oy] = [S.view.ox, S.view.oy];
  const sw = S.cur.width * S.view.scale;
  const sh = S.cur.height * S.view.scale;
  ctx.drawImage(S.img, ox, oy, sw, sh);

  S.boxes.forEach((b, i) => {
    const [ix, iy, iw, ih] = boxRect(b);
    const [x, y] = toScreen(ix, iy);
    const w = iw * S.view.scale, h = ih * S.view.scale;
    const color = classColor(b.cls);
    const selected = i === S.sel;

    ctx.lineWidth = selected ? 2.5 : 1.5;
    ctx.strokeStyle = color;
    // 자동 라벨 중 신뢰도가 낮은 것은 점선으로 눈에 띄게 한다.
    ctx.setLineDash(b.conf != null && b.conf < 0.5 ? [6, 4] : []);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);

    if (selected) {
      ctx.fillStyle = color.replace(")", " / 0.12)");
      ctx.globalAlpha = 0.15;
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1;
    }

    // 라벨 칩
    const name = S.classes[b.cls] ?? `cls_${b.cls}`;
    const text = b.conf != null ? `${name} ${b.conf.toFixed(2)}` : name;
    ctx.font = "11px 'Malgun Gothic', Consolas, monospace";
    const tw = ctx.measureText(text).width + 8;
    const ty = y - 15 < 0 ? y + 1 : y - 15;
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw, 14);
    ctx.fillStyle = "#14161a";
    ctx.fillText(text, x + 4, ty + 11);

    if (selected) {
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "#14161a";
      ctx.lineWidth = 1;
      for (const [, hx, hy] of handlePositions(b)) {
        ctx.fillRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
        ctx.strokeRect(hx - HANDLE / 2, hy - HANDLE / 2, HANDLE, HANDLE);
      }
    }
  });

  if (S.predOn) drawFindings();

  // 변경 내역에 마우스를 올린 동안: 그 저장 직전의 박스를 흰 점선으로 겹쳐 그린다.
  if (S.ghost) {
    ctx.save();
    ctx.setLineDash([5, 3]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
    ctx.font = "11px 'Malgun Gothic', Consolas, monospace";
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    for (const b of S.ghost) {
      const [ix, iy, iw, ih] = boxRect(b);
      const [x, y] = toScreen(ix, iy);
      ctx.strokeRect(x, y, iw * S.view.scale, ih * S.view.scale);
      ctx.fillText(clsName(b.cls), x + 2, y + ih * S.view.scale - 3);
    }
    ctx.restore();
  }
}

const clsName = (i) => S.classes[i] ?? `cls_${i}`;

/* ───────────────────────────── 마우스 ───────────────────────────── */

let drag = null;

function mousePos(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

const MODE_LABEL = { box: "새 박스", pan: "화면 끌기", sam: "SAM 클릭" };

/** Ctrl 을 누르고 있으면 화면 끌기와 현재 편집 모드가 뒤바뀐다. */
function effectiveMode() {
  if (!S.ctrl) return S.mode;
  return S.mode === "pan" ? "box" : "pan";
}

function setMode(mode) {
  if (!MODE_LABEL[mode]) return;
  if (mode === "sam" && S.sam && S.sam.enabled === false) {
    toast("SAM 이 꺼져 있습니다 (config.yaml 의 sam.enabled)", true);
    return;
  }
  S.mode = mode;
  renderMode();
}

function renderMode() {
  const eff = effectiveMode();
  canvas.style.cursor =
    eff === "pan" ? (drag && drag.mode === "pan" ? "grabbing" : "grab")
    : S.samBusy ? "progress"
    : "crosshair";

  const el = $("#mode-badge");
  if (!el) return;
  el.textContent = S.samBusy && eff === "sam" ? "SAM…" : MODE_LABEL[eff];
  el.className = "mode-badge " + eff + (S.ctrl ? " temp" : "");
  el.title = S.ctrl
    ? "Ctrl 을 누르고 있는 동안 임시로 바뀐 상태입니다"
    : "B = 새 박스, C = SAM 클릭, D = 화면 끌기 (Ctrl 을 누르면 잠시 화면 끌기로)";
}

/* ───────────────────────────── SAM ───────────────────────────── */

async function loadSamStatus() {
  try {
    S.sam = await api("/api/sam");
  } catch {
    S.sam = { enabled: false, error: "상태를 확인할 수 없습니다" };
  }
  const btn = $("#btn-sam");
  if (btn) btn.disabled = !S.sam.enabled;
}

/**
 * SAM 에게 박스 하나를 받아 추가한다.
 * point 만 주면 클릭 한 번으로, box 를 주면 그 범위를 다듬는다.
 * replace >= 0 이면 그 인덱스의 박스를 갈아끼운다.
 */
async function samBox({ point, box, replace = -1 }) {
  if (!S.cur || S.samBusy) return;
  S.samBusy = true;
  renderMode();
  // 응답을 기다리는 사이에 다른 이미지로 넘어가거나 박스 목록이 바뀔 수 있다. 요청한
  // 이미지와 박스를 붙잡아 두고, 달라졌으면 결과를 버린다 — 예전에는 넘어간 이미지에
  // SAM 박스가 붙어 그 이미지에 저장됐고, 다듬기는 번호가 밀린 엉뚱한 박스를 갈아끼웠다.
  const id = S.cur.id;
  const target = replace >= 0 ? S.boxes[replace] : null;
  try {
    const body = { cls: target ? target.cls : S.curClass };
    if (point) body.point = point;
    if (box) body.box = box;
    const r = await api(`/api/images/${id}/segment`, { method: "POST", body });
    if (!S.cur || S.cur.id !== id) return;
    const at = target ? S.boxes.indexOf(target) : -1;
    if (target && at < 0) { toast("다듬는 사이에 박스가 바뀌어 SAM 결과를 버렸습니다"); return; }
    pushUndo();
    if (target) {
      S.boxes[at] = { ...r.box, cls: target.cls };
      selectBox(at);
    } else {
      S.boxes.push(r.box);
      selectBox(S.boxes.length - 1);
    }
    markDirty();
    refresh();
  } catch (err) {
    toast(`SAM 실패: ${err.message}`, true);
  } finally {
    S.samBusy = false;
    renderMode();
  }
}

/** 선택한 박스를 SAM 으로 다듬는다. */
function samRefineSelected() {
  if (S.sel < 0) { toast("다듬을 박스를 먼저 고르세요"); return; }
  const b = S.boxes[S.sel];
  samBox({
    box: [b.cx - b.w / 2, b.cy - b.h / 2, b.cx + b.w / 2, b.cy + b.h / 2],
    replace: S.sel,
  });
}

function hitHandle(sx, sy) {
  if (S.sel < 0) return null;
  for (const [k, hx, hy] of handlePositions(S.boxes[S.sel])) {
    if (Math.abs(sx - hx) <= HANDLE && Math.abs(sy - hy) <= HANDLE) return k;
  }
  return null;
}

function hitBox(sx, sy) {
  const [ix, iy] = toImage(sx, sy);
  let best = -1, bestArea = Infinity;
  S.boxes.forEach((b, i) => {
    const [x, y, w, h] = boxRect(b);
    if (ix >= x && ix <= x + w && iy >= y && iy <= y + h && w * h < bestArea) {
      best = i; bestArea = w * h;
    }
  });
  return best;
}

canvas.addEventListener("mousedown", (e) => {
  if (!S.cur) return;
  const [sx, sy] = mousePos(e);

  // 가운데 버튼은 모드와 무관하게 항상 화면 끌기
  if (e.button === 1) {
    drag = { mode: "pan", sx, sy, ox: S.view.ox, oy: S.view.oy, moved: 0 };
    renderMode();
    return;
  }
  if (e.button !== 0) return;

  if (effectiveMode() === "pan") {
    // 끌기 모드에서는 어디를 잡아도 화면이 움직인다.
    // 끌지 않고 클릭만 하면 박스 선택으로 쳐준다(보면서 고르기 위해).
    drag = { mode: "pan", sx, sy, ox: S.view.ox, oy: S.view.oy, moved: 0 };
    renderMode();
    return;
  }

  // SAM 모드에서는 모든 좌클릭이 프롬프트다. 기존 박스 위를 클릭했다고 선택·이동으로
  // 새면, 객체가 빽빽한 장면에서는 대부분의 클릭이 SAM 에 닿지 않는다.
  // 선택·이동이 필요하면 B(새 박스) 모드나 오른쪽 박스 목록을 쓴다.
  if (effectiveMode() === "sam") {
    selectBox(-1);
    const [ix, iy] = toImage(sx, sy);
    drag = { mode: "new", ix, iy, sam: true };
    return;
  }

  const handle = hitHandle(sx, sy);
  if (handle) {
    pushUndo();
    drag = { mode: "resize", handle, start: boxRect(S.boxes[S.sel]), sx, sy };
    return;
  }

  const hit = hitBox(sx, sy);
  if (hit >= 0) {
    selectBox(hit);
    pushUndo();
    const [ix, iy] = toImage(sx, sy);
    const [x, y] = boxRect(S.boxes[hit]);
    drag = { mode: "move", dx: ix - x, dy: iy - y, sx, sy };
    return;
  }

  selectBox(-1);
  const [ix, iy] = toImage(sx, sy);
  drag = { mode: "new", ix, iy };
});

window.addEventListener("mousemove", (e) => {
  if (!drag || !S.cur) return;
  const [sx, sy] = mousePos(e);
  const [ix, iy] = toImage(sx, sy);

  if (drag.mode === "pan") {
    drag.moved = Math.max(drag.moved || 0, Math.abs(sx - drag.sx) + Math.abs(sy - drag.sy));
    S.view.ox = drag.ox + (sx - drag.sx);
    S.view.oy = drag.oy + (sy - drag.sy);
    draw();
    return;
  }

  if (drag.mode === "new") {
    drag.cur = [ix, iy];
    draw();
    // 그리는 중인 사각형 미리보기
    const [x1, y1] = toScreen(Math.min(drag.ix, ix), Math.min(drag.iy, iy));
    const [x2, y2] = toScreen(Math.max(drag.ix, ix), Math.max(drag.iy, iy));
    ctx.strokeStyle = classColor(S.curClass);
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    return;
  }

  // 박스를 고르려고 누른 손이 1~2px 떨린 것은 이동·크기 조절로 치지 않는다.
  if ((drag.mode === "move" || drag.mode === "resize") && !drag.changed) {
    if (Math.abs(sx - drag.sx) + Math.abs(sy - drag.sy) < 3) return;
    drag.changed = true;
  }

  if (drag.mode === "move") {
    const b = S.boxes[S.sel];
    const [, , w, h] = boxRect(b);
    // 이동은 크기를 지킨다. 잘라내면 이미지 밖으로 끌었을 때 박스가 쪼그라든다.
    Object.assign(b, rectToBoxInside(ix - drag.dx, iy - drag.dy, w, h));
    markDirty();
    draw();
    return;
  }

  if (drag.mode === "resize") {
    let [x, y, w, h] = drag.start;
    let x2 = x + w, y2 = y + h;
    if (drag.handle.includes("n")) y = iy;
    if (drag.handle.includes("s")) y2 = iy;
    if (drag.handle.includes("w")) x = ix;
    if (drag.handle.includes("e")) x2 = ix;
    Object.assign(
      S.boxes[S.sel],
      rectToBox(Math.min(x, x2), Math.min(y, y2), Math.abs(x2 - x), Math.abs(y2 - y))
    );
    markDirty();
    draw();
  }
});

window.addEventListener("mouseup", (e) => {
  if (!drag) return;

  // 끌기 모드에서 거의 움직이지 않았으면 '클릭'으로 보고 박스를 선택한다.
  if (drag.mode === "pan") {
    if ((drag.moved || 0) < 4) {
      const [sx, sy] = mousePos(e);
      selectBox(hitBox(sx, sy));
    }
    drag = null;
    renderMode();
    draw();
    return;
  }

  if (drag.mode === "new") {
    const W = S.cur.width, H = S.cur.height;
    const big = drag.cur
      && (Math.abs(drag.cur[0] - drag.ix) * S.view.scale > 4)
      && (Math.abs(drag.cur[1] - drag.iy) * S.view.scale > 4);

    if (drag.sam) {
      // SAM 모드: 끌었으면 그 범위를, 그냥 클릭했으면 그 점을 프롬프트로 보낸다.
      // 이미지 밖은 새 박스와 마찬가지로 받지 않는다 — 가장자리로 끌어다 붙이면
      // 사용자가 찍지도 않은 곳에 박스가 생긴다.
      const d = drag;
      drag = null;
      if (big) {
        const x1 = Math.min(d.ix, d.cur[0]), y1 = Math.min(d.iy, d.cur[1]);
        const [rx, ry, rw, rh] = clipRect(x1, y1,
          Math.abs(d.cur[0] - d.ix), Math.abs(d.cur[1] - d.iy));
        if (rw > 0 && rh > 0) {
          samBox({ box: [rx / W, ry / H, (rx + rw) / W, (ry + rh) / H] });
        } else {
          toast("이미지 바깥에는 박스를 만들 수 없습니다");
        }
      } else if (d.ix >= 0 && d.ix <= W && d.iy >= 0 && d.iy <= H) {
        samBox({ point: [d.ix / W, d.iy / H] });
      } else {
        toast("이미지 안을 클릭하세요");
      }
      draw();
      return;
    }

    // 클릭 실수로 점 크기 박스가 생기지 않도록 최소 크기를 둔다.
    if (big) {
      const [x1, y1] = [Math.min(drag.ix, drag.cur[0]), Math.min(drag.iy, drag.cur[1])];
      const [x2, y2] = [Math.max(drag.ix, drag.cur[0]), Math.max(drag.iy, drag.cur[1])];
      const nb = rectToBox(x1, y1, x2 - x1, y2 - y1);
      // 이미지 밖에서만 그렸으면 잘라낸 결과가 비어 있다. 그때는 만들지 않는다.
      if (nb.w > 0 && nb.h > 0) {
        pushUndo();
        S.boxes.push({ cls: S.curClass, ...nb, conf: null, origin: "human" });
        selectBox(S.boxes.length - 1);
        markDirty();
      } else {
        toast("이미지 바깥에는 박스를 만들 수 없습니다");
      }
    }
  }
  if (drag.mode === "move" || drag.mode === "resize") {
    if (drag.changed) {
      // 사람이 손댄 박스는 더 이상 모델 예측이 아니다. 끄는 도중에 자동 저장이 한 번
      // 돌았을 수 있으므로 이 변경도 저장 대상으로 올린다.
      const b = S.boxes[S.sel];
      if (b) { b.origin = "human"; b.conf = null; }
      markDirty();
    } else {
      // 클릭해서 고르기만 했다. 누를 때 쌓은 되돌리기를 거둔다. 예전에는 고르기만 해도
      // 자동 라벨이 '수동' 으로 바뀌고 신뢰도가 지워져, 다음 저장 때 그대로 DB 로 갔다.
      S.undo.pop();
    }
    renderBoxList();      // 옮긴 박스로 예측 비교도 다시 맞춘다
  }
  drag = null;
  draw();
});

canvas.addEventListener("wheel", (e) => {
  if (!S.cur) return;
  e.preventDefault();
  // 가로 스크롤(틸트 휠·트랙패드 좌우)은 무시한다. deltaY 가 0 이라 아래 삼항식이
  // '축소' 쪽으로 떨어져서, 옆으로 굴릴 때마다 이미지가 작아졌다.
  // preventDefault 는 이미 했으므로 브라우저의 좌우 스크롤·뒤로가기 제스처도 안 먹는다.
  if (!e.deltaY) return;
  const [sx, sy] = mousePos(e);
  const [ix, iy] = toImage(sx, sy);
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  S.view.scale = Math.min(40, Math.max(0.05, S.view.scale * factor));
  // 커서 아래 지점이 그대로 있도록 오프셋 보정
  S.view.ox = sx - ix * S.view.scale;
  S.view.oy = sy - iy * S.view.scale;
  draw();
}, { passive: false });

canvas.addEventListener("contextmenu", (e) => {
  // 우클릭으로 박스 삭제
  e.preventDefault();
  const [sx, sy] = mousePos(e);
  const hit = hitBox(sx, sy);
  if (hit >= 0) { pushUndo(); S.boxes.splice(hit, 1); selectBox(-1); markDirty(); refresh(); }
});

/* ───────────────────────── 편집 상태 ───────────────────────── */

function pushUndo() {
  S.undo.push(JSON.stringify(S.boxes));
  if (S.undo.length > 60) S.undo.shift();
}

function undo() {
  const prev = S.undo.pop();
  if (!prev) return;
  S.boxes = JSON.parse(prev);
  S.sel = -1;
  markDirty();
  refresh();
}

function selectBox(i) {
  S.sel = i;
  if (i >= 0) S.curClass = S.boxes[i].cls;
  renderBoxList();
  renderClassList();
  draw();
}

function markDirty() {
  S.dirty = true;
  S.rev += 1;
  const el = $("#save-state");
  el.textContent = "수정됨";
  el.className = "save-state dirty";
  clearTimeout(S._saveTimer);
  S._saveTimer = setTimeout(saveBoxes, 600);
}

/**
 * 지금 박스를 저장한다. **저장은 한 번에 하나만 돌고**, 도는 동안 생긴 수정은 그 저장이
 * 끝난 뒤 이어서 저장한다.
 *
 * 예전에는 저장 중에 들어온 요청을 그냥 버리고(`if (S.saving) return`), 저장이 끝나면
 * 수정 표시를 무조건 내렸다. 그래서 저장이 도는 동안 한 수정(되돌리기 포함)은 '저장됨'
 * 이 뜬 채로 사라졌고, 그 틈에 다음 이미지로 넘어가면 flushSave 도 그냥 지나쳤다.
 * 저장을 늦춘 서버로 재현했다 — 클래스를 두 번 바꾸고 바로 넘기면 첫 값만 DB 에 남았다
 * (2026-09-23). 09-21 에 박스 2개가 경고 없이 사라진 것도 이 경로였을 수 있다
 * (실수로 지우고 곧바로 Ctrl+Z 를 누르면 되돌리기만 사라진다).
 */
async function saveBoxes() {
  while (S.saving) await S.saving.catch(() => {});
  if (!S.cur || !S.dirty) return;
  const id = S.cur.id, rev = S.rev;
  const payload = S.boxes.map((b) => ({
    cls: b.cls, cx: b.cx, cy: b.cy, w: b.w, h: b.h,
    conf: b.conf, origin: b.origin || "human",
  }));
  S.saving = api(`/api/images/${id}/boxes`, { method: "PUT", body: { boxes: payload } });
  try {
    const res = await S.saving;
    // 저장하는 사이에 또 고쳤으면 수정 표시를 그대로 둔다 — 걸어 둔 타이머나 flushSave 가
    // 다시 저장한다. 이미지가 바뀐 뒤라면 그 화면의 표시를 건드리지 않는다.
    if (S.rev === rev && S.cur && S.cur.id === id) {
      S.dirty = false;
      showSaved(res.n_boxes);
    }
    if (S.cur && S.cur.id === id) {
      setHistCount(res.n_history);
      if (S.histOpen) loadHistory();
    }
    // 박스를 고쳤으면 클래스별 집계도 달라진다.
    loadState().catch(() => {});
  } catch (err) {
    toast(`저장 실패: ${err.message}`, true);
  } finally {
    S.saving = null;
  }
}

/**
 * '저장됨' 표시. 이 이미지를 연 뒤로 박스가 줄었으면 주황으로 '박스 5→4' 를 붙인다.
 * 지우려고 지운 것이면 그냥 넘어가면 되고, 아니면 '변경 내역' 에서 되돌린다 —
 * 09-21 에 박스가 경고 없이 사라진 뒤 넣었다.
 */
function showSaved(n) {
  const el = $("#save-state");
  const shrunk = n < S.loadedCount;
  el.textContent = shrunk ? `저장됨 · 박스 ${S.loadedCount}→${n}` : "저장됨";
  el.className = "save-state saved" + (shrunk ? " shrunk" : "");
  el.title = shrunk
    ? "이 이미지를 연 뒤로 박스가 줄었습니다. 지우려던 게 아니면 오른쪽 '변경 내역'(H)에서 되돌리세요."
    : "";
}

/** 남은 수정을 전부 저장한다. 저장에 실패해 수정이 남아 있으면 false. */
async function flushSave() {
  clearTimeout(S._saveTimer);
  for (;;) {
    const rev = S.rev;
    await saveBoxes();                 // 도는 저장이 있으면 끝나기를 기다린 뒤 남은 것을 저장
    if (!S.dirty) return true;
    if (S.rev === rev) return false;   // 그 사이 새 수정이 없는데도 남았다 = 저장 실패
  }
}

/* ───────────────────────── 목록 렌더링 ───────────────────────── */

function renderClassList() {
  const ul = $("#class-list");
  ul.innerHTML = "";
  const totals = S.classes.map((_, i) => (S.classBoxes[i] || {}).total || 0);
  const max = Math.max(1, ...totals);

  // 지금 보고 있는 이미지의 클래스별 개수
  const here = S.classes.map(() => 0);
  S.boxes.forEach((b) => { if (b.cls < here.length) here[b.cls] += 1; });

  S.classes.forEach((name, i) => {
    const li = document.createElement("li");
    li.className = i === S.curClass ? "active" : "";
    const n = totals[i];
    const info = S.classBoxes[i] || {};
    // 막대는 가장 많은 클래스를 100%로 본 상대 비율. 불균형이 한눈에 보인다.
    li.style.setProperty("--bar", `${(n / max) * 100}%`);
    li.style.setProperty("--barcolor", classColor(i));
    li.innerHTML =
      `<span class="swatch" style="background:${classColor(i)}"></span>` +
      `<span class="nm"></span>` +
      `<span class="here${here[i] ? "" : " zero"}">${here[i] || "·"}</span>` +
      `<span class="cnt">${n.toLocaleString()}</span>` +
      `<span class="idx">${i < 9 ? i + 1 : ""}</span>`;
    li.querySelector(".nm").textContent = name;

    const STATUS_KO = {
      approved: "승인", pending: "검수 대기",
      skipped: "건너뜀", unlabeled: "미라벨",
    };
    const breakdown = Object.keys(STATUS_KO)
      .filter((s) => info[s])
      .map((s) => `  ${STATUS_KO[s]} ${info[s].toLocaleString()}`);
    li.title = [
      `${name}`,
      `이 이미지: ${here[i]}개`,
      `전체: ${n.toLocaleString()}개`,
      ...breakdown,
    ].join("\n");

    li.onclick = () => setClass(i);
    ul.appendChild(li);
  });
}

function confClass(b) {
  if (b.conf == null) return "human";
  if (b.conf >= S.confidentAt) return "high";
  if (b.conf >= 0.5) return "mid";
  return "low";
}

/** 라벨끼리 사실상 같은 자리(IoU ≥ 0.85)인 쌍 — 같은 물체를 두 번 그렸거나, 클래스를 바꾸려다
 *  새로 하나 더 그린 흔적이다. 모델은 예측과만 견주므로 이것은 따로 본다 (2026-10-25).
 *  오토바이와 탄 사람(person·motorcycle) 처럼 진짜로 겹치는 물체도 0.85 를 넘을 수 있다. */
const SELF_OVERLAP = 0.85;
function selfOverlaps(boxes = S.boxes) {
  const out = [];
  for (let i = 0; i < boxes.length; i++)
    for (let j = i + 1; j < boxes.length; j++) {
      const v = iouN(boxes[i], boxes[j]);
      if (v >= SELF_OVERLAP) out.push([i, j, v]);
    }
  return out;
}

function renderBoxList() {
  const ul = $("#box-list");
  ul.innerHTML = "";
  const overlaps = selfOverlaps();
  $("#box-count").textContent = (S.boxes.length ? `${S.boxes.length}개` : "")
    + (overlaps.length ? ` · 겹침 ${overlaps.length}쌍` : "");
  $("#btn-same").disabled = S.sel < 0;
  // 박스 목록이 바뀌는 곳(추가·삭제·클래스 변경·되돌리기)은 전부 여기를 지난다.
  if (S.predOn) updateFindings();
  const partner = new Map();      // 박스 번호 -> [짝 번호, IoU] (IoU 가 가장 큰 짝 하나)
  for (const [i, j, v] of overlaps) {
    if (!partner.has(i) || partner.get(i)[1] < v) partner.set(i, [j, v]);
    if (!partner.has(j) || partner.get(j)[1] < v) partner.set(j, [i, v]);
  }
  S.boxes.forEach((b, i) => {
    const li = document.createElement("li");
    li.className = i === S.sel ? "active" : "";
    li.innerHTML =
      `<span class="swatch" style="background:${classColor(b.cls)}"></span>` +
      `<span class="nm"></span>` +
      (partner.has(i) ? `<span class="dup"></span>` : "") +
      `<span class="conf ${confClass(b)}">${b.conf == null ? "수동" : b.conf.toFixed(2)}</span>` +
      `<span class="del" title="삭제">✕</span>`;
    li.querySelector(".nm").textContent = S.classes[b.cls] ?? `cls_${b.cls}`;
    if (partner.has(i)) {
      const [j, v] = partner.get(i);
      const el = li.querySelector(".dup");
      el.textContent = `⚠ 겹침 ${v.toFixed(2)}`;
      el.title = `${clsName(S.boxes[j].cls)} 박스와 거의 같은 자리입니다 (IoU ${v.toFixed(3)}). 같은 물체를 두 번 그렸는지 보세요.`;
    }
    li.onclick = (e) => {
      if (e.target.classList.contains("del")) {
        pushUndo(); S.boxes.splice(i, 1); S.sel = -1; markDirty(); refresh();
      } else {
        selectBox(i);
      }
    };
    ul.appendChild(li);
  });
}

/** pos = 지금 필터·정렬에서 몇 번째인지(1부터). 묶음을 나눠도 이어서 센다. */
function itemRow(im, pos) {
  const li = document.createElement("li");
  li.className = "item" + (S.cur && im.id === S.cur.id ? " active" : "");
  li.dataset.id = im.id;
  const hue = 120 - Math.round(im.uncertainty * 120); // 초록(확실) -> 빨강(애매)
  li.innerHTML =
    `<span class="unc" style="background:hsl(${hue} 70% 50%)" ` +
    `title="불확실도 ${im.uncertainty}"></span>` +
    `<span class="pos" title="지금 필터·정렬에서 ${pos}번째">${pos}</span>` +
    `<span class="nm"></span>`
    + (im.rare_cls != null ? '<span class="rare"></span>' : "")
    + (im.domain ? `<span class="dom dom-${im.domain}">${DOMAIN_NAME[im.domain] || im.domain}</span>` : "")
    + (im.split === "test" || im.split === "val" ? `<span class="spl spl-${im.split}">${im.split}</span>` : "")
    + ((im.tags || []).length ? '<span class="tg"></span>' : "")
    + `<span class="nb">${im.n_boxes}</span>`;
  li.querySelector(".nm").textContent = im.name;
  if ((im.tags || []).length) {   // 첫 태그 이름 + 나머지 개수. 전부는 마우스를 올리면
    const names = im.tags.map((t) => S.tagName[t] || `#${t}`);
    const tg = li.querySelector(".tg");
    tg.textContent = `#${names[0]}${names.length > 1 ? ` +${names.length - 1}` : ""}`;
    tg.title = names.map((n) => `#${n}`).join(" ");
  }
  // '드문 클래스 먼저' 정렬: 이 이미지를 앞으로 끌어온 클래스를 붙인다
  if (im.rare_cls != null) {
    const tag = li.querySelector(".rare");
    tag.textContent = clsName(im.rare_cls);
    tag.style.color = classColor(im.rare_cls);
    tag.title = `${clsName(im.rare_cls)} — 승인 데이터에서 이 클래스가 나온 장면 묶음 ${im.rare_groups}개`
      + ` · 점수 ${im.rare_score} (신뢰도 × 희소도)`;
  }
  li.onclick = () => loadImage(im.id);
  return li;
}

function renderImageList() {
  const ul = $("#image-list");
  ul.innerHTML = "";

  if (!S.total) {
    const li = document.createElement("li");
    li.className = "item";
    li.innerHTML = `<span class="nm dim">해당 상태의 이미지가 없습니다</span>`;
    ul.appendChild(li);
    return;
  }

  S.chunks.forEach((ch, idx) => {
    const head = document.createElement("li");
    const hasCurrent = S.cur && ch.images.some((im) => im.id === S.cur.id);
    head.className = "group" + (ch.open ? " open" : "") + (hasCurrent ? " has-current" : "");
    head.innerHTML =
      `<span class="caret">${ch.open ? "▾" : "▸"}</span>` +
      `<span class="rng">${ch.start + 1} – ${ch.end}</span>` +
      `<span class="cnt">${ch.end - ch.start}장</span>`;
    head.onclick = () => toggleChunk(idx);
    ul.appendChild(head);

    if (ch.open) ch.images.forEach((im, i) => ul.appendChild(itemRow(im, ch.start + i + 1)));
  });
}

/** 이미지를 바꿀 때: 목록을 다시 그리지 않고 지금 이미지 표시만 옮긴 뒤, 그 줄이 목록 밖에
 * 있으면 보이는 곳까지 굴린다. '전체' 로 6천 줄을 펼치면 다시 그리기가 한 번에 0.1초쯤 걸려
 * 화살표를 누를 때마다 끊겼다(2026-11-03). 이미지 항목(도메인·태그)을 바꿨으면 부르는 쪽이
 * 그 줄만 refreshRows 로 고친다. 묶음을 펼칠 때는 굴리지 않는다 — 손으로 내려 본 자리에서
 * 튕겨 나간다. */
function markCurrent() {
  const ul = $("#image-list");
  const id = S.cur ? S.cur.id : null;
  ul.querySelectorAll("li.item.active").forEach((li) => li.classList.remove("active"));
  const li = id == null ? null : ul.querySelector(`li.item[data-id="${id}"]`);
  if (li) li.classList.add("active");
  ul.querySelectorAll("li.group").forEach((head, i) => {
    const ch = S.chunks[i];
    head.classList.toggle("has-current", !!ch && ch.images.some((im) => im.id === id));
  });
  if (li) li.scrollIntoView({ block: "nearest" });
}

/** fn 으로 목록을 다시 받아도 지금 이미지 줄이 화면의 같은 높이에 남게 굴린다. 지금 이미지가
 * 목록에 없었으면 아무것도 하지 않는다.
 * next 를 주면(다시 받은 뒤 곧 그 줄로 넘어간다): 지금 줄이 그 줄 바로 위에 그대로 남지 않았으면
 * — 빠졌거나(승인해서 '검수 대기' 에서) 옮겨 갔으면('최근 수정순' 맨 위로) — 대신 그 줄을 지금 줄이
 * 있던 높이에 둔다. 아래 줄이 한 칸 올라오고 파란 줄 자리는 그대로다. */
function keepCurrentRow(fn, next = null) {
  return keepRowInPlace(S.cur ? S.cur.id : null, fn, next);
}

/** keepCurrentRow 와 같은데 지금 이미지 대신 id 의 줄을 제자리에 둔다. */
async function keepRowInPlace(id, fn, next = null) {
  const ul = $("#image-list");
  const top = (x) => {
    const li = ul.querySelector(`li.item[data-id="${x}"]`);
    return li ? li.getBoundingClientRect().top - ul.getBoundingClientRect().top : null;
  };
  const y = id == null ? null : top(id);
  const result = await fn();
  if (y == null) return result;
  let anchor = id;
  if (next) {
    const n = next();
    const i = n == null ? -1 : S.images.findIndex((im) => im.id === n);
    if (i >= 0 && !(i > 0 && S.images[i - 1].id === id)) anchor = n;
  }
  const y2 = top(anchor);
  if (y2 != null) ul.scrollTop += y2 - y;
  return result;
}

/** 묶음 보기(100개씩…)에서 다시 받은 뒤, 보던 이미지가 펼친 묶음에 없으면 그 이미지가 든 묶음을
 * 펼친다. 앞쪽 줄이 빠지거나 늘면(필터에서 빠진 채 남겨 둔 줄, 2026-11-04) 묶음 경계가 밀려 보던
 * 이미지가 닫힌 묶음으로 넘어갈 수 있다 — 그대로 두면 다음 이미지를 펼친 묶음에서만 찾아 그만큼
 * 건너뛴다. cands = 찾을 이미지 id, 앞의 것부터. 첫 후보가 펼친 묶음에 있으면 아무것도 하지 않고,
 * 없을 때만 전체 목록을 한 번 받아 남은 첫 후보의 묶음을 펼친다. */
async function openChunkWith(cands) {
  if (S.chunks.length < 2 || !cands.length) return;
  if (S.images.some((im) => im.id === cands[0])) return;
  const all = (await api(listUrl(1e9, 0))).images;
  const pos = new Map(all.map((im, i) => [im.id, i]));
  const hit = cands.find((x) => pos.has(x));
  if (hit == null) return;
  const p = pos.get(hit);
  const ch = S.chunks.find((c) => p >= c.start && p < c.end);
  if (!ch || ch.open) return;
  ch.images = all.slice(ch.start, ch.end);
  ch.open = true;
  S.images = flatImages();
  renderImageList();
}

/** 열려 있는 묶음들의 이미지를 순서대로 이어붙인 것이 곧 탐색 대상이다. */
function flatImages() {
  return S.chunks.filter((c) => c.open).flatMap((c) => c.images);
}

async function toggleChunk(idx) {
  const ch = S.chunks[idx];
  if (ch.open) {
    ch.open = false;
    S.images = flatImages();
    renderImageList();
    return;
  }
  // 펼칠 때는 펼친 묶음까지 한꺼번에 새로 받는다. 그새 필터에 맞지 않게 된 채 남겨 둔 줄(Q·W·E·T,
  // 박스 고침)이 있으면 서버 쪽 묶음 경계가 그만큼 밀려 있어, 그 묶음만 받으면 사이의 이미지가 어느
  // 묶음에도 나오지 않았다(2026-11-04 — 첫 묶음에서 T 29번 뒤 둘째 묶음을 펼치자 29장이 빠졌다).
  // 화면 맨 위에 보이던 줄은 제자리에 둔다.
  const keep = new Set([...S.chunks.filter((c) => c.open).map((c) => c.start), ch.start]);
  const top = firstVisibleRowId();
  const before = S.images.map((im) => im.id);
  await keepRowInPlace(top, () => loadImageList(keep), top == null ? null : () => nextAfter(top, before));
}

/** 목록 화면 맨 위에 보이는 이미지 줄의 id(붙박이 묶음 머리에 가린 줄은 건너뛴다). 없으면 null. */
function firstVisibleRowId() {
  const ul = $("#image-list");
  const top = ul.getBoundingClientRect().top + 40;   // 묶음 머리 높이(약 35px)만큼 아래부터
  for (const li of ul.querySelectorAll("li.item[data-id]")) {
    if (li.getBoundingClientRect().bottom > top) return Number(li.dataset.id);
  }
  return null;
}

async function fetchChunk(idx) {
  const ch = S.chunks[idx];
  const data = await api(listUrl(ch.end - ch.start, ch.start));
  ch.images = data.images;
}

/** 현재 필터의 목록 API 주소. "지정한 id" 필터면 상태·정렬 대신 id 목록을 보낸다. */
function listUrl(limit, offset) {
  const status = $("#filter-status").value;
  if (status === "ids") {
    return `/api/images?ids=${(S.ids || []).join(",")}&limit=${limit}&offset=${offset}`;
  }
  const order = $("#filter-order").value;
  let url = `/api/images?status=${status}&order=${order}&limit=${limit}&offset=${offset}`;
  if (S.filterEmpty) url += "&empty=1";
  else if (S.filterClasses.size) url += `&cls=${[...S.filterClasses].sort((a, b) => a - b).join(",")}`;
  if (S.filterCamera) url += `&camera=${encodeURIComponent(S.filterCamera)}`;
  if (S.filterDomain) url += `&domain=${encodeURIComponent(S.filterDomain)}`;
  if (S.filterSplit) url += `&split=${encodeURIComponent(S.filterSplit)}`;
  if (S.filterTag) url += `&tag=${encodeURIComponent(S.filterTag)}`;
  return url;
}

/** 개수를 붙인 필터 select 를 다시 채운다. 고른 값이 아직 있으면 그대로 두고, 없어졌으면 전체로. */
function fillCountFilter(sel, keep, allText, items, names, noneText, none, noneKey) {
  sel.innerHTML = "";
  const add = (value, text) => {
    const o = document.createElement("option");
    o.value = value; o.textContent = text;
    sel.appendChild(o);
  };
  add("", allText);
  for (const d of items) add(d.key, `${names[d.key] || d.key} (${d.n})`);
  add(noneKey, `${noneText} (${none})`);
  const value = [...sel.options].some((o) => o.value === keep) ? keep : "";
  sel.value = value;
  sel.classList.toggle("on", !!value);
  return value;
}

/* ── 분할(train·val·test) ── 보기만 한다. 분할을 바꾸는 것은 labeler.py split 뿐이다(2026-11-03). */
async function loadSplits() {
  let data;
  try { data = await api("/api/splits"); } catch { return; }    // 실패해도 검수는 되어야 한다
  S.filterSplit = fillCountFilter($("#filter-split"), S.filterSplit, "분할: 전체",
    data.splits, {}, "미배정", data.none, data.none_key);
}

/* ── 도메인(촬영 형태) ── images.domain, 2026-11-03
 * legacy 는 출처가 없어 CCTV·비CCTV·드론을 사람이 정한다. 필터에 개수를 함께 보여 준다. */
const DOMAIN_NAME = { cctv: "CCTV", noncctv: "비CCTV", drone: "드론" };

async function loadDomains() {
  let data;
  try { data = await api("/api/domains"); } catch { return; }   // 실패해도 검수는 되어야 한다
  S.filterDomain = fillCountFilter($("#filter-domain"), S.filterDomain, "도메인: 전체",
    data.domains, DOMAIN_NAME, "미분류", data.none, data.none_key);
}

/** 지금 이미지의 도메인 버튼 표시. */
function renderDomainBar() {
  const d = S.cur ? S.cur.domain : null;
  document.querySelectorAll(".domain-bar [data-domain]").forEach((b) => {
    b.classList.toggle("on", b.dataset.domain === d);
    b.disabled = !S.cur;
  });
}

/** 목록에서 그 이미지들의 줄만 다시 그린다(순번은 그대로). 6천 줄을 통째로 다시 그리면 0.1초쯤 걸린다. */
function refreshRows(ids) {
  const want = new Set(ids);
  const byId = new Map(S.images.map((im) => [im.id, im]));
  $("#image-list").querySelectorAll("li.item").forEach((li) => {
    const id = Number(li.dataset.id);
    if (!want.has(id) || !byId.has(id)) return;
    li.replaceWith(itemRow(byId.get(id), Number(li.querySelector(".pos").textContent)));
  });
}

/** 도메인을 정하고 다음 이미지로. 이미 그 값이어도 풀지 않는다(다른 키를 누르면 바뀐다).
 *
 * 하나가 끝나기 전에 들어온 누름은 버린다. 키를 꾹 누르면 초당 30번쯤 반복 입력이 들어오는데,
 * 예전에는 그때마다 같은 이미지에 저장·목록 다시 그리기·이미지 받기가 겹쳐 쌓여, 2초 누른 것을
 * 처리하는 데 25초 동안 화면이 멈췄다(2026-11-03, 6,229줄 '전체' 보기). 이제는 꾹 누르면
 * 처리가 끝나는 대로 한 장씩 넘어간다. */
async function setDomain(domain) {
  if (!S.cur || S.domainBusy || S.listBusy) return;
  S.domainBusy = true;
  try {
    await setDomainNow(domain);
  } finally {
    S.domainBusy = false;
  }
}

async function setDomainNow(domain) {
  if (!(await saveBeforeLeaving())) return;
  const id = S.cur.id;
  // 이미 그 값이면 그대로 두고 넘어가기만 한다. 예전에는 같은 키를 다시 누르면 미분류로 되돌렸는데,
  // 연달아 누르다 지정이 풀려 버렸다(2026-11-03 사용자 요청으로 뺌).
  if (S.cur.domain === domain) { await advanceFrom(id); return; }
  try {
    await api(`/api/images/${id}/domain`, { method: "POST", body: { domain } });
  } catch (err) {
    toast(`도메인을 저장하지 못했습니다: ${err.message}`, true);
    return;
  }
  if (S.cur && S.cur.id === id) S.cur.domain = domain;
  for (const im of S.images) if (im.id === id) im.domain = domain;
  loadDomains();
  renderDomainBar();
  // 도메인 필터에서 빠져도 목록은 다시 받지 않고 그 줄을 제자리에 둔다(꼬리표만 바꾼다). 예전에는
  // 다시 받아 줄이 빠지면서 목록이 움직였다(2026-11-04 사용자 요청). '↻ 새로 받기'(refreshList)가 걷어 낸다.
  refreshRows([id]);
  await advanceFrom(id);
}

/* ── 태그 ── 2026-11-03
 * 비슷한 이미지끼리 사람이 이름을 붙여 묶는다(같은 카메라·같은 크롭 방식 등). 한 이미지에 여러 개.
 * 라벨이 아니라 학습·변경 이력에는 들어가지 않는다. 오른쪽 목록: 줄을 누르면 T 로 붙일 태그로 고르고,
 * 칸을 누르면 이 이미지에 붙이거나 뗀다. ✎ 이름 바꾸기, × 지우기(이미지는 그대로). */

// T 로 붙일 태그는 다시 열어도 남긴다. 탭(프로젝트)마다 태그 번호가 따로라 주소별로 둔다.
const CUR_TAG_KEY = `curTag:${BASE}`;
function saveCurTag() {
  try { localStorage.setItem(CUR_TAG_KEY, S.curTag == null ? "" : String(S.curTag)); } catch { /* 그만 */ }
}

async function loadTags() {
  let data;
  try { data = await api("/api/tags"); } catch { return; }      // 실패해도 검수는 되어야 한다
  S.tags = data.tags;
  S.tagName = Object.fromEntries(data.tags.map((t) => [t.id, t.name]));
  if (S.curTag != null && !(S.curTag in S.tagName)) S.curTag = null;   // 다른 창에서 지웠다
  S.filterTag = fillCountFilter($("#filter-tag"), S.filterTag, "태그: 전체",
    data.tags.map((t) => ({ key: String(t.id), n: t.n })), S.tagName, "태그 없음", data.none, data.none_key);
  renderTagList();
  if (VIEW === "gallery") galleryTagsChanged();   // 이름·개수가 바뀌었을 수 있다 (gallery.js)
}

/** 태그 찾기 칸의 글자를 담은 태그만 (대소문자 무시). 비었으면 전부. */
function tagMatches(t, q) {
  return !q || t.name.toLowerCase().includes(q.trim().toLowerCase());
}

function renderTagList() {
  const ul = $("#tag-list");
  ul.innerHTML = "";
  $("#tag-count").textContent = S.tags.length ? `${S.tags.length}개` : "";
  $("#tag-cur").textContent = S.curTag != null && S.tagName[S.curTag] ? `→ #${S.tagName[S.curTag]}` : "";
  const mine = new Set(S.cur ? S.cur.tags || [] : []);
  $("#img-tags").textContent = S.cur && mine.size
    ? [...mine].map((t) => `#${S.tagName[t] || t}`).join(" ") : "";
  if (!S.tags.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "태그가 없습니다 — 아래에 이름을 넣어 만드세요";
    ul.appendChild(li);
    return;
  }
  const q = $("#tag-q").value;
  for (const t of S.tags) {
    if (!tagMatches(t, q)) continue;
    const li = document.createElement("li");
    li.className = t.id === S.curTag ? "active" : "";
    li.title = `${t.name} — 붙은 이미지 ${t.n.toLocaleString()}장\n줄을 누르면 T 로 붙일 태그로 고릅니다. 칸은 이 이미지에 붙이기/떼기.`;
    li.innerHTML =
      `<input type="checkbox" title="이 이미지에 붙이기/떼기"${mine.has(t.id) ? " checked" : ""}${S.cur ? "" : " disabled"}>` +
      `<span class="nm"></span><span class="cnt">${t.n.toLocaleString()}</span>` +
      `<button class="linkbtn ed" title="이름 바꾸기">✎</button>` +
      `<button class="linkbtn del" title="태그 지우기 (이미지는 그대로)">×</button>`;
    li.querySelector(".nm").textContent = t.name;
    const box = li.querySelector("input");
    box.onclick = (e) => e.stopPropagation();
    box.onchange = () => { box.blur(); toggleImageTag(t.id, box.checked); };
    li.querySelector(".ed").onclick = (e) => { e.stopPropagation(); renameTag(t); };
    li.querySelector(".del").onclick = (e) => { e.stopPropagation(); deleteTag(t); };
    li.onclick = () => {
      S.curTag = S.curTag === t.id ? null : t.id;
      saveCurTag();
      renderTagList();
    };
    ul.appendChild(li);
  }
}

/** 붙이거나 뗀 결과를 화면에 반영한다(목록 줄·지금 이미지·개수). */
function applyTagChange(tagId, on, ids) {
  const hit = new Set(ids);
  const upd = (im) => {
    const cur = new Set(im.tags || []);
    if (on) cur.add(tagId); else cur.delete(tagId);
    im.tags = [...cur].sort((a, b) => a - b);
  };
  if (S.cur && hit.has(S.cur.id)) upd(S.cur);
  for (const im of S.images) if (hit.has(im.id)) upd(im);
  loadTags();               // 개수
}

/** 목록(펼친 묶음)에서 그 태그가 붙은 이미지 id. */
function taggedIds(tagId) {
  return S.images.filter((im) => (im.tags || []).includes(tagId)).map((im) => im.id);
}

/** 오른쪽 칸으로 이 이미지에 붙이거나 뗀다. 목록은 다시 받지 않는다(필터에서 빠져도 그 자리에 둔다). */
async function toggleImageTag(tagId, on) {
  if (!S.cur) return;
  const id = S.cur.id;
  try {
    await api(`/api/images/${id}/tags`, { method: "POST", body: { tag_id: tagId, on } });
  } catch (err) {
    toast(`태그를 저장하지 못했습니다: ${err.message}`, true);
    loadTags();
    return;
  }
  applyTagChange(tagId, on, [id]);
  refreshRows([id]);
}

/** T: 고른 태그를 이 이미지에 붙이고 다음으로. 이미 붙어 있으면 넘어가기만 한다(떼지 않는다 —
 * Q·W·E 와 같다). 꾹 누르면 하나씩 처리한다(setDomain 참고). */
async function tagAndNext() {
  if (!S.cur || S.tagBusy || S.listBusy) return;
  if (S.curTag == null) { toast("오른쪽 태그 목록에서 붙일 태그를 먼저 누르세요 (없으면 만드세요)", true); return; }
  S.tagBusy = true;
  try {
    if (!(await saveBeforeLeaving())) return;
    const id = S.cur.id, tagId = S.curTag;
    if (!(S.cur.tags || []).includes(tagId)) {
      try {
        await api(`/api/images/${id}/tags`, { method: "POST", body: { tag_id: tagId, on: true } });
      } catch (err) {
        toast(`태그를 저장하지 못했습니다: ${err.message}`, true);
        loadTags();
        return;
      }
      applyTagChange(tagId, true, [id]);
      // '태그 없음' 등 필터에서 빠져도 그 줄은 제자리에 둔다 — 다시 받으면 목록이 움직였다
      // (2026-11-04 사용자 요청). '↻ 새로 받기'(refreshList)가 걷어 낸다.
      refreshRows([id]);
    }
    await advanceFrom(id);
  } finally {
    S.tagBusy = false;
  }
}

/** input = 이름을 넣은 칸 (라벨링 오른쪽 #new-tag, 이미지 보기 #gs-new-tag). */
async function createTag(input = $("#new-tag")) {
  const name = input.value.trim();
  if (!name) { input.focus(); return; }
  let res;
  try {
    res = await api("/api/tags", { method: "POST", body: { name } });
  } catch (err) {
    toast(err.message, true);
    return;
  }
  input.value = "";
  input.blur();
  S.curTag = res.id;      // 만든 태그를 바로 T 로 붙일 수 있게
  saveCurTag();
  await loadTags();
  toast(`태그 '${name}' 를 만들었습니다 — T 로 붙입니다`);
}

async function renameTag(t) {
  const name = prompt(`'${t.name}' 의 새 이름`, t.name);
  if (name == null || name.trim() === t.name) return;
  try {
    await api(`/api/tags/${t.id}`, { method: "PUT", body: { name } });
  } catch (err) {
    toast(err.message, true);
    return;
  }
  await loadTags();
  // 그 태그가 붙은 줄만 고친다. 예전에는 목록을 통째로 다시 그려, '전체' 에서 지금 줄을 놓쳤다(2026-11-04).
  refreshRows(taggedIds(t.id));
}

async function deleteTag(t) {
  if (!confirm(`태그 '${t.name}' 를 지웁니다. 붙어 있던 이미지 ${t.n.toLocaleString()}장에서도 떨어집니다`
    + " (이미지·라벨은 그대로입니다). 되돌릴 수 없습니다. 지울까요?")) return;
  try {
    await api(`/api/tags/${t.id}`, { method: "DELETE" });
  } catch (err) {
    toast(err.message, true);
    return;
  }
  const hit = taggedIds(t.id);
  const drop = (im) => { if (im.tags) im.tags = im.tags.filter((x) => x !== t.id); };
  if (S.cur) drop(S.cur);
  S.images.forEach(drop);
  const filtered = S.filterTag === String(t.id);
  await loadTags();       // 고른 필터가 없어지면 '전체' 로 돌아간다
  // 그 태그로 거르던 중이면 '전체' 로 다시 받되 지금 줄은 화면의 같은 높이에 둔다. 아니면 그 줄만 고친다.
  if (filtered) await keepCurrentRow(() => loadImageList());
  else refreshRows(hit);
}

/* ── 카메라(출처) 필터 ── docs/provenance.md
 * 카메라가 붙은 이미지가 하나도 없으면(legacy 뿐이면) 숨긴다. 고른 카메라가 사라지면 '전체' 로. */
async function loadCameras() {
  let data;
  try { data = await api("/api/cameras"); } catch { return; }   // 실패해도 검수는 되어야 한다
  const sel = $("#filter-camera");
  const keep = S.filterCamera;
  sel.innerHTML = "";
  const add = (value, text, title = "") => {
    const o = document.createElement("option");
    o.value = value; o.textContent = text; o.title = title;
    sel.appendChild(o);
  };
  add("", "카메라: 전체");
  for (const c of data.cameras) {
    add(c.key, `${c.key}${c.place ? " · " + c.place : " · (등록 파일에 없음)"} (${c.n})`,
        [c.kind, c.mount, c.split && `split 고정 ${c.split}`].filter(Boolean).join(" · "));
  }
  if (data.cameras.length) add(data.none_key, `출처 없음 (${data.no_camera})`);
  $("#camera-filter").classList.toggle("hidden", !data.cameras.length);
  sel.title = data.error ? `카메라 등록 파일 오류: ${data.error}` : "카메라(출처)로 거릅니다. 상태·클래스 필터와 함께 걸립니다.";
  S.filterCamera = [...sel.options].some((o) => o.value === keep) ? keep : "";
  sel.value = S.filterCamera;
  sel.classList.toggle("on", !!S.filterCamera);
}

/* ── 클래스 필터 ──────────────────────────────────────
 * 고른 클래스를 **전부** 가진 이미지만 남긴다(교집합). '박스 없음' 은 그 반대쪽이라
 * 같이 고를 수 없게 했다 — 교집합이 언제나 비기 때문이다. */
function cfLabel() {
  if (S.filterEmpty) return "클래스: 박스 없음";
  if (!S.filterClasses.size) return "클래스: 전체";
  const names = [...S.filterClasses].sort((a, b) => a - b).map((i) => S.classes[i] ?? `cls_${i}`);
  return `클래스: ${names.join(" + ")}`;
}

/** 버튼 글자와 체크 상태만 손본다. 패널 DOM 은 건드리지 않는다 — 고를 때마다 다시 그리면
 *  방금 누른 체크박스가 교체돼 연속으로 고르기 어렵다 (2026-09-21 에 겪음). */
function syncClassFilter() {
  const active = S.filterEmpty || S.filterClasses.size > 0;
  const btn = $("#cf-toggle");
  btn.textContent = cfLabel();
  btn.classList.toggle("on", active);
  $("#cf-clear").classList.toggle("hidden", !active);
  $("#cf-panel").querySelectorAll("input[type=checkbox]").forEach((cb) => {
    const i = cb.dataset.cls;
    if (i === undefined) cb.checked = S.filterEmpty;         // '박스 없음'
    else {
      cb.checked = S.filterClasses.has(Number(i));
      cb.disabled = S.filterEmpty;                            // 둘은 함께 못 쓴다
    }
  });
}

function renderClassFilter() {
  const panel = $("#cf-panel");
  panel.innerHTML = "";
  S.classes.forEach((name, i) => {
    const row = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.dataset.cls = String(i);
    cb.checked = S.filterClasses.has(i);
    cb.disabled = S.filterEmpty;
    cb.onchange = () => {
      cb.checked ? S.filterClasses.add(i) : S.filterClasses.delete(i);
      syncClassFilter();
      reloadList();
    };
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = (S.classBoxes[i]?.total ?? 0).toLocaleString();
    row.append(cb, document.createTextNode(name), n);
    panel.appendChild(row);
  });

  const sep = document.createElement("div");
  sep.className = "sep";
  panel.appendChild(sep);

  const row = document.createElement("label");
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = S.filterEmpty;
  cb.onchange = () => {
    S.filterEmpty = cb.checked;
    if (S.filterEmpty) S.filterClasses.clear();   // 둘은 함께 쓸 수 없다
    syncClassFilter();
    reloadList();
  };
  row.append(cb, document.createTextNode("박스 없음 (빈 이미지)"));
  row.title = "박스가 하나도 없는 이미지만 봅니다. 클래스 선택과 함께 쓸 수 없습니다.";
  panel.appendChild(row);
  syncClassFilter();
}

/* ── 필터 칸 (2026-11-04) ── 클래스·카메라·도메인·분할·태그·목록 묶음을 접히는 한 칸에 모았다. 접혀 있으면
 * 고른 필터만 한 줄로 보인다(상태·정렬은 늘 위에). 이미지 보기에서는 늘 펼친다(style.css). */
function optText(sel) {        // 고른 항목의 글자에서 끝의 '(장수)' 를 뗀다
  const o = sel.selectedOptions[0];
  return o ? o.textContent.replace(/\s*\([\d,]+\)$/, "") : "";
}

function renderFilterSummary() {
  const parts = [];
  if (S.filterEmpty || S.filterClasses.size) parts.push(cfLabel());
  if (S.filterCamera) parts.push(`카메라 ${optText($("#filter-camera"))}`);
  if (S.filterDomain) parts.push(`도메인 ${optText($("#filter-domain"))}`);
  if (S.filterSplit) parts.push(`분할 ${optText($("#filter-split"))}`);
  if (S.filterTag) parts.push(S.filterTag === "__none__" ? "태그 없음" : `#${optText($("#filter-tag"))}`);
  const sum = $("#filters-sum");
  sum.textContent = parts.length ? parts.join(" · ") : "필터 없음";
  sum.title = parts.length ? `걸린 필터: ${sum.textContent}` : "";
  sum.classList.toggle("on", parts.length > 0);
  $("#filters-clear").classList.toggle("hidden", !parts.length);
}

function toggleFilters(open = $("#filters-body").classList.contains("hidden")) {
  $("#filters-body").classList.toggle("hidden", !open);
  $("#filters-caret").textContent = open ? "▾" : "▸";
  try { localStorage.setItem("filtersOpen", open ? "1" : "0"); } catch { /* 저장 못 해도 그만 */ }
}

/** 클래스·카메라·도메인·분할·태그 필터를 모두 푼다(상태·정렬은 그대로). */
function clearFilters() {
  S.filterClasses.clear();
  S.filterEmpty = false;
  S.filterCamera = S.filterDomain = S.filterSplit = S.filterTag = "";
  for (const s of ["#filter-camera", "#filter-domain", "#filter-split", "#filter-tag"]) {
    $(s).value = "";
    $(s).classList.remove("on");
  }
  syncClassFilter();
  reloadList();
}

/** "812, 904 1001" -> [812, 904, 1001]. 숫자가 아닌 조각이 있으면 null. */
function parseIds(text) {
  const tokens = String(text).split(/[\s,]+/).filter(Boolean);
  if (!tokens.length || !tokens.every((t) => /^\d+$/.test(t))) return null;
  return [...new Set(tokens.map(Number))];
}

/** 주소의 #ids=812,904 (또는 #id=812). 없거나 잘못됐으면 null. */
function idsFromHash() {
  const m = decodeURIComponent(location.hash).match(/^#ids?=(.+)$/);
  return m ? parseIds(m[1]) : null;
}

/** 지정한 id 들만 목록으로 만들어 첫 장을 연다. 화살표·승인은 이 목록 안에서 움직인다. */
async function openIds(ids) {
  S.ids = ids;
  const sel = $("#filter-status");
  sel.querySelector('option[value="ids"]').hidden = false;
  sel.value = "ids";
  $("#goto-ids").value = ids.join(", ");
  await loadImageList(new Set([0]));
  if (S.images.length) await loadImage(S.images[0].id);
  // 알림은 한 번에 하나만 보이므로 없는 id 경고를 장수와 한 줄로 합친다.
  const miss = S.missingIds;
  if (miss.length) {
    toast(`지정한 id ${S.total}장 · 없는 id ${miss.length}개: ${miss.slice(0, 10).join(", ")}`, true);
  } else {
    toast(`지정한 id ${S.total}장`);
  }
}

function refresh() {
  renderBoxList();
  renderClassList();
  draw();
}

/* ───────────────────────── 변경 내역 (2026-09-23) ─────────────────────────
 * 서버가 박스를 저장할 때마다 '바뀌기 전·뒤' 를 box_history 에 남긴다. 여기서는 그걸
 * 보여 주고, 한 줄을 골라 '그 저장 전' 으로 되돌린다. 되돌린 것도 이력에 남는다. */

const SRC_KO = { ui: "수정", predict: "자동 라벨", restore: "되돌림", bulk: "일괄 수정", revert: "일괄 취소" };

function fmtTime(iso) {
  const d = new Date(iso);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/** 두 박스 목록의 차이를 한 줄로: 'car −1 · van +1', 개수가 같으면 무엇이 바뀌었는지. */
function histDiff(before, after) {
  const count = (list) => list.reduce((m, b) => { m[b.cls] = (m[b.cls] || 0) + 1; return m; }, {});
  const a = count(before), z = count(after);
  const parts = [];
  for (const k of new Set([...Object.keys(a), ...Object.keys(z)])) {
    const d = (z[k] || 0) - (a[k] || 0);
    if (d) parts.push(`${clsName(Number(k))} ${d > 0 ? "+" : "−"}${Math.abs(d)}`);
  }
  if (parts.length) return parts.join(" · ");
  const moved = before.some((b, i) => after[i]
    && Math.abs(b.cx - after[i].cx) + Math.abs(b.cy - after[i].cy)
     + Math.abs(b.w - after[i].w) + Math.abs(b.h - after[i].h) > 1e-6);
  if (moved) return "위치·크기";
  const recls = before.some((b, i) => after[i] && b.cls !== after[i].cls);
  return recls ? "클래스 (개수 같음)" : "신뢰도·출처만";
}

function setHistCount(n) {
  if (n !== undefined) $("#hist-count").textContent = n ? String(n) : "";
}

function toggleHistory(open = !S.histOpen) {
  S.histOpen = open;
  try { localStorage.setItem("histOpen", open ? "1" : "0"); } catch { /* 저장 못 해도 그만 */ }
  $("#hist-list").classList.toggle("hidden", !open);
  $("#hist-caret").textContent = open ? "▾" : "▸";
  if (open) loadHistory();
  else if (S.ghost) { S.ghost = null; draw(); }
}

async function loadHistory() {
  if (!S.cur || !S.histOpen) return;
  const id = S.cur.id;
  try {
    const r = await api(`/api/images/${id}/history`);
    if (!S.cur || S.cur.id !== id) return;
    S.history = r.entries;
    setHistCount(r.entries.length);
    renderHistory();
  } catch (err) {
    toast(`변경 내역을 못 읽었습니다: ${err.message}`, true);
  }
}

function renderHistory() {
  const ul = $("#hist-list");
  ul.innerHTML = "";
  if (!S.history.length) {
    ul.innerHTML = '<li><span class="empty">아직 바뀐 적이 없습니다</span></li>';
    return;
  }
  for (const h of S.history) {
    const li = document.createElement("li");
    li.className = h.n_after < h.n_before ? "shrunk" : "";
    li.innerHTML = '<span class="when"></span><span class="what"></span><span class="cnt"></span>'
      + '<span class="diff"></span><button class="btn" title="이 저장이 일어나기 직전의 박스로 되돌립니다">이 저장 전으로</button>';
    li.querySelector(".when").textContent = fmtTime(h.at);
    li.querySelector(".what").textContent = SRC_KO[h.source] || h.source;
    li.querySelector(".cnt").textContent = `${h.n_before}→${h.n_after}`;
    li.querySelector(".diff").textContent = histDiff(h.before, h.after);
    li.onmouseenter = () => { S.ghost = h.before; draw(); };
    li.onmouseleave = () => { S.ghost = null; draw(); };
    li.querySelector(".btn").onclick = () => restoreEntry(h);
    ul.appendChild(li);
  }
}

async function restoreEntry(h) {
  if (!S.cur) return;
  const msg = `${fmtTime(h.at)} 저장 직전 상태(박스 ${h.n_before}개)로 되돌립니다.\n`
    + "되돌린 것도 변경 내역에 남으므로 다시 되돌릴 수 있습니다.";
  if (!confirm(msg)) return;
  if (!(await saveBeforeLeaving())) return;
  const id = S.cur.id;
  try {
    const r = await api(`/api/images/${id}/history/${h.id}/restore`,
      { method: "POST", body: { which: "before" } });
    if (!S.cur || S.cur.id !== id) return;
    pushUndo();                       // Ctrl+Z 로 되돌리기 전으로 다시 갈 수 있다
    S.boxes = r.boxes.map((b) => ({ ...b, cls0: b.cls }));
    S.sel = -1;
    S.ghost = null;
    showSaved(r.n_boxes);
    setHistCount(r.n_history);
    refresh();
    loadHistory();
    loadState().catch(() => {});
    toast(`박스 ${r.n_before}개 → ${r.n_boxes}개로 되돌렸습니다`);
  } catch (err) {
    toast(`되돌리기 실패: ${err.message}`, true);
  }
}

/* ───────────────────────── 대화상자 ───────────────────────── */

const modalOpen = () => !$("#modal").classList.contains("hidden");

function openModal(title, html) {
  $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = html;
  $("#modal").classList.remove("hidden");
}

function closeModal() {
  $("#modal").classList.add("hidden");
  $("#modal-body").innerHTML = "";
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/**
 * '변경 기록' — 여러 이미지에 걸친 최근 변경. 기본은 **박스가 줄어든 저장만** 보여 준다.
 * 여러 장을 연달아 고친 뒤 사라진 박스가 없는지 보는 용도다(09-21 에는 백업과 대조했다).
 */
async function openAudit(mode = "shrunk") {
  openModal("변경 기록", `
    <div class="row">
      <select id="audit-mode">
        <option value="shrunk">박스가 줄어든 저장</option>
        <option value="all">모든 저장</option>
        <option value="bulk">일괄 수정 묶음</option>
      </select>
      <span id="audit-note" class="dim"></span>
      <span class="spacer"></span>
      <button id="audit-open" class="btn btn-sm" title="표의 이미지들을 왼쪽 목록으로 엽니다">목록으로 열기</button>
    </div>
    <div id="audit-body">불러오는 중…</div>`);
  const sel = $("#audit-mode");
  sel.value = mode;
  sel.onchange = () => openAudit(sel.value);
  let data;
  try {
    data = await api(`/api/history?limit=500${mode === "shrunk" ? "&shrunk=1" : ""}`);
  } catch (err) {
    $("#audit-body").textContent = `불러오지 못했습니다: ${err.message}`;
    return;
  }
  if (!modalOpen() || $("#audit-mode")?.value !== mode) return;   // 그 사이 닫았거나 바꿨다

  if (mode === "bulk") {
    renderBatches(data.batches);
    return;
  }
  const rows = data.entries;
  const ids = [...new Set(rows.map((r) => r.image_id))];
  $("#audit-note").textContent = rows.length
    ? `${rows.length}건 · 이미지 ${ids.length}장 (최근 500건까지)`
    : "";
  $("#audit-open").disabled = !ids.length;
  $("#audit-open").onclick = () => { closeModal(); openIds(ids); };
  if (!rows.length) {
    $("#audit-body").innerHTML = mode === "shrunk"
      ? "<p>박스 수가 줄어든 저장이 없습니다.</p>"
      : "<p>아직 기록된 변경이 없습니다.</p>";
    return;
  }
  $("#audit-body").innerHTML = `<table class="tbl"><thead><tr>
      <th>시각</th><th>id</th><th>이름</th><th>분할</th><th>상태</th><th>박스</th><th>종류</th>
    </tr></thead><tbody>${rows.map((r) => `
      <tr class="link" data-id="${r.image_id}" title="이 이미지 열기">
        <td>${esc(fmtTime(r.at))}</td><td class="num">${r.image_id}</td>
        <td class="name">${esc(r.name)}</td><td>${esc(r.split || "-")}</td><td>${esc(r.status)}</td>
        <td class="num${r.n_after < r.n_before ? " shrunk" : ""}">${r.n_before}→${r.n_after}</td>
        <td>${esc(SRC_KO[r.source] || r.source)}</td>
      </tr>`).join("")}</tbody></table>`;
  $("#audit-body").querySelectorAll("tr.link").forEach((tr) => {
    tr.onclick = () => { closeModal(); loadImage(Number(tr.dataset.id)); };
  });
}

/** '같은 자리 일괄 수정' 묶음 목록. 묶음째 되돌릴 수 있다 (그 뒤 또 고친 이미지는 건너뛴다). */
function renderBatches(batches) {
  $("#audit-open").disabled = true;
  const done = batches.filter((b) => b.source === "bulk");
  const undone = new Set(batches.filter((b) => b.source === "revert").map((b) => b.batch.replace(/^취소: /, "")));
  $("#audit-note").textContent = done.length ? `${done.length}건 (최근 30건까지)` : "";
  if (!done.length) {
    $("#audit-body").innerHTML = "<p>일괄 수정 기록이 없습니다. (박스를 고르고 <kbd>G</kbd> 로 만든다)</p>";
    return;
  }
  $("#audit-body").innerHTML = `<table class="tbl"><thead><tr>
      <th>시각</th><th>내용</th><th>이미지</th><th></th>
    </tr></thead><tbody>${done.map((b, i) => `
      <tr><td>${esc(fmtTime(b.at))}</td><td class="name">${esc(b.batch)}</td>
        <td class="num">${b.images}</td>
        <td>${undone.has(b.batch) ? '<span class="dim">되돌림</span>'
          : `<button class="btn btn-sm" data-i="${i}">되돌리기</button>`}</td></tr>`).join("")}
    </tbody></table>`;
  $("#audit-body").querySelectorAll("button[data-i]").forEach((btn) => {
    btn.onclick = async () => {
      const b = done[Number(btn.dataset.i)];
      if (!confirm(`일괄 수정 '${b.batch}' (이미지 ${b.images}장)을 되돌립니다.\n`
        + "그 뒤에 또 고친 이미지는 건드리지 않고 건너뜁니다.")) return;
      if (!(await saveBeforeLeaving())) return;
      try {
        const r = await api("/api/history/batch/revert", { method: "POST", body: { batch: b.batch } });
        toast(`되돌림 ${r.reverted}장` + (r.skipped ? ` · 그 뒤 고쳐서 건너뜀 ${r.skipped}장` : ""));
        if (S.cur) await reloadCurrentBoxes();
        loadState().catch(() => {});
        openAudit("bulk");
      } catch (err) {
        toast(`되돌리기 실패: ${err.message}`, true);
      }
    };
  });
}

/* ───────────────────── 같은 자리 일괄 수정 (2026-09-23) ─────────────────────
 * 고정 카메라 묶음에서는 같은 차가 수백 프레임 같은 자리에 있다(van 박스의 83% 가
 * 고정 CCTV 한 대의 두 자리). 한 장에서 클래스를 고친 뒤 G 를 누르면, 같은 묶음의 다른 이미지에서
 * 같은 자리 박스를 찾아 **무엇이 바뀔지 먼저 보여 주고**, 적용하면 한꺼번에 바꾼다.
 * 되돌리기는 상단 '변경 기록' → '일괄 수정 묶음'. */

async function openSamePlace() {
  if (!S.cur || S.sel < 0) { toast("박스를 먼저 고르세요 (클래스를 바꾼 뒤 G)"); return; }
  if (!(await saveBeforeLeaving())) return;          // 이 이미지의 수정부터 저장
  const b = S.boxes[S.sel];
  if (!b) return;
  const from0 = b.cls0 ?? b.cls;
  const st = {
    imageId: S.cur.id,
    box: [b.cx, b.cy, b.w, b.h],
    to: b.cls,
    from: from0,
    // 원래 클래스 그대로라면(클래스를 안 바꾸고 G) 다른 클래스로 적힌 것을 끌어오려는 뜻이다
    any: from0 === b.cls,
    iou: 0.85,
    preview: null,
  };
  openModal("같은 자리 일괄 수정", `
    <p>같은 묶음(같은 장면)의 <b>다른 이미지</b>에서 이 박스와 같은 자리에 있는 박스를
      <b class="to"></b> 로 바꿉니다. 이미지마다 가장 많이 겹치는 박스 하나만 봅니다.
      지금 이미지는 이미 저장된 그대로입니다.</p>
    <div class="row">
      <label>겹침(IoU) ≥
        <select id="sp-iou">
          <option value="0.95">0.95 (거의 똑같은 자리)</option>
          <option value="0.85" selected>0.85</option>
          <option value="0.7">0.7</option>
          <option value="0.5">0.5 (느슨하게)</option>
        </select></label>
      <label><input type="checkbox" id="sp-any"> 다른 클래스 박스도 포함</label>
      <span class="dim" id="sp-from"></span>
    </div>
    <div id="sp-preview">찾는 중…</div>
    <div class="row">
      <span class="spacer"></span>
      <button id="sp-list" class="btn" disabled title="바뀔 이미지들을 왼쪽 목록으로 엽니다 (적용하지 않음)">목록으로 보기</button>
      <button id="sp-apply" class="btn btn-primary" disabled>적용</button>
    </div>`);
  $("#modal-body .to").textContent = clsName(st.to);
  $("#sp-any").checked = st.any;
  $("#sp-from").textContent = `원래 클래스: ${clsName(st.from)}`;

  const body = () => ({
    box: st.box, to_cls: st.to, iou: st.iou,
    from_cls: st.any ? null : [st.from],
  });

  const preview = async () => {
    $("#sp-apply").disabled = $("#sp-list").disabled = true;
    $("#sp-preview").textContent = "찾는 중…";
    try {
      st.preview = await api(`/api/images/${st.imageId}/same-place`, { method: "POST", body: body() });
    } catch (err) {
      $("#sp-preview").innerHTML = `<p class="warn"></p>`;
      $("#sp-preview .warn").textContent = err.message;
      return;
    }
    const p = st.preview;
    const fmt = (o) => Object.entries(o).map(([k, v]) => `${esc(k)} ${v.toLocaleString()}`).join(" · ") || "-";
    const risky = (p.by_split.test || 0) + (p.by_split.val || 0);
    $("#sp-preview").innerHTML = `
      <p>묶음 <b>${esc(p.grp)}</b> (이미지 ${p.group_size.toLocaleString()}장) 에서 같은 자리 박스
        <b>${p.matched.toLocaleString()}</b>개를 찾았습니다.
        그중 <b>${p.to_change.toLocaleString()}</b>개를 ${esc(clsName(st.to))} 로 바꿉니다
        (이미 ${esc(clsName(st.to))}: ${p.already.toLocaleString()}개).</p>
      <p class="dim">바뀔 박스의 지금 클래스: ${fmt(p.by_class)}<br>
        분할: ${fmt(p.by_split)} &nbsp;·&nbsp; 상태: ${fmt(p.by_status)}</p>
      ${risky ? `<p class="warn">val·test 이미지 ${risky}장이 바뀝니다 — 그 분할의 점수가 움직입니다.</p>` : ""}`;
    $("#sp-apply").disabled = !p.to_change;
    $("#sp-list").disabled = !p.to_change;
  };

  $("#sp-iou").onchange = (e) => { st.iou = Number(e.target.value); preview(); };
  $("#sp-any").onchange = (e) => { st.any = e.target.checked; preview(); };
  $("#sp-list").onclick = () => { const ids = st.preview.ids; closeModal(); openIds(ids); };
  $("#sp-apply").onclick = async () => {
    $("#sp-apply").disabled = true;
    try {
      const r = await api(`/api/images/${st.imageId}/same-place/apply`,
        { method: "POST", body: { ...body(), expected: st.preview.to_change } });
      closeModal();
      toast(`${r.changed.toLocaleString()}장을 바꿨습니다 — 되돌리려면 '변경 기록' → 일괄 수정 묶음`);
      loadState().catch(() => {});
      loadImageList().catch(() => {});
    } catch (err) {
      toast(err.message, true);
      preview();                 // 대상이 바뀌었으면 새로 보여 준다
    }
  };
  preview();
}

/* ───────────────────── 예측 비교 P (2026-09-23) ─────────────────────
 * 운영 모델을 지금 이미지에 돌려(서버 /predict, DB 에는 안 씀) 라벨과 어긋나는 곳을
 * 점선으로 겹쳐 보여 준다. 분류는 tools/label_qa.py 와 같은 규칙이다:
 *   빠짐  — 모델이 확신(≥0.5)하는데 근처에 라벨이 없다
 *   클래스 — 라벨과 크게 겹치는데(IoU≥0.5) 클래스가 다르다
 *   박스  — 같은 클래스인데 IoU 가 0.3~0.5 뿐이다
 *   못 봄 — 라벨이 있는데 모델이 거기서 아무것도 못 본다 (짧은 변 24px 미만은 뺀다)
 * **모델이 모르는 클래스의 라벨은 비교하지 않는다** — visible-3 은 pickup 을 모른다.
 * 비교는 브라우저에서 한다: 박스를 고칠 때마다 모델을 다시 돌리지 않게. */

const FIND = {
  MATCH: 0.5, NEAR: 0.3, HIGH: 0.5, MIN_SIDE: 24,
  label: { missing: "빠짐", class: "클래스", box: "박스", unseen: "못 봄" },
  color: { missing: "#ff6fe0", class: "#ffae52", box: "#57e2ff", unseen: "#ff6b6b" },
  accept: { missing: "추가", class: "바꾸기", box: "맞추기" },
};

function iouN(a, b) {
  const iw = Math.max(0, Math.min(a.cx + a.w / 2, b.cx + b.w / 2) - Math.max(a.cx - a.w / 2, b.cx - b.w / 2));
  const ih = Math.max(0, Math.min(a.cy + a.h / 2, b.cy + b.h / 2) - Math.max(a.cy - a.h / 2, b.cy - b.h / 2));
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

/** 라벨(S.boxes)과 예측(S.pred.preds)을 견줘 어긋나는 곳을 뽑는다. */
function computeFindings() {
  if (!S.pred || !S.cur) return [];
  const labels = S.boxes;
  const preds = S.pred.preds;          // 서버가 신뢰도 내림차순으로 준다
  const known = new Set(S.pred.model_classes);
  const W = S.cur.width, H = S.cur.height;
  const gHit = labels.map(() => false);
  const pHit = preds.map(() => false);

  // 1) 짝짓기: 같은 클래스끼리, 신뢰도 높은 예측부터. 모델이 모르는 클래스의 라벨은
  //    어느 클래스 예측과도 짝이 될 수 있다 (그 자리를 '빠짐' 으로 띄우지 않게).
  preds.forEach((p, i) => {
    let best = -1, bestV = FIND.MATCH;
    labels.forEach((g, j) => {
      if (gHit[j] || (g.cls !== p.cls && known.has(g.cls))) return;
      const v = iouN(p, g);
      if (v >= bestV) { best = j; bestV = v; }
    });
    if (best >= 0) { gHit[best] = true; pHit[i] = true; }
  });

  const out = [];
  const used = new Set();
  // 2) 짝 없는 확신 예측
  preds.forEach((p, i) => {
    if (pHit[i] || p.conf < FIND.HIGH) return;
    let jo = -1, io = 0, js = -1, is = 0;
    labels.forEach((g, j) => {
      const v = iouN(p, g);
      if (g.cls === p.cls) { if (v > is) { is = v; js = j; } }
      else if (v > io) { io = v; jo = j; }
    });
    if (io >= FIND.MATCH) {
      if (!gHit[jo] && !used.has(jo) && known.has(labels[jo].cls)) {
        used.add(jo);
        out.push({ kind: "class", pred: i, label: jo, iou: io });
      }
      return;                                   // 다른 클래스로 잡힌 라벨 위의 가설 — 후보 아님
    }
    if (is >= FIND.NEAR) {
      if (!gHit[js] && !used.has(js)) {
        used.add(js);
        out.push({ kind: "box", pred: i, label: js, iou: is });
      }
      return;                                   // 군중 속 중복 예측
    }
    out.push({ kind: "missing", pred: i, label: -1, iou: Math.max(io, is) });
  });
  // 3) 모델이 아무것도 못 보는 라벨
  labels.forEach((g, j) => {
    if (gHit[j] || used.has(j) || !known.has(g.cls)) return;
    if (Math.min(g.w * W, g.h * H) < FIND.MIN_SIDE) return;
    const best = preds.reduce((m, p) => Math.max(m, iouN(p, g)), 0);
    if (best < FIND.NEAR) out.push({ kind: "unseen", pred: -1, label: j, iou: best });
  });
  return out.filter((f) => !S.predIgnore.has(findKey(f)));
}

/** '무시' 를 기억하는 키. 박스 번호는 고칠 때마다 밀리므로 좌표로 만든다. */
function findKey(f) {
  const r = (b) => (b ? [b.cx, b.cy, b.w, b.h].map((v) => v.toFixed(3)).join(",") : "-");
  return `${f.kind}|${r(S.pred.preds[f.pred])}|${r(S.boxes[f.label])}`;
}

function findText(f) {
  const p = S.pred.preds[f.pred], g = S.boxes[f.label];
  switch (f.kind) {
    case "missing": return `${clsName(p.cls)} ${p.conf.toFixed(2)} — 라벨 없음`;
    case "class":   return `${clsName(g.cls)} → ${clsName(p.cls)} ${p.conf.toFixed(2)}`;
    case "box":     return `${clsName(p.cls)} ${p.conf.toFixed(2)} · 겹침 ${f.iou.toFixed(2)}`;
    default:        return `${clsName(g.cls)} — 모델이 못 봄`;
  }
}

function updateFindings() {
  S.findings = S.predOn ? computeFindings() : [];
  if (S.findHot >= S.findings.length) S.findHot = -1;
  renderFindings();
}

function renderFindings() {
  const ul = $("#pred-list");
  ul.innerHTML = "";
  $("#pred-count").textContent = S.pred && S.findings.length ? String(S.findings.length) : "";
  if (!S.pred) {
    ul.innerHTML = `<li class="empty">${S.predLoading ? "예측 중…" : "예측이 없습니다"}</li>`;
    return;
  }
  if (!S.findings.length) {
    ul.innerHTML = '<li class="empty">라벨과 어긋나는 곳이 없습니다</li>';
    return;
  }
  S.findings.forEach((f, k) => {
    const li = document.createElement("li");
    li.className = k === S.findHot ? "hot" : "";
    li.innerHTML = `<span class="kind ${f.kind}">${FIND.label[f.kind]}</span><span class="txt"></span>`
      + (FIND.accept[f.kind] ? `<button class="btn ok">${FIND.accept[f.kind]}</button>` : "")
      + '<button class="btn no" title="이 이미지에서 이 항목을 숨깁니다">무시</button>';
    li.querySelector(".txt").textContent = findText(f);
    li.onmouseenter = () => { S.findHot = k; draw(); };
    li.onmouseleave = () => { S.findHot = -1; draw(); };
    li.onclick = (e) => {
      if (e.target.classList.contains("ok")) acceptFinding(f);
      else if (e.target.classList.contains("no")) ignoreFinding(f);
      else if (f.label >= 0) selectBox(f.label);
    };
    ul.appendChild(li);
  });
}

function drawFindings() {
  if (!S.pred || !S.findings.length) return;
  ctx.save();
  ctx.font = "11px 'Malgun Gothic', Consolas, monospace";
  S.findings.forEach((f, k) => {
    const b = f.kind === "class" || f.kind === "unseen" ? S.boxes[f.label] : S.pred.preds[f.pred];
    if (!b) return;
    const [ix, iy, iw, ih] = boxRect(b);
    const [x, y] = toScreen(ix, iy);
    const w = iw * S.view.scale, h = ih * S.view.scale;
    const hot = k === S.findHot;
    ctx.strokeStyle = ctx.fillStyle = FIND.color[f.kind];
    ctx.lineWidth = hot ? 3.5 : 2;
    ctx.setLineDash(hot ? [] : [7, 4]);
    ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
    const t = `${FIND.label[f.kind]}: ${findText(f)}`;
    const tw = ctx.measureText(t).width + 8;
    const ty = y + h + 4;
    ctx.fillRect(x - 2, ty, tw, 15);
    ctx.fillStyle = "#14161a";
    ctx.fillText(t, x + 2, ty + 11);
  });
  ctx.restore();
}

function acceptFinding(f) {
  const p = f.pred >= 0 ? S.pred.preds[f.pred] : null;
  if (!p) return;
  pushUndo();
  let at;
  if (f.kind === "missing") {
    S.boxes.push({ cls: p.cls, cx: p.cx, cy: p.cy, w: p.w, h: p.h, conf: null, origin: "human", cls0: p.cls });
    at = S.boxes.length - 1;
  } else if (f.kind === "class") {
    Object.assign(S.boxes[f.label], { cls: p.cls, conf: null, origin: "human" });
    at = f.label;
  } else if (f.kind === "box") {
    Object.assign(S.boxes[f.label], { cx: p.cx, cy: p.cy, w: p.w, h: p.h, conf: null, origin: "human" });
    at = f.label;
  } else {
    S.undo.pop();
    return;
  }
  S.findHot = -1;
  markDirty();
  selectBox(at);
}

function ignoreFinding(f) {
  S.predIgnore.add(findKey(f));
  S.findHot = -1;
  updateFindings();
  draw();
}

async function fetchPreds() {
  if (!S.predOn || !S.cur) return;
  const id = S.cur.id;
  S.pred = null;
  S.predLoading = true;
  $("#pred-info").textContent = "예측 중…";
  $("#pred-info").title = "";
  updateFindings();
  try {
    const r = await api(`/api/images/${id}/predict`);
    if (!S.cur || S.cur.id !== id || !S.predOn) return;
    S.pred = r;
    $("#pred-info").textContent = `${r.weights} · ${r.device === "cpu" ? "CPU" : "GPU"} ${Math.round(r.ms)}ms`;
    $("#pred-info").title = (r.device === "cpu" ? "GPU 에서 학습·자동 라벨링이 도는 중이라 CPU 로 돌렸습니다.\n" : "")
      + (r.unknown_classes.length
        ? `이 모델은 ${r.unknown_classes.join("·")} 를 모릅니다 — 그 클래스의 라벨은 비교하지 않습니다.`
        : "");
  } catch (err) {
    if (S.cur && S.cur.id === id) {
      $("#pred-info").textContent = "실패";
      toast(`예측 실패: ${err.message}`, true);
    }
  } finally {
    if (S.cur && S.cur.id === id) {
      S.predLoading = false;
      updateFindings();
      draw();
    }
  }
}

function togglePred(on = !S.predOn) {
  S.predOn = on;
  try { localStorage.setItem("predOn", on ? "1" : "0"); } catch { /* 저장 못 해도 그만 */ }
  $("#btn-pred").classList.toggle("on", on);
  $("#pred-head").classList.toggle("hidden", !on);
  $("#pred-list").classList.toggle("hidden", !on);
  if (on) fetchPreds();
  else { S.pred = null; S.findings = []; S.findHot = -1; draw(); }
}

/** 서버에서 박스가 바뀌었을 때(일괄 수정·되돌리기) 지금 이미지의 박스를 다시 읽는다. */
async function reloadCurrentBoxes() {
  if (!S.cur || !(await saveBeforeLeaving())) return;
  const id = S.cur.id;
  const meta = await api(`/api/images/${id}`);
  if (!S.cur || S.cur.id !== id) return;
  S.boxes = meta.boxes.map((b) => ({ ...b, cls0: b.cls }));
  S.sel = -1;
  S.undo = [];
  setHistCount(meta.n_history);
  refresh();
  if (S.histOpen) loadHistory();
}

/* ───────────────────────── 데이터 로딩 ───────────────────────── */

async function loadState() {
  const st = await api("/api/state");
  S.classes = st.classes;
  S.classBoxes = st.class_boxes || [];
  S.confidentAt = st.confident_at;
  S.counts = st.counts;
  S.paths = st.paths || {};
  const c = st.counts;
  $("#counts").innerHTML =
    `<span>대기 <b>${c.pending}</b></span>` +
    `<span>승인 <b>${c.approved}</b></span>` +
    `<span>미라벨 <b>${c.unlabeled}</b></span>` +
    `<span>건너뜀 <b>${c.skipped}</b></span>` +
    (c.trashed ? `<span class="trash-count">휴지통 <b>${c.trashed}</b></span>` : "");
  $("#btn-predict").disabled = false;
  renderClassList();
  // 클래스가 바뀌면(스캔·클래스 추가) 필터 목록도 다시 그린다. 없어진 클래스는 선택에서 뺀다.
  [...S.filterClasses].filter((i) => i >= S.classes.length).forEach((i) => S.filterClasses.delete(i));
  renderClassFilter();
  loadCameras();        // 스캔으로 새 카메라가 들어올 수 있다. 실패해도 검수는 되어야 하므로 기다리지 않는다
  loadDomains();
  loadSplits();         // 학습·split 이 미배정을 배정하면 바뀐다
  loadTags();
  if (st.job && st.job.status === "running" && !S.jobPoll) watchJob(st.job.id);
}

/** 이어서 학습할 체크포인트가 있을 때만 버튼을 보여준다.
 *
 * 페이지를 열 때와 작업이 끝날 때만 부른다(WSL 버튼과 같다). 예전에는 loadState 에서 불러
 * 박스를 저장할 때마다 서버가 runs/ 의 체크포인트(last.pt 여러 개)를 열었다 — 한 번에
 * 0.24~0.36초 (2026-09-24 측정). 체크포인트는 학습이 끝나거나 멈출 때만 바뀐다. */
async function refreshResumeButton() {
  const btn = $("#btn-resume");
  try {
    const p = await api("/api/resume-point");
    S.resume = p.available ? p : null;
  } catch {
    S.resume = null;
  }
  btn.classList.toggle("hidden", !S.resume);
  if (S.resume) {
    btn.textContent = `이어서 학습 (${S.resume.epoch}/${S.resume.total})`;
    btn.title = S.resume.summary;
  }
  syncModelDot();
}

/** '모델' 메뉴 단추의 점 — 이어서 학습할 것이 있으면 메뉴를 열지 않아도 보이게 (2026-11-04). */
function syncModelDot() {
  const any = ["#btn-resume", "#btn-resume-wsl"].some((s) => !$(s).classList.contains("hidden"));
  $("#model-dot").classList.toggle("hidden", !any);
}

/** 목록을 묶음 단위로 다시 구성한다. keepOpen 에 있는 묶음 번호는 열린 채로 둔다. */
async function loadImageList(keepOpen = null) {
  const status = $("#filter-status").value;
  // '전체' 는 묶음 하나에 다 담는다 (2026-11-03 사용자 요청)
  const chunkValue = $("#chunk-size").value;
  const size = chunkValue === "all" ? 1e9 : Number(chunkValue) || 500;
  $("#filter-order").disabled = status === "ids";   // 지정한 id 는 넣은 순서 그대로
  $("#cf-toggle").disabled = status === "ids";      // id 목록은 클래스 필터를 무시한다
  $("#filter-camera").disabled = status === "ids";  // 카메라 필터도 마찬가지
  $("#filter-domain").disabled = status === "ids";
  $("#filter-split").disabled = status === "ids";
  $("#filter-tag").disabled = status === "ids";
  if (status === "ids") $("#cf-panel").classList.add("hidden");
  renderFilterSummary();

  const openBefore = keepOpen ?? new Set(
    S.chunks.filter((c) => c.open).map((c) => c.start)
  );

  // 첫 페이지를 받아 전체 개수를 알아낸다.
  const first = await api(listUrl(size, 0));
  S.total = first.total;
  S.missingIds = first.missing || [];

  S.chunks = [];
  for (let start = 0; start < S.total; start += size) {
    S.chunks.push({
      start,
      end: Math.min(start + size, S.total),
      open: false,
      images: [],
    });
  }
  if (S.chunks.length) {
    S.chunks[0].images = first.images;
  }

  // 이전에 열려 있던 묶음을 복원. 아무것도 없으면 첫 묶음만 연다.
  const wanted = openBefore.size ? openBefore : new Set([0]);
  for (let i = 0; i < S.chunks.length; i++) {
    if (!wanted.has(S.chunks[i].start)) continue;
    if (!S.chunks[i].images.length) await fetchChunk(i);
    S.chunks[i].open = true;
  }
  if (!S.chunks.some((c) => c.open) && S.chunks.length) S.chunks[0].open = true;

  S.images = flatImages();
  renderImageList();
  if (VIEW === "gallery") await galleryLoad();   // 두 보기는 같은 필터를 쓴다 — 다시 받을 때 함께
  return S.images;
}

/** 저장하지 못한 수정이 있으면 알리고 false. 넘어가면 그 수정은 사라지므로 멈춘다. */
async function saveBeforeLeaving() {
  if (await flushSave()) return true;
  toast("저장하지 못한 수정이 있어 넘어가지 않았습니다. 서버를 확인한 뒤 다시 시도하세요.", true);
  return false;
}

async function loadImage(id) {
  if (!(await saveBeforeLeaving())) return;
  const meta = await api(`/api/images/${id}`);
  // 받아 오는 사이에 지금 이미지를 또 고쳤을 수 있다. 바꾸기 직전에 한 번 더 저장한다.
  if (!(await saveBeforeLeaving())) return;
  S.cur = meta;
  // cls0 = 열었을 때의 클래스. '같은 자리 일괄 수정' 이 원래 클래스로 짝을 찾는 데 쓴다.
  // 저장 페이로드에는 들어가지 않는다 (saveBoxes 가 필드를 골라 보낸다).
  S.boxes = meta.boxes.map((b) => ({ ...b, cls0: b.cls }));
  S.sel = -1;
  S.undo = [];
  S.dirty = false;
  S.loadedCount = meta.boxes.length;
  S.ghost = null;
  S.history = [];
  setHistCount(meta.n_history);
  if (S.histOpen) { renderHistory(); loadHistory(); }
  S.pred = null;
  S.findings = [];
  S.predIgnore = new Set();
  S.findHot = -1;
  if (S.predOn) fetchPreds();

  const skipBtn = $("#btn-skip");
  if (meta.status === "pending") {
    skipBtn.innerHTML = '건너뛰기 <kbd>S</kbd>';
    skipBtn.title = "이 이미지를 '건너뜀'으로 표시하고 다음으로";
  } else {
    skipBtn.innerHTML = '다음 <kbd>S</kbd>';
    skipBtn.title = "상태를 바꾸지 않고 다음 이미지로";
  }

  // 휴지통에 있는 이미지는 '복원'과 '비우기'를, 그 외에는 '휴지통'을 보여준다.
  const inTrash = meta.status === "trashed";
  $("#btn-trash").classList.toggle("hidden", inTrash);
  $("#btn-restore").classList.toggle("hidden", !inTrash);
  $("#btn-purge").classList.toggle("hidden", !inTrash);
  $("#btn-approve").classList.toggle("hidden", inTrash);

  $("#img-name").textContent = meta.name;
  $("#img-meta").textContent =
    `${meta.width}×${meta.height} · ${meta.status} · ${meta.split || "미배정"} · 불확실도 ${meta.uncertainty}` +
    (meta.model_version ? ` · ${meta.model_version}` : "") +
    // 출처(docs/provenance.md): 시각은 카메라 현지 시각 그대로(시간대 표기는 뗀다)
    (meta.camera ? ` · 카메라 ${meta.camera}${meta.camera_place ? `(${meta.camera_place})` : ""}` +
      (meta.captured_at ? ` ${meta.captured_at.replace("T", " ").slice(0, 19)}` : " · 촬영 시각 모름") : "");
  renderDomainBar();
  renderTagList();
  $("#save-state").textContent = "";
  $("#save-state").className = "save-state";
  $("#empty-hint").classList.add("hidden");

  const img = new Image();
  img.onload = () => {
    // 빨리 넘기면 앞 이미지가 늦게 도착할 수 있다. 지금 이미지가 아니면 버린다.
    if (!S.cur || S.cur.id !== id) return;
    // 화면 좌표는 **브라우저가 그린 크기**로 계산한다. EXIF 회전이 있는 사진은 DB 크기
    // (회전 전 원시 크기, 예: 4000x3000)와 브라우저가 그린 크기(3000x4000)가 뒤바뀌어
    // 찌그러져 보였다. 박스는 정규화 좌표이고 학습(Ultralytics)도 회전을 적용한 화면
    // 기준이라, 이 크기로 그리면 라벨 값은 그대로 두고 모양만 바로잡힌다
    // (2026-09-23, 정사각이 아닌 4장: id 1050·1126·1692·3181).
    S.cur.width = img.naturalWidth || S.cur.width;
    S.cur.height = img.naturalHeight || S.cur.height;
    S.img = img; S.pendingFit = true; fitView(); refresh();
  };
  img.onerror = () => { if (S.cur && S.cur.id === id) toast("이미지를 불러오지 못했습니다.", true); };
  // 예전 스키마에서는 id 가 재사용됐다(CLAUDE.md 함정 11). URL 에 갱신 시각을 붙여 캐시를
  // 가른다. 서버도 Cache-Control: no-cache 를 보내지만, 그 헤더가 없던 시절에 이미 캐시된
  // 항목은 브라우저가 서버에 묻지도 않고 내주므로 URL 자체가 달라져야 한다.
  const ver = encodeURIComponent(meta.updated_at || "");
  img.src = url(`/api/images/${id}/file${ver ? `?v=${ver}` : ""}`);

  markCurrent();
}

/* ───────────────────────── 검수 동작 ───────────────────────── */

/** 상태를 바꾸고 다음 이미지로. status=null 이면 상태는 그대로 두고 넘어가기만 한다. */
async function setStatus(status) {
  if (!S.cur) return;
  // 저장에 실패한 채로 승인하면 화면과 다른 박스가 학습에 들어간다.
  if (!(await saveBeforeLeaving())) return;

  const id = S.cur.id;
  const before = S.images.map((im) => im.id);

  if (status === "approved") {
    const overlaps = selfOverlaps();
    if (overlaps.length) {
      const lines = overlaps.slice(0, 5).map(([i, j, v]) =>
        `  ${clsName(S.boxes[i].cls)} · ${clsName(S.boxes[j].cls)}  (IoU ${v.toFixed(2)})`);
      const msg = `같은 자리에 박스가 겹쳐 있습니다 ${overlaps.length}쌍:\n${lines.join("\n")}\n\n`
        + "같은 물체를 두 번 그렸다면 하나를 지우세요. 오토바이와 탄 사람처럼 진짜로 겹친 것이면 "
        + "그대로 승인해도 됩니다.\n\n그래도 승인할까요?";
      if (!confirm(msg)) return;
    }
  }

  if (status) {
    await api(`/api/images/${id}/status`, { method: "POST", body: { status } });
    await loadState();
    // 다시 받으면 그새 필터에 맞지 않게 된 줄(Q·W·E·T)도 함께 빠진다. 보던 자리는 화면에서 그대로
    const after = before.slice(before.indexOf(id) + 1);
    await keepCurrentRow(async () => {
      await loadImageList();
      await openChunkWith(after);       // 넘어갈 줄이 닫힌 묶음으로 밀렸으면 펼친다
    }, () => nextAfter(id, before));
  }
  await advanceFrom(id, before);
}

/** 다시 받기 전 목록 before(id 순서)에서 id 뒤에 있던 줄 가운데 지금 목록에 남은 첫 줄의 id.
 * id 가 before 에 없으면 맨 앞부터 찾는다. */
function nextAfter(id, before) {
  const now = new Set(S.images.map((im) => im.id));
  return before.slice(before.indexOf(id) + 1).find((x) => now.has(x)) ?? null;
}

/** id 다음 이미지로. 목록을 다시 받았으면 before 에 다시 받기 전 순서를 준다. 예전에는 '빠졌으면
 * 같은 순번, 남았으면 한 칸 뒤' 로 골라, 그새 빠진 줄(필터에서 빠진 채 남겨 둔 줄, 2026-11-04)이
 * 앞에 있으면 그만큼 건너뛰었고 '최근 수정순' 에서 승인하면(맨 위로 옮겨 간다) 맨 위 둘째 줄로 갔다. */
async function advanceFrom(id, before = S.images.map((im) => im.id)) {
  const next = nextAfter(id, before);
  if (next != null) {
    await loadImage(next);
  } else if (S.images.length) {
    toast("이 묶음의 마지막입니다.");
  } else {
    toast("표시할 이미지가 없습니다.");
  }
}

/** 건너뛰기: 검수 대기 중인 이미지만 '건너뜀'으로 내리고, 그 외에는 단순히 다음으로. */
async function skipOrAdvance() {
  if (!S.cur) return;
  await setStatus(S.cur.status === "pending" ? "skipped" : null);
}

function step(delta) {
  if (!S.images.length) return;
  const idx = S.images.findIndex((im) => S.cur && im.id === S.cur.id);
  const next = Math.min(S.images.length - 1, Math.max(0, (idx < 0 ? 0 : idx) + delta));
  loadImage(S.images[next].id);
}

function setClass(i) {
  if (i < 0 || i >= S.classes.length) return;
  S.curClass = i;
  const b = S.sel >= 0 ? S.boxes[S.sel] : null;
  // 클래스를 실제로 바꿨을 때만 수정이다. 사람이 정한 클래스이므로 신뢰도도 지운다 — 예전에는
  // 옛 클래스의 신뢰도가 남아, 0.5 미만이면 점선으로 그려지고 검수 순서 점수에도 섞였다
  // (README 는 '수동' 이 된다고 적고 있었다, 2026-09-24). 같은 클래스를 다시 누르면 아무것도 안 한다.
  if (b && b.cls !== i) {
    pushUndo();
    Object.assign(b, { cls: i, conf: null, origin: "human" });
    markDirty();
  }
  refresh();
}

/* ───────────────────────── 작업(job) ───────────────────────── */

async function startJob(kind, body = {}) {
  try {
    const job = await api(`/api/jobs/${kind}`, { method: "POST", body });
    $("#job").classList.remove("hidden");
    watchJob(job.id);
  } catch (err) {
    toast(err.message, true);
  }
}

function watchJob(id) {
  clearInterval(S.jobPoll);
  $("#job").classList.remove("hidden");
  const tick = async () => {
    let job;
    try {
      job = await api(`/api/jobs/${id}`);
    } catch {
      clearInterval(S.jobPoll);
      return;
    }
    $("#job-kind").textContent = { scan: "폴더 스캔", predict: "자동 라벨링", train: "학습", resume: "이어서 학습", train_wsl: "WSL 학습", resume_wsl: "WSL 이어하기", tidy: "정리", purge: "휴지통 비우기", export: "내보내기" }[job.kind] || job.kind;
    $("#job-status").textContent = { running: "진행 중…", done: "완료", error: "실패" }[job.status];
    $("#job-status").className = job.status;
    const stopBtn = $("#job-stop");
    // 되돌릴 수 있는 작업만 중단을 노출한다. 학습은 중간 가중치가 남는다.
    const isWsl = job.kind === "train_wsl" || job.kind === "resume_wsl";
    const stoppable = job.status === "running"
      && (job.kind === "train" || job.kind === "resume" || isWsl);
    stopBtn.classList.toggle("hidden", !stoppable);
    stopBtn.disabled = job.cancel_requested;
    stopBtn.textContent = job.cancel_requested ? "중단 중…" : "중단";
    stopBtn.onclick = async () => {
      const msg = "학습을 중단할까요?\n"
        + "현재 배치까지 마친 뒤 검증과 가중치 저장을 끝내고 종료합니다."
        + (isWsl ? "\n\n(WSL 쪽에 STOP 파일을 만들어 멈춥니다. 이어 학습할 수 있습니다.)" : "");
      if (!confirm(msg)) return;
      try { await api(`/api/jobs/${job.id}/stop`, { method: "POST" }); }
      catch (err) { toast(err.message, true); }
    };

    $("#job-progress").textContent = job.progress || "";
    const log = $("#job-log");
    log.textContent = job.log.join("\n") + (job.error ? `\n\n${job.error}` : "");
    log.scrollTop = log.scrollHeight;

    if (job.status !== "running") {
      clearInterval(S.jobPoll);
      $("#job-stop").classList.add("hidden");
      const early = job.result && job.result.stopped_early;
      toast(
        job.status === "done"
          ? `${$("#job-kind").textContent} ${early ? "중단됨 (가중치는 저장됨)" : "완료"}`
          : `실패: ${job.error}`,
        job.status === "error"
      );
      await loadState();
      refreshResumeButton();
      await loadImageList();
      if (!S.cur && S.images.length) await loadImage(S.images[0].id);
      if (isWsl) refreshWslButtons();
    }
  };
  tick();
  S.jobPoll = setInterval(tick, 1200);
}

/**
 * WSL 학습 버튼 상태를 맞춘다. loadState 마다 부르지 않는 이유: 서버가 WSL 쪽
 * 체크포인트(60MB)를 \\wsl.localhost 로 열어 보는데, loadState 는 박스를 저장할
 * 때마다 불린다. 페이지를 열 때와 WSL 작업이 끝날 때만 새로 고친다.
 */
async function refreshWslButtons() {
  const train = $("#btn-train-wsl");
  const resume = $("#btn-resume-wsl");
  try {
    S.wsl = await api("/api/wsl");
  } catch {
    S.wsl = { available: false, reason: "WSL 상태를 확인하지 못했습니다" };
  }
  train.disabled = !S.wsl.available;
  train.title = S.wsl.available
    ? "WSL(Linux)에서 학습합니다. 같은 설정에서 epoch 258초 → 62초로 약 4배 빠릅니다."
    : `WSL 학습을 쓸 수 없습니다: ${S.wsl.reason}`;
  const r = S.wsl.available ? S.wsl.resume : null;
  resume.classList.toggle("hidden", !r);
  if (r) resume.textContent = `WSL 이어하기 (${r.epoch}/${r.total})`;
  syncModelDot();
}

/* ───────────────────────── 이벤트 배선 ───────────────────────── */

$("#btn-scan").onclick = () => startJob("scan");
$("#btn-predict").onclick = () => startJob("predict", { include_pending: false });
$("#btn-train").onclick = () => {
  if (confirm("승인된 라벨로 모델을 재학습합니다. GPU를 오래 점유합니다. 계속할까요?")) {
    startJob("train", {});
  }
};
$("#btn-resume").onclick = () => {
  const p = S.resume;
  if (!p) { toast("이어서 학습할 체크포인트가 없습니다."); return; }
  let msg = `${p.epoch}/${p.total} epoch 까지 마친 학습을 ${p.epoch + 1} epoch 부터 이어서 진행합니다.\n\n`
    + `· 체크포인트: ${p.kind}\n`
    + "· 데이터셋 폴더(config 의 data.dataset)가 남아 있으면 그대로 씁니다 — 그 사이 검수한 라벨은 반영되지 않습니다\n"
    + "  (지워졌으면 DB 로 다시 만듭니다. DB 가 그대로면 원래와 같습니다)\n"
    + "· 하이퍼파라미터는 체크포인트에 저장된 값을 씁니다 (config.yaml 수정은 대부분 무시됩니다)\n";
  if (p.lost) {
    msg += `\n주의: 이 체크포인트 이후 ${p.lost} epoch 분이 버려집니다.\n`;
  }
  msg += "\n계속할까요?";
  if (confirm(msg)) startJob("resume", {});
};

$("#btn-train-wsl").onclick = () => {
  const msg = "WSL(Linux)에서 모델을 재학습합니다.\n\n"
    + "· 승인된 라벨로 데이터셋을 새로 만들어 WSL 로 복사한 뒤 학습합니다\n"
    + "· Windows 학습보다 약 4배 빠릅니다 (epoch 258초 → 62초, 실측)\n"
    + "· GPU 를 오래 점유합니다. 끝나면 models/best.pt 가 교체됩니다 (이전 것은 best.prev.pt)\n\n"
    + "계속할까요?";
  if (confirm(msg)) startJob("train_wsl", {});
};
$("#btn-resume-wsl").onclick = () => {
  const r = S.wsl && S.wsl.resume;
  if (!r) { toast("WSL 쪽에 이어서 학습할 체크포인트가 없습니다."); return; }
  let msg = `WSL 에서 ${r.epoch}/${r.total} epoch 까지 마친 학습을 ${r.epoch + 1} epoch 부터 이어갑니다.\n\n`
    + `· 폴더: ${r.run}   · 체크포인트: ${r.kind}\n`
    + "· 데이터셋을 다시 만들지 않습니다 — 그 사이 검수한 라벨은 반영되지 않습니다\n";
  if (r.lost) msg += `\n주의: 이 체크포인트 이후 ${r.lost} epoch 분이 버려집니다.\n`;
  if (confirm(msg + "\n계속할까요?")) startJob("resume_wsl", {});
};

$("#btn-tidy").onclick = () => {
  // 폴더는 서버가 알려 준 이 프로젝트의 실제 경로 (09-21 폴더 이동 뒤 옛 경로가 박혀 있었다)
  const p = S.paths || {};
  const src = p.unlabeled || "자동 라벨링 폴더", dst = p.labeled || "정리 폴더";
  const msg = "검수 완료된 이미지를 자동 라벨링 폴더 밖으로 옮깁니다.\n\n"
    + `· 대상: ${src} 안의 '승인' 이미지 (시드 데이터는 건드리지 않습니다)\n`
    + `· 옮길 곳: ${dst}/images, 라벨은 ${dst}/labels 에 함께 저장\n`
    + "· 파일을 실제로 이동하며, DB 경로도 같이 갱신됩니다\n"
    + "· 원본 파일이 사라진 DB 항목도 함께 삭제됩니다\n\n"
    + "계속할까요?";
  if (confirm(msg)) startJob("tidy", {});
};

$("#btn-export").onclick = () => {
  // 기본은 export/<프로젝트>. 세 프로젝트가 'export' 하나를 같이 쓰면 라벨이 섞였다 (2026-09-24).
  const dir = prompt("내보낼 폴더 (저장소 기준 상대경로).\n"
    + "비어 있는 폴더나 예전 내보내기 폴더여야 합니다 — 예전 내보내기면 비우고 새로 씁니다.",
    (S.paths && S.paths.export) || "export");
  if (dir) startJob("export", { out_dir: dir, with_images: confirm("이미지도 함께 복사할까요?") });
};
$("#job-close").onclick = () => { clearInterval(S.jobPoll); $("#job").classList.add("hidden"); };

$("#btn-audit").onclick = () => openAudit();
$("#btn-same").onclick = () => openSamePlace();
$("#btn-pred").onclick = () => togglePred();
$("#hist-head").onclick = () => toggleHistory();
$("#modal-close").onclick = closeModal;
// 바깥(어두운 배경)을 누르면 닫는다. 안쪽을 누른 것은 무시한다.
$("#modal").addEventListener("mousedown", (e) => { if (e.target.id === "modal") closeModal(); });

$("#btn-approve").onclick = () => setStatus("approved");
$("#btn-trash").onclick = () => setStatus("trashed");
$("#btn-restore").onclick = () => setStatus("pending");
$("#btn-purge").onclick = () => {
  const n = (S.counts && S.counts.trashed) || 0;
  if (!n) { toast("휴지통이 비어 있습니다."); return; }
  const msg = `휴지통의 ${n}장을 영구 삭제합니다.\n\n`
    + "· 원본 이미지 파일이 디스크에서 삭제됩니다\n"
    + "· 라벨과 검수 이력도 함께 사라집니다\n"
    + "· 되돌릴 수 없습니다\n\n"
    + "계속할까요?";
  if (confirm(msg)) startJob("purge", {});
};
$("#btn-skip").onclick = () => skipOrAdvance();
$("#btn-sam").onclick = () => setMode(S.mode === "sam" ? "box" : "sam");
$("#btn-prev").onclick = () => step(-1);
$("#btn-fit").onclick = fitView;

async function reloadList() {
  await loadImageList(new Set([0]));   // 필터가 바뀌면 첫 묶음부터 다시
  if (S.images.length) await loadImage(S.images[0].id);
}

/** '새로 받기' (2026-11-04 사용자 요청): 목록을 지금 필터로 다시 받는다 — Q·W·E·T 로 필터에 맞지
 * 않게 된 채 남겨 둔 줄이 빠진다. 보던 자리는 화면에서 그대로 두고, 지금 이미지가 빠졌으면 원래
 * 그 뒤에 있던 줄 가운데 남은 첫 줄로 간다. */
async function refreshList() {
  if (S.listBusy || S.domainBusy || S.tagBusy) return;
  S.listBusy = true;
  try {
    await loadState();
    const id = S.cur ? S.cur.id : null;
    const before = S.images.map((im) => im.id);
    const n0 = S.total;
    await keepCurrentRow(async () => {
      await loadImageList();
      if (id != null) await openChunkWith([id, ...before.slice(before.indexOf(id) + 1)]);
    }, id == null ? null : () => nextAfter(id, before));
    toast(`목록을 새로 받았습니다 — ${n0.toLocaleString()} → ${S.total.toLocaleString()}장`);
    if (id != null && !S.images.some((im) => im.id === id)) await advanceFrom(id, before);
  } catch (err) {
    toast(`목록을 받지 못했습니다: ${err.message}`, true);
  } finally {
    S.listBusy = false;
  }
}
$("#btn-reload-list").onclick = (e) => {
  e.currentTarget.blur();   // 포커스가 남으면 Space·Enter 가 승인과 함께 이 단추까지 누를 수 있다
  refreshList();
};
$("#filter-status").onchange = reloadList;
$("#filter-order").onchange = reloadList;
$("#chunk-size").onchange = reloadList;
$("#filter-camera").onchange = () => {
  S.filterCamera = $("#filter-camera").value;
  $("#filter-camera").classList.toggle("on", !!S.filterCamera);
  reloadList();
};
$("#filter-domain").onchange = () => {
  S.filterDomain = $("#filter-domain").value;
  $("#filter-domain").classList.toggle("on", !!S.filterDomain);
  reloadList();
};
$("#filter-split").onchange = () => {
  S.filterSplit = $("#filter-split").value;
  $("#filter-split").classList.toggle("on", !!S.filterSplit);
  reloadList();
};
$("#filter-tag").onchange = () => {
  S.filterTag = $("#filter-tag").value;
  $("#filter-tag").classList.toggle("on", !!S.filterTag);
  reloadList();
};
$("#btn-add-tag").onclick = () => createTag();
$("#new-tag").addEventListener("keydown", (e) => {
  if (e.key === "Enter") createTag();
  if (e.key === "Escape") e.target.blur();
});
$("#tag-q").addEventListener("input", renderTagList);
$("#tag-q").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.target.value = ""; renderTagList(); e.target.blur(); }
});
$("#filters-head").onclick = (e) => { if (!e.target.closest("button")) toggleFilters(); };
$("#filters-clear").onclick = clearFilters;

/* 상단 메뉴 (2026-11-04) — 항목을 누르면 닫히고, 밖을 누르거나 Esc 로도 닫힌다. 항목의 onclick(위
 * btn-scan 등)이 먼저 돌고 그 뒤 닫는다. 포커스도 놓는다 — 남으면 Space·Enter 가 그 항목을 또 누를 수 있다. */
function closeMenus(except = null) {
  document.querySelectorAll(".menu").forEach((m) => {
    if (m === except) return;
    m.querySelector(".menu-list").classList.add("hidden");
    m.querySelector(".menu-btn").setAttribute("aria-expanded", "false");
  });
}
document.querySelectorAll(".menu").forEach((m) => {
  const btn = m.querySelector(".menu-btn"), list = m.querySelector(".menu-list");
  btn.onclick = (e) => {
    e.stopPropagation();
    const open = list.classList.contains("hidden");
    closeMenus(m);
    list.classList.toggle("hidden", !open);
    btn.setAttribute("aria-expanded", String(open));
    btn.blur();
  };
  list.addEventListener("click", (e) => {
    const item = e.target.closest(".menu-item");
    if (item) { item.blur(); closeMenus(); }
  });
});
document.addEventListener("click", (e) => { if (!e.target.closest(".menu")) closeMenus(); });

/* 오른쪽 칸 접기 (2026-11-04) — 머리를 누르면 접고 편다. 접은 것은 다시 열어도 남는다(변경 내역은 따로 H). */
const SEC_KEY = "secClosed";
function loadSecClosed() {
  try { return JSON.parse(localStorage.getItem(SEC_KEY) || "{}") || {}; } catch { return {}; }
}
function setSecClosed(sec, closed) {
  sec.classList.toggle("collapsed", closed);
  sec.querySelector(".sec-head .caret").textContent = closed ? "▸" : "▾";
}
document.querySelectorAll(".sec").forEach((sec) => {
  setSecClosed(sec, !!loadSecClosed()[sec.dataset.sec]);
  sec.querySelector(".sec-head").addEventListener("click", (e) => {
    if (e.target.closest("button, input, kbd")) return;   // 머리 안의 '같은 자리' 단추 등
    const closed = !sec.classList.contains("collapsed");
    setSecClosed(sec, closed);
    const all = loadSecClosed();
    all[sec.dataset.sec] = closed;
    try { localStorage.setItem(SEC_KEY, JSON.stringify(all)); } catch { /* 저장 못 해도 그만 */ }
  });
});

/* 보기 전환 (2026-11-04) — 라벨링 ↔ 이미지 보기(gallery.js). 한 페이지 안이라 필터·지금 이미지·되돌리기가
 * 그대로 남는다. 마지막 보기는 다시 열어도 남긴다(프로젝트마다). */
const VIEW_KEY = `view:${BASE}`;
async function setView(v) {
  if (v === VIEW) return;
  if (VIEW === "gallery") galleryHide();
  VIEW = v;
  document.body.classList.toggle("view-gallery", v === "gallery");
  document.querySelectorAll("#view-switch .seg-btn").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  try { localStorage.setItem(VIEW_KEY, v); } catch { /* 저장 못 해도 그만 */ }
  closeMenus();
  if (v === "gallery") await galleryShow();
  else { resizeCanvas(); markCurrent(); }
}
document.querySelectorAll("#view-switch .seg-btn").forEach((b) => {
  b.onclick = () => { b.blur(); setView(b.dataset.view); };
});
document.querySelectorAll(".domain-bar [data-domain]").forEach((b) => {
  b.onclick = () => setDomain(b.dataset.domain);
});
// 목록 필터는 고른 뒤 포커스를 놓는다 — 남아 있으면 ↑·↓ 가 이미지 대신 필터 값을 바꾼다
document.querySelectorAll(".panel-left select").forEach((s) => s.addEventListener("change", () => s.blur()));

$("#cf-toggle").onclick = (e) => {
  e.stopPropagation();
  const panel = $("#cf-panel");
  const open = panel.classList.toggle("hidden");
  $("#cf-toggle").setAttribute("aria-expanded", String(!open));
};
$("#cf-clear").onclick = () => {
  S.filterClasses.clear();
  S.filterEmpty = false;
  syncClassFilter();
  reloadList();
};
// 패널 밖을 누르면 닫는다 (패널 안의 체크박스는 그대로 동작해야 하므로 걸러낸다)
document.addEventListener("click", (e) => {
  const panel = $("#cf-panel");
  if (panel.classList.contains("hidden")) return;
  if (panel.contains(e.target) || e.target.id === "cf-toggle") return;
  panel.classList.add("hidden");
  $("#cf-toggle").setAttribute("aria-expanded", "false");
});

$("#goto-ids").addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  const ids = parseIds(e.target.value);
  if (!ids) { toast("id 는 숫자만 쉼표로 구분해 넣어 주세요.", true); return; }
  e.target.blur();   // 곧바로 단축키(화살표·Space)를 쓸 수 있게
  try {
    await openIds(ids);
  } catch (err) {
    toast(err.message, true);
  }
});
window.addEventListener("hashchange", () => {
  const ids = idsFromHash();
  if (ids) openIds(ids).catch((err) => toast(err.message, true));
});

$("#btn-add-class").onclick = async () => {
  const name = $("#new-class").value.trim();
  if (!name) return;
  try {
    const res = await api("/api/classes", { method: "POST", body: { name } });
    S.classes = res.classes;
    $("#new-class").value = "";
    setClass(res.index);
  } catch (err) {
    toast(err.message, true);
  }
};
$("#new-class").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#btn-add-class").click();
});

window.addEventListener("keydown", (e) => {
  // 대화상자가 떠 있는 동안에는 Esc 말고는 아무것도 하지 않는다 — Space 가 뒤에서
  // 승인을 눌러 버리면 안 된다.
  if (modalOpen()) {
    if (e.key === "Escape") { e.preventDefault(); closeModal(); }
    return;
  }
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "Escape" && document.querySelector(".menu-list:not(.hidden)")) { closeMenus(); return; }
  // V: 라벨링 ↔ 이미지 보기. 이미지 보기에서는 키를 그쪽(gallery.js)에 넘긴다 — Space 가 뒤에 숨은
  // 라벨링 이미지를 승인하면 안 된다 (2026-11-04)
  if (!e.ctrlKey && !e.metaKey && !e.altKey && (e.key === "v" || e.key === "V" || e.key === "ㅍ")) {
    e.preventDefault();
    setView(VIEW === "gallery" ? "label" : "gallery");
    return;
  }
  if (VIEW === "gallery") { galleryKey(e); return; }

  // Ctrl 을 누르고 있는 동안만 모드를 반대로. (누르는 순간부터 커서도 바뀐다)
  if (e.key === "Control" || e.ctrlKey) {
    if (!S.ctrl) { S.ctrl = true; renderMode(); }
    if (e.key === "Control") return;
  }

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;

  switch (e.key) {
    case " ":
    case "Enter":
      e.preventDefault(); setStatus("approved"); break;
    case "s": case "S": case "ㄴ":
      e.preventDefault(); skipOrAdvance(); break;
    // 목록이 세로라 위·아래로 넘긴다 (2026-11-03 사용자 요청, 전에는 ←·→)
    case "ArrowDown": e.preventDefault(); step(1); break;
    case "ArrowUp":   e.preventDefault(); step(-1); break;
    case "Delete": case "Backspace":
      if (S.sel >= 0) { e.preventDefault(); pushUndo(); S.boxes.splice(S.sel, 1); S.sel = -1; markDirty(); refresh(); }
      break;
    case "Escape": selectBox(-1); break;
    case "f": case "F": fitView(); break;
    case "d": case "D": case "ㅇ": e.preventDefault(); setMode("pan"); break;
    case "b": case "B": case "ㅠ": e.preventDefault(); setMode("box"); break;
    case "c": case "C": case "ㅊ": e.preventDefault(); setMode("sam"); break;
    case "r": case "R": case "ㄱ": e.preventDefault(); samRefineSelected(); break;
    case "h": case "H": case "ㅗ": e.preventDefault(); toggleHistory(); break;
    case "g": case "G": case "ㅎ": e.preventDefault(); openSamePlace(); break;
    case "p": case "P": case "ㅔ": e.preventDefault(); togglePred(); break;
    case "q": case "Q": case "ㅂ": case "ㅃ": e.preventDefault(); setDomain("cctv"); break;
    case "w": case "W": case "ㅈ": case "ㅉ": e.preventDefault(); setDomain("noncctv"); break;
    case "e": case "E": case "ㄷ": case "ㄸ": e.preventDefault(); setDomain("drone"); break;
    case "t": case "T": case "ㅅ": case "ㅆ": e.preventDefault(); tagAndNext(); break;
    case "a": case "A": case "ㅁ":
      // 예측 비교의 맨 위 항목 받기 (빠짐 = 추가, 클래스 = 바꾸기, 박스 = 맞추기)
      if (S.predOn) {
        e.preventDefault();
        const f = S.findings.find((x) => FIND.accept[x.kind]);
        if (f) acceptFinding(f); else toast("받을 항목이 없습니다");
      }
      break;
    default:
      if (/^[1-9]$/.test(e.key)) { e.preventDefault(); setClass(Number(e.key) - 1); }
  }
});

window.addEventListener("keyup", (e) => {
  if (e.key === "Control" || !e.ctrlKey) {
    if (S.ctrl) { S.ctrl = false; renderMode(); }
  }
});
// 창 밖으로 포커스가 나가면 Ctrl 이 눌린 채로 남는 것을 막는다.
window.addEventListener("blur", () => {
  if (S.ctrl) { S.ctrl = false; renderMode(); }
});

window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe($("#canvas-wrap"));
window.addEventListener("beforeunload", (e) => {
  if (S.dirty || S.saving) { e.preventDefault(); e.returnValue = ""; }
});

/* ───────────────────────── 시작 ───────────────────────── */

/* 상단 프로젝트 탭. 목록은 루트 앱이 준다(/api/projects — BASE 를 붙이지 않는다).
 * `--config` 로 하나만 띄웠으면 그 주소가 없으므로 탭도 그리지 않는다. */
async function loadProjects() {
  const bar = $("#projects");
  if (!bar) return;
  let list = [];
  try {
    const res = await fetch("/api/projects");
    if (!res.ok) return;
    list = (await res.json()).projects || [];
  } catch (err) {
    return;                       // 탭이 없어도 검수는 되어야 한다
  }
  if (list.length < 2) return;
  bar.innerHTML = "";
  for (const p of list) {
    const a = document.createElement("a");
    a.className = "tab" + (p.base === BASE ? " on" : "");
    a.href = p.base;
    a.textContent = p.title;
    a.title = `${p.title} 프로젝트로 이동 (${p.base})`;
    bar.appendChild(a);
  }
  bar.classList.remove("hidden");
  const cur = list.find((p) => p.base === BASE);
  if (cur) document.title = `${cur.title} · 로컬 자동 라벨링`;
}

(async function main() {
  resizeCanvas();
  renderMode();
  try { if (localStorage.getItem("histOpen") === "1") toggleHistory(true); } catch { /* 없으면 접힌 채로 */ }
  try { if (localStorage.getItem("predOn") === "1") togglePred(true); } catch { /* 없으면 꺼진 채로 */ }
  try { const v = localStorage.getItem(CUR_TAG_KEY); if (v) S.curTag = Number(v); } catch { /* 없으면 안 고른 채로 */ }
  try { if (localStorage.getItem("filtersOpen") === "1") toggleFilters(true); } catch { /* 없으면 접힌 채로 */ }
  loadProjects();       // 실패해도 검수는 되어야 하므로 기다리지 않는다
  try {
    await loadState();
    loadSamStatus();      // 실패해도 검수는 되어야 하므로 기다리지 않는다
    refreshResumeButton();
    refreshWslButtons();  // 같은 이유. WSL 이 잠들어 있으면 깨우는 데 몇 초 걸린다

    // 주소에 #ids=812,904 가 있으면 그 목록부터 연다 (라벨 점검 결과를 바로 고칠 때).
    const hashIds = idsFromHash();
    if (hashIds) {
      await openIds(hashIds);
      return;
    }

    await loadImageList();

    // 검수 대기가 없으면(예: 시드만 넣은 첫 실행) 빈 화면 대신
    // 실제로 이미지가 있는 상태로 필터를 옮겨 준다.
    if (!S.images.length) {
      const fallback = ["approved", "pending", "unlabeled", "skipped", "trashed"]
        .find((s) => (S.counts?.[s] || 0) > 0);
      if (fallback) {
        $("#filter-status").value = fallback;
        await loadImageList();
      }
    }

    if (S.images.length) await loadImage(S.images[0].id);
    // 지난번에 이미지 보기에서 닫았으면 그대로 연다(#ids= 로 연 때는 라벨링 그대로)
    let last = null;
    try { last = localStorage.getItem(VIEW_KEY); } catch { /* 없으면 라벨링 */ }
    if (last === "gallery") await setView("gallery");
  } catch (err) {
    toast(`초기화 실패: ${err.message}`, true);
  }
})();
