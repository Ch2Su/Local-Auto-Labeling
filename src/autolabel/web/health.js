/* 데이터 점검 (2026-11-15 사용자 요청, 서버 health.py) — 지금 데이터가 어떻게 생겼는지 한 화면으로.
 *
 * 라벨링·이미지 보기·영상 캡처와 같은 페이지의 네 번째 보기다(app.js setView('health') · body.view-health).
 * **읽기만 한다** — GET /api/health 와 /api/health/heatmap 둘뿐이고 아무것도 저장하지 않는다. 키도 쓰지 않는다
 * (app.js keydown 이 VIEW === "health" 면 단축키를 넘기지 않는다 — 숨은 라벨링에 Space 승인이 먹으면 안 된다).
 * 막대·줄·칸을 누르면 그 이미지들을 이미지 보기로 연다: 클래스·카메라·태그·분할은 왼쪽 필터를 그렇게 맞추고(hcOpenFiltered),
 * 객체 수 막대·히트맵 칸처럼 맞는 필터가 없는 것은 id 목록으로 연다(hcOpenIds — 서버가 1,000개까지만 싣는다).
 * 수치의 뜻(범위·가려진 박스·박스 크기·묶음 수)은 health.py 머리말에 있다 — 화면 문구를 고칠 때 같이 볼 것.
 */

const H = {
  inited: false,
  data: null,       // 마지막 /api/health 응답
  heat: null,       // 마지막 히트맵 응답 — 화면에 그려진 것
  heatQ: null,      // 그 히트맵을 받은 조건(hcHeatParams) — 칸을 누르면 지금 조절 칸이 아니라 이것으로 묻는다
  seq: 0,           // 늦게 온 응답은 버린다(범위를 연달아 바꿀 때)
  heatSeq: 0,       // 히트맵을 받기 시작할 때·다른 보기로 갈 때 오른다 — 칸 누르기(hcOpenCell)의 응답도 이것으로 버린다
  cls: "",          // 히트맵 클래스("" = 전체)
  cam: "",          // 히트맵 카메라("" = 전체)
  hover: -1,        // 마우스가 올라간 칸 번호
  sizeReady: false, // 입력 크기 칸을 채웠나(첫 응답 뒤)
  savedSize: 0,
  opening: false,   // 이미지 보기로 넘기는 중 — 두 번 누르기를 버린다
};
const HC_KEY = `health:${BASE}`;
const HC_SIZES = [640, 960, 1280];
const HC_SIZE_COLORS = ["#ef5f6b", "#f0a742", "#e3d26f", "#6fb6e3", "#4c8dff"];   // 작을수록 붉게(검출하기 어려운 쪽)
const HC_SIZE_MARKS = [32, 16];   // '몇 px 미만이 몇 개' 로 따로 적을 경계 — 서버의 칸 경계에 있는 것만 적는다(hcUnder)
const HC_SPLIT_COLORS = { train: "#4c8dff", val: "#3ecf8e", test: "#f0a742", none: "#4a505c" };
const HC_SPLIT_NAMES = { train: "train", val: "val", test: "test", none: "미배정" };
const HC_DOMAIN_NAMES = { cctv: "CCTV", noncctv: "비CCTV", drone: "드론", none: "미분류" };

const hcN = (n) => Number(n || 0).toLocaleString();
function hcPct(n, total) {
  if (!total) return "–";
  const p = 100 * n / total;
  return `${p.toFixed(p > 0 && p < 10 ? 1 : 0)}%`;
}

/* ───────────────────────── 받기 ───────────────────────── */

function hcRestoreControls() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(HC_KEY) || "{}") || {}; } catch { /* 없으면 기본 */ }
  // 칸에 있는 값만 받는다(값 목록을 여기에 따로 적지 않는다 — index.html 의 option 이 서버의 SCOPES·masked 와 맞는다)
  for (const [sel, v] of [["#hc-scope", saved.scope], ["#hc-masked", saved.masked]]) {
    if ([...$(sel).options].some((o) => o.value === v)) $(sel).value = v;
  }
  H.savedSize = Number(saved.imgsz) || 0;
}

/** 조절 칸 값을 남긴다. 입력 크기 칸은 첫 응답 뒤에야 채워진다 — 그 전에는 지난번에 남긴 값을 그대로 둔다
 *  (2026-11-15 리뷰에서 고침: 첫 응답 전에 범위를 바꾸면 빈 칸을 읽어 0 을 남겼고, 다음에 열면 남겨 둔 크기가 사라졌다). */
function hcSaveControls() {
  const imgsz = H.sizeReady ? Number($("#hc-imgsz").value) || 0 : H.savedSize;
  const v = { scope: $("#hc-scope").value, masked: $("#hc-masked").value, imgsz };
  try { localStorage.setItem(HC_KEY, JSON.stringify(v)); } catch { /* 저장 못 해도 그만 */ }
}

/** 입력 크기 칸 — 학습 설정값(model.imgsz)과 640·960·1280. 첫 응답으로 설정값을 안 뒤에 채운다. */
function hcFillSizes(def, cur) {
  const sel = $("#hc-imgsz");
  const sizes = [...new Set([def, cur, ...HC_SIZES])].filter(Boolean).sort((a, b) => a - b);
  sel.innerHTML = sizes.map((s) => `<option value="${s}">${s}${s === def ? " (학습 설정)" : ""}</option>`).join("");
  sel.value = String(cur);
  H.sizeReady = true;
}

/** /api/health 의 조건. heat(hcHeatParams 의 꼴)를 주면 히트맵도 같은 요청으로 받는다 — 서버가 같은 스냅샷으로, 덮인 몫을
 *  한 번만 재서 함께 준다(2026-11-15 리뷰: 전에는 열 때마다 DB 를 두 번 읽고 두 번 쟀다). */
function hcQuery(heat = null) {
  const q = new URLSearchParams({ scope: $("#hc-scope").value, masked: $("#hc-masked").value });
  const size = H.sizeReady ? $("#hc-imgsz").value : (H.savedSize || "");
  if (size) q.set("imgsz", String(size));
  if (heat) {
    q.set("heat", "1");
    if (heat.cls !== "") q.set("cls", heat.cls);
    if (heat.cam !== "") q.set("camera", heat.cam);
  }
  return q;
}

async function hcLoad() {
  const seq = ++H.seq, heatSeq = ++H.heatSeq, p = hcHeatParams();
  $("#health").classList.add("loading");
  try {
    const d = await api(`/api/health?${hcQuery(p)}`);
    if (seq !== H.seq) return;
    H.data = d;
    if (!H.sizeReady || Number($("#hc-imgsz").value) !== d.imgsz) hcFillSizes(d.default_imgsz, d.imgsz);
    hcRender();     // 히트맵의 클래스·카메라 칸도 다시 채운다 — 고른 값이 새 범위에 없으면 '전체' 로 돌린다
    $("#hc-stamp").textContent = `${new Date().toLocaleTimeString()} 기준`;
    // 함께 온 히트맵은 물었을 때의 조건(p)으로 센 것이다. 그새 히트맵을 따로 받기 시작했거나 위에서 조건이 바뀌었으면 다시 받는다
    if (d.heatmap && heatSeq === H.heatSeq && H.cls === p.cls && H.cam === p.cam) hcShowHeat(d.heatmap, p);
    else await hcLoadHeat();
  } catch (err) {
    if (seq === H.seq) toast(`데이터 점검을 받지 못했습니다: ${err.message}`, true);
  } finally {
    if (seq === H.seq) $("#health").classList.remove("loading");
  }
}

/** setView('health') — 열 때마다 새로 센다(검수하다 돌아오면 수치가 달라져 있다). */
async function healthShow() {
  if (!H.inited) {
    H.inited = true;
    hcRestoreControls();
  }
  await hcLoad();
}

/** 다른 보기로 갈 때 — 돌고 있는 응답은 버린다. */
function healthHide() {
  H.seq++;
  H.heatSeq++;
  hcTip(null);
  $("#health").classList.remove("loading");
}

/* ───────────────────────── 그리기 ───────────────────────── */

function hcRender() {
  const d = H.data;
  hcRenderTiles(d);
  hcRenderNotes(d);
  hcRenderClasses(d);
  hcRenderSizes(d);
  hcRenderCounts(d);
  hcRenderHeatControls(d);
  hcRenderCameras(d);
  hcRenderTags(d);
  hcRenderMisc(d);
}

function hcRenderTiles(d) {
  const c = d.counts || {};
  const cams = d.cameras.filter((r) => r.key != null);
  const noCam = d.cameras.find((r) => r.key == null);
  const sp = d.splits.images;
  const assigned = sp.train + sp.val + sp.test;
  const per = cams.map((r) => r.images);
  const tiles = [
    { label: "이미지", value: hcN(d.images),
      // 검수 끝(작업 배치의 '모아서 한꺼번에' — 아직 승인 전, 2026-11-23)은 있을 때만 적는다
      sub: `승인 ${hcN(c.approved)}` + (c.reviewed ? ` · 검수 끝 ${hcN(c.reviewed)}` : "")
        + ` · 대기 ${hcN(c.pending)} · 미라벨 ${hcN(c.unlabeled)} · 건너뜀 ${hcN(c.skipped)} · 휴지통 ${hcN(c.trashed)}`,
      title: "큰 숫자는 지금 범위의 이미지 수, 아래는 범위와 상관없는 상태별 수입니다" },
    { label: "박스", value: hcN(d.boxes),
      sub: `이미지당 평균 ${d.per_image.mean} · 중앙값 ${d.per_image.median} · 최대 ${hcN(d.per_image.max)}`,
      act: d.per_image.max_id != null ? "one" : "", v: d.per_image.max_id,
      title: d.per_image.max_id != null ? `박스가 가장 많은 이미지(#${d.per_image.max_id})를 엽니다` : "" },
    { label: "박스 없는 이미지", value: hcN(d.empty),
      sub: d.images ? `${hcPct(d.empty, d.images)} — 학습에서는 배경 예시` : "",
      act: d.empty ? "bin" : "", v: 0, title: d.empty ? "박스 없는 이미지를 이미지 보기로" : "" },
    { label: "카메라", value: hcN(cams.length),
      sub: [per.length ? `많은 곳 ${hcN(Math.max(...per))}장 · 적은 곳 ${hcN(Math.min(...per))}장` : "",
            noCam ? `출처 없음 ${hcN(noCam.images)}장` : ""].filter(Boolean).join(" · ") },
    { label: "분할", value: assigned ? `${hcN(assigned)}장 배정` : "미배정",
      sub: `train ${hcN(sp.train)} · val ${hcN(sp.val)} · test ${hcN(sp.test)}${sp.none ? ` · 미배정 ${hcN(sp.none)}` : ""}` },
    { label: "가릴 영역", value: `${hcN(d.masks.images)}장`,
      sub: `영역 ${hcN(d.masks.rects)}개 · 학습에서 빠지는 박스 ${hcN(d.masks.dropped)}개` },
  ];
  $("#hc-tiles").innerHTML = tiles.map((t) => (
    `<div class="hc-tile${t.act ? " link" : ""}"${t.act ? ` data-act="${t.act}" data-v="${esc(t.v)}"` : ""}`
    + ` title="${esc(t.title || "")}"><div class="l">${esc(t.label)}</div><div class="v">${esc(t.value)}</div>`
    + `<div class="s">${esc(t.sub || "")}</div></div>`)).join("");
}

function hcRenderNotes(d) {
  const notes = [...d.notes];
  if (d.camera_error) notes.unshift({ level: "warn", text: `카메라 등록 파일을 읽지 못했습니다 — 장소·종류·고정 분할이 비어 보입니다: ${d.camera_error}` });
  $("#hc-notes").innerHTML = notes.map((n) => `<li class="${n.level === "warn" ? "warn" : "info"}">${esc(n.text)}</li>`).join("");
}

/** 분할별 몫을 한 줄 막대로. parts = {train, val, test, none}. */
function hcSplitBar(parts) {
  const total = parts.train + parts.val + parts.test + parts.none;
  if (!total) return "";
  const title = ["train", "val", "test", "none"].map((k) => `${HC_SPLIT_NAMES[k]} ${hcN(parts[k])}`).join(" · ");
  const segs = ["train", "val", "test", "none"].filter((k) => parts[k]).map((k) => (
    `<i style="width:${(100 * parts[k] / total).toFixed(2)}%;background:${HC_SPLIT_COLORS[k]}"></i>`)).join("");
  return `<span class="hc-stack" title="${esc(title)}">${segs}</span>`;
}

/** 박스 크기 칸의 이름 — **서버가 준 경계**(health.SIZE_EDGES)로 만든다: [8, 16, 32, 64] → "8px 미만"·"8–16"·…·"64px 이상".
 *  칸 k 는 [edges[k-1], edges[k]) 다(health.size_bin). 2026-11-15 리뷰: 전에는 이름과 '32px 미만' 의 칸 번호를 여기에 따로
 *  적어, 서버의 경계를 바꾸면 화면이 옛 범위로 적었다. */
function hcSizeNames(edges) {
  return [`${edges[0]}px 미만`, ...edges.slice(1).map((e, k) => `${edges[k]}–${e}`), `${edges[edges.length - 1]}px 이상`];
}

/** 짧은 변이 px 미만인 박스 수 — px 가 칸 경계에 없으면 null(셀 수 없다). */
function hcUnder(counts, edges, px) {
  const k = edges.indexOf(px);
  return k < 0 ? null : counts.slice(0, k + 1).reduce((a, b) => a + b, 0);
}

const hcSizeColor = (k) => HC_SIZE_COLORS[Math.min(k, HC_SIZE_COLORS.length - 1)];

function hcSizeBar(counts, names) {
  const total = counts.reduce((a, b) => a + b, 0);
  if (!total) return "";
  const title = counts.map((n, k) => `${names[k]} ${hcN(n)}`).join(" · ");
  const segs = counts.map((n, k) => (n ? `<i style="width:${(100 * n / total).toFixed(2)}%;background:${hcSizeColor(k)}"></i>` : "")).join("");
  return `<span class="hc-stack" title="${esc(title)}">${segs}</span>`;
}

function hcRenderClasses(d) {
  const rows = [...d.classes].sort((a, b) => b.boxes - a.boxes || a.idx - b.idx);
  const max = Math.max(1, ...rows.map((r) => r.boxes));
  // 묶음이 쓰는 분할 수(d.min_units — 서버의 health.used_splits)보다 적으면 그 분할들에 다 들어갈 수 없다
  const camHead = "이 클래스의 박스가 있는 카메라 수." + (d.min_units > 1
    ? ` 분할이 카메라째라 묶음이 ${d.min_units}곳 미만이면 ${d.splits.used.join("·")} 에 모두 들어갈 수 없습니다` : "");
  const body = rows.map((r) => {
    const few = r.boxes > 0 && r.units < d.min_units;       // 쓰는 분할 수보다 적은 묶음 — 서버(health.used_splits)가 준다
    const top = r.top ? `${hcPct(r.top.share, 1)}` : "";
    const topTitle = !r.top ? "" : r.top.cameras.length
      ? `박스가 가장 많은 곳: ${r.top.cameras.join(" + ")}` : "박스가 가장 많은 곳: 출처 없는 장면 묶음 하나";
    const cams = `${hcN(r.cameras)}${r.no_camera_boxes ? " +" : ""}`;
    const camTitle = (r.no_camera_boxes ? `출처 없는 이미지의 박스 ${hcN(r.no_camera_boxes)}개 — 분할 묶음으로는 ${hcN(r.units)}곳` : "")
      || (few ? `묶음이 ${d.min_units}곳 미만 — ${d.splits.used.join("·")} 에 모두 들어갈 수 없습니다` : "");
    return `<tr class="link${r.boxes ? "" : " zero"}" data-act="cls" data-v="${r.idx}">`
      + `<td class="nm"><i class="sw" style="background:${classColor(r.idx)}"></i>${esc(r.name)}</td>`
      + `<td class="bar"><i style="width:${(100 * r.boxes / max).toFixed(2)}%"></i></td>`
      + `<td class="num">${hcN(r.boxes)}</td><td class="num">${hcN(r.images)}</td>`
      + `<td class="num${few ? " warn" : ""}" title="${esc(camTitle)}">${cams}</td>`
      + `<td class="num" title="${esc(topTitle)}">${top}</td>`
      + `<td class="stk">${hcSplitBar(r.split)}</td>`
      + `<td class="num">${r.median_px == null ? "" : `${r.median_px}px`}</td>`
      + `<td class="num dim">${r.masked_out ? hcN(r.masked_out) : ""}</td></tr>`;
  }).join("");
  $("#hc-classes").innerHTML = `<table class="tbl hc-tbl"><thead><tr><th>클래스</th><th></th><th class="num">박스</th>`
    + `<th class="num">이미지</th><th class="num" title="${esc(camHead)}">카메라</th>`
    + `<th class="num" title="박스가 가장 많은 한 곳(분할이 통째로 옮기는 묶음 — 카메라, 출처 없는 이미지는 장면 묶음)이 이 클래스 박스에서 차지하는 몫 — 높으면 한 장면에 쏠려 있습니다">한 곳 몫</th>`
    + `<th title="이 클래스 박스의 분할별 몫 (train·val·test·미배정)">분할</th>`
    + `<th class="num" title="학습 입력(${d.imgsz})에서 박스 짧은 변의 중앙값">크기</th>`
    + `<th class="num" title="가릴 영역이 반 이상 덮어 학습 라벨에서 빠지는 박스 수${d.drop_masked ? " — 왼쪽 수치에는 들어 있지 않습니다" : " — 왼쪽 수치에 들어 있습니다"}">가려짐</th>`
    + `</tr></thead><tbody>${body}</tbody></table>`
    + `<div class="hc-legend">${["train", "val", "test", "none"].map((k) => `<span><i style="background:${HC_SPLIT_COLORS[k]}"></i>${HC_SPLIT_NAMES[k]}</span>`).join("")}</div>`;
}

function hcRenderSizes(d) {
  const counts = d.size_bins.counts, total = counts.reduce((a, b) => a + b, 0);
  $("#hc-size-sub").textContent = `— 긴 변을 ${d.imgsz} 에 맞췄을 때 박스의 짧은 변`;
  if (!total) { $("#hc-sizes").innerHTML = `<div class="dim">박스가 없습니다.</div>`; return; }
  const edges = d.size_bins.edges, names = hcSizeNames(edges);
  const legend = counts.map((n, k) => (
    `<span><i style="background:${hcSizeColor(k)}"></i>${names[k]} <b>${hcN(n)}</b> ${hcPct(n, total)}</span>`)).join("");
  const under = HC_SIZE_MARKS.map((px) => [px, hcUnder(counts, edges, px)]).filter(([, n]) => n != null).map(([px, n]) => (
    `${px}px 미만 <b>${hcN(n)}</b>개 (${hcPct(n, total)})`)).join(" · ");
  const rows = [...d.classes].filter((r) => r.boxes).sort((a, b) => a.median_px - b.median_px).map((r) => (
    `<div class="hc-row"><span class="nm"><i class="sw" style="background:${classColor(r.idx)}"></i>${esc(r.name)}</span>`
    + `${hcSizeBar(r.sizes, names)}<span class="num">${r.median_px}px</span></div>`)).join("");
  $("#hc-sizes").innerHTML = `<div class="hc-big">${hcSizeBar(counts, names)}</div><div class="hc-legend">${legend}</div>`
    + (under ? `<div class="hc-line">${under}</div>` : "")
    + `<div class="hc-sub">클래스별 (작은 것부터, 오른쪽은 중앙값)</div>${rows}`;
}

function hcBinLabel(b) {
  if (b.hi == null) return `${b.lo}+`;
  return b.lo === b.hi ? String(b.lo) : `${b.lo}–${b.hi}`;
}

function hcRenderCounts(d) {
  const max = Math.max(1, ...d.count_bins.map((b) => b.n));
  $("#hc-counts").innerHTML = `<div class="hc-cols">${d.count_bins.map((b, k) => (
    `<div class="hc-col${b.n ? " link" : ""}"${b.n ? ` data-act="bin" data-v="${k}"` : ""} title="객체 ${hcBinLabel(b)}개인 이미지 ${hcN(b.n)}장">`
    + `<span class="n">${hcN(b.n)}</span><i style="height:${Math.max(b.n ? 2 : 0, Math.round(120 * b.n / max))}px"></i>`
    + `<span class="l">${hcBinLabel(b)}</span></div>`)).join("")}</div>`
    + `<div class="hc-line">평균 <b>${d.per_image.mean}</b> · 중앙값 <b>${d.per_image.median}</b> · 최대 <b>${hcN(d.per_image.max)}</b>`
    + `${d.per_image.max_id != null ? ` <button class="linkbtn" data-act="one" data-v="${d.per_image.max_id}">#${d.per_image.max_id} 열기</button>` : ""}</div>`;
}

function hcRenderCameras(d) {
  if (!d.cameras.length) { $("#hc-cameras").innerHTML = `<div class="dim">이미지가 없습니다.</div>`; return; }
  const max = Math.max(1, ...d.cameras.map((r) => r.images));
  const body = d.cameras.map((r) => {
    const key = r.key == null ? d.none_camera : r.key;
    const split = r.pinned ? `<b class="pin">${esc(r.pinned)} 고정</b>` : "";
    const period = r.first ? (r.first === r.last ? r.first : `${r.first} ~ ${r.last}`) : "";
    return `<tr class="link" data-act="cam" data-v="${esc(key)}">`
      + `<td class="nm">${r.key == null ? `<span class="dim">출처 없음</span>` : esc(r.key)}${r.key != null && !r.registered ? ` <span class="warn" title="cameras.yaml 에 등록이 없습니다">미등록</span>` : ""}</td>`
      + `<td class="place dim" title="${esc(r.place || "")}">${esc(r.place || "")}</td>`
      + `<td class="bar"><i style="width:${(100 * r.images / max).toFixed(2)}%"></i></td>`
      + `<td class="num">${hcN(r.images)}</td><td class="num">${hcN(r.boxes)}</td>`
      + `<td class="num">${r.images ? (r.boxes / r.images).toFixed(1) : ""}</td><td class="num">${hcN(r.classes)}</td>`
      // 고정 글자와 막대를 한 줄(flex)에 — 전에는 막대가 칸 너비 100% 라 글자가 있는 줄만 다음 칸(가릴 영역)까지 밀려 나갔다(11-28)
      + `<td class="stk"><div class="hc-stk">${split}${hcSplitBar(r.splits)}</div></td>`
      + `<td class="num dim" title="가릴 영역이 있는 이미지 수 / 그 영역이 학습에서 빼는 박스 수">${r.masked ? `${hcN(r.masked)}장 / ${hcN(r.masked_out)}개` : ""}</td>`
      + `<td class="dim">${esc(period)}</td></tr>`;
  }).join("");
  $("#hc-cameras").innerHTML = `<table class="tbl hc-tbl"><thead><tr><th>카메라</th><th>장소</th><th></th><th class="num">이미지</th>`
    + `<th class="num">박스</th><th class="num">장당</th><th class="num" title="이 카메라에 박스가 있는 클래스 수">클래스</th>`
    + `<th title="이 카메라 이미지의 분할 — cameras.yaml 에서 고정했으면 그 분할">분할</th><th class="num">가릴 영역</th><th>촬영일</th>`
    + `</tr></thead><tbody>${body}</tbody></table>`;
}

function hcRenderTags(d) {
  const rows = d.tags.filter((t) => t.n).sort((a, b) => b.n - a.n || a.name.localeCompare(b.name));
  const line = (act, v, name, n, dim) => (
    `<div class="hc-row link" data-act="${act}" data-v="${esc(v)}"><span class="nm${dim ? " dim" : ""}">${esc(name)}</span>`
    + `<span class="hc-stack"><i style="width:${(100 * n / Math.max(1, d.images)).toFixed(2)}%;background:var(--accent)"></i></span>`
    + `<span class="num">${hcN(n)} <span class="dim">${hcPct(n, d.images)}</span></span></div>`);
  $("#hc-tags").innerHTML = (rows.map((t) => line("tag", t.id, t.name, t.n, false)).join("")
    + (d.untagged ? line("tag-none", "", "태그 없음", d.untagged, true) : ""))
    || `<div class="dim">이미지가 없습니다.</div>`;
}

function hcRenderMisc(d) {
  const sp = d.splits, total = d.images;
  const tg = sp.target;
  const splitLine = ["train", "val", "test", "none"].filter((k) => k !== "none" || sp.images.none).map((k) => (
    `<button class="hc-chip" data-act="split" data-v="${k === "none" ? esc(d.none_split) : k}" title="이 분할의 이미지를 이미지 보기로">`
    + `<i style="background:${HC_SPLIT_COLORS[k]}"></i>${HC_SPLIT_NAMES[k]} <b>${hcN(sp.images[k])}</b>장 `
    + `<span class="dim">${hcPct(sp.images[k], total)} · 박스 ${hcN(sp.boxes[k])}</span></button>`)).join("");
  const doms = Object.entries(d.domains).sort((a, b) => b[1] - a[1]).map(([k, n]) => (
    `${esc(HC_DOMAIN_NAMES[k] || k)} <b>${hcN(n)}</b>`)).join(" · ");
  const sizes = d.sizes.map((s) => `${s.w}×${s.h} <b>${hcN(s.n)}</b>`).join(" · ")
    + (d.sizes_more ? ` · 외 ${hcN(d.sizes_more)}가지` : "");
  $("#hc-misc").innerHTML = `<div class="hc-sub">분할 <span class="dim">— 설정의 목표 train ${hcPct(tg.train, 1)} · val ${hcPct(tg.val, 1)} · test ${hcPct(tg.test, 1)}</span></div>`
    + `<div class="hc-big">${hcSplitBar(sp.images)}</div><div class="hc-chips">${splitLine}</div>`
    + `<div class="hc-sub">도메인</div><div class="hc-line">${doms || `<span class="dim">–</span>`}</div>`
    + `<div class="hc-sub">이미지 크기</div><div class="hc-line">${sizes || `<span class="dim">–</span>`}</div>`;
}

/* ───────────────────────── 박스 위치 (히트맵) ───────────────────────── */

function hcRenderHeatControls(d) {
  const cls = $("#hc-heat-cls"), cam = $("#hc-heat-cam");
  cls.innerHTML = `<option value="">클래스: 전체</option>` + d.classes.filter((r) => r.boxes).map((r) => (
    `<option value="${r.idx}">${esc(r.name)} (${hcN(r.boxes)})</option>`)).join("");
  cam.innerHTML = `<option value="">카메라: 전체</option>` + d.cameras.map((r) => (
    `<option value="${esc(r.key == null ? d.none_camera : r.key)}">${r.key == null ? "출처 없음" : esc(r.key)} (${hcN(r.images)}장)</option>`)).join("");
  // 고른 값이 새 목록에 없으면(범위를 바꿔 사라짐) 전체로
  H.cls = [...cls.options].some((o) => o.value === H.cls) ? H.cls : "";
  H.cam = [...cam.options].some((o) => o.value === H.cam) ? H.cam : "";
  cls.value = H.cls;
  cam.value = H.cam;
}

/** 지금 조절 칸의 히트맵 조건. 받아서 그린 히트맵의 조건은 H.heatQ 에 남긴다(hcLoadHeat). */
function hcHeatParams() {
  return { scope: $("#hc-scope").value, masked: $("#hc-masked").value, cls: H.cls, cam: H.cam };
}

function hcHeatQuery(p, cell = null) {
  const q = new URLSearchParams({ scope: p.scope, masked: p.masked });
  if (p.cls !== "") q.set("cls", p.cls);
  if (p.cam !== "") q.set("camera", p.cam);
  if (cell) q.set("cell", `${cell[0]},${cell[1]}`);
  return q;
}

function hcHeatAspect(ratio) {
  const cv = $("#hc-heat-canvas");
  const r = ratio > 0.2 && ratio < 5 ? ratio : 16 / 9;
  $("#hc-heat").style.aspectRatio = String(r);
  const h = Math.round(cv.width / r);
  if (cv.height !== h) cv.height = h;
  hcDrawHeat();
}

/** 카메라를 골랐으면 그 카메라의 가장 새 그림을 바탕에 흐리게 — 박스가 도로의 어디에 몰렸는지 보인다. */
function hcHeatBackground(id) {
  const img = $("#hc-heat-bg");
  if (id == null) {
    img.classList.add("hidden");
    img.removeAttribute("src");
    hcHeatAspect(16 / 9);
    return;
  }
  const seq = H.heatSeq;
  img.onload = () => { if (seq === H.heatSeq) hcHeatAspect(img.naturalWidth / img.naturalHeight); };
  img.onerror = () => { if (seq === H.heatSeq) { img.classList.add("hidden"); hcHeatAspect(16 / 9); } };
  img.classList.remove("hidden");
  img.src = url(`/api/images/${id}/thumb?s=384`);
}

function hcHeatColor(t) {
  // 옅은 노랑 → 붉은색. 사진 바탕 위에서도 보이게 불투명도도 함께 올린다
  const g = Math.round(211 - 152 * t), b = Math.round(77 - 29 * t);
  return `rgba(255, ${g}, ${b}, ${(0.3 + 0.6 * t).toFixed(3)})`;
}

function hcDrawHeat() {
  const cv = $("#hc-heat-canvas"), g = cv.getContext("2d"), h = H.heat;
  g.clearRect(0, 0, cv.width, cv.height);
  if (!h) return;
  const [gw, gh] = h.grid, cw = cv.width / gw, ch = cv.height / gh;
  for (let k = 0; k < h.cells.length; k++) {
    const n = h.cells[k];
    if (!n) continue;
    g.fillStyle = hcHeatColor(Math.sqrt(n / h.max));
    g.fillRect(Math.floor((k % gw) * cw), Math.floor(Math.floor(k / gw) * ch), Math.ceil(cw), Math.ceil(ch));
  }
  if (H.hover >= 0) {
    g.strokeStyle = "#fff";
    g.lineWidth = 2;
    g.strokeRect((H.hover % gw) * cw + 1, Math.floor(H.hover / gw) * ch + 1, cw - 2, ch - 2);
  }
}

async function hcLoadHeat() {
  const seq = ++H.heatSeq, p = hcHeatParams();
  try {
    const h = await api(`/api/health/heatmap?${hcHeatQuery(p)}`);
    if (seq !== H.heatSeq) return;
    hcShowHeat(h, p);
  } catch (err) {
    if (seq === H.heatSeq) toast(`박스 위치를 받지 못했습니다: ${err.message}`, true);
  }
}

/** 받은 히트맵 h 를 그린다. p 는 그것을 물은 조건(hcHeatParams) — 칸을 누르면 이 조건으로 묻는다(hcOpenCell). */
function hcShowHeat(h, p) {
  H.heat = h;
  H.heatQ = p;
  H.hover = -1;
  hcTip(null);
  $("#hc-heat-info").textContent = h.boxes
    ? `박스 ${hcN(h.boxes)}개 · 이미지 ${hcN(h.images)}장 · 가장 많은 칸 ${hcN(h.max)}개`
    : "이 조건의 박스가 없습니다";
  hcHeatBackground(h.sample_id);
  hcDrawHeat();
}

/** 마우스 위치의 칸 [x, y, 번호] — 그림 밖이면 null. */
function hcHeatCell(e) {
  const h = H.heat, r = $("#hc-heat-canvas").getBoundingClientRect();
  if (!h || !r.width || !r.height) return null;
  const fx = (e.clientX - r.left) / r.width, fy = (e.clientY - r.top) / r.height;
  if (fx < 0 || fx >= 1 || fy < 0 || fy >= 1) return null;
  const x = Math.min(h.grid[0] - 1, Math.floor(fx * h.grid[0])), y = Math.min(h.grid[1] - 1, Math.floor(fy * h.grid[1]));
  return [x, y, y * h.grid[0] + x];
}

function hcTip(text, e) {
  const tip = $("#hc-heat-tip");
  if (!text) { tip.classList.add("hidden"); return; }
  const box = $("#hc-heat").getBoundingClientRect();
  tip.textContent = text;
  tip.classList.remove("hidden");
  const x = e.clientX - box.left, y = e.clientY - box.top;
  tip.style.left = `${Math.min(box.width - tip.offsetWidth - 4, x + 14)}px`;
  tip.style.top = `${Math.max(4, y - tip.offsetHeight - 8)}px`;
}

/* ───────────────────────── 이미지 보기로 넘기기 ───────────────────────── */

/** id 목록으로 이미지 보기를 연다 — 맞는 필터가 없는 것(객체 수 막대·히트맵 칸·한 장). */
async function hcOpenIds(ids, more = 0) {
  if (!ids || !ids.length || H.opening || navBusy()) return;
  H.opening = true;
  try {
    await openIds(ids);
    // 받는 사이 다른 보기로 갔으면 끌고 오지 않는다 — 목록은 그 보기에서도 이 id 들이다
    if (VIEW === "health") await setView("gallery");
    if (more) toast(`앞의 ${hcN(ids.length)}장만 열었습니다 (${hcN(more)}장 더 있습니다)`);
  } catch (err) {
    toast(`목록으로 열지 못했습니다: ${err.message}`, true);
  } finally {
    H.opening = false;
  }
}

/** 히트맵의 칸 (x, y) 에 박스 중심이 있는 이미지를 이미지 보기로 연다.
 *  **화면에 그려진 히트맵의 조건**(H.heatQ)으로 묻는다 — 조절 칸을 바꿔 새로 받는 중이어도 눈에 보이는 칸의 이미지가 열린다.
 *  받는 사이 다른 보기로 갔거나(healthHide) 히트맵을 다시 받기 시작했으면(H.heatSeq) 결과를 버리고, 받는 동안에도 H.opening
 *  이라 두 번 누르기는 한 번만 묻는다 (2026-11-15 리뷰에서 고침: 전에는 늦게 온 응답이 라벨링으로 간 사람을 이미지 보기로
 *  끌고 갔고, 조건은 지금 조절 칸 값이라 그려진 것과 다른 칸의 이미지가 열릴 수 있었다). */
async function hcOpenCell(x, y) {
  if (!H.heat || !H.heatQ || H.opening || navBusy()) return;
  const seq = H.heatSeq;
  let cell = null;
  H.opening = true;
  try {
    cell = (await api(`/api/health/heatmap?${hcHeatQuery(H.heatQ, [x, y])}`)).cell;
  } catch (err) {
    if (seq === H.heatSeq) toast(`그 칸의 이미지를 받지 못했습니다: ${err.message}`, true);
    return;
  } finally {
    H.opening = false;
  }
  if (seq !== H.heatSeq) return;
  if (!cell || !cell.ids.length) {
    toast("그 칸의 박스가 그새 없어졌습니다 — ↻ 새로 고침으로 다시 세어 보세요", true);
    return;
  }
  await hcOpenIds(cell.ids, cell.more);
}

/** 왼쪽 필터를 이 조건 하나로 맞추고(다른 필터는 푼다) 이미지 보기를 연다. 상태 칸은 지금 범위를 따른다.
 *  필터를 읽고 쓰는 것은 app.js 의 noFilters·filterState·setFilters 다(필터 풀기와 함께 쓴다 — 여기에 따로 적지 않는다).
 *  목록을 받지 못하면 필터를 앞 상태로 되돌린다 — 필터만 새 조건이고 목록은 옛 것인 채로 두지 않는다(openIds 와 같다,
 *  2026-11-15 리뷰에서 고침). 목록을 받은 뒤의 실패(첫 장 열기)는 되돌리지 않는다 — 목록이 이미 새 조건이다. */
async function hcOpenFiltered({ cls = null, camera = "", split = "", tag = null, tagNone = false } = {}) {
  if (H.opening || navBusy()) return;
  H.opening = true;
  const prev = filterState();
  try {
    // 고치던 박스(라벨링에 열어 둔 장)를 먼저 저장한다 — 못 하면 필터도 목록도 건드리지 않는다(app.js reloadList 와 같다)
    if (!(await saveBeforeLeaving())) return;
    // 상태 칸은 서버가 준 값 — 범위 '휴지통만 빼고 전부' 는 목록의 '전체'(휴지통 포함)가 아니라 'live' 다 (2026-11-15)
    const want = noFilters(H.data.list_status);
    if (cls != null) want.classes.push(cls);
    if (tag != null) want.tagF.inc.push(tag);
    want.tagF.none = tagNone;
    setFilters({ ...want, camera, split });
    try {
      await loadImageList(new Set([0]));     // 필터가 바뀌었으니 첫 묶음부터 (app.js reloadList 와 같다)
    } catch (err) {
      setFilters(prev);
      syncListControls();                    // loadImageList 가 새 조건으로 맞춰 둔 칸 잠금·필터 요약을 되돌린다
      throw err;
    }
    if (S.images.length) await loadImage(S.images[0].id);
    // 받는 사이 다른 보기로 갔으면 끌고 오지 않는다 — 필터와 목록은 그 보기에서도 이 조건이다
    if (VIEW === "health") await setView("gallery");
  } catch (err) {
    toast(`목록을 받지 못했습니다: ${err.message}`, true);
  } finally {
    H.opening = false;
  }
}

function hcAct(act, v) {
  const d = H.data;
  if (!d) return;
  if (act === "cls") hcOpenFiltered({ cls: Number(v) });
  else if (act === "cam") hcOpenFiltered({ camera: v });
  else if (act === "split") hcOpenFiltered({ split: v });
  else if (act === "tag") hcOpenFiltered({ tag: Number(v) });
  else if (act === "tag-none") hcOpenFiltered({ tagNone: true });
  else if (act === "bin") { const b = d.count_bins[Number(v)]; if (b) hcOpenIds(b.ids, b.more); }
  else if (act === "one") hcOpenIds([Number(v)]);
}

/* ───────────────────────── 이벤트 ───────────────────────── */

$("#hc-body").addEventListener("click", (e) => {
  const el = e.target.closest("[data-act]");
  if (el) hcAct(el.dataset.act, el.dataset.v);
});
for (const id of ["#hc-scope", "#hc-masked", "#hc-imgsz"]) {
  $(id).addEventListener("change", () => { $(id).blur(); hcSaveControls(); hcLoad(); });
}
$("#hc-reload").onclick = (e) => { e.currentTarget.blur(); hcLoad(); };
$("#hc-heat-cls").addEventListener("change", (e) => { e.target.blur(); H.cls = e.target.value; hcLoadHeat(); });
$("#hc-heat-cam").addEventListener("change", (e) => { e.target.blur(); H.cam = e.target.value; hcLoadHeat(); });

$("#hc-heat-canvas").addEventListener("mousemove", (e) => {
  const c = hcHeatCell(e), k = c ? c[2] : -1, n = c && H.heat ? H.heat.cells[k] : 0;
  // 표시할 칸이 바뀔 때만 다시 그린다 — 빈 칸 위에서는 표시가 없으니(-1) 그대로다(전에는 움직일 때마다 다시 그렸다)
  const hover = n ? k : -1;
  if (hover !== H.hover) { H.hover = hover; hcDrawHeat(); }
  hcTip(n ? `박스 ${hcN(n)}개 — 눌러서 이미지 보기` : null, e);
  $("#hc-heat-canvas").style.cursor = n ? "pointer" : "default";
});
$("#hc-heat-canvas").addEventListener("mouseleave", () => { H.hover = -1; hcTip(null); hcDrawHeat(); });
$("#hc-heat-canvas").addEventListener("click", (e) => {
  const c = hcHeatCell(e);
  if (c && H.heat && H.heat.cells[c[2]]) hcOpenCell(c[0], c[1]);
});
