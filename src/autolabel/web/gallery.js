/* 이미지 보기 (2026-11-04) — 여러 장을 한눈에 보고, 골라서 태그·도메인을 한꺼번에 바꾼다.
 *
 * 라벨링 화면(app.js)과 같은 페이지의 다른 보기다(setView). 왼쪽 필터(S.filter*·listUrl)와 태그 목록
 * (S.tags·S.tagName·S.curTag)을 라벨링과 함께 쓴다. 바꾼 것은 라벨링 목록 줄(S.images·refreshRows)과 지금
 * 이미지(S.cur)에도 곧바로 반영한다. 필터에 맞지 않게 된 칸도 다시 받기 전까지 제자리에 둔다(라벨링과 같다
 * — '↻ 새로 받기'). 사용자가 '이미지의 태그 확인·변경이 너무 불편하다' 고 해서 만들었다.
 *
 * 목록은 지금 필터의 전부를 한 번에 받는다(G.items — '전체 선택' 이 화면 밖까지 고르게). 칸은 내려가는 대로
 * GRID_STEP 개씩 그리고, 썸네일은 서버가 줄여 캐시한 JPEG(/api/images/<id>/thumb)를 loading=lazy 로 받는다.
 */

const G = {
  items: [],          // 지금 필터의 이미지 전부 (목록 API 의 행 — 라벨링 S.images 와는 다른 객체)
  pos: new Map(),     // id -> items 안의 자리
  cells: [],          // 그린 칸 (items 와 같은 자리, 앞에서부터 shown 개)
  shown: 0,
  sel: new Set(),     // 고른 id
  anchor: -1,         // Shift 범위를 시작하는 자리
  focus: -1,          // 지금 칸 (오른쪽 미리 보기·방향키)
  seq: 0,             // 목록 요청 번호 — 늦게 도착한 옛 응답을 버린다
  busy: false,        // 바꾸는 중 — 그동안의 Q·W·E·T 는 버린다(꾹 누르면 반복 입력이 쌓인다)
  size: 180,          // 칸 크기(px)
  previewId: null,    // 오른쪽 미리 보기에 띄운 이미지 — 같으면 사진을 다시 받지 않는다
  leftCur: undefined, // 라벨링으로 갈 때의 라벨링 지금 이미지 (galleryHide)
  tagSig: "",         // 태그 이름표 — 바뀌었을 때만 칸을 모두 다시 쓴다
};
const GRID_STEP = 400;          // 한 번에 그리는 칸 수
const BULK_CONFIRM = 30;        // 이보다 많이 고른 채로 바꾸면 한 번 묻는다(Ctrl+A 뒤 Q 같은 실수)
const G_SIZE_KEY = "gallerySize";

/** 칸 크기·화면 배율에 맞는 썸네일 크기 — 서버는 160·256·384 만 받는다. */
function thumbSize() {
  const px = G.size * (window.devicePixelRatio || 1);
  return px <= 170 ? 160 : px <= 270 ? 256 : 384;
}

function tagNames(im) {
  return (im.tags || []).map((t) => S.tagName[t]).filter(Boolean);
}

/* ───────────────────────── 칸 ───────────────────────── */

function gCell(im, i) {
  const el = document.createElement("div");
  el.className = "g-cell";
  el.dataset.idx = String(i);
  el.innerHTML = '<span class="g-check">✓</span><div class="g-img"><img loading="lazy" decoding="async" alt=""></div>'
    + '<div class="g-meta"></div><div class="g-tags"></div>';
  const img = el.querySelector("img");
  img.onerror = () => { img.remove(); el.querySelector(".g-img").textContent = "열 수 없음"; };
  img.src = url(`/api/images/${im.id}/thumb?s=${thumbSize()}`);
  fillCell(el, im);
  return el;
}

/** 칸의 글자(도메인·분할·박스 수·태그)와 고름·지금 표시. 바꾼 뒤에는 이것만 다시 부른다. */
function fillCell(el, im) {
  el.classList.toggle("sel", G.sel.has(im.id));
  el.classList.toggle("focus", Number(el.dataset.idx) === G.focus);
  el.querySelector(".g-meta").innerHTML = `<span class="g-id">${im.id}</span>`
    + `<span class="dom${im.domain ? ` dom-${im.domain}` : ""}">${im.domain ? DOMAIN_NAME[im.domain] || im.domain : "미분류"}</span>`
    + (im.split === "test" || im.split === "val" ? `<span class="spl spl-${im.split}">${im.split}</span>` : "")
    + `<span class="nb" title="박스 수">${im.n_boxes}</span>`;
  const names = tagNames(im);
  const tg = el.querySelector(".g-tags");
  tg.textContent = names.length ? names.map((n) => `#${n}`).join(" ") : "태그 없음";
  tg.classList.toggle("none", !names.length);
  el.title = `${im.id} · ${im.name}\n${names.length ? names.map((n) => `#${n}`).join(" ") : "(태그 없음)"}`;
}

function renderGrid() {
  const grid = $("#g-grid");
  gMore.disconnect();
  grid.innerHTML = "";
  G.cells = [];
  G.shown = 0;
  if (!G.items.length) {
    grid.innerHTML = '<div class="g-empty">이 필터에 맞는 이미지가 없습니다</div>';
    return;
  }
  renderMore();
}

/** 다음 GRID_STEP 칸을 그린다. 끝에 표시를 두고, 그게 보이면 또 그린다. */
function renderMore() {
  const grid = $("#g-grid");
  gMore.disconnect();
  grid.querySelector(".g-more")?.remove();
  const frag = document.createDocumentFragment();
  const end = Math.min(G.items.length, G.shown + GRID_STEP);
  for (let i = G.shown; i < end; i++) {
    G.cells[i] = gCell(G.items[i], i);
    frag.appendChild(G.cells[i]);
  }
  G.shown = end;
  let more = null;
  if (end < G.items.length) {
    more = document.createElement("div");
    more.className = "g-more";
    more.textContent = `${end.toLocaleString()} / ${G.items.length.toLocaleString()}장 — 내리면 더 그립니다`;
    frag.appendChild(more);
  }
  grid.appendChild(frag);
  if (more) gMore.observe(more);
}
const gMore = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) renderMore(); },
  { root: $("#g-grid"), rootMargin: "800px" });

function ensureShown(i) {
  while (i >= G.shown && G.shown < G.items.length) renderMore();
}

/** id 의 칸이 그리드 위에서 얼마나 내려와 있나(px). 그린 칸이 없으면 null. */
function cellTop(id) {
  const i = id == null ? undefined : G.pos.get(id);
  const el = i === undefined ? null : G.cells[i];
  if (!el || !el.isConnected) return null;
  return el.getBoundingClientRect().top - $("#g-grid").getBoundingClientRect().top;
}

/* ───────────────────────── 받기 ───────────────────────── */

/** 지금 필터의 이미지를 모두 다시 받는다. 고른 것 가운데 남은 것과 지금 칸은 그대로 두고, 지금 칸은 화면의
 * 같은 높이에 둔다(새로 받아도 보던 자리가 움직이지 않게 — 라벨링 keepCurrentRow 와 같다).
 * focusId 를 주면 그 이미지를 지금 칸으로 하고 가운데로 굴린다(라벨링에서 넘어올 때). */
async function galleryLoad(focusId = undefined) {
  const seq = ++G.seq;
  $("#g-count").textContent = "불러오는 중…";
  let data;
  try {
    data = await api(listUrl(1e9, 0));
  } catch (err) {
    if (seq === G.seq) {
      $("#g-count").textContent = "";
      toast(`이미지 목록을 받지 못했습니다: ${err.message}`, true);
    }
    return;
  }
  if (seq !== G.seq) return;                      // 그새 또 받았다
  let keepId = focusId !== undefined ? focusId : G.items[G.focus]?.id ?? null;
  const keepTop = focusId !== undefined ? null : cellTop(keepId);
  if (focusId === undefined && keepId != null) {
    // 지금 칸이 새 목록에 없으면(필터에서 빠졌다) 옛 순서에서 그 뒤의 첫 생존 칸(없으면 앞쪽의 마지막 생존 칸)을
    // 지금 칸으로 하고, 빠진 칸이 있던 높이에 둔다 — 예전에는 맨 위로 튀었다(2026-11-05, 라벨링 keepCurrentRow 와 같다).
    const alive = new Set(data.images.map((im) => im.id));
    if (!alive.has(keepId)) {
      const next = G.items.slice(G.focus + 1).find((im) => alive.has(im.id))
        ?? G.items.slice(0, G.focus).reverse().find((im) => alive.has(im.id));
      keepId = next ? next.id : null;
    }
  }
  G.items = data.images;
  G.pos = new Map(G.items.map((im, i) => [im.id, i]));
  for (const id of [...G.sel]) if (!G.pos.has(id)) G.sel.delete(id);
  G.focus = keepId != null && G.pos.has(keepId) ? G.pos.get(keepId) : -1;
  G.anchor = G.focus;
  renderGrid();
  const grid = $("#g-grid");
  if (G.focus >= 0) {
    ensureShown(G.focus);
    const now = cellTop(keepId);
    if (keepTop != null && now != null) grid.scrollTop += now - keepTop;
    else G.cells[G.focus].scrollIntoView({ block: "center" });
  } else {
    grid.scrollTop = 0;
  }
  renderGalleryHead();
  renderGallerySide();
}

/** 이미지 보기로 올 때(setView). 라벨링에서 바꾼 것이 있을 수 있어 늘 새로 받는다. 라벨링에서 다른 이미지로
 * 옮겨 왔으면 그 이미지를 지금 칸으로 하고 그것만 고른다 — 바로 태그를 보고 고칠 수 있게, 그리고 미리 보기와
 * 오른쪽 태그 칸이 같은 이미지를 가리키게. 그대로 돌아왔으면(V 를 두 번) 고른 것·지금 칸을 그대로 둔다. */
async function galleryShow() {
  const curId = S.cur ? S.cur.id : null;
  const moved = curId !== G.leftCur;
  await galleryLoad(moved || G.focus < 0 ? curId : undefined);
  if ((moved || !G.sel.size) && G.focus >= 0) {
    G.sel = new Set([G.items[G.focus].id]);
    G.anchor = G.focus;
    syncSel();
  }
}

/** 라벨링으로 갈 때(setView) — 돌아왔을 때 라벨링에서 이미지를 옮겼는지 가린다. */
function galleryHide() {
  G.leftCur = S.cur ? S.cur.id : null;
}

/* ───────────────────────── 고르기 ───────────────────────── */

function renderGalleryHead() {
  $("#g-count").textContent = `${G.items.length.toLocaleString()}장`;
  $("#g-sel-n").textContent = G.sel.size.toLocaleString();
}

/** 고른 것이 바뀐 뒤 — 칸 표시·개수·오른쪽을 맞춘다. */
function syncSel() {
  for (let i = 0; i < G.shown; i++) G.cells[i].classList.toggle("sel", G.sel.has(G.items[i].id));
  renderGalleryHead();
  renderGallerySide();
}

function setSel(ids) {
  G.sel = new Set(ids);
  syncSel();
}

function setFocus(i, scroll = true) {
  if (G.focus >= 0 && G.cells[G.focus]) G.cells[G.focus].classList.remove("focus");
  G.focus = i;
  if (i < 0) return;
  ensureShown(i);
  G.cells[i].classList.add("focus");
  if (scroll) G.cells[i].scrollIntoView({ block: "nearest" });
}

function rangeIds(a, b) {
  const [lo, hi] = a < b ? [a, b] : [b, a];
  return G.items.slice(lo, hi + 1).map((im) => im.id);
}

/** 고른 id — 그리드 순서대로. */
function selIds() {
  return G.items.filter((im) => G.sel.has(im.id)).map((im) => im.id);
}

$("#g-grid").addEventListener("click", (e) => {
  const el = e.target.closest(".g-cell");
  if (!el) return;
  const i = Number(el.dataset.idx), id = G.items[i].id;
  const add = e.ctrlKey || e.metaKey;
  if (e.shiftKey && G.anchor >= 0) {          // 범위 (Ctrl 도 누르면 지금 고른 것에 더한다)
    if (!add) G.sel.clear();
    for (const x of rangeIds(G.anchor, i)) G.sel.add(x);
  } else if (add) {                            // 하나 더하기/빼기
    if (G.sel.has(id)) G.sel.delete(id); else G.sel.add(id);
    G.anchor = i;
  } else {                                     // 이것만
    G.sel = new Set([id]);
    G.anchor = i;
  }
  setFocus(i, false);
  syncSel();
});
$("#g-grid").addEventListener("dblclick", (e) => {
  const el = e.target.closest(".g-cell");
  if (el) openInLabel(G.items[Number(el.dataset.idx)].id);
});

/** 한 줄에 칸이 몇 개인가 — ↑·↓ 가 한 줄씩 움직이게. */
function gridCols() {
  const grid = $("#g-grid");
  const cs = getComputedStyle(grid);
  const inner = grid.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const gap = parseFloat(cs.columnGap) || 8;
  return Math.max(1, Math.floor((inner + gap) / (G.size + gap)));
}

/** 방향키: 지금 칸을 옮기고 그 칸만 고른다. Shift 면 시작한 칸부터 범위로 고른다. */
function moveFocus(d, extend) {
  if (!G.items.length) return;
  const i = G.focus < 0 ? 0 : Math.max(0, Math.min(G.items.length - 1, G.focus + d));
  setFocus(i);
  if (extend && G.anchor >= 0) {
    G.sel = new Set(rangeIds(G.anchor, i));
  } else {
    G.anchor = i;
    G.sel = new Set([G.items[i].id]);
  }
  syncSel();
}

function toggleFocusSel() {
  if (G.focus < 0) return;
  const id = G.items[G.focus].id;
  if (G.sel.has(id)) G.sel.delete(id); else G.sel.add(id);
  G.anchor = G.focus;
  syncSel();
}

/* ───────────────────────── 라벨링으로 ───────────────────────── */

/** 그 이미지를 라벨링 보기에서 연다 — 지금 필터의 목록 안에서(↑·↓ 가 이어진다). 라벨링 목록을 먼저 다시 받는다 —
 *  여기서 바꾼 것(Q·W·E·T)으로 서버 쪽 묶음 경계가 밀려 있으면, 옛 묶음으로 펼친 목록에서 줄이 빠졌다 (2026-11-05). */
async function openInLabel(id) {
  try {
    await setView("label");
    await loadImageList();
    await openChunkWith([id]);      // 묶음 보기에서 그 이미지가 닫힌 묶음에 있으면 펼친다
    await loadImage(id);
  } catch (err) {
    toast(`라벨링에서 열지 못했습니다: ${err.message}`, true);
  }
}

/** 고른 이미지만 라벨링 목록으로 넘긴다('지정한 id', 그리드 순서). 고른 것이 없으면 지금 칸 하나를 연다. */
async function openSelInLabel() {
  const ids = selIds();
  if (!ids.length) {
    if (G.focus >= 0) await openInLabel(G.items[G.focus].id);
    else toast("이미지를 먼저 고르세요");
    return;
  }
  try {
    await setView("label");
    await openIds(ids);           // 목록을 못 받으면 앞 상태로 되돌리고 던진다
  } catch (err) {
    toast(`라벨링 목록으로 넘기지 못했습니다: ${err.message}`, true);
  }
}

/* ───────────────────────── 한꺼번에 바꾸기 ───────────────────────── */

/** 바꾼 이미지의 행(이미지 보기·라벨링 목록·지금 이미지)을 fn 으로 고치고 다시 그린다. */
function applyToRows(ids, fn) {
  const hit = new Set(ids);
  for (const im of G.items) if (hit.has(im.id)) fn(im);
  for (const im of S.images) if (hit.has(im.id)) fn(im);
  if (S.cur && hit.has(S.cur.id)) { fn(S.cur); renderDomainBar(); renderTagList(); }
  refreshRows(ids);
  for (const id of ids) {
    const i = G.pos.get(id);
    if (i !== undefined && i < G.shown) fillCell(G.cells[i], G.items[i]);
  }
  renderGallerySide();
}

async function bulkDomain(domain) {
  if (G.busy) return;
  const ids = selIds();
  if (!ids.length) { toast("이미지를 먼저 고르세요 (클릭·Shift+클릭·Ctrl+A)"); return; }
  if (ids.length > BULK_CONFIRM
      && !confirm(`고른 ${ids.length.toLocaleString()}장의 도메인을 '${DOMAIN_NAME[domain]}' 로 바꿉니다. 계속할까요?`)) return;
  G.busy = true;
  try {
    await api("/api/bulk/domain", { method: "POST", body: { ids, domain } });
    applyToRows(ids, (im) => { im.domain = domain; });
    loadDomains();
    toast(`${ids.length.toLocaleString()}장을 ${DOMAIN_NAME[domain]} 로 정했습니다`);
  } catch (err) {
    toast(`도메인을 바꾸지 못했습니다: ${err.message}`, true);
  } finally {
    G.busy = false;
  }
}

async function bulkTag(tagId, on) {
  if (G.busy) return;
  if (tagId == null || !S.tagName[tagId]) {
    toast("오른쪽 태그 목록에서 붙일 태그를 먼저 누르세요 (없으면 만드세요)", true);
    return;
  }
  const ids = selIds();
  if (!ids.length) { toast("이미지를 먼저 고르세요 (클릭·Shift+클릭·Ctrl+A)"); renderGallerySide(); return; }
  const name = S.tagName[tagId];
  if (ids.length > BULK_CONFIRM && !confirm(`고른 ${ids.length.toLocaleString()}장${on
    ? `에 태그 '${name}' 를 붙입니다` : `에서 태그 '${name}' 를 뗍니다`}. 계속할까요?`)) {
    renderGallerySide();          // 눌러 바뀐 칸 표시를 되돌린다
    return;
  }
  G.busy = true;
  try {
    const r = await api("/api/bulk/tag", { method: "POST", body: { ids, tag_id: tagId, on } });
    applyToRows(ids, (im) => {
      const s = new Set(im.tags || []);
      if (on) s.add(tagId); else s.delete(tagId);
      im.tags = [...s].sort((a, b) => a - b);
    });
    loadTags();                   // 개수
    toast(`#${name} ${on ? "붙임" : "뗌"} — ${r.changed.toLocaleString()}장이 바뀜 (고른 ${ids.length.toLocaleString()}장)`);
  } catch (err) {
    toast(`태그를 바꾸지 못했습니다: ${err.message}`, true);
    loadTags();
  } finally {
    G.busy = false;
  }
}

/* ───────────────────────── 오른쪽 ───────────────────────── */

function renderGallerySide() {
  const n = G.sel.size;
  $("#gs-n").textContent = n ? `${n.toLocaleString()}장` : "없음";

  // 미리 보기: 지금 칸(없으면 고른 것 가운데 첫 장)
  const pv = $("#gs-preview");
  let im = G.focus >= 0 ? G.items[G.focus] : null;
  if (!im && n) im = G.items[G.pos.get(selIds()[0])];
  if (!im) {
    G.previewId = null;
    pv.innerHTML = '<div class="empty">칸을 누르면 여기서 크게 봅니다</div>';
  } else {
    if (G.previewId !== im.id) {
      G.previewId = im.id;
      pv.innerHTML = '<img alt=""><div class="nm"></div><div class="meta"></div><div class="tags"></div>';
      pv.querySelector("img").src = url(`/api/images/${im.id}/thumb?s=384`);
    }
    pv.querySelector(".nm").textContent = im.name;
    pv.querySelector(".nm").title = im.name;
    pv.querySelector(".meta").textContent = `${im.id} · ${im.width}×${im.height} · ${im.status} · `
      + `${im.split || "미배정"} · ${DOMAIN_NAME[im.domain] || "미분류"} · 박스 ${im.n_boxes}`;
    const names = tagNames(im);
    pv.querySelector(".tags").textContent = names.length ? names.map((x) => `#${x}`).join(" ") : "태그 없음";
  }

  // 도메인: 고른 것의 분포. 모두 한 도메인이면 그 단추를 켠다
  const dc = new Map();
  const tc = new Map();     // 태그 id -> 고른 것 가운데 붙은 장수
  for (const id of G.sel) {
    const it = G.items[G.pos.get(id)];
    if (!it) continue;
    dc.set(it.domain, (dc.get(it.domain) || 0) + 1);
    for (const t of it.tags || []) tc.set(t, (tc.get(t) || 0) + 1);
  }
  $("#gs-dom").textContent = [...dc].sort((a, b) => b[1] - a[1])
    .map(([d, c]) => `${DOMAIN_NAME[d] || "미분류"} ${c.toLocaleString()}`).join(" · ");
  document.querySelectorAll("[data-gdomain]").forEach((b) => {
    b.disabled = !n;
    b.classList.toggle("on", n > 0 && dc.get(b.dataset.gdomain) === n);
  });

  // 태그: 칸 ✓ = 고른 것 모두에 붙음, − = 일부. 고른 것에 붙은 태그를 위로(무엇이 붙었는지 먼저 보이게)
  $("#gs-tag-count").textContent = S.tags.length ? `${S.tags.length}개` : "";
  $("#gs-tag-cur").textContent = S.curTag != null && S.tagName[S.curTag] ? `→ #${S.tagName[S.curTag]}` : "";
  const ul = $("#gs-tags");
  ul.innerHTML = "";
  if (!S.tags.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "태그가 없습니다 — 아래에 이름을 넣어 만드세요";
    ul.appendChild(li);
    return;
  }
  const q = $("#gs-tag-q").value;
  const list = S.tags.filter((t) => tagMatches(t, q)).sort((a, b) => (tc.get(b.id) || 0) - (tc.get(a.id) || 0));
  for (const t of list) {
    const c = tc.get(t.id) || 0;
    const li = document.createElement("li");
    li.className = t.id === S.curTag ? "active" : "";
    li.title = `${t.name} — 붙은 이미지 ${t.n.toLocaleString()}장${n ? `, 고른 것 중 ${c.toLocaleString()}장` : ""}\n`
      + "칸을 누르면 고른 이미지 모두에 붙입니다(모두에 붙어 있으면 뗍니다). 줄을 누르면 T 로 붙일 태그로 고릅니다.";
    li.innerHTML = `<input type="checkbox"${n ? "" : " disabled"}><span class="nm"></span>`
      // 고른 것 중 붙은 장수/고른 장수 · 전체 장수 (고른 것에 없으면 전체만)
      + `<span class="cnt">${c ? `<b>${c.toLocaleString()}</b>/${n.toLocaleString()} · ` : ""}${t.n.toLocaleString()}</span>`
      + '<button class="linkbtn ed" title="이름 바꾸기">✎</button>'
      + '<button class="linkbtn del" title="태그 지우기 (이미지는 그대로)">×</button>';
    li.querySelector(".nm").textContent = t.name;
    const box = li.querySelector("input");
    box.checked = n > 0 && c === n;
    box.indeterminate = c > 0 && c < n;
    box.onclick = (e) => e.stopPropagation();
    box.onchange = () => { box.blur(); bulkTag(t.id, c !== n); };
    li.querySelector(".ed").onclick = (e) => { e.stopPropagation(); renameTag(t); };
    li.querySelector(".del").onclick = (e) => { e.stopPropagation(); deleteTag(t); };
    li.onclick = () => {
      S.curTag = S.curTag === t.id ? null : t.id;
      saveCurTag();
      renderTagList();
      renderGallerySide();
    };
    ul.appendChild(li);
  }
}

/** 태그 목록이 바뀐 뒤(loadTags) — 지운 태그는 칸에서도 떼고, 이름이 바뀌었으면 칸 글자를 다시 쓴다. */
function galleryTagsChanged() {
  const sig = S.tags.map((t) => `${t.id}:${t.name}`).join("|");
  if (sig !== G.tagSig) {
    G.tagSig = sig;
    for (const im of G.items) {
      if (im.tags && im.tags.some((t) => !(t in S.tagName))) im.tags = im.tags.filter((t) => t in S.tagName);
    }
    for (let i = 0; i < G.shown; i++) fillCell(G.cells[i], G.items[i]);
  }
  renderGallerySide();
}

/* ───────────────────────── 키 ───────────────────────── */

/** 이미지 보기의 단축키 (app.js keydown 이 넘긴다 — 입력 칸·대화상자·V 는 거기서 먼저 걸렀다). */
function galleryKey(e) {
  if (e.ctrlKey || e.metaKey) {
    if (e.key === "a" || e.key === "A" || e.key === "ㅁ") { e.preventDefault(); setSel(G.items.map((im) => im.id)); }
    return;
  }
  if (e.altKey) return;
  switch (e.key) {
    case "Escape": setSel([]); break;
    case "ArrowRight": e.preventDefault(); moveFocus(1, e.shiftKey); break;
    case "ArrowLeft": e.preventDefault(); moveFocus(-1, e.shiftKey); break;
    case "ArrowDown": e.preventDefault(); moveFocus(gridCols(), e.shiftKey); break;
    case "ArrowUp": e.preventDefault(); moveFocus(-gridCols(), e.shiftKey); break;
    case "Home": e.preventDefault(); moveFocus(-G.items.length, e.shiftKey); break;
    case "End": e.preventDefault(); moveFocus(G.items.length, e.shiftKey); break;
    case " ": e.preventDefault(); toggleFocusSel(); break;
    case "Enter": e.preventDefault(); openSelInLabel(); break;
    case "q": case "Q": case "ㅂ": case "ㅃ": e.preventDefault(); bulkDomain("cctv"); break;
    case "w": case "W": case "ㅈ": case "ㅉ": e.preventDefault(); bulkDomain("noncctv"); break;
    case "e": case "E": case "ㄷ": case "ㄸ": e.preventDefault(); bulkDomain("drone"); break;
    case "t": case "T": case "ㅅ": case "ㅆ": e.preventDefault(); bulkTag(S.curTag, true); break;
    default: break;
  }
}

/* ───────────────────────── 배선 ───────────────────────── */

function applySize(px) {
  G.size = px;
  $("#g-grid").style.setProperty("--g-size", `${px}px`);
  $("#g-size").value = String(px);
}
$("#g-size").addEventListener("input", (e) => {
  const before = thumbSize();
  applySize(Number(e.target.value));
  if (thumbSize() !== before) {             // 칸이 커져 더 큰 썸네일이 필요하다
    for (let i = 0; i < G.shown; i++) {
      const img = G.cells[i].querySelector("img");
      if (img) img.src = url(`/api/images/${G.items[i].id}/thumb?s=${thumbSize()}`);
    }
  }
  if (G.focus >= 0) G.cells[G.focus]?.scrollIntoView({ block: "nearest" });
  try { localStorage.setItem(G_SIZE_KEY, String(G.size)); } catch { /* 저장 못 해도 그만 */ }
});
$("#g-size").addEventListener("change", (e) => e.target.blur());   // 포커스가 남으면 ←·→ 가 크기를 바꾼다
$("#g-select-all").onclick = (e) => { e.currentTarget.blur(); setSel(G.items.map((im) => im.id)); };
$("#g-select-none").onclick = (e) => { e.currentTarget.blur(); setSel([]); };
$("#g-open").onclick = (e) => { e.currentTarget.blur(); openSelInLabel(); };
document.querySelectorAll("[data-gdomain]").forEach((b) => {
  b.onclick = () => { b.blur(); bulkDomain(b.dataset.gdomain); };
});
$("#gs-tag-q").addEventListener("input", renderGallerySide);
$("#gs-tag-q").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.target.value = ""; renderGallerySide(); e.target.blur(); }
});
$("#gs-add-tag").onclick = () => createTag($("#gs-new-tag"));
$("#gs-new-tag").addEventListener("keydown", (e) => {
  if (e.isComposing || e.keyCode === 229) return;     // 한글 조합 중의 Enter (app.js #new-tag 와 같다, 2026-11-05)
  if (e.key === "Enter") createTag($("#gs-new-tag"));
  if (e.key === "Escape") e.target.blur();
});
try {
  const v = Number(localStorage.getItem(G_SIZE_KEY));
  if (v >= 120 && v <= 320) applySize(v);
} catch { /* 없으면 기본 크기 */ }
