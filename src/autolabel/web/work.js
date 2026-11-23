/* 작업 배치 (2026-11-23, 사용자 요청) — 한 번에 넣은 묶음(영상 캡처의 넣기·이미지 넣기·폴더 스캔)을 '작업 N' 으로 따로 본다.
 *
 * app.js 뒤에 읽힌다. 같은 페이지의 다섯 번째 보기다(setView('work') · body.view-work — app.js VIEWS 표에 key 없이 적혀 있어
 * 이 보기에서 키는 전부 버려진다: 숨은 라벨링에 Space 승인이 먹지 않는다). 배치 목록은 app.js 가 loadState 마다 받아 둔
 * S.works 를 그대로 그린다(loadWorks → workRender) — 여기서 따로 들고 있지 않는다.
 *
 * - 카드를 누르면(workOpen) 왼쪽 필터를 그 배치 하나로 바꾸고 라벨링으로 간다. 이미지 보기도 같은 필터를 쓴다.
 * - '자동 라벨링' 은 그 배치의 미라벨만 돌린다(POST /api/jobs/predict {work_batch}).
 * - '모아서 한꺼번에' 를 켠 배치는 승인(Space)이 '검수 끝' 으로 가고(app.js approveTarget), '데이터셋에 추가' 가 검수 끝을
 *   전부 승인으로 올린다(POST /api/work-batches/{id}/add). 검수 대기·미라벨은 그대로다 — 보지 않은 장은 승인되지 않는다.
 * - '＋ 이미지 넣기'(서버 upload.py): 고른 파일을 임시 폴더에 올리고, 넣기 때만 unlabeled/<카메라>/ 로 옮겨 배치 하나로 등록한다.
 *   **카메라(출처)는 반드시 고른다** — 모르면 '자동'(이 묶음용 카메라 vid-NN 등록). 분할이 카메라째라 출처 없이는 넣지 않는다.
 */

const W = {
  seq: 0,            // 들어올 때·나갈 때·다시 받을 때마다 +1 — 늦게 온 응답을 버린다
  opening: false,    // 배치를 여는 중(목록 받기) — 두 번 누르기는 한 번만
  busy: new Set(),   // 이름·방식·추가 요청이 도는 배치 번호
  doneOpen: false,   // '끝난 배치' 를 펼쳤나
  limits: null,      // /api/upload 의 한도·형식 {max_files, max_bytes, exts}
  left: [],          // 서버에 남은 넣지 않은 묶음 (창을 새로 고쳐 잃어버린 올리기)
  up: null,          // 고른 파일 묶음 {files, skipped: [글], sid, sending, cancel, done, failed: [{name, why}]}
};

const WK_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"];   // 서버가 알려 주기 전의 기본값 (yolo_io.IMG_EXTS)
const WK_CAM_KEY = `wkCam:${BASE}`;       // 지난번에 고른 카메라 — 탭(프로젝트)마다 카메라가 따로라 주소별로
const WK_PICK = "?";                      // 카메라 칸의 '고르세요' — 이 값이면 넣지 않는다
const WK_STATUS = [["unlabeled", "미라벨"], ["pending", "검수 대기"], ["reviewed", "검수 끝"], ["approved", "데이터셋"],
                   ["skipped", "건너뜀"], ["trashed", "휴지통"]];

function wkBytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

/** 서버의 UTC 시각 글을 이 PC 의 시각 'YYYY-MM-DD HH:MM' 으로. 읽을 수 없으면 그대로. */
function wkWhen(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso || "");
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

/** 배치를 열 때 처음 보일 상태 — 할 일이 있는 것부터(검수 대기 → 미라벨 → 검수 끝), 없으면 휴지통 빼고 전체. */
function workFirstStatus(b) {
  const c = (b && b.counts) || {};
  return ["pending", "unlabeled", "reviewed"].find((s) => c[s]) || "live";
}

/* ───────────────────────── 보기 들어오고 나가기 ───────────────────────── */

async function workShow() {
  W.seq++;
  workRender();          // 받아 둔 것을 먼저 그리고
  await workReload();    // 새로 받는다
}

function workHide() {
  W.seq++;
}

async function workReload() {
  const seq = ++W.seq;
  $("#work").classList.add("loading");
  try {
    // 배치 목록은 app.js 의 loadWorks 가 S.works 에 받고(끝나면 workRender 를 부른다), 카메라 목록은 loadCameras 가 S.camReg 에
    const [, , up] = await Promise.all([loadWorks(), loadCameras(), api("/api/upload").catch(() => null)]);
    if (seq !== W.seq) return;
    if (up) { W.limits = up.limits; W.left = up.sessions || []; W.unlabeled = up.unlabeled || ""; }
  } finally {
    if (seq === W.seq) $("#work").classList.remove("loading");
  }
  workRender();
}

/* ───────────────────────── 그리기 ───────────────────────── */

function workCard(b) {
  const c = b.counts, live = b.total - (c.trashed || 0);
  const el = document.createElement("div");
  el.className = "wk-card" + (W.busy.has(b.id) ? " busy" : "");
  el.dataset.id = b.id;
  const cams = b.cameras || [];
  const camText = !cams.length ? "" : cams.length > 1 ? `카메라 ${cams.length}대`
    : cams[0].key ? camLabel(cams[0]) : "출처 없음";
  const camTitle = cams.map((x) => `${x.key || "출처 없음"}${x.place ? ` · ${x.place}` : ""} (${x.n}장)`).join("\n");
  // 진행 막대 — 상태별 몫(휴지통 제외)
  const segs = WK_STATUS.filter(([k]) => k !== "trashed" && c[k])
    .map(([k]) => `<i class="st-${k}" style="flex:${c[k]}"></i>`).join("");
  const chips = WK_STATUS.filter(([k]) => c[k]).map(([k, t]) =>
    `<button class="wk-chip st-${k}" data-act="open" data-status="${k}" title="이 배치의 '${t}' ${c[k]}장을 라벨링에서 엽니다">`
    + `<i></i>${t} <b>${c[k]}</b></button>`).join("");
  el.innerHTML =
    '<div class="wk-top">'
    + `<button class="wk-title" data-act="open" title="이 배치를 라벨링에서 엽니다">작업 ${b.id}</button>`
    + '<span class="wk-name"></span><span class="wk-meta dim"></span><span class="spacer"></span>'
    + '<button class="linkbtn" data-act="rename" title="이 배치의 이름을 바꿉니다(비우면 이름 없음)">이름…</button>'
    + "</div>"
    + `<div class="wk-bar" title="상태별 몫 (휴지통 제외 ${live}장)">${segs}</div>`
    + `<div class="wk-chips">${chips || '<span class="dim">이미지 없음</span>'}</div>`
    + '<div class="wk-acts">'
    + '<button class="btn btn-sm btn-primary" data-act="open">열기</button>'
    + `<button class="btn btn-sm" data-act="predict" ${c.unlabeled ? "" : "disabled"} title="이 배치의 미라벨 이미지만 자동 라벨링합니다(검수 대기로 들어갑니다). 미라벨이 없으면 할 것이 없습니다">자동 라벨링${c.unlabeled ? ` (미라벨 ${c.unlabeled}장)` : ""}</button>`
    + `<button class="btn btn-sm" data-act="predict-tiles" ${c.unlabeled ? "" : "disabled"} title="같은 일을 타일 추론으로 — 멀리 있는 작은 물체를 더 잡지만 오검출이 늘고 시간이 약 2배입니다">타일</button>`
    + (c.reviewed ? `<button class="btn btn-sm wk-add" data-act="add" title="이 배치에서 '검수 끝' 인 이미지를 전부 승인(데이터셋)으로 올립니다 — 검수 대기·미라벨은 그대로 둡니다. 한 번 묻습니다">검수 끝 ${c.reviewed}장 데이터셋에 추가</button>` : "")
    + '<span class="spacer"></span>'
    + '<label class="wk-hold" title="켜면 이 배치에서 승인(Space)이 \'검수 끝\' 으로 갑니다 — 데이터셋(승인)에는 \'데이터셋에 추가\' 를 눌러 한꺼번에 넣습니다. 끄면 지금까지처럼 승인하는 대로 데이터셋에 들어갑니다. 이미 검수 끝·승인인 이미지는 바꾸지 않습니다">'
    + `<input type="checkbox" data-act="hold" ${b.hold ? "checked" : ""}> 모아서 한꺼번에 추가</label>`
    + "</div>";
  el.querySelector(".wk-name").textContent = b.name || b.kind_name || b.kind;
  const meta = el.querySelector(".wk-meta");
  meta.textContent = [b.name ? b.kind_name : "", camText, wkWhen(b.created_at), `${live}장`].filter(Boolean).join(" · ");
  meta.title = [b.note, camTitle].filter(Boolean).join("\n");
  return el;
}

function workRender() {
  const shown = S.works.filter((b) => b.total > 0);
  const open = shown.filter((b) => workTodo(b) > 0), done = shown.filter((b) => workTodo(b) === 0);
  const list = $("#wk-list");
  list.innerHTML = "";
  if (!shown.length) {
    list.innerHTML = '<div class="hc-card dim">아직 작업 배치가 없습니다 — <b>＋ 이미지 넣기</b>, <b>영상 캡처</b>의 넣기, '
      + "<b>데이터 → 폴더 스캔</b> 가운데 어느 것으로든 이미지를 넣으면 그 묶음이 <b>작업 N</b> 으로 여기에 생깁니다.</div>";
  } else if (!open.length) {
    list.innerHTML = '<div class="hc-card dim">할 일이 남은 배치가 없습니다 — 모든 배치의 이미지가 데이터셋(승인)·건너뜀·휴지통입니다.</div>';
  }
  for (const b of open) list.appendChild(workCard(b));
  const head = $("#wk-done-head"), doneEl = $("#wk-done");
  head.classList.toggle("hidden", !done.length);
  doneEl.classList.toggle("hidden", !done.length || !W.doneOpen);
  head.textContent = `${W.doneOpen ? "▾" : "▸"} 끝난 배치 ${done.length}개 — 미라벨·검수 대기·검수 끝이 남지 않은 배치`;
  doneEl.innerHTML = "";
  if (W.doneOpen) for (const b of done) doneEl.appendChild(workCard(b));
  // 배치 없는 이미지 — 배치 기능 전에 넣은 것·도구(tools/import_*)가 넣은 것
  const n = S.workNone;
  const none = $("#wk-none");
  none.innerHTML = "";
  if (n && n.total) {
    const parts = WK_STATUS.filter(([k]) => n[k]).map(([k, t]) => `${t} ${n[k]}`).join(" · ");
    const live = n.total - (n.trashed || 0);
    none.innerHTML = `배치 없는 이미지 <b>${live}</b>장 (${esc(parts)}) — 배치 기능 전에 넣었거나 도구가 넣은 이미지입니다. `
      + '<button class="linkbtn" id="wk-none-open">라벨링에서 보기</button>';
    $("#wk-none-open").onclick = () => workOpen(S.workNoneKey, workFirstStatus({ counts: n }));
  }
  workRenderLeft();
  workRenderUp();
}

/** 서버에 남은 넣지 않은 묶음 — 올리다 창을 새로 고쳤을 때. 지금 올리는 중인 묶음은 빼고 보인다. */
function workRenderLeft() {
  const mine = W.up && W.up.sid;
  const left = W.left.filter((s) => s.sid !== mine && s.state === "open" && s.n);
  const el = $("#wk-left");
  el.classList.toggle("hidden", !left.length);
  if (!left.length) return;
  const n = left.reduce((a, s) => a + s.n, 0);
  el.innerHTML = `넣지 않은 올린 묶음 ${left.length}개(${n}장)가 서버에 남아 있습니다 — 올리다 창을 닫았거나 새로 고친 것입니다. `
    + '<button class="linkbtn" id="wk-left-commit">그대로 넣기</button> · <button class="linkbtn" id="wk-left-drop">버리기</button>';
  $("#wk-left-commit").onclick = () => workLeft(left, true);
  $("#wk-left-drop").onclick = () => workLeft(left, false);
}

async function workLeft(sessions, commit) {
  if (!confirm(commit
    ? `남아 있는 묶음 ${sessions.length}개를 올릴 때 고른 카메라로 그대로 넣을까요?`
    : `남아 있는 묶음 ${sessions.length}개를 버릴까요?\n서버의 임시 사본만 지웁니다 — 원본 파일은 그대로입니다.`)) return;
  for (const s of sessions) {
    try {
      if (commit) workShowResult(await api(`/api/upload/${s.sid}/commit`, { method: "POST" }), []);
      else await api(`/api/upload/${s.sid}`, { method: "DELETE" });
    } catch (err) {
      if (err.status !== 404) toast(err.message, true);
    }
  }
  try { await loadState(); await loadWorks(); } catch { /* 아래에서 다시 받는다 */ }
  await workReload();
}

/* ───────────────────────── 배치 동작 ───────────────────────── */

/** 그 배치(또는 '배치 없음')만 라벨링 목록에 걸고 라벨링으로 간다. 고치던 박스를 먼저 저장하고, 못 하면 아무것도 바꾸지
 *  않는다. 목록을 못 받으면 필터를 되돌린다(app.js reloadList·health.js hcOpenFiltered 와 같은 약속). */
async function workOpen(id, status = null) {
  if (W.opening || navBusy()) return;
  W.opening = true;
  const prev = filterState(), from = VIEW;
  try {
    if (!(await saveBeforeLeaving())) return;
    const find = () => S.works.find((x) => String(x.id) === String(id));
    // 방금 넣은 배치는 목록(S.works)에 아직 없을 수 있다 — 한 번 받고 연다(필터 칸의 항목·머리줄이 그 배치를 알아야 한다)
    if (!find() && String(id) !== S.workNoneKey) await loadWorks();
    const b = find();
    const want = noFilters(status || workFirstStatus(b));
    want.work = String(id);
    setFilters(want);
    try {
      await loadImageList(new Set([0]));     // 필터가 바뀌었으니 첫 묶음부터
    } catch (err) {
      setFilters(prev);
      syncListControls();                    // loadImageList 가 새 조건으로 맞춰 둔 칸 잠금·필터 요약을 되돌린다
      throw err;
    }
    if (S.images.length) await loadImage(S.images[0].id);
    // 연 곳(작업 배치·영상 캡처의 넣은 결과)에서 라벨링으로 간다. 받는 사이 다른 보기로 갔으면 끌고 오지 않는다 —
    // 필터와 목록은 그 보기에서도 이 조건이다
    if (VIEW === from && from !== "label") await setView("label");
  } catch (err) {
    toast(`목록을 받지 못했습니다: ${err.message}`, true);
  } finally {
    W.opening = false;
  }
}

/** 이름·'모아서 한꺼번에' 를 고친다. 끝나면 목록을 다시 받는다(지금 이미지의 승인 단추도 loadWorks 가 맞춘다). */
async function workEdit(b, body) {
  if (W.busy.has(b.id)) return;
  W.busy.add(b.id);
  try {
    await api(`/api/work-batches/${b.id}`, { method: "PUT", body });
  } catch (err) {
    toast(err.message, true);
  } finally {
    W.busy.delete(b.id);
  }
  await loadWorks();       // 실패했어도 칸(체크)을 서버의 값으로 되돌린다
}

function workRename(b) {
  const name = prompt(`작업 ${b.id} 의 이름 (비우면 이름 없음 — 종류와 시각으로 보입니다)`, b.name || "");
  if (name === null) return;
  workEdit(b, { name: name.trim() });
}

async function workSetHold(b, hold) {
  await workEdit(b, { hold });
  const now = S.works.find((x) => x.id === b.id);
  if (now && now.hold === hold) {
    toast(hold ? `작업 ${b.id}: 이제 승인(Space)이 '검수 끝' 으로 갑니다 — '데이터셋에 추가' 로 한꺼번에 넣으세요`
      : `작업 ${b.id}: 이제 승인하는 대로 데이터셋에 들어갑니다` + (now.counts.reviewed ? ` (검수 끝 ${now.counts.reviewed}장은 그대로 — '데이터셋에 추가' 로 넣으세요)` : ""));
  }
}

/** 그 배치의 '검수 끝' 을 전부 승인(데이터셋)으로 올린다 — 한 번 묻는다. 라벨링의 머리줄 단추도 이것을 부른다(app.js). */
async function workAddToDataset(b) {
  if (W.busy.has(b.id) || navBusy()) return;
  const c = b.counts;
  if (!c.reviewed) { toast("이 배치에는 '검수 끝' 인 이미지가 없습니다"); return; }
  const rest = [["pending", "검수 대기"], ["unlabeled", "미라벨"]].filter(([k]) => c[k]).map(([k, t]) => `${t} ${c[k]}장`);
  if (!confirm(`작업 ${b.id} 의 '검수 끝' ${c.reviewed}장을 데이터셋(승인)에 넣습니다 — 다음 학습·분할부터 들어갑니다.\n`
    + (rest.length ? `${rest.join(" · ")}은 그대로 남습니다(보지 않은 이미지는 넣지 않습니다).\n` : "")
    + "\n넣을까요?")) return;
  W.busy.add(b.id);
  try {
    // 지금 열어 둔 장이 이 배치의 검수 끝이면 고치던 박스부터 저장한다 — 화면과 다른 박스가 승인되지 않게
    if (!(await saveBeforeLeaving())) return;
    const r = await api(`/api/work-batches/${b.id}/add`, { method: "POST" });
    toast(`작업 ${b.id}: ${r.added}장을 데이터셋에 넣었습니다`);
    await loadState();
    await loadWorks();
    try {
      await loadImageList();               // 라벨링 목록(검수 끝으로 보던 중이면 그 줄들이 빠진다)
      await syncCurrentAfterJob();         // 지금 이미지의 상태가 바뀌었으면 다시 읽는다
      if (!S.cur && S.images.length) await loadImage(S.images[0].id);
    } catch (err) {
      toast(`넣은 뒤 목록·지금 이미지를 다시 받지 못했습니다: ${err.message}`, true);
    }
  } catch (err) {
    toast(err.message, true);
  } finally {
    W.busy.delete(b.id);
    if (VIEW === "work") workRender();
  }
}

function workAct(e) {
  const t = e.target.closest("[data-act]");
  const card = e.target.closest(".wk-card");
  if (!t || !card) return;
  const b = S.works.find((x) => x.id === Number(card.dataset.id));
  if (!b) return;
  const act = t.dataset.act;
  // 체크 칸은 누르면 click 과 change 가 둘 다 온다 — change 에서만 받는다(두 번 보내지 않게)
  if (act === "hold") { if (e.type === "change") workSetHold(b, t.checked); return; }
  if (e.type !== "click") return;
  if (t.blur) t.blur();
  if (act === "open") workOpen(b.id, t.dataset.status || null);
  else if (act === "rename") workRename(b);
  else if (act === "add") workAddToDataset(b);
  else if (act === "predict" || act === "predict-tiles") {
    // 자동 라벨링 작업 — 진행은 위쪽 작업 칸에 뜨고, 끝나면 app.js 가 상태·배치 목록을 다시 받는다(watchJob → loadState)
    startJob("predict", { include_pending: false, tiles: act === "predict-tiles", work_batch: b.id });
  }
}

/* ───────────────────────── 이미지 넣기 ───────────────────────── */

/** 서버 upload.clean_name 과 같은 규칙으로 이름이 받을 형식인지 본다(화면은 거르기만 한다 — 이름을 다듬는 것은 서버다). */
function upExtOk(name) {
  const exts = (W.limits && W.limits.exts) || WK_EXTS;
  const base = String(name || "").replace(/\\/g, "/").split("/").pop().trim().replace(/[. ]+$/, "");
  const dot = base.lastIndexOf(".");
  return dot >= 0 && exts.includes(base.slice(dot).toLowerCase());     // '.jpg' 뿐인 이름도 서버가 받는다(image.jpg 로)
}

/** 고른 파일들을 넣기 칸에 올린다(아직 서버로 보내지 않는다). 형식이 아닌 것·너무 큰 것·한도를 넘는 것은 빼고 알린다. */
function workPick(files) {
  if (W.up && W.up.sending) { toast("넣는 중입니다 — 끝난 뒤 다시 고르세요", true); return; }
  const maxBytes = (W.limits && W.limits.max_bytes) || 200 * 1024 ** 2;
  const maxFiles = (W.limits && W.limits.max_files) || 5000;
  const ok = [], skipped = [];
  let badExt = 0, tooBig = 0;
  for (const f of files) {
    if (!upExtOk(f.name)) badExt++;
    else if (f.size > maxBytes) tooBig++;
    else if (!f.size) skipped.push(`${f.name}: 빈 파일`);
    else ok.push(f);
  }
  if (badExt) skipped.push(`이미지가 아닌 파일 ${badExt}개는 뺐습니다 (받는 형식: ${((W.limits && W.limits.exts) || WK_EXTS).join(" ")})`);
  if (tooBig) skipped.push(`${wkBytes(maxBytes)} 를 넘는 파일 ${tooBig}개는 뺐습니다`);
  if (ok.length > maxFiles) {
    skipped.push(`한 묶음은 ${maxFiles.toLocaleString()}장까지라 뒤의 ${(ok.length - maxFiles).toLocaleString()}장은 뺐습니다 — 나눠서 넣으세요`);
    ok.length = maxFiles;
  }
  if (!ok.length) {
    toast(skipped[0] || "넣을 이미지가 없습니다", true);
    return;
  }
  // 이름순으로 — 폴더째 고르면 순서가 뒤섞여 온다. 숫자는 수로 견준다(frame_2 < frame_10)
  ok.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  W.up = { files: ok, skipped, sid: null, sending: false, cancel: false, done: 0, failed: [] };
  $("#wk-result").classList.add("hidden");
  workRenderUp(true);
  $("#wk-up").scrollIntoView({ block: "nearest" });
}

/** 모든 파일 이름이 `<등록된 키>_…` 로 시작하면 그 키 — 카메라 칸의 처음 값으로 제안한다(영상 캡처가 영상 이름에서 읽는 것과 같다). */
function upGuessCamera(files, cams) {
  const keys = cams.map((c) => c.key).sort((a, b) => b.length - a.length);      // 긴 키부터 (hw-1 과 hw-10)
  const first = keys.find((k) => files[0].name.startsWith(`${k}_`));
  return first && files.every((f) => f.name.startsWith(`${first}_`)) ? first : null;
}

function workRenderUp(fresh = false) {
  const up = W.up, box = $("#wk-up");
  box.classList.toggle("hidden", !up);
  if (!up) return;
  const bytes = up.files.reduce((a, f) => a + f.size, 0);
  $("#wk-up-n").textContent = `— ${up.files.length.toLocaleString()}장 · ${wkBytes(bytes)}`;
  $("#wk-up-skip").textContent = up.skipped.join(" · ");
  const sel = $("#wk-up-camera");
  const cams = S.camReg || [];
  if (fresh || !sel.options.length) {
    let saved = null;
    try { saved = localStorage.getItem(WK_CAM_KEY); } catch { /* 없으면 고르게 */ }
    const guess = upGuessCamera(up.files, cams);
    sel.innerHTML = "";
    const add = (value, text, title = "") => {
      const o = document.createElement("option");
      o.value = value; o.textContent = text; o.title = title;
      sel.appendChild(o);
    };
    add(WK_PICK, "카메라를 고르세요");
    add("", "자동 — 이 묶음용 카메라를 새로 등록 (vid-NN)", "출처를 모를 때 — 이 묶음용 카메라를 cameras.yaml 에 등록해 넣습니다(장소 '모름'). 묶음 전체가 한 카메라가 됩니다");
    for (const c of cams) {
      add(c.key, camLabel(c, [c.kind, c.split && `${c.split} 고정`].filter(Boolean).join(" · ")),
          [c.place, c.kind, c.split && `split 고정 ${c.split}`, `이미지 ${c.n}장`].filter(Boolean).join(" · "));
    }
    // 이름이 전부 한 카메라의 키로 시작하면 그 카메라, 아니면 지난번에 고른 카메라(등록에 남아 있을 때만). '자동' 은 제안하지 않는다
    const want = guess || (saved && cams.some((c) => c.key === saved) ? saved : WK_PICK);
    sel.value = want;
  }
  const cam = sel.value;
  const note = $("#wk-up-cam-note");
  const hit = cams.find((c) => c.key === cam);
  note.textContent = cam === WK_PICK ? "출처(카메라)를 골라야 넣을 수 있습니다 — 분할이 카메라째로 정해집니다. 모르면 '자동'."
    : cam === "" ? "이 묶음 전체를 새 카메라 하나(vid-NN, 장소 '모름' · cctv)로 등록합니다 — 서로 다른 장소의 그림이 섞여 있으면 "
      + "장소별로 나눠 넣거나, 넣은 뒤 이미지 보기의 '옮기기' 로 나누세요."
      : `→ ${W.unlabeled || "unlabeled"}/${cam}/` + (hit && hit.split ? ` · ${hit.split} 고정 카메라` : "");
  note.classList.toggle("warn", cam === WK_PICK);
  $("#wk-up-go").disabled = up.sending || cam === WK_PICK;
  $("#wk-up-go").textContent = up.sending ? "넣는 중…" : `${up.files.length.toLocaleString()}장 넣기`;
  $("#wk-up-cancel").textContent = up.sending ? "멈추기" : "취소";
  sel.disabled = up.sending;
  $("#wk-up-name").disabled = up.sending;
  $("#wk-up-progress").classList.toggle("hidden", !up.sending);
}

async function workUpload() {
  const up = W.up;
  if (!up || up.sending) return;
  const cam = $("#wk-up-camera").value, name = $("#wk-up-name").value.trim();
  if (cam === WK_PICK) { toast("카메라를 고르세요 — 모르면 '자동'", true); return; }
  const n = up.files.length;
  if (!confirm(`${n.toLocaleString()}장을 ${W.unlabeled || "unlabeled"}/${cam || "(새 자동 카메라)"}/ 에 넣고 작업 배치 하나로 등록합니다(미라벨).\n`
    + (cam ? "" : "카메라를 '자동' 으로 두어 이 묶음용 카메라를 cameras.yaml 에 새로 등록합니다(장소 '모름' · cctv — 나중에 영상 캡처 탭의 '고치기…' 로 바꿀 수 있습니다).\n")
    + "이미 있는 이미지와 내용이 같은 파일은 넣지 않습니다. 원본 파일은 그대로입니다.\n\n계속할까요?")) return;
  try { if (cam) localStorage.setItem(WK_CAM_KEY, cam); } catch { /* 저장 못 해도 그만 */ }
  up.sending = true; up.cancel = false; up.done = 0; up.failed = [];
  workRenderUp();
  const bar = $("#wk-up-bar"), text = $("#wk-up-text");
  const progress = (msg) => { bar.style.width = `${(up.done / n) * 100}%`; text.textContent = msg; };
  let sid = null, result = null, error = null;
  try {
    progress("묶음을 시작하는 중…");
    sid = up.sid = (await api("/api/upload", { method: "POST", body: { camera: cam || null, name: name || null } })).sid;
    for (const f of up.files) {
      if (up.cancel) break;
      progress(`${up.done + 1} / ${n}  ${f.name}`);
      let res, data = {};
      try {
        res = await fetch(url(`/api/upload/${sid}/file?name=${encodeURIComponent(f.name)}&size=${f.size}`), { method: "POST", body: f });
        data = await res.json().catch(() => ({}));
      } catch {
        throw new Error("서버와 연결이 끊겼습니다 — 올린 것은 버립니다");
      }
      if (!res.ok) {
        const why = errorText(data.detail) || `HTTP ${res.status}`;
        // 묶음이 없어졌거나(404) 닫혔으면(409) 더 올릴 수 없다. 그 밖(형식·크기·읽지 못함)은 그 파일만 건너뛴다
        if (res.status === 404 || res.status === 409 || res.status === 507 || res.status >= 500) throw new Error(why);
        up.failed.push({ name: f.name, why });
      }
      up.done++;
    }
    if (up.cancel) throw new Error("멈췄습니다 — 올린 것은 버립니다");
    if (up.failed.length === n) throw new Error(`한 장도 받지 못했습니다: ${up.failed[0].why}`);
    progress("넣는 중 — 옮기고 스캔합니다…");
    result = await api(`/api/upload/${sid}/commit`, { method: "POST" });
    sid = null;                           // 넣었다 — 서버가 세션을 지웠다
  } catch (err) {
    error = err;
  }
  if (sid) await api(`/api/upload/${sid}`, { method: "DELETE" }).catch(() => {});     // 넣지 못한 묶음은 버린다
  const failed = up.failed;
  if (result) {
    W.up = null;
    $("#wk-up-name").value = "";
    try { await loadState(); await loadWorks(); await loadImageList(); } catch { /* 라벨링으로 가면 다시 받는다 */ }
    workShowResult(result, failed);
  } else {
    up.sending = false; up.sid = null;
    toast(error ? error.message : "넣지 못했습니다", true);
  }
  await workReload();
}

/** 넣은 결과를 남긴다(다음 묶음을 고를 때까지). r 은 서버 upload.commit 의 결과, failed 는 올리다 건너뛴 파일. */
function workShowResult(r, failed) {
  const lines = [];
  const main = r.work && r.work.main;
  if (main) {
    lines.push(`<b>작업 ${main.id}</b> — ${esc(r.camera)} 에 <b>${main.n}</b>장을 넣었습니다(미라벨).`);
  } else if (r.moved) {
    lines.push(`${esc(r.camera)} 에 ${r.moved}장을 옮겼지만 등록된 것이 없습니다.`);
  } else {
    lines.push("넣은 이미지가 없습니다.");
  }
  if (r.auto === "new") lines.push(`카메라를 '자동' 으로 두어 <b>${esc(r.camera)}</b> 를 새로 등록했습니다(장소 '모름') — 알게 되면 영상 캡처 탭의 '고치기…' 로 바꾸세요.`);
  if (r.duplicates) {
    const ex = (r.duplicate_names || []).map((d) => `${esc(d.name)}${d.same_as ? ` = id ${d.same_as}` : " = 같은 묶음의 앞 파일"}`).join(", ");
    lines.push(`내용이 이미 있는 이미지와 같아 넣지 않음 ${r.duplicates}장` + (ex ? ` (${ex}${r.duplicates > 5 ? " …" : ""})` : "") + ".");
  }
  if (r.scan_duplicates) lines.push(`스캔의 중복 검사에 걸려 등록하지 않음 ${r.scan_duplicates}장.`);
  if (r.exists) lines.push(`같은 이름이 이미 있어 건너뜀 ${r.exists}장.`);
  if (r.missing) lines.push(`받은 파일이 없어 건너뜀 ${r.missing}장.`);
  if (failed && failed.length) {
    lines.push(`올리지 못한 파일 ${failed.length}개: ` + failed.slice(0, 5).map((f) => esc(f.why)).join(" / ") + (failed.length > 5 ? " …" : ""));
  }
  const other = r.work && r.work.other;
  if (other) lines.push(`폴더에 있던 다른 새 이미지 ${other.n}장은 작업 ${other.id}(폴더 스캔)으로 따로 묶었습니다.`);
  if (r.scan_error) lines.push(`<span class="err-text">파일은 옮겼지만 스캔이 멈췄습니다: ${esc(r.scan_error)}<br>고친 뒤 데이터 → 폴더 스캔.</span>`);
  if (r.left) lines.push(esc(r.left));
  if (main) {
    lines.push('다음: <button class="linkbtn" id="wk-res-predict">자동 라벨링</button> 을 돌리거나 '
      + `<button class="linkbtn" id="wk-res-open">작업 ${main.id} 열기</button> 로 들어가 직접 그립니다.`);
  }
  const box = $("#wk-result");
  box.innerHTML = lines.join("<br>");
  box.classList.toggle("err", !!r.scan_error || !main);
  box.classList.remove("hidden");
  if (main) {
    $("#wk-res-open").onclick = () => workOpen(main.id);
    $("#wk-res-predict").onclick = () => startJob("predict", { include_pending: false, tiles: false, work_batch: main.id });
  }
  toast(main ? `작업 ${main.id}: ${main.n}장을 넣었습니다` : "넣은 것이 없습니다", !main || !!r.scan_error);
}

function workUpCancel() {
  const up = W.up;
  if (!up) return;
  if (up.sending) { up.cancel = true; $("#wk-up-text").textContent = "멈추는 중…"; return; }
  W.up = null;
  workRenderUp();
}

/** 끌어다 놓은 것에서 파일을 모은다 — 폴더면 그 안까지. 항목(entry)은 drop 처리기가 끝나기 전에 꺼내야 한다. */
async function workDropFiles(dt) {
  const entries = [...(dt.items || [])].map((it) => (it.webkitGetAsEntry ? it.webkitGetAsEntry() : null)).filter(Boolean);
  if (!entries.length) return [...(dt.files || [])];
  const out = [];
  const walk = async (entry) => {
    if (entry.isFile) {
      out.push(await new Promise((ok, no) => entry.file(ok, no)));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {                         // readEntries 는 한 번에 100개쯤만 준다 — 빈 목록이 올 때까지
        const part = await new Promise((ok, no) => reader.readEntries(ok, no));
        if (!part.length) break;
        for (const e of part) await walk(e);
      }
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

/* ───────────────────────── 이벤트 ───────────────────────── */

$("#wk-list").addEventListener("click", workAct);
$("#wk-done").addEventListener("click", workAct);
$("#wk-list").addEventListener("change", workAct);
$("#wk-done").addEventListener("change", workAct);
$("#wk-done-head").onclick = () => { W.doneOpen = !W.doneOpen; workRender(); };
$("#wk-reload").onclick = (e) => { e.currentTarget.blur(); workReload().catch((err) => toast(err.message, true)); };
$("#wk-go-capture").onclick = () => setView("capture");
$("#wk-add").onclick = () => $("#wk-files").click();
$("#wk-add-dir").onclick = () => $("#wk-dir").click();
for (const id of ["#wk-files", "#wk-dir"]) {
  $(id).addEventListener("change", (e) => {
    const files = [...e.target.files];
    e.target.value = "";                 // 같은 파일을 다시 골라도 change 가 나게
    if (files.length) workPick(files);
  });
}
$("#wk-up-camera").onchange = () => workRenderUp();
$("#wk-up-go").onclick = () => workUpload().catch((err) => toast(err.message, true));
$("#wk-up-cancel").onclick = workUpCancel;
$("#wk-up-name").addEventListener("keydown", (e) => {
  if (e.isComposing || e.keyCode === 229) return;     // 한글 조합 중의 Enter
  if (e.key === "Enter") { e.preventDefault(); $("#wk-up-go").click(); }
});
// 이 화면에 파일·폴더를 끌어다 놓으면 넣기 칸으로. 페이지 안의 그림을 끈 것은 받지 않는다(밖에서 온 파일만 'Files' 를 싣고
// 온다 — 이 보기에는 끌 수 있는 그림이 없다)
$("#work").addEventListener("dragover", (e) => {
  if ([...(e.dataTransfer?.types || [])].includes("Files")) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; }
});
$("#work").addEventListener("drop", (e) => {
  if (![...(e.dataTransfer?.types || [])].includes("Files")) return;
  e.preventDefault();
  workDropFiles(e.dataTransfer).then((files) => { if (files.length) workPick(files); else toast("넣을 파일이 없습니다", true); })
    .catch((err) => toast(`끌어온 것을 읽지 못했습니다: ${err.message || err}`, true));
});
