// tests/test_js_health.py 가 node 로 돌린다:  node health_harness.js <health.js 에서 뗀 함수들> <입력 json> <출력 json>
//
// health.js 의 '이미지 보기로 넘기기'(hcOpenCell·hcOpenIds·hcOpenFiltered 와 그것들이 쓰는 함수)를 고치지 않고 그대로 돌리고,
// 그 둘레(서버 응답·목록 받기·보기 바꾸기·화면 칸)만 대역으로 바꾼다. 기다리는 일(api·loadImageList·openIds)은 전부 한 줄로
// 세워 두고, 시나리오의 단계가 하나씩 답하거나(answer) 실패시킨다(fail) — 그 사이에 '다른 보기로 감'·'조절 칸을 바꿈' 을 끼워
// 넣어 늦게 온 응답을 만든다. 단계마다 밀린 await 를 끝까지 돌린 뒤 다음으로 간다(nav_harness.js 와 같은 방식).
//
// 'use strict' 를 쓰지 않는다 — 아래의 with 로, 뗀 함수가 찾는 이름(S·H·api·$·VIEW …)을 대역에서 찾게 한다.

const fs = require("fs");

const [codePath, inPath, outPath] = process.argv.slice(2);
const CODE = fs.readFileSync(codePath, "utf8");

// 무엇을 시키든 조용히 받는 대역 — 여기서 보지 않는 화면 함수(hcTip·hcDrawHeat·hcHeatBackground …) 자리에 선다
const anything = new Proxy(function () {}, {
  get: (t, k) => (k === "then" ? undefined : k === Symbol.toPrimitive ? () => "" : anything),
  set: () => true,
  apply: () => anything,
  construct: () => anything,
});

async function scenario(sc) {
  const settle = () => new Promise((done) => setImmediate(done));      // 밀린 await(마이크로태스크)가 다 돈 뒤에 풀린다
  const pending = [];                                                  // 답을 기다리는 일 — 먼저 부른 것부터
  const wait = (kind, what) => new Promise((resolve, reject) => { pending.push({ kind, what, resolve, reject }); });

  // 화면 칸 — 값과 'on' 표시만 든다
  const els = {};
  const $ = (sel) => els[sel] || (els[sel] = {
    value: "", on: false, textContent: "",
    classList: { toggle: (name, v) => { if (name === "on") els[sel].on = !!v; }, add() {}, remove() {} },
  });
  // 처음 상태: 왼쪽 필터가 여럿 걸려 있고, 점검 화면의 조절 칸은 그려진 히트맵(H.heatQ)과 **다른** 값이다
  Object.assign($("#filter-status"), { value: "pending" });
  Object.assign($("#filter-camera"), { value: "old-cam", on: true });
  Object.assign($("#filter-domain"), { value: "cctv", on: true });
  Object.assign($("#filter-split"), { value: "train", on: true });
  Object.assign($("#hc-scope"), { value: "all" });
  Object.assign($("#hc-masked"), { value: "keep" });

  const S = { filterClasses: new Set([1]), filterEmpty: false, filterCamera: "old-cam", filterDomain: "cctv",
              filterSplit: "train", tagF: { inc: [7], exc: [9], any: true, none: false }, images: [{ id: 11 }] };
  const H = { data: { list_status: "live", scope: "all" }, heat: { cells: [] }, seq: 3, heatSeq: 5, opening: false,
              heatQ: { scope: "approved", masked: "drop", cls: "", cam: "cam-01" }, cls: "2", cam: "", hover: -1 };
  const filters = () => ({
    status: $("#filter-status").value, classes: [...S.filterClasses], empty: S.filterEmpty,
    camera: S.filterCamera, domain: S.filterDomain, split: S.filterSplit, tagF: JSON.parse(JSON.stringify(S.tagF)),
    selects: ["#filter-camera", "#filter-domain", "#filter-split"].map((s) => [$(s).value, $(s).on]),
  });
  const log = { api: [], opened: [], views: [], toasts: [], listedWith: [], loaded: [], syncs: 0, renders: 0 };

  const world = {
    S, H, $,
    VIEW: "health",
    hcN: (n) => String(n),
    navBusy: () => false,
    toast: (msg, bad) => { log.toasts.push([String(msg), !!bad]); },
    api: (path) => { log.api.push(String(path)); return wait("api", path); },
    openIds: (ids) => { log.opened.push([...ids]); return wait("ids", ids); },
    loadImageList: () => {
      world.syncListControls();                    // 진짜 loadImageList 도 받기 전에 칸을 지금 조건으로 맞춘다
      log.listedWith.push(filters());
      return wait("list").then((images) => { S.images = images; return images; });
    },
    loadImage: async (id) => { log.loaded.push(id); },
    setView: async (v) => { log.views.push(v); world.VIEW = v; },
    syncListControls: () => { log.syncs++; },
    syncClassFilter: () => {},
    tagFilter: { render: () => { log.renders++; } },
  };
  const env = new Proxy(world, {
    // node 의 전역(Promise·URLSearchParams·Set …)은 그대로 쓰고, 그 밖의 이름은 전부 대역에서 찾는다
    has: (t, k) => typeof k === "string" && (k in t || !(k in globalThis)),
    get: (t, k) => (typeof k !== "string" ? undefined : k in t ? t[k] : anything),
  });
  const hc = new Function("env", `with (env) {\n${CODE}\nreturn { hcOpenCell, hcOpenIds, hcOpenFiltered, hcLoadHeat };\n}`)(env);

  const answer = (ok, value) => {
    const p = pending.shift();
    if (!p) throw new Error(`${sc.name}: 답할 일이 없습니다`);
    if (!ok) p.reject(new Error(value));
    else p.resolve(value !== undefined ? value : p.kind === "list" ? [{ id: 42 }] : undefined);
  };
  for (const [step, ...args] of sc.steps) {
    if (step === "cell") hc.hcOpenCell(...args);
    else if (step === "ids") hc.hcOpenIds(...args);
    else if (step === "filtered") hc.hcOpenFiltered(...args);
    else if (step === "loadheat") hc.hcLoadHeat();
    else if (step === "leave") { H.seq++; H.heatSeq++; world.VIEW = "label"; }      // healthHide + setView('label')
    else if (step === "reheat") H.heatSeq++;                                         // 히트맵을 다시 받기 시작했다
    else if (step === "controls") { H.cls = args[0].cls; H.cam = args[0].cam; }      // 조절 칸을 바꿨다
    else if (step === "answer") answer(true, args[0]);
    else if (step === "fail") answer(false, args[0]);
    else throw new Error(`${sc.name}: 모르는 단계 ${step}`);
    await settle();
  }
  return { ...log, view: world.VIEW, filters: filters(), opening: H.opening, heatQ: H.heatQ, waiting: pending.length };
}

(async () => {
  const out = {};
  for (const sc of JSON.parse(fs.readFileSync(inPath, "utf8"))) out[sc.name] = await scenario(sc);
  fs.writeFileSync(outPath, JSON.stringify(out));
})().catch((err) => {
  console.error((err && err.stack) || err);
  process.exit(1);
});
