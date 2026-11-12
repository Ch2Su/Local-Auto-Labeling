// tests/test_js_nav.py 가 node 로 돌린다:  node nav_harness.js <app.js 에서 뗀 함수들> <입력 json> <출력 json>
//
// app.js 의 step·imageLoading·openImage·loadImage·loadImageNow 를 고치지 않고 그대로 돌리고, 그 둘레(서버 응답·사진·화면
// 그리기)만 대역으로 바꾼다. 시간은 가짜 시계다 — 키 입력·서버 응답·사진 도착을 시각 순서대로 하나씩 일으키고, 일으킬
// 때마다 밀린 await 를 끝까지 돌린 뒤 다음으로 간다. 그래서 실제 시간과 상관없이 늘 같은 결과가 나온다.
//
// 'use strict' 를 쓰지 않는다 — 아래의 with 로, 뗀 함수가 찾는 이름(S·api·Image·$·renderTagList …)을 대역에서 찾게 한다.

const fs = require("fs");

const [codePath, inPath, outPath] = process.argv.slice(2);
const CODE = fs.readFileSync(codePath, "utf8");

// 무엇을 시키든 조용히 받는 대역 — loadImageNow 가 부르는 화면 함수($·renderTagList·fitView·markCurrent …) 자리에 선다.
// loadImageNow 가 화면 함수를 새로 부르게 되어도 이 파일은 그대로 돈다. (then 이 없어야 await 가 값으로 받는다)
const anything = new Proxy(function () {}, {
  get: (t, k) => (k === "then" ? undefined : k === Symbol.toPrimitive ? () => "" : anything),
  set: () => true,
  apply: () => anything,
  construct: () => anything,
});

async function scenario(sc) {
  let now = 0;
  let order = 0;
  const queue = [];
  const at = (ms, fn) => { queue.push({ t: now + ms, n: order++, fn }); };
  const settle = () => new Promise((done) => setImmediate(done));      // 밀린 await(마이크로태스크)가 다 돈 뒤에 풀린다

  const metaOf = (id) => ({ id, name: `f${id}.jpg`, width: 2560, height: 1440, status: "pending", split: null,
                            uncertainty: 0, boxes: [], masks: [], n_history: 0, updated_at: "t" });
  // 화면 상태 — 목록 rows 줄(id 1..rows), start 번 이미지를 보고 있고 그 사진이 떠 있다
  const S = { images: Array.from({ length: sc.rows }, (_, i) => ({ id: i + 1 })), cur: metaOf(sc.start), img: {},
              imgId: sc.start, loadSeq: 0, loading: 0, imgPending: null, mode: "box" };
  const log = { requests: 0, leftBeforePicture: 0, toasts: [] };

  const world = {
    S,
    url: (path) => path,
    toast: (msg) => { log.toasts.push(String(msg)); },
    saveBeforeLeaving: async () => !sc.saveFails,
    // GET /api/images/{id} — sc.meta ms 뒤에 답한다
    api: (path) => new Promise((resolve, reject) => {
      const id = Number(String(path).split("/").pop());
      log.requests++;
      // 떠나려는 장의 사진이 화면에 있었나 (받지 못하는 사진은 뜰 것이 없으니 세지 않는다)
      if (S.cur.id !== id && S.imgId !== S.cur.id && !sc.failPic.includes(S.cur.id)) log.leftBeforePicture++;
      at(sc.meta, () => (sc.failApi.includes(id) ? reject(new Error("응답 없음")) : resolve(metaOf(id))));
    }),
    // 사진 — src 를 정하면 sc.pic ms 뒤에 onload(failPic 이면 onerror)
    Image: class {
      set src(v) {
        const id = Number(/images\/(\d+)\/file/.exec(String(v))[1]);
        at(sc.pic, () => {
          if (sc.failPic.includes(id)) { if (this.onerror) this.onerror(); return; }
          this.naturalWidth = 2560;
          this.naturalHeight = 1440;
          if (this.onload) this.onload();
        });
      }
    },
  };
  const env = new Proxy(world, {
    // node 의 전역(Promise·Math·encodeURIComponent …)은 그대로 쓰고, 그 밖의 이름은 전부 대역에서 찾는다
    has: (t, k) => typeof k === "string" && (k in t || !(k in globalThis)),
    get: (t, k) => (typeof k !== "string" ? undefined : k in t ? t[k] : anything),
  });
  const app = new Function("env", `with (env) {\n${CODE}\nreturn { step, imageLoading };\n}`)(env);

  const delta = sc.key === "up" ? -1 : 1;
  at(0, () => app.step(delta, false));                                   // 누른 순간 (e.repeat = false)
  for (let t = sc.delay; t < sc.delay + sc.holdMs; t += sc.repeat) {     // 누르고 있는 동안의 반복 입력
    // keydown 처리기는 e.repeat 를 넘긴다. passRepeat=false 는 2026-11-12 전의 처리기(반복도 그냥 step)
    at(t, () => app.step(delta, sc.passRepeat));
  }
  let atRelease = null;
  at(sc.delay + sc.holdMs, () => { atRelease = S.cur.id; });             // 손을 뗐다
  while (queue.length) {
    queue.sort((a, b) => a.t - b.t || a.n - b.n);
    const ev = queue.shift();
    now = ev.t;
    ev.fn();
    await settle();
  }
  return { end: S.cur.id, atRelease, pictureUp: S.imgId === S.cur.id, loading: S.loading, stillLoading: app.imageLoading(),
           requests: log.requests, leftBeforePicture: log.leftBeforePicture, toasts: [...new Set(log.toasts)] };
}

(async () => {
  const out = [];
  for (const sc of JSON.parse(fs.readFileSync(inPath, "utf8"))) out.push(await scenario(sc));
  fs.writeFileSync(outPath, JSON.stringify(out));
})().catch((err) => {
  console.error((err && err.stack) || err);
  process.exit(1);
});
