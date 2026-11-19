// tests/test_js_pan.py 가 node 로 돌린다:  node pan_harness.js <app.js 에서 뗀 코드> <고치기 전 꼴의 코드> <입력 json> <출력 json>
//
// app.js 의 keydown 처리기(onKeydown)와 다각형을 그리는 중의 W·A·S·D 화면 이동(panCode·panKeyDown·panKeyUp·panStop·panFrame·
// panBy·panClamp)·polyKey·clampImg·toImage 를 고치지 않고 그대로 돌린다. 가짜인 것은 시계(performance.now)·프레임
// (requestAnimationFrame, 60fps)·키 입력뿐이다 — 시각 순서대로 하나씩 일으키므로 늘 같은 결과가 나온다. 처리기가 부르는 그 밖의
// 화면 함수는 대역이 받고, '원래 뜻'(도메인·건너뛰기·모드 바꾸기·예측 받기 …)과 다각형 닫기·버리기는 불렸는지만 적는다.
// keyup 은 app.js 의 keyup 처리기처럼 panKeyUp 으로, 창이 포커스를 잃은 것(blur)은 panStop 으로 보낸다.
//
// 'use strict' 를 쓰지 않는다 — 아래의 with 로, 뗀 함수가 찾는 이름(S·canvas·drag·draw …)을 대역에서 찾게 한다.

const fs = require("fs");

const [codePath, oldPath, inPath, outPath] = process.argv.slice(2);
const CODES = { new: fs.readFileSync(codePath, "utf8"), old: fs.readFileSync(oldPath, "utf8") };

// 무엇을 시키든 조용히 받는 대역 — onKeydown 이 부르는 그 밖의 화면 함수(closeModal·renderMode …) 자리에 선다
const anything = new Proxy(function () {}, {
  get: (t, k) => (k === "then" ? undefined : k === Symbol.toPrimitive ? () => "" : anything),
  set: () => true,
  apply: () => anything,
  construct: () => anything,
});

const FRAME = 1000 / 60;
// 불렸는지 적어 둘 화면 함수 — 인자는 글자·숫자만 남긴다
const RECORDED = ["setDomain", "skipOrAdvance", "setMode", "acceptFinding", "setStatus", "step", "fitView", "undo",
  "selectBox", "togglePred", "tagAndNext", "samRefineSelected", "toggleHistory", "openSamePlace", "setClass", "deleteMask",
  "polyBackspace", "toast", "closeMenus"];

function keyEvent(ev) {
  return {
    type: ev.type === "up" ? "keyup" : "keydown", key: ev.key, code: ev.code ?? "", repeat: !!ev.repeat,
    shiftKey: !!ev.shift, ctrlKey: !!ev.ctrl, metaKey: false, altKey: false, target: { tagName: "CANVAS" },
    prevented: false,
    preventDefault() { this.prevented = true; },
  };
}

function scenario(sc) {
  let now = 0;
  let draws = 0;
  let rafId = 0;
  const frames = new Map();
  const calls = [];
  // 이미지 sc.W×sc.H 를 scale 배로 (ox, oy) 에 놓고 본다. drawing 이면 점 두 개를 찍은 다각형을 그리는 중이다
  const S = {
    cur: { id: 7, width: sc.W, height: sc.H }, imgId: 7, view: { scale: sc.scale, ox: sc.ox, oy: sc.oy },
    poly: sc.drawing ? { id: 7, pts: [[1000, 500], [1100, 520]], cur: null, sp: sc.mouse || null } : null,
    mode: "mask", ctrl: false, predOn: !!sc.predOn, findings: [{ kind: "missing" }], msel: -1, sel: -1, boxes: [],
  };
  const world = {
    S,
    drag: sc.dragPan ? { mode: "pan", sx: 0, sy: 0, ox: sc.ox, oy: sc.oy } : null,
    canvas: { clientWidth: sc.cw, clientHeight: sc.ch },
    performance: { now: () => now },
    requestAnimationFrame: (fn) => { frames.set(++rafId, fn); return rafId; },
    cancelAnimationFrame: (id) => { frames.delete(id); },
    draw: () => { draws++; },
    modalOpen: () => false,
    viewKeydown: () => false,
    renderMode: () => {},
    FIND: { accept: { missing: true } },
    document: { querySelector: () => null },
    // 닫기·버리기는 실제처럼 그리던 다각형을 비운다
    polyClose: () => { calls.push(["polyClose"]); S.poly = null; },
    polyCancel: () => { calls.push(["polyCancel"]); S.poly = null; },
  };
  for (const name of RECORDED) {
    world[name] = (...args) => { calls.push([name, ...args.filter((a) => typeof a === "string" || typeof a === "number")]); };
  }
  const env = new Proxy(world, {
    // node 의 전역(Math·Set …)은 그대로 쓰고, 그 밖의 이름은 전부 대역에서 찾는다
    has: (t, k) => typeof k === "string" && (k in t || !(k in globalThis)),
    get: (t, k) => (typeof k !== "string" ? undefined : k in t ? t[k] : anything),
  });
  const app = new Function("env", `with (env) {\n${CODES[sc.code || "new"]}\n`
    + "return { onKeydown, panKeyUp, panStop, PAN, PAN_SPEED, PAN_FAST };\n}")(env);

  // 키 입력 등은 적힌 순서(n)대로, 프레임은 같은 시각의 키 입력 뒤에. noFrames 사이에는 프레임이 없다(탭이 멈춤)
  const queue = sc.events.map((ev, n) => ({ ...ev, n }));
  for (let k = 1; k * FRAME <= sc.end; k++) {
    const t = k * FRAME;
    if (!(sc.noFrames && t > sc.noFrames[0] && t < sc.noFrames[1])) queue.push({ t, type: "frame", n: 1e9 + k });
  }
  queue.sort((a, b) => a.t - b.t || a.n - b.n);
  const marks = {};
  let keydowns = 0, prevented = 0;
  for (const ev of queue) {
    now = ev.t;
    if (ev.type === "frame") {
      const due = [...frames.values()];
      frames.clear();
      for (const fn of due) fn(now);
    } else if (ev.type === "down") {
      const e = keyEvent(ev);
      app.onKeydown(e);
      keydowns++;
      if (e.prevented) prevented++;
    } else if (ev.type === "up") {
      app.panKeyUp(keyEvent(ev));
    } else if (ev.type === "blur") {
      app.panStop();
    } else if (ev.type === "closeByMouse") {           // 첫 점을 눌러 닫았다(polyClick → polyClose)
      S.poly = null;
    } else if (ev.type === "mark") {
      marks[ev.name] = { ox: S.view.ox, oy: S.view.oy };
    }
  }
  return {
    ox: S.view.ox, oy: S.view.oy, marks, calls, keydowns, prevented, draws,
    held: app.PAN.keys.size, rafPending: frames.size, raf: app.PAN.raf,
    poly: S.poly ? { n: S.poly.pts.length, cur: S.poly.cur } : null,
    drag: world.drag ? { ox: world.drag.ox, oy: world.drag.oy } : null,
    speed: app.PAN_SPEED, fast: app.PAN_FAST,
  };
}

try {
  const out = JSON.parse(fs.readFileSync(inPath, "utf8")).map(scenario);
  fs.writeFileSync(outPath, JSON.stringify(out));
} catch (err) {
  console.error((err && err.stack) || err);
  process.exit(1);
}
