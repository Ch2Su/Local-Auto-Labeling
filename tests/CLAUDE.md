# tests/ — 테스트 파일마다 지키는 것

2026-11-17 에 저장소 루트 CLAUDE.md '회귀 테스트' 의 뒤쪽 항목을 낱말 그대로 옮겨 왔다(그 파일이 매 세션 읽혀 크기를
줄이느라). 이 파일은 `tests/` 아래 파일을 다룰 때만 읽힌다. 돌리는 법·node 찾기·'버그를 고치면 테스트로 남긴다'·`cfg`
규칙은 루트에 남겼다. 테스트를 더하면 여기와 README '회귀 테스트' 표에 적을 것.

- `tests/test_js_parity.py` 가 '고치면 둘 다' 인 식(`maskCover`·`maskSlabs`·`sameRect`·`applyMaskTemplate`·`capNoTimeName`·`capName`·
  `MASK_MAX`·`MASK_DROP`·`MASK_SAME`·`MASK_MAX_PTS`)을 node 로 돌려 Python 과 견주고, 화면 스크립트(app·gallery·capture·health)의 문법도 본다. JS 선언을 글자로
  떼므로(맨 왼쪽 `function 이름(` … 맨 왼쪽 `}`, 한 줄 `const`) 그 모양을 바꾸면 테스트의 규칙도 맞춘다. 짝이 새로 생기면 거기에 더할 것.
  다각형 영역(2026-11-19)은 꼬이지 않은·꼬인·픽셀에 맞춘 다각형과 사각형을 섞은 708건을 끝자리까지 견준다(`poly_cover_cases`,
  다각형 입력은 `helpers.random_poly`) — 화면 쪽은 `ptsBox`·`polyMask`·`maskFromWire`·`maskToWire` 를 함께 뗀다.
- **다각형이 없는 박스의 덮는 몫이 그대로인지**는 `tests/test_masks.py` 의 `old_rect_cover`(다각형 전의 식을 옮겨 둔 것)로 본다 —
  사각형 식을 고치면 그 함수도 함께 고칠 것(그대로 두면 '옛 값과 같다' 가 틀린 기준이 된다).
- `tests/test_js_nav.py`(11-12)는 같은 규칙으로 app.js 의 이미지 넘기기(`step`·`imageLoading`·`openImage`·`loadImage`·
  `loadImageNow`)를 떼어 `tests/nav_harness.js` 에서 **그대로** 돌린다 — 키 반복·서버 응답·사진 도착만 가짜 시계로 일으키고,
  이 함수들이 부르는 그 밖의 화면 함수는 아무것도 안 하는 대역이 받는다. ↑·↓ 를 누르고 있으면 계속 넘어가는지(src/autolabel/CLAUDE.md)
  를 지킨다. 이 다섯 함수의 이름을 바꾸면 테스트의 `PARTS` 도 맞춘다.
- `tests/test_js_health.py`(11-15)는 같은 방식으로 health.js 의 화면 열기·이미지 보기로 넘기기(`hcLoad`·`hcLoadHeat`·
  `hcOpenCell`·`hcOpenIds`·`hcOpenFiltered` 와 그것들이 쓰는 함수 — 테스트의 `PARTS`)와 app.js 의 `noFilters`·`filterState`·
  `setFilters`(`APP_PARTS`)를 `tests/health_harness.js` 에서 돌린다 — 기다리는 일(서버 응답·목록 받기)에 단계마다 답하며 그
  사이에 '다른 보기로 감'·'조절 칸을 바꿈' 을 끼워 넣는다. 이름을 바꾸면 그 목록도 맞춘다.
- `tests/test_js_views.py`(11-15)는 app.js 의 보기 표 `VIEWS`(맨 왼쪽 `const VIEWS = {` … `};`)와 `viewOf`·`setView`·
  `viewKeydown` 을 돌린다 — 표에 `key` 없는 보기를 하나 더 넣어 Space·S·V 가 숨은 라벨링으로 새지 않는지, 표의 이름이
  index.html 의 단추·style.css 와 같은지 본다(src/autolabel/CLAUDE.md '보기는 `VIEWS` 표 한 곳').
- 11-16 에 더한 것(Gemini 리뷰 묶음 5): `tests/test_js_filters.py`(app.js `reloadList`·`clearFilters` 와 `listControls`·
  `setListControls` — 목록을 다시 받기 전에 저장하고, 못 하면 필터 칸을 되돌리는가), `tests/test_js_capture.py`(capture.js
  `capReset`·`capFlushStart`·`captureHide`, 11-17 부터 카메라 고르는 칸의 `키 · 장소` — app.js `camLabel` 과 `capRenderCameras`·`capCamEditOpen`·`capCamDelRender`), `tests/test_capture_commit.py`(넣는 도중 죽은 서버 — 손으로 만든 세션 폴더로),
  `tests/test_label_qa.py`(가릴 영역·모델이 모르는 클래스 — 써 둔 `preds.jsonl` 을 `--reuse` 로). JS 쪽은 함수를 글자로
  떼니 이름을 바꾸면 그 파일의 `PARTS` 도 맞춘다.
- `tests/test_health_dataset.py`(11-15)는 **진짜 학습 세트(`trainer.build_dataset`)와 내보내기(`export_yolo`)를 임시 폴더에
  만들어** 라벨 줄 수를 분할·클래스별로 `health.report` 와 견준다 — 가릴 영역이 빼는 박스의 판정(`masks.drops`·`applies_to`)을
  셋 중 하나가 따로 적기 시작하면 여기서 어긋난다. 그림 파일을 실제로 쓰는 몇 안 되는 테스트다(작은 JPEG 넉 장).
- **좌표가 '같은 자리' 인지는 반올림한 값을 견주지 말고 `masks.same_rect`(화면 `sameRect`)로 본다** — 네 좌표의 차가 모두
  `SAME`(7.5e-8) 이하. 11-12 에 테스트로 찾아 고쳤다: 소수 7자리 반올림 키는 픽셀에 딱 맞춘 좌표(2560·1280 폭)가 반올림 경계에
  자주 걸려, 화면이 영역을 {cx, cy, w, h} 로 들었다 되돌리는 끝자리 잡음만으로 다른 자리가 됐다(같은 템플릿을 두 번 더하면 겹쳐
  들어가고 빼기가 남겼다 — 손으로 그린 영역은 0건). 이유·수치는 `masks.py` 의 `SAME`. 좌표를 견주는 새 코드에서도 반올림 키를 만들지 말 것.
- `tests/test_js_pan.py`(11-20)는 같은 규칙으로 app.js 의 라벨링 keydown 처리기(`onKeydown`)와 다각형을 그리는 중의 W·A·S·D 화면
  이동(`panCode`·`panKeyDown`·`panKeyUp`·`panStop`·`panFrame`·`panBy`·`panClamp`, 상수 `PAN_DIR`·`PAN_KEY`·`PAN_SPEED`·`PAN_FAST`·`PAN`)과
  `polyKey`·`clampImg`·`toImage` 를 `tests/pan_harness.js` 에서 **그대로** 돌린다 — 시계·프레임(60fps)·키 입력(500ms 뒤 33ms 마다
  반복)만 가짜다. keyup 은 `panKeyUp`, blur 는 `panStop` 으로 보낸다(app.js 의 keyup·blur 처리기는 이름 없는 함수라 떼지 않는다 —
  연결은 브라우저로만 봤다). 대조로 onKeydown 에서 `PAN_LINE`(W·A·S·D 를 받는 줄)을 뺀 꼴도 돌려, 그리는 중의 W·A·S·D 가 도메인·
  예측 받기·건너뛰기·화면 끌기 모드로 새는 것을 이 모의가 실제로 가르는지 본다. 이름이나 그 줄을 바꾸면 `PARTS`·`CONSTS`·
  `PAN_LINE` 도 맞춘다.
- `tests/test_tiling.py`(11-17)는 타일 추론의 합치기(`tiling.merge_cross`)를 검출 배열만으로 본다 — 차량 클래스끼리는 클래스가
  달라도 겹치면 하나만 남고, 사람과 탈것·덜 겹친 차는 남는가. 모델·GPU 를 쓰지 않는다.
- `tests/test_eval_test.py`(11-15)는 `wsl/eval_test.py` 의 `at_conf`(놓침·헛 박스)가 conf 0.25·IoU 0.45 로 세어지는지 본다 —
  Ultralytics 8.4.212 의 검증기는 혼동 행렬을 val 의 conf 0.001·IoU 0.5 로 세서, 같은 119장에서 헛 박스가 에러 없이 20 → 988 이
  됐다(mAP 는 8.4.142 와 같았다 — docs/experiments.md '새 풀 프레임 데이터'). **Ultralytics 를 올리면 이 테스트부터 볼 것.** val 에
  `conf=0.25` 를 넘겨 고치지 말 것(mAP 까지 0.25 위에서만 센다) — 혼동 행렬이 받는 값만 바꾼다(`pin_confusion_thresholds`).
- `tools/test_baseline.py`·`test_tiles.py` 는 테스트가 아니다(`pytest.ini` 가 `tests/` 만 모은다). 테스트가 못 보는 것: 화면 조작(이벤트 처리기의 연결·그림),
  학습·추론, `trainer.py`·`train_wsl.py` 에 따로 있는 이어하기 로직(함정 8), WSL·Runpod, `tools/`.
