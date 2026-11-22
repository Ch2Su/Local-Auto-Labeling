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
  `PAN_LINE` 도 맞춘다. 그리는 동안 Q·E·T 를 막는 것(`polyKey`)도 여기서 본다 — 그리는 중이 아닐 때 원래 뜻으로 가는 대조와 함께.
- `tests/test_tiling.py`(11-17)는 타일 추론의 합치기(`tiling.merge_cross`)를 검출 배열만으로 본다 — 차량 클래스끼리는 클래스가
  달라도 겹치면 하나만 남고, 사람과 탈것·덜 겹친 차는 남는가. 모델·GPU 를 쓰지 않는다. 11-20 부터 통째 자동 라벨링의 같은 정리
  (`cross_class_keep`·`plain_class_iou`)도 본다 — 같은 클래스는 건드리지 않고, 차량 규칙은 merge_cross 에 같은 클래스가 겹칠 수
  없는 문턱(1.1)을 준 것과 무작위 400벌에서 남는 것이 같아야 한다(지워지는 박스가 수백 개인 표본이다 — 표본을 바꾸면 그 수를 볼 것).
- `tests/test_predictor.py`(11-20)는 `AutoLabeler.run`(통째)을 가짜 모델로 돌려 DB 에 저장된 박스를 본다 — 모델은 정해 둔 검출을
  Ultralytics 의 `Boxes` 로 돌려준다(`.xyxy`·`.xywhn`·`.cls`·`.conf` 를 진짜와 같게). `labeler._model` 에 넣어 YOLO 를 올리지 않고,
  run 이 파일이 있는지만 보므로 빈 파일을 둔다. 2026-11-20 에 정리를 뺀 꼴로 돌려 실패하는 것을 봤다.
- `tests/test_server.py` 의 하루 사본(11-21)은 `server.datetime`·`store.datetime` 을 naive `now()` 만 바꾸는 가짜 시계(`_Clock` —
  UTC 기록은 진짜 시각)로 갈아 날짜를 넘긴다: 켤 때·같은 날·읽기 요청·날이 바뀐 뒤 첫 변경(사본이 그 변경 **앞**의 상태인지 박스 수로)·
  keep 0·사본을 못 떠도 저장되고 그날은 다시 안 함·동시 요청에 하나만(`_DailyBackup` 을 스레드 여덟로)·프로젝트 탭(`create_multi_app`)으로
  띄운 서버. store 쪽 시계도 갈아 두어
  고치기 전 코드(켤 때만 뜨던)로 돌리면 '날이 바뀐 뒤 첫 변경' 에서만 실패한다 — 2026-11-21 에 그렇게 실패하는 것을 봤다.
- `tests/test_import_aihub_pm.py`(11-22)는 `tools/import_aihub_pm.py` 를 본다 — PM 줄 거르기(`clean_pm`: NaN·화면 밖·같은 물체에
  또 하나), **PM 박스를 단서로만 쓰는 규칙**(`plan_hints`: 탈것·탄 사람 후보의 자리, 모델 클래스를 AI Hub 종류로, 코드 15 는
  그대로, 모델 박스 하나는 PM 박스 하나에만, PM 박스 밖의 세워 둔 탈것은 그대로), 같은 카메라 자리 판정(`same_camera` —
  밝기 상관으로는 다른 화면인 주간·야간을 특징점 정합으로, 특징점도 안 맞는 것을 윤곽 무늬로), 그리고 임시 폴더의 작은 가짜
  AI Hub 폴더(연출 그림은 zip 안에)와 가짜 모델로 색인 → 고르기 → 넣기(파일 바이트·카메라 등록·검수 대기 행·scan 이 건너뛰는지·
  다시 돌리면 이어지는 번호)와 실패했을 때 되돌리기를 끝까지 돌린다. 가짜 프레임은 장면 번호로 정해지는 다각형·선 무늬다 —
  **칸 무늬(가로·세로 변뿐)로 만들면 윤곽 방향 무늬가 장면마다 같아져 전부 같은 자리가 된다**(11-22 에 겪음). 왼쪽 위 16×16
  칸의 밝기가 가짜 모델이 낼 차 수다. 2026-11-22 에 규칙 여섯(코드 15·특징점 정합·윤곽 무늬·같은 물체 중복·후보 자리·
  '모양만 다름')을 하나씩 끄고 돌려 해당 테스트가 실패하는 것을 봤다. 목록 파일은 `ah.resolve` 를 갈아 임시 폴더에 쓴다
  (안 갈면 저장소의 `runs/aihub_pm/` 에 남는다). **11-23 — AI Hub 는 전부 train:** 넣을 때 새 카메라에 `split: train` 이
  적히는가, `pin_plan`·`pin_train`(`--pin-train`)이 고정이 빈 `ah-NNNN` 만 고치는가(사람이 test 로 적은 것·이미지가 이미 val 에
  든 카메라·`hw-01`·키 꼴이 다른 `ah-x` 는 그대로, 이미 train 에 든 것은 막지 않는다), 보기만일 때 안 쓰는가, AI Hub 폴더 없이
  도는가(`main()` 을 `sys.argv` 로). 여러 카메라를 한 번에 고치는 `provenance.set_splits` 는 `tests/test_provenance.py` 가
  본다 — 그 줄만 바뀌는가(LF·CRLF), `edit_camera` 를 하나씩 부른 것과 바이트까지 같은가, 쓰기 전 파일 하나, 없는 키·못 고치는
  한 줄 꼴 항목이 하나라도 있으면 아무것도 안 쓰는가, 다시 읽어 확인하는 그물(`_edit_camera_text` 를 다른 카메라까지 건드리게
  갈아 끼워 본다). 그날 규칙 열세 가지를 메모리에서만 끄고 돌려 전부 실패하는 것을 봤다. **같은 날 — 손으로 고른 그림 점검
  (`--check`):** DB 에 도구가 넣은 둘·사람이 `ah-0101` 로 넣은 하나·직접 모은 카메라의 하나를 두고(`put` — 파일과 행을 함께),
  고른 폴더 13장이 같은 그림·같은 영상·같은 자리(주간·야간)·준 것끼리(영상·자리·바이트)·못 읽음·새 것으로 갈리는지, 직접 모은
  카메라와는 자리를 견주지 않고 바이트만 견주는지, `main()` 이 DB·등록 파일·그림을 그대로 두고 목록만 쓰는지 본다. 그리고
  사람이 `ah-NNNN` 으로 손으로 넣은 영상을 넣는 실행이 다시 고르지 않는지(`known`). **파일 이름은 전부 `T…`·`zz_…` 로 지었다 —
  Windows 는 경로를 대소문자 없이 정렬해서, 소문자로 시작하는 이름을 섞으면 '앞의 파일' 이 OS 마다 달라진다.** 장면 번호는
  서로 다른 것으로 확인된 것(1·2·22–27)만 썼다. 그날 규칙 열한 가지를 메모리에서만 끄고 돌려 전부 실패하는 것을 봤다.
- `tests/test_label_qa_claude.py`(11-23)는 `tools/label_qa_claude.py` 가 **밖으로 보내면 안 되는 그림(AI Hub — 출처 `aihub_pm`
  이거나 카메라 `ah-NNNN`)을 요청에 싣지 않는지**만 본다. `anthropic` 모듈을 가짜로 갈아 끼우고(`sys.modules`) 키 읽기
  (`load_api_key`)도 갈아 `main()` 을 끝까지 돌린다 — 보내도 되는 그림은 회색, 안 되는 그림은 자홍색 JPEG 이라 **요청에 실린
  그림의 평균 색**으로 어디서 왔는지 안다. 본다: 후보·양성·음성 대조군 어디에도 안 들어가는가, `--dry-run` 이 그 크롭을 안
  만드는가, 이 규칙 전에 남긴 `items.json` 의 항목도 빼는가(그리고 '지운 이미지' 로 잘못 알리지 않는가), 그 그림의 후보뿐이면
  아무것도 안 보내고 멈추는가, 카메라 열이 없는 옛 DB, 규칙이 `import_aihub_pm` 의 `SOURCE`·`KEY_RE` 와 같은가. 2026-11-23 에
  규칙 일곱 가지(전체·출처만·키만·그릴 수 있는 그림에 넣기·후보 안 빼기·남은 목록 안 빼기·뒤의 둘을 함께)를 메모리에서만 끄고
  돌려 실패하는 것을 봤다 — 그림이 실제로 나가는 두 경우는 색 단언에서 걸렸다.
- `tests/test_eval_test.py`(11-15)는 `wsl/eval_test.py` 의 `at_conf`(놓침·헛 박스)가 conf 0.25·IoU 0.45 로 세어지는지 본다 —
  Ultralytics 8.4.212 의 검증기는 혼동 행렬을 val 의 conf 0.001·IoU 0.5 로 세서, 같은 119장에서 헛 박스가 에러 없이 20 → 988 이
  됐다(mAP 는 8.4.142 와 같았다 — docs/experiments.md '새 풀 프레임 데이터'). **Ultralytics 를 올리면 이 테스트부터 볼 것.** val 에
  `conf=0.25` 를 넘겨 고치지 말 것(mAP 까지 0.25 위에서만 센다) — 혼동 행렬이 받는 값만 바꾼다(`pin_confusion_thresholds`).
- `tools/test_baseline.py`·`test_tiles.py` 는 테스트가 아니다(`pytest.ini` 가 `tests/` 만 모은다). 테스트가 못 보는 것: 화면 조작(이벤트 처리기의 연결·그림),
  학습·추론, `trainer.py`·`train_wsl.py` 에 따로 있는 이어하기 로직(함정 8), WSL·Runpod, `tools/`.
