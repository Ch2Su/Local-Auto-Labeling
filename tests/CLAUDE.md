# tests/ — 테스트 파일마다 지키는 것

2026-11-17 에 저장소 루트 CLAUDE.md '회귀 테스트' 의 뒤쪽 항목을 낱말 그대로 옮겨 왔다(그 파일이 매 세션 읽혀 크기를
줄이느라). 이 파일은 `tests/` 아래 파일을 다룰 때만 읽힌다. 돌리는 법·node 찾기·'버그를 고치면 테스트로 남긴다'·`cfg`
규칙은 루트에 남겼다. 테스트를 더하면 여기와 README '회귀 테스트' 표에 적을 것.

- `tests/test_js_parity.py` 가 '고치면 둘 다' 인 식(`maskCover`·`maskSlabs`·`sameRect`·`applyMaskTemplate`·`capNoTimeName`·`capName`·
  `MASK_MAX`·`MASK_DROP`·`MASK_SAME`·`MASK_MAX_PTS`)을 node 로 돌려 Python 과 견주고, 화면 스크립트(app·gallery·capture·health·work)의 문법도 본다. JS 선언을 글자로
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
  만들어** 라벨 줄 수를 분할·클래스별로 `health.report` 와 견준다 — 가릴 영역이 빼는 박스의 판정(`masks.drops` — 11-24 부터 test 도)과
  칠하는가(`masks.paints`)를 셋 중 하나가 따로 적기 시작하면 여기서 어긋난다. test 장의 `ignore/` 그림이 칠할 바로 그 픽셀(`paint_array`)
  이고 평가 도구가 찾는 자리(`eval_test.ignore_path`)에 있는지도 본다. 그림 파일을 실제로 쓰는 몇 안 되는 테스트다(작은 JPEG 넉 장).
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
  **11-23 Gemini 리뷰:** 사본을 뜨는 동안 들어온 변경 요청이 기다리는지는 미들웨어(`_BackupOnWrite`)를 두 요청으로 불러 본다
  (사본 대역이 Event 로 멈춰 있는 사이 둘째 요청이 쓰지 않는가) — 위의 '스레드 여덟' 은 `run` 만 불러 `due()` 길을 지나지 않아,
  `day` 를 뜨기 전에 바꾸던 경합을 못 잡았다. 뜨다 실패한 사본이 파일을 남기지 않고 그날 다시 뜨는지는 `tests/test_store.py`
  (원본 쪽 연결을, 사본에 조금 쓰고 예외를 내는 대역으로 간다 — `store.copy_db`). 둘 다 고치기 전 코드에서 실패하는 것을 봤다.
- `tests/test_import_aihub_pm.py`(11-22)는 `tools/import_aihub_pm.py` 를 본다 — PM 줄 거르기(`clean_pm`: NaN·화면 밖·같은 물체에
  또 하나), **PM 박스를 단서로만 쓰는 규칙**(`plan_hints`: 탈것·탄 사람 후보의 자리, 모델 클래스를 AI Hub 종류로, 코드 15 는
  그대로, 모델 박스 하나는 PM 박스 하나에만, PM 박스 밖의 세워 둔 탈것은 그대로), 같은 카메라 자리 판정(`same_camera` —
  밝기 상관으로는 다른 화면인 주간·야간을 특징점 정합으로, 특징점도 안 맞는 것을 윤곽 무늬로), 그리고 임시 폴더의 작은 가짜
  AI Hub 폴더(연출 그림은 zip 안에)와 가짜 모델로 색인 → 고르기 → 넣기(파일 바이트·카메라 등록·검수 대기 행·scan 이 건너뛰는지·
  다시 돌리면 이어지는 번호)와 실패했을 때 되돌리기를 끝까지 돌린다. 가짜 프레임은 장면 번호로 정해지는 다각형·선 무늬다 —
  **칸 무늬(가로·세로 변뿐)로 만들면 윤곽 방향 무늬가 장면마다 같아져 전부 같은 자리가 된다**(11-22 에 겪음). 왼쪽 위 16×16
  칸의 밝기가 가짜 모델이 낼 차 수다. 2026-11-22 에 규칙 여섯(코드 15·특징점 정합·윤곽 무늬·같은 물체 중복·후보 자리·
  '모양만 다름')을 하나씩 끄고 돌려 해당 테스트가 실패하는 것을 봤다. 목록 파일은 `ah.resolve` 를 갈아 임시 폴더에 쓴다
  (안 갈면 저장소의 `runs/aihub_pm/` 에 남는다). **11-23 — AI Hub 는 전부 train(11-27 에 풀었다 — 지금은 넣을 때 새 카메라에 고정이 **안** 적히는가, 넣는 실행이 고정 안 된
  카메라를 알리지 않는가를 본다):** 그때는 넣을 때 새 카메라에 `split: train` 이
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
  **같은 날 Gemini 리뷰:** 이름만 바꾼 사본(`zz_a`·`zz_b`)이 앞의 파일의 판정(같은 자리·같은 영상)을 받는가, 깨진 파일과 그 사본이
  둘 다 '못 읽음' 이고 같은 영상의 읽히는 파일을 겹침으로 만들지 않는가(제 폴더를 쓴다 — `picked` 는 두 테스트가 장수까지
  단언해서 건드리지 않았다), DB 가 없으면 네 실행 모두 멈추고 아무것도 만들지 않는가, 보기만 하는 실행이 다각형 열을 뺀 옛 DB
  (`ALTER TABLE … DROP COLUMN` 으로 만든다)에 열을 더하지 않는가, 꼴이 다른 `ah-x` 를 알리는가. 고치기 전 도구로 돌려 넷 다
  실패하는 것을 봤다. 따옴표 없이 쓴 숫자 키(`101:`)의 고정·고치기·빼기는 `tests/test_provenance.py`.
- `tests/test_label_qa_claude.py`(11-23)는 `tools/label_qa_claude.py` 가 **밖으로 보내면 안 되는 그림(AI Hub — 출처 `aihub_pm`
  이거나 카메라 키가 `ah-` 로 시작)을 요청에 싣지 않는지**를 본다(같은 날 리뷰 뒤로 대조군·클래스 설명도 조금 — 끝에). `anthropic` 모듈을 가짜로 갈아 끼우고(`sys.modules`) 키 읽기
  (`load_api_key`)도 갈아 `main()` 을 끝까지 돌린다 — 보내도 되는 그림은 회색, 안 되는 그림은 자홍색 JPEG 이라 **요청에 실린
  그림의 평균 색**으로 어디서 왔는지 안다. 본다: 후보·양성·음성 대조군 어디에도 안 들어가는가, `--dry-run` 이 그 크롭을 안
  만드는가, 이 규칙 전에 남긴 `items.json` 의 항목도 빼는가(그리고 '지운 이미지' 로 잘못 알리지 않는가), 그 그림의 후보뿐이면
  아무것도 안 보내고 멈추는가, 카메라 열이 없는 옛 DB, 규칙이 `import_aihub_pm` 의 `SOURCE` 와 같고 키는 그 도구의 `KEY_RE` 를
  전부 덮으며 더 넓은가(`ah-12`·`ah-0101a` 도 막는다 — 같은 날 Gemini 리뷰: 전에는 `ah-NNNN` 만 막았다). 2026-11-23 에
  규칙 일곱 가지(전체·출처만·키만·그릴 수 있는 그림에 넣기·후보 안 빼기·남은 목록 안 빼기·뒤의 둘을 함께)를 메모리에서만 끄고
  돌려 실패하는 것을 봤다 — 그림이 실제로 나가는 두 경우는 색 단언에서 걸렸다. **대조군과 가릴 영역(같은 날 리뷰):** 보내도 되는
  그림마다 오른쪽 라벨을 통째로 덮는 영역을 두고 씨앗 여덟 개로 뽑아, 양성이 반 넘게 덮인 라벨이 아니고 음성이 영역 밖인지 본다
  (고치기 전에는 씨앗 0 의 음성 대조가 영역 한가운데였다). responder·cone 에 설명이 있고 판정 문구가 라바콘을 말하는지도.
- **작업 배치 (2026-11-23)** — `tests/test_workbatch.py`(store·workbatch·서버 API·스캔 작업·`labeler.py scan`·영상 캡처의 넣기를
  서버째로), `tests/test_upload.py`(웹 이미지 넣기 — 서버 API 와 `UploadManager` 를 따로), `tests/test_js_work.py`(app.js `approveTarget`·
  `approveText`·`workLabel`·`workTodo`·`setStatus`·`setStatusNow`·`nextAfter`·`listUrl`·`tagQuery`, work.js `workFirstStatus`·`workOpen`·
  `upExtOk` — `APP_PARTS`·`OPEN_APP_PARTS`·`WORK_PARTS`), `tests/test_work_batch_tool.py`(`tools/work_batch.py` 를 `sys.argv` 로).
  그림은 작은 진짜 JPEG 이다(스캔·올리기가 열어 본다 — 색이 다르면 바이트가 달라 중복 검사에 걸리지 않는다). 옛 DB 는
  `test_workbatch.OLD_IMAGES`(work_batch 열이 없는 images 표)로 만든다. 2026-11-23 에 규칙 스물일곱 가지를 하나씩 끄고 돌려 전부
  실패하는 것을 봤다(하나는 원래 코드와 뜻이 같은 변형이라 그 대신 '승인 전 확인보다 먼저 갈 상태를 정하는' 변형을 돌려 봤다).
  `test_js_filters.py`·`test_js_health.py`(+`health_harness.js`)·`test_js_views.py` 도 배치 필터·보기에 맞춰 늘렸다.
- **테스트는 `YOLO_AUTOINSTALL=false` 로 돈다**(conftest, 2026-11-23) — Ultralytics 가 올라온 프로세스에서 못 알아보는 그림을
  열면 `pi-heif` 를 이 `.venv` 에 설치하려 들었다(올리기 테스트가 전체 실행에서만 그렇게 실패했다 — 앞선 테스트가 Ultralytics 를
  올려 둔다). `tests/test_yolo_io.py` 가 `image_size` 는 원래의 `Image.open` 으로 여는지 본다(갈아 끼운 것을 흉내 낸 대역으로).
- `tests/test_ignore_regions.py`(11-24)는 **test 의 가릴 영역 = 채점에서 뺄 자리**(사용자 결정 — masks.py 'test 의 영역')를 본다:
  쓰는 곳(`masks.write_ignore` — 학습 세트·내보내기·`test_baseline.build_split`)과 찾는 곳(`wsl/eval_test.ignore_path`)이 같은가,
  박스 안 픽셀 몫(`covered_px`·`drop_ignored`), 그리고 `IgnoreValidator.update_metrics` 가 레터박스 입력 좌표(배율 0.5·위아래 여백
  140)를 원본으로 되돌려 그 자리에 반 넘게 든 예측만 빼는가 — 검증기는 `__new__` 로 만들고 부모 `DetectionValidator.update_metrics`
  를 받아 적는 대역으로 간다(모델·데이터 없이). 그날 `ops.scale_boxes` 줄을 빼고 돌려 이 테스트가 실패하는 것을 봤고, 진짜
  `model.val`(yolo26n·Ultralytics 예제 그림 bus.jpg·CPU)로도 돌려 영역이 없으면 mAP50 0.995, 왼쪽 반을 덮으면 그 자리의 예측이
  빠지는 것을 봤다(스크래치 스크립트 — 모델이 필요해 테스트로 두지 않았다). **Ultralytics 를 올리면 이 테스트도 볼 것** —
  `update_metrics`·예측 사전 꼴·`scale_boxes` 에 기댄다(꼴이 다르면 검증기가 멈춘다).
- `tests/test_its_incident_clips.py`(11-24)는 `tools/its_incident_clips.py` 의 고르기를 네트워크·키 없이 본다 — 반경 안 가장
  가까운 카메라, 칸(약 2km)보다 넓은 반경, 돌발과 같은 도로 유형의 카메라만, 드문 유형·최근 것 먼저, 같은 카메라는 한 번,
  끝난 돌발·현장이 없는 돌발(정체·이동 작업) 빼기, 좌표·주소 없는 줄 버리기, 오류 글에서 키 지우기, mp4 가 아닌 응답은 파일을
  남기지 않기(`urlopen` 을 대역으로).
- `tests/test_segmenter.py`(11-25)는 SAM 클릭의 '작게 나오면 주변만 잘라 다시'(`segmenter._ask_crop`)를 가짜 predictor 로 본다 —
  set_image 에 들어온 것(경로 = 통째, 배열 = 잘라 낸 칸)과 프롬프트에 따라 원본 픽셀 좌표의 사각형 하나를 마스크로 돌려준다
  (잘라 낸 칸이 원본의 어디였는지는 `_image_array` 를 감싸 배열에 붙인다). 본다: 큰 결과는 다시 묻지 않음, 작은 결과·마스크 없음은
  칸으로 다시 묻고 칸 좌표로 옮긴 프롬프트를 넘김, 칸을 그림 안으로 밂, 칸 가장자리에 닿은 결과는 버리고 통째 결과, 좌표를
  원본으로 되돌림, 잘라 넣은 뒤 다음 클릭은 통째 그림을 다시 인코딩, `crop_below: 0` 이면 끔, 잘라 낼 그림을 Ultralytics 와
  같은 함수로 읽음(EXIF 회전 사진). 2026-11-25 에 규칙 여섯을 하나씩 빼고 돌려 전부 실패하는 것을 봤다. 진짜 모델로는
  `runs/sam_click_bench/check_segmenter.py` 가 측정(thybrid 64)과 같은 IoU 가 나오는지 본다(테스트 아님 — 모델·DB 가 필요하다).
- `tests/test_eval_test.py`(11-15)는 `wsl/eval_test.py` 의 `at_conf`(놓침·헛 박스)가 conf 0.25·IoU 0.45 로 세어지는지 본다 —
  Ultralytics 8.4.212 의 검증기는 혼동 행렬을 val 의 conf 0.001·IoU 0.5 로 세서, 같은 119장에서 헛 박스가 에러 없이 20 → 988 이
  됐다(mAP 는 8.4.142 와 같았다 — docs/experiments.md '새 풀 프레임 데이터'). **Ultralytics 를 올리면 이 테스트부터 볼 것.** val 에
  `conf=0.25` 를 넘겨 고치지 말 것(mAP 까지 0.25 위에서만 센다) — 혼동 행렬이 받는 값만 바꾼다(`pin_confusion_thresholds`).
  11-28 에 IoU 문턱별 AP(`ap_iou`, Ultralytics `box.all_ap` — 행은 `ap_class_index` 순서)를 더했다: 클래스마다·전체 평균이 맞는가,
  `all_ap` 가 비면 칸만 빼는가. 같은 날 더한 `--heads`(one2many+NMS·one2one)는 모델이 필요해 테스트가 없다 — 헤드마다 모델을
  새로 읽는 이유(`Detect.fuse` 가 안 쓰는 헤드를 지운다)는 yolo26n 으로 손으로 봤다(docs/experiments.md 11-28 절).
- **Gemini 리뷰 묶음 6 (2026-11-25)** — 전부 고치기 전 코드로 돌려 실패하는 것을 봤다. `tests/test_workbatch.py` 의 꺼진 넣기
  다시 하기(`cut_short_session` — 프레임의 일부·전부를 unlabeled 로 옮겨 두고 상태가 `committing` 인 세션을 손으로 만든다:
  전부 옮긴 때도 스캔하는가, `registered`·`other_new`·배치 장수, 이미 등록된 이름뿐이면 스캔하지 않는가), `tests/test_js_capture.py`
  의 `capShowResult`(옮겨 둔 프레임을 등록한 결과를 '넣은 것이 없습니다' 로 적지 않는가), `tests/test_upload.py` 의 받는 중인
  파일(넣기 409 · 확인하는 사이에 다 받은 파일이 묶음에 드는가 · 늦게 닿은 파일의 이름 예약 · 도중에 죽은 요청),
  `tests/test_capture_commit.py` 의 영상 올리기 상한, `tests/test_store.py`·`tests/test_split.py` 의 '라벨에 남는 박스로 세기'
  (분할 쪽은 가려진 박스 20개짜리 장이 라벨 있는 두 장을 한 분할로 모는 네 장짜리 경우 — 라벨에 남는 수는 테스트가
  `masks.split_boxes` 로 따로 센다), `tests/test_health.py` 의 고정 분할 카메라, `tests/test_predictor.py` 의 타일 모드
  (`tiling.run_component` 만 정해 둔 검출로 갈아 끼운다), `tests/test_yolo_io.py` 의 그림 열기(패키지에 `Image.open` 을 직접
  부르는 줄이 없는가 — **주석·문서 글에도 그 글자를 여는 괄호까지 쓰면 걸린다**, 그리고 새 프로세스로 `import autolabel` 의
  `YOLO_AUTOINSTALL` 기본값), `tests/test_its_incident_clips.py` 의 번호·주소 거르기. **본문을 그대로 받는 처리 함수는
  `helpers.post_endpoint` 로 꺼내 `helpers.StreamBody` 를 넘겨 직접 부른다** — TestClient 는 본문을 한 번에 줘서 '몇 조각째에서
  멈췄나'·'도중에 취소됐다' 를 만들 수 없다. `wsl/sync_dataset.sh` 의 `ignore/` 맞추기는 테스트가 없다(WSL 의 임시 폴더에서
  손으로 돌려 봤다 — 복사 · 지운 그림 걷기 · 원본에 폴더가 없을 때 비우기).
- **Gemini 리뷰 묶음 7 (2026-11-27)** — 전부 고치기 전 코드로 돌려 실패하는 것을 봤다. `tests/test_store.py` 의 '드문 클래스 먼저'
  점수(덮인 자리의 pickup 0.99 가 통째로·꼭 반(다각형)·40% 덮인 세 장과 다 덮인 한 장), `tests/test_health_dataset.py` 의 내보내기
  (라벨만 내보낸 요약 문구 · 영역이 있는 train·test 그림을 깨뜨리면 그 id 를 말하고 멈추는가 · 너무 큰 그림 예외는 `write_masked`
  를 갈아 끼워), `tests/test_jobs.py`(진짜 `JobManager` — 조각은 `_install_tees()` 가 준 tee 에 직접 쓴다: pytest 가 테스트마다
  sys.stdout 을 갈아서 print 는 첫 테스트에서만 로그에 닿는다. CR·LF 는 `chr()` 로 적었다 — 함정 6), `tests/test_js_sync.py`
  (`syncCurrentAfterJob`·`sameBoxes`·한 줄 `const CUR_META_KEYS` — 이름·모양을 바꾸면 `PARTS` 도), `tests/test_yolo_io.py` 의
  모델을 올리는 도구 목록(`import_aihub_pm`·`label_qa`·`import_virat`·`legacy_groups`·`wsl/eval_test.py` — 모델을 올리는 도구를
  더하면 그 목록에 넣을 것).
- **Gemini 리뷰 묶음 8 (2026-11-27)** — 전부 고치기 전 코드로 돌려 실패하는 것을 봤다. `tests/test_tidy.py`('도는 사이' 는 진행
  알림 `progress` 에서 만든다 — 장마다 그 장을 다루기 직전에 불린다. DB 실패는 `autolabel.store._now` 를 예외로 갈아, 파일을
  옮긴 뒤 경로를 고치는 자리에서 낸다), `tests/test_upload.py` 의 카메라 키 바꾸기·지우기, `tests/test_crop_images.py`(원시
  100×60 에 Orientation 6 을 적은 진짜 JPEG — 화면 60×100), `tests/test_workbatch.py` 의 등록 파일 읽는 횟수
  (`server.load_cameras` 를 세는 것으로 갈아), `tests/test_js_lists.py`(`loadState`·`loadSplits`·`listRequest`·`listStale`·
  한 줄 `const LIST_LAST` — 요청마다 풀어 줄 손잡이를 쥐고 응답 순서를 뒤집는다), `tests/test_server.py` 의 목록 상한,
  `tests/test_yolo_io.py` 의 '모델을 올리는 도구의 함수를 가져다 쓰는 도구'(가져오는 줄로 찾는다 — `domain_review.py`).
- `tests/test_wsl_paths.py`(11-28)는 WSL 스크립트가 저장소 자리를 제 위치에서 구하는지 본다 — 파이썬은 ast 로 문서 글(docstring)을 뺀
  문자열에, 셸은 `#` 로 시작하지 않는 줄에 `/mnt/<글자>/` 가 있으면 실패한다(주석·머리말의 노트북 예시는 괜찮다). `train_wsl.main()` 을
  없는 데이터로 불러 안내가 `REPO` 경로를 찍는지도. 셸 스크립트를 실제로 돌리지는 않는다(돌리면 데이터셋이 있는 PC 에서 WSL 홈으로
  복사한다) — 11-28 에 WSL 에서 손으로 돌려 데스크탑의 `/mnt/c/…` 를 찍는 것을 봤다. 고치기 전 코드로 돌려 셋 다 실패하는 것을 봤다.
- `tests/test_paths.py`(11-28, 포터블 앱 1단계)는 `paths.AppPaths` 를 본다 — 개발 모드의 값이 그 전의 `ROOT` 와 같은가(설정 파일·
  상대경로·`weights/`·`web/`), 패키지에서 `__file__` 로 자리를 구하는 곳이 `paths.py`(와 2단계에서 옮길 `wsl_train.py`)뿐인가,
  `LAL_USER_DATA`, 배포판(얼린 실행)의 포터블 표식·OS 기본 자리 — 실행 파일·번들·OS·환경 변수는 `discover()` 인자로 흉내 낸다.
  고치기 전 코드로 돌리면 `__file__` 검사와 `.gitignore` 검사 둘만 실패한다(나머지는 값이 같아서 — 이 단계는 동작을 바꾸지 않는다).
- `tests/test_wsl_optional.py`(11-28, 포터블 앱 2단계)는 WSL 학습이 Windows 의 선택 기능인지 본다 — 새 프로세스로 `import autolabel.server`
  뒤 `autolabel.wsl_train` 이 올라오지 않았는가, `sys.platform` 을 `linux` 로 갈아 `check()`·`/api/wsl`·작업 요청이 WSL 을 부르지
  않는가(`subprocess.run` 을 부르면 실패하는 대역), 작업이 GPU 를 잡지 않는가(`jobs._gpu_owner`), 화면의 `refreshWslButtons`(node —
  `supported: false` 면 감춤, 옛 서버·오류면 보이되 꺼짐). 다른 OS 에서 실제로 돌려 본 것은 아니다. 고치기 전 코드로는 따로 만든
  worktree 에서 돌려 다섯 다 실패하는 것을 봤다(작업 트리를 되돌리는 동안 다른 세션의 자동 커밋이 옛 코드를 담지 않게).
- `tests/test_devices.py`(11-28, 포터블 앱 3단계)는 `devices.resolve_device`·`Config.device` 를 본다 — `torch.cuda.is_available`·
  `torch.backends.mps.is_available` 을 갈아 CUDA·MPS 가 있고 없는 PC 를 흉내 낸다(실제 GPU 없는 PC·macOS 는 아니다). 새 프로세스로
  `import autolabel.config, autolabel.server` 뒤 `torch` 가 올라오지 않았는지도. 고치기 전 config.py 로는 worktree 에서 '키가 없으면
  auto' 테스트가 실패하는 것을 봤다(`devices.py` 만 옮겨 놓고).
- `tests/test_build_vehicle_crops.py`(11-28)는 `tools/build_vehicle_crops.py`(VehicleRefiner 크롭 — docs/vehicle-refiner.md)를
  `main()` 으로 돌린다: 가릴 영역이 반 넘게 덮은 박스는 빼고 세는가(원본 DB 바이트는 그대로), Orientation 6 사진(원시 100×60 → 화면
  60×100, 빨간 자리)을 화면 기준으로 잘라 크롭 평균색이 빨강인가, 넓히기(15%)가 그림 안에서 멈추고 가장자리 표시, 그림 크기가 DB 와
  다르면 건너뛰기, 기준 세트(`images/<분할>/<이름>__<해시>.jpg`)의 test 에 든 카메라의 다른 장까지 `fold=test`, 같은 자리 짝은 한
  그룹·한 fold·`local_only=1`, `assign_folds` 가 그룹을 쪼개지 않고 responder 하나짜리 그룹 다섯을 fold 다섯에 하나씩, 표식 없는
  폴더는 안 쓰고 표식 있는 폴더는 지난 크롭을 지움, 보기만이면 폴더를 안 만듦. 그날 규칙 여덟(가려짐 빼기·EXIF·test 빼기·짝·
  local_only·드문 클래스 먼저·표식·크기 확인)을 메모리에서만 끄고 돌려 전부 실패하는 것을 봤다.
- `tools/test_baseline.py`·`test_tiles.py` 는 테스트가 아니다(`pytest.ini` 가 `tests/` 만 모은다). 테스트가 못 보는 것: 화면 조작(이벤트 처리기의 연결·그림),
  학습·추론, `trainer.py`·`train_wsl.py` 에 따로 있는 이어하기 로직(함정 8), WSL·Runpod, `tools/`.
