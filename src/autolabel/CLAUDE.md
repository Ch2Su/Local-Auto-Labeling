# src/autolabel — UI 조작·저장 경로에서 지킬 것

2026-11-10 에 저장소 루트 CLAUDE.md 에서 낱말 그대로 옮겨 왔다(그 파일이 매 세션 읽혀 크기를 줄이느라). 글 속의 '위·아래 …' 는
루트 CLAUDE.md 의 절을 가리킨다.
이 파일은 `src/autolabel/` 아래 파일을 다룰 때만 읽힌다.

사용자 문서·단축키는 README '검수 UI 사용법'. 작업자가 알아야 할 것만:
- **이 폴더를 고친 뒤에는 회귀 테스트를 돌릴 것**(2026-11-12, `tests/` — `.venv\Scripts\python.exe -m pytest`, 루트 CLAUDE.md
  '회귀 테스트'). 아래의 '고치면 둘 다' 가운데 `maskCover`·`maskSlabs`·`sameRect`·`applyMaskTemplate`·`capNoTimeName`·`MASK_MAX`·
  `MASK_SAME`·`MASK_MAX_PTS` 는 `tests/test_js_parity.py` 가 화면과 서버를 대조한다.
- **편집은 0.6초 뒤 자동 저장된다**(`app.js` `markDirty`·`flushSave`) — UI 를 시험하면 실제 작업물이 바뀐다. 먼저
  `GET /api/images/{id}` 로 떠 두고 끝나면 `PUT /api/images/{id}/boxes` 로 되돌릴 것. 경합 시험은 **DB 복사본**(sqlite `backup()`)
  으로 띄운 서버에서 한다.
- **저장은 한 번에 하나만 돈다**(`S.saving`·`S.rev`). 저장이 실패하면 이미지를 넘기거나 승인하지 않는다(`saveBeforeLeaving`).
  응답을 기다리는 곳(SAM·예측·이미지 로드·변경 내역)은 **요청한 이미지 id 를 붙잡아 두고 바뀌었으면 결과를 버린다** — 새
  비동기 호출을 넣을 때도 그렇게 할 것.
  - **목록을 바꾸는 길도 저장이 먼저다 (2026-11-16)** — `reloadList`(필터·정렬·묶음 크기 칸)·`openIds`·데이터 점검의
    `hcOpenFiltered`. 전에는 목록부터 받고 첫 장을 열 때에야 저장을 봐서, 저장이 거절되면(test 이미지의 가릴 영역 409·잘못된 박스
    422) 목록은 새 조건인데 가운데에는 그 목록에 없는 옛 이미지가 남았다. 못 하면 목록을 그대로 두고 칸을 **목록을 받았을 때의
    조건**(`S.listAt` — `loadImageList` 가 받을 때마다 `listControls()` 로 적는다)으로 되돌린다(`setListControls`). 목록을 다시
    받는 새 길은 `reloadList` 를 거치고, 필터 칸을 새로 만들면 `filterState`·`setFilters` 에 넣을 것(안 넣으면 되돌릴 때 빠진다).
    `tests/test_js_filters.py` 가 이 함수들을 그대로 돌린다.
- 여러 장을 연달아 고친 뒤에는 **박스가 줄어든 저장을 확인할 것**(상단 '변경 기록' 이나 `GET /api/history?shrunk=1`).
- **박스를 바꾸는 길은 전부 `store._replace_boxes` 를 지난다**(변경 이력 `box_history` 에 그 이미지의 박스 목록 전후를 JSON 으로
  남긴다). 새 경로도 그걸 거치게 하고, `_write_lock` 은 **부르는 쪽이** 쥔다(안에서 다시 잡으면 멈춘다).
  - **이력의 목록은 2026-11-11 부터 `hist_lists` 에 한 번만 둔다**(사용자 요청 — 이력 줄이기). `box_history` 의 before·after·
    masks_before·masks_after 는 새 줄이면 `#<hist_lists.id>`, 그 전 줄이면 JSON 글자다 — 쓰는 길은 `_hist_put`, 읽는 길은
    `_hist_bodies` → `_hist_text`(`box_history`·`history_entry`·`revert_batch`). 줄은 저장마다 그대로 남아(되돌릴 수 있는 시점·
    `n_before`/`n_after`·'줄어든 저장' 은 전과 같다) 같은 목록만 함께 쓴다 — 한 줄의 '전' 은 앞 줄의 '후', 영역만 바꾼 저장의
    박스 전후는 같은 목록이다. 이력 JSON 은 `_box_key` 와 같은 자리(좌표 7·conf 6)로 반올림하고(영역도 7자리 — 그 차는
    `masks.SAME` 안이라 되읽은 영역이 원래와 같은 자리다. 자릿수를 줄이면 `SAME` 도 볼 것) 빈칸 없이 쓴다(템플릿
    `mask_templates.rects` 는 반올림하지 않는다). 목록에는 FOREIGN KEY 가 없어 `delete_images` 가 아무 줄도 안 가리키는 목록을
    지운다(`_hist_gc`) — 이미지를 지우는 새 길을 만들면 그것을 거칠 것. **11-11 전 코드는 새 줄을 못 읽는다**(`json.loads('#12')`).
    11-11 실측(DB 사본): 박스 100개 장에 영역을 하나씩 300개까지 더하는 저장 298번이 옛 모양 11.4MB → 1.9MB, 오늘 이력 636줄을
    이 모양으로 바꿨다면 9.85MB → 약 1.85MB. 옛 줄은 바꾸지 않았다(읽기는 그대로 된다).
  - 파일을 바꾼 이미지(`store.replace_file` — VIRAT `--recrop` 은 source predict, `tools/crop_images.py` 는 crop)는 그 전 이력의
    박스가 옛 화면 좌표다. crop 은 서버가 그 전 이력으로 되돌리기를 막지만(`store.crop_entries`), VIRAT 자르기 이력은 가려내지 못한다.
    가릴 영역이 있는 이미지에 `masks=`(새 그림 기준으로 옮긴 영역)를 안 주면 `replace_file` 이 ValueError 를 낸다(아무것도 안 쓴다).
- **가릴 영역 (2026-11-09, `src/autolabel/masks.py`)** — 라벨하기 애매한 물체를 검수 화면 `M` 모드로 덮는 사각형·다각형(다각형은
  11-19 — 아래 '다각형')(무엇을 덮는지는 docs/label-rules.md 한 곳에 적는다 — 기준을 바꾸면 거기와 화면 문구 `index.html`·`app.js` 를 볼 것).
  DB `masks` 표(정규화 xyxy + 다각형 꼭짓점 `poly`, EXIF 회전을 적용한 화면 기준)에만 있고 **원본 그림은 그대로**다. 학습 세트(`trainer.build_dataset`)와
  이미지를 함께 내보내기(`export_yolo`)만 그 자리를 회색 114 로 칠한 JPEG 을 쓰고, 영역들의 합집합이 박스 넓이의 반 이상을 덮는
  박스는 그 라벨 파일에서만 뺀다(DB 박스는 그대로). **test 는 칠하지 않는다** — 서버가 test 이미지(분할 test·test 고정 카메라)에
  영역을 받지 않고(409, 지우기만 받는다) 학습 세트도 test 분할은 원본 그대로다. 원본을 직접 읽는 도구(test_baseline·
  test_tiles 등)는 영역을 모른다. **`tools/label_qa.py` 는 11-16 부터 영역을 본다** — 반 이상 덮인 라벨과 예측을 견주지
  않는다(학습 세트와 같은 `masks.drops`·`applies_to`, 전부 보려면 `--keep-masked`). 그날 실제 DB(사본)로 재니 후보가 162건 →
  81건이 됐고, 모델이 모르는 클래스의 라벨도 견주지 않게 하니(아래 '예측 비교') 53건이 됐다.
  - 저장(PUT boxes)의 `masks` 는 **그 화면에서 영역을 손댄 뒤로만** 싣는다(`S.masksTouched`) — 안 실으면(옛 화면 포함) 서버가
    그대로 둔다. 영역이 바뀐 저장은 박스가 그대로여도 이력에 남고(`box_history.masks_before·masks_after`, 박스만 바뀐 줄은 NULL),
    되돌리기는 그 줄에 영역이 있을 때만 영역도 되돌린다. 되돌리기(Ctrl+Z) 한 칸은 박스와 영역을 함께 쌓는다.
  - 가릴 영역 모드에서는 영역만 고르고 옮긴다(박스는 B). 덮는 몫은 `masks.covered` 와 `app.js` 의 `maskCover` 가 같은 식이다
    (좌표 압축으로 합집합 — 영역마다 제 칸에 표시하고 표시된 칸을 줄·칸 순서로 더한다. 다각형이 걸린 박스는 세로 띠 — 아래
    '다각형') — 고치면 둘 다. 화면은 그릴 때마다
    박스마다 부른다. 11-11 전의 '칸마다 모든 영역 견주기' 는 한 박스에 걸친 영역 수의 세제곱이라 되살리지 말 것.
  - **다각형 (2026-11-19, 사용자 요청 — 대각선 모양을 덮으려면 작은 사각형을 여러 개 그려야 했다. 설계는 사용자와 정했다)**.
    영역은 사각형 `(x1, y1, x2, y2)` 와 다각형 `((x, y), …)`(꼭짓점 3개 이상, 그린 순서) 두 꼴이고 `masks.is_poly`(첫 값이 점인가)
    하나로 가른다. JSON(이력 `hist_lists`·템플릿 `mask_templates.rects`·API)도 같은 꼴 — 사각형 `[x1, y1, x2, y2]`, 다각형
    `[[x, y], …]`(`masks.to_wire`·`as_mask`). **DB 는 `masks.poly` 열(꼭짓점 JSON, 반올림 안 함)이고 x1·y1·x2·y2 에는 바깥
    사각형을 둔다**(`masks.row_values`·`from_row`, store 의 `_MASK_COLS`·`_read_masks`) — `Store` 가 DB 를 열 때 `ALTER TABLE ADD
    COLUMN` 으로 더한다(사용자 결정 — 실제 DB 는 서버를 새 코드로 다시 켤 때). 옛 코드는 바깥 사각형만 읽고, 옛 코드로 그 장의
    영역을 다시 저장하면 다각형이 바깥 사각형으로 바뀐다. **`masks` 를 SQL 로 바로 읽는 새 코드는 poly 도 읽을 것**(label_qa 는
    열이 없는 옛 DB 를 NULL 로 읽는다. `runs/compare_1118_pretrain/make_extra.py` 는 다른 세션의 실험 스크립트라 바꾸지 않았다 —
    다각형이 있는 DB 에 쓰면 다각형을 바깥 사각형으로 칠한다).
    - **덮는 몫**: 다각형이 걸리지 않은 박스는 옛 좌표 압축 그대로(끝자리까지 같다 — `tests/test_masks.py` 가 옛 식을 옮겨 두고
      견준다), 다각형이 하나라도 걸리면 세로 띠(`masks._slab_union` ↔ `app.js maskSlabs`, 연산 순서까지 같다): 띠 경계 = 박스·
      사각형의 x, 꼭짓점 x, 변끼리·변과 가로선(박스·사각형 위아래)이 엇갈리는 x — 그 사이에서는 단면 길이가 직선이라 '가운데
      단면 × 폭' 이 정확한 넓이다. 다각형의 안은 **홀짝 규칙**(칠하기·화면 그림 `fill("evenodd")`·고르기 `inPoly` 가 모두 같다).
      **칠하기는 픽셀 중심이 안이면**(`masks.poly_pixels` — 줄마다 중심 높이에서 켜고 끄기). 사각형 칠하기(`pixel_rect`)는 그대로.
    - **같은 자리**: 다각형끼리는 꼭짓점 수가 같고 순서대로 모든 좌표가 `SAME` 안(`same_rect`·`sameRect` — 이름은 사각형만 있던
      때의 것), 사각형과 다각형은 늘 다르다. 그래서 템플릿 더하기·빼기·이력 되돌리기·묶음 되돌리기가 사각형과 같은 규칙으로 돈다.
      화면은 다각형 꼭짓점을 받은 값 그대로 든다(사각형처럼 {cx, cy, w, h} 로 오가지 않는다 — cx·cy·w·h 는 바깥 사각형으로 따로
      센다, `polyMask`).
    - **한도**: 꼭짓점 100개(`masks.MAX_POINTS`·`app.js MASK_MAX_PTS` — 덮는 몫이 변 수의 제곱으로 는다. 서버는 받을 때 400,
      `clean` 은 이것으로 버리지 않는다), 한 장 500개에 다각형도 하나. 11-19 실측(1920×1080, 박스 101개): 다각형 20개(꼭짓점 10)
      얹어 화면 한 번 그리기 4.4ms·오른쪽 목록 2.6ms, 60개(꼭짓점 20) 30ms·24ms, Python `covered` 101박스 4ms·56ms.
    - **화면 조작(사용자 결정)**: M 모드에서 끌면 사각형, 끌지 않고 누르면 다각형 첫 꼭짓점(빈 곳·이미지 안 — `polyStart`).
      그리는 중인 것은 `S.poly`(이미지 픽셀 꼭짓점)에만 있고 닫을 때 `S.masks` 로 간다(되돌리기 한 칸·자동 저장). 닫기 = 첫 점 누르기·
      마지막 점 한 번 더(4px 안)·Enter, Esc 버리기, Backspace·Del·Ctrl+Z·우클릭 마지막 점 지우기(`polyKey` 가 keydown 의 Ctrl+Z 보다
      먼저 받는다). **그리는 동안 승인하지 않는다** — Space 는 `polyKey`, 승인 버튼은 `setStatus` 가 막는다(덮으려던 자리 없이 학습에
      들어가지 않게). 다른 이미지(`loadImageNow`·`reloadCurrentBoxes`)·다른 모드(`setMode`)로 가면 버린다(`polyCancel`). 고르기는
      다각형 **안쪽**만(`hitMask` — 바깥 사각형의 빈 모서리를 누르면 그 자리에 새 영역을 그릴 수 있게), 고른 다각형은 꼭짓점 네모
      (끌기·우클릭 지우기 `deleteVertex`, 3개는 남김)와 변 가운데 동그라미(끌면 꼭짓점 더하기)를 받는다(`drag.mode` `vertex`).
      옮기기는 처음 꼭짓점(`drag.p0`)에서 옮긴 만큼이다(끝자리 잡음이 쌓이지 않게). 한 줄로 늘어서거나 1px 아래로 줄어드는
      다각형은 만들지도 끌지도 않는다(`polyTooSmall` — 서버 `clean` 보다 넉넉하게: 화면에 남은 것이 저장에서 조용히 빠지지 않게).
    - **그리는 중의 W·A·S·D 화면 이동 (2026-11-20, 사용자 요청 — 크게 확대해 넓은 자리를 덮을 때 마우스는 꼭짓점에 두고 화면만
      옮기려고)**: `onKeydown`(라벨링 keydown — 이 일로 이름을 붙였다)이 `polyKey` 보다 먼저 `panKeyDown` 에 묻는다. 누르고 있는
      동안 `requestAnimationFrame`(`panFrame`)이 그쪽으로 옮긴다 — 1초에 캔버스 짧은 변 × `PAN_SPEED`(1.0), Shift 면 × `PAN_FAST`(2.5),
      대각선도 같은 빠르기, 한 프레임 0.1초까지. 이미지 가장자리가 화면 가운데를 넘어가게는 옮기지 않는다(`panBy`·`panClamp` —
      이미 넘어가 있으면 더 멀어지지만 않게). 끄는 화면 끌기(가운데 버튼·Ctrl)와 겹치면 `drag.ox/oy` 도 같이 옮기고, 커서까지의 점선은
      `S.poly.sp`(커서의 화면 자리 — mousemove 가 둔다)로 다시 센다. 키는 자리(`e.code`, 없으면 글자·자모 `PAN_KEY`)로 보고
      Ctrl·Alt·Cmd 와 함께면 받지 않는다. **그리는 동안 누르기 시작한 키는 뗄 때까지 `PAN.keys` 에 남아 그 반복 입력을 받아
      버린다** — 누른 채 다각형을 닫아도 W(도메인)·A(예측 받기)·D(화면 끌기 모드)로 새지 않게(닫은 뒤에는 움직이지 않는다).
      그리는 중이 아니면 W·A·S·D 는 원래 뜻이다. ↑·↓ 의 '입력이 올 때만 움직인다'(아래)와 달리 누르고 있음을 기억하므로,
      keyup(`panKeyUp`)과 창 blur(`panStop`) 둘 다 놓는다 — keyup·blur 처리기를 고치면 이 둘을 지킬 것. 그래도 keyup 을 놓치면
      (네이티브 대화상자 등) 이미지 가장자리에서 서고, 그 키를 한 번 눌렀다 떼면 풀린다. `tests/test_js_pan.py`(`pan_harness.js`).
      11-20 에 사본 서버로 확인: 캔버스 280px 에서 D 0.6초 → 170px(예상 171), Shift+A 434px(437), 가장자리에서 정확히 섬,
      blur 뒤 0px — 이 창이 화면에 안 보이면 requestAnimationFrame 이 초당 1번쯤으로 줄어 그때는 16ms 타이머로 갈음해 쟀다.
      **그리는 동안 Q·E(도메인)·T(태그) 키는 `polyKey` 가 막는다**(같은 날 사용자 결정 — 정하고 다음 장으로 가 다각형을 버리는데
      W·A·S·D 바로 옆이라). 글자는 keydown 의 switch 와 같은 자모·쌍자음까지, 알림은 누른 첫 입력에만. **키만 막는다** — 도메인·태그
      버튼(`setDomain`·`tagAndNext`)은 그대로(승인은 `setStatus` 가 버튼까지 막는 것과 다르다). B·C(모드 바꾸기 — 다각형을 버림)는
      11-19 설계 그대로 두었다.
    - **옛 서버와 섞일 때**: 서버를 다시 켜기 전에 페이지만 새로 고치면 새 app.js 가 옛 서버에 붙는다 — `/api/images/{id}` 의
      `mask_poly` 가 참일 때만(`S.maskPoly`) 다각형을 시작하고 다각형 템플릿을 적용한다(옛 서버는 pts 를 몰라 422, 사각형 칸만
      받으면 바깥 사각형으로 저장한다). 11-19 에 옛 코드 서버 + 새 화면으로 사각형 저장·다각형 거절을 확인했다. 거꾸로(옛 화면 + 새
      서버)는 다각형을 숫자가 아닌 값으로 받아 그 영역을 손대 저장하면 422 로 막힌다(조용히 바뀌지 않는다) — 새로 고칠 것.
    - 시험: `tests/test_masks.py`(덮는 몫·칠하기·자르기·같은 자리), `test_js_parity.py`(화면과 끝자리까지), `test_store.py`(정확한
      저장·옛 DB 의 열 더하기·템플릿·되돌리기), `test_server.py`(API·검사·test 막기), `test_health_dataset.py`(진짜 학습 세트·내보내기의
      칠한 픽셀과 라벨 수), `test_label_qa.py`. 도구: `tools/crop_images.py` 는 다각형을 남길 영역으로 잘라(`masks.clip_poly`) 옮기고,
      `tools/mask_images.py` 는 사각형만 넣되 있던 다각형을 지키고 대조 그림에 칠한다.
  - **한 장 한도는 500개**(`masks.MAX_MASKS`·`app.js` 의 `MASK_MAX` — 같은 값, 11-11 에 50 → 500, 사용자 요청: 애매한 물체를
    하나씩 촘촘히 덮으려고). 실수로 수천 개가 들어오는 것만 막는다. 11-11 실측(2560×1440, 박스 95개 + 영역 500개): 화면 한 번
    그리기 2.4ms·오른쪽 목록 다시 그리기 5ms·저장 21ms·학습 세트 한 장 0.1초.
  - **가릴 영역 템플릿 (2026-11-10, 사용자 요청)** — 이름 붙인 영역 묶음(DB `mask_templates`, 정규화 xyxy JSON + 만든 그림의 화면
    크기). 적용하면 이미지의 `masks` 로 **복사**된다(템플릿을 고치거나 지워도 적용한 이미지는 그대로). 한 장 적용은 화면이
    `S.masks` 에 더하고 평소 저장(PUT boxes, `S.masksTouched`)으로 간다 — 되돌리기·이력이 그대로 먹는다. 여러 장은 이미지 보기의
    `/api/bulk/mask-template`(add·remove) → `store.apply_mask_template`: 이미지마다 `_replace_boxes(source='bulk', batch=…, masks=…)`
    라 일괄 수정 묶음으로 되돌린다. `revert_batch` 는 `masks_after` 가 있고 박스 전후가 같은 줄(영역만 바꾼 줄)은 **박스를 견주지
    않고 지금 박스를 그대로 둔 채** 영역만 되돌린다(지금 영역이 적용 직후와 같을 때만) — 템플릿을 넣은 뒤 박스를 검수한 장도
    되돌려야 해서다. 되돌리면 영역이 새로 생기는 장이 그새 test 가 됐으면 건너뛴다.
    **test 판정은 server 의 `mask_block` 한 곳에만 둔다** — store(`apply_mask_template(blocked=)`·`revert_batch(mask_blocked=)`)에는
    `mask_blocker()` 로 넘긴다(store 는 cameras.yaml 을 모른다). 넘기지 않으면 store 는 막지 않는다.
    같은 자리 판정은 `masks.same_rect` — 네 좌표의 차가 모두 `SAME`(7.5e-8) 이하(다각형은 꼭짓점마다 — 아래 '다각형')(2026-11-12 에 소수 7자리 반올림 키
    `mask_key` 에서 바꿨다: 픽셀에 딱 맞춘 2560·1280 폭 좌표가 반올림 경계에 걸려 화면을 한 번 오가면 다른 자리가 됐다 — 이유·
    수치는 masks.py 의 `SAME`). 더하기는 같은 자리를 건너뛰고(`masks.merge`), 빼기는 같은 자리만 뺀다(`masks.subtract`), 영역이
    '바뀌었나' 는 `same_rects`. 화면의 `sameRect`·`MASK_SAME` 이 같은 식이다 — 고치면 둘 다. 더하기는 test·화면 비율이 1% 넘게 다른 그림·한도(500개) 넘김을
    건너뛴다. **비율은 장마다 파일을 열어 EXIF 회전을 적용한 화면 크기로 본다**(`_display_ratio`, 11-16 — 전에는 DB 의 원시 비율이
    맞으면 파일을 안 열어서, 16:9 로 찍어 90도 돌려 보는 사진에 영역이 엉뚱한 자리로 들어갔다. 파일을 못 열면 DB 크기로 보고
    그것도 어긋나면 unreadable. 사본 서버 실측: 141장 더하기 0.86초). 화면은 적용 직전에 목록을 다시 받는다(`freshMaskTemplate`).
  - `_replace_boxes` 는 **박스가 그대로면(같은 `_box_key` 목록) boxes 행을 지우고 다시 넣지 않는다**(2026-11-10) — 영역만 바꾸는
    저장·템플릿 적용에서 박스 행 id 가 그대로 남는다. 이력은 전과 같이 남긴다.
- DB 자동 백업(`store.backup_daily`)은 서버를 켤 때와, 켜 둔 채 날이 바뀐 뒤 **첫 변경 요청**(GET·HEAD·OPTIONS 밖) 앞에서
  `<DB 폴더>/backups/` 에 뜬다(`server._DailyBackup`·`_BackupOnWrite`, 2026-11-21 — 전에는 켤 때만이라 자정을 넘겨 켜 둔 서버는
  다시 켤 때까지 사본이 없었다). 그날 것이 있으면 건너뛰어 하루 최대 한 번, 최근 `backup.keep`(14)개 — 그 이름 꼴의 파일만 지운다.
  **시험 서버는 `backup.keep: 0` 으로 끌 것.**
- 같은 자리 일괄 수정(`store.same_place` → `set_class_bulk`)은 적용 때 다시 찾아 미리 본 수와 다르면 409, 묶음째 되돌리기는 그
  뒤 손대지 않은 이미지만 되돌린다. 묶음(grp)이 정해지기 전의 새 이미지는 쓸 수 없다. '변경 기록' 의 묶음 목록
  (`store.batches(limit)`)은 **일괄 수정(source bulk)을 `limit` 건** 싣고 그보다 새로운 다른 줄(되돌리기 revert)은 전부 싣는다
  (11-16 — 전에는 둘을 함께 30줄로 잘라, 되돌린 묶음이 많을수록 일괄 수정이 덜 보였다). 화면은 revert 줄로 `되돌림` 을 표시한다.
- **예측 비교(P)** 는 운영 가중치로 한 장 추론만 하고 DB 에 쓰지 않는다(비교는 브라우저 `computeFindings`, 규칙은
  `tools/label_qa.py` 의 `analyze_image` 와 같고 예측 하한만 다르다 — label_qa 0.10, P 는 `predict.conf` 0.25. **고치면 둘 다**).
  모델이 모르는 클래스(지금 운영 8클래스 모델의 responder·cone)의 라벨은 비교하지 않는다 — 어느 클래스 예측과도 짝이 되고
  `class`·`unseen` 으로 내지 않는다. label_qa 는 11-16 까지 이 규칙이 빠져 있었다(모델이 아는 클래스를 `meta.json` 의
  `model_classes` 로 남기고 `--reuse` 가 읽는다). 가릴 영역이 반 넘게 덮은 자리도 둘 다 견주지 않는다. 올릴 때 빈 이미지로 예열하는데, **자동 라벨링(`AutoLabeler`)은 예열하지 않는다** — 첫 추론이 CUDA
  초기화로 길어지면 NMS 시간 제한에 걸려 검출이 'NMS time limit exceeded' 경고만 남기고 잘릴 수 있다(한 번 봤고 조건은 모른다).
- **타일 추론(2026-11-10, `tiling.py`)** 은 자동 라벨링의 선택지다(`AutoLabeler.run(tiles=True)` ← `PredictIn.tiles` ← 메뉴
  '자동 라벨링 (타일)', CLI `label --tiles`, 설정 `predict.tiles`). **`tools/test_tiles.py` 와 같은 함수를 쓴다** — 고치면 측정도
  바뀐다. 운영 조합은 `tiling.detect`(측정 변형 `g2+full:big` + `--cross-iou`): 성분마다 RAW_*(0.001·0.7·300)로 원검출을 받아
  잘린 조각 박스를 버리고 `merge_cross` 로 합친 **뒤에** predict.conf·max_det 로 거른다 — 그래서 조각을 안 쓰는 작은 이미지도
  보통 모드와 아주 조금 다를 수 있다. 그림은 `ultralytics.utils.patches.imread`(Ultralytics 가 경로로 읽을 때와 같은 함수)로 읽고,
  타일 모드만 빈 이미지로 예열한다. 타일로 라벨링한 이미지는 `model_version` 끝에 `+tiles<grid>`. 예측 비교(P)는 통째 그대로다.
  **차량 클래스끼리는 클래스가 달라도 `class_iou`(0.6) 이상 겹치면 신뢰도 높은 쪽만 남긴다 (11-17 사용자 요청,
  `merge_cross(group=, iou_class=)`·`tiling.VEHICLES`)** — 클래스별 NMS 라 통째 car·타일 truck 인 같은 차가 둘 다 남았다(실측은
  docs/experiments.md '타일 추론'). 사람·자전거·오토바이·킥보드·라바콘은 묶지 않는다(탄 사람과 탈것은 실제로 겹친다) — 차량
  클래스를 더하면 `VEHICLES` 에 넣을 것. `class_iou: 0` 이면 끈다. `tests/test_tiling.py`.
  **보통(통째) 자동 라벨링도 11-20 부터 같은 차량 규칙으로 정리한다**(사용자 결정 — 새 풀 프레임 평가 175장에서 통째도 같은 차에
  car·responder 처럼 클래스만 다른 박스를 겹쳐 냈다: 운영 가중치 중복 42 → 10, 찾음·클래스까지 그대로, docs/experiments.md
  '타일 모드로 중복이 얼마나 주는지'). 설정은 `predict.class_iou`(0.6, 없으면 `tiling.PLAIN_CLASS_IOU`, 0 이면 끔 —
  `tiling.plain_class_iou`), 함수는 `tiling.cross_class_keep`: **같은 클래스 NMS 는 하지 않고**(모델이 낸 그대로) 클래스를 넘는
  차량 겹침만 merge_cross 와 같은 순서로 지운다(무작위 400벌로 견주는 테스트가 있다). `AutoLabeler.run` 의 통째 경로가 xyxy(픽셀)로
  표를 받아 xywhn 행을 거르고, 뺀 수는 `PredictReport.merged`(요약 줄 '겹친 차량 박스 N개'). 타일 모드는 그대로 detect 안에서
  `predict.tiles.class_iou` 로 하고 그 수는 세지 않는다. **예측 비교(P)·label_qa 는 걸지 않는다**(모델이 낸 그대로 견준다 —
  P 의 `LiveDetector` 머리말). `model_version` 에는 표시를 붙이지 않았다(11-20 이후 통째로 라벨링한 이미지는 정리된 것).
  `tests/test_predictor.py`(가짜 모델로 저장까지). 그날 DB 사본(승인 175장, 운영 가중치)으로 끝까지 돌려 보니 끈 쪽 1,743개 중
  52개만 빠졌고 그 수가 `merged` 와 같았으며, 클래스가 다른 차량 박스 쌍(IoU ≥ 0.6)이 55 → 0(`runs/compare_1118_pretrain/plain_merge_check.py`).
- **영상 캡처 탭 (2026-11-10, 사용자 요청 — `capture.py`·`web/capture.js`, 사용법은 README '영상 캡처')** — 라벨링·이미지 보기와
  같은 페이지의 세 번째 보기(`setView('capture')`·`body.view-capture`). 영상을 올려(`POST /api/capture/upload` — multipart 가
  아니라 **본문 그대로** `request.stream()` 을 8MB 씩 쓴다, `size` 와 다르면·끊기면(ClientDisconnect) 지운다) 서버가 OpenCV 로
  뽑고, 고른 것만 `unlabeled/<카메라>/<날짜>/` 로 옮겨 스캔한다.
  - 세션 = `<DB 폴더>/capture/<YYYYMMDD_HHMMSS_hex4>/`(`video.<확장자>` 사본·`frames/`·`thumbs/`·`session.json`). 넣거나 닫으면
    폴더째 지운다. 서버를 켤 때 `session.json` 이 없거나 '올리는 중' 이던 세션 폴더는 지우고(그 이름 꼴만), 뽑던·넣던 것은 '멈춤'.
    세션 id 는 `SID_RE` 로만 받는다(경로 탈출 막기) — 파일은 세션·번호로만 서빙한다(`_capture_file`).
  - **작업 큐(`jobs`)를 쓰지 않는다** — 학습 중에도 캡처하려고. 뽑기는 프로젝트에 한 번에 하나(`_extract_lock`, 나머지는 queued),
    넣기는 하나씩(`_commit_lock`). **넣기의 `ingest` 는 `server.ingest_lock` 을 쥔다**(`job_scan` 도 같은 잠금) — 스캔을 부르는
    새 길을 만들면 같은 잠금을 쥘 것. 정리·휴지통 비우기가 돌면 넣지 않는다(`_capture_blocker`, 409). 넣는 동안 세션은
    `committing` 이라 다시 뽑기·닫기가 프레임을 지우지 못한다. **`committing` 은 `session.json` 에도 남긴다(11-16)** — 옮기는
    도중 서버가 죽으면 다시 켤 때 `_load` 가 그 상태를 보고 '넣는 도중 서버가 꺼졌습니다' 를 붙인다(전에는 메모리에서만 바꿔 그
    안내가 한 번도 안 나왔다). 같은 영상에서 다시 넣으면 남은 프레임만 옮기고(옮긴 것은 같은 이름이라 `exists`) 자동 카메라도
    지문으로 다시 찾는다 — 사본에서 넣기를 3장 옮긴 뒤 `os._exit` 로 죽여 보고 화면에서 확인했다(다시 넣기: 옮김 5·있음 3·
    스캔 8장). `tests/test_capture_commit.py`. 화면 쪽 `capReset` 은 타이머뿐 아니라 들어 둔 시작 시각(`C.startPending`)도
    버린다 — 안 버리면 넣거나 닫은 뒤 다른 보기로 갈 때 없어진 세션에 PUT 을 보냈다(`tests/test_js_capture.py`).
  - 뽑는 방식·이름은 `tools/extract_frames.py` 와 **바이트까지 같다**(11-10 에 10장 대조) — `frame_name`·`parse_start`·`small_gray`·
    `changed_pct` 를 capture.py 한 곳에 두고 extract_frames·import_roadplus 가 가져간다. 이름의 시각을 scan 이 `captured_at` 으로
    읽으니(provenance `_COMPACT`) 꼴을 바꾸면 둘 다 본다. 넣을 때 같은 이름이 있으면 덮지 않는다(`_rename` — Windows rename +
    다른 드라이브면 `xb` 복사). 시작 시각은 이름(`_from_name`) → 사람이 적은 값. 수정 시각 − 길이(`start_mtime`)는 제안만 한다.
  - **시작 시각을 비우면(11-10 사용자 요청) 시각 없는 이름** `notime_name` = `<카메라>_v<지문 8자>_t00h01m05s500.jpg`(폴더 `v<지문>/`,
    영상 안 시각) — `captured_at` 은 NULL. **숫자 묶음 사이마다 글자를 둔 것은 scan 이 시각으로 읽지 않게 하려는 것**이다
    (provenance `_COMPACT`·`_DASHED` — 지문이 '20251231' 같은 숫자뿐이어도, 카메라 키에 날짜가 있어도 안 걸린다. 꼴을 바꾸면
    `_from_name` 이 None 인지 다시 볼 것). 지문은 올릴 때 센 영상 내용 sha1(`video.sha1`, 화면에는 `video.tag`) — 같은 영상이면 같은
    이름이라 다시 넣어도 건너뛴다. 이 기능 전에 올린 세션은 넣을 때 센다(`video_tag`). 화면의 미리 보기 `capNoTimeName` 이 같은 식
    이다(반올림은 반을 올림 — 고치면 둘 다).
  - **카메라를 고르지 않으면 영상마다 자동 카메라(11-11 사용자 요청, `_auto_camera`).** 넣기의 `camera` 가 비면(없음·`""`) 이
    영상(지문)을 전에 자동 등록한 카메라 — **note 에 `영상 지문 <8자>`** 가 있는 키(`find_auto_camera`, 키를 바꿔도 note 가 따라간다)
    — 에 넣고, 없으면 다음 번호 `vid-NN`(`next_auto_key` — 등록·`camera_fields`·DB 가 쓰는 키(`used_cameras`)·카메라 폴더 이름을
    건너뛴다)을 `add_camera` 로 등록한다: kind cctv, place `모름 — 영상 <이름>`, 해상도, note `auto_note`. 결과의 `auto` 는
    new·reused·dropped(옮긴 프레임이 없으면 만든 등록을 `remove_cameras` 로 되돌린다). **카메라를 아예 비우지 않는 것은 분할이
    카메라째라서다** — 출처 없이 넣으면 한 영상의 프레임이 train·test 로 갈릴 수 있다. 화면 `capAutoCamera` 가 같은 식으로 미리
    보인다(목록 `fields.note` 로 — DB·폴더 번호는 몰라 드물게 다를 수 있고, 결과가 실제 키를 알린다).
  - 프레임 항목 `{i, t, c, s, k}` — `c` 는 앞 장과 바뀐 픽셀 %, `s` 는 **JPEG 바이트까지 같은** 앞 프레임(멈춘 화면 — 처음부터
    `k=false`). 화면이 남기는 빼기 목록은 `PUT …/state {drop, upto}` — `upto` 밖(화면이 아직 못 받은 뒤쪽)은 건드리지 않는다.
  - **`tools/auto_capture.py`(11-12)가 이 API 를 그대로 부른다** — `POST capture/upload?name&size&mtime`(본문 그대로)·
    `POST capture/<세션>/extract {every}`·`GET capture/<세션>?since=`·`PUT capture/<세션>/state {drop, upto, camera, start}`·
    `POST capture/<세션>/commit {camera, start, keep}`·`DELETE capture/<세션>`, 그리고 `GET capture`(limits·cameras)·`GET state`
    (classes·class_boxes·job)·`POST jobs/predict`. 꼴을 바꾸면 그 도구도 고칠 것. 세션 프레임·썸네일은 서버와 같은 PC 면
    `<DB 폴더>/capture/<세션>/` 에서 바로 읽는다(없으면 `frame/<i>` 로 받는다).
  - 화면은 뽑는 동안 0.7초마다 `?since=` 로 새 프레임만 받고, `run`(다시 뽑을 때마다 +1)이 바뀌면 처음부터 받는다. 썸네일·원본
    주소에 `?r=<run>` 을 붙여 서버가 1시간 캐시를 허락한다. 폴링 응답의 카메라·시작 시각은 받지 않는다(고치는 중인 값을 덮지 않게).
  - **새 카메라 등록(`＋ 새로` → `POST /api/cameras` → `provenance.add_camera`, 11-10 사용자 요청).** `cameras.yaml` 은 사람이
    손으로 쓰는 파일이라(묶음 설명 주석·`crop`·순서) **다시 쓰지 않고 `cameras:` 묶음 끝에 글자로 덧붙인다**(들여쓰기·줄끝·BOM 은
    파일을 따른다 — `read_text` 는 CRLF 를 LF 로 바꿔 읽으니 바이트로 읽는다). 쓰기 전에 덧붙인 글을 다시 읽어 '기존 카메라는
    속성까지 그대로·새 키 하나만 늘었다' 가 아니면 아무것도 쓰지 않고, 쓰기 전 파일을 `<DB 폴더>/backups/cameras_<시각>.yaml` 에
    남긴 뒤 한 번에 바꿔 끼운다(DB 하루 사본 정리는 `labeling_*.db` 꼴만 지워 이 파일은 남는다). 이미 깨진 등록 파일·`cameras: {}`
    한 줄 꼴이면 손대지 않는다(400). 값 규칙은 `load_cameras` 와 같고, 기본 시간대(+09:00)는 적지 않으며 `split` 은 비어도 적는다.
    시간대는 `normalize_tz` 가 `+09:00` 꼴로 바꿔 적는다(11-10 — KST·JST·UTC·GMT·Z·`UTC+9`·`+9`·Asia/Seoul. `parse_tz` 도 같은
    표를 써서 손으로 쓴 `tz: KST` 도 읽힌다. CST·IST·BST·AST 는 뜻이 여럿, EST·PST 는 여름에 바뀌어 이유와 함께 거절한다 — 약어를
    더할 때는 뜻이 하나뿐인지 볼 것. 화면 목록의 `tz` 는 `tz_text`).
    장소는 서버 로그에 찍지 않는다.
  - **카메라 고치기(`고치기…` → `PUT /api/cameras/{key}` → `provenance.edit_camera`, 11-10 사용자 요청).** 보낸 칸만 보고 지금 값과
    **뜻이 같은 칸은 건드리지 않는다**(`_same_value` — mount 없음 = unknown, tz 는 `normalize_tz` 로 견줌, 빈 값 = 없음). 바뀐 칸의 줄만
    고친다: 한 줄 값은 그 줄을 바꾸고(줄 끝 주석은 `_line_comment` — 떼어도 같게 읽히는 첫 ' #'), 없던 칸은 `_FIELD_ORDER` 자리에 넣고,
    비운 칸은 줄을 뺀다(split 은 `split:`). 키는 키 줄의 이름만 바꾼다. 다시 읽어 그 카메라만 기대대로(`merged`) 바뀌었을 때만 쓴다 —
    flow 한 줄 항목처럼 못 고치는 모양은 400(손으로). `dry_run` 은 확인까지만 한다. **server 가 막는 것(409)**: 분할이 정해진
    이미지(`store.camera_image_stats` 의 assigned)가 있는 카메라의 고정 분할(split.py 는 정한 분할을 안 옮겨 두 분할로 갈린다), 가릴
    영역이 있는 이미지(masked)가 있는데 test 고정. **시간대를 바꾸면** 그 카메라 이미지의 `captured_at` 을 `capture_time` 으로 다시
    센다(`store.set_captured_at` — updated_at 은 그대로, 다시 셀 수 없으면 그대로). 지우기와 같은 두 잠금을 같은 순서로 쥔다.
    키를 바꾸면 캡처 세션도 새 키로(`rename_camera`). 화면 목록의 `fields`(`camera_fields`)가 고치기 칸을 채운다.
  - **여러 카메라의 고정 분할만 한 번에 (11-23, `provenance.set_splits` — AI Hub 카메라 백 대를 train 으로 고정하려고).**
    `edit_camera` 를 카메라마다 부르면 쓰기 전 사본이 카메라 수만큼 생기고 도중에 멈추면 반만 고친 파일이 남는다 — 이 함수는
    같은 줄 고치기(`_edit_camera_text`)로 전부 고친 뒤 다시 읽어 '그 카메라들의 split 만 바뀌고 나머지는 속성·순서까지 그대로'
    일 때만 사본 하나를 남기고 한 번에 바꿔 끼운다(하나씩 고친 것과 바이트까지 같다 — `tests/test_provenance.py`). 없는 키·못
    고치는 한 줄 꼴 항목이 하나라도 있으면 아무것도 안 쓴다. **서버 API 는 없다** — 부르는 것은 `tools/import_aihub_pm.py
    --pin-train` 뿐이고, 서버의 잠금(`commit_lock`·`ingest_lock`)을 쥐지 않으니 화면에서 카메라를 더하는 바로 그 순간과 겹치면
    한쪽이 덮일 수 있다(쓰기 전 사본은 남는다). **이미지가 이미 다른 분할에 든 카메라를 막는 것은 부르는 쪽 몫이다**(그 도구의
    `pin_plan` — 서버의 409 와 같은 일). 서버에서 부르게 되면 그 두 잠금과 `use["assigned"]` 검사를 함께 넣을 것.
  - **이미지가 있는 카메라의 키 바꾸기(11-11 사용자 요청, `camera_rename.py` — 전에는 409).** 키가 든 세 곳을 함께 바꾼다:
    `cameras.yaml` 키 줄, 카메라 폴더 `unlabeled`·`labeled/images`·`labeled/labels` 의 `<키>/` 와 파일 이름 앞머리 `<키>_`(앞머리가
    다른 이름은 옮기기만), DB `images.camera`·`path`·`name` 과 `mask_templates.camera`. **순서가 핵심이다 — DB 행이 없는 파일을
    가리키는 때가 없어야 한다**(그러면 정리 뒤의 prune 이 그 행을 라벨째 지운다): 계획(`plan_rename` — 새 폴더에 파일·DB 에 새 키
    행·새 경로 겹침·DB 의 다른 행이 쓰는 경로면 409) → DB 사본 `backups/labeling_camrename_<시각>.db`(하루 사본 정리 대상 아님) → 새
    경로에 **하드링크**(안 되는 디스크면 복사 — 수정 시각·크기가 같아 썸네일 캐시가 그대로 맞는다) → DB 한 트랜잭션
    (`store.rename_camera_rows` — `BEGIN IMMEDIATE`, 지금 경로가 그사이 바뀐 행이 있으면 안 바꿈, updated_at 그대로) →
    `edit_camera` → 옛 이름 지우기. 링크·DB·yaml 중 하나가 실패하면 앞의 것을 되돌린다(`undo_camera_rows`·만든 링크 지우기 — 11-11
    시험에서 셋 다 '파일·DB·등록이 처음과 같다'). 옛 이름을 못 지우면(다른 프로그램이 열어 둠) 알린다 — 등록이 빠진 옛 폴더라 scan 이
    멈추니 손으로 지울 것. **작업 큐에서 돈다**(`camera_rename` — 파일 경로를 읽는 학습·자동 라벨링·정리와 겹치지 않게, 도는 작업이
    있으면 409). 요청은 끝날 때까지 기다리고(10분 넘으면 504), 값 확인(`dry_run`)·계획은 작업을 만들기 전에 한 번, 작업 안에서
    잠금을 쥐고 한 번 더 한다. 키만 다시 바꾸면 경로 글자까지 처음과 같다(11-11 시험). 화면은 바꾼 뒤 라벨링 카메라 필터를 새 키로 옮긴다.
  - **고른 이미지만 다른 카메라로 (11-15 사용자 요청, `camera_move.py` — 이미지 보기 오른쪽 `카메라` 칸 → `POST /api/bulk/camera`).**
    키 바꾸기가 카메라를 통째로 바꾼다면 이것은 몇 장만 **이미 등록된** 다른 카메라로 옮긴다(화면을 돌린 동안의 프레임을 다른
    키로 나눠 두는 때). 한 장마다 파일(`unlabeled/<키>/…`, 정리된 것은 `labeled/images/<키>/` 와 짝 라벨 `.txt`)의 폴더·이름
    앞머리, DB `camera`·`path`·`name`, 촬영 시각(새 카메라 시간대로 `capture_time` — 못 세면 그대로)을 바꾸고 **`grp` 를 비운다**
    — 묶음은 카메라째라 옛 묶음 키를 달고 가면 다음 분할이 옛 카메라 쪽으로 보낸다(비워 두면 `ensure_splits` 가 묶음을 전부 다시
    센다. `tests/test_camera_move.py` 가 이 줄을 빼면 실패한다). 순서는 키 바꾸기와 같다(하드링크 → `store.move_camera_rows` 한
    트랜잭션, 그사이 경로·카메라가 바뀐 행이 있으면 안 바꿈 → 옛 이름 지우기). DB 를 통째로 뜨지 않고 `backups/camera_moves.jsonl`
    에 옮긴 것을 남긴다. **막는 것(`MoveBlocked` → 409)**: 분할이 정해진 이미지를 옮겨 새 카메라가 두 분할에 걸치는 때(고정 분할·
    이미 있는 이미지의 분할과 다름 — 미배정은 막지 않는다), 가릴 영역이 있는 이미지를 test 고정 카메라로(`mask_block` 과 같은
    뜻 — 규칙을 바꾸면 둘 다), 새 경로가 디스크·DB 에 이미 있음, 출처 없는 이미지. `dry_run` 은 세기만 한다(화면의 확인 문구).
    키 바꾸기처럼 **작업 큐에서 돈다**(`camera_move`, 도는 작업이 있으면 409)·같은 두 잠금. 화면은 옮긴 줄의 이름·카메라만 고치고
    (`applyToRows`), 라벨링의 지금 이미지는 다시 열지 않고 정보 줄만 맞춘다(`refreshCurMeta` — 되돌리기·확대를 지킨다). 고를
    목록은 `/api/cameras` 의 `registered`(등록 전부 — 필터용 `cameras` 는 이미지가 있는 카메라만이다).
  - **카메라 지우기(`지우기…` → `POST /api/cameras/remove` → `provenance.remove_cameras`, 11-10 사용자 요청).** 같은 방식으로
    **그 카메라의 줄만** 뺀다 — 키 줄, 그 아래 더 깊이 들여쓴 줄(속성·안쪽 주석), 키 바로 위에 **같은 들여쓰기로** 붙은 주석(등록
    주석). 맨 왼쪽의 묶음 설명 주석과 끝의 얕은 주석·빈 줄은 다음 키 것으로 보고 남긴다. 다시 읽어 '그 키들만 없어지고 나머지는
    속성까지 그대로' 일 때만 쓰고(백업은 add 와 같은 곳), 없는 키가 하나라도 있으면 하나도 안 뺀다. 더했다 빼면 바이트까지 처음과
    같다(11-10 시험). **이미지가 남은 카메라는 server 가 막는다(409)** — `store.camera_counts(with_trashed=True)`(휴지통도 그
    카메라를 가리킨다)와 `unlabeled/<키>`·`labeled/images/<키>`·`labeled/labels/<키>` 의 파일. 검사와 빼기 사이에 넣기·스캔이 끼지
    않게 `capman.commit_lock` → `ingest_lock` 순서로 쥔다(넣기도 같은 순서 — 바꾸면 서로 막힌다). 지운 카메라를 고른 캡처 세션은
    카메라를 비운다(`forget_cameras`). 빈 카메라 폴더는 지우지 않는다. 화면 목록의 `db`·`files` 는 `_camera_rows` 가 붙인다. 화면은 계열(접두어+번호)마다 다음 번호를 제안하고(공개 데이터셋만 있는
    계열은 뺌), 처음 보는 접두어·`test` 고정은 한 번 더 묻는다(키에 지명이 드러나면 문서·커밋에 쓸 수 없다 — 루트 CLAUDE.md git 절).
  - **카메라를 고르는 칸은 `키 · 장소 (꼬리)` 로 적는다 (11-17 사용자 요청, app.js `camLabel`)** — 키가 번호뿐이라 키만
    보여서는 영상에 어느 카메라를 줄지 알 수 없었다(장소는 `title` 에만 있었다). 영상의 카메라(`#cap-camera`)·`고칠 것`
    (`#cc-target`)·지우기 목록과 라벨링의 카메라 필터·이미지 보기의 '옮기기'(꼬리 = 이미지 수)가 **한 함수를 쓴다**(같은 날
    리뷰 — 전에는 세 곳이 따로 이었고 줄임은 캡처 탭에만 있었다). 장소가 `CAM_PLACE_MAX`(40자)를 넘으면 **가운데**를 줄인다
    (펼친 목록은 가장 긴 줄만큼 넓어진다. 자동 카메라의 `모름 — 영상 <이름>` 은 구별되는 곳이 이름 끝이라 앞만 남기면 같은
    앞머리의 영상들이 똑같이 보였다). 전체는 option 의 `title` 과, 닫힌 칸에서는 뜨지 않으니 `#cap-camera` 자체의 `title`
    (`capCamTitle` — `capRenderCommit` 이 고를 때마다 맞춘다). **닫힌 칸은 좁아 꼬리가 잘린다** — 고른 카메라의 고정 분할·
    한국이 아닌 시간대(`capCamFlags`)는 시작 시각 안내에도 한 줄로 적는다(`capCamNote`). 왼쪽 칸이 300px 라 지우기 목록은
    `키 장소 종류` 한 줄에 '지울 수 없는 이유' 가 안 들어가면 아랫줄로 내린다(style.css `.cap-cam-del-list li` 의
    `flex-wrap` — 한 줄에 눌러 담으면 장소가 한두 글자만 보인다). 카메라를 고르는 칸을 새로 만들면 `camLabel` 을 쓸 것.
    **장소는 화면에만 쓴다** — 서버 로그·git 에 넣지 않는다(테스트의 장소는 지어낸 것). `tests/test_js_capture.py`.
  - **영상 캡처에서 키는 `captureKey` 로만 간다 — V 도**(app.js `VIEWS` 표의 `capture.key`, `vToggle` 없음 → `viewKeydown`) —
    숨은 라벨링에 Space(승인)·S·Q 가 먹으면 안 된다(11-10 에 확인). 크게 보기는 공용 대화상자가 아니라 `#cap-preview`(키도 captureKey).
  - **뽑은 프레임은 끌지 않는다 (11-12 사용자 요청).** 크롬은 페이지 안의 `<img>` 를 끌 때도 dataTransfer 에 'Files' 와 그 JPEG
    을 실어, `#capture` 의 끌어다 놓기가 영상인 줄 알고 `capUpload` 로 넘겼다(실측: 칸을 다른 칸에 놓으면 이름 '0'·image/jpeg 파일).
    칸·크게 보기 그림은 `draggable=false`, `#cap-grid`·`#cap-preview` 의 dragstart 는 막는다. 끌기가 없으면 끌었다 놓은 것이
    클릭이 되므로 `capWasDrag`(누른 곳에서 6px 넘게)면 칸 클릭(넣기/빼기)과 크게 보기 바깥 클릭(닫기)을 버린다 — 칸 클릭에
    새 처리를 더하면 이 검사 뒤에 둘 것. 밖에서 끌어온 영상 파일은 그대로 받는다.
- **데이터 점검 보기 (2026-11-15 사용자 요청 — `health.py`·`web/health.js`, 사용법은 README '데이터 점검')** — 같은 페이지의 네
  번째 보기(`setView('health')`·`body.view-health`). **읽기만 한다**: `GET /api/health?scope=approved|all&masked=drop|keep&imgsz=`
  와 `GET /api/health/heatmap?…&cls=&camera=&cell=x,y` 둘뿐이고, `store.health_snapshot`(읽기 트랜잭션 하나)이 준 줄을
  `health.report`·`heatmap`·`cell_images` 가 센다(파일은 열지 않는다 — 141장 0.02초, 캐시 없음, 열 때마다 새로).
  - **덮인 몫은 한 요청에 한 번만, 필요한 박스만 잰다 (11-15 리뷰).** 화면을 열 때는 `/api/health?heat=1&cls=&camera=` 한 번 —
    같은 스냅샷·같은 판정(`masked_out` → `report`·`heatmap` 의 `gone=`)으로 히트맵을 `heatmap` 에 함께 싣는다(화면 `hcLoad` →
    `hcShowHeat`. 함께 온 것이 지금 조건과 안 맞을 때만 따로 받는다). 히트맵만 다시 받을 때는 태그 수를 세지 않고
    (`health_snapshot(tags=False)`) 거른 뒤 남은 박스만, 칸을 누를 때(`cell=`)는 **격자 없이 그 칸의 박스만** 잰다. 히트맵
    요청을 새로 만들면 수치 요청과 따로 재게 하지 말 것.
  - **가릴 영역이 반 이상 덮은 박스는 기본으로 어느 수치에도 넣지 않는다**(`health.masked_out` — 학습 세트의
    `masks.split_boxes` 와 같은 판정). 그날 DB 는 승인 박스 1,277개 가운데 619개가 그런 박스였다(영역으로 덮은 자리에 자동 라벨
    박스가 남는다) — 640 입력에서 짧은 변 32px 미만이 넣고 세면 79%, 빼면 62% 다. **이 DB 의 박스를 SQL 로 바로 세어 말할
    때도 가려진 박스를 뺐는지 밝힐 것**(11-15 에 내가 79% 를 그냥 말했다). **test 분할 이미지의 박스는 빼지 않는다** —
    학습 세트·내보내기가 test 는 칠하지도 빼지도 않는다(`trainer.build_dataset`·`export_yolo` 의 `split == "test"`). 영역을 먼저
    덮고 나중에 test 로 배정된 이미지가 있으면 `masks.test_images` 와 '눈여겨볼 것' 에 나온다(11-15 리뷰에서 고침 — 그 전에는
    분할을 안 보고 빼서 test 박스가 적게 세어졌다). **판정은 `masks.py` 한 곳에 있다** — 박스가 빠지는가는 `masks.drops`,
    test 예외는 `masks.applies_to` 이고 trainer·export·health 가 함께 쓴다(11-15 — 전에는 셋이 따로 적었다. 다시 따로 적지 말
    것). `tests/test_health_dataset.py` 가 진짜 학습 세트·내보내기를 임시 폴더에 만들어 라벨 줄 수를 `health.report` 와 견준다.
  - 박스 크기는 긴 변을 imgsz 에 맞췄을 때의 짧은 변(`short_side`)이다. DB 의 원시 크기로 세서 EXIF 로 돌려 보는 그림은
    어긋난다(파일을 열어야 알 수 있어 그대로 뒀다). 클래스의 `units` 는 분할이 통째로 옮기는 단위다(`health.unit_ids` — 같은
    카메라, **또는 `grp` 가 같은 이미지**: dHash 로 이어진 두 카메라는 한 묶음이다. 둘 다 없으면 그 한 장. grp 는 분할을 한 번
    돌린 뒤에야 생긴다). '한 곳 몫'(`top`)도 이 묶음으로 센다 — 출처 없는 이미지 전부를 한 곳으로 치지 않는다. 묶음이 **쓰는
    분할 수**(`health.used_splits` — 비율 0 인 val·test 는 뺀다)보다 적으면 알리고, 그 수는 응답의 `min_units`·`splits.used` 로
    화면에 준다. **화면에 서버의 값을 따로 적지 말 것** — 박스 크기 칸의 이름과 '32px 미만' 도 `size_bins.edges` 로 만든다
    (`hcSizeNames`·`hcUnder`, 11-15 리뷰). '눈여겨볼 것'(`_notes`)에는 숫자로 확인되는 것만 넣는다 — 많다·적다 같은 판단은 넣지 말 것.
  - 눌러 넘기기: 클래스·카메라·태그·분할은 왼쪽 필터를 그 조건 하나로 바꾼다(`hcOpenFiltered` — app.js 의 `noFilters`·
    `setFilters` 로 맞추고 목록을 받은 뒤 `setView('gallery')`). 상태 칸은 서버가 준
    `list_status` 로 맞춘다 — 범위 all(휴지통만 뺀 전부)은 목록의 `all`(휴지통도 넣는다)이 아니라 `live`(`store.LIVE`, 상태
    필터의 '휴지통 빼고 전체')다. `_list_where` 의 그 조건과 `health_snapshot` 의 WHERE 는 같은 이미지를 줘야 한다
    (`tests/test_health.py` 가 범위마다 견준다 — 11-15 리뷰에서 고침: `all` 로 넘겨 휴지통 이미지가 섞였다). 객체 수 막대·히트맵 칸은 id
    목록으로 연다(`openIds`) — 목록 API 가 GET `?ids=` 라 서버가 `health.ID_CAP`(1,000)개까지만 싣는다.
  - **넘기는 길은 기다리는 동안 바뀐 것을 본다 (11-15 리뷰에서 고침, `tests/test_js_health.py`).** 히트맵 칸(`hcOpenCell`)은
    **화면에 그려진 히트맵의 조건**(`H.heatQ` — `hcLoadHeat` 가 물을 때의 `hcHeatParams` 를 남긴다)으로 묻고, 받는 사이 다른
    보기로 갔거나(`healthHide`) 히트맵을 다시 받기 시작했으면(`H.heatSeq`) 버린다 — 전에는 지금 조절 칸 값으로 물었고, 늦게 온
    응답이 라벨링으로 간 사람을 이미지 보기로 끌고 갔다. 받는 동안에도 `H.opening` 이라 두 번 누르기는 한 번만 묻는다.
    `hcOpenFiltered` 는 필터를 바꾸기 전 값(`filterState`)을 들고 있다가 **목록을 못 받으면 되돌린다**(`setFilters` +
    `syncListControls` — `openIds` 와 같은 약속: 필터 칸만 새 조건이고 목록은 옛 것인 채로 두지 않는다. 목록을 받은 뒤의
    실패는 되돌리지 않는다). 세 길 모두 끝에 `VIEW === "health"` 일 때만 이미지 보기로 바꾼다.
  - 키: 데이터 점검은 키를 쓰지 않는다 — app.js `VIEWS` 표의 `health` 에 `key` 가 없어 `viewKeydown` 이 버린다(V 도. 아래
    '보기는 `VIEWS` 표 한 곳'). 응답은 요청 번호(`H.seq`·`H.heatSeq`)로 마지막 것만 쓴다. 조절 칸은 `hcSaveControls` 가
    남기는데 입력 크기 칸은 첫 응답 뒤에야 채워지므로 그 전에는 지난 값(`H.savedSize`)을 그대로 남긴다. 히트맵은 표시할 칸이
    바뀔 때만 다시 그린다(빈 칸 위에서 마우스를 움직여도 그대로). 수치의 뜻을 바꾸면 health.py 머리말·README·화면
    문구(index.html 의 `title`)를 함께 고칠 것.
- **학습 요약의 '분할마다 박스가 없는 클래스' (11-16, `trainer.BuildReport.class_gaps` — Gemini 의 '사전 학습 게이트' 제안에서)** —
  `build_dataset` 이 라벨 파일에 실제로 쓴 박스를 클래스·분할별로 세어(`per_split`) 요약 끝에 붙인다: train 에 없는 클래스
  `[경고]`(배우지 못한다), val·test 에 없는 클래스와 박스가 아예 없는 클래스 `[알림]`. Windows·WSL 학습과 이어하기의 데이터셋
  다시 만들기가 모두 `summary()` 를 찍으니 따로 부를 것이 없다. **막지는 않는다** — 드문 클래스가 한 분할에만 있는 것은 데이터가
  적을 때 늘 있는 일이다. 최소 박스 수 같은 문턱도 두지 않았다(데이터 점검의 `_notes` 와 같이 숫자로 확인되는 것만). val 이
  비어 train 한 장을 빌린 때(`val_borrowed`)는 val 줄을 내지 않는다. 수치는 `health.report` 의 분할별 수와 같아야 한다
  (`tests/test_health_dataset.py`).
- **GPU 작업(학습·자동 라벨링)이 도는 동안 예측 비교와 SAM(C·R)은 CPU 로 돈다**(`jobs.gpu_busy`). 시작할 때 `server.free_gpu()`
  가 모든 탭의 SAM·예측 모델을 내린다(함정 9). SAM 은 외접 박스만 낸다(마스크는 저장하지 않는다, `origin='sam'`). `C` 모드에서는
  모든 좌클릭이 SAM 으로 간다(고르거나 옮기려면 `B` 모드).
- '드문 클래스 먼저'·'태그 순'(11-10, `_list_by_tag`) 정렬·클래스 필터는 목록과 **같은 WHERE**(`_list_where`·`count_images`)로 센다 — 어긋나면 목록 묶음이 틀어진다.
  카메라·도메인·분할 필터도 같은 WHERE 에 있다. **분할 필터는 보기만 한다**(`/api/splits`) — 웹 UI 에 분할을 바꾸는 길을 넣지 말 것
  (분할은 `split.py` 만 쓴다, 위 '묶음 단위 계층화').
- **목록은 '전체' 로 6천 줄을 펼칠 수 있다** — 이미지를 바꿀 때 목록을 통째로 다시 그리지 말 것(`loadImage` 는 `markCurrent()`,
  한 장의 꼬리표는 `refreshRows()`). 통째로 다시 그리면 0.1초, 화면 밖 줄까지 다시 배치하면 0.4초가 걸려 Q·W·E 를 꾹 누르면
  25초 멈췄다 — `li.item` 의 `content-visibility: auto` 를 빼지 말 것. 반복 입력이 겹치지 않게 `S.domainBusy` 가 막는다.
  **그 짝으로 `li.item` 의 높이(32px)를 못 박아 둘 것** — 내용에 맡기면 건너뛴 줄(자리 표시 크기)과 그린 줄의 높이가 달라,
  다시 그릴 때 같은 스크롤 위치가 백 줄 넘게 다른 곳을 가리킨다(11-04, 태그 이름을 바꾸면 지금 줄을 놓쳤다).
- **Q·W·E·T·태그 칸으로 필터에서 빠진 줄도 목록을 다시 받기 전까지 제자리에 둔다**(11-04 사용자 요청 — 다시 받으면 목록이
  움직였다). 걷어 내는 것은 목록 위 `↻ 새로 받기`(`refreshList`). 태그 이름 바꾸기·지우기도 그 줄만 고친다. 승인 등으로 다시 받을
  때는 `keepCurrentRow()` 가 보던 자리를 화면에 그대로 두고, 다음 이미지는 **다시 받기 전 순서로** 고른다(`advanceFrom(id, before)` —
  순번으로 고르면 그새 빠진 줄만큼 건너뛰고, '최근 수정순' 에서 승인하면 맨 위로 옮겨 간 탓에 맨 위 둘째 줄로 갔다).
  **그런 줄이 남아 있으면 서버 쪽 묶음(100개씩…) 경계가 밀려 있다** — 묶음을 펼칠 때는 펼친 묶음을 전부 새로 받고(`toggleChunk` —
  그 묶음만 받으면 사이의 이미지가 어느 묶음에도 안 나왔다), 다시 받은 뒤 보던·넘어갈 이미지가 닫힌 묶음으로 밀렸으면
  `openChunkWith()` 가 그 묶음을 펼친다.
- **보기는 app.js 의 `VIEWS` 표 한 곳에 적는다 (11-15 리뷰).** 라벨링 말고의 보기(이미지 보기·영상 캡처·데이터 점검)마다
  `show`·`hide`(setView 가 부른다)·`key`(그 보기의 키 처리 — **없으면 키를 버린다**)·`vToggle`(V 가 라벨링과 오간다 — 이미지
  보기만)·`full`(세 칸을 통째로 쓴다 — `body.view-full` 이 왼쪽 목록·가운데·오른쪽 칸을 숨긴다, style.css 한 줄)을 적는다.
  `setView`·`viewKeydown`·처음 열 때의 마지막 보기(`viewOf`)가 이 표만 본다 — **보기를 더할 때 그 셋과 keydown 을 따로 고치지
  말 것**(전에는 다섯 곳을 나란히 고쳤고, keydown 을 빠뜨리면 숨은 라벨링에 Space 승인이 먹었다). 더할 것은 표의 한 줄,
  index.html 의 단추(`data-view`)와 `<section id="<이름>">`, style.css 의 `body:not(.view-<이름>) #<이름> { display: none }`
  뿐이다 — `tests/test_js_views.py` 가 표·단추·CSS 의 이름을 견주고, 표에 `key` 없는 보기를 하나 더 넣어 키가 새지 않는지 본다.
- **왼쪽 필터를 통째로 읽고 쓰는 것은 app.js 의 `noFilters`·`filterState`·`setFilters` 다 (11-15 리뷰)** — 필터 풀기
  (`clearFilters`)와 데이터 점검의 눌러 넘기기·되돌리기(`hcOpenFiltered`)가 함께 쓴다. **목록 필터를 새로 만들면 이 세 함수에
  넣을 것**(안 넣으면 필터 풀기·눌러 넘기기에서 안 풀리고, 실패했을 때 안 돌아온다 — 목록 주소는 `listUrl`, 요약은 아래).
- **화면 구성(11-04 개편)** — 작업 버튼은 상단 `데이터`·`모델` 메뉴(`.menu-item`, id 는 그대로라 onclick 도 그대로),
  왼쪽 필터는 접히는 `필터` 칸(`renderFilterSummary` 가 한 줄 요약 — 새 필터를 넣으면 거기도 넣을 것), 태그 필터는
  여러 개 넣고(포함) 빼는(제외) 찾기 칸(`makeTagFilter`, 11-04 — select 가 아니고 상태는 `S.tagF` 하나다. 코드로 바꾸면
  `tagFilter.render()`, 목록은 `applyTagFilter()` 로 다시 받는다 — 받는 사이 또 바뀌면 끝난 뒤 마지막 조건으로 한 번
  더. API 는 `tag=3,7`(모두, `tagmode=any` 면 하나라도)·`xtag=5`(빼기)·`tag=__none__`, SQL 은 `store.TagFilter`), 오른쪽은
  `.sec` 칸 접기(`localStorage` `secClosed`). **이미지 보기(`gallery.js`)는 같은 페이지의 다른 보기다**(`setView`·`VIEW`·
  `body.view-gallery`) — 필터·태그·`S.curTag` 를 함께 쓰고, 이미지 보기일 때 키는 `galleryKey` 로만 간다(라벨링 단축키가
  숨은 이미지에 먹으면 안 된다 — Space 승인). 목록은 지금 필터 전부를 받고 칸은 400개씩 그린다. 여러 장 바꾸기는
  `/api/bulk/tag`·`/api/bulk/domain`(**`/api/images/...` 밑에 두지 말 것** — `{image_id}` 경로와 겹치면 FastAPI 가 422
  에서 멈춘다), 바꾼 뒤 `G.items`·`S.images`·`S.cur` 를 함께 고친다(`applyToRows`). 30장 넘으면 묻는다. 썸네일은
  `/api/images/{id}/thumb?s=160|256|384` 가 `<DB 폴더>/thumbs/` 에 캐시한다(원본 수정 시각·크기가 이름에 — 백업에서 뺀다).
- **오른쪽 칸의 높이 나누기(11-15 사용자 화면 — style.css `.sec.grow`·`.hist-list`)** — 클래스·태그·템플릿 칸은 내용만큼, 박스 칸
  (`.sec.grow`, `flex: 1` = 바탕 높이 0)이 남는 높이를 갖는다. 그래서 변경 내역(H, 최대 38%)을 펴면 박스 칸이 0 으로 줄어 넘친
  '박스' 머리가 '변경 내역' 머리 위에 겹쳐 그려졌고 내역 끝은 화면 밖으로 잘렸다(2560×1440 창에서 칸을 다 편 채 H — 1500×950 과
  1600×1257 에서 재현). 이제 박스 칸은 편 동안 136px(머리 + 네 줄) 아래로 안 줄고, 변경 내역·예측 목록은 그만큼 **줄어든다**
  (`flex: 0 1 auto` — 세 줄 넘는 내역만 120px 바닥. `:has()` 로 가르는 것은 min-height 를 그냥 주면 한 줄짜리 내역 아래에 빈칸이
  생겨서다). 그래도 못 담는 낮은 창은 오른쪽 칸이 통째로 구른다(`#panel-label { overflow-y: auto }`) — 내역을 펴면 받은 뒤 보이게
  내린다(`toggleHistory`). **오른쪽에 칸을 더하면 낮은 창에서 박스 칸이 최소 높이로 가는지 볼 것.**
  **박스 칸은 11-15 까지 접히지 않았다**(같은 날 사용자 화면) — `.sec.grow .sec-body { display: flex }` 가 `.sec.collapsed .sec-body
  { display: none }` 과 같은 무게로 뒤에 와서 이겼다(머리의 ▸ 만 바뀜). 지금은 `.sec.grow:not(.collapsed) .sec-body` 다 — `.sec` 의
  몸통에 `display` 를 주는 줄을 더할 때는 접힌 것을 빼고 줄 것. 박스 칸을 접으면 변경 내역이 38% 한도 없이 남는 높이를 쓴다.
  칸 머리(`.sec-head`)는 한 줄이다(`nowrap` — 개수 글 `.dim` 만 말줄임, 전체는 `title`): 박스 머리의 개수 글이 길어지면 제목이
  '박/스' 로 꺾였다.
- **왼쪽 목록 폭(11-07)** — 경계 손잡이(`#list-resizer`)를 끌면 app.js 가 `.layout` 의 `--left-w` 를 바꾸고(localStorage
  `listWidth`), CSS 의 `--left-eff`(clamp)가 창 폭에 맞춰 줄인다(가운데 420px·오른쪽 270px 은 남김, 250~900px). 창 크기
  이벤트에 기대지 말 것 — 에뮬레이션으로 창을 줄이면 resize 이벤트 없이 목록이 화면을 다 덮었다. 이미지 보기에서는
  늘 250px 다. 왼쪽 칸 폭을 다른 곳에 숫자로 박지 말 것.
- **태그(11-03)** — `tags`·`image_tags` 표(이미지 하나에 여러 개, 이미지를 지우면 CASCADE 로 떨어진다). 라벨이 아니라
  `updated_at`·변경 이력에 남기지 않고, 학습·분할·`test_baseline` 은 아직 보지 않는다. 검수 UI 의 T(고른 태그 붙이고 다음,
  `S.tagBusy`)·칸(붙이기/떼기)·✎·×·태그 필터. legacy 의 카메라·크롭 분류를 사용자가 여기에 담는다 — 분할에 쓰게 되면 태그
  이름이 바뀔 수 있으니 id 로 가리킬 것.
- 이미지를 밖에서 편집해 확장자·크기가 바뀌면 조용히 어긋난다 — **`labeler.py check`**(파일 없음·열리지 않음·크기 불일치·경로
  표기), 크기만 바뀌었으면 `--fix-size`(박스 값은 그대로 둔다).
- **사람이 보지 않은 것은 승인되지 않게 (11-05).** 승인·건너뛰기·Q·W·E·T·새로 받기는 서로 겹치지 않는다(`navBusy()` — 도는 중에
  들어온 것은 버린다), Space·Enter·S 는 키 반복(`e.repeat`)을 버린다. 승인은 지금 이미지의 사진이 뜬 뒤에만(`S.imgId` — 로드는
  `S.loadSeq` 로 마지막 것만, 사진이 뜨기 전에는 앞 장 그림을 두고 캔버스 입력도 받지 않는다), 휴지통 이미지는 승인하지 않는다(검수
  대기로 먼저). 자동 라벨링이 도는 중(`S.jobKind`)에는 승인 전에 서버 박스를 다시 보고 다르면 다시 읽는다. 이미지를 바꾸는 작업
  (자동 라벨링·휴지통 비우기·정리, `IMAGE_JOBS`)이 끝나면 `syncCurrentAfterJob` 이 지금 이미지를 맞춘다(404 면 비우고 첫 장) —
  학습·스캔은 넣지 말 것(긴 학습이 끝날 때 되돌리기·확대가 날아간다). 서버 쪽 `AutoLabeler.run` 도 시작 때의 (상태, updated_at)
  과 같을 때만 쓴다(`store.apply_prediction`).
- **↑·↓ 는 누르고 있으면 계속 넘어간다 (11-12 사용자 요청) — Space·Enter·S 와 달리 키 반복을 받는다.** keydown 이 `e.repeat` 를
  `step(delta, held)` 에 넘기고, 반복 입력은 앞 장이 다 뜬 뒤에만 넘긴다(`imageLoading()` — `S.loading`: 돌고 있는 `loadImage`
  수, `S.imgPending`: 사진을 받는 중인 메타). 그 전에는 반복(약 33ms)마다 넘기려다, 정보 받기(실측 40~50ms)가 끝나기 전에 같은
  장을 또 요청하고 앞 요청은 `S.loadSeq` 에 버려져 누르는 동안 한 장만 넘어갔다(시험 서버 3초: 1장 → 32장, 장마다 사진이 뜬 뒤).
  **입력이 올 때만 움직인다** — '누르고 있음' 을 따로 기억해 타이머로 넘기지 말 것(창이 포커스를 잃어 keyup 을 놓치면 혼자 계속
  넘어간다). `loadImage` 는 `loadImageNow` 를 감싸 어느 길로 끝나든 `S.loading` 을 내리고, `S.imgPending` 은 onload·onerror 가
  내린다 — 새 끝내기 길을 만들면 둘 다 볼 것(안 내리면 누르고 있는 화살표가 그 뒤로 먹지 않는다). 목록 끝에서 반복 입력은 같은
  장을 다시 열지 않는다(한 번 누르기는 전처럼 다시 연다). `tests/test_js_nav.py` 가 이 함수들을 node 로 돌린다.
- **건너뛰기와 '검수 대기로' 는 한 번 묻는다 (11-11 사용자 요청).** `skipOrAdvance` 는 건너뜀으로 내릴 때만 `confirm` 한다(상태를
  바꾸지 않는 '다음' 은 묻지 않는다). 아래 줄의 `검수 대기로`(`#btn-to-pending` → `backToPending`)는 승인됨·건너뜀·휴지통에서
  보이고(`BACK_TO_PENDING` — 전의 휴지통 '복원' 을 넓혔다), 상태만 pending 으로 바꾸고 다음으로 간다. 둘 다 묻기 전에 `navBusy()`
  를 보고(물어 놓고 setStatus 가 버리지 않게), 답한 뒤 지금 이미지가 그대로인지 다시 본다. 서버의 `/status` 는 어느 상태로든
  받는다 — 막는 것은 화면뿐이다. pending 으로 돌린 승인 이미지는 `label --all` 이 예측으로 덮는다(이력은 남는다, README).
- 끄는 박스는 `drag.box` 로 붙잡고 되돌리기는 처음 바꿀 때 쌓는다(`editDragBox`), 그 박스가 없어지면 끌기를 끝낸다(`dragLost`).
  크기 조절·새 박스는 1px 아래로 만들지 않는다(`tooSmall` — 서버는 w·h 0 을 422, 1e-4 이하를 조용히 버린다). 예측(P) 요청은
  200ms 미뤄 마지막 장만 보내고(`S.predSeq`) 이미지 보기에서는 보내지 않는다.
- **검수 서버는 다른 사이트에서 부르기를 막는다**(11-05, `server._RequestGuard` — 바깥 앱에 한 번). GET·HEAD·OPTIONS 가 아닌
  요청은 `Sec-Fetch-Site` 가 same-origin·none 이 아니거나 `Origin` 이 Host 와 다르면 403, Host 가 루프백·bind 주소가 아니면 400
  (`--host 0.0.0.0` 이면 Host 는 안 본다). 헤더 없는 요청(urllib·curl)은 통과한다. **TestClient 는 `base_url="http://127.0.0.1:7860"`**
  으로(기본 testserver 는 400). 엔드포인트의 `-> dict` 반환 주석은 지우지 말 것 — 그 덕에 결과의 NaN 이 null 로 나간다.
- **오류 응답(4xx·5xx)은 `Cache-Control: no-store` 로 나간다 (11-15, `no_cache_web` 미들웨어).** 크롬은 캐시 헤더가 없는 410 을
  기한 없이 캐시한다 — `--config` 로 하나만 띄운 시험 서버에서 '원본 파일이 사라졌습니다'(410)가 파일이 돌아온 뒤에도 그 썸네일
  주소에 남았다(같은 주소가 캐시로는 410, 캐시를 건너뛰면 200. 썸네일 주소에는 `?v=` 가 없다). 탭으로 띄운 서버는 경로가
  `/p/<이름>/api/…` 라 그 전에도 no-cache 가 붙어 있었다. 파일이 없을 때의 응답 코드를 바꾸거나 캐시 헤더를 직접 다는 길을
  만들면 `tests/test_server.py` 의 두 테스트를 볼 것.
