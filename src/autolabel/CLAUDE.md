# src/autolabel — UI 조작·저장 경로에서 지킬 것

2026-11-10 에 저장소 루트 CLAUDE.md 에서 낱말 그대로 옮겨 왔다(그 파일이 매 세션 읽혀 크기를 줄이느라). 글 속의 '위·아래 …' 는
루트 CLAUDE.md 의 절을 가리킨다.
이 파일은 `src/autolabel/` 아래 파일을 다룰 때만 읽힌다.

사용자 문서·단축키는 README '검수 UI 사용법'. 작업자가 알아야 할 것만:
- **편집은 0.6초 뒤 자동 저장된다**(`app.js` `markDirty`·`flushSave`) — UI 를 시험하면 실제 작업물이 바뀐다. 먼저
  `GET /api/images/{id}` 로 떠 두고 끝나면 `PUT /api/images/{id}/boxes` 로 되돌릴 것. 경합 시험은 **DB 복사본**(sqlite `backup()`)
  으로 띄운 서버에서 한다.
- **저장은 한 번에 하나만 돈다**(`S.saving`·`S.rev`). 저장이 실패하면 이미지를 넘기거나 승인하지 않는다(`saveBeforeLeaving`).
  응답을 기다리는 곳(SAM·예측·이미지 로드·변경 내역)은 **요청한 이미지 id 를 붙잡아 두고 바뀌었으면 결과를 버린다** — 새
  비동기 호출을 넣을 때도 그렇게 할 것.
- 여러 장을 연달아 고친 뒤에는 **박스가 줄어든 저장을 확인할 것**(상단 '변경 기록' 이나 `GET /api/history?shrunk=1`).
- **박스를 바꾸는 길은 전부 `store._replace_boxes` 를 지난다**(변경 이력 `box_history` 에 그 이미지의 박스 목록 전후를 JSON 으로
  남긴다). 새 경로도 그걸 거치게 하고, `_write_lock` 은 **부르는 쪽이** 쥔다(안에서 다시 잡으면 멈춘다).
  - 파일을 바꾼 이미지(`store.replace_file` — VIRAT `--recrop` 은 source predict, `tools/crop_images.py` 는 crop)는 그 전 이력의
    박스가 옛 화면 좌표다. crop 은 서버가 그 전 이력으로 되돌리기를 막지만(`store.crop_entries`), VIRAT 자르기 이력은 가려내지 못한다.
    가릴 영역이 있는 이미지에 `masks=`(새 그림 기준으로 옮긴 영역)를 안 주면 `replace_file` 이 ValueError 를 낸다(아무것도 안 쓴다).
- **가릴 영역 (2026-11-09, `src/autolabel/masks.py`)** — 라벨하기 애매한 물체를 검수 화면 `M` 모드로 덮는 사각형
  (무엇을 덮는지는 docs/label-rules.md 한 곳에 적는다 — 기준을 바꾸면 거기와 화면 문구 `index.html`·`app.js` 를 볼 것).
  DB `masks` 표(정규화 xyxy, EXIF 회전을 적용한 화면 기준)에만 있고 **원본 그림은 그대로**다. 학습 세트(`trainer.build_dataset`)와
  이미지를 함께 내보내기(`export_yolo`)만 그 자리를 회색 114 로 칠한 JPEG 을 쓰고, 영역들의 합집합이 박스 넓이의 반 이상을 덮는
  박스는 그 라벨 파일에서만 뺀다(DB 박스는 그대로). **test 는 칠하지 않는다** — 서버가 test 이미지(분할 test·test 고정 카메라)에
  영역을 받지 않고(409, 지우기만 받는다) 학습 세트도 test 분할은 원본 그대로다. 원본을 직접 읽는 도구(test_baseline·label_qa·
  test_tiles 등)는 영역을 모른다.
  - 저장(PUT boxes)의 `masks` 는 **그 화면에서 영역을 손댄 뒤로만** 싣는다(`S.masksTouched`) — 안 실으면(옛 화면 포함) 서버가
    그대로 둔다. 영역이 바뀐 저장은 박스가 그대로여도 이력에 남고(`box_history.masks_before·masks_after`, 박스만 바뀐 줄은 NULL),
    되돌리기는 그 줄에 영역이 있을 때만 영역도 되돌린다. 되돌리기(Ctrl+Z) 한 칸은 박스와 영역을 함께 쌓는다.
  - 가릴 영역 모드에서는 영역만 고르고 옮긴다(박스는 B). 덮는 몫은 `masks.covered` 와 `app.js` 의 `maskCover` 가 같은 식이다
    (좌표 압축으로 합집합) — 고치면 둘 다.
  - **가릴 영역 템플릿 (2026-11-10, 사용자 요청)** — 이름 붙인 영역 묶음(DB `mask_templates`, 정규화 xyxy JSON + 만든 그림의 화면
    크기). 적용하면 이미지의 `masks` 로 **복사**된다(템플릿을 고치거나 지워도 적용한 이미지는 그대로). 한 장 적용은 화면이
    `S.masks` 에 더하고 평소 저장(PUT boxes, `S.masksTouched`)으로 간다 — 되돌리기·이력이 그대로 먹는다. 여러 장은 이미지 보기의
    `/api/bulk/mask-template`(add·remove) → `store.apply_mask_template`: 이미지마다 `_replace_boxes(source='bulk', batch=…, masks=…)`
    라 일괄 수정 묶음으로 되돌린다. `revert_batch` 는 `masks_after` 가 있고 박스 전후가 같은 줄(영역만 바꾼 줄)은 **박스를 견주지
    않고 지금 박스를 그대로 둔 채** 영역만 되돌린다(지금 영역이 적용 직후와 같을 때만) — 템플릿을 넣은 뒤 박스를 검수한 장도
    되돌려야 해서다. 되돌리면 영역이 새로 생기는 장이 그새 test 가 됐으면 건너뛴다.
    **test 판정은 server 의 `mask_block` 한 곳에만 둔다** — store(`apply_mask_template(blocked=)`·`revert_batch(mask_blocked=)`)에는
    `mask_blocker()` 로 넘긴다(store 는 cameras.yaml 을 모른다). 넘기지 않으면 store 는 막지 않는다.
    같은 자리 판정은 `masks.mask_key`(소수 7자리) — 더하기는 같은 좌표를 건너뛰고(`masks.merge`), 빼기는 같은 좌표만 뺀다
    (`masks.subtract`). 화면의 `xyxyKey` 가 같은 규칙이다 — 고치면 둘 다. 더하기는 test·화면 비율이 1% 넘게 다른 그림(DB 크기로
    먼저 보고 어긋날 때만 `display_size` 로 연다)·50개 넘김을 건너뛴다. 화면은 적용 직전에 목록을 다시 받는다(`freshMaskTemplate`).
  - `_replace_boxes` 는 **박스가 그대로면(같은 `_box_key` 목록) boxes 행을 지우고 다시 넣지 않는다**(2026-11-10) — 영역만 바꾸는
    저장·템플릿 적용에서 박스 행 id 가 그대로 남는다. 이력은 전과 같이 남긴다.
- DB 자동 백업(`store.backup_daily`)은 `<DB 폴더>/backups/` 에 하루 한 번, 최근 `backup.keep`(14)개 — 그 이름 꼴의 파일만 지운다.
  **시험 서버는 `backup.keep: 0` 으로 끌 것.**
- 같은 자리 일괄 수정(`store.same_place` → `set_class_bulk`)은 적용 때 다시 찾아 미리 본 수와 다르면 409, 묶음째 되돌리기는 그
  뒤 손대지 않은 이미지만 되돌린다. 묶음(grp)이 정해지기 전의 새 이미지는 쓸 수 없다.
- **예측 비교(P)** 는 운영 가중치로 한 장 추론만 하고 DB 에 쓰지 않는다(비교는 브라우저 `computeFindings`, 규칙은
  `tools/label_qa.py` 와 같고 예측 하한만 다르다 — label_qa 0.10, P 는 `predict.conf` 0.25). 모델이 모르는 클래스(11-08 전 운영 visible-3 은
  pickup)는 비교하지 않는다. 올릴 때 빈 이미지로 예열하는데, **자동 라벨링(`AutoLabeler`)은 예열하지 않는다** — 첫 추론이 CUDA
  초기화로 길어지면 NMS 시간 제한에 걸려 검출이 'NMS time limit exceeded' 경고만 남기고 잘릴 수 있다(한 번 봤고 조건은 모른다).
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
  `S.loadSeq` 로 마지막 것만, 사진이 뜨기 전에는 앞 장 그림을 두고 캔버스 입력도 받지 않는다), 휴지통 이미지는 승인하지 않는다(복원
  먼저). 자동 라벨링이 도는 중(`S.jobKind`)에는 승인 전에 서버 박스를 다시 보고 다르면 다시 읽는다. 이미지를 바꾸는 작업
  (자동 라벨링·휴지통 비우기·정리, `IMAGE_JOBS`)이 끝나면 `syncCurrentAfterJob` 이 지금 이미지를 맞춘다(404 면 비우고 첫 장) —
  학습·스캔은 넣지 말 것(긴 학습이 끝날 때 되돌리기·확대가 날아간다). 서버 쪽 `AutoLabeler.run` 도 시작 때의 (상태, updated_at)
  과 같을 때만 쓴다(`store.apply_prediction`).
- 끄는 박스는 `drag.box` 로 붙잡고 되돌리기는 처음 바꿀 때 쌓는다(`editDragBox`), 그 박스가 없어지면 끌기를 끝낸다(`dragLost`).
  크기 조절·새 박스는 1px 아래로 만들지 않는다(`tooSmall` — 서버는 w·h 0 을 422, 1e-4 이하를 조용히 버린다). 예측(P) 요청은
  200ms 미뤄 마지막 장만 보내고(`S.predSeq`) 이미지 보기에서는 보내지 않는다.
- **검수 서버는 다른 사이트에서 부르기를 막는다**(11-05, `server._RequestGuard` — 바깥 앱에 한 번). GET·HEAD·OPTIONS 가 아닌
  요청은 `Sec-Fetch-Site` 가 same-origin·none 이 아니거나 `Origin` 이 Host 와 다르면 403, Host 가 루프백·bind 주소가 아니면 400
  (`--host 0.0.0.0` 이면 Host 는 안 본다). 헤더 없는 요청(urllib·curl)은 통과한다. **TestClient 는 `base_url="http://127.0.0.1:7860"`**
  으로(기본 testserver 는 400). 엔드포인트의 `-> dict` 반환 주석은 지우지 말 것 — 그 덕에 결과의 NaN 이 null 로 나간다.
