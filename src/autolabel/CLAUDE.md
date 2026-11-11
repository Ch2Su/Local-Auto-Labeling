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
- **타일 추론(2026-11-10, `tiling.py`)** 은 자동 라벨링의 선택지다(`AutoLabeler.run(tiles=True)` ← `PredictIn.tiles` ← 메뉴
  '자동 라벨링 (타일)', CLI `label --tiles`, 설정 `predict.tiles`). **`tools/test_tiles.py` 와 같은 함수를 쓴다** — 고치면 측정도
  바뀐다. 운영 조합은 `tiling.detect`(측정 변형 `g2+full:big` + `--cross-iou`): 성분마다 RAW_*(0.001·0.7·300)로 원검출을 받아
  잘린 조각 박스를 버리고 `merge_cross` 로 합친 **뒤에** predict.conf·max_det 로 거른다 — 그래서 조각을 안 쓰는 작은 이미지도
  보통 모드와 아주 조금 다를 수 있다. 그림은 `ultralytics.utils.patches.imread`(Ultralytics 가 경로로 읽을 때와 같은 함수)로 읽고,
  타일 모드만 빈 이미지로 예열한다. 타일로 라벨링한 이미지는 `model_version` 끝에 `+tiles<grid>`. 예측 비교(P)는 통째 그대로다.
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
    `committing` 이라 다시 뽑기·닫기가 프레임을 지우지 못한다.
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
  - **카메라 지우기(`지우기…` → `POST /api/cameras/remove` → `provenance.remove_cameras`, 11-10 사용자 요청).** 같은 방식으로
    **그 카메라의 줄만** 뺀다 — 키 줄, 그 아래 더 깊이 들여쓴 줄(속성·안쪽 주석), 키 바로 위에 **같은 들여쓰기로** 붙은 주석(등록
    주석). 맨 왼쪽의 묶음 설명 주석과 끝의 얕은 주석·빈 줄은 다음 키 것으로 보고 남긴다. 다시 읽어 '그 키들만 없어지고 나머지는
    속성까지 그대로' 일 때만 쓰고(백업은 add 와 같은 곳), 없는 키가 하나라도 있으면 하나도 안 뺀다. 더했다 빼면 바이트까지 처음과
    같다(11-10 시험). **이미지가 남은 카메라는 server 가 막는다(409)** — `store.camera_counts(with_trashed=True)`(휴지통도 그
    카메라를 가리킨다)와 `unlabeled/<키>`·`labeled/images/<키>`·`labeled/labels/<키>` 의 파일. 검사와 빼기 사이에 넣기·스캔이 끼지
    않게 `capman.commit_lock` → `ingest_lock` 순서로 쥔다(넣기도 같은 순서 — 바꾸면 서로 막힌다). 지운 카메라를 고른 캡처 세션은
    카메라를 비운다(`forget_cameras`). 빈 카메라 폴더는 지우지 않는다. 화면 목록의 `db`·`files` 는 `_camera_rows` 가 붙인다. 화면은 계열(접두어+번호)마다 다음 번호를 제안하고(공개 데이터셋만 있는
    계열은 뺌), 처음 보는 접두어·`test` 고정은 한 번 더 묻는다(키에 지명이 드러나면 문서·커밋에 쓸 수 없다 — 루트 CLAUDE.md git 절).
  - **키는 app.js keydown 이 `VIEW === "capture"` 면 V 보다 먼저 `captureKey` 로만 넘긴다** — 숨은 라벨링에 Space(승인)·S·Q 가
    먹으면 안 된다(11-10 에 확인). 크게 보기는 공용 대화상자가 아니라 `#cap-preview`(키도 captureKey).
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
