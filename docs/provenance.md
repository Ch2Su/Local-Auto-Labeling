# 출처 기록 설계 (2026-11-03 — 1~4단계 구현됨, 5단계(공개 데이터셋 가져오기)는 따로)

11-03 에 그때까지 모인 이미지를 `data/visible/legacy/` 로 옮겼다(출처 모름). 그 뒤로 새로 모으는 이미지는
**어느 출처 단위(카메라)에서 언제 찍혔는지** 를 DB 에 남긴다. 이 문서는 그 규칙과 구현 계획이다.

## 사용자 결정 (11-03)

| | 결정 |
|---|---|
| 수집 형태 | 영상 프레임 추출 · 카메라/폰 사진 · 화면 캡처 · 공개 데이터셋 — 전부 |
| 분할 단위 | **카메라째 한 분할** — 같은 카메라는 train/val/test 중 한쪽에만 |
| 위치 정밀도 | **장소 이름만** — 좌표는 적지 않는다 |
| 등록 파일 | **YAML** — `data/visible/cameras.yaml` |

## 출처 단위('카메라')

분할과 평가의 단위다. 형태마다 무엇을 한 단위로 볼지가 다르다.

| kind | 한 단위 | 예 |
|---|---|---|
| `cctv` | 고정 카메라 한 대 | 교차로 기둥 카메라 |
| `phone` | 한 장소에서 한 번 나가 찍은 묶음 | 어느 날 한 주차장에서 찍은 사진들 |
| `screen` | 캡처한 화면의 원 카메라(모르면 캡처 원천 하나) | 공개 CCTV 웹 뷰어의 한 채널 |
| `dataset` | 공개 데이터셋의 장면 | VIRAT 장면 `0002` |

같은 장소라도 카메라가 다르면 다른 단위다. 폰으로 같은 장소를 다른 날 다시 찍으면 **같은 단위로 등록할 것** —
다른 단위로 두면 같은 배경이 train 과 test 로 갈린다.

## 폴더 규칙

```text
data/visible/unlabeled/<카메라키>/[<YYYYMMDD>/]<파일>
```

- 카메라키는 `cameras.yaml` 의 키 그대로, 영문 소문자·숫자·`-` 만(`^[a-z0-9][a-z0-9-]*$`) — 폴더 이름이 되므로
  한글·공백을 피한다.
- 날짜 폴더는 정리용이고 없어도 된다(시각은 아래 순서로 따로 읽는다).
- `unlabeled/` 바로 아래 둔 파일은 지금처럼 출처 없이(`source='unlabeled'`) 들어가고 scan 이 경고한다.
- 지금 tidy 는 승인분을 `labeled/images/` **한 폴더에 평평하게** 옮긴다(`tidy.tidy_approved` → `_free_name`). DB 의
  `camera` 열은 남지만, DB 를 잃으면 `labeled/` 만으로는 카메라를 되찾을 수 없다 — 그래서 tidy 가 카메라 폴더를
  유지하게 바꾼다(`labeled/images/<카메라>/…` ↔ `labels/<카메라>/…`, 구현 순서 1).

## 등록 파일 `data/visible/cameras.yaml`

**저장소(public)에 담지 않는다** — `data/` 아래라 이미 `.gitignore` 대상이고, 다른 디스크 백업(robocopy)에는 들어간다.
설치 장소를 적으므로 이 파일과 그 내용을 커밋·업로드하지 말 것.

```yaml
cameras:
  pg-cross-01:            # 키 = 폴더 이름
    kind: cctv            # cctv | phone | screen | dataset
    place: 판교 ○○ 사거리   # 장소 이름만. 좌표 금지
    mount: pole           # pole | rooftop | wall | indoor | drone | handheld | unknown
    view: 교차로 북쪽, 부감 약 30도   # 자유 서술
    resolution: 1920x1080
    tz: '+09:00'          # 시각 해석 기준. 기본 +09:00 (이 PC 의 Python 에 tzdata 가 없어 Asia/Seoul 같은 이름은 안 풀린다)
    rights: 직접 촬영      # 권리자·라이선스·사용 조건
    split:                # 비우면 자동(카메라째 한 분할). train|val|test 로 고정할 수 있다
    note:
  virat-0002:
    kind: dataset
    place: VIRAT scene 0002
    mount: rooftop
    rights: VIRAT Release 2.0 — 개인별 동의 약관, 재배포 금지로 볼 것
```

- 속성은 이 파일 하나에만 둔다. DB 에는 키만 남긴다(두 곳에 두면 어긋난다).
- 키를 바꾸려면 폴더 이름과 DB `images.camera` 를 같이 바꿔야 한다 — 되도록 바꾸지 말 것.

## DB 변경

`images` 에 열 둘을 더한다(`ALTER TABLE ADD COLUMN` — `split`·`grp`·`dhash` 때와 같은 방식, 기존 행은 그대로).

| 열 | 값 | legacy |
|---|---|---|
| `camera` | 카메라키 | NULL |
| `captured_at` | 촬영 시각 ISO 8601, 시간대 포함(`2026-11-03T14:05:12+09:00`) | NULL |

- 카메라 폴더로 들어온 이미지의 `source` 는 **`collected`** — 지금 `source='unlabeled'` 는 `tools/test_baseline.py` 에서
  '비CCTV' 를 뜻하므로 새 CCTV 를 거기 섞지 않는다. 공개 데이터셋은 지금처럼 데이터셋 이름(`o365`, `virat`)을 쓴다.
- `ingest_batch` 는 경로가 이미 있으면 `source` 를 다시 쓴다(`store.py`) — `camera`·`captured_at` 은 **처음 넣을 때만**
  쓰고 덮지 않는다.

## 촬영 시각 읽는 순서

처음 걸리는 것을 쓴다. 못 찾으면 NULL(모름) — **파일 수정 시각은 쓰지 않는다**(복사·다운로드 때 바뀐다).

1. 파일 이름의 시각 — `..._YYYYMMDD_HHMMSS[_mmm]`(아래 프레임 추출 도구가 쓰는 형식), 폰 `YYYYMMDD_HHMMSS`,
   Windows 캡처 `스크린샷 YYYY-MM-DD HHMMSS`(`grouping.py` 가 이미 읽는 꼴).
2. EXIF `DateTimeOriginal`(+ 있으면 `OffsetTimeOriginal`) — 폰·카메라 사진.
3. 없음.

시간대가 없는 값은 카메라의 `tz` 로 해석한다.

## 영상 → 프레임: `tools/extract_frames.py`

```bash
.venv/Scripts/python.exe tools/extract_frames.py <영상> --camera pg-cross-01 --start "2026-11-03 14:00:00" --every 5
```

- `unlabeled/<카메라>/<YYYYMMDD>/<카메라>_<YYYYMMDD_HHMMSS>_<mmm>.jpg` 로 쓴다(시각 = 시작 + 프레임 시각).
- `--start` 를 빼면 영상 이름에서 시각을 찾고, 없으면 멈춘다(시각 없는 프레임을 만들지 않는다).
- 간격은 기본 5초(`--every`, 0·5·10초… 격자). 같은 이름이 있으면 건너뛰어 다시 돌려도 결과가 같다. `--dry-run` 은 셈만 한다.
- **거의 같은 프레임 건너뛰기는 기본으로 끈다.** dHash 거리 2 이하로 건너뛰게 짰다가, VIRAT 옥상 영상에서 5초 간격
  138장 중 125장이 걸렸다 — 8x8 dHash 는 화면 속 작은 사람·차를 못 본다. 대신 `--min-change <퍼센트>` 를 주면 직전
  저장 프레임과 비교해 밝기가 20 넘게 바뀐 픽셀 비율(가로 320px)이 그보다 작을 때 건너뛴다. 같은 영상 137쌍의 비율은
  중앙값 0.77%, 0.05% 미만(사실상 같은 화면) 22쌍 — 똑같은 화면만 빼려면 0.05 정도다. 바이트까지 같은 프레임은
  scan 의 중복 검사(dupcheck)가 어차피 막는다.
- 실측(11-03): VIRAT `VIRAT_S_000001.mp4`(1080p, 29.97fps, 11.5분)를 끝까지 훑는 데 32초.

## 분할

- `grouping.group_images` 에서 **같은 카메라는 하나로 합친다**(지금의 연번·연사·dHash 합집합에 하나를 더한다).
  그래서 카메라 하나 = 묶음 하나(dHash 로 다른 이미지와 이어지면 더 커진다).
- `cameras.yaml` 의 `split` 이 있으면 그 묶음은 그 분할로 고정한다. 없으면 지금의 계층화가 묶음째 배정한다.
- **기존 이미지는 움직이지 않는다**는 규칙은 그대로다. 새 카메라 묶음이 dHash 로 legacy 묶음과 이어지면 옮기지 않고
  `[알림] … 두 분할에 걸칩니다` 로 알린다.
- 카메라가 적은 동안은 15% test 가 카메라 한두 대로 정해져 비율이 크게 틀어질 수 있다 — split 보고에 카메라별
  배정을 찍어 보이고(`카메라별: …`, 두 분할에 걸친 카메라는 `[경고]`), 필요하면 `split:` 으로 직접 고정한다.
- 고정은 다수결보다 앞선다. 다만 **이미 다른 분할에 있던 이미지는 옮기지 않고** `[알림] … 옮기지 않았습니다` 로
  알린다(기존 배정 고정 규칙). 고정이 서로 다른 카메라가 dHash 로 한 묶음이 되면 어느 쪽도 지킬 수 없어
  **분할을 멈춘다** — `split` 도 `grp` 도 쓰기 전에 멈춘다. 등록 파일이 깨졌을 때도 같다.
- 확인(11-03): 실제 DB 사본 두 개에 옛 코드(`b149bcb`)와 새 코드로 `ensure_splits` 를 돌려 승인 6,567장의 `grp`·`split`
  이 한 행도 다르지 않았다(legacy 는 카메라가 없어 결과가 같아야 한다).

## 평가

`tools/test_baseline.py --domains` 가 고정 도메인에 더해 `legacy`(seed·unlabeled)·`collected` 를 재고, 그 분할에 카메라
이미지가 있으면 `cam:<키>`·`kind:<형태>`·`mount:<설치>` 도 잰다(`--domain cam:pg-cross-01` 처럼 하나만도 된다).
`all` 은 collected 도 담으므로 09-27 기준선과 같은 범위는 `legacy` 로 견준다. 출처 열이 없는 옛 DB 도 그대로 읽는다.

## 검수 UI

- 왼쪽 목록 위에 **카메라 필터**(`카메라: 전체 / <키> · 장소 (장수) / 출처 없음`) — 카메라가 붙은 이미지가 하나라도
  있을 때만 보인다. 상태·클래스 필터와 AND 로 걸린다(`GET /api/images?camera=<키>`, 출처 없음은 `__none__`).
  목록은 `GET /api/cameras`(DB 의 카메라별 장수 + 등록 파일의 장소·형태·고정 분할)로 채운다.
- 이미지 머리줄에 `카메라 <키>(장소) <촬영 시각>` — 시각은 카메라 현지 시각 그대로, 모르면 `촬영 시각 모름`.
- 카메라 등록·수정 화면은 없다 — `cameras.yaml` 을 손으로 고친다(등록 파일이 깨지면 필터 툴팁에 오류가 뜬다).

## 구현 순서

1. **(11-03 구현)** `store.py` 열 추가 · `config.yaml` 에 `data.cameras` · `provenance.py`(등록 파일·시각) · `ingest.py`
   가 카메라 폴더를 읽고 시각을 채운다. 등록 파일에 없는 카메라키가 있으면 **그 scan 전체를 멈춘다**(오타가 새
   카메라로 들어가는 것을 막는다). tidy 가 카메라 폴더를 유지한다.
2. **(11-03 구현)** `grouping.group_images` 가 같은 카메라를 한 묶음으로, `split.ensure_splits` 가 `split:` 고정과
   카메라별 보고를 한다.
3. **(11-03 구현)** `tools/extract_frames.py`.
4. **(11-03 구현)** `tools/test_baseline.py` 출처 도메인 · 검수 UI 카메라 필터와 카메라·시각 표시.
5. (따로) 공개 데이터셋 가져오기 — VIRAT 은 장면(`XXYY`)을 카메라로, 라벨은 **검수 대기**로 넣는다
   (행동과 무관한 객체 박스가 빠져 있다 — VIRAT 배포본 `docs/VIRAT_Video_Dataset_Release2.0_Introduction_v1.0.pdf` 3.1).

## 정하지 않은 것

- 검수 UI 에서 카메라를 등록·수정할지 — 지금은 YAML 을 손으로 고친다.
- legacy 이미지 중 출처를 아는 것(예: 795프레임 고정 CCTV)에 나중에 카메라키를 붙일지.
