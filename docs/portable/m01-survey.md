# Milestone 0.1 — 포터블 기반 조사 (2026-11-28, HEAD `33a4a55` 뒤, 코드는 바꾸지 않았다)

**방향 (2026-11-28 사용자 결정 — '0.1 조사 시작').** 지금까지는 혼자 쓰는 로컬 도구였다. 사용자가 ChatGPT 와 나눈 논의로
"다른 사람도 받아 쓰는 Windows·macOS·Linux 포터블 앱(압축을 풀고 바로 실행)" 을 장기 목표로 정했고, 그 인수인계 문서를
[handoff-chatgpt-20261128.md](handoff-chatgpt-20261128.md) 에 그대로 두었다(그 문서의 코드 현황은 이전 정적 리뷰라 이 조사로
다시 확인했다 — 아래 1절). CCTV 검출부를 만드는 일(CLAUDE.md 머리의 방향)과의 우선순위는 사용자가 정한다.

이 문서는 그 문서 15절의 첫 요청(Milestone 0.1 — 경로·저장 위치·OS 전용 코드·장치·DB·테스트를 조사하고 위험·순서·호환·되돌리기
계획을 세운다)의 결과다. **조사만 했고 코드는 바꾸지 않았다.** 수치는 이 날 이 저장소에서 grep·읽기 전용 DB 조회로 센 것이다.

---

## 1. 인수인계 문서의 지적 — 확인 결과

| 지적 | 지금 코드 | 판단 |
|---|---|---|
| A. 경로가 저장소 위치에 묶임 | `config.py:12` `ROOT = Path(__file__).parents[2]`, 상대경로는 전부 `resolve()` 가 `ROOT` 기준으로 푼다. DB `images.path` 는 절대경로 | 맞음 (2절) |
| B. 서버가 WSL 모듈을 맨 위에서 import | `server.py:57` `from . import wsl_train` | 사실. 다만 `wsl_train` 이 표준 라이브러리·프로젝트 모듈만 import 해서 **import 자체는 다른 OS 에서도 될 것으로 본다**(돌려 보지는 않았다). OS 를 타는 것은 `wsl.exe` 를 부를 때다 (3절) |
| C. 기본 장치가 GPU 0 | `config.py:125` `model.device` 기본 `0` | 맞음. 장치 판단은 흩어져 있지 않다 — `cfg.device` 를 쓰는 곳 다섯(학습 2·예측 비교·자동 라벨링·SAM)과 `gpu_busy()` 면 CPU 로 가는 두 곳뿐 (4절) |
| D. 큰 모듈 | `server.py` 2,088줄, `store.py` 2,039줄 | 맞음 |
| E. 데이터셋을 지운 뒤 검사 | `trainer.build_dataset` 이 `rmtree(root)`(124–125줄) 뒤에 클래스·승인 수를 본다 | 사실이지만 **위험은 작다** — `dataset/` 은 학습마다 DB 로 새로 만드는 산출물이라 실패해도 DB·원본은 그대로다. 포터블에서 순서를 바꾸는 것은 싸다 |
| F. 헤더 없는 요청 허용·`0.0.0.0` 이면 Host 생략 | `server._RequestGuard` 그대로 | 맞음. 헤더 없는 요청은 **일부러** 받는다(`tools/auto_capture.py`·`scheduled_stop.py` 가 urllib 로 부른다). 배포판에서는 토큰으로 다시 설계할 곳 |
| G. 개발 설정과 제품 설정이 섞임 | CLAUDE.md·`.claude/`(자동 커밋 훅)·`wsl/`·Runpod 문서·Gemini 칠판 | 맞음 |
| 테스트 약 698개 | 지금 **701개**(파일 41·함수 549, 매개변수화 포함), 화면 JS 는 node 로 | — |
| pyproject | 없다. `requirements.txt` 는 노트북 기준(Python 3.14·torch 2.14.0+cu130)이고 이 PC(3.15.2·cu134)와도 다르다 | 문서의 '그 버전을 배포 기준으로 삼지 말 것' 이 맞다 |

---

## 2. 경로 의존 — 무엇이 어디에 묶여 있나

### 2.1 저장소 루트(`ROOT`) 기준으로 푸는 것

`config.resolve()` 한 함수가 거의 전부를 받는다(`resolve(` 122곳 / 41파일 — 도구 포함). `ROOT` 를 직접 쓰는 곳은 6파일 17곳:

| 곳 | 무엇 |
|---|---|
| `config.py` | `ROOT`·`DEFAULT_CONFIG`(= `ROOT/config.yaml`)·`resolve()`·`pretrained_path()`(**`ROOT/weights/<이름>` 을 직접 찾는다**) |
| `server.py:601` | 화면에 보일 경로를 `ROOT` 기준 상대로 줄인다(표시용) |
| `export.py:130` | 내보낼 폴더가 저장소 안이면 `.gitignore` 를 둔다(저장소 전용 동작) |
| `tools/build_subset_dataset.py`·`its_incident_clips.py`·`scheduled_stop.py` | 도구 — 개발용 |

설정에 **키가 없는** 경로(코드에 박힌 이름):

| 경로 | 쓰는 곳 | 비고 |
|---|---|---|
| `runs/` | `resolve("runs")` 11곳 — 학습 결과·`runs/STOP`(정지 파일)·이어하기 찾기 | 설정 키 없음 |
| `weights/` | `pretrained_path` | 설정 키 없음. **DetectView 가 이 경로를 그대로 부른다**(6절) |
| `export/` | `resolve("export")` | |
| `src/autolabel/web/` | `server.py:71` `WEB_DIR = Path(__file__).parent / "web"` | 패키징하면 번들 리소스(읽기 전용) |
| `labeler.py:7` | `sys.path.insert(0, …/src)` | 진입점 |

### 2.2 DB 폴더 기준으로 두는 것 (`cfg.db.parent / …`, 26곳 / 10파일)

`backups/`(12곳 — 하루 사본·`before_workbatch`·`camrename`·`cameras_*.yaml`·`crop_*`·`camera_moves.jsonl`), `thumbs/`, `capture/`,
`upload/`, `aihub_pm/`, `autocapture/`, 그리고 기본값이 DB 옆인 `classes.txt`·`cameras.yaml`. **이것들은 이미 '워크스페이스 = DB
폴더' 에 가깝다** — DB 폴더를 옮기면 같이 옮겨진다.

### 2.3 DB·파일에 적힌 절대경로

| 어디 | 무엇 | 옮기면 |
|---|---|---|
| DB `images.path` | 이미지 원본의 절대경로(383행) | 전부 '파일 없음' — 지금은 README 5단계의 손 SQL |
| `images.path` 를 직접 읽는 곳 | **115곳 / 24파일** — 패키지 13파일(`store` 15·`tidy` 8·`camera_move` 8·`dupcheck` 7·`server` 7·`ingest` 2·`capture`·`export`·`grouping`·`predictor`·`split`·`trainer`·`upload` 각 1), 나머지는 `tools/` | 상대경로로 바꾸면 이 전부가 대상 |
| 학습 체크포인트(`runs/*/weights/*.pt` 의 args) | 데이터셋 절대경로 — `trainer.uses_dataset` 이 그것으로 '이 설정의 학습인가' 를 가른다 | 옮기면 이어하기가 그 학습을 못 찾는다(README 에 적혀 있다) |
| `export/*/export_info.json` | `"db": str(cfg.db)` — 다른 DB 의 내보내기는 지우지 않는 표식 | 옮기면 '다른 DB' 로 본다(안전한 쪽으로 틀린다) |
| 캡처 세션 `session.json` | 영상 이름·상태(경로는 세션 폴더 기준) | 영향 작음 |

### 2.4 사용자 PC 밖 위치 (Ultralytics·WSL)

- **Ultralytics 는 자기 설정·폰트를 사용자 설정 폴더에 쓰고**(`settings.json`·플롯용 글꼴), AMP 확인 때 `yolo26n.pt` 를
  작업 폴더에서 찾다 없으면 받으려 한다(`wsl/probe_vram.py` 머리말에 겪은 기록). 포터블·오프라인에서 정할 곳 — Ultralytics 쪽
  동작이라 어느 버전에서 어디에 쓰는지는 다시 볼 것.
- WSL 학습은 `~/datasets/<이름>`·`~/new_env/yolo/runs_wsl`(설정 `wsl:`).
- **파일 로그는 없다** — `logging` 을 쓰는 모듈이 없고 서버·작업 로그는 콘솔과 메모리(작업 로그 400줄)뿐이다.

---

## 3. OS 전용 코드

| 곳 | 무엇 | 다른 OS 에서 |
|---|---|---|
| `wsl_train.py` | `wsl.exe`(`shutil.which("wsl")` 없으면 `C:\Windows\System32\wsl.exe`), `creationflags=_NO_WINDOW`, Windows→WSL 경로 변환 `to_linux` | 호출하면 실패. 버튼을 감춰야 한다 |
| `wsl/` | 리눅스 쪽 스크립트(학습·동기화·VRAM·평가) | 이미 리눅스용. 11-28 에 저장소 경로를 스스로 구하게 고쳤다 |
| `stopper.py:71·96` | `sys.platform != "win32"` 일 때만 SIGTERM 처리 | 분기돼 있다 |
| `trainer.py:273` | Windows 면 `cache: ram` 경고(함정 2) | 분기돼 있다 |
| `os.link`(`trainer.py:115`·`camera_rename.py:142`) | 하드링크 — 안 되면 복사로 | 같은 디스크면 어느 OS 든 된다 |
| `open(…, "xb")` 세 곳 | 덮지 않게 만들기 | 어느 OS 든 된다 |
| 썸네일 `os.replace` 의 `PermissionError` 처리 | Windows 에서 다른 요청이 같은 파일을 보내는 중일 때 | 다른 OS 에서는 안 걸린다 |
| `run.bat` | 실행 배치 파일(함정 10 — ASCII 만) | macOS·Linux 용 실행 스크립트가 없다 |

`sys.platform` 을 보는 곳은 셋뿐이다 — OS 분기는 적고, OS 를 타는 것은 거의 WSL 하나다.

---

## 4. 장치

- 설정: `model.device`(기본 `0`). 쓰는 곳: `trainer.py:646·812`(학습), `predictor.py`(자동 라벨링 168·214·222, 예측 비교 332),
  `segmenter.py:301`(SAM), `tiling.py`(받은 값을 넘김).
- GPU 작업(학습·자동 라벨링)이 도는 동안 예측 비교·SAM 은 `"cpu"` 로 간다(`jobs.gpu_busy`).
- `torch.cuda` 를 직접 부르는 곳은 `empty_cache` 두 곳(`predictor.py:321`·`segmenter.py:272`)뿐이다.
- 장치 문자열은 그대로 Ultralytics 에 넘어간다 — `0`·`"cpu"`·`"cuda:0"`·`"mps"` 모두 Ultralytics 가 받는 꼴이다. **CUDA 가 없는
  PC 에서 기본값 `0` 이면 Ultralytics 가 오류를 낸다**(돌려 보지는 않았다 — Ultralytics `select_device` 의 동작으로 본 것).
- `wsl/train_wsl.py` 는 `device=0` 고정(WSL 안. 두 번째 GPU 는 `CUDA_VISIBLE_DEVICES` 로 — 11-27).

---

## 5. DB — 스키마·마이그레이션·백업

- **스키마 버전이 없다** — `PRAGMA user_version` 0, 버전 표도 없다. `Store.__init__` 이 열 때마다 `CREATE TABLE IF NOT EXISTS` 와
  빠진 열의 `ALTER TABLE ADD COLUMN`(images 의 split·grp·dhash·camera·captured_at·domain·work_batch, box_history 의
  masks_before·after, masks 의 poly)으로 맞춘다. 열 하나를 더하는 데는 충분했지만, 경로처럼 **값을 바꾸는** 마이그레이션에는
  버전과 되돌리기가 필요하다.
- 표 11개: images(383) · boxes(4,409) · box_history(18,252) · hist_lists(17,041) · masks(2,008) · mask_templates(5) · tags(9) ·
  image_tags(884) · work_batches(2) · classes(10) (이 날 실제 DB 를 읽기 전용으로 셌다). 색인 8개, 트리거 없음, WAL.
- 백업: `store.copy_db`(SQLite 백업 API 로 임시 이름에 뜬 뒤 바꿔 끼움) — 하루 사본(`backup_daily`, 최근 14개), 작업 배치 열을
  더하기 전(`labeling_before_workbatch_*`), 카메라 키 바꾸기 전(`labeling_camrename_*`). **값을 바꾸는 마이그레이션에 그대로 쓸 수
  있는 부품이 이미 있다.**
- 경로를 다루는 길의 안전장치: 휴지통 비우기·정리·prune 은 한 장마다 상태·경로를 다시 본다(11-27, `purge_row`·`move_row`·
  `delete_missing`). scan·prune 은 '파일 없음' 이 절반을 넘으면 멈춘다. **경로를 바꾸는 마이그레이션이 틀리면 이 '절반' 문턱
  아래로만 틀려도 prune 이 그만큼 행을 라벨째 지운다** — 마이그레이션 중·직후에는 prune 을 막아야 한다.

---

## 6. 이 저장소에만 있는 제약 (인수인계 문서에 없는 것)

- **DetectView(`E:\DetectView`, 검출 앱)가 이 저장소를 부른다** — `wsl/train_wsl.py` 의 인자·`<project>/STOP`·`RESULT_DIR` 줄,
  WSL `yolo_final` 환경, `weights/yolo26s.pt`·`yolo26m.pt`. 문서의 제안대로 `weights/` 를 UserData 로 옮기거나 `wsl/` 을
  `integrations/` 로 옮기면 DetectView 가 깨진다 — 옮긴다면 옛 자리를 남기거나 DetectView 쪽과 함께 바꾼다.
- **배포판에 데이터가 섞이면 안 된다** — `data/`(DB·이미지), `cameras.yaml`(실제 설치 장소), AI Hub 그림(약관상 제3자 제공 금지 —
  docs/provenance.md), `runs/`·`models/`(이 데이터로 학습한 가중치). 빌드가 `src/`·`web/` 만 담는다는 것을 검사로 지켜야 한다.
- **라이선스** — 코드가 Ultralytics(AGPL-3.0)를 import 해서 저장소가 AGPL 이다. 배포판도 AGPL 조건(소스 제공)을 따라야 하고,
  YOLO26 가중치·SAM 2.1 가중치의 조건은 따로 확인할 것(확인하지 않았다).
- **개인 작업 흐름** — 자동 커밋 훅·Gemini 칠판·Runpod·WSL·`tools/` 의 데이터별 도구는 개발 모드에 남기고 배포판에서 뺀다.
- **`.venv` 를 다른 프로젝트가 빌려 쓴다** — 패키지 구성을 바꾸는 실험은 따로 만든 venv 에서 한다.

---

## 7. 계획 — 순서·위험·호환·되돌리기

원칙: 한 단계 = 한 커밋 묶음, **기존 Windows 개발 모드는 매 단계 그대로 돈다**(설정을 바꾸지 않으면 동작이 같다), 단계마다 회귀
테스트를 더하고 전체를 돌린다.

| # | 단계 | 바꾸는 것 | 위험 | 호환 | 되돌리기 |
|---|---|---|---|---|---|
| 1 | **경로 한 곳으로** | `AppPaths`(설치·리소스·사용자 데이터·워크스페이스 자리)를 `config.py` 옆에 두고, `resolve`·`pretrained_path`·`resolve("runs")`·`WEB_DIR` 이 그것을 거치게. 개발 모드에서는 지금과 같은 값(저장소 루트) | 낮음 — 값이 같다 | `ROOT` 는 그대로 남겨 두고 호출부만 옮긴다 | 커밋 되돌리기 |
| 2 | **WSL 을 선택 기능으로** | `server.py` 의 `wsl_train` import 를 처리 함수 안으로, `wsl_train.available()`(Windows 이고 wsl.exe 가 있을 때)로 버튼을 감춤 | 낮음 | 이 PC 동작 그대로. DetectView 인터페이스(`train_wsl.py`)는 건드리지 않는다 | 커밋 되돌리기 |
| 3 | **장치 `auto`** | `model.device: auto` 를 받아 CUDA → MPS → CPU 순으로 고르는 한 함수, 고른 결과를 화면·로그에 표시. **이 저장소의 config.yaml 은 `0` 그대로** | 낮음 | 숫자·문자열 장치는 지금처럼 그대로 넘긴다 | 설정을 `0` 으로 |
| 4 | **데이터셋 만들기 순서** | 검사 → 임시 폴더에 만들기 → 바꿔 끼우기 | 낮음 | 결과 폴더가 같다 | 커밋 되돌리기 |
| 5 | **스키마 버전** | `PRAGMA user_version` 으로 지금 스키마를 1 로 적기(값을 바꾸지 않는 첫 단계), 그 뒤 마이그레이션은 버전으로 가른다 | 낮음 | 옛 코드는 user_version 을 보지 않는다 | 0 으로 되돌리면 같다 |
| 6 | **이미지 경로를 워크스페이스 기준으로** | 경로 읽기를 `store` 한 함수로 모은 뒤(115곳 → 한 곳), 새 열(워크스페이스 기준 상대경로)을 더하고, 사본 DB 에서 바꿔 보고 검증, 실제 DB 는 사본을 뜬 뒤 · 옛 절대경로 열은 지우지 않고 남긴다 | **높음** — 틀리면 prune 이 행을 라벨째 지운다(5절) | 상대경로가 없으면 절대경로를 쓴다 | 사본 DB 로 바꿔 끼우기, 옛 열이 남아 있다 |
| 7 | **의존성 나누기** | `pyproject.toml` + 플랫폼별 잠금(CPU·CUDA·macOS) — 이 PC·노트북의 지금 조합은 개발용 잠금으로 | 중간 — `.venv` 를 다른 프로젝트가 쓴다 | 새 venv 에서만 시험 | 파일만 더하는 단계 |
| 8 | **macOS·Linux 에서 테스트** | GitHub Actions 로 세 OS 에서 CPU 테스트(GPU·모델 안 쓰는 지금 테스트가 그대로 대상) | 낮음 | — | — |

**진행 (같은 날 사용자 '1단계 시작'):** 1단계를 했다 — `src/autolabel/paths.py`(`AppPaths`·`APP`), `config.ROOT`·`DEFAULT_CONFIG`·
`resolve`·`pretrained_path` 는 `APP.user_data_dir`, `server.WEB_DIR` 은 `APP.resource_dir`, 내보내기의 `.gitignore` 는
`APP.source_dir` 기준. 개발 모드 값은 그대로다(사본 서버를 켜서 화면·`/api/state` 확인, 테스트 717개). `ROOT` 는 이름을 그대로 두었다.
배포판 쪽 규칙(포터블 표식·OS 기본 자리·번들 리소스 자리)은 패키징 전이라 규칙만 테스트했다. `wsl_train.REPO` 는 2단계에서.

**2단계도 했다 (같은 날 사용자 '2단계 시작'):** `server.py` 의 `wsl_train` import 를 WSL 요청 안으로, `wsl_train.supported()`
(Windows·WSL 스크립트 — 서브프로세스 없이), Windows 가 아니면 `/api/wsl` 이 `supported: false` 를 주고 화면은 버튼을 감추며
작업 요청은 GPU 를 잡기 전에 400. `wsl_train.REPO` 대신 `SCRIPTS`(`paths.APP`). 이 PC 에서는 사본 서버로 WSL 버튼이 그대로
보이고 켜지는 것을 봤다(`supported`·`available` 참). 다른 OS 는 `sys.platform` 을 갈아 흉내 냈을 뿐이다(8단계에서 실제로).
`wsl/train_wsl.py`(DetectView 가 부른다)는 건드리지 않았다.

**3단계도 했다 (같은 날 사용자 '3단계 시작'):** `src/autolabel/devices.py`(`resolve_device`·`auto_device` — CUDA → MPS → CPU),
`Config.device` 가 그것을 거치고 **키가 없으면 `auto`**(그 전에는 `0`). 사람이 적은 장치는 그대로 넘긴다. 이 저장소의
`config.yaml` 은 `0` 그대로라 이 PC 의 동작은 같다(주석에 `auto`·`mps` 를 적었다). torch 는 장치를 쓸 때만 불러 온다.
화면에 고른 장치를 보여 주는 것은 하지 않았다(`/api/state` 에서 고르면 첫 화면이 torch 를 불러 오느라 느려진다 — Ultralytics 가
학습·추론 로그에 장치를 찍는다). MPS 에서 SAM·타일 추론이 되는지는 macOS 에서 볼 것.

**4단계도 했다 (같은 날 사용자 '4단계 시작'):** `trainer.build_dataset` 이 클래스·승인 라벨을 먼저 보고, `<dataset>.building` 에
만든 뒤 옛 세트를 `.old` 로 비켜 두고 이름을 바꿔 끼운다(실패하면 되돌린다). 옛 세트의 학습 캐시(`.npy`·`.cache`)만 먼저 지워 디스크를
두 벌만큼 쓰지 않게 했다. DB 사본으로 실제로 두 번 만들어 봤다(승인 383장 → 파일 811개, 14초·12초, 옆 폴더 안 남음).

**5단계도 했다 (같은 날 사용자 '5단계 시작'):** `store.SCHEMA_VERSION = 1` — 새 코드가 `Store` 를 열면 열 맞추기를 마친 뒤
`PRAGMA user_version` 에 1 을 적는다(값은 그대로 — DB 사본에서 0 → 1, 이미지 383·박스 4,409·이력 18,252 그대로). 더 새 번호의 DB 는
쓰기 전에 거절한다. `schema_version()` 은 읽기 전용. **실제 DB 는 새 코드가 처음 열 때(검수 서버를 다시 켤 때 등) 1 이 된다** —
이 조사 때는 0 이었다. 옛 코드는 이 번호를 보지 않으니 노트북 등 옛 코드와 섞여도 지장이 없다.

**6단계도 했다 (같은 날 사용자 '6단계 시작') — 표의 방법과 다르게 했다.** 표는 '경로 읽기 115곳을 한 함수로 모으고 새 열을
읽게' 였는데, 그러면 읽는 곳(도구의 SQL 포함)을 전부 고쳐야 하고 하나라도 빠뜨리면 옮긴 PC 에서만 조용히 틀린다. 대신 **path 는
절대경로 그대로 두고 읽는 곳은 하나도 안 바꿨다** — 옆에 `images.rel`(DB 폴더 기준 상대경로)을 더해 path 를 쓰는 길 7곳이 함께
적고, `Store` 가 열 때 '`path` 의 파일이 없고 `DB 폴더/rel` 에 파일이 있는 행' 만 path 를 그 자리로 바꾼다(그 전에 사본). 그래서
이 PC 에서는 값이 바뀌지 않고(rel 만 채운다), 옮긴 PC 에서는 열자마자 path 가 맞아 백여 곳이 그대로 맞는 값을 읽는다. 표의 약속
셋(상대경로가 없으면 절대경로 · 옛 절대경로 열이 남는다 · 사본)은 그대로다. 실측(DB 사본과 그림 사본 263MB): 사본을 A 에 두고
열면 0 → 2·rel 383개(0.16초), 폴더째 B 로 옮기면 '파일 없음' 383 → 열자 383개를 고쳐 0(0.24초), 행·박스·이력·updated_at 그대로,
`check` 이상 없음(경로 표기 0), 다시 열기 0.018초, DB 파일만 C 로 옮기면 아무것도 안 바꿈. **실제 DB 는 아직 0 이다** — 검수 서버를
새 코드로 다시 켤 때(또는 새 코드의 명령·도구가 열 때) `labeling_before_relpath_<시각>.db` 를 뜨고 2 가 된다(383장 모두 DB 폴더 아래라
경로는 그대로). 남은 것: rel 이 없는 행(그 전 코드가 넣고 새 코드가 안 연 DB)은 README 5단계의 손 SQL · 학습 체크포인트의 데이터셋
절대경로(`trainer.uses_dataset` — 2.3절)는 이 단계 밖 · 두 PC 가 같은 DB 를 동기화 폴더로 함께 쓰면 열 때마다 서로의 자리로 고친다
(그때마다 사본 — 그렇게 쓰지 말 것).

**7단계도 했다 (같은 날 사용자 '7단계 시작'):** `pyproject.toml`(범위 — 아래쪽은 노트북 requirements.txt 의 판, Ultralytics 만 위쪽을
함정 8 을 확인한 8.4.212 로 막음, `requires-python >=3.14`, 묶음 test·claude·roadplus, `labeler` 명령, 편집 설치만)과 `locks/`(uv 로 고른 잠금
넷 — 데스크탑 개발 `win-cu134-py315`(이 `.venv` 의 판 그대로 58개) · 배포 `win-cpu`·`linux-cpu`·`macos-arm64`(모두 py314, 개발 잠금과
같은 판), 다시 만드는 `locks/compile.py`). `requirements.txt` 는 노트북 잠금으로 그대로 뒀다(주석만). **`.venv` 에는 아무것도 깔지
않았다** — 새 가상 환경(Python 3.14.10)에 `win-cpu-py314` 와 `pip install -e . --no-deps` 만 깔아 `pip check` 깨끗 · `labeler --help` ·
회귀 테스트 801개 통과(건너뜀 0). Linux·macOS 잠금은 고르기만 했다(8단계). 배포 Python 3.14 는 잠정(8절 2번). macOS 는 torch 2.15.0 휠이
macOS 14 부터라 그 아래는 안 된다. `tests/test_deps.py` 가 import·범위·잠금을 파일만으로 견준다. 해시 고정은 아직이다.

**8단계도 했다 (같은 날 사용자 '8단계 시작') — 워크플로까지, 아직 GitHub 에서 돌지는 않았다(push 해야 돈다).**
`.github/workflows/tests.yml`: push(main)·PR·손 실행에 windows/ubuntu/macos-latest 세 러너가 Python 3.14 로 그 OS 의 배포용 잠금을 새
venv 에 깔고 `pip install -e . --no-deps` → pytest. 건너뛴 테스트가 있으면 실패, 읽기 권한만, 액션은 커밋 해시(checkout v7.0.1 ·
setup-python v7.1.0), Windows 러너의 `core.autocrlf` 를 체크아웃 전에 끈다, Linux 는 OpenCV 용 libgl1·glib. 공개 저장소라 러너는 무료.
**push 전에 Linux 를 WSL 로 미리 돌렸다** — Ubuntu 의 Python 3.14.4 로 새 venv(ensurepip 이 없어 pip 만 `.venv` 에서 옮겨 깔았다)에
`linux-cpu-py314` → 처음 703 통과·3 실패, 셋 다 테스트가 Windows 를 가정한 것이라 테스트만 고쳐 706 통과(화면 JS 101개는 node 가 없어
건너뜀 — CI 에는 node 가 있다). Windows `.venv` 807개 통과. `tests/test_ci_workflow.py` 가 워크플로를 파일만으로 본다. **같은 날 사용자 'push 하기' 로 올려 첫 실행이 세 OS
모두 통과했다**(run 46139301829, 커밋 4e3139a — Python 3.14.10 · torch 2.15.0(+cpu) · Ultralytics 8.4.212, 각 807개 통과·건너뜀 0:
ubuntu 51초 · macos 61초 · windows 145초, node 24·24·22). **macOS 러너의 GPU 는 가상 장치라 MPS 를 못 쓴다**(torch 가 'Apple Paravirtual
device … does not support Metal 2.0' 경고 — `auto` 가 CPU 를 골랐다) — MPS 로 추론·학습이 되는지는 실제 Mac 에서만 알 수 있다. **남은 것**:
MPS 실기 확인 · 다른 OS 에서 서버를 띄워 화면을 본 것은 아니다 · 패키징(PyInstaller)은 이 다음이다(인수인계 문서 14절).

6번이 유일하게 위험한 단계였다. 1–5번은 동작을 바꾸지 않거나 설정으로 되돌릴 수 있고, 6번은 그 위에서 사본으로 먼저 했다.
패키징(PyInstaller)은 8번까지 소스 모드로 세 OS 가 도는 것을 본 뒤다(인수인계 문서 14절 8번과 같다).

---

## 8. 사용자가 정할 것

1. 이 일과 지금 하던 일(데이터 수집·첫 분할·학습)의 순서.
2. 배포 대상 Python 버전 — 지금 두 PC 가 3.14·3.15 로 다르다.
3. macOS·Linux 실기기가 있는지(없으면 GitHub Actions 로 CPU 만 본다 — 그곳에서 MPS 가 되는지는 확인하지 않았다).
4. 개인 작업 흐름(WSL·Runpod·Gemini·자동 커밋)을 이 저장소에 개발 모드로 둘지, 제품과 저장소를 나눌지.
5. DetectView 가 부르는 경로(`wsl/train_wsl.py`·`weights/`)를 그대로 둘지, 함께 옮길지.
