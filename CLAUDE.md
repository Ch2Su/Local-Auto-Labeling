# 로컬 자동 이미지 라벨링 — 작업 인수인계

소량의 손라벨 데이터로 YOLO를 학습시키고 → 나머지를 자동 라벨링 → 웹에서 검수 →
검수분을 합쳐 재학습하는 루프. 전부 로컬에서 돈다.

**방향 (2026-09-18 사용자 확인): CCTV 중심이 아니라, 어떤 이미지든 들어 있는 객체를 다
탐지하는 쪽으로 바뀌었다.** 드론 부감·스크린샷·영화 장면 같은 비CCTV 이미지도 운영 대상이다.
그래서 분할·평가에서 특정 도메인이 한쪽(train/val/test)에 몰리지 않게 할 것.

사용자 문서는 [README.md](README.md). 이 파일은 **작업자가 알아야 할 것**만 담는다.

**Gemini 와 함께 쓴다 (2026-10-25 사용자 결정).** 코드·설정·DB 수정은 **Claude**, 코드 리뷰·읽기 전용
실측 검증·아이디어는 **Gemini** 가 맡는다. 주고받는 칠판은 `docs/collab.md` — **git 에 담지 않는 로컬
파일**(`.gitignore`)이고 150줄 안쪽으로 유지한다. 턴을 시작할 때 그 파일의 `[1. 현재 작업 지시]` 를 읽고,
끝나면 `[2. Claude 작업 보고]` 에 시작 시점 `HEAD`·수정 파일·내용을 적어 상태를 `WAITING_VERIFY` 로 둔다.
오래 남길 규칙·실험 기록은 칠판이 아니라 이 파일과 docs/experiments.md 에 적는다.

---

## 환경 (이미 구성됨)

| | |
|---|---|
| Python | 3.14.3 (`.venv`) |
| PyTorch | 2.14.0+cu130 |
| Ultralytics | 8.4.142 |
| GPU | RTX 5060 Laptop 8GB, sm_120 (Blackwell) |
| OS | Windows 11 |

`python`은 반드시 `.venv\Scripts\python.exe`를 쓴다. 시스템 Python에는 아무것도 없다.

**블루스크린·예기치 않은 종료가 있었다 — 원인은 모른다.** 09-12 `0x133 DPC_WATCHDOG_VIOLATION`
(Windows 에서 Ultralytics `val` 을 돌리던 무렵, 덤프 `C:\WINDOWS\Minidump\091226-19796-01.dmp` 는
WinDbg 가 없어 분석 안 함), 09-16 버그체크 0 으로 꺼짐(WSL 학습을 정상 중단한 뒤). 같은 검증·
학습을 그 뒤 여러 번 돌렸을 때는 문제없었다.

Python 3.14 + sm_120 조합이라 **PyTorch는 cu130 인덱스에서만** 받을 수 있다.
재설치가 필요하면 (버전은 `requirements.txt` 와 같게 — 10-31 에 고정했다. 빼면 인덱스 최신(그날 2.14.1)이
깔리고, 뒤이은 `pip install -r requirements.txt` 가 `==2.14.0` 에 맞추려고 PyPI 의 CPU 빌드로 바꾼다 — dry-run 확인):

```bash
.venv/Scripts/python.exe -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
```

**Runpod (클라우드 GPU 대여) — MCP 서버 `runpod` 로 붙어 있다 (2026-09-18).** 키는 Windows 사용자
환경 변수 `RUNPOD_API_KEY` 에 사용자가 넣었다 — **키 값을 읽거나 출력하지 말 것.** 공식 플러그인의
OAuth 는 이 데스크톱 앱에서 안 돼서(redirect_uri 가 claude.ai 라 거절) API 키 방식이다. MCP 는
pod·볼륨 관리와 가격 조회만 한다. 파일 전송·SSH 는 Windows 기본 `ssh`·`scp`·`tar` 로 한다
(`~/.ssh/id_ed25519` 공개키 등록됨).
- **pod 를 만들기 전에 GPU·예상 시간·비용을 사용자에게 말하고 허락받을 것.** 초 단위 과금이고
  멈춘 pod 도 볼륨 요금($0.20/GB/월)이 나간다. **멈춘 pod 는 다시 안 켜질 수 있다**(09-20 '남는 GPU
  없음') — 결과는 바로 받고 다 쓰면 지운다. pod 는 스스로 못 멈추므로(pod 안의 키는 Unauthorized)
  대화에서 `pod-action stop`.
- **지금 pod·볼륨은 없다** (09-24 확인). A40 pod(09-18~20, 약 $4.2)는 지웠고 결과(best.pt 3개·
  results.csv·로그)는 `runs/compare_a40/` 에 있다. resume.pt 는 pod 와 함께 사라졌다 — 300 epoch 을
  돌리려면 처음부터.
- **GPU 를 고른다면 RTX 6000 Ada.** x960 b16 한 스텝: A40 869ms / Ada 513ms($0.84/시간) / L40S
  582ms / A100 423ms — Ada 가 A40 과 총비용은 같고 시간은 60%. 표와 재는 법은 docs/experiments.md 끝
  (**RAM 캐시로 잴 것** — 아니면 디코딩을 잰다).
- **A40 비교 결과** (300 스케줄의 60 epoch, 7클래스 test 전체): s640 0.6515 / l960 0.7264 / x960
  0.7243 — 960 + 큰 모델이 같은 조건 대조군보다 +0.07. 설정끼리만 비교할 것(60 epoch 이라 절대값은
  낮다). 표와 해석은 docs/experiments.md.
- **새 pod 절차는 `runs/runpod/`** (묶음 tar 만들기 → scp → `pod_probe.sh` → `pod_compare.sh`).
  **묶음을 만든 `make_bundle.py` 는 저장소 어디에도 없다**(10-23 확인 — 세션 scratchpad 에 있었던 것으로
  보인다). 다시 쓰려면 `data/visible/dataset` 을 DB 로 만들고 tar 로 묶는 스크립트를 새로 짤 것.
  09-20 에 만든 묶음 tar 3개(4.5GB)는 라벨이 바뀌어 낡았기에 10-24 에 지웠다(사용자 확인, 휴지통으로).
  학습 전에 **`pip install --break-system-packages -U pillow` 와 `labels/*.cache` 삭제** — 이미지의
  Pillow 11.0.0 은 확장자만 .png 인 삼성 폰 JPEG 29장을 '손상'으로 빼고(EXIF 쓰다 ZeroDivisionError),
  AVIF 도 못 읽는다. `/workspace` 는 네트워크 FS 라 데이터셋은 `/root` 에 풀고, 컨테이너 RAM 한도
  (A40 46.6GiB)에 ram 캐시·워커가 겹치면 SIGKILL 된다(1280 시도가 그렇게 죽었다).
- SSH 로 `a && b && setsid nohup ... &` 처럼 목록째 띄우면 ssh 가 안 돌아온다(학습은 산다).
- 로컬에서 x·l 960 을 재면 VRAM 이 넘쳐 느리다 — 사용자가 GPU 를 쓰는 중이면 먼저 물을 것.

**Roboflow MCP — claude.ai 커넥터로 붙어 있다 (2026-09-28).** 도구 이름은 `mcp__<커넥터 id>__projects_list`
꼴(프로젝트·이미지·라벨·버전 내보내기·학습·Universe 검색·휴지통 등). OAuth 라 키가 없다.
- **워크스페이스는 비어 있다** — 스토리지만으로 크레딧이 나가 사용자가 09-28 에 데이터를 전부 지웠다.
  `projects_list` 0개. 휴지통에 남아 있던 'My First Project'(object-detection, 09-28 삭제)도 **10-02 에
  사용자가 직접 비웠다**(`trash_list` 0개로 확인). MCP 에는 휴지통을 비우는 도구가 없다(목록·복원뿐).
- **그래서 로컬이 유일한 사본이다.** 외부 백업(09-11)은 같은 물리 디스크(E: = 디스크 0)에 있어서,
  09-28 에 **다른 디스크(C: = 디스크 1)로 한 벌 떴다** — 아래 '외부 백업'. 10-31 에 보니 그 C: 사본이 없어져서
  **같은 날 새로 떴다**(`C:\local_auto_labeling_backup\20261031\`).
- 로보플로에 무언가를 올리거나 프로젝트를 만들고 복원하면 **스토리지 크레딧이 다시 나간다** — 먼저 물을 것.
  Universe 검색(`universe_search`·`universe_dataset_images_search`)은 조회만 한다(크레딧이 드는지는
  확인하지 않았다). Universe 데이터를 받을 때도 **COCO 로** 받을 것(함정 4).
- **남의 데이터셋을 넣을 때 (10-02 조사).** 시드(COCO) 경로는 라벨을 **검수 없이 곧바로 approved** 로
  넣고, 이름이 config 의 `classes` 와 다른 카테고리(`pedestrian`·`SUV`·`Pickup Truck` 등)는 리포트에만
  남기고 **버린다**(`coco_io.read_coco`). 미라벨 폴더로 넣으면 라벨을 아예 읽지 않는다. 그래서 카테고리
  이름을 우리 9클래스로 먼저 바꾸고, 우리 클래스를 **빠짐없이** 라벨한 데이터인지 확인할 것 — 사람이
  라벨 안 된 이미지가 승인으로 들어가면 모델이 그 사람을 배경으로 배운다. Universe 의 차종 데이터셋은
  대부분 사람을 라벨하지 않았고, 한 장에 박스 하나뿐인 것도 많다. Objects365 의 `Scooter` 는 오토바이형
  스쿠터이고 이 프로젝트의 scooter 는 전동 킥보드다(`Hoverboard` 는 세그웨이류).
- **Objects365 v2 train 을 BAAI 에서 직접 받아 풀었다 (10-02~03).** `E:\datasets\objects365\` — 저장소
  밖이다. 라벨 `zhiyuan_objv2_train.json`(5.14GB), 압축 `patchN.tar.gz`(v1 p0\~15 약 3.4GB, v2 p16\~50 약
  7.7GB, 안에 `patchN/` 접두사)를 `train\patchN\` 으로 풀었다. json 은 51묶음 **1,742,292장**, 디스크에는
  **49묶음 1,686,812장** — p44·p45(55,477장)는 받지 않았고, 압축 자체에 3장이 없고(p6 `…00320532`·
  `…00320534`, p16 `…00908726`) p4 에 0바이트 쓰레기 `objects365_v1_00215749.jpg.ddEA04F9` 가 들어 있다.
  나머지는 파일 이름까지 json 과 같다(tar 반환값 0).
  - json 은 **뒤쪽 주석의 필드 순서가 다르다**(`image_id` 가 앞) — 정규식으로 흘려 읽을 때 객체째 잘라
    `json.loads` 할 것. 이미지 항목 끝에 `"url": ""` 이 있다. 주석에 `iscrowd`·`isfake`(포스터·장난감)·
    `isreflected`(반사) 플래그가 있다. 우리 클래스 관련 박스만 뽑은 색인이 `o365_index.pkl` 이다
    (images: id→(patch, 이름, w, h), boxes: id→[(cat, x, y, w, h, crowd, fake, refl)]).
  - 라벨 라이선스는 license.txt 에 CC BY 4.0(링크는 by-sa), **이미지는 Flickr CC 로 NonCommercial 이 섞여
    있다**(json `licenses`).
  - **O365 의 Van 은 우리 van 과 다르다** — 무작위 크롭 48장에 SUV(볼보 XC60)·시에나·캐노피 픽업·버스
    뒷면이 섞여 있었다. Pickup Truck 은 대부분 미국식 픽업이라 우리 1a 와 맞지만 세단·SUV 오표기가 있다.
    그대로 승인으로 넣지 말고 검수를 거칠 것.
  - **가져오기는 `tools/import_o365.py`** (10-03). van·pickup 이 든 사진을 반씩 골라 `unlabeled/o365/` 로
    복사하고 **검수 대기**(source·origin `o365`)로 넣는다. 기본은 보기만, `--apply` 때 DB 를 먼저 떠 둔다.
    사용자 결정: Scooter → motorcycle, isfake·isreflected 박스는 버린다. 우리 클래스에 iscrowd 가 있는
    사진(후보의 절반)과 Ambulance 가 있는 사진은 뺀다. **넣은 뒤 `labeler.py label --all` 을 쓰지 말 것** —
    검수 대기 라벨을 모델 예측으로 덮어쓴다(웹 UI 의 자동 라벨링 버튼은 `include_pending: false` 라 안전하다).
  - **10-03 에 500장(seed 0)을 넣었다** — id 6726~7225, 박스 3,292개(car 1,252·person 1,107·van 371·
    pickup 350 …), 전부 검수 대기. 넣기 전 DB: `data/visible/labeling.db.bak_o365_20261003_024915`, 목록:
    `runs/o365_import/20261003_024915.tsv`. 한 장 검수에 걸리는 시간을 재 보고 더 넣을지 정하기로 했다.
  - **`tools/test_baseline.py` 의 all·nodrone 은 10-03 부터 source='o365' 를 뺀다**(이름표는 그대로라
    --history 가 이어진다). Objects365 만은 `o365`, 전부는 `all_o365` 도메인.
  - **`objects365_v1_00420917` 같은 이름은 연번 계열로 묶지 않는다** (`grouping._INDEPENDENT`, 10-03) —
    그대로면 O365 사진 전부가 묶음 하나가 되어 한 분할에 갇힌다.

**git — 2026-10-02 에 저장소로 만들었다 (사용자 허락, 브랜치 `main`).** 담는 것은 코드·설정·
문서와 `runs/` 의 텍스트 기록(results.csv·args.yaml·`test_baseline/history.jsonl`·스크립트)뿐이다.
`data/`·`models/`·`weights/`·`export/`·`.venv`·`graft/`·그림·로그·tar 는 `.gitignore` 로 뺐다 —
**데이터와 DB 는 git 이 아니라 백업으로 지킨다**(아래 '외부 백업'). 저장소 설정 `core.autocrlf=false`
(파일을 바이트 그대로 둔다 — CR 이 섞여 겪은 함정 6 때문). 사용자 전역 설정의 기본 브랜치는
`KNUT_APP` 이지만 이 저장소는 `main` 으로 만들었다.
- **턴이 끝날 때마다 자동으로 커밋된다** (사용자 요청, 10-02). `.claude/settings.json` 의 Stop 훅이
  `.claude/hooks/autocommit.sh` 를 부른다 — 바뀐 게 있으면 `git add -A` 후 '자동 커밋: <파일 이름들>'
  으로 커밋한다. 5MB 넘는 파일은 담지 않고 알린다. 사용자가 직접 고친 파일도 같이 담긴다. 끄려면
  settings.json 에서 그 훅을 지운다. 훅이 추적 파일이라 **이 저장소를 클론한 다른 PC 에서도 돈다.**
  - 머지·리베이스·cherry-pick·revert 중이거나, HEAD 가 브랜치에 있지 않거나, **충돌이 남아 있으면** 알리고
    건너뛴다(10-31). stash pop 충돌은 MERGE_HEAD 를 남기지 않아서 옛 훅은 충돌 표시째 커밋했다(시험 저장소에서
    재현 — 실제 기록에 그런 커밋은 없다). 색인에서만 풀린 충돌(`git restore --staged`·`git reset` 뒤)도 파일에 남은
    표시로 거른다. `git add`·`git commit` 이 실패하면(다른 프로그램이 파일을 잡고 있음, 새 PC 에 `user.name`·
    `user.email` 이 없음 등) 알린다.
- 의미 있는 단위로 남기고 싶으면 턴이 끝나기 전에 직접 메시지를 붙여 커밋한다 — 그러면 훅은 할 일이 없다.
- 코드 리뷰·수정 범위는 `git diff HEAD~N` 이나 `git log` 로 잡는다(작업 트리는 턴마다 비워진다).
- **원격: GitHub `chisu0208/Local-Auto-Labeling` (private, 2026-10-31 연결, 사용자 허락).** 첫 push 때 GitHub 이
  만든 'Initial commit'(README 한 줄)을 `--force-with-lease` 로 덮어써서, 로컬 이력이 해시 그대로 올라갔다.
  - **훅은 push 하지 않는다 — push 는 사용자가 요청할 때만.** 올린 커밋은 다시 쓰지 말 것(amend·rebase 뒤
    force push 금지 — Gemini 칠판이 커밋을 해시로 가리킨다).
  - **GitHub 은 개인 private 저장소에서 비밀을 걸러 주지 않는다**(push protection 은 public 저장소나 조직의 유료
    플랜 몫이고, 개인 Pro 로는 안 된다). 그래서 `.gitignore` 맨 끝에 비밀 파일 이름(`.env`·`*.pem`·`id_ed25519*`·
    `credentials.*`·`.mcp.json`·`.gemini/settings.json` 등)과 runs/ 밖의 DB 사본·영상·로그·압축을 막아 두었다
    (10-31). `runs/runpod/` 의 json(pod 조회 결과)도 뺀다. 키는 환경 변수에만 둔다. 새 비밀 패턴은 그 블록에 넣을
    것 — runs/ 화이트리스트보다 **뒤에** 있어야 runs/ 아래에서도 먹는다. `.claude/worktrees/` 도 무시한다(담으면
    워크트리가 gitlink 로 커밋된다).
  - **줄끝: `.gitattributes` 가 `*.sh` 만 LF 로 고정한다**(10-31). 그 전에는 Windows 기본값(core.autocrlf=true)
    으로 클론하면 .sh 가 CRLF 로 풀려 WSL bash 가 `$'do\r'` 문법 오류를 냈다(재현함). 나머지 파일은 CRLF(32개)와
    LF 가 섞인 채 바이트 그대로 둔다 — 전체 정규화는 하지 않았다. **그 대신 작업 트리의 .sh 를 CRLF 로 저장하면
    git 에는 보이지 않는다**(status 가 깨끗하고 `git checkout --` 으로도 안 고쳐진다). WSL 은 이 작업 트리를 직접
    실행하므로, WSL 에서 `$'\r'` 오류가 나면 `git ls-files --eol -- '*.sh'` 의 `w/crlf` 로 찾아
    `sed -i 's/\r$//' <파일>` 로 CR 을 지운다.
  - **GitHub 은 한 문단(표는 한 칸) 안의 `~` 두 개를 취소선으로 그린다**(10-31 에 이 파일·docs/experiments.md 의
    6곳을 `\~` 로 고쳤다). 범위를 `~` 로 적을 때 같은 문단에 `~` 가 또 있으면 `\~` 로 쓸 것 — 인라인 코드와 코드
    블록 안은 괜찮다. 확인은 그 파일을 `gh api markdown`(mode `markdown`)으로 그려 `<del>` 이 없는지 본다.

**코드는 graft 로 먼저 찾는다 (2026-09-20 도입).** `graft/` 에 이 저장소의 코드 그래프가 있다
(29개 파일·385 심볼·1,164 간선: `src/autolabel/`·`tools/`·`wsl/`·`labeler.py`·`app.js`).
파일을 통째로 읽기 전에 여기서 위치와 호출 관계를 본다 — 전부 로컬·무료·1초 안팎이다.
`graft` 도 `node` 도 PATH 에 없어서 전체 경로로 부른다:

```bash
PATH="/c/Program Files/nodejs:$PATH" node "C:/Users/chisu/AppData/Roaming/npm/node_modules/@nanonets/graft/dist/cli.js" <명령>
#   ask "<질문>" --source | grep "<심볼>" | skeleton <파일> | callers <심볼> --depth 2 | map
```

- **편집 뒤 `build` 를 다시 돌릴 필요가 없다** — 조회할 때마다 바뀐 파일만 다시 읽는다(실측:
  임시 파일을 만들자 바로 30개 파일로 잡혔고, 지우자 29개로 돌아왔다. 바뀐 게 있으면 6~7초).
- **`graft/` 아래 마크다운 카드는 턴 끝에야 갱신된다** — 직접 grep 하지 말고 명령으로 물을 것.
- **`callers` 는 이름이 겹치면 호출처를 빠뜨린다**(`set_classes` 에서 `_ingest_coco` 를 놓쳤다).
  빠짐없이 봐야 하면 `graft grep`.
- **인덱스에 없는 것**: `config.yaml`, `CLAUDE.md`·`docs/`, `index.html`·`style.css`,
  `runs/runpod/*.sh`. 이것들은 평소처럼 읽고 grep 한다.
- `graft/` 는 재생성 가능한 로컬 캐시라 이날 만든 `.gitignore` 에 들어 있다.

## 프로젝트(데이터셋) 탭 — 2026-09-21 추가

**지금 프로젝트는 가시광 하나다 — 열화상·화재 감지는 2026-09-28 에 뺐다** (사용자 결정: 쓰지 않을
것 같다). `config.thermal.yaml`·`config.fire.yaml`·`data/thermal`(빈 DB)·`data/fire`(로보플로 시드
358장·박스 531개, UI 에서 고친 적 없음)를 **외부 백업 없이 지웠다.** 지울 때는 로보플로에 원본이 있다고
여겼는데, **같은 날 사용자가 로보플로 데이터를 전부 지웠다고 알려 왔다**(아래 'Roboflow MCP'). 로보플로
휴지통의 'My First Project' 가 화재 시드였는지는 열어 보지 않은 채 10-02 에 사용자가 휴지통을 비웠다 —
**화재 데이터는 이제 어디에도 없다.** 학습한 적이 없어 가중치·runs·WSL 폴더는 원래 없었다.
탭 구조(코드)는 그대로 두었다. `projects.yaml` 에 항목이 하나면 탭 줄을 그리지 않는다(`app.js
loadProjects`). 목록은 `projects.yaml` 이고, 프로젝트 하나 = config 파일 하나다 (데이터 폴더·DB·
가중치·`runs` 이름이 전부 config 로 갈린다).

```
labeler.py review            -> projects.yaml 의 프로젝트를 모두 띄운다 (기본)
   /             -> 첫 프로젝트로 보냄       /api/projects -> 탭 목록
   /p/visible/   (/p/<이름>/ ...)          각 프로젝트의 UI·API
labeler.py --config <설정> review  -> 그 설정 하나만 (예전과 같음, 탭 없음)
```

- **서버 안에서 앱을 프로젝트마다 따로 만들어 `/p/<이름>` 에 붙인다** (`server.create_multi_app`).
  Store·작업큐·SAM 이 앱마다 따로라 섞이지 않는다. **'현재 프로젝트' 를 서버에 하나 두고
  바꾸는 방식은 쓰지 말 것** — 검수 UI 는 박스를 0.6초 뒤 자동 저장하므로, 브라우저 탭을
  두 개 열면 엉뚱한 프로젝트에 저장된다.
- **GPU 작업은 프로젝트를 가로질러 하나만 돈다** (`jobs._claim_gpu`). `predict`·`train`·
  `resume`·`train_wsl`·`resume_wsl` 이 대상이고, 막히면 "'가시광' 프로젝트에서 'train' 작업이
  GPU 를 쓰는 중입니다" 로 409 가 나간다. scan·tidy·export 는 서로 막지 않는다.
- **SAM 은 프로세스에 하나만 올린다** (`segmenter._active`). 다른 탭에서 SAM 을 쓰면 이전
  것을 내린다 — 0.55GB 씩 세 개를 물고 있을 이유가 없다.
- **웹 UI 의 API 주소는 전부 현재 경로 기준이다** (`app.js` 의 `BASE`·`url()`). 절대경로
  `/api/...` 를 새로 쓰면 탭에서 깨진다. 탭 목록만 루트의 `/api/projects` 를 직접 부른다.
- `labeler.py` 의 다른 명령(scan·train·label…)은 여전히 `--config` 로 고른다. 다른 프로젝트는
  `labeler.py --config <설정> scan` 처럼 쓴다 (`--config` 는 하위 명령 **앞**에).
- **새 프로젝트의 클래스는 첫 `scan` 때 DB 에 들어간다.** 그 전까지 UI 클래스 목록이 비어 있는
  것은 정상이다(`discover_classes` 가 config 의 `classes` 를 DB 에 쓴다).
- 새 프로젝트를 만들 때는 `data:`·`model.weights`·`train.overrides.name`·`wsl.dataset`·`wsl.name`
  을 **전부** 새 이름으로 바꿀 것 — 하나라도 겹치면 가중치·학습 폴더·WSL 데이터셋을 서로 덮는다.

## 명령

명령 목록은 `labeler.py --help` 로 본다 (scan·train·label·resume·review·tidy·
check·split·dupcheck·prune·purge·export·status).

보조 도구는 `tools/` 에 있다 (2026-09-11 루트에서 옮김). 어느 폴더에서 실행해도
저장소의 `config.yaml`·`src/` 를 찾는다.

| | |
|---|---|
| `tools/finetune.py` | 해상도를 올리고 mosaic 을 끈 채 짧게 미세조정 |
| `tools/probe_scale.py` | 모델 크기(n/s/m)를 바꿔 같은 조건으로 비교 (n→s 결정에 썼다) |
| `tools/scheduled_stop.py` | 정해진 시각에 웹 UI 학습을 정상 중단 (`POST /api/jobs/{id}/stop`). 탭 서버에서는 `/api/projects` 로 학습 중인 프로젝트를 찾는다(09-23 전에는 탭 서버에서 아무것도 못 멈췄다) |
| `tools/label_qa.py` | 승인 라벨과 모델 예측이 어긋나는 곳을 뽑아 후보 CSV·대조 시트로 (DB 읽기 전용). 10-25 부터 **라벨끼리 겹친 곳**(`duplicate`, IoU ≥ 0.85)도 — `--dup-only` 면 모델 없이 그것만. 검수 UI 도 박스 목록에 '⚠ 겹침' 을 띄우고 승인할 때 묻는다 |
| `tools/label_qa_claude.py` | 위 missing 후보를 Claude API 로 거르는 파일럿 (대조군 포함). API 키 필요 — 아직 한 번도 호출 안 함 |
| `tools/test_baseline.py` | 지금 DB 라벨로 test 점수를 다시 잰다 (DB 읽기 전용). **`--domains` 로 전체·드론·비CCTV·드론 제외를 한 번에 재고 `runs/test_baseline/history.jsonl` 에 클래스별로 쌓는다** (09-23). `--history` 로 표 |

```bash
.venv/Scripts/python.exe tools/finetune.py          # 960 / batch 16 / 30 epoch
.venv/Scripts/python.exe tools/finetune.py --resume # 중단된 미세조정 이어하기
.venv/Scripts/python.exe tools/finetune.py --help   # imgsz·batch·epochs·lr-scale 조절
```

finetune 은 `models/best.pt` 를 건드리지 않고 `models/best_ft960.pt` 로 저장한 뒤, 옛
가중치와 나란히 검증해 보여준다. 교체는 `--promote`(이전 것은 `best.prev.pt`) 또는 사람이 직접.

**라벨 점검 (2026-09-18, `runs/label_qa/20260918_best/`) — 끝났다.** visible-3 의 missing 후보
366건을 눈으로 가려(대조군을 섞어 가린 채로) 진짜 누락 74곳/70장을 뽑았고, 사용자가 전부 검수했다.
남긴 45곳(잘린 조각·야간 번짐·유리 너머 사람·기둥 뒤 차량)은 일관된 판단이라 더 고칠 것은 없다.
그 폴더의 대조 시트·크롭 그림 492장은 10-24 에 지웠다(사용자 확인) — csv·json·판정 txt 는 남아 있다.
- 로컬에서 지운 시드 49장을 로보플로에서도 지울 할 일은 **없어졌다** — 09-28 에 사용자가 로보플로
  데이터를 전부 지웠다. 목록 `runs/label_qa/roboflow_delete_20260927.tsv` 는 기록으로만 남는다. 49장 모두
  09-11 외부 백업(`E:\local_auto_labeling_backup\20260911`)에 원본이 있다.

**라벨 기준: 배달 로봇은 라벨하지 않는다** (사용자 확인, 2026-09-18). 드론 주차장 장면
(`stabilized_*back_interaction_01`·`front_interaction_03`)의 흰 소형 차량이 그것인데,
모델이 car 로 자주 잡는다. 라벨 점검에서 missing 후보로 나와도 무시할 것.

**차량 클래스 판정 순서 (2026-09-21 확정).** 기준은 크기나 용도가 아니라 **차체가 한
덩어리인가**다. 위에서부터 차례로 보고 처음 걸리는 것으로 정한다.

| | 묻는 것 | 결과 | 예 (이미지 id) |
|---|---|---|---|
| 1 | 적재함이 캡과 **나뉘어** 있고 **열려** 있나 (호로·탈착 덮개 포함) | | |
| 1a | └ **1톤급까지** (포터·봉고·캐리·다마스, 뒷바퀴 홑바퀴) | **pickup** | 4982·2392·4051 포터 카고, 4413 F-150, 1839 글래디에이터, **2671 아발란치**, **6631 적재함을 새로 단 포터 2세대** |
| 1b | └ **그보다 크면** (마이티·메가트럭·평판·덤프, 뒷바퀴 겹바퀴) | **truck** | 5218 대형 평판, 4462 덤프, 5173 마이티 평판 |
| 2 | 적재함이 캡과 **나뉘어** 있고 **상자로 덮여** 있나 | **truck** | 5297·4979·4855·4543 포터/마이티 탑차, 대형 화물차·트레일러, **117 TAB 이동식 베팅차** |
| 3 | 한 덩어리인데 **승객 출입문이 일반 버스처럼 따로** 있나 | **bus** | 시내·관광·2층·스쿨버스, 93·8·112·2866 소형 셔틀버스 |
| 4 | 한 덩어리이고 **지붕이 앞유리 위로 곧게 이어지며 옆면이 수직**인가 | **van** | 611 무창 화물밴, 936 FedEx, 4266 하이에이스, 6490 트랜스포터, 3706 스타렉스급, **3428 스타리아**, 4341 일본식 경형·소형 캡오버 밴, **기아 레이·박스형 경차**(N-BOX·탄토·스페이시아류), **토요타 알파드**(상자형 대형 MPV), **다치아 도커**(소형 밴) |
| 5 | 그 밖 (보닛이 낮고 지붕이 뒤로 흐름) | **car** | 5978 이보크(SUV), 4513 레조로 보이는 소형 MPV, **기아 카니발**(전 세대), **토요타 시에나**, **혼다 오딧세이**, 모닝·스파크 같은 뒤가 기울어진 경차, 캐스퍼(경형 SUV) |

- **캡섀시에 얹은 특장 바디는 2번(truck)이다** — 창문·해치가 있어도 승객이 타지 않으면
  bus 가 아니다 (117 TAB 베팅차). 3번은 **사람이 타고 내리는 문**이 있을 때만이다.
- **개조차는 몸통은 얹은 것으로, 크기(1a/1b)는 원래 차종(캡)으로 본다** (사용자 결정,
  2026-09-24). 몸통은 117·1489(F-350 + 박스 바디)처럼 지금 얹혀 있는 것으로 가지를 고르고,
  1톤급인지는 캡으로 본다. **6631** 은 포터 2세대 캡에 적재함을 뜯고 평판을 새로 단 차라
  **뒷바퀴가 겹바퀴인데도 pickup** 이다 — 개조 때 바퀴도 바꿨을 수 있어 원래 급을 말해 주지
  못하고, CCTV 에서는 겹바퀴가 거의 안 보여 모델이 배울 단서도 캡 쪽이다. 새 적재함이 원래보다
  눈에 띄게 길면 5746·5754 처럼 truck 으로 갈 여지는 남는다(6631 은 비스듬해서 길이를 못 쟀다).
- **세일 패널이 있는 픽업(아발란치)은 옆에서 SUV 처럼 보인다** — 적재함이 열려 있는지
  뒤쪽을 볼 것 (2671).
- **스타리아는 보닛이 거의 없고 앞유리가 코까지 흐른다** — 부감·야간이면
  세단으로 보이기 쉽다 (3428).

- **1·2 와 4 의 갈림이 핵심이다.** 같은 포터 섀시라도 적재함이 따로 얹혀 있으면
  truck/pickup, 스프린터처럼 캡과 화물칸이 이어진 한 몸통이면 van 이다. **여기서는 크기를
  보지 않는다** — 936(FedEx 배송밴)이 5297(포터 탑차)보다 큰데도 van 이다.
- **1a/1b 의 크기 기준은 사용자가 정했다 (2026-09-21).** 열린 적재함은 **1톤급까지만
  pickup**, 그 위는 truck 이다. 이 단서가 없으면 5톤 평판(5218)까지 pickup 이 된다.
  **크기를 보는 곳은 여기 하나뿐이다.**
- **미니버스는 3번으로 가른다** (사용자 결정). 승객 출입문이 일반 버스처럼 따로 있으면
  bus, 슬라이딩 도어로 타면 van 이다. 그래서 카운티·레스타급은 bus, 스타렉스·솔라티
  연장형은 van 이다.
- **SUV 는 car 다. 승용 MPV 는 모양으로 가른다** (09-24 사용자 결정으로 고쳐 씀). 7인승이어도
  지붕이 뒤로 흐르면 car (4513, 레조로 보이는 소형 MPV), 지붕이 뒤까지 평평하고 뒷면·옆면이
  수직인 상자형이면 van (**토요타 알파드**). 승용이냐 화물이냐는 보지 않는다 — 레이와 같은 이유다.
  **기아 카니발은 세대와 상관없이 car 다** (사용자 결정, 2026-09-24). 4세대도 알파드보다 차체가
  낮고 앞코가 SUV 처럼 길어, 상자형 van 들과 모양이 다르다. 1~3세대는 지붕이 뒤로 흘러 원래 car
  라서, 세대를 가리기 어려운 부감에서도 한 클래스로 갈 수 있다. 기존 car 라벨도 고칠 필요가 없다.
  - **MPV 를 가르는 요령:** 키가 크고 앞면이 서 있는 상자(알파드·벨파이어·노아·세레나류)는 van,
    앞코가 SUV 처럼 길고 차체가 낮은 미니밴(카니발·**토요타 시에나**·**혼다 오딧세이**)은 car.
    셋 다 09-24 사용자 결정.
  - **다치아 도커는 van 이다** (사용자 결정, 2026-09-24). 앞코는 승용차처럼 나와 있지만 그 뒤
    몸통이 높은 상자다 — 지붕이 뒤까지 평평하고 뒷문·옆면이 수직이다. **앞코 길이가 아니라 뒤
    몸통으로 본다**는 점이 카니발류(차체가 낮은 미니밴)와 갈리는 곳이다. 같은 꼴의 소형 밴(르노
    캉구·시트로엥 베를링고·푸조 파트너·폭스바겐 캐디·피아트 도블로·포드 트랜짓 커넥트)은
    아직 정하지 않았다.
- **기아 레이와 같은 모양의 박스형 경차는 van 이다** (사용자 결정, 2026-09-24). 지붕이 뒤까지
  평평하고 뒷면·옆면이 수직이라 4번에 맞는다. 짧은 보닛은 가르는 기준이 아니다 — 6490
  트랜스포터·936 FedEx 도 앞코가 짧게 나와 있는데 van 이다. 같은 차체로 **레이 밴(화물)** 이
  나오고 판정표는 용도가 아니라 모양으로 가르므로 둘은 같은 클래스여야 하며, CCTV 에서 뒤에서
  본 레이는 4341 의 경형 밴과 거의 같은 상자다. 일본 박스형 경차(N-BOX·탄토·스페이시아류)도
  같다. 모닝·스파크처럼 뒤가 기울어진 경차와 캐스퍼(경형 SUV)는 5번 car 그대로다.
  - **그 전 라벨에는 레이가 car 로 들어가 있을 가능성이 크다** — van 클래스가 09-20 에 생겼다.
    몇 장인지는 세지 않았다. visible-3 은 van 을 몰라 예측 비교로는 못 찾는다. 9클래스로
    재학습한 뒤 car 라벨을 van 으로 예측하는 곳을 `tools/label_qa.py` 로 뽑거나, 고정 CCTV 에서
    반복되면 '같은 자리' 일괄 수정으로 바꾼다.
  - 4341 은 레이가 아니다 — 박스 둘 다 앞코가 거의 없는 일본식 캡오버 밴이다(09-24 확인).

**눈으로 전수 검사했다 (2026-09-21).** 같은 묶음에서 위치·크기가 같은 박스를 같은 차로 접고
(van 1,967 → 307개) 크기 내림차순으로 크롭을 만들어 봤다. 만드는 스크립트는 세션 scratchpad 에
있었고 남기지 않았다 — 다시 하려면 `boxes`+`images` 조인에 `(grp, round(cx,2), round(cy,2),
round(w,2), round(h,2))` 로 접고 PIL 로 10×8 대조 시트를 찍으면 된다.

| 클래스 | 본 것 / 대상 | 크기 하한 |
|---|---|---|
| van | **307 / 307 (전수)** | 60px |
| truck | 400 / 945 | 176px |
| pickup | 80 / 397 | 220px |
| car | 80 / 1,072 | 570px |

- **내가 오판한 3건 — 지금 라벨이 맞다. 다시 오류로 찍지 말 것.** (사용자 정정, 2026-09-21)

  | id | 지금 | 내가 잘못 본 것 | 실제 |
  |---|---|---|---|
  | **117** | truck | '시내버스의 승객 출입문' | **TAB 이동식 베팅차.** 피아트 두카토 캡섀시에 박스 바디를 얹은 것이고, 내가 문으로 본 것은 **베팅 단말기 서비스 해치**다. 지붕에 위성 안테나가 있고 승객은 타지 않는다 → 2번(상자 바디) = truck |
  | **3428** | van | '지붕이 트렁크로 흘러내리는 세단' | **현대 스타리아.** 야간 부감이라 거의 없는 보닛과 코까지 흐르는 앞유리를 세단 실루엣으로 봤다 → 4번 = van |
  | **2671** | pickup | '셰보레 서버번(SUV)' | **셰보레 아발란치 2세대.** 캡과 적재함을 잇는 **세일 패널** 때문에 SUV 처럼 보이지만 적재함은 열려 있다 → 1a = pickup |

  **세 번 다 같은 실수다 — 낯선 각도에서 실루엣만 보고 판정하고, 가르는 세부를 확인하지
  않았다.** 승객 출입문인지 서비스 해치인지, 지붕인지 앞유리인지, 세일 패널 뒤가 적재함인지
  짐칸인지. 애매하면 **원본 해상도로 키워서** 보고, 그래도 모르면 사용자에게 물을 것.
- **32 도 의심했다가 평범한 박스밴으로 판단해 뺐다** (van 그대로).
- **그래서 내가 올린 7건 중 4건만 맞았다.** 아래 truck 후보도 같은 눈으로 뽑은 것이라
  다시 보니 11장 중 2장이 틀렸다 (1489·4544, 박스 바디를 열린 적재함으로 봤다).
  **축소 대조 시트(150px 칸)로는 적재함이 열렸는지 덮였는지도 자주 틀린다** — 후보를
  올리기 전에 반드시 원본 해상도로 한 번 더 볼 것.
- **1톤급 열린 적재함 후보 25장 — 사용자가 전부 차종을 식별해 판정을 끝냈다 (2026-09-21).**
  내가 뽑고 사용자가 차종으로 갈랐다. **이 목록이 1a/1b 경계의 실물 기준이다.**

  | 판정 | 이미지 id — 차종 |
  |---|---|
  | **pickup (1톤급)** | **3135·5937** 토요타 다이나 · **4573** 포터2 · **4365** 이스즈 엘프 · **1041** 1톤급 · **5715·5749·5831** 1톤급 · **5326**(맨 오른쪽 작은 것) 포터/봉고 · **2694**(작은 흰 픽업) |
  | **truck (2톤급 이상)** | **3565·3566·3567·3569**·**5110·5173·5191**·**5326**(맨 왼쪽)·**4550** 마이티 · **4342** 히노 듀트로 · **2901** 포톤 2~3톤 · **5746·5754·5765·5938** 적재함이 길다 · **5834** 급수차 |

  - **차종을 아는 것이 제일 빠르다.** 나는 뒷바퀴(홑/겹)로 가르려 했지만 **CCTV 부감에서는
    겹바퀴가 거의 안 보여서 25장 중 대부분을 못 가렸다.** 앞모습만 보이면 **캡 형상**으로
    마이티/포터를 가르는 편이 낫다(5110·5173·5191 이 그 경우다).
  - **호로는 '덮개' 로 보지 않는다 (사용자 결정, 2026-09-21).** 1톤급이면 호로가 씌워져
    있어도 pickup 이다 — 5831 을 truck 에서 pickup 으로 옮겨 5937(초록 호로 다이나)·5749 와
    맞췄다. 위 1번의 '호로·탈착 덮개 포함' 이 그 뜻이다. **2번(truck)은 차체에 고정된
    상자 바디일 때만이다.**
  - **덮개·캐노피를 씌운 픽업도 pickup 이다 (사용자 확인, 2026-09-24).** 평평한 적재함 커버든
    캡 높이까지 올라오는 하드탑(캐노피)이든 떼어 낼 수 있는 부착물이라 차체는 그대로 픽업이다.
    높은 캐노피는 탑차처럼 보이니 **원래 적재함이 남아 있는지**로 가른다 — 캐노피 아래로 픽업
    적재함의 옆판·뒷문(테일게이트)이 보이면 pickup, 적재함을 떼고 상자 바디를 얹었으면 truck
    (1489 F-350 특장차, 4544 포터 탑차).
  - **내가 틀려서 뺀 2장 — truck 이 맞다**: **1489** 는 포드 F-350 에 **박스 바디를 얹은
    특장차**, **4544** 는 **현대 포터 탑차**(옆면에 1899-40xx 광고)다. 둘 다 축소 시트에서
    열린 적재함으로 잘못 봤다.

- **검수 UI 로 여러 장을 연달아 고친 뒤에는 박스가 줄어든 저장을 확인할 것** — 상단
  **'변경 기록'**(기본이 '박스가 줄어든 저장') 이나 `GET /api/history?shrunk=1`. 09-21 에 25장을
  고치는 동안 박스 2개(1041·2901)가 경고 없이 사라져 백업과 대조해 되살린 뒤 만든 것이다.
  원인으로 의심되는 자동 저장 경합은 09-23 에 고쳤다(아래 'UI 조작').
- **176px 아래는 재지 않았다.** CCTV 화면에서 1톤급과 2.5톤급은 그보다 작으면 사람이 봐도
  못 가른다. 남은 truck 545개·pickup 317개·car 992개는 **9클래스로 재학습한 뒤
  `tools/label_qa.py` 로 모델과 대조해 거르는 편이 낫다.**
- **EXIF 회전 이미지 49장이 있다** (Orientation=6, 삼성 폰 사진 47 + 1126 + 3181, train 48/val 1).
  PIL 로 그냥 열면 90도 돌아간 채 나와서 박스가 엉뚱한 자리로 보인다 — **라벨은 정상이다.**
  브라우저도 Ultralytics(`exif_transpose`)도 회전을 적용한다. 크롭을 뜰 때는
  `ImageOps.exif_transpose` 를 꼭 거칠 것. 이걸 모르고 id1022·1702 를 오류로 의심했다.
  - **라벨은 회전을 적용한 화면 기준 정규화 좌표이고, DB 의 width/height 는 회전 전 원시
    크기다** (2026-09-23 확인, `labeler.py check` 도 원시 크기와 견준다). 49장 중 45장은
    정사각(2992·1496)이라 차이가 없고, **정사각이 아닌 4장**(id 1050 4000x3000, 1126,
    1692, 3181 val)만 가로·세로가 뒤바뀐다. 화면·모델이 보는 크기는 `yolo_io.display_size`.
  - 그래서 **화면·SAM·크롭 좌표는 DB 크기가 아니라 회전 적용 크기로 계산해야 한다** — 09-23 전에는
    그 4장이 검수 UI 에서 찌그러져 보였고 SAM 클릭이 엉뚱한 곳을 찍었다. 지금 UI 는 브라우저가
    그린 크기, SAM 과 `tools/label_qa*.py` 는 `display_size`·`exif_transpose` 를 쓴다.
    학습·자동 라벨링은 원래 맞았다.

**일관성은 확인했다.** 같은 묶음 안에서 사실상 같은 자리(IoU ≥ 0.85)인데 클래스가 다른
박스를 전수 검사했더니 차량 쪽 충돌은 2건뿐이었고, 열어 보니 **둘 다 오탐**이었다 —
같은 주차칸에 다른 차가 서 있던 프레임(6488 파란 세단 / 6490 흰 트랜스포터),
같은 도로 지점의 다른 차(4979 탑차 / 4982 개방 적재함). 진짜 불일치는 0건이다.
(`motorcycle`/`person` 3건은 오토바이와 탄 사람이 겹쳐서 나온 것으로 역시 정상이다.)

**van 은 한 장면에 쏠려 있다 (2026-09-27 측정).** 박스 2,218개 중 **1,590개(72%)가 고정 CCTV
한 대의 795프레임**(640×640, 6자리 번호 이름)에서 나오고, 그 안의 위치는 **단 두 곳**이다 —
(0.90, 0.12)·(0.98, 0.12) 에 795개씩. **세워진 van 두 대가 795프레임 반복된 것**이라, 나머지
628개(09-24 의 483 에서 늘었다)가 실질적인 다양성이다. 분할은 van train 2,056 / val 58 / **test 60** /
미배정 44 — **test 가 60개뿐이라 van AP 는 아직 믿기 어렵다.** 그 카메라가 통째로 train 이라 생기는
당연한 결과다(묶음 분할). pickup 은 712개(test 99 / val 92)다.
- 이 795프레임은 **지금 DB 에서 묶음 `g000001`(1,433장)에 다른 출처와 섞여 있다** — 이름이 숫자뿐인
  파일을 전부 한 계열로 본 탓이다(아래 '묶음 단위 계층화' 와 `grouping.py` 머리말). 09-24 에 규칙을
  고쳐 **다음 학습(또는 `labeler.py split`) 때 이 카메라는 묶음 `g000120` 이 된다**(DB 복사본으로
  확인, 분할이 옮겨지는 이미지 0장).
- 모델이 '밴' 이 아니라 '저 카메라의 저 흰 밴' 을 외울 여지가 크다. 기준을 더 다듬는
  것보다 **다른 장면의 van 을 채우는 쪽**이 효과가 크다.

### WSL 학습 (`wsl/`, 웹 UI 'WSL 학습' 버튼)

사용자의 원래 학습 환경은 WSL Ubuntu 의 miniconda 다 (`~/miniconda3/envs/yolo_final`).
ultralytics 는 09-11 에 8.4.142 로 올려 Windows 와 맞췄고 torch 는 그대로다(WSL 2.11+cu130).
올리기 전 상태는 `~/yolo_final_freeze_before_upgrade.txt`. **패키지를 올릴 때는 드라이런으로
torch 를 안 건드리는지 먼저 볼 것** — PyPI 의 torch 가 cu130 빌드(sm_120 지원)를 덮어쓸 수 있다.

**웹 UI 의 'WSL 학습' 버튼** (`wsl_train.py`) 은 이렇게 돈다:
DB 로 `data/visible/dataset` 빌드(Windows) → `sync_dataset.sh` 로 WSL 에 복사 → `wsl` 로
`train_wsl.py` 실행하며 출력을 작업 로그로 중계 → 끝나면 `promote_weights()` 로
`models/best.pt` 교체(이전 것은 `best.prev.pt`, Windows 버튼과 같은 함수).
[중단] 은 WSL 쪽 `STOP` 파일을 만든다. 'WSL 이어하기' 는 WSL 쪽 체크포인트를 본다.
설정은 `config.yaml` 의 `wsl:` — 프로젝트를 늘리면 **WSL 데이터셋 폴더를 반드시 다르게**
둘 것(sync 가 `--delete` 로 맞추므로 같으면 서로 덮어쓴다).

- `train_wsl.py` 는 끝에 `RESULT_DIR=` / `STOPPED=` 줄을 찍는다. Ultralytics 가 같은
  이름이 있으면 name2, name3 로 바꿔서, 서버가 결과 폴더를 추측할 수 없기 때문이다.
- WSL 체크포인트(60MB)를 `\\wsl.localhost` 로 여는 건 비싸다. 웹 UI 는 박스를
  저장할 때마다 상태를 새로 고치므로, `find_resume()` 은 파일 (크기, 수정 시각) 이
  그대로면 직전 결과를 쓴다. 버튼 갱신도 페이지 로드·WSL 작업 종료 때만 한다.
- `wsl.exe` 자체 메시지는 UTF-16 이라 `WSL_UTF8=1` 로 받고, 한글이 청크 경계에서
  잘리지 않게 증분 디코더로 읽는다.
- **서버를 죽이면 WSL 학습은 계속 돈다** (별개 프로세스). 그때는 STOP 파일을 직접 만들 것.

WSL 로 돌리면 Linux 라 `cache: ram` 을 쓸 수 있다.

`train_wsl.py --stop-at N` 은 N epoch 을 마치면 정지 파일처럼 멈춘다(2026-09-18 추가).
epochs 는 그대로라 lr 스케줄이 본 학습과 같고, resume.pt 가 남아 `--resume` 으로 이어간다.

**여러 설정을 차례로 돌리는 비교 학습은 `runs/compare_0918/run_compare.sh`** (웹 UI 버튼과 같은
인자, 300 스케줄의 `--stop-at 60`). 09-18 에 사용자 요청으로 cmp_s640 epoch 54 에서 멈췄고 A40
비교로 대신했다. 그 WSL 폴더는 09-24 에 지웠다(사용자 확인) — 로그·`results.csv`·`args.yaml` 은
`runs/compare_0918/` 에 있다.
- **순서 스크립트를 `pkill` 하지 말 것** — wsl 세션이 닫히며 학습도 같이 죽는다(그때 '[중단]' 없이
  끊겼다). 전부 멈출 때는 `touch ~/new_env/yolo/runs_cmp/HALT ~/new_env/yolo/runs_cmp/STOP`.
- 결과는 `runs_cmp/` 에 둔다(`runs_wsl/` 이 아님 — 'WSL 이어하기' 가 집어 가 운영 가중치를 바꾸지
  않게). 도는 동안 웹 UI 로 학습을 시작하지 말 것(GPU 를 나눠 쓰고, WSL 버튼은 데이터셋을
  `--delete` 로 다시 동기화한다). 평가는 `tools/test_baseline.py --weights <best> --domains`.

`wsl/probe_vram.py` 는 모델 크기 × 해상도 × batch 별 학습 VRAM 을 잰다(구조만으로 모델을
만들어 가중치를 받지 않는다). 결과 표는 `docs/experiments.md` 맨 끝. 8GB 에서 들어가는
최대치: s 640 b20 / s 960 b8 / s 1280 b4 / m 640 b10 / m 960 b4 / l 640 b8 / x 640 b5.
여러 설정을 돌릴 때는 셸 스크립트 파일로 돌릴 것 — `wsl -- bash -c '... $변수 ...'` 는
변수가 먼저 풀려 빈 값이 된다(아래 '스크립트 파일로 돌릴 것'과 같은 함정, 09-18 에 또 겪음).

```bash
bash /mnt/e/local_auto_labeling/wsl/sync_dataset.sh        # data/visible/dataset -> ~/datasets/visible
~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/train_wsl.py
touch ~/new_env/yolo/runs_wsl/STOP                          # 정상 중단
~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/train_wsl.py --resume
```

Git Bash 에서 `wsl` 을 부를 때는 `MSYS_NO_PATHCONV=1` 을 앞에 붙일 것 — 안 붙이면
`/mnt/e/...` 를 `C:/Program Files/Git/mnt/e/...` 로 바꿔버린다.

**WSL 이 약 4배 빠르다 (2026-09-12, 전체 학습으로 실측).** 버튼 두 개를 같은 인자로
맞춘 뒤(아래 참고) 같은 데이터·같은 설정으로 끝까지 돌린 두 학습의 비교다.

| | Windows `runs/train-3` | WSL `runs_wsl/visible` |
|---|---|---|
| 조건 | yolo26s / 640 / batch 20 / Ultralytics 8.4.142 | 같음 |
| epoch 소요 (중앙값) | 258초 (n=199) | **62초** (n=287, 61~78초) |
| 스텝 (240 스텝/epoch) | 약 1 s/it | 0.25\~0.29 s/it (3.5\~4.0 it/s) |
| 캐시 | disk `.npy` 14GB, 준비 수 분 / 워커 4 | ram 5.3GB, 준비 11초 / 워커 8 |
| 결과 | 202 epoch 에서 정지 파일로 멈춤 | 288 epoch 조기 종료, **5.0시간** |

남은 차이는 캐시(disk/ram)·워커(4/8)·spawn/fork 뿐인데, **셋 중 어느 것이 몇 배를
냈는지는 가르지 않았다.** mAP 는 같은 epoch 끼리 ±0.02 안에서 따라갔다 — 속도 말고는 같은 학습이다.
(**epoch 수가 다른 학습끼리 같은 epoch 의 mAP 를 견주지 말 것** — `cos_lr` 이 총 epoch 에 묶여
있어 3 epoch 학습의 epoch 3 과 300 epoch 학습의 epoch 3 은 lr 이 전혀 다르다.)

- `train_wsl.py` 는 **Windows 프로젝트 코드를 import 하지 않는다.** WSL 의 Ultralytics
  에만 기대도록 정지·스냅샷·이어하기를 자체로 담았다(시작→STOP→`--resume`→STOP 을 실제로 돌려
  확인).
- **WSL 버튼은 Windows 학습 버튼과 같은 인자로 학습한다 (2026-09-11).** 두 버튼 모두
  `trainer.train_args(cfg)` 로 인자를 만들고, WSL 은 그중 data·project·name·workers·
  cache·device 만 `wsl:` 설정으로 바꿔 `train_wsl.py --bare --hyp KEY=VALUE ...` 로
  넘긴다. `--bare` 는 스크립트 자체 기본값(DEFAULTS)을 끄라는 뜻이라, config 에 없는
  키는 양쪽 다 Ultralytics 기본값이다(optimizer=auto 등). `--dry-run` 으로 최종 인자를 볼 수 있다.
  - **optimizer 를 `MuSGD` 로 이름 짓는 것과 `auto` 가 MuSGD 를 고르는 것은 다르다** — `auto` 는
    `warmup_bias_lr` 을 0 으로 바꾼다(engine/trainer.py `build_optimizer`). 그래서 직접 실행
    (DEFAULTS, MuSGD 고정)과 `tools/probe_scale.py`·`finetune.py` 는 본 학습과 이 값 하나가 다르다.
  - `wsl.exe` 는 명령줄을 bash 로 넘긴다(`$HOME` 이 풀리고 공백 없는 `a;b` 는 명령
    두 개가 된다). 그래서 영숫자와 `_ . + - =` 로만 된 값만 넘기고, 리스트 같은 값이
    overrides 에 있으면 WSL 을 띄우기 전에 멈춘다(`wsl_train.hyp_tokens`).
- **DB 는 WSL 로 가져가지 않는다.** `data/visible/labeling.db` 는 SQLite 인데 `/mnt/e` 로는
  파일 잠금이 제대로 안 먹어, Windows 검수 서버와 동시에 쓰면 손상될 수 있다.
  학습엔 이미지·라벨만 필요하므로 `sync_dataset.sh` 가 그것만 복사한다
  (`.npy`·`.cache` 는 뺀다 — 전자는 Windows 의 disk 캐시, 후자는 Windows 절대경로를 담는다).
- **사용자의 원래 스크립트 `~/new_env/yolo/train_yolo_v3_1.py` 는 다시 쓰지 말 것** — `dfl=0.0`
  이다(함정 1). 그 밖에도 yolo26n / batch 40 / patience 75 / close_mosaic 15 로 지금 설정과 다르다.

## 현재 데이터 상태

```
승인 6,567장 / 박스 41,554개 / 클래스 9개      (2026-09-27 확인)
bicycle 1,511  bus 985  car 9,270  motorcycle 1,459  person 23,321
scooter 1,031  truck 1,047  pickup 712  van 2,218
검수 대기 0, 미라벨 0, 건너뜀 0, 휴지통 0, 박스 없는 승인 이미지 127
(10-03: Objects365 500장을 검수 대기로 더 넣었다 — 위 'Roboflow MCP' 의 Objects365 항목)
분할: train 4,642 / val 822 / test 968(박스 6,508) / 미배정 135 — DB 의 images.split 에 고정
      (미배정은 다음 학습 때 배정된다. 09-24 에 DB 복사본으로 돌려 봤을 때 기존 배정 이동은 0장)
```

**09-24~27 사용자가 데이터셋을 크게 손봤다** (09-24 백업과 대조, `box_history` 2,732건).
- 기존 이미지 782장을 고쳤다 — 클래스 **car→van 208, truck→pickup 72, car→pickup 46** 등 09-24
  판정표(레이·알파드·도커 van, 캐노피 픽업 등)를 따른 것이다. 새로 그린 박스 162(person 133), 지운 76,
  위치·크기 749. **test 이미지 129장이 바뀌었다.**
- 새 이미지 109장(비CCTV, pickup 63·van 42 박스 포함), 지운 이미지 46장(시드 `ala_2_*` 등 id
  1897~1932 와 휴지통 5240·5241 — 파일도 지워졌다. 로보플로 목록은 아래 '라벨 점검').
- 점검(09-27): 파일 없음 0, 바이트 중복 0, 같은 묶음·같은 자리 차량 클래스 엇갈림 0(후보 2곳은 다른 차),
  dHash 로 본 파일 내용 변경은 알려진 1131 뿐. **정리(tidy) 덮어쓰기로 사진 2장을 잃었다** — 함정 13.
  4428 에 person 박스가 두 번 그려진 곳이 새로 생겼다(IoU 0.94, 아직 그대로).

**클래스는 9개다 — `pickup`(7번)·`van`(8번)을 09-20 에 사용자가 검수 UI 에서 추가했다.** 판정
기준은 위 '차량 클래스 판정 순서'.
- **7클래스 시절의 점수(visible-3 기준선 0.6868, A40 비교 등)는 이제 그대로 비교할 수 없다.** 평가
  클래스가 늘어 평균이 달라지고, 옛 가중치는 pickup·van 을 아예 모른다(AP 0). 기존 car·truck 박스를
  pickup·van 으로 바꾼 것도 섞여 있다. 비교는 **클래스별**로 할 것. 7클래스 기준선 표는
  docs/experiments.md 로 옮겼다.
- **지금 기준선은 09-27 라벨로 다시 잰 것이다 (`tools/test_baseline.py --domains`, visible-3 =
  `models/best.pt` sha256 `ca1f9f5c…`, 640).** `runs/test_baseline/history.jsonl` 에 09-23 기록과 함께
  있다. 다음 모델은 `--weights <새 가중치> --domains` 로 재면 이 기록과 클래스별로 나란히 나온다.

  | 도메인 | 장 / 박스 | mAP50-95 (9클래스 평균) | 09-23 | 클래스별 mAP50-95 (09-27) |
  |---|---|---|---|---|
  | 전체 | 968 / 6,508 | **0.5174** | 0.5208 | bus 0.887 · car 0.753 · person 0.625 · truck 0.618 · bicycle 0.611 · motorcycle 0.582 · scooter 0.581 · **pickup 0 (99) · van 0 (60)** |
  | 드론 | 36 / 491 | 0.5215 | 0.5215 | car 0.750 · **person 0.293** |
  | 비CCTV | 139 / 488 | 0.3780 | 0.3788 | bus 0.851 · bicycle 0.543 · car 0.530 · person 0.430 · truck 0.395 · motorcycle 0.355 · scooter 0.298 |
  | 드론 제외 | 932 / 6,017 | 0.5212 | 0.5251 | |

  **같은 가중치인데 09-23 보다 낮은 것은 라벨이 바뀐 몫이다** — car −0.025·truck −0.017 이 대부분이다.
  car·truck 박스를 van·pickup 으로 옮기자, 그 둘을 모르는 visible-3 이 여전히 car·truck 으로 잡아서
  오검출이 됐다. 평균에 pickup·van 의 0 이 들어가 옛 7클래스 값보다 낮게 나온다. 드론 person 0.2929 는
  09-18 부터 소수 4자리까지 같다.
  **비CCTV 가 가장 약하다** — 바뀐 방향(어떤 이미지든)에서는 여기가 가장 큰 약점이다.
  비CCTV 는 `source='unlabeled'`(나중에 넣은 이미지)로 가른다. 옛 `--like "data\labeled"` 는 09-21
  폴더 이동 뒤로 **0장**에 걸린다 — 쓰지 말 것.
- 클래스 번호는 순서가 전부다. config.yaml 의 `classes` 에 **뒤에만 붙이고** 중간에 끼워
  넣거나 빼지 말 것 — 박스가 번호로 가리킨다.

`labeler.py split` 은 읽기 전용이 아니다 — 보여주기 전에 미배정 이미지를 배정해 DB 에 쓴다.
DB 만 보려면 `sqlite3` 를 `mode=ro` 로 열 것.

**묶음 단위 계층화 3분할로 바꿨다 (2026-09-16, `split.py`·`grouping.py`).**

- **test 를 뗐다 (15%).** 학습에도 best.pt 고르기에도 쓰지 않는다. val 은 300 epoch 중 가장
  좋은 지점을 고르는 데 쓰이므로 그 이득이 점수에 섞인다. 모델끼리 비교는 test 로 한다
  (`tools/test_baseline.py`).
- **같은 장면은 통째로 한쪽에만 넣는다.** 갈라진 묶음 0개. 가장 큰 묶음은 **고정 CCTV 한 대의
  795프레임**이다(van 쏠림의 원인). 09-23 까지는 이름이 숫자뿐인 파일이 출처와 상관없이 전부 한
  묶음(`g000001`, 1,433장)이었다 — COCO 식 12자리 사진·256px 이미지·폰 사진이 CCTV 와 섞여 묶음째
  train 에 갇혔다. **09-24 에 자릿수·이미지 크기가 같아야 한 계열이 되도록 고쳤다**
  (`grouping.name_series`). 다음 학습(또는 `labeler.py split`) 때 그 1,433장이 360묶음으로 나뉘고
  CCTV 는 `g000120` 이 된다. 이미 전부 train 이라 **옮겨지는 이미지는 0장**이다(DB 복사본으로 확인)
  — 그 사진들을 val·test 로 보내려면 `split --reset` 이 필요하고, 그러면 점수 비교가 끊긴다.
- **계층화.** 클래스별 비중이 목표에서 최대 5.5%p 벗어난다(person: train 64.5% /
  test 18.5%). 이미지 단위로 나눌 때의 0.1%p 보다 나쁘지만, 묶음을 쪼갤 수 없어서
  생기는 값이다. 큰 묶음부터 '넣었을 때 가장 덜 틀어지는 쪽'에 넣고, 순서를 12번
  흔들어 가장 나은 배정을 고른다.
- **잔여 누수.** 묶음 기준(dHash ≤2)으로는 val·test 모두 0%. 느슨한 기준(≤6)으로 보면
  val 5.6% / test 4.7% 가 train 과 비슷하다 (묶기 전에는 val 17.5% / test 22.4%였다).
  남은 것은 '같은 카메라를 다른 날·다른 조명에서 찍은 것'이다. 거리 3~6 쌍을 열어 보면
  진짜 같은 장면과 오탐(둘 다 어두운 화면)이 섞여 있어서, 기준을 더 낮추면 상관없는
  것까지 묶인다. `labeler.py split --leak` 으로 다시 잴 수 있다.
- **도메인 치우침 — 계층화는 클래스 비중만 맞추고 출처는 보지 않는다** (`tools/label_qa.py` 로
  발견). 09-18 에 둘을 손으로 고쳤다(사용자 결정).
  - 드론 부감 영상(`stabilized_*`)이 전부 test 에 있었다 — 묶음 3개(`g003272`·`g003290`·`g003323`,
    49장)를 train 으로 옮겼다. test 에 남긴 **`g003254`·`g003305` 는 dHash 거리 4 라 사실상 같은
    장면이므로 반드시 같은 쪽에 둘 것.** 되돌리기: `runs/label_qa/drone_split_backup_20260918.json`.
  - 비CCTV 가 val 에 쏠렸던 원인 `g006004`(T 시리즈 126장)와 test 의 같은 시리즈 7장을 train 으로
    옮겼다(`runs/label_qa/noncctv_split_backup_20260918.json`).
  - **옮기는 방향은 val·test -> train 뿐이어야 한다** — 반대로 옮기면 이미 학습한 이미지가 평가셋에
    들어가 점수가 부풀려진다.

**09-16 재배정 전의 모델(train-3·visible·visible-2)의 test 점수는 쓰면 안 된다** — 지금 val/test 에
있는 이미지를 학습에 썼다(visible-2 는 test 에서 0.880 이 나온다). 옛 모델과 비교할 때는 두 모델 모두
**장면째** 못 본 이미지만 쓸 것. 방법과 결과는 [docs/experiments.md](docs/experiments.md).

앞으로 새로 승인되는 이미지는 자기 묶음을 따라가고(묶음이 이미 배정돼 있으면 그쪽),
새 묶음이면 계층화로 들어간다. 기존 이미지는 움직이지 않는다 — 시험에서 새 이미지
200장을 넣었을 때 기존 배정 이동 0장을 확인했다.
**예외였던 것도 막았다 (2026-10-25, Gemini 지적).** 새 이미지가 서로 다른 split 의 두 묶음을 dHash 로
이어 붙이면, 예전에는 합쳐진 묶음의 다수결로 소수 쪽 기존 이미지가 옮겨졌다(합성 시험에서 test 1장이
train 으로 갔다). 이제 `reset=False` 면 **기존 이미지는 무조건 제자리**이고(새 이미지만 다수결을 따른다),
그런 묶음은 `SplitReport.mixed` 로 **[알림] … 두 분할에 걸칩니다** 를 띄운다 — 옮기는 대신 그 장면이
두 분할에 걸치는 누수를 감수한 것이다. `moved` 경고는 reset 이 아니면 0 이어야 한다. 10-25 DB 복사본으로
돌려 기존 6,432장 이동 0·mixed 0 을 확인했다(새 164장은 train 58·val 85·test 21). 전체를 다시 나누려면
`labeler.py split --reset --yes` 를 사람이 돌려야 한다.
- **장면 묶기에서 떼는 꼬리표**: 로보플로(`_jpg.rf.<해시>`)와 tidy(`__<sha1 8자>`·`_2`)를 뗀 이름으로
  연번을 본다(10-25 — `maxresdefault__7bfb8759` 가 해시 끝 '59' 로 연번이 됐었다).
- **dHash 는 EXIF 회전을 적용해 잰다**(10-25). 회전 태그로 저장된 사진과 돌려 저장된 같은 사진의 거리가
  35·37 → 0·1 이 됐다. DB 에 캐시된 49장의 dHash 도 새 방식으로 바꿨다(그 전 DB:
  `data/visible/labeling.db.bak_collab1_20261025_040203`).

**폴더는 프로젝트마다 따로다 (2026-09-21 에 가시광을 `data/visible/` 로 모았다).**
지금은 `data/visible/` 하나다(열화상·화재는 09-28 에 지웠다). **DB 는 이미지를 절대경로로 들고 있어서,
폴더를 옮기면 `images.path` 도 같이 치환해야 한다** — 아니면 전부 '파일 없음' 이 된다(09-21 에
6,478행을 치환했다. 옮기기 직전 DB: `data/labeling.db.bak_20260921_125612`).

- `data/visible/seed/seed1`, `seed2` — Roboflow 내보내기를 회차별로 쌓는다. 각 폴더에
  `coco/*.json` + `images/`. `config.yaml` 의 `seed_coco: data/visible/seed` 가 하위 json 을
  전부 읽는다. **로보플로에는 이제 없다(09-28 사용자가 전부 지움) — 이 폴더가 유일한 사본이다.**
- `data/visible/labeled/` — 검수 완료 후 `tidy`로 옮겨진 것 (이미지+라벨)
- `data/visible/dataset/` — **학습 시 자동 생성되는 폴더.** 학습을 시작할 때마다(Windows·WSL
  버튼 모두) DB 로 새로 만든다. 지금 것은 마지막 학습 무렵의 옛것이다. **test 를 잴 때는 이 폴더 말고
  `tools/test_baseline.py` 를 쓸 것**(DB 에서 그때그때 만든다). 이어하기는 없으면 다시 만든다
  (`ensure_resume_dataset`). Windows 학습은 여기에 `.npy` 디스크 캐시(약 14GB)를 더 얹는다.
- `data/visible/labeling.db` — **작업물 그 자체.** 아래 주의사항 참고. 옆의 `backups/` 에 검수
  서버가 하루 한 번 사본을 남긴다(최근 14개).
- `data/visible/classes.txt` — `Config.classes_txt` 는 **DB 옆**을 본다(09-21 전에는 세 프로젝트가
  `data/classes.txt` 하나를 같이 썼다).
- `export/<프로젝트>/` — 내보내기 기본 폴더(09-24 부터). 예전 기본값 `export` 는 세 프로젝트가 같이
  써서 섞였다. 이 함수가 만든 폴더(`export_info.json`)만 비우고 다시 쓰고, 표식 없는 폴더가 비어
  있지 않으면 쓰지 않고 멈춘다. 표식은 **쓰기 전에** 남기고(`finished: false` → 끝나면 true) — 첫
  내보내기가 도중에 죽어도 다시 내보낼 수 있다. 지난 라벨을 못 지우면 멈추고, 대소문자만 다른 이름
  (`IMG_1`·`img_1`)은 Windows 에서 같은 파일이라 해시를 붙여 가른다(10-02 코드 리뷰).

**받는 이미지 형식은 `yolo_io.IMG_EXTS` 하나로 정해진다 (2026-09-22 에 `.avif` 추가).**
`iter_images()` 가 이 집합으로 거르고, `ingest` 가 **시드 폴더와 미라벨 폴더를 훑을 때 둘 다**
그것을 쓴다. **여기 없는 확장자는 `scan` 이 조용히 건너뛴다 — 에러도 경고도 없다.**
지금 목록: `.jpg .jpeg .png .bmp .webp .tif .tiff .avif`.

- **AVIF 는 Pillow 11.3+ 에 코덱이 내장돼 있어 된다** (09-22 실측: Windows 12.3.0·WSL 12.1.1 왕복
  픽셀 일치, 검수 UI 도 뜬다). Windows 의 cv2 는 AVIF 를 못 읽지만 Ultralytics `imread` 의 PIL
  폴백이 받는다. **렌탈 pod 의 Pillow 11.0.0 에는 AVIF 가 없다** — Pillow 를 먼저 올릴 것(Runpod 항목).
- **`.heic`·`.heif` 는 넣지 않았다** — 이 Pillow 에 HEIF 코덱이 없다. 아이폰 사진을 그대로 넣으려면
  **`pillow-heif` 를 먼저 깔고** IMG_EXTS 에 추가할 것.

`weights/` 에 사전학습 가중치를 모아 둔다 — `yolo26n/s/m/l.pt`, `sam2.1_t.pt` (m·l 은 09-18 에
받았고 아직 학습에 쓰지 않았다). config 에 이름만 적어도(`yolo26s.pt`) `config.pretrained_path()`
가 여기서 찾는다. 이게 없으면 Ultralytics 가 **실행한 폴더**에 받아 버린다.

`models/` — `best.pt`(운영) = **`best_visible3.pt`** = visible-3 epoch 156 (val 0.654, 기준선을 잰
가중치, SHA-256 `ca1f9f5c…`). `best_visible2.pt`(visible-2 e202), `best_ft960.pt`(옛 n 960 미세조정).
`best.prev.pt` 는 지금 없다 — 다음 학습이 끝나면 생긴다.

`best.pt`·`best.prev.pt` 는 학습이 끝날 때뿐 아니라 **중단할 때마다** 그 시점 best 로 밀려난다
(09-18 에 visible-4 를 2.5분 만에 끊자 `best.pt` 가 val 0.525 짜리로 바뀌었다). **그래서 운영할
가중치는 늘 `best_<이름>.pt` 로 따로 둔다** — 이름이 다른 파일은 자동으로 덮이지 않는다.

**visible-3 은 visible-2 보다 처음 보는 장면에서 약 0.05 낮았다** (09-16 공정 비교, 7개 클래스 전부
같은 방향). 원인은 가르지 않았다 — 묶음 분할로 train 이 5,098 → 4,526장으로 줄어든 것이 유력하다.
데이터를 늘리면 새 종류 장면에서 크게 오른다(09-14: 0.537 → 0.800). 둘 다 docs/experiments.md.

**외부 백업: `E:\local_auto_labeling_backup\20260911\`** (09-11, `.venv`·`data/dataset` 빼고 전부,
md5 대조). 그날 정리로 지운 것(주기 체크포인트·옛 DB·버린 학습 폴더 등)도 여기 있다.

**다른 디스크 백업: `C:\local_auto_labeling_backup\20261031\`** (10-31, E: 와 다른 물리 디스크 — C: = 디스크 1).
09-28 에 뜬 `…\20260928\`(17,346개 5.15GB)이 10-31 에 보니 없어져서(C:\ 에도 휴지통에도 없었다) 사용자 확인을
받고 같은 날 새로 떴다. `.venv`·`graft`·`data/visible/dataset`·`__pycache__`·`*.tar`·살아 있는 DB 파일을 빼고
전부 — 8,127개 5.11GB(코드·`.git`·`data/visible` 의 seed·unlabeled·labeled·backups·옛 DB `.bak` 2개·models·
weights·runs)를 원본과 md5 전수 대조했다(불일치·누락 0, 복사 직후 원본이 고쳐진 requirements.txt 1개뿐). DB 는
sqlite `backup()` 스냅샷을 `data/visible/labeling.db` 에 넣었다(journal_mode=delete, integrity ok, 이미지 7,067 /
박스 44,844 / 이력 4,105 — 원본과 같음). DB 가 가리키는 이미지 7,067장은 전부 복사 범위 안에 있다.
`E:\datasets\objects365`(O365 원본)는 저장소 밖이라 들어 있지 않다. 뺀 것·되살리는 법·다시 뜨는 명령은 그 폴더의
`BACKUP_INFO.txt`, 로그는 `C:\local_auto_labeling_backup\robocopy_20261031.log`. 09-28 방법(`/XF labeling.db*`)과
달리 옛 DB 백업(`labeling.db.bak_*`)도 담는다. **C: 는 남은 공간이 적다**(10-31 백업 뒤 25.2GB) — 다음 백업은
날짜 폴더를 새로 만들기 전에 이전 것을 지울지 물을 것.

---

## 반드시 알아야 할 함정

이 프로젝트에서 실제로 겪고 원인까지 확인한 것들이다. 모두 **에러 없이 조용히
성능만 깎거나 데이터를 잃는** 종류라 재발하기 쉽다.

### 1. `dfl` 을 0으로 두지 말 것

`train.overrides.dfl` 은 이름과 달리 DFL 전용이 아니라 **박스 거리 손실 게인**이다.
`ultralytics/cfg/default.yaml` 주석 그대로: *"DFL when reg_max > 1, L1 on DFL-free YOLO26"*.

- yolo11 (reg_max=16) → `loss[2]` = DFL
- yolo26 (reg_max=1)  → `loss[2]` = ltrb 거리의 **L1**
- 두 경우 모두 `loss[2] *= hyp.dfl` 을 거친다 (`utils/loss.py`)

0으로 두면 아키텍처와 무관하게 거리 회귀 감독이 꺼진다. 합성 데이터 20 epoch A/B:
**mAP50은 0.995로 동일, mAP50-95는 0.902 → 0.953.** 검출은 CIoU로 학습되므로
망가지지 않고 **좌표 정밀도만** 손해다. `trainer.check_box_distance_gain()` 이 경고한다.

### 2. Windows 에서 `cache: ram` 금지

Windows는 `spawn`만 가능하다(`fork` 없음). DataLoader 워커마다 데이터셋 객체를
**피클해서 파이프로 보내는데**, RAM 캐시(수 GB numpy)가 그 안에 있어 죽는다.

```
OSError: [Errno 22] Invalid argument
  ... when serializing BaseDataset._ImageCache state
```

뒤따르는 `_MultiProcessingDataLoaderIter ... _workers_status` 는 뒷정리 잡음이지 원인이 아니다.
현재 `cache: disk` (`.npy` 경로만 피클). `trainer.check_ram_cache()` 가 경고한다.

> 사용자의 원래 코드는 리눅스에서 `cache=True`를 썼고 그건 옳다 — fork는 copy-on-write라 공짜다.
> 리눅스로 옮기면 `cache: true` 로 되돌리는 게 더 빠르다.

### 3. 하드링크 금지 (`hardlink_images: false`)

Ultralytics는 스캔 중 깨진 JPEG을 발견하면 **그 자리에 다시 저장한다**
(`corrupt JPEG restored and saved`). `data/<프로젝트>/dataset` 을 하드링크로 만들면 같은 inode라
**원본까지 덮어써진다.** 실제로 8개 파일이 그렇게 변경됐다. 지금은 복사가 기본.

### 4. Roboflow YOLO 내보내기는 박스를 자른다

세그 폴리곤을 한 줄에 하나만 담는 형식이라, **가려져서 조각난 객체는 큰 조각만 남고
나머지가 버려진다.** COCO 내보내기의 `bbox`와 전수 대조한 결과:

| IoU | 비율 |
|---|---|
| ≥0.95 | 65.5% |
| 0.90~0.95 | 23.1% |
| **<0.70** | **6.0%** (2,365개) |

IoU 0.90 미만 4,481개에서 COCO 박스가 평균 2.4배 컸다. **그래서 COCO로 갈아탔다.**
데이터를 다시 받을 일이 있으면 반드시 **COCO 또는 Pascal VOC**로 받을 것.

또한 Ultralytics는 한 파일에 5열 행과 7열 이상 행이 섞이면 `labels mix segment and
detection rows` 로 **이미지째 버린다**. 그 데이터에서 1,085장(18%)이 그랬다.
`yolo_io._row_to_box()` 는 행 단위로 처리해 이를 살린다.

### 5. `labeling.db` 를 함부로 지우지 말 것

**검수 대기·건너뜀·휴지통 라벨은 DB에만 있다.** `.txt` 로 파일에 쓰이는 시점은
`train` / `tidy` / `export` 세 곳뿐이고 전부 **승인된 것만** 대상이다.

DB를 재생성해야 하면 **먼저 사용자에게 물을 것.** 이전에 데이터 교체 과정에서
여러 번 지웠는데, 검수가 본격화된 지금은 작업물 손실이 된다.

### 6. 파일에 코드를 쓸 때 heredoc 을 쓰지 말 것

이 세션에서 `bash heredoc + python 패치 스크립트`로 JS/Python 을 수정하다
`\n`·`\r` 이스케이프가 **실제 제어문자로 파일에 들어가** 문법 오류를 여러 번 냈다.
CR이 들어가면 `sed` 출력이 멀쩡해 보여 진단도 어렵다.

**코드 파일 수정은 Write / Edit 도구를 쓸 것.** 특히 문자열 리터럴이 있는 곳.

### 7. Windows 에서 스크립트를 짤 때 `if __name__ == "__main__":` 가드를 뺄 것

Ultralytics 의 `train()`/`val()` 을 모듈 최상단에서 부르면 DataLoader 워커가
spawn 으로 뜨면서 **그 스크립트를 통째로 다시 import** 한다. 가드가 없으면
`RuntimeError: An attempt has been made to start a new process before the current
process has finished its bootstrapping phase` 로 죽거나, 부모–자식이 서로 기다리며
**CPU 0% 로 조용히 멈춘다.** 이 세션에서 두 번 냈다.

일회용 분석 스크립트라면 `workers=0` 을 주는 편이 더 간단하다.

### 8. 학습이 끝나면 `last.pt` 로는 이어 학습할 수 없다

`final_eval()` 이 `strip_optimizer()` 를 불러 `last.pt` 와 `best.pt` 를
**덮어쓴다** (`utils/torch_utils.py`). optimizer/ema/scaler/best_fitness 가 None 이
되고 `epoch` 가 -1 이 되며 모델은 FP16 + `requires_grad=False` 가 된다.
정상 종료든 중단이든 똑같이 적용된다. 파일 크기로 구분된다 — yolo26s 는 60MB=온전,
20MB=제거됨 (yolo26n 은 15.5MB / 5.4MB).

그래서 `trainer.attach_callbacks()` 가 **중단이 확정된 epoch 의 `last.pt` 를
`resume.pt` 로 복사해 둔다.** `on_fit_epoch_end` 는 `save_model()` 직후이자
`final_eval()` 이전이라 이 자리가 유일한 기회다.

강제 종료(프로세스 kill, 정전)로 죽은 경우는 오히려 안전하다. `final_eval()` 이
돌지 않아 `last.pt` 가 온전하게 남는다.

**정상 중단은 정지 파일로 한다 (`stopper.py`).** `runs/STOP` 을 만들면 **다음
배치 경계에서** 멈추고 resume.pt 를 남긴다. 진행 중이던 epoch 의 나머지는 건너뛰고
그 epoch 을 완료로 기록한다(검증·저장은 한다) — 1초 안에 반응하는 대신 한 epoch
일부를 덜 학습한다. 웹 UI 중단 버튼도 원래 이렇게 동작한다. CLI(`train`/`resume`)와 웹 UI 로 시작한
학습 모두 이 파일을 본다. WSL·렌탈 서버에서도 똑같이 쓴다.

    touch runs/STOP          # WSL / Linux
    New-Item runs\STOP       # PowerShell

**Ctrl+C 는 가로채지 않는다.** 터미널의 Ctrl+C 는 DataLoader 워커에게도 가고,
워커는 KeyboardInterrupt 를 받으면 루프를 빠져나간다
(`torch/utils/data/_utils/worker.py`: *"Main process will raise KeyboardInterrupt
anyways"*). 메인에서 신호를 삼키고 배치 경계까지 가려 해도 워커가 이미 죽어
에러로 끝난다. 그래서 Ctrl+C 는 즉시 종료로 두었다 — `last.pt` 는 온전하므로
진행 중이던 epoch 하나만 잃는다. Linux 에서는 `kill <메인 PID>`(SIGTERM)도
정상 중단으로 받는다.

**스냅샷은 사람이 멈춘 때만 남기고, 끝난 학습은 이어하기 대상에서 뺀다 (2026-09-11).**
WSL 버튼을 시험하다 세 겹의 문제가 드러났다. 모두 '끝난 학습에 이어하기 버튼이 뜨고,
누르면 epoch 을 버리는' 증상이다.

1. **조기 종료도 스냅샷을 남겼다.** Ultralytics 는 patience 조기 종료에도
   `self.stop |= self.stopper(...)` 로 `trainer.stop` 을 세운다. 스냅샷 조건이 그걸
   봤기 때문에 `runs/train` 이 epoch 259 조기 종료 때 resume.pt 를 남겼다.
   → 조건을 `state["stopped"]`(정지 파일·중단 버튼으로만 섬)로 바꿨다.
2. **중단 → 이어하기 → 완주하면 첫 스냅샷이 남았다.**
   → 사람이 멈춘 게 아니면 끝날 때 resume.pt 를 지운다 (`clear_stale_snapshot`).
3. **주기 체크포인트로 내려가는 폴백이 끝난 학습을 되살렸다.** `resume_point_in` 은
   resume.pt·last.pt 를 못 쓰면 `epochN.pt` 로 내려가는데, 그 파일들은 옵티마이저가
   살아 있다. `save_period` 로 epoch0.pt 가 늘 남으므로 끝난 학습마다 버튼이 떴다.
   → last.pt 가 strip 된 흔적(`epoch == -1`)이 있으면 끝난 학습으로 보고 None.

**끝난 학습의 주기 체크포인트(`epochN.pt`)는 지워도 된다** — `best.pt`·`last.pt`·그래프·
`results.csv` 만 남긴다(09-11·09-14 에 그렇게 정리했다. train-3 의 202/300 상태는 외부 백업에 있다).
09-24 에 WSL `runs_wsl/visible-3` 의 14개(805MB)와 멈춘 비교 학습 `runs_cmp/cmp_s640`(290MB)도
지웠다(사용자 확인). 앞으로도 **지우기 전에 사용자에게 물을 것.** WSL 안에서 지워도 가상 디스크
파일(`E:\WSL\Ubuntu\ext4.vhdx`)은 저절로 줄지 않는다.

WSL 에서 파일을 지울 때는 **스크립트 파일로 돌릴 것.** `wsl -- sh -c '... $d ...'` 로
넘기면 바깥 bash 가 `$d`·`$(...)` 를 먼저 풀어 빈 값이 된다 — 이날 `rm` 이 `/weights/...`
를 향해 헛돌았다(없는 경로라 피해는 없었다). 위 `wsl.exe` 항목과 같은 원인이다.

**이어하기는 이 설정의 데이터셋으로 돈 학습만 고른다 (2026-09-11).** 가시광과 열화상이
`runs/` 를(WSL 은 `runs_wsl/` 를) 같이 쓰는데, 예전에는 설정과 무관하게 **가장 최근
폴더**를 집었다. 열화상 학습을 한 번 돌리면 가시광 UI 에 그 이어하기가 뜨고, 누르면
끝날 때 `promote_weights()` 가 열화상 결과로 `models/best.pt` 를 덮는 구조였다(열화상
설정으로 확인하니 실제로 train-3 이 떴다). 체크포인트의 `train_args["data"]` 가
`cfg.dataset/data.yaml` 과 같은 것만 받는다 (`trainer.uses_dataset`). 폴더 이름은
config 로 바뀌므로 믿지 않는다.

**`data/<프로젝트>/dataset` 이 없으면 이어하기가 DB 로 다시 만든다 (`ensure_resume_dataset`).**
정리 때 그 폴더를 지우자 '이어서 학습' 버튼은 뜨는데 누르면 "데이터셋이
없습니다" 로 실패하게 됐었다. 분할이 DB(`images.split`)에 고정돼 있어 DB 가 그대로면
다시 만든 결과가 원래와 같다 (경로 해시로 나누던 09-11 에도 확인했다 — DB 로 다시 계산한
라벨 5,961개가 train-3 과 같은 DB 로 만든 WSL 복사본과 한 글자도 다르지 않았다). 그래서
다시 만들 때 '학습 폴더가 생긴 뒤 updated_at 이 바뀐 승인 이미지 수'를 같이 찍는다. 0 이
아니면(그 사이 새로 승인돼 배정된 것 포함) 데이터셋이 달라져 이전 epoch 과 점수를 바로
비교할 수 없다고 경고한다. DB 에서 지운 행은 흔적이 없어 이 수에 안 잡힌다.

그 폴더가 **남아 있으면** 다시 만들지 않고, 승인 수와 데이터셋 장수(train+val+test)가 다르면
알리기만 한다.

**정상 중단은 운영 가중치를 바꾼다.** `_finish()` 가 끝날 때마다 그 학습의
best.pt 를 `models/best.pt` 로 복사하는데, 도중에 끊어도 이 경로를 탄다. 그래서
덮어쓰기 직전에 기존 것을 `models/best.prev.pt` 로 남기도록 했다. 되돌리려면
그 파일을 `best.pt` 로 복사하면 된다.

### 9. Windows 는 VRAM 이 모자라도 OOM 을 내지 않고 **조용히 느려진다**

리눅스라면 `CUDA out of memory` 로 죽을 상황에서, Windows(WDDM)는 할당을
**시스템 RAM("공유 GPU 메모리")으로 옮기고 계속 돌린다.** PCIe 를 통해 오가므로
배치당 1.3초가 **45초**까지 늘어난다. 에러는 한 줄도 안 난다.

실제로 겪은 값 (batch 40, 8GB 카드). 사람이 PC 를 쓰는 동안에만 터진다:

```
epoch 22~25    136~144초    정상
epoch 26~59    134~3612초   폭주. 사람이 PC 를 쓰던 시간대 (약 2.3시간 손해)
epoch 60~89    134~136초    사람이 안 쓰자 완전히 안정
```

**증상 판별은 `results.csv` 의 epoch 소요시간으로 한다.** 아래 지표는 다 못 쓴다.

| 지표 | 왜 못 쓰나 |
|---|---|
| `utilization.gpu` | 정상·붕괴 모두 99~100% |
| `power.draw` | 정상일 때 36W 도 69W 도 나온다. 붕괴 때 42W. 겹친다 |
| `Shared Usage` 카운터 | 정상일 때도 2.8GB. '예약'이지 '전송 중'이 아니다 |
| VRAM 사용량 총합 | 7.7GB 로도 멀쩡히 돈다. 고정 임계선이 없다 |

**원인 규명에 쓸 수 있는 건 이것 하나다** — 프로세스별 전용 VRAM 을 어댑터(LUID)로
쪼개 본다. 학습 프로세스와 같은 LUID 에 뭐가 있는지가 전부다.

```powershell
(Get-Counter "\GPU Process Memory(*)\Dedicated Usage").CounterSamples |
  Where-Object {$_.CookedValue -gt 30MB} | ForEach-Object {
    if($_.InstanceName -match 'pid_(\d+)_luid_(0x\w+_0x\w+)'){
      "{0,-16} {1} {2} MB" -f (Get-Process -Id $Matches[1] -EA SilentlyContinue).ProcessName,
        $Matches[2], [math]::Round($_.CookedValue/1MB) } }
```

**실제 범인은 `dwm`(데스크톱 창 관리자) 이었다.** 하이브리드 노트북(AMD 840M +
RTX 5060)인데 Chrome 이 dGPU 에서 돌고 있었고, dwm 이 그 창을 합성하려고 RTX 에
**2.3GB** 를 잡았다. Chrome 자체는 71MB 였다 — 유발자와 점유자가 다르다.

```
붕괴 시  python 7,084 + dwm 2,291 + chrome 71 + claude 41 = 9,526MB / 8,151MB
해결 후  python 6,332MB 단독 (dwm·chrome·claude 는 전부 AMD 쪽 LUID)
```

고친 방법: `설정 → 시스템 → 디스플레이 → 그래픽` 에서 **chrome.exe 를 '절전'(iGPU)**
으로. 레지스트리 `HKCU\Software\Microsoft\DirectX\UserGpuPreferences` 에 남아 영구적이다.
단 **`GpuPreference` 는 앱이 시작할 때만 읽히므로** 바꾼 뒤 그 앱을 재시작해야 한다
(이때 재부팅이 필요했던 이유). 상시 켜두는 Electron 앱은 미리 지정해 둘 것.

빠른 확인: `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 가
학습 프로세스 몫과 비슷하면 정상, 7.5GB 이상이면 뭔가 올라온 것이다. 위의 약 6.3GB 는
yolo26n batch 40 때 값이다. 지금 설정(yolo26s batch 20)의 Windows 학습 몫은 따로 재지
않았다 — 프로브로는 b16 4.47GiB / b24 6.61GiB (`docs/experiments.md`).

원인은 데스크톱이 VRAM 을 같이 쓰기 때문이다. 이 기기에서는 M365 Copilot
(`WorkloadsSessionHost`) 2.4GB, `dwm` 0.6GB, 그 외 Chrome·NVIDIA Overlay·
msedgewebview2 가 붙어 있었다. 그때의 `batch: 40` 은 사용자가 **리눅스에서** 잡은 값이라
데스크톱 몫이 없었다(지금 설정은 yolo26s 에 20).

대처: GPU 쓰는 앱을 닫거나, `labeler.py resume --batch 16` 처럼 줄여 이어간다
(`nbs=64` 누적이 유효 배치를 맞춰준다).

**WSL 도 똑같다 (2026-09-18 실측).** VRAM 을 넘기면 OOM 없이 공유 메모리로 넘어가
느려지기만 한다 — s 960 b16 이 11.13 GiB 를 잡고 5,704 ms/step(정상의 25배). 모델·해상도별
한도 표는 `docs/experiments.md` 맨 끝 (`wsl/probe_vram.py` 로 잰 것).

### 10. `run.bat` 에 한글을 넣지 말 것

`chcp 65001` 이 걸린 상태에서 cmd.exe 는 배치 파일 위치를 **바이트로 세면서 문자는
멀티바이트로 읽는다.** 그래서 비ASCII 가 든 줄 다음부터 파일 포인터가 줄 중간에
떨어지고, 남은 조각을 명령으로 실행한다.

```
'브라우저에서' is not recognized as an internal or external command
```

안내 문구는 `cli.cmd_review()` 에서 출력한다. `run.bat` 은 **주석까지 ASCII 로** 유지할 것.

### 11. `images.id` 는 AUTOINCREMENT 여야 한다 (2026-09-08 수정됨)

`INTEGER PRIMARY KEY` 만 쓰면 rowid 별칭이라 **마지막 행을 지우면 그 id 가 다음
이미지에 다시 붙는다.** 웹 UI 는 이미지를 id 로만 가리키므로(`/api/images/{id}/file`)
URL 이 그대로인 채 내용만 바뀌고, 브라우저는 캐시해 둔 **예전 이미지**를 새 이미지
자리에 보여준다. 실제로 겪었다 — 파일도 DB 도 멀쩡한데 화면만 다른 사진이었다.

세 겹으로 막아 두었다. 하나라도 되돌리지 말 것.

- `store.py` 스키마: `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `server.py` 이미지 응답: `Cache-Control: no-cache` (ETag 로 304 되니 비용 없음)
- `app.js`: `?v=<updated_at>` 로 URL 자체를 가름 (이미 오염된 캐시 대응)

(옛 DB 는 새 테이블로 복사해 옮겼고 id 값은 그대로다. 옮기기 전 DB 는 외부 백업에 있다.)

### 12. scan 이 검수 UI 에서 추가한 클래스를 지웠다 (2026-09-20 고침)

검수 UI 의 '클래스 추가'(`POST /api/classes` → `store.add_class`)는 **DB 의 classes 표에만**
쓴다. 그런데 `discover_classes` 가 config.yaml 의 `classes` 로 **표를 통째로 지우고 다시 써서**,
scan 을 한 번 돌리면 UI 로 추가한 클래스 이름이 사라졌다(박스는 남아 `cls_7`·`cls_8` 로 보인다).
사용자가 하루에 1,696개를 그린 뒤 겪었고, 이름을 **사용자에게 물어서** 되살렸다(pickup·van).
이름이 없으면 그 박스가 있는 이미지는 저장이 안 되고(`PUT /boxes` 400) 학습에서도 빠진다.

고친 곳:
- `store.set_classes(names)` 는 이제 **목록보다 뒤에 있는 클래스를 지우지 않는다.**
  진짜로 줄일 때만 `drop_extra=True` (그 번호를 쓰는 박스가 있으면 `PUT /api/classes` 가 막는다).
- `discover_classes`·`_ingest_coco` 는 합친 목록(`store.get_classes()`)을 쓴다.
- `POST /api/classes` 가 `classes.txt`(DB 옆)에 전체 목록을 적어 둔다 — 이름을 잃었을 때
  되살릴 기록이다(config.yaml 이 우선이라 평소엔 읽히지 않는다).
- config.yaml 의 `classes` 에도 pickup·van 을 적었다. **뒤에만 붙일 것.**

### 13. 정리(tidy)가 같은 이름의 파일을 덮어썼다 (2026-09-27 고침)

인터넷에서 받은 사진은 `images.jpg`·`images (1).jpg` 처럼 이름이 겹친다. 정리는 겹치면
`이름__<원래 경로 해시>` 로 한 번만 바꿨는데, **같은 이름으로 받은 사진이 같은 자리
(`unlabeled/images (1).jpg`)를 두 번 거쳐 오면 해시까지 같아서** 두 번째 정리가 첫 사진 파일을
덮어썼다(Windows `shutil.move` 는 대상이 있으면 복사로 넘어가 덮어쓴다). 이어서 DB 경로 갱신이
경로 중복으로 실패해 그 행은 '파일 없음' 이 됐고, 함께 도는 prune 이 행을 라벨째 지웠다.

- **피해 (09-27 21:18 정리):** 6639·6641 의 **원래 사진 두 장을 잃었다**(pickup·car 가 있던 웹 사진).
  그 자리에 6676·6686 의 사진이 들어갔고 두 행(라벨 포함)은 사라졌다. 사진과 라벨이 전혀 안 맞는
  상태였다.
- **복구 (09-27, 사용자 확인):** 6676·6686 의 라벨이 09-26 백업과 `labeled/labels/` 에 남아 있어서,
  6639·6641 을 그 라벨(밴+사람)과 실제 크기로 바꿨다(`source='restore'`, 변경 이력에 옛 라벨이 남는다
  — 원래 사진을 찾으면 되살릴 수 있다). 전체 6,567장의 크기·dHash 를 대조해 다른 피해는 없었다.
- **고친 것 (`tidy._free_name`):** 이미지·라벨 이름이 **둘 다** 비어 있을 때까지 `_2`, `_3` 을 더 붙이고,
  옮기기 직전에 대상이 있으면 멈추고, DB 갱신에 실패하면 파일을 제자리로 되돌린다. 라벨까지 보는
  이유는 `a.jpg`·`a.png` 가 같은 `a.txt` 를 쓰기 때문이다. 재현 시험(같은 이름 세 번 정리) 통과.
- **10-02 코드 리뷰로 더 막았다:** 되돌리기마저 실패하면 라벨 .txt 를 파일 옆에 남기고 예외로 멈춘다
  (웹 UI 작업의 뒤이은 prune 이 그 행을 지우지 않게). 라벨 .txt 쓰기 실패는 그 한 장의 오류로만 남기고
  정리를 계속한다(DB 가 원본이다). **코드를 고친 뒤에는 검수 서버를 다시 켜야 반영된다.**

---

## 아키텍처

코드는 `src/autolabel/` 에 있다. 각 파일 맨 위 docstring 에 그 파일이 왜 그렇게
되어 있는지 적어 두었으니 파일 목록은 `ls` 로 보고 docstring 을 읽을 것.

**설계 의도 몇 가지**

- 학습은 매번 사전학습 가중치(`model.base`, 지금 `weights/yolo26s.pt`)에서 **새로**
  미세조정한다. 이어 학습하면 자기학습 루프에서 모델이 자기 오류를 굳힌다(confirmation bias).
- train/val/test 분할은 **DB(`images.split`)에 고정**하고, 같은 장면은 묶어서 한쪽에만
  넣는다 (`split.py`·`grouping.py`, 2026-09-16). 새로 승인된 것만 계층화로 채우므로
  기존 배정은 움직이지 않는다 — 사이클 간 성능 비교가 유효하다. 그 전에는 경로 해시로
  매번 계산해서 `tidy` 가 경로를 바꾸면 배정이 뒤집혔고, 같은 영상의 인접 프레임이
  train·val 로 갈려 점수가 부풀려졌다.
- 자동 승인은 넣지 않았다. 신뢰도가 높아도 사람이 봐야 학습에 들어간다.
- 재스캔은 이미 등록된 이미지를 건너뛴다. UI에서 손본 라벨이 원본으로 되덮이지 않는다.
  같은 스캔 안에서 같은 경로가 두 번 나와도 첫 것만 넣는다. **`INSERT ... ON CONFLICT DO UPDATE`
  뒤에 `lastrowid` 를 쓰지 말 것** — UPDATE 로 끝나면 직전에 넣은 다른 행의 id 가 남는다(sqlite
  3.50 실측, 예전 `ingest_batch` 가 그렇게 박스를 엉뚱한 이미지에 붙일 수 있었다. 지금은
  `DO NOTHING` + `rowcount`). 내용(바이트)이 같은 새 파일은 등록하지 않는다(`dupcheck.py`).

## UI 조작

단축키는 README.md 의 '검수 UI 사용법' 표와 화면 아래 안내줄에 있다.

학습 중 **중단** 버튼은 `trainer.stop` 을 세워 배치 경계에서 빠져나온다.
강제 종료가 아니라 최종 검증·가중치 저장까지 마치고 정상 종료된다.

**편집은 자동 저장된다.** 박스를 하나 그리면 0.6초 뒤(다른 이미지로 넘어갈 때는 그 전에
바로) DB 에 쓰인다 (`app.js` `markDirty`·`flushSave`). UI 를
테스트할 때는 실제 작업물이 바뀐다는 뜻이므로, 먼저 `GET /api/images/{id}` 로
스냅샷을 떠 두고 끝나면 `PUT /api/images/{id}/boxes` 로 되돌릴 것.

**저장은 한 번에 하나만 돈다** (09-23 에 경합을 고쳤다 — 예전에는 저장이 도는 동안 한 수정이
'저장됨' 과 함께 사라졌다). `S.saving` 이 진행 중인 Promise 이고, 도는 동안의 수정은 끝난 뒤
이어서 저장하며, 수정마다 `S.rev` 가 올라 '저장 중에 또 고쳤나' 를 가린다. `flushSave()` 가
false 면(저장 실패) **이미지를 넘기거나 승인하지 않는다** (`saveBeforeLeaving`).
- 응답을 기다리는 동안 이미지가 바뀔 수 있는 곳(SAM·예측·이미지 로드·변경 내역)은 전부 **요청한
  이미지 id 를 붙잡아 두고 달라졌으면 결과를 버린다.** 새로 비동기 호출을 넣을 때도 그렇게 할 것.
- 박스를 **고르기만 하는 클릭**(3px 미만 움직임)은 수정이 아니다. **클래스를 실제로 바꾸면**
  신뢰도를 지우고 '수동' 이 된다(09-24 전에는 옛 신뢰도가 남았다).
- 이어하기 버튼(`/api/resume-point`)은 페이지를 열 때와 작업이 끝날 때만 새로 고친다 — 저장마다
  부르면 서버가 runs/ 의 체크포인트를 열어 0.24~0.36초씩 들었다(09-24 에 바꿈).
- 이런 경합을 다시 시험할 때는 `Store.set_boxes` 에 `time.sleep` 을 끼운 서버를 **DB 복사본**
  으로 띄우면 된다 (sqlite `backup()` 으로 뜨면 검수 서버가 떠 있어도 일관된 스냅샷이다).

### 변경 이력·같은 자리 일괄 수정·예측 비교 (2026-09-23 추가)

사용자 문서는 README '검수 UI 사용법'. 작업자가 알아야 할 것만:

- **박스 변경 이력은 `box_history` 표 하나다** (`store._replace_boxes`). 박스를 바꾸는 길은
  전부 여기를 지난다 — `set_boxes`(UI 저장·자동 라벨링·되돌리기), `set_class_bulk`(일괄),
  `revert_batch`. 한 줄에 그 이미지의 박스 목록 **전후를 JSON 으로 통째로** 둔다
  (`[[cls,cx,cy,w,h,conf,origin], ...]`) — boxes 행 id 는 저장마다 바뀌어 기준이 못 된다.
  **바뀌기 전이 비어 있으면 남기지 않는다**(되돌릴 게 없고, 자동 라벨링이 미라벨 수천 장에
  쓰는 몫이 쌓일 뿐). 내용이 같은 저장(`_box_key` 로 1e-7 반올림 비교)도 남기지 않는다.
- **새로 박스를 바꾸는 경로를 만들면 `_replace_boxes` 를 거치게 할 것.** 직접 `DELETE/INSERT`
  하면 이력에서 빠진다. `_replace_boxes` 는 `_write_lock` 을 **부르는 쪽이** 쥐고 있어야
  한다(안에서 다시 잡으면 멈춘다 — threading.Lock 은 재진입이 안 된다).
- **DB 자동 백업** (`store.backup_daily`, `create_app` 에서): 하루 한 번 `<DB 폴더>/backups/
  <DB 이름>_날짜_시각.db`, 최근 `backup.keep`(기본 14)개만 남긴다 — **이 이름 꼴의 파일만**
  지운다. 시험 서버를 띄울 때는 `backup.keep: 0` 으로 끌 것(복사본 옆에 쌓인다).
- **같은 자리 일괄 수정** (`store.same_place` → `set_class_bulk`, `/api/images/{id}/same-place`
  미리 보기 · `/apply`): 같은 `grp` 의 **다른** 이미지에서 이미지마다 IoU 최고인 박스 하나를
  짝으로 본다. 기본은 원래 클래스(`cls0`, 이미지를 열 때의 클래스)가 같은 것만. 적용 때는
  같은 조건으로 **다시 찾고**, 미리 본 수(`expected`)와 다르면 409 — 그 사이 누가 고쳤으면
  사람이 다시 보게. 묶음(`batch`)째 되돌리기는 **지금 박스가 일괄 직후와 같은 이미지만**
  되돌린다. 고정 CCTV 의 van 자리(0.978, 0.123)로 시험하니 0.85 에서 789장(0.95 725 / 0.5 794)
  이었고, 적용·되돌리기가 정확히 맞았다(복사본). **묶음이 정해지기 전(다음 학습 전)의 새 이미지는
  쓸 수 없다**(grp 가 없다).
- **예측 비교** (`predictor.LiveDetector`, `GET /api/images/{id}/predict`): 운영 가중치로 한 장
  추론만 하고 **DB 에는 안 쓴다.** 라벨과의 비교는 브라우저(`app.js computeFindings`)가 한다 —
  박스를 고칠 때마다 모델을 다시 돌리지 않게. 규칙은 `tools/label_qa.py` 와 같다(짝 IoU 0.5,
  가까움 0.3, 확신 0.5, '못 봄' 은 짧은 변 24px 이상). 다른 점 둘: 예측 하한이 label_qa 의
  0.10 이 아니라 `predict.conf`(0.25)이고, **모델이 모르는 클래스의 라벨은 비교하지 않는다**
  (visible-3 은 pickup·van 을 모른다 — `unknown_classes`).
  - **GPU 작업(학습·자동 라벨링)이 도는 중이면 CPU 로** 돈다(`jobs.gpu_busy`). Ultralytics 는
    첫 추론 때 장치를 정하고 바꾸지 않으므로 장치가 바뀌면 모델을 다시 올린다. 가중치 파일의
    수정 시각이 바뀌어도 다시 올린다. 실측: GPU 65\~160ms, CPU 100\~280ms.
  - **올릴 때 빈 이미지로 예열한다.** 한 번은 첫 추론이 CUDA 초기화로 16초 걸리며 NMS 시간
    제한(2.05초)을 넘겨 **Ultralytics 가 검출을 도중에 잘랐다**('NMS time limit exceeded' 경고만).
    다시 재니 1.8초·잘림 없음이라 자동 라벨링(`AutoLabeler`)은 손대지 않았다 — 조건은 모른다.
  - 학습·자동 라벨링을 시작하면 `server.free_gpu()` 가 SAM 과 이 모델을 **모든 탭 것까지** 내린다.
- **'드문 클래스 먼저' 정렬** (`order=rare`, `store._list_rare`·`class_rarity`): 희소도는 박스 수가
  아니라 **그 클래스가 나온 승인 이미지의 장면 묶음(grp) 수**로 센다 — 가장 드문 클래스를 1 로,
  나머지는 (가장 적은 묶음 수 / 자기 묶음 수). 묶음 규칙을 고친 09-24 복사본 기준: van 309묶음(1.0),
  pickup 401, scooter 545, truck 549, motorcycle 556, bus 601, bicycle 1,024, car 1,710, person 2,455
  (DB 는 다음 학습 때 이 값이 된다). 이미지 점수 = 박스마다 (신뢰도 × 희소도)의 최댓값(사람이 그린
  박스는 신뢰도 1).
  SQL 로는 못 매겨서 같은 조건의 전체를 파이썬으로 줄 세운 뒤 잘라 준다 — 목록 묶음(100개씩)과
  맞도록 `count_images` 와 같은 WHERE(`_list_where`)를 쓴다.
- **클래스 필터** (`GET /api/images?cls=2,4` · `&empty=1`, 09-21): 둘 이상 고르면 **전부 가진**
  이미지만(교집합), '박스 없음' 은 클래스 선택과 함께 못 쓴다. 거르면 상태별 개수를 못 쓰므로
  `count_images` 로 **같은 WHERE 로** 다시 센다 — 목록 묶음이 그 수로 나뉘어서 어긋나면 틀어진다.

### 이미지를 밖에서 편집했다면 `labeler.py check`

개인정보 블러 같은 이유로 원본을 편집 툴로 손보는 일이 있다. 파일명·확장자·픽셀
크기만 그대로면 문제없지만, 셋 중 하나가 바뀌면 조용히 어긋난다.

- **확장자가 바뀌면**(.png -> .jpg 로 저장) 경로가 달라져 DB 참조가 끊긴다.
- **크기가 바뀌면** 박스는 정규화 좌표라 값은 살아 있지만, 검수 UI 가 DB 의
  width/height 로 화면 좌표를 계산하므로(`app.js` `boxRect`) 박스가 어긋나 보인다.
  종횡비까지 달라지면 박스 자체가 틀어진다.

`labeler.py check` 가 이 셋(파일 없음 / 열리지 않음 / 크기 불일치)을 전수 검사한다.
크기만 달라졌다면 `--fix-size` 로 DB 를 실제 파일에 맞춘다(박스 값은 건드리지 않는다).

(09-21 에 id 1131 한 건을 `--fix-size` 로 고쳤다 — 박스 값은 원래 맞았고 화면 표시만 틀어져 있었다.)

(예전에는 로보플로 원본도 같이 고쳐야 했다 — 다음 회차 내보내기에서 원본이 다시 들어오므로. 09-28 부터
로보플로에 데이터가 없어 로컬만 고치면 된다.)

### SAM (`C` 모드)

객체를 한 번 클릭하면 박스가 생기고, 끌면 그 범위를 다듬는다. `R` 은 이미 고른
박스를 다듬는다. `segmenter.py` 참고 — 모델 선택 근거와 실측치가 거기 있다.

- **마스크는 저장하지 않는다.** DB 는 박스만 들고 있고(`boxes` 테이블), SAM 은
  외접 박스만 내놓는다. 그래서 스키마 변경이 없다. 나온 박스는 `origin='sam'`,
  `conf=null` 이라 UI 에서 사람이 그린 것과 같은 취급을 받는다.
- **`C` 모드에서는 모든 좌클릭이 SAM 으로 간다.** 기존 박스 위 클릭을 선택·이동으로
  넘기면 객체가 빽빽한 장면에서 대부분의 클릭이 SAM 에 닿지 않아서 그렇게 했다.
  고르거나 옮기려면 `B` 모드나 오른쪽 박스 목록을 쓴다.
- **학습·자동 라벨링을 시작하면 SAM 을 내린다** (`server.free_gpu`, 모든 탭 것까지).
  8GB 카드에서 0.55GB 도 아깝고, 모자라면 에러 없이 느려지기만 한다(함정 9).
  수동으로 내리려면 `POST /api/sam/release`.
- **GPU 작업이 도는 동안 C·R 은 CPU 로 돈다** (10-25, `jobs.gpu_busy`, 예측 비교와 같은 방식). 예전에는
  내린 SAM 이 다음 클릭에 GPU 로 다시 올라갔다. 장치가 바뀌면 내리고 다시 올린다. 시험: 학습 중 0 MiB,
  끝난 뒤 GPU 284 MiB, 다시 학습 중 32 MiB(파이썬에서 닿는 CUDA 텐서 0개 — cuBLAS 작업 공간으로 보임).
- Ultralytics 의 SAM Predictor 는 한 번에 이미지 하나만 문다. 그래서 인코딩 캐시
  깊이가 1 이고 락으로 직렬화한다. 한 사람이 한 이미지를 보며 여러 번 찍는
  패턴이라 충분하다.

---

## 다음에 할 일

0~2. **지난 학습 기록은 [docs/experiments.md](docs/experiments.md) 로 옮겼다.**
   결론만: yolo26n → **yolo26s** 로 갈아탔고(같은 조건 60 epoch 에서 mAP50-95
   0.6495 → 0.7288), s 는 **`predict.imgsz` 640 이 정점**이다. scooter 붕괴는
   학습 부족이었고 해결됐다. close_mosaic 마지막 30 epoch 은 두 번 연속 이득이
   없었다. 측정표·VRAM 실측·학습별 상세는 그 파일에 있다.

3. **미검토 제안**
   - **해상도.** 640 입력에서 박스의 **23.3%가 32px 미만**이다(960 에서 12.2%, 1280 에서
     7.2%, 09-08 측정). 추론 해상도만 바꿔 본 결과는 모델마다 반대였다 — yolo26n 은 960 이
     정점(640 0.7050 → 960 0.7193)이라 한때 `predict.imgsz: 960` 을 썼지만, **yolo26s 는
     640 이 정점**(960 −0.0018, 1280 −0.0501)이라 지금은 640 이다. 표는 `config.yaml` 주석과
     `docs/experiments.md`.

     학습 자체를 960 으로 올리는 것은 n 으로만 해 봤다(`tools/finetune.py`,
     `models/best_ft960.pt`). s 로는 스텝 속도만 쟀고(b8 5.15GiB) 학습은 안 했다. 해 본다면
     batch 16 이하로(메모리 ∝ batch × imgsz²), epoch 시간은 imgsz² 비율대로 약 2.3배를
     예상할 것(추정, 실측 아님).

   - SAHI: 작은 박스가 문제라면 960 학습을 먼저 시험해 볼 것. 그래도 부족하면 검토.
   - **중복 유입 — 재압축된 내보내기는 아직 못 잡는다.** `scan` 은 바이트가 같은 새 파일을 막고
     (`dupcheck.py`, 09-23), 박스가 다른 중복은 등록하지 않고 리포트에서 따로 센다(건너뛸지 갱신할지
     사람이 정한다). 로보플로에서 전처리를 켜고 내보내면 바이트가 달라지므로 그때는 **디코딩된 픽셀
     배열의 해시**가 필요하다. 파일명·dHash 로는 안 된다(설계 근거는 `dupcheck.py` 머리말).

4. **9클래스로 다시 학습하기 — 지금 운영 모델(visible-3)은 pickup·van 을 모른다.** 학습하면
   `tools/test_baseline.py --weights <새 가중치> --domains` 로 09-27 기준선과 클래스별로 비교하고,
   그 모델로 `tools/label_qa.py` 를 돌려 미뤄 둔 것을 거른다 — car 로 들어간 레이·박스형 경차
   (van 후보), 176px 아래의 truck·pickup·car 판정. 검수 UI 의 예측 비교(P)도 그때부터 pickup·van
   을 본다. 학습이 묶음을 다시 계산하므로 `g000001` 이 360묶음으로 나뉘는 것도 이때다.
   - **사용자 결정 거리:** 예전 `g000001` 에 섞여 train 에 갇혔던 CCTV 가 아닌 사진(약 640장)을
     val·test 로도 보내려면 `labeler.py split --reset --yes` 가 필요하다 — 그러면 지금까지의 점수
     비교가 끊긴다.

## 사용자에 대해

- 한국어로 소통한다.
- 검증을 중시한다. 수치와 근거를 제시하면 잘 받아들이고, 추측을 사실처럼 말하면
  정확히 짚어낸다. **모르면 코드나 실측으로 확인하고 답할 것.**
- 원본 데이터를 옮기거나 지우는 작업은 반드시 먼저 확인을 받을 것.
