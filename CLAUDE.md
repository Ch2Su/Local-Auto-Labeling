# 로컬 자동 이미지 라벨링 — 작업 인수인계

소량의 손라벨 데이터로 YOLO 를 학습시키고 → 나머지를 자동 라벨링 → 웹에서 검수 → 검수분을 합쳐 재학습하는 루프.
전부 로컬에서 돈다. **방향 (2026-11-03 사용자 확인): CCTV 객체 검출 고도화 — LLM 연동 유고(이상 상황) 감지 시스템의 검출부를 만든다.**
유고 감지 시스템 자체(LLM 연동)는 다른 세션에서 만든다 — 이 저장소는 검출 모델·라벨·학습만 맡는다. 09-18 의 '어떤 이미지든 탐지' 방향은 대체됐고, 드론·비CCTV 데이터와 평가 도메인을 어떻게 다룰지는 정해진 것이 없으니 필요하면 사용자에게 물을 것. 분할·평가에서 특정 도메인·카메라가 한쪽(train/val/test)에 몰리지 않게 할 것은 그대로다.

사용자 문서는 [README.md](README.md). 이 파일은 작업자가 **매번** 알아야 할 규칙만 담는다(11-02 에 105KB 에서 줄였다).
경위·실측·사건 기록은 줄이기 전 판 `git show f7ea49a:CLAUDE.md`(함정 번호와 큰 절 이름은 같다 — SAM·변경 이력·check 는
'UI 조작' 에 합쳤다), 학습·평가 기록은 [docs/experiments.md](docs/experiments.md), 라벨 판정 기준 전문과 차종별 실물 예시는
[docs/label-rules.md](docs/label-rules.md).

**Gemini 와 함께 쓴다 (2026-10-25 사용자 결정).** 코드·설정·DB 수정은 **Claude**, 코드 리뷰·읽기 전용 실측 검증·아이디어는
**Gemini** 가 맡는다. 칠판은 `docs/collab.md` — git 에 담지 않는 로컬 파일이고 150줄 안쪽으로 유지한다. 턴을 시작할 때
`[1. 현재 작업 지시]` 를 읽고, 끝나면 `[2. Claude 작업 보고]` 에 시작 시점 `HEAD`·수정 파일·내용을 적어 상태를
`WAITING_VERIFY` 로 둔다. 오래 남길 규칙·기록은 칠판이 아니라 이 파일과 docs/ 에 적는다.

## 먼저 물을 것

- 원본 데이터(이미지·시드·라벨)를 옮기거나 지우는 일, **DB 재생성**(함정 5), 학습 체크포인트·결과 폴더를 지우는 일.
- **push** — 훅은 push 하지 않는다. 사용자가 요청할 때만이고, 저장소가 public 이라 올리는 순간 누구나 본다.
- Runpod pod 만들기 — GPU·예상 시간·비용을 말하고 허락받는다. 로보플로에 올리기·프로젝트 만들기·복원 — 스토리지
  크레딧이 다시 나간다.
- C: 에 백업을 새로 뜨기 전에 이전 날짜 폴더를 지울지(C: 는 남은 공간이 적다).
- 로컬에서 x·l 960 을 재는 일 — VRAM 이 넘쳐 느려지니, 사용자가 GPU 를 쓰는 중이면.
- GitHub 의 옛 저장소 `Local-Auto-Labeling-old` 를 지우거나 공개로 바꾸는 일 — 옛 이력·원래 메일이 그대로다.
- **`labeler.py split --reset --yes` 는 사용자가 직접 돌린다** — 리셋하면 학습에 쓴 이미지가 test 로 갈 수 있어 지금까지의
  모델(visible-3 포함) test 점수를 쓸 수 없게 된다.

키는 환경 변수에만 둔다 — **`RUNPOD_API_KEY` 같은 키 값을 읽거나 출력하지 말 것.**

---

## 환경 (이미 구성됨)

| | |
|---|---|
| Python | 3.14.3 (`.venv`) |
| PyTorch | 2.14.0+cu130 |
| Ultralytics | 8.4.142 |
| GPU | RTX 5060 Laptop 8GB, sm_120 (Blackwell) |
| OS | Windows 11 |

`python` 은 반드시 `.venv\Scripts\python.exe` 를 쓴다 — 시스템 Python 에는 아무것도 없다. Python 3.14 + sm_120 이라
**PyTorch 는 cu130 인덱스에서만** 받는다. `requirements.txt` 가 `+cu130` 까지 고정해서(11-01) CPU 빌드로 조용히 바뀌는 대신
'No matching distribution' 으로 멈춘다. 다시 깔 때는 아래를 돌린다(CPU 빌드가 깔렸어도 이것으로 바뀐다 — `==2.14.0` 만
적으면 '이미 설치됨' 으로 넘어간다):

```bash
.venv/Scripts/python.exe -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

블루스크린(09-12 `0x133 DPC_WATCHDOG_VIOLATION`, Ultralytics `val` 무렵)과 버그체크 0 종료(09-16, WSL 학습을 정상 중단한
뒤)가 있었다 — 원인은 모르고, 같은 작업을 그 뒤 여러 번 돌렸을 때는 문제없었다.

**Runpod (클라우드 GPU 대여).** MCP 서버 `runpod` 는 11-02 에 뺐다(필요 없는 MCP 정리 — 그날 pod·볼륨 0개를 확인했다).
다시 쓰려면 PowerShell 에서 아래를 돌리고 새 세션을 연다 — 키는 화면에 안 나오고 `~/.claude.json` 에 그대로 적힌다(공식
플러그인의 OAuth 는 이 데스크톱 앱에서 안 된다). 파일 전송·SSH 는 Windows 기본 `ssh`·`scp`·`tar`(`~/.ssh/id_ed25519` 등록됨).
`claude mcp add --scope user --transport http runpod https://mcp.getrunpod.io/ --header "Authorization: Bearer $env:RUNPOD_API_KEY"`
- 초 단위 과금이고 멈춘 pod 도 볼륨 요금($0.20/GB/월)이 나간다. **멈춘 pod 는 다시 안 켜질 수 있다** — 결과는 바로 받고 다
  쓰면 지운다. pod 는 스스로 못 멈춘다(pod 안의 키는 Unauthorized) — MCP 의 `pod-action stop` 이나 콘솔에서 멈춘다.
- **고른다면 RTX 6000 Ada** — x960 b16 한 스텝이 A40 869ms / Ada 513ms / A100 423ms 라 A40 과 총비용은 같고 시간은 60%다
  (스텝은 RAM 캐시로 잴 것 — 아니면 디코딩을 잰다). 09-18 A40 비교(300 스케줄의 60 epoch) 결과는 `runs/compare_a40/`·
  docs/experiments.md — resume.pt 는 pod 와 함께 사라져 300 epoch 은 처음부터다(비용도 그 기준으로 말할 것).
- 절차는 `runs/runpod/`(묶음 tar → scp → `pod_probe.sh` → `pod_compare.sh`). 묶음을 만든 `make_bundle.py` 는 저장소에 없다 —
  DB 로 `data/visible/dataset` 을 만들어 tar 로 묶는 스크립트를 새로 짤 것.
- 학습 전에 **`pip install --break-system-packages -U pillow` 와 `labels/*.cache` 삭제** — 이미지의 Pillow 11.0.0 은 확장자만
  .png 인 삼성 폰 JPEG 29장을 '손상'으로 빼고 AVIF 도 못 읽는다. `/workspace` 는 네트워크 FS 라 데이터셋은 `/root` 에 풀고,
  컨테이너 RAM 한도(A40 46.6GiB)에 ram 캐시·워커가 겹치면 SIGKILL 된다. SSH 로 `a && b && setsid nohup ... &` 처럼 목록째
  띄우면 ssh 가 안 돌아온다(학습은 산다).

**Roboflow.** 09-28 에 claude.ai 커넥터로 붙였는데 11-02 에는 커넥터 목록에 없다(다시 쓰려면 claude.ai 커넥터 설정에서).
**워크스페이스는 비어 있다** — 스토리지 크레딧 때문에 사용자가 09-28 에 데이터를 전부 지웠고 휴지통도 10-02 에 비웠다.
**그래서 로컬이 유일한 사본이다**(아래 '다른 디스크 백업').
- 데이터를 받을 때는 **COCO 로**(함정 4). Universe 검색은 조회만 한다(크레딧이 드는지는 확인하지 않았다).
- **남의 데이터셋을 넣을 때:** 시드(COCO) 경로는 라벨을 **검수 없이 곧바로 approved** 로 넣고, config 의 `classes` 와 이름이
  다른 카테고리(`pedestrian`·`SUV`·`Pickup Truck` 등)는 **버린다**(`coco_io.read_coco`). 미라벨 폴더로 넣으면 라벨을 읽지
  않는다. 그래서 카테고리 이름을 먼저 우리 클래스로 바꾸고, 우리 클래스를 **빠짐없이** 라벨한 데이터인지 볼 것 — 사람이
  빠진 이미지가 승인되면 모델이 사람을 배경으로 배운다. Objects365 의 `Scooter` 는 오토바이형이고 우리 scooter 는 전동 킥보드다.

**Objects365 (10-02 BAAI 에서 받음) — 11-03 에 프로젝트에서 뺐다**(사용자 결정 — 백업한 뒤 휴지통으로 보내 비웠다). 지금 DB 에
o365 행은 없고 `data/visible/unlabeled/o365/` 도 없다. 원본은 `E:\datasets\objects365.tar`(331GB, 저장소 밖 — patch 압축 51개·json·
`o365_index.pkl`·license.txt, 풀린 사진 없음)에 있었는데 **11-04 에는 그 폴더에 없었다**(옮겼는지 지웠는지 모름) — **`tools/import_o365.py` 는 그 tar 를 찾아 다시 풀기 전에는 못 돈다.** 다시 넣게 되면 아래를 볼 것.
- 10-03 에 500장을 넣었었다(id 6726–7225, source·origin `o365`, 넣기 전 DB `labeling.db.bak_o365_20261003_024915`) — 뺄 때 승인 29
  / 대기 429 / 건너뜀 42. 11-03 에 뺄 무렵 뜬 DB 백업은 `labeling.db.bak_trash_20261103_004945`·`labeling.db.bak_trash29_20261103_125351`.
  **넣은 뒤 `labeler.py label --all` 을 쓰지 말 것** — 검수 대기 라벨을 예측으로 덮는다(웹 UI 의 자동 라벨링은
  `include_pending: false` 라 안전하다).
- 넣을 때의 사용자 결정: Scooter → motorcycle, isfake·isreflected 박스는 버리고, 우리 클래스에 iscrowd 가 있거나 Ambulance 가
  있는 사진은 뺀다. O365 의 Van 은 우리 기준과 다르고(SUV·캐노피 픽업·버스 뒷면이 섞임 — 11-02 부터 car 로 넣는다) Pickup
  Truck 에도 세단·SUV 오표기가 있으니 **검수를 거칠 것**(Van 으로 들어온 캐노피 픽업은 pickup, 버스 뒷면은 bus 로). O365 의
  `Hoverboard`(세그웨이류)도 우리 scooter 가 아니다. 이미지는 Flickr CC 라 NonCommercial 이 섞여 있다.
- `tools/test_baseline.py` 의 all·nodrone 은 source='o365' 를 뺀다(O365 만은 `o365`, 전부는 `all_o365`).
  `objects365_v1_…` 이름은 연번 계열로 묶지 않는다(`grouping._INDEPENDENT` — 아니면 전부 한 묶음이 된다).
- json(5.14GB)을 흘려 읽을 때는 객체째 잘라 `json.loads` 할 것 — 뒤쪽 주석은 필드 순서가 다르다.

## git

**2026-10-02 에 저장소로 만들었다**(브랜치 `main`, `core.autocrlf=false` — 파일을 바이트 그대로 둔다). 담는 것은 코드·설정·
문서와 `runs/` 의 텍스트 기록(results.csv·args.yaml·`test_baseline/history.jsonl`·스크립트)뿐이다. `data/`·`models/`·
`weights/`·`export/`·`.venv`·`graft/`·그림·로그·tar 는 `.gitignore` 로 뺐다 — **데이터와 DB 는 git 이 아니라 백업으로 지킨다.**
- **원격은 GitHub `Ch2Su/Local-Auto-Labeling` — public 이다(11-02).** **push 하는 순간 누구나 본다** — 올리기 전에 담긴
  내용을 볼 것. **올린 커밋은 다시 쓰지 말 것**(amend·rebase 뒤 force push 금지 — 칠판이 커밋을 해시로 가리킨다). 11-02 에
  커밋 메일을 감추느라 이력을 한 번 다시 쓴 것(사용자 결정)만 예외다. 그 옛 이력은 로컬 `refs/archive/pre-noreply/main` 에
  있다 — **이 ref 는 push 하지 말 것**(옛→새 해시 대응표 `.git/noreply-rewrite-20261102.map`, 옛 저장소
  `Local-Auto-Labeling-old` 는 private). **public 에 올라간 것은 이력을 고쳐 force push 해도 GitHub 에 옛 커밋이 해시로
  남는다**(11-02 에 새 저장소로 옮긴 이유) — 키·개인정보가 올라갔으면 지워졌다고 여기지 말고 바로 사용자에게 알릴 것.
  11-08 에 GitHub 계정 이름이 `chisu0208` → `Ch2Su` 로 바뀌어(id 42445310 그대로) origin 주소도 바꿨다(사용자 결정). 옛 주소는
  아직 리다이렉트되지만 옛 이름을 남이 가져가 같은 이름의 저장소를 만들면 끊긴다 — 다른 클론도 `git remote set-url` 할 것.
- **이 저장소의 `user.email` 은 `.git/config` 의 noreply 다**(`42445310+chisu0208@users.noreply.github.com`). 새로 클론한 곳에서는
  자동 커밋 훅이 돌기 전에 같은 설정을 할 것 — 안 하면 원래 메일이 다시 커밋에 들어간다. 계정 이름이 바뀐 뒤에도 이 메일의
  커밋은 앞 번호(id)로 `Ch2Su` 계정에 연결된다(11-08 에 7e5f275 로 확인).
- **턴이 끝날 때마다 자동 커밋된다**(`.claude/settings.json` 의 Stop 훅 → `.claude/hooks/autocommit.sh`, 추적 파일이라 클론한
  PC 에서도 돈다). 바뀐 것을 임시 색인으로 `git add -A` 해 '자동 커밋: <파일 이름들>' 으로 남기고, 5MB 넘는 파일은 빼고
  알린다. 머지·리베이스·cherry-pick·revert 중, HEAD 가 브랜치에 없을 때, 충돌이 남았거나 담을 내용에 충돌 표시·**키 모양
  줄**이 있을 때, 200개 넘게 바뀌었을 때는 알리고 건너뛴다. 워크트리에서 일하면 그 워크트리를 커밋한다. 사용자가 직접 고친
  파일도 같이 담긴다. 의미 있는 단위로 남기려면 턴이 끝나기 전에 직접 커밋한다 — 그러면 훅은 할 일이 없다.
- **훅을 시험할 때는** 시험 저장소를 만들어 `CLAUDE_PROJECT_DIR=<시험 저장소> bash <시험 저장소>/.claude/hooks/autocommit.sh`
  로만 돌릴 것 — 기본값이 이 저장소라 그냥 돌리면 실제로 커밋한다.
- 코드 리뷰·수정 범위는 `git diff HEAD~N`·`git log` 로 잡는다(작업 트리는 턴마다 비워진다).
- **LICENSE 는 GNU AGPL-3.0 이다**(11-02 — 코드가 Ultralytics(AGPL-3.0)를 import 한다). 본문은 GitHub `licenses/agpl-3.0`
  그대로이고 고치면 안 된다.
- **비밀:** GitHub 비밀 검사·push 보호를 켰다(11-02) — GitHub 이 아는 종류의 키만 막는다. 그 밖에는 `.gitignore` 맨 끝의 비밀
  파일 이름 블록(`.env`·`*.pem`·`id_ed25519*`·`credentials.*`·`.mcp.json`·`**/.gemini/settings.json` 등, runs/ 밖의 데이터
  사본·영상·로그·압축)과 자동 커밋 훅의 키 모양 검사(정해진 패턴만)뿐이다. **이름 블록은 추적되는 파일을 못 막는다** —
  `.claude/settings.json` 의 `env`·runs/ 화이트리스트의 .json·.yaml 메모에 키를 적으면 훅의 검사만 남고, 손으로 하는 커밋은 그것도
  거치지 않는다. 그래서 키는 환경 변수에만 둔다. 새 비밀 패턴은 그 블록에 넣을 것 — runs/ 화이트리스트보다 **뒤에** 있어야
  runs/ 아래에서도 먹는다. `runs/runpod/` 의 json·`.claude/worktrees/` 도 뺀다.
- **줄끝:** `.gitattributes` 가 `*.sh` 만 LF 로 고정하고, 나머지는 CRLF·LF 가 섞인 채 바이트 그대로다(이 파일은 CRLF).
  작업 트리의 .sh 를 CRLF 로 저장하면 git 에는 안 보이는데 WSL 이 `$'\r'` 오류를 낸다 — `git ls-files --eol -- '*.sh'` 의
  `w/crlf` 로 찾아 `sed -i 's/\r$//' <파일>`.
- **GitHub 은 한 문단(표는 한 칸) 안의 `~` 두 개를 취소선으로 그린다** — 같은 문단에 `~` 가 또 있으면 `\~` 로 쓸 것(인라인
  코드·코드 블록 안은 괜찮다). 확인은 그 파일을 `gh api markdown`(mode `markdown`)으로 그려 `<del>` 이 없는지 본다.

## 코드 찾기 — graft

`graft/` 에 이 저장소의 코드 그래프가 있다(`src/autolabel/`·`tools/`·`wsl/`·`labeler.py`·`app.js`). 파일을 통째로 읽기 전에
위치와 호출 관계를 본다 — 로컬·무료·1초 안팎이다. `graft` 도 `node` 도 PATH 에 없어서 전체 경로로 부른다:

```bash
PATH="/c/Program Files/nodejs:$PATH" node "C:/Users/chisu/AppData/Roaming/npm/node_modules/@nanonets/graft/dist/cli.js" <명령>
#   ask "<질문>" --source | grep "<심볼>" | skeleton <파일> | callers <심볼> --depth 2 | map
```

- 편집 뒤 `build` 는 필요 없다(조회할 때 바뀐 파일만 다시 읽는다). `graft/` 아래 카드는 턴 끝에야 갱신되니 직접 grep 하지 말 것.
- `callers` 는 이름이 겹치면 호출처를 빠뜨린다 — 빠짐없이 보려면 `graft grep`.
- 인덱스에 없는 것(`config.yaml`·`CLAUDE.md`·`docs/`·`index.html`·`style.css`·`runs/runpod/*.sh`)은 평소처럼 읽는다.

## 프로젝트(데이터셋) 탭

지금 프로젝트는 **가시광 하나다**(열화상·화재는 09-28 에 뺐고, 화재 데이터는 이제 어디에도 없다). 목록은 `projects.yaml`,
프로젝트 하나 = config 파일 하나다(데이터 폴더·DB·가중치·`runs` 이름이 config 로 갈린다). 항목이 하나면 탭 줄을 그리지
않는다. `labeler.py review` 는 모든 프로젝트를 `/p/<이름>/` 에 띄우고(`/` 는 첫 프로젝트로, `/api/projects` 는 탭 목록),
`labeler.py --config <설정> review` 는 그 하나만 띄운다(탭 없음).
- 서버 안에서 앱을 프로젝트마다 따로 만든다(`server.create_multi_app` — Store·작업큐·SAM 이 따로다). **'현재 프로젝트' 를 서버에
  하나 두고 바꾸는 방식은 쓰지 말 것** — 자동 저장 때문에 브라우저 탭 두 개가 엉뚱한 프로젝트에 저장한다.
- **GPU 작업(predict·train·resume·train_wsl·resume_wsl)은 프로젝트를 가로질러 하나만 돈다**(`jobs._claim_gpu`, 막히면 409).
  SAM 도 프로세스에 하나만 올린다(`segmenter._active`).
- **웹 UI 의 API 주소는 전부 현재 경로 기준이다**(`app.js` 의 `BASE`·`url()`) — 절대경로 `/api/...` 를 새로 쓰면 탭에서 깨진다
  (탭 목록만 루트의 `/api/projects` 를 부른다).
- 다른 명령은 `labeler.py --config <설정> scan` 처럼 고른다(`--config` 는 하위 명령 **앞**). 새 프로젝트의 클래스는 첫 `scan`
  때 DB 에 들어간다. 새 프로젝트는 `data:`·`model.weights`·`train.overrides.name`·`wsl.dataset`·`wsl.name` 을 **전부** 새
  이름으로 — 하나라도 겹치면 가중치·학습 폴더·WSL 데이터셋을 서로 덮는다.

## 명령·도구

명령은 `labeler.py --help`(scan·train·label·resume·review·tidy·check·split·dupcheck·prune·purge·export·status). 보조 도구는
`tools/` 에 있다(어느 폴더에서 돌려도 저장소의 `config.yaml`·`src/` 를 찾는다).

| | |
|---|---|
| `tools/test_baseline.py` | 지금 DB 라벨로 test 를 잰다(DB 읽기 전용). `--domains` 로 전체·CCTV·드론·비CCTV·드론 제외를 한 번에 재 `runs/test_baseline/history.jsonl` 에 클래스별로 쌓고, `--history` 로 표를 본다. CCTV·드론·비CCTV 는 11-04 부터 `images.domain`(손 분류, `tools/test_tiles.py` 도 11-05 부터 같다). 모델 클래스 순서가 DB 와 다르면 멈춘다 |
| `tools/label_qa.py` | 승인 라벨과 예측이 어긋나는 곳, 라벨끼리 겹친 곳(IoU ≥ 0.85 — `--dup-only` 면 모델 없이)을 후보 CSV·대조 시트로(DB 읽기 전용) |
| `tools/label_qa_claude.py` | missing 후보를 Claude API 로 거르는 파일럿 — API 키 필요, 아직 한 번도 호출 안 함 |
| `tools/merge_class.py` | 맨 끝 클래스를 다른 클래스로 합친다(이력을 남긴다. 기본은 보기만, `--apply` 때 DB 를 먼저 뜬다. config.yaml 에서 먼저 빼야 돈다) |
| `tools/import_o365.py` | Objects365 사진을 검수 대기로 넣는다(위 'Objects365') |
| `tools/import_virat.py` | VIRAT 영상에서 장면마다 프레임을 뽑아 검수 대기로 넣는다 — 운영 모델 예측 + 모델이 그 자리에서 못 찾은 VIRAT 박스, 장면 = 카메라 `virat-XXYY`(docs/provenance.md 'VIRAT 가져오기'). 기본은 보기만·`--apply`, `--per-scene` 을 올려 다시 돌리면 모자란 만큼만 기존 프레임과 떨어진 곳에서 더 넣는다. 이미 넣은 검수 대기를 지금 모델로 다시 라벨링하는 것은 `--relabel`(사람이 고친 이미지는 건너뛴다, VIRAT 박스 없이는 `--model-only` — 0503 은 그렇게 했다). `cameras.yaml` 의 VIRAT 카메라에 `crop:` 이 있으면 그 영역만 쓴다(0503·0500·0101) — 이미 넣은 이미지는 `--recrop`(같은 행의 파일·크기를 바꾼다, `store.replace_file` — **돌린 뒤 검수 화면을 새로 고칠 것**: 열려 있던 옛 화면이 옛 기준 박스를 잘린 이미지에 저장할 수 있다). 이미 자른 장면을 더 좁힐 때는 `--keep-boxes`(다시 라벨링하지 않고 박스만 옮긴다 — 폭이 바뀌면 모델 예측이 달라진다). **`labeler.py label --all` 금지**(VIRAT 박스까지 예측으로 덮는다) |
| `tools/import_roadplus.py` | ROAD PLUS(고속도로 교통정보) CCTV 를 OBS 로 녹화한 파일에서 **전체 화면으로 재생된 구간만** 골라(사이트 화면·빈 화면은 뺀다) 영상 영역을 잘라 높이 720 으로, `--every` 초마다 `unlabeled/<카메라>/` 에 extract_frames 와 같은 이름으로 쓴다(2026-11-09). 시각은 파일 수정 시각 − 길이. 다시 튼 구간은 건너뛴다. 카메라는 `cameras.yaml` 에 먼저(kind screen). 그다음 scan → label → OWLv2 미리 라벨 |
| `tools/ovd_propose.py` | OWLv2(이름만 주면 찾는 검출기)로 출동 차량·라바콘 후보를 JSONL 로(2026-11-09). **다른 가상 환경**(`E:\OpenVocab\.venv` — transformers)으로 돌고 DB 를 열지 않는다. 입력은 아래 `list` 가 쓴 TSV |
| `tools/import_proposals.py` | 그 후보로 responder·cone 박스를 만들어 검수 대기로(11-09). `list`(이미지 목록) → `apply`(기본은 보기만, `--apply` 때 DB 를 뜨고 `source='bulk'` 묶음 'OWLv2 후보 …' 로 써서 검수 UI '변경 내역' 에서 되돌릴 수 있다 — **승인 이미지는 검수 대기로 간다**) → `restore`(되돌린 뒤 옛 승인 이미지를 승인으로). responder 후보가 기존 차량 박스와 IoU 0.5 이상이면 그 박스 클래스만 바꾼다. 결과·목록은 `runs/ovd/`(git 밖 — 이미지 경로 목록이라) |
| `tools/finetune.py` | 해상도를 올리고 mosaic 을 끈 짧은 미세조정 → `models/best_ft960.pt`(운영 가중치는 `--promote` 때만 바꾼다). `runs/STOP` 으로 정상 중단(그때는 `--promote` 안 함), 기본 batch 는 모델 크기·imgsz 별 8GB 실측 한도 |
| `tools/probe_scale.py` | 모델 크기(n/s/m)를 같은 조건으로 비교 |
| `tools/extract_frames.py` | 영상에서 프레임을 뽑아 `unlabeled/<카메라>/<날짜>/<카메라>_<시각>_<밀리초>.jpg` 로(카메라는 `cameras.yaml` 에 먼저 등록, 시작 시각은 `--start` 나 영상 이름) — docs/provenance.md |
| `tools/legacy_groups.py` | legacy 이미지의 장면 종류·카메라 후보·크롭 추정 초안(DB 읽기 전용) — 사전학습 임베딩을 평균 연결로 묶어 `data/visible/legacy_groups/` 에 CSV·대조 시트. 카메라 후보는 거칠다(웹 사진과 CCTV 크롭이 한 후보에 섞이기도 한다) |
| `tools/domain_review.py` | 도메인(CCTV·비CCTV·드론)·태그 점검(DB 읽기 전용) — 같은 태그·장면 묶음과 다른 도메인, 비슷한 이미지(임베딩 이웃)와 다른 도메인, EXIF 제조사, 태그 포함 관계·오타 후보, 같은 카메라 누수를 `data/visible/domain_review/` 에 id 목록·대조 시트로. **의심일 뿐이다** — 11-04 에 의심 이미지를 다 눈으로 보니 고칠 것은 6장이었다. 본 것은 `--mark-seen` 으로 그 폴더의 `seen.csv` 에 적고(11-04 점검분 465건), 다음부터 같은 이미지가 같은 도메인으로 같은 검사에 걸리면 목록·시트에서 뺀다(`--all` 이면 다) — 표시는 사람이 다 본 뒤에만 할 것. '일부가 빠진 태그' 표(3장 이상 장면 묶음의 절반 이상에 붙은 태그가 나머지에 빠진 곳)는 따로 — 한 태그의 줄을 다 본 뒤 `--mark-tag <태그>` 로 (이미지, `tagmiss:<태그 id>`) 를 적는다(`--mark-seen` 은 이 표를 적지 않는다) |
| `tools/prune_seed_json.py` | 시드 COCO json 에서 파일이 없는 이미지 항목과 그 annotations 를 뺀다(scan 의 'JSON 에 있으나 파일이 없는 이미지' 를 없앤다, DB 는 안 건드림). 기본은 보기만·`--apply`. 원본 json 은 `seed_coco` 밖 `seed_json_backup/<시각>/` 에 뜨고, 다시 읽어 남은 이미지·박스가 같은지 본 뒤 다르면 되돌린다. 한 json 의 절반 넘게 없거나 id 가 겹치거나 원래 모양으로 다시 저장할 수 없으면 아무것도 쓰지 않는다(절반은 `--force`) |
| `tools/scheduled_stop.py` | 정해진 시각에 웹 UI 학습을 정상 중단(탭 서버에서는 `/api/projects` 로 학습 중인 프로젝트를 찾는다) |

## 라벨 기준

전문·차종별 실물 예시·전수 검사 기록은 [docs/label-rules.md](docs/label-rules.md) — 판정 기준을 바꾸면 그 파일도 같이 고칠
것. **배달 로봇은 라벨하지 않는다**(드론 주차장 장면의 흰 소형 차량 — 모델이 car 로 잡아 missing 후보로 나와도 무시할 것).

**차량 클래스 판정 순서** — 크기나 용도가 아니라 **차체가 한 덩어리인가**로 본다. 위에서부터 처음 걸리는 것으로 정한다.
**11-02 에 van 을 car 로 합쳤다**(사용자 결정) — 가를 것은 pickup·truck·bus 와 그 밖(car)이다.

| | 묻는 것 | 결과 |
|---|---|---|
| 0 | 경광등·출동 도색·표지가 **보이는** 출동 차량 — 소방·구급·경찰·도로공사 순찰·견인차, 뒤에 LED 화살표판을 단 차(11-09 추가 — 애매하면 아래 차종으로) | **responder** |
| 1a | 적재함이 캡과 **나뉘어** 있고 **열려** 있다(호로·탈착 덮개·캐노피 포함) — **1톤급까지**(포터·봉고·다마스·미국식 픽업) | **pickup** |
| 1b | 위와 같은데 **그보다 크다**(마이티·메가트럭·평판·덤프) | **truck** |
| 2 | 적재함이 나뉘어 있고 차체에 고정된 **상자 바디**다(탑차·트레일러·캡섀시에 얹은 특장차) | **truck** |
| 3 | 한 덩어리인데 **승객 출입문이 일반 버스처럼 따로** 있다(카운티·레스타급 포함) | **bus** |
| 4 | 그 밖 — 승용차·SUV·미니밴, 슬라이딩 도어 승합차, 한 덩어리 상자형 밴(스타렉스·스타리아·하이에이스·레이 등, 11-02 전 van) | **car** |

- **크기를 보는 곳은 1a/1b 하나뿐이다.** 같은 섀시라도 적재함이 따로 얹혔으면 truck/pickup, 캡과 화물칸이 한 몸통이면 car 다.
- 개조차는 몸통은 지금 얹힌 것으로, 1톤급인지는 원래 차종(캡)으로 본다. 캐노피 아래로 원래 적재함(옆판·테일게이트)이
  보이면 pickup, 적재함을 떼고 상자를 얹었으면 truck. 창문·해치가 있어도 승객이 안 타면 bus 가 아니다. 세일 패널 픽업
  (아발란치)은 옆에서 SUV 처럼 보이니 뒤를 볼 것.
- **CCTV 부감에서는 겹바퀴가 거의 안 보인다** — 차종(캡 형상)을 아는 것이 제일 빠르다.
- **낯선 각도에서 실루엣만 보고 판정하지 말 것.** 내가 117(베팅차)·3428(스타리아)·2671(아발란치)을 오판했고 지금 라벨이
  맞다 — 다시 오류로 찍지 말 것. 축소 대조 시트로는 적재함이 열렸는지도 자주 틀린다. 후보를 올리기 전에 **원본 해상도로**
  다시 보고, 그래도 모르면 사용자에게 물을 것.
- van 을 다시 나누게 되면 그때의 기준(`git show e788f84:CLAUDE.md`)과 합치기 직전 DB
  (`data/visible/labeling.db.bak_merge_20261102_133352`)를 볼 것.

**이미지 도메인(`images.domain`) — 공중에서 찍었거나 드론이 찍었으면 드론이다**(11-04 사용자 결정, 예시는 docs/label-rules.md).
영화의 공중 장면·헬기 추격 화면도, 드론 영상에서 눈높이로 내려온 프레임도 드론이다(뒤쪽은 '일단'). 이런 이미지는
`tools/domain_review.py` 가 이웃·태그·장면 묶음과 다르다고 계속 띄운다 — 다시 오류로 올리지 말 것. CCTV 와 비CCTV 의
경계는 아직 적어 둔 기준이 없다(애매하면 물을 것).

**EXIF 회전 이미지가 49장 있다**(Orientation=6). 라벨은 회전을 적용한 화면 기준 정규화 좌표이고, **DB 의 width/height 는 회전
전 원시 크기다**(`labeler.py check` 도 원시 크기와 견준다). 정사각이 아닌 4장(id 1050·1126·1692·3181)만 가로·세로가 뒤바뀐다.
그래서 **화면·SAM·크롭 좌표는 회전 적용 크기로**(`yolo_io.display_size`, 크롭은 `ImageOps.exif_transpose`) 계산한다 — PIL 로
그냥 열면 박스가 엉뚱한 자리로 보이지만 라벨은 정상이다.

## 현재 데이터 상태

```text
승인 5,203장(VIRAT 37 포함) / 박스 30,066개 / 클래스 8개      (2026-11-08 아침 확인 — 11-07 에 '어울리지 않는 이미지' 949장을 뺐다)
bicycle 1,518  bus 988  car 10,344  motorcycle 1,505  person 12,860
scooter 1,031  truck 1,050  pickup 770
검수 대기 478(전부 VIRAT), 미라벨 0, 건너뜀 0, 휴지통 0, 박스 없는 승인 이미지 60 · DB 행 5,681
분할: train 3,725 / val 729 / test 749(박스 4,192) / 미배정 0 — DB 의 images.split 에 고정(미배정 169장은 11-08 학습 때 배정됐다)
도메인(승인): CCTV 3,847 / 비CCTV 1,098 / 드론 258 / 미분류 0 · 휴지통이 아닌 이미지에 붙은 태그 84개·3,753장
11-09: 클래스 10개(responder·cone — 승인 박스는 아직 0) · DB 행 5,898 · 승인 4,598 / 검수 대기 1,300 =
       VIRAT 478 + ROAD PLUS 사고 처리 현장 train 162(카메라 hw-* 5대, 묶음 'OWLv2 후보 11-09 11:32'·'… 11:41')
       + **test 55(11-07 사고 영상 6개 — 카메라 hw-06·hw-07·hw-08·hw-09, DetectView 평가 장면의 출처)**
       + 옛 승인 605(사용자 "넣기 (문턱 0.5)" — 라바콘 후보 1,315 더함·출동 차량으로 90 바꿈·1 더함, 묶음 'OWLv2 후보 11-09 11:34',
       분할 train 473·val 48·test 50 + VIRAT 34). **검수를 끝내기 전에는 학습·test 점수가 이 605장을 빼고 돈다**
       (출동 차량으로 바꾼 90개는 대부분 S.RIDE 택시·흰 SUV 오검출 — 검수에서 원래 차종으로 되돌릴 것)
```

- **11-07 저녁 '프로젝트에 어울리지 않는 이미지' 를 휴지통으로 보냈다**(사용자 결정, VIRAT 제외 승인 6,115장 점검 — 목록·시트는
  git 밖 `data/visible/fit_review/`).
  - 보낸 것 949장
    - A 영상물 243: 영화·뮤직비디오·드라마·앨범 표지
    - B 우리 클래스 물체가 없는 풍경·동물·빈 화면 48: 해외 라이브캠 39 등
    - C PETS 2009 S2.L1 View_001 연속 프레임 658: 프레임 번호 8개마다 1장, 96장만 남김
  - 이로써 test 는 802 → 693장이 됐다.
  - 기록: 전 DB `labeling.db.bak_fittrash_20261107_201344`, 목록 `data/visible/trashed_fit_20261107_201344.txt`.
  - **같은 날 사용자가 휴지통을 비웠다** — VIRAT 휴지통 7장과 함께 956행과 원본 파일을 지웠다(seed 701 · legacy/labeled 248 ·
    VIRAT 7). 남은 행은 파일이 다 있다.
    - 되살리려면 위 백업 DB 의 행과 파일이 필요하다. 파일은 사용자 클라우드 백업이나 C: `20261031`(legacy) 에서 가져온다.
    - 그 701장은 seed COCO json 에 이름이 남아 scan 이 'JSON 에 있으나 파일이 없는 이미지' 로 알렸다. 11-08 에 json 에서 뺐다
      (아래 `legacy/seed`).
    - 지운 248장의 라벨 사본(legacy/labeled/labels/*.txt)도 사용자 요청으로 윈도우 휴지통에 보냈다.
    - 그 폴더에는 그 전부터 이미지 짝이 없던 txt 3개가 남아 있다(one_10344…·two_plus_3352…·스크린샷 2026-09-13 204935).
  - **휴지통 비우기는 원본 파일까지 지운다**(`tidy.purge_trashed`). legacy/seed 는 유일한 사본이니 비우기 전에 물을 것.
  - 남긴 것(사용자가 정할 것): 거리 수준 비CCTV 사진 1,098(bicycle 의 40%·bus 18%·scooter 17% 가 여기), 드론 258,
    실내 CCTV 크롭 약 120.

- **11-07 에 VIRAT 프레임 550장을 검수 대기로 넣었다**(id 7229–7778, source `virat`, 카메라 `virat-XXYY` 11대 × 50,
  도메인 CCTV, 분할 미배정 — docs/provenance.md 'VIRAT 가져오기'). 박스는 같은 날 사용자 결정으로 다시 라벨링해 운영 모델
  + 모델이 못 찾은 VIRAT 이고, 0503 은 아래 주차장 띠(1920×650)만 잘라 `split: train` 으로 고정한 뒤 모델 예측만으로
  다시 라벨링했다(`--model-only`). 승인 전에는 학습에 안 들어간다.
  - 같은 날 저녁 점검에서 확실한 오류만 고쳤다(사용자 승인): 박스 211개(116장), 휴지통 6장. 이력 묶음 'VIRAT 정리 A~D',
    백업 `labeling.db.bak_viratfix_20261107_195710`. 자세한 것은 provenance.
  - 0500 도 잘라 x 700 오른쪽·y 270 아래(1220×810)만 쓰고 `split: train` 으로 고정했다(사용자 결정, 11-07 밤).
    - 위쪽 인도에 겹쳐 선 오토바이 뭉치를 눈으로 가를 수 없고, 왼쪽은 나무 그늘(가린 차의 VIRAT 박스)과 막은 공사 차선이다.
    - 폭이 줄어 640 입력에서 물체가 통째 화면보다 약 1.6배 크게 보인다(train 전용이라 평가에는 영향 없음).
    - 백업 `labeling.db.bak_viratrecrop_20261107_210025`(y 자르기 전)·`…_215155`(x 자르기 전).
    - 이어서 흰 밴 뒤 오토바이 라벨이 빠진 27장에 박스를 채웠다(이력 묶음 'VIRAT 정리 E', 백업 `…bak_viratmoto_20261107_215540`).
    - 11-08 새벽 사용자가 0500 을 검수했다: 승인 26. 나머지 24장은 휴지통을 비워 행·파일이 없다.
  - 0101 도 y 210 위를 잘라(1280×510) `split: train` 으로 고정했다(사용자 결정, 11-08).
    - 왼쪽 위로 주차장 너머(운동장·세워 둔 트레일러·먼 인도에 앉은 사람들)가 멀리 보여 장마다 라벨이 갈렸다.
    - 폭이 그대로라 물체 크기는 같다. 박스는 다시 라벨링하지 않고 옮겼다(`--keep-boxes`, 2,569 → 2,325).
    - 먼 인도·도로의 VIRAT 보행자·자전거도 같이 빠졌다. 백업 `labeling.db.bak_viratrecrop_20261108_034644`.
  - 자른 장면(0500·0101)은 박스 좌표가 바뀌어 'VIRAT 정리 A~D' 묶음 되돌리기에서 건너뛴다.
  - 지금 VIRAT(11-08 저녁): 승인 37(0101 3 · 0500 26 · 0503 8) / 검수 대기 478. 검수 대기 박스는 모델 7,243 · VIRAT 1,000
    — 새 운영 m 으로 407장을 다시 라벨링하고 확실한 오검출 25개를 지운 뒤다(아래 '다음에 할 일' 4).
  - 휴지통을 비운 것은 행·파일이 없다. 그래서 0000·0002·0502 는 48장, 0101 은 46장, 0503 은 49장, 0500 은 26장이다.
  - `import_virat.py --per-scene 50 --apply` 를 다시 돌리면 모자란 만큼 다시 채운다(지운 것과 같은 프레임을 다시 고를 수 있다).
  - 남은 후보 목록은 git 밖 `data/visible/virat_qa/`.
  - VIRAT 이미지는 묶음(grp)이 없어 '같은 자리 일괄 수정' 이 400 이다.
- **클래스는 10개다** — 11-09 에 responder(8, 출동 차량)·cone(9, 라바콘)을 붙였다(사용자 결정 — DetectView 유고 단서용, 판정은
  docs/label-rules.md '출동 차량·라바콘'). **그날 박스는 아직 없고, 옛 승인 라벨에는 출동 차량이 truck·car 로, 라바콘은 배경으로 들어 있다** —
  고치기 전에 이 둘을 넣어 학습하지 말 것(후보 찾기·미리 라벨은 `tools/ovd_propose.py` → `tools/import_proposals.py`).
  그 전 8개: pickup 은 09-20 에 사용자가 검수 UI 에서 추가했고, 같은 날 추가한 van 은 11-02 에 car 로 합쳤다.
  **클래스 번호는 순서가 전부다** — config.yaml 의 `classes` 에 **뒤에만 붙이고** 중간에 끼우거나 빼지 말 것(박스가 번호로
  가리킨다). 맨 끝 클래스를 없앨 때는 `tools/merge_class.py` — van 합침 묶음은 van 을 되살리기 전에는 되돌리기가 409 로
  막힌다. 합치자 같은 자리에 car 가 둘이 된 5185(test)는 칠판 묶음 1 의 DB 6건에 들어 있다.
- **점수는 클래스별로 비교할 것.** 7클래스 시절 점수(visible-3 0.6868, A40 비교)도, 9클래스 라벨로 잰 09-27 기준선(전체 0.5174)도
  지금 라벨과 바로 견줄 수 없다. **11-08 에 visible-3 을 지금 라벨(test 749장)로 다시 쟀다** — 전체 0.5762(pickup 의 0 이 든
  8클래스 평균, pickup 뺀 7 은 0.6585). 같은 test 로 잰 새 m·DetectView y26m 과의 표는 docs/experiments.md '8클래스 첫 학습',
  기록은 `--history`. 7클래스 모델은 pickup 을 truck·car 로 잡아 그 둘이 깎이니, 8클래스 모델과는 pickup 영향이 없는 클래스로도
  견줄 것. **비CCTV 가 가장 약하다**(11-08 새 m 도 0.58 — 09-27 의 0.378 은 비CCTV 를 `source='unlabeled'` 로 가른 값). **11-04
  부터 test_baseline 의 CCTV·드론·비CCTV 는 사람이 정한 `images.domain`** 이다 — 이름표에 '(손 분류)' 가 붙고 기록에 `domain_rule`
  이 남아 옛 기록과 견주지 않는다.
- car 의 약 13% 는 **고정 CCTV 한 대(PETS 2009 View_001)의 754프레임에 세워진 두 대**였다(옛 van 쏠림 — 그 카메라는 통째로
  train). 11-07 에 96프레임만 남겨 car 192개(2%)로 줄였다.
- `labeler.py split` 은 읽기 전용이 아니다(미배정을 배정해 DB 에 쓴다) — DB 만 보려면 `sqlite3` 를 `mode=ro` 로 열 것.

**묶음 단위 계층화 3분할 (2026-09-16, `split.py`·`grouping.py`).**
- **test(15%)는 학습에도 best.pt 고르기에도 쓰지 않는다** — 모델끼리 비교는 test 로 한다(`tools/test_baseline.py`).
- **같은 장면(dHash·이름 연번 묶음)은 통째로 한쪽에만 넣는다.** 계층화는 클래스 비중만 맞추고 출처는 보지 않으니 도메인 쏠림은
  따로 볼 것. **기존 이미지는 움직이지 않는다** — 새 이미지만 자기 묶음을 따르거나 계층화로 들어가고, 두 분할의 묶음을 이어
  붙이는 새 이미지가 나오면 옮기지 않고 `[알림] … 두 분할에 걸칩니다` 만 띄운다(`moved` 는 reset 이 아니면 0 이어야 한다).
- **손으로 옮길 때는 val·test → train 방향만** — 반대면 학습한 이미지가 평가셋에 들어간다. test 의 드론 묶음 `g003254`·
  `g003305` 는 사실상 같은 장면이라 반드시 같은 쪽에 둘 것.
- 09-24 에 이름 계열 규칙을 고쳐(자릿수·이미지 크기가 같아야 한 계열) `g000001`(그때 1,433장)이 나뉘게 됐고, **11-08 학습 때 실제로
  다시 묶였다**(묶음 2,885개, 가장 큰 묶음 180장 — 기존 배정은 그대로).
- 묶을 때 로보플로(`_jpg.rf.<해시>`, 두 번 내보낸 `_png_png.rf.<해시>` 까지 전부)·tidy(`__<sha1 8자>`·`_2`) 꼬리표는 떼고 보고,
  dHash 는 EXIF 회전을 적용해 잰다. 영문 `Screenshot …` 도 30초 스크린샷 규칙을 탄다. **이름 계열은 카메라 안에서만 잇는다**
  (다른 카메라끼리는 dHash 로만). 누수는 `labeler.py split --leak` 으로 다시 잰다.
- **11-05 에 꼬리표 규칙을 고쳐, 두 분할에 걸친 묶음이 알림으로 뜬다** — 11-08 학습 때 27개였다(11-05 시뮬레이션은 29개·366장, 그
  사이 휴지통으로 뺀 이미지가 있다). 예: `duque_de_caxias` 110장(한 카메라, g002471)이 train 98·val 9·test 3. 기존 배정은 고정이라
  옮기지 않는다.
  계층화는 섞인 묶음을 다수결 분할로 센다(비중이 조금 어긋남 — 고치지 않았다). 누수를 없애려면 분할 재배정(사용자 결정).
- `train.test_ratio`(또는 val_ratio)를 0 으로 바꿔 이미 배정된 분할이 비율에서 빠지면, reset 이 아닌 한 **아무것도 쓰기 전에
  멈춘다**(11-05 — 전에는 test 가 경고 없이 train·val 로 흡수됐다).
- **09-16 재배정 전 모델(train-3·visible·visible-2)의 test 점수는 쓰면 안 된다**(지금 val/test 를 학습에 썼다). 옛 모델과는 두
  모델 모두 장면째 못 본 이미지로만 비교한다(docs/experiments.md).

**폴더는 `data/visible/` 다.** DB 는 이미지를 **절대경로**로 들고 있어 폴더를 옮기면 `images.path` 도 치환해야 한다(스니펫은
README '새 PC 에서 시작하기' 5단계). **경로를 고치기 전에는 스캔·학습·`prune`·정리 버튼을 쓰지 말 것** — 스캔은 시드를 검수
전 라벨로 다시 넣고(그래서 DB 의 절반 넘게 '파일 없음' 이면 멈춘다, `scan --force` 로 강행), 웹 UI 의 정리(tidy)는 뒤이어
`prune` 을 돌려 '파일 없음' 행을 **검수 상태·라벨째 지운다**(절반 넘게 없을 때만 멈추니, 일부만 되살린 상태면 그만큼 지운다).
- **`legacy/` (2026-11-03)** — 촬영 위치·카메라·시각 같은 출처 없이 모인 지난 수집분(`seed/` 5,849장 + `labeled/` 718장, DB
  `images.path` 6,567행도 같이 치환했다. 옮기기 전 DB: `labeling.db.bak_legacy_20261103_124639`). **앞으로의 수집은 출처를 밝혀
  `legacy/` 밖에 둔다.** 새 이미지의 tidy 결과는 `legacy/` 가 아니라 새 `labeled/` 에 쌓인다(`config.yaml` 의 `labeled`).
- `legacy/seed/seed1`·`seed2` — 로보플로 COCO 내보내기(**유일한 사본**). `seed_coco: data/visible/legacy/seed` 는 하위 폴더까지 COCO 주석
  json 을 **전부** 찾아(`coco_io.find_annotation_files`) 검수 없이 승인으로 넣으니, 검토 전 데이터나 json 사본을 `legacy/seed/` 아래에
  두지 말 것(새 회차는 `seedN/` 에 `coco/*.json` + `images/`).
  - **11-08 에 seed1 json 3개에서 파일을 지운 이미지 1,274장·박스 12,024개를 뺐다**(`tools/prune_seed_json.py`, 사용자 승인).
    원본 json 은 `legacy/seed_json_backup/20261108_045140/`(뺀 목록 `removed.tsv` 함께)과 C: `20261031` 에 있다.
  - 휴지통 비우기로 시드 사진을 지우면 json 에 이름이 남아, scan 이 'JSON 에 있으나 파일이 없는 이미지' 로 알리고
    건너뛴다. 지운 게 맞으면 이 도구로 뺀다. 지운 사진을 되살릴 때는 떠 둔 json 으로 되돌리고, 라벨은 DB 백업 쪽을 쓴다.
- `legacy/labeled/` — 11-03 전에 tidy 가 옮긴 검수분(그 `labels/*.txt` 는 정리 시점 사본이라 지금 DB 와 다르다 — 11-05 에
  716장 중 98장, 옛 van 8번 44개. 되살리기는 DB 백업으로). `unlabeled/` — 새로 넣는 사진.
- **출처 기록 (11-03, [docs/provenance.md](docs/provenance.md))** — 새 수집분은 `unlabeled/<카메라키>/…` 에 두고 카메라를
  `data/visible/cameras.yaml`(실제 장소 — git 에 넣지 말 것)에 먼저 등록한다. 등록 안 된 폴더가 있으면 scan 이 아무것도 넣지
  않고 멈춘다. 들어온 이미지는 `source='collected'`·`images.camera`·`captured_at`(파일 이름 → EXIF, 못 찾으면 NULL), tidy 는
  `labeled/images/<카메라>/` 로 옮긴다. 분할은 **카메라째 한 묶음**이고 `cameras.yaml` 의 `split:` 으로 고정할 수 있다
  (고정이 다른 카메라가 dHash 로 이어지면 분할이 멈춘다). split 보고의 `카메라별:` 줄로 어디로 갔는지 볼 것.
  검수 UI 는 카메라 필터·카메라·촬영 시각을 보여 주고, `test_baseline.py --domains` 는 `legacy`·`collected`·`cam:<키>`·
  `kind:`·`mount:` 를 잰다 — **09-27 기준선과 같은 범위는 이제 `all` 이 아니라 `legacy`** 다(`all` 은 새 수집분도 담는다). `cctv`·`drone`·`noncctv` 는 `images.domain`, 비어 있으면 카메라 등록으로 짐작한다(설치 drone → 드론, kind cctv → CCTV, phone → 비CCTV, 그 밖은 `undomain`).
- `dataset/` — 학습할 때마다 DB 로 새로 만든다(Windows 학습은 `.npy` 디스크 캐시 약 14GB 를 더 얹는다). **test 는 이 폴더 말고
  `tools/test_baseline.py` 로 잴 것.** 이어하기는 없으면 다시 만든다(`ensure_resume_dataset`).
- `labeling.db` — **작업물 그 자체**(함정 5). 옆의 `backups/` 에 검수 서버가 하루 한 번 사본을 남긴다(최근 14개).
  `classes.txt` 는 DB 옆이다. `thumbs/`(11-04)는 이미지 보기 썸네일 캐시 — 지워도 되고 백업하지 않는다.
- `export/<프로젝트>/` — 내보내기 기본 폴더. 표식(`export_info.json`)이 있고 그 DB 가 지금 DB 인 폴더만 비우고 다시 쓴다. 저장소
  안의 다른 폴더로 내보내면 그 폴더에 `.gitignore`(`*`)를 둔다. `data.yaml` 은 DB 분할 그대로의 목록(`train.txt`·`val.txt`·
  `test.txt`, 미배정은 빠짐)을 가리킨다(11-05 — 전에는 train = val = images 라 test 가 섞였다).

**받는 이미지 형식은 `yolo_io.IMG_EXTS` 하나로 정한다** — 여기 없는 확장자는 `scan` 이 **조용히 건너뛴다.** 지금
`.jpg .jpeg .png .bmp .webp .tif .tiff .avif`. AVIF 는 Pillow 11.3+ 에서 된다. `.heic` 를 받으려면 `pillow-heif` 를 먼저 깔고 추가할 것.

**가중치.** `weights/` 에 사전학습 가중치(`yolo26n/s/m/l.pt`, `sam2.1_t.pt`)가 있다 — config 에 이름만 적으면
`config.pretrained_path()` 가 여기서 찾는다(없으면 Ultralytics 가 실행한 폴더에 받아 버린다). **`models/best.pt`(운영)는 11-08
부터 `best_y26m_8cls.pt`** 다(8클래스 첫 학습, yolo26m 640, epoch 84, val 0.693, sha256 `f09f02f4…` — 사용자 결정). 그 전 운영
visible-3(yolo26s 7클래스, epoch 156, val 0.654, sha256 `ca1f9f5c…`)은 `best_visible3.pt`·`best.prev.pt` 에 있다.
`best_detectview_y26m.pt` 는 DetectView 배포 모델 원본의 사본(7클래스, sha256 `785b3542…`)이다 — 셋의 test 비교는
docs/experiments.md '8클래스 첫 학습'. **학습 설정도 같은 날 운영 m 과 같게 바꿨다**(`model.base` `weights/yolo26m.pt`·
`train.batch` 10·epochs 100·close_mosaic 10 — 경로를 빼면 Ultralytics 전체 설정이 그 학습과 같다).
**`best.pt`·`best.prev.pt` 는 학습이 끝날 때뿐 아니라 중단할 때마다 그 시점 best 로 밀려난다** — 운영할 가중치는 늘
`best_<이름>.pt` 로 따로 둔다.

**다른 디스크 백업: `C:\local_auto_labeling_backup\20261031\`**(C: = 디스크 1, E: 와 다른 물리 디스크). `.venv`·`graft`·
`data/visible/dataset`·`__pycache__`·`*.tar`·살아 있는 DB 파일을 뺀 전부(8,127개 5.11GB, md5 전수 대조)와 sqlite `backup()` 으로
뜬 DB 다. 자세한 것은 그 폴더의 `BACKUP_INFO.txt`. Objects365 tar 는 들어 있지 않다. E: 의 외부 백업(09-11)은 11-02 에 사용자가
지웠다. **그 위에 더하는 `C:\local_auto_labeling_backup\20261105\`**(11-05, 28MB — 사용자 확인): 그날의 DB 스냅샷(integrity ok,
행 수 원본과 같음), 10-31 뒤에 생긴 이미지 3장(id 7226–7228)·`domain_review/`(seen.csv)·classes.txt·구내식당 id 목록·
`docs/collab.md`, push 전 커밋까지 담은 `repo_main.bundle`. 이미지는 10-31 백업에 6,115장 중 6,112장이 크기까지 같게 있어 다시
뜨지 않았다. 되살릴 때는 10-31 의 옛 폴더 구조(`seed/`·`labeled/`)를 `legacy/` 아래로 옮긴 뒤 이것을 덮는다(그 폴더의
`BACKUP_INFO.txt`). 하루 사본·`labeling.db.bak_*`(11-02·03, 약 210MB)는 넣지 않았다. 다시 뜰 때는(PowerShell, `$src`=저장소
`$dst`=새 날짜 폴더 — **이전 것을 지울지 먼저 물을 것**):

```powershell
robocopy $src $dst /E /XJ /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:8 /XD "$src\.venv" "$src\graft" "$src\data\visible\dataset" "$src\data\visible\thumbs" __pycache__ /XF labeling.db labeling.db-wal labeling.db-shm labeling.db-journal *.tar
```

종료 코드 0–7 이 성공이고, 그 뒤 DB 를 sqlite `backup()` 으로 `$dst\data\visible\labeling.db` 에 뜬다. **되살리기는 README '새
PC 에서 시작하기' 4·5단계대로 네 폴더만** 가져온다 — 백업 폴더를 통째로 덮는 `BACKUP_INFO.txt` 1단계는 저장소까지 잃었을
때만이다(안의 `.git` 은 메일을 바꾸기 전 이력이고, 옛 문서까지 덮인다).

## WSL 학습 (`wsl/`, 웹 UI 'WSL 학습' 버튼)

학습은 WSL 이 **약 4배 빠르다**(09-12 같은 조건 전체 학습: epoch 258초 → 62초 — ram 캐시·워커 8·fork 덕인데 몫은 가르지
않았다). 환경은 WSL Ubuntu 의 miniconda `~/miniconda3/envs/yolo_final`(ultralytics 8.4.142, torch 2.11+cu130). **패키지를 올릴
때는 드라이런으로 torch 를 안 건드리는지 먼저 볼 것** — PyPI 의 torch 가 cu130 빌드를 덮어쓸 수 있다. 되돌릴 기준은
`~/yolo_final_freeze_before_upgrade.txt`(09-11 업그레이드 전 freeze).

버튼(`wsl_train.py`)은 DB 로 `data/visible/dataset` 을 만들고 → `sync_dataset.sh` 로 WSL 에 복사(`--delete`) → `train_wsl.py`
를 돌리며 로그를 중계하고 → 끝나면 `promote_weights()` 로 `models/best.pt` 를 바꾼다. [중단] 은 WSL 쪽 `STOP` 파일을 만든다.
설정은 `config.yaml` 의 `wsl:`.
- **WSL 버튼은 Windows 학습 버튼과 같은 인자로 학습한다**(`trainer.train_args(cfg)` → `train_wsl.py --bare --hyp KEY=VALUE`,
  `--dry-run` 으로 확인). `train_wsl.py` 는 Windows 프로젝트 코드를 import 하지 않는다. `wsl.exe` 가 명령줄을 bash 로 넘기므로
  값은 영숫자와 `_ . + - =` 만 넘긴다(`wsl_train.hyp_tokens`).
- **DB 는 WSL 로 가져가지 않는다** — `/mnt/e` 로는 SQLite 잠금이 안 먹어 검수 서버와 함께 쓰면 손상될 수 있다. 이미지·라벨만
  복사한다. 프로젝트를 늘리면 WSL 데이터셋 폴더를 **반드시 다르게** 둘 것(`--delete` 동기화가 서로 덮는다).
- **서버를 죽여도 WSL 학습은 계속 돈다** — 그때는 STOP 파일을 직접 만든다. `--stop-at N` 은 N epoch 을 마치면 정지 파일처럼
  멈춘다(lr 스케줄은 그대로, `--resume` 으로 잇는다).
- [중단] 을 누른 뒤에는 학습이 끝날 때까지 STOP 이 없으면 0.5초마다 다시 만든다(11-05 — train_wsl.py 는 시작할 때 STOP 을 지워
  시작 직후의 중단이 씹혔다. 그 동작은 그대로 둔다). 'WSL 이어하기' 는 train_wsl.py 가 쓸 수 있는 resume.pt·last.pt 만 띄우고,
  `--resume` 은 results.csv 를 체크포인트 epoch 까지 자른다(원본 `.csv.bak`).
- 비교 학습은 `runs/compare_0918/run_compare.sh`(결과는 `runs_cmp/` — 'WSL 이어하기' 가 집어 가지 않게). **순서 스크립트를
  `pkill` 하지 말 것**(학습도 같이 죽는다) — 멈출 때는 `touch ~/new_env/yolo/runs_cmp/HALT ~/new_env/yolo/runs_cmp/STOP`. 도는
  동안 웹 UI 로 학습을 시작하지 말 것.
- **`wsl -- bash -c '... $변수 ...'` 는 바깥 셸이 변수를 먼저 풀어 빈 값이 된다** — 변수가 든 일(파일 지우기 포함)은 셸
  스크립트 파일로 돌린다. Git Bash 에서 `wsl` 을 부를 때는 `MSYS_NO_PATHCONV=1` 을 붙인다(아니면 `/mnt/e/...` 가 바뀐다).
- 사용자의 원래 스크립트 `~/new_env/yolo/train_yolo_v3_1.py` 는 다시 쓰지 말 것 — `dfl=0.0` 이다(함정 1).
- **optimizer `auto` 는 `warmup_bias_lr` 을 0 으로 바꾼다** — MuSGD 를 이름으로 고정한 직접 실행·`probe_scale.py`·`finetune.py`
  는 본 학습과 이 값이 다르다. **epoch 수가 다른 학습끼리 같은 epoch 의 mAP 를 견주지 말 것**(`cos_lr` 이 총 epoch 에 묶인다).
- 8GB 에서 들어가는 최대치(`wsl/probe_vram.py`): s 640 b20 / s 960 b8 / s 1280 b4 / m 640 b10 / m 960 b4 / l 640 b8 / x 640 b5.
  WSL 도 넘치면 OOM 없이 느려진다(함정 9).

```bash
bash /mnt/e/local_auto_labeling/wsl/sync_dataset.sh        # data/visible/dataset -> ~/datasets/visible
~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/train_wsl.py
touch ~/new_env/yolo/runs_wsl/STOP                          # 정상 중단
~/miniconda3/envs/yolo_final/bin/python /mnt/e/local_auto_labeling/wsl/train_wsl.py --resume
```

---

## 반드시 알아야 할 함정

실제로 겪고 원인까지 확인한 것들이다. 모두 **에러 없이 조용히 성능만 깎거나 데이터를 잃는** 종류라 재발하기 쉽다.

1. **`dfl` 을 0으로 두지 말 것.** `train.overrides.dfl` 은 DFL 전용이 아니라 **박스 거리 손실 게인**이다 — yolo11(reg_max=16)은
   DFL 에, yolo26(reg_max=1)은 ltrb 거리 L1 에 곱한다(근거: `ultralytics/cfg/default.yaml` 의 dfl 주석 *"DFL when reg_max > 1,
   L1 on DFL-free YOLO26"*, `utils/loss.py` 의 `loss[2] *= hyp.dfl`). 0이면 좌표 정밀도만 조용히 깎인다(합성 A/B: mAP50 은 같고
   mAP50-95 가 0.953 대신 0.902). `trainer.check_box_distance_gain()` 이 경고한다.
2. **Windows 에서 `cache: ram` 금지.** spawn 이라 워커마다 RAM 캐시를 피클하다 `OSError: [Errno 22] … _ImageCache` 로 죽는다
   (뒤따르는 `_workers_status` 는 잡음). 지금 `cache: disk`. Linux(WSL·pod)에서는 ram 이 맞다. `trainer.check_ram_cache()` 가 경고한다.
3. **하드링크 금지(`hardlink_images: false`).** Ultralytics 는 깨진 JPEG 을 그 자리에 다시 저장해서, 하드링크면 **원본이 덮인다**
   (8개 겪음). 지금은 복사가 기본이다.
4. **Roboflow YOLO 내보내기는 박스를 자른다**(가려져 조각난 객체는 큰 조각만 남는다 — COCO 와 IoU 0.70 미만이 6%). 데이터는
   반드시 **COCO 나 Pascal VOC** 로 받을 것. Ultralytics 는 한 파일에 5열·7열 이상 행이 섞이면 이미지째 버린다 —
   `yolo_io._row_to_box()` 가 행 단위로 살린다.
5. **`labeling.db` 를 함부로 지우지 말 것.** 검수 대기·건너뜀·휴지통 라벨은 DB 에만 있다(`.txt` 는 train·tidy·export 때 승인분만
   쓰인다). 재생성해야 하면 **먼저 사용자에게 물을 것.**
6. **코드 파일은 Write / Edit 도구로 고칠 것** — bash heredoc + 패치 스크립트가 `\n`·`\r` 을 실제 제어문자로 넣어 여러 번
   깨졌다(CR 이 섞이면 진단도 어렵다). 특히 문자열 리터럴이 있는 곳.
7. **Windows 에서 Ultralytics `train()`/`val()` 을 부르는 스크립트는 `if __name__ == "__main__":` 가드 안에 둘 것** — 없으면 spawn
   워커가 스크립트를 다시 import 해 죽거나 CPU 0% 로 조용히 멈춘다. 일회용 분석이면 `workers=0` 이 간단하다.
8. **학습이 끝나면 — 정지 파일·중단 버튼으로 멈춰도 — `last.pt` 로는 이어 학습할 수 없다.** 어느 쪽이든 `final_eval()` 의
   `strip_optimizer()` 가 `last.pt`·`best.pt` 를 덮는다(epoch -1, yolo26s 60MB → 20MB). 그래서 `trainer.attach_callbacks()` 가
   중단이 확정된 epoch 의 `last.pt` 를 `resume.pt` 로 복사해 두고, 이어하기는 그것으로 한다. 강제 종료(kill·정전)는 오히려
   `last.pt` 가 온전하다.
   - 복사할 수 있는 자리는 `on_fit_epoch_end`(`save_model()` 직후·`final_eval()` 전) 하나뿐이고, Ultralytics 는 patience 조기
     종료에도 `trainer.stop` 을 세운다 — **Ultralytics 를 올리면 이 두 가지를 소스로 다시 볼 것**(requirements.txt 주석). 이
     로직은 `trainer.py` 와 `train_wsl.py`(Windows 코드를 import 하지 않는다)에 따로 있어, 고칠 때는 둘 다 고친다.
   - **정상 중단은 정지 파일로** — `runs/STOP`(WSL 은 `~/new_env/yolo/runs_wsl/STOP`)을 만들면 다음 배치 경계에서 멈추고
     resume.pt 를 남긴다(그 epoch 의 나머지는 건너뛴다). 웹 UI 중단 버튼도 같다. **Ctrl+C 는 가로채지 않는다**(즉시 종료 —
     워커가 먼저 죽기 때문이다. epoch 하나만 잃는다).
   - 스냅샷은 **사람이 멈춘 때만** 남기고, 끝난 학습(조기 종료·완주, strip 된 `last.pt`)은 이어하기 대상에서 뺀다. 이어하기는
     **이 설정의 데이터셋으로 돈 학습만** 고른다(`trainer.uses_dataset`).
   - **정상 중단도 운영 가중치를 바꾼다** — `_finish()` 가 그 학습의 best.pt 로 `models/best.pt` 를 덮고 이전 것은
     `best.prev.pt` 로 남긴다(되돌리려면 그것을 복사한다).
   - 끝난 학습의 `epochN.pt` 는 지워도 되지만(`best.pt`·`last.pt`·그래프·`results.csv` 만 남긴다) **지우기 전에 물을 것.** WSL
     안에서 지워도 `E:\WSL\Ubuntu\ext4.vhdx` 는 저절로 줄지 않는다.
9. **Windows 는 VRAM 이 모자라도 OOM 을 내지 않고 공유 메모리로 넘겨 조용히 느려진다**(배치당 1.3초 → 45초, 사람이 PC 를 쓸
   때만). 판별은 **`results.csv` 의 epoch 소요시간**으로 한다 — GPU 사용률·전력·공유 메모리 카운터·VRAM 총합은 못 쓴다. 원인은
   프로세스별 전용 VRAM 을 어댑터(LUID)로 쪼개 본다(학습 프로세스와 같은 LUID 에 뭐가 있나):

   ```powershell
   (Get-Counter "\GPU Process Memory(*)\Dedicated Usage").CounterSamples |
     Where-Object {$_.CookedValue -gt 30MB} | ForEach-Object {
       if($_.InstanceName -match 'pid_(\d+)_luid_(0x\w+_0x\w+)'){
         "{0,-16} {1} {2} MB" -f (Get-Process -Id $Matches[1] -EA SilentlyContinue).ProcessName,
           $Matches[2], [math]::Round($_.CookedValue/1MB) } }
   ```

   예전 범인은 dGPU 에서 돈 Chrome 창을 합성하던 `dwm`(2.3GB)이었다 — `설정 → 시스템 → 디스플레이 → 그래픽` 에서 chrome.exe 를
   '절전'(iGPU)으로 지정해 고쳤다(앱을 다시 켜야 먹는다. M365 Copilot 등도 VRAM 을 잡는다). 대처는 GPU 쓰는 앱을 닫거나
   `labeler.py resume --batch 16`(`nbs=64` 누적이 유효 배치를 맞춘다). 빠른 확인:
   `nvidia-smi --query-gpu=memory.used --format=csv,noheader` 가 7.5GB 이상이면 뭔가 올라온 것이다. WSL 도 똑같다.
10. **`run.bat` 에 한글을 넣지 말 것**(주석까지 ASCII) — `chcp 65001` 에서 cmd 가 줄 중간부터 조각을 명령으로 실행한다. 안내
    문구는 `cli.cmd_review()` 에서 낸다.
11. **`images.id` 는 AUTOINCREMENT 여야 한다** — 아니면 마지막 행을 지운 id 가 새 이미지에 붙고, 브라우저는 캐시한 옛 사진을
    그 자리에 보여준다. 세 겹을 하나도 되돌리지 말 것: `store.py` 의 `AUTOINCREMENT` · `server.py` 의 `Cache-Control: no-cache` ·
    `app.js` 의 `?v=<updated_at>`.
12. **검수 UI 로 추가한 클래스는 DB 의 classes 표에만 있다** — 예전 scan 이 그 이름을 지웠다(09-20 고침, 이름이 없으면 그
    이미지는 저장(400)도 학습도 안 된다). `store.set_classes` 는 목록보다 뒤의 클래스를 지우지 않고(`drop_extra=True` 일 때만),
    `discover_classes`·`_ingest_coco` 는 합친 목록을 쓰고, `POST /api/classes` 가 `classes.txt` 에 전체 목록을 남긴다.
13. **정리(tidy)가 같은 이름 파일을 덮어썼다**(09-27, 사진 2장을 잃었다 — 6639·6641 은 지금 다른 사진과 그 라벨이고, 변경
    이력에 원래 라벨이 남아 원래 사진을 찾으면 되살릴 수 있다). 이제 `tidy._free_name` 이 이미지·라벨 이름이 **둘 다** 빌 때까지
    번호를 붙이고, 옮길 대상이 있으면 멈추고, DB 갱신이 실패하면 파일을 되돌린다. **코드를 고친 뒤에는 검수 서버를 다시 켜야
    반영된다.**

---

## 아키텍처·설계 의도

코드는 `src/autolabel/` 에 있다. 각 파일 맨 위 docstring 에 그렇게 된 이유를 적어 두었으니 고치기 전에 읽을 것.
- 학습은 매번 사전학습 가중치(`model.base`, 11-08 부터 `weights/yolo26m.pt`)에서
  **새로** 미세조정한다 — 이어 학습하면 자기학습 루프에서 모델이 자기 오류를 굳힌다.
- 분할은 DB 에 고정한다(위 '묶음 단위 계층화') — 그래야 사이클 간 비교가 유효하다.
- **자동 승인은 없다** — 신뢰도가 높아도 사람이 봐야 학습에 들어간다.
- 재스캔은 이미 등록된 이미지를 건너뛴다(UI 에서 손본 라벨을 원본으로 되덮지 않는다). **`INSERT ... ON CONFLICT DO UPDATE`
  뒤에 `lastrowid` 를 쓰지 말 것** — UPDATE 로 끝나면 다른 행의 id 가 남는다(지금은 `DO NOTHING` + `rowcount`).
- **중복 유입:** 바이트가 같은 새 파일은 등록하지 않는다(`dupcheck.py`). 재압축된 같은 사진은 아직 못 잡는다 — 디코딩한 픽셀의
  해시가 필요하다(`dupcheck.py` 머리말).

## UI 조작

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
- '드문 클래스 먼저' 정렬·클래스 필터는 목록과 **같은 WHERE**(`_list_where`·`count_images`)로 센다 — 어긋나면 목록 묶음이 틀어진다.
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

---

## 다음에 할 일

지난 학습의 결론은 docs/experiments.md 에 있다 — yolo26n → **yolo26s**(같은 조건 60 epoch 0.6495 → 0.7288) → 11-08
**yolo26m**(8클래스 첫 학습), s 는 **`predict.imgsz` 640 이 정점**(960·1280 은 낮다), close_mosaic 은 세 번 연속 이득이 없었다
(s 마지막 30 두 번, m 마지막 10 은 −0.010).

3. **미검토 제안.** 640 입력에서 박스의 23.3% 가 32px 미만이다. 학습 자체를 960 으로 올리는 것은 s 로는 해 보지 않았다 —
   해 본다면 batch 16 이하로, epoch 시간은 약 2.3배를 예상할 것(추정). SAHI 식 타일 추론은 `tools/test_tiles.py` 로 재 봤다(11-03, visible-3·CCTV test) — mAP50-95 는 +0.035\~0.038 이지만 같은 conf 에서 FP 가 늘고(대부분 라벨 누락이 아니라 진짜 오검출), FP 를 맞추면 이득이 16px 미만 객체(+4.5pt)로 좁혀져 `AutoLabeler` 에는 아직 넣지 않았다 — 타일↔통째 사이에만 NMS IoU 0.3 을 걸면 F1 이 통째와 같고(0.890) 재현율 +1.8pt·mAP +0.035(CCTV)라 넣을 만하지만, 사용자가 FP 표본 48개를 직접 확인한 뒤 정한다(표·한계는 docs/experiments.md '타일 추론' — 그 CCTV 는 11-04 전 규칙으로 가른 것이라, 다시 잴 때는 손 분류 도메인으로 바뀐다).
4. **8클래스 학습 — 11-08 에 한 번 했고(yolo26m, `models/best_y26m_8cls.pt`) 같은 날 운영 가중치로 올렸다(사용자 결정).** 그 전
   운영 visible-3 은 pickup 을 모르고 car 에 옛 van 이 섞였다. test 749장에서 새 m 은 visible-3 보다 모든 클래스가 높고(pickup 뺀
   7: 0.6585 → 0.7398), 같은 설정의 DetectView y26m 과는 pickup 과 상관없는 클래스에서 거의 같다(+0.013). 표는
   docs/experiments.md '8클래스 첫 학습'. 예측 비교(P)·자동 라벨링도 이제 pickup 을 보고, 웹 UI 학습도 같은 설정(m·batch 10·100 epoch·close_mosaic 10)으로 돈다. 그날 멈춘 s 학습
   `~/new_env/yolo/runs_wsl/visible` 은 사용자 요청으로 지웠다('WSL 이어하기' 후보 없음). 남은 것:
   - 8클래스 모델로 `tools/label_qa.py` 를 돌려 176px 아래의 truck·pickup·car 판정을 거른다(눈으로는 176px 위만 봤다 — 그 아래는
     사람도 1톤급과 2.5톤급을 못 가른다).
   - 검수 대기 VIRAT 478장 중 407장은 11-08 저녁 새 m 으로 다시 라벨링했다(사용자 요청 — pickup 263개가 새로 나옴, provenance).
     사람 이력이 있는 71장('VIRAT 정리 A·B·D' 로 고친 것)은 건너뛰어 아직 visible-3 박스다. 새 모델은 그중 car·truck 63개를
     pickup 이라 한다(검수 때 P 로 볼 것). 새 m 은 한 차에 car·pickup 을 겹쳐 내는 일이 많다(검수 화면 '⚠ 겹침' 101쌍).
     새 m 의 scooter 오검출 21개·정산기 사람 4개는 지웠다(이력 묶음 'VIRAT 정리 F·G', 사용자 승인 — 그 21장도 이제 건너뛴다).
   - **사용자 결정 거리:** `g000001` 에 섞여 train 에 갇혔던 CCTV 가 아닌 사진(약 640장)을 val·test 로도 보내려면
     `labeler.py split --reset --yes` 가 필요하다 — 그러면 지금까지의 점수 비교가 끊긴다.
5. **기준선을 처음부터 새로 잰다 (11-03 사용자 결정).** 순서: test 점검 → 압축 백업 뒤 구내식당 CCTV 338장(seed1
   `cam1_1hour_*`, person 만 4,992박스, test 162장) 삭제(11-03 에 했다) → 이미지 추가 → legacy 를 **크롭 여부 → 카메라·출처** 순으로 다시 묶기 →
   (사용자가) 분할 재배정 → visible-3 으로 새 기준선. 그 전의 test 점수(11-03 점검 전 `history.jsonl` 6줄 포함)와는 견주지 않는다.
   - legacy CCTV 5,426장의 크기는 3,501가지다(한 번만 나온 크기가 3,396) — `스크린샷 …`(1,959장)·`T#_#_#_C_D_R_#`(443)·
     `cctv#_#`·`duque_de_caxias-#` 등은 크기가 제각각이라 화면을 잘라 낸 크롭이고, 높이만 1080·720 으로 같은 계열은 한 카메라
     화면을 가로로 자른 것일 수 있다. `640×640`(1,190장)은 로보플로가 늘린 것이라 크기로는 크롭 여부를 모른다.
   - **test_baseline 의 `cctv`(seed 중 드론 아님)에는 CCTV 가 아닌 사진이 섞여 있다** — 숫자 이름 계열의 웹 사진(COCO 류)·
     `img-bicycles-…`(217장)·폰 사진. 초안은 `tools/legacy_groups.py`(장면 종류 116개 중 2번 728장이 대부분 지상·근접 사진).
     새 기준선의 CCTV 도메인은 분류가 끝난 뒤 그 결과로 다시 정할 것.
   - **'이미지 추가' 후보 — 공개 데이터셋 (11-04 조사, docs/provenance.md '공개 데이터셋')** — VIRAT 은 영상 329클립·
     11장면을 11-07 에 다 받아 장면당 50장(550장)을 검수 대기로 넣었다(`tools/import_virat.py` — 운영 모델 예측 + 모델이
     못 찾은 VIRAT 박스, 장면 = 카메라, 주석 파일이 없는 영상 14개는 뺐다). 검수는 사용자가 하고, 평가에 쓸 장면은 다음 학습 전에
     `cameras.yaml` 의 `split:` 으로 정한다. MOT17 은 보류(사람을 가려진 부분까지 몸 전체로 그린 라벨이다).
   - **분류는 `images.domain`(cctv·noncctv·drone, NULL=미분류)에 담는다** — 검수 UI 의 Q·W·E 버튼과 도메인 필터(같은 장면
     묶음째 정하던 `묶음째` 는 11-04 사용자 결정으로 뺐다). 라벨이 아니라 `updated_at`·변경 이력에 남기지 않는다.
     `test_baseline.py` 는 11-04 부터 이 칸으로 가른다. 분할(`split.py`)은 아직 이 칸을 보지 않는다.
   - **11-04 점검(`tools/domain_review.py`, 시트로 눈 확인 — 메모 `data/visible/domain_review/검토_20261104.md`)** — 손으로
     나눈 도메인은 거의 일관됐다. 고칠 것 6장(1038·1553·1554·1589 → 비CCTV, 6689·6697 → 드론)과 기준 두 가지(위 '라벨
     기준' 의 이미지 도메인)는 사용자가 같은 날 정리했고, 다시 돌리면 의심 440장·새로 나온 것 0이다 — 남은 것은 임베딩이
     기기(폰·라이브캠)를 못 가린 것, 숫자 이름 묶음 g000942(180장 — CCTV 크롭 117·창밖 폰 사진 63)가 두 출처를 합친 것,
     그 기준대로 둔 드론이다. 옛 test_baseline 규칙과는 크게 다르다(옛 CCTV 5,312장 중 913장이 비CCTV·128장이 드론, 옛
     비CCTV 716장 중 236장이 CCTV). **11-04 분할은 test 802장 중 198장이 train 과 같은 잎 태그(카메라)를 썼다**(드론 55/95 —
     주차장 드론 영상·Shibuya 라이브캠 등) — 다시 나눌 때는 태그를 묶음으로 쓸 것(장면 묶음은 30초 스크린샷·숫자 이름에서
     다른 카메라를 합친다). 태그 오타는 `Moive`·`stablilized` 를 고쳤고 `deque de caxias`(파일은 duque_de_caxias, 110장)가 남았다.

## 사용자에 대해

- 한국어로 소통한다.
- 검증을 중시한다. 수치와 근거를 제시하면 잘 받아들이고, 추측을 사실처럼 말하면 정확히 짚어낸다. **모르면 코드나 실측으로
  확인하고 답할 것.**
- 원본 데이터를 옮기거나 지우는 작업은 반드시 먼저 확인을 받을 것(위 '먼저 물을 것').
