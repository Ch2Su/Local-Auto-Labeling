# 로컬 자동 이미지 라벨링 — 작업 인수인계

소량의 손라벨 데이터로 YOLO 를 학습시키고 → 나머지를 자동 라벨링 → 웹에서 검수 → 검수분을 합쳐 재학습하는 루프.
전부 로컬에서 돈다. **방향 (2026-11-03 사용자 확인): CCTV 객체 검출 고도화 — LLM 연동 유고(이상 상황) 감지 시스템의 검출부를 만든다.**
유고 감지 시스템 자체(LLM 연동)는 다른 세션에서 만든다 — 이 저장소는 검출 모델·라벨·학습만 맡는다. 09-18 의 '어떤 이미지든 탐지' 방향은 대체됐다. **11-10 에 이미지를 전부 비우고 풀 해상도 CCTV 캡처로 처음부터 다시 모으기 시작했다**(아래 '현재 데이터 상태') — 드론·비CCTV 를 다시 넣을지와 평가 도메인은 정해진 것이 없으니 필요하면 사용자에게 물을 것. 분할·평가에서 특정 도메인·카메라가 한쪽(train/val/test)에 몰리지 않게 할 것은 그대로다.

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
- **`labeler.py split --reset --yes` 는 사용자가 직접 돌린다** — 리셋하면 학습에 쓴 이미지가 test 로 갈 수 있어 그때까지 잰
  test 점수를 쓸 수 없게 된다.
- 보관 폴더 `E:\local_auto_labeling_archive\` 를 지우거나 옮기는 일 — 11-10 에 비운 옛 데이터가 있고, ROAD PLUS 구간 영상
  (`videos/`)은 거기가 유일한 사본이다.
- **AI Hub 에서 받은 그림(카메라 `ah-*`, 11-22)을 밖으로 올리는 일**(Runpod pod·git) — AI Hub 약관이 승인 없는 제3자 제공을 막고
  국외 반출에 합의를 요구한다(docs/provenance.md '공개 데이터셋'). **Roboflow 에는 올리지 않는다**(11-23 확인 — 비공개
  프로젝트도 Roboflow 가 남에게 라이선스하는 데이터셋에 넣을 수 있고, 올리는 사람이 권리를 보증해야 한다). 학습 세트 묶음을
  pod 로 보낼 때 섞이지 않게 볼 것.
- **Claude 는 AI Hub 그림(`ah-*` 원본·크롭·대조 시트)을 열지 않는다 (11-23 사용자 결정 '열지 마세요').** Read 로 연 그림은 대화와
  함께 모델 서버로 나간다 — 보조 작업(Agent)이 여는 것, 그 그림이 뜬 검수 화면의 화면 캡처도 같은 길이다. 수치·코드로
  확인하고, 눈으로 볼 것은 시트를 만들어 **경로만 알려** 사용자가 보게 한다. 저장소에서 그림을 인터넷으로 보내는 코드는
  `tools/label_qa_claude.py`(크롭을 Claude API 로) 하나이고(11-23 에 찾아봄) 그날부터 그 그림을 건너뛴다(`local_only`) — 그런
  도구를 새로 만들면 같은 규칙을 넣을 것.

키는 환경 변수에만 둔다 — **`RUNPOD_API_KEY` 같은 키 값을 읽거나 출력하지 말 것.**

---

## 환경 (이미 구성됨)

`python` 은 반드시 `.venv\Scripts\python.exe` 를 쓴다 — 시스템 Python 에는 아무것도 없다. Python 3.14 + sm_120 이라
**PyTorch 는 cu130 인덱스에서만** 받는다. `requirements.txt` 가 `+cu130` 까지 고정해서(11-01) CPU 빌드로 조용히 바뀌는 대신
'No matching distribution' 으로 멈춘다. 다시 깔 때는 아래를 돌린다(CPU 빌드가 깔렸어도 이것으로 바뀐다 — `==2.14.0` 만
적으면 '이미 설치됨' 으로 넘어간다):

```bash
.venv/Scripts/python.exe -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

블루스크린(09-12 `0x133 DPC_WATCHDOG_VIOLATION`, Ultralytics `val` 무렵)과 버그체크 0 종료(09-16, WSL 학습을 정상 중단한
뒤)가 있었다 — 원인은 모르고, 같은 작업을 그 뒤 여러 번 돌렸을 때는 문제없었다.

**Runpod (클라우드 GPU 대여).** MCP 서버 `runpod` 는 11-02 에 뺐다. 다시 붙이는 명령·과금·GPU 고르기·pod 에서의 준비(Pillow·
`/root`·RAM 한도)는 [docs/runpod.md](docs/runpod.md) — **pod 를 띄우기 전에 읽을 것.** 고른다면 RTX 6000 Ada 이고, 멈춘 pod 도 볼륨
요금이 나가며 다시 안 켜질 수 있다. **집 데스크탑(i5-14400F·RTX 5060 Ti 16GB, 2026-11-12 구매 검토)에서 직접 세팅·운영하는 길은 [docs/desktop-setup.md](docs/desktop-setup.md)** — 사용자가 Claude·MCP 없이 직접 해 보려고 쓴 런북이다(노트북 8GB 의 한도·경로 `/mnt/e` 는 그 PC 에서 `/mnt/c` 로 읽는다).

**Roboflow.** 09-28 에 claude.ai 커넥터로 붙였는데 11-02 에는 커넥터 목록에 없다(다시 쓰려면 claude.ai 커넥터 설정에서).
**워크스페이스는 비어 있다** — 스토리지 크레딧 때문에 사용자가 09-28 에 데이터를 전부 지웠고 휴지통도 10-02 에 비웠다.
**그래서 옛 시드는 E: 보관 폴더와 C: 백업이 유일한 사본이다**(아래 '현재 데이터 상태'·'다른 디스크 백업').
- 데이터를 받을 때는 **COCO 로**(함정 4). Universe 검색은 조회만 한다(크레딧이 드는지는 확인하지 않았다).
- **다시 올리게 되면 먼저 볼 것 (11-23 요금표·요금제 문서로 확인 — docs/provenance.md 의 AI Hub 약관 항목):** 새 워크스페이스는
  설정 'Data Licensing and Model Training' 이 켜진 채 시작해, 비공개 프로젝트의 그림도 Roboflow 가 제품 개선과 남에게
  라이선스하는 데이터셋에 쓸 수 있다(끄는 것은 무료 — 이미 모은 것은 안 지운다). 옛 'Public' 워크스페이스는 전부 Universe 에
  공개된다. 우리 워크스페이스가 어느 쪽인지는 확인하지 않았다 — CCTV 캡처(실제 장소가 찍힌다)를 올리기 전에 볼 것.
- **남의 데이터셋을 넣을 때:** 시드(COCO) 경로는 라벨을 **검수 없이 곧바로 approved** 로 넣고, config 의 `classes` 와 이름이
  다른 카테고리(`pedestrian`·`SUV`·`Pickup Truck` 등)는 **버린다**(`coco_io.read_coco`). 미라벨 폴더로 넣으면 라벨을 읽지
  않는다. 그래서 카테고리 이름을 먼저 우리 클래스로 바꾸고, 우리 클래스를 **빠짐없이** 라벨한 데이터인지 볼 것 — 사람이
  빠진 이미지가 승인되면 모델이 사람을 배경으로 배운다. Objects365 의 `Scooter` 는 오토바이형이고 우리 scooter 는 전동 킥보드다.

**Objects365 는 11-03 에 프로젝트에서 뺐다**(DB 에 o365 행 없음, 원본 tar 도 11-04 에는 그 자리에 없었다). 다시 넣게 되면
[docs/provenance.md](docs/provenance.md) 'CLAUDE.md 에서 옮겨 온 메모' 의 Objects365 를 볼 것 — 넣은 뒤 `labeler.py label --all` 금지.

## git

**2026-10-02 에 저장소로 만들었다**(브랜치 `main`, `core.autocrlf=false` — 파일을 바이트 그대로 둔다). **데이터와 DB 는 git 이 아니라 백업으로 지킨다**(빼는 것은 `.gitignore`).
- **원격은 GitHub `Ch2Su/Local-Auto-Labeling` — public 이다(11-02).** **push 하는 순간 누구나 본다** — 올리기 전에 담긴
  내용을 볼 것. **올린 커밋은 다시 쓰지 말 것**(amend·rebase 뒤 force push 금지 — 칠판이 커밋을 해시로 가리킨다). 11-02 에
  커밋 메일을 감추느라 이력을 한 번 다시 쓴 것(사용자 결정)만 예외다. 그 옛 이력은 로컬 `refs/archive/pre-noreply/main` 에
  있다 — **이 ref 는 push 하지 말 것**(옛→새 해시 대응표 `.git/noreply-rewrite-20261102.map`, 옛 저장소
  `Local-Auto-Labeling-old` 는 private). **public 에 올라간 것은 이력을 고쳐 force push 해도 GitHub 에 옛 커밋이 해시로
  남는다**(11-02 에 새 저장소로 옮긴 이유) — 키·개인정보가 올라갔으면 지워졌다고 여기지 말고 바로 사용자에게 알릴 것.
  11-08 에 GitHub 계정 이름이 `chisu0208` → `Ch2Su` 로 바뀌어(id 42445310 그대로) origin 주소도 바꿨다(사용자 결정). 옛 주소는
  아직 리다이렉트되지만 옛 이름을 남이 가져가 같은 이름의 저장소를 만들면 끊긴다 — 다른 클론도 `git remote set-url` 할 것.
- **실제 장소 이름은 git 에 넣지 말 것 — 문서·코드 주석·커밋 메시지 모두.** `cameras.yaml` 의 `place`(도로·교량·휴게소·IC·
  지사 이름, 화면에 찍힌 지명 글자)는 카메라 키(`hw-04` 처럼)로 쓴다. 11-09 에 안 올린 커밋 16개 중 13개에서 이것을 빼느라
  이력을 다시 썼다(사용자 요청 — 올리기 전이라 가능했다). 빼기 전 이력은 로컬 `refs/archive/pre-placename/main` 에 있다 —
  **이 ref 도 push 하지 말 것**(대응표 `.git/placename-rewrite-20261109.map`). **카메라 키도 지명이 드러나지 않게 짓는다** —
  같은 날 옛 키(로마자 지명)였던 고속도로 카메라 9대를 `hw-01`–`hw-09` 로 바꾸고(DB·폴더·파일 이름까지 — docs/provenance.md)
  안 올린 커밋의 옛 키도 한 번 더 다시 썼다(대응표 `.git/camkey-rewrite-20261109.map`). 새 카메라는 다음 번호로 등록한다.
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

도구마다 하는 일·옵션·조심할 것은 [tools/CLAUDE.md](tools/CLAUDE.md) 의 표에 있다(2026-11-17 에 이 파일에서 옮겼다 — `tools/` 의
파일을 다룰 때는 저절로 읽힌다). **도구를 돌리기만 할 때도 먼저 그 표에서 그 도구의 줄을 읽을 것.** 있는 도구:
`test_baseline`·`label_qa`·`label_qa_claude`·`merge_class`·`import_o365`·`import_virat`·`import_aihub_pm`·`import_roadplus`·`ovd_propose`·`import_proposals`·`crop_images`·`mask_images`·`finetune`·`probe_scale`·`extract_frames`·`auto_capture`·`its_incident_clips`·`legacy_groups`·`domain_review`·`prune_seed_json`·`build_subset_dataset`·`scheduled_stop`. 늘 지킬 것:
- **VIRAT·Objects365·AI Hub(`import_aihub_pm`)를 넣은 뒤 `labeler.py label --all` 금지**(넣어 둔 박스를 예측으로 덮는다).
- 그림 파일을 바꾸는 도구(`crop_images`·`import_virat --recrop`)를 돌린 뒤에는 검수 화면을 새로 고칠 것.
- `import_proposals --apply` 는 승인 이미지를 검수 대기로 돌린다. `mask_images` 는 11-24 부터 test 이미지에도 넣는다(아래 '가릴 영역').

## 회귀 테스트

`tests/`(2026-11-12 — 표·사용법은 README '회귀 테스트'). **코드를 고친 뒤 돌릴 것:** `.venv\Scripts\python.exe -m pytest`
(11-24 에 514개·75초쯤. 임시 폴더의 DB 만 쓰고 GPU·모델은 안 쓴다. pytest 는 기본 설치에 없다 — `requirements.txt` 끝).
- **화면(JS) 테스트는 node 가 없어도 돈다 (11-16).** `tests/test_js_parity.find_node` 가 `NODE_BIN` → PATH 의 node →
  `C:\Program Files\nodejs\node.exe` → VS Code(`Code.exe` 를 `ELECTRON_RUN_AS_NODE=1` 로)를 차례로 찾는다. 하나도 없을 때만
  건너뛴다 — **건너뛴 테스트가 있으면 화면 코드를 확인한 것이 아니니 그렇게 보고할 것**(집 데스크탑은 그 전까지 38개가 늘 건너뛰어졌다).
- **버그를 고치면 그 버그를 다시 내는 입력을 테스트로 남긴다** — 일회용 스크립트로 확인하고 버리지 말 것.
- 테스트의 `cfg`(`tests/conftest.py`)는 `data.*` 경로를 전부 임시 폴더의 절대경로로 적는다. 새 경로 설정을 쓰는 코드를 시험하면
  거기에도 넣을 것 — 빼면 `config.resolve` 가 저장소의 `data/` 를 가리킨다.
- 테스트 파일마다 무엇을 지키는지와 화면(JS) 함수를 글자로 떼는 규칙은 [tests/CLAUDE.md](tests/CLAUDE.md) 로 옮겼다(2026-11-17 —
  `tests/` 의 파일을 다룰 때 읽힌다). 화면 함수의 이름·선언 모양을 바꿔 JS 테스트가 깨지면 그 테스트 파일의 `PARTS` 를 맞춘다.
  늘 지킬 것 둘: **좌표가 '같은 자리' 인지는 반올림한 값을 견주지 말고 `masks.same_rect`(화면 `sameRect`)로 본다** ·
  **Ultralytics 를 올리면 `tests/test_eval_test.py` 부터 볼 것**(val 에 `conf=0.25` 를 넘겨 고치지 말 것).

## 라벨 기준

전문·차종별 실물 예시·전수 검사 기록은 [docs/label-rules.md](docs/label-rules.md) — 판정 기준을 바꾸면 그 파일도 같이 고칠
것. **배달 로봇은 라벨하지 않는다**(드론 주차장 장면의 흰 소형 차량 — 모델이 car 로 잡아 missing 후보로 나와도 무시할 것).

**사람을 포함해 라벨하기 애매한 물체는 가릴 영역(`M`)으로 덮는다**(11-09 사용자 결정, 전문은 docs/label-rules.md) — 박스를
짐작해 그리지도, 비워 두지도 않는다. 몇 px 조각이라도 덮고, 라벨된 박스와 조금 겹쳐도 괜찮다. 차종만 애매한 차는 여기에 들지
않는다(아래 판정 순서대로).

**사고 난 차도 원래 차종으로 그리고 이미지 태그 `incident` 를 붙인다** — crashed 클래스는 11-15 에 보류했다(사용자 결정, 이유는
docs/label-rules.md). 사용자가 다시 꺼내기 전에는 권하지 말 것.

**차량 클래스 판정 순서** — 크기나 용도가 아니라 **차체가 한 덩어리인가**로 본다. 위에서부터 처음 걸리는 것으로 정한다.
**11-02 에 van 을 car 로 합쳤다**(사용자 결정) — 가를 것은 pickup·truck·bus 와 그 밖(car)이다.

| | 묻는 것 | 결과 |
|---|---|---|
| 0 | 경광등·출동 도색·표지가 **보이는** 출동 차량 — 소방·구급·경찰·도로공사 순찰·견인차, 뒤에 LED 화살표판을 단 차(11-09 추가 — 애매하면 아래 차종으로). 작업용 회전등만 단 화물·공사 트럭(카고 크레인 포함)은 아니다(11-23). 경찰 오토바이는 motorcycle + 이미지 태그 `police-motorcycle`(11-23 — 더 모이면 다시 정한다) | **responder** |
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
- **낯선 각도에서 실루엣만 보고 판정하지 말 것.** 내가 옛 DB 의 117(베팅차)·3428(스타리아)·2671(아발란치)을 오판했고 그
  라벨이 맞았다. 축소 대조 시트로는 적재함이 열렸는지도 자주 틀린다. 후보를 올리기 전에 **원본 해상도로**
  다시 보고, 그래도 모르면 사용자에게 물을 것.
- van 을 다시 나누게 되면 그때의 기준(`git show e788f84:CLAUDE.md`)과 합치기 직전 DB(보관 폴더의
  `data/visible/labeling.db.bak_merge_20261102_133352`)를 볼 것.

**이미지 도메인(`images.domain`) — 공중에서 찍었거나 드론이 찍었으면 드론이다**(11-04 사용자 결정, 예시는 docs/label-rules.md).
영화의 공중 장면·헬기 추격 화면도, 드론 영상에서 눈높이로 내려온 프레임도 드론이다(뒤쪽은 '일단'). 이런 이미지는
`tools/domain_review.py` 가 이웃·태그·장면 묶음과 다르다고 계속 띄운다 — 다시 오류로 올리지 말 것. CCTV 와 비CCTV 의
경계는 아직 적어 둔 기준이 없다(애매하면 물을 것).

**EXIF 회전 이미지**(옛 데이터에 Orientation=6 이 49장 있었다)의 라벨은 회전을 적용한 화면 기준 정규화 좌표이고, **DB 의
width/height 는 회전 전 원시 크기다**(`labeler.py check` 도 원시 크기와 견준다) — 정사각이 아니면 가로·세로가 뒤바뀐다.
그래서 **화면·SAM·크롭 좌표는 회전 적용 크기로**(`yolo_io.display_size`, 크롭은 `ImageOps.exif_transpose`) 계산한다 — PIL 로
그냥 열면 박스가 엉뚱한 자리로 보이지만 라벨은 정상이다.

## 현재 데이터 상태

**2026-11-10 에 이미지를 전부 비우고 처음부터 다시 모은다**(사용자 결정 — 가릴 영역 `M` 이 생겨, CCTV 화면을 잘라 낸 크롭·
스크린샷 대신 풀 해상도 캡처를 통째로 쓰고 필요 없는 곳은 덮으려고).

- **옛 데이터는 `E:\local_auto_labeling_archive\20261110\` 에 있다**(같은 디스크로 옮겼다 — 담긴 것·되살리기는 그 폴더의
  `ARCHIVE_INFO.txt`). 이미지 5,931장(legacy 시드·VIRAT·ROAD PLUS 프레임)과 비우기 직전 DB·작업별 DB 사본·하루 사본·분석
  결과, ROAD PLUS 구간 영상(`videos/` — **C: 에 없는 유일한 사본**)이다. 이미지와 DB 는 C: 백업 세 폴더에도 있다(아래).
- DB 는 같은 파일을 비웠다 — 클래스 10개와 id 번호(AUTOINCREMENT)는 이어지고 태그는 지웠다. **문서에 적힌 이미지 번호(8,031
  이하)는 옛 DB 의 것**이다 — 보관 폴더의 `labeling.db` 에서 찾는다.
- 비우기 전의 이 절(옛 데이터 수·legacy·VIRAT·분할 묶음·점수 메모)은 `git show 25a3de6:CLAUDE.md`. 옛 test 점수
  (`runs/test_baseline/history.jsonl`, docs/experiments.md)는 새 test 와 견주지 않는다.
- 운영 가중치(`models/best.pt` — 아래 '가중치')는 그대로 두고 새 이미지의 자동 라벨링에 쓴다. responder·cone 은 모르니 그 둘은
  OWLv2 후보(`tools/ovd_propose.py` → `tools/import_proposals.py`)나 손으로 채운다.
- **11-22 에 AI Hub '개인형 이동장치 안전 데이터' 100장을 검수 대기로 넣었다**(사용자 결정 — 시범. source `aihub_pm`, 카메라
  `ah-0001`–`ah-0100`, `tools/import_aihub_pm.py`, 경위·수치는 docs/provenance.md '공개 데이터셋'). 박스는 운영 모델 예측 + AI Hub
  단서로 넣은 후보(출처 `aihub`)라 전부 사람이 봐야 한다. **카메라 키는 '화면이 다른 것만 골랐다' 는 어림이다** — 이 데이터에는
  카메라 번호가 없고 같은 장소를 여러 자리에서 찍었다. **그래서 AI Hub 그림은 전부 train 에 둔다 (11-23 사용자 결정)** — 그날
  등록 파일의 `ah-*` 100대를 `split: train` 으로 고정했고(`import_aihub_pm.py --pin-train --apply`), 도구가 새로 등록하는
  카메라도 train 으로 적는다. 사용자가 손으로 더 넣을 수 있다 — **카메라 키를 `ah-NNNN` 으로 등록해야** train 고정·겹침
  점검이 걸리고(scan 은 출처를 `collected` 로 넣는다), 등록한 뒤 `--pin-train` 을 다시 돌린다. `ah-` 로 시작하는데 그 꼴이
  아닌 키(`ah-101`)는 고정·점검에서 빠지니 그 도구가 알린다 — 밖으로 보내지 않는 규칙(`local_only`)만은 `ah-` 로 시작하는
  키를 전부 막는다(11-23 Gemini 리뷰). 넣기 전에
  겹침(같은 그림·같은 영상·같은 자리)은 `import_aihub_pm.py --check <폴더>` 로 잰다 — Claude 가 그림을 못 보니 이 숫자가 전부다
  (docs/provenance.md 의 그 항목). **고정한 그림도 분할 비율에 든다** — AI Hub 가 전부 승인되면 직접 모은 그림의 train 몫이
  66% → 55% 로 준다(사본 실측, docs/next.md 5번): 첫 분할 때 사용자와 정할 것.
- **휴지통 비우기는 원본 파일까지 지운다**(`tidy.purge_trashed`) — 새로 모은 사진은 다른 사본이 없을 수 있으니 비우기 전에 물을 것.
- **클래스는 10개다** — 11-09 에 responder(8, 출동 차량)·cone(9, 라바콘)을 붙였다(사용자 결정 — DetectView 유고 단서용, 판정은
  docs/label-rules.md '출동 차량·라바콘'). 그 전 8개: pickup 은 09-20 에 사용자가 검수 UI 에서 추가했고, 같은 날 추가한 van 은
  11-02 에 car 로 합쳤다. **클래스 번호는 순서가 전부다** — config.yaml 의 `classes` 에 **뒤에만 붙이고** 중간에 끼우거나 빼지
  말 것(박스가 번호로 가리킨다). 맨 끝 클래스를 없앨 때는 `tools/merge_class.py`.
- **점수는 클래스별로 비교할 것** — 클래스 수·라벨 기준이 다른 모델끼리는 전체 평균이 다른 것을 잰다(7클래스 모델은 pickup 을
  truck·car 로 잡아 그 둘까지 깎였다). test_baseline 의 CCTV·드론·비CCTV 는 사람이 정한 `images.domain` 이다(비어 있으면 카메라
  등록으로 짐작 — 아래 '출처 기록').
- `labeler.py split` 은 읽기 전용이 아니다(미배정을 배정해 DB 에 쓴다) — DB 만 보려면 `sqlite3` 를 `mode=ro` 로 열 것.

**묶음 단위 계층화 3분할 (2026-09-16, `split.py`·`grouping.py`).**
- **test(15%)는 학습에도 best.pt 고르기에도 쓰지 않는다** — 모델끼리 비교는 test 로 한다(`tools/test_baseline.py`).
- **같은 장면(dHash·이름 연번 묶음)은 통째로 한쪽에만 넣는다.** 계층화는 클래스 비중만 맞추고 출처는 보지 않으니 도메인 쏠림은
  따로 볼 것. **기존 이미지는 움직이지 않는다** — 새 이미지만 자기 묶음을 따르거나 계층화로 들어가고, 두 분할의 묶음을 이어
  붙이는 새 이미지가 나오면 옮기지 않고 `[알림] … 두 분할에 걸칩니다` 만 띄운다(`moved` 는 reset 이 아니면 0 이어야 한다).
- **손으로 옮길 때는 val·test → train 방향만** — 반대면 학습한 이미지가 평가셋에 들어간다.
- **이름 계열은 카메라 안에서만 잇는다**(다른 카메라끼리는 dHash 로만 — 묶는 규칙의 나머지는 `grouping.py` 머리말). 누수는
  `labeler.py split --leak` 으로 다시 잰다.
- 두 분할에 걸친 묶음은 알림으로 뜨고 기존 배정은 고정이라 옮기지 않는다(11-05). 계층화는 섞인 묶음을 다수결 분할로 센다(비중이
  조금 어긋남 — 고치지 않았다). 누수를 없애려면 분할 재배정(사용자 결정).
- `train.test_ratio`(또는 val_ratio)를 0 으로 바꿔 이미 배정된 분할이 비율에서 빠지면, reset 이 아닌 한 **아무것도 쓰기 전에
  멈춘다**(11-05 — 전에는 test 가 경고 없이 train·val 로 흡수됐다).

**폴더는 `data/visible/` 다.** DB 는 이미지를 **절대경로**로 들고 있어 폴더를 옮기면 `images.path` 도 치환해야 한다(스니펫은
README '새 PC 에서 시작하기' 5단계). **경로를 고치기 전에는 스캔·학습·`prune`·정리 버튼을 쓰지 말 것** — 스캔은 시드를 검수
전 라벨로 다시 넣고(그래서 DB 의 절반 넘게 '파일 없음' 이면 멈춘다, `scan --force` 로 강행), 웹 UI 의 정리(tidy)는 뒤이어
`prune` 을 돌려 '파일 없음' 행을 **검수 상태·라벨째 지운다**(절반 넘게 없을 때만 멈추니, 일부만 되살린 상태면 그만큼 지운다).
- **새 수집은 전부 출처(카메라)를 밝힌다.**
- 시드: `seed_coco: data/visible/seed`(지금은 없다 — 없으면 시드 없이 돈다)는 하위 폴더까지 COCO 주석 json 을 **전부** 찾아
  (`coco_io.find_annotation_files`) 검수 없이 승인으로 넣으니, 검토 전 데이터나 json 사본을 그 아래에 두지 말 것(회차마다
  `seedN/` 에 `coco/*.json` + `images/`). 시드 사진을 휴지통 비우기로 지우면 json 에 이름이 남아 scan 이 알리니
  `tools/prune_seed_json.py` 로 뺀다.
- **출처 기록 (11-03, [docs/provenance.md](docs/provenance.md))** — 새 수집분은 `unlabeled/<카메라키>/…` 에 두고 카메라를
  `data/visible/cameras.yaml`(실제 장소 — git 에 넣지 말 것)에 먼저 등록한다. 등록 안 된 폴더가 있으면 scan 이 아무것도 넣지
  않고 멈춘다. 분할은 **카메라째 한 묶음**이고 `cameras.yaml` 의 `split:` 으로 고정할 수 있다
  (고정이 다른 카메라가 dHash 로 이어지면 분할이 멈춘다). split 보고의 `카메라별:` 줄로 어디로 갔는지 볼 것.
  `test_baseline.py --domains` 의 `cctv`·`drone`·`noncctv` 는 `images.domain`, 비어 있으면 카메라 등록으로 짐작한다(설치 drone → 드론, kind cctv → CCTV, phone → 비CCTV, 그 밖은 `undomain`).
- `dataset/` — 학습할 때마다 DB 로 새로 만든다(Windows 학습은 `.npy` 디스크 캐시 약 14GB 를 더 얹는다). **test 는 이 폴더 말고
  `tools/test_baseline.py` 로 잴 것.** 이어하기는 없으면 다시 만든다(`ensure_resume_dataset`).
- `labeling.db` — **작업물 그 자체**(함정 5). 옆의 `backups/` 에 검수 서버가 하루 한 번 사본을 남긴다(최근 14개 — 켤 때, 켜 둔 채 날이 바뀌면 그날 첫 변경 요청 앞에서).
  `classes.txt` 는 DB 옆이다. `thumbs/`(11-04)는 이미지 보기 썸네일 캐시 — 지워도 되고 백업하지 않는다. `capture/`(11-10)는
  영상 캡처 탭의 임시 세션(올린 영상 사본·뽑은 프레임 — 넣거나 닫으면 서버가 지운다)이라 백업하지 않는다(수 GB 일 수 있다).
  `autocapture/`(11-12)는 `tools/auto_capture.py` 의 처리 기록 `videos.jsonl`(지우면 처리한 영상을 다시 올린다)·후보 프레임별
  기록 `frames/`·대조 시트 `sheets/` 다 — 영상 경로·검출 기록이라 git 밖(data/)에 둔다.
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
**`best.pt`·`best.prev.pt` 는 학습이 끝까지 갈 때마다 밀려난다**(11-16 부터 중단한 학습은 건드리지 않는다 — 함정 8) —
`best.prev.pt` 는 한 벌뿐이니 운영할 가중치는 늘 `best_<이름>.pt` 로 따로 둔다.

**다른 디스크 백업은 `C:\local_auto_labeling_backup\20261031\` 와 그 위에 차례로 더하는 `20261105\`·`20261110\`** 다(C: 는 E: 와 다른 물리 디스크).
셋을 차례로 덮으면 11-10 에 비우기 직전의 이미지·DB 가 된다. **그 뒤에 새로 모은 데이터는 아직 C: 에 없다.**
담긴 것·다시 뜨는 명령·되살리는 순서는 [docs/backup.md](docs/backup.md) — 새로 뜨기 전에 이전 날짜 폴더를 지울지 먼저 물을 것.

## WSL 학습 (`wsl/`, 웹 UI 'WSL 학습' 버튼)

학습은 WSL 이 **약 4배 빠르다**(09-12 같은 조건 전체 학습: epoch 258초 → 62초 — ram 캐시·워커 8·fork 덕인데 몫은 가르지
않았다). 환경은 WSL Ubuntu 의 miniconda `~/miniconda3/envs/yolo_final`(ultralytics 8.4.142, torch 2.11+cu130). **패키지를 올릴
때는 드라이런으로 torch 를 안 건드리는지 먼저 볼 것** — PyPI 의 torch 가 cu130 빌드를 덮어쓸 수 있다. 되돌릴 기준은
`~/yolo_final_freeze_before_upgrade.txt`(09-11 업그레이드 전 freeze).

버튼의 흐름(데이터셋 → 복사 → 학습 → 승격)·[중단] 은 `wsl_train.py` 머리말, 설정은 `config.yaml` 의 `wsl:`.
- **WSL 버튼은 Windows 학습 버튼과 같은 인자로 학습한다**(`trainer.train_args(cfg)` → `train_wsl.py --bare --hyp KEY=VALUE`,
  `--dry-run` 으로 확인). `train_wsl.py` 는 Windows 프로젝트 코드를 import 하지 않는다. `wsl.exe` 가 명령줄을 bash 로 넘기므로
  값은 영숫자와 `_ . + - =` 만 넘긴다(`wsl_train.hyp_tokens`).
- **DB 는 WSL 로 가져가지 않는다** — `/mnt/e` 로는 SQLite 잠금이 안 먹어 검수 서버와 함께 쓰면 손상될 수 있다. 이미지·라벨만
  복사한다. 프로젝트를 늘리면 WSL 데이터셋 폴더를 **반드시 다르게** 둘 것(`--delete` 동기화가 서로 덮는다).
- **서버를 죽여도 WSL 학습은 계속 돈다** — 그때는 STOP 파일을 직접 만든다. `--stop-at N` 은 N epoch 을 마치면 정지 파일처럼
  멈춘다(lr 스케줄은 그대로, `--resume` 으로 잇는다).
- 'WSL 이어하기' 는 train_wsl.py 가 쓸 수 있는 resume.pt·last.pt 만 띄우고, `--resume` 은 results.csv 를 체크포인트 epoch
  까지 자른다(원본 `.csv.bak`).
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
- 손으로 돌리는 명령(동기화·학습·STOP·`--resume`)은 `wsl/train_wsl.py` 머리말에 있다.

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
   - **사람이 멈춘 학습은 운영 가중치를 바꾸지 않는다 (11-16 사용자 결정).** 끝까지 간 학습(조기 종료 포함)만 올리고 이전
     것을 `best.prev.pt` 로 남긴다. 멈춘 학습은 로그 끝에 그 학습의 best.pt 경로를 찍는다 — 쓰려면 손으로 복사한다. 규칙은
     `trainer.promote_if_finished` 한 곳이고, 예외 둘(운영 가중치가 없을 때·마지막 epoch 에서 멈춘 때)은 그 머리말에 있다
     (`tests/test_promote.py`).
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

전문은 [src/autolabel/CLAUDE.md](src/autolabel/CLAUDE.md) 로 옮겼다(11-10 — 그 폴더의 파일을 다룰 때 읽힌다. `tools/` 만 고치더라도
박스 저장·가릴 영역·검수 서버를 건드리면 먼저 읽을 것). 늘 알아야 할 것만:
- **편집은 0.6초 뒤 자동 저장된다** — UI 를 시험하면 실제 작업물이 바뀐다. 시험은 **DB 복사본** 서버에서, `backup.keep: 0` 으로.
- **박스를 바꾸는 길은 전부 `store._replace_boxes` 를 지난다**(변경 이력). 그림 파일을 바꿀 때는 `store.replace_file` —
  가릴 영역이 있는 이미지는 `masks=` 를 줘야 한다. 이력의 박스·영역 목록은 11-11 부터 `hist_lists` 에 한 번만 두고
  `box_history` 는 `#id` 로 가리킨다 — 도구에서 before·after 를 SQL 로 바로 읽지 말고 `store.box_history`·`history_entry` 로.
- **가릴 영역**은 DB `masks` 표에만 있고 학습 세트·이미지 내보내기만 칠한다. **test 는 칠하지 않고 '채점에서 빼는 자리' 로 쓴다**
  (11-24 사용자 결정 — 반 넘게 덮인 라벨은 test 라벨 파일에서도 빼고, 칠할 자리를 `ignore/…png` 로 남겨 평가 도구가 그 자리의
  예측을 세지 않는다: `wsl/eval_test.py` 의 `IgnoreValidator` 를 `tools/test_baseline.py`·`test_tiles.py` 도 쓴다. 전에는 test
  이미지에 영역을 둘 수 없었다 — masks.py 머리말). 덮는 몫은 `masks.covered` 와
  `app.js` 의 `maskCover` 가 같은 식이다 — 고치면 둘 다. **사각형과 다각형(11-19)** 이 있다 — 다각형은 `poly` 열의 꼭짓점
  JSON(x1·y1·x2·y2 는 바깥 사각형)이라 **`masks` 를 SQL 로 바로 읽지 말고** `store.get_masks`·`masks_by_image`(행은 `masks.from_row`)로.
- **영상 캡처 탭·카메라 등록/고치기/지우기/키 바꾸기(`capture.py`·`provenance.py`·`camera_rename.py`)·몇 장만 다른 카메라로
  옮기기(`camera_move.py`)·데이터 점검 보기(`health.py`)** 의 규칙은 src/autolabel/CLAUDE.md 에 있다(2026-11-17 에 여기의 요약을
  뺐다 — 같은 글이 거기에 더 자세히 있다). 늘 지킬 것만: 출처(카메라) 없이는 이미지를 넣지 않는다(분할이 카메라째다 — 모르면
  자동 카메라 `vid-NN`) · **DB 의 `camera` 만 SQL 로 바꾸지 말 것**(파일은 옛 카메라 폴더에 남아 정리(tidy)·분할 묶음이
  어긋난다 — 이미지 보기의 `옮기기`) · **데이터의 박스 수·크기를 말할 때는 데이터 점검(`health.report`)의 수치를 쓰고, SQL 로
  바로 셌으면 가려진 박스를 뺐는지 밝힐 것.**
- 코드를 고친 뒤에는 검수 서버를 다시 켜야 반영된다. TestClient 는 `base_url="http://127.0.0.1:7860"`.

---

## 다음에 할 일

목록·수치·경위는 [docs/next.md](docs/next.md) 로 옮겼다(2026-11-17 — 이 파일을 줄이느라. 번호는 그대로다). **다음 일을 고르거나
학습·데이터 수집·분할·Runpod·자동 캡처를 손대기 전에 읽을 것.** 늘 지킬 것만:
- **첫 학습 전에 분할을 사용자와 정할 것** — 11-16 에 DB 사본으로 미리 돌리니 val 이 카메라 한 대였고 val·test 에 박스가
  하나도 없는 클래스가 여럿이었다(5번).
- **Runpod 로 x·1280 을 재 보는 일은 사용자가 다시 꺼내기 전에는 띄우자고 권하지 말 것**(6번).
- 새로 모은 사진은 C: 백업에 없다 — 새 날짜 폴더로 뜨기 전에 물을 것(5번).
- `tools/auto_capture.py` 는 진짜 서버·진짜 DB 로는 아직 안 돌렸다 — 사용자가 한다(7번).

## 사용자에 대해

- 한국어로 소통한다.
- 검증을 중시한다. 수치와 근거를 제시하면 잘 받아들이고, 추측을 사실처럼 말하면 정확히 짚어낸다. **모르면 코드나 실측으로
  확인하고 답할 것.**
- 원본 데이터를 옮기거나 지우는 작업은 반드시 먼저 확인을 받을 것(위 '먼저 물을 것').
