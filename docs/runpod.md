# Runpod (클라우드 GPU 대여) — 절차·함정

2026-11-10 에 저장소 루트 CLAUDE.md 에서 낱말 그대로 옮겨 왔다(그 파일이 매 세션 읽혀 크기를 줄이느라). 글 속의 '위·아래 …' 는
루트 CLAUDE.md 의 절을 가리킨다.

pod 를 만들기 전에 GPU·예상 시간·비용을 말하고 허락받는 것, 키 값을 읽거나 출력하지 않는 것은 루트 CLAUDE.md '먼저 물을 것' 에 있다.

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
