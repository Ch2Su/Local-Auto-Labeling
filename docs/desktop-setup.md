# 집 데스크탑에서 직접 학습하기 — 세팅·운영 런북

2026-11-12 에 썼다(사용자 요청 — Runpod 를 MCP 로 Claude 에게 맡기던 것을 빼고, 직접 세팅하고 직접 돌려 보려고). **Claude 없이 이 문서만
보고 따라 할 수 있게** 썼다. 명령은 `powershell` 블록이면 Windows PowerShell, `bash` 블록이면 WSL(Ubuntu) 안에서 돌린다.

**전제 부품(2026-11-12 구매 검토)**: i5-14400F · DDR4-3200 32GB · NVMe 1TB · RTX 5060 Ti 16GB · PSU 850W. 부품이 달라졌으면 해당하는 곳만 고친다.
GPU 가 노트북(5060 Laptop 8GB)과 **같은 Blackwell(sm_120)** 이라 환경(cu130·Ultralytics)을 그대로 옮긴다.

## 0. 큰 그림

- **집 PC = 본진.** 저장소·DB·이미지·학습이 전부 여기 있다. 노트북은 학교에서 브라우저로 집 PC 의 검수 UI 에 접속해 검수한다(7절).
- 학습은 노트북과 같다 — 검수 UI 의 `모델 → WSL 학습` 버튼. 코드·`config.yaml` 은 그대로이고, 집 PC 에서 달라지는 것은 이것뿐이다:

| | 노트북 | 집 PC |
|---|---|---|
| 저장소 위치 | `E:\local_auto_labeling` | `C:\local_auto_labeling` (드라이브가 하나라) |
| WSL 메모리(`.wslconfig`) | 48GB | **24GB** (RAM 32GB 의 3/4) |
| VRAM | 8GB | 16GB — 8GB 에서 못 돌리던 l/x·960 이 들어갈 수 있다(5.7) |
| WSL 안의 `/mnt/e/...` | `/mnt/e/local_auto_labeling` | **`/mnt/c/local_auto_labeling`** (문서·주석에 적힌 `/mnt/e` 는 `/mnt/c` 로 읽는다. 웹 UI 버튼은 경로를 자동 변환한다) |

- **데이터는 복사하지 노트북 것을 지우지 않는다.** 노트북 원본이 그대로 백업이 된다(4절).
- 키·토큰은 문서·config 에 적지 않는다(저장소가 public).

## 1. 조립 직후 (BIOS)

- 모니터 케이블은 **그래픽카드**에 꽂는다 — 14400F 는 내장 그래픽이 없어 메인보드 단자에 꽂으면 화면이 안 나온다.
- GPU 보조 전원 케이블은 카드의 단자(8핀 또는 16핀 12V-2x6)에 끝까지. 어댑터보다 PSU 의 전용 케이블이 낫다.
- BIOS 를 최신으로 올린다(13·14세대 CPU 안정성 마이크로코드 패치 — 14400F 가 대상인지는 확인하지 못했다, 최신이 손해는 없다).
- **XMP/DOCP 프로파일이 있는 램이면 켠다**(안 켜면 2133–2666 으로 돌 수 있다). 프로파일이 없는 JEDEC 3200 램(아이티젠 삼성 칩 PC4-25600 등)은 보통 자동으로 3200 으로 뜬다. 14400F 의 공식 최대가 DDR4-3200 이다. 어느 쪽이든 BIOS 에서 3200MHz 인지 본다. 램이 2개면 매뉴얼이 가리키는 슬롯 두 개(보통 A2·B2)에 꽂아야 듀얼 채널이다.
- **가상화(Intel VT-x / Virtualization)가 켜져 있는지** 본다 — WSL2 가 필요로 한다(보통 기본 켬).
- Resizable BAR(Above 4G Decoding)는 켠다. 선택: `AC Power Loss` = `Power On`(정전 뒤 전기가 오면 저절로 켜진다).
- 확인: BIOS 첫 화면에 메모리 **32GB / 3200MHz**, NVMe 1TB, CPU 14400F 가 보이는가.

## 2. Windows·드라이버·기본 도구

1. 새 SSD 에 Windows 11 설치(USB 설치 미디어), Windows Update 를 끝까지.
2. **NVIDIA 드라이버를 최신으로**(Windows 쪽에만 — WSL 안에는 깔지 않는다). 노트북은 621.14 에서 돈다. CUDA 13.0 을 쓰는 PyTorch 라 드라이버가 CUDA 13 을 지원해야 한다.

```powershell
nvidia-smi
```

   GPU 이름·메모리 16GB(약 1만 6천 MiB)·`CUDA Version: 13.x` 가 보이면 된다.
3. **절전을 끈다**(학습·원격 접속 중에 잠들면 안 된다).

```powershell
powercfg /change standby-timeout-ac 0
```

```powershell
powercfg /change hibernate-timeout-ac 0
```

   설정 → Windows Update → 고급 옵션에서 **활성 시간**을 넓게 잡아 학습 중 자동 재부팅을 줄인다.
4. **Git**(git-scm.com)과 **Python 3.14.3**(python.org — **3.14.1 은 안 된다**, torchvision 이 그 버전만 받지 않는다. 설치할 때 `py launcher` 를 켠다)을 깐다.

## 3. 저장소와 Windows 쪽 환경

저장소는 **`C:\local_auto_labeling`** 에 둔다. OneDrive·문서 폴더 아래는 피한다(동기화가 SQLite 파일을 건드린다).

```powershell
git clone https://github.com/Ch2Su/Local-Auto-Labeling.git C:\local_auto_labeling
```

```powershell
cd C:\local_auto_labeling
```

```powershell
py -3.14 -m venv .venv
```

**PyTorch 를 먼저, cu130 인덱스에서**(순서를 바꾸면 CPU 빌드가 들어온다 — README '새 PC 에서 시작하기' 3):

```powershell
.venv\Scripts\python.exe -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

기대: `2.14.0+cu130 True NVIDIA GeForce RTX 5060 Ti`. `+cu130` 이 없거나 `False` 면 첫 설치 명령을 다시 돌린다.

> **여기서부터 데이터를 옮기기 전에는 `labeler.py` 를 돌리지 않는다**(검수 UI 포함) — 빈 `labeling.db` 를 새로 만든다.
> Claude Code 를 이 PC 에서 쓸 때만 `git config user.name`·`user.email`(GitHub noreply 주소)을 먼저 정한다(README 1단계 — 자동 커밋 훅이 쓴다).

## 4. 노트북 → 집 PC 데이터 옮기기

노트북에서 **검수 서버·학습을 끄고** 한다. 외장 드라이브(`X:` 로 적었다)를 거친다. 옮길 것은 README 의 네 폴더다(git 에 없다).

**① DB 는 파일을 그냥 복사하지 말고 sqlite `backup()` 으로 뜬다**(노트북, 저장소 폴더에서 — 먼저 폴더를 만든다):

```powershell
mkdir X:\move
```

```powershell
.venv\Scripts\python.exe -c "import sqlite3; s=sqlite3.connect('data/visible/labeling.db'); d=sqlite3.connect(r'X:\move\labeling.db'); s.backup(d); d.close(); s.close()"
```

**② 나머지는 robocopy**(노트북). `data\visible` 은 DB 파일과 다시 만들 수 있는 폴더를 뺀다 — `docs/backup.md` 와 같은 제외 목록이다. 종료 코드 0–7 이 성공이다:

```powershell
robocopy E:\local_auto_labeling\data\visible X:\move\data\visible /E /XJ /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:8 /XD thumbs capture dataset __pycache__ /XF labeling.db labeling.db-wal labeling.db-shm labeling.db-journal *.tar
```

```powershell
robocopy E:\local_auto_labeling\models X:\move\models /E /COPY:DAT /R:1 /W:1
```

```powershell
robocopy E:\local_auto_labeling\weights X:\move\weights /E /COPY:DAT /R:1 /W:1
```

(`runs\` 는 중단한 학습을 이어 하려는 때만 — 체크포인트가 데이터셋의 절대경로를 들고 있어 같은 경로가 아니면 이어지지 않는다. 텍스트 기록은 git 에 있다.)

**③ 집 PC 에서 제자리로**(`X:` 를 꽂은 뒤):

```powershell
robocopy X:\move\data\visible C:\local_auto_labeling\data\visible /E /COPY:DAT /R:1 /W:1 /MT:8
```

```powershell
copy X:\move\labeling.db C:\local_auto_labeling\data\visible\labeling.db
```

```powershell
robocopy X:\move\models C:\local_auto_labeling\models /E /COPY:DAT /R:1 /W:1
```

```powershell
robocopy X:\move\weights C:\local_auto_labeling\weights /E /COPY:DAT /R:1 /W:1
```

**④ 경로 맞추기 — 반드시 한다.** DB 가 이미지를 `E:\local_auto_labeling\…` 절대경로로 들고 있어, 이대로 두면 전부 '파일 없음' 이다. **README '새 PC 에서 시작하기' 5단계의 파이썬 스니펫**을 그대로 쓴다(`old` 는 `E:\\local_auto_labeling\\`, `new` 는 스니펫이 `ROOT` 에서 알아서 가져온다). **고치기 전에는 scan·학습·prune·정리 버튼을 쓰지 않는다**(파일 없음이 절반 넘으면 scan 이 멈추고, prune·정리는 그 행을 라벨째 지운다). 끝나면:

```powershell
.venv\Scripts\python.exe labeler.py check
```

`이상 없음.` 이 나오고, `labeler.py status` 의 장수가 노트북과 같으면 된다. 같아지기 전에는 **노트북 원본을 지우지 않는다.**

> 2026-11-10 에 DB 를 비웠다 — 옛 데이터는 `E:\local_auto_labeling_archive\20261110\`(ARCHIVE_INFO.txt)에 있다. 옛 것이 필요 없으면 그 뒤에 새로 모은 것만 옮기면 된다.

## 5. WSL2 학습 환경

### 5.1 설치

관리자 PowerShell 에서(끝나면 재부팅):

```powershell
wsl --install -d Ubuntu
```

처음 켜면 리눅스 사용자 이름·비밀번호를 정한다(이 문서에서는 `<리눅스사용자>`).

### 5.2 메모리·CPU 한도

`C:\Users\<윈도우사용자>\.wslconfig` 를 만든다:

```ini
[wsl2]
memory=24GB
processors=12
```

기본값은 RAM 의 절반(16GB)이라 모자란다. 데이터셋 RAM 캐시가 약 5GB(`config.yaml` 의 `wsl` 주석)이고 워커가 따로 쓴다. 나머지 8GB 는 Windows·검수 서버·브라우저 몫이다. 적용:

```powershell
wsl --shutdown
```

### 5.3 conda

WSL(Ubuntu)을 열고:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

설치가 끝나면 WSL 창을 닫았다 다시 연다. **환경 이름은 `yolo_final`, 위치는 `~/miniconda3` 그대로**여야 한다 — `config.yaml` 의 `wsl.python` 이 `~/miniconda3/envs/yolo_final/bin/python` 이다.

```bash
~/miniconda3/bin/conda create -n yolo_final -c conda-forge --override-channels python=3.11 -y
```

(`--override-channels` 는 conda 의 기본 채널 약관 동의 요구를 피한다. 노트북은 Python 3.11.15.)

### 5.4 패키지 — 노트북에서 실측한 버전(2026-11-12)

PyTorch 를 먼저, cu130 인덱스에서. 뒤의 Ultralytics 가 torch 를 바꾸지 않는지 `--dry-run` 으로 먼저 본다(CLAUDE.md 'WSL 학습'):

```bash
~/miniconda3/envs/yolo_final/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

```bash
~/miniconda3/envs/yolo_final/bin/python -m pip install --dry-run ultralytics==8.4.142 pillow==12.1.1
```

출력에 `torch` 를 다시 받는다는 줄이 없으면 실제로:

```bash
~/miniconda3/envs/yolo_final/bin/python -m pip install ultralytics==8.4.142 pillow==12.1.1
```

Pillow 는 11.3 이상이어야 AVIF 를 읽고, 11.0 은 일부 JPEG 을 손상으로 뺀다(requirements.txt 주석).

### 5.5 확인

```bash
~/miniconda3/envs/yolo_final/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

기대: `2.11.0+cu130 True NVIDIA GeForce RTX 5060 Ti (12, 0)`. `False` 면 9절.

### 5.6 사전학습 가중치

`C:\local_auto_labeling\weights\` 에 `yolo26m.pt` 등이 있어야 한다(4절에서 옮겼다). 없으면 Ultralytics 가 학습 시작 때 인터넷에서 받으려 한다.

### 5.7 16GB 에서 어디까지 되는지 재기 (학습 데이터가 생긴 뒤)

노트북 8GB 의 한도(CLAUDE.md '8GB 에서 들어가는 최대치')를 면적으로 늘려 본 추정은 **l960 ≈ batch 7, x960 ≈ batch 4–5** 이지만 실측이 아니다. 실측은 학습으로 한 번 데이터셋이 WSL 에 복사된 뒤에:

```bash
~/miniconda3/envs/yolo_final/bin/python /mnt/c/local_auto_labeling/wsl/probe_vram.py --models m l x --imgsz 640 960 --batches 4 8 16
```

결과는 `runs/vram_probe/results.jsonl`. 16GB 를 넘으면 OOM 없이 공유 메모리로 넘어가 느려지니(함정 9) ms/step 이 튀는 설정은 빼고 고른다.

## 6. 학습 돌리기

```powershell
.venv\Scripts\python.exe labeler.py review
```

`http://127.0.0.1:7860` 에서 `모델 → WSL 학습`. 버튼이 DB 로 데이터셋을 만들고 → WSL 로 복사하고 → 학습하고 → 끝나면 `models/best.pt` 를 바꾼다.

- **끝나면 — 중간에 멈춰도 — `models/best.pt` 가 그 학습의 결과로 바뀌고 직전 것은 `best.prev.pt` 로 간다.** 시험 학습이면 먼저 지킬 가중치를 `best_<이름>.pt` 로 복사해 둔다(지금 운영 가중치는 `models\best_y26m_8cls.pt` 가 따로 있다). 되돌리려면 그 파일을 `best.pt` 로 복사한다.
- **학습 중에는 GPU 를 쓰는 앱(게임·영상·크롬 하드웨어 가속)을 닫는다.** 집 PC 는 모니터가 학습 GPU 에 붙어 있다. epoch 소요시간이 갑자기 몇 배로 늘면 VRAM 이 넘친 것이다(CLAUDE.md 함정 9, 아래 9절).
- **멈추기:** 작업 창의 `[중단]`. 서버를 껐다면 탐색기에서 `\\wsl.localhost\Ubuntu\home\<리눅스사용자>\new_env\yolo\runs_wsl\` 에 **`STOP`** 이라는 빈 파일을 만든다. 다음 배치에서 멈추고 `resume.pt` 를 남긴다. Ctrl+C 는 즉시 종료다.
- **이어하기:** `모델 → WSL 이어하기`.
- 결과 그래프·`results.csv` 는 `\\wsl.localhost\Ubuntu\home\<리눅스사용자>\new_env\yolo\runs_wsl\visible\`.

## 7. 학교에서 쓰기 (원격)

### 7.1 Tailscale

집 PC 와 노트북 **양쪽에** Tailscale(tailscale.com)을 깔고 같은 계정으로 로그인한다. 집 PC 의 주소:

```powershell
tailscale ip -4
```

`100.x.y.z` 꼴이다(이하 `<TS-IP>`). 노트북에서 `tailscale status` 에 집 PC 가 보이면 연결된 것이다.

### 7.2 검수 서버를 그 주소로 연다

집 PC 에서:

```powershell
.venv\Scripts\python.exe labeler.py review --host <TS-IP>
```

노트북 브라우저에서 **`http://<TS-IP>:7860`**. 학습 시작·중단·검수가 전부 이 화면에서 된다.

- **이름(MagicDNS)이나 다른 주소로 열면 400 `Invalid host header`** — 서버는 Host 가 루프백이거나 bind 한 주소일 때만 받는다(`server.py` `_RequestGuard`). 그 주소 그대로 쓴다.
- **`--host 0.0.0.0` 은 쓰지 않는다** — Host 검사가 꺼지고 집 공유기 안의 모든 기기에 열린다. **검수 UI 에는 로그인이 없다.** 공유기에서 7860 을 포트포워딩하는 것은 절대 금지.
- 처음 열 때 Windows 방화벽 허용 창이 뜨면 `개인 네트워크` 로 허용한다. 접속이 시간 초과면 서버가 켜져 있는지·방화벽·`tailscale status` 순서로 본다.
- **검수 서버 창을 닫으면 UI 가 죽는다**(WSL 에서 도는 학습은 계속 돈다 — 멈추려면 STOP 파일).

### 7.3 집을 나서기 전 체크

- [ ] 집 PC 에서 검수 서버를 `--host <TS-IP>` 로 켜 두었다
- [ ] **노트북에서** `http://<TS-IP>:7860` 이 열리는지 집에서 먼저 확인했다
- [ ] 절전 끔(2절 3), Windows Update 대기 재부팅 없음
- [ ] 학습을 걸고 갈 때는 시작 후 첫 epoch 이 정상 속도로 도는 것까지 봤다

PC 가 꺼지면 학교에서는 못 켠다 — 정전 복구(1절 `AC Power Loss`)는 전기가 돌아왔을 때 켜 주기만 하고 서버는 다시 켜야 한다. 자동으로 올리고 싶어지면 그때 윈도우 자동 로그인·시작 프로그램을 더한다(지금은 만들지 않았다).

### 7.4 (선택) 비상용 SSH

검수 UI 가 안 뜰 때 집 PC 에 들어가려는 용도다. 관리자 PowerShell, 집 PC 에서 한 번만:

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
```

```powershell
Start-Service sshd; Set-Service -Name sshd -StartupType Automatic
```

노트북에서 `ssh <윈도우사용자>@<TS-IP>`. 들어가면 `powershell` 로 셸을 바꾼다. Tailscale 안에서만 쓰고 공유기에서 22 번을 열지 않는다.

## 8. 백업

집 PC 가 본진이 되면 **그곳에만 있는 데이터**가 생긴다 — 새로 모은 사진·검수한 라벨(`labeling.db`)이다.

- 검수 서버가 하루 한 번 `data/visible/backups/` 에 DB 사본 14개를 남기지만 **같은 디스크**라 SSD 가 죽으면 같이 간다.
- **주 1회 정도 다른 디스크(노트북 또는 외장)로 뜬다** — 명령은 `docs/backup.md` 의 robocopy(`$src` = `C:\local_auto_labeling`)와 sqlite `backup()` 으로 DB 를 뜨는 것. 새 날짜 폴더를 만들 때 이전 것을 지울지는 그때 정한다.
- 학습 결과 중 지킬 가중치는 `models\best_<이름>.pt` 로 Windows 쪽에 복사해 둔다(WSL 의 `ext4.vhdx` 는 한 파일이라 같이 백업하기 어렵다).

## 9. 문제가 생기면

| 증상 | 원인·조치 |
|---|---|
| Windows `torch.cuda.is_available()` 가 False, 버전에 `+cu130` 없음 | CPU 빌드가 깔렸다 — 3절의 PyTorch 설치 명령을 다시 |
| `No matching distribution found for torch==…+cu130` | Python 이 3.14.3 이 아니거나(3.14.1 불가) `--index-url` 을 빼먹었다 |
| WSL 에서 `torch.cuda.is_available()` 가 False | Windows 드라이버를 최신으로 → `wsl --update` → `wsl --shutdown` 후 다시. **WSL 안에 NVIDIA 드라이버를 깔지 않는다** |
| `scan` 이 '파일 없음이 절반 넘는다' 며 멈춤 / 이미지가 전부 안 열림 | 4절 ④(경로 치환)를 안 했다 — scan·prune·정리 금지, 스니펫부터. 이미 지워졌으면 노트북 원본에서 DB 를 다시 뜬다 |
| 검수 UI 의 DB 가 비었다(0장) | 데이터를 옮기기 전에 `labeler.py` 를 돌려 빈 DB 가 만들어졌다 — 4절 ① 의 DB 를 덮어쓴다(빈 DB 는 먼저 따로 치운다) |
| WSL 학습이 `Killed`·메모리 부족 | `.wslconfig` 의 `memory` 를 올리거나(Windows 몫이 8GB 아래로 내려가지 않게), `config.yaml` 의 `wsl.cache` 를 `disk` 로. `wsl --shutdown` 뒤 다시 |
| epoch 시간이 갑자기 몇 배로 늘었다 | VRAM 이 넘쳐 공유 메모리로 갔다(함정 9). GPU 쓰는 앱을 닫고 `nvidia-smi` 로 사용량을 본다 — 16GB 에 거의 차 있으면 batch 를 줄여 이어한다(`labeler.py resume --batch 8` 또는 WSL 이어하기) |
| 노트북 브라우저에서 400 `Invalid host header` | bind 한 `<TS-IP>` 가 아닌 주소로 열었다 — 7.2 |
| 노트북 브라우저에서 403 `다른 출처에서 온 요청` | 검수 UI 밖의 페이지에서 보낸 요청이다. 주소창에서 직접 열린 UI 로만 조작한다 |
| 접속 시간 초과 | 서버가 꺼졌다 / 방화벽 / Tailscale 끊김(`tailscale status`) |
| `wsl --install` 이 가상화 오류 | BIOS 의 가상화(VT-x)를 켠다(1절) |

## 10. 다 됐는지 확인

- [ ] BIOS 에 32GB·3200MHz(XMP 프로파일이 있으면 켬), 램 2개면 듀얼 채널
- [ ] `nvidia-smi` 에 16GB·CUDA 13.x
- [ ] Windows `.venv` 의 torch 가 `2.14.0+cu130 True`
- [ ] WSL 의 torch 가 `2.11.0+cu130 True … (12, 0)`
- [ ] `labeler.py check` 가 `이상 없음.`, `status` 장수가 노트북과 같다
- [ ] 노트북 브라우저에서 `http://<TS-IP>:7860` 이 열린다
- [ ] 승인 이미지가 쌓인 뒤: `WSL 학습` 이 끝까지 돌고 epoch 시간이 정상(5.7 의 probe 도 한 번)
- [ ] 8절 백업을 한 번 떴다

## 11. Runpod 는

이 PC 가 못 하는 큰 학습(x960 한 번에 하루 가까이 — 6000 Ada 대비 3–4배 느릴 것으로 본다, 스펙 기반 추정)만 맡긴다. 직접 하는 길은 `docs/runpod.md`(MCP 는 11-02 에 뺐다 — 콘솔에서 pod 를 만들고 `ssh`·`scp` 로 올린다). **pod 를 만들기 전 GPU·예상 시간·비용을 따져 보고, 멈춘 pod 도 볼륨 요금이 나간다는 점을 기억한다.**
