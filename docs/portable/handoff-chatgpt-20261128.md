# Local Auto Labeling — Claude 작업 인수인계 문서

> 목적: 이 문서는 ChatGPT와 사용자가 논의한 프로젝트 현황, 목표, 기존 코드 리뷰 요약, 배포 방향 및 아키텍처 제안을 현재 작업 중인 Claude 모델에 전달하기 위한 문서다.
>
> **중요:** 아래의 코드 현황과 줄 수, 동작에 관한 내용은 이전 대화에서 수행한 저장소 정적 리뷰를 요약한 것이다. 저장소는 계속 변경될 수 있으므로, 실제 수정에 착수하기 전에 반드시 현재 `main` 브랜치와 작업 트리를 다시 확인하라. 이 문서의 아키텍처와 코드 예시는 제안이며, 이미 구현된 것으로 간주하면 안 된다.

---

## 1. 프로젝트 개요

- 프로젝트명: **Local-Auto-Labeling**
- GitHub: <https://github.com/Ch2Su/Local-Auto-Labeling>
- 기존 목적: 개발자 본인이 사용하는 로컬 이미지 자동 라벨링 및 YOLO 학습 도구
- 확정된 장기 목적: **다른 사용자도 다운로드해 사용할 수 있는 크로스 플랫폼 포터블 데스크톱 프로그램으로 발전**
- 사용자와 합의한 배포 형태: **압축 파일을 풀고 바로 실행하는 포터블 배포**
- 사용자와 합의한 OS 정책: **Windows, macOS, Linux를 초기부터 모두 지원**
- 기본 제품 성격: 로컬 우선(local-first), 단일 사용자 중심, 이미지 및 학습 데이터가 기본적으로 사용자 PC 밖으로 전송되지 않는 애플리케이션

### 제품 방향의 핵심

1. 기존 Python / FastAPI / SQLite / YOLO 중심의 핵심 구현을 최대한 재사용한다.
2. Electron 또는 Tauri로 프런트엔드와 백엔드를 전면 재작성하는 일은 당장 하지 않는다.
3. 사용자가 Python, CUDA, 개발 도구를 직접 설치하지 않아도 실행할 수 있는 플랫폼별 포터블 패키지를 제공한다.
4. 프로그램 코드와 사용자 데이터, 워크스페이스, 모델을 분리한다.
5. CPU 환경에서도 수동 라벨링과 데이터 관리 기능은 동작해야 한다.
6. OS별 차이는 플랫폼 추상화 계층에 모으고, 핵심 비즈니스 로직이 OS를 직접 판별하지 않게 한다.
7. 기존 데이터 무결성, 검수 이력, 안전한 학습 결과 승격, 데이터 누수 방지 기능을 퇴보시키지 않는다.

---

## 2. 이전 정적 코드 리뷰에서 파악한 프로젝트 특성

아래 내용은 이전 세션에서 살펴본 코드와 README를 요약한 것이다. 모든 항목은 현재 저장소에서 재확인해야 한다.

### 2.1 기존 기능 및 구성

프로젝트는 대략 다음 흐름을 갖는다.

1. 이미지 및 수동 라벨 데이터 등록
2. YOLO 모델 학습
3. 자동 추론을 통한 라벨 생성
4. 웹 UI에서 라벨 검수 및 수정
5. 승인된 라벨을 학습 데이터로 편입
6. 반복 학습

주요 구성요소로 다음 파일/모듈이 확인되었다.

- `config.py`: 설정 로드 및 경로 처리
- `projects.py`: 프로젝트별 설정 및 앱 인스턴스 구성
- `store.py`: SQLite 기반 이미지 상태, 라벨, 이력, 데이터 관리
- `predictor.py`: YOLO 추론
- `trainer.py`: 데이터셋 생성 및 학습 실행
- `split.py`: train/val/test 데이터 분할
- `server.py`: FastAPI 서버 및 여러 API/서비스 연결
- `cli.py`: CLI 진입점 및 리뷰 서버 실행
- `wsl_train.py` 및 `wsl/`: Windows WSL 학습 통합
- SAM 관련 보조 라벨링 기능

### 2.2 기존 설계에서 유지할 강점

- 이미지의 `unlabeled`, `pending`, `reviewed`, `approved`, `skipped`, `trashed` 등 상태를 구분한다.
- 자동 추론 도중 사용자가 수정한 라벨이 추론 결과에 의해 덮어써지지 않도록 DB 기록 전에 상태/수정 시각을 재확인하는 보호 로직이 있다.
- 데이터 분할 시 개별 프레임이 아니라 그룹 단위로 묶어 동일 영상/유사 프레임의 train/validation/test 간 누수를 줄이려 한다.
- 데이터셋 분할 결과를 기록하고 재사용하는 설계가 있다.
- 학습 중단 또는 실패 결과가 기존 운영 가중치를 무조건 덮어쓰지 않도록 결과 승격을 보호한다.
- SQLite 백업, 휴지통, 변경 이력 등 데이터 복구를 고려한 기능이 있다.
- 여러 프로젝트를 동시에 다루는 구조에서 프로젝트별 Store/작업 상태 등을 분리하려는 설계가 있다.
- YOLO 등 무거운 라이브러리를 필요 시점에 import하는 지연 로딩 방식이 일부 적용되어 있다.
- README 기준 이전 세션에서 회귀 테스트 약 698개를 확인했다고 보고되었다. 현재 개수와 실행 결과는 재확인할 것.

**지침:** 위 기능은 배포 구조를 개편할 때 가능한 한 그대로 보존하고, 리팩터링 전후에 회귀 테스트를 수행한다.

### 2.3 이전 리뷰에서 발견한 개선 대상

#### A. 경로가 소스 저장소 루트에 결합됨

이전 리뷰에서 `config.py`가 전역 `ROOT`를 기준으로 상대 경로를 해석하고, 일부 DB/데이터 경로도 저장소 루트와 결합된 것으로 파악되었다.

문제:
- PyInstaller 등으로 패키징하면 소스 파일 위치, 번들 리소스 위치, 실행 파일 위치가 서로 다르다.
- 설치 위치가 읽기 전용일 수 있다.
- 포터블 폴더를 이동하거나 App 파일만 업데이트할 때 사용자 데이터가 손상될 수 있다.
- 이미지 경로가 절대경로로 저장되어 워크스페이스 이동이 불편하다.

#### B. WSL 의존성이 서버 import에 연결됨

이전 리뷰에서는 `server.py`가 WSL 학습 모듈을 최상위 import하는 부분을 확인했다. WSL 구현은 Windows 전용 명령과 경로에 의존한다.

문제:
- macOS/Linux에서 WSL 모듈 import 자체가 플랫폼 호환성 문제를 일으킬 수 있다.
- WSL은 제품 핵심 기능이 아니라 Windows 선택 확장 기능이어야 한다.

#### C. 하드웨어 기본값이 NVIDIA 중심

이전 리뷰에서 설정의 device 기본값이 `0`인 것으로 확인되었다. 이는 NVIDIA GPU를 기본 가정하는 방식이다.

문제:
- CPU 전용 PC에서 기본 동작이 실패할 수 있다.
- macOS Apple Silicon의 MPS를 표현할 공통 장치 추상화가 필요하다.
- CUDA/MPS/CPU별 지원 여부와 fallback 정책이 분산될 수 있다.

#### D. 대형 모듈의 책임 집중

이전 리뷰 당시 `server.py` 약 2,067줄, `store.py` 약 2,021줄로 파악되었다. 최신 줄 수는 재확인해야 한다.

문제:
- API 라우팅, 검증, 작업 관리, UI 연결, 파일 처리 등이 한곳에 집중될 가능성이 있다.
- DB 관련 관심사도 한 모듈에 많이 모여 있다.

방향:
- 전면 재작성 대신 새 계층을 추가하고 기존 코드를 조금씩 이전한다.
- 기능별 테스트를 만든 뒤 라우트 및 저장소 책임을 점진적으로 분리한다.

#### E. 데이터셋 생성의 안전성

이전 리뷰에서 데이터셋 디렉터리를 지우고 새로 만드는 동작이 검증보다 먼저 수행될 가능성을 지적했다.

권장:
1. 데이터 및 클래스 유효성 검사
2. 임시 디렉터리에 데이터셋 생성
3. 생성물 검증
4. 기존 데이터셋과 안전하게 교체

기존 파일을 삭제하기 전 실패 복구 경로가 있어야 한다.

#### F. 보안 및 네트워크 접근

이전 리뷰에서는 CSRF/DNS rebinding 관련 요청 방어가 있으나, wildcard bind에서 Host 제한이 완화되는 분기 및 Origin 없는 요청을 허용하는 동작을 확인했다고 보고되었다.

권장:
- 기본 서버 bind는 `127.0.0.1` 유지
- LAN/외부 접근은 명시적으로 opt-in
- 외부 접근 시 별도 인증/권한 계층 필요
- 로컬 브라우저에서 접근하는 API에도 세션 토큰 또는 동등한 방어 추가 검토
- 배포 버전의 API 문서/디버그 엔드포인트 정책 확정
- 민감한 파일 경로와 토큰이 로그에 노출되지 않도록 처리

#### G. 개발자 환경 설정과 제품 설정 혼재

이전 리뷰에서는 개발용 WSL/Runpod 경로, 개발 자동화 설정, 개인 환경 관련 문서가 저장소에 포함되어 있다고 파악했다.

권장:
- 개인 경로 및 기기별 환경값을 기본 배포 설정에서 제거
- 개발 도구 설정은 유지하더라도 제품 기본 설정/사용자 문서에서 분리
- 실제 비밀값, 토큰, 개인정보가 없는지 별도 스캔

---

## 3. 합의된 배포 모델

### 3.1 플랫폼별 ZIP

하나의 ZIP으로 세 OS를 모두 실행하는 방식은 사용하지 않는다. 각 운영체제/CPU 아키텍처별 네이티브 바이너리와 런타임이 다르기 때문이다.

초기 목표 패키지:

| 아티팩트 이름(가칭) | 환경 | 가속 |
|---|---|---|
| `LocalAutoLabeling-win-x64-cpu.zip` | Windows x64 | CPU |
| `LocalAutoLabeling-win-x64-cuda.zip` | Windows x64 + NVIDIA | CUDA |
| `LocalAutoLabeling-macos-arm64.zip` | Apple Silicon | MPS 및 CPU |
| `LocalAutoLabeling-linux-x64-cpu.zip` | Linux x64 | CPU |
| `LocalAutoLabeling-linux-x64-cuda.zip` | Linux x64 + NVIDIA | CUDA |

추가 플랫폼은 후속 검증 대상으로 둔다.

- macOS Intel x86_64: 후속 검증
- Linux ARM64: 후속 검증
- Linux AMD GPU/ROCm: 후속 검토
- Windows AMD GPU: 후속 검토
- 기타 Intel GPU: 후속 검토

OS별 동작과 GPU 가속은 별도 축이다. 예를 들어 macOS CPU 모드에서도 이미지 수동 검수, 데이터 관리 및 내보내기는 정상 작동해야 한다.

### 3.2 포터블 폴더 구조

권장 배포 폴더:

```text
LocalAutoLabeling/
├── App/
│   ├── LocalAutoLabeling.exe / .app / native launcher
│   ├── _internal/ 또는 번들 런타임
│   ├── resources/
│   ├── licenses/
│   └── version.json
├── UserData/
│   ├── config/
│   ├── workspaces/
│   ├── models/
│   ├── logs/
│   └── cache/
├── portable.flag
└── README_FIRST_RUN.txt
```

원칙:
- `App/`에는 실행 코드, Python 런타임, 라이브러리 및 읽기 전용 리소스만 둔다.
- `UserData/`에는 설정, DB, 이미지/라벨, 모델 캐시, 학습 결과, 로그 등 사용자 데이터를 둔다.
- App 교체/업데이트가 UserData를 건드리지 않도록 한다.
- 폴더 이동 후에도 Managed Workspace가 작동해야 한다.
- `portable.flag`가 있으면 사용자 데이터는 기본적으로 배포 폴더 내부 `UserData/`를 사용한다.
- 해당 위치가 읽기 전용/쓰기 불가인 경우 오류를 명확히 안내하고, 사용자가 다른 데이터 위치를 선택할 수 있게 한다.
- 사용자가 포터블 폴더 밖에 워크스페이스를 만들 수 있도록 한다.

주의: 운영체제별 실행 파일 형태는 달라도 사용자가 경험하는 실행 흐름은 동일하게 만든다.

---

## 4. 목표 아키텍처

### 4.1 계층 구조

```text
Portable Launcher
    |
    +-- Runtime Manager
    |     +-- Start/Stop backend
    |     +-- Process lifecycle
    |     +-- Logs and diagnostics
    |     +-- Single-instance handling
    |
    +-- Platform Manager
    |     +-- OS/CPU architecture detection
    |     +-- GPU/device detection
    |     +-- Feature capability checks
    |
    +-- Workspace Manager
    |     +-- Workspace registry
    |     +-- Workspace create/open/move
    |     +-- Managed/Linked storage
    |     +-- DB migration/backup
    |
    +-- Model Manager
    |     +-- Model registry
    |     +-- Download/resume/checksum
    |     +-- Source/license metadata
    |
    +-- Application Core
          +-- FastAPI
          +-- Existing browser review UI
          +-- Store / SQLite
          +-- Job Manager
          +-- Dataset Engine
          +-- YOLO Predictor/Trainer
          +-- SAM segmentation
```

핵심 원칙:
- OS별 코드는 `platform/` 및 명시적 `integrations/` 아래에 모은다.
- Core는 `sys.platform`이나 WSL 명령어를 직접 사용하지 않고 인터페이스에 의존한다.
- 서버는 사용자의 프로젝트/워크스페이스를 명확히 전달받아 처리한다.
- 기존 프로젝트별 Store/작업 컨텍스트 분리 구조는 유지한다.
- 모델 로딩은 가능하면 지연 로딩한다.

### 4.2 제안 디렉터리 구조

아래는 최종 목표 구조의 예시다. 한 번에 이 구조로 파일을 이동하지 말고, 기능별 테스트를 추가하며 점진적으로 옮긴다.

```text
Local-Auto-Labeling/
├── src/
│   └── autolabel/
│       ├── __main__.py
│       ├── launcher.py
│       ├── platform/
│       │   ├── __init__.py
│       │   ├── paths.py
│       │   ├── system.py
│       │   ├── devices.py
│       │   ├── capabilities.py
│       │   └── diagnostics.py
│       ├── config/
│       │   ├── loader.py
│       │   ├── schema.py
│       │   └── defaults.yaml
│       ├── workspace/
│       │   ├── manager.py
│       │   ├── registry.py
│       │   ├── manifest.py
│       │   ├── storage.py
│       │   └── migration.py
│       ├── core/
│       │   ├── predictor.py
│       │   ├── trainer.py
│       │   ├── store.py
│       │   ├── split.py
│       │   ├── ingest.py
│       │   └── dataset.py
│       ├── ml/
│       │   ├── device_manager.py
│       │   ├── model_registry.py
│       │   ├── model_loader.py
│       │   ├── backends/
│       │   └── services/
│       ├── api/
│       │   ├── app.py
│       │   ├── middleware.py
│       │   ├── routes/
│       │   └── schemas/
│       ├── runtime/
│       │   ├── server.py
│       │   ├── process.py
│       │   ├── logging.py
│       │   └── lifecycle.py
│       ├── integrations/
│       │   ├── wsl.py
│       │   └── remote_training.py
│       └── web/
│           └── static/
├── packaging/
│   ├── pyinstaller/
│   │   ├── windows.spec
│   │   ├── macos.spec
│   │   └── linux.spec
│   └── scripts/
│       ├── build_windows.ps1
│       ├── build_macos.sh
│       ├── build_linux.sh
│       ├── package.py
│       └── verify_release.py
├── requirements/
│   ├── base.in 또는 base.txt
│   ├── win-cpu.lock
│   ├── win-cuda.lock
│   ├── macos-arm64.lock
│   ├── linux-cpu.lock
│   ├── linux-cuda.lock
│   └── dev.lock
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── platform/
│   ├── migration/
│   ├── ui/
│   └── packaging/
├── docs/
├── pyproject.toml
└── README.md
```

실제 폴더 구조는 기존 `src/autolabel/` 및 테스트 구조를 먼저 확인한 뒤 최소 변경으로 결정한다. 새 디렉터리 구성을 위해 불필요한 이동을 만들지 않는다.

---

## 5. 경로 및 데이터 저장 설계

### 5.1 서로 다른 경로 개념을 분리

반드시 별개로 취급해야 하는 경로:

1. **install_dir**: 실행 파일과 App 폴더의 위치
2. **resource_dir**: PyInstaller 번들 내부의 읽기 전용 리소스 위치
3. **user_data_dir**: 앱 설정, 로그, 캐시, 모델 캐시 등 사용자 전역 데이터 위치
4. **workspace_dir**: 개별 프로젝트의 DB, 이미지, 라벨, 학습 결과 위치
5. **external_storage_root**: Linked Workspace가 참조하는 외부 이미지 디렉터리

PyInstaller 사용 시 `sys.executable`의 위치와 `sys._MEIPASS`의 위치가 같은 개념이라고 가정하지 않는다. 개발 모드와 frozen 모드 양쪽에서 리소스 경로를 검증한다.

### 5.2 AppPaths 인터페이스 제안

```python
@dataclass(frozen=True)
class AppPaths:
    install_dir: Path
    resource_dir: Path
    user_data_dir: Path
    portable: bool
```

`AppPaths.discover()` 같은 단일 진입점에서 경로를 결정하되, 다음을 지원한다.

- 개발 모드
- 패키징 모드
- 포터블 모드
- 사용자 지정 UserData 위치
- 쓰기 권한 오류 처리
- 기존 데이터 경로 호환

OS별 기본 데이터 위치(비포터블/폴백 모드):

| OS | 기본 위치 예시 |
|---|---|
| Windows | `%LOCALAPPDATA%/Ch2Su/LocalAutoLabeling` |
| macOS | `~/Library/Application Support/LocalAutoLabeling` |
| Linux | `$XDG_DATA_HOME/local-auto-labeling`, 없으면 `~/.local/share/local-auto-labeling` |

### 5.3 Workspace 종류

**Managed Workspace** (기본 권장): 이미지, DB, 라벨, 모델 산출물 등을 워크스페이스 아래 관리한다. 폴더 전체 복사/이동을 통해 프로젝트를 옮기기 쉽다.

**Linked Workspace** (고급): 외부 이미지 폴더는 원위치에 두고, DB/라벨/학습 결과만 워크스페이스에서 관리한다. 외부 경로가 바뀌면 재연결 절차가 필요하다.

### 5.4 이미지 경로 스키마 방향

현재 DB의 `images.path` 절대경로 저장 방식이 실제로 어떻게 사용되는지 확인한 뒤, 다음 개념으로 점진적으로 이전한다.

```sql
CREATE TABLE storage_roots (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    root_path TEXT NOT NULL,
    display_name TEXT
);

-- images 테이블에 다음 필드를 단계적으로 추가하는 설계 예시
ALTER TABLE images ADD COLUMN storage_root_id INTEGER;
ALTER TABLE images ADD COLUMN relative_path TEXT;
```

실제 스키마는 기존 테이블, 인덱스, 외래키, 트리거, 마이그레이션 규칙을 조사한 뒤 설계한다. 위 SQL을 그대로 실행하지 말 것.

마이그레이션 원칙:
1. SQLite 온라인 백업 API 또는 안전한 백업 절차로 원본 DB 스냅샷 생성
2. 이미지 원본 파일 존재 여부 확인
3. 기존 절대경로를 적절한 storage root + relative path로 매핑
4. 레코드 수, 라벨, 이력, 관계 검증
5. 이전/새 위치에서 이미지 로드 검증
6. 실패 시 원본 DB와 파일을 그대로 유지하고 롤백
7. 성공 후에도 이전 버전 백업을 사용자가 복구할 수 있게 보존

DB 스키마에 `schema_version` 또는 이에 준하는 명시적인 버전 관리 체계를 둔다.

---

## 6. 플랫폼 및 장치 관리

### 6.1 1차 지원 기준

| OS/하드웨어 | 초기 정책 |
|---|---|
| Windows x64 + NVIDIA | CUDA 추론/학습 지원 |
| Windows x64 + GPU 없음 | CPU 기능 지원 |
| macOS ARM64 + Apple Silicon | MPS 및 CPU 지원 |
| Linux x64 + NVIDIA | CUDA 추론/학습 지원 |
| Linux x64 + GPU 없음 | CPU 기능 지원 |
| macOS Intel | 추가 검증 후 지원 여부 확정 |
| Linux ARM64 | 후속 검토 |
| AMD/ROCm | 후속 검토 |

세 운영체제 모두에서 최소한 다음 기능은 동작해야 한다.
- 앱 실행/종료
- 이미지 가져오기 및 표시
- 수동 라벨 생성/수정/저장
- SQLite 데이터 관리 및 백업
- 데이터셋 내보내기

GPU 학습/추론은 지원 백엔드와 하드웨어가 있을 때 활성화한다.

### 6.2 Device 설정

기존의 NVIDIA 장치 번호 기본값 `0`을 그대로 유지하지 말고, 설정 스키마를 확인한 뒤 기본값을 `auto`로 전환하는 방안을 검토한다.

권장 사용자 표현:
- `auto`
- `cpu`
- `cuda:0`, `cuda:1` 등
- `mps`

내부 변환은 실제 사용 라이브러리(Ultralytics/PyTorch)가 요구하는 device 형식과 일치시킨다.

자동 선택 기본 우선순위 예시:
1. 사용 가능한 CUDA
2. 사용 가능한 MPS
3. CPU

다만 아래 원칙을 따른다.
- 사용자가 수동 선택한 장치는 임의로 바꾸지 않는다.
- 학습 중 CUDA OOM이 발생했다고 자동으로 CPU 학습으로 몰래 전환하지 않는다.
- 장치 오류 시 원인과 사용 가능한 대안을 보여주고 사용자가 재시도 여부를 선택한다.
- 특정 모델/연산이 MPS에서 지원되지 않을 수 있으므로 기능별 capability를 판단한다.
- CPU는 수동 검수와 데이터 관리의 fallback이며, CPU 학습은 느릴 수 있음을 명시한다.

### 6.3 플랫폼 모듈 API 예시

```python
class DeviceManager:
    def detect(self): ...
    def select(self, preference="auto"): ...
    def health_check(self, device): ...
    def available_devices(self): ...

class PlatformCapabilities:
    def supports(self, capability: str) -> bool: ...
```

핵심 코드에서 `torch.cuda.is_available()` 등을 제각각 호출하지 않고, 공통 서비스에서 장치를 결정한다.

---

## 7. WSL 및 원격 학습 통합

기본 학습은 각 OS의 Native PyTorch를 사용한다.

권장 구조:

```text
Training Service
    |
    +-- Native Training Provider
    |     +-- Windows CUDA/CPU
    |     +-- macOS MPS/CPU
    |     +-- Linux CUDA/CPU
    |
    +-- WSL Provider (Windows 선택 기능)
    |
    +-- Remote Provider (향후 Runpod 등)
```

- WSL 관련 모듈은 Windows에서 명시적으로 기능이 선택될 때만 import한다.
- `server.py`의 최상위 import로 WSL 구현이 모든 OS에 로드되지 않도록 한다.
- WSL은 핵심 서비스의 필수 의존성이 아니다.
- Runpod 등 원격 학습도 향후 Provider 형태로 분리한다.
- 기본 사용자 경험은 WSL이나 원격 계정 없이 동작해야 한다.

---

## 8. 런처 및 서버 수명주기

### 8.1 사용자 실행 흐름

1. 사용자가 OS별 launcher를 실행한다.
2. 런처가 OS/아키텍처/장치/쓰기 권한을 검사한다.
3. AppPaths를 계산하고 UserData를 준비한다.
4. 첫 실행이면 기본 워크스페이스를 생성하거나 위치 선택을 안내한다.
5. 로컬 FastAPI 서버를 loopback 주소에 시작한다.
6. OS에서 사용 가능한 임시 포트를 할당받는다. 고정 포트 충돌에 의존하지 않는다.
7. health endpoint로 서버 준비 상태를 확인한다.
8. 기본 브라우저를 자동 실행한다.
9. 종료 요청 시 실행 중인 작업에 graceful shutdown을 요청한다.
10. 작업 정리/체크포인트/DB close 후 서버를 종료한다.

### 8.2 프로세스 구조

우선은 런처가 백엔드 프로세스를 감독하는 구조를 권장한다. 기존 JobManager와 학습/추론 내부 구조를 첫 단계부터 모두 별도 프로세스로 바꾸지 않는다.

초기 필수 기능:
- single-instance 또는 중복 실행 감지
- 백엔드 프로세스 시작/종료 관리
- 서버 readiness 확인
- 종료 요청 및 timeout 처리
- 비정상 종료 로그
- 재실행 시 복구 가능한 작업 상태 확인

### 8.3 서버 보안

- 기본 bind: `127.0.0.1`
- 동적 포트 할당
- LAN/외부 접속은 기본 비활성화
- 외부 접속 기능은 향후 명시적 설정 및 인증이 있을 때만 제공
- 로컬 브라우저 API에도 세션 토큰/동등한 방어 도입 검토
- 기존 CSRF, Host, DNS rebinding 방어를 재검토하고 유지/보강
- wildcard bind 시 보안 제약을 명확하게 적용
- API docs/debug 엔드포인트의 배포 모드 정책 수립
- 로그에 토큰, 사용자 이미지 전체 경로 등 민감한 정보가 과도하게 기록되지 않도록 한다.

---

## 9. 모델 관리

모든 모델을 기본 ZIP에 포함하지 않는다. 모델 가중치는 애플리케이션 런타임과 분리한다.

권장 분류:
- 사전학습 모델: UserData 공용 모델 캐시
- SAM 모델: UserData 공용 모델 캐시
- 사용자가 학습한 `best.pt` 등: 워크스페이스 내부

모델 관리 기능:
- 모델 이름/버전/작업 유형 표시
- 모델 출처 및 라이선스 표시
- 다운로드 진행률
- 다운로드 취소 및 재개
- SHA-256 무결성 확인
- 디스크 공간 확인
- 기존 가중치 보호
- 삭제 전 사용 중인 워크스페이스 참조 검사

모델 매니페스트 예시(실제 URL/해시/라이선스가 확정된 것이 아님):

```yaml
models:
  - id: yolo26n
    type: detection
    version: "1"
    filename: yolo26n.pt
    sha256: "REPLACE_WITH_VERIFIED_HASH"
    source: "VERIFIED_SOURCE_URL"
    license: "VERIFY_MODEL_LICENSE"
```

모델 버전/출처/라이선스는 실제 공급자의 최신 조건을 확인한 후 명시한다.

---

## 10. 의존성 및 패키징

### 10.1 Python 및 ML 의존성

이전 개발 설정에서 Python 3.14.x와 특정 CUDA용 PyTorch 버전이 사용된 것으로 보고되었다. 이것을 3개 OS 전체의 배포 기준으로 그대로 채택하지 않는다.

먼저 Python 버전 후보를 선정하고, 각 대상 플랫폼에서 아래 패키지가 실제로 호환되는지 검증한다.
- PyTorch / torchvision
- Ultralytics
- OpenCV
- Pillow
- FastAPI / Uvicorn
- SAM 관련 라이브러리
- 기타 네이티브 확장 의존성

정확한 Python/PyTorch/CUDA 조합은 현재 공식 호환표와 실제 CI 결과를 바탕으로 확정한다.

권장 lock 분리 예시:

```text
requirements/
├── base.in
├── win-cpu.lock
├── win-cuda.lock
├── macos-arm64.lock
├── linux-cpu.lock
├── linux-cuda.lock
└── dev.lock
```

간접 의존성까지 재현 가능하도록 lock 파일/해시를 관리한다.

### 10.2 PyInstaller

초기 검토 도구: **PyInstaller `onedir` 모드**.

이유:
- 대형 PyTorch 및 네이티브 라이브러리 디버깅이 상대적으로 쉬움
- 실행 때마다 거대한 onefile 임시 추출을 할 필요가 없음
- 사용자 데이터 및 모델 파일을 분리하기 쉬움

주의:
- PyInstaller는 운영체제별 네이티브 빌드가 필요하다. Windows 빌드는 Windows, macOS 빌드는 macOS, Linux 빌드는 Linux 환경에서 수행한다.
- 동적 import, PyTorch 네이티브 라이브러리, Ultralytics 리소스, 정적 웹 파일 포함 여부를 검증한다.
- macOS는 ARM64 빌드/서명/공증을 별도 검증한다.
- Linux는 배포 대상의 glibc/시스템 라이브러리 호환 범위를 명시한다.
- PyInstaller가 문제를 해결하지 못하면 그때 런타임 번들링 방식 또는 다른 패키징 도구를 비교한다. 초기에 여러 빌드 시스템을 혼합하지 않는다.

### 10.3 모델과 패키지 크기

PyTorch/CUDA가 포함된 ZIP은 매우 커질 수 있다.
- CPU와 CUDA 패키지를 분리한다.
- SAM 가중치와 사전학습 모델은 필요 시 별도 다운로드한다.
- 첫 실행에서 필요한 컴포넌트를 선택할 수 있게 한다.
- GPU 런타임이 사용자 시스템과 실제 호환되는지 검사한다.
- NVIDIA 드라이버는 사용자 시스템 요소이므로, 번들 CUDA runtime과 드라이버 호환 조건을 명확히 설명한다.

---

## 11. UI 및 사용자 경험

초기에는 기존 브라우저 기반 검수 UI를 재사용한다.

추가할 UX:

1. 첫 실행 환경 진단
2. GPU/CPU 장치 및 기능 상태 표시
3. 워크스페이스 생성/열기/이동/재연결
4. 모델 설치/삭제/상태 화면
5. 다운로드/추론/학습 진행 상태
6. 오류 메시지와 진단 로그 내보내기
7. 프로그램 버전 및 DB schema 버전 표시
8. 실행 중인 작업을 고려한 종료 UX

사용자가 개발자 설정을 직접 만지지 않도록 기본/고급 설정을 구분한다.

첫 실행에서 보여줄 정보는 아래 정도로 제한한다.
- 현재 OS와 CPU 아키텍처
- 감지된 장치
- 선택된 실행 장치
- UserData 저장 위치 및 사용 가능 공간
- 필요한 모델 설치 상태

---

## 12. 테스트 및 릴리스 기준

기존 회귀 테스트는 유지한다. GPU/실제 모델 테스트를 모두 빠른 CI에 넣으려 하지 말고 계층화한다.

### 12.1 테스트 계층

| 계층 | 검증 내용 | 실행 환경 |
|---|---|---|
| Unit | DB/상태/경로/설정/분할 | 일반 CI |
| Migration | 기존 DB → 새 schema, rollback | 임시 DB 및 복사본 |
| Integration | 이미지 등록 → 추론 결과/검수 → 승인 → dataset 생성 | CPU 테스트 모델 또는 fixture |
| UI | 브라우저에서 이미지 표시/라벨 수정/저장 | Playwright 등 |
| Platform | OS 경로/권한/프로세스 동작 | 각 OS runner |
| GPU smoke | CUDA/MPS 초기화와 소규모 추론 | 실제 GPU runner 또는 검증 PC |
| Packaging | ZIP 실행, resource loading, 종료 | clean VM/실기기 |
| Relocation | UserData/Workspace 폴더 이동 후 재오픈 | OS별 clean test |

### 12.2 최소 공개 베타 합격 기준

- 세 OS에서 Python 사전 설치 없이 앱 실행
- CPU 모드에서 수동 라벨링 및 데이터 관리 기능 정상 동작
- 플랫폼별로 지원하는 GPU에서 추론 Smoke Test 통과
- 지원한다고 선언한 GPU 백엔드에서 학습 Smoke Test 통과
- 워크스페이스를 다른 폴더/드라이브로 이동해도 복구 가능
- App 폴더만 교체해도 UserData 유지
- DB 마이그레이션 실패 시 원본 복구 가능
- 중복 실행 및 서버 종료가 안전함
- 외부 네트워크에 기본 노출되지 않음
- 다운로드한 ZIP의 SHA-256 및 릴리스 버전 확인 가능
- 사용 중인 라이선스/모델 출처 고지 제공

### 12.3 릴리스 아티팩트

각 ZIP과 함께 다음을 게시한다.
- 플랫폼/아키텍처 명시 파일명
- 버전 및 빌드 정보
- SHA-256 체크섬
- 라이선스 고지와 third-party notices
- 지원 OS/하드웨어 목록
- 알려진 제한 사항
- 데이터 백업 및 업그레이드 안내

macOS 외부 배포는 Developer ID 서명과 공증을 검토한다. 서명/공증 없는 ZIP은 Gatekeeper 경고 등 사용성 문제가 발생할 수 있다.

---

## 13. 우선순위별 구현 로드맵

### Milestone 0.1 — Portable Foundation (첫 작업)

목표: 소스 코드가 저장소 위치, 개발자 PC, 특정 GPU를 가정하지 않도록 만든다.

작업:
- [ ] 현재 main/working tree 및 실제 파일 구조 확인
- [ ] `ROOT`, `resolve()`, `__file__`, 현재 디렉터리 의존 위치 전수 조사
- [ ] `AppPaths` 구현
- [ ] 개발 모드/패키징 모드/포터블 모드 경로 테스트
- [ ] Config 상대경로 기준 명확화
- [ ] UserData와 Workspace 경계 정리
- [ ] WSL 최상위 import 제거 및 선택적 로딩
- [ ] GPU 기본값/장치 감지 인터페이스 도입
- [ ] 기존 DB 경로 마이그레이션 설계와 백업/롤백 테스트

완료 조건:
- 기존 Windows 개발 모드의 기능 회귀 없음
- 프로젝트 폴더가 이동되어도 workspace 설정으로 재연결 가능
- 쓰기 권한이 없는 설치 경로에서도 사용자 데이터 위치를 별도로 사용 가능
- OS 의존 코드가 플랫폼 계층/통합 모듈에 모이기 시작함

### Milestone 0.2 — Cross-platform Python Runtime

- [ ] Windows 네이티브 개발 실행
- [ ] macOS ARM64 개발 실행
- [ ] Linux x64 개발 실행
- [ ] CPU 모드 검증
- [ ] CUDA 모드 검증(Windows/Linux)
- [ ] MPS 모드 검증(macOS ARM64)
- [ ] SAM 기능별 backend compatibility 조사
- [ ] WSL 선택 기능의 Windows 한정 동작 확인
- [ ] 경로, 파일 잠금, 프로세스 종료의 OS 차이 테스트

### Milestone 0.3 — Portable Packaging

- [ ] PyInstaller `onedir` 빌드 실험
- [ ] OS별 spec/config
- [ ] CPU/GPU 패키지별 의존성 고정
- [ ] ZIP layout 생성
- [ ] launcher 실행 및 서버 readiness
- [ ] 브라우저 자동 실행
- [ ] 로그/진단 정보
- [ ] portable UserData 보존 확인
- [ ] clean machine 설치/실행 테스트

### Milestone 0.4 — Public Beta

- [ ] GitHub Actions 플랫폼별 build/test
- [ ] 세 OS에서 실제 실행 테스트
- [ ] CUDA/MPS smoke tests
- [ ] macOS 서명/공증 파이프라인
- [ ] 릴리스 체크섬/라이선스 문서
- [ ] 사용자 가이드 및 FAQ
- [ ] 오류 신고 템플릿/진단 로그 내보내기

### Milestone 1.0 — Stable Release

- [ ] 호환성 범위 명확히 공지
- [ ] 안정적 데이터 마이그레이션 및 복구
- [ ] 수동 업데이트 가이드
- [ ] 모델 관리 및 버전 정책
- [ ] 자동 업데이트는 안정성 확보 후 검토
- [ ] 사용자 기여/버그 신고 프로세스 정리

---

## 14. Claude가 작업을 시작할 때 지켜야 할 원칙

1. **현재 저장소 상태를 먼저 확인한다.** 이 문서의 파일 경로, 코드 구조, 줄 수, 설정 기본값이 현재 HEAD와 다를 수 있다.
2. **한 번에 전체 재작성하지 않는다.** 작은 변경 단위로 진행하고 각 단계에서 기존 테스트를 실행한다.
3. **기존 사용자 데이터는 절대 위험에 노출하지 않는다.** DB migration 전에 복사본/백업을 만들고 rollback 테스트부터 작성한다.
4. **`ROOT`를 제거하기 전에 모든 참조를 조사한다.** 임시 호환 계층을 두고 점진적으로 호출부를 이전한다.
5. **기존 데이터의 절대경로를 임의로 삭제/변환하지 않는다.** 매핑 검증이 되기 전까지 기존 경로를 보존한다.
6. **WSL 구현을 삭제하지 않는다.** Windows 전용 선택 provider로 격리한다.
7. **CUDA/MPS/CPU를 하나의 문자열 치환으로 처리하지 않는다.** 장치 탐지, capability, 모델별 호환성 및 오류 정책을 명시적으로 구현한다.
8. **PyInstaller 패키징부터 시작하지 않는다.** 먼저 source mode에서 세 OS 경로와 기능을 검증한다.
9. **정확한 Python/PyTorch/CUDA 버전을 추측해 고정하지 않는다.** 공식 호환 정보와 각 OS 빌드 테스트를 통해 결정한다.
10. **기존 테스트 수를 품질의 유일한 지표로 삼지 않는다.** DB migration, UI, GPU, packaging, relocation 검증을 추가한다.
11. **사용자에게 보이는 동작을 유지한다.** 기존 CLI/웹 UI를 갑자기 제거하지 말고 개발자 모드로 유지할 방법을 고려한다.
12. **구현하지 않은 항목을 구현했다고 보고하지 않는다.** 계획/실제 구현/테스트 통과 결과를 분리해 기록한다.
13. **변경 전후 아키텍처와 주요 의사결정을 짧은 ADR 또는 문서로 남긴다.**

---

## 15. 첫 번째 Claude 작업 요청(복사해서 사용할 프롬프트)

아래 프롬프트를 Claude에 전달해 첫 작업을 시작할 수 있다.

```text
이 문서를 Local Auto Labeling 프로젝트의 제품 방향 및 아키텍처 인수인계 문서로 사용해 주세요.

먼저 현재 저장소의 HEAD, git status, 실제 디렉터리 구조, pyproject/requirements, 테스트를 확인하세요. 이 문서에 적힌 기존 코드 현황은 이전 정적 리뷰 결과이므로 현재 코드와 다를 수 있습니다. 확인하지 않은 내용을 사실로 가정하지 마세요.

첫 번째 작업은 Milestone 0.1 Portable Foundation의 설계 및 구현 계획을 구체화하는 것입니다. 당장 대규모 코드를 수정하기 전에 다음을 수행해 주세요.

1. 전역 ROOT 및 상대/절대 경로 해석 지점 전체 검색
2. 설정 파일, DB, 이미지, 모델, 캐시, 로그, 학습 결과의 현재 저장 위치와 경로 의존 관계 분석
3. WSL 및 기타 OS 전용 모듈의 import/호출 지점 분석
4. CUDA/장치 기본값 및 장치 선택 코드 분석
5. 기존 DB 스키마와 migration/backup 방식 분석
6. 현재 테스트 구조 및 테스트 실행 방법 확인
7. 위험도, 변경 순서, 호환성 계획, 롤백 계획 작성

그 후 작은 단위로 구현을 시작하세요. 사용자 데이터의 원본을 수정하거나 삭제하지 말고, migration은 복사본 fixture에서 먼저 검증하세요. 기존 기능 회귀를 방지하는 테스트를 변경과 함께 추가하세요.

목표는 기존 Windows 개인 개발 환경을 깨뜨리지 않으면서 Windows/macOS/Linux에서 공통으로 실행 가능한 Portable Foundation을 만드는 것입니다. 패키징은 이 기반이 검증된 후 진행합니다.
```

---

## 16. 아직 결정하지 않은 사항

다음은 추후 저장소 및 공식 라이브러리 호환성을 확인한 후 결정해야 한다.

- 배포 대상 Python 버전
- 각 OS별 PyTorch/torchvision/Ultralytics/SAM 정확한 버전 조합
- macOS Intel 지원 여부 및 시점
- Linux 배포판/glibc 최소 지원 기준
- CUDA 런타임 번들 방식과 NVIDIA 드라이버 최소 조건
- PyInstaller가 모든 ML 의존성을 안정적으로 번들할 수 있는지 여부
- MPS에서 YOLO 및 SAM의 기능별 호환 범위
- Linked Workspace에서 외부 이미지 경로가 변경되었을 때 재연결 UX
- 세션 토큰 등 로컬 API 인증의 구체적인 구현 방식
- macOS Developer ID 서명/공증 및 배포 비용/운영 정책
- 자동 업데이트를 1.0에 포함할지 이후 버전으로 미룰지

이 항목들은 미리 단정하지 말고 테스트 및 라이선스/공식 문서 확인을 통해 결정한다.

---

## 17. 요약

프로젝트의 목표는 개인용 Windows 개발 도구를 단순히 EXE로 만드는 것이 아니다.

**기존의 로컬 이미지 라벨링 및 ML 파이프라인을 보존하면서, Windows/macOS/Linux에서 실행 가능한 포터블 데스크톱 애플리케이션으로 발전시키는 것**이다.

가장 중요한 첫 단계는 다음 세 가지다.

1. 실행 파일/번들 리소스/UserData/Workspace 경로를 분리한다.
2. OS 및 GPU 차이를 Platform Manager와 Provider 구조로 격리한다.
3. 기존 DB와 사용자 이미지 데이터를 안전하게 이동·마이그레이션할 수 있게 한다.

이 기반이 완성된 다음 PyInstaller 등으로 OS별 `onedir` 패키지를 만들고, ZIP 배포 및 GitHub Actions 릴리스 파이프라인을 구축한다.

**권장 진행 순서: Portable Foundation → Cross-platform Python Runtime → Portable Packaging → Public Beta.**
