# 국가교통정보센터(ITS) CCTV Open API 조사 (2026-11-24)

사용자가 받은 인증키로 'CCTV 화상자료'·'돌발상황정보' API 가 무엇을 주는지 잰 스크립트다. 결과와 판단은
[docs/provenance.md](../../docs/provenance.md) '국가교통정보센터(ITS) CCTV Open API' 에 적었다 — 이 폴더는 그 수치를 다시 낼 때만 본다.
**가져온 그림은 없다**(정지 영상은 머리 4KB 만 받아 크기를 읽고, 영상은 프레임을 세기만 한다 — 디스크에 쓰는 것은 API 응답뿐이다).

- **키:** 환경 변수 `ITS_API_KEY`(프로세스에 없으면 Windows 사용자 환경 변수에서 읽는다). 값을 찍지 않고 오류 글에서도
  지운다(`scrub`). 키는 요청 주소의 `apiKey=` 에 들어간다 — 그 주소를 로그·파일에 남기지 말 것.
- **API 를 부르는 것은 `list`·`events` 둘뿐이고 한 번 돌리면 1회다**(월 한도는 개발용 100건·운영용 1,000건). 나머지 명령은
  저장해 둔 목록의 영상 주소(CDN, 24시간 유효)만 읽거나 파일끼리 견준다.
- **결과는 저장소 밖에 둘 것** — 목록 JSON 에 실제 지명(`cctvname`)과 영상 주소가 들어 있다. `runs/` 아래의 `.json` 은 git 에
  담기니 `--out` 을 이 폴더로 주면 안 된다(`data/` 아래나 저장소 밖 작업 폴더로).
- 아래에서 `OUT` = 저장소 밖 작업 폴더. 돌리기: `.venv\Scripts\python.exe runs\its_probe\its_probe.py <명령> …`
  (PowerShell 에서는 먼저 `[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)` — 한글 출력이 UTF-8 이다).

| 명령 | 하는 일 | 인자 |
|---|---|---|
| `list` | **API 1회.** 전국 상자(124–132°E · 33–39°N)의 카메라 목록을 `OUT\list_<type>_<cctvType>.json` 에 쓰고 칸 채움·주소 꼴·번호 범위를 찍는다 | `--type ex\|its` · `--cctv-type 1..5` · `--out OUT` · (`--box minX maxX minY maxY`) |
| `summary` | 저장한 목록을 다시 요약 | `--in` 목록 |
| `stills` | 정지 영상(cctvType 3) 목록에서 그림 머리만 받아 크기 분포. `--n 0` 이면 전부 | `--in` 목록 · `--n` · `--seed` · `--range 4096`(앞 N 바이트만) · `--rps`(초당 요청 한도) · `--workers` · `--out` 결과 json |
| `fresh` | 같은 정지 영상 주소를 몇 번 받아 그림이 바뀌는지(메모리에서 해시만) | `--in` 목록 · `--n` · `--times` · `--gap` 초 |
| `hls` | 실시간(cctvType 4) 주소의 재생 목록 태그·조각 길이, 끝 조각 둘로 잰 비트레이트, OpenCV 로 연 프레임 크기. 카메라는 무작위, 번호(`--ids`), 또는 `stills --out` 결과에서 그 크기인 것(`--size-from` · `--size`)으로 고른다 | `--in` 목록 · `--n` · `--seed` · (`--ids 12,34`) · (`--size-from` json · `--size 1920x1080`) · (`--no-cv`) |
| `mp4` | 동영상(cctvType 5) 주소의 파일 크기·길이·프레임 크기 | `--in` 목록 · `--n` · `--seed` |
| `soak` | 실시간 한 대를 N초 이어 읽어 끊기는지·3초 넘게 멈추는지 | `--in` 목록 · `--seconds` · `--seed` |
| `concurrent` | 실시간 K대를 한꺼번에 N초 | `--in` 목록 · `--k` · `--seconds` · `--seed` |
| `events` | **API 1회.** 돌발상황정보 전체(`type=all`·`eventType=all`)를 `OUT\events_all.json` 에 쓰고 도로·유형별 수를 찍는다 | `--out OUT` |
| `events-summary` | 저장한 돌발 목록을 다시 요약 | `--in` |
| `near` | 돌발 지점마다 가장 가까운 CCTV 까지의 거리(좌표끼리 — 그 카메라에 현장이 보이는지는 모른다) | `--events` · `--lists` 목록들 |

`its_ids.py OUT list_ex_3.json list_ex_4.json …` — 주소 첫 칸(번호)이 목록끼리 같은 카메라(이름·좌표)를 가리키는지,
이름·좌표가 여러 번호에 겹치는지(API 를 부르지 않는다).

**11-24 에 돌린 것** (API 6회 — `list` ex·its × cctvType 3·4, ex × 5, `events`): 결과 파일은 git 밖
`data/visible/its/20261124/`(목록 5개 · `events_all.json` · `stills --n 200 --seed 1124` 의 `stills_*.json` ·
`stills --n 0 --range 4096 --rps 40 --workers 8` 의 `census_*.json`). 그 밖에 `hls --n 3 --seed 5`(ex·its),
`hls --size-from census_its.json --size 1920x1080 --n 3 --seed 3`, `hls --size-from census_ex.json --size 1280x720 --n 2 --seed 3`,
`mp4 --n 3 --seed 5`(ex), `fresh --n 3 --seed 21 --times 4 --gap 12`(ex), `soak --seconds 300 --seed 11`(ex),
`concurrent --k 4 --seconds 30 --seed 31`(ex), `near`(ex·its 의 cctvType 3 목록). 목록 속 영상 주소는 24시간 뒤 죽으니 영상을
읽는 명령을 다시 돌리려면 `list` 부터 다시 받아야 한다(API 횟수가 든다).
