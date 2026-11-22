# AI Hub '개인형 이동장치 안전 데이터' 조사 (2026-11-22)

사용자가 받아 둔 AI Hub 데이터(집 데스크탑 `E:\datasets\120.개인형 이동장치 안전 데이터\01.데이터\`)를 프로젝트에 쓸 수
있는지 읽기 전용으로 본 스크립트다. 결과와 판단은 [docs/provenance.md](../../docs/provenance.md) '공개 데이터셋' 의 같은 이름
항목에 적었다 — 이 폴더는 그 수치를 다시 낼 때만 본다. **아직 프로젝트 DB 에는 아무것도 넣지 않았다.**

- 전부 데이터·DB 를 읽기만 한다. 결과(색인 JSONL·예측 JSON·대조 시트)는 인자로 준 폴더에 쓴다 — **저장소 밖에 둘 것**
  (색인은 130MB, 시트·크롭은 AI Hub 그림이라 git 에 넣으면 안 된다 — 약관이 제3자 제공을 막는다).
- 내려받은 데이터를 읽는 스크립트는 `python -I` 로 돌린다(모델을 쓰는 둘은 `-I` 없이, `YOLO_AUTOINSTALL=false`).
- 아래에서 `D` = `…\01.데이터`, `OUT` = 저장소 밖 작업 폴더.

| 스크립트 | 하는 일 | 인자 |
|---|---|---|
| `zip_check.py` | zip 목록(폴더별 장수·확장자)을 읽는다 — 끝까지 받아졌는지 | zip 경로들 |
| `zip_sample.py` | zip 을 처음부터 끝까지 고르게 N개 읽어 CRC·크기 확인 + 폴더별 예시 2장(PM 박스 그림) | zip · `D\1.Training\라벨링데이터` · OUT · N |
| `name_stats.py` | 파일 이름만으로 폴더별 장수·영상·클립 수 | `D\1.Training\라벨링데이터` |
| `vl_stats.py` | 검증 라벨(`VL1.zip`) 전수 — 키·코드·박스 크기·이상값 | `VL1.zip` |
| `index_labels.py` | 학습 라벨 58만 개를 읽어 PM 박스만 JSONL 한 파일로(7분쯤) | 라벨 폴더 · OUT 의 jsonl |
| `index_stats.py` | 그 색인으로 폴더별 종류(오토바이·자전거·킥보드) 박스·장·영상 수, 박스 높이 | jsonl |
| `dup_and_split.py` | 같은 물체에 박스가 둘인 곳, 검증 영상이 학습에도 있는지, 검증 라벨 이상값 | jsonl · `VL1.zip` |
| `code_sheets.py` | `PM_code` 별 크롭 대조 시트(영상마다 큰 박스 하나) | 라벨 폴더 · 그림 폴더 · OUT · 표본 수 |
| `rider_rule.py` | **우리** DB 에서 탈것 박스와 탄 사람 박스의 관계(탈것 박스가 사람을 포함하는지) + 예시 시트 | DB · 보관본이면 `1` · 옛 경로 앞부분 · 새 경로 앞부분 · OUT |
| `model_vs_pm.py` | 풀린 폴더에서 영상마다 1장씩 N장에 가중치를 돌려 예측을 JSON 으로 | 라벨 폴더 · 그림 폴더 · 가중치 · OUT 의 json · N · imgsz |
| `staged_sample.py` | 연출(주간·맑음) 킥보드 프레임을 영상마다 1장씩 zip 에서 꺼내 같은 일을 | jsonl · `TS3.zip\|TS4.zip` · OUT 폴더 · 가중치 · OUT 의 json · N |
| `analyze_pred.py` | 그 JSON 으로 PM 박스 자리에서 탈것·탄 사람을 찾았는지, 코드↔모델 클래스, 프레임당 예측 수 | json |
| `outside_sheet.py` | 모델이 찾은 이륜 박스 중 PM 박스 밖인 것의 크롭 시트(세워 둔 탈것이 라벨에 없는지) | json · 그림 폴더 · OUT 의 jpg |
| `disagree_sheet.py` | 킥보드 코드인데 모델이 bicycle 로 잡았거나 못 찾은 곳의 크롭 시트 | json · 그림 폴더 · OUT 폴더 |
| `scene_thumbs.py` | 영상마다 한 장의 축소본 상관으로 같은 화면끼리 묶어 카메라 자리 수를 어림 | 그림 폴더 · OUT 의 json |

탈것 종류는 코드 묶음으로 본다(13–19 오토바이 · 20–27 자전거 · 28–36 킥보드 — 그림과 모델 예측으로 확인한 것이고 AI Hub
페이지에는 코드표가 없다). `analyze_pred.py`·`index_stats.py`·`dup_and_split.py` 의 `group()` 이 그 대응이다.
