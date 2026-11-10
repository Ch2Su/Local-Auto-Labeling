# 다른 디스크 백업 — 담긴 것·다시 뜨기·되살리기

2026-11-10 에 저장소 루트 CLAUDE.md 에서 낱말 그대로 옮겨 왔다(그 파일이 매 세션 읽혀 크기를 줄이느라). 글 속의 '위·아래 …' 는
루트 CLAUDE.md 의 절을 가리킨다.

**다른 디스크 백업: `C:\local_auto_labeling_backup\20261031\`**(C: = 디스크 1, E: 와 다른 물리 디스크). `.venv`·`graft`·
`data/visible/dataset`·`__pycache__`·`*.tar`·살아 있는 DB 파일을 뺀 전부(8,127개 5.11GB, md5 전수 대조)와 sqlite `backup()` 으로
뜬 DB 다. 자세한 것은 그 폴더의 `BACKUP_INFO.txt`. Objects365 tar 는 들어 있지 않다. E: 의 외부 백업(09-11)은 11-02 에 사용자가
지웠다. **그 위에 더하는 `C:\local_auto_labeling_backup\20261105\`**(11-05, 28MB — 사용자 확인): 그날의 DB 스냅샷(integrity ok,
행 수 원본과 같음), 10-31 뒤에 생긴 이미지 3장(id 7226–7228)·`domain_review/`(seen.csv)·classes.txt·구내식당 id 목록·
`docs/collab.md`, push 전 커밋까지 담은 `repo_main.bundle`. 이미지는 10-31 백업에 6,115장 중 6,112장이 크기까지 같게 있어 다시
뜨지 않았다. 되살릴 때는 10-31 의 옛 폴더 구조(`seed/`·`labeled/`)를 `legacy/` 아래로 옮긴 뒤 이것을 덮는다(그 폴더의
`BACKUP_INFO.txt`). 하루 사본·`labeling.db.bak_*`(11-02·03, 약 210MB)는 넣지 않았다. **다시 그 위에 더하는
`C:\local_auto_labeling_backup\20261110\`**(11-10, 파일 812개 299MB + DB 57MB — 사용자가 이미지를 크게 지우기 전, 사용자 확인):
그때 DB 이미지 5,931장 중 10-31·11-05 백업에 없던 768장(VIRAT 515·ROAD PLUS hw-01–09 253 — legacy 5,163장은 크기까지 같게
있었다), 자르기 전 hw-04 원본 36장(`backups/crop_20261109_154300`), 지우기 전 DB 스냅숏(`labeling.db.bak_predelete_20261110_160048`
을 다시 `backup()`, journal_mode=delete)과 같은 시점 라벨의 텍스트 사본(`backups/labels_predelete_20261110_160048/` —
`images.jsonl.gz` 한 줄에 이미지 하나·박스·태그·가릴 영역), `cameras.yaml`·`classes.txt`·`docs/collab.md`·`repo_main.bundle`.
md5 는 `md5.tsv` 로 전부 원본과 같다. 원본 영상(VIRAT `E:\datasets`, hw-01–05 클립 `data/visible/videos/`)은 넣지 않았다. 되살릴
때는 20261031 → 20261105 → 20261110 순서로 덮는다(그 폴더의 `BACKUP_INFO.txt`). 다시 뜰 때는(PowerShell, `$src`=저장소
`$dst`=새 날짜 폴더 — **이전 것을 지울지 먼저 물을 것**):

```powershell
robocopy $src $dst /E /XJ /COPY:DAT /DCOPY:DAT /R:1 /W:1 /MT:8 /XD "$src\.venv" "$src\graft" "$src\data\visible\dataset" "$src\data\visible\thumbs" __pycache__ /XF labeling.db labeling.db-wal labeling.db-shm labeling.db-journal *.tar
```

종료 코드 0–7 이 성공이고, 그 뒤 DB 를 sqlite `backup()` 으로 `$dst\data\visible\labeling.db` 에 뜬다. **되살리기는 README '새
PC 에서 시작하기' 4·5단계대로 네 폴더만** 가져온다 — 백업 폴더를 통째로 덮는 `BACKUP_INFO.txt` 1단계는 저장소까지 잃었을
때만이다(안의 `.git` 은 메일을 바꾸기 전 이력이고, 옛 문서까지 덮인다).
