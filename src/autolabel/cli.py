"""명령줄 진입점.

    python labeler.py scan       # 폴더 훑어서 DB 등록 (+ 클래스 확정)
    python labeler.py train      # 승인된 라벨로 학습
    python labeler.py resume     # 중단된 학습을 이어서 진행
    python labeler.py label      # 미라벨 이미지 자동 라벨링
    python labeler.py review     # 검수 웹 UI 실행
    python labeler.py tidy       # 검수 완료 이미지를 data/labeled 로 정리
    python labeler.py check      # DB 와 실제 이미지가 맞는지 확인 (편집 후 점검)
    python labeler.py split      # train/val/test 배정 확인 (--reset 으로 전체 재배정)
    python labeler.py dupcheck   # 내용이 같은 이미지 찾기 (--folder 로 넣기 전 폴더 검사)
    python labeler.py prune      # 원본이 사라진 DB 항목 삭제
    python labeler.py purge      # 휴지통 비우기 (원본 파일까지 영구 삭제)
    python labeler.py export     # YOLO 포맷으로 내보내기
    python labeler.py status     # 현재 상태 요약
"""

from __future__ import annotations

import argparse
import sys

from .config import Config, resolve
from .export import export_yolo
from .ingest import discover_classes, ingest
from .predictor import AutoLabeler
from .split import ensure_splits, leak_report
from .stopper import StopSignal
from .store import Store
from .tidy import check_images, prune_missing, purge_trashed, tidy_approved
from .trainer import find_resume_point
from .trainer import resume_train as run_resume
from .trainer import train as run_train


def _setup(args) -> tuple[Config, Store]:
    cfg = Config.load(args.config)
    return cfg, Store(cfg.db)


def cmd_scan(args) -> int:
    cfg, store = _setup(args)
    # ingest 를 먼저 돌려야 클래스 추론이 등록된 박스를 볼 수 있다.
    report = ingest(cfg, store, force=args.force)
    classes = discover_classes(cfg, store)
    print(report.summary())
    print(f"클래스({len(classes)}개): {classes}")
    if not classes:
        print(
            "\n경고: 클래스가 비어 있습니다.\n"
            f"  config.yaml의 classes 항목을 채우거나 {cfg.classes_txt} 에 한 줄에 하나씩 적으세요.",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_train(args) -> int:
    cfg, store = _setup(args)
    stop = StopSignal().install()
    print(stop.banner() + "\n", flush=True)
    try:
        info = run_train(cfg, store, epochs=args.epochs, should_stop=stop)
    finally:
        stop.clear()
    print("\n학습 중단됨 (이어하기: labeler.py resume)" if stop.reason else "\n학습 완료")
    print(f"  결과 폴더 : {info['run_dir']}")
    print(f"  가중치    : {info['weights']}")
    for k, v in info["metrics"].items():
        print(f"  {k}: {v:.4f}")
    return 0


def cmd_resume(args) -> int:
    cfg, store = _setup(args)

    if args.check:
        point = find_resume_point(cfg)
        if point is None:
            print("이어서 학습할 체크포인트가 없습니다.")
            return 1
        print(point.summary())
        return 0

    stop = StopSignal().install()
    print(stop.banner() + "\n", flush=True)
    try:
        info = run_resume(cfg, store, batch=args.batch, should_stop=stop)
    finally:
        stop.clear()
    print("\n학습 중단됨 (다시 이어하기: labeler.py resume)" if stop.reason
          else "\n학습 완료 (이어서 진행)")
    print(f"  결과 폴더 : {info['run_dir']}")
    print(f"  가중치    : {info['weights']}")
    print(f"  재개 지점 : {info['resumed_at_epoch']}/{info['total_epochs']} epoch")
    for k, v in info["metrics"].items():
        print(f"  {k}: {v:.4f}")
    return 0


def cmd_label(args) -> int:
    cfg, store = _setup(args)
    labeler = AutoLabeler(cfg, store)

    def progress(n: int, total: int, name: str) -> None:
        print(f"\r[{n}/{total}] {name[:50]:<50}", end="", flush=True)

    ids = None
    if args.all:
        ids = [r["id"] for r in store.list_images(status="unlabeled", limit=10**9)]
        ids += [r["id"] for r in store.list_images(status="pending", limit=10**9)]
    report = labeler.run(ids, progress=progress)
    print("\n" + report.summary())
    print("\n다음: python labeler.py review  로 검수하세요.")
    return 0


def cmd_review(args) -> int:
    import uvicorn

    from .projects import load_projects
    from .server import create_app, create_multi_app

    # `--config` 를 직접 준 것은 '이 설정 하나만' 이라는 뜻이다. 안 줬으면 projects.yaml 의
    # 프로젝트를 모두 띄우고 상단 탭으로 오간다 (2026-09-21).
    if args.config:
        cfg, projects = Config.load(args.config), None
    else:
        projects = load_projects()
        cfg = projects[0].cfg

    host = args.host or str(cfg.get("server.host", "127.0.0.1"))
    port = args.port or int(cfg.get("server.port", 7860))
    # 연 주소를 앱에 알린다 — 다른 Host 로 들어온 요청(DNS rebinding)을 막는 데 쓴다 (server._RequestGuard, 2026-11-05)
    app = create_app(cfg, bind_host=host) if projects is None else create_multi_app(projects, bind_host=host)
    # 안내 문구는 여기서 낸다. run.bat 에 한글을 넣으면 chcp 65001 상태의 cmd.exe 가
    # 줄 조각을 명령으로 실행해 버린다 (run.bat 주석 참고).
    print(f"검수 UI: http://{host}:{port}   (Ctrl+C 로 종료)")
    if projects:
        for p in projects:
            print(f"  - {p.title}: http://{host}:{port}/p/{p.name}/   ({p.config_path.name})")
    print("브라우저에서 위 주소를 여세요. 이 창을 닫으면 진행 중인 작업도 함께 종료됩니다.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def cmd_tidy(args) -> int:
    cfg, store = _setup(args)

    def progress(n: int, total: int, name: str) -> None:
        print(f"\r[{n}/{total}] {name[:50]:<50}", end="", flush=True)

    report = tidy_approved(cfg, store, progress=progress)
    print("\n" + report.summary())
    return 0


def cmd_prune(args) -> int:
    cfg, store = _setup(args)
    report = prune_missing(
        cfg, store, dry_run=args.dry_run,
        max_ratio=1.1 if args.force else 0.5,
    )
    print(report.summary())
    if args.dry_run and report.missing:
        print("\n실제로 지우려면 --dry-run 없이 다시 실행하세요.")
    return 1 if report.aborted else 0


def cmd_check(args) -> int:
    cfg, store = _setup(args)
    report = check_images(cfg, store, fix_size=args.fix_size)
    print(report.summary())
    # --fix-size 로 크기를 전부 맞췄다면 남은 문제로 치지 않는다.
    unresolved = bool(report.missing) or bool(report.unreadable) or bool(report.noncanonical) or (
        bool(report.resized) and report.fixed < len(report.resized)
    )
    return 1 if unresolved else 0


def cmd_split(args) -> int:
    """train/val/test 배정을 보여준다. --reset 이면 전부 다시 나눈다."""
    cfg, store = _setup(args)
    if args.reset and not args.yes:
        print("--reset 은 전체를 다시 나눕니다. 이전 학습들과 점수 비교가 끊깁니다\n"
              "  (그 모델들이 새 val/test 이미지를 학습에 썼을 수 있습니다).\n"
              "  실행하려면 --yes 를 함께 주세요.")
        return 1
    _, report = ensure_splits(cfg, store, reset=args.reset, regroup=args.regroup)
    print(report.summary())
    if args.leak:
        print()
        print(leak_report(cfg, store, max_dist=args.leak_dist))
    return 0


def cmd_dupcheck(args) -> int:
    """내용(바이트)이 같은 이미지를 찾는다. DB 에는 아무것도 쓰지 않는다."""
    from .dupcheck import check

    cfg, store = _setup(args)
    folder = resolve(args.folder) if args.folder else None
    if folder is not None and not folder.exists():
        print(f"폴더가 없습니다: {folder}", file=sys.stderr)
        return 1
    report = check(store, folder)
    print(report.summary())
    return 0


def cmd_purge(args) -> int:
    cfg, store = _setup(args)
    pending = store.counts().get("trashed", 0)
    if not pending:
        print("휴지통이 비어 있습니다.")
        return 0
    if not args.yes:
        print(f"휴지통에 {pending}장이 있습니다. 원본 파일과 라벨을 영구 삭제합니다.")
        print("되돌릴 수 없습니다. 실행하려면 --yes 를 붙이세요.")
        return 1

    def progress(n: int, total: int, name: str) -> None:
        print(f"\r[{n}/{total}] {name[:50]:<50}", end="", flush=True)

    report = purge_trashed(cfg, store, progress=progress)
    print("\n" + report.summary())
    return 0


def cmd_export(args) -> int:
    cfg, store = _setup(args)
    report = export_yolo(cfg, store, resolve(args.out) if args.out else None,
                         with_images=args.with_images)
    print(report.summary())
    return 0


def cmd_status(args) -> int:
    cfg, store = _setup(args)
    counts = store.counts()
    classes = store.get_classes()
    print(f"DB        : {cfg.db}")
    print(f"가중치    : {cfg.active_weights()}" + ("" if cfg.weights.exists() else "  (아직 학습 전)"))
    print(f"클래스    : {classes or '(비어 있음)'}")
    print("이미지 상태:")
    for key, label in [
        ("unlabeled", "미라벨    "),
        ("pending", "검수 대기 "),
        ("approved", "승인      "),
        ("skipped", "건너뜀    "),
    ]:
        print(f"  {label}: {counts[key]:>6}")
    print(f"  {'합계      '}: {counts['total']:>6}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="labeler",
        description="로컬 자동 이미지 라벨링 (YOLO 바운딩 박스)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=None, help="설정 파일 경로 (기본: config.yaml)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("scan", help="시드/미라벨 폴더를 훑어 DB에 등록")
    sp.add_argument("--force", action="store_true",
                    help="DB 이미지의 절반 넘게 디스크에 없어도 스캔한다 (경로를 고치기 전이면 같은 사진이 다시 등록된다)")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("train", help="승인된 라벨로 모델 학습")
    sp.add_argument("--epochs", type=int, default=None)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("resume", help="중단된 학습을 이어서 진행")
    sp.add_argument(
        "--check", action="store_true",
        help="이어 학습할 체크포인트만 찾아 보여주고 끝낸다",
    )
    sp.add_argument(
        "--batch", type=int, default=None,
        help="배치 크기를 바꿔서 이어간다 (VRAM 부족으로 느려질 때 줄인다)",
    )
    sp.set_defaults(func=cmd_resume)

    sp = sub.add_parser("label", help="미라벨 이미지에 자동 라벨 생성")
    sp.add_argument(
        "--all", action="store_true", help="이미 검수 대기 중인 이미지도 다시 추론"
    )
    sp.set_defaults(func=cmd_label)

    sp = sub.add_parser("review", help="검수 웹 UI 실행")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("tidy", help="검수 완료 이미지를 자동 라벨링 폴더 밖으로 정리")
    sp.set_defaults(func=cmd_tidy)

    sp = sub.add_parser(
        "check", help="DB 와 실제 이미지 파일이 맞는지 확인 (편집 후 점검용)")
    sp.add_argument(
        "--fix-size", action="store_true",
        help="픽셀 크기가 달라진 항목을 실제 파일에 맞춰 DB 를 갱신한다",
    )
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser(
        "split", help="train/val/test 배정 확인 (--reset 으로 전체 재배정)")
    sp.add_argument("--reset", action="store_true",
                    help="전체를 다시 나눈다. 이전 학습과 점수 비교가 끊긴다")
    sp.add_argument("--yes", action="store_true", help="--reset 확인")
    sp.add_argument("--regroup", action="store_true", help="같은 장면 묶음을 다시 계산한다")
    sp.add_argument("--leak", action="store_true",
                    help="val·test 가 train 과 얼마나 비슷한지(잔여 누수) 함께 잰다")
    sp.add_argument("--leak-dist", type=int, default=6,
                    help="잔여 누수 판정 dHash 거리 (기본 6, 묶음 기준은 2)")
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser(
        "dupcheck", help="내용이 같은 이미지 찾기 (DB 안, 또는 넣으려는 폴더 대 DB). 읽기만 한다")
    sp.add_argument("--folder", default=None,
                    help="이 폴더의 이미지가 이미 등록된 것과 같은지 본다 (예: 새 로보플로 내보내기)")
    sp.set_defaults(func=cmd_dupcheck)

    sp = sub.add_parser("prune", help="원본 파일이 사라진 DB 항목 삭제")
    sp.add_argument("--dry-run", action="store_true", help="지우지 않고 대상만 보여줌")
    sp.add_argument("--force", action="store_true", help="절반 넘게 사라져도 강행")
    sp.set_defaults(func=cmd_prune)

    sp = sub.add_parser("purge", help="휴지통 비우기 (원본 파일까지 영구 삭제)")
    sp.add_argument("--yes", action="store_true", help="확인 없이 삭제")
    sp.set_defaults(func=cmd_purge)

    sp = sub.add_parser("export", help="승인된 라벨을 YOLO 포맷으로 내보내기")
    sp.add_argument("--out", default=None,
                    help="출력 폴더 (기본: export/<프로젝트>, 예: export/visible). "
                         "비어 있거나 예전 내보내기 폴더여야 한다")
    sp.add_argument("--with-images", action="store_true", help="이미지도 함께 복사")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("status", help="현재 상태 요약")
    sp.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
