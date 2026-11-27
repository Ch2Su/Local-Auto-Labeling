"""작업 큐의 로그 (jobs.py) — 진행 표시줄과 사람에게 알리는 줄 (2026-11-27, Gemini 리뷰 묶음 7).

tqdm 진행 표시줄은 같은 줄을 덮어쓰며 갱신하고, 작업 로그도 그 줄을 덮어쓴다(`sink`). 그런데 중단 버튼이 넣는
`[중단 요청]` 줄도 그냥 마지막 줄이어서, 진행 표시줄이 도는 중에 누르면 바로 다음 갱신이 그 줄을 지웠다.

작업은 진짜 `JobManager` 로 돌리고(GPU 를 안 쓰는 종류), 작업 함수가 정해 둔 조각을 차례로 낸다 — 조각 사이에 중단을
누르는 것도 작업 스레드가 한다(순서가 정해진다).
"""

from __future__ import annotations

import threading

from autolabel.jobs import JobManager, _install_tees

NOTICE = "[중단 요청] 진행 중인 단계를 마치는 대로 종료합니다..."
CR, LF = chr(13), chr(10)       # 줄 되돌리기·줄바꿈 — 글자로 적지 않는다(루트 CLAUDE.md 함정 6)


def bar(text: str) -> str:
    """tqdm 이 내는 꼴 — 줄 앞에 CR 을 붙여 같은 줄을 다시 쓴다(그래서 한 조각은 다음 조각이 올 때 로그에 들어간다)."""
    return CR + text


def wait(job) -> None:
    for _ in range(500):
        if job.status != "running":
            return
        threading.Event().wait(0.01)


def run_job(steps) -> list[str]:
    """steps = 조각(글자)이나 호출할 것(관리자·job 을 받는다)의 목록 — 작업 스레드가 차례로 낸다. 끝난 뒤의 로그."""
    man = JobManager()
    out, _ = _install_tees()

    def work(job) -> dict:
        for step in steps:
            if callable(step):
                step(man, job)
            else:
                out.write(step)         # 작업이 print 한 것과 같다 — pytest 가 테스트마다 sys.stdout 을 갈아서 직접 쓴다
        return {}

    job = man.start("tidy", work)
    wait(job)
    assert job.status == "done", job.error
    return list(job.log)


def cancel(man, job) -> None:
    assert man.cancel(job.id) is True


def test_a_progress_bar_keeps_overwriting_one_line():
    log = run_job(["start" + LF, bar(" 10%|#"), bar(" 50%|##"), bar("100%|###"), LF + "finished" + LF])
    assert log == ["start", "100%|###", "finished"]


def test_the_cancel_notice_survives_the_next_progress_update():
    """진행 표시줄이 도는 중에 중단을 눌러도 안내 줄이 남는다 — 그 뒤의 진행 표시줄은 새 줄에서 이어 간다.
    고치기 전에는 안내 줄이 마지막 줄이라 바로 다음 갱신(20%)이 그 줄을 덮어썼다."""
    log = run_job(["start" + LF, bar(" 10%|#"), bar(" 20%|#"), cancel, bar(" 50%|##"), bar("100%|###"),
                   LF + "stopped" + LF])
    assert log == ["start", " 10%|#", NOTICE, "100%|###", "stopped"]


def test_the_cancel_notice_after_a_full_line_is_kept_too():
    log = run_job(["start" + LF, cancel, "next" + LF, bar(" 10%|#"), bar(" 20%|#"), LF])
    assert log == ["start", NOTICE, "next", " 20%|#"]


def test_cancel_reports_the_flag_and_refuses_finished_jobs():
    man = JobManager()
    seen: list[bool] = []

    def work(job) -> dict:
        assert man.cancel(job.id) is True
        seen.append(job.cancel_requested)
        return {}

    job = man.start("tidy", work)
    wait(job)
    assert seen == [True] and man.cancel(job.id) is False and man.cancel(999) is False
    assert job.log.count(NOTICE) == 1
