"""학습을 '정상 중단'시키는 신호.

웹 UI 의 중단 버튼과 같은 효과를 CLI 에서도 내기 위한 것이다. `trainer.stop` 을
세우면 **다음 배치 경계에서** 배치 루프를 빠져나온다(engine/trainer.py 의
`on_train_batch_end` 직후 `if self.stop:`). 진행 중이던 epoch 의 **나머지는
건너뛰고**, 검증과 저장은 그대로 한 뒤 끝난다. 그 epoch 은 '완료'로 기록되므로
이어 학습은 다음 epoch 부터 시작한다 — 한 epoch 의 일부를 덜 학습하는 셈이지만
300 epoch 에서는 무시할 만하다. `final_eval()` 이 last.pt 를 훼손하기 전에
resume.pt 가 남는다 (CLAUDE.md 함정 8).

경로는 셋이다.

1. **정지 파일** — `runs/STOP` 이 생기면 멈춘다. **가장 확실하다.**
   어느 터미널·탐색기·원격 접속에서든 만들 수 있고, DataLoader 워커를 전혀
   건드리지 않는다. WSL·렌탈 서버에서도 똑같이 쓴다.

       touch runs/STOP               (WSL / Linux)
       type nul > runs\\STOP          (Windows cmd)
       New-Item runs\\STOP            (PowerShell)

   웹 UI 로 시작한 학습도 이 파일을 본다.

2. **SIGTERM** — `kill <메인 PID>`. 메인 프로세스에만 가므로 정상 중단된다.
   Linux/WSL 전용이다. Windows 의 Stop-Process 는 신호가 아니라 강제 종료다.

3. **Ctrl+C 는 가로채지 않는다.** 터미널의 Ctrl+C 는 DataLoader 워커에게도
   전달되고, 워커는 KeyboardInterrupt 를 받으면 루프를 빠져나간다
   (torch/utils/data/_utils/worker.py 의 주석: "Main process will raise
   KeyboardInterrupt anyways"). 메인에서 신호를 삼키고 배치 경계까지 가려 해도
   워커가 이미 죽어 있어 에러로 끝난다. 그래서 Ctrl+C 는 '즉시 종료'로 둔다.
   그래도 `final_eval()` 이 돌지 않으므로 last.pt 는 온전하다 — 잃는 것은
   진행 중이던 epoch 하나이고 `labeler.py resume` 으로 이어갈 수 있다.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

from .config import resolve


def stop_file() -> Path:
    """정지 파일 위치. 프로젝트가 여럿이어도 같은 runs/ 를 쓰므로 하나로 둔다.

    두 학습을 동시에 돌리지 않는 전제다 (VRAM 8GB).
    """
    return resolve("runs") / "STOP"


class StopSignal:
    """호출하면 '지금 멈춰야 하는가'를 돌려준다. trainer 의 should_stop 에 넘긴다."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or stop_file()
        self.reason: str | None = None
        self._flag = False
        # 시작할 때 남아 있던 STOP 은 지난 학습의 잔재다. 그대로 두면 새 학습이
        # 시작하자마자 멈춘다.
        self.clear()

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def install(self) -> "StopSignal":
        """SIGTERM 을 정상 중단으로 받는다. 메인 스레드에서만 부를 수 있다."""
        if sys.platform != "win32" and hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, self._on_sigterm)
        return self

    def _on_sigterm(self, signum, frame) -> None:
        if self._flag:
            return
        self._flag = True
        self.reason = "SIGTERM"
        print("\n[중단] SIGTERM 을 받았습니다. 다음 배치에서 멈추고 검증·저장합니다.",
              flush=True)

    def __call__(self) -> bool:
        if self._flag:
            return True
        if self.path.exists():
            self._flag = True
            self.reason = "정지 파일"
            print(f"\n[중단] {self.path} 가 생겼습니다. 다음 배치에서 멈추고 검증·저장합니다.",
                  flush=True)
            return True
        return False

    def banner(self) -> str:
        how = [f"  정지 파일 : {self.path} 를 만들면 됩니다"]
        if sys.platform != "win32":
            how.append("  SIGTERM   : kill <이 프로세스 PID>")
        how.append("  (Ctrl+C 는 즉시 종료입니다. 진행 중이던 epoch 만 잃고 이어할 수 있습니다)")
        return "정상 중단 방법:\n" + "\n".join(how)
