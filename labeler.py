#!/usr/bin/env python
"""프로젝트 진입점. `python labeler.py <명령>` 으로 실행한다."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from autolabel.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
