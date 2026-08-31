from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_RETRY_ATTEMPTS = 10
_RETRY_DELAY_SECONDS = 0.1


def move_dir(source: Path, target: Path) -> None:
    if sys.platform == "win32":
        max_attempts = _RETRY_ATTEMPTS
        delay = _RETRY_DELAY_SECONDS
    else:
        max_attempts = 1
        delay = 0.0
    for attempt in range(max_attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(delay)
