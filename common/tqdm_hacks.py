#!/usr/bin/env python3
import sys
import time
from typing import Any
from typing import Callable
from typing import List
from typing import Optional

import tqdm.std
import tqdm.utils

from common.even_earlier_hacks import is_notebook

TQDM_LAST_DISPLAY_TIME = 0.0
TQDM_MAX_UPDATE_PERIOD = 1.0


def tqdm_disable_output(
    self: tqdm.std.tqdm,
    tqdm_instance: tqdm.std.tqdm,
    func: Callable[..., Any],
) -> Callable[..., Any]:
    tqdm_instance._last_line = ""

    def new_write(text: str):
        tqdm_instance._last_line = text

    def new_flush():
        return

    if func.__name__ == "write":
        return new_write
    elif func.__name__ == "flush":
        return new_flush
    else:
        raise NotImplementedError()


def tqdm_moveto(self: tqdm.std.tqdm, n: int):
    return


def tqdm_refresh(self: tqdm.std.tqdm, nolock: bool = False, lock_args: Optional[List[Any]] = None):
    # NOTE: the locking logic looks bad, but it was copied verbatim
    if self.disable:
        return
    if not nolock:
        if lock_args:
            if not self._lock.acquire(*lock_args):
                return False
        else:
            self._lock.acquire()

    if tqdm_display_time():
        tqdm_display_all()

    if not nolock:
        self._lock.release()
    return True


def tqdm_display_time():
    global TQDM_LAST_DISPLAY_TIME
    time_now = time.time()
    if time_now > TQDM_LAST_DISPLAY_TIME + TQDM_MAX_UPDATE_PERIOD:
        TQDM_LAST_DISPLAY_TIME = time_now
        return True
    else:
        return False


def tqdm_display_all():
    lines = ["\n"]
    for bar in sorted(tqdm.std.tqdm._instances):
        bar.display()
        lines.append(bar._last_line.replace("\r", ""))
    lines.append("\n")
    sys.stdout.writelines("".join(lines))


def setup():
    if is_notebook():
        return
    if sys.stdout.isatty():
        return
    tqdm.utils.DisableOnWriteError.disable_on_exception = tqdm_disable_output
    tqdm.std.tqdm._last_display_time = 0
    tqdm.std.tqdm.moveto = tqdm_moveto
    tqdm.std.tqdm.refresh = tqdm_refresh
