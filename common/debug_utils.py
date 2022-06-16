import sys
import threading
from types import TracebackType
from typing import Optional
from typing import Type

IS_NAMING_SUPPORTED = True


def dbg():
    # noinspection PyUnresolvedReferences
    import pydevd

    # import pydevd_pycharm
    pydevd.settrace("localhost", port=22345, stdoutToServer=True, stderrToServer=True, suspend=True)


# TASK 631605b5-5144-4334-8bd6-54081a7f3c19: this code is from our previous sourceress project, but no longer works correctly
# got this code from here: https://github.com/jlubcke/pytest-pycharm/blob/master/pytest_pycharm.py
def fix_stack_and_stop(exc_type: Type, value: Exception, traceback: Optional[TracebackType]):
    # noinspection PyUnresolvedReferences
    import pydevd

    # noinspection PyUnresolvedReferences
    from pydevd import pydevd_tracing

    frames = []
    while traceback:
        frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    thread = threading.current_thread()
    frames_by_id = dict([(id(frame), frame) for frame in frames])
    frame = frames[-1]
    exception = (exc_type, value, traceback)
    try:
        debugger = pydevd.debugger
    except AttributeError:
        debugger = None
        while debugger is None:
            debugger = pydevd.get_global_debugger()
    pydevd_tracing.SetTrace(None)  # no tracing from here
    try:
        debugger.stop_on_unhandled_exception(thread, frame, frames_by_id, exception)
    except AttributeError:
        # fallback to pre PyCharm 2019.2 API
        debugger.handle_post_mortem_stop(thread, frame, frames_by_id, exception)


def set_breakpoint_fixed_stack():
    # noinspection PyUnresolvedReferences
    import pydevd

    pydevd.settrace("localhost", port=22345, stdoutToServer=True, stderrToServer=True, suspend=True)
    exc_type, exc_value, tb = sys.exc_info()
    fix_stack_and_stop(exc_type, exc_value, tb)
