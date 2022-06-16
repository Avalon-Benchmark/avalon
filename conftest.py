import sys
import threading
from typing import Optional

import pytest
from _pytest.config.argparsing import Parser
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.python import Module
from _pytest.reports import CollectReport
from _pytest.runner import CallInfo


# noinspection SpellCheckingInspection
def pytest_sessionstart(session: Session):
    assert session, "to satisfy linter"
    if "--pycharm" in sys.argv:
        # noinspection PyUnresolvedReferences
        import pydevd_pycharm

        pydevd_pycharm.settrace("localhost", port=22345, stdoutToServer=True, stderrToServer=True, suspend=False)


# noinspection SpellCheckingInspection
def pytest_addoption(parser: Parser):
    parser.addoption("--pycharm", action="store_true", default=False, help="enable pycharm debugging")
    parser.addoption("--with-regression", action="store_const", const=True, dest="run_regression")
    parser.addoption("--without-regression", action="store_const", const=False, dest="run_regression")


# noinspection SpellCheckingInspection
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: Item):
    """
    Checks whether tests should be skipped based on integration settings
    """

    if item.get_closest_marker("regression_test"):
        if item.config.getoption("run_regression") in (None, True):
            return
        pytest.skip("Regression tests skipped")


debug_thread: Optional[threading.Thread] = None


# TASK 631605b5-5144-4334-8bd6-54081a7f3c19: this code actually causes you to stop in pycharm... but there are no variables
# see the other occurrence of this task id for our earlier sourceress way of doing this (which also no longer works)
def pytest_exception_interact(node: Module, call: CallInfo, report: CollectReport) -> CollectReport:
    """
    Drop into PyCharm debugger, if available, on uncaught exceptions.
    """
    assert node, "to satisfy linter"
    if "--pycharm" in sys.argv:

        exception_info = call.excinfo
        assert exception_info is not None
        # noinspection PyUnresolvedReferences,PyProtectedMember
        exc_type, value, traceback = exception_info._excinfo  # type: ignore

        # noinspection PyUnresolvedReferences
        import pydevd

        # noinspection PyUnresolvedReferences
        import pydevd_tracing

        debugger = None
        while debugger is None:
            debugger = pydevd.get_global_debugger()
        additional_info = pydevd.set_additional_thread_info(pydevd.thread)
        additional_info.suspended_at_unhandled = True
        pydevd_tracing.SetTrace(None)  # no tracing from here

        def actually_stop():
            debugger.stop_on_unhandled_exception(
                debugger, pydevd.thread, additional_info, (exc_type, value, traceback)
            )

        global debug_thread

        # stop on the first error, but allow the report to return so that you can see the error
        if debug_thread is None:
            debug_thread = threading.Thread(target=actually_stop)
            debug_thread.start()

    return report
