import shutil
import tarfile
import time
from pathlib import Path
from typing import Callable
from typing import Generator
from typing import Optional

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from avalon.contrib.testing_utils import fixture
from avalon.contrib.testing_utils import integration_test
from avalon.contrib.testing_utils import slow_integration_test
from avalon.contrib.testing_utils import use
from avalon.datagen.avalon_godot_tests.conftest import AvalonEnv
from avalon.datagen.avalon_godot_tests.conftest import godot_env_
from avalon.datagen.avalon_godot_tests.scenario import get_vr_action
from avalon.datagen.errors import GodotError
from avalon.datagen.godot_env._bridge import GodotEnvBridge
from avalon.datagen.godot_env._bridge import _BridgeKillSwitch
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.godot_env.actions import _to_bytes
from avalon.datagen.godot_env.interactive_godot_process import InteractiveGodotProcess
from avalon.datagen.godot_env.interactive_godot_process import wait_until_true
from avalon.datagen.godot_generated_types import READY_LOG_SIGNAL
from avalon.datagen.godot_utils import create_env_from_artifacts


def assert_eventually_true(
    callback: Callable[[], Optional[bool]],
    max_wait_sec: float = 1,
    sleep_inc: float = 0.001,
):
    try:
        wait_until_true(callback, max_wait_sec, sleep_inc)
    except TimeoutError as e:
        assert callback(), e.args[0]


@fixture
@use(godot_env_)
def godot_env_with_reset_(godot_env: AvalonEnv) -> AvalonEnv:
    godot_env.reset()
    return godot_env


@fixture
def caplog_(caplog: LogCaptureFixture):
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)


@fixture
def fast_kill_switch_patch_(monkeypatch: pytest.MonkeyPatch):
    mock_timeout = 0.5

    def mock_watch_fast_timeout(self: _BridgeKillSwitch, _: Optional[float] = None):
        self._kill_time = time.time() + mock_timeout
        return self

    monkeypatch.setattr(_BridgeKillSwitch, "watch_blocking_action", mock_watch_fast_timeout)
    monkeypatch.setattr(_BridgeKillSwitch.__init__, "__defaults__", (mock_timeout, mock_timeout))


@fixture
def fast_open_close_kill_switch_patch_(monkeypatch: pytest.MonkeyPatch):
    mock_timeout = 0.5
    monkeypatch.setattr(GodotEnvBridge.__init__, "__defaults__", (mock_timeout, mock_timeout))


@fixture
def never_ready_godot_process_patch_(monkeypatch: pytest.MonkeyPatch):
    def mock_never_ready(self: InteractiveGodotProcess):
        process = self.process
        assert process is not None, "Cannot wait_for_ready_signal() before start()"

        def just_raise():
            if process.returncode:
                self._raise_error()

        wait_until_true(just_raise, max_wait_sec=2)

    monkeypatch.setattr(InteractiveGodotProcess, "wait_for_ready_signal", mock_never_ready)


@slow_integration_test
@use(
    godot_env_,
    caplog_,
    fast_kill_switch_patch_,
    never_ready_godot_process_patch_,
    fast_open_close_kill_switch_patch_,
)
def test_kill_switch_watches_startup(
    godot_env: AvalonEnv,
    caplog: LogCaptureFixture,
):
    godot_env.close()

    with pytest.raises(GodotError):
        godot_env._restart_process()

    assert "killing godot" in caplog.text
    assert_eventually_true(lambda: "SIGKILL" in godot_env.process._error_code_repr(), max_wait_sec=1)


@slow_integration_test
@use(
    godot_env_with_reset_,
    fast_kill_switch_patch_,
)
def test_kill_switch_watches_messages(
    godot_env_with_reset: AvalonEnv,
    monkeypatch: pytest.MonkeyPatch,
):
    godot_env = godot_env_with_reset

    def blocking_to_bytes(_: VRAction) -> bytes:
        return _to_bytes(int, 4)

    monkeypatch.setattr(VRAction, "to_bytes", blocking_to_bytes)

    with pytest.raises(GodotError, match=r"Invalid number of bytes.*") as e:
        godot_env.act(get_vr_action())

    cause = e.value.__cause__
    assert cause is not None and isinstance(cause, GodotError)

    assert_eventually_true(lambda: "SIGKILL" in godot_env.process._error_code_repr(), max_wait_sec=1)

    with pytest.raises(GodotError, match=r"return.*code.*SIGKILL"):
        godot_env.process._poll_for_exit()
    godot_env._bridge.after_close()


@slow_integration_test
@use(godot_env_)
def test_env_opens_and_closes_nicely(godot_env: AvalonEnv):
    assert godot_env.process.is_running
    assert godot_env._bridge.is_open

    logs, err_logged = godot_env.process._read_log()
    assert not err_logged
    assert any(READY_LOG_SIGNAL in log for log in logs)

    godot_env.close()
    assert godot_env.process.process is None or godot_env.process.process.returncode == 0
    assert not godot_env.process.is_running
    assert not godot_env._bridge.is_open
    assert_eventually_true(lambda: godot_env.process.is_closed)

    logs, err_logged = godot_env.process._read_log()
    assert not err_logged
    assert any("CLOSE_MESSAGE" in log for log in logs)


@integration_test
@use(godot_env_)
def test_requires_reset(godot_env: AvalonEnv):
    with pytest.raises(Exception, match=r"reset.*act"):
        godot_env.act(get_vr_action())
    godot_env.close()


@fixture
def godot_env_bridge_with_godot_error_on_act_(monkeypatch: pytest.MonkeyPatch):
    def mock_act_to_throw_exception(*args):  # type: ignore[no-untyped-def]
        raise GodotError

    monkeypatch.setattr(GodotEnvBridge, "act", mock_act_to_throw_exception)


@fixture
def interactive_godot_process_with_godot_error_on_check_for_errors_(monkeypatch: pytest.MonkeyPatch):
    def mock_check_for_errors_to_throw_exception(*args):  # type: ignore[no-untyped-def]
        raise GodotError

    monkeypatch.setattr(InteractiveGodotProcess, "check_for_errors", mock_check_for_errors_to_throw_exception)


@fixture
def artifact_path_() -> Generator[Path, None, None]:
    artifact_parent_path = Path("/tmp/test_replay_from_error_artifacts")
    artifact_parent_path.mkdir(parents=True)
    yield artifact_parent_path / "godot_env_artifacts.tar.gz"
    shutil.rmtree(artifact_parent_path)


@fixture
@use(artifact_path_)
def mock_create_error_artifact_path_(artifact_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "avalon.datagen.godot_env.interactive_godot_process.create_error_artifact_path", lambda: artifact_path
    )


@slow_integration_test
@use(
    godot_env_with_reset_,
    artifact_path_,
    godot_env_bridge_with_godot_error_on_act_,
    interactive_godot_process_with_godot_error_on_check_for_errors_,
    mock_create_error_artifact_path_,
)
def test_replay_from_error_artifacts(
    godot_env_with_reset: AvalonEnv, artifact_path: Path, monkeypatch: pytest.MonkeyPatch
):
    godot_env = godot_env_with_reset

    monkeypatch.setattr(godot_env.process, "artifact_path", artifact_path)

    with pytest.raises(GodotError):
        godot_env.act(get_vr_action())
    godot_env.close()

    assert godot_env.process.artifact_path.exists()

    unpacked_artifacts_path = artifact_path.parent
    tar = tarfile.open(artifact_path)
    tar.extractall(path=unpacked_artifacts_path)
    tar.close()
    artifact_path.unlink()

    godot_env_replay = create_env_from_artifacts(unpacked_artifacts_path)

    with pytest.raises(GodotError):
        list(godot_env_replay())
