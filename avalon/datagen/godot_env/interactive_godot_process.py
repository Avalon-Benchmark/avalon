import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path
from posixpath import basename
from signal import SIGKILL
from signal import Signals
from signal import valid_signals
from typing import Deque
from typing import Final
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple

from avalon.common.log_utils import log_to_sentry
from avalon.common.log_utils import logger
from avalon.common.utils import AVALON_PACKAGE_DIR
from avalon.common.utils import wait_until_true
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import TEMP_DIR
from avalon.datagen.data_config import AbstractDataConfig
from avalon.datagen.errors import GodotError
from avalon.datagen.godot_generated_types import READY_LOG_SIGNAL
from avalon.datagen.godot_generated_types import SimSpec
from avalon.datagen.world_creation.world_generator import GenerateAvalonWorldParams

# TODO: Consider linking binaries to `godot_editor` and `godot_runner` allowing for more fine-grained control/inspection from python
GODOT_BINARY_PATH_ENV_FLAG: Final = "GODOT_BINARY_PATH"
GODOT_BINARY_PATH: Final = os.environ.get(GODOT_BINARY_PATH_ENV_FLAG, f"{AVALON_PACKAGE_DIR}/bin/godot")

GODOT_EDITOR_PATH_ENV_FLAG: Final = "GODOT_EDITOR_PATH"
GODOT_EDITOR_PATH: Final = os.environ.get("GODOT_EDITOR_PATH", f"{GODOT_BINARY_PATH}_editor")

GODOT_ERROR_LOG_PATH: Final = f"{FILESYSTEM_ROOT}/godot"

AVALON_GODOT_PROJECT_PATH: Final = f"{AVALON_PACKAGE_DIR}/datagen/godot"
AVALON_HUMAN_WORLDS_PATH: Final = f"{AVALON_GODOT_PROJECT_PATH}/worlds"

DATAGEN_SCRIPT_PATH: Final = f"{AVALON_GODOT_PROJECT_PATH}/datagen.sh"

_ACTION_REPLAY_FILENAME: Final = "actions_replay.out"


def _create_json_from_config(config: AbstractDataConfig) -> str:
    return json.dumps(config.to_dict(), indent=2)


ERROR_ALLOWLIST = (
    "Basis must be normalized",
    "tree_exiting",
    "libgodot_openxr.so",
    "No library set for this platform",
)

SYSTEM_FINALIZING_NO_GODOT_RERAISE = "Could not check Godot log during Python finalization"


def _read_log(log_path: str) -> Tuple[List[str], bool]:
    """Read the godot log and check for errors"""
    try:
        with open(log_path, "r") as infile:
            log_lines = infile.readlines()
    except NameError:
        # We check Godot logs during GodotEnv.__del__ and depending on the exact script + Python internals, it may get
        # called during Python finalization, when open() and other builtins are not available. Can't use a custom
        # error class since they can't be caught during finalization.
        # This tweak can possibly be removed when we're running Python 3.10; ref: https://bugs.python.org/issue26789
        # also see: https://pythondev.readthedocs.io/finalization.html
        if sys.is_finalizing():
            raise RuntimeError(SYSTEM_FINALIZING_NO_GODOT_RERAISE)

    is_disallowed_error_logged = False
    for l in log_lines:
        if "ERROR" in l and not any(x in l for x in ERROR_ALLOWLIST):
            is_disallowed_error_logged = True
    return log_lines, is_disallowed_error_logged


def _raise_godot_error(
    log_path: str,
    artifact_path: Path,
    log_content: Optional[str] = None,
    details: str = "",
):
    if log_content is None:
        try:
            with open(log_path, "r") as infile:
                log_content = infile.read()
        except FileNotFoundError as e:
            log_content = str(e)

    if details != "":
        details = details + " "
    else:
        match = re.search(r"^ERROR: (.*)$", log_content, re.MULTILINE)
        if match is not None:
            details = match.group(1) + " "

    error_message = f"{details}\nLog: {log_path}\nArtifacts: {artifact_path}"

    logger.error(f"Godot error: {error_message}")
    logger.error("Attempting to dump log file here...")
    logger.error(log_content)

    logger.error("Attempting to log error to sentry...")
    log_to_sentry(GodotError(error_message))

    raise GodotError(error_message)


def _popen_process_group(bash_cmd: Iterable[str], log_path: str, env: Mapping[str, str] = {}):
    with open(log_path, "wb") as log:
        return subprocess.Popen(
            list(bash_cmd),
            start_new_session=True,
            stdout=log,
            stderr=log,
            env=env,
        )


def _kill_process_group(process: subprocess.Popen, is_lookup_error_ok: bool = True):
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        if not is_lookup_error_ok:
            raise
        return
    assert pgid == process.pid, f"Cannot _kill_process_group of a child process (pgid={pgid}, pid={process.pid})"
    return os.killpg(pgid, SIGKILL)


def get_first_run_action_record_path(config_path: str) -> Path:
    return Path(config_path).parent.joinpath(f"actions.out")


def _derive_pipe_paths(config_path: str, log_path: str):
    action_record_path = get_first_run_action_record_path(config_path)
    if action_record_path.is_file():
        action_record_path = Path(log_path).parent.joinpath(_ACTION_REPLAY_FILENAME)
    action_pipe_path = log_path + ".actions"
    observation_pipe_path = log_path + ".observations"
    return str(action_record_path), action_pipe_path, observation_pipe_path


def create_error_artifact_path() -> Path:
    root_path = Path(GODOT_ERROR_LOG_PATH)
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path / f"godot_env_artifacts__{str(uuid.uuid4())}.tar.gz"


class InteractiveGodotProcess:
    def __init__(
        self,
        config: SimSpec,
        gpu_id: int,
        keep_log: bool = True,
        is_dev_flag_added: bool = False,
        run_uuid: Optional[str] = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.config = config
        self.keep_log = keep_log
        self.is_dev_flag_added = is_dev_flag_added
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.log_uuid = str(uuid.uuid4())
        log_root = os.path.join(TEMP_DIR, self.log_uuid)
        os.makedirs(log_root, exist_ok=True)
        self.log_path = os.path.join(log_root, f"godot.log")
        self.prefix = f"Worker (PID={os.getpid()}): "

        if run_uuid is None:
            self.run_uuid = str(uuid.uuid4())
        else:
            self.run_uuid = run_uuid

        self.action_record_path, self.action_pipe_path, self.observation_pipe_path = _derive_pipe_paths(
            self.config_path, self.log_path
        )

        self.output_file_read_loc = 0
        self.error_marker = "ERROR"
        self.output_file_buffer = (" " * (len(self.error_marker) + 1)).encode("UTF-8")
        self.artifact_path = create_error_artifact_path()

    @property
    def config_path(self) -> str:
        return os.path.join(self.config.get_dir_root(), str(self.run_uuid), "config.json")

    @property
    def _config_done_path(self) -> str:
        return self.config_path.replace(".json", ".done")

    @property
    def is_replay(self):
        return basename(self.action_record_path) == _ACTION_REPLAY_FILENAME

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    @property
    def is_finished(self) -> bool:
        return self.process is not None and self.process.returncode is not None

    @property
    def is_closed(self) -> bool:
        if not self.is_finished:
            return False
        is_config_still_present = Path(self.config_path).is_file()
        does_config_done_exist = Path(self._config_done_path).is_file()

        config_has_been_renamed = not is_config_still_present and does_config_done_exist
        if config_has_been_renamed:
            return config_has_been_renamed

        has_been_cleaned_up = not (is_config_still_present or does_config_done_exist)
        if has_been_cleaned_up:
            return True
        return False

    def _read_log(self):
        return _read_log(self.log_path)

    def _get_godot_command(
        self,
        input_pipe_path: str,
        output_pipe_path: str,
        extra_flags: Tuple[str, ...] = tuple(),
    ) -> Tuple[List[str], Mapping[str, str]]:
        assert self.gpu_id is not None, "Refusing to run godot process without a gpu_id"
        extra_flags = extra_flags + tuple([f"--cuda-gpu-id={self.gpu_id}"])

        resolution = (self.config.recording_options.resolution_x, self.config.recording_options.resolution_y)
        if self.config.recording_options.is_adding_debugging_views:
            resolution = (resolution[0] * 2, resolution[1] * 2)

        bash_args = [
            DATAGEN_SCRIPT_PATH,
            f"--thread_count=4",
            f"-U",
            f"--input_pipe_path={input_pipe_path}",
            f"--output_pipe_path={output_pipe_path}",
            f"--resolution={resolution[0]}x{resolution[1]}",
            *extra_flags,
            self.config_path,
        ]
        bash_env = {
            GODOT_BINARY_PATH_ENV_FLAG: GODOT_BINARY_PATH,
        }
        return bash_args, bash_env

    def start(self) -> None:
        assert os.path.exists(GODOT_BINARY_PATH), (
            f"Cannot run avalon: Godot binary has not been installed to to {GODOT_BINARY_PATH_ENV_FLAG}={GODOT_BINARY_PATH}. "
            f"Please run `python -m avalon.install_godot_binary` and try again."
        )
        # we open a file here so that we can watch
        # create the file because we want to open it now
        Path(self.log_path).touch()
        self.output_file = open(self.log_path, "rb")

        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as config_file:
            config_file.write(_create_json_from_config(self.config))

        extra_flags = ("--dev",) if self.is_dev_flag_added else tuple()
        godot_bash_args, godot_bash_env = self._get_godot_command(
            self.action_pipe_path, self.observation_pipe_path, extra_flags
        )
        logger.debug(f"{self.prefix} process group: {' '.join(godot_bash_args)}' &>> {self.log_path}")

        debug_bash_cmd = " ".join(self._get_godot_command(self.action_record_path, "/tmp/debug_output")[0])
        logger.debug(f"{self.prefix} TO DEBUG RUN: {debug_bash_cmd}")

        self.process = _popen_process_group(godot_bash_args, self.log_path, godot_bash_env)

    def save_artifacts(self, recent_worlds: Deque[GenerateAvalonWorldParams]) -> Path:
        tar_path = Path(self.artifact_path)
        with tarfile.open(tar_path, "x:gz") as tar:
            tar.add(self.action_record_path, arcname=Path(self.action_record_path).name)
            tar.add(self.log_path, arcname=Path(self.log_path).name)
            tar.add(self.config_path, arcname=Path(self.config_path).name)

            generated_root_path_by_id = {}
            for i, world in enumerate(recent_worlds):
                generation_id = uuid.uuid4()
                for world_path in world.output_path.rglob("*.tscn"):
                    generated_root_path_by_id[str(generation_id)] = str(world.output_path)
                    tar.add(world_path, arcname=f"worlds/{generation_id}/{world_path.name}")

            with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as tmp_file:
                json.dump(
                    {"run_uuid": str(self.run_uuid), "generated_root_path_by_id": generated_root_path_by_id},
                    tmp_file,
                )
                tmp_file.flush()
                tar.add(tmp_file.name, arcname=f"meta.json")
        return tar_path

    def check_for_errors(self) -> None:
        """Raises a GodotError if an error was encountered"""
        file_size = os.path.getsize(self.log_path)
        unread_byte_count = file_size - self.output_file_read_loc
        if unread_byte_count > 0:
            new_bytes = self.output_file.read(unread_byte_count)
            self.output_file_buffer = self.output_file_buffer + new_bytes
            line_buffer = self.output_file_buffer.decode("UTF-8", errors="ignore")
            lines = line_buffer.split("\n")
            # NOTE: we may have dropped something due to the "ignore" above but that should be fine
            self.output_file_buffer = lines.pop().encode("UTF-8")
            for line in lines:
                if self.error_marker in line and not any(x in line for x in ERROR_ALLOWLIST):
                    self._raise_error()

    def close(self, kill: bool = True, raise_logged_errors: bool = True):
        assert not self.is_closed, f"{self} already closed, cannot close again"

        self.output_file.close()

        if self.process is None:
            return

        if kill:
            _kill_process_group(self.process)

        if raise_logged_errors:
            self.raise_any_logged_godot_errors()

        try:
            wait_until_true(self._poll_for_exit)
        except Exception as e:
            _kill_process_group(self.process)
            if raise_logged_errors:
                raise
            logger.error(f"Got error {e} but continuing to exit")

    def _error_code_repr(self) -> str:
        if self.process is None or not self.process.returncode:
            return ""
        signal = abs(self.process.returncode)
        matching_known_signal = (s.name for s in valid_signals() if isinstance(s, Signals) and s.value == signal)
        name = next(matching_known_signal, "UNKNOWN")
        return f"returncode={name}({self.process.returncode})"

    def _raise_error(self, log_content: Optional[str] = None) -> None:
        _raise_godot_error(
            log_path=self.log_path,
            artifact_path=self.artifact_path,
            log_content=log_content,
            details=self._error_code_repr(),
        )

    def raise_any_logged_godot_errors(self) -> None:
        lines, is_error_logged = _read_log(self.log_path)
        if is_error_logged:
            self._raise_error("\n".join(lines))

    def wait_for_log_signal(self, log_signal: str) -> None:
        process = self.process
        assert process is not None, "Cannot wait_for_ready_signal() before start()"

        def wait_for_ready() -> bool:
            assert process is not None
            process.poll()
            if process.returncode:
                self._raise_error()
            log_lines, is_error_logged = self._read_log()
            if is_error_logged:
                _raise_godot_error(
                    log_path=self.log_path,
                    artifact_path=self.artifact_path,
                    log_content="\n".join(log_lines),
                )
            return any(log_signal in l for l in log_lines)

        try:
            wait_until_true(wait_for_ready, sleep_inc=0.0001, max_wait_sec=20)
        except TimeoutError as e:
            err = f"did not observe ready message {log_signal}: {str(e)}"
            _raise_godot_error(log_path=self.log_path, artifact_path=self.artifact_path, details=err)

    def wait_for_ready_signal(self) -> None:
        self.wait_for_log_signal(READY_LOG_SIGNAL)

    def _poll_for_exit(self) -> bool:
        assert self.process is not None, "Cannot call _poll_for_exit() before start()"
        self.raise_any_logged_godot_errors()
        self.process.poll()
        if self.process.returncode is None:
            return False
        self.process.wait()
        if self.process.returncode:
            self._raise_error()

        # TODO: this whole mess is not multiprocess-safe
        os.rename(self.config_path, self._config_done_path)
        if not self.keep_log:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)

        return True
