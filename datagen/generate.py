import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from posixpath import basename
from signal import SIGKILL
from signal import Signals
from signal import valid_signals
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from uuid import uuid4

from tqdm.auto import tqdm

from common.log_utils import logger
from contrib.utils import TEMP_DIR
from datagen.data_config import AbstractDataConfig
from datagen.errors import GodotError
from datagen.godot_generated_types import READY_LOG_SIGNAL
from datagen.godot_generated_types import SimSpec

DATAGEN_PATH = os.path.dirname(__file__)
EXPERIMENT_DIR = os.path.dirname(DATAGEN_PATH)
OLD_GODOT_PATH = f"{DATAGEN_PATH}/old_godot"
NEW_GODOT_PATH = f"{DATAGEN_PATH}/godot"

_ACTION_REPLAY_FILENAME = "actions_replay.out"


def _create_json_from_config(config: AbstractDataConfig) -> str:
    return json.dumps(config.to_dict(), indent=2)


def _is_test_env() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def _move_dir_and_ignore_if_exists(tmp_dir: str, local_cache_dir: str):
    try:
        os.rename(tmp_dir, local_cache_dir)
    except OSError as e:
        if e.errno == 39:
            logger.info(
                f"{local_cache_dir} already exists and was probably created by someone else. Our datagen should be deterministic so we'll skip updating {local_cache_dir}"
            )
        else:
            raise


ERROR_ALLOWLIST = (
    "Basis must be normalized",
    "tree_exiting",
    "libgodot_openxr.so",
    "No library set for this platform",
)


def _read_log(log_path: str) -> Tuple[List[str], bool]:
    """Read the godot log and check for errors"""
    with open(log_path, "r") as infile:
        log_lines = infile.readlines()
    is_disallowed_error_logged = False
    for l in log_lines:
        if "ERROR" in l and not any(x in l for x in ERROR_ALLOWLIST):
            is_disallowed_error_logged = True
    return log_lines, is_disallowed_error_logged


def _raise_godot_error(log_path: str, log_content: Optional[str] = None, details: str = ""):
    if log_content is None:
        try:
            with open(log_path, "r") as infile:
                log_content = infile.read()
        except FileNotFoundError as e:
            log_content = str(e)
    if details != "":
        details = details + " "
    error_message = f"Godot error: {details}(see {log_path})"
    logger.error(error_message)
    logger.error("Attempting to dump log file here...")
    logger.error(log_content)
    raise GodotError(error_message)


def _popen_process_group(bash_cmd: Iterable[str], log_path: str):
    with open(log_path, "wb") as log:
        return subprocess.Popen(
            list(bash_cmd),
            start_new_session=True,
            stdout=log,
            stderr=log,
        )


def _kill_process_group(process: subprocess.Popen):
    pgid = os.getpgid(process.pid)
    assert pgid == process.pid, f"Cannot to _kill_process_group of a child process (pgid={pgid}, pid={process.pid})"
    return os.killpg(pgid, SIGKILL)


class GodotProcess:
    def __init__(self, config: AbstractDataConfig, progress: tqdm, keep_log: bool = True):
        self.config = config
        self.keep_log = keep_log
        self.progress = progress
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.is_cleanup_done = False
        log_uuid = str(uuid4())
        self.log_path = os.path.join(TEMP_DIR, f"godot-{log_uuid}.log")
        self.prefix = f"Worker (PID={os.getpid()}): "

        progress.total = config.get_video_range().size
        progress.n = 0

    @property
    def config_path(self) -> str:
        return os.path.join(self.config.get_dir_root(), "config.json")

    def _get_godot_command(self):
        return [
            f"{NEW_GODOT_PATH}/datagen.sh",
            "--thread_count=4",
            self.config_path,
        ]

    def start(self) -> bool:
        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as config_file:
            config_file.write(_create_json_from_config(self.config))
        godot_bash_args = self._get_godot_command()
        # logger.info(f"{self.prefix} process group: {' '.join(godot_bash_args)}' &>> {self.log_path}")
        self.process = _popen_process_group(godot_bash_args, self.log_path)
        return True

    def raise_error(self):
        _raise_godot_error(self.log_path)

    def poll(self):
        assert self.process is not None, "Cannot call poll() before start()"
        if self.process.returncode is None:
            if os.path.exists(self.log_path):
                lines, is_error_logged = _read_log(self.log_path)
                if is_error_logged:
                    _raise_godot_error(self.log_path, "\n".join(lines))
                video_lines = [x for x in lines if x.startswith("video:")]
                video_count = len(video_lines)
                if video_count:
                    self.progress.set_description(
                        f"{self.prefix} Created {video_count} of {self.progress.total} videos"
                    )
                    self.progress.n = video_count
                else:
                    self.progress.set_description(f"{self.prefix} Initializing...")
            else:
                self.progress.set_description(f"{self.prefix} Starting process...")

            self.progress.update(0)
            self.process.poll()
            return False
        else:
            if not self.is_cleanup_done:
                self.process.wait()
                if self.process.returncode:
                    self.raise_error()
                os.rename(self.config_path, self.config_path.replace(".json", ".done"))
                if not self.keep_log:
                    if os.path.exists(self.log_path):
                        os.remove(self.log_path)
                self.is_cleanup_done = True
            return True

    def wait_for(self):
        if self.process is None:
            return
        while not self.poll():
            time.sleep(0.5)


def get_first_run_action_record_path(config_path: str) -> Path:
    return Path(config_path).parent.joinpath(f"actions.out")


def _derive_pipe_paths(config_path: str, log_path: str):
    action_record_path = get_first_run_action_record_path(config_path)
    if action_record_path.is_file():
        action_record_path = Path(log_path).parent.joinpath(_ACTION_REPLAY_FILENAME)
    action_pipe_path = log_path + ".actions"
    observation_pipe_path = log_path + ".observations"
    return str(action_record_path), action_pipe_path, observation_pipe_path


class InteractiveGodotProcess:
    def __init__(self, config: SimSpec, keep_log: bool = True, gpu_id: Optional[int] = None):
        self.gpu_id = gpu_id
        self.config = config
        self.keep_log = keep_log
        self.process: Optional[subprocess.Popen[bytes]] = None
        log_uuid = str(uuid4())
        log_root = os.path.join(TEMP_DIR, log_uuid)
        os.makedirs(log_root, exist_ok=True)
        self.log_path = os.path.join(log_root, f"godot.log")
        self.prefix = f"Worker (PID={os.getpid()}): "
        # Maybe this is what prefix was supposed to be for?
        self.uuid = uuid.uuid4()
        self.action_record_path, self.action_pipe_path, self.observation_pipe_path = _derive_pipe_paths(
            self.config_path, self.log_path
        )

        self.output_file_read_loc = 0
        self.error_marker = "ERROR".encode("UTF-8")
        self.output_file_buffer = (" " * (len(self.error_marker) + 1)).encode("UTF-8")

    @property
    def config_path(self) -> str:
        # TODO (hackathon) fix recording option flags here so this makes sense
        if self.config.recording_options.is_recording_rgb:
            return os.path.join(self.config.get_dir_root(), str(self.uuid), "config.json")
        else:
            return os.path.join(Path(self.log_path).parent, str(self.uuid), "config.json")

    @property
    def _config_done_path(self) -> str:
        return self.config_path.replace(".json", ".done")

    @property
    def is_replay(self):
        return basename(self.action_record_path) == _ACTION_REPLAY_FILENAME

    @property
    def is_running(self):
        return self.process is not None and self.process.returncode is None

    @property
    def is_finished(self):
        return self.process is not None and self.process.returncode is not None

    @property
    def is_closed(self):
        if not self.is_finished:
            return False
        config_has_been_renamed = (not Path(self.config_path).is_file()) and Path(self._config_done_path).is_file()
        return config_has_been_renamed

    def _read_log(self):
        return _read_log(self.log_path)

    def _get_godot_command(
        self,
        input_pipe_path: str,
        output_pipe_path: str,
        extra_flags: Tuple[str, ...] = tuple(),
    ):
        if self.gpu_id is not None:
            extra_flags = extra_flags + tuple([f"--cuda-gpu-id={self.gpu_id}"])
        else:
            assert False
        return [
            f"{NEW_GODOT_PATH}/datagen.sh",
            f"--thread_count=4",
            f"-U",
            f"--input_pipe_path={input_pipe_path}",
            f"--output_pipe_path={output_pipe_path}",
            *extra_flags,
            self.config_path,
        ]

    def start(self):
        # we open a file here so that we can watch
        # create the file because we want to open it now
        Path(self.log_path).touch()
        self.output_file = open(self.log_path, "rb")

        Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as config_file:
            config_file.write(_create_json_from_config(self.config))

        godot_bash_args = self._get_godot_command(self.action_pipe_path, self.observation_pipe_path)
        logger.info(f"{self.prefix} process group: {' '.join(godot_bash_args)}' &>> {self.log_path}")

        debug_bash_cmd = " ".join(self._get_godot_command(self.action_record_path, "/tmp/debug_output"))
        logger.info(f"{self.prefix} TO DEBUG RUN: {debug_bash_cmd}")

        self.process = _popen_process_group(godot_bash_args, self.log_path)

    def check_for_errors(self):
        """Raises a GodotError if an error was encountered"""
        file_size = os.path.getsize(self.log_path)
        unread_byte_count = file_size - self.output_file_read_loc
        if unread_byte_count > 0:
            new_bytes = self.output_file.read(unread_byte_count)
            self.output_file_buffer = self.output_file_buffer + new_bytes
            if self.error_marker in self.output_file_buffer and not any(
                x.encode("UTF-8") in self.output_file_buffer for x in ERROR_ALLOWLIST
            ):
                self._raise_error()
            else:
                self.output_file_buffer = self.output_file_buffer[-len(self.error_marker) + 1 :]

    def close(self, kill: bool = True, raise_logged_errors: bool = True):
        assert not self.is_closed, f"{self} already closed, cannot close again"

        self.output_file.close()

        if self.process is None:
            return

        if raise_logged_errors:
            self._raise_any_logged_godot_errors()

        if kill:
            _kill_process_group(self.process)

        try:
            wait_until_true(self._poll_for_exit)
        except Exception as e:
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

    def _raise_error(self, log_content: Optional[str] = None):
        _raise_godot_error(
            self.log_path,
            log_content=log_content,
            details=self._error_code_repr(),
        )

    def _raise_any_logged_godot_errors(self):
        lines, is_error_logged = _read_log(self.log_path)
        if is_error_logged:
            self._raise_error("\n".join(lines))

    def wait_for_ready_signal(self):
        process = self.process
        assert process is not None, "Cannot wait_for_ready_signal() before start()"

        def wait_for_ready():
            process.poll()
            if process.returncode:
                self._raise_error()
            log_lines, is_error_logged = self._read_log()
            if is_error_logged:
                _raise_godot_error(self.log_path, "\n".join(log_lines))
            if any(READY_LOG_SIGNAL in l for l in log_lines):
                return True

        try:
            wait_until_true(wait_for_ready, sleep_inc=0.0001)
        except TimeoutError as e:
            err = f"did not observe ready message {READY_LOG_SIGNAL}: {str(e)}"
            _raise_godot_error(self.log_path, details=err)

    def _poll_for_exit(self):
        assert self.process is not None, "Cannot call _poll_for_exit() before start()"
        self._raise_any_logged_godot_errors()
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


def call_godot(config: AbstractDataConfig, progress: tqdm, keep_log: bool = True) -> bool:
    process = GodotProcess(config, progress, keep_log)
    process.start()
    process.wait_for()
    return False


def wait_until_true(
    callback: Callable[[], Optional[bool]],
    max_wait_sec: float = 5,
    sleep_inc: float = 0.25,
):
    """Repeatedly call callback() until it returns True or max_wait_sec is reached"""
    waited_for_sec = 0.0000
    while waited_for_sec <= max_wait_sec:
        if callback():
            return
        time.sleep(sleep_inc)
        waited_for_sec += sleep_inc
    raise TimeoutError(f"could not complete within {max_wait_sec} seconds")
