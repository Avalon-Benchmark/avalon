import os
import platform
import random
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir
from typing import Any
from typing import ContextManager
from typing import Generator
from uuid import uuid4

import numpy as np
import sh
import torch


def _get_filesystem_root() -> str:
    env_value = os.getenv("SCIENCE_FILESYSTEM_ROOT")
    if not env_value:
        return str(Path(gettempdir()) / "science")
    return env_value


def compact(x: Any) -> Any:
    return list(filter(None, x))


_SPACES_RE = re.compile(" +")
_git_command = sh.git.bake(_cwd=".", _tty_out=False)


class NotARepositoryError(Exception):
    pass


def get_current_git_version() -> str:
    try:
        return str(re.split(_SPACES_RE, _git_command.log("-n1").splitlines()[0], 2)[1])
    except sh.ErrorReturnCode as e:
        if e.exit_code == 128 and "fatal: not a git repository" in e.stderr.decode("UTF-8"):
            raise NotARepositoryError() from e
        else:
            raise


def is_git_repo_clean_on_notebook() -> bool:
    try:
        output = _git_command("status", "-s").splitlines()
    except sh.ErrorReturnCode as e:
        if e.exit_code == 128 and "fatal: not a git repository" in e.stderr.decode("UTF-8"):
            return True
        if e.exit_code == 1:
            return False
        else:
            raise
    else:
        non_notebook_lines = [x for x in output if not x.endswith(".ipynb")]
        return len(non_notebook_lines) == 0


FILESYSTEM_ROOT = _get_filesystem_root()
TEMP_DIR = os.path.join(FILESYSTEM_ROOT, "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)


def _get_tests_folder() -> str:
    env_value = os.getenv("SCIENCE_TESTS_FOLDER")
    if not env_value:
        return str(Path(gettempdir()) / "tests")
    return env_value


TESTS_FOLDER = _get_tests_folder()


def is_notebook() -> bool:
    """This specific snippet checks if the file is run in interactive mode."""
    import __main__ as main

    return not hasattr(main, "__file__")


def set_all_seeds(seed: int) -> None:
    # make deterministic. was getting weird behavior
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 2)
    random.seed(seed + 3)


# if training multiple networks, may want to call this before training each!
def make_deterministic(seed: int, num_intraop_threads: int = 1) -> None:
    # This bit is necessary to get torch.linalg to be deterministic in some cases, eg
    # https://github.com/pytorch/pytorch/issues/71222
    # Used in orthogonal initializer.
    # This might need to happen before torch loads, though? Not sure.
    # eg with `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_all_seeds(seed)
    # ref: https://github.com/pytorch/pytorch/issues/88718
    torch.set_num_threads(num_intraop_threads)
    torch.use_deterministic_algorithms(True)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.deterministic = True


def run_local_command(
    command: str, is_checked: bool = True, trace_output: bool = True
) -> subprocess.CompletedProcess[bytes]:
    if trace_output:
        print(command)
    # NOTE: do we really have to jump through all these hoops to get live line output?
    # yes. yes we do. yes, this is ridiculous. no, things aren't better in Python 3.9.
    # no, this isn't even enough - downstream things *will* still buffer on their own.
    # yes, we can make them not buffer... with a series of progressively sadder hacks.
    # the 80% solution is to run ssh with -tt, THEN DO NOT PIPE TO A FILE PLEASE STOP.
    # (that'd make our commands detect an interactive terminal and use line buffering)
    # DO NOT CHANGE STDIN TO ANYTHING BESIDES DEV NULL, that'll cause race conditions.
    process = subprocess.Popen(
        command,
        shell=True,
        executable="/bin/bash",
        bufsize=1,
        encoding="UTF-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "TERM": "dumb"},
    )
    all_lines = []
    exit_code = None
    while exit_code is None:
        exit_code = process.poll()
        while process.stdout:
            next_line = process.stdout.readline()
            if next_line:
                all_lines.append(next_line)
                if trace_output:
                    print(next_line.rstrip("\n"))
            else:
                break
        time.sleep(0.1)

    while next_line := process.stdout.readline():
        all_lines.append(next_line)
        if trace_output:
            print(next_line.rstrip("\n"))

    if is_checked:
        assert exit_code == 0

    str_lines = "".join(all_lines)
    return subprocess.CompletedProcess(command, exit_code, str_lines.encode("UTF-8"), b"")


def is_on_osx():
    return platform.system().lower() == "darwin"


def create_temp_file_path(cleanup: bool = True) -> ContextManager[Path]:
    @contextmanager
    def context() -> Generator[Path, None, None]:
        random_id = uuid4()
        output_path = os.path.join(TEMP_DIR, str(random_id))
        try:
            yield Path(output_path)
        finally:
            if cleanup and os.path.exists(output_path):
                if os.path.isfile(output_path):
                    os.remove(output_path)
                else:
                    shutil.rmtree(output_path)

    # noinspection PyTypeChecker
    return context()


def temp_dir(base_dir: str, is_uuid_concatenated: bool = False) -> ContextManager[Path]:
    @contextmanager
    def context() -> Generator[Path, None, None]:
        random_id = uuid4()
        if is_uuid_concatenated:
            output_path = Path(base_dir.rstrip("/") + "_" + str(random_id))
        else:
            output_path = Path(base_dir) / str(random_id)
        output_path.mkdir(parents=True, exist_ok=True)
        try:
            yield output_path
        finally:
            if output_path.exists():
                try:
                    shutil.rmtree(str(output_path))
                except OSError:
                    os.unlink(str(output_path))

    # noinspection PyTypeChecker
    return context()
