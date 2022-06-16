import json
import os
import platform
import random
import subprocess
import time
import urllib
from pathlib import Path
from typing import Optional

import ipykernel
import numpy as np
import torch
from notebook import notebookapp


def _get_filesystem_root() -> str:
    env_value = os.getenv("SCIENCE_FILESYSTEM_ROOT")
    if not env_value:
        if platform.system().lower() == "darwin":
            return "/tmp/science"
        else:
            return "/mnt/private"
    return env_value


FILESYSTEM_ROOT = _get_filesystem_root()
TEMP_DIR = os.path.join(FILESYSTEM_ROOT, "tmp")
os.makedirs(TEMP_DIR, exist_ok=True)

SHARED_FOLDER = "/mnt/shared"


def _get_tests_folder() -> str:
    env_value = os.getenv("SCIENCE_TESTS_FOLDER")
    if not env_value:
        if platform.system().lower() == "darwin":
            return "/tmp/tests"
        else:
            return os.path.join(SHARED_FOLDER, "tests")
    return env_value


TESTS_FOLDER = _get_tests_folder()


def is_notebook() -> bool:
    """This specific snippet checks if the file is run in interactive mode."""
    import __main__ as main

    return not hasattr(main, "__file__")


# from here: https://stackoverflow.com/questions/12544056/how-do-i-get-the-current-ipython-jupyter-notebook-name
def get_notebook_path() -> Optional[Path]:
    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    try:
        connection_file = os.path.basename(ipykernel.get_connection_file())
    except RuntimeError as e:
        if "app not specified, and not in a running Kernel" in str(e):
            return None
        raise
    else:
        kernel_id = connection_file.split("-", 1)[1].split(".")[0]

        for srv in notebookapp.list_running_servers():
            if srv["token"] == "" and not srv["password"]:  # No token and no password, ahem...
                # noinspection PyUnresolvedReferences
                req = urllib.request.urlopen(srv["url"] + "api/sessions")
            else:
                # noinspection PyUnresolvedReferences
                req = urllib.request.urlopen(srv["url"] + "api/sessions?token=" + srv["token"])
            sessions = json.load(req)
            for sess in sessions:
                if sess["kernel"]["id"] == kernel_id:
                    return Path(os.path.join(srv["notebook_dir"], sess["notebook"]["path"]))
    return None


def set_all_seeds(seed: int):
    # make determinisitc. was getting weird behavior
    torch.manual_seed(seed + 1)
    np.random.seed(seed + 2)
    random.seed(seed + 3)


# if training multiple networks, may want to call this before training each!
def make_deterministic(seed: int):
    set_all_seeds(seed)
    # noinspection PyUnresolvedReferences
    torch.use_deterministic_algorithms(True)
    # noinspection PyUnresolvedReferences
    torch.backends.cudnn.benchmark = False


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
