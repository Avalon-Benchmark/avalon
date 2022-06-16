"""
This module is imported before anything else is run and can be used to monkeypatch any necessary libraries.
"""
import common.even_earlier_hacks  # isort:skip
import io
import json
import os
import resource
import sys
from typing import Optional

# We have to import numpy before pytorch to solve this issue:
# https://github.com/pytorch/pytorch/issues/37377
import numpy  # noqa
import psutil
import torch
import torch.utils.data

import common.tqdm_hacks


def _set_gpu_max_memory():
    mem_limit = None
    _machine_spec_file = os.path.expanduser("~/machine_spec.txt")
    if os.path.exists(_machine_spec_file):
        with open(_machine_spec_file, "r") as infile:
            data = json.loads(infile.read().strip())
            if data.get("gpu_mem_gb"):
                mem_limit = int(data["gpu_mem_gb"]) * 1024 ** 3

    if mem_limit is None:
        return

    if not torch.cuda.is_available():
        return

    torch.cuda.set_per_process_memory_fraction(min(mem_limit / torch.cuda.get_device_properties(0).total_memory, 1.0))


def _is_pid_running(pid: Optional[int]):
    for proc in psutil.process_iter():
        try:
            if proc.pid == pid:
                return proc.status() != "zombie"
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


# PyTorch, seriously, we don't really need 2137 dataloader processes, it's FINE
torch.utils.data.DataLoader.check_worker_number_rationality = lambda self: None  # type: ignore

resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))
_set_gpu_max_memory()

_is_notebook_debugging = False


def set_notebook_debug_flag(flag: bool):
    global _is_notebook_debugging
    _is_notebook_debugging = flag


def get_notebook_debug_flag():
    global _is_notebook_debugging
    return _is_notebook_debugging


def fix_stream_buffering():
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(line_buffering=True)


fix_stream_buffering()
common.tqdm_hacks.setup()


def setup():
    """A noop, just so that the import is not optimized away"""
