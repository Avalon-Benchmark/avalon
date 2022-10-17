import os
import platform
import sys
from pathlib import Path
from typing import Dict
from typing import Optional

from loguru import logger
from tqdm import tqdm

from avalon.common.error_utils import capture_exception
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import is_notebook

__all__ = [
    "logger",
    "enable_debug_logging",
    "get_log_path",
    "get_user_name",
    "get_experiment_name",
    "log_to_sentry",
    "log_to_sentry_and_reraise",
]


def get_user_name() -> str:
    return os.getenv("EXPERIMENT_USERNAME", "default_user")


def get_experiment_name() -> str:
    return os.getenv("EXPERIMENT_NAME", "default_experiment")


def get_log_path() -> Path:
    if platform.system().lower() == "darwin":
        log_folder = Path("/tmp")
    else:
        log_folder = Path(FILESYSTEM_ROOT)
    assert log_folder.is_dir(), "Log dir does not exist. Please create it."
    return log_folder


def _log_with_tqdm(msg: str) -> None:
    tqdm.write(msg, end="")


def log_to_sentry(exception: BaseException, extra: Optional[Dict[str, str]] = None):
    if is_notebook():
        raise exception

    if extra is None:
        extra = {}

    user = f"{get_user_name()}-sentry"
    experiment = get_experiment_name()
    # passing in extra fingerprint of user so that issues on sentry are unique per user
    capture_exception(exception, extra=dict(username=user, experiment=experiment, **extra), extra_fingerprint=user)


def log_to_sentry_and_reraise(exception: BaseException):
    log_to_sentry(exception)
    raise exception


def _configure_logger(default_level: str = "INFO"):
    logger.remove()
    log_file = str(get_log_path() / "log.txt")
    # logger.add(_log_with_tqdm, level=default_level, format="{message}")
    logger.add(sys.stdout, level=default_level, format="{message}")
    logger.add(log_file, level="DEBUG", format="{time}|{process.id}|{thread.id}| {level} | {file}:{line} | {message}")


_configure_logger()
tqdm.get_lock()
# disable monitoring thread because it is not totally safe, caused some issues with multithreading.
tqdm.monitor_interval = 0


def enable_debug_logging():
    _configure_logger("DEBUG")
