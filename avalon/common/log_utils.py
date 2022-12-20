import json
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
    "configure_local_logger",
    "configure_remote_logger",
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
        log_folder = Path("/tmp/logs")
    else:
        log_folder = Path(FILESYSTEM_ROOT) / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder


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


tqdm.get_lock()
# disable monitoring thread because it is not totally safe, caused some issues with multithreading.
tqdm.monitor_interval = 0


def configure_local_logger(level: str = "DEBUG", format: Optional[str] = None) -> None:
    try:
        from computronium.common.log_utils import configure_local_logger as inner_configure_local_logger
    except ImportError:
        logger.remove()
        logger.add(sys.stdout, level=level, format="{message}")
        os.environ["_LOG_CONFIG"] = json.dumps(dict(level=level, format=format))
    else:
        if format is None:
            inner_configure_local_logger(level)
        else:
            inner_configure_local_logger(level, format)


def configure_remote_logger(level: str = "DEBUG", format: Optional[str] = None) -> None:
    try:
        from computronium.common.log_utils import configure_remote_logger as inner_configure_remote_logger
    except ImportError:
        logger.remove()
        logger.add(sys.stdout, level=level, format="{message}")
        os.environ["_LOG_CONFIG"] = json.dumps(dict(level=level, format=format))
    else:
        if format is None:
            inner_configure_remote_logger(level)
        else:
            inner_configure_remote_logger(level, format)


def configure_parent_logging() -> None:
    try:
        from computronium.common.log_utils import configure_parent_logging as inner_configure_parent_logging
    except ImportError:
        if "_LOG_CONFIG" not in os.environ:
            raise Exception(
                "No parent process logging configuration specified. You are likely attempting to call configure_parent_logging without having already called configure_remote_logger or configure_local_logger in a parent process. If this is not a spawned subprocess, you should call one of those two functions rather than calling this function."
            )
        log_config = json.loads(os.environ["_LOG_CONFIG"])
        configure_local_logger(**log_config)
    else:
        inner_configure_parent_logging()
