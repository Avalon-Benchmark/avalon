import os
import platform
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import psutil
import wandb
from loguru import logger
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from common import slack_service
from common.error_utils import capture_exception
from common.process_utils import get_process
from common.utils import get_path_to_current_dir
from contrib.serialization import Serializable
from contrib.utils import FILESYSTEM_ROOT
from contrib.utils import get_notebook_path
from contrib.utils import is_notebook

__all__ = [
    "logger",
    "enable_debug_logging",
    "is_tqdm_impossible",
    "get_log_path",
    "get_user_name",
    "get_experiment_name",
    "create_test_wandb_logger_from_notebook",
    "log_to_sentry",
    "log_to_sentry_and_reraise",
    "log_to_slack",
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


# no need to do this manually as the slack integration will do it automatically
def log_to_slack(exception: BaseException, user: str, experiment_name: str):
    exception_traceback = "".join(traceback.TracebackException.from_exception(exception).format())
    slack_message = f"@{user} - your experiment `{experiment_name}` failed. Here is the exception traceback: ```{exception_traceback}```"
    slack_service.post_text_to_slack_channel(slack_message)


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


def create_test_wandb_logger_from_notebook(
    logger_name: str, offline: bool = True, experiment_name: Optional[str] = None
) -> WandbLogger:
    notebook_path = get_notebook_path()
    assert notebook_path is not None
    test_name = notebook_path.parts[-1].replace(".ipynb", "")
    save_dir = str(get_log_path() / "tests" / test_name / logger_name / "wandb")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="test", name=experiment_name, offline=offline, save_dir=save_dir, entity="sourceress"
    )
    wandb_logger.experiment.name  # this initializes wandb, but is slightly hacky...
    return wandb_logger


def is_tqdm_impossible() -> bool:
    if "PYTEST_CURRENT_TEST" in os.environ:
        return True
    # TASK 811d011a-66a9-4960-bfa3-eab503992b74: disable tqdm over ssh
    try:
        process = get_process(os.getpid())
        if process is not None:
            cmd = " ".join(process.cmdline())
            is_pytest = "pytest" in cmd and "-s" in cmd.split("pytest ", 1)[-1]
            if is_pytest:
                parent = process.parent()
                if parent is not None:
                    parent_cmd = " ".join(parent.cmdline())
                    is_parent_ipython_shell = "python -m ipykernel_launcher" in parent_cmd
                    if is_parent_ipython_shell:
                        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return False


def log_config_to_wandb(config: Dict[str, Any]):
    current_dir_path = get_path_to_current_dir()

    # add git hash to the wandb config
    with open(current_dir_path / "current_git_hash") as file:
        git_hash = file.readline()[:-1]
        wandb.config.update(config)
        wandb.config.update({"git_hash": git_hash})

    # upload the file with the git diff to wandb
    wandb.save(str(current_dir_path / "current_git_diff"))


def aggregate_mean(x: List[float]):
    return sum(x) / len(x)


class LogAggregator(Serializable):
    def __init__(self, prefix: str):
        self.logged_values: Dict[str, List[Any]] = defaultdict(list)
        self.prefix = prefix

    def add_logs(self, *args: Dict[str, float]):
        for log in args:
            for k, v in log.items():
                self.logged_values[k].append(v)

    def dump_logs(self) -> Dict[str, float]:
        output: Dict[str, float] = {}
        for key, value in self.logged_values.items():
            output[f"{self.prefix}_{key}"] = aggregate_mean(value)
        return output
