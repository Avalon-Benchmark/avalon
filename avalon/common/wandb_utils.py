import os
import re
from pathlib import Path
from typing import List
from typing import Optional

import wandb
from loguru import logger
from wandb.apis.public import Run
from wandb.apis.public import Runs

EVAL_TEMP_PATH = "/tmp/avalon_eval"

WANDB_ORGANIZATION = "sourceress"

WANDB_TEST_PROJECT = "test"


class RemoteWandbFileNotFound(Exception):
    pass


class RunDoesNotExist(Exception):
    pass


class ProjectDoesNotExist(Exception):
    pass


def get_wandb_file_by_partial_name(run: Run, file_name: str) -> wandb.apis.public.File:
    name_regex = re.compile(pattern=file_name)
    files: List[wandb.apis.public.File] = [f for f in run.files() if name_regex.search(f.name) is not None]
    if len(files) == 0:
        raise RemoteWandbFileNotFound(f"File that matches `{file_name}` does not exist.")
    if len(files) != 1:
        raise Exception(f"Multiple files found matching {file_name}.")
    return files[0]


def get_wandb_runs_by_tag(project: str, tag: str) -> list[Run]:
    api = wandb.Api()

    if project not in [p.name for p in api.projects()]:
        raise ProjectDoesNotExist()

    runs: Runs = api.runs(path=f"{WANDB_ORGANIZATION}/{project}", filters={"tags": {"$in": [tag]}})
    return list(runs)


def get_wandb_run_by_name(project: str, experiment_name: str) -> Run:
    api = wandb.Api()

    if project not in [p.name for p in api.projects()]:
        raise ProjectDoesNotExist()

    # this will get the most recent run, this could be a potential issue if you have experiments with the same name
    runs: Runs = api.runs(path=f"{WANDB_ORGANIZATION}/{project}", filters={"displayName": experiment_name})
    if len(runs) == 0:
        raise RunDoesNotExist(f"Could not find run `{experiment_name}` in project `{project}`.")
    if len(runs) != 1:
        logger.warning(
            f"INFO: Multiple runs found with the same experiment name {experiment_name}. Using the most recent run."
        )
    run: Run = runs[0]
    return run


def get_wandb_run_by_suggestion_uuid(project: str, uuid: str) -> Run:
    """Set a configuration attribute `uuid` to a unique string, adn this function will fetch that run for you."""
    api = wandb.Api()

    if project not in [p.name for p in api.projects()]:
        raise ProjectDoesNotExist()

    # this will get the most recent run, this could be a potential issue if you have experiments with the same name
    runs: Runs = api.runs(path=f"{WANDB_ORGANIZATION}/{project}", filters={"config.suggestion_uuid": uuid})
    if len(runs) == 0:
        raise RunDoesNotExist(f"Could not find suggestion_uuid `{uuid}` in project `{project}`.")
    if len(runs) != 1:
        logger.warning(f"INFO: Multiple runs found with the same suggestion_uuid {uuid}. Using the most recent run.")
    run: Run = runs[0]
    return run


def get_file_from_wandb(project: str, experiment_name: str, file_name: str) -> wandb.apis.public.File:
    run = get_wandb_run_by_name(project, experiment_name)
    return get_wandb_file_by_partial_name(run, file_name)


def download_wandb_checkpoint_from_run(project: str, experiment_name: str, checkpoint_name: str) -> str:
    file = get_file_from_wandb(project=project, experiment_name=experiment_name, file_name=checkpoint_name)
    path = os.path.join(".", file.name)
    wandb.util.download_file_from_url(path, file.url, wandb.Api().api_key)
    return path


def wandb_ensure_api_key() -> None:
    assert os.getenv("WANDB_API_KEY"), "WANDB_API_KEY not defined, make sure it's included in your bashenv.sh!"


def load_checkpoint_from_wandb_run(run_path: str, filename: str) -> str:
    api = wandb.Api()
    run = api.run(run_path)
    run_root = Path(EVAL_TEMP_PATH) / run_path
    os.makedirs(run_root, exist_ok=True)
    bones_checkpoint_path = wandb.restore(filename, run_path=run_path, replace=True, root=str(run_root))
    assert bones_checkpoint_path is not None, "Could not load checkpoint"
    return bones_checkpoint_path.name


def get_latest_checkpoint_filename(run_path: str, prefix: str = "", suffix: str = "") -> Optional[str]:
    api = wandb.Api()
    run = api.run(run_path)
    checkpoint_filenames: List[str] = [
        file.name for file in run.files() if file.name.startswith(prefix) and file.name.endswith(suffix)
    ]
    checkpoint_filenames = sorted(checkpoint_filenames, key=lambda x: int(x[len(prefix) : -len(suffix)]))
    if len(checkpoint_filenames) == 0:
        return None

    return checkpoint_filenames[-1]
