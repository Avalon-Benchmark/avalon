import os
import re
from typing import Any
from typing import List

import pytorch_lightning
import torch
import wandb
from loguru import logger
from torch import Tensor

WANDB_ORGANIZATION = "sourceress"

WANDB_TEST_PROJECT = "test"


class RemoteWandbFileNotFound(Exception):
    pass


class RunDoesNotExist(Exception):
    pass


class ProjectDoesNotExist(Exception):
    pass


def add_scalar(trainer: Any, tag: str, value: Tensor, freq: int = 20):
    i = trainer.global_step
    if i % freq != 0:
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    trainer.logger.experiment.log({tag: value, "epoch": trainer.current_epoch}, commit=False)


def get_wandb_file_by_partial_name(run: wandb.apis.public.Run, file_name: str) -> wandb.apis.public.File:
    name_regex = re.compile(pattern=file_name)
    files: List[wandb.apis.public.File] = [f for f in run.files() if name_regex.search(f.name) is not None]
    if len(files) == 0:
        raise RemoteWandbFileNotFound(f"File that matches `{file_name}` does not exist.")
    if len(files) != 1:
        raise Exception(f"Multiple files found matching {file_name}.")
    return files[0]


def get_wandb_run_by_name(project: str, experiment_name: str) -> wandb.apis.public.Run:
    api = wandb.Api()

    if project not in [p.name for p in api.projects()]:
        raise ProjectDoesNotExist()

    # this will get the most recent run, this could be a potential issue if you have experiments with the same name
    runs: wandb.apis.public.Runs = api.runs(
        path=f"{WANDB_ORGANIZATION}/{project}", filters={"displayName": experiment_name}
    )
    if len(runs) == 0:
        raise RunDoesNotExist(f"Could not find run `{experiment_name}` in project `{project}`.")
    if len(runs) != 1:
        logger.warning(
            f"INFO: Multiple runs found with the same experiment name {experiment_name}. Using the most recent run."
        )
    run: wandb.apis.public.Run = runs[0]
    return run


def get_file_from_wandb(project: str, experiment_name: str, file_name: str) -> wandb.apis.public.File:
    run = get_wandb_run_by_name(project, experiment_name)
    return get_wandb_file_by_partial_name(run, file_name)


def download_wandb_checkpoint_from_run(project: str, experiment_name: str, checkpoint_name: str) -> str:
    file = get_file_from_wandb(project=project, experiment_name=experiment_name, file_name=checkpoint_name)
    path = os.path.join(".", file.name)
    wandb.util.download_file_from_url(path, file.url, wandb.Api().api_key)
    return path


def wandb_ensure_api_key():
    assert os.getenv("WANDB_API_KEY"), "WANDB_API_KEY not defined, make sure it's included in your bashenv.sh!"


def log_histogram(trainer: pytorch_lightning.Trainer, tag: str, value: Tensor, freq: int = 20):
    if trainer is None or trainer.logger is None or trainer.logger.experiment is None:
        return
    if not trainer.is_global_zero:
        return
    if trainer.global_step % (freq) != 0:
        return
    if type(value) == torch.Tensor:
        value = value.cpu().detach()
    trainer.logger.experiment.log({tag: value}, step=trainer.global_step)
