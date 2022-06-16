# DEAD_CODE: this entire file is crap ...
import os
import shutil
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List

import torch
import wandb
from IPython import display
from pytorch_lightning import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor

from common.imports import np
from common.log_utils import logger
from common.type_utils import WandbCurrentRun
from common.utils import TMP_DATA_DIR
from common.visual_utils import get_path_to_video_png
from common.visual_utils import get_tensor_as_video
from common.visual_utils import make_video_grid
from contrib.utils import FILESYSTEM_ROOT

# from wandb.sdk.data_types import JSONMetadata


_SURPRISE_METADATA_PARTIAL_NAME = "surprise_data_test"


def _flatten_dict_of_arrays(lst: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    grouped = defaultdict(list)
    for out in lst:
        for k, v in out.items():
            grouped[k].append(v)
    return {k: np.concatenate(v) for k, v in grouped.items()}


def _tensor_dict_to_numpy(tensor_dict: Dict[str, Tensor]):
    return {k: v.cpu().numpy() for k, v in tensor_dict.items() if isinstance(v, Tensor)}


def log_image_or_video_to_wandb(
    run: WandbCurrentRun,
    input: Tensor,
    name: str,
    caption: str = "",
    file_dir: str = TMP_DATA_DIR,
    commit: bool = False,
    normalize: bool = True,
) -> str:
    video_png_path = get_path_to_video_png(input, file_dir=file_dir, normalize=normalize)
    run.log(
        {name: wandb.Image(video_png_path, caption=caption)},
        commit=commit,
    )
    return video_png_path


def is_wandb_logger_available_in_trainer(trainer: Trainer) -> bool:
    return bool(trainer.logger) and not trainer.running_sanity_check and isinstance(trainer.logger, WandbLogger)


def is_wandb_logger_available_in_module(module: LightningModule) -> bool:
    return bool(module.logger) and isinstance(module.logger, WandbLogger)


class ModelCheckpointCallback(Callback):
    def __init__(self, checkpoint_name: str, save_dir: str = FILESYSTEM_ROOT, interval: int = 1):
        self.checkpoint_name = checkpoint_name
        self.save_dir = save_dir
        self.epoch = 0
        self.interval = interval

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any):
        if is_wandb_logger_available_in_trainer(trainer):
            if self.epoch == 0 or (self.epoch + 1) % self.interval == 0:
                # used to be this, but that doesn't type check: run = wandb.run
                run = trainer.logger.experiment
                logger.info(f"INFO: Saving model checkpoint of last epoch to {run.dir}")
                # base_path = os.path.join(self.save_dir, "checkpoints")
                path = os.path.join(run.dir, f"{self.checkpoint_name}__{self.epoch}.ckpt")
                trainer.save_checkpoint(path)
                # wandb.save(path, base_path=base_path)
            self.epoch += 1


class LastEpochModelCheckpoint(Callback):
    def __init__(self, checkpoint_name: str, save_dir: str = FILESYSTEM_ROOT):
        self.checkpoint_name = checkpoint_name
        self.save_dir = save_dir

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: Any):
        if is_wandb_logger_available_in_trainer(trainer):
            logger.info(f"INFO: Saving model checkpoint of last epoch to {self.save_dir}")
            path = os.path.join(self.save_dir, "checkpoints", self.checkpoint_name)
            trainer.save_checkpoint(path)
            wandb.save(path)


class CleanUpTempDataDir(Callback):
    """ the temporary date directory is used to store all the images that we will upload to wandb """

    # note: this has to be on init start as it runs before train and test, most other callbacks run at the start
    #  of test or fit
    def on_init_start(self, trainer: Trainer):
        logger.info(f"INFO: Removing {TMP_DATA_DIR} and creating an empty directory.")
        if os.path.exists(TMP_DATA_DIR):
            shutil.rmtree(TMP_DATA_DIR)
        os.makedirs(TMP_DATA_DIR)


class IPythonVideoGridCallback(Callback):
    def __init__(
        self,
        num_conditioning_frames: int = 40,
        is_display_enabled: bool = True,
    ):
        super().__init__()
        self.is_display_enabled = is_display_enabled
        self.display_handle = None
        self.num_conditioning_frames = num_conditioning_frames

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        display.clear_output(wait=True)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        batch = next(iter(trainer.val_dataloaders[0]))
        batch = [x.to(pl_module.device) for x in batch]
        videos = pl_module.generate_video(batch, num_conditioning_frames=self.num_conditioning_frames)  # type: ignore
        videos = torch.clip(videos, min=-1, max=1)
        if self.is_display_enabled:
            video = make_video_grid(videos, 4)
            s, c, h, w = video.shape
            display_video = get_tensor_as_video(video, (h, w))
            if self.display_handle is None:
                self.display_handle = display.display(display_video, display_id=True)
            else:
                self.display_handle.update(display_video)
