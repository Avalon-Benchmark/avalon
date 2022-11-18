"""Utilities for logging to experiment tracking tools."""
import math
import queue
import secrets
import time
from types import ModuleType
from typing import Optional
from typing import Union

import numpy as np
import torch
import wandb
from loguru import logger
from numpy import typing as npt
from torch import Tensor
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run


class TrackerBackend:
    def log_scalar(self, key, value, step):
        raise NotImplementedError()

    def log_histogram(self, key, histogram, step):
        raise NotImplementedError()

    def log_frames(self, key, frames, fps, step):
        raise NotImplementedError()

    def log_image(self, key, image, step):
        raise NotImplementedError()

    def log_table(self, key, column_names, rows, step):
        raise NotImplementedError()

    def watch_torch_model(self, model, log_freq):
        raise NotImplementedError()

    def download_file(self, file_identifier: str) -> str:
        raise NotImplementedError()

    def get_run_dir(self):
        return None

    def upload_file(self, filepath):
        raise NotImplementedError()


class Wandb(TrackerBackend):
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run

    def log_scalar(self, key, value, step):
        self.wandb_run.log({key: value}, step=step)

    def log_histogram(self, key, histogram, step):
        self.wandb_run.log({key: wandb.Histogram(histogram)}, step=step)

    def log_frames(self, key, frames, fps, step):
        # TODO: make this work for non-tensor
        # Wandb doesn't like 1-channel greyscale videos
        shape = list(frames.shape)
        if shape[1] == 1:
            shape[1] = 3
            frames = frames.expand(shape)
        self.wandb_run.log({key: wandb.Video(frames, fps=fps, format="webm")}, step=step)  # type: ignore

    def log_image(self, key, image, step):
        wandb.log({key: wandb.Image(image)}, step=step)

    def log_table(self, key, column_names, rows, step):
        table = wandb.Table(data=rows, columns=column_names)
        self.wandb_run.log({key: table}, step=step)

    def watch_torch_model(self, model, log_freq):
        self.wandb_run.watch(model, log="all", log_freq=log_freq, log_graph=False)

    def download_file(self, file_identifier: str) -> str:
        project_name, run_id, filename = file_identifier.split("/")
        api = wandb.Api()
        run = api.run(f"sourceress/{project_name}/{run_id}")
        # We save to a random directory to avoid contention issues if there's
        # potentially multiple processes downloading the same file at the same time.
        # TODO: clean these files up
        path = run.file(filename).download(replace=True, root=f"./data/{secrets.token_hex(10)}").name
        assert isinstance(path, str)
        return path

    def get_run_dir(self):
        return Path(self.wandb_run.dir)

    def upload_file(self, filepath):
        self.wandb_run.save(str(filepath), policy="now")


# TODO: instead of every call including the step (and thus having to know the step in all code),
# we should just update the step once per iteration with a special call.
# Why didn't I do it that way in the first place? Does that limit any capabilities?

class ExperimentTracker:
    def __init__(
        self,
        tracker_backend: TrackerBackend,
        log_backoff_point: int = 5000,
        log_backoff_factor: int = 20,
        scalar_freq: int = 5,
        media_freq: int = 100,
        hist_freq: int = 100,
    ):
        self.tracker_backend = tracker_backend
        self.log_backoff_point = log_backoff_point
        self.log_backoff_factor = log_backoff_factor
        # backoff does apply to scalars and media
        self.scalar_freq = scalar_freq
        self.media_freq = media_freq
        # backoff does not apply to hists
        self.hist_freq = hist_freq

        self.last_time: Optional[float] = None
        self.last_step = 0

    def effective_freq(self, step: int, freq: int) -> int:
        # Simple logging backoff logic.
        if step > self.log_backoff_point and freq != 1:
            freq *= self.log_backoff_factor
        return freq

    def check_log_interval(self, step: int, freq: int, backoff: bool = True) -> bool:
        if backoff:
            freq = self.effective_freq(step, freq)
        return step % freq == 0

    def log_from_queue(self, tracking_queue, prefix: str = ""):  # type: ignore
        """Allows logging from another process."""
        try:
            mode, key, value = tracking_queue.get_nowait()
            key = f"{prefix}{key}"
            if mode == "scalar":
                self.log_scalar(key, value, freq=1)
            elif mode == "histogram":
                self.log_histogram(key, value, mean_freq=1, hist_freq=1)
            elif mode == "video":
                self.log_video(key, value, freq=1)
            elif mode == "image":
                self.log_image(key, value, freq=1)
            else:
                logger.warning(f"unrecognized mode in log_from_pipe: {mode}, {key}")
        except queue.Empty:
            pass

    def log_scalar(
        self,
        tag: str,
        value: Union[int, float, Tensor],
        step: Optional[int] = None,
        freq: Optional[int] = None,
        wandb_run: Union[ModuleType, Run, RunDisabled] = wandb,
    ) -> None:
        freq = self.scalar_freq if freq is None else freq
        if step is None:
            step = self.last_step
        if not self.check_log_interval(step, freq):
            return
        if type(value) == torch.Tensor:
            value = value.cpu().detach()
        self.tracker_backend.log_scalar(tag, value, step)

    def log_histogram(
        self,
        tag: str,
        value: Union[list, npt.NDArray, Tensor],
        step: Optional[int] = None,
        mean_freq: Optional[int] = None,
        hist_freq: Optional[int] = None,
        log_mean: bool = True,
    ) -> None:
        mean_freq = self.scalar_freq if mean_freq is None else mean_freq
        hist_freq = self.hist_freq if hist_freq is None else hist_freq

        if step is None:
            step = self.last_step

        # Allow backoff to apply to the scalars
        if log_mean:
            try:
                # works on torch and tf
                mean = value.mean()  # type: ignore
            except AttributeError:
                # hopefully works on most everything else
                mean = np.mean(value)
            self.log_scalar(tag + "_mean", mean, step, mean_freq)
        # Backoff doesn't work well with hists, messes up the viz
        if not self.check_log_interval(step, hist_freq, backoff=False):
            return
        if type(value) == torch.Tensor:
            value = value.cpu().detach()
        self.tracker_backend.log_histogram(tag + "_hist", value, step)

    def log_video(
        self,
        tag: str,
        batch: Tensor,
        step: Optional[int] = None,
        freq: Optional[int] = None,
        normalize: bool = False,
        num_images_per_row: int = 4,
    ) -> None:
        freq = self.media_freq if freq is None else freq
        if step is None:
            step = self.last_step
        # Expects b, t, c, h, w
        # or t, c, h, w
        if not self.check_log_interval(step, freq):
            return

        if normalize:
            min_v = torch.min(batch)
            range_v = torch.max(batch) - min_v
            if range_v > 0:
                batch = (batch - min_v) / range_v
            else:
                batch = torch.zeros(batch.size())

        # batch = preprocess_batch(batch).permute(0, 2, 1, 3, 4) + .5
        if len(batch.shape) == 5:
            frames = make_video_grid(batch, num_images_per_row=num_images_per_row, pad_value=1)
        else:
            assert len(batch.shape) == 4
            frames = batch

        # This should be in range 0-1
        if type(frames) == torch.Tensor:
            frames = frames.detach()
            if torch.is_floating_point(frames):
                frames = (frames * 255).clamp(0, 255).to(torch.uint8)
            frames = frames.cpu()
        elif isinstance(frames, np.ndarray):
            if np.issubdtype(frames.dtype, np.floating):
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
        else:
            assert False

        self.tracker_backend.log_frames(tag, frames, fps=4, step=step)

    def log_image(self, tag: str, value: Tensor, step: Optional[int] = None, freq: Optional[int] = None) -> None:
        if step is None:
            step = self.last_step
        freq = self.media_freq if freq is None else freq
        if not self.check_log_interval(step, freq):
            return
        if type(value) == torch.Tensor:
            value = value.cpu().detach()
        self.tracker_backend.log_image(tag, value, step)

    def log_table(self, tag: str, value: Tensor, step: int, freq: Optional[int] = None):
        freq = self.media_freq if freq is None else freq
        # value should be a 1d tensor, in this current implementation. can add more columns in the future.
        if not self.check_log_interval(step, freq):
            return
        if type(value) == torch.Tensor:
            value = value.cpu().detach()
        columns = ["test"]
        rows = [[x] for x in value]
        self.tracker_backend.log_table(tag, columns, rows, step)

    def log_iteration_time(self, batch_size: int, step: int, freq: Optional[int] = None) -> None:
        """Batch size should be the number of samples processed per step of i."""
        freq = self.scalar_freq if freq is None else freq
        if not self.check_log_interval(step, freq):
            return

        if self.last_time is None:
            self.last_time = time.time()
            self.last_step = step
        else:
            if step == self.last_step:
                return
            dt = (time.time() - self.last_time) / (step - self.last_step)
            self.last_time = time.time()
            self.last_step = step
            self.log_scalar("timings/iterations-per-sec", 1 / dt, step, freq=1)
            self.log_scalar("timings/samples-per-sec", batch_size / dt, step, freq=1)

    def watch(self, model: torch.nn.Module, freq: Optional[int] = None) -> None:
        freq = self.hist_freq if freq is None else freq
        self.tracker_backend.watch_torch_model(model, freq)

    def download_file(self, file_identifier: str) -> str:
        return self.tracker_backend.download_file(file_identifier)

    def get_run_dir(self):
        return self.tracker_backend.get_run_dir()

    def upload_file(self, filepath):
        return self.tracker_backend.upload_file(filepath)


def make_video_grid(
    tensor: Tensor,
    num_images_per_row: int = 10,
    padding: int = 2,
    pad_value: int = 0,
) -> Tensor:
    """
    This is a repurposed implementation of `make_grid` from torchvision to work with videos.

    Expects shape (batch_size, timesteps, channels, h, w).
    """
    n_maps, sequence_length, num_channels, height, width = tensor.size()
    x_maps = min(num_images_per_row, n_maps)
    y_maps = int(math.ceil(float(n_maps) / x_maps))
    height, width = int(height + padding), int(width + padding)
    grid = tensor.new_full(
        (sequence_length, num_channels, height * y_maps + padding, width * x_maps + padding), pad_value
    )
    k = 0
    for y in range(y_maps):
        for x in range(x_maps):
            if k >= n_maps:
                break
            grid.narrow(2, y * height + padding, height - padding).narrow(
                3, x * width + padding, width - padding
            ).copy_(tensor[k])
            k += 1
    return grid
