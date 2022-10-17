from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

import numpy as np
import wandb
from IPython import display
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

TensorOrNDArray = Union["Tensor", NDArray]


def encode_video(
    frames: TensorOrNDArray,
    normalize: bool = False,
    fps: int = 10,
    video_format: str = "webm",
) -> str:
    """
    This is a stripped down version of wandb_lib.log_video. Can pull in more options from there as needed,
    like batching or normalization.
    Keeping them separate so we don't need Torch here.
    Expects an array of shape (t c h w), in uint range (0, 255) or float range (0, 1).
    I think wandb.Video works with a batch of videos natively, too.
    - `format` accepts "gif", "mp4", "webm" or "ogg"
    """
    if normalize:
        raise NotImplementedError

    if isinstance(frames, np.ndarray):
        if np.issubdtype(frames.dtype, np.floating):
            frames = (frames * 255).clip(0, 255).astype(np.uint8)
    else:
        # Assume Torch tensor
        import torch

        frames = frames.detach()
        if torch.is_floating_point(frames):
            frames = (frames * 255).clamp(0, 255).to(torch.uint8)
        frames = frames.cpu()

    return wandb.Video(frames, format=video_format, fps=fps)._path  # type: ignore


def visualize_arraylike_as_video(
    frames: TensorOrNDArray,
    normalize: bool = False,
    fps: int = 10,
    size: tuple[int, int] = (256, 256),
    video_format: str = "webm",
) -> None:
    """Render a video in a jupyter notebook. See `encode_video` for options.
    - `size`: size to display the video
    """
    video_path = encode_video(frames, normalize=normalize, fps=fps, video_format=video_format)
    display.display(
        display.Video(video_path, embed=True, width=size[0], height=size[1], html_attributes="controls autoplay")
    )
