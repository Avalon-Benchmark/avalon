import math
import os
import tempfile
from pathlib import Path
from typing import IO
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

import PIL
import matplotlib.pyplot as plt
import torch
from IPython.core.display import Image
from PIL import Image as PILImage
from PIL import ImageSequence
from ipywidgets import widgets
from matplotlib import animation
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image

from common.imports import np
from common.log_utils import logger
from common.utils import TMP_DATA_DIR
from common.utils import show

T = TypeVar("T")


def safe_pil_image(x: Tensor) -> PILImage:
    # TASK: 98c5c5c4-5d3b-4f7b-9a53-c156544c0063
    # Since this expects images tensors to be in the 0 - 1 range, most callers rescale them right before the call.
    # Seems saner to add an extra parameter and optionally renormalize in here?
    return to_pil_image(torch.clamp(x, min=0, max=1))


def make_video_grid(
    tensor: Tensor,
    num_images_per_row: int = 10,
    padding: int = 2,
    pad_value: int = 0,
) -> Tensor:
    """
    This is a repurposed implementation of `make_grid` from torchvision to work with videos.
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


def load_video_to_tensor(fp: Union[str, Path, IO[str], IO[bytes]]) -> Tensor:
    image = PIL.Image.open(fp)
    tensor = torch.from_numpy(np.stack([np.array(frame) for frame in ImageSequence.Iterator(image)]))
    return tensor


def get_path_to_surprise_graph_png(loss: Tensor, ylim: Optional[Tuple[int, int]] = None) -> str:
    fig, ax = plt.subplots()
    ax.set_xlabel("frame")
    ax.set_ylabel("Surprise")
    if ylim:
        ax.set_ylim(ylim)
    frame_count = loss.shape[0]
    frames = np.linspace(0, frame_count - 1, frame_count)
    if not isinstance(loss, np.ndarray):
        kl_loss = loss.cpu().detach().numpy()
    else:
        kl_loss = loss
    (line,) = ax.plot([0, frame_count], [0, kl_loss.max() * 1.1])
    animate = lambda i: line.set_data(frames[:i], kl_loss[:i])
    interval = 40  # the time to display the current frame
    ani = animation.FuncAnimation(
        fig, animate, frames=range(1, len(frames) + 1), interval=interval, repeat_delay=0, repeat=True
    )
    surprise_file = tempfile.NamedTemporaryFile(suffix=".png", dir=TMP_DATA_DIR, delete=False)
    ani.save(surprise_file.name, writer="pillow")
    plt.close()  # no reason to show an empty plot
    surprise_file.close()
    return surprise_file.name


def save_pil_images_as_video(images: List[PIL.Image.Image], file_dir: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png", dir=file_dir, delete=False) as f:
        images[0].save(
            f,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=40,
            compress_level=0,
            loop=0,
        )
        f.close()
        return f.name


def get_path_to_video_png(video: Tensor, file_dir: str = TMP_DATA_DIR, normalize: bool = True) -> str:
    os.makedirs(file_dir, exist_ok=True)
    if normalize:
        images = [safe_pil_image(v * 0.5 + 0.5) for v in video]
    else:
        images = [safe_pil_image(v) for v in video]
    return save_pil_images_as_video(images, file_dir)


def get_surprise_graph_as_np_array(loss: Tensor) -> np.ndarray:
    path_to_png = get_path_to_surprise_graph_png(loss)
    surprise_image = PILImage.open(path_to_png)
    animation_frames = []
    for frame in range(surprise_image.n_frames):
        surprise_image.seek(frame)
        animation_frames.append(np.transpose(np.asarray(surprise_image), (2, 0, 1)))
    return np.stack(animation_frames)


def get_combined_surprise_video(video: Tensor, surprise: Tensor, ylim: Optional[Tuple[int, int]] = None) -> Tensor:
    path_to_surprise_graph_png = get_path_to_surprise_graph_png(surprise, ylim=ylim)
    im = PIL.Image.open(path_to_surprise_graph_png)
    surprise_images = (
        torch.from_numpy(np.stack([np.array(frame) for frame in ImageSequence.Iterator(im)]))
        .permute(0, 3, 1, 2)
        .type(torch.float32)
        .div(255)
    )
    sequence_length, num_channels, height, width = surprise_images.shape
    new_image_dimension = min(height, width)
    upsampled_video = F.interpolate(video, size=(new_image_dimension, new_image_dimension)) * 0.5 + 0.5
    upsampled_video = torch.cat(
        [upsampled_video, torch.ones((sequence_length, 1, new_image_dimension, new_image_dimension))], dim=1
    )
    return torch.cat([surprise_images, upsampled_video], dim=-1)


def visualize_surprise_combined(video: Tensor, surprise: Tensor) -> None:
    combined_video = get_combined_surprise_video(video, surprise)
    visualize_tensor_as_video(combined_video, normalize=False)


def visualize_surprise_for_video(video: Tensor, surprise: Tensor) -> None:
    path_to_surprise_graph_png = get_path_to_surprise_graph_png(surprise)
    path_to_video_png = get_path_to_video_png(video)
    show(Image(path_to_surprise_graph_png))
    show(Image(path_to_video_png))


def visualize_surprise_graphs(
    data: Tensor,
    labels: Tensor,
    surprises: Tensor,
) -> None:
    for video, label, surprise in zip(data, labels, surprises):
        logger.info(f"label: {label.item()}")
        visualize_surprise_for_video(video, surprise)


def visualize_tensor_as_video(
    x: Tensor,
    file_format: str = "png",
    normalize: bool = True,
    size: Optional[Tuple[int, int]] = None,
    duration: int = 40,
):
    if normalize:
        images = [safe_pil_image(x[t] * 0.5 + 0.5) for t in range(x.shape[0])]
    else:
        images = [safe_pil_image(x[t]) for t in range(x.shape[0])]

    with tempfile.NamedTemporaryFile(suffix=f".{file_format}") as fp:
        images[0].save(
            fp,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=duration,
            loop=0,
        )
        width = None
        height = None
        if size is not None:
            width, height = size
        show(Image(fp.name, width=width, height=height), display_id=True)


def visualize_tensor_as_image(
    x: Tensor, file_format: str = "png", normalize: bool = True, size: Optional[Tuple[int, int]] = None
):
    if normalize:
        image = safe_pil_image(x * 0.5 + 0.5)
    else:
        image = safe_pil_image(x)

    with tempfile.NamedTemporaryFile(suffix=f".{file_format}") as fp:
        image.save(fp)
        width = None
        height = None
        if size is not None:
            width, height = size
        show(Image(fp.name, width=width, height=height), display_id=True)


def get_tensor_as_video(x: Tensor, resize: Tuple[int, int] = (256, 256), normalize: bool = True) -> Image:
    h, w = resize
    if normalize:
        images = [
            safe_pil_image(x[t] * 0.5 + 0.5).resize((w, h), resample=PIL.Image.NEAREST) for t in range(x.shape[0])
        ]
    else:
        images = [safe_pil_image(x[t]).resize((w, h), resample=PIL.Image.NEAREST) for t in range(x.shape[0])]

    with tempfile.NamedTemporaryFile(suffix=f".png") as fp:
        images[0].save(
            fp,
            save_all=True,
            append_images=images[1:],
            optimize=False,
            duration=100,
            loop=0,
        )
        return Image(fp.name)


def get_surprise_combined(video: Tensor, surprise: Tensor) -> Image:
    path_to_surprise_graph_png = get_path_to_surprise_graph_png(surprise)
    im = PIL.Image.open(path_to_surprise_graph_png)
    surprise_images = (
        torch.from_numpy(np.stack([np.array(frame) for frame in ImageSequence.Iterator(im)]))
        .permute(0, 3, 1, 2)
        .type(torch.float32)
        .div(255)
    )
    sequence_length, num_channels, height, width = surprise_images.shape
    new_image_dimension = min(height, width)
    upsampled_video = F.interpolate(video, size=(new_image_dimension, new_image_dimension)) * 0.5 + 0.5
    upsampled_video = torch.cat(
        [upsampled_video, torch.ones((sequence_length, 1, new_image_dimension, new_image_dimension))], dim=1
    )
    combined_video = torch.cat([surprise_images, upsampled_video], dim=-1)
    return get_tensor_as_video(combined_video, resize=(256, 512), normalize=False)


def widget_compare_videos_in_dataset(
    ds: Dataset,
    labels: np.ndarray,
    video_title_func: Callable[[int, int], str],
    video_getter_func: Callable[[Dataset, int], Tensor],
):
    """
    Very hacky function to display two images at the same time and update the images with a dropdown
    """
    options = [(f"{i} (label={label})", i) for i, label in zip(list(range(len(labels))), labels)]
    video1_idx = options[0][1]
    video2_idx = options[-1][1]

    dropdown1 = widgets.Dropdown(
        options=options,
        value=video1_idx,
        description="Video 1:",
        disabled=False,
    )
    dropdown2 = widgets.Dropdown(
        options=options,
        value=video2_idx,
        description="Video 2:",
        disabled=False,
    )

    video1 = get_tensor_as_video(video_getter_func(ds, video1_idx))
    video2 = get_tensor_as_video(video_getter_func(ds, video2_idx))

    video1_title = show(video_title_func(1, labels[video1_idx]), display_id=True)
    video1_display = show(video1, display_id=True)
    video2_title = show(video_title_func(2, labels[video2_idx]), display_id=True)
    video2_display = show(video2, display_id=True)

    def on_change1(change: Any):
        if change["type"] == "change" and change["name"] == "value":
            video1_idx = change["new"]
            title = video_title_func(1, labels[video1_idx])
            video1_title.update(title)
            video1 = get_tensor_as_video(video_getter_func(ds, video1_idx))
            video1_display.update(video1)

    def on_change2(change: Any):
        if change["type"] == "change" and change["name"] == "value":
            video2_idx = change["new"]
            title = video_title_func(2, labels[video2_idx])
            video2_title.update(title)
            video2 = get_tensor_as_video(video_getter_func(ds, video2_idx))
            video2_display.update(video2)

    dropdown1.observe(on_change1)
    dropdown2.observe(on_change2)
    show([dropdown1, dropdown2])


def widget_view_videos_in_dataset(
    ds: Dataset,
    labels: np.ndarray,
    video_title_func: Callable[[int, int], str],
    video_getter_func: Callable[[Dataset, int], Tensor],
    surprise_getter_func: Callable,
):
    """
    Very hacky function to display images in a dataset.
    """
    size = len(labels)
    state = {"index": 0}

    prev_button = widgets.Button(
        description="Previous Video",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click me",
        icon="check",
    )
    next_button = widgets.Button(
        description="Next Video",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Click me",
        icon="check",
    )

    video = get_surprise_combined(
        video=video_getter_func(ds, state["index"]), surprise=surprise_getter_func(state["index"])
    )
    video_title = show(video_title_func(state["index"], labels[state["index"]]), display_id=True)
    video_display = show(video, display_id=True)

    def _update_display(index: int):
        video = get_surprise_combined(video=video_getter_func(ds, index), surprise=surprise_getter_func(index))
        video_title.update(f"Video {index} (label={labels[index]})")
        video_display.update(video)

    def on_prev_button_clicked(_: Any):
        if state["index"] == 0:
            state["index"] = size - 1
        else:
            state["index"] -= 1
        _update_display(state["index"])

    def on_next_button_clicked(_: Any):
        if state["index"] == size - 1:
            state["index"] = 0
        else:
            state["index"] += 1
        _update_display(state["index"])

    prev_button.on_click(on_prev_button_clicked)
    next_button.on_click(on_next_button_clicked)
    show(prev_button)
    show(next_button)


def get_indices_at_thresholds(values: np.ndarray, threshold: np.ndarray):
    """
    Note: skips the first and last two thresholds as these are required for plotting the ROC curve but may
    not be meaningful to us
    """
    near_indices = []
    t = list(np.flip(threshold))[1:-2]
    for i in range(len(values)):
        v = values[i]
        if v == t[0]:
            t.pop(0)
            near_indices.append(i)
        if len(t) == 0:
            break
    return list(set(near_indices))


def get_worst_values_by_labels(
    all_loss_values: np.ndarray, all_labels: np.ndarray, selected_labels: Set[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition = [bool(label in selected_labels) for label in all_labels]
    indices_for_labels = np.squeeze(np.argwhere(condition))
    subset_labels = all_labels[indices_for_labels]
    # TODO not sure how to fix the typing here
    subset_data = all_loss_values[indices_for_labels]
    sorted_indices = np.argsort(subset_data)
    data_indices = np.arange(0, len(all_labels))[indices_for_labels][sorted_indices]
    return (
        subset_data[sorted_indices],
        subset_labels[sorted_indices],
        data_indices,
    )


def convert_dataset_to_tensor(ds: Dataset, length: int) -> Tensor:
    return torch.stack([ds[i] for i in range(length)], dim=0)
