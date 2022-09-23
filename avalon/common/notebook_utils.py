import io
from typing import Any
from typing import Optional
from typing import Tuple

import PIL
import ipywidgets as widgets
from matplotlib import pyplot as plt
from torch import Tensor

from avalon.common.log_utils import enable_debug_logging
from avalon.common.visual_utils import safe_pil_image
from avalon.contrib.utils import make_deterministic


def configure_sensible_defaults(seed: int = 0) -> int:
    plt.style.use("seaborn-whitegrid")
    enable_debug_logging()
    make_deterministic(seed)
    return seed


def plot_to_image_widget(plot: Any, size: Optional[Tuple[int, int]] = None, svg: bool = False):
    if size is None:
        size = (450, 450)

    # TODO: 77a341ac-ab68-42e4-88e3-bff6fce06c44 - make sure we can save SVG plots to wandb ;_;
    if svg:
        save_format = "svg"
        read_format = "svg+xml"
    else:
        save_format = "png"
        read_format = "png"

    pic_buffer = io.BytesIO()
    plt.savefig(pic_buffer, format=save_format, bbox_inches="tight")
    plt.close()
    pic_buffer.seek(0)

    return widgets.Image(
        value=pic_buffer.getvalue(),
        format=read_format,
        width=size[0],
        height=size[1],
    )


def video_tensor_to_widget(
    data: Tensor, is_normalization_requred: bool, size: Tuple[int, int] = (450, 450)
) -> widgets.Image:
    # TASK 98c5c5c4-5d3b-4f7b-9a53-c156544c0063
    if is_normalization_requred:
        images = [safe_pil_image(data[t] * 0.5 + 0.5) for t in range(data.shape[0])]
    else:
        images = [safe_pil_image(data[t]) for t in range(data.shape[0])]
    pic_IObytes = io.BytesIO()
    images[0].save(
        pic_IObytes,
        format="png",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=40,
        loop=0,
    )
    pic_IObytes.seek(0)
    widget = widgets.Image(
        value=pic_IObytes.read(),
        format="png",
        width=size[0],
        height=size[1],
    )
    return widget


def widget_to_pil_image(image: widgets.Image) -> PIL.Image:
    buf = io.BytesIO()
    buf.write(image.value)
    buf.seek(0)
    return PIL.Image.open(buf)
