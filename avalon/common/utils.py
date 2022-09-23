import glob
import hashlib
import itertools
import os
import random
from collections import OrderedDict
from collections import defaultdict
from enum import Enum
from hashlib import md5
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import KeysView
from typing import List
from typing import Optional
from typing import Protocol
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import ValuesView
from typing import cast

import ipyplot
import numpy as np
import torch
from IPython.core.display import display
from PIL import Image
from torch import Tensor
from torch.types import Device
from torch.utils.data.dataset import Dataset

from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import TEMP_DIR

TMP_DATA_DIR = os.path.join(TEMP_DIR, "wandb")


class _SupportsLessThan(Protocol):
    def __lt__(self, __other: Any) -> bool:
        ...


T = TypeVar("T")
TK = TypeVar("TK", bound=_SupportsLessThan)
TV = TypeVar("TV")


class EvaluationStage(Enum):
    TRAINING = "TRAIN"
    VALIDATION = "VAL"
    TEST = "TEST"
    GENERATION = "GEN"


def show(x: Any, **kwargs):  # type: ignore
    """This wrapper is just here so it shuts up about the type signature, it can display anything"""
    return display(x, **kwargs)


def create_random_image():
    import numpy
    from PIL import Image

    image_array = numpy.random.rand(512, 512, 3) * 255
    image = Image.fromarray(image_array.astype("uint8")).convert("RGBA")
    image.save("random_512.png")


def partition_dict(pred_fn: Callable, d: Dict) -> Tuple[Dict, Dict]:
    """
    Use a predicate to partition a dictionary into two mutually exclusive sub-dicts with values that tested False, True
    partition_dict(is_odd, {'a': 0, 'b': 1}) --> ({'a': 0}, {'b': 1})
    """
    return {k: v for k, v in d.items() if not pred_fn(k, v)}, {k: v for k, v in d.items() if pred_fn(k, v)}


def flatten(iterable: Iterable[Iterable[T]]) -> List[T]:
    return list(itertools.chain(*iterable))


def compact(x: Any) -> Any:
    return list(filter(None, x))


def check(x: torch.Tensor, expected_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    true_shape = x.shape
    if len(true_shape) != len(expected_shape):
        raise Exception(
            f"Tensor doesn't even have the right number of dimensions:\nExpected shape: {expected_shape}\nTrue shape:    {true_shape}"
        )
    for i in range(0, len(expected_shape)):
        if expected_shape[i] == -1:
            continue
        if expected_shape[i] != true_shape[i]:
            raise Exception(
                f"Tensor shape does not match at dimension={i}:\nExpected shape: {expected_shape}\nTrue shape:    {true_shape}"
            )
    return true_shape


def group_by_helper(data: Iterable[TV], get_key: Callable[[TV], TK]) -> Dict[TK, List[TV]]:
    data = sorted(data, key=get_key)
    return {k: list(g) for k, g in groupby(data, get_key)}


def first(iterator: Union[Iterable[T], Iterator[T]]) -> Optional[T]:
    if isinstance(iterator, (tuple, list, KeysView, ValuesView)):
        return next(iter(iterator), None)
    elif hasattr(iterator, "__next__"):
        return next(cast(Iterator[T], iterator), None)
    elif isinstance(iterator, np.ndarray):
        if len(iterator) == 0:
            return None
        return cast(T, iterator[0])
    elif iterator is None:
        raise Exception("Not a sequence, is None!")
    raise Exception("Ooops, handle this")


def only(x: Iterable[T]) -> T:
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


# TASK 98c5c5c4-5d3b-4f7b-9a53-c156544c0063 -- remove this extra complexity from transforming to the [-1,1] range and back again
# In particular, this class is *especially* bad because it is data dependent--if you have an image that happens to have no really
# low values, you will end up transforming differently than something else from the exact same set of data!
class ClampImage(object):
    def __call__(self, tensor: Tensor):
        tensor = tensor.clone()
        img_min = float(tensor.min())
        img_max = float(tensor.max())
        tensor.clamp_(min=img_min, max=img_max)
        tensor.add_(-img_min).div_(img_max - img_min + 1e-5)
        return tensor


def visualize_images_in_filepath_glob(filepath_glob: str, max_images: int = 100, shuffle: bool = True):
    paths = sorted(glob.glob(filepath_glob))
    if shuffle:
        random.shuffle(paths)
    images = [Image.open(x) for x in paths[:max_images]]
    ipyplot.plot_images(images, max_images=max_images, img_width=50)


def get_root_experiment_dir() -> Path:
    return Path(os.getcwd().split("/current")[0])


def get_path_to_current_dir() -> Path:
    return get_root_experiment_dir() / "current"


def get_relative_project_path() -> str:
    project_dir = find_root_dir()
    git_dir = find_git_dir()
    return str(Path(project_dir).relative_to(Path(git_dir)))


def find_root_dir() -> str:
    return find_dir_by_file_name("root_dir_marker.txt")


def find_git_dir() -> str:
    return find_dir_by_file_name(".gitignore")


def find_dir_by_file_name(target_file_name: str) -> str:
    filepath_parts = str(Path(__file__).absolute()).split("/")
    for dirname in filepath_parts[1:]:
        partial_dirpath = filepath_parts[: filepath_parts.index(dirname)] + [target_file_name]
        if os.path.exists("/".join(partial_dirpath)):
            return os.path.dirname("/".join(partial_dirpath))
    raise Exception(f"Can't find {target_file_name}")


def hash_dirpath(dirpath: str) -> str:
    files = sorted(glob.glob(dirpath, recursive=True))
    return md5(
        "".join([md5(open(x, "rb").read()).hexdigest() for x in files if os.path.isfile(x)]).encode("utf-8")
    ).hexdigest()


def get_dataset_as_batch(ds: Dataset, length: int, device: Optional[Device] = None) -> List[Tensor]:
    features = defaultdict(list)
    for index in range(length):
        # TODO this resets the numpy seed for moving mnist
        data = ds[index]
        for i, feature in enumerate(data):
            features[i].append(feature)
    batch = [torch.stack(features[i]) for i in range(len(features))]
    if device:
        batch = [x.to(device) for x in batch]
    return batch


def assert_shape(actual: Union[torch.Size, Tuple[int, ...]], expected: Tuple[int, ...], message: str = ""):
    assert actual == expected, f"Expected shape: {expected} but passed shape: {actual}. {message}"


def conv_transpose_out_size(
    in_size: int, stride: int, padding: int, kernel_size: int, out_padding: int, dilation: int = 1
) -> int:
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def conv_out_size(in_size: int, stride: int, padding: int, kernel_size: int) -> int:
    # note: this is missing dilation but we don't normally change it
    return int((in_size - kernel_size + 2 * padding) / stride + 1.0)


# so named because we append to this path later, so this is the dir root of dir roots
DATA_FOLDER = os.path.join(FILESYSTEM_ROOT, "data")


def _freeze_params(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = True


FIX_ENCODER_STRINGS = {"0": "conv", "1": "bn"}


def _fix_key(key: str, prefix: str) -> str:
    parts_without_prefix = key[len(prefix) + 1 :].split(".")
    if prefix == "encoder" and len(parts_without_prefix) > 2:
        layer_part = parts_without_prefix[2]
        if layer_part in FIX_ENCODER_STRINGS:
            parts_without_prefix[2] = FIX_ENCODER_STRINGS[layer_part]
    return ".".join(parts_without_prefix)


def load_module_from_checkpoint(
    model: torch.nn.Module, checkpoint: OrderedDict[str, Tensor], prefix: str, freeze: bool = True
):
    assert "state_dict" in checkpoint, "Must have state dict key to load module from"
    state_dict = checkpoint["state_dict"]
    assert isinstance(state_dict, OrderedDict)
    model_dict = OrderedDict((k[len(prefix) + 1 :], v) for k, v in state_dict.items() if k.split(".")[0] == prefix)
    model.load_state_dict(model_dict)
    if freeze:
        _freeze_params(model)


def get_cuda_device_if_available() -> str:
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        return f"cuda:{device_id}"
    return "cpu"


def float_to_str(f: float) -> str:
    # rstrip is needed to account for int cases
    return np.format_float_positional(f).rstrip(".").replace(".", "_")


def to_immutable_array(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def file_checksum(file_path: str) -> str:
    with open(file_path, "rb") as output_file:
        data = output_file.read()
        return hashlib.md5(data).hexdigest()


def dir_checksum(dir_path: Path) -> str:
    hash_md5 = hashlib.md5()
    for path in sorted(dir_path.glob("*")):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(2 ** 20), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()
