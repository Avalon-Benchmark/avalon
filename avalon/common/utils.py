import glob
import hashlib
import itertools
import os
import time
from enum import Enum
from hashlib import md5
from itertools import groupby
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
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

import numpy as np
from IPython.display import display

from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import TEMP_DIR

TMP_DATA_DIR: Final = os.path.join(TEMP_DIR, "wandb")

AVALON_PACKAGE_DIR: Final = os.path.abspath(f"{__file__}/../..")


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


def create_random_image() -> None:
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


def conv_transpose_out_size(
    in_size: int, stride: int, padding: int, kernel_size: int, out_padding: int, dilation: int = 1
) -> int:
    return (in_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + out_padding + 1


def conv_out_size(in_size: int, stride: int, padding: int, kernel_size: int) -> int:
    # note: this is missing dilation but we don't normally change it
    return int((in_size - kernel_size + 2 * padding) / stride + 1.0)


DATA_FOLDER = os.path.join(FILESYSTEM_ROOT, "data")


FIX_ENCODER_STRINGS = {"0": "conv", "1": "bn"}


def _fix_key(key: str, prefix: str) -> str:
    parts_without_prefix = key[len(prefix) + 1 :].split(".")
    if prefix == "encoder" and len(parts_without_prefix) > 2:
        layer_part = parts_without_prefix[2]
        if layer_part in FIX_ENCODER_STRINGS:
            parts_without_prefix[2] = FIX_ENCODER_STRINGS[layer_part]
    return ".".join(parts_without_prefix)


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


def dir_checksum(dir_path: Path, glob: str = "*") -> str:
    hash_md5 = hashlib.md5()
    for path in sorted(dir_path.glob(glob)):
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def wait_until_true(
    callback: Callable[[], Optional[bool]],
    max_wait_sec: float = 5,
    sleep_inc: float = 0.25,
):
    """Repeatedly call callback() until it returns True or max_wait_sec is reached"""
    waited_for_sec = 0.0000
    while waited_for_sec <= max_wait_sec:
        if callback():
            return
        time.sleep(sleep_inc)
        waited_for_sec += sleep_inc
    raise TimeoutError(f"could not complete within {max_wait_sec} seconds")
