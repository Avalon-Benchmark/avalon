import enum
from typing import Any
from typing import Callable
from typing import Protocol
from typing import Tuple
from typing import TypeVar
from typing import Union

import wandb
from torch import Tensor

T = TypeVar("T")


# used to indicate attrs class attributes that are actually required, but because of options, have to come later
def required():
    raise NotImplementedError()


class WorkerInfo(Protocol):
    # technically this is a long, but I don't know how to express that in python typing :(
    seed: int
    id: int


class DatasetTypeEnum(enum.Enum):
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    TRAIN = "TRAIN"


TensorTransformFunction = Callable[..., Any]
WandbCurrentRun = Union[wandb.sdk.wandb_run.Run, wandb.sdk.lib.RunDisabled]
WandbAPIRun = wandb.apis.public.Run


_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]


PAIRED_VIDEO_VARIANT_ID_LOOKUP = {"NOT_PAIRED": 0, "AA": 1, "BB": 2, "AB": 3, "BA": 4}


class SimpleModule(Protocol):
    def forward(self, x: Tensor) -> Tensor:
        ...

    def __call__(self, x: Tensor) -> Tensor:
        ...
