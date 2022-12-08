from random import Random
from typing import Dict
from typing import Generic
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import attr

from avalon.contrib.serialization import Serializable
from avalon.datagen.data_config import AbstractDataConfig
from avalon.datagen.data_config import AbstractRange


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class IntRange(AbstractRange):
    min_ge: int
    max_lt: int

    @staticmethod
    def n(n: int) -> "IntRange":
        return IntRange(0, n)

    @property
    def size(self) -> int:
        return self.max_lt - self.min_ge

    @property
    def midpoint(self) -> float:
        return self.min_ge + self.size / 2.0

    def contains(self, value: float) -> bool:
        return self.min_ge <= value < self.max_lt

    def overlap(self, other: "IntRange") -> Optional["IntRange"]:
        min_ge = max(self.min_ge, other.min_ge)
        max_lt = min(self.max_lt, other.max_lt)
        if min_ge >= max_lt:
            return None
        return IntRange(min_ge, max_lt)


# TODO: this was missing, add it and verify nothing breaks
# @attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class DataConfigImplementation(AbstractDataConfig):
    video_min: int
    video_max: int
    frame_max: int
    dir_root: str
    random_key: str

    output_file_name_format: str
    is_using_shared_caches: bool
    is_generating_paired_videos: bool

    def get_is_using_shared_caches(self) -> bool:
        return self.is_using_shared_caches

    def set_is_using_shared_caches(
        self: "DataConfigImplementation", is_using_shared_caches: bool
    ) -> "DataConfigImplementation":
        with self.mutable_clone() as updated_copy:
            updated_copy.is_using_shared_caches = is_using_shared_caches
            return updated_copy

    def get_random_key(self) -> str:
        return self.random_key

    def get_dir_root(self) -> str:
        return self.dir_root

    def set_dir_root(self: "DataConfigImplementation", dir_root: str) -> "DataConfigImplementation":
        with self.mutable_clone() as updated_copy:
            updated_copy.dir_root = dir_root
            return updated_copy

    def get_output_file_name_format(self) -> str:
        return self.output_file_name_format

    def get_is_generating_paired_videos(self) -> bool:
        return self.is_generating_paired_videos

    def get_video_range(self) -> AbstractRange:
        return IntRange(self.video_min, self.video_max)

    def get_frame_range(self) -> AbstractRange:
        return IntRange(0, self.frame_max)

    def create_data_generation_request(
        self: "DataConfigImplementation", video_range: AbstractRange, random_key: str, dir_root: str
    ) -> "DataConfigImplementation":
        with self.mutable_clone() as new_config:
            new_config.video_min = video_range.min_ge
            new_config.video_max = video_range.max_lt
            new_config.random_key = random_key
            new_config.dir_root = dir_root
            return new_config

    # TODO: I wish data config were not expected to know which features would be generated...
    def get_features(self) -> Tuple[str, ...]:
        raise Exception("New data configs do not know about features")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Vector2(Serializable):
    x: float
    y: float

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: int) -> float:
        if item == 0:
            return self.x
        if item == 1:
            return self.y
        raise IndexError(f"Vector2 only has 2 items, cannot access index {item}")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Vector3(Serializable):
    x: float
    y: float
    z: float

    def __len__(self) -> int:
        return 3

    def __getitem__(self, item: int) -> float:
        if item == 0:
            return self.x
        if item == 1:
            return self.y
        if item == 2:
            return self.z
        raise IndexError(f"Vector2 only has 3 items, cannot access index {item}")


AnyVec2 = Union[Vector2, float, int]
AnyVec3 = Union[Vector3, float, int]

T = TypeVar("T")
FloatT = TypeVar("FloatT", AnyVec2, AnyVec3, float)


class BaseDistribution(Serializable, Generic[T]):
    def sample(self, rand: Random) -> T:
        raise NotImplementedError()


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True, init=True)
class ChoicesDistribution(BaseDistribution[T]):
    value_choice: Tuple[T, ...]
    value_chance: Tuple[float, ...]

    @staticmethod
    def create_from_tuple(
        value_choice: Tuple[T, ...],
    ):
        return ChoicesDistribution(
            value_choice=tuple(value_choice),
            value_chance=(1.0,) * len(value_choice),
        )

    @staticmethod
    def create_from_dict(
        value_choice: Dict[T, float],
    ):
        return ChoicesDistribution(
            value_choice=tuple(value_choice),
            value_chance=tuple(value_choice.values()),
        )

    def sample(self, rand: Random) -> T:
        return rand.choices(self.value_choice, self.value_chance)[0]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True, init=False)
class ConstDistribution(BaseDistribution[T]):
    value: T

    def __init__(self, value: T) -> None:
        self.value = value

    def sample(self, rand: Random) -> T:
        return self.value


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True, init=False)
class GaussianDistribution(BaseDistribution[FloatT]):
    value_mean: FloatT
    value_std: FloatT

    def __init__(self, value_mean: FloatT, value_std: FloatT) -> None:
        self.value_mean = value_mean
        self.value_std = value_std

    def sample(self, rand: Random) -> FloatT:
        if isinstance(self.value_mean, float):
            assert isinstance(self.value_std, float)
            return rand.gauss(self.value_mean, self.value_std)
        else:
            assert hasattr(self.value_mean, "__len__")
            assert not isinstance(self.value_std, float)
            values = []
            for i in range(0, len(self.value_mean)):
                value = rand.gauss(self.value_mean[i], self.value_std[i])
                values.append(value)
            return (self.value_mean.__class__)(*values)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True, init=False)
class BernoulliDistribution(BaseDistribution[bool]):
    probability: float
    value: bool

    def __init__(self, probability: float, value: bool = True) -> None:
        self.probability = probability
        self.value = value


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True, init=False)
class UniformDistribution(BaseDistribution[FloatT]):
    value_min: FloatT
    value_max: FloatT

    def __init__(self, value_min: FloatT, value_max: FloatT) -> None:
        self.value_min = value_min
        self.value_max = value_max

    def sample(self, rand: Random) -> FloatT:
        if isinstance(self.value_min, (bool, str)):
            raise Exception("Cannot use Uniform with bools or strings, that is nonsense")
        elif isinstance(self.value_min, (float, int)):
            assert isinstance(self.value_max, (float, int))
            value = (rand.random() * (self.value_max - self.value_min)) + self.value_min
            if isinstance(self.value_min, int):
                return round(value)
            return value
        else:
            assert hasattr(self.value_min, "__len__")
            assert not isinstance(self.value_max, float)
            values = []
            for i in range(0, len(self.value_min)):
                value = (rand.random() * (self.value_max[i] - self.value_min[i])) + self.value_min[i]
                values.append(value)
            return (self.value_min.__class__)(*values)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class FloatRange(IntRange):
    min_ge: float  # type: ignore
    max_lt: float  # type: ignore

    def overlap(self, other: "FloatRange") -> Optional["FloatRange"]:  # type: ignore
        min_ge = max(self.min_ge, other.min_ge)
        max_lt = min(self.max_lt, other.max_lt)
        if min_ge >= max_lt:
            return None
        return FloatRange(min_ge, max_lt)
