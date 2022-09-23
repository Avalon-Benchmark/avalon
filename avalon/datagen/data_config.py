import json
from hashlib import md5
from numbers import Number
from typing import Generic
from typing import Tuple
from typing import TypeVar

from avalon.common.utils import find_root_dir
from avalon.common.utils import hash_dirpath
from avalon.contrib.serialization import Serializable

RangeT = TypeVar("RangeT", bound=Number)


class AbstractRange(Generic[RangeT], Serializable):
    min_ge: RangeT
    max_lt: RangeT

    @property
    def size(self) -> RangeT:
        return self.max_lt - self.min_ge  # type: ignore


DataT = TypeVar("DataT", bound=Serializable)


class AbstractDataConfig(Serializable):
    def get_is_using_shared_caches(self) -> bool:
        raise NotImplementedError()

    def set_is_using_shared_caches(self: DataT, is_using_shared_caches: bool) -> DataT:
        raise NotImplementedError()

    def get_random_key(self) -> str:
        raise NotImplementedError()

    def get_dir_root(self) -> str:
        raise NotImplementedError()

    def set_dir_root(self: DataT, dir_root: str) -> DataT:
        raise NotImplementedError()

    def get_video_range(self) -> AbstractRange:
        raise NotImplementedError()

    def get_features(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    def get_frame_range(self) -> AbstractRange:
        raise NotImplementedError()

    def create_data_generation_request(
        self: DataT, video_range: AbstractRange, random_key: str, dir_root: str
    ) -> DataT:
        raise NotImplementedError()

    def get_output_file_name_format(self) -> str:
        raise NotImplementedError()

    def get_is_generating_paired_videos(self) -> bool:
        raise NotImplementedError()


def get_datagen_hash(config: Serializable) -> str:
    # TODO: make sure there's no extra usages, since the Godot binary path is different for old datagen
    hashed_config = md5(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()
    hashed_godot_binary = md5(open("/usr/bin/godot-old", "rb").read()).hexdigest()
    hashed_godot_dir = hash_dirpath(find_root_dir() + "/datagen/godot/**/*")
    return md5((hashed_godot_dir + hashed_config + hashed_godot_binary).encode("utf-8")).hexdigest()
