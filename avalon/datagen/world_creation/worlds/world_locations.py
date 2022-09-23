from typing import cast

import attr
import numpy as np

from avalon.common.utils import to_immutable_array
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import to_2d_point


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WorldLocations:
    island: MapBoolNP = attr.ib(converter=to_immutable_array)
    spawn: Point3DNP = attr.ib(converter=to_immutable_array)
    goal: Point3DNP = attr.ib(converter=to_immutable_array)

    def get_2d_spawn_goal_distance(self) -> float:
        return cast(float, np.linalg.norm(to_2d_point(self.spawn) - to_2d_point(self.goal)))
