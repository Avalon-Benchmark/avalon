import attr
import numpy as np

from datagen.world_creation.heightmap import MapBoolNP
from datagen.world_creation.heightmap import Point2DNP
from datagen.world_creation.heightmap import Point3DNP


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class WorldLocationData:
    island: MapBoolNP
    spawn: Point3DNP
    goal: Point3DNP

    def get_2d_spawn_goal_distance(self) -> float:
        return np.linalg.norm(to_2d_point(self.spawn) - to_2d_point(self.goal))


def to_2d_point(point: Point3DNP) -> Point2DNP:
    return np.array([point[0], point[2]])
