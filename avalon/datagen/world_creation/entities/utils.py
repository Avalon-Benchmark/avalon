from typing import Union
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.worlds.height_map import HeightMap


def normalized(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def facing_2d(from_point: Point3DNP, to_point: Point3DNP) -> Rotation:
    dir = to_point - from_point
    dir[1] = 0
    dir = -normalized(dir)
    yaw = np.arctan2(dir[0], dir[2])
    return Rotation.from_euler("y", yaw)


def get_random_ground_points(
    rand: np.random.Generator,
    center: Point3DNP,
    count: int,
    map: HeightMap,
    min_radius: float,
    max_radius: float,
    offset: Union[float, np.ndarray],
    island_mask: MapBoolNP,
) -> Point3DListNP:
    assert count > 0
    acceptable_points = None
    scale = 1.0
    for i in range(10):
        radius = rand.uniform(min_radius, max_radius, size=count * 2 + 5)
        angle = rand.uniform(0, np.pi * 2, size=count * 2 + 5)
        x = np.cos(angle) * radius + center[0]
        y = np.sin(angle) * radius + center[2]
        points = np.stack([x, y], axis=1)
        points = map.restrict_points_to_region(points)
        idx_x, idx_y = map.points_to_indices(points)
        points = points[island_mask[idx_x, idx_y]]
        heights = map.get_heights(points)
        points_3d = np.stack([points[:, 0], heights, points[:, 1]], axis=1)
        if acceptable_points is None:
            acceptable_points = points_3d
        else:
            acceptable_points = np.concatenate([acceptable_points, points_3d], axis=0)
        if acceptable_points.shape[0] >= count:
            acceptable_points = acceptable_points[:count, :]
            acceptable_points[:, 1] += scale * offset
            return cast(Point3DListNP, acceptable_points)
    raise ImpossibleWorldError("Unable to create enough random points. Likely in the water.")
