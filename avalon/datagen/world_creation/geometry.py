import math
from enum import Enum
from typing import Set
from typing import Tuple
from typing import cast

import attr
import numpy as np
from scipy.spatial import Delaunay

from avalon.datagen.godot_base_types import FloatRange
from avalon.datagen.godot_base_types import Vector2
from avalon.datagen.world_creation.types import Point3DNP


class Axis(Enum):
    """Values are loweracase for easy use in getattr/setattr on objects with xyz properties"""

    X = "x"
    Y = "y"
    Z = "z"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BuildingTile:
    x: int
    z: int


def global_to_local_coords(global_coords: Point3DNP, offset: Point3DNP) -> Point3DNP:
    assert len(global_coords) == len(offset) == 3
    return cast(Point3DNP, global_coords - offset)


def local_to_global_coords(local_coords: Point3DNP, offset: Point3DNP) -> Point3DNP:
    assert len(local_coords) == len(offset) == 3
    return cast(Point3DNP, local_coords + offset)


def euclidean_distance(position_a: BuildingTile, position_b: BuildingTile) -> float:
    return math.sqrt(abs(position_a.x - position_b.x) ** 2 + abs(position_a.z - position_b.z) ** 2)


def midpoint(point_a: Point3DNP, point_b: Point3DNP) -> Point3DNP:
    return cast(Point3DNP, np.min([point_a, point_b], axis=0) + abs(point_a - point_b) / 2)


def get_triangulation_edges(triangulation: Delaunay) -> Set[Tuple[np.ndarray, np.ndarray]]:
    edges = set()
    point_indices = ((0, 1), (0, 2), (1, 2))
    for simplex in triangulation.simplices:
        for start_idx, end_idx in point_indices:
            point1 = cast(np.ndarray, simplex[start_idx])
            point2 = cast(np.ndarray, simplex[end_idx])
            edge = (point1, point2) if point1 < point2 else (point2, point1)
            edges.add(edge)
    return edges


def squares_overlap(centroid_a: Vector2, size_a: float, centroid_b: Vector2, size_b: float) -> bool:
    ax = FloatRange(centroid_a.x - size_a / 2, centroid_a.x + size_a / 2)
    ay = FloatRange(centroid_a.y - size_a / 2, centroid_a.y + size_a / 2)
    bx = FloatRange(centroid_b.x - size_b / 2, centroid_b.x + size_b / 2)
    by = FloatRange(centroid_b.y - size_b / 2, centroid_b.y + size_b / 2)
    return bool(ax.overlap(bx) and ay.overlap(by))
