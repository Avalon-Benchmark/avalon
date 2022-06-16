import math
from enum import Enum
from typing import Tuple
from typing import Union

import attr
import numpy as np

from datagen.godot_base_types import Vector2
from datagen.godot_base_types import Vector3
from datagen.world_creation.region import FloatRange


class Axis(Enum):
    X = "x"
    Y = "y"
    Z = "z"


# todo: just use Vector2?
@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Position:
    x: int
    z: int


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Geometry:
    x: float
    y: float
    z: float


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Plane(Geometry):
    width: float
    length: float
    pitch: float
    yaw: float
    roll: float


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class Box(Geometry):
    width: float
    length: float
    height: float
    pitch: float
    yaw: float
    roll: float


def global_to_local_coords(global_coords: np.ndarray, offset: Union[np.ndarray, Vector3]) -> np.ndarray:
    assert len(global_coords) == len(offset) == 3
    return global_coords - offset


def local_to_global_coords(local_coords: np.ndarray, offset: Union[np.ndarray, Vector3]) -> np.ndarray:
    assert len(local_coords) == len(offset) == 3
    return local_coords + offset


def euclidean_distance(position_a: Position, position_b: Position) -> float:
    return math.sqrt(abs(position_a.x - position_b.x) ** 2 + abs(position_a.z - position_b.z) ** 2)


def middle(position_a: Position, position_b: Position) -> Position:
    return Position(
        x=min(position_a.x, position_b.x) + abs(position_a.x - position_b.x) // 2,
        z=min(position_a.z, position_b.z) + abs(position_a.z - position_b.z) // 2,
    )


def midpoint(point_a: Vector3, point_b: Vector3) -> Vector3:
    return Vector3(
        min(point_a.x, point_b.x) + abs(point_a.x - point_b.x) / 2,
        min(point_a.y, point_b.y) + abs(point_a.y - point_b.y) / 2,
        min(point_a.z, point_b.z) + abs(point_a.z - point_b.z) / 2,
    )


def get_triangulation_edges(triangulation):
    edges = set()
    point_indices = ((0, 1), (0, 2), (1, 2))
    for simplex in triangulation.simplices:
        for start_idx, end_idx in point_indices:
            point1 = simplex[start_idx]
            point2 = simplex[end_idx]
            edge = (point1, point2) if point1 < point2 else (point2, point1)
            edges.add(edge)
    return edges


def rotate_around_origin(xy: Tuple[float, float], radians: float) -> Tuple[float, float]:
    """from: https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302"""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)
    return xx, yy


Point3 = Tuple[float, float, float]


def squares_overlap(centroid_a: Vector2, size_a: float, centroid_b: Vector2, size_b: float):
    ax = FloatRange(centroid_a.x - size_a / 2, centroid_a.x + size_a / 2)
    ay = FloatRange(centroid_a.y - size_a / 2, centroid_a.y + size_a / 2)
    bx = FloatRange(centroid_b.x - size_b / 2, centroid_b.x + size_b / 2)
    by = FloatRange(centroid_b.y - size_b / 2, centroid_b.y + size_b / 2)
    return ax.overlap(bx) and ay.overlap(by)
