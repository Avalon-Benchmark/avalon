import math
from typing import Optional
from typing import Tuple

import attr
import numpy as np

from avalon.datagen.world_creation.types import HeightMode
from avalon.datagen.world_creation.types import MapFloatNP
from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.worlds.obstacles.harmonics import EdgeConfig
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HeightObstacle:
    is_inside_ring: bool
    edge_config: EdgeConfig
    traversal_length: float
    # the traversal width, ie, as you are facing it, how wide (left to right) is the area that you can cross
    traversal_width: float
    # over how many multiples of traversal_width should we interpolate from "no noise" to "full noise"
    traversal_noise_interpolation_multiple: float
    is_default_climbable: bool
    detail_radius: float
    # not particularly useful... too easy for things to end up off the map
    traversal_theta_offset: float = 0.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RingObstacleConfig:
    height: float
    height_mode: HeightMode
    edge: EdgeConfig
    center_point: Point2DNP
    traversal_point: Point2DNP
    inner_safety_radius: float
    outer_safety_radius: float
    inner_obstacle: Optional[HeightObstacle] = None
    outer_obstacle: Optional[HeightObstacle] = None
    inner_solution: Optional[HeightSolution] = None
    outer_solution: Optional[HeightSolution] = None
    dual_solution: Optional[HeightSolution] = None
    chasm_bottom_size: float = 0.0
    expansion_meters: float = 0.0
    terrain_blurs: Tuple[Tuple[float, float], ...] = tuple()

    def get_inner_solution_points(self) -> Tuple[Point2DNP, Point2DNP]:
        obstacle = self.inner_obstacle
        assert obstacle is not None
        delta = self.traversal_point - self.center_point
        r = np.linalg.norm(delta)
        theta = math.atan2(delta[1], delta[0])
        theta += obstacle.traversal_theta_offset
        inner_r = r - max([obstacle.traversal_length, 0.01])
        outer_r = r
        inner_point = np.array([inner_r * math.cos(theta), inner_r * math.sin(theta)])
        outer_point = np.array([outer_r * math.cos(theta), outer_r * math.sin(theta)])
        return inner_point + self.center_point, outer_point + self.center_point

    def get_outer_solution_points(self) -> Tuple[Point2DNP, Point2DNP]:
        obstacle = self.outer_obstacle
        assert obstacle is not None
        delta = self.traversal_point - self.center_point
        r = np.linalg.norm(delta)
        theta = math.atan2(delta[1], delta[0])
        theta += obstacle.traversal_theta_offset
        inner_r = r + self.chasm_bottom_size
        outer_r = r + self.chasm_bottom_size + max([obstacle.traversal_length, 0.01])
        # logger.debug(inner_r)
        # logger.debug(outer_r)
        # logger.debug(obstacle.traversal_length)
        # logger.debug(self.chasm_bottom_size)
        inner_point = np.array([inner_r * math.cos(theta), inner_r * math.sin(theta)])
        outer_point = np.array([outer_r * math.cos(theta), outer_r * math.sin(theta)])
        return inner_point + self.center_point, outer_point + self.center_point


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class RingObstacle:
    config: RingObstacleConfig
    mid_z: float
    z: MapFloatNP
    r: MapFloatNP
    theta: MapFloatNP
    outer_mid_z: float
