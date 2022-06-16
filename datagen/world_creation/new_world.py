# %%
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import attr
import numpy as np
from godot_parser import GDObject
from godot_parser import Node as GDNode
from godot_parser import Vector3 as GDVector3
from nptyping import assert_isinstance
from scipy import ndimage
from scipy import stats
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from skimage import morphology

from common.errors import SwitchError
from common.log_utils import logger
from common.utils import only
from datagen.godot_base_types import Vector3
from datagen.world_creation.biome_map import BiomeHeightMap
from datagen.world_creation.biome_map import signed_line_distance
from datagen.world_creation.constants import AGENT_HEIGHT
from datagen.world_creation.constants import DEFAULT_SAFETY_RADIUS
from datagen.world_creation.constants import FLORA_REMOVAL_METERS
from datagen.world_creation.constants import HALF_AGENT_HEIGHT_VECTOR
from datagen.world_creation.constants import METERS_OF_TREE_CLEARANCE_AROUND_LINE_OF_SIGHT
from datagen.world_creation.constants import UP_VECTOR
from datagen.world_creation.heightmap import WATER_LINE
from datagen.world_creation.heightmap import BiomeConfig
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.heightmap import FloraConfig
from datagen.world_creation.heightmap import HeightMap
from datagen.world_creation.heightmap import HeightMode
from datagen.world_creation.heightmap import MapBoolNP
from datagen.world_creation.heightmap import MapFloatNP
from datagen.world_creation.heightmap import Point2DNP
from datagen.world_creation.heightmap import Point3DListNP
from datagen.world_creation.heightmap import Point3DNP
from datagen.world_creation.heightmap import SpecialBiomes
from datagen.world_creation.heightmap import clamp
from datagen.world_creation.heightmap import get_flora_config_by_file
from datagen.world_creation.heightmap import perlin
from datagen.world_creation.heightmap import selected_distance_weighted_points
from datagen.world_creation.indoor.objects import Building
from datagen.world_creation.indoor.objects import Story
from datagen.world_creation.items import CANONICAL_FOOD
from datagen.world_creation.items import CANONICAL_FOOD_CLASS
from datagen.world_creation.items import Boulder
from datagen.world_creation.items import ColoredSphere
from datagen.world_creation.items import Entity
from datagen.world_creation.items import Food
from datagen.world_creation.items import Item
from datagen.world_creation.items import Log
from datagen.world_creation.items import Placeholder
from datagen.world_creation.items import Predator
from datagen.world_creation.items import Prey
from datagen.world_creation.items import Scenery
from datagen.world_creation.items import SpawnPoint
from datagen.world_creation.items import Stone
from datagen.world_creation.items import Tool
from datagen.world_creation.items import Weapon
from datagen.world_creation.new_godot_scene import ImprovedGodotScene
from datagen.world_creation.region import FloatRange
from datagen.world_creation.region import Region
from datagen.world_creation.task_generators import IdGenerator
from datagen.world_creation.tasks.constants import IS_ALL_SCENERY_IN_MAIN
from datagen.world_creation.tasks.constants import IS_DEBUGGING_IMPOSSIBLE_WORLDS
from datagen.world_creation.terrain import Terrain
from datagen.world_creation.terrain import _create_static_body
from datagen.world_creation.terrain import create_multimesh_instance
from datagen.world_creation.utils import WORLD_RAISE_AMOUNT
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.utils import plot_terrain
from datagen.world_creation.utils import plot_value_grid
from datagen.world_creation.utils import plot_value_grid_multi_marker
from datagen.world_creation.world import build_outdoor_world_map
from datagen.world_creation.world_config import BuildingConfig
from datagen.world_creation.world_config import WorldConfig
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point

EntityType = TypeVar("EntityType", bound=Entity)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HarmonicsConfig:
    harmonics: Tuple[int, ...]
    weights: Tuple[float, ...]
    # just used for debugging
    is_deterministic: bool = False


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class EdgeConfig:
    # edges are scaled based on their radius so that the size of the noise is constant
    # this factor changes that automatic scaling
    # a higher number will effectively scale up a smaller circle by that much
    scale: float = 1.0
    # noise of 0.0 means "no noise, smooth line", noise of 1.0 means "very noisy"
    # going much beyond 1.0 is probably a bad idea
    noise: float = 0.5
    # 1.0 means a perfect circle, 0.0 means weirdly blobby
    circularity: float = 0.5

    def to_harmonics(self, radius: float) -> HarmonicsConfig:
        noise = self.noise ** 0.5
        base_radius = 20.6
        base_harmonic_count = 20 * noise
        scale_factor = (radius / base_radius) * (1.0 / self.scale)
        noise_harmonics = list(range(1, round(base_harmonic_count * scale_factor ** 0.5)))
        scaled_harmonics = tuple([round(2 * scale_factor) + i for i in noise_harmonics])
        noise_weights = tuple([noise * 0.75 / x for x in scaled_harmonics])

        shape_factor = (1.0 - self.circularity) * 1.75
        shape_weights = tuple(x * shape_factor for x in [0.1, 0.3, 0.2, 0.1, 0.05])
        shape_harmonics = (1, 2, 3, 4, 5)

        return HarmonicsConfig(
            harmonics=scaled_harmonics + shape_harmonics,
            weights=noise_weights + shape_weights,
        )


def create_harmonics(
    rand: np.random.Generator, theta: MapFloatNP, config: HarmonicsConfig, is_normalized: bool
) -> MapFloatNP:
    variation = np.zeros_like(theta)
    for harmonic, weight in zip(config.harmonics, config.weights):
        # noise = rand.normal()
        noise = rand.uniform(-1, 1)
        if config.is_deterministic:
            noise = np.ones_like(noise)
        variation += np.sin(harmonic * theta) * ((noise) * weight)
        # noise = rand.normal()
        noise = rand.uniform(-1, 1)
        if config.is_deterministic:
            noise = np.ones_like(noise)
        variation += np.cos(harmonic * theta) * ((noise) * weight)
    if is_normalized:
        variation_min = np.min(variation)
        variation = (variation - variation_min) / (np.max(variation) - variation_min)
    return variation


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PointPath:
    points: np.ndarray

    def create_height_mask(
        self, map: HeightMap, path_width: float, height_bucket_count: int, gap_count: int, gap_width: float
    ) -> MapFloatNP:
        is_debugging = False

        if height_bucket_count > 0 and gap_count > 0:
            assert height_bucket_count == gap_count, "Dont support unequal numbers of buckets and gaps"

        segment_lengths = self._get_segment_lengths()
        total_length = sum(segment_lengths)

        # create each segment of the path
        mask = np.zeros_like(map.Z)
        path_length_so_far = 0
        for i in range(0, len(self.points) - 1):
            start_point = self.points[i]
            end_point = self.points[i + 1]
            segment_length = segment_lengths[i]
            start_val = path_length_so_far / total_length
            end_val = (path_length_so_far + segment_length) / total_length
            # gradient_mask = np.ones_like(map.Z) * start_val
            # x_delta = end_point[0] - start_point[0]
            # if abs(x_delta) > 0.0001:
            #     x_slope = (end_val - start_val) / (x_delta)
            # else:
            #     x_slope = 0
            # y_delta = end_point[1] - start_point[1]
            # if abs(y_delta) > 0.0001:
            #     y_slope = (end_val - start_val) / (y_delta)
            # else:
            #     y_slope = 0
            # x_diffs = map.X - start_point[0]
            # y_diffs = map.Y - start_point[1]
            # if x_slope > 0 and y_slope > 0:
            #     x_slope, y_slope = _normalized(np.array([x_slope, y_slope]))
            # gradient_mask += x_diffs * x_slope + y_diffs * y_slope

            # TODO: I am a bad person. I couldnt figure out a better way to make a nice gradient from the start to the end
            #  so I just did this...
            center = 1000 * (start_point - end_point) + end_point
            gradient_mask = np.sqrt(map.get_dist_sq_to(center))
            gradient_mask -= gradient_mask[map.point_to_index(start_point)]
            gradient_mask /= gradient_mask[map.point_to_index(end_point)]

            gradient_mask = np.clip(gradient_mask, start_val, end_val)
            line_seg_distances = map.get_lineseg_distances(start_point, end_point)
            close_enough = line_seg_distances < (path_width / 2.0)
            mask[close_enough] = gradient_mask[close_enough]

            if is_debugging:
                print("Path segment:")
                print(start_val, end_val)
                print(start_point, end_point)
                plot_value_grid(gradient_mask, "Path construction gradient")
                plot_value_grid(line_seg_distances, "Path construction line seg distances")
                plot_value_grid(close_enough, "Path construction close enough")
                plot_value_grid(
                    mask,
                    "Path construction of height mask",
                    markers=[map.point_to_index(x) for x in [start_point, end_point]],
                )

            # if gap_count > 0:
            #     gaps = gap_count * min(1, round(segment_length / total_length))
            # else:
            #     gaps = 0
            # gap_locations = asdfsda
            # t = asdfdsaf
            # gap_mask = asdfdsaf
            # for gap_location in gap_locations:
            #     gap_start = gap_location - gap_width / 2.0
            #     gap_end = gap_location + gap_width / 2.0
            #     gap_mask[np.logical_and(t > gap_start, t < gap_end)] = True
            # solid_mask = np.logical_not(gap_mask)

            path_length_so_far += segment_length

        # with the requested number of gaps for that segment
        # create the composite path mask
        # bucket the values
        # if height_bucket_count > 0:
        #     mask = asdfdsa

        return mask

    def get_length(self) -> float:
        return sum(self._get_segment_lengths())

    def _get_segment_lengths(self) -> Tuple[float, ...]:
        segments = []
        for i in range(1, len(self.points)):
            dist = np.linalg.norm(self.points[i] - self.points[i - 1])
            segments.append(dist)
        return tuple(segments)


class PathFailure(Exception):
    pass


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HeightPath:
    # this is the width of the full path, ie, half of this width from the middle to each edge
    width: float = 1.0
    extra_point_count: int = 0
    is_path_climbable: bool = False
    is_height_affected: bool = True
    is_solution_flattened: bool = False
    is_path_restricted_to_land: bool = False
    flattening_mode: str = "midpoint"
    is_chasm_bottom_flattened: bool = False
    is_detailed: bool = True
    is_outside_edge_unclimbable: bool = False
    min_desired_2d_length: Optional[float] = None
    max_desired_2d_length: Optional[float] = None
    height_bucket_count: int = 0
    gap_count: int = 0
    gap_width: float = 0.0
    max_retries: int = 10
    ocean_radius: float = 20.0
    is_path_failure_allowed: bool = False
    is_debug_graph_printing_enabled: bool = False

    def apply(
        self,
        rand: np.random.Generator,
        map: HeightMap,
        is_climbable: MapBoolNP,
        is_detail_important: MapBoolNP,
        start_point: Point2DNP,
        end_point: Point2DNP,
        obstacle_weight_map: MapFloatNP,
        inner_solution_mask: MapBoolNP,
        outer_solution_mask: MapBoolNP,
    ):

        full_mask = np.logical_or(np.logical_or(inner_solution_mask, outer_solution_mask), obstacle_weight_map > 0.0)
        if self.is_debug_graph_printing_enabled:
            plot_value_grid(full_mask, "Full mask", markers=[map.point_to_index(x) for x in [start_point, end_point]])
            # plot_value_grid(inner_solution_mask)
            # plot_value_grid(outer_solution_mask)

        start_to_end_2d_dist = np.linalg.norm(start_point - end_point)
        min_len = start_to_end_2d_dist if self.min_desired_2d_length is None else self.min_desired_2d_length
        max_len = 3 * min_len if self.max_desired_2d_length is None else self.max_desired_2d_length

        # the basic path. Used if cannot make path
        path = PointPath(np.array([start_point, end_point]))
        # different cases for each:
        if self.extra_point_count == 0:
            # if there are 0 extra points, just a straight shot
            pass
        elif self.extra_point_count == 1:
            one_point_path = self._create_one_point_path(
                rand, map, full_mask, start_point, end_point, min_len, max_len
            )
            if one_point_path is not None:
                path = one_point_path
            elif not self.is_path_failure_allowed:
                raise PathFailure()
            else:
                map.log_simplicity_warning("Reduce from 1 extra point path to direct path")
        elif self.extra_point_count == 2:
            two_point_path = self._create_two_point_path(
                rand, map, full_mask, start_point, end_point, min_len, max_len
            )
            if two_point_path is not None:
                path = two_point_path
            elif not self.is_path_failure_allowed:
                raise PathFailure()
            else:
                map.log_simplicity_warning("Reduce from 1 extra point path to direct path")
        else:
            raise SwitchError(f"Only support paths with <= 2 points, sorry")

        # figure out the height differences
        start_height = map.get_rough_height_at_point(start_point)
        end_height = map.get_rough_height_at_point(end_point)
        height_delta = end_height - start_height
        # print("Height delta", height_delta)

        if self.is_solution_flattened or self.is_chasm_bottom_flattened:
            assert not self.is_height_affected, "Flattening and affecting height are mutually exclusive"
            assert self.extra_point_count == 0, "Only support flattening single segment paths"
            assert not self.is_outside_edge_unclimbable, "Doesnt make sense when flattening"
            assert not self.is_path_climbable, "Doesnt make sense when flattening"
            path_distances = map.get_lineseg_distances(start_point, end_point)
            radius = self.width / 2.0
            path_mask = path_distances < radius

            solution_only_mask = np.logical_or(inner_solution_mask, outer_solution_mask)

            if self.is_debug_graph_printing_enabled:
                plot_value_grid(path_mask, "Path Mask")

            if self.is_path_restricted_to_land:
                if np.any(np.logical_and(np.logical_not(map.get_land_mask()), path_mask)):
                    if IS_DEBUGGING_IMPOSSIBLE_WORLDS:
                        plot_terrain(
                            map.Z, "Path endpoints", markers=[map.point_to_index(x) for x in [start_point, end_point]]
                        )
                        plot_value_grid(path_mask, "Path")
                    raise ImpossibleWorldError("Path ended up in the ocean, not good. Try again plz.")

            # even if we're not restricted to land, we cannot affect the water
            path_mask = np.logical_and(path_mask, full_mask)

            # actually apply the flattening to the solution parts
            if self.is_solution_flattened:
                if self.flattening_mode == "midpoint":
                    height = (start_height + end_height) / 2.0
                elif self.flattening_mode == "min":
                    height = min([start_height, end_height])
                elif self.flattening_mode == "max":
                    height = max([start_height, end_height])
                else:
                    raise SwitchError(f"Unsupported {self.flattening_mode}")
                falloff_mask = np.logical_and(path_distances < self.width, solution_only_mask)
                mixing = np.clip(((self.width - path_distances[falloff_mask]) / self.width) * 2, 0, 1)
                map.Z[falloff_mask] = map.Z[falloff_mask] * (1.0 - mixing) + height * mixing

            # and apply the flattening to the obstacle parts
            if self.is_chasm_bottom_flattened:
                mid_point = (start_point + end_point) / 2.0
                height = map.get_rough_height_at_point(mid_point)
                falloff_mask = np.logical_and(path_distances < self.width, obstacle_weight_map >= 1.0)
                mixing = np.clip(((self.width - path_distances[falloff_mask]) / self.width) * 2, 0, 1)
                map.Z[falloff_mask] = map.Z[falloff_mask] * (1.0 - mixing) + height * mixing
        else:
            height_mask = path.create_height_mask(
                map, self.width, self.height_bucket_count, self.gap_count, self.gap_width
            )
            path_mask = np.logical_and(height_mask > 0, height_mask < 1)

            if self.is_path_restricted_to_land:
                if np.any(np.logical_and(np.logical_not(map.get_land_mask()), path_mask)):
                    if IS_DEBUGGING_IMPOSSIBLE_WORLDS:
                        plot_terrain(
                            map.Z, "Path endpoints", markers=[map.point_to_index(x) for x in [start_point, end_point]]
                        )
                        plot_value_grid(path_mask, "Path")
                    raise ImpossibleWorldError("Path ended up in the ocean, not good. Try again plz.")

            # only apply the part that overlaps with our solution zones
            path_mask = np.logical_and(path_mask, full_mask)

            if self.is_debug_graph_printing_enabled:
                plot_value_grid(path_mask, "Path Mask")

            # actually apply the path
            if self.is_height_affected:
                map.Z[path_mask] = height_mask[path_mask] * height_delta + start_height
            if self.is_path_climbable:
                is_climbable[path_mask] = True
            if self.is_outside_edge_unclimbable:
                expanded_mask = ndimage.binary_dilation(path_mask, structure=morphology.disk(5))
                edge_mask = np.logical_and(expanded_mask, np.logical_not(path_mask))
                is_climbable[edge_mask] = False
                is_detail_important[edge_mask] = True

        if self.is_detailed:
            is_detail_important[path_mask] = True

        if self.is_debug_graph_printing_enabled:
            plot_value_grid(is_climbable, "Is climbable after path")
            plot_value_grid(is_detail_important, "Is detail after path")

    def _create_one_point_path(
        self,
        rand: np.random.Generator,
        map: HeightMap,
        full_mask: MapBoolNP,
        start_point: Point2DNP,
        end_point: Point2DNP,
        min_len: float,
        max_len: float,
    ) -> Optional[PointPath]:
        # try a bunch of splits for the desired distance
        start_dist_sq = map.get_dist_sq_to(start_point)
        end_dist_sq = map.get_dist_sq_to(end_point)
        for mask in self._masks_to_try(map, full_mask, start_dist_sq, end_dist_sq):
            for i in range(self.max_retries):
                overall_len = rand.uniform(min_len, max_len)
                path = _make_one_point_path(rand, map, mask, overall_len, start_point, end_point, self.width)
                if path is not None:
                    return path
        return None

    def _create_two_point_path(
        self,
        rand: np.random.Generator,
        map: HeightMap,
        full_mask: MapBoolNP,
        start_point: Point2DNP,
        end_point: Point2DNP,
        min_len: float,
        max_len: float,
    ) -> Optional[PointPath]:
        start_dist_sq = map.get_dist_sq_to(start_point)
        end_dist_sq = map.get_dist_sq_to(end_point)
        max_len_in_one_point = (max_len - min_len) * 0.6
        for mask in self._masks_to_try(map, full_mask, start_dist_sq, end_dist_sq):
            for i in range(self.max_retries):
                overall_len = rand.uniform(min_len, min_len + max_len_in_one_point * rand.uniform())
                start_len = overall_len * rand.uniform(0.1, 0.45)
                end_len = start_len
                mid_points = _get_circle_intersections(start_point, start_len, end_point, end_len)
                if len(mid_points) == 2:
                    rand.shuffle(mid_points)
                    path = PointPath(np.array([start_point, *mid_points, end_point]))
                    if _check_full_path(path, map, mask, self.width):
                        if min_len < path.get_length() < max_len:
                            return path
                        else:
                            logger.debug("Rejected because path length")

                    # fine, whatever, let's just try again
        # if we've tried a bunch and still failed, bail
        return None

    # allows us to try again cutting a little bit into the ocean if strictly necessary to find a working path
    def _masks_to_try(
        self, map: HeightMap, full_mask: MapBoolNP, start: MapFloatNP, end: MapFloatNP
    ) -> List[MapBoolNP]:
        if self.ocean_radius <= 0.0:
            return [full_mask]
        alt_mask = full_mask.copy()
        ocean_radius_sq = self.ocean_radius * self.ocean_radius
        within_radius = np.logical_or(start < ocean_radius_sq, end < ocean_radius_sq)
        alt_mask[np.logical_and(map.Z < WATER_LINE, within_radius)] = True
        return [full_mask, alt_mask]


def _make_one_point_path(
    rand, map: HeightMap, mask, overall_len, start_point, end_point, width
) -> Optional[PointPath]:
    start_len = overall_len * 0.5
    end_len = start_len
    for mid_point in _get_circle_intersections(start_point, start_len, end_point, end_len):
        is_start_path_clear = _check_path(map, mask, start_point, mid_point, width, shrink_start=True)
        is_end_path_clear = _check_path(map, mask, start_point, mid_point, width, shrink_end=True)
        if is_start_path_clear and is_end_path_clear:
            return PointPath(np.array([start_point, mid_point, end_point]))


def _check_full_path(path: PointPath, map: HeightMap, mask: MapBoolNP, width: float) -> bool:
    # plot_value_grid(mask, markers=[map.point_to_index(x) for x in path.points])

    for i in range(1, len(path.points)):
        start = path.points[i - 1]
        end = path.points[i]
        if not _check_path(map, mask, start, end, width, shrink_start=i == 1, shrink_end=i == (len(path.points) - 1)):
            return False
        # # ensure that non of the other points are co-linear
        # other_points = list(path.points[: i - 1]) + list(path.points[i + 1 :])
        # for point in other_points:
        #     if abs(convenient_signed_line_distance(point, start, end)) < 0.5 * width:
        #         logger.debug("Rejected because colinear")
        #         return False
    return True


def _check_path(
    map: HeightMap,
    mask: MapBoolNP,
    start_point: Point2DNP,
    end_point: Point2DNP,
    width: float,
    shrink_start=False,
    shrink_end=True,
) -> bool:
    if shrink_start:
        points_2d = np.stack([map.X, map.Y], axis=2)
        path_mask = _make_triangular_mask(points_2d, start_point, end_point, width)
    elif shrink_end:
        points_2d = np.stack([map.X, map.Y], axis=2)
        path_mask = _make_triangular_mask(points_2d, end_point, start_point, width)
    else:
        # shrink the points in so they dont intersect at the ends
        start_to_end_dist = np.linalg.norm(start_point - end_point)
        if start_to_end_dist < width:
            midpoint = (start_point + end_point) / 2.0
            path_mask = map.get_dist_sq_to(midpoint) < width ** 2
        else:
            start_to_end_vec = (end_point - start_point) / start_to_end_dist
            start_point = start_point + start_to_end_vec * (width / 2.0)
            end_point = end_point + -1.0 * start_to_end_vec * (width / 2.0)
            line_seg_distances = map.get_lineseg_distances(start_point, end_point)
            path_mask = line_seg_distances < width / 2.0

    # plot_value_grid(path_mask, "mask", markers=[map.point_to_index(x) for x in [start_point, end_point]])

    # TODO: this could adaptively threshold as well I suppose...
    if np.any(np.logical_and(path_mask, np.logical_not(mask))):
        # plot_value_grid(mask, "mask", markers=[map.point_to_index(x) for x in [start_point, end_point]])
        # plot_value_grid(line_seg_distances < width, "line_seg_distances")
        # plot_value_grid(np.logical_and(line_seg_distances < width, np.logical_not(mask)), "failure points")
        # logger.debug("Rejected because not in mask")
        return False
    return True


def _make_triangular_mask(points_2d, start: Point2DNP, end: Point2DNP, width: float) -> MapBoolNP:
    start_to_end_dist = np.linalg.norm(start - end)
    start_to_end_vec = (end - start) / start_to_end_dist
    rot_vec = np.array([-start_to_end_vec[1], start_to_end_vec[0]])
    e1 = end + rot_vec * width / 2.0
    e2 = end + -1.0 * rot_vec * width / 2.0

    triangle_mask = np.logical_and(
        np.logical_and(
            _bulk_signed_line_distance(points_2d, start, e1) > 0, _bulk_signed_line_distance(points_2d, e1, e2) > 0
        ),
        _bulk_signed_line_distance(points_2d, e2, start) > 0,
    )

    # plot_value_grid(triangle_mask)

    return triangle_mask


def _bulk_signed_line_distance(points: np.ndarray, a: Point2DNP, b: Point2DNP) -> np.ndarray:
    ab_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return (((b[0] - a[0]) * (a[1] - points[:, :, 1])) - ((a[0] - points[:, :, 0]) * (b[1] - a[1]))) / ab_dist


def _get_circle_intersections(
    center_a: Point2DNP, radius_a: float, center_b: Point2DNP, radius_b: float
) -> List[Point2DNP]:
    result = get_intersections(center_a[0], center_a[1], radius_a, center_b[0], center_b[1], radius_b)
    if result is None:
        return []
    x1, y1, x2, y2 = result
    if abs(x1 - x2) < 0.001 and abs(y1 - y2) < 0.001:
        return [np.array([x1, y1])]
    return [np.array([x1, y1]), np.array([x2, y2])]


def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = math.sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return (x3, y3, x4, y4)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HeightSolution:
    paths: Tuple[HeightPath, ...] = tuple()
    inside_items: Tuple[Tool, ...] = tuple()
    outside_items: Tuple[Tool, ...] = tuple()
    inside_item_randomization_distance: float = 0.0
    outside_item_randomization_distance: float = 0.0
    # how far away from the brink to push solution points
    # needs this high of a default so that we can find something in the solution mask that works
    solution_point_brink_distance: float = 1.0
    # TODO: this parameterization is lame. It assumes all items are the same type and size, basically
    inside_item_radius: float = 0.0
    outside_item_radius: float = 0.0
    is_debug_graph_printing_enabled: bool = False

    def reverse(self) -> "HeightSolution":
        return attr.evolve(
            self,
            inside_items=self.outside_items,
            outside_items=self.inside_items,
            inside_item_randomization_distance=self.outside_item_randomization_distance,
            outside_item_randomization_distance=self.inside_item_randomization_distance,
            inside_item_radius=self.outside_item_radius,
            outside_item_radius=self.inside_item_radius,
        )

    # path (pillar, plateau, land, climb), how high to start, how complex
    # items (items in solution positions, whether to destack, how far to move, which zone they're in)
    # is walking path enabled (for being able to retry jumps, basically)
    def apply(
        self,
        rand: np.random.Generator,
        map: HeightMap,
        is_climbable: MapBoolNP,
        is_detail_important: MapBoolNP,
        obstacle_weight_map: MapFloatNP,
        inner_solution_mask: MapBoolNP,
        outer_solution_mask: MapBoolNP,
        inner_solution_point: Point2DNP,
        outer_solution_point: Point2DNP,
    ) -> Tuple[Tool, ...]:
        # see above TODO--we dont support different types of items in the same solution
        if self.inside_item_radius > 0.0:
            radii = set([x.get_offset() for x in self.inside_items])
            assert (
                len(radii) <= 1
            ), "Not allowed to have multiple radii of items in your solution because the implementation is lazy"
        if self.outside_item_radius > 0.0:
            radii = set([x.get_offset() for x in self.outside_items])
            assert (
                len(radii) <= 1
            ), "Not allowed to have multiple radii of items in your solution because the implementation is lazy"

        if self.is_debug_graph_printing_enabled:
            plot_value_grid(obstacle_weight_map, "Solution obstacle weights")
            plot_value_grid(inner_solution_mask, "Solution inner")
            plot_value_grid(outer_solution_mask, "Solution outer")

        # inner_solution_point = self._fix_solution_point(map, inner_solution_point, inner_solution_mask)
        # outer_solution_point = self._fix_solution_point(map, outer_solution_point, outer_solution_mask)

        # if self.is_climb_path_created:
        #     _create_climb_path(map, is_climbable, inner_solution_point, outer_solution_point, 5.0)

        # force the points sufficiently far apart that they dont overlap to start with
        # and they are in their solution zones
        start_point = inner_solution_point
        end_point = outer_solution_point
        if self.solution_point_brink_distance > 0.0:
            # step the points away from midpoint until they hit the edge of their solution masks
            num_steps = 40
            step_size = self.solution_point_brink_distance / num_steps

            step_vector = normalized(start_point - end_point)
            point = start_point
            has_entered_mask = inner_solution_mask[map.point_to_index(point)]
            for i in range(1, num_steps + 1):
                point = point + step_vector * step_size
                if inner_solution_mask[map.point_to_index(point)]:
                    start_point = point
                    has_entered_mask = True
                else:
                    if has_entered_mask:
                        break
            if not has_entered_mask:
                if IS_DEBUGGING_IMPOSSIBLE_WORLDS:
                    plot_terrain(map.Z, markers=[map.point_to_index(x) for x in [start_point, end_point]])
                raise ImpossibleWorldError("Could not find start point within mask")
                # fine, I guess we start where we were told then
                # start_point = inner_solution_point
            step_vector = normalized(end_point - start_point)
            point = end_point
            has_entered_mask = inner_solution_mask[map.point_to_index(point)]
            for i in range(1, num_steps + 1):
                point = point + step_vector * step_size
                if outer_solution_mask[map.point_to_index(point)]:
                    end_point = point
                    has_entered_mask = True
                else:
                    if has_entered_mask:
                        break
            if not has_entered_mask:
                # fine, I guess we start where we were told then
                end_point = outer_solution_point
            if not has_entered_mask:
                if IS_DEBUGGING_IMPOSSIBLE_WORLDS:
                    plot_terrain(map.Z, markers=[map.point_to_index(x) for x in [start_point, end_point]])
                raise ImpossibleWorldError("Could not find end point within mask")
            if tuple(start_point) == tuple(end_point):
                raise ImpossibleWorldError("HeightSolution cannot separate start and end points")

        if self.is_debug_graph_printing_enabled:
            plot_terrain(map.Z, "Path endpoints", markers=[map.point_to_index(x) for x in [start_point, end_point]])
            # plot_terrain(map.Z, "Path endpoints", markers=[map.point_to_index(start_point)])

        land = map.get_land_mask()
        if not land[map.point_to_index(start_point)]:
            if IS_DEBUGGING_IMPOSSIBLE_WORLDS:
                plot_terrain(
                    map.Z, "Path endpoints", markers=[map.point_to_index(x) for x in [start_point, end_point]]
                )
            raise ImpossibleWorldError("Solution start point is not on land, bad")
        if not land[map.point_to_index(end_point)]:
            if IS_DEBUGGING_IMPOSSIBLE_WORLDS:
                plot_terrain(
                    map.Z, "Path endpoints", markers=[map.point_to_index(x) for x in [start_point, end_point]]
                )
            raise ImpossibleWorldError("Solution end point is not on land, bad")

        # create any defined path
        for path in self.paths:
            # print("mask components")
            # plot_value_grid(inner_solution_mask)
            # plot_value_grid(obstacle_weight_map)
            # plot_value_grid(outer_solution_mask)

            path.apply(
                rand,
                map,
                is_climbable,
                is_detail_important,
                start_point,
                end_point,
                obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
            )

        # TODO: create the nice walking area if requested (get nearest point on both shapes, get heights and distances, then interpolate based on your distance)

        # place any requested items
        items = []
        if len(self.inside_items) > 0:
            new_items, important_points = self._place_items(
                rand,
                map,
                self.inside_items,
                self.inside_item_randomization_distance,
                self.inside_item_radius,
                start_point,
                end_point,
                inner_solution_mask,
            )
            is_detail_important[important_points] = True
            items.extend(new_items)
        if len(self.outside_items) > 0:
            new_items, important_points = self._place_items(
                rand,
                map,
                self.outside_items,
                self.outside_item_randomization_distance,
                self.outside_item_radius,
                end_point,
                start_point,
                outer_solution_mask,
            )
            is_detail_important[important_points] = True
            items.extend(new_items)

        return tuple(items)

    def _fix_solution_point(self, map: HeightMap, solution_point: Point2DNP, solution_mask: MapBoolNP) -> Point2DNP:
        index = map.point_to_index(solution_point)
        if solution_mask[index]:
            return solution_point
        # move the point into the masked region, since that is not guaranteed by the outer code
        dist_sq = map.get_dist_sq_to(solution_point)
        dist_sq[np.logical_not(solution_mask)] = np.inf
        near_index = tuple(np.unravel_index(np.argmin(dist_sq), dist_sq.shape))
        return map.index_to_point_2d(near_index)

    def _place_items(
        self,
        rand: np.random.Generator,
        map: HeightMap,
        items: Tuple[Tool, ...],
        randomization_distance: float,
        item_radius: float,
        inside_solution_point: Point2DNP,
        outside_solution_point: Point2DNP,
        solution_mask: MapBoolNP,
    ) -> Tuple[List[Tool], MapBoolNP]:
        solution_forward = normalized(outside_solution_point - inside_solution_point)
        solution_yaw = np.arctan2(solution_forward[1], solution_forward[0])

        item_size_radius_in_grid_unit = round(item_radius * map.cells_per_meter) + 1
        # reduce the size of the solution mask so we dont end up inside of walls
        reduced_solution_mask = np.logical_not(
            ndimage.binary_dilation(
                np.logical_not(solution_mask), structure=morphology.disk(item_size_radius_in_grid_unit)
            )
        )

        transformed_items = []
        for item in items:

            # figure out the new base position
            item_local_pos_2d = to_2d_point(item.position)
            item_r = np.linalg.norm(item_local_pos_2d)
            item_yaw = np.arctan2(item_local_pos_2d[1], item_local_pos_2d[0])
            item_yaw += solution_yaw
            rotated_item_pos_2d = np.array([item_r * np.cos(item_yaw), item_r * np.sin(item_yaw)])
            item_pos_2d = rotated_item_pos_2d + inside_solution_point
            height = map.get_rough_height_at_point(item_pos_2d)
            new_position = np.array([item_pos_2d[0], item.position[1] + height, item_pos_2d[1]])

            # figure out the new rotation
            yaw_rotation = Rotation.from_euler("y", -solution_yaw)
            old_rotation = Rotation.from_matrix(item.rotation.reshape((3, 3)))
            new_rotation = (yaw_rotation * old_rotation).as_matrix().flatten()

            # create the updated item
            extra_kwargs = {}
            if isinstance(item, Tool):
                extra_kwargs["solution_mask"] = reduced_solution_mask
            updated_item = attr.evolve(item, position=new_position, rotation=new_rotation, **extra_kwargs)
            transformed_items.append(updated_item)

        # add some noise to the position, ensuring that it stays within the solution mask
        if randomization_distance > 0.0:
            # in case the solution area is too small
            if np.any(reduced_solution_mask):
                for i, item in enumerate(transformed_items):
                    item_pos_2d = to_2d_point(item.position)
                    item_dist_sq = map.get_dist_sq_to(item_pos_2d)
                    item_jitter_mask = item_dist_sq < randomization_distance ** 2
                    possible_spawn_mask = np.logical_and(item_jitter_mask, reduced_solution_mask)
                    if not np.any(possible_spawn_mask):
                        possible_spawn_mask = reduced_solution_mask
                    if not np.any(reduced_solution_mask):
                        raise ImpossibleWorldError(f"No place to put item: {item}")
                    other_items_mask = np.zeros_like(possible_spawn_mask)
                    # prevent items from moving inside of other items on the same height level as them
                    if item_size_radius_in_grid_unit > 0:
                        for j, other_item in enumerate(transformed_items):
                            # ignore yourself
                            if i == j:
                                continue
                            # ignore other items that are not on your same level
                            if items[i].position[1] != items[j].position[1]:
                                continue
                            other_items_mask[map.point_to_index(to_2d_point(other_item.position))] = True
                        other_items_mask = ndimage.binary_dilation(
                            other_items_mask, structure=morphology.disk(item_size_radius_in_grid_unit)
                        )
                    spawn_points_without_other_items = np.logical_and(
                        possible_spawn_mask, np.logical_not(other_items_mask)
                    )
                    if self.is_debug_graph_printing_enabled:
                        plot_value_grid(reduced_solution_mask, "Reduced solution graph")
                        plot_value_grid(possible_spawn_mask, "Possible tool spawn")
                        plot_value_grid(spawn_points_without_other_items, "Final placement graph for tools")
                    if np.any(spawn_points_without_other_items):
                        new_point_2d = map.index_to_point_2d(
                            tuple(rand.choice(np.argwhere(spawn_points_without_other_items)))
                        )
                        old_height = map.get_rough_height_at_point(item_pos_2d)
                        new_position_2d = np.array([new_point_2d[0], new_point_2d[1]])
                        new_height = map.get_rough_height_at_point(new_position_2d)
                        height_offset = new_height - old_height

                        item.position[0] = new_point_2d[0]
                        item.position[2] = new_point_2d[1]
                        item.position[1] = item.position[1] + height_offset

        # verify that all items have been moved out of the ocean
        for item in transformed_items:
            if map.get_rough_height_at_point(to_2d_point(item.position)) < WATER_LINE:
                raise ImpossibleWorldError(
                    "Some items could not find spawn points on the land. Try increasing their randomization distance or solution zone size"
                )

        # set is_detail_important around these items as well
        important_points = add_detail_near_items(map, transformed_items, item_radius)

        return transformed_items, important_points


def add_detail_near_items(map: HeightMap, items: List[Item], item_radius: float):
    item_size_radius_in_grid_unit = round(item_radius * map.cells_per_meter) + 1
    # set is_detail_important around these items as well
    new_detail_areas = np.zeros_like(map.Z, dtype=np.bool_)
    for item in items:
        item_pos_2d = to_2d_point(item.position)
        new_detail_areas[map.point_to_index(item_pos_2d)] = True

    importance_radius = max([2, item_size_radius_in_grid_unit])
    return ndimage.binary_dilation(new_detail_areas, structure=morphology.disk(importance_radius))


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
        # print(inner_r)
        # print(outer_r)
        # print(obstacle.traversal_length)
        # print(self.chasm_bottom_size)
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


@attr.s(auto_attribs=True, collect_by_mro=True)
class NewWorld:
    """Enables creation of a series of Zones separated by Obstacles on a HeightMap"""

    config: WorldConfig
    map: HeightMap
    export_config: ExportConfig
    entity_id_gen: IdGenerator
    items: List[Entity]
    obstacle_zones: List[Tuple[MapBoolNP, MapBoolNP]]
    building_by_id: Dict[int, Building]
    empty_space: MapBoolNP
    view_mask: MapBoolNP
    is_climbable: MapBoolNP
    is_detail_important: MapBoolNP
    full_obstacle_mask: MapBoolNP
    flora_mask: MapFloatNP
    is_debug_graph_printing_enabled: bool
    biome_config: Optional[BiomeConfig] = None
    special_biomes: Optional[SpecialBiomes] = None

    is_safe_mode_enabled: bool = True

    # TODO: these old functions should be moved...
    def get_spawn_location(self) -> np.ndarray:
        spawn_height = self.get_height_at((0.0, 0.0)) + AGENT_HEIGHT / 2.0
        return np.array([0, spawn_height, 0])

    @staticmethod
    def build(
        config: WorldConfig,
        export_config: ExportConfig,
        biome_config: Optional[BiomeConfig] = None,
        is_debug_graph_printing_enabled: bool = True,
    ) -> "NewWorld":
        map = build_outdoor_world_map(config, is_debug_graph_printing_enabled)
        true_array = np.ones_like(map.Z, dtype=np.bool_)
        return NewWorld(
            config,
            map,
            export_config,
            IdGenerator(),
            [],
            [],
            {},
            empty_space=true_array,
            view_mask=np.ones_like(true_array),
            is_climbable=true_array.copy(),
            is_detail_important=np.zeros(map.Z.shape, dtype=np.bool_),
            full_obstacle_mask=np.zeros(map.Z.shape, dtype=np.bool_),
            is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
            biome_config=biome_config,
            flora_mask=np.ones_like(map.Z),
        )

    def make_biome_noise(
        self, rand: np.random.Generator, resource_file: str, noise_min: float, noise_scale: float = 0.05
    ):
        if self.biome_config.is_independent_noise_per_scenery:
            noise = perlin(self.map.Z.shape, noise_scale, rand, is_normalized=True, noise_min=noise_min)
        else:
            noise = np.ones_like(self.map.Z)
        is_tree = "trees/" in resource_file
        if is_tree:
            noise *= self.view_mask
        return noise

    def flatten(self, point: Point2DNP, radius: float, importance_radius: float):
        dist_sq = self.map.radial_flatten(point, radius)
        nearby = dist_sq < importance_radius * importance_radius
        self.is_detail_important[nearby] = True

    def get_height_at(self, point: Union[Tuple[float, float], Point2DNP]) -> float:
        if isinstance(point, tuple):
            point = np.array(point)
        return self.map.Z[self.map.point_to_index(point)]

    def reset_height_offset(self, item: EntityType, offset: float) -> EntityType:
        pos = item.position.copy()
        pos[1] = self.get_height_at(to_2d_point(item.position)) + offset
        return attr.evolve(item, position=pos)

    def add_item(self, item: EntityType, reset_height_offset: Optional[float] = None) -> EntityType:
        item = attr.evolve(item, entity_id=self.entity_id_gen.get_next_id())
        if reset_height_offset is not None:
            item = self.reset_height_offset(item, reset_height_offset)
        self.items.append(item)
        return item

    def add_spawn(
        self,
        rand: np.random.Generator,
        difficulty: float,
        spawn_location: Point3DNP,
        food_location: Point3DNP,
        is_visibility_required: bool = True,
    ):
        spawn_location = spawn_location.copy()
        height_at_location = self.map.Z[self.map.point_to_index(to_2d_point(spawn_location))]
        spawn_location[1] = height_at_location + HALF_AGENT_HEIGHT_VECTOR[1] * 1.1
        spawn_item = get_spawn(self.entity_id_gen, rand, difficulty, spawn_location, food_location)
        spawn_item = attr.evolve(spawn_item, is_visibility_required=is_visibility_required)
        self.items.append(spawn_item)

        # update the flora mask to prevent things from spawning in the special areas
        flora_removal_cells = round(self.map.cells_per_meter * FLORA_REMOVAL_METERS) + 1
        flora_removal_mask = morphology.dilation(self.is_detail_important, morphology.disk(flora_removal_cells))
        self.flora_mask[flora_removal_mask] = 0.0

        # update our view_mask to prevent trees from blocking our view of the goal
        if is_visibility_required:
            visibility_line_width = METERS_OF_TREE_CLEARANCE_AROUND_LINE_OF_SIGHT
            segment_dist = self.map.get_lineseg_distances(to_2d_point(spawn_location), to_2d_point(food_location))
            close_to_sight_line = segment_dist < visibility_line_width
            self.view_mask[close_to_sight_line] = False

    def add_spawn_and_food(
        self,
        rand: np.random.Generator,
        difficulty: float,
        spawn_location: Point3DNP,
        food_location: Point3DNP,
        food_class=None,
        is_visibility_required: bool = True,
    ):
        if food_class is None:
            food_class = CANONICAL_FOOD_CLASS
        # TODO: not quite right, this is the wrong offset...
        food_location[1] = (
            self.map.Z[self.map.point_to_index(to_2d_point(food_location))] + CANONICAL_FOOD.get_offset()
        )
        food = food_class(entity_id=self.entity_id_gen.get_next_id(), position=food_location)
        self.items.append(food)
        self.add_spawn(rand, difficulty, spawn_location, food_location, is_visibility_required=is_visibility_required)

    def replace_weapon_placeholders(self, replacements: Iterable[Type[Weapon]]):
        fixed_items = []
        weapon_types = iter(replacements)
        for item in self.items:
            if isinstance(item, Placeholder):
                weapon_type = next(weapon_types)
                new_item = weapon_type(position=item.position, entity_id=item.entity_id, rotation=item.rotation)
                new_item = self.reset_height_offset(new_item, new_item.get_offset())
                fixed_items.append(new_item)
            else:
                fixed_items.append(item)
        self.items = fixed_items

    def get_random_point_for_weapon_or_predator(
        self, rand: np.random.Generator, point: Point2DNP, radius: float, island_mask: MapBoolNP
    ) -> Optional[Point2DNP]:
        safe_mask = self.get_safe_mask(island_mask=island_mask)
        safe_places = np.argwhere(safe_mask)
        possible_points = self.map.get_2d_points()[safe_places[:, 1], safe_places[:, 0]]
        distances = np.linalg.norm(possible_points - point, axis=1)
        for std_dev in (1.0, 5.0):
            location_weights = stats.norm(radius, std_dev).pdf(distances)
            total_weights = location_weights.sum()
            if total_weights <= 0.0:
                continue
        if total_weights <= 0.0:
            return None
        location_probabilities = location_weights / total_weights
        return rand.choice(possible_points, p=location_probabilities)

    def add_random_predator_near_point(
        self, rand: np.random.Generator, predator_type, location: Point2DNP, radius: float, island_mask: MapBoolNP
    ):
        predator_pos_2d = self.get_random_point_for_weapon_or_predator(rand, location, radius, island_mask)
        if predator_pos_2d is None:

            raise ImpossibleWorldError("Nowhere to stick a single predator. Weird")
        # add predator near food
        predator_position = np.array([predator_pos_2d[0], 0.0, predator_pos_2d[1]])
        # TODO: animals should probably have offsets too
        predator = predator_type(entity_id=0, position=predator_position)
        self.add_item(predator, reset_height_offset=predator.get_offset())

    def carry_tool_randomly(self, rand: np.random.Generator, item: Tool, distance_preference: stats.norm):

        assert (
            item.solution_mask is not None
        ), "How was this tool placed? Ideally we should be setting the solution_mask when placing tools, so that we can carry them later"

        # don't accidentally stomp on anything else
        possible_spawn_mask = np.logical_and(item.solution_mask, np.logical_not(self.is_detail_important))

        # if there is somewhere to go...
        if np.any(possible_spawn_mask):

            # figure out the new position
            center_point = to_2d_point(item.position)
            possible_points = self.map.get_2d_points()[possible_spawn_mask]
            new_point_2d = only(
                selected_distance_weighted_points(rand, possible_points, center_point, distance_preference, 1)
            )
            height = self.map.get_rough_height_at_point(new_point_2d)
            new_position = np.array([new_point_2d[0], height, new_point_2d[1]])

            # print(f"Carried object {np.linalg.norm(new_point_2d - center_point)}m")

            # replace the item with one that has an updated position
            prev_len = len(self.items)
            self.items = [x for x in self.items if x.entity_id != item.entity_id]
            assert len(self.items) == prev_len - 1, "Accidentally removed multiple items, that's bad"
            item = attr.evolve(item, position=new_position)
            self.items.append(item)

            important_points = add_detail_near_items(self.map, [item], item.get_offset())
            self.is_detail_important[important_points] = True

    def get_safe_point(
        self,
        rand: np.random.Generator,
        sq_distances: Optional[MapFloatNP] = None,
        max_sq_dist: Optional[float] = None,
        island_mask: Optional[MapBoolNP] = None,
    ) -> Optional[Point3DNP]:
        mask = self.get_safe_mask(island_mask, max_sq_dist, sq_distances)
        return self._get_safe_point(rand, mask)

    def _get_safe_point(self, rand: np.random.Generator, mask: MapBoolNP) -> Optional[Point3DNP]:
        if not np.any(mask):
            return None
        selected_coords = tuple(rand.choice(np.argwhere(mask)))
        point_2d = self.map.index_to_point_2d(selected_coords)
        return np.array([point_2d[0], self.map.get_rough_height_at_point(point_2d), point_2d[1]])

    def get_safe_mask(
        self,
        island_mask: Optional[MapBoolNP] = None,
        max_sq_dist: Optional[float] = None,
        sq_distances: Optional[MapFloatNP] = None,
    ) -> MapBoolNP:
        # start with all land
        mask = self.map.get_land_mask()
        # remove all obstacles
        mask = np.logical_and(mask, np.logical_not(self.full_obstacle_mask))
        # remove places where detail is important
        mask = np.logical_and(mask, np.logical_not(self.is_detail_important))
        # remove unclimbable places
        mask = np.logical_and(mask, self.is_climbable)
        # if specified, restrict to a single continent
        if island_mask is not None:
            mask = np.logical_and(mask, island_mask)
        # if specified, restrict to points nearby
        if sq_distances is not None:
            assert max_sq_dist is not None
            mask = np.logical_and(mask, sq_distances < max_sq_dist)
        return mask

    def add_height_obstacle(self, rand: np.random.Generator, ring_config: RingObstacleConfig, island_mask: MapBoolNP):

        if self.is_debug_graph_printing_enabled:
            plot_terrain(self.map.Z, "Terrain (before obstacle)")

        if ring_config.inner_obstacle is None or ring_config.outer_obstacle is None:
            assert ring_config.chasm_bottom_size == 0.0, "Cannot make a chasm with a single obstacle"

        assert ring_config.chasm_bottom_size >= 0, "chasm_bottom_size cannot be negative"

        # TODO: deal with zones for compositional tasks
        # if around_zone is None:
        #     around_zone = Zone(zone_id=len(self.zone_by_id), mask=inner_zone)
        # elif self.is_safe_mode_enabled:
        #     around_zone.check_if_contained_by(inner_zone)

        if ring_config.inner_obstacle and ring_config.outer_obstacle:
            ring = self._create_ring(
                rand,
                ring_config,
                ring_config.inner_safety_radius + ring_config.inner_obstacle.traversal_length,
                ring_config.outer_safety_radius
                - (ring_config.outer_obstacle.traversal_length + ring_config.chasm_bottom_size),
            )
        elif ring_config.inner_obstacle:
            ring = self._create_ring(
                rand,
                ring_config,
                ring_config.inner_safety_radius + ring_config.inner_obstacle.traversal_length,
                ring_config.outer_safety_radius,
            )
        elif ring_config.outer_obstacle:
            ring = self._create_ring(
                rand,
                ring_config,
                ring_config.inner_safety_radius,
                ring_config.outer_safety_radius - ring_config.outer_obstacle.traversal_length,
            )
        else:
            raise Exception("Must define one of inner or outer obstacle")

        inner_mask = None
        outer_mask = None

        if ring_config.inner_obstacle:
            inner_obstacle_weight_map, inner_mask, outside_inner_obstacle_mask = self._create_obstacle_masks(
                rand, ring, ring_config.inner_obstacle, island_mask
            )

        if ring_config.outer_obstacle:
            # TODO: mutating mid_z here is kinda weird, dont do that
            if ring_config.inner_obstacle:
                # ring.mid_z += ring_config.chasm_bottom_size
                orig = ring.mid_z
                ring.mid_z = ring.outer_mid_z
            outer_obstacle_weight_map, inside_outer_obstacle_mask, outer_mask = self._create_obstacle_masks(
                rand, ring, ring_config.outer_obstacle, island_mask
            )
            if ring_config.inner_obstacle:
                # ring.mid_z -= ring_config.chasm_bottom_size / 2.0
                ring.mid_z = (orig + ring.outer_mid_z) / 2.0

        if ring_config.inner_obstacle:
            inner_point_for_inner_solution, outer_point_for_inner_solution = ring_config.get_inner_solution_points()
            inner_traversal_midpoint = (inner_point_for_inner_solution + outer_point_for_inner_solution) / 2.0
        if ring_config.outer_obstacle:
            inner_point_for_outer_solution, outer_point_for_outer_solution = ring_config.get_outer_solution_points()
            outer_traversal_midpoint = (inner_point_for_outer_solution + outer_point_for_outer_solution) / 2.0

        # when applying for chasm or ridge, need to be really careful that the edges of the masks work out
        if ring_config.inner_obstacle and ring_config.outer_obstacle:
            if self.is_debug_graph_printing_enabled:
                plot_value_grid(
                    inner_obstacle_weight_map - outer_obstacle_weight_map, "Inner and outer obstacle delta"
                )
            map_new, is_climbable_new = self._apply_height_obstacle(
                inner_obstacle_weight_map - outer_obstacle_weight_map,
                ring,
                island_mask,
                inner_point_for_inner_solution,
                outer_point_for_outer_solution,
            )
        elif ring_config.inner_obstacle:
            map_new, is_climbable_new = self._apply_height_obstacle(
                inner_obstacle_weight_map,
                ring,
                island_mask,
                inner_point_for_inner_solution,
                outer_point_for_inner_solution,
            )
        elif ring_config.outer_obstacle:
            map_new, is_climbable_new = self._apply_height_obstacle(
                outer_obstacle_weight_map,
                ring,
                island_mask,
                inner_point_for_outer_solution,
                outer_point_for_outer_solution,
            )
        else:
            raise Exception("Pointless obstacle")

        is_detail_important_new = self.is_detail_important.copy()

        # ignore borders with the ocean, they should not be marked as important
        not_island_borders = np.logical_not(map_new.get_outline(island_mask, 3))
        for i in range(2):
            if i == 0:
                obstacle = ring_config.inner_obstacle
            else:
                obstacle = ring_config.outer_obstacle
            if obstacle:
                if i == 0:
                    mask = inner_obstacle_weight_map
                    traversal_point = inner_traversal_midpoint
                else:
                    mask = outer_obstacle_weight_map
                    traversal_point = outer_traversal_midpoint
                # update the climbability where the height changed if necessary
                if not obstacle.is_default_climbable:
                    masked_climbing_region = np.logical_and(mask < 1.0, mask > 0.0)
                    # applies a 3x3 pass to ensure that nearby elements are also considered unclimbable
                    masked_climbing_region = ndimage.binary_dilation(
                        masked_climbing_region, structure=ndimage.generate_binary_structure(2, 2)
                    )
                    # mask out anything that is faded
                    # plot_value_grid(masked_climbing_region, "masked_climbing_region")
                    is_climbable_new[masked_climbing_region] = False

                # we mark things as important if they are sufficiently close to a traversal point
                # AND there is a big delta in height. Otherwise we lose important detail on some obstacles (namely the bottoms of chasms)
                locations = np.stack([self.map.X, self.map.Y], axis=2)
                dist_sq = (locations[:, :, 0] - traversal_point[0]) ** 2 + (
                    locations[:, :, 1] - traversal_point[1]
                ) ** 2
                obstacle_radius = obstacle.detail_radius
                near_important_point = dist_sq < obstacle_radius * obstacle_radius

                # also union this with any place where a cel is 0 or 1, but neighbors are not
                for val in (0, 1):
                    extra_mask = np.zeros_like(mask)
                    extra_mask[mask == val] = 1.0
                    expanded_extra_mask = ndimage.binary_dilation(extra_mask, structure=morphology.disk(3))
                    final_extra_mask = np.logical_and(expanded_extra_mask, np.logical_not(extra_mask))
                    final_extra_mask = np.logical_and(final_extra_mask, not_island_borders)
                    is_detail_important_new[np.logical_and(final_extra_mask, near_important_point)] = True
                    # plot_value_grid(final_extra_mask, "final_extra_mask")
                    # plot_value_grid(np.logical_and(final_extra_mask, not_island_borders), "Fixed?")
                    # plot_value_grid(near_important_point, "near_important_point")
                    # plot_value_grid(is_detail_important_new, "detail")
                    if not obstacle.is_default_climbable:
                        # plot_value_grid(final_extra_mask, "final_extra_mask")
                        is_climbable_new[final_extra_mask] = False

        if self.is_debug_graph_printing_enabled:
            plot_terrain(self.map.Z, "Height before obstacle")
            plot_terrain(map_new.Z, "Height after obstacle")
            plot_value_grid(is_detail_important_new, "Is detail important after obstacle")

        items = []

        if ring_config.inner_solution:
            inside_inner_safety_region_mask = ring.r < ring_config.inner_safety_radius
            inner_solution_mask = np.logical_and(inner_mask, np.logical_not(inside_inner_safety_region_mask))
            inside_mid_z = ring.z < ring.mid_z
            inside_inner_obstacle = inner_mask
            only_inner_obstacle_weight_map = inner_obstacle_weight_map.copy()
            only_inner_obstacle_weight_map[inner_obstacle_weight_map == 1.0] = 0.0
            if ring_config.outer_obstacle:
                outer_solution_mask = np.logical_and(inside_mid_z, np.logical_not(inside_inner_obstacle))
            else:
                outer_solution_mask = np.logical_and(
                    ring.r < ring_config.outer_safety_radius, np.logical_not(inside_inner_obstacle)
                )
                outer_solution_mask = np.logical_and(
                    outer_solution_mask, np.logical_not(only_inner_obstacle_weight_map > 0.0)
                )
            outer_solution_mask = np.logical_and(outer_solution_mask, island_mask)

            if ring.config.height_mode == HeightMode.MIDPOINT_RELATIVE:
                map_new.interpolate_heights(
                    outer_solution_mask,
                    np.logical_and(only_inner_obstacle_weight_map > 0.0, island_mask),
                )

            new_items = ring_config.inner_solution.apply(
                rand,
                map_new,
                is_climbable_new,
                is_detail_important_new,
                only_inner_obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
                inner_point_for_inner_solution,
                outer_point_for_inner_solution,
            )
            items.extend(new_items)

        if ring_config.outer_solution:
            outside_mid_z = ring.z > ring.mid_z
            inside_outer_obstacle = inside_outer_obstacle_mask
            if ring_config.inner_obstacle:
                inner_solution_mask = np.logical_and(outside_mid_z, inside_outer_obstacle)
            else:
                inner_solution_mask = np.logical_and(ring.r > ring_config.inner_safety_radius, inside_outer_obstacle)
                assert False, "This really needs to be better tested!  Test before using."
            inner_solution_mask = np.logical_and(inner_solution_mask, island_mask)

            outside_outer_safety_region_mask = ring.r > ring_config.outer_safety_radius
            outer_solution_mask = np.logical_and(outer_mask, np.logical_not(outside_outer_safety_region_mask))

            only_outer_obstacle_weight_map = outer_obstacle_weight_map.copy()
            only_outer_obstacle_weight_map[outer_obstacle_weight_map == 1.0] = 0.0

            if ring.config.height_mode == HeightMode.MIDPOINT_RELATIVE:
                assert (
                    False
                ), "Not implemented. See above, but I think this needs to change to specify on which side heights should be blended"

            new_items = ring_config.outer_solution.apply(
                rand,
                map_new,
                is_climbable_new,
                is_detail_important_new,
                only_outer_obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
                inner_point_for_outer_solution,
                outer_point_for_outer_solution,
            )
            items.extend(new_items)

        if ring_config.dual_solution:
            # markers = [
            #     map_new.point_to_index(x)
            #     for x in [
            #         inner_point_for_inner_solution,
            #         outer_point_for_inner_solution,
            #         inner_point_for_outer_solution,
            #         outer_point_for_outer_solution,
            #     ]
            # ]

            inside_inner_safety_region_mask = ring.r < ring_config.inner_safety_radius
            inner_solution_mask = np.logical_and(inner_mask, np.logical_not(inside_inner_safety_region_mask))
            outside_outer_safety_region_mask = ring.r > ring_config.outer_safety_radius
            outer_solution_mask = np.logical_and(outer_mask, np.logical_not(outside_outer_safety_region_mask))

            # outer_obstacle_weight_map, inside_outer_obstacle_mask,

            # plot_value_grid(inner_obstacle_weight_map - outer_obstacle_weight_map, markers=markers)
            # print(ring.mid_z)
            # print(np.linalg.norm(outer_point_for_outer_solution - inner_point_for_inner_solution))
            # plot_value_grid(
            #     outer_solution_mask,
            #     "ARGGG",
            #     markers=markers,
            # )

            new_items = ring_config.dual_solution.apply(
                rand,
                map_new,
                is_climbable_new,
                is_detail_important_new,
                inner_obstacle_weight_map - outer_obstacle_weight_map,
                inner_solution_mask,
                outer_solution_mask,
                inner_point_for_inner_solution,
                outer_point_for_outer_solution,
            )
            items.extend(new_items)

        for item in items:
            self.add_item(item)
            # flatten near each of the items, otherwise you end up with some pretty impossible stuff
            map_new.radial_flatten(to_2d_point(item.position), item.get_offset() * 2.0, island_mask)

        # validate that nothing went horribly wrong:
        assert (
            map_new.Z[island_mask].min() > WORLD_RAISE_AMOUNT - 1000
        ), "Hmmm, seems like we accidentally connected to a point outside of the island, not good"

        extended_island_mask = morphology.dilation(island_mask, morphology.disk(6))
        self.is_climbable[extended_island_mask] = is_climbable_new[extended_island_mask]
        self.is_detail_important = is_detail_important_new
        self.map = map_new

        # keeps track of all of the places that we made into obstacles, so we can place things later
        if ring_config.inner_obstacle and ring_config.outer_obstacle:
            is_obstacle = (inner_obstacle_weight_map - outer_obstacle_weight_map) > 0.0
        elif ring_config.inner_obstacle:
            is_obstacle = np.logical_and(inner_obstacle_weight_map > 0.0, inner_obstacle_weight_map < 1.0)
            if not np.any(is_obstacle):
                is_obstacle = self.map.get_outline(inner_obstacle_weight_map, 1)
        else:
            is_obstacle = np.logical_and(outer_obstacle_weight_map > 0.0, outer_obstacle_weight_map < 1.0)
            if not np.any(is_obstacle):
                is_obstacle = self.map.get_outline(outer_obstacle_weight_map, 1)
        self.full_obstacle_mask = np.logical_or(self.full_obstacle_mask, is_obstacle)

        self.obstacle_zones.append((inner_mask, outer_mask))

        if self.is_debug_graph_printing_enabled:
            plot_value_grid(is_climbable_new, "New is climbable")
            plot_terrain(self.map.Z, "Terrain (after obstacle)")

    # # TODO: clearly a placeholder implementation for scenery right now...
    # def add_scenery(self, rand: np.random.Generator, biome_map: BiomeHeightMap):
    #     # create bushes randomly in a particular biome
    #     biome_id = 11
    #     self.items.extend(
    #         biome_map.get_random_points_in_biome(
    #             rand,
    #             config=SceneryConfig(
    #                 resource_file="res://scenery/bush.tscn",
    #                 biome_id=biome_id,
    #                 density=0.5,
    #                 border_distance=0.0,
    #                 border_mode=BorderMode.HARD,
    #             ),
    #             is_debug_graph_printing_enabled=True,
    #         )
    #     )
    #
    #     # create trees in a particular biome proportional to distance from the edge of the biome
    #     # we cap the distance so that it is uniform random as long as it is away from the border
    #     self.items.extend(
    #         biome_map.get_random_points_in_biome(
    #             rand,
    #             config=SceneryConfig(
    #                 resource_file="res://scenery/tree.tscn",
    #                 biome_id=biome_id,
    #                 density=0.2,
    #                 border_distance=5.0,
    #                 border_mode=BorderMode.LINEAR,
    #                 is_oriented_to_surface=True,
    #             ),
    #             is_debug_graph_printing_enabled=True,
    #         )
    #     )

    def generate_terrain(self, rand: np.random.Generator, biome_map: BiomeHeightMap) -> Terrain:
        is_climbable_fixed = self.is_climbable.copy()
        # plot_value_grid(is_climbable_fixed, "final is climbable")
        # if self.is_coast_unclimbable:
        #     is_climbable_fixed = np.logical_and(is_climbable_fixed, np.logical_not(biome_map.map.get_all_coast()))
        #     is_climbable_fixed[self.is_detail_important] = self.is_climbable[self.is_detail_important]
        return Terrain(
            biome_map,
            is_climbable_fixed,
            self.is_detail_important,
            self.config.point_density_in_points_per_square_meter,
            rand,
        )

    def add_building(self, building, mask: MapBoolNP):
        building = attr.evolve(building, id=len(self.building_by_id))
        self.building_by_id[building.id] = building
        self.full_obstacle_mask = np.logical_or(self.full_obstacle_mask, mask)
        return building

    def export(self, terrain: Terrain, output_folder: Path):
        if self.export_config.debug_visualization_config is not None:
            spawn_item = only([x for x in self.items if isinstance(x, SpawnPoint)])
            food_items = [x for x in self.items if isinstance(x, Food)]
            if len(food_items) == 0:
                food_items = [x for x in self.items if isinstance(x, Prey)]
            spawn_position = spawn_item.position
            food_distances = [(np.linalg.norm(spawn_position - x.position), x) for x in food_items]
            food_position = sorted(food_distances)[-1][1].position
            mid_position = (spawn_position + food_position) / 2.0
            spawn_to_goal_vec = food_position - spawn_position
            new_spawn_item = None
            for off_center in (1.0, 0.5, 0.0):
                for orientation in (1.0, -1.0):
                    sideways = np.array([-spawn_to_goal_vec[2], 0.0, spawn_to_goal_vec[0]]) * orientation * off_center
                    full_len = np.linalg.norm(spawn_to_goal_vec)
                    new_spawn = mid_position + sideways + UP_VECTOR * full_len / 2.0
                    if not self.map.region.contains_point_3d(new_spawn, epsilon=0.0):
                        continue
                    rand = np.random.default_rng(0)
                    new_spawn_item = get_spawn(self.entity_id_gen, rand, 0.0, new_spawn, mid_position)
                    # markers = [
                    #     self.map.point_to_index(to_2d_point(x))
                    #     for x in [food_position, mid_position, spawn_position, new_spawn_item.position]
                    # ]
                    # plot_value_grid(self.map.Z, markers=markers)
                    break
                if new_spawn_item is not None:
                    break
            assert new_spawn_item is not None, "Failed to find a point in the region to view from"
            self.items = [x for x in self.items if not isinstance(x, SpawnPoint)]
            self.items.append(new_spawn_item)

            # replace all items with markers
            points_by_type = defaultdict(list)
            new_items = []
            collected_types = tuple([Food, Prey, Predator, Log, Weapon, Boulder, Stone])
            for item in self.items:
                if isinstance(item, collected_types):
                    for dtype in collected_types:
                        if isinstance(item, dtype):
                            points_by_type[dtype.__name__].append(item.position)
                else:
                    new_items.append(item)
            SPAWN_COLOR = "#FF00F8"
            color_by_type_name = {
                Food.__name__: "#FF0000",
                Prey.__name__: "#FDFF00",
                Predator.__name__: "#AE00FF",
                Log.__name__: "#FF9E00",
                Weapon.__name__: "#000000",
                Boulder.__name__: "#FF0094",
                Stone.__name__: "#FF0094",
            }
            for type_name, points in points_by_type.items():
                color = color_by_type_name[type_name]
                for point in points:
                    new_items.append(
                        ColoredSphere(entity_id=self.entity_id_gen.get_next_id(), position=point, color=color)
                    )
            new_items.append(
                ColoredSphere(entity_id=self.entity_id_gen.get_next_id(), position=spawn_position, color=SPAWN_COLOR)
            )
            self.items = new_items

            if self.export_config.debug_visualization_config.is_2d_graph_drawn:
                marker_lists = []
                for type_name, points in points_by_type.items():
                    color = color_by_type_name[type_name]
                    markers = [self.map.point_to_index(to_2d_point(x)) for x in points]
                    marker_lists.append((markers, color))
                marker_lists.append(([self.map.point_to_index(to_2d_point(spawn_position))], SPAWN_COLOR))
                height_with_water = self.map.Z.copy()
                height_with_water[height_with_water < WATER_LINE] = self.map.Z.max(initial=1.0)
                tag = ""
                if self.config.is_indoor_only:
                    tag = "INDOOR"
                    height_with_water[:, :] = 0.5
                plot_value_grid_multi_marker(height_with_water, tag + str(output_folder), marker_lists)

        is_exported_with_absolute_paths = self.export_config.is_exported_with_absolute_paths
        biome_map = terrain.biome_map
        map = biome_map.map
        building_name_by_id = {}
        for i, building in self.building_by_id.items():
            building_name = f"building_{i}"
            building_name_by_id[i] = building_name
        start_tile: Optional[Path] = None
        assert map.region.x.size == map.region.z.size

        # safety check
        items_outside_world = [x for x in self.items if not map.region.contains_point_2d(x.point2d)]
        if len(items_outside_world) > 0:
            raise Exception("Some items were spawned outside of the world, should never happen")

        x_tile_size = map.region.x.size / self.config.x_tile_count
        z_tile_size = map.region.x.size / self.config.z_tile_count
        x_min = map.region.x.min_ge
        z_min = map.region.z.min_ge

        items = []
        items_by_tile_ids = {}
        trees_by_tile_ids_and_resource = {}
        flora_by_tile_ids_and_resource = {}
        for item in self.items:
            zx_tile_ids = item_to_tile_ids(item, self.map.region, self.config.x_tile_count, self.config.z_tile_count)
            if isinstance(item, Scenery):
                scenery_by_tile_ids_and_resource = (
                    trees_by_tile_ids_and_resource
                    if "trees/" in item.resource_file
                    else flora_by_tile_ids_and_resource
                )
                if zx_tile_ids not in scenery_by_tile_ids_and_resource:
                    scenery_by_tile_ids_and_resource[zx_tile_ids] = {}
                resource = item.resource_file
                if resource not in scenery_by_tile_ids_and_resource[zx_tile_ids]:
                    scenery_by_tile_ids_and_resource[zx_tile_ids][resource] = []
                scenery_by_tile_ids_and_resource[zx_tile_ids][resource].append(item)
            else:
                if zx_tile_ids not in items_by_tile_ids:
                    items_by_tile_ids[zx_tile_ids] = []
                items_by_tile_ids[zx_tile_ids].append(item)
                items.append(item)

        all_tile_ids = set()
        for x_idx in range(0, self.config.x_tile_count):
            for z_idx in range(0, self.config.z_tile_count):
                all_tile_ids.add((z_idx, x_idx))

        terrain_export_results = []
        spawn_point = None
        for x_idx in range(0, self.config.x_tile_count):
            for z_idx in range(0, self.config.z_tile_count):
                region = Region(
                    x=FloatRange(x_min + x_idx * x_tile_size, x_min + (x_idx + 1) * x_tile_size),
                    z=FloatRange(z_min + z_idx * z_tile_size, z_min + (z_idx + 1) * z_tile_size),
                )
                region_items = items_by_tile_ids.get((z_idx, x_idx), [])
                region_building_names = [
                    building_name_by_id[i] for i, x in self.building_by_id.items() if region.overlaps_region(x.region)
                ]
                fine_output_file = output_folder / f"tile_{x_idx}_{z_idx}.tscn"
                coarse_output_file = output_folder / f"distant_{x_idx}_{z_idx}.tscn"
                # extend the outside regions a tiny bit so that no triangles are left behind.
                adapted_region = region.epsilon_expand(
                    x_idx == 0,
                    x_idx == self.config.x_tile_count - 1,
                    z_idx == 0,
                    z_idx == self.config.z_tile_count - 1,
                )
                neighboring_and_this_tile_ids = set()
                for i in range(-self.config.tile_radius, self.config.tile_radius + 1):
                    other_tile_x = x_idx + i
                    if other_tile_x < 0 or other_tile_x >= self.config.x_tile_count:
                        continue
                    for j in range(-self.config.tile_radius, self.config.tile_radius + 1):
                        other_tile_z = z_idx + j
                        if other_tile_z < 0 or other_tile_z >= self.config.z_tile_count:
                            continue
                        neighboring_and_this_tile_ids.add((other_tile_z, other_tile_x))
                distant_tile_ids = set([x for x in all_tile_ids if x not in neighboring_and_this_tile_ids])
                terrain_export_results.append(
                    terrain.export(
                        fine_output_file,
                        coarse_output_file,
                        adapted_region,
                        x_idx,
                        z_idx,
                        self.config.tile_radius,
                        region_building_names,
                        trees_by_tile_ids_and_resource if not IS_ALL_SCENERY_IN_MAIN else {},
                        flora_by_tile_ids_and_resource if not IS_ALL_SCENERY_IN_MAIN else {},
                        distant_tile_ids,
                        neighboring_and_this_tile_ids,
                    )
                )
                spawn_items = [x for x in region_items if isinstance(x, SpawnPoint)]
                if len(spawn_items) > 0:
                    assert spawn_point is None, "Should only be one spawn point"
                    spawn_point = only(spawn_items)
                    start_tile = fine_output_file
        assert start_tile is not None, "No tiles contained a spawn point--that's not good!"

        starting_terrain_tile_path = (
            str(output_folder / start_tile.name) if is_exported_with_absolute_paths else "./" + start_tile.name
        )

        # export tscn file
        scene = ImprovedGodotScene()
        level_resource = scene.add_ext_resource(starting_terrain_tile_path, "PackedScene")
        terrain_manager_script = scene.add_ext_resource("res://terrain/terrain_manager.gd", "Script")

        if self.export_config.is_meta_data_exported:
            meta_output_path = output_folder / "meta.json"
            with open(meta_output_path, "w") as outfile:
                # spawn_point
                food = [x for x in items if isinstance(x, Food)]
                prey = [x for x in items if isinstance(x, Prey)]
                food_count = len(food) + len(prey)
                if food_count == 1:
                    only_food = only(food + prey)
                    total_distance = np.linalg.norm(only_food.position - spawn_point.position)
                else:
                    all_food = food + prey
                    total_distance = sum([np.linalg.norm(x.position - spawn_point.position) for x in all_food])
                meta_data = dict(
                    size_in_meters=self.config.size_in_meters,
                    food_count=food_count,
                    total_distance=total_distance,
                    is_visibility_required=spawn_point.is_visibility_required,
                )
                outfile.write(json.dumps(meta_data))

        main_output_path = output_folder / "main.tscn"

        with scene.use_tree() as tree:
            tree.root = GDNode("Avalon", type="Spatial")
            if biome_map.config.export_config is None or biome_map.config.export_config.is_sun_enabled:
                tree.root.add_child(
                    GDNode(
                        "Sun",
                        type="DirectionalLight",
                        properties={
                            "transform": get_transform_from_pitch_and_yaw(
                                *get_pitch_and_yaw_from_sky_params(biome_map.config.godot_sky_config)
                            ),
                            **biome_map.config.godot_sun_config,
                        },
                    )
                )
            camera_transform = GDObject(
                "Transform", 0.5, 0.594, -0.629, 0, 0.727, 0.686, 0.866, -0.343, 0.363, -6.525, 5.377, 4.038
            )
            tree.root.add_child(
                GDNode(
                    "Camera",
                    type="Camera",
                    properties={"transform": camera_transform},
                )
            )
            tree.root.add_child(
                _create_godot_world_environment(
                    scene, biome_map.config.godot_env_config, biome_map.config.godot_sky_config
                )
            )

            if IS_ALL_SCENERY_IN_MAIN:
                scenery_by_resource = {}
                for scenery_by_tile_ids_and_resource in (
                    trees_by_tile_ids_and_resource,
                    flora_by_tile_ids_and_resource,
                ):
                    for tile_id, inner_dict in scenery_by_tile_ids_and_resource.items():
                        for resource, scenery_list in inner_dict.items():
                            if resource not in scenery_by_resource:
                                scenery_by_resource[resource] = []
                            scenery_by_resource[resource].extend(scenery_list)
                    for resource, scenery_list in scenery_by_resource.items():
                        tree.root.add_child(
                            create_multimesh_instance(
                                scene,
                                "main",
                                scenery_list,
                                resource.replace(".tscn", ".res"),
                                biome_map.config.flora_config,
                            )
                        )

            data = np.logical_not(self.is_climbable)
            if data.shape[1] % 8 != 0:
                data = np.pad(data, pad_width=[(0, 0), (0, 8 - data.shape[1] % 8)])
            bytes = np.packbits(data, bitorder="little")

            boundary = self.config.size_in_meters / 2.0

            export_path = "."
            if is_exported_with_absolute_paths:
                export_path = f'"{str(output_folder)}"'
            if self.export_config.world_id is not None:
                export_path = f"res://worlds/{self.export_config.world_id}"

            terrain_manager = GDNode(
                "TerrainManager",
                type="Node",
                properties=dict(
                    script=terrain_manager_script.reference,
                    export_path=export_path,
                    x_tile_count=self.config.x_tile_count,
                    z_tile_count=self.config.z_tile_count,
                    tile_radius=self.config.tile_radius,
                    x_min=-boundary,
                    z_min=-boundary,
                    x_max=boundary,
                    z_max=boundary,
                    climb_map=GDObject("PoolByteArray", *list(bytes)),
                    climb_map_x=len(self.is_climbable[0, :]),
                    climb_map_y=len(self.is_climbable[:, 0]),
                ),
            )
            tiles_node = GDNode("tiles", type="Node")
            tiles_node.add_child(GDNode(str(start_tile.name).split(".")[0], instance=level_resource.reference.id))
            terrain_manager.add_child(tiles_node)

            building_group = GDNode("buildings", type="Node")
            terrain_manager.add_child(building_group)
            for i, building in self.building_by_id.items():
                building.export(scene, building_group, building_name_by_id[i])

            terrain_manager.add_child(GDNode("walls", type="Node"))
            collision_mesh_group = GDNode("terrain_collision_meshes", type="Node")

            # for testing if collision is faster with a single object
            if biome_map.config.export_config and biome_map.config.export_config.is_single_collision_mesh:
                (
                    global_triangle_indices_for_region,
                    region_triangles,
                    region_triangle_normals,
                    region_vertex_normals,
                    region_vertices,
                    region_colors,
                ) = terrain._get_export_data_for_region(map.region.epsilon_expand(True, True, True, True))
                terrain_export_results = [(region_triangles, region_vertices)]

            for i, (region_triangles, region_vertices) in enumerate(terrain_export_results):
                collision_mesh_group.add_child(_create_static_body(i, scene, region_triangles, region_vertices))
            terrain_manager.add_child(collision_mesh_group)
            tree.root.add_child(terrain_manager)

            tracker_node = GDNode("dynamic_tracker", type="Node")
            for item in items:
                tracker_node.add_child(item.get_node(scene))
            tree.root.add_child(tracker_node)

            # figure out all of the trees
            tree_by_resource = {}
            for tile_id, inner_dict in trees_by_tile_ids_and_resource.items():
                for resource, scenery_list in inner_dict.items():
                    if resource not in tree_by_resource:
                        tree_by_resource[resource] = []
                    tree_by_resource[resource].extend(scenery_list)

            # TODO: test whether we can scale staticbodies. I dont think so actually :(
            #  bc godot is buggy with scaling of collision shaapes
            # # figure out all of the trees that we're using and make their shapes
            # shapes_by_resource = {}
            # for resource, scenery_list in tree_by_resource.items():
            #     shapes_by_resource[resource] = _create_shape_sub_resource_for_tree(resource, scene)

            # add all collision shapes for the trees
            tree_collision_mesh_group = GDNode("tree_collision_meshes", type="Spatial")
            for i, (resource, scenery_list) in enumerate(tree_by_resource.items()):
                for j, tree_obj in enumerate(scenery_list):
                    tree_collider = _create_tree_shape(scene, tree_obj, i, j, self.biome_config.flora_config, resource)
                    tree_collision_mesh_group.add_child(tree_collider)
            tree.root.add_child(tree_collision_mesh_group)

            # is big enough that you dont really notice that it isn't infinite
            tree.root.add_child(create_ocean_node(scene, "large_ocean", ocean_size=1_000, offset=-2))

            # if we ONLY make a huge ocean, it has a rendering bug near the shore
            # so we also make a small one that is slightly higher
            # UNLESS it is a completely flat world (probably indoor-only), since we'd get overlapping planes
            world_size = round(max([map.region.x.size, map.region.z.size]) * 1.2)
            if not (map.Z == 0).all():
                tree.root.add_child(create_ocean_node(scene, "near_ocean", ocean_size=world_size, offset=-1))

            tree.root.add_child(
                _create_ocean_collision(
                    scene, "ocean", position=np.array([0, -1.5, 0]), size=np.array([world_size, 1.0, world_size])
                )
            )

        scene.write(str((main_output_path).absolute()))

    def plot_visibility(self, spawn_point: Point3DNP, point_offset: Point3DNP, markers):
        visibility_calculator = self.map.generate_visibility_calculator()
        visibility = np.zeros_like(self.map.Z)
        spawn_point_2d = np.array([spawn_point[0], spawn_point[2]])
        for i, j in np.ndindex(*self.map.Z.shape):
            x = self.map.X[i, j]
            y = self.map.Y[i, j]
            z = self.map.Z[i, j]
            # for the ocean
            if z < 0:
                z = 0.0
            point = np.array([x, z, y])
            if np.isclose(np.array([x, y]), spawn_point_2d).all():
                visibility[i, j] = 1.0
            else:
                visibility[i, j] = visibility_calculator.is_visible_from(
                    spawn_point + HALF_AGENT_HEIGHT_VECTOR, point + point_offset
                )
                # if (i, j) in markers:
                #     print(spawn_point + HALF_AGENT_HEIGHT_VECTOR, point + point_offset)
                #     print(visibility[i, j])
                # if (i, j) == (32, 22):
                #     print(point)
                #     visibility[i, j] = 1.0
                #     visibility_calculator.is_visible_from(spawn_point + HALF_AGENT_HEIGHT_VECTOR, point,
                #     is_plotted=True)
        plot_value_grid(visibility, markers=markers)
        return visibility

    # TODO: do we really need to do this copying here?
    def _apply_height_obstacle(
        self,
        mask: np.ndarray,
        ring: RingObstacle,
        island_mask: MapBoolNP,
        start_point: Optional[Point2DNP],
        end_point: Optional[Point2DNP],
    ) -> Tuple[HeightMap, np.ndarray]:
        map_new = self.map.copy()
        mask = mask.copy()
        mask[np.logical_not(island_mask)] = 0.0

        if (
            ring.config.height_mode == HeightMode.MIDPOINT_ABSOLUTE
            or ring.config.height_mode == HeightMode.MIDPOINT_RELATIVE
        ):
            if ring.config.inner_obstacle and ring.config.outer_obstacle:
                assert ring.config.height_mode == HeightMode.MIDPOINT_ABSOLUTE
                start_height = map_new.get_rough_height_at_point(start_point)
                end_height = map_new.get_rough_height_at_point(end_point)
                if ring.config.height < 0.0:
                    base_height = max([start_height, end_height])
                else:
                    base_height = min([start_height, end_height])
                # this is pretty gross, but required for current tasks...
                if ring.config.expansion_meters > 0.0:
                    expanded_mask = mask.copy()
                    cell_units = int(ring.config.expansion_meters * map_new.cells_per_meter) + 1
                    # print("cell units", cell_units)
                    nearby = morphology.dilation(expanded_mask > 0.0, morphology.disk(cell_units))
                    expanded_mask[nearby] = np.clip(expanded_mask[nearby], 0.001, None)
                    expanded_mask[np.logical_not(island_mask)] = 0.0
                else:
                    expanded_mask = mask
                map_new.Z[expanded_mask > 0.0] = base_height
                # plot_value_grid(expanded_mask)
                # plot_value_grid(mask > 0.0)
                # plot_value_grid(expanded_mask > 0.0)
                map_new.apply_height_mask(expanded_mask, ring.config.height, HeightMode.RELATIVE)
            else:
                # only makes sense if this is not double sided
                boolean_obstacle_mask = np.logical_and(mask > 0.0, mask < 1.0)
                if ring.config.height_mode == HeightMode.MIDPOINT_ABSOLUTE:
                    # set everything to the start height
                    start_height = map_new.get_rough_height_at_point(start_point)
                    map_new.Z[boolean_obstacle_mask] = start_height
                else:
                    start_z = ring.z[map_new.point_to_index(start_point)]
                    near_start_z = map_new.get_outline(ring.z < start_z, 1)
                    near_start_z = np.logical_and(near_start_z, island_mask)
                    # plot_value_grid(near_start_z, "Near mid z for crazy heightmap thing")
                    ring_indices = np.argwhere(near_start_z)
                    assert ring_indices.shape[-1] == 2
                    ring_points = np.stack(
                        [map_new.X[0, ring_indices[:, 1]], map_new.Y[ring_indices[:, 0], 0]], axis=1
                    )
                    ring_heights = map_new.get_heights(ring_points)
                    # interp = NearestNDInterpolator(ring_points, ring_heights)
                    # # set everything to the height at the nearest midpoint
                    # plot_value_grid(boolean_obstacle_mask, "boolean_obstacle_mask")
                    # map_new.Z[boolean_obstacle_mask] = interp(
                    #     map_new.X[boolean_obstacle_mask], map_new.Y[boolean_obstacle_mask]
                    # )
                    ring_thetas = ring.theta[ring_indices[:, 0], ring_indices[:, 1]]
                    sorted_tuples = sorted(tuple(x) for x in zip(ring_thetas, ring_heights))
                    first_tuple = sorted_tuples[0]
                    last_tuple = sorted_tuples[-1]
                    twopi = np.pi * 2.0
                    sorted_tuples = (
                        [(last_tuple[0] - twopi, last_tuple[1])]
                        + sorted_tuples
                        + [(first_tuple[0] + twopi, first_tuple[1])]
                    )
                    sorted_tuples = np.array(sorted_tuples)
                    # wrap with 2pi and pi versions of the endpoints
                    height_by_theta = interp1d(sorted_tuples[:, 0], sorted_tuples[:, 1])
                    for idx in np.argwhere(boolean_obstacle_mask):
                        idx = tuple(idx)
                        theta = ring.theta[idx]
                        map_new.Z[idx] = height_by_theta(theta)
                # apply the mask to apply the delta
                map_new.apply_height_mask(mask, ring.config.height, HeightMode.RELATIVE)
        else:
            map_new.apply_height_mask(mask, ring.config.height, ring.config.height_mode)
        is_climbable_new = self.is_climbable.copy()
        return map_new, is_climbable_new

    def _create_ring(
        self, rand: np.random.Generator, config: RingObstacleConfig, safety_radius_min: float, safety_radius_max: float
    ) -> RingObstacle:

        # print(safety_radius_min)
        # print(safety_radius_max)

        edge_config = config.edge

        # shift to account for the center point
        x_diff = config.center_point[0] - self.map.X
        y_diff = config.center_point[1] - self.map.Y

        # convert to polar coordinates
        r = np.sqrt(x_diff * x_diff + y_diff * y_diff)
        traversal_distance = np.sqrt(np.sum((config.center_point - config.traversal_point) ** 2))
        if traversal_distance < safety_radius_min or traversal_distance > safety_radius_max:
            raise Exception(
                f"Impossible to create a ring between {safety_radius_min} and {safety_radius_max} when the traversal point is {traversal_distance} meters away!"
            )
        normalized_r = r / traversal_distance
        theta = np.arctan2(y_diff, x_diff)

        # figure out where we cross
        traversal_indices = self.map.point_to_index(config.traversal_point)  # y, x

        outer_traversal_indices = None
        outer_mid_z = 0.0
        if config.inner_obstacle and config.outer_obstacle:
            outer_r = traversal_distance + config.chasm_bottom_size
            traversal_theta = theta[traversal_indices]
            outer_traversal_point = (
                -1.0 * np.array([outer_r * math.cos(traversal_theta), outer_r * math.sin(traversal_theta)])
                + config.center_point
            )
            outer_traversal_indices = self.map.point_to_index(outer_traversal_point)

        # create the "z" field of values.
        # we use various isolines from this z field to define the edges of the obstacle
        # do this repeatedly until we end up with a circle that fits between the safety margins
        for i in range(10):
            harmonics = edge_config.to_harmonics(traversal_distance)
            variation = create_harmonics(rand, theta, config=harmonics, is_normalized=False)
            z = np.sqrt(np.clip(normalized_r ** 2 + normalized_r * variation, 0.01, np.inf))

            mid_z = z[traversal_indices]
            z = (z / mid_z) * traversal_distance
            mid_z = traversal_distance
            if outer_traversal_indices:
                outer_mid_z = z[outer_traversal_indices]

                # plot_value_grid(
                #     np.logical_and(z < outer_mid_z, z > mid_z),
                #     "sigh world",
                #     markers=[traversal_indices, outer_traversal_indices],
                # )

            for margin in (0.1, 0.5):
                outline = np.bitwise_and(safety_radius_min + margin > r, r > safety_radius_min - margin)
                if np.any(outline):
                    break
            # plot_value_grid(outline)
            if np.any(outline):
                inner_z_at_safety_radius = z[outline].max()

                for margin in (0.1, 0.5):
                    outline = np.bitwise_and(safety_radius_max + margin > r, r > safety_radius_max - margin)
                    if np.any(outline):
                        break
                # plot_value_grid(outline)
                if np.any(outline):
                    outer_z_at_safety_radius = z[outline].min()
                    if mid_z - inner_z_at_safety_radius > 0.0 and outer_z_at_safety_radius - mid_z > 0.0:
                        break

            # if we get here, sigh... means that it was hard to make a circle like what we wanted, try again with something more circular
            edge_config = attr.evolve(
                edge_config, circularity=(edge_config.circularity + 1.0) / 2.0, noise=(edge_config.noise + 0.0) / 2.0
            )
            # TODO: track these warnings in a better way
            # logger.warning(f"Reduced noise and increased circularity: {edge_config.noise}, {edge_config.circularity}")
        else:
            raise ImpossibleWorldError("Tried to make a circle many times, bounds are too tight!")

        # safety check: if you cross too close, it will be sad (divide by zero below)
        if self.is_safe_mode_enabled:
            center_indices = self.map.point_to_index(config.center_point)  # y, x
            if center_indices == traversal_indices:
                raise Exception(
                    f"Cannot set traversal point and center point too close: center={config.center_point} and traversal={config.traversal_point}"
                )

        # figure out where the middle of the chasm is (the z value) and rescale z
        # such that it roughly puts z into world units (ie, meters)
        if self.is_debug_graph_printing_enabled:
            margin = 2.0
            outline = np.bitwise_and(mid_z + margin > z, z > mid_z - margin).astype(np.float)

            outline[r < safety_radius_min] = 0.5
            outline[r > safety_radius_max] = 0.5

            plot_value_grid(outline, title=f"Mid Z +/- {margin}m")

            plot_value_grid(z, title="Raw Z function")

        return RingObstacle(config=config, r=r, z=z, theta=theta, mid_z=mid_z, outer_mid_z=outer_mid_z)

    def _create_obstacle_masks(
        self,
        rand: np.random.Generator,
        ring: RingObstacle,
        config: HeightObstacle,
        island_mask: MapBoolNP,
        # whether debugging is enabled
        is_debug_graph_printing_enabled: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        The high level purpose of this function is to create obstacles (chasms, cliffs, and ridges).

        Using a chasm as a concrete example, we want to be able to finely control the exact distance
        that an agent would need to jump in order to cross, in addition to what fraction of the chasm
        is crossable at that width.

        Of course, we could simply draw a line or a circle for the chasm which is exactly the right size,
        but this looks extremely unnatural.

        The purpose of this function is to make it easy to create more natural looking chasms.

        At a high level, it works as follows:
        - Use spherical harmonic noise to make a wiggly circle that represents the bottom middle of the chasm
        - on either side, add purely additive spherical harmonic noise in order to vary the sizes of the chasm
            With respect to theta, this noise is varied smoothly such that within traversal_width / 2.0 of the
            traversal point, there is no noise, and at (traversal_width / 2.0) *  traversal_noise_interpolation_multiple
            the noise is the maximum value (which is defined by the safety_radius of the inner and outer rings)
        - linearly interpolate the transitions between the bottom and top of the chasm over edge_distance meters (roughly)
        """

        if config.is_inside_ring:
            assert config.traversal_theta_offset == 0.0, "Not allowed to vary theta inside the ring"

        # figure out where the traversal point is in polar coordinates
        traversal_offset = ring.config.traversal_point - ring.config.center_point
        traversal_theta = np.arctan2(traversal_offset[1], traversal_offset[0]) + config.traversal_theta_offset

        # figure out the interpolation bounds for our traversal
        # they are represented in "normalized theta delta" space, eg, 0.5 means "90 degrees away from the traversal
        # angle"
        traversal_distance = np.sqrt(np.sum((ring.config.center_point - ring.config.traversal_point) ** 2))
        if config.is_inside_ring:
            distance_delta = -config.traversal_length
        else:
            distance_delta = config.traversal_length + ring.config.chasm_bottom_size
        traversal_width_theta = np.arctan2(config.traversal_width / 2.0, traversal_distance + distance_delta)
        theta_width_interp_start = traversal_width_theta / np.pi
        theta_width_interp_end = min(config.traversal_noise_interpolation_multiple * theta_width_interp_start, 1.0)
        theta_interp_width = theta_width_interp_end - theta_width_interp_start

        # create the normalized theta space
        normalized_theta_delta = ring.theta.copy()
        normalized_theta_delta -= traversal_theta
        normalized_theta_delta[normalized_theta_delta < -np.pi] += 2 * np.pi
        normalized_theta_delta[normalized_theta_delta > np.pi] -= 2 * np.pi
        normalized_theta_delta[normalized_theta_delta > 0] *= -1.0
        normalized_theta_delta = (normalized_theta_delta / np.pi) + 1

        # create the mask (0 - 1) that can perform the actual interpolation
        theta_mask = normalized_theta_delta.copy()
        theta_mask[normalized_theta_delta >= theta_width_interp_end] = 1.0
        theta_mask[normalized_theta_delta < theta_width_interp_end] = 0.0
        theta_region = np.logical_and(
            normalized_theta_delta >= theta_width_interp_start, normalized_theta_delta < theta_width_interp_end
        )
        theta_mask[theta_region] = (
            normalized_theta_delta[theta_region] - theta_width_interp_start
        ) / theta_interp_width

        # create the output obstacle mask (which we will update below)
        # 0.0 will represent the region that we start in, and 1.0 the region of the obstacle
        obstacle_mask = ring.z.copy()

        # TODO: actually tie this constant to the other
        # this is intimately related to how many grid points per meter there are
        MARGIN = 0.25

        if config.is_inside_ring:
            # figure out the maximum z that could be added without exceeding the safety radius
            safety_radius = ring.config.inner_safety_radius
            outline = np.bitwise_and(safety_radius + MARGIN > ring.r, ring.r > safety_radius - MARGIN)
            if is_debug_graph_printing_enabled:
                plot_value_grid(outline, title="Inner safety radius")
            z_at_safety_radius = ring.z[outline].max()
            max_z_to_add = (ring.mid_z - config.traversal_length) - z_at_safety_radius
            if max_z_to_add <= 0.0:
                max_z_to_add = 0.01

            # calculate the noised up inner edge and bottom cutoffs
            harmonics = (
                create_harmonics(
                    rand, ring.theta, config.edge_config.to_harmonics(traversal_distance), is_normalized=True
                )
                * max_z_to_add
            )
            inner_theta_dependent_noise = harmonics * theta_mask
            inner_edge_z = (ring.mid_z - config.traversal_length) - inner_theta_dependent_noise
            inner_bottom_z = ring.mid_z - inner_theta_dependent_noise

            # everything less than the inner edge is the inner region, mask should be zero
            inner_ring_inner_region = ring.z < inner_edge_z
            obstacle_mask[inner_ring_inner_region] = 0.0

            if config.traversal_length > 0.01:
                # interpolate smoothly between inner_edge_z and inner_bottom_z
                interpolating_region = np.logical_and(ring.z >= inner_edge_z, ring.z < inner_bottom_z)
                obstacle_mask[interpolating_region] = ((ring.z - inner_edge_z) / (inner_bottom_z - inner_edge_z))[
                    interpolating_region
                ]
                if is_debug_graph_printing_enabled:
                    plot_value_grid(interpolating_region, title="Inner ring interpolation region")

            # if there is no outer ring, we're basically done--just clamp the rest of the values.
            # if outer_ring is None:
            obstacle_mask = np.clip(obstacle_mask, 0, 1)

            inner_region = inner_ring_inner_region
            outer_region = ring.z > inner_bottom_z
        else:
            # figure out the maximum z that could be added without exceeding the safety radius
            safety_radius = ring.config.outer_safety_radius
            outline = np.bitwise_and(safety_radius + MARGIN > ring.r, ring.r > safety_radius - MARGIN)
            if is_debug_graph_printing_enabled:
                plot_value_grid(outline, title="Outer safety radius")
            z_at_safety_radius = ring.z[outline].min()
            max_z_to_add = z_at_safety_radius - (ring.mid_z + config.traversal_length)
            # print(max_z_to_add)
            if max_z_to_add <= 0.0:
                max_z_to_add = 0.01

            # calculate the noised up outer edge and bottom cutoffs
            harmonics = (
                create_harmonics(
                    rand, ring.theta, config.edge_config.to_harmonics(traversal_distance), is_normalized=True
                )
                * max_z_to_add
            )
            outer_theta_dependent_noise = harmonics * theta_mask
            outer_bottom_z = ring.mid_z + outer_theta_dependent_noise
            outer_edge_z = (ring.mid_z + config.traversal_length) + outer_theta_dependent_noise

            # everything greater than the outer edge is the outer region, mask should be zero
            outer_ring_outer_region = ring.z > outer_edge_z
            obstacle_mask[outer_ring_outer_region] = 0.0

            if config.traversal_length > 0.01:
                # interpolate smoothly between outer_bottom_z and outer_edge_z
                interpolating_region = np.logical_and(ring.z >= outer_bottom_z, ring.z < outer_edge_z)
                obstacle_mask[interpolating_region] = ((outer_edge_z - ring.z) / (outer_edge_z - outer_bottom_z))[
                    interpolating_region
                ]
                if is_debug_graph_printing_enabled:
                    plot_value_grid(interpolating_region, title="Outer ring interpolation region")

            # if there is no inner ring, we're basically done
            # if inner_ring is None:
            # we just set anything in the inner region to one as well (since we're going to invert below)
            obstacle_mask[ring.z < ring.mid_z] = 1.0
            # invert and clip
            obstacle_mask = 1.0 - np.clip(obstacle_mask, 0, 1)

            inner_region = ring.z < outer_bottom_z
            outer_region = outer_ring_outer_region

        # # if both the inner and outer rings are define, make sure we set all of the values in the bottom of the chasm
        # # to 1.0
        # if inner_ring and outer_ring:
        #     chasm_bottom = np.logical_and(ring.z >= inner_bottom_z, ring.z < outer_bottom_z)
        #     obstacle_mask[chasm_bottom] = 1.0
        #     if is_debug_graph_printing_enabled:
        #         plot_value_grid(chasm_bottom, title="Chasm bottom")
        #     inner_region = inner_ring_inner_region
        #     outer_region = outer_ring_outer_region
        # elif inner_ring:
        #     inner_region = inner_ring_inner_region
        #     outer_region = ring.z > inner_bottom_z
        # else:
        #     assert outer_ring
        #     inner_region = ring.z < outer_bottom_z
        #     outer_region = outer_ring_outer_region

        # print the result if debugging is enabled
        if is_debug_graph_printing_enabled:
            plot_value_grid(inner_region, title="Inner region (full)")
            plot_value_grid(outer_region, title="Outer region (full)")
            plot_value_grid(obstacle_mask, title="Final mask (full)")

        not_island = np.logical_not(island_mask)
        obstacle_mask[not_island] = 0.0
        inner_region[not_island] = 0
        outer_region[not_island] = 0

        if is_debug_graph_printing_enabled:
            plot_value_grid(inner_region, title="Inner region (minus island)")
            plot_value_grid(outer_region, title="Outer region (minus island)")
            plot_value_grid(obstacle_mask, title="Final mask (minus island)")

        return obstacle_mask, inner_region, outer_region

    def reset_beaches(self, island: MapBoolNP):
        # remove beaches for anything that has gone above the height on our island
        self.special_biomes.swamp_mask[island] = self.map.Z[island] < self.biome_config.swamp_elevation_max
        self.special_biomes.fresh_water_mask[island] = self.map.Z[island] < 0.0
        self.special_biomes.beach_mask[island] = self.map.Z[island] < self.biome_config.max_shore_height
        squared_slope = self.map.get_squared_slope()
        cliffs = squared_slope > self.biome_config.force_cliff_square_slope
        cliffs = morphology.dilation(cliffs, morphology.disk(5))
        self.special_biomes.beach_mask[cliffs] = False

    def begin_height_obstacles(self, locations: WorldLocationData) -> WorldLocationData:
        self.map.raise_island(locations.island, WORLD_RAISE_AMOUNT)
        delta = UP_VECTOR * WORLD_RAISE_AMOUNT

        for item in self.items:
            item.position[1] = item.position[1] + WORLD_RAISE_AMOUNT

        for building_id in self.building_by_id.keys():
            building = self.building_by_id[building_id]
            new_position = attr.evolve(building.position, y=building.position.y + WORLD_RAISE_AMOUNT)
            self.building_by_id[building_id] = attr.evolve(building, position=new_position)

        return attr.evolve(locations, goal=locations.goal + delta, spawn=locations.spawn + delta)

    def end_height_obstacles(
        self,
        locations: WorldLocationData,
        is_accessible_from_water: bool,
        spawn_region: Optional[MapBoolNP] = None,
        is_spawn_region_climbable: bool = True,
    ):
        # player can't jump up here. Is a little higher than stricly necessary bc dont want to grab stuff
        MIN_INACCESSIBLE_HEIGHT = 3.0

        if is_accessible_from_water:
            sea_height = 0.0
        else:
            sea_height = MIN_INACCESSIBLE_HEIGHT
        # lower the island back down
        lowered_amount = self.map.lower_island(locations.island, sea_height)
        # lower all of the resulting positions as well
        fixed_items = []
        for item in self.items:
            item = attr.evolve(item, position=item.position.copy())
            item.position[1] = item.position[1] - lowered_amount
            # TODO: remove this! Will certainly be annoying to debug when we add things underwater...
            if item.position[1] < 0.0:
                item.position[1] = 0.5
            fixed_items.append(item)
        self.items = fixed_items
        for building_id in self.building_by_id.keys():
            building = self.building_by_id[building_id]
            new_position = attr.evolve(building.position, y=building.position.y - lowered_amount)
            self.building_by_id[building_id] = attr.evolve(building, position=new_position)
        locations.spawn[1] = locations.spawn[1] - lowered_amount
        locations.goal[1] = locations.goal[1] - lowered_amount

        if len(self.obstacle_zones) == 0:
            return

        assert len(self.obstacle_zones) >= 1
        spawn_region = self.get_spawn_zone(locations, spawn_region)

        assert spawn_region is not None

        # set unclimbability around the island (except our spawn zone, that can be climbable)
        is_climbable_fixed = self.is_climbable.copy()

        climbability_cell_radius = 4
        island_cliff_edge = self.map.get_outline(locations.island, climbability_cell_radius - 1)
        is_climbable_fixed[island_cliff_edge] = False
        # plot_value_grid(island_cliff_edge, "clif edge")
        # plot_value_grid(is_climbable_fixed, "is climb")
        # plot_value_grid(self.is_detail_important, "base is important")

        if is_spawn_region_climbable:
            # we want to be able to retry tasks by walking back to our spawn
            # but only makes sense to make the shore nearby unclimbable, not all cliffs
            # since otherwise tasks like climb and stack, which start you in a pit, are broken
            near_spawn_climbable = np.logical_and(
                spawn_region,
                np.logical_not(
                    morphology.dilation(self.full_obstacle_mask, morphology.disk(climbability_cell_radius + 1))
                ),
            )
            shore_mask = self.map.get_outline(self.map.get_land_mask(), 4)
            near_spawn_climbable = np.logical_and(near_spawn_climbable, shore_mask)
            near_spawn_climbable = morphology.dilation(near_spawn_climbable, morphology.disk(climbability_cell_radius))
            is_climbable_fixed[near_spawn_climbable] = True

        # no matter what, dont mess up important climbability
        # print(is_accessible_from_water)
        if not is_accessible_from_water:
            is_climbable_fixed[self.is_detail_important] = self.is_climbable[self.is_detail_important]

        if self.is_debug_graph_printing_enabled:
            plot_value_grid(is_climbable_fixed, "no but for realz")

        self.is_climbable = is_climbable_fixed

        self.reset_beaches(locations.island)

    def get_spawn_zone(
        self, locations: WorldLocationData, spawn_region: Optional[MapBoolNP] = None
    ) -> Optional[MapBoolNP]:
        if len(self.obstacle_zones) == 1:
            # find the closest one to the spawn
            zone_a, zone_b = only(self.obstacle_zones)
            if zone_a is None:
                spawn_region = zone_b
            elif zone_b is None:
                spawn_region = zone_a
            else:
                dist_sq_map = self.map.get_dist_sq_to(to_2d_point(locations.spawn))
                spawn_region = zone_a if dist_sq_map[zone_a].min() < dist_sq_map[zone_b].min() else zone_b
        return spawn_region

    def get_critical_distance(
        self,
        locations: WorldLocationData,
        min_distance: float,
        desired_distance: Optional[float] = None,
    ) -> Optional[float]:
        max_distance = locations.get_2d_spawn_goal_distance() - 2 * DEFAULT_SAFETY_RADIUS - min_distance
        if max_distance < 0.0:
            self.map.log_simplicity_warning("World is too small for even the smallest obstacle of this type")
            return None
        if desired_distance is None:
            return max_distance
        if max_distance < desired_distance:
            self.map.log_simplicity_warning(
                f"The requested obstalce is too far for the space available: {desired_distance} jump does not fit "
                f"because goal and spawn are {locations.get_2d_spawn_goal_distance()} away (and there needs to be "
                f"room for margins)"
            )
            return max_distance
        return desired_distance

    def mask_flora(self, mask: MapFloatNP):
        self.flora_mask *= mask


def create_ocean_node(scene: ImprovedGodotScene, name: str, ocean_size: int, offset: float):
    ocean_location = np.array([0, offset, 0])
    ocean_node = GDNode(
        name,
        type="MeshInstance",
        properties={
            "transform": GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, *ocean_location),
            "mesh": scene.add_sub_resource(
                "CubeMesh",
                material=scene.add_ext_resource("res://materials/ocean.material", "Material").reference,
                size=GDVector3(ocean_size, 2.0, ocean_size),
            ).reference,
            "material/0": None,
        },
    )
    return ocean_node


def item_to_tile_ids(item: Item, world_region: Region, x_tile_count: int, z_tile_count: int) -> Tuple[float, float]:
    if x_tile_count == 1:
        assert z_tile_count == 1, "lazy implementation"
        return (0, 0)
    max_x_index = x_tile_count - 1
    one_cell_x = world_region.x.size / (max_x_index)
    x = round((item.position[0] - world_region.x.min_ge) / one_cell_x)
    max_y_index = z_tile_count - 1
    one_cell_y = world_region.z.size / (max_y_index)
    y = round((item.position[2] - world_region.z.min_ge) / one_cell_y)
    return clamp(y, 0, max_y_index), clamp(x, 0, max_x_index)


def get_spawn(
    entity_id_gen: IdGenerator,
    rand: np.random.Generator,
    difficulty: float,
    spawn_location: np.ndarray,
    target_location: np.ndarray,
):
    direction_to_target = normalized(target_location - spawn_location)
    # noinspection PyTypeChecker
    target_yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
    spawn_view_yaw = (target_yaw + rand.uniform(-180, 180) * difficulty) % 360
    # don't start with super weird pitches
    # noinspection PyTypeChecker

    dist_to_target_2d = np.linalg.norm(
        np.array([target_location[0], target_location[2]]) - np.array([spawn_location[0], spawn_location[2]])
    )
    eye_location = spawn_location + HALF_AGENT_HEIGHT_VECTOR
    height_dist_to_target = target_location[1] - eye_location[1]
    target_pitch = np.arctan2(height_dist_to_target, dist_to_target_2d) * 180.0 / np.pi
    spawn_view_pitch = np.clip(target_pitch + difficulty * rand.uniform(-60, 60), -70, 70)
    return SpawnPoint(
        entity_id_gen.get_next_id(),
        position=spawn_location,
        yaw=spawn_view_yaw,
        pitch=float(spawn_view_pitch),
    )


def get_random_positions_along_path(
    visible_locations: Point3DListNP,
    start: Point3DNP,
    end: Point3DNP,
    target_location_distribution,
    rand: np.random.Generator,
    count: int,
) -> Point3DListNP:
    """Difficulty scales with how far away the point is from the straight line path between start and end"""
    path_length = np.linalg.norm(start - end)
    start_point = (start[0], start[2])
    end_point = (end[0], end[2])
    location_weights = np.array(
        [
            target_location_distribution.pdf(signed_line_distance((x[0], x[2]), start_point, end_point, path_length))
            for x in visible_locations
        ]
    )
    location_weights /= location_weights.sum()
    return rand.choice(visible_locations, p=location_weights, size=count)


def get_difficulty_based_value(
    difficulty: float, min_val: float, max_val: float, variability: float, rand: np.random.Generator
) -> float:
    total_delta = max_val - min_val
    delta = variability * total_delta
    remainder = total_delta - delta
    return min_val + (remainder * difficulty) + (rand.uniform() * delta)


def normalized(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def build_building(config: BuildingConfig, building_id: int, rand: np.random.Generator) -> Building:
    rng = random.Random(rand.random())
    stories = []
    for story_num in range(config.story_count):
        footprint_below = None
        if story_num != 0:
            footprint_below = stories[story_num - 1].footprint.copy()
        footprint = config.footprint_builder.build(
            config.width, config.length, story_num, rand, footprint_below=footprint_below
        )

        rooms = config.room_builder.build(footprint, rand)
        hallways = config.hallway_builder.build(rooms, rng)
        story = Story(
            story_num,
            width=config.width,
            length=config.length,
            footprint=footprint,
            rooms=rooms,
            hallways=hallways,
            has_ceiling=True,
        )
        stories.append(story)

    # todo: make clear link_stories is mutating
    if config.story_linker is not None:
        linked_stories, links = config.story_linker.link_stories(stories, rng)
    else:
        linked_stories = stories
        links = []

    if config.obstacle_builder is not None:
        obstacles = config.obstacle_builder.generate(linked_stories, rand)
        config.obstacle_builder.apply(linked_stories, obstacles)

    if config.entrance_builder is not None:
        stories_with_entrances = config.entrance_builder.build(linked_stories, rand)
    else:
        stories_with_entrances = stories

    if config.window_builder is not None:
        stories_with_windows = config.window_builder.build(stories_with_entrances, rand, config.aesthetics)
    else:
        stories_with_windows = stories_with_entrances

    return Building(
        id=building_id,
        position=Vector3(0, 0, 0),
        stories=stories_with_windows,
        story_links=links,
        is_climbable=config.is_climbable,
    )


def _get_angle_value_for_difficulty(target_yaw: float, difficulty: float, rand: np.random.Generator):
    possible_yaws = range(0, 360, 10)
    difficulty_degrees = difficulty * 180
    difficulty_adjusted_target_yaw = round(target_yaw + rand.choice([1, -1]) * difficulty_degrees) % 360
    yaw_distribution = stats.norm(difficulty_adjusted_target_yaw, 5)
    angle_weights = np.array([yaw_distribution.pdf(x) for x in possible_yaws])
    angle_weights /= angle_weights.sum()
    spawn_view_yaw = rand.choice(possible_yaws, p=angle_weights)
    return spawn_view_yaw


def _get_spawn(
    entity_id_gen: IdGenerator,
    rand: np.random.Generator,
    difficulty: float,
    spawn_location: np.ndarray,
    target_location: np.ndarray,
):
    direction_to_target = normalized(target_location - spawn_location)
    # noinspection PyTypeChecker
    target_yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
    spawn_view_yaw = _get_angle_value_for_difficulty(target_yaw, difficulty, rand)
    # don't start with super weird pitches
    # noinspection PyTypeChecker

    dist_to_target_2d = np.linalg.norm(
        np.array([target_location[0], target_location[2]]) - np.array([spawn_location[0], spawn_location[2]])
    )
    eye_location = spawn_location + HALF_AGENT_HEIGHT_VECTOR
    height_dist_to_target = target_location[1] - eye_location[1]
    target_pitch = np.arctan2(height_dist_to_target, dist_to_target_2d) * 180.0 / np.pi
    pitch_variance = 100
    spawn_view_pitch = np.clip(target_pitch + difficulty * pitch_variance * rand.uniform(), -70, 70)
    return SpawnPoint(
        entity_id_gen.get_next_id(),
        position=spawn_location,
        yaw=spawn_view_yaw,
        pitch=float(spawn_view_pitch),
    )


def random_point_within_radius(rand: np.random.Generator, point: Point2DNP, radius: float) -> Point2DNP:
    assert_isinstance(point, Point2DNP)
    r = rand.uniform() * radius
    theta = rand.uniform() * np.pi * 2.0
    x = math.cos(theta) * r
    y = math.sin(theta) * r
    return np.array([x + point[0], y + point[1]])


def get_pitch_and_yaw_from_sky_params(sky_config: Dict[str, Any]):
    pitch = (180.0 - sky_config.get("sun_latitude", 170.0)) * -1.0
    yaw = sky_config.get("sun_longitude", 0.0) * -1.0
    return pitch, yaw


def get_transform_from_pitch_and_yaw(pitch, yaw):
    yaw_rotation = Rotation.from_euler("y", yaw, degrees=True)
    pitch_rotation = Rotation.from_euler("x", pitch, degrees=True)
    rotation = (yaw_rotation * pitch_rotation).as_matrix().flatten()
    return GDObject("Transform", *rotation, 0, 0, 0)


def _create_godot_world_environment(
    scene: ImprovedGodotScene, env_config: Dict[str, Any], sky_config: Dict[str, Any]
) -> GDNode:
    sky_resource = scene.add_sub_resource("ProceduralSky", **sky_config)
    full_env_config = {**env_config}
    full_env_config["background_sky"] = sky_resource.reference
    env_resource = scene.add_sub_resource("Environment", **full_env_config)
    return GDNode(
        "WorldEnvironment",
        type="WorldEnvironment",
        properties={
            "environment": env_resource.reference,
        },
    )


def _create_ocean_collision(scene: ImprovedGodotScene, name: str, position: np.ndarray, size: np.ndarray) -> GDNode:
    transform = GDObject("Transform", 1, 0, 0, 0, 1, 0, 0, 0, 1, *position)
    root = GDNode(
        name,
        type="StaticBody",
        properties={
            "transform": transform,
        },
    )
    root.add_child(
        GDNode(
            f"{name}_collision",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape", resource_local_to_scene=True, extents=GDVector3(*size / 2.0)
                ).reference
            },
        )
    )

    return root


def _create_tree_shape(
    scene: ImprovedGodotScene, tree_obj, i, j, flora_config: Dict[str, FloraConfig], resource: str
) -> GDNode:
    config, resource_name = get_flora_config_by_file(flora_config, resource)
    size = config.collision_extents * tree_obj.scale * config.default_scale
    offset_position = tree_obj.position.copy()
    offset_position[1] += config.collision_extents[1] * tree_obj.scale[1] * config.default_scale
    transform = GDObject("Transform", *tree_obj.rotation, *offset_position)
    tree_collider = GDNode(f"tree_{i}_collision_{j}", type="StaticBody", properties={"transform": transform})
    tree_collider.add_child(
        GDNode(
            f"tree_{i}_collision_{j}_shape",
            type="CollisionShape",
            properties={
                "shape": scene.add_sub_resource(
                    "BoxShape", resource_local_to_scene=True, extents=GDVector3(*size)
                ).reference
            },
        )
    )

    return tree_collider
