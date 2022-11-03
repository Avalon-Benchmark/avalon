import math
from typing import List
from typing import Optional
from typing import Tuple
from typing import cast

import attr
import numpy as np
from loguru import logger
from scipy import ndimage
from skimage import morphology

from avalon.common.errors import SwitchError
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.constants import IS_DEBUGGING_IMPOSSIBLE_WORLDS
from avalon.datagen.world_creation.constants import WATER_LINE
from avalon.datagen.world_creation.debug_plots import plot_terrain
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import MapFloatNP
from avalon.datagen.world_creation.types import Point2DListNP
from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.worlds.height_map import HeightMap


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
        path_length_so_far = 0.0
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

            # this is just a stupid way to make a gradient between the start and end point...
            center = 1000 * (start_point - end_point) + end_point
            gradient_mask = np.sqrt(map.get_dist_sq_to(center))
            gradient_mask -= gradient_mask[map.point_to_index(start_point)]
            end_dist_val = gradient_mask[map.point_to_index(end_point)]
            if end_dist_val > 0.00001:
                gradient_mask /= end_dist_val

                gradient_mask = np.clip(gradient_mask, start_val, end_val)
                line_seg_distances = map.get_lineseg_distances(start_point, end_point)
                close_enough = line_seg_distances < (path_width / 2.0)
                mask[close_enough] = gradient_mask[close_enough]

                if is_debugging:
                    logger.debug("Path segment:")
                    logger.debug((start_val, end_val))
                    logger.debug((start_point, end_point))
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
            dist = float(np.linalg.norm(self.points[i] - self.points[i - 1]))
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

        start_to_end_2d_dist = float(np.linalg.norm(start_point - end_point))
        if self.min_desired_2d_length is None:
            min_len = start_to_end_2d_dist
        else:
            min_len = self.min_desired_2d_length
        if self.max_desired_2d_length is None:
            max_len = 3 * min_len
        else:
            max_len = self.max_desired_2d_length

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
        # logger.debug(f"{height_delta=}")

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
                edge_mask = ndimage.binary_dilation(edge_mask, structure=morphology.disk(1))
                edge_mask = np.logical_and(edge_mask, obstacle_weight_map > 0.0)
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
    rand: np.random.Generator,
    map: HeightMap,
    mask: np.ndarray,
    overall_len: float,
    start_point: Point2DNP,
    end_point: Point2DNP,
    width: float,
) -> Optional[PointPath]:
    start_len = overall_len * 0.5
    end_len = start_len
    for mid_point in _get_circle_intersections(start_point, start_len, end_point, end_len):
        is_start_path_clear = _check_path(map, mask, start_point, mid_point, width, shrink_start=True)
        is_end_path_clear = _check_path(map, mask, start_point, mid_point, width, shrink_end=True)
        if is_start_path_clear and is_end_path_clear:
            return PointPath(np.array([start_point, mid_point, end_point]))
    return None


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
    shrink_start: bool = False,
    shrink_end: bool = True,
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
            path_mask = map.get_dist_sq_to(midpoint) < width**2
        else:
            start_to_end_vec = (end_point - start_point) / start_to_end_dist
            start_point = start_point + start_to_end_vec * (width / 2.0)
            end_point = end_point + -1.0 * start_to_end_vec * (width / 2.0)
            line_seg_distances = map.get_lineseg_distances(start_point, end_point)
            path_mask = line_seg_distances < width / 2.0

    # plot_value_grid(path_mask, "mask", markers=[map.point_to_index(x) for x in [start_point, end_point]])

    if np.any(np.logical_and(path_mask, np.logical_not(mask))):
        # plot_value_grid(mask, "mask", markers=[map.point_to_index(x) for x in [start_point, end_point]])
        # plot_value_grid(line_seg_distances < width, "line_seg_distances")
        # plot_value_grid(np.logical_and(line_seg_distances < width, np.logical_not(mask)), "failure points")
        # logger.debug("Rejected because not in mask")
        return False
    return True


def _make_triangular_mask(points_2d: Point2DListNP, start: Point2DNP, end: Point2DNP, width: float) -> MapBoolNP:
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

    return cast(MapBoolNP, triangle_mask)


def _bulk_signed_line_distance(points: np.ndarray, a: Point2DNP, b: Point2DNP) -> np.ndarray:
    ab_dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    return cast(
        np.ndarray, (((b[0] - a[0]) * (a[1] - points[:, :, 1])) - ((a[0] - points[:, :, 0]) * (b[1] - a[1]))) / ab_dist
    )


def _get_circle_intersections(
    center_a: Point2DNP, radius_a: float, center_b: Point2DNP, radius_b: float
) -> List[Point2DNP]:
    result = _get_intersections(center_a[0], center_a[1], radius_a, center_b[0], center_b[1], radius_b)
    if result is None:
        return []
    x1, y1, x2, y2 = result
    if abs(x1 - x2) < 0.001 and abs(y1 - y2) < 0.001:
        return [np.array([x1, y1])]
    return [np.array([x1, y1]), np.array([x2, y2])]


def _get_intersections(
    x0: float, y0: float, r0: float, x1: float, y1: float, r1: float
) -> Optional[Tuple[float, float, float, float]]:
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
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(r0**2 - a**2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d

        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        return (x3, y3, x4, y4)
