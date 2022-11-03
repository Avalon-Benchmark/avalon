from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

import attr
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation
from skimage import morphology

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.constants import IS_DEBUGGING_IMPOSSIBLE_WORLDS
from avalon.datagen.world_creation.constants import WATER_LINE
from avalon.datagen.world_creation.debug_plots import plot_terrain
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.entities.animals import Animal
from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import MapFloatNP
from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.utils import normalized
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.height_map import HeightMap
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class HeightSolution:
    paths: Tuple[HeightPath, ...] = tuple()
    inside_items: Tuple[Union[Animal, Tool], ...] = tuple()
    outside_items: Tuple[Union[Animal, Tool], ...] = tuple()
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
    ) -> Tuple[Union[Tool, Animal], ...]:
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
            # logger.debug("mask components")
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
        near_index = cast(Tuple[int, int], tuple(np.unravel_index(np.argmin(dist_sq), dist_sq.shape)))
        return map.index_to_point_2d(near_index)

    def _place_items(
        self,
        rand: np.random.Generator,
        map: HeightMap,
        items: Tuple[Union[Animal, Tool], ...],
        randomization_distance: float,
        item_radius: float,
        inside_solution_point: Point2DNP,
        outside_solution_point: Point2DNP,
        solution_mask: MapBoolNP,
    ) -> Tuple[List[Union[Animal, Tool]], MapBoolNP]:
        solution_forward = normalized(outside_solution_point - inside_solution_point)
        solution_yaw = np.arctan2(solution_forward[1], solution_forward[0])

        item_size_radius_in_grid_unit = round(item_radius * map.cells_per_meter) + 1
        # reduce the size of the solution mask so we dont end up inside of walls
        reduced_solution_mask = np.logical_not(
            ndimage.binary_dilation(
                np.logical_not(solution_mask), structure=morphology.disk(item_size_radius_in_grid_unit)
            )
        )

        transformed_items: List[Any] = []
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
            old_rotation = Rotation.from_matrix(item.rotation.copy().reshape((3, 3)))
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
                for i in range(len(transformed_items)):
                    item = transformed_items[i]
                    item_pos_2d = to_2d_point(item.position)
                    item_dist_sq = map.get_dist_sq_to(item_pos_2d)
                    item_jitter_mask = item_dist_sq < randomization_distance**2
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
                            cast(Tuple[int, int], tuple(rand.choice(np.argwhere(spawn_points_without_other_items))))
                        )
                        old_height = map.get_rough_height_at_point(item_pos_2d)
                        new_position_2d = np.array([new_point_2d[0], new_point_2d[1]])
                        new_height = map.get_rough_height_at_point(new_position_2d)
                        height_offset = new_height - old_height

                        position_new = item.position.copy()
                        position_new[0] = new_point_2d[0]
                        position_new[2] = new_point_2d[1]
                        position_new[1] = position_new[1] + height_offset
                        item = attr.evolve(item, position=position_new)
                        transformed_items[i] = item

        # verify that all items have been moved out of the ocean
        for item in transformed_items:
            if map.get_rough_height_at_point(to_2d_point(item.position)) < WATER_LINE:
                raise ImpossibleWorldError(
                    "Some items could not find spawn points on the land. Try increasing their randomization distance or solution zone size"
                )

        # set is_detail_important around these items as well
        important_points = add_detail_near_items(map, transformed_items, item_radius)

        return transformed_items, important_points


def add_detail_near_items(map: HeightMap, items: Sequence[Item], item_radius: float):
    item_size_radius_in_grid_unit = round(item_radius * map.cells_per_meter) + 1
    # set is_detail_important around these items as well
    new_detail_areas = np.zeros_like(map.Z, dtype=np.bool_)
    for item in items:
        item_pos_2d = to_2d_point(item.position)
        new_detail_areas[map.point_to_index(item_pos_2d)] = True

    importance_radius = max([2, item_size_radius_in_grid_unit])
    return ndimage.binary_dilation(new_detail_areas, structure=morphology.disk(importance_radius))
