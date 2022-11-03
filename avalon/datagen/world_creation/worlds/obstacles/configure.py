import math
from typing import Optional

import numpy as np
from loguru import logger

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.constants import DEFAULT_SAFETY_RADIUS
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.types import HeightMode
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.obstacles.harmonics import EdgeConfig
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.obstacles.ring_obstacle import HeightObstacle
from avalon.datagen.world_creation.worlds.obstacles.ring_obstacle import RingObstacleConfig
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.utils import add_offsets
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


def make_ring(
    rand: np.random.Generator,
    difficulty: float,
    world: World,
    locations: WorldLocations,
    gap_distance: float,
    height: float,
    traversal_width: float,
    constraint: Optional[CompositionalConstraint],
    is_inside_climbable: bool = True,
    is_outside_climbable: bool = False,
    inner_solution: Optional[HeightSolution] = None,
    outer_solution: Optional[HeightSolution] = None,
    dual_solution: Optional[HeightSolution] = None,
    inner_traversal_length: float = 0.0,
    outer_traversal_length: float = 0.0,
    safety_radius: float = DEFAULT_SAFETY_RADIUS,
    traversal_noise_interpolation_multiple: float = 2.0,
    is_single_obstacle: bool = False,
    probability_of_centering_on_spawn: Optional[float] = 0.5,
    max_additional_radius_multiple: float = 2.0,
    height_mode: HeightMode = HeightMode.RELATIVE,
    extra_safety_radius: float = 0.0,
    expansion_meters: float = 0.0,
    detail_radius: float = 20.0,
    traversal_difficulty_meters: float = 10.0,
) -> RingObstacleConfig:
    is_debugging = False

    if constraint is None:
        # should only be centered on you half the time
        assert probability_of_centering_on_spawn is not None
        is_centered_on_spawn = rand.uniform() < probability_of_centering_on_spawn
        if is_centered_on_spawn:
            center_point = to_2d_point(locations.spawn)
            other_point = to_2d_point(locations.goal)
        else:
            if is_debugging:
                logger.debug("Is reversed")
            center_point = to_2d_point(locations.goal)
            other_point = to_2d_point(locations.spawn)

            if is_single_obstacle:
                height *= -1
            else:
                # swap these because this is not the canonical ordering
                is_inside_climbable, is_outside_climbable = is_outside_climbable, is_inside_climbable
                inner_solution, outer_solution = outer_solution, inner_solution
                inner_traversal_length, outer_traversal_length = outer_traversal_length, inner_traversal_length

            if inner_solution is not None:
                inner_solution = inner_solution.reverse()

            if outer_solution is not None:
                outer_solution = outer_solution.reverse()

            if dual_solution is not None:
                dual_solution = dual_solution.reverse()
    else:
        center_point = to_2d_point(constraint.center)
        other_point = to_2d_point(locations.goal)

        if constraint.is_height_inverted:
            height *= -1

    middle_point = (center_point + other_point) / 2.0

    if constraint is None:
        # add some variation for the radius of the circle
        radius_scale = rand.uniform(1, max_additional_radius_multiple)
        center_point = middle_point + (center_point - middle_point) * radius_scale
    else:
        radius_scale = 1.0

    # and find a randomized traversal point that would actually work
    spawn_radius = np.linalg.norm(center_point - to_2d_point(locations.spawn))
    food_radius = np.linalg.norm(center_point - to_2d_point(locations.goal))
    min_traversal_radius = (
        min([spawn_radius, food_radius]) + safety_radius + inner_traversal_length + extra_safety_radius  # type: ignore
    )
    max_traversal_radius = max([spawn_radius, food_radius]) - (  # type: ignore
        safety_radius + outer_traversal_length + gap_distance + extra_safety_radius
    )
    radius_diff = max_traversal_radius - min_traversal_radius
    if radius_diff < 0:
        raise ImpossibleWorldError("Too small to make the requested ring")
    if is_debugging:
        logger.debug(f"Radius diff is {radius_diff}")
        logger.debug(spawn_radius)
        logger.debug(food_radius)
        logger.debug(min_traversal_radius)
        logger.debug(max_traversal_radius)
        logger.debug(safety_radius)
        logger.debug(inner_traversal_length)
        logger.debug(radius_scale)
    min_selectable_radius = min_traversal_radius + radius_diff * 0.3
    max_selectable_radius = max_traversal_radius - radius_diff * 0.3
    r_squared = world.map.get_dist_sq_to(center_point)
    acceptable_radius_mask = np.logical_and(
        r_squared < max_selectable_radius**2, r_squared > min_selectable_radius**2
    )
    if is_debugging:
        plot_value_grid(r_squared < max_selectable_radius**2, "Less than max radius")
        plot_value_grid(r_squared > min_selectable_radius**2, "Greater than min radius")
        plot_value_grid(acceptable_radius_mask, "Acceptable radius mask")

    max_dist = 2.0 + traversal_difficulty_meters * difficulty

    temp = middle_point - center_point
    middle_theta = np.arctan2(temp[1], temp[0])
    r = (min_traversal_radius + max_traversal_radius) / 2.0
    actual_middle = center_point + np.array([r * math.cos(middle_theta), r * math.sin(middle_theta)])

    # logger.debug(middle_theta)
    # logger.debug(middle_point)
    # logger.debug(center_point)
    # logger.debug(r)
    # logger.debug(actual_middle)

    if constraint is None:
        dist_sq_to_middle_point = world.map.get_dist_sq_to(actual_middle)
        near_traversal_center_mask = dist_sq_to_middle_point < max_dist**2
    else:
        near_traversal_center_mask = constraint.traversal_mask
    possible_points = np.logical_and(
        locations.island, np.logical_and(near_traversal_center_mask, acceptable_radius_mask)
    )
    if not np.any(possible_points):
        traversal_indices_array = None
    else:
        # biasing away from the ocean so we end up with failing paths less often
        if constraint is None:
            # TODO: max points happens to be reduced under us, so annoying :(
            water_dist = world.map.get_water_distance(
                rand, is_fresh_water_included_in_moisture=True, max_points=600, for_points=possible_points
            )
            if water_dist is not None:
                water_dist_for_points = water_dist[possible_points]
                power_to_raise = 8
                weights = (water_dist_for_points / water_dist_for_points.max()) ** power_to_raise
                weights /= weights.sum()
            else:
                weights = None
            traversal_indices_array = rand.choice(np.argwhere(possible_points), p=weights)

            # pd.DataFrame(water_dist[water_dist > 0.0]).hist()
            # plot_value_grid(water_dist, "water dist", markers=[tuple(traversal_indices_array)])
            # plot_value_grid(
            #     (water_dist / water_dist.max()) ** power_to_raise,
            #     "probabilities",
            #     markers=[tuple(traversal_indices_array)],
            # )
            # plot_value_grid(world.map.Z)
        else:
            traversal_indices_array = rand.choice(np.argwhere(possible_points))
    if is_debugging:
        plot_value_grid(near_traversal_center_mask, "Near traversal center")
        plot_value_grid(possible_points, "Possible points")

    if traversal_indices_array is None:
        raise ImpossibleWorldError("Could not find valid traversal point")
        # fine, whatever, just set to the midpoint
        # ring_traversal_point = to_2d_point((locations.spawn + locations.goal) / 2.0)
    else:
        traversal_indices = tuple(traversal_indices_array)
        ring_traversal_point = np.array([world.map.X[traversal_indices], world.map.Y[traversal_indices]])

    if is_debugging:
        logger.debug(f"Selected traversal point {ring_traversal_point}")
        logger.debug(np.linalg.norm(ring_traversal_point - center_point))

    if is_single_obstacle:
        outer_obstacle = None
        assert outer_solution is None, "Well that doesnt make sense"
    else:
        outer_obstacle = HeightObstacle(
            edge_config=EdgeConfig(),
            is_inside_ring=False,
            traversal_length=outer_traversal_length,
            traversal_width=traversal_width,
            traversal_noise_interpolation_multiple=traversal_noise_interpolation_multiple,
            is_default_climbable=is_outside_climbable,
            detail_radius=detail_radius,
        )

    ring_config = RingObstacleConfig(
        center_point=center_point,
        traversal_point=ring_traversal_point,
        edge=EdgeConfig(),
        height=height,
        height_mode=height_mode,
        inner_safety_radius=min([spawn_radius, food_radius]) + (safety_radius + extra_safety_radius),  # type: ignore
        outer_safety_radius=max([spawn_radius, food_radius]) - (safety_radius + extra_safety_radius),  # type: ignore
        inner_obstacle=HeightObstacle(
            edge_config=EdgeConfig(),
            is_inside_ring=True,
            traversal_length=inner_traversal_length,
            traversal_width=traversal_width,
            traversal_noise_interpolation_multiple=traversal_noise_interpolation_multiple,
            is_default_climbable=is_inside_climbable,
            detail_radius=detail_radius,
        ),
        outer_obstacle=outer_obstacle,
        chasm_bottom_size=gap_distance,
        inner_solution=inner_solution,
        outer_solution=outer_solution,
        dual_solution=dual_solution,
        expansion_meters=expansion_meters,
    )

    return ring_config


def create_outer_placeholder_solution(
    count: int = 5, offset: float = 0.5, randomization_dist: float = 2.0
) -> HeightSolution:
    items = [Placeholder(offset=offset, position=np.array([-0.5 * i, 0.0, 0.0])) for i in range(count)]
    return HeightSolution(
        outside_items=tuple(add_offsets(items)),
        outside_item_randomization_distance=randomization_dist,
        outside_item_radius=items[0].get_offset(),
        solution_point_brink_distance=0.5,
    )


def create_inner_placeholder_solution(
    count: int = 5, offset: float = 0.5, randomization_dist: float = 2.0
) -> HeightSolution:
    items = [Placeholder(offset=offset, position=np.array([-0.5 * i, 0.0, 0.0])) for i in range(count)]
    return HeightSolution(
        inside_items=tuple(add_offsets(items)),
        inside_item_randomization_distance=randomization_dist,
        inside_item_radius=items[0].get_offset(),
        solution_point_brink_distance=0.5,
    )
