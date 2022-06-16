from pathlib import Path
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from common.utils import only
from datagen.world_creation.constants import UP_VECTOR
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.constants import get_min_task_distance
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.heightmap import HeightMode
from datagen.world_creation.items import CANONICAL_FOOD_HEIGHT_ON_TREE
from datagen.world_creation.items import Placeholder
from datagen.world_creation.new_world import HeightPath
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import WorldLocationData

_MIN_CLIMB_HORIZONTAL_DIST = 1.0
_MAX_CLIMB_HORIZONTAL_DIST = 7.0
_MIN_CLIFF_HEIGHT_TO_PREVENT_JUMPING = 3.2
_MAX_CLIFF_HEIGHT = 20
_MIN_CLIMB_SLOPE = 1.43  # 55 degrees min
_MAX_CLIMB_SLOPE = 4.0  # 76 degrees max (mostly to keep climb_distance higher)
MIN_CLIMB_TASK_DISTANCE = get_min_task_distance(_MIN_CLIMB_HORIZONTAL_DIST)


def generate_climb_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_climb_obstacle(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=True, is_spawn_region_climbable=False)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


# TODO: we should probalby make a training wheels climb task that starts you in a tiny pit that you can ALMOST jump out of
#  the pit can eventually get a little bigger and a little taller, and eventually get a very wide climb path
def create_climb_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:

    is_everywhere_climbable, difficulty = select_boolean_difficulty(difficulty, rand)
    # for the agent we may want to move height here because its difficulty may be more important than path_point_count?
    path_point_count, difficulty = select_categorical_difficulty([0, 1, 2], difficulty, rand)

    # you aren't technically going to have to move this far--the food gets updated to be close to the cliff edge
    # this mostly just gives a bit of space for the climb paths
    desired_goal_dist = difficulty_variation(
        _MAX_CLIMB_HORIZONTAL_DIST, _MAX_CLIMB_HORIZONTAL_DIST * 4.0, rand, difficulty
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    max_climb_distance = min(
        [world.get_critical_distance(locations, _MIN_CLIMB_HORIZONTAL_DIST), _MAX_CLIMB_HORIZONTAL_DIST]
    )

    if max_climb_distance is None:
        raise WorldTooSmall(AvalonTask.CLIMB, _MIN_CLIMB_HORIZONTAL_DIST, locations.get_2d_spawn_goal_distance())

    height = normal_distrib_range(_MIN_CLIFF_HEIGHT_TO_PREVENT_JUMPING, _MAX_CLIFF_HEIGHT, 1.0, rand, difficulty)
    slope = normal_distrib_range(_MIN_CLIMB_SLOPE, _MAX_CLIMB_SLOPE, 0.5, rand, difficulty)
    climb_distance = min(
        normal_distrib_range(
            _MIN_CLIMB_HORIZONTAL_DIST, max_climb_distance, max_climb_distance / 2.0, rand, difficulty
        ),
        height / slope,
    )
    path_width = normal_distrib_range(10.0, 1.0, 0.5, rand, difficulty)

    outside_items = tuple([Placeholder()])
    inside_items = tuple()
    if constraint and constraint.is_height_inverted:
        outside_items, inside_items = inside_items, outside_items

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        constraint=constraint,
        gap_distance=0.0,
        height=height,
        traversal_width=normal_distrib_range(10.0, 1.0, 1.0, rand, difficulty),
        inner_traversal_length=climb_distance,
        is_single_obstacle=True,
        is_inside_climbable=is_everywhere_climbable,
        is_outside_climbable=is_everywhere_climbable,
        inner_solution=HeightSolution(
            paths=tuple()
            if is_everywhere_climbable
            else (
                HeightPath(
                    is_path_restricted_to_land=True,
                    extra_point_count=path_point_count,
                    width=path_width,
                    is_path_climbable=True,
                    is_height_affected=False,
                    # sometimes your paths will be slightly simpler than you expected.
                    # in those cases, a simplicity warning will be logged
                    is_path_failure_allowed=True,
                ),
            ),
            # this will be replaced with the food below
            outside_items=outside_items,
            inside_items=inside_items,
            # so that the food ends up away from the edge a little bit
            outside_item_radius=0.5,
            inside_item_radius=0.5,
            solution_point_brink_distance=1.5,
        ),
        height_mode=HeightMode.MIDPOINT_RELATIVE,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)

    new_locations = replace_placeholder_with_goal(locations, world)

    return world, new_locations, difficulty


def replace_placeholder_with_goal(locations, world):
    # reset the location of the food to be where the placeholder ended up
    # height will be reset below
    placeholder_position = only([x for x in world.items if isinstance(x, Placeholder)]).position
    new_locations = attr.evolve(locations, goal=placeholder_position + UP_VECTOR * CANONICAL_FOOD_HEIGHT_ON_TREE)
    world.items = [x for x in world.items if not isinstance(x, Placeholder)]
    return new_locations
