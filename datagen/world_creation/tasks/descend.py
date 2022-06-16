from pathlib import Path
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from common.utils import only
from datagen.world_creation.constants import MAX_FALL_DISTANCE_TO_DIE
from datagen.world_creation.constants import MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import FoodTreeBase
from datagen.world_creation.items import Placeholder
from datagen.world_creation.new_world import HeightPath
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point


def generate_descend_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_descend_obstacle(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=False, is_spawn_region_climbable=False)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_descend_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:
    is_platform_mode, difficulty = select_boolean_difficulty(difficulty, rand)
    is_everywhere_climbable, difficulty = select_boolean_difficulty(difficulty, rand)

    desired_goal_dist = 8 + 8 * difficulty * rand.uniform()
    extra_fall_distance = normal_distrib_range(
        0.0, MAX_FALL_DISTANCE_TO_DIE - MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE + 5.0, 4.0, rand, difficulty
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    height_delta = MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE + extra_fall_distance
    climbable_path_width = normal_distrib_range(10.0, 1.0, 0.5, rand, difficulty)

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        constraint=constraint,
        gap_distance=0.0,
        # height=-1.0 * (min_fall_distance_to_cause_damage + extra_fall_distance),
        # this causes a tiny bit of fall damage and feels fair:
        # height=-10.0,
        # this feels obviously lethal:
        # height=-28.0,
        height=-height_delta,
        traversal_width=10.0 - 9.0 * difficulty * rand.uniform(),
        inner_traversal_length=0.0,
        is_single_obstacle=True,
        is_inside_climbable=is_everywhere_climbable,
        is_outside_climbable=is_everywhere_climbable,
        inner_solution=HeightSolution(
            paths=(
                HeightPath(
                    width=climbable_path_width,
                    is_path_climbable=True,
                    is_height_affected=False,
                    is_path_restricted_to_land=False,
                ),
                HeightPath(
                    width=1.0,
                    is_path_climbable=False,
                    is_height_affected=False,
                    is_path_restricted_to_land=True,
                ),
            ),
            # this will be replaced with the food below
            inside_items=tuple([Placeholder(entity_id=0, position=np.array([0.0, 0.0, 0.0]))]),
            # so that the food ends up away from the edge a little bit
            inside_item_radius=0.75,
            solution_point_brink_distance=1.5,
        ),
        probability_of_centering_on_spawn=1.0,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)

    # reset the location of the spawn to be where the placeholder ended up
    # height will be reset below
    new_locations = attr.evolve(locations, spawn=only([x for x in world.items if isinstance(x, Placeholder)]).position)
    world.items = [x for x in world.items if not isinstance(x, Placeholder)]

    # move the food height down
    down_vector = np.array([0.0, -1.0, 0.0])
    new_locations = attr.evolve(new_locations, goal=new_locations.goal + down_vector * height_delta)

    # if this is platform mode, make a platform to fall on so it's easier
    if is_platform_mode:
        # find the cliff edges
        sq_slope = world.map.get_squared_slope()
        cliff_edges = sq_slope > ((height_delta * 0.8) ** 2)
        island_edges = world.map.get_outline(locations.island, 3)
        cliff_edges = np.logical_and(cliff_edges, np.logical_not(island_edges))
        cliff_edges = np.logical_and(cliff_edges, locations.island)
        cliff_movement_distance = scale_with_difficulty(difficulty, 1.0, 6.0)
        for i in range(2):
            nearby = world.map.get_dist_sq_to(to_2d_point(new_locations.spawn)) < cliff_movement_distance ** 2
            cliff_edges = np.logical_and(cliff_edges, nearby)
            if np.any(cliff_edges):
                cliff_indices = tuple(rand.choice(np.argwhere(cliff_edges)))
                cliff_point_2d = world.map.index_to_point_2d(cliff_indices)
                cliff_position = np.array(
                    [cliff_point_2d[0], new_locations.spawn[1] - height_delta / 2.0, cliff_point_2d[1]]
                )
                cliff = FoodTreeBase(entity_id=0, position=cliff_position)
                world.add_item(cliff)
                break

    return world, new_locations, difficulty
