from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import stats

from datagen.world_creation.constants import BRIDGE_LENGTH
from datagen.world_creation.constants import CLIMBING_REQUIRED_HEIGHT
from datagen.world_creation.constants import MAX_BRIDGE_DIST
from datagen.world_creation.constants import MIN_BRIDGE_DIST
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import Log
from datagen.world_creation.new_world import HeightPath
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import add_offsets
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import WorldLocationData


def generate_bridge_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_bridge_obstacle(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=False, is_spawn_region_climbable=False)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_bridge_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    is_for_carry: bool = False,
) -> Tuple[NewWorld, WorldLocationData, float]:

    is_solved, difficulty = select_boolean_difficulty(difficulty, rand)

    extra_safety_radius = scale_with_difficulty(difficulty, 2.0, 6.0)
    desired_gap_distance = scale_with_difficulty(difficulty, MIN_BRIDGE_DIST, MAX_BRIDGE_DIST)
    desired_goal_dist = MIN_BRIDGE_DIST * 3.0 + extra_safety_radius * 3.0 + MIN_BRIDGE_DIST * 3.0 * difficulty

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, desired_goal_dist / 5), rand, difficulty, export_config, constraint
    )

    gap_distance = world.get_critical_distance(locations, MIN_BRIDGE_DIST, desired_gap_distance)
    height = scale_with_difficulty(difficulty, CLIMBING_REQUIRED_HEIGHT, BRIDGE_LENGTH)
    inside_item_radius = 0.5
    solution_point_brink_distance = 1.0

    if gap_distance is None:
        raise WorldTooSmall(AvalonTask.BRIDGE, MIN_BRIDGE_DIST, locations.get_2d_spawn_goal_distance())

    if is_solved:
        randomization_dist = 0
        log_dist = inside_item_radius + (solution_point_brink_distance + gap_distance) / 2
        log_height = height
    else:
        randomization_dist = difficulty_variation(1.0, extra_safety_radius, rand, difficulty)
        log_dist = -difficulty_variation(1.5, 4.0, rand, difficulty)
        log_height = 0

    # log_dist = difficulty_variation(1.0, 3.0, rand, difficulty)
    # randomization_dist = difficulty_variation(0.0, 4.0, rand, difficulty)
    log = Log(entity_id=0, position=np.array([log_dist, log_height, 0.0]))

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance,
        constraint=constraint,
        height=-height,
        traversal_width=normal_distrib_range(10.0, 2.0, 1.0, rand, difficulty),
        is_inside_climbable=True,
        is_outside_climbable=False,
        dual_solution=HeightSolution(
            inside_items=tuple(add_offsets([log])),
            inside_item_randomization_distance=randomization_dist,
            inside_item_radius=inside_item_radius,
            paths=(
                HeightPath(
                    is_solution_flattened=True,
                    is_height_affected=False,
                    width=normal_distrib_range(8.0, 4.0, 1.0, rand, difficulty),
                ),
            ),
            solution_point_brink_distance=solution_point_brink_distance,
        ),
        extra_safety_radius=extra_safety_radius,
        probability_of_centering_on_spawn=0.0 if is_for_carry else 0.5,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)

    return world, locations, difficulty
