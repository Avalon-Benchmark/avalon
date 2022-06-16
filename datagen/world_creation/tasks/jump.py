from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import stats

from datagen.world_creation.constants import JUMPING_REQUIRED_HEIGHT
from datagen.world_creation.constants import MAX_EFFECTIVE_JUMP_DIST
from datagen.world_creation.constants import MAX_FALL_DISTANCE_TO_DIE
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.constants import get_min_task_distance
from datagen.world_creation.heightmap import ExportConfig
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
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import WorldLocationData

_MIN_DIST_FOR_JUMP = 1.0
_MIN_HEIGHT_REQUIRING_JUMP = 0.7
MIN_JUMP_TASK_DISTANCE = get_min_task_distance(_MIN_DIST_FOR_JUMP)


def generate_jump_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_jump_obstacle(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=False, is_spawn_region_climbable=False)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


# TODO: select from entirely different jump setups (across, up to grab, up platforms, multi hop path, across to a building)
# TODO: sometimes requires multiple jumps
# TODO: apply some height deltas if we want some extra variation
def create_jump_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:

    is_depth_jumpable, difficulty = select_boolean_difficulty(difficulty, rand)
    is_climbing_possible, difficulty = select_boolean_difficulty(difficulty, rand, initial_prob=1, final_prob=0.1)
    desired_goal_dist = difficulty_variation(7.0, 12.0, rand, difficulty)

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, desired_goal_dist / 10), rand, difficulty, export_config, constraint
    )

    desired_jump_distance = scale_with_difficulty(
        difficulty, _MIN_DIST_FOR_JUMP, MAX_EFFECTIVE_JUMP_DIST
    )  # make it not impossibly hard to jump
    min_dist_for_actual_gap = 1.0
    jump_distance = world.get_critical_distance(locations, min_dist_for_actual_gap, desired_jump_distance)
    jumpable_width = normal_distrib_range(10.0, 3.0, 1.0, rand, difficulty)

    if jump_distance is None:
        raise WorldTooSmall(AvalonTask.JUMP, min_dist_for_actual_gap, locations.get_2d_spawn_goal_distance())

    if is_depth_jumpable:
        depth = difficulty_variation(_MIN_HEIGHT_REQUIRING_JUMP, JUMPING_REQUIRED_HEIGHT, rand, difficulty)
    else:
        # we need deeper gaps for the compositional tasks--because they can be on more difficult terrain,
        # the gap needs to be quite deep to prevent you from glitching your way across
        depth = scale_with_difficulty(difficulty, JUMPING_REQUIRED_HEIGHT, MAX_FALL_DISTANCE_TO_DIE)

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        jump_distance,
        constraint=constraint,
        height=-depth,
        traversal_width=jumpable_width,
        is_inside_climbable=True,
        is_outside_climbable=is_climbing_possible,
        dual_solution=HeightSolution(
            paths=(
                HeightPath(
                    is_solution_flattened=True,
                    is_height_affected=False,
                    width=normal_distrib_range(8.0, 4.0, 1.0, rand, difficulty),
                ),
            ),
            solution_point_brink_distance=1.0,
        ),
        extra_safety_radius=0.5,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)

    return world, locations, difficulty
