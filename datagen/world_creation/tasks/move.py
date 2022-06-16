from pathlib import Path
from typing import Optional
from typing import Tuple

import numpy as np
from scipy import stats

from datagen.world_creation.constants import CLIMBING_REQUIRED_HEIGHT
from datagen.world_creation.constants import TIGHT_DIST_STD_DEV
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
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import WorldLocationData

_MIN_GAP_DISTANCE = 3.0
MIN_MOVE_TASK_DISTANCE = get_min_task_distance(_MIN_GAP_DISTANCE)


def generate_move_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_move_obstacle(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=False)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_move_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:

    if constraint is None:
        # this is a sort of "training wheels" condition, which is the only reason it comes before
        # the categoricals below
        is_path_direct, difficulty = select_boolean_difficulty(difficulty, rand)
    else:
        is_path_direct = False

    if is_path_direct:
        desired_goal_dist = scale_with_difficulty(difficulty, 0.5, 40.0)
        world, locations = create_world_from_constraint(
            stats.norm(desired_goal_dist, TIGHT_DIST_STD_DEV), rand, difficulty, export_config, constraint
        )
    else:
        path_point_count, difficulty = select_categorical_difficulty([0, 1, 2], difficulty, rand)
        desired_goal_dist = get_desired_move_task_distance(difficulty)
        traversal_width = normal_distrib_range(10.0, 2.0, 1.0, rand, difficulty)

        world, locations = create_world_from_constraint(
            stats.norm(desired_goal_dist, TIGHT_DIST_STD_DEV), rand, difficulty, export_config, constraint
        )

        max_gap_dist = world.get_critical_distance(locations, _MIN_GAP_DISTANCE)

        if max_gap_dist is None:
            raise WorldTooSmall(AvalonTask.MOVE, _MIN_GAP_DISTANCE, locations.get_2d_spawn_goal_distance())

        if max_gap_dist is not None:
            gap_distance = normal_distrib_range(
                max_gap_dist * 0.3, max_gap_dist * 0.8, max_gap_dist * 0.1, rand, difficulty
            )
            depth = difficulty_variation(CLIMBING_REQUIRED_HEIGHT, CLIMBING_REQUIRED_HEIGHT * 3, rand, difficulty)
            ring_config = make_ring(
                rand,
                difficulty,
                world,
                locations,
                height=-depth,
                gap_distance=gap_distance,
                traversal_width=traversal_width,
                dual_solution=HeightSolution(
                    paths=(
                        HeightPath(
                            is_path_restricted_to_land=True,
                            extra_point_count=path_point_count,
                            width=traversal_width,
                            # sometimes your paths will be slightly simpler than you expected.
                            # in those cases, a simplicity warning will be logged
                            is_path_failure_allowed=True,
                        ),
                    ),
                    solution_point_brink_distance=1.0,
                ),
                traversal_noise_interpolation_multiple=4.0,
                constraint=constraint,
            )

            world.add_height_obstacle(rand, ring_config, locations.island)

    return world, locations, difficulty


def get_desired_move_task_distance(difficulty: float) -> float:
    return scale_with_difficulty(difficulty, 6.0, 40.0)
