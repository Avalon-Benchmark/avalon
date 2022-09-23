from pathlib import Path
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import CLIMBING_REQUIRED_HEIGHT
from avalon.datagen.world_creation.constants import TIGHT_DIST_STD_DEV
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations

_MIN_GAP_DISTANCE = 3.0


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class MoveTaskConfig(TaskConfig):
    # move has two different modes: direct and indirect.
    # in the direct mode, it's just a normal world and you walk somewhere, very straightforward
    # in the indirect mode, there is a very particular path that must be followed in order to get to the destination
    # these values control how far away the goal (food) is from the starting point in each of the two modes
    direct_goal_dist_easy: float = 0.5
    direct_goal_dist_hard: float = 40.0
    indirect_goal_dist_easy: float = 6.0
    indirect_goal_dist_hard: float = 40.0
    # these parameters control how many turns there are in the path (for the indirect mode)
    # due to implementation constraints, it cannot be greater than 2, but we dont want to be making weird mazes anyway
    indirect_path_point_count_easy: int = 0
    indirect_path_point_count_hard: int = 2
    indirect_path_point_count_std_dev: float = 0.25
    # controls how wide the path is. As this becomes more narrow, the task gets closer to walking a tightrope
    indirect_path_width_easy: float = 10.0
    indirect_path_width_hard: float = 1.75
    indirect_path_width_std_dev: float = 0.1
    # how deep to make the chasm that must be crossed in indirect mode. Should be deep enough that you can't just jump
    # out of it, otherwise won't learn to actually walk along the path
    indirect_chasm_depth_easy: float = CLIMBING_REQUIRED_HEIGHT
    indirect_chasm_depth_hard: float = CLIMBING_REQUIRED_HEIGHT * 3.0


def generate_move_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: MoveTaskConfig = MoveTaskConfig(),
) -> None:
    world, locations, difficulty = create_move_obstacle(rand, difficulty, export_config, task_config=task_config)
    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=False)
    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)


def create_move_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    task_config: MoveTaskConfig = MoveTaskConfig(),
) -> Tuple[World, WorldLocations, float]:

    if constraint is None:
        # this is a sort of "training wheels" condition, which is the only reason it comes before
        # the categoricals below
        is_path_direct, difficulty = select_boolean_difficulty(difficulty, rand)
    else:
        is_path_direct = False

    if is_path_direct:
        desired_goal_dist = scale_with_difficulty(
            difficulty, task_config.direct_goal_dist_easy, task_config.direct_goal_dist_hard
        )
        world, locations = create_world_from_constraint(
            stats.norm(desired_goal_dist, TIGHT_DIST_STD_DEV), rand, difficulty, export_config, constraint
        )
    else:
        path_point_count = round(
            normal_distrib_range(
                task_config.indirect_path_point_count_easy - 0.49,
                task_config.indirect_path_point_count_hard + 0.49,
                task_config.indirect_path_point_count_std_dev,
                rand,
                difficulty,
            )
        )
        desired_goal_dist = scale_with_difficulty(
            difficulty, task_config.indirect_goal_dist_easy, task_config.indirect_goal_dist_hard
        )
        traversal_width = normal_distrib_range(
            task_config.indirect_path_width_easy,
            task_config.indirect_path_width_hard,
            task_config.indirect_path_width_std_dev,
            rand,
            difficulty,
        )

        world, locations = create_world_from_constraint(
            stats.norm(desired_goal_dist, TIGHT_DIST_STD_DEV), rand, difficulty, export_config, constraint
        )

        max_gap_dist, _warnings = world.get_critical_distance(locations, _MIN_GAP_DISTANCE)

        if max_gap_dist is None:
            raise WorldTooSmall(AvalonTask.MOVE, _MIN_GAP_DISTANCE, locations.get_2d_spawn_goal_distance())

        if max_gap_dist is not None:
            gap_distance = normal_distrib_range(
                max_gap_dist * 0.3, max_gap_dist * 0.8, max_gap_dist * 0.1, rand, difficulty
            )
            depth = difficulty_variation(
                task_config.indirect_chasm_depth_easy, task_config.indirect_chasm_depth_hard, rand, difficulty
            )
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
                            is_path_climbable=True,
                            is_outside_edge_unclimbable=True,
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

            world = world.add_height_obstacle(rand, ring_config, locations.island)

    return world, locations, difficulty
