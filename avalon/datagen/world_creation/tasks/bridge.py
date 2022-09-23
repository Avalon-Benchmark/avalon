from pathlib import Path
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import BRIDGE_LENGTH
from avalon.datagen.world_creation.constants import CLIMBING_REQUIRED_HEIGHT
from avalon.datagen.world_creation.constants import HALF_AGENT_HEIGHT_VECTOR
from avalon.datagen.world_creation.constants import MAX_BRIDGE_DIST
from avalon.datagen.world_creation.constants import MIN_BRIDGE_DIST
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.tools.log import Log
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.utils import to_2d_point
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
from avalon.datagen.world_creation.worlds.utils import add_offsets
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class BridgeTaskConfig(TaskConfig):
    # how large the gap to bridge should be (ie, how far would you have to jump if you could jump that far)
    # at difficulty 0.0 and 1.0 respectively
    bridge_dist_easy: float = MIN_BRIDGE_DIST
    bridge_dist_hard: float = MAX_BRIDGE_DIST
    # minimum required starting distance between the log and player at difficulty 0.0 and 1.0 respectively
    player_to_log_dist_easy: float = 2.0
    player_to_log_dist_hard: float = 6.0
    # minimum required starting distance between the log and the gap at difficulty 0.0 and 1.0 respectively
    gap_to_log_dist_easy: float = 1.5
    gap_to_log_dist_hard: float = 4.0
    # how wide the gap area is (ie, how much room to the left and right do you have as you are facing across the gap)
    # at difficulty 0.0 and 1.0 respectively
    gap_width_easy: float = 10.0
    gap_width_hard: float = 2.0
    # how wide of an area (in the same sense as gap_width) is flattened. This is required
    # at difficulty 0.0 and 1.0 respectively
    flatten_width_easy: float = 8.0
    flatten_width_hard: float = 4.0


def generate_bridge_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: BridgeTaskConfig = BridgeTaskConfig(),
) -> None:
    world, locations, difficulty = create_bridge_obstacle(rand, difficulty, export_config, task_config=task_config)
    world, locations = world.end_height_obstacles(
        locations, is_accessible_from_water=False, is_spawn_region_climbable=False
    )
    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal, is_spawn_height_reset=False)
    export_world(output_path, rand, world)


def create_bridge_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    is_for_carry: bool = False,
    task_config: BridgeTaskConfig = BridgeTaskConfig(),
) -> Tuple[World, WorldLocations, float]:

    is_solved, difficulty = select_boolean_difficulty(difficulty, rand)

    extra_safety_radius = scale_with_difficulty(
        difficulty, task_config.player_to_log_dist_easy, task_config.player_to_log_dist_hard
    )
    desired_gap_distance = scale_with_difficulty(
        difficulty, task_config.bridge_dist_easy, task_config.bridge_dist_hard
    )
    desired_goal_dist = (
        task_config.bridge_dist_easy * 3.0
        + extra_safety_radius * 3.0
        + task_config.bridge_dist_easy * 3.0 * difficulty
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, desired_goal_dist / 5), rand, difficulty, export_config, constraint
    )

    gap_distance, _warnings = world.get_critical_distance(
        locations, task_config.bridge_dist_easy, desired_gap_distance
    )
    height = scale_with_difficulty(difficulty, CLIMBING_REQUIRED_HEIGHT, BRIDGE_LENGTH)
    inside_item_radius = 0.5
    solution_point_brink_distance = 1.0

    if gap_distance is None:
        raise WorldTooSmall(AvalonTask.BRIDGE, task_config.bridge_dist_easy, locations.get_2d_spawn_goal_distance())

    if is_solved:
        randomization_dist = 0.0
        log_dist = inside_item_radius + (solution_point_brink_distance + gap_distance) / 2
        log_height = height
    else:
        randomization_dist = difficulty_variation(1.0, extra_safety_radius, rand, difficulty)
        log_dist = -difficulty_variation(
            task_config.gap_to_log_dist_easy, task_config.gap_to_log_dist_hard, rand, difficulty
        )
        log_height = 0

    log = Log(position=np.array([log_dist, log_height, 0.0]))

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance,
        constraint=constraint,
        height=-height,
        traversal_width=normal_distrib_range(
            task_config.gap_width_easy, task_config.gap_width_hard, 1.0, rand, difficulty
        ),
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
                    width=normal_distrib_range(
                        task_config.flatten_width_easy, task_config.flatten_width_hard, 1.0, rand, difficulty
                    ),
                ),
            ),
            solution_point_brink_distance=solution_point_brink_distance,
        ),
        extra_safety_radius=extra_safety_radius,
        probability_of_centering_on_spawn=0.0 if is_for_carry else 0.5,
    )
    world = world.add_height_obstacle(rand, ring_config, locations.island)

    # if this is non-compositional (ie, normal) task, then make sure we didn't accidentally overlap the spawn and log
    if constraint is None:
        # reset the spawn no matter what:
        spawn_location = locations.spawn.copy()
        idx = world.map.point_to_index(to_2d_point(spawn_location))
        height_at_location = world.map.Z[idx]
        spawn_location[1] = height_at_location + HALF_AGENT_HEIGHT_VECTOR[1] * 1.1

        # fix up the spawn if too close to the log
        item = world.items[-1]
        assert isinstance(item, Log), "Huh, only expected a single item to be spawned in this case..."
        log = item
        spawn_log_dist = np.linalg.norm(locations.spawn - log.position)
        safety_margin = 0.5
        if spawn_log_dist < (BRIDGE_LENGTH / 2.0) + safety_margin:
            spawn_location = spawn_location + UP_VECTOR * (AGENT_HEIGHT + safety_margin)

        # set the spawn
        locations = attr.evolve(locations, spawn=spawn_location)

    return world, locations, difficulty
