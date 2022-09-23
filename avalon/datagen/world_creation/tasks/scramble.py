from pathlib import Path

import attr
import numpy as np
from scipy import stats

from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import HALF_AGENT_HEIGHT_VECTOR
from avalon.datagen.world_creation.constants import TIGHT_DIST_STD_DEV
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class ScrambleTaskConfig(TaskConfig):
    # how steep the world will be, in general, between the spawn and goal
    # needs to be relatively close to this value, otherwise the task is simply walk or climb...
    desired_slope: float = 0.8
    # how far away the goal should be for difficulty 0.0 and 1.0 respectively
    goal_dist_easy: float = 2.0
    goal_dist_hard: float = 20.0
    # how "diverse" the world will be, ie, how crazy the terrain will be
    world_diversity_easy: float = 0.5
    world_diversity_hard: float = 1.0
    # worlds will be at least this large
    world_size_easy: float = 40.0
    world_size_hard: float = 60.0


def generate_scramble_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: ScrambleTaskConfig = ScrambleTaskConfig(),
) -> None:
    world_diversity = scale_with_difficulty(
        difficulty, task_config.world_diversity_easy, task_config.world_diversity_hard
    )
    world_min_size = scale_with_difficulty(difficulty, task_config.world_size_easy, task_config.world_size_hard)
    desired_goal_dist = scale_with_difficulty(difficulty, task_config.goal_dist_easy, task_config.goal_dist_hard)
    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, TIGHT_DIST_STD_DEV),
        rand,
        world_diversity,
        export_config,
        None,
        min_size_in_meters=world_min_size,
    )
    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=False)

    # boost up the terrain around the food
    horizontal_distance = locations.get_2d_spawn_goal_distance()
    spawn_height = locations.spawn[1] - HALF_AGENT_HEIGHT_VECTOR[1]
    desired_height = horizontal_distance * task_config.desired_slope + spawn_height
    height_at_food = locations.goal[1] - CANONICAL_FOOD_HEIGHT_ON_TREE
    actual_height_delta = desired_height - height_at_food
    height_scale = desired_height / height_at_food
    map_new = world.map.copy()
    map_new.add_hill(to_2d_point(locations.goal), height_scale, horizontal_distance, locations.island)

    # flatten the top just a little for the purposes of tree placement
    flatten_size = min([horizontal_distance * 0.2, 2.0])
    map_new.radial_flatten(to_2d_point(locations.goal), flatten_size * 2.0)

    world = attr.evolve(world, map=map_new)

    # adjust the final food height by the delta
    locations = attr.evolve(locations, goal=locations.goal + UP_VECTOR * actual_height_delta)

    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)
