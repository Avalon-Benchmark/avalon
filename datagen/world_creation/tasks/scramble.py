from pathlib import Path

import attr
import numpy as np
from scipy import stats

from datagen.world_creation.constants import HALF_AGENT_HEIGHT_VECTOR
from datagen.world_creation.constants import TIGHT_DIST_STD_DEV
from datagen.world_creation.constants import UP_VECTOR
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import CANONICAL_FOOD_HEIGHT_ON_TREE
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.world_location_data import to_2d_point


def generate_scramble_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    desired_slope = 0.8
    world_diversity = scale_with_difficulty(difficulty, 0.5, 1.0)
    world_min_size = scale_with_difficulty(difficulty, 40.0, 60.0)
    desired_goal_dist = scale_with_difficulty(difficulty, 2.0, 20.0)
    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, TIGHT_DIST_STD_DEV),
        rand,
        world_diversity,
        export_config,
        None,
        min_size_in_meters=world_min_size,
    )
    world.end_height_obstacles(locations, is_accessible_from_water=False)

    # boost up the terrain around the food
    horizontal_distance = locations.get_2d_spawn_goal_distance()
    spawn_height = locations.spawn[1] - HALF_AGENT_HEIGHT_VECTOR[1]
    desired_height = horizontal_distance * desired_slope + spawn_height
    height_at_food = locations.goal[1] - CANONICAL_FOOD_HEIGHT_ON_TREE
    actual_height_delta = desired_height - height_at_food
    height_scale = desired_height / height_at_food
    world.map.add_hill(to_2d_point(locations.goal), height_scale, horizontal_distance, locations.island)

    # flatten the top just a little bit for the purposes of tree placement
    flatten_size = min([horizontal_distance * 0.2, 2.0])
    world.map.radial_flatten(to_2d_point(locations.goal), flatten_size * 2.0)

    # adjust the final food height by the delta
    locations = attr.evolve(locations, goal=locations.goal + UP_VECTOR * actual_height_delta)

    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))
