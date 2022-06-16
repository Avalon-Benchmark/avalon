from pathlib import Path

import numpy as np
from scipy import stats

from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.indoor_task_generators import BuildingTask
from datagen.world_creation.indoor_task_generators import create_building_obstacle
from datagen.world_creation.indoor_task_generators import get_radius_for_building_task
from datagen.world_creation.indoor_task_generators import make_indoor_task_world
from datagen.world_creation.items import FOOD_TREE_VISIBLE_HEIGHT
from datagen.world_creation.items import TREE_FOOD_OFFSET
from datagen.world_creation.tasks.constants import IS_WORLD_DIVERSITY_ENABLED
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import WorldType
from datagen.world_creation.tasks.task_worlds import create_world_for_skill_scenario
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty

# max distance to make any of the compositional tasks. In meters.
_MAX_EXPLORE_DISTANCE = 600.0


def generate_explore_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    is_indoor = rand.uniform() < 0.2
    # is_indoor = False
    if is_indoor:
        building_radius = get_radius_for_building_task(rand, BuildingTask.NAVIGATE, difficulty)
        building, extra_items, spawn_location, target_location = create_building_obstacle(
            rand, difficulty, BuildingTask.NAVIGATE, building_radius, location=np.array([0, 2, 0]), yaw_radians=0.0
        )
        make_indoor_task_world(
            building, extra_items, difficulty, spawn_location, target_location, output_path, rand, export_config
        )
    else:
        desired_goal_dist = scale_with_difficulty(difficulty, 0.5, _MAX_EXPLORE_DISTANCE / 2.0)
        world, locations = create_world_for_skill_scenario(
            rand,
            difficulty if IS_WORLD_DIVERSITY_ENABLED else 0.0,
            TREE_FOOD_OFFSET,
            stats.norm(desired_goal_dist, desired_goal_dist / 5),
            export_config,
            is_visibility_required=False,
            visibility_height=FOOD_TREE_VISIBLE_HEIGHT,
            max_size_in_meters=_MAX_EXPLORE_DISTANCE,
            world_type=WorldType.CONTINENT,
        )
        add_food_tree_for_simple_task(world, locations)
        world.add_spawn(rand, difficulty, locations.spawn, locations.goal, is_visibility_required=False)
        export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))
