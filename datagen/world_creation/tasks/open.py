from pathlib import Path

import numpy as np

from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.indoor_task_generators import BuildingTask
from datagen.world_creation.indoor_task_generators import create_building_obstacle
from datagen.world_creation.indoor_task_generators import get_radius_for_building_task
from datagen.world_creation.indoor_task_generators import make_indoor_task_world
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty


def generate_open_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    building_radius = get_radius_for_building_task(rand, BuildingTask.OPEN, difficulty)
    building, extra_items, spawn_location, target_location = create_building_obstacle(
        rand, difficulty, BuildingTask.OPEN, building_radius, location=np.array([0, 2, 0]), yaw_radians=0.0
    )
    make_indoor_task_world(
        building, extra_items, difficulty, spawn_location, target_location, output_path, rand, export_config
    )

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))
