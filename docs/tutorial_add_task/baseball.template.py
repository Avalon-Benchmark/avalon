from pathlib import Path
from typing import Tuple

import attr
import numpy as np

from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BaseballIndoorTaskConfig(IndoorTaskConfig):
    """Our task config. Any constants and other task boundary-defining attributes can go here."""

    ...


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BaseballTaskGenerator(BuildingTaskGenerator):
    """
    Our building task generator. It takes our task config and is able to spit out buildings based on difficulty and a
    seed. See `create_building_for_skill_scenario` to see how the generator is used.
    """

    config: BaseballIndoorTaskConfig = BaseballIndoorTaskConfig()

    def get_site_radius(self, rand: np.random.Generator, difficulty: float) -> float:
        """
        For outdoor worlds, buildings are placed on a site. This function determines how large that site must be for a
        specific level of difficulty.
        """
        ...

    def get_building_config(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        allowed_auxiliary_tasks: Tuple[BuildingTask, ...] = tuple(),
        aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig(),
    ) -> BuildingConfig:
        """
        The initial config used to create the building used for this level. Obstacles and modifications are added later.
        """
        ...

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        """
        The key function that maps difficulty to concrete obstacle parameters. For clarity and easier tuning, this is
        separate from actually applying the obstacles. You can return as many or few parameters as needed for the task.
        """
        return ()

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple
    ) -> IndoorTaskParams:
        """
        This is the "meat" of the generation process, where parameters get applied to the base building to arrive at a
        building with obstacles and modifications needed for the task.
        """
        ...


def generate_baseball_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: BaseballIndoorTaskConfig = BaseballIndoorTaskConfig(),
) -> None:
    """
    This part is mostly boilerplate that actually drives the task generator and exports the final level to a set of
    .tscn files that can be played in Godot (directly for human testing, or via GodotEnv for agent training).

    If we were to add an outdoor version of this task for variety, we would add that logic here as well (See other task
    files for examples).
    """
    building, entities, spawn_location, target_location = create_building_for_skill_scenario(
        rand,
        difficulty,
        BaseballTaskGenerator(task_config),
        position=CANONICAL_BUILDING_LOCATION,
        is_indoor_only=True,
    )
    world = make_indoor_task_world(
        building, entities, difficulty, spawn_location, target_location, rand, export_config
    )
    export_world(output_path, rand, world)
