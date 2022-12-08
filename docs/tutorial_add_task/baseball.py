from itertools import product
from pathlib import Path
from typing import Tuple
from typing import cast

import attr
import numpy as np

from avalon.common.utils import only
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import FOOD_HOVER_DIST
from avalon.datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from avalon.datagen.world_creation.constants import STANDING_REACH_HEIGHT
from avalon.datagen.world_creation.entities.tools.weapons import LargeStick
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import local_to_global_coords
from avalon.datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import add_food_island
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.indoor.task_generator import rectangle_dimensions_within_radius
from avalon.datagen.world_creation.indoor.tiles import decide_tiles_by_distance
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BaseballIndoorTaskConfig(IndoorTaskConfig):
    """Our task config. Any constants and other task boundary-defining attributes can go here."""

    min_site_radius = 15
    max_site_radius = 30
    site_radius_std_dev = 3
    min_plinth_height = (MAX_JUMP_HEIGHT_METERS + STANDING_REACH_HEIGHT) * 1.25
    max_plinth_height = 7.9  # determined empirically
    plinth_height_std_dev = 0.2


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
        return normal_distrib_range(
            self.config.min_site_radius, self.config.max_site_radius, self.config.site_radius_std_dev, rand, difficulty
        )

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
        width, length = rectangle_dimensions_within_radius(radius)
        return BuildingConfig(
            width=width,
            length=length,
            story_count=1,
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(max_rooms=1),
            hallway_builder=DefaultHallwayBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        """
        The key function that maps difficulty to concrete obstacle parameters. For clarity and easier tuning, this is
        separate from actually applying the obstacles. You can return as many or few parameters as needed for the task.
        """
        story = only(building.stories)
        room = only(story.rooms)
        plinth_height = normal_distrib_range(
            self.config.min_plinth_height,
            self.config.max_plinth_height,
            self.config.plinth_height_std_dev,
            rand,
            difficulty,
        )
        plinth_tile = cast(Tuple[int, int], tuple(rand.choice(room.get_tile_positions())))
        all_viable_bat_tiles = list(product(range(1, room.width - 1), range(1, room.length - 1)))
        viable_bat_tiles = [tile for tile in all_viable_bat_tiles if tile != plinth_tile]
        bat_tile = only(decide_tiles_by_distance(viable_bat_tiles, plinth_tile, difficulty, rand))
        viable_spawn_tiles = [tile for tile in viable_bat_tiles if tile != bat_tile]
        spawn_tile = only(decide_tiles_by_distance(viable_spawn_tiles, bat_tile, difficulty, rand))
        return plinth_height, BuildingTile(*plinth_tile), BuildingTile(*bat_tile), BuildingTile(*spawn_tile)

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple
    ) -> IndoorTaskParams:
        """
        This is the "meat" of the generation process, where parameters get applied to the base building to arrive at a
        building with obstacles and modifications needed for the task.
        """
        plinth_height, plinth_tile, bat_tile, spawn_tile = obstacle_params

        story = only(building.stories)
        room = only(story.rooms)
        updated_room, island_tiles = add_food_island(room, plinth_tile, 1, plinth_height)
        story.rooms[0] = updated_room
        room = updated_room

        bat_thickness = 0.5
        room_position_3d = np.array([room.position.x, 0, room.position.z])
        bat_position_in_room = self._position_from_tile(room, bat_tile, at_height=bat_thickness / 2)
        bat_position = local_to_global_coords(bat_position_in_room, room_position_3d)
        extra_entities = [LargeStick(position=bat_position)]

        spawn_location_in_room_space = self._position_from_tile(room, spawn_tile, at_height=AGENT_HEIGHT / 2)
        spawn_location = local_to_global_coords(spawn_location_in_room_space, room_position_3d)

        target_location_in_room_space = self._position_from_tile(room, plinth_tile, at_height=FOOD_HOVER_DIST)
        target_location = local_to_global_coords(target_location_in_room_space, room_position_3d)

        valid_entrance_sites = ((0, 0, tuple(Azimuth)),)
        return building, extra_entities, spawn_location, target_location, valid_entrance_sites


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
