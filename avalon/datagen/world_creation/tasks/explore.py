import math
from pathlib import Path
from typing import List
from typing import Tuple

import attr
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import FOOD_HOVER_DIST
from avalon.datagen.world_creation.debug_plots import IS_DEBUG_VIS
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_VISIBLE_HEIGHT
from avalon.datagen.world_creation.entities.constants import TREE_FOOD_OFFSET
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import DefaultStoryLinker
from avalon.datagen.world_creation.indoor.builders import FootprintBuilder
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import TLShapeFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import WindowBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingNavGraph
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import get_room_centroid_in_building_space
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.indoor.task_generator import rectangle_dimensions_within_radius
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.types import WorldType
from avalon.datagen.world_creation.worlds.creation import create_world_for_skill_scenario
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ExploreIndoorTaskConfig(IndoorTaskConfig):
    # Largest possible site radius for a building to be placed on at difficulty=1
    max_site_radius: float = 25.0
    # Standard deviation for the distribution deciding the site radius: higher means more variability at same difficulty
    site_radius_std_dev: float = 2.5
    # Min/max number of stories used for the building at difficulty=0 and 1, respectively
    min_story_count: int = 1
    max_story_count: int = 3


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class ExploreTaskConfig(TaskConfig):
    # likelihood that the indoor variant of the task will be used
    indoor_probability: float = 0.2
    # max distance to make any of the compositional tasks. In meters.
    outdoor_max_distance: float = 600.0
    indoor_config: ExploreIndoorTaskConfig = ExploreIndoorTaskConfig()


def generate_explore_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: ExploreTaskConfig = ExploreTaskConfig(),
) -> None:
    is_indoor = rand.uniform() < task_config.indoor_probability
    if is_indoor:
        building, entities, spawn_location, target_location = create_building_for_skill_scenario(
            rand,
            difficulty,
            ExploreTaskGenerator(task_config.indoor_config),
            position=CANONICAL_BUILDING_LOCATION,
            is_indoor_only=True,
        )
        world = make_indoor_task_world(
            building, entities, difficulty, spawn_location, target_location, rand, export_config
        )
    else:
        desired_goal_dist = scale_with_difficulty(difficulty, 0.5, task_config.outdoor_max_distance / 2.0)
        world, locations = create_world_for_skill_scenario(
            rand,
            difficulty,
            TREE_FOOD_OFFSET,
            stats.norm(desired_goal_dist, desired_goal_dist / 5),
            export_config,
            is_visibility_required=False,
            visibility_height=FOOD_TREE_VISIBLE_HEIGHT,
            max_size_in_meters=task_config.outdoor_max_distance,
            world_type=WorldType.CONTINENT,
        )
        world = add_food_tree_for_simple_task(world, locations)
        world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal, is_visibility_required=False)

    export_world(output_path, rand, world)


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ExploreTaskGenerator(BuildingTaskGenerator):
    config: ExploreIndoorTaskConfig = ExploreIndoorTaskConfig()

    def get_site_radius(self, rand: np.random.Generator, difficulty: float) -> float:
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
        story_count = self.config.min_story_count + round(
            normal_distrib_range(0, self.config.max_story_count - self.config.min_story_count, 0.1, rand, difficulty)
        )
        width, length = rectangle_dimensions_within_radius(radius)
        footprint_builder: FootprintBuilder
        if width <= 10 or length <= 10:
            footprint_builder = RectangleFootprintBuilder()
            min_room_size = 2
        else:
            footprint_builder = TLShapeFootprintBuilder()
            min_room_size = 3

        return BuildingConfig(
            width=width,
            length=length,
            story_count=story_count,
            footprint_builder=footprint_builder,
            room_builder=HouseLikeRoomBuilder(min_room_size=min_room_size),
            hallway_builder=DefaultHallwayBuilder(proportion_additional_edges=1),
            story_linker=DefaultStoryLinker(
                raise_on_failure=False, allow_ladders=BuildingTask.CLIMB in allowed_auxiliary_tasks
            ),
            window_builder=WindowBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        spawn_story = building.stories[0]
        if len(spawn_story.rooms) == 1:
            raise ImpossibleWorldError("Building too small for explore task")

        exterior_wall_azimuths_by_room_id = spawn_story.get_exterior_wall_azimuths_by_room_id()
        periphery_room_ids = [
            room_id
            for room_id, exterior_wall_azimuths in exterior_wall_azimuths_by_room_id.items()
            if len(exterior_wall_azimuths) > 0
        ]
        initial_room_id = rand.choice(periphery_room_ids)

        nav_graph = BuildingNavGraph(building)
        spawn_room = spawn_story.rooms[initial_room_id]
        spawn_node = nav_graph.get_room_node(spawn_story, spawn_room)
        distance_by_location_id = nx.single_source_dijkstra_path_length(nav_graph, spawn_node, weight="distance")
        max_distance = max(distance_by_location_id.values())
        desired_distance = difficulty * max_distance

        # The target location distribution uses a standard deviation derived from the actual distribution derivation,
        # since otherwise you can end up with all-zero weights for sparse/bimodal distributions
        # (e.g. [10, 100] at difficulty 0.5 = 50, for which weights are [0,0] if we use a hard-coded std of say 0.5
        distribution_std = np.std(list(distance_by_location_id.values()))
        target_location_distribution = stats.norm(desired_distance, math.sqrt(distribution_std))
        location_weights = np.array(
            [
                target_location_distribution.pdf(distance) if distance > 0 else 0
                for distance in distance_by_location_id.values()
            ]
        )
        location_weights /= location_weights.sum()
        target_node = rand.choice(list(distance_by_location_id.keys()), p=location_weights)
        return nav_graph, spawn_node, target_node

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple
    ) -> IndoorTaskParams:
        nav_graph, spawn_node, target_node = obstacle_params
        stories_and_rooms_by_node_id = nav_graph.get_stories_and_rooms_by_node_id(building.stories)

        spawn_story, spawn_room = stories_and_rooms_by_node_id[spawn_node]
        spawn_location = get_room_centroid_in_building_space(
            building, spawn_story, spawn_room, at_height=AGENT_HEIGHT / 2
        )
        target_location = np.array(nav_graph.nodes[target_node]["position"])
        target_location[1] += FOOD_HOVER_DIST

        if IS_DEBUG_VIS:
            ax = nav_graph.plot()
            ax.text(
                spawn_location[2], spawn_location[0], spawn_location[1] - (AGENT_HEIGHT // 2), "spawn", color="blue"
            )
            ax.text(target_location[2], target_location[0], target_location[1], "target", color="green")
            plt.show()

        extra_items: List[Entity] = []
        entrance_sites = ((spawn_story.num, spawn_room.id, tuple(Azimuth)),)
        return building, extra_items, spawn_location, target_location, entrance_sites
