from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from avalon.common.utils import only
from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import FOOD_HOVER_DIST
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import DefaultStoryLinker
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import WindowBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.components import Room
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import get_room_centroid_in_building_space
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.indoor.task_generator import rectangle_dimensions_within_radius
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.types import HeightMode
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ClimbIndoorTaskConfig(IndoorTaskConfig):
    # Min/max site radii for a building with this task to be fit in
    min_site_radius: float = 7.0
    max_site_radius: float = 15.0
    # Min and max stories for the building to have at difficulty=0 and 1, respectively
    min_story_count: int = 2
    max_story_count: int = 5


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class ClimbOutdoorTaskConfig(TaskConfig):
    # only a particular path is made climbable, not the entire surface, which makes the task more challenging as this
    # path is made more narrow
    # the maximum number of "points" to include in the path (ie, how many segments it should have at difficulty 1.0)
    # note that this value cannot be larger than 2 due to implementation reasons (and making much more convoluted
    # paths starts to feel like a weird maze task, which isn't what we're going for)
    # a value of 0 means the climbable path will be a single straight line
    # easy corresponds to difficulty 0.0, hard to difficulty 1.0
    path_point_count_easy: int = 0
    path_point_count_hard: int = 2
    path_point_count_std_dev: float = 0.25
    # how wide to make the climbable path at difficulty 0.0 and 1.0 respectively
    path_width_easy: float = 10.0
    path_width_hard: float = 1.5
    path_width_std_dev: float = 0.1
    # bounds for over how much distance (in the Godot XZ plane) the climbing should take place
    min_climb_horizontal_dist: float = 1.0
    max_climb_horizontal_dist: float = 7.0
    # how high to make the cliff to be climbed. Note that it must be at least this tall in order to prevent just jumping
    # in order to accomplish the task
    cliff_height_easy: float = 3.2
    cliff_height_hard: float = 20.0
    # how steep to make the cliff for difficulty 0.0 and 1.0 respectively.
    # note that it cannot be less than this, or else you can walk up it, and it cannot be too steep, or else it becomes
    # difficult to have enough space for the multi-point paths on harder difficulties
    cliff_slope_easy: float = 1.43  # 55 degrees min
    cliff_slope_hard: float = 4.0  # 76 degrees max (mostly to keep climb_distance higher)


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class ClimbTaskConfig(TaskConfig):
    # likelihood that the indoor variant of the task will be used
    indoor_probability: float = 0.2
    # the configs for each variant of the task
    indoor_config: ClimbIndoorTaskConfig = ClimbIndoorTaskConfig()
    outdoor_config: ClimbOutdoorTaskConfig = ClimbOutdoorTaskConfig()


def generate_climb_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: ClimbTaskConfig = ClimbTaskConfig(),
) -> None:
    is_indoor = rand.uniform() < task_config.indoor_probability
    if is_indoor:
        building, entities, spawn_location, target_location = create_building_for_skill_scenario(
            rand,
            difficulty,
            ClimbTaskGenerator(task_config.indoor_config),
            position=CANONICAL_BUILDING_LOCATION,
            is_indoor_only=True,
        )
        world = make_indoor_task_world(
            building, entities, difficulty, spawn_location, target_location, rand, export_config
        )
    else:
        world, locations, difficulty = create_climb_obstacle(
            rand, difficulty, export_config, task_config=task_config.outdoor_config
        )
        world, locations = world.end_height_obstacles(
            locations, is_accessible_from_water=True, is_spawn_region_climbable=False
        )
        world = add_food_tree_for_simple_task(world, locations)
        world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)

    export_world(output_path, rand, world)


def create_climb_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    task_config: ClimbOutdoorTaskConfig = ClimbOutdoorTaskConfig(),
) -> Tuple[World, WorldLocations, float]:

    is_everywhere_climbable, difficulty = select_boolean_difficulty(difficulty, rand)
    # for the agent we may want to move height here because its difficulty may be more important than path_point_count?
    path_point_count = round(
        normal_distrib_range(
            task_config.path_point_count_easy - 0.49,
            task_config.path_point_count_hard + 0.49,
            task_config.path_point_count_std_dev,
            rand,
            difficulty,
        )
    )

    # you aren't technically going to have to move this far--the food gets updated to be close to the cliff edge
    # this mostly just gives a bit of space for the climb paths
    desired_goal_dist = difficulty_variation(
        task_config.max_climb_horizontal_dist, task_config.max_climb_horizontal_dist * 4.0, rand, difficulty
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    critical_distance, _warnings = world.get_critical_distance(locations, task_config.min_climb_horizontal_dist)
    if critical_distance is None:
        raise WorldTooSmall(
            AvalonTask.CLIMB, task_config.min_climb_horizontal_dist, locations.get_2d_spawn_goal_distance()
        )

    max_climb_distance = min([critical_distance, task_config.max_climb_horizontal_dist])

    height = normal_distrib_range(task_config.cliff_height_easy, task_config.cliff_height_hard, 1.0, rand, difficulty)
    slope = normal_distrib_range(task_config.cliff_slope_easy, task_config.cliff_slope_hard, 0.5, rand, difficulty)
    climb_distance = min(
        normal_distrib_range(
            task_config.min_climb_horizontal_dist, max_climb_distance, max_climb_distance / 2.0, rand, difficulty
        ),
        height / slope,
    )
    path_width = normal_distrib_range(
        task_config.path_width_easy, task_config.path_width_hard, task_config.path_width_std_dev, rand, difficulty
    )

    outside_items: Tuple[Tool, ...] = tuple([Placeholder()])
    inside_items: Tuple[Tool, ...] = tuple()
    if constraint and constraint.is_height_inverted:
        outside_items, inside_items = inside_items, outside_items

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        constraint=constraint,
        gap_distance=0.0,
        height=height,
        traversal_width=normal_distrib_range(
            task_config.path_width_easy, task_config.path_width_hard, 1.0, rand, difficulty
        ),
        inner_traversal_length=climb_distance,
        is_single_obstacle=True,
        is_inside_climbable=is_everywhere_climbable,
        is_outside_climbable=is_everywhere_climbable,
        inner_solution=HeightSolution(
            paths=tuple()
            if is_everywhere_climbable
            else (
                HeightPath(
                    is_path_restricted_to_land=True,
                    extra_point_count=path_point_count,
                    width=path_width,
                    is_path_climbable=True,
                    is_height_affected=False,
                    # sometimes your paths will be slightly simpler than you expected.
                    # in those cases, a simplicity warning will be logged
                    is_path_failure_allowed=True,
                ),
            ),
            # this will be replaced with the food below
            outside_items=outside_items,
            inside_items=inside_items,
            # so that the food ends up away from the edge a little bit
            outside_item_radius=0.5,
            inside_item_radius=0.5,
            solution_point_brink_distance=1.5,
        ),
        height_mode=HeightMode.MIDPOINT_RELATIVE,
    )
    world = world.add_height_obstacle(rand, ring_config, locations.island)

    new_locations, world = replace_placeholder_with_goal(locations, world)

    return world, new_locations, difficulty


def replace_placeholder_with_goal(locations: WorldLocations, world: World) -> Tuple[WorldLocations, World]:
    # reset the location of the food to be where the placeholder ended up
    # height will be reset below
    placeholder_position = only([x for x in world.items if isinstance(x, Placeholder)]).position
    new_locations = attr.evolve(locations, goal=placeholder_position + UP_VECTOR * CANONICAL_FOOD_HEIGHT_ON_TREE)
    return new_locations, attr.evolve(world, items=tuple([x for x in world.items if not isinstance(x, Placeholder)]))


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ClimbTaskGenerator(BuildingTaskGenerator):
    config: ClimbIndoorTaskConfig = ClimbIndoorTaskConfig()

    def get_site_radius(self, rand: np.random.Generator, difficulty: float) -> float:
        return difficulty_variation(self.config.min_site_radius, self.config.max_site_radius, rand, difficulty)

    def get_building_config(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        allowed_auxiliary_tasks: Tuple[BuildingTask, ...] = tuple(),
        aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig(),
    ) -> BuildingConfig:
        width, length = rectangle_dimensions_within_radius(radius)
        story_count = self.config.min_story_count + round(
            normal_distrib_range(0, self.config.max_story_count - self.config.min_story_count, 0.1, rand, difficulty)
        )
        return BuildingConfig(
            width=width,
            length=length,
            story_count=story_count,
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(max_rooms=1),
            story_linker=DefaultStoryLinker(allow_ladders=True, allow_ramps=False),
            hallway_builder=DefaultHallwayBuilder(),
            window_builder=WindowBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        spawn_story = building.stories[0]
        spawn_room = only(spawn_story.rooms)
        target_story = building.stories[-1]
        target_room = only(target_story.rooms)
        return spawn_story, spawn_room, target_story, target_room

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple[Story, Room, Story, Room]
    ) -> IndoorTaskParams:
        spawn_story, spawn_room, target_story, target_room = obstacle_params
        entities: List[Entity] = []
        spawn_location = get_room_centroid_in_building_space(
            building, spawn_story, spawn_room, at_height=AGENT_HEIGHT / 2
        )
        target_location = get_room_centroid_in_building_space(
            building, target_story, target_room, at_height=FOOD_HOVER_DIST
        )
        entrance_sites = ((spawn_story.num, spawn_room.id, tuple(Azimuth)),)
        return building, entities, spawn_location, target_location, entrance_sites
