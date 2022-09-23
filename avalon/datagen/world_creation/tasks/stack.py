import math
from itertools import product
from pathlib import Path
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import attr
import numpy as np
from scipy import stats

from avalon.common.errors import SwitchError
from avalon.common.utils import only
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import BOX_HEIGHT
from avalon.datagen.world_creation.constants import FOOD_HOVER_DIST
from avalon.datagen.world_creation.constants import ITEM_FLATTEN_RADIUS
from avalon.datagen.world_creation.constants import JUMPING_REQUIRED_HEIGHT
from avalon.datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from avalon.datagen.world_creation.entities.item import Item
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.entities.tools.stone import Stone
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import local_to_global_coords
from avalon.datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import WindowBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.constants import TileIdentity
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import _decide_boulder_mass
from avalon.datagen.world_creation.indoor.task_generator import add_food_island
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import decide_spawn_room_and_tile
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.indoor.task_generator import rectangle_dimensions_within_radius
from avalon.datagen.world_creation.indoor.tiles import decide_tiles_by_distance
from avalon.datagen.world_creation.indoor.utils import inset_borders
from avalon.datagen.world_creation.tasks.climb import replace_placeholder_with_goal
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.utils import add_offsets
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StackIndoorTaskConfig(IndoorTaskConfig):
    # Smallest possible site radius for a building with this task to be placed on (largest size is inherited)
    min_site_radius: float = 8.0
    # Standard deviation for the distribution deciding the site radius: higher means more variability at same difficulty
    site_radius_std_dev: float = 2.5
    # Total number of stories for the building. Note that this task is a single-story task, all other stories will
    # be purely aesthetic (e.g. to help with being viewed in the distance outdoors).
    story_count: int = 2
    # Min/max number of stones that need to be vertically stacked in order to solve the task
    min_stackable_height: int = 1
    max_stackable_height: int = 4
    # The task will create enough crates to stack a staircase + this many surplus stones (in case some get lost / stuck)
    surplus_stone_count: int = 0
    # Square island side length; should be large enough to prevent knocking food off by reaching or jumping
    food_island_size: int = 5
    # Min/max stone masses used to determine exact stone size; lighter stones are less stable
    min_stone_mass: float = 50
    max_stone_mass: float = 500


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class StackOutdoorTaskConfig(TaskConfig):
    # how much to move the boxes from their "canonical" (ie, solved) configuration at difficulty 0.0 and 1.0
    randomization_dist_easy: float = 0.0
    randomization_dist_hard: float = 6.0
    # makes the cliff at least this tall (with zero boxes). Final height will be num_boxes * box_height taller than this
    base_height: float = JUMPING_REQUIRED_HEIGHT - 0.5
    # how wide to make the region of the cliff where  the height is precisely correct for stacking. Other areas could
    # be taller, so making this value smaller often makes the task harder
    cliff_width_easy: float = 10.0
    cliff_width_hard: float = 1.0
    cliff_width_std_dev: float = 1.0
    # controls how much space will be present for stacking boxes at the base of the cliff
    cliff_space: float = 3.0


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class StackTaskConfig(TaskConfig):
    # likelihood that the indoor variant of the task will be used
    indoor_probability: float = 0.2
    # the configs for each variant of the task
    indoor_config: StackIndoorTaskConfig = StackIndoorTaskConfig()
    outdoor_config: StackOutdoorTaskConfig = StackOutdoorTaskConfig()


def generate_stack_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: StackTaskConfig = StackTaskConfig(),
) -> None:
    is_indoor = rand.uniform() < task_config.indoor_probability
    if is_indoor:
        building, entities, spawn_location, target_location = create_building_for_skill_scenario(
            rand,
            difficulty,
            StackTaskGenerator(task_config.indoor_config),
            position=CANONICAL_BUILDING_LOCATION,
            is_indoor_only=True,
        )
        world = make_indoor_task_world(
            building, entities, difficulty, spawn_location, target_location, rand, export_config
        )
    else:
        world, locations, difficulty = create_stack_obstacle(
            rand, difficulty, export_config, task_config=task_config.outdoor_config
        )
        world, locations = world.end_height_obstacles(
            locations, is_accessible_from_water=False, is_spawn_region_climbable=False
        )
        world = add_food_tree_for_simple_task(world, locations)
        world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)

    export_world(output_path, rand, world)


def create_stack_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    is_for_carry: bool = False,
    task_config: StackOutdoorTaskConfig = StackOutdoorTaskConfig(),
) -> Tuple[World, WorldLocations, float]:

    stack_height = round(normal_distrib_range(0.51, 3.49, 0.25, rand, difficulty))
    spare_boxes, difficulty = select_categorical_difficulty([3, 2, 1, 0], difficulty, rand)

    desired_height = stack_height * BOX_HEIGHT + task_config.base_height

    # how far to jiggle the boxes from their "solved" configuration
    randomization_dist = scale_with_difficulty(
        difficulty, task_config.randomization_dist_easy, task_config.randomization_dist_hard
    )

    # you aren't technically going to have to move this far--the food gets updated to be close to the cliff edge
    # this mostly just gives a bit of space for the climb paths
    base_dist = 2.0 * desired_height + 2.0 * randomization_dist
    desired_goal_dist = difficulty_variation(base_dist, 2.0 * base_dist, rand, difficulty)

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    # this ensures there is space at the base of the cliff
    if 2 * randomization_dist > locations.get_2d_spawn_goal_distance() - task_config.cliff_space:
        randomization_dist = (locations.get_2d_spawn_goal_distance() - task_config.cliff_space) / 2.0
    extra_safety_radius = randomization_dist

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance=0.0,
        height=desired_height,
        constraint=constraint,
        traversal_width=normal_distrib_range(
            task_config.cliff_width_easy,
            task_config.cliff_width_hard,
            task_config.cliff_width_std_dev,
            rand,
            difficulty,
        ),
        inner_traversal_length=0.0,
        is_single_obstacle=True,
        is_inside_climbable=False,
        inner_solution=HeightSolution(
            inside_items=_get_boxes(stack_height, spare_boxes),
            inside_item_randomization_distance=randomization_dist,
            inside_item_radius=0.5,
            solution_point_brink_distance=2.0,
            outside_items=tuple([Placeholder()]),
            outside_item_radius=1.0,
        ),
        extra_safety_radius=extra_safety_radius,
        probability_of_centering_on_spawn=0.0 if is_for_carry else 0.5,
    )
    ring_config = attr.evolve(ring_config, terrain_blurs=((8.0, 0.5), (6.0, 0.75), (4.0, 1.0)))
    earlier_item_count = len(world.items)
    world = world.add_height_obstacle(rand, ring_config, locations.island)
    world = flatten_places_under_items(world, earlier_item_count, Stone)

    # move the food to the edge so that it is visible
    new_locations, world = replace_placeholder_with_goal(locations, world)

    return world, new_locations, difficulty


def flatten_places_under_items(
    world: World, earlier_item_count: int, filter_to_classes: Union[Type[Item], Tuple[Type[Item], ...]]
):
    new_stones = []
    for i in range(earlier_item_count, len(world.items)):
        item = world.items[i]
        if isinstance(item, filter_to_classes):
            new_stones.append(item)
    stone_positions = [to_2d_point(x.position) for x in new_stones]
    for i, stone in enumerate(new_stones):
        distances = []
        for j, other_stone in enumerate(new_stones):
            if i == j:
                continue
            distances.append(np.linalg.norm(stone_positions[i] - stone_positions[j]))
        if len(distances) > 0 and np.min(distances) > ITEM_FLATTEN_RADIUS:
            world = world.flatten(stone_positions[i], ITEM_FLATTEN_RADIUS, ITEM_FLATTEN_RADIUS)
    return world


def _get_boxes(stack_height: int, spare_boxes: int) -> Tuple[Tool, ...]:
    if stack_height == 1:
        boxes = [
            Stone(position=np.array([0.0, 0.0, 0.0])),
        ]
    elif stack_height == 2:
        boxes = [
            Stone(position=np.array([0.0, 0.0, 0.0])),
            Stone(position=np.array([0.0, 1.0, 0.0])),
            Stone(position=np.array([-1.0, 0.0, 0.0])),
        ]
    elif stack_height == 3:
        boxes = [
            Stone(position=np.array([0.0, 0.0, 0.0])),
            Stone(position=np.array([0.0, 1.0, 0.0])),
            Stone(position=np.array([0.0, 2.0, 0.0])),
            Stone(position=np.array([-1.0, 0.0, 0.0])),
            Stone(position=np.array([-1.0, 1.0, 0.0])),
            Stone(position=np.array([-2.0, 0.0, 0.0])),
        ]
    else:
        raise SwitchError(f"Unhandled number of boxes: {stack_height}")

    if spare_boxes >= 1:
        boxes.append(Stone(position=np.array([0.0, 0.0, -1.0])))
    if spare_boxes >= 2:
        boxes.append(Stone(position=np.array([0.0, 0.0, 1.0])))
    if spare_boxes >= 3:
        boxes.append(Stone(position=np.array([-1.0, 0.0, 1.0])))
    assert spare_boxes <= 3

    return tuple(add_offsets(boxes))


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StackTaskGenerator(BuildingTaskGenerator):
    config: StackIndoorTaskConfig = StackIndoorTaskConfig()

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
        width, length = rectangle_dimensions_within_radius(radius)
        return BuildingConfig(
            width=width,
            length=length,
            story_count=self.config.story_count,
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(max_rooms=1),
            story_linker=None,
            hallway_builder=DefaultHallwayBuilder(),
            window_builder=WindowBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        first_story = building.stories[0]
        room = only(first_story.rooms)
        border_value = 0
        permitted_tiles = inset_borders(room.tiles, void_value=TileIdentity.ROOM.value, border_value=border_value)
        tile_zs, tile_xs = np.where(permitted_tiles == border_value)
        permitted_tile_tuples = list(zip(*(tile_zs, tile_xs)))
        spawn_room, spawn_tile = decide_spawn_room_and_tile(rand, first_story, {room.id: permitted_tile_tuples})

        min_stackable_height = self.config.min_stackable_height
        max_stackable_height = self.config.max_stackable_height
        stackable_height = round(min_stackable_height + difficulty * (max_stackable_height - min_stackable_height))
        required_stone_count = (stackable_height * (stackable_height + 1)) // 2
        surplus_stone_count = self.config.surplus_stone_count

        food_island_size = self.config.food_island_size
        food_island_height = (stackable_height * BOX_HEIGHT) + MAX_JUMP_HEIGHT_METERS * 0.95

        min_room_size = math.ceil(BOX_HEIGHT) * 2 + food_island_size
        if spawn_room.width < min_room_size or spawn_room.length < min_room_size:
            raise ImpossibleWorldError("Room too small for stack task")

        viable_food_island_tiles = list(
            product(
                range(1, spawn_room.width - food_island_size - 2), range(1, spawn_room.length - food_island_size - 2)
            )
        )
        if len(viable_food_island_tiles) == 0:
            raise ImpossibleWorldError("Building too small for push task")
        food_island_top_left_tile = only(
            decide_tiles_by_distance(viable_food_island_tiles, spawn_tile, difficulty, rand)
        )
        food_island_position = BuildingTile(*food_island_top_left_tile)
        food_island_tiles = list(
            product(
                range(food_island_position.x, food_island_position.x + food_island_size),
                range(food_island_position.z, food_island_position.z + food_island_size),
            )
        )

        free_tiles = [
            tile
            for tile in list(product(range(spawn_room.width), range(spawn_room.length)))
            if tile not in food_island_tiles and tile != spawn_tile
        ]
        food_island_center_tile = (
            food_island_top_left_tile[0] + food_island_size // 2,
            food_island_top_left_tile[1] + food_island_size // 2,
        )
        stone_tiles = decide_tiles_by_distance(
            free_tiles,
            food_island_center_tile,
            difficulty,
            rand,
            tile_count=required_stone_count + surplus_stone_count,
        )
        stone_masses = [
            _decide_boulder_mass(
                rand,
                difficulty,
                min_mass=self.config.min_stone_mass,
                max_mass=self.config.max_stone_mass,
                lighter_is_harder=True,
            )
            for i in range(len(stone_tiles))
        ]

        return (
            spawn_tile,
            food_island_top_left_tile,
            food_island_size,
            food_island_height,
            stone_tiles,
            stone_masses,
            BOX_HEIGHT,
        )

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple
    ) -> IndoorTaskParams:
        first_story = building.stories[0]
        main_room = only(first_story.rooms)
        (
            spawn_tile,
            food_island_top_left_tile,
            food_island_size,
            food_island_height,
            stone_tiles,
            stone_masses,
            stone_size,
        ) = obstacle_params
        food_island_position = BuildingTile(*food_island_top_left_tile)
        room_with_island, food_island_tiles = add_food_island(
            main_room, food_island_position, food_island_size, food_island_height
        )

        building.stories[0].rooms[0] = room_with_island

        tile_size = 1
        stone_info = []
        for i, stone_tile in enumerate(stone_tiles):
            floor_elevation = room_with_island.floor_heightmap[stone_tile[1], stone_tile[0]] - DEFAULT_FLOOR_THICKNESS
            stone_location_in_room = np.array(
                [
                    stone_tile[0] + tile_size / 2,
                    floor_elevation + stone_size / 2,
                    stone_tile[1] + tile_size / 2,
                ]
            )
            stone_info.append(StoneInfo(stone_location_in_room, stone_size, stone_masses[i]))

        food_island_center_tile = (
            food_island_position.x + food_island_size // 2,
            food_island_position.z + food_island_size // 2,
        )
        target_location_in_room = np.array(
            [
                food_island_center_tile[0] + tile_size / 2,
                room_with_island.floor_heightmap[food_island_center_tile[1], food_island_center_tile[0]]
                + FOOD_HOVER_DIST,
                food_island_center_tile[1] + tile_size / 2,
            ]
        )

        spawn_location_in_room = np.array(
            [
                spawn_tile[0] + tile_size / 2,
                room_with_island.floor_heightmap[spawn_tile[1], spawn_tile[0]] + AGENT_HEIGHT / 2,
                spawn_tile[1] + tile_size / 2,
            ]
        )

        room_offset = np.array([main_room.position.x, 0, main_room.position.z])
        spawn_location_in_building = local_to_global_coords(np.array(spawn_location_in_room), room_offset)
        target_location_in_building = local_to_global_coords(np.array(target_location_in_room), room_offset)

        stones = []
        for stone_location_in_room, stone_size, stone_mass in stone_info:
            stone_location_in_building = local_to_global_coords(np.array(stone_location_in_room), room_offset)
            stone_location = local_to_global_coords(stone_location_in_building, building.position)
            stones.append(Stone(position=stone_location, size=stone_size, mass=stone_mass))

        entrance_sites = ((first_story.num, main_room.id, tuple(Azimuth)),)
        return building, stones, spawn_location_in_building, target_location_in_building, entrance_sites


class StoneInfo(NamedTuple):
    location: Point3DNP
    size: float
    mass: float
