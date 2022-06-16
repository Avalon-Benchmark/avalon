import math
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.spatial.transform import Rotation
from shapely.geometry import Point

from common.errors import SwitchError
from common.utils import only
from datagen.godot_base_types import Vector2
from datagen.godot_base_types import Vector3
from datagen.world_creation.constants import AGENT_HEIGHT
from datagen.world_creation.constants import BOULDER_MAX_MASS
from datagen.world_creation.constants import BOULDER_MIN_MASS
from datagen.world_creation.constants import BOX_HEIGHT
from datagen.world_creation.constants import FOOD_HOVER_DIST
from datagen.world_creation.constants import MAX_FLAT_JUMP_METERS
from datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from datagen.world_creation.geometry import Axis
from datagen.world_creation.geometry import Position
from datagen.world_creation.geometry import euclidean_distance
from datagen.world_creation.geometry import local_to_global_coords
from datagen.world_creation.geometry import midpoint
from datagen.world_creation.geometry import squares_overlap
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.indoor.builders import DefaultEntranceBuilder
from datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from datagen.world_creation.indoor.builders import DefaultStoryLinker
from datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from datagen.world_creation.indoor.builders import ObstacleBuilder
from datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from datagen.world_creation.indoor.builders import TLShapeFootprintBuilder
from datagen.world_creation.indoor.builders import WindowBuilder
from datagen.world_creation.indoor.constants import CEILING_THICKNESS
from datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from datagen.world_creation.indoor.helpers import draw_line_in_grid
from datagen.world_creation.indoor.objects import Azimuth
from datagen.world_creation.indoor.objects import Building
from datagen.world_creation.indoor.objects import BuildingAestheticsConfig
from datagen.world_creation.indoor.objects import BuildingNavGraph
from datagen.world_creation.indoor.objects import Room
from datagen.world_creation.indoor.objects import Story
from datagen.world_creation.indoor.objects import Wall
from datagen.world_creation.indoor.objects import Window
from datagen.world_creation.items import CANONICAL_FOOD_CLASS
from datagen.world_creation.items import Boulder
from datagen.world_creation.items import Door
from datagen.world_creation.items import DoorOpenButton
from datagen.world_creation.items import Entity
from datagen.world_creation.items import HingeDoor
from datagen.world_creation.items import HingeSide
from datagen.world_creation.items import LatchingMechanics
from datagen.world_creation.items import MountSlot
from datagen.world_creation.items import RotatingBar
from datagen.world_creation.items import SlidingBar
from datagen.world_creation.items import SlidingDoor
from datagen.world_creation.items import Stone
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.new_world import _get_spawn
from datagen.world_creation.new_world import build_building
from datagen.world_creation.task_generators import IdGenerator
from datagen.world_creation.tasks.biome_settings import generate_biome_config
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.utils import IS_DEBUG_VIS
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.utils import decompose_weighted_mean
from datagen.world_creation.world_config import BuildingConfig
from datagen.world_creation.world_config import WorldConfig

MIN_NAVIGATE_BUILDING_SIZE = 5.0


class BuildingTask(Enum):
    NAVIGATE = "NAVIGATE"
    OPEN = "OPEN"
    PUSH = "PUSH"
    STACK = "STACK"


IndoorTaskParams = Tuple[Building, List[Entity], np.ndarray, np.ndarray]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class IndoorTaskBuilder:
    def get_build_params(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def build(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        aesthetics: BuildingAestheticsConfig,
        add_entrance: bool,
        is_climbable: bool,
    ) -> IndoorTaskParams:
        raise NotImplementedError

    def decide_spawn_room_and_tile(
        self,
        building: Building,
        story: Story,
        rand: np.random.Generator,
        permitted_tiles: Optional[List[Tuple[int, int]]] = None,
    ):
        if entrances := story.entrances:
            entrance = rand.choice(entrances)
            spawn_room, landing_position = entrance.get_connected_room_and_landing_position(building)
            spawn_tile = landing_position.x, landing_position.z
            if permitted_tiles is not None and spawn_tile not in permitted_tiles:
                raise ImpossibleWorldError("Entrance was spawned in a location outside the spawn region")
        else:

            spawn_room: Room = rand.choice(story.rooms)
            if permitted_tiles is not None:
                spawn_tile = rand.choice(permitted_tiles)
            else:
                spawn_tile = rand.integers(0, spawn_room.width - 1), rand.integers(0, spawn_room.length - 1)
        return spawn_room, spawn_tile


def tile_centroid(tile_position: Tuple[int, int], tile_size: float = 1):
    half_size = tile_size / 2
    return Vector2(tile_position[0] + half_size, tile_position[1] + half_size)


def rectangle_dimensions_within_radius(radius: float):
    size = math.floor(2 * radius / math.sqrt(2))
    return size, size


def decide_tiles_by_distance(
    free_tiles: List[Tuple[int, int]],
    target_tile: Tuple[int, int],
    difficulty: float,
    rand: np.random.Generator,
    tile_count=1,
):
    target_tile_position = Position(*target_tile)
    tile_distances = [
        euclidean_distance(Position(*tile_position), target_tile_position) for tile_position in free_tiles
    ]
    desired_distance = difficulty * max(tile_distances)
    std = np.std(tile_distances)
    std = std if std != 0 else 1  # std=0 yields invalid distribution
    distance_distribution = stats.norm(desired_distance, math.sqrt(std))
    weights = np.array([distance_distribution.pdf(distance) for distance in tile_distances])
    weights /= weights.sum()
    return [tuple(tile) for tile in rand.choice(free_tiles, tile_count, p=weights)]


def _decide_boulder_mass(
    rand: np.random.Generator,
    difficulty: float,
    min_mass: float = BOULDER_MIN_MASS,
    max_mass: float = BOULDER_MAX_MASS,
    lighter_is_harder: bool = False,
) -> float:
    if lighter_is_harder:
        difficulty = 1 - difficulty
    target_mass = min_mass + (max_mass - min_mass) * difficulty
    mass_std = 2.5
    boulder_mass_distribution = stats.norm(target_mass, mass_std)
    return min(max(boulder_mass_distribution.rvs(random_state=rand), min_mass), max_mass)


class PushTaskVariant(Enum):
    BLOCKED_ENTRANCE = "blocked_entrance"
    CLIMB_HIGHER = "climb_higher"
    CROSS_CHASM = "cross_chasm"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PushTaskBuilder(IndoorTaskBuilder):
    def get_build_params(self, difficulty: float, rand: np.random.Generator, building: Building) -> Any:
        difficulty_weights = [2, 1]
        variant_difficulty, param_difficulty = decompose_weighted_mean(difficulty, difficulty_weights, rand=rand)
        variants_by_difficulty = {
            0.25: PushTaskVariant.BLOCKED_ENTRANCE,
            0.5: PushTaskVariant.CLIMB_HIGHER,
            0.75: PushTaskVariant.CROSS_CHASM,
        }
        variant_difficulty_distribution = stats.norm(variant_difficulty, 0.125)
        difficulty_weights = np.array([variant_difficulty_distribution.pdf(x) for x in variants_by_difficulty.keys()])
        difficulty_weights /= difficulty_weights.sum()
        task_variant = rand.choice(list(variants_by_difficulty.values()), p=difficulty_weights)

        boulder_mass = _decide_boulder_mass(rand, difficulty)
        if task_variant == PushTaskVariant.BLOCKED_ENTRANCE:
            build_fn = self._place_obstacles_and_boulders_blocked_entrance
            extra_build_params = self.get_blocked_entrance_build_params(building, difficulty, rand)
        elif task_variant == PushTaskVariant.CLIMB_HIGHER:
            build_fn = self._place_obstacles_and_boulders_climb_higher
            extra_build_params = self.get_climb_higher_build_params(building, difficulty, rand)
        elif task_variant == PushTaskVariant.CROSS_CHASM:
            build_fn = self._place_obstacles_and_boulders_cross_chasm
            extra_build_params = self.get_cross_chasm_build_params(building, difficulty, rand)
        else:
            raise SwitchError(task_variant)

        return build_fn, extra_build_params, boulder_mass

    def get_blocked_entrance_build_params(
        self, building: Building, difficulty: float, rand: np.random.Generator
    ) -> Dict:
        tile_size = 1
        boulder_size = 2
        min_enclosure_width = math.ceil(boulder_size) + tile_size  # >1 extra tile to be able to walk around it

        first_story = building.stories[0]
        room = only(first_story.rooms)

        # todo: leftward = harder?
        free_sites = [
            site
            for site in ObstacleBuilder.get_free_sites(building.stories[:1])
            if min_enclosure_width <= site.index <= room.width - min_enclosure_width - 1 and site.vertical
        ]
        if len(free_sites) == 0:
            raise ImpossibleWorldError("Room too small to create task")
        site_idx = rand.choice(range(len(free_sites)))
        site = free_sites[site_idx]
        hole_tile = site.index, site.length // 2
        boulder_tile = hole_tile[0] + 1, hole_tile[1]

        # Spawn region is always on the east, so we can add a positive-x aligned entrance there easily
        target_region_tiles = list(product(range(site.index - 1), range(room.length)))
        spawn_region_tiles = list(product(range(site.index + 1, room.width), range(room.length)))
        free_spawn_region_tiles = [
            tile
            for tile in spawn_region_tiles
            if not squares_overlap(
                tile_centroid(tile), tile_size, tile_centroid(boulder_tile, boulder_size), boulder_size
            )
        ]
        spawn_room, spawn_tile = self.decide_spawn_room_and_tile(building, first_story, rand, free_spawn_region_tiles)
        target_tile = only(decide_tiles_by_distance(target_region_tiles, spawn_tile, difficulty, rand))

        extra_build_params = dict(
            hole_tile=hole_tile,
            spawn_tile=spawn_tile,
            boulder_size=boulder_size,
            target_tile=target_tile,
        )
        return extra_build_params

    def get_climb_higher_build_params(self, building: Building, difficulty: float, rand: np.random.Generator) -> Dict:
        first_story = building.stories[0]
        spawn_room, spawn_tile = self.decide_spawn_room_and_tile(building, first_story, rand)

        food_island_size = 5  # prevent knocking food off by reaching or jumping
        viable_food_island_tiles = list(
            product(range(spawn_room.width - food_island_size - 1), range(spawn_room.length - food_island_size - 1))
        )
        if len(viable_food_island_tiles) == 0:
            raise ImpossibleWorldError("Building too small for push task")
        food_island_top_left_tile = only(
            decide_tiles_by_distance(viable_food_island_tiles, spawn_tile, difficulty, rand)
        )
        food_island_position = Position(*food_island_top_left_tile)
        food_island_tiles = list(
            product(
                range(food_island_position.x, food_island_position.x + food_island_size),
                range(food_island_position.z, food_island_position.z + food_island_size),
            )
        )

        leeway_factor = 0.95
        boulder_size = MAX_JUMP_HEIGHT_METERS * leeway_factor
        boulder_size_in_tiles = math.ceil(boulder_size)
        food_island_height = boulder_size + MAX_JUMP_HEIGHT_METERS * leeway_factor
        viable_boulder_tiles = [
            tile
            for tile in list(
                product(
                    range(boulder_size_in_tiles, spawn_room.width - boulder_size_in_tiles + 1),
                    range(boulder_size_in_tiles, spawn_room.length - boulder_size_in_tiles + 1),
                )
            )
            if tile not in food_island_tiles and tile != spawn_tile
        ]
        if len(viable_boulder_tiles) == 0:
            raise ImpossibleWorldError("Building too small for push task")
        boulder_tile = only(decide_tiles_by_distance(viable_boulder_tiles, spawn_tile, difficulty, rand))

        extra_build_params = dict(
            spawn_tile=spawn_tile,
            food_island_top_left_tile=food_island_top_left_tile,
            food_island_size=food_island_size,
            food_island_height=food_island_height,
            boulder_tile=boulder_tile,
            boulder_size=boulder_size,
        )
        return extra_build_params

    def get_cross_chasm_build_params(self, building: Building, difficulty: float, rand: np.random.Generator) -> Dict:
        boulder_size = MAX_JUMP_HEIGHT_METERS * 0.9
        chasm_depth = (MAX_JUMP_HEIGHT_METERS * 1.8) - DEFAULT_FLOOR_THICKNESS
        chasm_size = round(MAX_FLAT_JUMP_METERS * 2)
        min_distance_from_wall = math.ceil(boulder_size)

        tile_size = 1
        first_story = building.stories[0]
        room = only(first_story.rooms)

        viable_x_positions = list(range(min_distance_from_wall, room.width - min_distance_from_wall - chasm_size))
        if len(viable_x_positions) == 0:
            raise ImpossibleWorldError("Room too small to place a chasm")
        chasm_left_x = rand.choice(viable_x_positions)

        spawn_region_tiles = list(product(range(chasm_left_x + chasm_size, room.width), range(room.length)))
        target_region_tiles = list(product(range(chasm_left_x), range(room.length)))
        spawn_room, spawn_tile = self.decide_spawn_room_and_tile(
            building, first_story, rand, permitted_tiles=spawn_region_tiles
        )

        target_tile = only(decide_tiles_by_distance(target_region_tiles, spawn_tile, difficulty, rand))
        free_spawn_region_tiles = [
            tile
            for tile in spawn_region_tiles
            if not squares_overlap(
                tile_centroid(spawn_tile), tile_size, tile_centroid(tile, boulder_size), boulder_size
            )
            and (boulder_size <= tile_size or (tile[0] not in {0, room.width - 1}))
        ]
        boulder_tile = only(decide_tiles_by_distance(free_spawn_region_tiles, target_tile, difficulty, rand))

        extra_build_params = dict(
            chasm_left_x=chasm_left_x,
            chasm_size=chasm_size,
            chasm_depth=chasm_depth,
            spawn_tile=spawn_tile,
            boulder_tile=boulder_tile,
            boulder_size=boulder_size,
            target_tile=target_tile,
        )
        return extra_build_params

    def _place_obstacles_and_boulders_blocked_entrance(
        self,
        building: Building,
        room: Room,
        *,
        hole_tile: Tuple[int, int],
        spawn_tile: Tuple[int, int],
        boulder_size: float,
        target_tile: Tuple[int, int],
    ):
        tile_size = 1
        hole_tile_x, hole_tile_z = hole_tile

        wall_point_set = [
            [Position(x=hole_tile_x, z=0), Position(x=hole_tile_x, z=hole_tile_z - 1)],
            [Position(x=hole_tile_x, z=hole_tile_z + 1), Position(x=hole_tile_x, z=room.length - 1)],
        ]

        # Make wall unjumpable, but low enough to potentially see food on other side
        wall_height = room.outer_height
        new_heightmap = room.floor_heightmap.copy()
        windows = []
        for points in wall_point_set:
            wall = Wall(0, 0, points, 1, wall_height)
            draw_line_in_grid(wall.points, new_heightmap, wall.height, filter_values=None)

            window_y = DEFAULT_FLOOR_THICKNESS + (wall_height - DEFAULT_FLOOR_THICKNESS) / 2
            window_position_room_coords = midpoint(
                Vector3(points[0].x + tile_size / 2, window_y, points[0].z),
                Vector3(points[1].x + tile_size / 2, window_y, points[1].z + tile_size),
            )
            window_length = round((euclidean_distance(points[0], points[1]) + tile_size) * 0.8)
            window_size = wall.thickness, wall_height * 0.5, window_length
            room_offset = np.array([room.position.x, 0, room.position.z])
            window_position_room_coords = np.array(
                [window_position_room_coords.x, window_position_room_coords.y, window_position_room_coords.z]
            )
            window_position = local_to_global_coords(window_position_room_coords, room_offset)
            window = Window(window_position, np.array(window_size))
            windows.append(window)

        # todo: cleaner mutation?
        initial_room_id = 0
        building.stories[0].rooms[initial_room_id] = room.with_heightmap(new_heightmap)
        building.stories[0] = attr.evolve(building.stories[0], windows=windows)

        boulder_location_in_room = (
            hole_tile_x + 1 + boulder_size / 2,
            room.floor_heightmap[hole_tile_z, hole_tile_x] + boulder_size / 2 - DEFAULT_FLOOR_THICKNESS,
            hole_tile_z + tile_size / 2,
        )

        spawn_height = room.floor_heightmap[spawn_tile[1], spawn_tile[0]] + AGENT_HEIGHT / 2
        spawn_location_in_room = spawn_tile[0] + tile_size / 2, spawn_height, spawn_tile[1] + tile_size / 2
        target_height = room.floor_heightmap[target_tile[1], target_tile[0]] + FOOD_HOVER_DIST
        target_location_in_room = target_tile[0] + tile_size / 2, target_height, target_tile[1] + tile_size / 2
        return spawn_location_in_room, target_location_in_room, boulder_location_in_room, boulder_size

    def _place_obstacles_and_boulders_cross_chasm(
        self,
        building: Building,
        room: Room,
        *,
        chasm_left_x: int,
        chasm_size: int,
        chasm_depth: float,
        spawn_tile: Tuple[int, int],
        boulder_tile: Tuple[int, int],
        boulder_size: float,
        target_tile: Tuple[int, int],
    ):
        tile_size = 1
        new_heightmap = room.floor_heightmap.copy()
        new_heightmap[0 : room.length, chasm_left_x : chasm_left_x + chasm_size] = -chasm_depth
        building.stories[0].rooms[0] = room.with_heightmap(new_heightmap)

        spawn_height = room.floor_heightmap[spawn_tile[1], spawn_tile[0]] + AGENT_HEIGHT / 2
        spawn_location_in_room = spawn_tile[0] + tile_size / 2, spawn_height, spawn_tile[1] + tile_size / 2
        target_height = room.floor_heightmap[target_tile[1], target_tile[0]] + FOOD_HOVER_DIST
        target_location_in_room = target_tile[0] + tile_size / 2, target_height, target_tile[1] + tile_size / 2
        boulder_height = (
            room.floor_heightmap[boulder_tile[1], boulder_tile[0]] + boulder_size / 2 - DEFAULT_FLOOR_THICKNESS
        )
        boulder_location_in_room = boulder_tile[0] + tile_size / 2, boulder_height, boulder_tile[1] + tile_size / 2
        return spawn_location_in_room, target_location_in_room, boulder_location_in_room, boulder_size

    def _place_obstacles_and_boulders_climb_higher(
        self,
        building: Building,
        room: Room,
        *,
        spawn_tile: Tuple[int, int],
        food_island_top_left_tile: Tuple[int, int],
        food_island_size: int,
        food_island_height: float,
        boulder_tile: Tuple[int, int],
        boulder_size: float,
    ):
        tile_size = 1
        min_room_size = math.ceil(boulder_size) * 2 + food_island_size
        if room.width < min_room_size or room.length < min_room_size:
            raise ImpossibleWorldError("Room too small to fit a food island and boulder")

        food_island_position = Position(*food_island_top_left_tile)
        room_with_island, food_island_tiles = add_food_island(
            room, food_island_position, food_island_size, food_island_height
        )
        room_with_island.outer_height = (
            room_with_island.floor_heightmap.max() + CEILING_THICKNESS + 2 * MAX_JUMP_HEIGHT_METERS
        )
        building.stories[0].rooms[0] = room_with_island

        spawn_location_in_room = (
            spawn_tile[0] + tile_size / 2,
            room_with_island.floor_heightmap[spawn_tile[1], spawn_tile[0]] + AGENT_HEIGHT / 2,
            spawn_tile[1] + tile_size / 2,
        )

        boulder_location_in_room = (
            boulder_tile[0] + tile_size / 2,
            room_with_island.floor_heightmap[boulder_tile[1], boulder_tile[0]]
            - DEFAULT_FLOOR_THICKNESS
            + boulder_size / 2,
            boulder_tile[1] + tile_size / 2,
        )

        food_island_center_tile = (
            food_island_position.x + food_island_size // 2,
            food_island_position.z + food_island_size // 2,
        )
        target_location_in_room = (
            food_island_center_tile[0] + tile_size / 2,
            room_with_island.floor_heightmap[food_island_center_tile[1], food_island_center_tile[0]] + FOOD_HOVER_DIST,
            food_island_center_tile[1] + tile_size / 2,
        )

        return spawn_location_in_room, target_location_in_room, boulder_location_in_room, boulder_size

    def build(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        aesthetics: BuildingAestheticsConfig,
        add_entrance: bool,
        is_climbable: bool,
    ) -> IndoorTaskParams:

        width, length = rectangle_dimensions_within_radius(radius)
        building_config = BuildingConfig(
            width=width,
            length=length,
            story_count=aesthetics.desired_story_count,
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(max_rooms=1),
            hallway_builder=DefaultHallwayBuilder(),
            window_builder=WindowBuilder(),
            is_climbable=is_climbable,
            aesthetics=aesthetics,
        )
        building = build_building(building_config, 0, rand)
        main_room_id = 0
        if add_entrance:
            # TODO: We're forcing EAST as the entrance azimuth here and making the levels have the spawn region on the east
            #  to avoid having to rotate these levels which is a bit of a pain. However, this introduces coupling between
            #  the two layers, which is nasty and unforgivable, and we should fix it
            building = rebuild_with_aligned_entrance(building, rand, main_room_id, Azimuth.EAST)
        main_room = building.stories[0].rooms[main_room_id]

        build_fn, extra_build_params, boulder_mass = self.get_build_params(difficulty, rand, building)
        spawn_location_in_room, target_location_in_room, boulder_location_in_room, boulder_size = build_fn(
            building, main_room, **extra_build_params
        )

        room_offset = np.array([main_room.position.x, 0, main_room.position.z])
        spawn_location_in_building = local_to_global_coords(np.array(spawn_location_in_room), room_offset)
        target_location_in_building = local_to_global_coords(np.array(target_location_in_room), room_offset)
        boulder_location_in_building = local_to_global_coords(np.array(boulder_location_in_room), room_offset)

        entities = [Boulder(position=boulder_location_in_building, mass=boulder_mass, size=boulder_size)]
        return building, entities, spawn_location_in_building, target_location_in_building


class StoneInfo(NamedTuple):
    location: np.array
    size: float
    mass: float


def add_food_island(room: Room, position: Position, size: int, height: float) -> Tuple[Room, List[Tuple[int, int]]]:
    """place an elevated square platform in the room; returns all its tiles as (x,z) coordinates"""
    if size >= room.width or size >= room.length:
        raise ImpossibleWorldError(f"Room {room.id} is too small to place a food island of size {size}")

    new_heightmap = room.floor_heightmap.copy()
    new_heightmap[position.z : position.z + size, position.x : position.x + size] = DEFAULT_FLOOR_THICKNESS + height
    if height > 3:
        center_z = position.z + size // 2
        center_x = position.x + size // 2
        new_heightmap[center_z, center_x] += MAX_JUMP_HEIGHT_METERS * 0.95
    tiles = list(product(range(position.x, position.x + size), range(position.z, position.z + size)))
    new_outer_height = room.outer_height + (new_heightmap.max() - DEFAULT_FLOOR_THICKNESS)
    updated_room = attr.evolve(room, outer_height=new_outer_height, floor_heightmap=new_heightmap)
    return updated_room, tiles


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class StackTaskBuilder(IndoorTaskBuilder):
    def get_build_params(self, difficulty: float, rand: np.random.Generator, building: Building) -> Any:
        first_story = building.stories[0]
        spawn_room, spawn_tile = self.decide_spawn_room_and_tile(building, first_story, rand)

        min_stackable_height = 1
        max_stackable_height = 4  # as number of vertically stacked stones
        stackable_height = round(min_stackable_height + difficulty * (max_stackable_height - min_stackable_height))
        required_stone_count = (stackable_height * (stackable_height + 1)) // 2
        surplus_stone_count = 0

        food_island_size = 5  # prevent knocking food off by reaching or jumping
        food_island_height = (stackable_height * BOX_HEIGHT) + MAX_JUMP_HEIGHT_METERS * 0.95

        min_room_size = math.ceil(BOX_HEIGHT) * 2 + food_island_size
        if spawn_room.width < min_room_size or spawn_room.length < min_room_size:
            raise ImpossibleWorldError("Room too small for stack task")

        food_island_size = 5  # prevent knocking food off by reaching or jumping
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
        food_island_position = Position(*food_island_top_left_tile)
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
        # todo: make task upper difficulty bound higher by spreading stones further apart?
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

        return spawn_tile, food_island_top_left_tile, food_island_size, food_island_height, stone_tiles, BOX_HEIGHT

    def build(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        aesthetics: BuildingAestheticsConfig,
        add_entrance: bool,
        is_climbable: bool,
    ) -> IndoorTaskParams:
        width, length = rectangle_dimensions_within_radius(radius)
        building_config = BuildingConfig(
            width=width,
            length=length,
            story_count=aesthetics.desired_story_count,
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(max_rooms=1),
            hallway_builder=DefaultHallwayBuilder(),
            window_builder=WindowBuilder(),
            is_climbable=is_climbable,
            aesthetics=aesthetics,
        )
        building = build_building(building_config, 1, rand)
        initial_room_id = 0
        if add_entrance:
            rebuild_with_aligned_entrance(building, rand, initial_room_id)

        main_room = only(building.stories[0].rooms)
        (
            spawn_tile,
            food_island_top_left_tile,
            food_island_size,
            food_island_height,
            stone_tiles,
            stone_size,
        ) = self.get_build_params(difficulty, rand, building)
        food_island_position = Position(*food_island_top_left_tile)
        room_with_island, food_island_tiles = add_food_island(
            main_room, food_island_position, food_island_size, food_island_height
        )

        building.stories[0].rooms[0] = room_with_island

        tile_size = 1
        stone_info = []
        for stone_tile in stone_tiles:
            # todo: consistent naming!!
            floor_elevation = room_with_island.floor_heightmap[stone_tile[1], stone_tile[0]] - DEFAULT_FLOOR_THICKNESS
            stone_location_in_room = (
                stone_tile[0] + tile_size / 2,
                floor_elevation + stone_size / 2,
                stone_tile[1] + tile_size / 2,
            )
            # todo: pull out to other fn
            stone_mass = _decide_boulder_mass(
                rand, difficulty, min_mass=50, max_mass=500, lighter_is_harder=True
            )  # lighter stones = more unstable?
            stone_info.append(StoneInfo(stone_location_in_room, stone_size, stone_mass))

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

        room_offset = Vector3(main_room.position.x, 0, main_room.position.z)
        spawn_location_in_building = local_to_global_coords(np.array(spawn_location_in_room), room_offset)
        target_location_in_building = local_to_global_coords(np.array(target_location_in_room), room_offset)

        stones = []
        for stone_location_in_room, stone_size, stone_mass in stone_info:
            stone_location_in_building = local_to_global_coords(np.array(stone_location_in_room), room_offset)
            stone_location = local_to_global_coords(stone_location_in_building, building.position)
            stones.append(Stone(position=stone_location, size=stone_size, mass=stone_mass))

        return building, stones, spawn_location_in_building, target_location_in_building


def make_sliding_door(
    story: Story,
    door_tile: Position,
    door_azimuth: Azimuth,
    door_face_axis: Axis,
    slide_right: bool = True,
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH,
) -> SlidingDoor:
    tile_width = 1
    door_width = 1
    positive_floor_depth = story.floor_heightmap[door_tile.z, door_tile.x]
    extra_positive_floor_depth = positive_floor_depth - DEFAULT_FLOOR_THICKNESS
    door_height = (story.inner_height - extra_positive_floor_depth) / 1.1
    door_thickness = 0.1
    door_wall_gap = 0.0

    if door_azimuth == Azimuth.NORTH:
        door_centroid_2d = (door_tile.x + door_width / 2, door_tile.z + door_thickness / 2 + door_wall_gap)
        door_rotation_degrees = 0
        side_tiles = [Vector2(door_tile.x - 1, door_tile.z), Vector2(door_tile.x + 1, door_tile.z)]
    elif door_azimuth == Azimuth.EAST:
        door_centroid_2d = (
            door_tile.x + tile_width - door_thickness / 2 - door_wall_gap,
            door_tile.z + door_width / 2,
        )
        door_rotation_degrees = -90
        side_tiles = [Vector2(door_tile.x, door_tile.z - 1), Vector2(door_tile.x + 1, door_tile.z + 1)]
    elif door_azimuth == Azimuth.SOUTH:
        door_centroid_2d = (
            door_tile.x + door_width / 2,
            door_tile.z + tile_width - door_thickness / 2 - door_wall_gap,
        )
        door_rotation_degrees = 180
        side_tiles = [Vector2(door_tile.x + 1, door_tile.z), Vector2(door_tile.x - 1, door_tile.z)]
    elif door_azimuth == Azimuth.WEST:
        door_centroid_2d = (door_tile.x + door_thickness / 2 + door_wall_gap, door_tile.z + door_width / 2)
        door_rotation_degrees = 90
        side_tiles = [Vector2(door_tile.x, door_tile.z + 1), Vector2(door_tile.x, door_tile.z - 1)]
    else:
        raise SwitchError(door_azimuth)
    can_slide_left = story.get_room_at_point(side_tiles[0]) is not None
    can_slide_right = story.get_room_at_point(side_tiles[1]) is not None
    assert can_slide_right or can_slide_left, "Can't place sliding door - no free space on either side"

    door_floor_gap = 0.1
    door_location = np.array(
        [
            door_centroid_2d[0],
            story.floor_negative_depth + door_height / 2 + door_floor_gap,
            door_centroid_2d[1],
        ]
    )
    door_rotation = Rotation.from_euler("y", door_rotation_degrees, degrees=True).as_matrix().flatten()
    return SlidingDoor(
        0,
        door_location,
        size=np.array([door_width, door_height, door_thickness]),
        rotation=door_rotation,
        face_axis=door_face_axis,
        slide_right=slide_right,
        latching_mechanics=latching_mechanics,
    )


def make_hinge_door(
    story: Story,
    door_tile: Position,
    door_azimuth: Azimuth,
    door_face_axis: Axis,
    hinge_side: HingeSide,
    is_pushable: bool = True,
    is_pullable: bool = True,
    latching_mechanics: LatchingMechanics = LatchingMechanics.NO_LATCH,
) -> HingeDoor:
    hinge_radius = 0.05
    tile_size = 1
    wall_gap = 0.025
    door_width = tile_size - hinge_radius * 2 - wall_gap
    door_vertical_gap = 0.1
    extra_positive_floor_depth = story.floor_heightmap[door_tile.z, door_tile.x] - DEFAULT_FLOOR_THICKNESS
    door_height = story.inner_height - extra_positive_floor_depth - 2 * door_vertical_gap
    door_thickness = 0.075

    # Align the centroid such that the outer frame of the door aligns with the walls
    door_centroid_2d = [door_tile.x, door_tile.z]
    if door_azimuth == Azimuth.NORTH:
        door_rotation_degrees = 0
        door_centroid_2d[0] += tile_size / 2
        door_centroid_2d[1] -= door_thickness / 2
    elif door_azimuth == Azimuth.EAST:
        door_rotation_degrees = -90
        door_centroid_2d[0] += tile_size + door_thickness / 2
        door_centroid_2d[1] += tile_size / 2
    elif door_azimuth == Azimuth.SOUTH:
        door_rotation_degrees = 180
        door_centroid_2d[0] += tile_size / 2
        door_centroid_2d[1] += tile_size + door_thickness / 2
    elif door_azimuth == Azimuth.WEST:
        door_rotation_degrees = 90
        door_centroid_2d[0] -= door_thickness / 2
        door_centroid_2d[1] += tile_size / 2
    else:
        raise SwitchError(door_azimuth)

    door_position = np.array(
        [
            door_centroid_2d[0],
            story.floor_negative_depth + door_height / 2 + door_vertical_gap,
            door_centroid_2d[1],
        ]
    )
    door_rotation = Rotation.from_euler("y", door_rotation_degrees, degrees=True).as_matrix().flatten()
    return HingeDoor(
        0,
        door_position,
        size=np.array([door_width, door_height, door_thickness]),
        rotation=door_rotation,
        hinge_side=hinge_side,
        hinge_radius=hinge_radius,
        face_axis=door_face_axis,
        latching_mechanics=latching_mechanics,
        max_inwards_angle=90 if is_pushable else 0,
        max_outwards_angle=90 if is_pullable else 0,
    )


def make_open_button(door: Door):
    door_width, door_height, door_thickness = door.size
    if isinstance(door, HingeDoor):
        door_width += door.hinge_radius * 2
        multiplier = 1 if door.hinge_side == HingeSide.LEFT else -1
    elif isinstance(door, SlidingDoor):
        multiplier = -1 if door.slide_right else 1
    else:
        raise NotImplementedError(type(door))
    button_width = button_height = 0.2 * door_width
    button_thickness = 0.25
    offset_from_door = 0.25
    button_size = np.array([button_width, button_height, button_thickness])
    button_position = np.array(
        [multiplier * (door_width / 2 + button_width / 2 + offset_from_door), 0, button_thickness / 2]
    )
    return DoorOpenButton(0, is_dynamic=True, position=button_position, size=button_size)


def make_rotating_bar(door: Door):
    door_width, door_height, door_thickness = door.size
    unlatch_angle = 10
    if isinstance(door, HingeDoor):
        door_width += door.hinge_radius * 2
        if door.max_outwards_angle > 0 and door.max_inwards_angle == 0:
            unlatch_angle = 75
        anchor_side = HingeSide.RIGHT if door.hinge_side == HingeSide.LEFT else HingeSide.LEFT
    elif isinstance(door, SlidingDoor):
        anchor_side = HingeSide.LEFT if door.slide_right else HingeSide.RIGHT
    else:
        raise SwitchError(f"Unknown door type: {door.__class__}")
    bar_width = door_width * 0.75
    bar_height = door_height * 0.0375
    bar_thickness = 0.25
    bar_size = np.array([bar_width, bar_height, bar_thickness])
    bar_position_x = -door_width / 2 - bar_width / 4
    if anchor_side == HingeSide.RIGHT:
        bar_position_x = -bar_position_x
    bar_position = np.array([bar_position_x, door_height / 4, bar_thickness / 2 + door_thickness / 2])
    rotation_axis = Axis.Z if door.face_axis == Axis.X else Axis.X
    return RotatingBar(
        0,
        is_dynamic=True,
        position=bar_position,
        size=bar_size,
        rotation_axis=rotation_axis,
        anchor_side=anchor_side,
        unlatch_angle=unlatch_angle,
    )


def make_sliding_bar(door: Door, slot=MountSlot.BOTTOM) -> SlidingBar:
    door_width, door_height, door_thickness = door.size
    if isinstance(door, HingeDoor):
        door_width += door.hinge_radius * 2
        if door.hinge_side == HingeSide.RIGHT:
            x_multiplier = -1
            mount_side = HingeSide.LEFT
        else:
            x_multiplier = 1
            mount_side = HingeSide.RIGHT
    elif isinstance(door, SlidingDoor):
        if door.slide_right:
            x_multiplier = -1
            mount_side = HingeSide.LEFT
        else:
            x_multiplier = 1
            mount_side = HingeSide.RIGHT
    else:
        raise SwitchError(f"Unknown door type {door.__class__}")
    bar_width = door_width * 0.075
    bar_height = door_height * 0.25
    bar_thickness = 0.125
    bar_size = np.array([bar_width, bar_height, bar_thickness])
    if slot == MountSlot.BOTTOM:
        y_multiplier = -1.1
    elif slot == MountSlot.TOP:
        y_multiplier = 1.1 if isinstance(door, HingeDoor) else 1.15  # account for sliding door rail
    else:
        raise SwitchError(slot)

    bar_position = np.array(
        [
            x_multiplier * door_width / 2,
            y_multiplier * (door_height / 2 - bar_height / 2),
            -door_thickness / 2 + bar_thickness / 2,
        ]
    )
    return SlidingBar(
        0,
        is_dynamic=True,
        position=bar_position,
        size=bar_size,
        door_face_axis=door.face_axis,
        mount_slot=slot,
        mount_side=mount_side,
    )


def get_room_centroid_in_building_space(story, room, at_height=None):
    height = at_height if at_height else room.outer_height / 2
    return np.array([room.center.x, story.floor_negative_depth + height, room.center.z])


def get_room_centroid_in_world_space(building, story, room, at_height=None):
    centroid_in_building_space = get_room_centroid_in_building_space(story, room, at_height)
    centroid_in_world_space = local_to_global_coords(centroid_in_building_space, building.position)
    return centroid_in_world_space


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class OpenTaskBuilder(IndoorTaskBuilder):
    def get_build_params(
        self, difficulty: float, rand: np.random.Generator
    ) -> Tuple[Type[Door], Dict, LatchingMechanics, List[Tuple]]:
        difficulty_weights = [4, 0.5, 1]
        difficulties = decompose_weighted_mean(difficulty, difficulty_weights, rand=rand)
        lock_difficulty, latching_difficulty, door_mechanics_difficulty = difficulties

        hinge_side = rand.choice([HingeSide.LEFT, HingeSide.RIGHT])
        door_mechanics_by_difficulty = {
            0.25: (HingeDoor, dict(is_pushable=True, is_pullable=True, hinge_side=hinge_side)),
            0.5: (HingeDoor, dict(is_pushable=True, is_pullable=False, hinge_side=hinge_side)),
            0.75: (HingeDoor, dict(is_pushable=False, is_pullable=True, hinge_side=hinge_side)),
            1: (SlidingDoor, dict(slide_right=rand.choice([False, False]))),
        }
        door_mechanics_difficulty_distribution = stats.norm(door_mechanics_difficulty, 0.125)
        difficulty_weights = np.array(
            [door_mechanics_difficulty_distribution.pdf(x) for x in door_mechanics_by_difficulty.keys()]
        )
        difficulty_weights /= difficulty_weights.sum()
        door_type, door_mechanics_params = tuple(
            rand.choice(list(door_mechanics_by_difficulty.values()), p=difficulty_weights)
        )

        latching_mechanics_by_difficulty = {
            0.25: LatchingMechanics.NO_LATCH,
            0.5: LatchingMechanics.LATCH_ONCE,
            0.75: LatchingMechanics.AUTO_LATCH,
        }
        latching_mechanics_difficulty_distribution = stats.norm(latching_difficulty, 0.125)
        difficulty_weights = np.array(
            [latching_mechanics_difficulty_distribution.pdf(x) for x in latching_mechanics_by_difficulty.keys()]
        )
        difficulty_weights /= difficulty_weights.sum()
        latching_mechanics = rand.choice(list(latching_mechanics_by_difficulty.values()), p=difficulty_weights)

        variants_by_difficulty = {
            0.125: [],
            0.250: [(make_open_button,)],
            0.675: [(make_rotating_bar,)],
            0.725: [(make_rotating_bar,), (make_open_button,)],
        }
        if door_type != HingeDoor or door_mechanics_params["is_pullable"] == False:
            # Sliding bars cannot be used with pullable doors
            variants_by_difficulty.update(
                {
                    0.375: [(make_sliding_bar, dict(slot=MountSlot.BOTTOM))],
                    0.500: [(make_sliding_bar, dict(slot=MountSlot.TOP))],
                    0.750: [
                        (make_sliding_bar, dict(slot=MountSlot.BOTTOM)),
                        (make_sliding_bar, dict(slot=MountSlot.TOP)),
                    ],
                    0.800: [(make_rotating_bar,), (make_sliding_bar, dict(slot=MountSlot.BOTTOM))],
                    0.925: [
                        (make_rotating_bar,),
                        (make_sliding_bar, dict(slot=MountSlot.BOTTOM)),
                        (make_open_button,),
                    ],
                }
            )

        lock_difficulty_distribution = stats.norm(difficulty, 0.125)
        difficulty_weights = np.array([lock_difficulty_distribution.pdf(x) for x in variants_by_difficulty.keys()])
        difficulty_weights /= difficulty_weights.sum()
        variant_indices = range(len(list(variants_by_difficulty.items())))
        variant_index = rand.choice(variant_indices, p=difficulty_weights)
        _variant_difficulty, variant_locks = list(variants_by_difficulty.items())[variant_index]

        return door_type, door_mechanics_params, latching_mechanics, variant_locks

    def build(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        aesthetics: BuildingAestheticsConfig,
        add_entrance: bool,
        is_climbable: bool,
    ) -> IndoorTaskParams:
        width, length = rectangle_dimensions_within_radius(radius)
        if width < 5 and length < 5:
            raise ImpossibleWorldError("Building too small to fit open task")

        building_config = BuildingConfig(
            width=width,
            length=length,
            story_count=aesthetics.desired_story_count,
            # TODO(mx): get T/L working here - needs to build a wall between separate rooms
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(min_room_size=1, max_rooms=2),
            hallway_builder=DefaultHallwayBuilder(),
            # All the action takes place on the first story; any extra stories are purely aesthetic and not connected
            story_linker=None,
            window_builder=WindowBuilder(),
            is_climbable=is_climbable,
            aesthetics=aesthetics,
        )

        building = build_building(building_config, 1, rand)
        initial_room_id = 0
        if add_entrance:
            building = rebuild_with_aligned_entrance(building, rand, initial_room_id)

        story = building.stories[0]
        initial_room = story.rooms[initial_room_id]
        spawn_location = get_room_centroid_in_building_space(story, initial_room, at_height=AGENT_HEIGHT / 2)

        other_rooms = [room for room in story.rooms if room != initial_room]
        target_room = only(other_rooms)
        target_location = get_room_centroid_in_building_space(story, target_room, at_height=AGENT_HEIGHT / 2)

        hallway = only(
            [
                hallway
                for hallway in story.hallways
                if {initial_room.id, target_room.id} == {hallway.from_room_id, hallway.to_room_id}
            ]
        )

        if hallway.from_room_id == initial_room.id:
            door_tile = hallway.points[0]
            door_azimuth = hallway.from_room_azimuth
        else:
            door_tile = hallway.points[-1]
            door_azimuth = hallway.to_room_azimuth

        if door_azimuth in {Azimuth.NORTH, Azimuth.SOUTH}:
            door_face_axis = Axis.X
        else:
            door_face_axis = Axis.Z

        door_type, door_mechanics_params, latching_mechanics, variant_locks = self.get_build_params(difficulty, rand)
        if door_type is HingeDoor:
            make_door = make_hinge_door
        elif door_type is SlidingDoor:
            make_door = make_sliding_door
        else:
            raise NotImplementedError(door_type)

        door = make_door(
            story,
            door_tile,
            door_azimuth,
            door_face_axis,
            latching_mechanics=latching_mechanics,
            **door_mechanics_params,
        )

        locks = []
        for make_lock, *params in variant_locks:
            params = only(params) if params else {}
            locks.append(make_lock(door, **params))
        with door.mutable_clone() as locked_door:
            locked_door.locks = locks
            door = locked_door

        return building, [door], spawn_location, target_location


def rebuild_with_aligned_entrance(
    building: Building, rand: np.random.Generator, initial_room_id: int, entrance_azimuth: Optional[Azimuth] = None
) -> Building:
    entrance_builder = DefaultEntranceBuilder(top=False, bottom=True)
    permitted_azimuths = [entrance_azimuth] if entrance_azimuth else None
    entrance = entrance_builder.add_entrance(
        building.stories[0], rand, permitted_room_ids=[initial_room_id], permitted_azimuths=permitted_azimuths
    )
    # todo: this is confusing to reason about; fix
    # If we're going from the interior room, what direction is the hallway pointing?
    required_yaw_rotation = -entrance.azimuth.angle_from_positive_x
    return building.rebuild_rotated(required_yaw_rotation)


def get_radius_for_building_task(rand: np.random.Generator, task: BuildingTask, difficulty: float) -> float:
    if task == BuildingTask.OPEN:
        return difficulty_variation(5.0, 12.0, rand, difficulty)
    elif task == BuildingTask.NAVIGATE:
        return difficulty_variation(5.0, 25.0, rand, difficulty)
    elif task == BuildingTask.STACK:
        return difficulty_variation(8.0, 12.0, rand, difficulty)
    elif task == BuildingTask.PUSH:
        return difficulty_variation(10.0, 15.0, rand, difficulty)
    else:
        raise SwitchError(f"Unknown BuildingTask: {task}")


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class NavigateTaskBuilder(IndoorTaskBuilder):
    def get_build_params(
        self, difficulty: float, rand: np.random.Generator, nav_graph: BuildingNavGraph, initial_node: str
    ) -> Any:
        distance_by_location_id = nx.single_source_dijkstra_path_length(nav_graph, initial_node, weight="distance")
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
        return target_node

    def build(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        aesthetics: BuildingAestheticsConfig,
        add_entrance: bool,
        is_climbable: bool,
    ) -> IndoorTaskParams:
        # todo: move building config generation out of build() as well
        story_count = 1 + round(rand.normal(loc=difficulty, scale=0.1))
        width, length = rectangle_dimensions_within_radius(radius)
        if width <= 10 or length <= 10:
            footprint_builder = RectangleFootprintBuilder()
            min_room_size = 2
        else:
            footprint_builder = TLShapeFootprintBuilder()
            min_room_size = 3

        building_config = BuildingConfig(
            width=width,
            length=length,
            story_count=story_count,
            footprint_builder=footprint_builder,
            room_builder=HouseLikeRoomBuilder(min_room_size=min_room_size),
            hallway_builder=DefaultHallwayBuilder(proportion_additional_edges=1),
            story_linker=DefaultStoryLinker(raise_on_failure=False),
            window_builder=WindowBuilder(),
            is_climbable=is_climbable,
            aesthetics=aesthetics,
        )

        building = build_building(building_config, 1, rand)
        first_story = building.stories[0]
        if len(first_story.rooms) == 1:
            raise ImpossibleWorldError("Building too small for navigate task")

        exterior_wall_azimuths_by_room_id = first_story.get_exterior_wall_azimuths_by_room_id()
        periphery_room_ids = [
            room_id
            for room_id, exterior_wall_azimuths in exterior_wall_azimuths_by_room_id.items()
            if len(exterior_wall_azimuths) > 0
        ]
        initial_room_id = rand.choice(periphery_room_ids)
        if add_entrance:
            building = rebuild_with_aligned_entrance(building, rand, initial_room_id)

        first_story = building.stories[0]
        initial_room = first_story.rooms[initial_room_id]
        spawn_location = get_room_centroid_in_building_space(first_story, initial_room, at_height=AGENT_HEIGHT / 2)

        nav_graph = BuildingNavGraph(building)
        initial_node = nav_graph.get_room_node(first_story, initial_room)
        target_node = self.get_build_params(difficulty, rand, nav_graph, initial_node)

        target_location = np.array(nav_graph.nodes[target_node]["position"])
        target_location[1] += FOOD_HOVER_DIST

        if IS_DEBUG_VIS:
            ax = nav_graph.plot()
            ax.text(
                spawn_location[2], spawn_location[0], spawn_location[1] - (AGENT_HEIGHT // 2), "spawn", color="blue"
            )
            ax.text(target_location[2], target_location[0], target_location[1], "target", color="green")
            plt.show()

        extra_items = []
        return building, extra_items, spawn_location, target_location


def translate_entities_to_world_space(building: Building, entities: List[Entity]):
    translated_entities = []
    for entity in entities:
        with entity.mutable_clone() as offset_entity:
            offset_entity.position = get_location_in_world_space(entity.position, building)
            if hasattr(offset_entity, "rotation") and building.yaw_degrees != 0:
                offset_entity.rotation = (
                    Rotation.from_euler("y", building.yaw_degrees, degrees=True).as_matrix().flatten()
                )
            translated_entities.append(offset_entity)
    return translated_entities


def make_indoor_task_world(
    building: Building,
    extra_entities: List[Entity],
    difficulty: float,
    spawn_location: np.ndarray,
    target_location: np.ndarray,
    output_path: Path,
    rand: np.random.Generator,
    export_config: ExportConfig,
):
    if IS_DEBUG_VIS:
        building.plot()

    id_generator = IdGenerator()
    world = create_indoor_only_world(rand, building, export_config)
    world.items.extend(
        [
            *[attr.evolve(entity, entity_id=id_generator.get_next_id()) for entity in extra_entities],
            _get_spawn(id_generator, rand, difficulty, spawn_location, target_location),
            CANONICAL_FOOD_CLASS(entity_id=id_generator.get_next_id(), position=target_location),
        ]
    )

    export_skill_world(output_path, rand, world)


def get_location_in_world_space(location_in_building: np.ndarray, building: Building):
    translated = local_to_global_coords(location_in_building, building.offset_point)
    offset_from_centroid = translated - building.position
    rotation = Rotation.from_euler("y", building.yaw_degrees, degrees=True)
    rotated = rotation.apply(offset_from_centroid)
    rotated_and_offset = rotated + building.position
    return rotated_and_offset


def create_building_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    task: BuildingTask,
    site_radius: float,
    location: np.ndarray,
    yaw_radians: float,
    is_climbable: bool = True,
    aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig(),
) -> Tuple[Building, List[Entity], np.ndarray, np.ndarray]:
    # todo: entrance_difficulty (None = no door)
    """radius: the building should fit within a circle of this size, centered on the building location"""
    builder_by_task = {
        BuildingTask.NAVIGATE: NavigateTaskBuilder,
        BuildingTask.OPEN: OpenTaskBuilder,
        BuildingTask.PUSH: PushTaskBuilder,
        BuildingTask.STACK: StackTaskBuilder,
    }
    builder = builder_by_task[task]()
    building, extra_entities, spawn_location_in_building, target_location_in_building = builder.build(
        rand, difficulty, site_radius, aesthetics, add_entrance=True, is_climbable=is_climbable
    )
    building = building.with_transform(new_position=Vector3(*location), new_yaw_degrees=math.degrees(yaw_radians))

    spawn_location = get_location_in_world_space(spawn_location_in_building, building)
    target_location = get_location_in_world_space(target_location_in_building, building)
    translated_entities = translate_entities_to_world_space(building, extra_entities)
    return building, translated_entities, spawn_location, target_location


def get_building_positive_height_at_point(point: np.ndarray, building: Building) -> float:
    """
    Positive height refers to the exterior building height from it's y origin.
    Returns np.nan if point is not within building footprint.
    NOTE: both point and building must be in world space (rotated and positioned)
    """
    outline = building.get_footprint_outline()
    if IS_DEBUG_VIS:
        x, z = zip(*outline.exterior.coords)
        ax = plt.axes()
        ax.plot(x, z)
        ax.plot(point[0], point[2], "ro")
        ax.invert_yaxis()
        plt.gca().set_aspect("equal")
        plt.show()
    point = Point(point[0], point[2])
    if not outline.contains(point):
        # Note: points ON the edge are not included
        return np.nan
    else:
        return building.get_positive_height_at_point(point)


def create_indoor_only_world(rand: np.random.Generator, building: Building, export_config: ExportConfig):
    first_story = building.stories[0]
    furthest_distance = max(
        abs(building.position.x) + first_story.width / 2, abs(building.position.z) + first_story.length / 2
    )
    config = WorldConfig(
        seed=rand.integers(0, np.iinfo(np.int64).max),
        x_tile_count=1,
        z_tile_count=1,
        size_in_meters=2 * furthest_distance,
        point_density_in_points_per_square_meter=0.2,
        initial_point_count=1,
        initial_noise_scale=0,
        fractal_iteration_count=0,
        noise_scale_decay=0,
        mountain_noise_count=0,
        mountain_radius=0,
        mountain_offset=0,
        mountain_radius_decay=0,
        mountain_noise_scale=0,
        is_mountain_placement_normal_distribution=True,
        final_max_altitude_meters=0,
        is_indoor_only=True,
    )
    biome_config = generate_biome_config(rand, export_config, 10.0, 0.0)
    export_config = attr.evolve(export_config, is_biome_fast=True)
    world = NewWorld.build(config, export_config, biome_config, is_debug_graph_printing_enabled=False)
    world.add_building(building, np.ones_like(world.map.Z))
    return world
