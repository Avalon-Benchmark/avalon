import math
from itertools import product
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import cast

import attr
import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation

from avalon.common.utils import only
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.biome import generate_biome_config
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.world import WorldConfig
from avalon.datagen.world_creation.constants import BOULDER_MAX_MASS
from avalon.datagen.world_creation.constants import BOULDER_MIN_MASS
from avalon.datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from avalon.datagen.world_creation.debug_plots import IS_DEBUG_VIS
from avalon.datagen.world_creation.entities.doors.hinge_door import HingeDoor
from avalon.datagen.world_creation.entities.doors.sliding_door import SlidingDoor
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.entities.food import CANONICAL_FOOD_CLASS
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import local_to_global_coords
from avalon.datagen.world_creation.indoor.builders import DefaultEntranceBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.components import Entrance
from avalon.datagen.world_creation.indoor.components import Room
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import TILE_SIZE
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.doors import get_door_params_from_difficulty
from avalon.datagen.world_creation.indoor.doors import make_entrance_hinge_door
from avalon.datagen.world_creation.indoor.doors import make_entrance_sliding_door
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world import _get_spawn
from avalon.datagen.world_creation.worlds.world import build_building

EntranceSite = Tuple[int, int, Tuple[Azimuth, ...]]
IndoorTaskParams = Tuple[Building, Sequence[Entity], Point3DNP, Point3DNP, Tuple[EntranceSite, ...]]

CANONICAL_BUILDING_LOCATION = np.array([0.0, 2.0, 0.0])

PrincipalObstacleParams = TypeVar("PrincipalObstacleParams", bound=Tuple[Any, ...])


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class BuildingTaskGenerator(Generic[PrincipalObstacleParams]):
    config: IndoorTaskConfig = IndoorTaskConfig()

    def get_site_radius(self, rand: np.random.Generator, difficulty: float) -> float:
        raise NotImplementedError

    def get_building_config(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        allowed_auxiliary_tasks: Tuple[BuildingTask, ...] = tuple(),
        aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig(),
    ) -> BuildingConfig:
        raise NotImplementedError

    def get_principal_obstacle_params(
        self, rand: np.random.Generator, difficulty: float, building: Building
    ) -> PrincipalObstacleParams:
        raise NotImplementedError

    def add_principal_obstacles(
        self,
        rand: np.random.Generator,
        building: Building,
        obstacle_params: PrincipalObstacleParams,
    ) -> IndoorTaskParams:
        raise NotImplementedError

    def get_entrance_obstacle_params(
        self,
        rand: np.random.Generator,
        difficulty: float,
        building: Building,
        entrance: Entrance,
        allowed_entrance_tasks: Tuple[BuildingTask, ...],
    ) -> Optional[Tuple]:
        if difficulty == 0:
            return None
        if BuildingTask.OPEN in allowed_entrance_tasks:
            return get_door_params_from_difficulty(rand, difficulty)
        return None

    def add_entrance(
        self, rand: np.random.Generator, building: Building, viable_entrance_sites: Tuple[EntranceSite, ...]
    ) -> None:
        site_idx = rand.choice(range(len(viable_entrance_sites)))
        story_id, room_id, permitted_azimuths = viable_entrance_sites[site_idx]
        entrance_builder = DefaultEntranceBuilder(top=False, bottom=True)
        entrance_builder.add_entrance(
            building.stories[story_id], rand, permitted_room_ids=[room_id], permitted_azimuths=permitted_azimuths
        )

    def add_entrance_obstacles(
        self, rand: np.random.Generator, building: Building, story: Story, entrance: Entrance, obstacle_params: Tuple
    ) -> Tuple[Building, List[Entity]]:
        door_type, door_mechanics_params, latching_mechanics, variant_locks = obstacle_params
        if door_type is HingeDoor:
            make_entrance_door = make_entrance_hinge_door
        elif door_type is SlidingDoor:
            make_entrance_door = make_entrance_sliding_door  # type: ignore
        else:
            raise NotImplementedError(door_type)

        door = make_entrance_door(
            story,
            entrance,
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

        return building, [door]

    @staticmethod
    def _position_from_tile(room: Room, tile: BuildingTile, at_height: float, tile_size: int = TILE_SIZE) -> Point3DNP:
        return np.array(
            [
                tile.x + tile_size / 2,
                room.floor_heightmap[tile.z, tile.x] + at_height,
                tile.z + tile_size / 2,
            ]
        )


def decide_spawn_room_and_tile(
    rand: np.random.Generator,
    story: Story,
    permitted_tiles_by_room_id: Optional[Dict[int, List[Tuple[int, int]]]] = None,
) -> Tuple[Room, Tuple[int, int]]:
    if entrances := story.entrances:
        entrance = rand.choice(entrances)  # type: ignore[arg-type]
        spawn_room, landing_position = entrance.get_connected_room_and_landing_position(story)
        spawn_tile = landing_position.x, landing_position.z
        if permitted_tiles_by_room_id is not None:
            if (
                spawn_room.id not in permitted_tiles_by_room_id
                or spawn_tile not in permitted_tiles_by_room_id[spawn_room.id]
            ):
                raise ImpossibleWorldError("Entrance was spawned in a location outside the spawn region")
    else:
        if permitted_tiles_by_room_id is not None:
            spawn_room_id = rand.choice(list(permitted_tiles_by_room_id.keys()))
            spawn_room = story.rooms[spawn_room_id]
            spawn_tile = tuple(rand.choice(permitted_tiles_by_room_id[spawn_room_id]))  # type: ignore
        else:
            spawn_room: Room = rand.choice(story.rooms)  # type: ignore
            spawn_tile = rand.integers(0, spawn_room.width - 1), rand.integers(0, spawn_room.length - 1)
    return spawn_room, spawn_tile


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
    return cast(float, min(max(boulder_mass_distribution.rvs(random_state=rand), min_mass), max_mass))


def add_food_island(
    room: Room, position: BuildingTile, size: int, height: float
) -> Tuple[Room, List[Tuple[int, int]]]:
    """
    Place an elevated square platform in the room with `position` as its top-left tile.
    Position should be passed in ROOM coordinate space.
    Returns all its tiles as (x,z) coordinates (in room space).
    """
    if size >= room.width or size >= room.length:
        raise ImpossibleWorldError(f"Room {room.id} is too small to place a food island of size {size}")

    new_heightmap = room.floor_heightmap.copy()
    new_heightmap[position.z : position.z + size, position.x : position.x + size] = DEFAULT_FLOOR_THICKNESS + height
    if height > 3 and size > 1:
        center_z = position.z + size // 2
        center_x = position.x + size // 2
        new_heightmap[center_z, center_x] += MAX_JUMP_HEIGHT_METERS * 0.95
    tiles = list(product(range(position.x, position.x + size), range(position.z, position.z + size)))
    new_outer_height = room.outer_height + (new_heightmap.max() - DEFAULT_FLOOR_THICKNESS)
    updated_room = attr.evolve(room, outer_height=new_outer_height, floor_heightmap=new_heightmap)
    return updated_room, tiles


def translate_entities_to_world_space(building: Building, entities: Sequence[Entity]):
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


MIN_NAVIGATE_BUILDING_SIZE = 5.0


def rectangle_dimensions_within_radius(radius: float) -> Tuple[int, int]:
    size = math.floor(2 * radius / math.sqrt(2))
    return size, size


def get_room_centroid_in_building_space(
    building: Building, story: Story, room: Room, at_height: Optional[float] = None
) -> Point3DNP:
    tile_size = 1
    height = at_height if at_height else room.outer_height / 2
    story_y_offset = building.get_story_y_offset(story.num)
    center_tile_elevation = room.floor_heightmap[room.length // 2, room.width // 2]
    return np.array(
        [room.center.x + tile_size / 2, story_y_offset + center_tile_elevation + height, room.center.z + tile_size / 2]
    )


def get_room_centroid_in_world_space(
    building: Building, story: Story, room: Room, at_height: Optional[float] = None
) -> Point3DNP:
    centroid_in_building_space = get_room_centroid_in_building_space(building, story, room, at_height)
    centroid_in_world_space = local_to_global_coords(centroid_in_building_space, building.position)
    return centroid_in_world_space


def rebuild_with_aligned_entrance(
    building: Building, rand: np.random.Generator, initial_room_id: int, entrance_azimuth: Optional[Azimuth] = None
) -> Building:
    entrance_builder = DefaultEntranceBuilder(top=False, bottom=True)
    permitted_azimuths = [entrance_azimuth] if entrance_azimuth else None
    entrance = entrance_builder.add_entrance(
        building.stories[0], rand, permitted_room_ids=[initial_room_id], permitted_azimuths=permitted_azimuths
    )
    required_yaw_rotation = -entrance.azimuth.angle_from_positive_x
    return building.rebuild_rotated(required_yaw_rotation)


def make_indoor_task_world(
    building: Building,
    extra_entities: List[Entity],
    difficulty: float,
    spawn_location: Point3DNP,
    target_location: Point3DNP,
    rand: np.random.Generator,
    export_config: ExportConfig,
):
    if IS_DEBUG_VIS:
        building.plot()

    first_story = building.stories[0]
    furthest_distance = max(
        abs(building.position[0]) + first_story.width / 2, abs(building.position[2]) + first_story.length / 2
    )
    config = WorldConfig(
        seed=rand.integers(0, np.iinfo(np.int64).max),
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
    world = World.build(config, export_config, biome_config, is_debug_graph_printing_enabled=False)
    building, world = world.add_building(building, np.ones_like(world.map.Z))
    world = world.add_items(
        [
            *extra_entities,
            _get_spawn(rand, difficulty, spawn_location, target_location),
            CANONICAL_FOOD_CLASS(position=target_location),
        ]
    )

    return world


def get_location_in_world_space(location_in_building: Point3DNP, building: Building):
    translated = local_to_global_coords(location_in_building, building.offset_point)
    offset_from_centroid = translated - building.position
    rotation = Rotation.from_euler("y", building.yaw_degrees, degrees=True)
    rotated = rotation.apply(offset_from_centroid)
    rotated_and_offset = rotated + building.position
    return rotated_and_offset


def create_building_for_skill_scenario(
    rand: np.random.Generator,
    difficulty: float,
    task_generator: BuildingTaskGenerator,
    position: Point3DNP,
    site_radius: Optional[float] = None,
    allowed_auxiliary_tasks: Tuple[BuildingTask, ...] = tuple(),
    allowed_entrance_tasks: Tuple[BuildingTask, ...] = tuple(),
    aesthetics_config: BuildingAestheticsConfig = BuildingAestheticsConfig(),
    is_indoor_only: bool = False,
):
    site_radius = task_generator.get_site_radius(rand, difficulty) if site_radius is None else site_radius

    default_building_id = 0
    building_config = task_generator.get_building_config(
        rand, difficulty, site_radius, allowed_auxiliary_tasks, aesthetics_config
    )
    building = build_building(building_config, default_building_id, rand)
    obstacle_params = task_generator.get_principal_obstacle_params(rand, difficulty, building)
    (
        building,
        entities,
        spawn_location_in_building,
        target_location_in_building,
        entrance_sites,
    ) = task_generator.add_principal_obstacles(rand, building, obstacle_params)

    if not is_indoor_only:
        task_generator.add_entrance(rand, building, entrance_sites)

        for story in building.stories:
            for entrance in story.entrances:
                add_entrance_obstacles = len(allowed_entrance_tasks) > 0 and rand.uniform() < difficulty
                if not add_entrance_obstacles:
                    continue

                entrance_obstacle_params = task_generator.get_entrance_obstacle_params(
                    rand, difficulty, building, entrance, allowed_entrance_tasks
                )
                if entrance_obstacle_params:
                    building, additional_entities = task_generator.add_entrance_obstacles(
                        rand, building, story, entrance, entrance_obstacle_params
                    )
                    entities = [*entities, *additional_entities]

    building = building.with_transform(new_position=position)
    spawn_location = get_location_in_world_space(spawn_location_in_building, building)
    target_location = get_location_in_world_space(target_location_in_building, building)
    translated_entities = translate_entities_to_world_space(building, entities)
    return building, translated_entities, spawn_location, target_location
