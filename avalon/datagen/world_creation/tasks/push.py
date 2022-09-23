import math
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from avalon.common.errors import SwitchError
from avalon.common.log_utils import logger
from avalon.common.utils import only
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import BOULDER_HEIGHT
from avalon.datagen.world_creation.constants import BOULDER_MAX_MASS
from avalon.datagen.world_creation.constants import BOULDER_MIN_MASS
from avalon.datagen.world_creation.constants import FOOD_HOVER_DIST
from avalon.datagen.world_creation.constants import JUMPING_REQUIRED_HEIGHT
from avalon.datagen.world_creation.constants import MAX_FLAT_JUMP_DIST
from avalon.datagen.world_creation.constants import MAX_FLAT_JUMP_METERS
from avalon.datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from avalon.datagen.world_creation.constants import MIN_BRIDGE_DIST
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.tools.boulder import Boulder
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import euclidean_distance
from avalon.datagen.world_creation.geometry import local_to_global_coords
from avalon.datagen.world_creation.geometry import midpoint
from avalon.datagen.world_creation.geometry import squares_overlap
from avalon.datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import WindowBuilder
from avalon.datagen.world_creation.indoor.builders import get_free_obstacle_sites
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.components import Room
from avalon.datagen.world_creation.indoor.components import Wall
from avalon.datagen.world_creation.indoor.components import Window
from avalon.datagen.world_creation.indoor.constants import CEILING_THICKNESS
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import Azimuth
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
from avalon.datagen.world_creation.indoor.tiles import draw_line_in_grid
from avalon.datagen.world_creation.indoor.tiles import tile_centroid
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.types import HeightMode
from avalon.datagen.world_creation.utils import decompose_weighted_mean
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.utils import add_offsets
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations

_EXTRA_SAFETY_RADIUS = 3.0
_MIN_PUSH_TASK_DISTANCE = MIN_BRIDGE_DIST + 2 * _EXTRA_SAFETY_RADIUS


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PushIndoorTaskConfig(IndoorTaskConfig):
    # Smallest and largest possible site radius for a building with this task to be placed on.
    min_site_radius: float = 10.0
    max_site_radius: float = 15.0
    # Standard deviation for the distribution deciding the site radius: higher means more variability at same difficulty
    site_radius_std_dev: float = 1.0
    # Total number of stories for the building. Note that this task is a single-story task, all other stories will
    # be purely aesthetic (e.g. to help with being viewed in the distance outdoors).
    story_count: int = 2
    # Boulder size that needs to be pushed; mainly determines visibility while pushing
    blocked_entrance_boulder_size: float = 2
    # Square island side length; should be large enough to prevent knocking food off by reaching or jumping
    climb_higher_food_island_size: int = 5


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class PushOutdoorTaskConfig(TaskConfig):
    # how large of a gap to use for the push task in difficulty 0.0 and 1.0 worlds respectively
    gap_dist_easy: float = _MIN_PUSH_TASK_DISTANCE
    gap_dist_hard: float = _MIN_PUSH_TASK_DISTANCE + MAX_FLAT_JUMP_DIST
    # how high to make the chasm. Unfortunately needs to be really quite a narrow range because you need to be able to
    # push the boulder in, then jump on it, and still have enough space to be able to jump out on the other side
    chasm_depth_easy: float = JUMPING_REQUIRED_HEIGHT
    chasm_depth_hard: float = BOULDER_HEIGHT + JUMPING_REQUIRED_HEIGHT - 0.5
    # how far away to possibly put the boulder
    boulder_dist_easy: float = 0.0
    boulder_dist_hard: float = 10.0
    # how heavy to make the boulder
    boulder_mass_easy: float = BOULDER_MIN_MASS
    boulder_mass_hard: float = BOULDER_MAX_MASS
    boulder_mass_std_dev: float = 10.0
    # the minimum chance (ie, at difficulty=1.0) that the inside wall will be climbable
    final_is_inside_climbable_probability: float = 0.2


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class PushTaskConfig(TaskConfig):
    # likelihood that the indoor variant of the task will be used
    indoor_probability: float = 0.2
    # the configs for each variant of the task
    indoor_config: PushIndoorTaskConfig = PushIndoorTaskConfig()
    outdoor_config: PushOutdoorTaskConfig = PushOutdoorTaskConfig()


def generate_push_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: PushTaskConfig = PushTaskConfig(),
) -> None:
    is_indoor = rand.uniform() < task_config.indoor_probability
    if is_indoor:
        building, entities, spawn_location, target_location = create_building_for_skill_scenario(
            rand,
            difficulty,
            PushTaskGenerator(task_config.indoor_config),
            position=CANONICAL_BUILDING_LOCATION,
            is_indoor_only=True,
        )
        world = make_indoor_task_world(
            building, entities, difficulty, spawn_location, target_location, rand, export_config
        )
    else:
        world, locations, difficulty = create_push_obstacle(
            rand, difficulty, export_config, task_config=task_config.outdoor_config
        )
        world, locations = world.end_height_obstacles(
            locations, is_accessible_from_water=False, is_spawn_region_climbable=False
        )
        world = add_food_tree_for_simple_task(world, locations)
        world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)

    export_world(output_path, rand, world)


def create_push_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    task_config: PushOutdoorTaskConfig = PushOutdoorTaskConfig(),
) -> Tuple[World, WorldLocations, float]:

    is_inside_climbable, difficulty = select_boolean_difficulty(
        difficulty, rand, final_prob=task_config.final_is_inside_climbable_probability
    )

    desired_goal_dist = (task_config.gap_dist_easy * 3.0) + (
        task_config.gap_dist_easy * 4.0 * difficulty * rand.uniform()
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    desired_gap_distance = scale_with_difficulty(difficulty, task_config.gap_dist_easy, task_config.gap_dist_hard)
    gap_distance, _warnings = world.get_critical_distance(locations, task_config.gap_dist_easy, desired_gap_distance)

    height = scale_with_difficulty(difficulty, task_config.chasm_depth_easy, task_config.chasm_depth_hard)

    if gap_distance is None:
        raise WorldTooSmall(AvalonTask.PUSH, task_config.gap_dist_easy, locations.get_2d_spawn_goal_distance())
    gap_distance -= 2 * _EXTRA_SAFETY_RADIUS
    if gap_distance < MIN_BRIDGE_DIST:
        raise WorldTooSmall(AvalonTask.PUSH, gap_distance, MIN_BRIDGE_DIST)
    logger.trace(f"Creating a {gap_distance} meter gap")
    randomization_dist = scale_with_difficulty(
        difficulty, task_config.boulder_dist_easy, task_config.boulder_dist_hard
    )
    boulder_mass = normal_distrib_range(
        task_config.boulder_mass_easy,
        task_config.boulder_mass_hard,
        task_config.boulder_mass_std_dev,
        rand,
        difficulty,
    )
    boulder = Boulder(position=np.array([-1.0, 0.0, 0.0]), mass=boulder_mass)
    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance,
        constraint=constraint,
        height=-height,
        traversal_width=normal_distrib_range(10.0, 3.0, 1.0, rand, difficulty),
        is_inside_climbable=True,
        is_outside_climbable=False,
        dual_solution=HeightSolution(
            solution_point_brink_distance=1.0,
            inside_items=tuple(add_offsets([boulder])),
            inside_item_randomization_distance=randomization_dist,
            inside_item_radius=1.0,
            paths=(
                HeightPath(
                    is_solution_flattened=True,
                    is_path_restricted_to_land=False,
                    is_chasm_bottom_flattened=True,
                    is_height_affected=False,
                    width=randomization_dist * 2.0,
                    flattening_mode="min",
                ),
                HeightPath(
                    is_solution_flattened=False,
                    is_path_restricted_to_land=True,
                    is_chasm_bottom_flattened=False,
                    is_height_affected=False,
                    width=2.0,
                ),
            ),
        ),
        probability_of_centering_on_spawn=0.0 if constraint is None else None,
        height_mode=HeightMode.MIDPOINT_ABSOLUTE,
        expansion_meters=_EXTRA_SAFETY_RADIUS,
    )
    world = world.add_height_obstacle(rand, ring_config, locations.island)
    new_goal = locations.goal.copy()
    new_goal[1] = world.get_height_at(to_2d_point(new_goal)) + CANONICAL_FOOD_HEIGHT_ON_TREE
    locations = attr.evolve(locations, goal=new_goal)
    return world, locations, difficulty


class PushTaskVariant(Enum):
    BLOCKED_ENTRANCE = "blocked_entrance"
    CLIMB_HIGHER = "climb_higher"
    CROSS_CHASM = "cross_chasm"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class PushTaskGenerator(BuildingTaskGenerator):
    config: PushIndoorTaskConfig = PushIndoorTaskConfig()

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
            hallway_builder=DefaultHallwayBuilder(),
            window_builder=WindowBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        difficulty_weights = np.array([2, 1])
        variant_difficulty, param_difficulty = decompose_weighted_mean(difficulty, difficulty_weights, rand=rand)
        variants_by_difficulty = {
            0.25: PushTaskVariant.BLOCKED_ENTRANCE,
            0.5: PushTaskVariant.CLIMB_HIGHER,
            0.75: PushTaskVariant.CROSS_CHASM,
        }
        variant_difficulty_distribution = stats.norm(variant_difficulty, 0.125)
        difficulty_weights = np.array([variant_difficulty_distribution.pdf(x) for x in variants_by_difficulty.keys()])
        difficulty_weights /= difficulty_weights.sum()
        task_variant = rand.choice(list(variants_by_difficulty.values()), p=difficulty_weights)  # type: ignore[arg-type]

        build_fn: Callable[..., Any]
        boulder_mass = _decide_boulder_mass(rand, difficulty)
        if task_variant == PushTaskVariant.BLOCKED_ENTRANCE:
            build_fn = self._add_obstacles_blocked_entrance
            extra_build_params = self._get_blocked_entrance_build_params(building, param_difficulty, rand)
        elif task_variant == PushTaskVariant.CLIMB_HIGHER:
            build_fn = self._add_obstacles_climb_higher
            extra_build_params = self._get_climb_higher_build_params(building, param_difficulty, rand)
        elif task_variant == PushTaskVariant.CROSS_CHASM:
            build_fn = self._add_obstacles_cross_chasm
            extra_build_params = self._get_cross_chasm_build_params(building, param_difficulty, rand)
        else:
            raise SwitchError(task_variant)

        entrance_azimuth = Azimuth.EAST
        return build_fn, extra_build_params, boulder_mass, entrance_azimuth

    def _get_blocked_entrance_build_params(
        self, building: Building, difficulty: float, rand: np.random.Generator
    ) -> Dict:
        tile_size = 1
        boulder_size = self.config.blocked_entrance_boulder_size
        min_enclosure_width = math.ceil(boulder_size) + tile_size  # >1 extra tile to be able to walk around it

        first_story = building.stories[0]
        room = only(first_story.rooms)

        free_sites = [
            site
            for site in get_free_obstacle_sites(building.stories[:1])
            if min_enclosure_width <= site.site_index <= room.width - min_enclosure_width - 1 and site.vertical
        ]
        if len(free_sites) == 0:
            raise ImpossibleWorldError("Room too small to create task")
        site_idx = rand.choice(range(len(free_sites)))
        site = free_sites[site_idx]
        hole_tile = site.site_index, site.length // 2
        boulder_tile = hole_tile[0] + 1, hole_tile[1]

        # Spawn region is always on the east, so we can add a positive-x aligned entrance there easily
        # Target region excludes edges when possible to avoid grabbing food through walls
        target_region_x_range = range(site.site_index - 1)
        if len(target_region_x_range) > 1:
            target_region_x_range = range(1, site.site_index - 1)
        target_region_y_range = range(room.length - 1)
        if len(target_region_y_range) > 1:
            target_region_y_range = range(1, room.length - 1)
        target_region_tiles = list(product(target_region_x_range, target_region_y_range))
        spawn_region_tiles = list(product(range(site.site_index + 1, room.width), range(room.length)))
        free_spawn_region_tiles = [
            tile
            for tile in spawn_region_tiles
            if not squares_overlap(
                tile_centroid(tile), tile_size, tile_centroid(boulder_tile, boulder_size), boulder_size
            )
        ]
        spawn_room, spawn_tile = decide_spawn_room_and_tile(rand, first_story, {room.id: free_spawn_region_tiles})
        target_tile = only(decide_tiles_by_distance(target_region_tiles, spawn_tile, difficulty, rand))

        extra_build_params = dict(
            hole_tile=hole_tile,
            spawn_tile=spawn_tile,
            boulder_size=boulder_size,
            target_tile=target_tile,
        )
        return extra_build_params

    def _get_climb_higher_build_params(self, building: Building, difficulty: float, rand: np.random.Generator) -> Dict:
        first_story = building.stories[0]
        spawn_room, spawn_tile = decide_spawn_room_and_tile(rand, first_story)

        food_island_size = self.config.climb_higher_food_island_size
        viable_food_island_tiles = list(
            product(range(spawn_room.width - food_island_size - 1), range(spawn_room.length - food_island_size - 1))
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
        food_island_center_tile = (
            food_island_position.x + food_island_size // 2,
            food_island_position.z + food_island_size // 2,
        )
        if len(viable_boulder_tiles) == 0:
            raise ImpossibleWorldError("Building too small for push task")
        boulder_tile = only(decide_tiles_by_distance(viable_boulder_tiles, food_island_center_tile, difficulty, rand))

        extra_build_params = dict(
            spawn_tile=spawn_tile,
            food_island_top_left_tile=food_island_top_left_tile,
            food_island_size=food_island_size,
            food_island_height=food_island_height,
            boulder_tile=boulder_tile,
            boulder_size=boulder_size,
        )
        return extra_build_params

    def _get_cross_chasm_build_params(self, building: Building, difficulty: float, rand: np.random.Generator) -> Dict:
        boulder_size = MAX_JUMP_HEIGHT_METERS * 0.9
        chasm_depth = (MAX_JUMP_HEIGHT_METERS * 1.8) - DEFAULT_FLOOR_THICKNESS
        chasm_size = round(MAX_FLAT_JUMP_METERS * 2)
        min_launch_side_distance_from_wall = math.ceil(boulder_size)
        min_landing_side_distance_from_wall = 3  # at least 1 tile away from wall and at least 1 tile away from chasm

        tile_size = 1
        first_story = building.stories[0]
        room = only(first_story.rooms)

        viable_x_positions = list(
            range(min_landing_side_distance_from_wall, room.width - min_launch_side_distance_from_wall - chasm_size)
        )
        if len(viable_x_positions) == 0:
            raise ImpossibleWorldError("Room too small to place a chasm")
        chasm_left_x = rand.choice(viable_x_positions)

        spawn_region_tiles = list(product(range(chasm_left_x + chasm_size, room.width), range(room.length)))
        # Target range excludes positions next to walls to avoid grabbing food through walls
        target_x_range = range(1, chasm_left_x - 1)
        target_region_tiles = list(product(target_x_range, range(1, room.length - 1)))
        spawn_room, spawn_tile = decide_spawn_room_and_tile(rand, first_story, {room.id: spawn_region_tiles})

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

    def _add_obstacles_blocked_entrance(
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

        wall_point_set = (
            (BuildingTile(x=hole_tile_x, z=0), BuildingTile(x=hole_tile_x, z=hole_tile_z - 1)),
            (BuildingTile(x=hole_tile_x, z=hole_tile_z + 1), BuildingTile(x=hole_tile_x, z=room.length - 1)),
        )

        # Make wall unjumpable, but low enough to potentially see food on other side
        wall_height = room.outer_height
        new_heightmap = room.floor_heightmap.copy()
        windows = []
        for points in wall_point_set:
            wall = Wall(0, 0, points, 1, wall_height)
            draw_line_in_grid(wall.points, new_heightmap, wall.height, drawable_grid_value=None)

            window_y = DEFAULT_FLOOR_THICKNESS + (wall_height - DEFAULT_FLOOR_THICKNESS) / 2
            window_position_room_coords = midpoint(
                np.array([points[0].x + tile_size / 2, window_y, points[0].z]),
                np.array([points[1].x + tile_size / 2, window_y, points[1].z + tile_size]),
            )
            window_length = round((euclidean_distance(points[0], points[1]) + tile_size) * 0.8)
            window_size = wall.thickness, wall_height * 0.5, window_length
            room_offset = np.array([room.position.x, 0, room.position.z])
            window_position = local_to_global_coords(window_position_room_coords, room_offset)
            window = Window(window_position, np.array(window_size))
            windows.append(window)

        initial_room_id = 0
        building.stories[0].rooms[initial_room_id] = room.with_heightmap(new_heightmap)
        building.stories[0] = attr.evolve(building.stories[0], windows=building.stories[0].windows + windows)

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

    def _add_obstacles_cross_chasm(
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
        room = room.with_heightmap(new_heightmap)
        building.stories[0].rooms[0] = room

        spawn_height = room.floor_heightmap[spawn_tile[1], spawn_tile[0]] + AGENT_HEIGHT / 2
        spawn_location_in_room = spawn_tile[0] + tile_size / 2, spawn_height, spawn_tile[1] + tile_size / 2

        # Prevent reaching by jump+grab
        room, _ = add_food_island(room, BuildingTile(*target_tile), 1, AGENT_HEIGHT)
        building.stories[0].rooms[0] = room
        target_height = room.floor_heightmap[target_tile[1], target_tile[0]] + FOOD_HOVER_DIST
        target_location_in_room = target_tile[0] + tile_size / 2, target_height, target_tile[1] + tile_size / 2

        boulder_height = (
            room.floor_heightmap[boulder_tile[1], boulder_tile[0]] + boulder_size / 2 - DEFAULT_FLOOR_THICKNESS
        )
        boulder_location_in_room = boulder_tile[0] + tile_size / 2, boulder_height, boulder_tile[1] + tile_size / 2
        return spawn_location_in_room, target_location_in_room, boulder_location_in_room, boulder_size

    def _add_obstacles_climb_higher(
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

        food_island_position = BuildingTile(*food_island_top_left_tile)
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

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple
    ) -> IndoorTaskParams:
        build_fn, extra_build_params, boulder_mass, entrance_azimuth = obstacle_params

        spawn_story = building.stories[0]
        spawn_room = spawn_story.rooms[0]
        spawn_location_in_room, target_location_in_room, boulder_location_in_room, boulder_size = build_fn(
            building, spawn_room, **extra_build_params
        )

        room_offset = np.array([spawn_room.position.x, 0, spawn_room.position.z])
        spawn_location_in_building = local_to_global_coords(np.array(spawn_location_in_room), room_offset)
        target_location_in_building = local_to_global_coords(np.array(target_location_in_room), room_offset)
        boulder_location_in_building = local_to_global_coords(np.array(boulder_location_in_room), room_offset)

        entities = [Boulder(position=boulder_location_in_building, mass=boulder_mass, size=boulder_size)]
        entrance_sites = ((spawn_story.num, spawn_room.id, tuple([Azimuth.EAST])),)
        return building, entities, spawn_location_in_building, target_location_in_building, entrance_sites
