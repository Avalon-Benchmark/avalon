import math
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple

import attr
import numpy as np
from scipy import stats

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.constants import JUMPING_REQUIRED_HEIGHT
from avalon.datagen.world_creation.constants import MAX_EFFECTIVE_JUMP_DIST
from avalon.datagen.world_creation.constants import MAX_FALL_DISTANCE_TO_DIE
from avalon.datagen.world_creation.constants import MAX_FLAT_JUMP_DIST
from avalon.datagen.world_creation.constants import MAX_JUMP_HEIGHT_METERS
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.geometry import BuildingTile
from avalon.datagen.world_creation.geometry import local_to_global_coords
from avalon.datagen.world_creation.indoor.builders import DefaultHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import WindowBuilder
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.components import FloorChasm
from avalon.datagen.world_creation.indoor.constants import CEILING_THICKNESS
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.constants import MIN_BUILDING_SIZE
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.indoor.task_generator import rectangle_dimensions_within_radius
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations

# smaller than this and there might not end up being much of a gap at all
# has to do with the resolution with which terrain is created
_MIN_DIST_FOR_JUMP = 1.0

# minimum height that requires the jump button to be pressed at all (less than this and you can just walk over it)
_MIN_HEIGHT_REQUIRING_JUMP = 0.7


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class JumpIndoorTaskConfig(IndoorTaskConfig):
    # Smallest and largest possible site radius for a building with this task to be placed on.
    min_site_radius: float = 6.5
    max_site_radius: float = 35.0
    # Standard deviation for the distribution deciding the site radius: higher means more variability at same difficulty
    site_radius_std_dev: float = 3.0
    # Total number of stories for the building. Note that this task is a single-story task, all other stories will
    # be purely aesthetic (e.g. to help with being viewed in the distance outdoors).
    story_count: int = 2


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class JumpOutdoorTaskConfig(TaskConfig):
    # the easier versions of the jump task allow for the agent to simply fall in the pit and jump out
    # these parameters control the min and max heights for that mode
    jumpable_mode_height_min: float = _MIN_HEIGHT_REQUIRING_JUMP
    jumpable_mode_height_max: float = JUMPING_REQUIRED_HEIGHT
    # the harder mode of the jump task requires that the agent pick the correct place to jump over a very specially
    # crafted gap that is exactly as wide as can be jumped over. These parameters control that jump distance
    jump_dist_easy: float = _MIN_DIST_FOR_JUMP
    jump_dist_hard: float = MAX_EFFECTIVE_JUMP_DIST
    # how wide the area is where the gap is the correct distance. Beyond this width, it can be much farther across
    jump_region_width_easy: float = 10.0
    jump_region_width_hard: float = 3.0
    jump_region_width_std_dev: float = 1.0
    # how much to flatten around the jump area. This is a bit unfortunate, as it makes it easy to tell where to jump.
    # howver, without this, jumping is too easy when the terrain is uneven, because you can just find a place where
    # one side of the chasm happens to be higher than the other
    jump_region_flatten_width_easy: float = 8.0
    jump_region_flatten_width_hard: float = 4.0
    jump_region_flatten_width_std_dev: float = 1.0
    # how far away the goal is
    goal_dist_min: float = 7.0
    goal_dist_max: float = 12.0
    # how likely climbing is to be allowed at difficulty 1.0  (at 0.0 it's always allowed)
    ending_climb_probability: float = 0.1


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class JumpTaskConfig(TaskConfig):
    # likelihood that the indoor variant of the task will be used
    indoor_probability: float = 0.2
    # the configs for each variant of the task
    indoor_config: JumpIndoorTaskConfig = JumpIndoorTaskConfig()
    outdoor_config: JumpOutdoorTaskConfig = JumpOutdoorTaskConfig()


def generate_jump_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: JumpTaskConfig = JumpTaskConfig(),
) -> None:
    is_indoor = rand.uniform() < task_config.indoor_probability
    if is_indoor:
        building, entities, spawn_location, target_location = create_building_for_skill_scenario(
            rand,
            difficulty,
            JumpTaskGenerator(task_config.indoor_config),
            position=CANONICAL_BUILDING_LOCATION,
            is_indoor_only=True,
        )
        world = make_indoor_task_world(
            building, entities, difficulty, spawn_location, target_location, rand, export_config
        )
    else:
        world, locations, difficulty = create_jump_obstacle(
            rand, difficulty, export_config, task_config=task_config.outdoor_config
        )
        world, locations = world.end_height_obstacles(
            locations, is_accessible_from_water=False, is_spawn_region_climbable=False
        )
        world = add_food_tree_for_simple_task(world, locations)
        world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)

    export_world(output_path, rand, world)


def create_jump_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    task_config: JumpOutdoorTaskConfig = JumpOutdoorTaskConfig(),
) -> Tuple[World, WorldLocations, float]:

    is_depth_jumpable, difficulty = select_boolean_difficulty(difficulty, rand)
    is_climbing_possible, difficulty = select_boolean_difficulty(
        difficulty, rand, initial_prob=1, final_prob=task_config.ending_climb_probability
    )
    desired_goal_dist = difficulty_variation(task_config.goal_dist_min, task_config.goal_dist_max, rand, difficulty)

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, desired_goal_dist / 10), rand, difficulty, export_config, constraint
    )

    # make it not impossibly hard to jump
    desired_jump_distance = scale_with_difficulty(difficulty, task_config.jump_dist_easy, task_config.jump_dist_hard)
    jump_distance, _warnings = world.get_critical_distance(locations, _MIN_DIST_FOR_JUMP, desired_jump_distance)
    jumpable_width = normal_distrib_range(
        task_config.jump_region_width_easy,
        task_config.jump_region_width_hard,
        task_config.jump_region_width_std_dev,
        rand,
        difficulty,
    )

    if jump_distance is None:
        raise WorldTooSmall(AvalonTask.JUMP, _MIN_DIST_FOR_JUMP, locations.get_2d_spawn_goal_distance())

    if is_depth_jumpable:
        depth = difficulty_variation(
            task_config.jumpable_mode_height_min, task_config.jumpable_mode_height_max, rand, difficulty
        )
    else:
        # we need deeper gaps for the compositional tasks--because they can be on more difficult terrain,
        # the gap needs to be quite deep to prevent you from glitching your way across
        depth = scale_with_difficulty(difficulty, JUMPING_REQUIRED_HEIGHT, MAX_FALL_DISTANCE_TO_DIE)

    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        jump_distance,
        constraint=constraint,
        height=-depth,
        traversal_width=jumpable_width,
        is_inside_climbable=True,
        is_outside_climbable=is_climbing_possible,
        dual_solution=HeightSolution(
            paths=(
                HeightPath(
                    is_solution_flattened=True,
                    is_height_affected=False,
                    width=normal_distrib_range(
                        task_config.jump_region_flatten_width_easy,
                        task_config.jump_region_flatten_width_hard,
                        task_config.jump_region_flatten_width_std_dev,
                        rand,
                        difficulty,
                    ),
                ),
            ),
            solution_point_brink_distance=1.0,
        ),
        extra_safety_radius=0.5,
    )
    world = world.add_height_obstacle(rand, ring_config, locations.island)

    return world, locations, difficulty


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class JumpTaskGenerator(BuildingTaskGenerator):
    config: JumpIndoorTaskConfig = JumpIndoorTaskConfig()

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
        width = max(width // 2, MIN_BUILDING_SIZE)
        room_height = ((MAX_JUMP_HEIGHT_METERS + AGENT_HEIGHT) * 1.1) + CEILING_THICKNESS + DEFAULT_FLOOR_THICKNESS

        return BuildingConfig(
            width=width,
            length=length,
            story_count=self.config.story_count,
            footprint_builder=RectangleFootprintBuilder(),
            room_builder=HouseLikeRoomBuilder(max_rooms=1, room_height=room_height),
            hallway_builder=DefaultHallwayBuilder(proportion_additional_edges=1),
            story_linker=None,
            window_builder=WindowBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(self, rand: np.random.Generator, difficulty: float, building: Building) -> Tuple:
        spawn_story = building.stories[0]
        spawn_room = spawn_story.rooms[0]
        # Note: crossing a 3-wide chasm is possible, but very difficult, so not included below
        max_chasm_width = math.floor(MAX_FLAT_JUMP_DIST)

        # We need one tile for launch and two for landing: they must be separate to enable climbing only on one side,
        # and landing needs two, so you can't stretch arm out through to the next climbable surface and escape the chasm
        launch_platform_width = 1
        landing_platform_width = 2
        platform_width = launch_platform_width + landing_platform_width
        max_obstacle_width = max_chasm_width + platform_width
        max_obstacle_count = (spawn_room.length - 2) // (max_obstacle_width)
        if max_obstacle_count == 0:
            raise ImpossibleWorldError("Building too small for jump task")

        free_site_indices = [i * max_obstacle_width for i in range(max_obstacle_count)]
        obstacle_count = 1 + round(normal_distrib_range(0, max_obstacle_count - 1, 0.1, rand, difficulty))
        site_indices = rand.choice(free_site_indices, size=obstacle_count)
        chasms = []
        for site_index in site_indices:
            # Note on indices: site index is the start of the "obstacle", which includes both launching and landing
            # platforms and the chasm. The launch platform is index 0, chasm is 1-2 and landing platform is 3
            # These numbers will differ if we change the max jumping distance / chasm width
            chasm_thickness = round(1 + normal_distrib_range(0, max_chasm_width - 1, 0.1, rand, difficulty))

            # Chasm must be deep enough to avoid winning by jumping + grabbing
            chasm_depth = (MAX_JUMP_HEIGHT_METERS * 2) - DEFAULT_FLOOR_THICKNESS
            depth_map = np.full_like(spawn_room.floor_heightmap, np.nan, dtype=np.float32)
            chasm_start_index = site_index + launch_platform_width
            chasm_end_index = chasm_start_index + chasm_thickness
            depth_map[chasm_start_index:chasm_end_index, :] = chasm_depth
            climbable_mask = np.zeros_like(spawn_room.floor_heightmap, dtype=np.bool_)
            climbable_mask[site_index, :] = True
            obstacle_mask = np.zeros_like(spawn_room.floor_heightmap, dtype=np.bool_)
            obstacle_mask[site_index : chasm_end_index + landing_platform_width, :] = True
            chasms.append(FloorChasm(spawn_story.num, spawn_room.id, obstacle_mask, climbable_mask, depth_map))

        spawn_tile = rand.choice([BuildingTile(i, 0) for i in range(spawn_room.width)])  # type: ignore[arg-type]
        target_tile = rand.choice([BuildingTile(i, spawn_room.length - 2) for i in range(1, spawn_room.width - 1)])  # type: ignore[arg-type]
        entrance_azimuth = Azimuth.NORTH
        return spawn_tile, entrance_azimuth, target_tile, chasms

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: Tuple
    ) -> IndoorTaskParams:
        spawn_tile, entrance_azimuth, target_tile, chasms = obstacle_params
        for chasm in chasms:
            chasm.apply(building.stories)

        first_chasm: FloorChasm = chasms[0]
        spawn_story = building.stories[first_chasm.story_id]
        spawn_room = spawn_story.rooms[first_chasm.room_id]
        spawn_location_in_room = self._position_from_tile(spawn_room, spawn_tile, AGENT_HEIGHT / 2)
        # Do not use FOOD_HOVER_DIST below as the food may fall into a nearby chasm if it is spawned close to a wall
        target_location_in_room = self._position_from_tile(spawn_room, target_tile, 0)

        room_offset = np.array([spawn_room.position.x, 0, spawn_room.position.z])
        spawn_location_in_building = local_to_global_coords(np.array(spawn_location_in_room), room_offset)
        target_location_in_building = local_to_global_coords(np.array(target_location_in_room), room_offset)

        extra_items: List[Entity] = []
        entrance_sites = ((spawn_story.num, spawn_room.id, tuple([entrance_azimuth])),)
        return building, extra_items, spawn_location_in_building, target_location_in_building, entrance_sites
