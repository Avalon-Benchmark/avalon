from pathlib import Path
from typing import Tuple

import attr
import numpy as np
from networkx.algorithms.approximation import traveling_salesman_problem

from avalon.common.utils import only
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.building import BuildingConfig
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.constants import AGENT_HEIGHT
from avalon.datagen.world_creation.entities.doors.hinge_door import HingeDoor
from avalon.datagen.world_creation.entities.doors.sliding_door import SlidingDoor
from avalon.datagen.world_creation.geometry import Axis
from avalon.datagen.world_creation.indoor.builders import HamiltonianHallwayBuilder
from avalon.datagen.world_creation.indoor.builders import HouseLikeRoomBuilder
from avalon.datagen.world_creation.indoor.builders import RectangleFootprintBuilder
from avalon.datagen.world_creation.indoor.builders import WindowBuilder
from avalon.datagen.world_creation.indoor.builders import get_room_overlap
from avalon.datagen.world_creation.indoor.building import Building
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingNavGraph
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.components import Room
from avalon.datagen.world_creation.indoor.components import Story
from avalon.datagen.world_creation.indoor.constants import Azimuth
from avalon.datagen.world_creation.indoor.doors import DoorParams
from avalon.datagen.world_creation.indoor.doors import get_door_params_from_difficulty
from avalon.datagen.world_creation.indoor.doors import make_hinge_door
from avalon.datagen.world_creation.indoor.doors import make_sliding_door
from avalon.datagen.world_creation.indoor.task_generator import CANONICAL_BUILDING_LOCATION
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import IndoorTaskParams
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.indoor.task_generator import get_room_centroid_in_building_space
from avalon.datagen.world_creation.indoor.task_generator import make_indoor_task_world
from avalon.datagen.world_creation.indoor.task_generator import rectangle_dimensions_within_radius
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class OpenTaskConfig(IndoorTaskConfig):
    # Smallest and largest possible site radius for a building with this task to be placed on.
    min_site_radius: float = 7.0
    max_site_radius: float = 15.0
    # Total number of stories for the building. Note that this task is a single-story task, all other stories will
    # be purely aesthetic (e.g. to help with being viewed in the distance outdoors).
    story_count: int = 2
    # Max number of rooms that will be created at difficulty=1. We'll have as many doors as rooms minus 1.
    # Note that going higher than 5 will increase number of impossible worlds, as it becomes harder to find a
    # Hamiltonian path in the building.
    max_room_count: int = 5


def generate_open_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: OpenTaskConfig = OpenTaskConfig(),
) -> None:
    building, entities, spawn_location, target_location = create_building_for_skill_scenario(
        rand,
        difficulty,
        OpenTaskGenerator(task_config),
        position=CANONICAL_BUILDING_LOCATION,
        is_indoor_only=True,
    )
    world = make_indoor_task_world(
        building, entities, difficulty, spawn_location, target_location, rand, export_config
    )
    export_world(output_path, rand, world)


StoryNum = int
FromRoom = int
ToRoom = int
PlacedDoorParams = Tuple[StoryNum, FromRoom, ToRoom, DoorParams]

FromStoryNum = int
FromRoomId = int
ToStoryNum = int
ToRoomId = int
OpenTaskObstacleParams = Tuple[FromStoryNum, FromRoomId, ToStoryNum, ToRoomId, Tuple[PlacedDoorParams, ...]]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class OpenTaskGenerator(BuildingTaskGenerator):
    config: OpenTaskConfig = OpenTaskConfig()

    def get_site_radius(self, rand: np.random.Generator, difficulty: float) -> float:
        return scale_with_difficulty(difficulty, self.config.min_site_radius, self.config.max_site_radius)

    def get_building_config(
        self,
        rand: np.random.Generator,
        difficulty: float,
        radius: float,
        allowed_auxiliary_tasks: Tuple[BuildingTask, ...] = tuple(),
        aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig(),
    ) -> BuildingConfig:
        width, length = rectangle_dimensions_within_radius(radius)
        if width < 5 and length < 5:
            raise ImpossibleWorldError("Building too small to fit open task")

        min_room_count = 2
        max_room_count = min_room_count + round(
            normal_distrib_range(0, self.config.max_room_count - min_room_count, 0.1, rand, difficulty)
        )

        return BuildingConfig(
            width=width,
            length=length,
            story_count=self.config.story_count,
            footprint_builder=RectangleFootprintBuilder(),
            # Min room size must be >3 to ensure we have space for locks on both sides
            room_builder=HouseLikeRoomBuilder(min_room_size=3, max_rooms=max_room_count),
            hallway_builder=HamiltonianHallwayBuilder(),
            # All the action takes place on the first story; any extra stories are purely aesthetic and not connected
            story_linker=None,
            window_builder=WindowBuilder(),
            aesthetics=aesthetics,
        )

    def get_principal_obstacle_params(
        self, rand: np.random.Generator, difficulty: float, building: Building
    ) -> OpenTaskObstacleParams:
        story_num = 0
        if len(building.stories[story_num].rooms) < 2:
            raise ImpossibleWorldError("Can't make open task in a single room")
        decorative_story_nums = tuple(s.num for s in building.stories[story_num + 1 :])
        nav_graph = BuildingNavGraph(building, excluded_stories=decorative_story_nums)
        path = traveling_salesman_problem(nav_graph, cycle=False)
        all_door_params = []
        stories_and_rooms_by_node_id = nav_graph.get_stories_and_rooms_by_node_id(building.stories)
        for from_node, to_node in zip(path[:-1], path[1:]):
            from_story, from_room = stories_and_rooms_by_node_id[from_node]
            to_story, to_room = stories_and_rooms_by_node_id[to_node]
            self._check_connection_is_valid(from_story, from_room, to_story, to_room)
            all_door_params.append(
                (story_num, from_room.id, to_room.id, get_door_params_from_difficulty(rand, difficulty))
            )
        initial_story, initial_room = stories_and_rooms_by_node_id[path[0]]
        target_story, target_room = stories_and_rooms_by_node_id[path[-1]]
        return initial_story.num, initial_room.id, target_story.num, target_room.id, tuple(all_door_params)

    @staticmethod
    def _check_connection_is_valid(from_story: Story, from_room: Room, to_story: Story, to_room: Room) -> None:
        # Sometimes hallways can only be fit in the corner of rooms - this does not work for the Open task since locks
        # must be placed on the sides and there might not be enough room.
        if from_story == to_story:
            x_overlap, z_overlap = get_room_overlap(from_room, to_room)
            wall_overlap = x_overlap or z_overlap
            is_hallway_going_to_be_forced_into_corner = wall_overlap is not None and wall_overlap.size < 3
            if is_hallway_going_to_be_forced_into_corner:
                raise ImpossibleWorldError(
                    f"Wall overlap between {from_room.id} and {to_room.id} is too small to fit a door with locks properly."
                )

    def add_principal_obstacles(
        self, rand: np.random.Generator, building: Building, obstacle_params: OpenTaskObstacleParams
    ) -> IndoorTaskParams:
        initial_story_num, initial_room_id, target_story_num, target_room_id, placed_door_param_set = obstacle_params
        from_story = building.stories[initial_story_num]
        from_room = from_story.rooms[initial_room_id]
        spawn_location = get_room_centroid_in_building_space(
            building, from_story, from_room, at_height=AGENT_HEIGHT / 2
        )

        to_story = building.stories[target_story_num]
        to_room = to_story.rooms[target_room_id]
        target_location = get_room_centroid_in_building_space(building, to_story, to_room, at_height=AGENT_HEIGHT / 2)

        doors = []
        for story_num, from_room_id, to_room_id, door_params in placed_door_param_set:
            story = building.stories[story_num]
            from_room = story.rooms[from_room_id]
            to_room = story.rooms[to_room_id]
            hallway = only(
                [
                    hallway
                    for hallway in story.hallways
                    if {from_room.id, to_room.id} == {hallway.from_room_id, hallway.to_room_id}
                ]
            )

            if hallway.from_room_id == from_room.id:
                door_tile = hallway.points[0]
                door_azimuth = hallway.from_room_azimuth
            else:
                door_tile = hallway.points[-1]
                door_azimuth = hallway.to_room_azimuth

            if door_azimuth in {Azimuth.NORTH, Azimuth.SOUTH}:
                door_face_axis = Axis.X
            else:
                door_face_axis = Axis.Z

            door_type, door_mechanics_params, latching_mechanics, variant_locks = door_params
            if door_type is HingeDoor:
                make_door = make_hinge_door
            elif door_type is SlidingDoor:
                make_door = make_sliding_door  # type: ignore
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
                final_params = only(params) if params else {}
                locks.append(make_lock(door, **final_params))  # type: ignore
            with door.mutable_clone() as locked_door:
                locked_door.locks = locks
                door = locked_door
            doors.append(door)

        entrance_sites = tuple([(initial_story_num, initial_room_id, tuple(Azimuth))])
        return building, doors, spawn_location, target_location, entrance_sites
