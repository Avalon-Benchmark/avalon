import math
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import cast

import attr
import networkx as nx
import numpy as np
from loguru import logger
from nptyping import assert_isinstance
from scipy import stats
from skimage import morphology

from avalon.common.errors import SwitchError
from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.building import generate_aesthetics_config
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import IndoorTaskConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import ITEM_FLATTEN_RADIUS
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.constants import WATER_LINE
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.debug_plots import plot_terrain
from avalon.datagen.world_creation.debug_plots import plot_value_grid
from avalon.datagen.world_creation.entities.animals import ALL_PREDATOR_CLASSES
from avalon.datagen.world_creation.entities.animals import Predator
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_VISIBLE_HEIGHT
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.entities.food import CANONICAL_FOOD_CLASS
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.entities.tools.weapons import LargeRock
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.entities.tools.weapons import Weapon
from avalon.datagen.world_creation.indoor.building import BuildingAestheticsConfig
from avalon.datagen.world_creation.indoor.building import BuildingTask
from avalon.datagen.world_creation.indoor.building import get_building_footprint_outline
from avalon.datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from avalon.datagen.world_creation.indoor.task_generator import MIN_NAVIGATE_BUILDING_SIZE
from avalon.datagen.world_creation.indoor.task_generator import BuildingTaskGenerator
from avalon.datagen.world_creation.indoor.task_generator import create_building_for_skill_scenario
from avalon.datagen.world_creation.tasks.bridge import BridgeTaskConfig
from avalon.datagen.world_creation.tasks.bridge import create_bridge_obstacle
from avalon.datagen.world_creation.tasks.climb import ClimbIndoorTaskConfig
from avalon.datagen.world_creation.tasks.climb import ClimbOutdoorTaskConfig
from avalon.datagen.world_creation.tasks.climb import ClimbTaskGenerator
from avalon.datagen.world_creation.tasks.climb import create_climb_obstacle
from avalon.datagen.world_creation.tasks.descend import DescendTaskConfig
from avalon.datagen.world_creation.tasks.descend import create_descend_obstacle
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.tasks.explore import ExploreIndoorTaskConfig
from avalon.datagen.world_creation.tasks.explore import ExploreTaskGenerator
from avalon.datagen.world_creation.tasks.fight import FightTaskConfig
from avalon.datagen.world_creation.tasks.fight import create_fight_obstacle
from avalon.datagen.world_creation.tasks.jump import JumpIndoorTaskConfig
from avalon.datagen.world_creation.tasks.jump import JumpOutdoorTaskConfig
from avalon.datagen.world_creation.tasks.jump import JumpTaskGenerator
from avalon.datagen.world_creation.tasks.jump import create_jump_obstacle
from avalon.datagen.world_creation.tasks.move import MoveTaskConfig
from avalon.datagen.world_creation.tasks.move import create_move_obstacle
from avalon.datagen.world_creation.tasks.open import OpenTaskConfig
from avalon.datagen.world_creation.tasks.open import OpenTaskGenerator
from avalon.datagen.world_creation.tasks.push import PushIndoorTaskConfig
from avalon.datagen.world_creation.tasks.push import PushOutdoorTaskConfig
from avalon.datagen.world_creation.tasks.push import PushTaskGenerator
from avalon.datagen.world_creation.tasks.push import create_push_obstacle
from avalon.datagen.world_creation.tasks.stack import StackIndoorTaskConfig
from avalon.datagen.world_creation.tasks.stack import StackOutdoorTaskConfig
from avalon.datagen.world_creation.tasks.stack import StackTaskGenerator
from avalon.datagen.world_creation.tasks.stack import create_stack_obstacle
from avalon.datagen.world_creation.types import MapBoolNP
from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.types import Point3DListNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.types import WorldType
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_for_skill_scenario
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.height_map import HeightMap
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations

# just need to be tuned for performance. Basically, more retries makes it more likely to generate a world with enough
# obstacles in worlds that are tricky but barely possible, while too many retries means you waste time continuing to
# try to place obstacles in worlds where it's basically impossible to do so and you just need to give up and start over
_OUTDOOR_OBSTACLE_MAX_RETRIES = 4
_INDOOR_OBSTACLE_MAX_RETRIES = 5


OUTDOOR_OBSTACLE_CONFIG_DICT: Dict[AvalonTask, Tuple[float, TaskConfig]] = {
    AvalonTask.MOVE: (1.0, MoveTaskConfig()),
    AvalonTask.JUMP: (1.0, JumpOutdoorTaskConfig()),
    AvalonTask.CLIMB: (1.0, ClimbOutdoorTaskConfig()),
    AvalonTask.PUSH: (1.0, PushOutdoorTaskConfig()),
    AvalonTask.FIGHT: (1.0, FightTaskConfig()),
    AvalonTask.STACK: (1.0, StackOutdoorTaskConfig()),
    AvalonTask.BRIDGE: (1.0, BridgeTaskConfig()),
    AvalonTask.DESCEND: (1.0, DescendTaskConfig()),
}

INDOOR_OBSTACLE_CONFIG_DICT: Dict[BuildingTask, Tuple[float, IndoorTaskConfig]] = {
    BuildingTask.EXPLORE: (1.0, ExploreIndoorTaskConfig()),
    BuildingTask.OPEN: (1.0, OpenTaskConfig()),
    BuildingTask.PUSH: (1.0, PushIndoorTaskConfig()),
    BuildingTask.STACK: (1.0, StackIndoorTaskConfig()),
    BuildingTask.CLIMB: (1.0, ClimbIndoorTaskConfig()),
    BuildingTask.JUMP: (1.0, JumpIndoorTaskConfig()),
}


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class CompositeTaskConfig(TaskConfig):
    # set this to the particular task being done
    task: AvalonTask
    # the largest number of obstacles that will ever be put in a single world. much more than 4 is pretty annoying to
    # play, and starts to require relatively large levels (if they're all height obstacles).
    # Higher difficulties will have more obstacles (up to this number)
    max_obstacles: int = 4
    # how frequently to include more or fewer obstacles (true obstacle count is sampled from:
    # NormalDistribution(max_obstacles, max_obstacles_std_dev)
    max_obstacles_std_dev: float = 1.0
    # how much world size is required for each obstacle. Rough number based on the max required by any individual
    # obstacle. Is required to determine how large the world should be
    goal_distance_per_obstacle: float = 21.0
    # how much to vary the size of the world. Worlds must be large enough so that even a two sigma small world
    # will still be big enough to fit all obstacles
    world_size_std_dev: float = 20.0
    # how high to boost the spawn to ensure that the goal is visible. If boosting up the spawn this much doesn't
    # result in the goal being visible, but it does need to be visible, an ImpossibleWorldError will be raised
    max_visibility_height_boost: float = 8.0
    # the probability that the food ends up in a building (which can have its own obstacles)
    is_goal_in_building_probability: float = 0.5
    # Compositional height obstacles present a problem: if the world is a normal island, you can just walk around
    # many of the obstacles. There are two different ways of dealing with this issue:
    # 1. raise the islands out of the water such that all land is surrounded by unclimbable cliffs
    # 2. only use obstalces that do not have this issue
    # This parameter controls the probability of the first mode. It is made more common because that way we get to use
    # most of the obstacles. The second mode has a very particular look because it can really only use a small number
    # of the obstacles (climb, descend, etc--things that change height)
    is_raised_island_mode_probability: float = 0.8
    # how many predators to include. At difficulty 1.0, will be a value randomly chosen between the two extremes
    predator_count_easy: int = 0
    predator_count_hard: int = 16
    # don't start predators within this radius of where you start
    min_predator_distance_from_spawn: float = 10.0
    #
    predator_path_dist_easy: float = 40.0
    predator_path_dist_hard: float = 15.0
    #
    weapon_ratio_easy: float = 3.0
    weapon_ratio_hard: float = 0.0
    weapon_path_dist_easy: float = 5.0
    weapon_path_dist_hard: float = 20.0
    large_rock_probability: float = 0.4
    rock_probability_easy: float = 0.5
    rock_probability_hard: float = 0.95
    # you can configure any of the inner tasks, as well as the probability of each of them occuring (will be normalized)
    outdoor_obstacle_configs: Dict[AvalonTask, Tuple[float, TaskConfig]] = OUTDOOR_OBSTACLE_CONFIG_DICT
    # you can also configure any of the indoor tasks that are used as well
    indoor_obstacle_configs: Dict[BuildingTask, Tuple[float, IndoorTaskConfig]] = INDOOR_OBSTACLE_CONFIG_DICT


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ForcedComposition:
    is_enabled: bool = False


# These are all the obstacles that create rings around a point
_ALLOWED_OBSTACLES = [
    AvalonTask.MOVE,
    AvalonTask.JUMP,
    AvalonTask.CLIMB,
    AvalonTask.PUSH,
    AvalonTask.FIGHT,
    AvalonTask.STACK,
    AvalonTask.BRIDGE,
    AvalonTask.DESCEND,
]


class FinalObstacleType(Enum):
    PREDATOR = "PREDATOR"
    CARRY = "CARRY"


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CompositionalSolutionPath:
    mask: MapBoolNP
    path_locations: np.ndarray
    points: Point3DListNP
    point_indices: np.ndarray
    start_2d_radii: np.ndarray
    end_2d_radii: np.ndarray

    def get_point_at_radius_from_goal(self, radius: float) -> Point3DNP:
        far_enough_away = self.end_2d_radii > radius
        if not np.any(far_enough_away):
            raise ImpossibleWorldError("The safety radius for the end point included the start point!")
        idx = int(np.argwhere(far_enough_away)[-1])
        return cast(Point3DNP, self.points[idx])

    def get_point_at_radius_from_spawn(self, radius: float) -> Point3DNP:
        far_enough_away = self.start_2d_radii > radius
        if not np.any(far_enough_away):
            raise ImpossibleWorldError("The safety radius for the start point included the end point!")
        idx = int(np.argwhere(far_enough_away)[0])
        return cast(Point3DNP, self.points[idx])


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CompositionalSolutionData:
    locations: WorldLocations
    full_path: CompositionalSolutionPath
    current_spawn_side_radius: float = 0.0
    current_goal_side_radius: float = 0.0

    @staticmethod
    def build(
        locations: WorldLocations, full_path: CompositionalSolutionPath, goal_safety_radius: float
    ) -> "CompositionalSolutionData":
        return CompositionalSolutionData(
            locations=locations,
            full_path=full_path,
            current_goal_side_radius=goal_safety_radius,
        )

    @property
    def spawn_endpoint(self) -> Point3DNP:
        return self.full_path.get_point_at_radius_from_spawn(self.current_spawn_side_radius)

    @property
    def goal_endpoint(self) -> Point3DNP:
        return self.full_path.get_point_at_radius_from_goal(self.current_goal_side_radius)

    def get_distance_to_use(self, remaining_height_obstacles: int) -> float:
        how_much_distance_to_use = 1 / remaining_height_obstacles
        distance_left = (
            self.locations.get_2d_spawn_goal_distance()
            - (self.current_goal_side_radius + self.current_spawn_side_radius) * 1.1
        )
        return how_much_distance_to_use * distance_left

    def update_spawn(self, radius: float) -> "CompositionalSolutionData":
        return attr.evolve(self, current_spawn_side_radius=radius)

    def update_goal(self, radius: float) -> "CompositionalSolutionData":
        return attr.evolve(self, current_goal_side_radius=radius)

    def plot(self, map: HeightMap) -> None:
        map.plot()
        heights = map.Z.copy()
        max_height = np.max(map.Z)
        heights[map.Z < WATER_LINE] = max_height
        interesting_points = [self.locations.spawn, self.locations.goal, self.spawn_endpoint, self.goal_endpoint]
        markers = [map.point_to_index(np.array([x[0], x[2]])) for x in interesting_points]
        for location, p in zip(list(self.full_path.path_locations), list(self.full_path.point_indices)):
            # if location < self.goal_path_location and location > self.spawn_path_location:
            #     heights[tuple(p)] = max_height
            heights[tuple(p)] = max_height
        plot_value_grid(heights, "Current solution", markers=markers)


def is_building_too_close_to_map_edge(building_center: Point2DNP, radius: float, map: HeightMap) -> bool:
    for delta in [
        (-radius, 0.0),
        (radius, 0.0),
        (0.0, -radius),
        (0.0, radius),
    ]:
        if not map.region.contains_point_2d(building_center + np.array(delta)):
            return True
    return False


def create_compositional_task(
    rand: np.random.Generator,
    difficulty: float,
    task_config: CompositeTaskConfig,
    export_config: ExportConfig,
    desired_goal_distance: Optional[stats.norm] = None,
    is_food_allowed: bool = True,
    is_debug_graph_printing_enabled: bool = False,
    min_size_in_meters: Optional[float] = None,
    max_size_in_meters: Optional[float] = None,
    _FORCED: Optional[ForcedComposition] = None,
) -> Tuple[World, WorldLocations]:
    """
    how this works, at a high level:

    first we use difficulty figure out how many obstacles we want, and how far to travel

    then we figure out how many of each type of obstacle we want
    obstacles come in three types:
        buildings
        outdoor rings
        final modifiers

    if the destination is a building, that is handled first.

    next we hand the ring obstacles, which are the most complicated to add

    first we try to add single small rings around the spawn and goal points (if desired)

    then we add the rest of the ring obstacles along the path inbetween.
    the path needs to be calculated so that we can move from the start to the end, staying on the land
    whenever a ring obstacle is added, we effectively are consuming part of that path

    after the ring obstacles are added, we add the buildings

    at the very end, we try to add enough final modifiers to bring us up to our desired total
    """

    # TODO: put back
    is_debug_graph_printing_enabled = False

    desired_obstacle_count, difficulty = _get_desired_obstacle_count(rand, difficulty, task_config, _FORCED)
    # figure out how many of each type of obstacle we want
    (building_count, desired_height_obstacle_count) = _get_obstacle_counts_per_type(
        rand, desired_obstacle_count, task_config, is_food_allowed
    )
    if task_config.task == AvalonTask.SURVIVE:
        # this is because it is handled outside of this function
        random_predator_count = 0
    else:
        random_predator_count = round(
            difficulty_variation(
                task_config.predator_count_easy,
                task_config.predator_count_hard,
                rand,
                difficulty,
            )
        )

    if desired_goal_distance is None:
        desired_goal_distance = (
            desired_obstacle_count * task_config.goal_distance_per_obstacle + 2 * task_config.world_size_std_dev
        )

    if max_size_in_meters is None:
        # 2x because we want space for the obstacles, and additional 1x to account fo the 50%
        # variation in world size from the way the fractal stuff works out
        max_size_in_meters = 3.0 * desired_goal_distance

        if building_count > 0:
            max_size_in_meters += 60.0

    if task_config.task == AvalonTask.NAVIGATE:
        is_visibility_required = True
    else:
        is_visibility_required = False

    if is_debug_graph_printing_enabled:
        logger.debug(
            f"Obstacles: {building_count} buildings, {desired_height_obstacle_count} height, {random_predator_count} predators"
        )
        logger.debug(f"Visibility: {is_visibility_required}")

    is_destination_building = False
    if task_config.task in (AvalonTask.GATHER, AvalonTask.SURVIVE):
        is_destination_building = building_count > 0
    elif building_count > 0:
        is_destination_building = True
        assert (
            building_count <= 1
        ), "Can only create a single food, and buildings must have food, thus cannot create extra buildings"

    min_visible_height_for_world_creation = FOOD_TREE_VISIBLE_HEIGHT
    if is_destination_building:
        # this is roughly the height of the smallest building
        min_visible_height_for_world_creation = 4.0

    goal_distance = stats.norm(desired_goal_distance, task_config.world_size_std_dev)
    world, starting_locations = create_world_for_skill_scenario(
        rand,
        difficulty,
        CANONICAL_FOOD_HEIGHT_ON_TREE,
        goal_distance,
        export_config,
        is_visibility_required=is_visibility_required,
        min_size_in_meters=min_size_in_meters,
        max_size_in_meters=max_size_in_meters,
        visibility_height=min_visible_height_for_world_creation,
        world_type=WorldType.CONTINENT,
    )

    if is_debug_graph_printing_enabled:
        logger.debug(f"WORLD SIZE: {world.config.size_in_meters ** 2}")
        logger.debug(
            f"Goal and spawn are {starting_locations.get_2d_spawn_goal_distance()} away, we need at least {desired_obstacle_count * task_config.goal_distance_per_obstacle}"
        )

    # we're going to keep adding obstacles until we hit our count
    obstacle_count = 0

    # generate a single AestheticsConfig that is shared among the buildings in the world
    aesthetics = generate_aesthetics_config()

    goal_safety_radius = 0.0
    is_food_required = True
    # you would expect this, but actually, we just care about being able to see to the top of the tree
    # goal_visible_height_delta = CANONICAL_FOOD_HEIGHT_ON_TREE
    goal_visible_height_delta = FOOD_TREE_VISIBLE_HEIGHT
    initial_building = None
    initial_building_items = None
    initial_building_starting_terrain_height = None
    if is_destination_building:
        building_task = _select_building_obstacle_type(rand, task_config)
        building_radius = get_radius_for_building_task(rand, building_task, difficulty, task_config)
        building_location_2d = to_2d_point(starting_locations.goal)
        if is_building_too_close_to_map_edge(building_location_2d, building_radius, world.map):
            building_obstacle = get_random_building_obstacle(rand, world, task_config, starting_locations, difficulty)
            if building_obstacle:
                building_task, building_radius, mask, new_difficulty = building_obstacle
                building_location_2d = world.map.index_to_point_2d(tuple(rand.choice(np.argwhere(mask))))  # type: ignore[arg-type]
            else:
                raise ImpossibleWorldError("Can not place initial building")
        new_goal_location, initial_building, initial_building_items, world = create_building(
            rand,
            world,
            task_config,
            building_location_2d,
            building_radius,
            building_task,
            difficulty,
            aesthetics,
            is_shore_created=True,
        )
        initial_building_starting_terrain_height = world.map.get_rough_height_at_point(building_location_2d)
        is_food_required = False
        # have to reset the island mask as well because the land might have been extended
        starting_locations = attr.evolve(
            starting_locations,
            goal=new_goal_location,
            island=world.map.get_island(to_2d_point(starting_locations.spawn))[-1],
        )
        obstacle_count += 1
        building_count -= 1
        if obstacle_count >= desired_obstacle_count:
            # just add the items now because we're done
            world = world.add_items(initial_building_items)
            world = world.add_item(CANONICAL_FOOD_CLASS(position=new_goal_location))
            full_path = _find_path(rand, world, starting_locations)
            solution = CompositionalSolutionData.build(starting_locations, full_path, goal_safety_radius)
            world = add_extra_predators(
                rand, world, task_config, solution, difficulty, random_predator_count, full_path
            )

            return _return_data(
                rand, world, difficulty, starting_locations, is_food_required, is_visibility_required, task_config
            )

        goal_safety_radius = building_radius * 1.5
        # this is how much above the goal is visible
        goal_visible_height_delta = initial_building.height.max_lt - starting_locations.goal

    # figure out how we're going to move from the start to the end
    full_path = _find_path(rand, world, starting_locations)
    solution = CompositionalSolutionData.build(starting_locations, full_path, goal_safety_radius)

    if is_debug_graph_printing_enabled:
        logger.debug(f"Initial safety radius: {goal_safety_radius}")
        logger.debug(f"Reaming distance: {solution.locations.get_2d_spawn_goal_distance()}")

    if is_debug_graph_printing_enabled:
        solution.plot(world.map)
        # plot_value_grid(
        #     world.map.Z,
        #     markers=[
        #         # world.map.point_to_index(to_2d_point(solution.full_path.get_point_at_radius_from_goal(x)))
        #         world.map.point_to_index(to_2d_point(solution.full_path.get_point_at_radius_from_spawn(x)))
        #         for x in range(50)
        #     ],
        # )

    if is_debug_graph_printing_enabled:
        if is_visibility_required:
            logger.debug("Visibility required!!!")
        logger.debug("Selected obstacle types:")
        logger.debug(f"{building_count + int(is_destination_building)} buildings")
        logger.debug(f"{desired_height_obstacle_count} height obstacles")
        logger.debug(f"{random_predator_count} predators")

    # raise the island up
    if desired_height_obstacle_count > 0:
        new_locations, world = world.begin_height_obstacles(solution.locations)
        solution = attr.evolve(solution, locations=new_locations)

        # see docs for is_raised_island_mode_probability
        is_raised_island_mode = rand.uniform() < task_config.is_raised_island_mode_probability

        if is_debug_graph_printing_enabled:
            logger.debug(f"{is_raised_island_mode=}")
        if is_raised_island_mode:
            # add as many ring obstacles as we can (up to the max we want)

            # keep track of which later obstacles are allowed for navigate, since we need to see the goal when swe start
            is_boosted = False
            if is_visibility_required:
                allowed_obstacles = tuple(
                    [
                        x
                        for x in _ALLOWED_OBSTACLES
                        if x
                        not in (
                            AvalonTask.DESCEND,
                            AvalonTask.CLIMB,
                            AvalonTask.FIGHT,
                        )
                    ]
                )
                is_boosted = rand.uniform() > 0.5
            else:
                allowed_obstacles = tuple([x for x in _ALLOWED_OBSTACLES])

            distance_to_use = solution.get_distance_to_use(desired_height_obstacle_count)
            spawn_region = None

            if is_boosted:
                obstacle_type = AvalonTask.CLIMB
                result, new_difficulty, solution, new_locations, world = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
                    task_config,
                    solution,
                    distance_to_use,
                    is_around_spawn=False,
                    is_height_inverted=True,
                    is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
                )
                if result:
                    desired_height_obstacle_count -= 1
                    allowed_obstacles += tuple([AvalonTask.CLIMB, AvalonTask.FIGHT])
                    # move the food to the ledge
                    if is_food_required:
                        updated_locations = attr.evolve(solution.locations, goal=new_locations.goal)
                        solution = attr.evolve(solution, locations=updated_locations)

                    # figure out the spawn region (in case this is the only obstacle)
                    spawn_region = world.obstacle_zones[0][0]

            is_first_successful_obstacle = True
            for i in range(desired_height_obstacle_count):
                obstacle_type = _select_ring_obstacle_type(rand, allowed_obstacles, task_config)
                is_around_spawn = True
                result, new_difficulty, solution, new_locations, world = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
                    task_config,
                    solution,
                    distance_to_use,
                    is_around_spawn=is_around_spawn,
                    is_height_inverted=False,
                    is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
                )
                # logger.debug(f"goal height {solution.locations.goal[1]}")
                if result:
                    obstacle_count += 1
                    if is_first_successful_obstacle:
                        is_first_successful_obstacle = False
                        # reset the spawn region
                        spawn_region = world.obstacle_zones[-1][0]

            if spawn_region is None:
                raise ImpossibleWorldError("Failed to add any height obstacles in raised island mode")
            world, new_locations = world.end_height_obstacles(
                solution.locations, is_accessible_from_water=False, spawn_region=spawn_region
            )
            solution = attr.evolve(solution, locations=new_locations)

        else:
            # add ring based obstacle around spawn
            fall_count, climb_count = _get_fall_climb_split(rand, desired_height_obstacle_count)

            is_first_successful_obstacle = True
            spawn_region = None

            distance_to_use = solution.get_distance_to_use(desired_height_obstacle_count)
            for i in range(fall_count):
                obstacle_type = AvalonTask.DESCEND
                result, new_difficulty, solution, new_locations, world = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
                    task_config,
                    solution,
                    distance_to_use,
                    is_around_spawn=True,
                    is_height_inverted=False,
                    is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
                )
                if result:
                    obstacle_count += 1
                    if is_first_successful_obstacle:
                        is_first_successful_obstacle = False
                        # reset the spawn region
                        spawn_region = world.obstacle_zones[-1][0]

            # add ring based obstacle around goal
            for i in range(climb_count):
                obstacle_type = AvalonTask.CLIMB
                result, new_difficulty, solution, new_locations, world = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
                    task_config,
                    solution,
                    distance_to_use,
                    is_around_spawn=False,
                    is_height_inverted=True,
                    is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
                )
                if result:
                    obstacle_count += 1
                    if is_first_successful_obstacle:
                        is_first_successful_obstacle = False
                        # reset the spawn region
                        spawn_region = world.obstacle_zones[-1][0]

            if spawn_region is None:
                raise ImpossibleWorldError("Failed to add any height obstacles in flat island mode")
            world, new_locations = world.end_height_obstacles(
                solution.locations, is_accessible_from_water=True, spawn_region=spawn_region
            )
            solution = attr.evolve(solution, locations=new_locations)

        # if necessary, boost JUST the spawn safety radius (until the end is visible)
        if is_visibility_required:
            height_increments = 0
            max_increments = 20
            height_delta_per_step = task_config.max_visibility_height_boost / max_increments
            visibility_height_offset = 0.0
            visibility_calculator = world.map.generate_visibility_calculator()
            # technically should add half agent offset to the first part, but want to be sure we can see it
            while not visibility_calculator.is_visible_from(
                solution.locations.spawn + UP_VECTOR * visibility_height_offset,
                solution.locations.goal + UP_VECTOR * goal_visible_height_delta,
            ):
                height_increments += 1
                if height_increments > max_increments:
                    # interesting_points = [solution.locations.spawn, solution.locations.goal]
                    # interesting_points = [solution.locations.goal]
                    # markers = [world.map.point_to_index(np.array([x[0], x[2]])) for x in interesting_points]
                    #
                    # logger.debug("busted")
                    # logger.debug((
                    #     solution.locations.spawn + up_vector * visibility_height_offset,
                    #     solution.locations.goal + up_vector * min_food_offset,
                    # ))
                    # logger.debug("works?")
                    # vis = world.plot_visibility(
                    #     solution.locations.spawn + up_vector * visibility_height_offset,
                    #     up_vector * min_food_offset,
                    #     markers,
                    # )
                    # world.plot_visibility(
                    #     locations.spawn + up_vector * visibility_height_offset + HALF_AGENT_HEIGHT_VECTOR,
                    #     up_vector * min_food_offset,
                    #     markers,
                    # )

                    # fix if we see this again, shouldn't really happen
                    raise ImpossibleWorldError("Could not boost spawn high enough to see goal. Very odd.")
                    # break
                visibility_height_offset += height_delta_per_step

            # actually boost the spawn
            spawn_radius = 2.0
            spawn_point_region = world.map.get_dist_sq_to(to_2d_point(solution.locations.spawn)) < spawn_radius**2
            if visibility_height_offset > 0.0:
                map_new = world.map.copy()
                map_new.raise_island(spawn_point_region, visibility_height_offset)
                world = attr.evolve(world, map=map_new)

    # figure out how much the height of the initial building changed, and update its (and its items0 positions
    if initial_building is not None:
        assert initial_building_items is not None
        assert initial_building_starting_terrain_height is not None
        new_height = world.map.get_rough_height_at_point(
            np.array([initial_building.position[0], initial_building.position[2]])
        )
        height_change = new_height - initial_building_starting_terrain_height
        items = move_all_items_up(initial_building_items, height_change)
        world = world.add_items(items)
        new_goal_location = starting_locations.goal + UP_VECTOR * height_change
        world = world.add_item(CANONICAL_FOOD_CLASS(position=new_goal_location))
        assert len(world.building_by_id) == 1
        new_building_position = initial_building.position + np.array([0, height_change, 0])
        new_building = attr.evolve(initial_building, position=new_building_position)
        world = attr.evolve(world, buildings=tuple([new_building]))

    if is_debug_graph_printing_enabled:  # TODO KJ
        logger.debug(f"{obstacle_count=}")
        logger.debug(f"{desired_obstacle_count=}")

    # add any required buildings
    for i in range(building_count):
        # randomly gets a task and location mask for where it can actually fit
        building_obstacle = get_random_building_obstacle(rand, world, task_config, solution.locations, difficulty)
        if building_obstacle:
            task, radius, mask, new_difficulty = building_obstacle
            building_location_2d = world.map.index_to_point_2d(tuple(rand.choice(np.argwhere(mask))))  # type: ignore[arg-type]
            food_location, _building, items, world = create_building(
                rand, world, task_config, building_location_2d, radius, task, difficulty, aesthetics
            )
            world = world.add_items(items)
            world = world.add_item(CANONICAL_FOOD_CLASS(position=food_location))
            obstacle_count += 1

    # finally, add a bunch of predators
    world = add_extra_predators(rand, world, task_config, solution, difficulty, random_predator_count, full_path)

    return _return_data(
        rand, world, difficulty, solution.locations, is_food_required, is_visibility_required, task_config
    )


def add_extra_predators(
    rand: np.random.Generator,
    world: World,
    task_config: CompositeTaskConfig,
    solution: CompositionalSolutionData,
    difficulty: float,
    random_predator_count: int,
    full_path: CompositionalSolutionPath,
) -> World:
    if random_predator_count > 0:

        weapon_ratio = round(
            difficulty_variation(
                task_config.weapon_ratio_easy,
                task_config.weapon_ratio_hard,
                rand,
                difficulty,
            )
        )
        weapon_count = round(weapon_ratio * random_predator_count)

        # probably will never need more than this resolution for path distances
        max_points = 1000
        sq_path_distances = (
            world.map.distances_from_points(
                full_path.point_indices, solution.locations.island, "solution path distances", max_points, rand
            )
            ** 2
        )

        max_distance = scale_with_difficulty(
            difficulty, task_config.predator_path_dist_easy, task_config.predator_path_dist_hard
        )
        max_sq_dist = max_distance**2
        for i in range(random_predator_count):
            position = world.get_safe_point(
                rand, sq_distances=sq_path_distances, max_sq_dist=max_sq_dist, island_mask=solution.locations.island
            )
            if position is not None:
                predator, output_difficulty = get_random_predator(rand, difficulty)
                predator = attr.evolve(predator, position=position)
                world = world.add_item(predator, reset_height_offset=predator.get_offset())

        max_distance = scale_with_difficulty(
            difficulty, task_config.weapon_path_dist_easy, task_config.weapon_path_dist_hard
        )
        max_sq_dist = max_distance**2
        weapon_classes: List[Type[Weapon]] = []
        for i in range(weapon_count):
            safe_mask = world.get_safe_mask(
                sq_distances=sq_path_distances, max_sq_dist=max_sq_dist, island_mask=solution.locations.island
            )
            position = world._get_safe_point(rand, safe_mask)
            rock_probability = scale_with_difficulty(
                difficulty, task_config.rock_probability_easy, task_config.rock_probability_hard
            )
            if rand.uniform() < rock_probability:
                if rand.uniform() < task_config.large_rock_probability:
                    weapon_classes.append(LargeRock)
                else:
                    weapon_classes.append(Rock)
            else:
                weapon_classes.append(Stick)
            if position is not None:
                tool = Placeholder(position=position)
                world = world.add_item(tool, reset_height_offset=tool.get_offset())
        world = world.replace_weapon_placeholders(weapon_classes, solution.locations.island, ITEM_FLATTEN_RADIUS)

    return world


def move_all_items_up(items: List[Entity], height_delta: float) -> List[Entity]:
    new_items = []
    for item in items:
        new_item_position = item.position.copy()
        new_item_position[1] += height_delta
        item = attr.evolve(item, position=new_item_position)
        new_items.append(item)
    return new_items


def remove_close_predators(world: World, spawn: Point2DNP, radius: float) -> World:
    assert_isinstance(spawn, Point2DNP)
    new_items = []
    for item in world.items:
        too_close = False
        if isinstance(item, Predator):
            if np.linalg.norm(to_2d_point(item.position) - spawn) < radius:
                too_close = True
        if not too_close:
            new_items.append(item)
    return attr.evolve(world, items=tuple(new_items))


def get_building_task_generator_class(building_task: BuildingTask) -> Type[BuildingTaskGenerator]:
    generator_by_task = {
        BuildingTask.EXPLORE: ExploreTaskGenerator,
        BuildingTask.OPEN: OpenTaskGenerator,
        BuildingTask.PUSH: PushTaskGenerator,
        BuildingTask.STACK: StackTaskGenerator,
        BuildingTask.CLIMB: ClimbTaskGenerator,
        BuildingTask.JUMP: JumpTaskGenerator,
    }
    if building_task not in generator_by_task:
        raise ValueError(f"No building task generator for {building_task}")
    return generator_by_task[building_task]


def create_building(
    rand: np.random.Generator,
    world: World,
    task_config: CompositeTaskConfig,
    point: Point2DNP,
    radius: float,
    building_task: BuildingTask,
    difficulty: float,
    aesthetics: BuildingAestheticsConfig = BuildingAestheticsConfig(),
    is_shore_created: bool = False,
):
    building_location = np.array([point[0], world.map.get_rough_height_at_point(point), point[1]])
    building_task_config = task_config.indoor_obstacle_configs[building_task][1]
    task_generator = get_building_task_generator_class(building_task)(building_task_config)
    building, items, _spawn_location, target_location = create_building_for_skill_scenario(
        rand,
        difficulty,
        task_generator,
        building_location,
        allowed_auxiliary_tasks=(BuildingTask.OPEN, BuildingTask.CLIMB),
        allowed_entrance_tasks=(BuildingTask.OPEN,),
        aesthetics_config=aesthetics,
        is_indoor_only=False,
    )

    # create a flat area around the building
    map_new = world.map.copy()
    map_new.radial_flatten(point, radius * 2)
    # plot_value_grid(map_new.Z, "After flattening for building")

    # also make space for the basement:
    within_radius_mask = map_new.get_dist_sq_to(point) < radius * radius
    points = map_new.get_2d_points()
    within_radius_points = points[within_radius_mask]

    # figure out where the building is in the grid
    building_mask = np.zeros_like(within_radius_mask)
    polygon = np.array([x for x in get_building_footprint_outline(building).exterior.coords])
    building_mask[within_radius_mask] = points_in_polygon(polygon, within_radius_points)

    # if requested, ensure all of those points are on land (other calls must guarantee this)
    if is_shore_created:
        map_new.Z[building_mask] = np.clip(map_new.Z[building_mask], WATER_LINE + 0.1, None)

    # set points near the building outline as special
    building_wall_cell_thickness = round(map_new.cells_per_meter * 2.0)
    building_outline_mask = map_new.get_outline(building_mask, building_wall_cell_thickness)
    is_detail_important_new = world.is_detail_important.copy()
    is_detail_important_new[building_outline_mask] = True
    # plot_value_grid(world.is_detail_important)

    # set height inside the building outline lower depending on the basement height
    basement_mask = map_new.shrink_region(building_mask, building_wall_cell_thickness // 2)
    basement_height = building.stories[0].floor_negative_depth + DEFAULT_FLOOR_THICKNESS + 0.1
    map_new.Z[basement_mask] -= basement_height
    # plot_value_grid(basement_mask)

    # if it is underwater, have to move everything up:
    building_min_height = map_new.Z[basement_mask].min()
    if building_min_height <= WATER_LINE:
        height_delta = -1.0 * building_min_height + 0.1
        map_new.Z[basement_mask] += height_delta
        items = move_all_items_up(items, height_delta)
        new_building_position = building.position + np.array([0, height_delta, 0])
        building = attr.evolve(building, position=new_building_position)

    # prevent trees and stuff from growing into buildings
    world = world.mask_flora(1.0 - map_new.create_radial_mask(point, radius * 2))
    world = attr.evolve(world, map=map_new, is_detail_important=is_detail_important_new)

    building, world = world.add_building(building, basement_mask)

    return target_location, building, items, world


def points_in_polygon(polygon: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype="float32")
    polygon = np.asarray(polygon, dtype="float32")
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2 - polygon
    mask1 = (pts[:, None] == polygon).all(-1).any(-1)
    m1 = (polygon[:, 1] > pts[:, None, 1]) != (contour2[:, 1] > pts[:, None, 1])
    slope = ((pts[:, None, 0] - polygon[:, 0]) * test_diff[:, 1]) - (
        test_diff[:, 0] * (pts[:, None, 1] - polygon[:, 1])
    )
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:, 1] < polygon[:, 1])
    m4 = m1 & m3
    count = np.count_nonzero(m4, axis=-1)
    mask3 = ~(count % 2 == 0)
    mask = mask1 | mask2 | mask3
    return cast(np.ndarray, mask)


def get_acceptable_building_placement_locations(world: World, safe_mask: MapBoolNP, radius: float) -> MapBoolNP:
    unsafe = np.logical_not(safe_mask)
    radius_cell_dist = int(round(world.map.cells_per_meter * radius)) + 1
    return np.logical_and(safe_mask, np.logical_not(morphology.dilation(unsafe, morphology.disk(radius_cell_dist))))


def get_random_building_obstacle(
    rand: np.random.Generator,
    world: World,
    task_config: CompositeTaskConfig,
    locations: WorldLocations,
    difficulty: float,
) -> Optional[Tuple[BuildingTask, float, MapBoolNP, float]]:
    safe_mask = world.get_safe_mask(locations.island)
    for i in range(_INDOOR_OBSTACLE_MAX_RETRIES):
        task = _select_building_obstacle_type(rand, task_config)
        radius = get_radius_for_building_task(rand, task, difficulty, task_config)
        mask = get_acceptable_building_placement_locations(world, safe_mask, radius)
        if np.any(mask):
            return task, radius, mask, difficulty

    # fine, try one last time with the simplest building possible:
    task = BuildingTask.EXPLORE
    radius = MIN_NAVIGATE_BUILDING_SIZE
    mask = get_acceptable_building_placement_locations(world, safe_mask, radius)
    if np.any(mask):
        return task, radius, mask, difficulty

    return None


def _return_data(
    rand: np.random.Generator,
    world: World,
    difficulty: float,
    locations: WorldLocations,
    is_food_required: bool,
    is_visibility_required: bool,
    task_config: CompositeTaskConfig,
) -> Tuple[World, WorldLocations]:
    if is_food_required:
        world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(
        rand, difficulty, locations.spawn, locations.goal, is_visibility_required=is_visibility_required
    )
    world = remove_close_predators(world, to_2d_point(locations.spawn), task_config.min_predator_distance_from_spawn)
    return world, locations


def _add_height_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    world: World,
    obstacle_type: AvalonTask,
    task_config: CompositeTaskConfig,
    solution: CompositionalSolutionData,
    radius_distance_to_use: float,
    is_around_spawn: bool,
    # TODO: looks like this might just be the negative of the above
    is_height_inverted: bool,
    is_debug_graph_printing_enabled: bool,
) -> Tuple[int, float, CompositionalSolutionData, WorldLocations, World]:
    errors = []

    radius_left = solution.locations.get_2d_spawn_goal_distance() - (
        solution.current_spawn_side_radius + solution.current_goal_side_radius
    )
    if radius_left <= 0.0:
        raise ImpossibleWorldError("Not enough radius left for another height obstacle")
    max_radius_distance_to_use = min([radius_left * 0.99, radius_distance_to_use])
    min_radius_distance_to_use = max_radius_distance_to_use * 0.8

    for i in range(_OUTDOOR_OBSTACLE_MAX_RETRIES):
        radius_step = rand.uniform(min_radius_distance_to_use, max_radius_distance_to_use)
        try:
            if is_around_spawn:
                center = solution.locations.spawn
                start_point = solution.spawn_endpoint
                final_radius = solution.current_spawn_side_radius + radius_step
                end_point = solution.full_path.get_point_at_radius_from_spawn(final_radius)
            else:
                center = solution.locations.goal
                start_point = solution.goal_endpoint
                final_radius = solution.current_goal_side_radius + radius_step
                end_point = solution.full_path.get_point_at_radius_from_goal(final_radius)

            constraint = CompositionalConstraint(
                attr.evolve(solution.locations, spawn=start_point, goal=end_point),
                world,
                center,
                solution.full_path.mask,
                is_height_inverted,
            )

            # if is_debug_graph_printing_enabled:
            #     interesting_points = [constraint.center, constraint.locations.spawn, constraint.locations.goal]
            #     markers = [world.map.point_to_index(np.array([x[0], x[2]])) for x in interesting_points]
            #     plot_terrain(world.map.Z, f"Solution constraint for {obstacle_type}", markers=markers)

            # remember the heights of the goal and spawn before we change stuff
            goal_height_start = world.map.get_rough_height_at_point(to_2d_point(solution.locations.goal))
            spawn_height_start = world.map.get_rough_height_at_point(to_2d_point(solution.locations.spawn))

            inner_task_config = task_config.outdoor_obstacle_configs[obstacle_type][1]

            export_config = world.export_config
            if obstacle_type == AvalonTask.MOVE:
                assert isinstance(inner_task_config, MoveTaskConfig), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_move_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.JUMP:
                assert isinstance(
                    inner_task_config, JumpOutdoorTaskConfig
                ), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_jump_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.CLIMB:
                assert isinstance(
                    inner_task_config, ClimbOutdoorTaskConfig
                ), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_climb_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.PUSH:
                assert isinstance(
                    inner_task_config, PushOutdoorTaskConfig
                ), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_push_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.FIGHT:
                assert isinstance(inner_task_config, FightTaskConfig), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_fight_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.STACK:
                assert isinstance(
                    inner_task_config, StackOutdoorTaskConfig
                ), f"Unexpected type: {type(inner_task_config)}"

                world, new_locations, difficulty = create_stack_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.BRIDGE:
                assert isinstance(inner_task_config, BridgeTaskConfig), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_bridge_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            elif obstacle_type == AvalonTask.DESCEND:
                assert isinstance(inner_task_config, DescendTaskConfig), f"Unexpected type: {type(inner_task_config)}"
                world, new_locations, difficulty = create_descend_obstacle(
                    rand, difficulty, export_config, constraint, task_config=inner_task_config
                )
            else:
                raise SwitchError(f"Unhandled obstacle type: {obstacle_type}")

            break
        except ImpossibleWorldError as e:
            if i >= _OUTDOOR_OBSTACLE_MAX_RETRIES - 1:
                raise
            errors.append(e)

    # fix up the heights
    goal_height_end = world.map.get_rough_height_at_point(to_2d_point(solution.locations.goal))
    spawn_height_end = world.map.get_rough_height_at_point(to_2d_point(solution.locations.spawn))
    goal_fixed = solution.locations.goal.copy()
    goal_fixed[1] += goal_height_end - goal_height_start
    spawn_fixed = solution.locations.spawn.copy()
    spawn_fixed[1] += spawn_height_end - spawn_height_start
    new_solution = attr.evolve(solution, locations=attr.evolve(solution.locations, spawn=spawn_fixed, goal=goal_fixed))

    # logger.debug(f"Goal height changed by {goal_height_end - goal_height_start}")

    if is_debug_graph_printing_enabled:
        interesting_points = [constraint.center, constraint.locations.spawn, constraint.locations.goal]
        markers = [world.map.point_to_index(np.array([x[0], x[2]])) for x in interesting_points]
        plot_terrain(world.map.Z, f"Solution constraint for {obstacle_type}", markers=markers)

    # new solution needs to figure out  max point along the path given the new radius
    if is_around_spawn:
        new_solution = new_solution.update_spawn(final_radius)
    else:
        new_solution = new_solution.update_goal(final_radius)
    return 1, difficulty, new_solution, new_locations, world


def _find_path(rand: np.random.Generator, world: World, locations: WorldLocations) -> CompositionalSolutionPath:
    start_2d = to_2d_point(locations.spawn)
    end_2d = to_2d_point(locations.goal)
    path = _find_path_from_points(rand, world, start_2d, end_2d)
    mask = np.zeros_like(world.is_climbable)
    point_indices = np.array(path)
    sel = point_indices[:, 0], point_indices[:, 1]
    mask[sel] = 1
    points_2d = np.stack([world.map.X[sel], world.map.Y[sel]], axis=1)
    points = np.stack([world.map.X[sel], world.map.Z[sel], world.map.Y[sel]], axis=1)
    start_2d_radii = np.linalg.norm(points_2d - start_2d, axis=-1)
    end_2d_radii = np.linalg.norm(points_2d - end_2d, axis=-1)
    path_locations = np.linspace(0.0, 1.0, num=len(end_2d_radii))
    return CompositionalSolutionPath(
        mask=mask,
        path_locations=path_locations,
        points=points,
        point_indices=point_indices,
        start_2d_radii=start_2d_radii,
        end_2d_radii=end_2d_radii,
    )


def _find_path_from_points(
    rand: np.random.Generator, world: World, start: Point2DNP, end: Point2DNP
) -> List[Tuple[int, int]]:
    ngrid = len(world.map.Z)
    # graph = nx.grid_graph(dim=[ngrid, ngrid])
    graph = nx.grid_2d_graph(ngrid, ngrid)

    max_water_dist = 20.0
    max_penalty = 100000
    sqrt2 = math.sqrt(2)

    water_distances = world.map.get_water_distance(
        rand,
        is_fresh_water_included_in_moisture=True,
        max_points=world.biome_config.max_kd_points if world.biome_config else None,
    )

    def cost(a: Tuple[int, int], b: Tuple[int, int], k1: float = 1.0, k2: float = 10.0, kind: str = "intsct") -> float:
        dist = 1 if (a[0] == b[0] or a[1] == b[1]) else sqrt2
        assert water_distances is not None
        water_distance = water_distances[b]
        if water_distance <= 0.01:
            penalty = np.inf
        elif water_distance < max_water_dist:
            factor = 1.0 - (water_distance / max_water_dist) ** 3
            penalty = max_penalty * factor
        else:
            penalty = 0.0
        return dist + penalty

    start_indices = world.map.point_to_index(start)
    end_indices = world.map.point_to_index(end)
    return cast(List[Tuple[int, int]], nx.astar_path(graph, start_indices, end_indices, _get_2d_dist, weight=cost))


def _get_2d_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    x1 = np.array(a, dtype=np.float32)
    x2 = np.array(b, dtype=np.float32)
    return cast(float, np.linalg.norm(x1 - x2))


def _select_ring_obstacle_type(
    rand: np.random.Generator,
    allowed_obstacles: Tuple[AvalonTask, ...],
    task_config: CompositeTaskConfig,
) -> AvalonTask:
    obstacles = allowed_obstacles
    probabilities = []
    for obstacle in allowed_obstacles:
        probabilities.append(task_config.outdoor_obstacle_configs[obstacle][0])
    probabilities_np = np.array(probabilities)
    probabilities_np /= probabilities_np.sum()
    return rand.choice(obstacles, p=probabilities_np)  # type: ignore


def _select_building_obstacle_type(
    rand: np.random.Generator,
    task_config: CompositeTaskConfig,
) -> BuildingTask:
    obstacles = [x for x in task_config.indoor_obstacle_configs.keys()]
    probabilities = []
    for obstacle in obstacles:
        probabilities.append(task_config.indoor_obstacle_configs[obstacle][0])
    probabilities_np = np.array(probabilities)
    probabilities_np /= probabilities_np.sum()
    return rand.choice(obstacles, p=probabilities_np)  # type: ignore


def _select_final_obstacle_type(
    rand: np.random.Generator,
    difficulty: float,
    obstacles: List[FinalObstacleType],
    _FORCED: Optional[FinalObstacleType] = None,
) -> FinalObstacleType:
    probabilities = [1.0] * len(obstacles)
    probabilities_np = np.array(probabilities)
    probabilities_np /= probabilities_np.sum()
    return rand.choice(obstacles, p=probabilities_np)  # type: ignore


def _get_desired_obstacle_count(
    rand: np.random.Generator,
    difficulty: float,
    task_config: CompositeTaskConfig,
    _FORCED: Optional[ForcedComposition],
) -> Tuple[int, float]:
    if task_config.task == AvalonTask.SURVIVE:
        if _FORCED and _FORCED.is_enabled:
            desired_obstacle_count = 1 if difficulty > 0.5 else 0
        else:
            is_obstacle_present, difficulty = select_boolean_difficulty(difficulty, rand)
            desired_obstacle_count = 1 if is_obstacle_present else 0
    else:
        if _FORCED and _FORCED.is_enabled:
            desired_obstacle_count = round(scale_with_difficulty(difficulty, 1.5, task_config.max_obstacles + 0.5))
        else:
            desired_obstacle_count = round(
                normal_distrib_range(
                    1.25, task_config.max_obstacles + 0.49, task_config.max_obstacles_std_dev, rand, difficulty
                )
            )
    return desired_obstacle_count, difficulty


def _get_obstacle_counts_per_type(
    rand: np.random.Generator, desired_obstacle_count: int, task_config: CompositeTaskConfig, is_food_allowed: bool
) -> Tuple[int, int]:
    if desired_obstacle_count == 0:
        return 0, 0
    is_food_in_building = rand.uniform() < task_config.is_goal_in_building_probability
    building_count = 1 if is_food_in_building else 0
    if not is_food_allowed:
        building_count = 0
    remaining = desired_obstacle_count - building_count
    # can only create new food (ie, buildings) if this is a gather or survive task
    if task_config.task in (AvalonTask.GATHER, AvalonTask.SURVIVE) and is_food_allowed:
        count_one = rand.integers(0, remaining, endpoint=True)
        count_two = remaining - count_one
        values = np.array([count_one, count_two])
        rand.shuffle(values)
        extra_building_count, height_obstacle_count = list(values)
        building_count += extra_building_count
    else:
        height_obstacle_count = remaining
    return building_count, height_obstacle_count


def _get_fall_climb_split(rand: np.random.Generator, desired_height_obstacle_count: int) -> Tuple[int, int]:
    if desired_height_obstacle_count == 0:
        return 0, 0
    fall_count = rand.integers(0, desired_height_obstacle_count, endpoint=True)
    climb_count = desired_height_obstacle_count - fall_count
    return fall_count, climb_count


def get_random_predator(rand: np.random.Generator, difficulty: float):
    predator_class, new_difficulty = select_categorical_difficulty(ALL_PREDATOR_CLASSES, difficulty, rand)
    predator = predator_class(position=np.array([0.0, 0.0, 0.0]))
    return predator, new_difficulty


def get_radius_for_building_task(
    rand: np.random.Generator, task: BuildingTask, difficulty: float, task_config: Optional[CompositeTaskConfig] = None
) -> float:
    generator_kwargs = {}
    if task_config is not None:
        generator_kwargs["config"] = task_config.indoor_obstacle_configs[task][1]
    generator_class = get_building_task_generator_class(task)
    return generator_class(**generator_kwargs).get_site_radius(rand, difficulty)
