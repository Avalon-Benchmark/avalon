import math
from enum import Enum
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import attr
import networkx as nx
import numpy as np
from nptyping import assert_isinstance
from scipy import stats
from skimage import morphology

from common.errors import SwitchError
from datagen.world_creation.constants import UP_VECTOR
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import WATER_LINE
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.heightmap import HeightMap
from datagen.world_creation.heightmap import MapBoolNP
from datagen.world_creation.heightmap import MapFloatNP
from datagen.world_creation.heightmap import Point2DNP
from datagen.world_creation.heightmap import Point3DListNP
from datagen.world_creation.heightmap import Point3DNP
from datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from datagen.world_creation.indoor.objects import BuildingAestheticsConfig
from datagen.world_creation.indoor_task_generators import MIN_NAVIGATE_BUILDING_SIZE
from datagen.world_creation.indoor_task_generators import BuildingTask
from datagen.world_creation.indoor_task_generators import create_building_obstacle
from datagen.world_creation.indoor_task_generators import get_radius_for_building_task
from datagen.world_creation.items import ALL_PREDATOR_CLASSES
from datagen.world_creation.items import ALL_PREY_CLASSES
from datagen.world_creation.items import CANONICAL_FOOD_CLASS
from datagen.world_creation.items import CANONICAL_FOOD_HEIGHT
from datagen.world_creation.items import CANONICAL_FOOD_HEIGHT_ON_TREE
from datagen.world_creation.items import FOODS
from datagen.world_creation.items import FOOD_TREE_VISIBLE_HEIGHT
from datagen.world_creation.items import InstancedDynamicItem
from datagen.world_creation.items import Predator
from datagen.world_creation.items import Rock
from datagen.world_creation.items import Stick
from datagen.world_creation.items import Tool
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.biome_settings import generate_aesthetics_config
from datagen.world_creation.tasks.bridge import create_bridge_obstacle
from datagen.world_creation.tasks.carry import get_carry_distance_preference
from datagen.world_creation.tasks.climb import create_climb_obstacle
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.constants import IS_WORLD_DIVERSITY_ENABLED
from datagen.world_creation.tasks.descend import create_descend_obstacle
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.eat import select_tool
from datagen.world_creation.tasks.fight import create_fight_obstacle
from datagen.world_creation.tasks.jump import create_jump_obstacle
from datagen.world_creation.tasks.move import create_move_obstacle
from datagen.world_creation.tasks.push import create_push_obstacle
from datagen.world_creation.tasks.stack import create_stack_obstacle
from datagen.world_creation.tasks.task_worlds import WorldType
from datagen.world_creation.tasks.task_worlds import create_world_for_skill_scenario
from datagen.world_creation.tasks.utils import get_rock_probability
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.utils import plot_terrain
from datagen.world_creation.utils import plot_value_grid
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point

# the largest number of obstacles that will ever be put in a single world
MAX_OBSTACLES = 4

# most obstacles need about this much
GOAL_DISTANCE_PER_OBSTACLE = 21.0

# don't start predators within this radius of where you start
MIN_PREDATOR_DISTANCE_FROM_SPAWN = 10.0

# how high to boost the spawn to ensure that the goal is visible
MAX_VISIBILITY_HEIGHT_BOOST = 8.0

WORLD_SIZE_STD_DEV = 20.0


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ForcedComposition:
    is_enabled: bool = False


class RingObstacleType(Enum):
    """These are all of the obstacles that create rings around a point"""

    MOVE_PATH = "MOVE_PATH"
    JUMP_GAP = "JUMP_GAP"
    CLIMB_PATH = "CLIMB_PATH"
    PUSH = "PUSH"
    FIGHT = "FIGHT"
    STACK_GAP = "STACK_GAP"
    BRIDGE_GAP = "BRIDGE_GAP"
    FALL = "FALL"


class FinalObstacleType(Enum):
    PREDATOR = "PREDATOR"
    CARRY = "CARRY"


class RadiateMode(Enum):
    SPAWN = "SPAWN"
    GOAL = "GOAL"
    RANDOM = "RANDOM"


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
        return self.points[idx]

    def get_point_at_radius_from_spawn(self, radius: float) -> Point3DNP:
        far_enough_away = self.start_2d_radii > radius
        if not np.any(far_enough_away):
            raise ImpossibleWorldError("The safety radius for the start point included the end point!")
        idx = int(np.argwhere(far_enough_away)[0])
        return self.points[idx]


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class CompositionalSolutionData:
    locations: WorldLocationData
    full_path: CompositionalSolutionPath
    current_spawn_side_radius: float = 0.0
    current_goal_side_radius: float = 0.0

    @staticmethod
    def build(
        locations: WorldLocationData, full_path: CompositionalSolutionPath, goal_safety_radius: float
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

    def plot(self, map: HeightMap):
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


def create_compositional_task(
    rand: np.random.Generator,
    difficulty: float,
    task: AvalonTask,
    export_config: ExportConfig,
    desired_goal_distance: Optional[stats.norm] = None,
    is_food_allowed: bool = True,
    is_debug_graph_printing_enabled: bool = False,
    min_size_in_meters: Optional[float] = None,
    max_size_in_meters: Optional[float] = None,
    _FORCED: Optional[ForcedComposition] = None,
) -> Tuple[NewWorld, WorldLocationData]:
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

    desired_obstacle_count, difficulty = _get_desired_obstacle_count(rand, difficulty, task, _FORCED)
    # figure out how many of each type of obstacle we want
    (building_count, desired_height_obstacle_count) = _get_obstacle_counts_per_type(
        rand, desired_obstacle_count, task, is_food_allowed
    )
    if task == AvalonTask.SURVIVE:
        # this is because it is handled outside of this function
        random_predator_count = 0
    else:
        predator_count_options = list(range(5))
        random_predator_count, _difficulty = select_categorical_difficulty(predator_count_options, difficulty, rand)
        # print(predator_count_options, random_predator_count, difficulty)

    if desired_goal_distance is None:
        desired_goal_distance, difficulty = _get_desired_goal_distance(rand, difficulty, desired_obstacle_count)

    # TODO: RESUME: check that these mins and maxes are correct
    if max_size_in_meters is None:
        # 2x because we want space for the obstacles, and additional 1x to account fo the 50%
        # variation in world size from the way the fractal stuff works out
        max_size_in_meters = 3.0 * desired_goal_distance

    if task == AvalonTask.NAVIGATE:
        is_visibility_required = True
    else:
        # is_visibility_required, difficulty = select_boolean_difficulty(difficulty, rand)
        is_visibility_required = False

    if is_debug_graph_printing_enabled:
        print(
            f"Obstacles: {building_count} buildings, {desired_height_obstacle_count} height, {random_predator_count} predators"
        )
        print(f"Visibility: {is_visibility_required}")

    is_destination_building = False
    if task in (AvalonTask.GATHER, AvalonTask.SURVIVE):
        is_destination_building = building_count > 0
    elif building_count > 0:
        is_destination_building = True
        assert (
            building_count <= 1
        ), "Can only create a single food, and buildings must have food, thus cannot create extra buildings"

    min_visible_height_for_world_creation = FOOD_TREE_VISIBLE_HEIGHT
    if is_destination_building:
        # TODO: this could be more controllable
        # this is roughly the height of the smallest building
        min_visible_height_for_world_creation = 4.0

    goal_distance = stats.norm(desired_goal_distance, WORLD_SIZE_STD_DEV)
    world, starting_locations = create_world_for_skill_scenario(
        rand,
        difficulty if IS_WORLD_DIVERSITY_ENABLED else 0.0,
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
        print("WORLD SIZE:", world.config.size_in_meters ** 2)  # TODO KJ
        print(
            f"Goal and spawn are {starting_locations.get_2d_spawn_goal_distance()} away, we need at least {desired_obstacle_count * GOAL_DISTANCE_PER_OBSTACLE}"
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
        possible_tasks = [x for x in BuildingTask]
        building_task, difficulty = select_categorical_difficulty(possible_tasks, difficulty, rand)
        building_radius = get_radius_for_building_task(rand, building_task, difficulty)
        building_location_2d = to_2d_point(starting_locations.goal)
        new_goal_location, initial_building, initial_building_items = create_building(
            rand,
            world,
            building_location_2d,
            building_radius,
            building_task,
            difficulty,
            aesthetics,
            is_shore_created=True,
        )
        initial_building_starting_terrain_height = world.map.get_rough_height_at_point(building_location_2d)
        is_food_required = False
        starting_locations = attr.evolve(starting_locations, goal=new_goal_location)
        obstacle_count += 1
        building_count -= 1
        if obstacle_count >= desired_obstacle_count:
            # just add the items now because we're done
            for item in initial_building_items:
                world.add_item(item)
            world.add_item(CANONICAL_FOOD_CLASS(entity_id=0, position=new_goal_location), CANONICAL_FOOD_HEIGHT)
            # TODO amazingly bad code make this less bad later
            full_path = _find_path(rand, world, starting_locations)
            solution = CompositionalSolutionData.build(starting_locations, full_path, goal_safety_radius)
            add_extra_predators(rand, world, solution, difficulty, random_predator_count, full_path)

            return _return_data(rand, world, difficulty, starting_locations, is_food_required, is_visibility_required)

        goal_safety_radius = building_radius
        # this is how much above the goal is visible
        goal_visible_height_delta = initial_building.height.max_lt - starting_locations.goal

    # figure out how we're going to move from the start to the end
    full_path = _find_path(rand, world, starting_locations)
    solution = CompositionalSolutionData.build(starting_locations, full_path, goal_safety_radius)

    if is_debug_graph_printing_enabled:
        print(f"Initial safety radius: {goal_safety_radius}")
        print(f"Reaming distance: {solution.locations.get_2d_spawn_goal_distance()}")

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
            print("Visibility required!!!")
        print("Selected obstacle types:")
        print(f"{building_count + int(is_destination_building)} buildings")
        print(f"{desired_height_obstacle_count} height obstacles")
        print(f"{random_predator_count} predators")

    # raise the island up
    if desired_height_obstacle_count > 0:
        solution = attr.evolve(solution, locations=world.begin_height_obstacles(solution.locations))

        # different ways of dealing with compositional height obstacles
        # in raise island mode, we ensure that you cannot work around height obstacles
        # in the other mode, we simply dont use obstacle combinations that can end up being circumventable
        is_raised_island_mode = rand.uniform() < 0.8

        if is_debug_graph_printing_enabled:
            print("is_raised_island_mode", is_raised_island_mode)
        if is_raised_island_mode:
            # add as many ring obstacles as we can (up to the max we want)

            # keep track of which later obstacles are allowed for navigate, since we need to see the goal when swe start
            is_boosted = False
            if is_visibility_required:
                allowed_obstacles = tuple(
                    [
                        x
                        for x in RingObstacleType
                        if x
                        not in (
                            RingObstacleType.FALL,
                            RingObstacleType.CLIMB_PATH,
                            RingObstacleType.FIGHT,
                        )
                    ]
                )
                is_boosted = rand.uniform() > 0.5
            else:
                allowed_obstacles = tuple([x for x in RingObstacleType])

            radiate_mode = rand.choice([x for x in RadiateMode])

            # TODO: put back!!
            radiate_mode = RadiateMode.SPAWN

            distance_to_use = solution.get_distance_to_use(desired_height_obstacle_count)
            spawn_region = None

            if is_boosted:
                obstacle_type = RingObstacleType.CLIMB_PATH
                result, new_difficulty, solution, new_locations = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
                    solution,
                    distance_to_use,
                    is_around_spawn=False,
                    is_height_inverted=True,
                    is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
                )
                if result:
                    desired_height_obstacle_count -= 1
                    allowed_obstacles += tuple([RingObstacleType.CLIMB_PATH, RingObstacleType.FIGHT])
                    # move the food to the ledge
                    if is_food_required:
                        updated_locations = attr.evolve(solution.locations, goal=new_locations.goal)
                        solution = attr.evolve(solution, locations=updated_locations)

                    # figure out the spawn region (in case this is the only obstacle)
                    spawn_region = world.obstacle_zones[0][0]

            is_first_successful_obstacle = True
            for i in range(desired_height_obstacle_count):
                obstacle_type = _select_ring_obstacle_type(rand, allowed_obstacles)
                if radiate_mode == RadiateMode.SPAWN:
                    is_around_spawn = True
                elif radiate_mode == RadiateMode.GOAL:
                    is_around_spawn = False
                elif radiate_mode == RadiateMode.RANDOM:
                    is_around_spawn = rand.uniform() < 0.5
                else:
                    raise SwitchError(f"Unknown RadiateMode: {radiate_mode}")
                result, new_difficulty, solution, new_locations = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
                    solution,
                    distance_to_use,
                    is_around_spawn=is_around_spawn,
                    is_height_inverted=False,
                    is_debug_graph_printing_enabled=is_debug_graph_printing_enabled,
                )
                # print("goal height", solution.locations.goal[1])
                if result:
                    obstacle_count += 1
                    if is_first_successful_obstacle:
                        is_first_successful_obstacle = False
                        # reset the spawn region
                        spawn_region = world.obstacle_zones[-1][0]

            if spawn_region is None:
                raise ImpossibleWorldError("Failed to add any height obstacles in raised island mode")
            world.end_height_obstacles(solution.locations, is_accessible_from_water=False, spawn_region=spawn_region)

        else:
            # add ring based obstacle around spawn
            fall_count, climb_count = _get_fall_climb_split(rand, desired_height_obstacle_count)

            is_first_successful_obstacle = True
            spawn_region = None

            distance_to_use = solution.get_distance_to_use(desired_height_obstacle_count)
            for i in range(fall_count):
                obstacle_type = RingObstacleType.FALL
                result, new_difficulty, solution, new_locations = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
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
                obstacle_type = RingObstacleType.CLIMB_PATH
                result, new_difficulty, solution, new_locations = _add_height_obstacle(
                    rand,
                    difficulty,
                    world,
                    obstacle_type,
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
            world.end_height_obstacles(solution.locations, is_accessible_from_water=True, spawn_region=spawn_region)

        # if necessary, boost JUST the spawn safety radius (until the end is visible)
        if is_visibility_required:
            height_increments = 0
            max_increments = 20
            height_delta_per_step = MAX_VISIBILITY_HEIGHT_BOOST / max_increments
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
                    # print("busted")
                    # print(
                    #     solution.locations.spawn + up_vector * visibility_height_offset,
                    #     solution.locations.goal + up_vector * min_food_offset,
                    # )
                    # print("works?")
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
            spawn_radius = 1.0
            spawn_point_region = world.map.get_dist_sq_to(to_2d_point(solution.locations.spawn)) < spawn_radius ** 2
            if visibility_height_offset > 0.0:
                world.map.raise_island(spawn_point_region, visibility_height_offset)

    # figure out how much the height of the initial building changed, and update its (and its items0 positions
    if initial_building is not None:
        assert initial_building_items is not None
        assert initial_building_starting_terrain_height is not None
        new_height = world.map.get_rough_height_at_point(
            np.array([initial_building.position.x, initial_building.position.z])
        )
        height_change = new_height - initial_building_starting_terrain_height
        items = move_all_items_up(initial_building_items, height_change)
        for item in items:
            world.add_item(item)
        new_goal_location = starting_locations.goal + UP_VECTOR * height_change
        world.add_item(CANONICAL_FOOD_CLASS(entity_id=0, position=new_goal_location))
        assert len(world.building_by_id) == 1
        world.building_by_id.clear()
        new_building_position = attr.evolve(initial_building.position, y=initial_building.position.y + height_change)
        new_building = attr.evolve(initial_building, position=new_building_position)
        world.building_by_id[new_building.id] = new_building

    if is_debug_graph_printing_enabled:  # TODO KJ
        print("actual obstacle_count", obstacle_count)
        print("desired_obstacle_count", desired_obstacle_count)
    # if obstacle_count >= desired_obstacle_count:
    #     return _return_data(rand, world, difficulty, solution.locations, is_food_required, is_visibility_required)

    # add any required buildings
    for i in range(building_count):
        # randomly gets a task and location mask for where it can actually fit
        building_obstacle = get_random_building_obstacle(rand, world, solution.locations, difficulty)
        if building_obstacle:
            task, radius, mask, new_difficulty = building_obstacle
            building_location_2d = world.map.index_to_point_2d(tuple(rand.choice(np.argwhere(mask))))
            food_location, _building, items = create_building(
                rand, world, building_location_2d, radius, task, difficulty, aesthetics
            )
            for item in items:
                world.add_item(item)
            world.add_item(CANONICAL_FOOD_CLASS(entity_id=0, position=food_location))
            obstacle_count += 1
            # if obstacle_count >= desired_obstacle_count:
            #     return _return_data(
            #         rand, world, difficulty, solution.locations, is_food_required, is_visibility_required
            #     )

    # finally, add a bunch of predators
    add_extra_predators(rand, world, solution, difficulty, random_predator_count, full_path)

    return _return_data(rand, world, difficulty, solution.locations, is_food_required, is_visibility_required)


def add_extra_predators(rand, world, solution, difficulty, random_predator_count, full_path):
    if random_predator_count > 0:
        # probably will never need more than this resolution for path distances
        max_points = 1000
        sq_path_distances = (
            world.map.distances_from_points(
                full_path.point_indices, solution.locations.island, "solution path distances", max_points, rand
            )
            ** 2
        )
        carried_ids = set()
        for i in range(random_predator_count):
            _add_final_obstacle(
                rand, difficulty, world, solution.locations, carried_ids, sq_path_distances, is_carry_allowed=False
            )


def move_all_items_up(items, height_delta):
    new_items = []
    for item in items:
        new_item_position = item.position.copy()
        new_item_position[1] += height_delta
        item = attr.evolve(item, position=new_item_position)
        new_items.append(item)
    return new_items


def remove_close_predators(world: NewWorld, spawn: Point2DNP, radius: float):
    assert_isinstance(spawn, Point2DNP)
    new_items = []
    for item in world.items:
        too_close = False
        if isinstance(item, Predator):
            if np.linalg.norm(to_2d_point(item.position) - spawn) < radius:
                too_close = True
        if not too_close:
            new_items.append(item)
    world.items = new_items


def create_building(
    rand: np.random.Generator,
    world: NewWorld,
    point: Point2DNP,
    radius: float,
    building_task: BuildingTask,
    difficulty: float,
    aesthetics=BuildingAestheticsConfig(),
    is_shore_created: bool = False,
):
    building_location = np.array([point[0], world.map.get_rough_height_at_point(point), point[1]])
    building, items, _spawn_location, target_location = create_building_obstacle(
        rand,
        difficulty,
        building_task,
        radius,
        building_location,
        yaw_radians=0.0,
        aesthetics=aesthetics,
    )

    # create a flat area around the building
    world.map.radial_flatten(point, radius * 2)
    # plot_value_grid(world.map.Z, "After flattening for building")

    # also make space for the basement:
    within_radius_mask = world.map.get_dist_sq_to(point) < radius * radius
    points = world.map.get_2d_points()
    within_radius_points = points[within_radius_mask]

    # figure out where the building is in the grid
    building_mask = np.zeros_like(within_radius_mask)
    polygon = np.array([x for x in building.get_footprint_outline().exterior.coords])
    building_mask[within_radius_mask] = points_in_polygon(polygon, within_radius_points)

    # if requested, ensure all of those points are on land (other calls must guarantee this)
    if is_shore_created:
        world.map.Z[building_mask] = np.clip(world.map.Z[building_mask], WATER_LINE + 0.1, None)

    # set points near the building outline as special
    building_wall_cell_thickness = round(world.map.cells_per_meter * 2.0)
    building_outline_mask = world.map.get_outline(building_mask, building_wall_cell_thickness)
    world.is_detail_important[building_outline_mask] = True
    # plot_value_grid(world.is_detail_important)

    # set height inside the building outline lower depending on the basement height
    basement_mask = world.map.shrink_region(building_mask, building_wall_cell_thickness // 2)
    basement_height = building.stories[0].floor_negative_depth + DEFAULT_FLOOR_THICKNESS + 0.1
    world.map.Z[basement_mask] -= basement_height
    # plot_value_grid(basement_mask)

    # if it is underwater, have to move everything up:
    building_min_height = world.map.Z[basement_mask].min()
    if building_min_height <= WATER_LINE:
        height_delta = -1.0 * building_min_height + 0.1
        world.map.Z[basement_mask] += height_delta
        items = move_all_items_up(items, height_delta)
        new_building_position = attr.evolve(building.position, y=building.position.y + height_delta)
        building = attr.evolve(building, position=new_building_position)

    # prevent trees and stuff from growing into buildings
    world.mask_flora(1.0 - world.map.create_radial_mask(point, radius * 2))

    building = world.add_building(building, basement_mask)

    return target_location, building, items


def points_in_polygon(polygon, pts):
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
    return mask


def get_acceptable_building_placement_locations(world: NewWorld, safe_mask: MapBoolNP, radius: float) -> MapBoolNP:
    unsafe = np.logical_not(safe_mask)
    radius_cell_dist = int(round(world.map.cells_per_meter * radius)) + 1
    return np.logical_and(safe_mask, np.logical_not(morphology.dilation(unsafe, morphology.disk(radius_cell_dist))))


def get_random_building_obstacle(
    rand, world: NewWorld, locations: WorldLocationData, difficulty: float, num_retries: int = 5
) -> Optional[Tuple[BuildingTask, float, MapBoolNP, float]]:
    possible_tasks = [x for x in BuildingTask]
    safe_mask = world.get_safe_mask(locations.island)
    for i in range(num_retries):
        task, difficulty = select_categorical_difficulty(possible_tasks, difficulty, rand)
        radius = get_radius_for_building_task(rand, task, difficulty)
        mask = get_acceptable_building_placement_locations(world, safe_mask, radius)
        if np.any(mask):
            return task, radius, mask, difficulty

    # fine, try one last time with the simplest building possible:
    task = BuildingTask.NAVIGATE
    radius = MIN_NAVIGATE_BUILDING_SIZE
    mask = get_acceptable_building_placement_locations(world, safe_mask, radius)
    if np.any(mask):
        return task, radius, mask, difficulty

    return None


def _return_data(
    rand: np.random.Generator,
    world: NewWorld,
    difficulty: float,
    locations: WorldLocationData,
    is_food_required: bool,
    is_visibility_required: bool,
) -> Tuple[NewWorld, WorldLocationData]:
    if is_food_required:
        add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal, is_visibility_required=is_visibility_required)
    remove_close_predators(world, to_2d_point(locations.spawn), MIN_PREDATOR_DISTANCE_FROM_SPAWN)
    return world, locations


def _add_height_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    world: NewWorld,
    obstacle_type: RingObstacleType,
    solution: CompositionalSolutionData,
    radius_distance_to_use: float,
    is_around_spawn: bool,
    # TODO: looks like this might just be the negative of the above
    is_height_inverted: bool,
    is_debug_graph_printing_enabled: bool,
) -> Tuple[int, float, CompositionalSolutionData, WorldLocationData]:
    errors = []

    radius_left = solution.locations.get_2d_spawn_goal_distance() - (
        solution.current_spawn_side_radius + solution.current_goal_side_radius
    )
    assert radius_left > 0.0
    max_radius_distance_to_use = min([radius_left * 0.99, radius_distance_to_use])
    min_radius_distance_to_use = max_radius_distance_to_use * 0.8

    is_error_allowed = False

    is_successful = False
    for i in range(4):
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

            # obstacle_type = RingObstacleType.STACK_GAP
            export_config = world.export_config
            if obstacle_type == RingObstacleType.MOVE_PATH:
                _world, new_locations, difficulty = create_move_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.JUMP_GAP:
                _world, new_locations, difficulty = create_jump_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.CLIMB_PATH:
                _world, new_locations, difficulty = create_climb_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.PUSH:
                _world, new_locations, difficulty = create_push_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.FIGHT:
                _world, new_locations, difficulty = create_fight_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.STACK_GAP:
                _world, new_locations, difficulty = create_stack_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.BRIDGE_GAP:
                _world, new_locations, difficulty = create_bridge_obstacle(rand, difficulty, export_config, constraint)
            # elif obstacle_type == RingObstacleType.AVOID_PREDATOR:
            #     _world, new_locations, difficulty = create_avoid_obstacle(rand, difficulty, export_config, constraint)
            elif obstacle_type == RingObstacleType.FALL:
                _world, new_locations, difficulty = create_descend_obstacle(
                    rand, difficulty, export_config, constraint
                )
            else:
                raise SwitchError(f"Unhandled obstacle type: {obstacle_type}")

            is_successful = True
            break
        # TODO: it is pretty unsafe that we're allowing this retry as things are being mutated... would feel much
        #  safer if immutable
        except ImpossibleWorldError as e:
            if not is_error_allowed:
                raise
            errors.append(e)

    assert is_successful
    # if not is_successful:
    #     # TODO: better handling for these. Could likely retry in some cases, and might want to make sure it's not
    #     #  happening too frequently
    #     print(f"Impossible: {errors}")
    #     return 0, difficulty, solution, solution.locations

    # fix up the heights
    goal_height_end = world.map.get_rough_height_at_point(to_2d_point(solution.locations.goal))
    spawn_height_end = world.map.get_rough_height_at_point(to_2d_point(solution.locations.spawn))
    solution.locations.goal[1] = solution.locations.goal[1] + (goal_height_end - goal_height_start)
    solution.locations.spawn[1] = solution.locations.spawn[1] + (spawn_height_end - spawn_height_start)

    # print(f"Goal height changed by {goal_height_end - goal_height_start}")

    if is_debug_graph_printing_enabled:
        interesting_points = [constraint.center, constraint.locations.spawn, constraint.locations.goal]
        markers = [world.map.point_to_index(np.array([x[0], x[2]])) for x in interesting_points]
        plot_terrain(world.map.Z, f"Solution constraint for {obstacle_type}", markers=markers)

    # new solution needs to figure out  max point along the path given the new radius
    if is_around_spawn:
        new_solution = solution.update_spawn(final_radius)
    else:
        new_solution = solution.update_goal(final_radius)
    return 1, difficulty, new_solution, new_locations


def _find_path(rand: np.random.Generator, world: NewWorld, locations: WorldLocationData) -> CompositionalSolutionPath:
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
    rand: np.random.Generator, world: NewWorld, start: Point2DNP, end: Point2DNP
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

    def cost(a, b, k1=1.0, k2=10.0, kind="intsct"):
        dist = 1 if (a[0] == b[0] or a[1] == b[1]) else sqrt2

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
    return nx.astar_path(graph, start_indices, end_indices, _get_2d_dist, weight=cost)


def _get_2d_dist(a, b):
    x1 = np.array(a, dtype=np.float32)
    x2 = np.array(b, dtype=np.float32)
    return np.linalg.norm(x1 - x2)


def _add_final_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    world: NewWorld,
    locations: WorldLocationData,
    carried_ids: Set[int],
    sq_path_distances: MapFloatNP,
    is_carry_allowed: bool = False,
) -> float:
    """
    add predator (along solution path)
    carry (move a solution object much farther away from the spawn)
    (for gather): add extra food, potentially even on another island
    random pits (in unmasked, unimportant areas along the solution path)
    increase foliage density?
    """
    allowed_obstacles = [FinalObstacleType.PREDATOR]
    uncarried_objects = [
        x
        for x in world.items
        if x.entity_id not in carried_ids and isinstance(x, Tool) and getattr(x, "solution_maks", None) is not None
    ]
    if len(uncarried_objects) > 0 and is_carry_allowed:
        allowed_obstacles.append(FinalObstacleType.CARRY)
    obstacle_type = _select_final_obstacle_type(rand, difficulty, allowed_obstacles)

    # add a random predator along the path
    if obstacle_type == FinalObstacleType.PREDATOR:
        predator_count, output_difficulty = select_categorical_difficulty([1, 2, 3, 4], difficulty, rand)

        for i in range(predator_count):
            max_distance = scale_with_difficulty(difficulty, 20.0, 3.0)
            max_sq_dist = max_distance ** 2
            position = world.get_safe_point(
                rand, sq_distances=sq_path_distances, max_sq_dist=max_sq_dist, island_mask=locations.island
            )
            predator, output_difficulty = get_random_predator(rand, difficulty)
            predator = attr.evolve(predator, position=position)
            world.add_item(predator, reset_height_offset=predator.get_offset())

            # sometimes spawn a weapon as well
            is_weapon_present, difficulty = select_boolean_difficulty(difficulty, rand)
            if is_weapon_present:
                safe_mask = world.get_safe_mask(
                    sq_distances=sq_path_distances, max_sq_dist=max_sq_dist, island_mask=locations.island
                )
                position = world._get_safe_point(rand, safe_mask)
                tool = (
                    Rock(solution_mask=safe_mask)
                    if rand.uniform() < get_rock_probability(difficulty)
                    else Stick(solution_mask=safe_mask)
                )
                tool = attr.evolve(tool, position=position)
                world.add_item(tool, reset_height_offset=tool.get_offset())

    # move one of the items somewhere else
    elif obstacle_type == FinalObstacleType.CARRY:
        object_to_carry = rand.choice(uncarried_objects)
        carried_ids.add(object_to_carry.entity_id)
        world.carry_tool_randomly(rand, object_to_carry, get_carry_distance_preference(difficulty))

    else:
        raise SwitchError(f"Unhandled obstacle type: {obstacle_type}")
    return difficulty


def _select_ring_obstacle_type(
    rand: np.random.Generator,
    allowed_obstacles: Tuple[RingObstacleType, ...],
    bias_obstacle: Optional[RingObstacleType] = None,
    bias_factor: Optional[float] = None,
) -> RingObstacleType:
    obstacles = allowed_obstacles
    probabilities = [1.0] * len(obstacles)
    if bias_obstacle is not None:
        obstacles = obstacles + tuple([bias_obstacle])
        assert bias_factor is not None
        probabilities.append(bias_factor)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return rand.choice(obstacles, p=probabilities)


def _select_final_obstacle_type(
    rand: np.random.Generator,
    difficulty: float,
    obstacles: List[FinalObstacleType],
    _FORCED: Optional[FinalObstacleType] = None,
) -> FinalObstacleType:
    probabilities = [1.0] * len(obstacles)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()
    return rand.choice(obstacles, p=probabilities)


def _get_desired_obstacle_count(
    rand: np.random.Generator,
    difficulty: float,
    task: AvalonTask,
    _FORCED: Optional[ForcedComposition],
) -> Tuple[int, float]:
    if task == AvalonTask.SURVIVE:
        if _FORCED and _FORCED.is_enabled:
            desired_obstacle_count = 1 if difficulty > 0.5 else 0
        else:
            is_obstacle_present, difficulty = select_boolean_difficulty(difficulty, rand)
            desired_obstacle_count = 1 if is_obstacle_present else 0
    else:
        if _FORCED and _FORCED.is_enabled:
            desired_obstacle_count = round(scale_with_difficulty(difficulty, 1.5, MAX_OBSTACLES + 0.5))
        else:
            desired_obstacle_count = round(normal_distrib_range(1.25, MAX_OBSTACLES + 0.49, 1.0, rand, difficulty))
    return desired_obstacle_count, difficulty


def _get_desired_goal_distance(
    rand: np.random.Generator, difficulty: float, desired_obstacle_count: int
) -> Tuple[float, float]:
    obstacle_count_based_goal_distance = (desired_obstacle_count) * GOAL_DISTANCE_PER_OBSTACLE + 2 * WORLD_SIZE_STD_DEV
    return obstacle_count_based_goal_distance, difficulty


def _get_obstacle_counts_per_type(
    rand: np.random.Generator, desired_obstacle_count: int, task: AvalonTask, is_food_allowed: bool
) -> Tuple[int, int]:
    if desired_obstacle_count == 0:
        return 0, 0
    is_food_in_building = rand.uniform() < 0.5
    building_count = 1 if is_food_in_building else 0
    if not is_food_allowed:
        building_count = 0
    remaining = desired_obstacle_count - building_count
    # can only create new food (ie, buildings) if this is a gather or survive task
    if task in (AvalonTask.GATHER, AvalonTask.SURVIVE) and is_food_allowed:
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
    predator = predator_class(entity_id=0, position=np.array([0.0, 0.0, 0.0]))
    return predator, new_difficulty


def get_random_prey(rand: np.random.Generator, difficulty: float):
    prey_class, new_difficulty = select_categorical_difficulty(ALL_PREY_CLASSES, difficulty, rand)
    prey = prey_class(entity_id=0, position=np.array([0.0, 0.0, 0.0]))
    return prey, new_difficulty


def get_random_food_and_tool(
    rand: np.random.Generator, difficulty: float
) -> Tuple[InstancedDynamicItem, Optional[Tool], float]:
    if rand.uniform() < 0.5:
        food, new_difficulty = select_categorical_difficulty(FOODS, difficulty, rand)
        tool = select_tool(food, rand, difficulty)
    else:
        prey_class, new_difficulty = select_categorical_difficulty(ALL_PREY_CLASSES, difficulty, rand)
        food = prey_class(entity_id=0, position=np.array([0.0, 0.0, 0.0]))
        tool = Rock() if rand.uniform() < get_rock_probability(difficulty) else Stick()
    return food, tool, new_difficulty
