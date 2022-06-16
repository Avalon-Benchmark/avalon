from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import attr
import numpy as np
from scipy import stats

from datagen.world_creation.heightmap import WATER_LINE
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.heightmap import Point2DNP
from datagen.world_creation.heightmap import Point3DNP
from datagen.world_creation.items import CANONICAL_FOOD
from datagen.world_creation.items import FOODS
from datagen.world_creation.items import FOOD_TREE_BASE_HEIGHT
from datagen.world_creation.items import Altar
from datagen.world_creation.items import Food
from datagen.world_creation.items import FoodTree
from datagen.world_creation.items import Rock
from datagen.world_creation.items import Stick
from datagen.world_creation.items import Tool
from datagen.world_creation.items import get_random_ground_points
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import get_rock_probability
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import plot_value_grid
from datagen.world_creation.world_location_data import WorldLocationData

# MIN_TREE_HEIGHT = 3.5
# MAX_TREE_HEIGHT = 7
#
#
# def get_tree_height(rand: np.random.Generator, difficulty: float, _FORCED: Optional[float] = None):
#     if _FORCED:
#         return _FORCED
#     max_addition = MAX_TREE_HEIGHT - MIN_TREE_HEIGHT
#     return MIN_TREE_HEIGHT + rand.uniform(0, max_addition) * difficulty


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ForcedFood:
    food: Optional[Type[Food]] = None


def generate_eat_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    is_plotting: bool = False,
    _FORCED: Optional[ForcedFood] = None,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_eat_obstacle(rand, difficulty, export_config, _FORCED=_FORCED)
    world.end_height_obstacles(locations, is_accessible_from_water=True)
    if is_plotting:
        world.map.plot()
        markers = [world.map.point_to_index(np.array([x[0], x[2]])) for x in [locations.spawn, locations.goal]]
        # plot_value_grid(world.map.Z, markers=markers)
        plot_value_grid(world.map.Z < WATER_LINE, markers=markers)
        # print(f"Spawn: {spawn_point}")
        # print(f"Food: {food_point}")

    world.end_height_obstacles(locations, is_accessible_from_water=True)
    # add the spawn location
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)
    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


# TODO: fix this--need to implement multiple food being on a tree before this will work
#  Keep this function, as it is used externally to this module
def get_food_tree_ground_probability(difficulty: float) -> float:
    return 1.0


def create_eat_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    _FORCED: Optional[ForcedFood] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:
    if _FORCED is not None and _FORCED.food:
        food_type = _FORCED.food()
    else:
        food_type, difficulty = select_categorical_difficulty(FOODS, difficulty, rand)
    # TODO configure amount of foods (esp if a multi instance food)
    is_single_food, difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.3)

    is_extra_tree_required = False
    on_object: Optional[Union[FoodTree, Altar]] = None
    if food_type.is_found_on_ground:
        is_on_altar, difficulty = select_boolean_difficulty(difficulty, rand)
    else:
        is_on_altar = False

    if is_on_altar:
        on_object = Altar.build(rand.uniform(-1, 0.5) * difficulty + 1.3, table_dim=1 + difficulty * rand.uniform())
        food_height = on_object.get_food_height(food_type)
        desired_food_dist = rand.uniform(0, 5) * difficulty + 0.5
    else:
        if food_type.is_grown_on_trees:
            is_on_ground = False
            if food_type.is_found_on_ground:
                is_on_ground, difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.1)

            is_extra_tree_required = is_on_ground
            if not is_on_ground:
                # TODO: randomize tree height
                # tree_height = get_tree_height(rand, difficulty, _FORCED=MAX_TREE_HEIGHT)
                tree_height = FOOD_TREE_BASE_HEIGHT
                on_object = FoodTree.build(
                    tree_height,
                    ground_probability=get_food_tree_ground_probability(difficulty),
                    is_food_on_tree=True,
                )
        elif food_type.is_openable:
            is_opened, difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.05)
            if is_opened:
                food_type = food_type.get_opened_version()

        desired_food_dist = rand.uniform(0, 10) * difficulty + 1
    # print(f"Scalar difficulty {difficulty}")

    food_height = on_object.get_food_height(food_type) if on_object else food_type.get_offset()

    world, locations = create_world_from_constraint(
        stats.norm(desired_food_dist, desired_food_dist / 5.0),
        rand,
        difficulty,
        export_config,
        constraint,
        food_height,
        ideal_shore_dist_options=(0.5,),
    )
    world.end_height_obstacles(locations, is_accessible_from_water=False)

    # TODO: muck with the foliage density parameters in the biome config
    #  This makes it more difficult when the object is on the ground because it would be harder to see
    # # continuous for ground foliage density or not
    # foliage_density = rand.uniform() * np.interp(difficulty, [0.0, 0.3, 0.6, 1.0], [0.0, 0.2, 0.4, 1.0])
    tree: Optional[FoodTree] = None
    if on_object and isinstance(on_object, Altar):
        actual_distance_to_food = np.linalg.norm(locations.goal - locations.spawn)
        # orient the altar towards you if you're really close
        if actual_distance_to_food < 1.6:
            direction_to_target = (locations.goal - locations.spawn) / actual_distance_to_food
            # noinspection PyTypeChecker
            yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
            on_object = attr.evolve(on_object, yaw=yaw)
        UP_VECTOR = np.array([0.0, 1.0, 0.0])
        world.add_item(
            on_object.place(locations.goal - FOOD_TREE_BASE_HEIGHT * UP_VECTOR),
            reset_height_offset=on_object.get_offset(),
        )
    elif on_object and isinstance(on_object, FoodTree):
        tree = on_object
    elif is_extra_tree_required:
        # TODO: randomize tree height
        # tree_height = get_tree_height(rand, difficulty)
        tree_height = FOOD_TREE_BASE_HEIGHT
        tree = FoodTree.build(tree_height, 1.0, is_food_on_tree=False)

    primary_food = food_type
    # add the tree if required, because we need to use it to figure out the spawn location of the other foods
    if tree is not None:
        # place the tree + optional base in case it would be floating.
        tree, tree_base = tree.place(locations.spawn, locations.goal, world.get_height_at)
        tree = world.add_item(tree)
        if tree_base is not None:
            world.add_item(tree_base)
        # attach the food + set on_object for future calculations
        if tree.is_food_on_tree:
            primary_food = primary_food.attached_to(tree)
            on_object = tree

    # add the foods
    food_count = 1
    if not is_single_food:
        food_count = 2 + round(difficulty * 3 * rand.uniform())

    # TODO: the size variation for tools and food should be baked into their spawn methods so it is consistent across all tasks
    #  except we do care here about making them larger, hmm...
    #  I guess they should at least have default ranges, and here we're just varying them based on difficulty
    # per instance size variation
    # TODO add back per dimension variation?
    food_scales = np.ones((food_count, 3))  # + np.random.uniform(low=-0.5, high=1, size=(food_count, 1)) * difficulty
    # figure out where to position the foods
    food_locations = np.array([locations.goal])
    min_radius = 0.75
    max_radius = 2.0
    food_hover = food_type.get_offset()
    other_food_count = food_count - 1
    if other_food_count > 0:
        if on_object is None:
            if tree is None:
                other_food_locations = get_random_ground_points(
                    rand,
                    locations.goal,
                    other_food_count,
                    world.map,
                    food_scales[1:],
                    min_radius,
                    max_radius,
                    food_hover,
                    locations.island,
                )
            else:
                other_food_locations = tree.get_food_locations(
                    rand,
                    locations.goal,
                    other_food_count,
                    world.map,
                    food_scales[1:],
                    min_radius,
                    max_radius,
                    food_hover,
                    locations.island,
                )
        else:
            other_food_locations = on_object.get_food_locations(
                rand,
                locations.goal,
                other_food_count,
                world.map,
                food_scales[1:],
                min_radius,
                max_radius,
                food_hover,
                locations.island,
            )
        food_locations = np.concatenate([food_locations, other_food_locations], axis=0)
    for i, (food_location, scale) in enumerate(zip(food_locations, food_scales)):
        base = primary_food if i == 0 else food_type
        world.add_item(attr.evolve(base, position=base.additional_offset + food_location, scale=scale))
    # figure out what tools we need
    tools: List[Tool] = []
    first_tool = select_tool(food_type, rand, difficulty)
    if first_tool:
        tools.append(first_tool)
    # some chance of including some rocks / sticks naturally anyway
    if on_object and on_object.is_tool_helpful:
        if len(tools) > 0:
            # default to more tools if we need one
            extra_tool_count = round(4 - 3 * difficulty - rand.uniform() * difficulty)
        else:
            # default to fewer tools if we don't
            extra_tool_count = round(rand.uniform(0, 3) * difficulty)
        for _ in range(extra_tool_count):
            if rand.uniform() < get_rock_probability(difficulty):
                tools.append(Rock())
            else:
                tools.append(Stick())
    # position and add any tools
    tool_count = len(tools)
    if tool_count > 0:
        # TODO: check visibility as well
        # TODO: scale these with difficulty
        min_radius = 1.0
        max_radius = min_radius + 1.0 + difficulty * 3
        # TODO: continuous size variation for the tools
        tool_offsets = np.array([x.get_offset() for x in tools])
        tool_scales = np.ones(
            (tool_count, 3)
        )  # + np.random.uniform(low=-0.5, high=1, size=(tool_count, 1)) * difficulty
        tool_locations = get_random_ground_points(
            rand,
            locations.spawn,
            tool_count,
            world.map,
            tool_scales,
            min_radius,
            max_radius,
            tool_offsets,
            locations.island,
        )
        for tool, tool_location in zip(tools, tool_locations):
            world.add_item(attr.evolve(tool, position=tool_location))

    return world, locations, difficulty


def add_food_tree(
    rand: np.random.Generator,
    difficulty: float,
    world: NewWorld,
    point: Point2DNP,
    is_tool_food_allowed: bool,
    spawn_point: Optional[Point2DNP] = None,
) -> Tuple[float, Optional[Tool]]:
    # figure out what foods we're allowed to create
    allowed_foods = [x for x in FOODS if x.is_grown_on_trees]
    if not is_tool_food_allowed:
        allowed_foods = [x for x in allowed_foods if not x.is_tool_required]

    # figure out the exact food and what that entails
    food, difficulty = select_categorical_difficulty(allowed_foods, difficulty, rand)
    is_on_ground = False
    if food.is_found_on_ground:
        is_on_ground, difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.1)

    tree = FoodTree.build(
        tree_height=FOOD_TREE_BASE_HEIGHT,
        ground_probability=get_food_tree_ground_probability(difficulty),
        is_food_on_tree=not is_on_ground,
    )
    tool: Optional[Tool] = None
    if is_tool_food_allowed:
        tool = select_tool(food, rand, difficulty)
    if is_on_ground:
        is_opened, difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.05)
        if is_opened:
            food = food.get_opened_version()

    # figure out positioning and orientation
    food_height = tree.get_food_height(food)
    food_location = np.array([point[0], world.get_height_at(point) + food_height, point[1]])
    if spawn_point is None:
        random_vec = np.array([rand.uniform(), rand.uniform(), rand.uniform(0.001)])
        spawn_location = food_location + random_vec
    else:
        spawn_location = np.array([spawn_point[0], 0, spawn_point[1]])
    food = attr.evolve(food, position=food_location)

    add_food_and_tree(food, spawn_location, tree, world)

    return difficulty, tool


def add_food_and_tree(
    food: Food, spawn_location: Point3DNP, tree: FoodTree, world: NewWorld, is_base_allowed: bool = True
):
    # put the tree (and any required base) into the world
    tree, tree_base = tree.place(spawn_location, food.position, world.get_height_at)
    tree = world.add_item(tree)
    if tree_base is not None and is_base_allowed:
        world.add_item(tree_base)
    # attach the food + set on_object for future calculations
    if tree.is_food_on_tree:
        food = food.attached_to(tree)
    # add the food to the world
    world.add_item(food)


def add_food_tree_for_simple_task(world: NewWorld, locations: WorldLocationData):
    food = attr.evolve(CANONICAL_FOOD, position=locations.goal)
    tree = FoodTree.build(
        tree_height=FOOD_TREE_BASE_HEIGHT,
        ground_probability=0.0,
        is_food_on_tree=True,
    )
    add_food_and_tree(food, locations.spawn, tree, world, is_base_allowed=False)


def select_tool(food: Food, rand: np.random.Generator, difficulty: float) -> Optional[Tool]:
    odds, options = food.get_tool_options()
    if odds == 0:
        return None
    if odds == 1.0 or difficulty_variation(0.0, 1.0, rand, difficulty) <= odds:
        if len(options) == 1:
            return options[0]()
        return rand.choice(options)()
    return None
