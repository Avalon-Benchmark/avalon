from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

import attr
import numpy as np
from scipy import stats

from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import UP_VECTOR
from avalon.datagen.world_creation.entities.altar import Altar
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_BASE_HEIGHT
from avalon.datagen.world_creation.entities.food import CANONICAL_FOOD
from avalon.datagen.world_creation.entities.food import FOODS
from avalon.datagen.world_creation.entities.food import Food
from avalon.datagen.world_creation.entities.food import FoodTree
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.entities.utils import get_random_ground_points
from avalon.datagen.world_creation.types import Point2DNP
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import get_rock_probability
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class EatTaskConfig(TaskConfig):
    # on the easiest difficulties, food starts out sitting on an altar which is right in front of you
    # this enables even random agents to succeed at the task and probably makes it much easier for agents to learn
    # this sparse reward setting.
    # these parameters control the size of the world in both modes (altar and non-altar, ie, normal)
    altar_extra_world_goal_dist_easy: float = 0.0
    altar_extra_world_goal_dist_hard: float = 5.0
    normal_extra_world_goal_dist_easy: float = 0.0
    normal_extra_world_goal_dist_hard: float = 10.0
    # parameters that control the size of the altar:
    altar_height_default: float = 1.3
    altar_height_variation_min: float = -1.0
    altar_height_variation_max: float = 0.5
    altar_width_default: float = 1.0
    altar_width_variation_min: float = 0.0
    altar_width_variation_max: float = 1.0
    # the probability that only a single food is spawned (vs multiple) at difficulty 0.0 and 1.0 respectively
    is_single_food_probability_easy: float = 1.0
    is_single_food_probability_hard: float = 0.3
    # how much food to create at the hardest difficulty. The agent must consume all of it to succed at the level
    max_food: int = 5
    # probability of whether food is on the ground or the tree at difficulty 0.0 and 1.0 respectively
    is_on_ground_probability_easy: float = 1.0
    is_on_ground_probability_hard: float = 0.1
    # probability of whether food is already opened at difficulty 0.0 and 1.0 respectively
    is_opened_probability_easy: float = 1.0
    is_opened_probability_hard: float = 0.05
    # how far away to spawn the foods beyond the first
    food_radius_min: float = 0.75
    food_radius_max: float = 2.0
    # at most how many extra tools to spawn
    extra_tool_count_max: int = 4


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ForcedFood:
    food: Optional[Type[Food]] = None


def generate_eat_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: EatTaskConfig = EatTaskConfig(),
    _FORCED: Optional[ForcedFood] = None,
) -> None:
    world, locations, difficulty = create_eat_obstacle(
        rand, difficulty, export_config, task_config=task_config, _FORCED=_FORCED
    )
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)


def create_eat_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    _FORCED: Optional[ForcedFood] = None,
    task_config: EatTaskConfig = EatTaskConfig(),
) -> Tuple[World, WorldLocations, float]:
    if _FORCED is not None and _FORCED.food:
        food_type = _FORCED.food()
    else:
        food_type, difficulty = select_categorical_difficulty(FOODS, difficulty, rand)
    is_single_food, difficulty = select_boolean_difficulty(
        difficulty,
        rand,
        initial_prob=task_config.is_single_food_probability_easy,
        final_prob=task_config.is_single_food_probability_hard,
    )
    # some things always come in bunches
    if food_type.is_always_multiple():
        is_single_food = False

    is_extra_tree_required = False
    on_object: Optional[Union[FoodTree, Altar]] = None
    if food_type.is_found_on_ground:
        is_on_altar, difficulty = select_boolean_difficulty(difficulty, rand)
    else:
        is_on_altar = False

    if is_on_altar:
        on_object = Altar.build(
            rand.uniform(task_config.altar_height_variation_min, task_config.altar_height_variation_max) * difficulty
            + task_config.altar_height_default,
            table_dim=task_config.altar_width_default
            + difficulty * rand.uniform(task_config.altar_width_variation_min, task_config.altar_width_variation_max),
        )
        desired_food_dist = (
            rand.uniform(task_config.altar_extra_world_goal_dist_easy, task_config.altar_extra_world_goal_dist_hard)
            * difficulty
            + 0.5
        )
    else:
        if food_type.is_grown_on_trees:
            is_on_ground = False
            if food_type.is_found_on_ground:
                is_on_ground, difficulty = select_boolean_difficulty(
                    difficulty,
                    rand,
                    initial_prob=task_config.is_on_ground_probability_easy,
                    final_prob=task_config.is_on_ground_probability_hard,
                )

            is_extra_tree_required = is_on_ground
            if not is_on_ground:
                tree_height = FOOD_TREE_BASE_HEIGHT
                on_object = FoodTree.build(
                    tree_height,
                    is_food_on_tree=True,
                )
        elif food_type.is_openable:
            is_opened, difficulty = select_boolean_difficulty(
                difficulty,
                rand,
                initial_prob=task_config.is_opened_probability_easy,
                final_prob=task_config.is_opened_probability_hard,
            )
            if is_opened:
                food_type = food_type.get_opened_version()

        desired_food_dist = (
            rand.uniform(task_config.normal_extra_world_goal_dist_easy, task_config.normal_extra_world_goal_dist_hard)
            * difficulty
            + 1
        )

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
    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=True)

    tree: Optional[FoodTree] = None
    if on_object and isinstance(on_object, Altar):
        actual_distance_to_food = np.linalg.norm(locations.goal - locations.spawn)
        # orient the altar towards you if you're really close
        if actual_distance_to_food < 1.6:
            direction_to_target = (locations.goal - locations.spawn) / actual_distance_to_food
            # noinspection PyTypeChecker
            yaw = np.angle(complex(direction_to_target[2], direction_to_target[0]), deg=True) + 180
            on_object = attr.evolve(on_object, yaw=yaw)
        world = world.add_item(
            on_object.place(locations.goal - FOOD_TREE_BASE_HEIGHT * UP_VECTOR),
            reset_height_offset=on_object.get_offset(),
        )
    elif on_object and isinstance(on_object, FoodTree):
        tree = on_object
    elif is_extra_tree_required:
        tree_height = FOOD_TREE_BASE_HEIGHT
        tree = FoodTree.build(tree_height, is_food_on_tree=False)

    primary_food = food_type
    # add the tree if required, because we need to use it to figure out the spawn location of the other foods
    if tree is not None:
        # place the tree + optional base in case it would be floating.
        tree, tree_base = tree.place(locations.spawn, locations.goal, world.get_height_at)
        world = world.add_item(tree)
        tree = cast(FoodTree, world.items[-1])
        if tree_base is not None:
            world = world.add_item(tree_base)
        # attach the food + set on_object for future calculations
        if tree.is_food_on_tree:
            primary_food = primary_food.attached_to(tree)
            on_object = tree

    # add the foods
    food_count = 1
    if not is_single_food:
        max_food_delta = task_config.max_food - 2
        food_count = 2 + round(difficulty * max_food_delta * rand.uniform())

    # figure out where to position the foods
    food_locations = np.array([locations.goal])
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
                    task_config.food_radius_min,
                    task_config.food_radius_max,
                    food_hover,
                    locations.island,
                )
            else:
                other_food_locations = tree.get_food_locations(
                    rand,
                    locations.goal,
                    other_food_count,
                    world.map,
                    task_config.food_radius_min,
                    task_config.food_radius_max,
                    food_hover,
                    locations.island,
                )
        else:
            other_food_locations = on_object.get_food_locations(
                rand,
                locations.goal,
                other_food_count,
                world.map,
                task_config.food_radius_min,
                task_config.food_radius_max,
                food_hover,
                locations.island,
            )
        food_locations = np.concatenate([food_locations, other_food_locations], axis=0)
    for i, food_location in enumerate(food_locations):
        base = primary_food if i == 0 else food_type
        world = world.add_item(attr.evolve(base, position=base.additional_offset + food_location))
    # figure out what tools we need
    tools: List[Tool] = []
    first_tool = select_tool(food_type, rand, difficulty)
    if first_tool:
        tools.append(first_tool)
    # some chance of including some rocks / sticks naturally anyway
    if on_object and on_object.is_tool_helpful:
        if len(tools) > 0:
            # default to more tools if we need one
            extra_tool_count = round(
                task_config.extra_tool_count_max
                - (task_config.extra_tool_count_max - 1) * difficulty
                - rand.uniform() * difficulty
            )
        else:
            # default to fewer tools if we don't. Don't want to confuse and distract the agent on easier difficulties
            extra_tool_count = round(rand.uniform(0, 3) * difficulty)
        for _ in range(extra_tool_count):
            if rand.uniform() < get_rock_probability(difficulty):
                tools.append(Rock())
            else:
                tools.append(Stick())
    # position and add any tools
    tool_count = len(tools)
    if tool_count > 0:
        min_radius = 1.0
        max_radius = min_radius + 1.0 + difficulty * 3
        tool_offsets = np.array([x.get_offset() for x in tools])
        tool_locations = get_random_ground_points(
            rand,
            locations.spawn,
            tool_count,
            world.map,
            min_radius,
            max_radius,
            tool_offsets,
            locations.island,
        )
        for tool, tool_location in zip(tools, tool_locations):
            world = world.add_item(attr.evolve(tool, position=tool_location))

    return world, locations, difficulty


def add_food_tree(
    rand: np.random.Generator,
    difficulty: float,
    world: World,
    point: Point2DNP,
    is_tool_food_allowed: bool,
    spawn_point: Optional[Point2DNP] = None,
) -> Tuple[float, Optional[Tool], World]:
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

    world = add_food_and_tree(food, spawn_location, tree, world)

    return difficulty, tool, world


def add_food_and_tree(
    food: Food, spawn_location: Point3DNP, tree: FoodTree, world: World, is_base_allowed: bool = True
) -> World:
    # put the tree (and any required base) into the world
    tree, tree_base = tree.place(spawn_location, food.position, world.get_height_at)
    world = world.add_item(tree)
    tree = cast(FoodTree, world.items[-1])
    if tree_base is not None and is_base_allowed:
        world = world.add_item(tree_base)
    # attach the food + set on_object for future calculations
    if tree.is_food_on_tree:
        food = food.attached_to(tree)
    # add the food to the world
    return world.add_item(food)


def add_food_tree_for_simple_task(world: World, locations: WorldLocations) -> World:
    food = attr.evolve(CANONICAL_FOOD, position=locations.goal)
    tree = FoodTree.build(
        tree_height=FOOD_TREE_BASE_HEIGHT,
        is_food_on_tree=True,
    )
    return add_food_and_tree(food, locations.spawn, tree, world, is_base_allowed=False)


def select_tool(food: Food, rand: np.random.Generator, difficulty: float) -> Optional[Tool]:
    odds, options = food.get_tool_options()
    if odds == 0:
        return None
    if odds == 1.0 or difficulty_variation(0.0, 1.0, rand, difficulty) <= odds:
        if len(options) == 1:
            return options[0]()
        return rand.choice(options)()  # type: ignore
    return None
