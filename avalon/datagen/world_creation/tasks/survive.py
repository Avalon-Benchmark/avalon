from pathlib import Path
from typing import Tuple

import attr
import numpy as np

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.animals import ALL_PREY_CLASSES
from avalon.datagen.world_creation.entities.food import NON_TREE_FOODS
from avalon.datagen.world_creation.entities.tools.tool import Tool
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.tasks.eat import add_food_tree
from avalon.datagen.world_creation.types import Point3DNP
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.compositional import CompositeTaskConfig
from avalon.datagen.world_creation.worlds.compositional import create_compositional_task
from avalon.datagen.world_creation.worlds.compositional import get_random_predator
from avalon.datagen.world_creation.worlds.compositional import remove_close_predators
from avalon.datagen.world_creation.worlds.difficulty import get_rock_probability
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.world import World


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class SurviveTaskConfig(CompositeTaskConfig):
    task: AvalonTask = AvalonTask.SURVIVE
    is_big_world_mode_probability: float = 0.2
    is_hunter_mode_probability_easy: float = 0.0
    is_hunter_mode_probability_hard: float = 0.5
    # how big to make worlds where we can only hunt
    # if much larger than this, we would have to spawn too many animals for the dnsity to be reasonable
    # for any human or agent to do in a short amount of time
    normal_max_hunting_world_length: float = 350.0
    big_max_hunting_world_length: float = 600.0
    # for the condition where there is fruit and prey
    normal_max_hunting_and_gathering_world_length: float = 350.0
    big_max_hunting_and_gathering_world_length: float = 600.0
    # how much food to create in hunting and gathering mode at difficulty 0.0 and 1.0 respectively
    gathering_food_density_easy: float = 0.0003
    gathering_food_density_hard: float = 0.0001
    gathering_food_density_std_dev: float = 0.00005
    #
    hunting_predator_density_easy: float = 0.00001
    hunting_predator_density_hard: float = 0.0008
    hunting_predator_density_std_dev: float = 0.0001
    hunting_prey_density_easy: float = 0.0005
    hunting_prey_density_hard: float = 0.0001
    hunting_prey_density_std_dev: float = 0.0001
    hunting_tool_density_easy: float = 0.0005
    hunting_tool_density_hard: float = 0.0004
    hunting_tool_density_std_dev: float = 0.0001
    hunting_forage_density_easy: float = 0.0002
    hunting_forage_density_hard: float = 0.00005
    hunting_forage_density_std_dev: float = 0.00005
    #
    gathering_predator_dist_easy: float = 20.0
    gathering_predator_dist_hard: float = 5.0
    gathering_prey_dist_easy: float = 5.0
    gathering_prey_dist_hard: float = 10.0
    gathering_tool_dist_easy: float = 5.0
    gathering_tool_dist_hard: float = 10.0
    gathering_forage_food_dist_easy: float = 10.0
    gathering_forage_food_dist_hard: float = 30.0
    gathering_predator_count_distribution: Tuple[int, ...] = (0, 1, 2, 3, 4)
    gathering_prey_count_distribution: Tuple[int, ...] = (2, 1, 0, 0)
    gathering_tool_count_distribution: Tuple[int, ...] = (4, 2, 0)
    gathering_forage_food_count_distribution: Tuple[int, ...] = (2, 1, 0, 0)


def generate_survive_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: SurviveTaskConfig = SurviveTaskConfig(),
) -> None:
    is_hunter_mode_probability = scale_with_difficulty(
        difficulty, task_config.is_hunter_mode_probability_easy, task_config.is_hunter_mode_probability_hard
    )
    is_hunter_mode = rand.uniform() < is_hunter_mode_probability
    is_big_world_mode = rand.uniform() < task_config.is_big_world_mode_probability

    # how big to make worlds where we can only hunt
    # if much larger than this, we would have to spawn too many animals for the dnsity to be reasonable
    # for any human or agent to do in a short amount of time
    max_hunting_world_length = task_config.normal_max_hunting_world_length

    # for the condition where there is fruit and prey
    max_hunting_and_gathering_world_length = task_config.normal_max_hunting_and_gathering_world_length

    if is_big_world_mode:
        max_hunting_world_length = task_config.big_max_hunting_world_length
        max_hunting_and_gathering_world_length = task_config.big_max_hunting_and_gathering_world_length

    if is_hunter_mode:
        # if we are hunting, there are only animal foods. Harder because they cannot be seen so easily
        (
            output_difficulty,
            predator_density,
            prey_density,
            tool_density,
            forage_food_density,
        ) = get_open_world_densities(rand, difficulty, task_config)

        # we restrict the size of these worlds to keep the density high enough that you can actually hunt something...
        desired_goal_distance = scale_with_difficulty(difficulty, 20.0, max_hunting_world_length / 2.0)
        world, locations = create_compositional_task(
            rand,
            difficulty,
            task_config,
            export_config,
            desired_goal_distance,
            is_food_allowed=False,
            max_size_in_meters=max_hunting_world_length,
        )
        predator_count, prey_count, tool_count, forage_count = [
            get_count_from_density(world, x)
            for x in [predator_density, prey_density, tool_density, forage_food_density]
        ]

        output_difficulty, world = add_random_predators(rand, difficulty, world, predator_count)
        # noinspection PyUnusedLocal
        output_difficulty, world = add_random_prey(rand, difficulty, world, prey_count)
        world = add_random_tools(rand, world, tool_count, difficulty)
        world = add_forage_food(rand, world, forage_count)
    else:
        # if we're not hunters, they we can both hunt and gather
        # we'll put predators and prey in increasingly large radii around fruit trees
        world, locations = create_compositional_task(
            rand,
            difficulty,
            task_config,
            export_config,
            is_food_allowed=True,
            min_size_in_meters=scale_with_difficulty(difficulty, 20.0, max_hunting_and_gathering_world_length / 2.0),
            max_size_in_meters=max_hunting_and_gathering_world_length,
        )

        # this is number of food/predator categoricals generated per square meter
        food_density = normal_distrib_range(
            task_config.gathering_food_density_easy,
            task_config.gathering_food_density_hard,
            task_config.gathering_food_density_std_dev,
            rand,
            difficulty,
        )
        food_count = get_count_from_density(world, food_density)

        for i in range(food_count):
            difficulty, world = add_fruit_tree_and_animals(rand, difficulty, world, task_config)

    # since we added a bunch of predators, prevent them from being too close to our spanw
    world = remove_close_predators(world, to_2d_point(locations.spawn), task_config.min_predator_distance_from_spawn)

    export_world(output_path, rand, world)


def get_open_world_densities(
    rand: np.random.Generator, difficulty: float, task_config: SurviveTaskConfig
) -> Tuple[float, float, float, float, float]:
    # numbers calculated as how many of a thing in a 500 x 500 world (the largest)
    is_forage_food_available, difficulty = select_boolean_difficulty(difficulty, rand)

    # this is number of animals/tools per square meter
    predator_density = normal_distrib_range(
        task_config.hunting_predator_density_easy,
        task_config.hunting_predator_density_hard,
        task_config.hunting_predator_density_std_dev,
        rand,
        difficulty,
    )
    prey_density = normal_distrib_range(
        task_config.hunting_prey_density_easy,
        task_config.hunting_prey_density_hard,
        task_config.hunting_prey_density_std_dev,
        rand,
        difficulty,
    )
    tool_density = normal_distrib_range(
        task_config.hunting_tool_density_easy,
        task_config.hunting_tool_density_hard,
        task_config.hunting_tool_density_std_dev,
        rand,
        difficulty,
    )

    if is_forage_food_available:
        # this is number of food per square meter
        forage_food_density = normal_distrib_range(
            task_config.hunting_forage_density_easy,
            task_config.hunting_forage_density_hard,
            task_config.hunting_forage_density_std_dev,
            rand,
            difficulty,
        )
    else:
        forage_food_density = 0.0
    return difficulty, predator_density, prey_density, tool_density, forage_food_density


def get_count_from_density(world: World, density: float) -> int:
    square_meters = world.config.size_in_meters * world.config.size_in_meters
    count = round(square_meters * density) + 1
    return count


def add_random_predators(
    rand: np.random.Generator, difficulty: float, world: World, count: int
) -> Tuple[float, World]:
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place predator")
        output_difficulty, world = create_predator(difficulty, position, rand, world)
    return difficulty, world


def create_predator(
    difficulty: float, position: Point3DNP, rand: np.random.Generator, world: World
) -> Tuple[float, World]:
    predator, difficulty = get_random_predator(rand, difficulty)
    predator = attr.evolve(predator, position=position)
    world = world.add_item(predator, reset_height_offset=predator.get_offset())
    return difficulty, world


def add_random_tools(rand: np.random.Generator, world: World, count: int, difficulty: float) -> World:
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place tool")
        world = create_tool(position, rand, world, difficulty)
    return world


def add_forage_food(rand: np.random.Generator, world: World, count: int) -> World:
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place food")
        world = create_forage_food(position, rand, world)
    return world


def create_tool(position: Point3DNP, rand: np.random.Generator, world: World, difficulty: float) -> World:
    tool: Tool
    if rand.uniform() < get_rock_probability(difficulty):
        tool = Rock()
    else:
        tool = Stick()
    tool = attr.evolve(tool, position=position)
    return world.add_item(tool, reset_height_offset=tool.get_offset())


def create_forage_food(position: Point3DNP, rand: np.random.Generator, world: World) -> World:
    food = rand.choice(NON_TREE_FOODS)  # type: ignore[arg-type]
    food = attr.evolve(food, position=position)
    return world.add_item(food, reset_height_offset=food.get_offset())


def add_random_prey(rand: np.random.Generator, difficulty: float, world: World, count: int) -> Tuple[float, World]:
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place prey")
        output_difficulty, world = create_prey(difficulty, position, rand, world)
    return difficulty, world


def create_prey(
    difficulty: float, position: Point3DNP, rand: np.random.Generator, world: World
) -> Tuple[float, World]:
    prey, difficulty = get_random_prey(rand, difficulty)
    prey = attr.evolve(prey, position=position)
    world = world.add_item(prey, reset_height_offset=prey.get_offset())
    return difficulty, world


def add_fruit_tree_and_animals(
    rand: np.random.Generator, difficulty: float, world: World, task_config: SurviveTaskConfig
) -> Tuple[float, World]:
    tree_position = world.get_safe_point(rand, island_mask=None)
    if tree_position is None:
        raise ImpossibleWorldError("Couldn't find safe point to place fruit tree")
    output_difficulty, tool, world = add_food_tree(
        rand, difficulty, world, to_2d_point(tree_position), is_tool_food_allowed=True
    )
    sq_distances_from_tree = world.map.get_dist_sq_to(to_2d_point(tree_position))

    predator_count, output_difficulty = select_categorical_difficulty(
        task_config.gathering_predator_count_distribution, difficulty, rand
    )
    predator_dist = scale_with_difficulty(
        difficulty, task_config.gathering_predator_dist_easy, task_config.gathering_predator_dist_hard
    )
    safe_mask = world.get_safe_mask(
        sq_distances=sq_distances_from_tree, max_sq_dist=predator_dist**2, island_mask=None
    )
    for i in range(predator_count):
        position = world._get_safe_point(rand, safe_mask)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place predator")
        output_difficulty, world = create_predator(difficulty, position, rand, world)

    prey_count, output_difficulty = select_categorical_difficulty(
        task_config.gathering_prey_count_distribution, difficulty, rand
    )
    prey_dist = scale_with_difficulty(
        difficulty, task_config.gathering_prey_dist_easy, task_config.gathering_prey_dist_hard
    )
    safe_mask = world.get_safe_mask(sq_distances=sq_distances_from_tree, max_sq_dist=prey_dist**2, island_mask=None)
    for i in range(prey_count):
        position = world._get_safe_point(rand, safe_mask)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place prey")
        output_difficulty, world = create_prey(difficulty, position, rand, world)

    tool_count, output_difficulty = select_categorical_difficulty(
        task_config.gathering_tool_count_distribution, difficulty, rand
    )
    tool_dist = scale_with_difficulty(
        difficulty, task_config.gathering_tool_dist_easy, task_config.gathering_tool_dist_hard
    )
    safe_mask = world.get_safe_mask(sq_distances=sq_distances_from_tree, max_sq_dist=tool_dist**2, island_mask=None)
    for i in range(tool_count):
        position = world._get_safe_point(rand, safe_mask)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place tool")
        world = create_tool(position, rand, world, difficulty)

    forage_food_count, output_difficulty = select_categorical_difficulty(
        task_config.gathering_forage_food_count_distribution, difficulty, rand
    )
    forage_food_dist = scale_with_difficulty(
        difficulty, task_config.gathering_forage_food_dist_easy, task_config.gathering_forage_food_dist_hard
    )
    safe_mask = world.get_safe_mask(
        sq_distances=sq_distances_from_tree, max_sq_dist=forage_food_dist**2, island_mask=None
    )
    for i in range(forage_food_count):
        position = world._get_safe_point(rand, safe_mask)
        if position is None:
            raise ImpossibleWorldError("Couldn't find safe point to place food")
        world = create_forage_food(position, rand, world)

    return difficulty, world


def get_random_prey(rand: np.random.Generator, difficulty: float):
    prey_class, new_difficulty = select_categorical_difficulty(ALL_PREY_CLASSES, difficulty, rand)
    prey = prey_class(position=np.array([0.0, 0.0, 0.0]))
    return prey, new_difficulty
