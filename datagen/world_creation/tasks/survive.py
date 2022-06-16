from pathlib import Path
from typing import Tuple

import attr
import numpy as np

from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import NON_TREE_FOODS
from datagen.world_creation.items import Rock
from datagen.world_creation.items import Stick
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional import MIN_PREDATOR_DISTANCE_FROM_SPAWN
from datagen.world_creation.tasks.compositional import create_compositional_task
from datagen.world_creation.tasks.compositional import get_random_predator
from datagen.world_creation.tasks.compositional import get_random_prey
from datagen.world_creation.tasks.compositional import remove_close_predators
from datagen.world_creation.tasks.eat import add_food_tree
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import get_rock_probability
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point


def generate_survive_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    is_hunter_mode_probability = scale_with_difficulty(difficulty, 0.0, 0.5)
    is_hunter_mode = rand.uniform() < is_hunter_mode_probability
    is_big_world_mode_probability = 0.2
    is_big_world_mode = rand.uniform() < is_big_world_mode_probability

    # how big to make worlds where we can only hunt
    # if much larger than this, we would have to spawn too many animals for the dnsity to be reasonable
    # for any human or agent to do in a short amount of time
    max_hunting_world_length = 350.0

    # for the condition where there is fruit and prey
    max_hunting_and_gathering_world_length = 350.0

    if is_big_world_mode:
        max_hunting_world_length = 600
        max_hunting_and_gathering_world_length = 600

    if is_hunter_mode:
        # if we are hunting, there are only animal foods. Harder because they cannot be seen so easily
        (
            output_difficulty,
            predator_density,
            prey_density,
            tool_density,
            forage_food_density,
        ) = get_open_world_densities(rand, difficulty)

        # we restrict the size of these worlds to keep the density high enough that you can actually hunt something...
        desired_goal_distance = scale_with_difficulty(difficulty, 20.0, max_hunting_world_length / 2.0)
        world, locations = create_compositional_task(
            rand,
            difficulty,
            AvalonTask.SURVIVE,
            export_config,
            desired_goal_distance,
            is_food_allowed=False,
            max_size_in_meters=max_hunting_world_length,
        )
        predator_count, prey_count, tool_count, forage_count = [
            get_count_from_density(world, x)
            for x in [predator_density, prey_density, tool_density, forage_food_density]
        ]

        output_difficulty = add_random_predators(rand, difficulty, world, predator_count)
        # noinspection PyUnusedLocal
        output_difficulty = add_random_prey(rand, difficulty, world, prey_count)
        add_random_tools(rand, world, tool_count, difficulty)
        add_forage_food(rand, world, forage_count)
    else:
        # if we're not hunters, they we can both hunt and gather
        # we'll put predators and prey in increasingly large radii around fruit trees
        world, locations = create_compositional_task(
            rand,
            difficulty,
            AvalonTask.SURVIVE,
            export_config,
            is_food_allowed=True,
            min_size_in_meters=scale_with_difficulty(difficulty, 20.0, max_hunting_and_gathering_world_length / 2.0),
            max_size_in_meters=max_hunting_and_gathering_world_length,
        )

        # this is number of food/predator categoricals generated per square meter
        food_density = normal_distrib_range(0.0003, 0.0001, 0.00005, rand, difficulty)
        food_count = get_count_from_density(world, food_density)

        for i in range(food_count):
            difficulty = add_fruit_tree_and_animals(rand, difficulty, world, locations)

    # since we added a bunch of predators, prevent them from being too close to our spanw
    remove_close_predators(world, to_2d_point(locations.spawn), MIN_PREDATOR_DISTANCE_FROM_SPAWN)

    export_skill_world(output_path, rand, world)

    # TODO tune hit points
    return TaskGenerationFunctionResult(starting_hit_points=1.0)


def get_open_world_densities(rand: np.random.Generator, difficulty: float) -> Tuple[float, float, float, float, float]:
    # numbers calculated as how many of a thing in a 500 x 500 world (the largest)
    is_forage_food_available, difficulty = select_boolean_difficulty(difficulty, rand)

    # this is number of animals/tools per square meter
    predator_density = normal_distrib_range(0.00001, 0.0008, 0.0001, rand, difficulty)
    prey_density = normal_distrib_range(0.0005, 0.0001, 0.0001, rand, difficulty)
    tool_density = normal_distrib_range(0.0005, 0.0004, 0.0001, rand, difficulty)

    if is_forage_food_available:
        # this is number of food per square meter
        forage_food_density = normal_distrib_range(0.0002, 0.00005, 0.00005, rand, difficulty)
    else:
        forage_food_density = 0.0
    return difficulty, predator_density, prey_density, tool_density, forage_food_density


def get_count_from_density(world: NewWorld, density: float) -> int:
    square_meters = world.config.size_in_meters * world.config.size_in_meters
    count = round(square_meters * density) + 1
    return count


def add_random_predators(rand: np.random.Generator, difficulty: float, world: NewWorld, count: int) -> float:
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        output_difficulty = create_predator(difficulty, position, rand, world)
    return difficulty


def create_predator(difficulty, position, rand, world):
    predator, difficulty = get_random_predator(rand, difficulty)
    predator = attr.evolve(predator, position=position)
    world.add_item(predator, reset_height_offset=predator.get_offset())
    return difficulty


def add_random_tools(rand: np.random.Generator, world: NewWorld, count: int, difficulty: float):
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        create_tool(position, rand, world, difficulty)


def add_forage_food(rand: np.random.Generator, world: NewWorld, count: int):
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        create_forage_food(position, rand, world)


def create_tool(position, rand, world, difficulty):
    tool = Rock() if rand.uniform() < get_rock_probability(difficulty) else Stick()
    tool = attr.evolve(tool, position=position)
    world.add_item(tool, reset_height_offset=tool.get_offset())


def create_forage_food(position, rand, world):
    food = rand.choice(NON_TREE_FOODS)
    food = attr.evolve(food, position=position)
    world.add_item(food, reset_height_offset=food.get_offset())


def add_random_prey(rand: np.random.Generator, difficulty: float, world: NewWorld, count: int) -> float:
    for i in range(count):
        position = world.get_safe_point(rand, island_mask=None)
        output_difficulty = create_prey(difficulty, position, rand, world)
    return difficulty


def create_prey(difficulty, position, rand, world):
    prey, difficulty = get_random_prey(rand, difficulty)
    prey = attr.evolve(prey, position=position)
    world.add_item(prey, reset_height_offset=prey.get_offset())
    return difficulty


def add_fruit_tree_and_animals(
    rand: np.random.Generator, difficulty: float, world: NewWorld, locations: WorldLocationData
) -> float:
    tree_position = world.get_safe_point(rand, island_mask=None)
    output_difficulty, tool = add_food_tree(
        rand, difficulty, world, to_2d_point(tree_position), is_tool_food_allowed=True
    )
    sq_distances_from_tree = world.map.get_dist_sq_to(to_2d_point(tree_position))

    predator_count, output_difficulty = select_categorical_difficulty([0, 1, 2, 3, 4], difficulty, rand)
    predator_dist = scale_with_difficulty(difficulty, 20.0, 5.0)
    safe_mask = world.get_safe_mask(
        sq_distances=sq_distances_from_tree, max_sq_dist=predator_dist ** 2, island_mask=None
    )
    for i in range(predator_count):
        position = world._get_safe_point(rand, safe_mask)
        output_difficulty = create_predator(difficulty, position, rand, world)

    prey_count, output_difficulty = select_categorical_difficulty([2, 1, 0, 0], difficulty, rand)
    prey_dist = scale_with_difficulty(difficulty, 5.0, 10.0)
    safe_mask = world.get_safe_mask(sq_distances=sq_distances_from_tree, max_sq_dist=prey_dist ** 2, island_mask=None)
    for i in range(prey_count):
        position = world._get_safe_point(rand, safe_mask)
        output_difficulty = create_prey(difficulty, position, rand, world)

    tool_count, output_difficulty = select_categorical_difficulty([4, 2, 0], difficulty, rand)
    tool_dist = scale_with_difficulty(difficulty, 5.0, 10.0)
    safe_mask = world.get_safe_mask(sq_distances=sq_distances_from_tree, max_sq_dist=tool_dist ** 2, island_mask=None)
    for i in range(tool_count):
        position = world._get_safe_point(rand, safe_mask)
        create_tool(position, rand, world, difficulty)

    forage_food_count, output_difficulty = select_categorical_difficulty([2, 1, 0, 0], difficulty, rand)
    forage_food_dist = scale_with_difficulty(difficulty, 10.0, 30.0)
    safe_mask = world.get_safe_mask(
        sq_distances=sq_distances_from_tree, max_sq_dist=forage_food_dist ** 2, island_mask=None
    )
    for i in range(forage_food_count):
        position = world._get_safe_point(rand, safe_mask)
        create_forage_food(position, rand, world)

    return difficulty
