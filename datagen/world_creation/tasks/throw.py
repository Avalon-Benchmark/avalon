from pathlib import Path
from typing import Final
from typing import List
from typing import Type

import numpy as np
from scipy import stats

from common.utils import first
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import Crow
from datagen.world_creation.items import Deer
from datagen.world_creation.items import Mouse
from datagen.world_creation.items import Pigeon
from datagen.world_creation.items import Prey
from datagen.world_creation.items import Rabbit
from datagen.world_creation.items import Rock
from datagen.world_creation.items import Squirrel
from datagen.world_creation.items import Turtle
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import add_offsets
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import to_2d_point

ALL_PREY_CLASSES_FOR_THROW: Final[List[Type[Prey]]] = [
    Turtle,
    Mouse,
    Rabbit,
    Squirrel,
    Deer,
    Pigeon,
    Crow,
]


def generate_throw_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:

    difficulty, food_class, locations, spawn_location, world = throw_obstacle_maker(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=False)
    world.add_spawn_and_food(rand, difficulty, spawn_location, locations.goal, food_class=food_class)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def throw_obstacle_maker(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    is_for_carry: bool = False,
):
    food_class, output_difficulty = select_categorical_difficulty(ALL_PREY_CLASSES_FOR_THROW, difficulty, rand)
    offset = food_class(entity_id=0, position=np.array([0.0, 0.0, 0.0])).get_offset()
    predator_type = None

    if is_for_carry:
        is_pit = True
    else:
        is_pit, output_difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.1)
    # TODO: this isn't exactly distance... because we reset the spawn position to be at the ledge below
    #  so we will actually end up closer
    desired_goal_dist = scale_with_difficulty(difficulty, 5.0, 20.0)
    incline_distance = 1.0
    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, desired_goal_dist / 10), rand, difficulty, export_config, None, offset
    )
    brink_distance = 0.5
    safety_radius = 1.0
    actual_goal_distance = locations.get_2d_spawn_goal_distance()
    max_possible_outer_radius = (actual_goal_distance - (safety_radius * 2 + incline_distance)) * 0.99

    min_obstacle_size = 0.5
    if max_possible_outer_radius < min_obstacle_size:
        raise WorldTooSmall(AvalonTask.THROW, min_obstacle_size, locations.get_2d_spawn_goal_distance())

    # TODO: check that this is sufficiently large: max_possible_outer_radius

    outer_radius = 0.0
    if is_pit:
        # can be increased up to max_possible_outer_radius
        # makes the task harder because then the pit is bigger
        extra_pit_size = normal_distrib_range(
            max_possible_outer_radius * 0.1, max_possible_outer_radius * 0.9, 0.5, rand, difficulty
        )
        outer_radius = max_possible_outer_radius - extra_pit_size

        # print(f"{extra_pit_size} out of {max_possible_outer_radius}")

        # disables extra large circles so we can make re;ally small pits and outcroppings
        max_additional_radius_multiple = 1.0
    else:
        # larger platforms are probably slightly easier, but doesn't make a huge difference
        max_additional_radius_multiple = 1.0
    rock_count = round(6 - difficulty * 2 * (1 + rand.uniform()))
    rocks = [Rock(entity_id=0, position=np.array([-0.5 * i, 0.0, 0.0])) for i in range(rock_count)]
    # TODO: tie this max to the tallest thing you can fall off
    depth = scale_with_difficulty(difficulty, 3.0, 10.0)
    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance=0.0,
        height=-depth,
        traversal_width=10.0 - 9.0 * difficulty * rand.uniform(),
        inner_traversal_length=incline_distance,
        is_single_obstacle=True,
        inner_solution=HeightSolution(
            inside_items=tuple(add_offsets(rocks)),
            inside_item_randomization_distance=2.0,
            inside_item_radius=first(rocks).get_offset(),
            solution_point_brink_distance=brink_distance,
        ),
        probability_of_centering_on_spawn=0.0 if is_pit else 1.0,
        outer_traversal_length=outer_radius,
        max_additional_radius_multiple=max_additional_radius_multiple,
        constraint=None,
    )
    world.add_height_obstacle(rand, ring_config, locations.island)
    # reset the spawn location to the edge of the pit so we can see the food
    average_rock_position = np.array([x.position for x in world.items if isinstance(x, Rock)]).mean(axis=0)
    # height will be reset below
    spawn_location = average_rock_position
    if predator_type is not None:
        world.add_random_predator_near_point(
            rand, predator_type, to_2d_point(locations.goal), incline_distance + safety_radius, locations.island
        )
    return difficulty, food_class, locations, spawn_location, world
