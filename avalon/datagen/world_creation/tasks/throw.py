from pathlib import Path
from typing import Final
from typing import List
from typing import Type

import attr
import numpy as np
from scipy import stats

from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.animals import Crow
from avalon.datagen.world_creation.entities.animals import Deer
from avalon.datagen.world_creation.entities.animals import Mouse
from avalon.datagen.world_creation.entities.animals import Pigeon
from avalon.datagen.world_creation.entities.animals import Prey
from avalon.datagen.world_creation.entities.animals import Rabbit
from avalon.datagen.world_creation.entities.animals import Squirrel
from avalon.datagen.world_creation.entities.animals import Turtle
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.utils import add_offsets

# a little wiggle room so that the pit and weapons and everything fit properly
_SAFETY_RADIUS = 1.0
# must have at least this much space
_MIN_OBSTACLE_SIZE = 0.5


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class ThrowTaskConfig(TaskConfig):
    # how far away the goal should be for difficulty 0.0 and 1.0 respectively
    # this implicitly controls the size of the pit, along with the next set of parameters
    goal_dist_easy: float = 5.0
    goal_dist_hard: float = 20.0
    # in pit mode, controls how much of the goal distance ends up being used in the pit vs outside the pit
    # by using less of the distance in the pit, the pit is smaller, which is easier, since the target cannot move
    # around as much
    pit_distance_fraction_easy: float = 0.1
    pit_distance_fraction_hard: float = 0.9
    pit_distance_std_dev: float = 0.5
    # how deep to make the pit
    pit_depth_easy: float = 3.0
    pit_depth_hard: float = MIN_FALL_DISTANCE_TO_CAUSE_DAMAGE
    # how many rocks are spawned at difficulty 0.0 and 1.0 respectively
    rock_count_easy: int = 6
    rock_count_hard: int = 2
    rock_count_std_dev: float = 0.25
    # the edge of the pit will span this much distance horizontally. Controls how steep it is
    incline_distance = 1.0
    # controls how far away from the edge of the pit you start
    brink_distance = 0.5


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
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: ThrowTaskConfig = ThrowTaskConfig(),
) -> None:

    difficulty, food_class, locations, spawn_location, world = throw_obstacle_maker(
        rand, difficulty, export_config, task_config=task_config
    )
    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=False)
    world = world.add_spawn_and_food(rand, difficulty, spawn_location, locations.goal, food_class=food_class)
    export_world(output_path, rand, world)


def throw_obstacle_maker(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    is_for_carry: bool = False,
    task_config: ThrowTaskConfig = ThrowTaskConfig(),
):
    food_class, output_difficulty = select_categorical_difficulty(ALL_PREY_CLASSES_FOR_THROW, difficulty, rand)
    offset = food_class(position=np.array([0.0, 0.0, 0.0])).get_offset()
    predator_type = None

    if is_for_carry:
        is_pit = True
    else:
        is_pit, output_difficulty = select_boolean_difficulty(difficulty, rand, final_prob=0.1)
    # this isn't exactly distance... because we reset the spawn position to be at the ledge below
    # so we will actually end up closer
    desired_goal_dist = scale_with_difficulty(difficulty, task_config.goal_dist_easy, task_config.goal_dist_hard)

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, desired_goal_dist / 10), rand, difficulty, export_config, None, offset
    )

    actual_goal_distance = locations.get_2d_spawn_goal_distance()
    max_possible_outer_radius = (actual_goal_distance - (_SAFETY_RADIUS * 2 + task_config.incline_distance)) * 0.99

    if max_possible_outer_radius < _MIN_OBSTACLE_SIZE:
        raise WorldTooSmall(AvalonTask.THROW, _MIN_OBSTACLE_SIZE, locations.get_2d_spawn_goal_distance())

    outer_radius = 0.0
    if is_pit:
        # can be increased up to max_possible_outer_radius
        # makes the task harder because then the pit is bigger
        extra_pit_size = normal_distrib_range(
            max_possible_outer_radius * task_config.pit_distance_fraction_easy,
            max_possible_outer_radius * task_config.pit_distance_fraction_hard,
            task_config.pit_distance_std_dev,
            rand,
            difficulty,
        )
        outer_radius = max_possible_outer_radius - extra_pit_size

        # logger.debug(f"{extra_pit_size} out of {max_possible_outer_radius}")

        # disables extra large circles so that we can make very small pits and outcroppings
        max_additional_radius_multiple = 1.0
    else:
        # larger platforms are probably slightly easier, but doesn't make a huge difference
        max_additional_radius_multiple = 1.0

    rock_count = round(
        normal_distrib_range(
            task_config.rock_count_easy + 0.49,
            task_config.rock_count_hard - 0.49,
            task_config.rock_count_std_dev,
            rand,
            difficulty,
        )
    )
    rocks = [Rock(position=np.array([-0.5 * i, 0.0, 0.0])) for i in range(rock_count)]
    depth = scale_with_difficulty(difficulty, task_config.pit_depth_easy, task_config.pit_depth_hard)
    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance=0.0,
        height=-depth,
        traversal_width=10.0 - 9.0 * difficulty * rand.uniform(),
        inner_traversal_length=task_config.incline_distance,
        is_single_obstacle=True,
        inner_solution=HeightSolution(
            inside_items=tuple(add_offsets(rocks)),
            inside_item_randomization_distance=2.0,
            inside_item_radius=rocks[0].get_offset(),
            solution_point_brink_distance=task_config.brink_distance,
        ),
        probability_of_centering_on_spawn=0.0 if is_pit else 1.0,
        outer_traversal_length=outer_radius,
        max_additional_radius_multiple=max_additional_radius_multiple,
        constraint=None,
    )
    world = world.add_height_obstacle(rand, ring_config, locations.island)
    # reset the spawn location to the edge of the pit so we can see the food
    average_rock_position = np.array([x.position for x in world.items if isinstance(x, Rock)]).mean(axis=0)
    # height will be reset below
    spawn_location = average_rock_position
    if predator_type is not None:
        world = world.add_random_predator_near_point(
            rand,
            predator_type,
            to_2d_point(locations.goal),
            task_config.incline_distance + _SAFETY_RADIUS,
            locations.island,
        )
    return difficulty, food_class, locations, spawn_location, world
