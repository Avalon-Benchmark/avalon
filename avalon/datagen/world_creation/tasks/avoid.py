from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import numpy as np
from scipy import stats

from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.entities.animals import ALL_PREDATOR_CLASSES
from avalon.datagen.world_creation.entities.animals import Predator
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.utils import normalized
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class AvoidTaskConfig(TaskConfig):
    # how many predators to spawn at difficulty 0.0 and 1.0 respectively
    # note that each predator may be a different type
    predator_count_easy: int = 1
    predator_count_hard: int = 5
    # the number of predators at a difficulty is not exact, but rather, is sampled from a normal distribution with
    # the mean being proportional to the difficulty, and the following standard deviation
    predator_count_std_dev: float = 0.5
    # the max and min dist between the spawn and goal (ie, how far the agents has to walk to complete the task)
    # larger worlds make more space for us to place the obstacles and predators
    min_dist: float = 13.0
    max_dist: float = 52.0
    # positioning for predators works by figuring out how far away they are, along the path (predator_path_fraction),
    # and then how far off the path they should be spawned (predator_distance_off_path)
    # predator_path_fraction is how far along the path the predators will spawn
    # 0.0 means "they will spawn right on top of you" and 1.0 means "they will spawn right on top of the food"
    # it is harder when they spawn near you
    predator_path_fraction_easy: float = 0.9
    predator_path_fraction_hard: float = 0.1
    predator_path_fraction_std_dev: float = 0.1
    # how far predators start to the side of the path at difficulty 0.0 and 1.0 respectively
    predator_distance_off_path_easy: float = 10.0
    predator_distance_off_path_hard: float = 1.0
    predator_distance_off_path_std_dev: float = 1.0
    # how large the "safety radius" around the player will be. This prevents predators from spawning that close
    # if this is too small, you will probably die before you can get your bearings
    safety_radius_easy: float = 11.0
    safety_radius_hard: float = 8.0


def generate_avoid_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: AvoidTaskConfig = AvoidTaskConfig(),
) -> None:
    world, locations, difficulty = create_avoid_obstacle(rand, difficulty, export_config, task_config=task_config)
    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=True)
    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)


def create_avoid_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    task_config: AvoidTaskConfig = AvoidTaskConfig(),
) -> Tuple[World, WorldLocations, float]:
    predator_types, difficulty = select_predator_types(
        task_config.predator_count_easy,
        task_config.predator_count_hard,
        task_config.predator_count_std_dev,
        difficulty,
        rand,
    )

    current_task_max = task_config.max_dist - (task_config.max_dist - task_config.min_dist) * difficulty
    current_task_delta = current_task_max - task_config.min_dist
    desired_goal_dist = task_config.min_dist + current_task_delta * rand.uniform()

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    # how far off the path we create the predator. Closer is harder
    predator_distance_mean = scale_with_difficulty(
        difficulty, task_config.predator_distance_off_path_easy, task_config.predator_distance_off_path_hard
    )

    food_dir_2d = normalized(to_2d_point(locations.goal) - to_2d_point(locations.spawn))
    food_dist_2d = np.linalg.norm(to_2d_point(locations.goal) - to_2d_point(locations.spawn))

    # make the player spawn radius a wee bit bigger for this task
    spawn_safety_radius = scale_with_difficulty(
        difficulty, task_config.safety_radius_easy, task_config.safety_radius_hard
    )
    spawn_dist_sq = world.map.get_dist_sq_to(to_2d_point(locations.spawn))
    nearby = spawn_dist_sq < spawn_safety_radius**2
    is_detail_important_new = world.is_detail_important.copy()
    is_detail_important_new[nearby] = True
    world = attr.evolve(world, is_detail_important=is_detail_important_new)

    for predator_type in predator_types:
        # figure out the randomization parameters for difficulty for this predator
        predator_dist = rand.normal(predator_distance_mean, task_config.predator_distance_off_path_std_dev)
        # how far along the line between food and spawn. Closer is harder
        predator_path_location = normal_distrib_range(
            task_config.predator_path_fraction_easy,
            task_config.predator_path_fraction_hard,
            task_config.predator_path_fraction_std_dev,
            rand,
            difficulty,
        )

        # figure out where to place the predator
        predator_offset_dir_2d = np.array([-food_dir_2d[1], food_dir_2d[0]])
        if rand.random() < 0.5:
            predator_offset_dir_2d *= -1

        path_point_2d = to_2d_point(locations.spawn) + (food_dir_2d * food_dist_2d * predator_path_location)
        predator_point_2d = path_point_2d + predator_dist * predator_offset_dir_2d

        # actually add the predator
        world = world.add_random_predator_near_point(rand, predator_type, predator_point_2d, 1, locations.island)

    return world, locations, difficulty


def select_predator_types(
    min_count: int,
    max_count: int,
    count_std_dev: float,
    difficulty: float,
    rand: np.random.Generator,
    exclude: Iterable[Type[Predator]] = tuple(),
    _FORCED: Optional[Iterable[Type[Predator]]] = None,
) -> Tuple[List[Type[Predator]], float]:
    if _FORCED:
        # ensure hardest is last
        _FORCED_LIST = list(sorted(_FORCED, key=ALL_PREDATOR_CLASSES.index))
        return _FORCED_LIST, difficulty

    acceptable_predators = [pred for pred in ALL_PREDATOR_CLASSES if pred not in exclude]

    max_predator_type, difficulty = select_categorical_difficulty(acceptable_predators, difficulty, rand, _FORCED=None)
    predator_count = round(normal_distrib_range(min_count - 0.45, max_count + 0.45, count_std_dev, rand, difficulty))

    acceptable_predators = acceptable_predators[: acceptable_predators.index(max_predator_type) + 1]
    predator_types = [max_predator_type]
    for i in range(predator_count - 1):
        p_type, _ = select_categorical_difficulty(acceptable_predators, difficulty, rand)
        predator_types.append(p_type)
    return predator_types, difficulty
