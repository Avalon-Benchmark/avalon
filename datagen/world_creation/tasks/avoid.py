from pathlib import Path
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import numpy as np
from scipy import stats

from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import ALL_PREDATOR_CLASSES
from datagen.world_creation.items import Predator
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.new_world import normalized
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point


def generate_avoid_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_avoid_obstacle(rand, difficulty, export_config)
    world.end_height_obstacles(locations, is_accessible_from_water=True)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_avoid_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:
    predator_types, difficulty = select_predator_types(1, 5, difficulty, rand)

    _AVOID_MAX_DIST = 52.0
    _AVOID_MIN_DIST = 13.0
    current_task_max = _AVOID_MAX_DIST - (_AVOID_MAX_DIST - _AVOID_MIN_DIST) * difficulty
    current_task_delta = current_task_max - _AVOID_MIN_DIST
    desired_goal_dist = _AVOID_MIN_DIST + current_task_delta * rand.uniform()

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_dist, 0.5), rand, difficulty, export_config, constraint
    )

    # how far off the path we create the predator. Closer is harder
    predator_distance_mean = scale_with_difficulty(difficulty, 10.0, 1.0)

    food_dir_2d = normalized(to_2d_point(locations.goal) - to_2d_point(locations.spawn))
    food_dist_2d = np.linalg.norm(to_2d_point(locations.goal) - to_2d_point(locations.spawn))

    # make the player spawn radius a wee bit bigger for this task
    spawn_safety_radius = scale_with_difficulty(difficulty, 11.0, 8.0)
    spawn_dist_sq = world.map.get_dist_sq_to(to_2d_point(locations.spawn))
    nearby = spawn_dist_sq < spawn_safety_radius ** 2
    world.is_detail_important[nearby] = True

    for predator_type in predator_types:
        # figure out the randomization parameters for difficulty for this predator
        predator_dist = rand.normal(predator_distance_mean, 1.0)
        # how far along the line between food and spawn. Closer is harder
        predator_path_location = 0.5 + difficulty_variation(0.4, -0.4, rand, difficulty)

        # figure out where to place the predator
        predator_offset_dir_2d = np.array([-food_dir_2d[1], food_dir_2d[0]])
        if rand.random() < 0.5:
            predator_offset_dir_2d *= -1

        path_point_2d = to_2d_point(locations.spawn) + (food_dir_2d * food_dist_2d * predator_path_location)
        predator_point_2d = path_point_2d + predator_dist * predator_offset_dir_2d

        # actually add the predator
        world.add_random_predator_near_point(rand, predator_type, predator_point_2d, 1, locations.island)

    return world, locations, difficulty


def select_predator_types(
    min_count: int,
    max_count: int,
    difficulty: float,
    rand: np.random.Generator,
    exclude: Iterable[Type[Predator]] = tuple(),
    _FORCED: Optional[Iterable[Type[Predator]]] = None,
) -> Tuple[List[Type[Predator]], float]:
    if _FORCED:
        # ensure hardest is last
        _FORCED = list(sorted(_FORCED, key=ALL_PREDATOR_CLASSES.index))

    acceptable_predators = [pred for pred in ALL_PREDATOR_CLASSES if pred not in exclude]

    max_predator_type, difficulty = select_categorical_difficulty(
        acceptable_predators, difficulty, rand, _FORCED=_FORCED[-1] if _FORCED else None
    )
    predator_count = round(normal_distrib_range(min_count - 0.45, max_count + 0.45, 0.5, rand, difficulty))
    if _FORCED:
        predator_count = len(_FORCED)

    if _FORCED:
        return _FORCED, difficulty

    acceptable_predators = acceptable_predators[: acceptable_predators.index(max_predator_type) + 1]
    predator_types = [max_predator_type]
    for i in range(predator_count - 1):
        p_type, _ = select_categorical_difficulty(acceptable_predators, difficulty, rand)
        predator_types.append(p_type)
    return predator_types, difficulty
