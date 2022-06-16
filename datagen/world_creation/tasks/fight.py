from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import numpy as np
from scipy import stats

from datagen.world_creation.constants import CLIMBING_REQUIRED_HEIGHT
from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import LARGEST_ANIMAL_SIZE
from datagen.world_creation.items import Predator
from datagen.world_creation.new_world import HeightPath
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.avoid import select_predator_types
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.hunt import ForceWeapons
from datagen.world_creation.tasks.hunt import select_weapons
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import create_inner_placeholder_solution
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import get_rock_probability
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import WorldTooSmall
from datagen.world_creation.world_location_data import WorldLocationData


@attr.define
class ForceFight(ForceWeapons):
    predators: Optional[Tuple[Type[Predator], ...]] = None
    predator_randomization_radius: Optional[float] = None
    gap_distance: Optional[float] = None
    path_width: Optional[float] = None
    desired_goal_distance: Optional[float] = None


def generate_fight_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    _FORCED: Optional[ForceFight] = None,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_fight_obstacle(rand, difficulty, export_config, _forced_fight=_FORCED)
    world.end_height_obstacles(locations, is_accessible_from_water=False, is_spawn_region_climbable=False)
    add_food_tree_for_simple_task(world, locations)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_fight_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    is_for_carry: bool = False,
    _forced_fight: Optional[ForceFight] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:
    if _forced_fight is None:
        _forced_fight = ForceFight()

    # prevents stuff from spawning this close to a predator
    predator_size = LARGEST_ANIMAL_SIZE

    # TODO: actually, allowing all animals
    # predator_types, difficulty = select_predator_types(
    #     1, 4, difficulty, rand, exclude=(Hippo, Bear), _FORCED=_forced_fight.predators
    # )
    predator_types, difficulty = select_predator_types(1, 5, difficulty, rand, _FORCED=_forced_fight.predators)

    # need some time and space to prepare...
    desired_goal_distance = scale_with_difficulty(difficulty, 12.0, 24.0, _FORCED=_forced_fight.desired_goal_distance)

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_distance, 0.5), rand, difficulty, export_config, constraint
    )

    predator_randomization_radius = scale_with_difficulty(
        difficulty, 10.0, 3.0, _FORCED=_forced_fight.predator_randomization_radius
    )

    weapons, difficulty = select_weapons(
        rand,
        difficulty,
        is_rock_necessary=False,
        rock_probability=_forced_fight.rock_probability or get_rock_probability(difficulty),
        _FORCED_WEAPON_VALUE=_forced_fight.weapon_value,
        _FORCED_LARGE_WEAPON_RATE=_forced_fight.large_weapon_probability,
    )

    weapon_radius = difficulty_variation(2.0, 5.0, rand, difficulty, _FORCED=_forced_fight.weapon_radius)

    min_wall_size = 1.0
    max_gap_dist = world.get_critical_distance(locations, min_wall_size)

    if max_gap_dist is None:
        raise WorldTooSmall(AvalonTask.FIGHT, min_wall_size, locations.get_2d_spawn_goal_distance())

    gap_distance = normal_distrib_range(
        0.1 * max_gap_dist,
        min([max_gap_dist * 0.5, 6.0]),
        max_gap_dist * 0.2,
        rand,
        difficulty,
        _FORCED=_forced_fight.gap_distance,
    )

    predator_distance = 6.0
    predators = []
    for i, predator_type in enumerate(predator_types):
        dist = predator_distance * ((i + 1) / predator_distance)
        predators.append(predator_type(entity_id=0, position=np.array([-dist, predator_size / 2.0, 0.0])))

    path_width = normal_distrib_range(10.0, 3.0, 1.0, rand, difficulty, _FORCED=_forced_fight.path_width)
    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance=gap_distance,
        constraint=constraint,
        height=CLIMBING_REQUIRED_HEIGHT + CLIMBING_REQUIRED_HEIGHT * 0.5 * rand.uniform(),
        traversal_width=path_width * 2,
        # don't want you to be able to just avoid the enemy
        is_inside_climbable=False,
        is_outside_climbable=False,
        # path
        dual_solution=HeightSolution(
            paths=(
                HeightPath(
                    is_path_restricted_to_land=True,
                    is_detailed=path_width < 3.0,
                    width=path_width,
                ),
            ),
            solution_point_brink_distance=1.0,
        ),
        # "closer to spawn"
        inner_solution=create_inner_placeholder_solution(
            count=len(weapons),
            offset=0,  # WEAPON_HEIGHT_OFFSET,
            randomization_dist=weapon_radius,
        ),
        # "further from spawn"
        outer_solution=HeightSolution(
            outside_items=tuple(predators),
            outside_item_randomization_distance=predator_randomization_radius,
            outside_item_radius=predator_size,
        ),
        traversal_noise_interpolation_multiple=4.0,
        probability_of_centering_on_spawn=0.0 if is_for_carry else 0.5,
    )

    world.add_height_obstacle(rand, ring_config, locations.island)
    world.replace_weapon_placeholders(weapons)

    return world, locations, difficulty
