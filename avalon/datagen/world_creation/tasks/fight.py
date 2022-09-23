from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import numpy as np
from scipy import stats

from avalon.datagen.errors import WorldTooSmall
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import CLIMBING_REQUIRED_HEIGHT
from avalon.datagen.world_creation.constants import ITEM_FLATTEN_RADIUS
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.animals import Predator
from avalon.datagen.world_creation.entities.constants import LARGEST_ANIMAL_SIZE
from avalon.datagen.world_creation.tasks.avoid import select_predator_types
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.tasks.hunt import ForceWeapons
from avalon.datagen.world_creation.tasks.hunt import select_weapons
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import create_inner_placeholder_solution
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.obstacles.height_path import HeightPath
from avalon.datagen.world_creation.worlds.obstacles.height_solution import HeightSolution
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class FightTaskConfig(TaskConfig):
    # how many predators to have at difficulty 0.0 and 1.0 respectively. Note that each one will be a different type.
    predator_count_easy: int = 1
    predator_count_hard: int = 5
    # how far the predators will be moved from their canonical location (ie, directly in the way) at difficulty 0.0
    # and 1.0 respectively.
    predator_randomization_dist_easy: float = 10.0
    predator_randomization_dist_hard: float = 3.0
    # how far away weapons will spawn. At max difficulty, they will spanw randomly between the min and max
    # At the lowest difficulty, they will be within the min radius
    weapon_radius_min: float = 2.0
    weapon_radius_max: float = 5.0
    # how far away our goal should be at difficulty 0.0 and 1.0 respectively
    # there needs to be enough room so that you dont start right next to the predators and get killed instantly
    goal_dist_easy: float = 12.0
    goal_dist_hard: float = 24.0
    # how long to make the corridor which requires you to actually fight enemies (on max difficulty)
    # on easier difficulties it will end up being a bit shorter (and may be shorter if the world is a bit too small
    # for a longer one to fit)
    path_length_max: float = 6.0
    # how wide to make the corridor in which the animals spawn at difficulty 0.0 and 1.0 respectively
    # controls how easy it is to avoid them. Is normally distributed with the mean based on the difficulty
    path_width_easy: float = 10.0
    path_width_hard: float = 2.5
    path_width_std_dev: float = 0.5
    # how tall the walls are. Always min height at difficulty 0.0, uniformly between the two at difficulty 1.0
    height_min: float = CLIMBING_REQUIRED_HEIGHT * 1.25
    height_max: float = CLIMBING_REQUIRED_HEIGHT * 2.0
    # these parameters control which weapons will be spawned
    # the probability that any given weapons will be a rock (vs a stick)
    rock_probability: float = 0.5
    # some weapons are more useful than others. They are spawned until enough "value" has been spawned
    weapon_value_easy: float = 5.0
    weapon_value_hard: float = 2.0
    weapon_value_std_dev: float = 0.5
    # there are large variants of weapons, which do more damage. They become less likely to spawn at higher difficulties
    large_weapon_probability_easy: float = 0.8
    large_weapon_probability_hard: float = 0.2


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
    task_config: FightTaskConfig = FightTaskConfig(),
    _FORCED: Optional[ForceFight] = None,
) -> None:
    world, locations, difficulty = create_fight_obstacle(
        rand, difficulty, export_config, _forced_fight=_FORCED, task_config=task_config
    )
    world, locations = world.end_height_obstacles(
        locations, is_accessible_from_water=False, is_spawn_region_climbable=False
    )
    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)


def create_fight_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    is_for_carry: bool = False,
    _forced_fight: Optional[ForceFight] = None,
    task_config: FightTaskConfig = FightTaskConfig(),
) -> Tuple[World, WorldLocations, float]:
    if _forced_fight is None:
        _forced_fight = ForceFight()

    predator_types, difficulty = select_predator_types(
        task_config.predator_count_easy,
        task_config.predator_count_hard,
        0.5,
        difficulty,
        rand,
        _FORCED=_forced_fight.predators,
    )

    # need some time and space to prepare...
    desired_goal_distance = scale_with_difficulty(
        difficulty, task_config.goal_dist_easy, task_config.goal_dist_hard, _FORCED=_forced_fight.desired_goal_distance
    )

    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_distance, 0.5), rand, difficulty, export_config, constraint
    )

    predator_randomization_radius = scale_with_difficulty(
        difficulty,
        task_config.predator_randomization_dist_easy,
        task_config.predator_randomization_dist_hard,
        _FORCED=_forced_fight.predator_randomization_radius,
    )

    weapons, difficulty = select_weapons(
        rand,
        difficulty,
        is_rock_necessary=False,
        rock_probability=_forced_fight.rock_probability or task_config.rock_probability,
        weapon_value_easy=task_config.weapon_value_easy,
        weapon_value_hard=task_config.weapon_value_hard,
        weapon_value_std_dev=task_config.weapon_value_std_dev,
        large_weapon_probability_easy=task_config.large_weapon_probability_easy,
        large_weapon_probability_hard=task_config.large_weapon_probability_hard,
        _FORCED_WEAPON_VALUE=_forced_fight.weapon_value,
        _FORCED_LARGE_WEAPON_RATE=_forced_fight.large_weapon_probability,
    )

    weapon_radius = difficulty_variation(
        task_config.weapon_radius_min,
        task_config.weapon_radius_max,
        rand,
        difficulty,
        _FORCED=_forced_fight.weapon_radius,
    )

    # if smaller than this, might not necessarily be a full wal.. Is sort of related to the constant for how
    # finely meshed the worlds are
    min_wall_size = 1.0

    max_gap_dist, _warnings = world.get_critical_distance(locations, min_wall_size)

    if max_gap_dist is None:
        raise WorldTooSmall(AvalonTask.FIGHT, min_wall_size, locations.get_2d_spawn_goal_distance())

    gap_distance = normal_distrib_range(
        0.1 * max_gap_dist,
        min([max_gap_dist * 0.5, task_config.path_length_max]),
        max_gap_dist * 0.2,
        rand,
        difficulty,
        _FORCED=_forced_fight.gap_distance,
    )

    predator_size = LARGEST_ANIMAL_SIZE
    predators = []
    for i, predator_type in enumerate(predator_types):
        dist = i + 1
        predators.append(predator_type(position=np.array([-dist, predator_size / 2.0, 0.0])))

    path_width = normal_distrib_range(
        task_config.path_width_easy,
        task_config.path_width_hard,
        task_config.path_width_std_dev,
        rand,
        difficulty,
        _FORCED=_forced_fight.path_width,
    )
    ring_config = make_ring(
        rand,
        difficulty,
        world,
        locations,
        gap_distance=gap_distance,
        constraint=constraint,
        height=difficulty_variation(task_config.height_min, task_config.height_max, rand, difficulty),
        traversal_width=path_width * 2,
        # don't want you to be able to just avoid the enemy
        is_inside_climbable=False,
        is_outside_climbable=False,
        # path
        dual_solution=HeightSolution(
            paths=(
                HeightPath(
                    is_path_climbable=True,
                    is_outside_edge_unclimbable=True,
                    is_path_restricted_to_land=True,
                    is_detailed=path_width < 3.0,
                    width=path_width,
                ),
            ),
            solution_point_brink_distance=1.0,
        ),
        # "closer to spawn"
        inner_solution=attr.evolve(
            create_inner_placeholder_solution(
                count=len(weapons),
                offset=0,  # WEAPON_HEIGHT_OFFSET,
                randomization_dist=weapon_radius,
            ),
            inside_item_radius=ITEM_FLATTEN_RADIUS,
        ),
        # "further from spawn"
        outer_solution=HeightSolution(
            outside_items=tuple(predators),
            outside_item_randomization_distance=predator_randomization_radius,
            # prevents stuff from spawning this close to a predator
            outside_item_radius=predator_size,
        ),
        traversal_noise_interpolation_multiple=4.0,
        probability_of_centering_on_spawn=0.0 if is_for_carry else 0.5,
    )

    world = world.add_height_obstacle(rand, ring_config, locations.island)
    world = world.replace_weapon_placeholders(weapons, locations.island, 0.0 if is_for_carry else ITEM_FLATTEN_RADIUS)

    return world, locations, difficulty
