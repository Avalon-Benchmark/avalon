from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import numpy as np
from scipy import stats

from avalon.datagen.errors import ImpossibleWorldError
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import DEFAULT_SAFETY_RADIUS
from avalon.datagen.world_creation.constants import ITEM_FLATTEN_RADIUS
from avalon.datagen.world_creation.constants import WEAPON_HEIGHT_OFFSET
from avalon.datagen.world_creation.entities.animals import ALL_PREY_CLASSES
from avalon.datagen.world_creation.entities.animals import Animal
from avalon.datagen.world_creation.entities.animals import Prey
from avalon.datagen.world_creation.entities.entity import Entity
from avalon.datagen.world_creation.entities.tools.placeholder import Placeholder
from avalon.datagen.world_creation.entities.tools.weapons import LargeRock
from avalon.datagen.world_creation.entities.tools.weapons import LargeStick
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.entities.tools.weapons import Weapon
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.creation import create_world_from_constraint
from avalon.datagen.world_creation.worlds.difficulty import difficulty_variation
from avalon.datagen.world_creation.worlds.difficulty import normal_distrib_range
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.difficulty import select_categorical_difficulty
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.obstacles.configure import create_inner_placeholder_solution
from avalon.datagen.world_creation.worlds.obstacles.configure import make_ring
from avalon.datagen.world_creation.worlds.types import CompositionalConstraint
from avalon.datagen.world_creation.worlds.world import World
from avalon.datagen.world_creation.worlds.world_locations import WorldLocations

TotalWeaponValue = Literal[4, 3, 2, 1]


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class HuntTaskConfig(TaskConfig):
    # how large of a world to create. Different bounds depending on if inside a pit or outside of the pit.
    # Note that it is much easier when the prey is in a pit because they cannot escape and ricochets can hit the prey
    inside_pit_dist_easy: float = 5.0
    inside_pit_dist_hard: float = 20.0
    outside_pit_dist_easy: float = 10.0
    outside_pit_dist_hard: float = 40.0
    # how big the pit should be. It will grow on harder difficulties up to the max possible in the world given the
    # pit distance defined above
    pit_size_min: float = 2.0
    # how steep to make the pit. The length is uniform random between these two values.
    pit_incline_length_min: float = 0.5
    pit_incline_length_max: float = 1.75
    # how high the pit should be. Is a bit weirdly parameterized because really the easiest thing is if the prey is just
    # baarely unable to escape, but not in such a deep pit that it makes it hard to throw stuff at it
    shallow_pit_depth_easy: float = 1.0
    shallow_pit_depth_hard: float = 2.0
    deep_pit_depth_easy: float = 3.0
    deep_pit_depth_hard: float = 2.0
    pit_depth_std_dev: float = 0.25
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
class ForceWeapons:
    # increments of 0.5
    weapon_value: Optional[TotalWeaponValue] = None
    weapon_radius: Optional[float] = None
    rock_probability: Optional[float] = None
    large_weapon_probability: Optional[float] = None
    desired_goal_distance: Optional[float] = None


@attr.define
class ForceHunt(ForceWeapons):
    prey: Optional[Type[Prey]] = None
    is_prey_in_pit: Optional[bool] = None
    is_pit_shallow: Optional[bool] = None
    pit_depth: Optional[float] = None


def generate_hunt_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: HuntTaskConfig = HuntTaskConfig(),
    _FORCED: Optional[ForceHunt] = None,
) -> None:
    world, locations, difficulty = create_hunt_obstacle(
        rand, difficulty, export_config, _forced_hunt=_FORCED, task_config=task_config
    )
    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=True)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_world(output_path, rand, world)


def create_hunt_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    _forced_hunt: Optional[ForceHunt] = None,
    task_config: HuntTaskConfig = HuntTaskConfig(),
) -> Tuple[World, WorldLocations, float]:
    if _forced_hunt is None:
        _forced_hunt = ForceHunt()

    is_prey_in_pit, difficulty = select_boolean_difficulty(difficulty, rand, _FORCED=_forced_hunt.is_prey_in_pit)
    prey_type, output_difficulty = select_categorical_difficulty(
        ALL_PREY_CLASSES, difficulty, rand, _FORCED=_forced_hunt.prey
    )

    if is_prey_in_pit:
        # this isn't exactly distance... because we reset the spawn position to be at the ledge below
        #  so we will actually end up closer
        desired_goal_distance = scale_with_difficulty(
            difficulty,
            task_config.inside_pit_dist_easy,
            task_config.inside_pit_dist_hard,
            _FORCED=_forced_hunt.desired_goal_distance,
        )
    else:
        # this WILL be how far away the prey is
        desired_goal_distance = scale_with_difficulty(
            difficulty,
            task_config.outside_pit_dist_easy,
            task_config.outside_pit_dist_hard,
            _FORCED=_forced_hunt.desired_goal_distance,
        )

    prey_offset = prey_type(position=np.array([0.0, 0.0, 0.0])).get_offset()
    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_distance, 0.5), rand, difficulty, export_config, constraint, prey_offset
    )

    pit_incline_delta = task_config.pit_incline_length_max - task_config.pit_incline_length_min
    pit_incline_distance = task_config.pit_incline_length_min + pit_incline_delta * rand.uniform()
    max_possible_outer_radius, _warnings = world.get_critical_distance(
        locations, pit_incline_distance + task_config.pit_size_min
    )

    prey = prey_type(position=locations.goal)

    weapons, output_difficulty = select_weapons(
        rand,
        difficulty,
        is_rock_necessary=is_flying_prey_present([prey]),
        rock_probability=_forced_hunt.rock_probability or task_config.rock_probability,
        weapon_value_easy=task_config.weapon_value_easy,
        weapon_value_hard=task_config.weapon_value_hard,
        weapon_value_std_dev=task_config.weapon_value_std_dev,
        large_weapon_probability_easy=task_config.large_weapon_probability_easy,
        large_weapon_probability_hard=task_config.large_weapon_probability_hard,
        _FORCED_WEAPON_VALUE=_forced_hunt.weapon_value,
        _FORCED_LARGE_WEAPON_RATE=_forced_hunt.large_weapon_probability,
    )

    weapon_radius = difficulty_variation(3.0, 6.0, rand, difficulty, _FORCED=_forced_hunt.weapon_radius)

    # if there is no pit, or the world is too small for a pit, fine, just stick some weapons and predator there
    if max_possible_outer_radius is None or not is_prey_in_pit:
        world = world.add_item(prey, reset_height_offset=prey.get_offset())
        midway = locations.spawn + (locations.goal - locations.spawn) / rand.uniform(1.5, 2.5)
        mid_point = to_2d_point(midway)
        for _ in range(len(weapons)):
            point = world.get_random_point_for_weapon_or_predator(rand, mid_point, weapon_radius, locations.island)
            if point is None:
                raise ImpossibleWorldError("No place for any weapons")
            position = np.array([point[0], 0.0, point[1]])
            world = world.add_item(Placeholder(position=position), reset_height_offset=WEAPON_HEIGHT_OFFSET)
        world = world.replace_weapon_placeholders(weapons, locations.island, ITEM_FLATTEN_RADIUS)

        new_locations = locations
    else:
        # can be increased up to max_possible_outer_radius
        # makes the task harder because then the pit is bigger
        extra_pit_size = scale_with_difficulty(difficulty, task_config.pit_size_min, max_possible_outer_radius)
        outer_radius = max_possible_outer_radius - extra_pit_size

        # Smaller walls are potentially harder because the prey can escape, but whatever
        is_pit_shallow, difficulty = select_boolean_difficulty(difficulty, rand, _FORCED=_forced_hunt.is_pit_shallow)
        if is_pit_shallow:
            pit_range = task_config.shallow_pit_depth_easy, task_config.shallow_pit_depth_hard
        else:
            pit_range = task_config.deep_pit_depth_easy, task_config.deep_pit_depth_hard
        pit_depth = -normal_distrib_range(
            pit_range[0], pit_range[1], task_config.pit_depth_std_dev, rand, difficulty, _FORCED=_forced_hunt.pit_depth
        )

        weapon_placeholders = attr.evolve(
            create_inner_placeholder_solution(
                count=len(weapons),
                offset=WEAPON_HEIGHT_OFFSET,
                randomization_dist=weapon_radius,
            ),
            inside_item_radius=ITEM_FLATTEN_RADIUS,
        )

        ring_config = make_ring(
            rand,
            difficulty,
            world,
            locations,
            constraint=constraint,
            gap_distance=0.0,
            height=pit_depth,
            traversal_width=normal_distrib_range(10.0, 1.0, 1.0, rand, difficulty),
            inner_traversal_length=pit_incline_distance,
            is_single_obstacle=True,
            inner_solution=weapon_placeholders,
            # always centered around the goal to ensure that it is a pit
            probability_of_centering_on_spawn=0.0,
            outer_traversal_length=outer_radius,
            # disables extra large circles so we can make really small pits
            max_additional_radius_multiple=1.0,
            # mostly this way so that YOU dont get stuck down there.
            # flying prey can always escape though
            is_inside_climbable=True,
        )
        world = world.add_height_obstacle(rand, ring_config, locations.island)
        world = world.replace_weapon_placeholders(weapons, locations.island, ITEM_FLATTEN_RADIUS)

        world = world.add_item(prey, reset_height_offset=prey.get_offset())

        # reset the spawn location to the bottom edge of the pit so we are right next to the prey
        weapon_positions = [x.position for x in world.items if is_weapon(x)]
        if len(weapon_positions) > 0:
            average_weapon_position = np.array(weapon_positions).mean(axis=0)
            new_locations = attr.evolve(locations, spawn=average_weapon_position)
        else:
            point = world.get_random_point_for_weapon_or_predator(
                rand,
                to_2d_point(locations.goal),
                extra_pit_size + DEFAULT_SAFETY_RADIUS + pit_incline_distance,
                locations.island,
            )
            if point is None:
                new_locations = locations
            else:
                position = np.array([point[0], 0.0, point[1]])
                new_locations = attr.evolve(locations, spawn=position)
    return world, new_locations, difficulty


def select_weapons(
    rand: np.random.Generator,
    difficulty: float,
    is_rock_necessary: bool,
    rock_probability: float,
    weapon_value_easy: float,
    weapon_value_hard: float,
    weapon_value_std_dev: float,
    large_weapon_probability_easy: float,
    large_weapon_probability_hard: float,
    _FORCED_WEAPON_VALUE: Optional[TotalWeaponValue] = None,
    _FORCED_LARGE_WEAPON_RATE: Optional[float] = None,
) -> Tuple[List[Type[Weapon]], float]:

    # Rocks/projectiles are harder to use and you lose them, so we give each weapon a "value" instead of having fixed counts
    weapon_value = int(
        normal_distrib_range(
            weapon_value_easy + 0.49, weapon_value_hard - 0.49, weapon_value_std_dev, rand, difficulty
        )
    )

    # Large weapons have the same value as their counterparts for now
    large_weapon_rate = scale_with_difficulty(
        difficulty, large_weapon_probability_easy, large_weapon_probability_hard, _FORCED=_FORCED_LARGE_WEAPON_RATE
    )

    weapons: List[Type[Weapon]] = []

    if is_rock_necessary:
        weapons.append(Rock)
        weapon_value -= Rock.WEAPON_VALUE
    weapon: Type[Weapon]
    while weapon_value > 0:
        if weapon_value < 1 or rand.uniform() < rock_probability:
            weapon = Rock
        else:
            weapon = Stick
        if rand.uniform() < large_weapon_rate:
            weapon = LargeRock if weapon is Rock else Stick
        weapon_value -= weapon.WEAPON_VALUE
        weapons.append(weapon)

    return weapons, difficulty


def is_flying_prey_present(animals: List[Animal]):
    return any(animal for animal in animals if isinstance(animal, Prey) and animal.is_flying)


def is_weapon(item: Entity):
    return isinstance(item, (Rock, Stick, Placeholder, LargeRock, LargeStick))
