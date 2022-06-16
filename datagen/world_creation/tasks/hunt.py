from pathlib import Path
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type

import attr
import numpy as np
from scipy import stats

from datagen.world_creation.constants import DEFAULT_SAFETY_RADIUS
from datagen.world_creation.constants import WEAPON_HEIGHT_OFFSET
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import ALL_PREY_CLASSES
from datagen.world_creation.items import Animal
from datagen.world_creation.items import Entity
from datagen.world_creation.items import LargeRock
from datagen.world_creation.items import LargeStick
from datagen.world_creation.items import Placeholder
from datagen.world_creation.items import Prey
from datagen.world_creation.items import Rock
from datagen.world_creation.items import Stick
from datagen.world_creation.items import Weapon
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.task_worlds import create_world_from_constraint
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import create_inner_placeholder_solution
from datagen.world_creation.tasks.utils import difficulty_variation
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import get_rock_probability
from datagen.world_creation.tasks.utils import make_ring
from datagen.world_creation.tasks.utils import normal_distrib_range
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.tasks.utils import select_categorical_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point

TotalWeaponValue = Literal[4, 3, 2, 1]

MIN_PIT_SIZE = 2.0


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
    _FORCED: Optional[ForceHunt] = None,
) -> TaskGenerationFunctionResult:
    world, locations, difficulty = create_hunt_obstacle(rand, difficulty, export_config, _forced_hunt=_FORCED)
    world.end_height_obstacles(locations, is_accessible_from_water=True)
    world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def create_hunt_obstacle(
    rand: np.random.Generator,
    difficulty: float,
    export_config: ExportConfig,
    constraint: Optional[CompositionalConstraint] = None,
    _forced_hunt: Optional[ForceHunt] = None,
) -> Tuple[NewWorld, WorldLocationData, float]:
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
            difficulty, 5.0, 20.0, _FORCED=_forced_hunt.desired_goal_distance
        )
    else:
        # this WILL be how far away the prey is
        desired_goal_distance = scale_with_difficulty(
            difficulty, 10.0, 40.0, _FORCED=_forced_hunt.desired_goal_distance
        )

    prey_offset = prey_type(entity_id=0, position=np.array([0.0, 0.0, 0.0])).get_offset()
    world, locations = create_world_from_constraint(
        stats.norm(desired_goal_distance, 0.5), rand, difficulty, export_config, constraint, prey_offset
    )

    pit_incline_distance = 0.5 + 1.25 * rand.uniform()
    max_possible_outer_radius = world.get_critical_distance(locations, pit_incline_distance + MIN_PIT_SIZE)

    prey = prey_type(entity_id=0, position=locations.goal)

    weapons, output_difficulty = select_weapons(
        rand,
        difficulty,
        is_rock_necessary=is_flying_prey_present([prey]),
        rock_probability=_forced_hunt.rock_probability or get_rock_probability(difficulty),
        _FORCED_WEAPON_VALUE=_forced_hunt.weapon_value,
        _FORCED_LARGE_WEAPON_RATE=_forced_hunt.large_weapon_probability,
    )

    weapon_radius = difficulty_variation(3.0, 6.0, rand, difficulty, _FORCED=_forced_hunt.weapon_radius)

    # if there is no pit, or the world is too small for a pit, fine, just stick some weapons and predator there
    if max_possible_outer_radius is None or not is_prey_in_pit:
        world.add_item(prey, reset_height_offset=prey.get_offset())
        midway = locations.spawn + (locations.goal - locations.spawn) / rand.uniform(1.5, 2.5)
        mid_point = to_2d_point(midway)
        for _ in range(len(weapons)):
            point = world.get_random_point_for_weapon_or_predator(rand, mid_point, weapon_radius, locations.island)
            if point is None:
                raise ImpossibleWorldError("No place for any weapons")
            position = np.array([point[0], 0.0, point[1]])
            world.add_item(Placeholder(entity_id=0, position=position), reset_height_offset=WEAPON_HEIGHT_OFFSET)
        world.replace_weapon_placeholders(weapons)

        new_locations = locations
    else:
        # can be increased up to max_possible_outer_radius
        # makes the task harder because then the pit is bigger
        extra_pit_size = scale_with_difficulty(difficulty, MIN_PIT_SIZE, max_possible_outer_radius)
        outer_radius = max_possible_outer_radius - extra_pit_size

        # Smaller walls are potentially harder because the prey can escape, but whatever
        is_pit_shallow, difficulty = select_boolean_difficulty(difficulty, rand, _FORCED=_forced_hunt.is_pit_shallow)
        pit_range = 3.0, 2.0
        if is_pit_shallow:
            pit_range = 1.0, 2.0
        pit_depth = -normal_distrib_range(*pit_range, 0.25, rand, difficulty, _FORCED=_forced_hunt.pit_depth)

        # TODO right?
        weapon_placeholders = create_inner_placeholder_solution(
            count=len(weapons),
            offset=WEAPON_HEIGHT_OFFSET,
            randomization_dist=weapon_radius,
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
        world.add_height_obstacle(rand, ring_config, locations.island)
        world.replace_weapon_placeholders(weapons)

        world.add_item(prey, reset_height_offset=prey.get_offset())

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
    rock_probability: float = 0.5,
    _FORCED_WEAPON_VALUE: Optional[TotalWeaponValue] = None,
    _FORCED_LARGE_WEAPON_RATE: Optional[float] = None,
) -> Tuple[List[Type[Weapon]], float]:

    # Rocks/projectiles are harder to use and you lose them, so we give each weapon a "value" instead of having fixed counts
    weapon_value, difficulty = select_categorical_difficulty(
        [5, 4, 3, 2],
        difficulty,
        rand,
        _FORCED=_FORCED_WEAPON_VALUE,
    )

    # Large weapons have the same value as their counterparts for now
    large_weapon_rate = scale_with_difficulty(difficulty, 0.8, 0.2, _FORCED=_FORCED_LARGE_WEAPON_RATE)

    weapons: List[Weapon] = []

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
            # TODO: sorry, the large stick is just too big. Maybe it can be sharp instead :-P
            # weapon = LargeRock if weapon is Rock else LargeStick
            weapon = LargeRock if weapon is Rock else Stick
        weapon_value -= weapon.WEAPON_VALUE
        weapons.append(weapon)

    return weapons, difficulty


def is_flying_prey_present(animals: List[Animal]):
    return any(animal for animal in animals if isinstance(animal, Prey) and animal.is_flying)


def is_weapon(item: Entity):
    return isinstance(item, (Rock, Stick, Placeholder, LargeRock, LargeStick))
