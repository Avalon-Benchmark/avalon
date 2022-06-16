import math
from pathlib import Path
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar

import attr
import numpy as np
from scipy import stats

from common.log_utils import logger
from common.utils import first
from contrib.serialization import Serializable
from datagen.world_creation.biome_map import make_fast_biome
from datagen.world_creation.biome_map import signed_line_distance
from datagen.world_creation.constants import DEFAULT_SAFETY_RADIUS
from datagen.world_creation.heightmap import HeightMode
from datagen.world_creation.items import Placeholder
from datagen.world_creation.items import Tool
from datagen.world_creation.new_world import EdgeConfig
from datagen.world_creation.new_world import HeightObstacle
from datagen.world_creation.new_world import HeightSolution
from datagen.world_creation.new_world import NewWorld
from datagen.world_creation.new_world import RingObstacleConfig
from datagen.world_creation.tasks.biome_settings import make_natural_biomes
from datagen.world_creation.tasks.compositional_types import CompositionalConstraint
from datagen.world_creation.tasks.constants import DEBUG_USE_FIRST_CATEGORICAL
from datagen.world_creation.tasks.constants import DEBUG_USE_TRUE_BOOLEAN
from datagen.world_creation.utils import ImpossibleWorldError
from datagen.world_creation.utils import plot_value_grid
from datagen.world_creation.world_location_data import WorldLocationData
from datagen.world_creation.world_location_data import to_2d_point


def export_skill_world(
    output_path: Path,
    rand: np.random.Generator,
    world: NewWorld,
):
    if world.export_config.is_biome_fast:
        biome_config = attr.evolve(world.biome_config, is_scenery_added=False)
        biome_map = make_fast_biome(world.map, biome_config)
    else:
        biome_map = make_natural_biomes(rand, world)

    # # TODO: remove
    # world.effective_height_map = world.map
    #
    # if is_scenery_added:
    #     world.add_scenery(rand, biome_map)

    terrain = world.generate_terrain(rand, biome_map)

    output_path.mkdir(parents=True, exist_ok=True)
    world.export(terrain, output_path)


def get_random_position_along_path(
    visible_locations: np.ndarray, start: np.ndarray, end: np.ndarray, difficulty: float, rand: np.random.Generator
) -> np.ndarray:
    """Difficulty scales with how far away the point is from the straight line path between start and end"""
    path_length = np.linalg.norm(start - end)
    desired_distance = rand.uniform() * difficulty * path_length * 2
    target_location_distribution = stats.norm(desired_distance, 0.5)
    start_point = (start[0], start[2])
    end_point = (end[0], end[2])
    location_weights = np.array(
        [
            target_location_distribution.pdf(signed_line_distance((x[0], x[2]), start_point, end_point, path_length))
            for x in visible_locations
        ]
    )
    location_weights /= location_weights.sum()
    return rand.choice(visible_locations, p=location_weights)


def get_difficulty_based_value(
    difficulty: float, min_val: float, max_val: float, variability: float, rand: np.random.Generator
) -> float:
    total_delta = max_val - min_val
    delta = variability * total_delta
    remainder = total_delta - delta
    return min_val + (remainder * difficulty) + (rand.uniform() * delta)


def make_ring(
    rand: np.random.Generator,
    difficulty: float,
    world: NewWorld,
    locations: WorldLocationData,
    gap_distance: float,
    height: float,
    traversal_width: float,
    constraint: Optional[CompositionalConstraint],
    is_inside_climbable: bool = True,
    is_outside_climbable: bool = False,
    inner_solution: Optional[HeightSolution] = None,
    outer_solution: Optional[HeightSolution] = None,
    dual_solution: Optional[HeightSolution] = None,
    inner_traversal_length: float = 0.0,
    outer_traversal_length: float = 0.0,
    safety_radius: float = DEFAULT_SAFETY_RADIUS,
    traversal_noise_interpolation_multiple=2.0,
    is_single_obstacle: bool = False,
    probability_of_centering_on_spawn: float = 0.5,
    max_additional_radius_multiple: float = 2.0,
    height_mode: HeightMode = HeightMode.RELATIVE,
    extra_safety_radius: float = 0.0,
    expansion_meters: float = 0.0,
    detail_radius: float = 20.0,
    traversal_difficulty_meters: float = 10.0,
) -> RingObstacleConfig:
    is_debugging = False

    if constraint is None:
        # should only be centered on you half the time
        is_centered_on_spawn = rand.uniform() < probability_of_centering_on_spawn
        if is_centered_on_spawn:
            center_point = to_2d_point(locations.spawn)
            other_point = to_2d_point(locations.goal)
        else:
            if is_debugging:
                logger.debug("Is reversed")
            center_point = to_2d_point(locations.goal)
            other_point = to_2d_point(locations.spawn)

            if is_single_obstacle:
                height *= -1
            else:
                # swap these because this is not the canonical ordering
                is_inside_climbable, is_outside_climbable = is_outside_climbable, is_inside_climbable
                inner_solution, outer_solution = outer_solution, inner_solution
                inner_traversal_length, outer_traversal_length = outer_traversal_length, inner_traversal_length

            if inner_solution is not None:
                inner_solution = inner_solution.reverse()

            if outer_solution is not None:
                outer_solution = outer_solution.reverse()

            if dual_solution is not None:
                dual_solution = dual_solution.reverse()
    else:
        center_point = to_2d_point(constraint.center)
        other_point = to_2d_point(locations.goal)

        if constraint.is_height_inverted:
            height *= -1

    middle_point = (center_point + other_point) / 2.0

    if constraint is None:
        # add some variation for the radius of the circle
        radius_scale = rand.uniform(1, max_additional_radius_multiple)
        center_point = middle_point + (center_point - middle_point) * radius_scale
    else:
        radius_scale = 1.0

    # and find a randomized traversal point that would actually work
    spawn_radius = np.linalg.norm(center_point - to_2d_point(locations.spawn))
    food_radius = np.linalg.norm(center_point - to_2d_point(locations.goal))
    min_traversal_radius = (
        min([spawn_radius, food_radius]) + safety_radius + inner_traversal_length + extra_safety_radius
    )
    max_traversal_radius = max([spawn_radius, food_radius]) - (
        safety_radius + outer_traversal_length + gap_distance + extra_safety_radius
    )
    radius_diff = max_traversal_radius - min_traversal_radius
    if radius_diff < 0:
        raise ImpossibleWorldError("Too small to make the requested ring")
    if is_debugging:
        print(f"Radius diff is {radius_diff}")
        print(spawn_radius)
        print(food_radius)
        print(min_traversal_radius)
        print(max_traversal_radius)
        print(safety_radius)
        print(inner_traversal_length)
        print(radius_scale)
    min_selectable_radius = min_traversal_radius + radius_diff * 0.3
    max_selectable_radius = max_traversal_radius - radius_diff * 0.3
    r_squared = world.map.get_dist_sq_to(center_point)
    acceptable_radius_mask = np.logical_and(
        r_squared < max_selectable_radius ** 2, r_squared > min_selectable_radius ** 2
    )
    if is_debugging:
        plot_value_grid(r_squared < max_selectable_radius ** 2, "Less than max radius")
        plot_value_grid(r_squared > min_selectable_radius ** 2, "Greater than min radius")
        plot_value_grid(acceptable_radius_mask, "Acceptable radius mask")

    max_dist = 2.0 + traversal_difficulty_meters * difficulty

    temp = middle_point - center_point
    middle_theta = np.arctan2(temp[1], temp[0])
    r = (min_traversal_radius + max_traversal_radius) / 2.0
    actual_middle = center_point + np.array([r * math.cos(middle_theta), r * math.sin(middle_theta)])

    # print(middle_theta)
    # print(middle_point)
    # print(center_point)
    # print(r)
    # print(actual_middle)

    if constraint is None:
        dist_sq_to_middle_point = world.map.get_dist_sq_to(actual_middle)
        near_traversal_center_mask = dist_sq_to_middle_point < max_dist ** 2
    else:
        near_traversal_center_mask = constraint.traversal_mask
    possible_points = np.logical_and(
        locations.island, np.logical_and(near_traversal_center_mask, acceptable_radius_mask)
    )
    if not np.any(possible_points):
        traversal_indices_array = None
    else:
        # biasing away from the ocean so we end up with failing paths less often
        if constraint is None:
            # TODO: max points happens to be reduced under us, so annoying :(
            water_dist = world.map.get_water_distance(
                rand, is_fresh_water_included_in_moisture=True, max_points=600, for_points=possible_points
            )
            water_dist_for_points = water_dist[possible_points]
            power_to_raise = 8
            weights = (water_dist_for_points / water_dist_for_points.max()) ** power_to_raise
            weights /= weights.sum()
            traversal_indices_array = rand.choice(np.argwhere(possible_points), p=weights)

            # pd.DataFrame(water_dist[water_dist > 0.0]).hist()
            # plot_value_grid(water_dist, "water dist", markers=[tuple(traversal_indices_array)])
            # plot_value_grid(
            #     (water_dist / water_dist.max()) ** power_to_raise,
            #     "probabilities",
            #     markers=[tuple(traversal_indices_array)],
            # )
            # plot_value_grid(world.map.Z)
        else:
            traversal_indices_array = rand.choice(np.argwhere(possible_points))
    if is_debugging:
        plot_value_grid(near_traversal_center_mask, "Near traversal center")
        plot_value_grid(possible_points, "Possible points")

    if traversal_indices_array is None:
        raise ImpossibleWorldError("Could not find valid traversal point")
        # fine, whatever, just set to the midpoint
        # ring_traversal_point = to_2d_point((locations.spawn + locations.goal) / 2.0)
    else:
        traversal_indices = tuple(traversal_indices_array)
        ring_traversal_point = np.array([world.map.X[traversal_indices], world.map.Y[traversal_indices]])

    if is_debugging:
        print(f"Selected traversal point {ring_traversal_point}")
        print(np.linalg.norm(ring_traversal_point - center_point))

    if is_single_obstacle:
        outer_obstacle = None
        assert outer_solution is None, "Well that doesnt make sense"
    else:
        outer_obstacle = HeightObstacle(
            edge_config=EdgeConfig(),
            is_inside_ring=False,
            traversal_length=outer_traversal_length,
            traversal_width=traversal_width,
            traversal_noise_interpolation_multiple=traversal_noise_interpolation_multiple,
            is_default_climbable=is_outside_climbable,
            detail_radius=detail_radius,
        )

    ring_config = RingObstacleConfig(
        center_point=center_point,
        traversal_point=ring_traversal_point,
        edge=EdgeConfig(),
        height=height,
        height_mode=height_mode,
        inner_safety_radius=min([spawn_radius, food_radius]) + (safety_radius + extra_safety_radius),
        outer_safety_radius=max([spawn_radius, food_radius]) - (safety_radius + extra_safety_radius),
        inner_obstacle=HeightObstacle(
            edge_config=EdgeConfig(),
            is_inside_ring=True,
            traversal_length=inner_traversal_length,
            traversal_width=traversal_width,
            traversal_noise_interpolation_multiple=traversal_noise_interpolation_multiple,
            is_default_climbable=is_inside_climbable,
            detail_radius=detail_radius,
        ),
        outer_obstacle=outer_obstacle,
        chasm_bottom_size=gap_distance,
        inner_solution=inner_solution,
        outer_solution=outer_solution,
        dual_solution=dual_solution,
        expansion_meters=expansion_meters,
    )

    return ring_config


def create_outer_placeholder_solution(count=5, offset=0.5, randomization_dist: float = 2.0):
    items = [Placeholder(offset=offset, entity_id=0, position=np.array([-0.5 * i, 0.0, 0.0])) for i in range(count)]
    return HeightSolution(
        outside_items=tuple(add_offsets(items)),
        outside_item_randomization_distance=randomization_dist,
        outside_item_radius=first(items).get_offset(),
        solution_point_brink_distance=0.5,
    )


def create_inner_placeholder_solution(count=5, offset=0.5, randomization_dist: float = 2.0):
    items = [Placeholder(offset=offset, entity_id=0, position=np.array([-0.5 * i, 0.0, 0.0])) for i in range(count)]
    return HeightSolution(
        inside_items=tuple(add_offsets(items)),
        inside_item_randomization_distance=randomization_dist,
        inside_item_radius=first(items).get_offset(),
        solution_point_brink_distance=0.5,
    )


def add_offsets(items: List[Tool]):
    new_items = []
    for item in items:
        new_position = item.position.copy().astype(np.float)
        new_position[1] = new_position[1] + item.get_offset()
        new_items.append(attr.evolve(item, position=new_position))
    return new_items


T = TypeVar("T")


def select_categorical_difficulty(
    choices: Sequence[T],
    difficulty: float,
    rand: np.random.Generator,
    _FORCED: Optional[T] = None,
) -> Tuple[T, float]:
    """
    returns selected choice, new difficulty
    """
    # TODO what if we want to actually force it to None? should we actually take index? But that seems less clear.
    if DEBUG_USE_FIRST_CATEGORICAL and _FORCED is None:
        _FORCED = choices[0]

    num_choices = len(choices)
    prob_coeff = 1 / sum(difficulty ** x for x in range(num_choices))
    choice_prob = [
        difficulty / num_choices + (1 - difficulty) * (difficulty ** x) * prob_coeff for x in range(num_choices)
    ]
    choice_idx = rand.choice(range(num_choices), p=choice_prob)
    if _FORCED is not None:
        assert _FORCED in choices
        choice_idx = choices.index(_FORCED)

    # TODO use other less arbitrary method to calculate?
    new_difficulty = difficulty ** 2 + (1 - difficulty) * (difficulty ** (choice_idx + 1))
    return choices[choice_idx], new_difficulty


def select_boolean_difficulty(
    difficulty: float,
    rand: np.random.Generator,
    initial_prob: float = 1,
    final_prob: float = 0.01,
    _FORCED: Optional[bool] = None,
) -> Tuple[bool, float]:
    """
    Uses log interpolation to scale initial prob (difficulty=0) to final prob (difficulty=1). Zero probability is not
    valid, but 1 is. Rescales difficulty to account for probability distribution.
    Tips:
    - Always finish boolean choices before using scalars
    - Somewhat better to have easier option be "true", since we can set p(True) = 1 at difficulty 0.

    debug_override_value that is not None will ALWAYS return that value!
    """
    if DEBUG_USE_TRUE_BOOLEAN and _FORCED is None:
        _FORCED = True
    sampled_value = rand.uniform()
    if _FORCED is not None:
        sampled_value = 0.0 if _FORCED else 1.0

    assert initial_prob <= 1 and final_prob <= 1, "Probability cannot be greater than 1!"
    if initial_prob == final_prob:
        return sampled_value < initial_prob, difficulty
    assert initial_prob > 0 and final_prob > 0, "Cannot have zero probability for True with log interpolation!"
    prob = (initial_prob ** (1 - difficulty)) * (final_prob ** difficulty)
    value = sampled_value < prob
    # the updated difficulty is the integral of the probability to d over the integral to 1
    if value:
        new_difficulty = (prob - initial_prob) / (final_prob - initial_prob)
    else:
        prob_coeff = np.log(initial_prob) - np.log(final_prob)
        new_difficulty = (prob - initial_prob + difficulty * prob_coeff) / (final_prob - initial_prob + prob_coeff)
    return value, new_difficulty


def get_rock_probability(difficulty: float) -> float:
    return scale_with_difficulty(difficulty, 0.5, 0.95)


def scale_with_difficulty(
    difficulty: float, start_val: float, end_val: float, _FORCED: Optional[float] = None
) -> float:
    if _FORCED:
        return _FORCED
    delta = end_val - start_val
    return start_val + delta * difficulty


def difficulty_variation(
    start_val: float,
    end_val: float,
    rand: np.random.Generator,
    difficulty: float,
    _FORCED: Optional[float] = None,
) -> float:
    if _FORCED:
        return _FORCED
    delta = end_val - start_val
    return start_val + delta * difficulty * rand.uniform()


def normal_distrib_range(
    start_val: float,
    end_val: float,
    std_dev: float,
    rand: np.random.Generator,
    difficulty: float,
    _FORCED: Optional[float] = None,
) -> float:
    if _FORCED:
        return _FORCED
    delta = end_val - start_val
    mean = start_val + delta * difficulty
    min_val = min([start_val, end_val])
    max_val = max([start_val, end_val])
    return float(np.clip(rand.normal(mean, std_dev), min_val, max_val))


def starting_hit_points_from_difficulty(difficulty: float, multiplier: float = 1.0) -> float:
    # goes from  multiplier/10 at difficulty 0 to multiplier at difficulty 1
    return multiplier * (10 ** difficulty) / 10


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class TaskGenerationFunctionResult(Serializable):
    starting_hit_points: float
