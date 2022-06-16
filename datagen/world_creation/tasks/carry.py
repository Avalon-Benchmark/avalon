from pathlib import Path

import attr
import numpy as np
from scipy import stats
from skimage import morphology
from skimage.morphology import flood_fill

from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.items import CANONICAL_FOOD_CLASS
from datagen.world_creation.tasks.bridge import create_bridge_obstacle
from datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from datagen.world_creation.tasks.fight import create_fight_obstacle
from datagen.world_creation.tasks.stack import create_stack_obstacle
from datagen.world_creation.tasks.throw import throw_obstacle_maker
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import scale_with_difficulty
from datagen.world_creation.tasks.utils import starting_hit_points_from_difficulty
from datagen.world_creation.world_location_data import to_2d_point


def generate_carry_task(
    rand: np.random.Generator, difficulty: float, output_path: Path, export_config: ExportConfig
) -> TaskGenerationFunctionResult:
    inner_generator = rand.choice(
        [
            throw_obstacle_maker,
            create_fight_obstacle,
            create_stack_obstacle,
            create_bridge_obstacle,
        ]
    )
    result = inner_generator(rand, difficulty, export_config, is_for_carry=True)
    if len(result) == 5:
        difficulty, food_class, locations, spawn_location, world = result
        locations = attr.evolve(locations, spawn=spawn_location)
        is_on_tree = False
    else:
        world, locations, difficulty = result
        food_class = CANONICAL_FOOD_CLASS
        is_on_tree = True

    world.end_height_obstacles(locations, is_accessible_from_water=False)

    # move the things to your spawn region

    # start with this island
    mask = locations.island
    # remove all obstacles
    mask = np.logical_and(mask, np.logical_not(world.full_obstacle_mask))
    # remove unclimbable places
    mask = np.logical_and(mask, world.is_climbable)
    # make a little smaller:
    mask = morphology.dilation(np.logical_not(mask), morphology.disk(1))

    # plot_value_grid(mask)

    # flood fill from your spanw. These are the points that you can reach
    index = world.map.point_to_index(to_2d_point(locations.spawn))
    flood_mask = mask.copy().astype(int)
    flood_fill(flood_mask, index, 2, in_place=True)
    spawn_region = flood_mask == 2

    # plot_value_grid(spawn_region, "spawn")

    if np.any(spawn_region):
        all_mobile_items = [x for x in world.items if getattr(x, "solution_mask", None) is not None]
        for item in all_mobile_items:
            full_position_mask = np.logical_or(item.solution_mask, spawn_region)
            item = attr.evolve(item, solution_mask=full_position_mask)
            carry_distance = get_carry_distance_preference(difficulty)
            world.carry_tool_randomly(rand, item, carry_distance)

    if is_on_tree:
        add_food_tree_for_simple_task(world, locations)
        world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    else:
        world.add_spawn_and_food(rand, difficulty, locations.spawn, locations.goal, food_class=food_class)

    export_skill_world(output_path, rand, world)

    return TaskGenerationFunctionResult(starting_hit_points_from_difficulty(difficulty))


def get_carry_distance_preference(difficulty: float):
    min_item_dist = 1.0
    max_item_dist = 10.0
    desired_distance = scale_with_difficulty(difficulty, min_item_dist, max_item_dist)
    distance_preference = stats.norm(desired_distance, desired_distance / 4.0)
    return distance_preference
