from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Tuple

import attr
import numpy as np
from scipy import stats
from skimage import morphology
from skimage.morphology import flood_fill

from avalon.common.utils import only
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.task import TaskConfig
from avalon.datagen.world_creation.constants import ITEM_FLATTEN_RADIUS
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.entities.food import CANONICAL_FOOD_CLASS
from avalon.datagen.world_creation.entities.tools.log import Log
from avalon.datagen.world_creation.entities.tools.stone import Stone
from avalon.datagen.world_creation.entities.tools.weapons import LargeRock
from avalon.datagen.world_creation.entities.tools.weapons import Rock
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.tasks.bridge import create_bridge_obstacle
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.tasks.fight import create_fight_obstacle
from avalon.datagen.world_creation.tasks.stack import create_stack_obstacle
from avalon.datagen.world_creation.tasks.stack import flatten_places_under_items
from avalon.datagen.world_creation.tasks.throw import throw_obstacle_maker
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.difficulty import scale_with_difficulty
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class CarryTaskConfig(TaskConfig):
    # how far should the item be moved away from its original spot on difficulty 0.0 and 1.0 respectively
    carry_dist_easy: float = 1.0
    carry_dist_hard: float = 10.0
    # in which tasks should we be carrying items away from their original spawn locations?
    # the way the carry task works is simply by moving the items farther away from their original locations
    tasks: Tuple[AvalonTask, ...] = (AvalonTask.THROW, AvalonTask.FIGHT, AvalonTask.STACK, AvalonTask.BRIDGE)


_GENERATION_FUNCTION_BY_TASK: Dict[AvalonTask, Callable] = {
    AvalonTask.THROW: throw_obstacle_maker,
    AvalonTask.FIGHT: create_fight_obstacle,
    AvalonTask.STACK: create_stack_obstacle,
    AvalonTask.BRIDGE: create_bridge_obstacle,
}


def generate_carry_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: CarryTaskConfig = CarryTaskConfig(),
) -> None:
    inner_task = rand.choice(task_config.tasks)  # type: ignore[arg-type]
    inner_generator = _GENERATION_FUNCTION_BY_TASK[inner_task]
    result = inner_generator(rand, difficulty, export_config, is_for_carry=True)
    if len(result) == 5:
        difficulty, food_class, locations, spawn_location, world = result
        locations = attr.evolve(locations, spawn=spawn_location)
        is_on_tree = False
    else:
        world, locations, difficulty = result
        food_class = CANONICAL_FOOD_CLASS
        is_on_tree = True

    world, locations = world.end_height_obstacles(locations, is_accessible_from_water=False)

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
            desired_distance = scale_with_difficulty(
                difficulty, task_config.carry_dist_easy, task_config.carry_dist_hard
            )
            carry_distance = stats.norm(desired_distance, desired_distance / 4.0)
            world = world.carry_tool_randomly(rand, item, carry_distance)

    if inner_task == AvalonTask.STACK:
        world = flatten_places_under_items(world, earlier_item_count=0, filter_to_classes=Stone)
    elif inner_task == AvalonTask.BRIDGE:
        log = only([x for x in world.items if isinstance(x, Log)])
        world = world.flatten(to_2d_point(log.position), ITEM_FLATTEN_RADIUS, ITEM_FLATTEN_RADIUS)
    else:
        world = flatten_places_under_items(world, earlier_item_count=0, filter_to_classes=(Rock, LargeRock, Stick))

    if is_on_tree:
        world = add_food_tree_for_simple_task(world, locations)
        world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    else:
        world = world.add_spawn_and_food(rand, difficulty, locations.spawn, locations.goal, food_class=food_class)

    export_world(output_path, rand, world)
