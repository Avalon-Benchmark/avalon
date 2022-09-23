from pathlib import Path
from typing import Optional

import attr
import numpy as np

from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.tasks.eat import add_food_tree
from avalon.datagen.world_creation.utils import to_2d_point
from avalon.datagen.world_creation.worlds.compositional import CompositeTaskConfig
from avalon.datagen.world_creation.worlds.compositional import ForcedComposition
from avalon.datagen.world_creation.worlds.compositional import create_compositional_task
from avalon.datagen.world_creation.worlds.difficulty import select_boolean_difficulty
from avalon.datagen.world_creation.worlds.export import export_world


@attr.s(auto_attribs=True, collect_by_mro=True, hash=True)
class GatherTaskConfig(CompositeTaskConfig):
    task: AvalonTask = AvalonTask.GATHER
    # how many extra fruit trees to create. Just uniform random between these two values (inclusive)
    extra_fruit_tree_count_min: int = 1
    extra_fruit_tree_count_max: int = 5


def generate_gather_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    task_config: GatherTaskConfig = GatherTaskConfig(),
    _FORCED: Optional[ForcedComposition] = None,
) -> None:
    world, locations = create_compositional_task(rand, difficulty, task_config, export_config, _FORCED=_FORCED)

    # food being everywhere is probably slightly easier
    is_food_everywhere = select_boolean_difficulty(difficulty, rand, final_prob=0.1)
    if is_food_everywhere:
        island_mask = None
    else:
        island_mask = locations.island

    # are more trees easier or harder? Not sure
    food_tree_count = rand.integers(
        task_config.extra_fruit_tree_count_min, task_config.extra_fruit_tree_count_max, endpoint=True
    )

    # just add some extra foods around. No guarantees about their reachability
    for i in range(food_tree_count):
        tree_position = world.get_safe_point(rand, island_mask=island_mask)
        if tree_position is not None:
            _difficulty, _tool, world = add_food_tree(
                rand, difficulty, world, to_2d_point(tree_position), is_tool_food_allowed=False
            )

    export_world(output_path, rand, world)
