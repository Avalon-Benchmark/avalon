from pathlib import Path
from typing import Optional

import numpy as np

from datagen.world_creation.constants import AvalonTask
from datagen.world_creation.heightmap import ExportConfig
from datagen.world_creation.tasks.compositional import ForcedComposition
from datagen.world_creation.tasks.compositional import create_compositional_task
from datagen.world_creation.tasks.eat import add_food_tree
from datagen.world_creation.tasks.utils import TaskGenerationFunctionResult
from datagen.world_creation.tasks.utils import export_skill_world
from datagen.world_creation.tasks.utils import select_boolean_difficulty
from datagen.world_creation.world_location_data import to_2d_point


def generate_gather_task(
    rand: np.random.Generator,
    difficulty: float,
    output_path: Path,
    export_config: ExportConfig,
    _FORCED: Optional[ForcedComposition] = None,
) -> TaskGenerationFunctionResult:
    world, locations = create_compositional_task(rand, difficulty, AvalonTask.GATHER, export_config, _FORCED=_FORCED)

    # food being everywhere is probably slightly easier
    is_food_everywhere = select_boolean_difficulty(difficulty, rand, final_prob=0.1)
    if is_food_everywhere:
        island_mask = None
    else:
        island_mask = locations.island

    # TODO: really not sure on the fruit trees... is more easier or harder? Who knows
    food_tree_count = rand.integers(1, 5, endpoint=True)

    # just add some extra foods around. No guarantees about their reachability
    for i in range(food_tree_count):
        tree_position = world.get_safe_point(rand, island_mask=island_mask)
        if tree_position is not None:
            add_food_tree(rand, difficulty, world, to_2d_point(tree_position), is_tool_food_allowed=False)

    export_skill_world(output_path, rand, world)

    # TODO tune hit points
    return TaskGenerationFunctionResult(starting_hit_points=1.0)
