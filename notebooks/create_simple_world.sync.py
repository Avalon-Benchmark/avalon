# %%
from pathlib import Path

import numpy as np
from scipy import stats

from avalon.common.log_utils import enable_debug_logging
from avalon.datagen.world_creation.configs.export import get_oculus_export_config
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_VISIBLE_HEIGHT
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.types import WorldType
from avalon.datagen.world_creation.worlds.creation import create_world_for_skill_scenario
from avalon.datagen.world_creation.worlds.export import export_world

enable_debug_logging()

# %%
difficulty = 0.5
size_in_meters = 50.0
world_type = WorldType.CONTINENT

# The export config you use depends on your use case, ee datagen/world_creation/configs/export.py for others
base = get_oculus_export_config()

export_configs = [
    base,
]

# %%
for i, export_config in enumerate(export_configs):
    rand = np.random.default_rng(0)

    world, locations = create_world_for_skill_scenario(
        rand,
        difficulty,
        CANONICAL_FOOD_HEIGHT_ON_TREE,
        stats.norm(size_in_meters, size_in_meters / 5),
        world_type=world_type,
        export_config=export_config,
        visibility_height=FOOD_TREE_VISIBLE_HEIGHT,
    )

    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)

    output_path = Path(f"/tmp/profiling_levels/level_{export_config.name}")
    output_path.mkdir(parents=True, exist_ok=True)
    export_world(output_path, rand, world)
