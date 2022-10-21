# %% [markdown]
# ### Custom Skill Worlds
# In this tutorial we'll cover:
# 1. Using avalon's world generation internals for creating custom tasks
# 2. Downloading generated worlds for inspection in the [avalon godot editor](https://github.com/Avalon-Benchmark/godot/releases).
# 3. Testing the world in mouse and keyboard mode with Godot's debugger.
# %%
from pathlib import Path
from typing import Final
from typing import Sequence

import attr
import numpy as np
from scipy import stats

from avalon.common.log_utils import enable_debug_logging
from avalon.datagen.world_creation.configs.export import ExportConfig
from avalon.datagen.world_creation.configs.export import get_mouse_and_keyboard_export_config
from avalon.datagen.world_creation.entities.animals import Frog
from avalon.datagen.world_creation.entities.constants import CANONICAL_FOOD_HEIGHT_ON_TREE
from avalon.datagen.world_creation.entities.constants import FOOD_TREE_VISIBLE_HEIGHT
from avalon.datagen.world_creation.entities.item import InstancedDynamicItem
from avalon.datagen.world_creation.entities.tools.weapons import Stick
from avalon.datagen.world_creation.tasks.eat import add_food_tree_for_simple_task
from avalon.datagen.world_creation.types import WorldType
from avalon.datagen.world_creation.worlds.creation import create_world_for_skill_scenario
from avalon.datagen.world_creation.worlds.export import export_world
from avalon.datagen.world_creation.worlds.export import get_world_slug

enable_debug_logging()

EXAMPLE_WORLDS_PATH: Final = "/tmp/example_worlds"


# %% [markdown]
# ### `create_simple_world`
# `create_world_for_skill_scenario` generates our terrain and scenery,
# varying foliage, biomes, and terrain based on `diversity` and `world_type`
# (`PLATONIC`, `ARCHIPELAGO`, or `CONTINENT`).
#
# Specific tasks will then customize the world further,
# adding task-specific animals, items, and terrain obstacles.
# %%
def create_simple_world(
    export_config: ExportConfig,
    world_type: WorldType = WorldType.CONTINENT,
    difficulty: float = 0.1,
    size_in_meters: float = 20.0,
    seed: int = 0,
    items_near_player: Sequence[InstancedDynamicItem] = [],
) -> Path:

    rand = np.random.default_rng(seed)

    # create_world_for_skill_scenario generates our terrain and scenery,
    # adding automatic diversity
    #
    # Available world types are PLATONIC, ARCHIPELAGO, and CONTINENT
    world, locations = create_world_for_skill_scenario(
        rand,
        diversity=difficulty,
        food_height=CANONICAL_FOOD_HEIGHT_ON_TREE,
        goal_distance=stats.norm(size_in_meters, size_in_meters / 5),
        world_type=world_type,
        export_config=export_config,
        visibility_height=FOOD_TREE_VISIBLE_HEIGHT,
    )

    world = add_food_tree_for_simple_task(world, locations)
    world = world.add_spawn(rand, difficulty, locations.spawn, locations.goal)
    for item in items_near_player:
        relative_to_player = item.position + locations.spawn
        item = attr.evolve(item, position=relative_to_player)
        world = world.add_item(item, item.get_offset())

    custom_task_name = f"simple_{world_type.name.lower()}_{size_in_meters}"
    world_id = get_world_slug(custom_task_name, seed, difficulty, is_practice=True)

    output_path = Path(f"{EXAMPLE_WORLDS_PATH}/{world_id}")
    output_path.mkdir(parents=True, exist_ok=True)
    export_world(output_path, rand, world)

    return output_path


export_config = get_mouse_and_keyboard_export_config()

items = [
    Stick(position=np.array([2.0, 0.0, 0.0])),
    Frog(position=np.array([0.0, 0.0, 3.0])),
]

# our default world
exported_path = create_simple_world(export_config, items_near_player=items)

# exploring some diversity
more_diverse_options = [
    (WorldType.ARCHIPELAGO, 0.1),
    (WorldType.CONTINENT, 0.5),
    (WorldType.PLATONIC, 0.75),
]

for world_type, difficulty in more_diverse_options:
    create_simple_world(export_config, world_type, difficulty, items_near_player=items)

# %% [markdown]
## Inspecting and running worlds
# Now that you've generated some worlds, you can inspect and test them in the
# [avalon godot editor](https://github.com/Avalon-Benchmark/godot/releases).
#
# First, install the editor for your platform and set the path:
# ```bash
# GODOT_BINARY_PATH=/path/to/avalon/godot
# ```
#
# > Note: Running in windows directly is unsupported
#
# Now we need to move our generated worlds in the avalon godot project so they'll be visible in the editor:
# ```bash
# # Moving worlds into position when running locally:
# #
# EXAMPLE_WORLDS_PATH=/tmp/example_worlds
# avalon_godot_project_path=avalon/datagen/godot
# mkdir $avalon_godot_project_path/worlds/
# mv $EXAMPLE_WORLDS_PATH/* "$avalon_godot_project_path/worlds/*"
# $GODOT_BINARY_PATH --editor $avalon_godot_project_path/project.godot
#
#
# # Copying worlds into position from a docker container
# #
# container_id=...
# avalon_godot_project_path=avalon/datagen/godot
# docker cp $container_id:$EXAMPLE_WORLDS_PATH ./example_worlds
# mkdir $avalon_godot_project_path/worlds/
# mv ./example_worlds/* "$avalon_godot_project_path/worlds" && rmdir ./example_worlds/
# $GODOT_BINARY_PATH --editor $avalon_godot_project_path/project.godot
# ````
#
# ### Inspecting the world
# Once you've opened the editor, your should be able to find your worlds' main scenes under `worlds`, such as:
# `worlds/practice__simple_continent_20__0_0_1/main.tscn`.
# Double clicking the scene will open it in the editor for inspection.
#
# > Note: the Player scene is added at runtime at `dynamic_tracker/SpawnPoint`.
#
# ### Testing out worlds
# Avalon comes with a mouse & keyboard mode for debugging.
# Press the "play" button in the upper right hand corner of the editor,
# which will spawn you in the entry scene.
#
# Go to the righthand teleporter, grab buttons on the sign to select a world,
# and grab the orb on the pillar to teleport to your world
# (The keybinding help panel can be toggled with `?`).
#
#
# > Note: If the player starts wobbling uncontrollably when the game starts,
# > you're running within a standard godot release and need to instead install and run avalon's
# > [custom build](https://github.com/Avalon-Benchmark/godot/releases).
