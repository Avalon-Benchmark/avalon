# %% [markdown]
# ## Inspecting Test Worlds
# Sometimes you might want to inspect the currently running world, or various nodes inside of it.
# This can be done using `env.debug_act`, which configures a debugging camera that will track the configured node.
#
# We'll also cover regenerating worlds for human inspection in the editor.
# %%
#%set_env PYTHONHASHSEED=0
# %%
import shutil
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict
from typing import Dict
from typing import List

from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.agent.godot.godot_gym import task_groups_from_training_protocol
from avalon.common.utils import flatten
from avalon.datagen.env_helper import DebugLogLine
from avalon.datagen.env_helper import display_video
from avalon.datagen.env_helper import get_debug_json_logs
from avalon.datagen.env_helper import get_null_vr_action
from avalon.datagen.generate_worlds import generate_evaluation_worlds
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.observations import AvalonObservation
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.constants import get_all_tasks_for_task_groups
from avalon.datagen.world_creation.world_generator import FixedWorldLoader
from avalon.datagen.world_creation.world_generator import GeneratedAvalonWorldParams

TEST_WORLD_OUTPUT_PATH = Path("/tmp/test_worlds")
TRAINING_PROTOCOL = TrainingProtocolChoice.MULTI_TASK_BASIC
# %% [markdown]
# ### Generating Test Worlds
# First we'll generate our worlds using the evaluation world generator.
#
# This generator always includes some specially tuned configurations to ensure
# agents are evaluated on a set of worlds with a minimum threshold of variety.
# %%
shutil.rmtree(TEST_WORLD_OUTPUT_PATH, ignore_errors=True)

min_difficulty = 0.0
worlds_per_task = 10


def tasks_from_protocol(protocol: TrainingProtocolChoice) -> List[AvalonTask]:
    groups = task_groups_from_training_protocol(protocol, is_meta_curriculum_used=False)
    return list(sorted(get_all_tasks_for_task_groups(groups), key=lambda g: g.name))


generated_world_params = generate_evaluation_worlds(
    base_output_path=TEST_WORLD_OUTPUT_PATH,
    tasks=tasks_from_protocol(TRAINING_PROTOCOL),
    min_difficulty=min_difficulty,
    num_worlds_per_task=worlds_per_task,
    is_generating_for_human=False,
    start_seed=0,
    num_workers=4,
    is_verbose=False,
)

# %%

hd_test_env = AvalonEnv(
    GodotEnvironmentParams(
        # It is possible to inspect any world while it is being run,
        # but if we want to inspect the evaluation worlds we just generated we need to 'test'
        mode="test",
        resolution=1024,
        seed=0,
        # necessary to get debug logs
        is_debugging_godot=True,
        #
        # Determines the tasks generated
        training_protocol=TRAINING_PROTOCOL,
        fixed_world_min_difficulty=min_difficulty,
        test_episodes_per_task=worlds_per_task,
        fixed_worlds_load_from_path=TEST_WORLD_OUTPUT_PATH,
    )
)
assert isinstance(hd_test_env.world_generator, FixedWorldLoader)

# %% [markdown]
# ### Inspecting worlds during training
#
# `env.debug_act` can be used to create and control a debug camera during training for inspecting nodes beyond the sight of the agent.
#
# When working with generated worlds, you can usually get a good higher-level view by focusing on `tile_0_0`,
# and find important items by inspecting the output of the debug log with `get_debug_json_logs` (as long as it has been requested).
# %%
def isometric_view(
    env: AvalonEnv, node: str = "tile_0_0", distance: float = 50, frames: int = 1
) -> List[AvalonObservation]:
    iso = DebugCameraAction.isometric(node, distance)
    view = [env.debug_act(iso)]
    view.extend([env.act(get_null_vr_action())[0] for _ in range(frames - 1)])
    return view


def good_viewing_distance(params: GeneratedAvalonWorldParams) -> int:
    if params.task in (AvalonTask.EXPLORE,):
        return 100
    return 50


def get_isometric_item_views(
    env: AvalonEnv, distance: int = 10, frames: int = 1
) -> Dict[str, List[AvalonObservation]]:
    # get our list of existing items from the debug log
    latest_frame: DebugLogLine = get_debug_json_logs(env)[-1]
    item_names = [i["name"] for i in latest_frame["items"]]

    return {
        item_name: isometric_view(hd_test_env, item_name, distance=distance, frames=frames) for item_name in item_names
    }


# %%
# Go through our generated worlds and collect the mid-difficulty world for each task for inspection:
generated_worlds_by_task: DefaultDict[AvalonTask, List[GeneratedAvalonWorldParams]] = defaultdict(lambda: [])
for params in hd_test_env.world_generator.worlds.values():
    generated_worlds_by_task[params.task].append(params)

worlds_to_inspect = []
for task, generated in generated_worlds_by_task.items():
    mid_difficulty_for_task_index = int(worlds_per_task / 2)
    params = sorted(generated, key=lambda p: p.difficulty)[mid_difficulty_for_task_index]
    worlds_to_inspect.append(params)

# %% [markdown]
# Now that we've decided on some worlds we'd like to look at, we'll iterate over them and:
# * Take a high-level overview snapshot
# * Query godot for the world's dynamic items
# * Attempt to take a snapshot of each item with the debug camera
# * Combine it all into a video
#
# Some of these samples won't get good item vantages here, especially indoor worlds.
# For those it is often easier to inspect them in the editor.
# %%
for params in worlds_to_inspect:
    player_view = hd_test_env.reset_nicely(world_id=params.index, episode_seed=params.seed)
    viewing_distance = good_viewing_distance(params)
    iso_world_view = isometric_view(hd_test_env, distance=viewing_distance, frames=1)

    item_views = get_isometric_item_views(hd_test_env)
    print(
        f"{params.task} (seed: {params.seed}, difficulty: {params.difficulty})\n  "
        f"{len(item_views)} items: {', '.join(item_views.keys())}"
    )

    iso_item_observations = flatten(item_views.values())

    display_video([*iso_world_view, player_view, *iso_item_observations, iso_world_view[0]], fps=1)

# %% [markdown]
# ### Opening in the editor
#
# To regenerate the above worlds for human inspection and playing, use `avalon.for_humans` utility:
# ```bash
# python -m avalon.for_humans generate_evaluation_worlds --tasks=SIMPLE --start_seed=0 --min_difficulty=0.0 --worlds_per_task=10
# python -m avalon.for_humans launch_editor
# ```
#
# You should now be able to see the worlds in the `worlds` directory:
#
# ![opened_in_editor](https://user-images.githubusercontent.com/8343799/198161192-c18fb7ab-192a-4638-b5ac-67bde17eba8c.png)
#
# > Note: Worlds generated for training use absolute paths internally, making them difficult to open in the editor.
