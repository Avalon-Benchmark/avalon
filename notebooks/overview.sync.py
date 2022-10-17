# %%
from typing import List

from loguru import logger

from avalon.datagen.env_helper import DebugLogLine
from avalon.datagen.env_helper import create_env
from avalon.datagen.env_helper import create_vr_benchmark_config
from avalon.datagen.env_helper import display_video
from avalon.datagen.env_helper import get_debug_json_logs
from avalon.datagen.env_helper import get_null_vr_action
from avalon.datagen.godot_env.actions import DebugCameraAction
from avalon.datagen.godot_env.actions import VRActionType
from avalon.datagen.godot_env.godot_env import GodotEnv
from avalon.datagen.godot_env.observations import AvalonObservationType
from avalon.datagen.world_creation.constants import SINGLE_TASK_GROUPS
from avalon.datagen.world_creation.constants import AvalonTaskGroup
from avalon.datagen.world_creation.constants import get_all_tasks_for_task_groups
from avalon.datagen.world_creation.world_generator import BlockingWorldGenerator

# %% [markdown]
# ### Simple random sampling with the standard gym interface

# %%
config = create_vr_benchmark_config()
vr_action_type = VRActionType
action_space = vr_action_type.to_gym_space()
env = create_env(config, vr_action_type)

observations = [env.reset()]

for _ in range(10):
    random_action = action_space.sample()
    observation, _reward, _is_done, _log = env.step(random_action)
    observations.append(observation)

display_video(observations)

# %% [markdown]
# ### Inspecting worlds during training
#
# `env.debug_act` can be used to create and control a debug camera during training for inspecting nodes beyond the sight of the agent.
#
# When working with generated worlds, you can usually get a good higher-level view by focusing on `tile_0_0`,
# and find important items by inspecting the output of the debug log with `get_debug_json_logs` (as long as it has been requested).

# %%
with create_vr_benchmark_config().mutable_clone() as hd_config:
    hd_config.recording_options.resolution_x = 1024
    hd_config.recording_options.resolution_y = 1024
    hd_config.recording_options.is_debugging_output_requested = True

hd_env = create_env(hd_config, VRActionType)
assert isinstance(hd_env.world_generator, BlockingWorldGenerator)

difficulty = 0.6
_all_tasks = get_all_tasks_for_task_groups(SINGLE_TASK_GROUPS)
hd_env.world_generator.difficulties = {task: difficulty for task in _all_tasks}


def isometric_view_of_current_world(
    env: GodotEnv[AvalonObservationType, VRActionType], distance: float = 50, frames: int = 1
) -> List[AvalonObservationType]:
    iso = DebugCameraAction.isometric("tile_0_0", distance)
    view = [env.debug_act(iso) for _ in range(frames)]
    env.act(get_null_vr_action())
    return view


task_groups_to_look_at = (
    (0, AvalonTaskGroup.HUNT, 50),
    (1, AvalonTaskGroup.AVOID, 50),
    (4, AvalonTaskGroup.FIND, 100),
    (5, AvalonTaskGroup.SURVIVE, 100),
    (6, AvalonTaskGroup.GATHER, 100),
    (3, AvalonTaskGroup.NAVIGATE, 100),
    (2, AvalonTaskGroup.EXPLORE, 100),
)

for world_id, task_group, distance in task_groups_to_look_at:
    hd_env.world_generator.task_groups = (task_group,)
    hd_env.reset_nicely(world_id=world_id, episode_seed=world_id)
    iso_view = isometric_view_of_current_world(hd_env, distance)

    latest_frame: DebugLogLine = get_debug_json_logs(hd_env)[-1]
    items = [i["name"] for i in latest_frame["items"]]
    logger.info(f"{task_group}\n {len(items)} items: {items}")

    display_video(isometric_view_of_current_world(hd_env, distance))

# %% [markdown]
# You can also inspect generated levels locally in the [Godot Editor](https://godotengine.org/download)
# By copying them into `datagen/godot/worlds` with `docker cp`.
#
# > NOTE: Each env uses a different unique path for world generation,
#         And also need those path references replaced to open properly locally
#
# To try this, run the command output by the following block locally,
# then open Godot and [import the project](https://docs.godotengine.org/en/latest/tutorials/editor/project_manager.html#opening-and-importing-projects).
# ```bash
# # outputs absolute world paths from most to least recent
# function recent_avaolon_worlds {
#   docker exec $container_id '/bin/sh' '-c' 'ls -td /tmp/level_gen/*/*'
# }
# function pull_avalon_worlds {
# }
# # Pull the last 10 generated worlds
# pull_avalon_worlds `recent_avalon_worlds | head -10`
#
# ```
# %%
unique_env_level_path = hd_env.world_generator.output_path
world_index = "1"
local_worlds_path = "datagen/godot/worlds"
print(
    f"# Copy world .tscn to local {local_worlds_path}"
    f"docker cp $container_id:{unique_env_level_path}/{world_index} datagen/godot/worlds\n"
    f"# Find and replace absolute generated world path to fix local references"
    f"perl -pi -e s,{unique_env_level_path},res://worlds, {local_worlds_path}/{world_index}/*\n"
)


# %%

# TODO Rerunning a recorded human action log
# from ..contrib.s3_utils import SimpleS3Client
# from ..datagen.human_playback import PlaybackResult

"""
AVALON_BUCKET_NAME = "avalon-benchmark"
OBSERVATION_KEY = "avalon__all_observations__935781fe-267d-4dcd-9698-714cc891e985.tar.gz"

def fetch_sample_playback(sample_human_playback_key: str) -> Path:
    s3_client = SimpleS3Client(bucket_name=AVALON_BUCKET_NAME)
    archive_name = "{sample_human_playback_key}.tar.gz"
    replay_path = Path(f"./replay_{sample_human_playback_key}")
    s3_client.download_to_file(key=f"sample_playback/{archive_name}", output_path=Path(archive_name))
    shutil.unpack_archive(archive_name, replay_path, "gztar")
    os.remove(archive_name)
    return replay_path

sample_replay_path = fetch_sample_playback("now_thats_a_good_replay")
validate_oculus_playback_recording(
    sample_replay_path,
    playback_path: Path,
    worlds_path: Path,
    threshold=0.0,
    is_using_human_input=False,
)
"""

# TODO running actions from pretrained weights
