# %%
import shutil
import time
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt

from avalon.common.log_utils import enable_debug_logging
from avalon.datagen.env_helper import create_env
from avalon.datagen.env_helper import create_vr_benchmark_config
from avalon.datagen.env_helper import display_video
from avalon.datagen.env_helper import get_null_vr_action
from avalon.datagen.godot_env import VRActionType
from avalon.datagen.world_creation.configs.export import get_agent_export_config
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.world_generator import GENERATION_FUNCTION_BY_TASK

enable_debug_logging()

# %%

task = AvalonTask.MOVE
difficulty = 0.6
world_seed = 0
NUM_ACTIONS = 5
RESOLUTION = 96

env_seed = 0


def create_world(output_path: Path, task: AvalonTask, difficulty: float, seed: int) -> Dict[str, Any]:
    start_time = time.time()
    rand = np.random.default_rng(seed)
    export_config = get_agent_export_config()
    generation_function = GENERATION_FUNCTION_BY_TASK[task]
    world_path = output_path / f"{task.value}__{seed}__{difficulty}"
    generation_function(rand, difficulty, world_path, export_config)
    end_time = time.time()
    total_time = end_time - start_time
    return dict(task=task, difficulty=difficulty, world_path=world_path, time=total_time)


base_output_path = Path("/tmp/levels/debug/")
if base_output_path.exists():
    shutil.rmtree(base_output_path)
base_output_path.mkdir(parents=True, exist_ok=True)

result = create_world(base_output_path, task, difficulty, world_seed)

config = create_vr_benchmark_config()

with config.mutable_clone() as config:
    config.recording_options.resolution_x = RESOLUTION
    config.recording_options.resolution_y = RESOLUTION

action_type = VRActionType
env = create_env(config, action_type)

observations = [
    env.reset_nicely_with_specific_world(
        episode_seed=env_seed,
        world_path=str(result["world_path"] / "main.tscn"),
    )
]

for i in range(NUM_ACTIONS):
    null_action = get_null_vr_action()
    obs, _ = env.act(null_action)
    observations.append(obs)

# %%
# display the video
display_video(observations, size=(RESOLUTION, RESOLUTION))

# %%
# display the first few frames as images
for obs in observations:
    plt.imshow(obs.rgbd[:, :, :3][::-1])
    plt.show()

# %%
