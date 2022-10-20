# %%
import shutil
from pathlib import Path

from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.datagen.env_helper import display_video
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.world_generator import GenerateAvalonWorldParams
from avalon.datagen.world_creation.world_generator import generate_world

# %%
OUTPUT_FOLDER = Path("./output/").absolute()
if OUTPUT_FOLDER.exists():
    shutil.rmtree(OUTPUT_FOLDER)

params = generate_world(
    GenerateAvalonWorldParams(
        AvalonTask.MOVE,
        difficulty=1,
        seed=42,
        index=0,
        output=str(OUTPUT_FOLDER),
    )
)

# %%
env_params = GodotEnvironmentParams(
    resolution=256,
    training_protocol=TrainingProtocolChoice.SINGLE_TASK_FIGHT,
    initial_difficulty=1,
)
env = AvalonEnv(env_params)
env.reset_nicely_with_specific_world(episode_seed=0, world_params=params)


def random_env_step():
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
    return obs


observations = [random_env_step() for _ in range(50)]

# %%
display_video(observations, fps=10)
