import pathlib as pathlib
import shutil

from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.datagen.env_helper import display_video
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.world_generator import GenerateAvalonWorldParams
from avalon.datagen.world_creation.world_generator import generate_world

# %%
# Initialize Avalon with the default parametric world generator

env_params = GodotEnvironmentParams(
    resolution=256,
    training_protocol=TrainingProtocolChoice.SINGLE_TASK_FIGHT,
    initial_difficulty=0,
)
env = AvalonEnv(env_params)
_ = env.reset()

# %%
# Take some environment steps and record the observations


def random_env_step():
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
    return obs


observations = [random_env_step() for _ in range(50)]
display_video(observations, fps=10)
# %%
# We can also generate and load a world manually

OUTPUT_FOLDER = pathlib.Path("./output/").absolute()

shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
params = generate_world(
    GenerateAvalonWorldParams(
        AvalonTask.MOVE,
        difficulty=1,
        seed=42,
        index=0,
        output=str(OUTPUT_FOLDER),
    )
)
env.reset_nicely_with_specific_world(episode_seed=0, world_params=params)

observations = [random_env_step() for _ in range(50)]
# %%
display_video(observations, fps=10)
