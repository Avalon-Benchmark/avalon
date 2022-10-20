# %%
import copy
from typing import Optional

import gym
import numpy as np
import ray
from einops import rearrange
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune import register_env

from avalon.agent.godot.godot_gym import AvalonGodotEnvWrapper
from avalon.agent.godot.godot_gym import CurriculumWrapper
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import GodotObsTransformWrapper
from avalon.agent.godot.godot_gym import TrainingProtocolChoice

# %% [markdown]
"""
This tutorial demonstrates how to integrate Avalon into an external reinforcement learning setup.
In this example, we're using RLLib,
as it's commonly used and has support for `gym.spaces.Dict` action and observation spaces,
which are required for `avalon`, since our action space has both continuous and discrete actions,
and our observation space has both images and scalar observations.

The goal of the tutorial is to demonstrate Avalon integration,
not to develop a full-featured and performant RL pipeline.
For that, check out the PPO implementation bundled with Avalon,
which is optimized and used for generating our baseline results.
"""

# %% [markdown]
"""
We have only tested this tutorial on linux (Ubuntu), as Avalon is optimized for linux environments.

There's a couple special dependences required for this tutorial, compared to the normal dependencies for `avalon`.
- gym==0.21.0; newer versions don't yet work with rllib.
- ray[rllib] installed from nightly (until they release a version with [this bugfix](https://github.com/ray-project/ray/issues/21921)).
    - for linux python 3.9: 
`pip install -U "ray[rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl"`
    - for mac python 3.9:
`pip install -U "ray[rllib] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-macosx_10_15_x86_64.whl`
    - otherwise grab the nightly from [here](https://docs.ray.io/en/latest/ray-overview/installation.html#daily-releases-nightlies).
    
Once you have those, let's get into the code!
"""
# %%
# We'll start with some gym wrappers to match Avalon's interface to the one expected by RLlib.


# RLLib can't handle the gym.spaces.MultiBinary spaces that we use in Avalon,
# so this wrapper splits them out into separate Discrete spaces.
# The result is equivalent in what it represents, just differently specified.
class RllibActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = copy.deepcopy(env.action_space)
        self.discrete_n = self.action_space.spaces["discrete"].n
        for i in range(self.discrete_n):
            self.action_space.spaces[f"discrete_{i}"] = gym.spaces.Discrete(n=2)
        del self.action_space.spaces["discrete"]

    def action(self, action):
        multibinary = np.zeros(self.discrete_n, dtype=int)
        for i in range(self.discrete_n):
            multibinary[i] = action[f"discrete_{i}"]
        return {"real": action["real"], "discrete": multibinary}


# RLLib expects image observations to be in (height, width, channels) shape,
# and we use (channels, height, width).
# This wrapper swaps those.
class RllibObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        rgbd = env.observation_space["rgbd"]
        self.observation_space.spaces["rgbd"] = gym.spaces.Box(
            low=rgbd.low.transpose(1, 2, 0),
            high=rgbd.high.transpose(1, 2, 0),
            shape=(96, 96, 4),
            dtype=rgbd.dtype,
        )

    def observation(self, x):
        x["rgbd"] = rearrange(x["rgbd"], "c h w -> h w c")
        return x


# %%
# This function creates an Avalon environment, to be called in each rollout worker.
def env_creator(env_config):
    # Each environment needs a unique seed so that they don't start off in an identical state.
    seed = env_config.worker_index * env_config.num_workers + env_config.vector_index

    # This is where you configure Avalon.
    # MULTI_TASK_EASY includes just the EAT and MOVE tasks;
    # change to MULTI_TASK_BASIC to get all non-compositional tasks.
    # See this class for other parameters that can be set.
    env_params = GodotEnvironmentParams(
        resolution=96,
        training_protocol=TrainingProtocolChoice.MULTI_TASK_EASY,
        task_difficulty_update=3e-4,
        energy_cost_coefficient=1e-8,
        seed=seed,
    )

    # This is the core Avalon environment
    env = AvalonGodotEnvWrapper(env_params)
    # This is a standard wrapper we use to scale/clip the observations; it should always be used.
    env = GodotObsTransformWrapper(env, greyscale=env_params.greyscale)
    if env_params.mode == "train":
        # In training, we use curriculum learning.
        # The environment starts off easy, and the difficulty increases adaptively if the agent is doing well.
        # This speeds up learning significantly. See our paper for a full explanation.
        env = CurriculumWrapper(
            env,
            task_difficulty_update=env_params.task_difficulty_update,
            meta_difficulty_update=env_params.meta_difficulty_update,
        )
    # These are the wrappers we created above
    env = RllibActionWrapper(env)
    env = RllibObservationWrapper(env)
    return env


# This registers our environment creation function with RLlib.
register_env("godot", env_creator)
# %%
# We store some extra information in the `info` dict returned on the last step of an episode.
# This includes the difficulty of that episode, and whether the agent succeeded in eating the food.
# This callback grabs that information and exposes it to RLlib to include in the logging.
class RayCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index: Optional[int] = None, **kwargs):
        info = episode.last_info_for()
        task = info["task"]
        episode.custom_metrics["success"] = info["success"]
        episode.custom_metrics[f"{task.lower()}_success"] = info["success"]
        episode.custom_metrics[f"{task.lower()}_difficulty"] = info["difficulty"]


# %%
# RLlib configuration.
# These hyperparams are not tuned,
# although they do work (the agent is able to learn at least in the `EASY` training protocol used here).
config = {
    "env_config": {},
    "env": "godot",
    "log_level": "INFO",
    "framework": "torch",
    "num_workers": 4,
    "num_envs_per_worker": 4,
    "num_gpus": 1,
    "num_gpus_per_worker": 0.25,
    "rollout_fragment_length": 200,
    "gamma": 0.99,
    "lr": 2.5e-5,
    "train_batch_size": 200 * 16,
    "sgd_minibatch_size": 200 * 16,
    "num_sgd_iter": 2,
    "clip_actions": False,
    "normalize_actions": True,
    "metrics_num_episodes_for_smoothing": 5,
    "callbacks": RayCallbacks,
    "kl_coeff": 0,
    "kl_target": 0.5,
    "lambda": 0.83,
    "clip_param": 0.03,
    "grad_clip": 0.5,
    "seed": 0,
}

# %%
# Now let's run the training!
ray.init(object_store_memory=4 * 10 ** 9)
trainer = PPOTrainer(config)
while True:
    result = trainer.train()
    if result["num_env_steps_sampled"] > 10_000_000:
        break


# %% [markdown]
"""
You can view the results of the training by running
`tensorboard serve --logdir ~/ray_results`

Some key metrics to look at are:
- `[eat/move]_success_mean`: this is the percentage of episodes that were a success (food was found)
- `[eat/move]_difficulty_mean`: this is the curriculum difficulty. it starts at 0 and increases when the respective task's `success_mean` is greater than .5.
- `time_this_iter_s`: the total time to collect and train one batch of rollouts. in this example, we have 16 workers * 200 steps per worker = 3200 frames.
On a machine with a 3090, this is ~17 seconds, for an effective throughput of 3200/17 = 190 env steps/sec. This is pretty bad; our implementation gets >800sps with the same number of workers.

We haven't tried to tune the speed of this implementation,
but one challenge with getting Avalon to run fast is that, while each `step()` is quite fast,
each `reset()` takes a meaningful amount of time (to load in the next world into godot).
In our PPO implementation, we have an intelligent rollout worker that will run inference on a partial batch
of environments if they're not all quickly ready.
This keeps the other environments from being blocked by one environment's slow `reset()`.
However, most RL implementations don't have this feature,
and so can get bogged down by Avalon's slow `reset()`s.
Check out our implementation in `avalon/common/worker.py` if you need to adapt something similar in your training setup.
"""
