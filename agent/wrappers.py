import copy
from collections import defaultdict
from typing import Dict

import gym
import numpy as np
import torch
from gym import Wrapper
from gym import spaces
from torch.nn import functional as F
from torchvision import transforms

from datagen.world_creation.constants import AvalonTask


class DictObsActionWrapper(Wrapper):
    """Give dictionary observations, and take dictionary actions."""

    def __init__(self, env, obs_key="wrapped", action_key="wrapped"):
        super().__init__(env)

        # Build observation space
        self.wrapped_obs = False
        if not isinstance(self.env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({obs_key: self.env.observation_space})
            self.wrapped_obs = True

        self.wrapped_action = False
        if not isinstance(self.action_space, gym.spaces.Dict):
            self.action_space = gym.spaces.Dict({action_key: self.action_space})
            self.wrapped_action = True

        self._env = env
        self.obs_key = obs_key
        self.action_key = action_key

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action: Dict[str, torch.Tensor]):
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def action(self, action):
        if self.wrapped_action:
            return action[self.action_key]
        else:
            return action

    def observation(self, observation):
        if self.wrapped_obs:
            return {self.obs_key: observation}
        else:
            return observation


# Advantage of doing this in wrapper: simple, maintains some diversity of difficulty.
# Disadvantage: the rate of update of difficulty (per global env step) depends on the number of workers.
# TODO: consider correcting for that by passing in the number of workers here.
class CurriculumWrapper(Wrapper):
    def __init__(self, env, task_difficulty_update, meta_difficulty_update):
        super().__init__(env)
        self._env = env
        self.difficulties = defaultdict(float)
        self.task_difficulty_update = task_difficulty_update
        self.meta_difficulty_update = meta_difficulty_update
        self.meta_difficulty = 0.0

    def step(self, action: Dict[str, torch.Tensor]):
        observation, reward, done, info = self.env.step(action)
        if done:
            task = AvalonTask[info["task"]]
            update_step = self.task_difficulty_update * np.random.uniform()
            if info["success"] == 1:
                self.difficulties[task] += update_step
                self.meta_difficulty += self.meta_difficulty_update
            elif info["success"] == 0:
                self.difficulties[task] -= update_step
                self.meta_difficulty -= self.meta_difficulty_update
            else:
                assert False, info["success"]
            self.difficulties[task] = max(min(self.difficulties[task], 1.0), 0.0)
            self.meta_difficulty = max(min(self.meta_difficulty, 1.0), 0.0)
            self._env.set_task_difficulty(task, self.difficulties[task])
            self._env.set_meta_difficulty(self.meta_difficulty)
        return observation, reward, done, info


class PixelObsWrapper(gym.ObservationWrapper):
    """Render state-based envs to pixels and use that as the observation."""

    def __init__(self, env):
        super().__init__(env)
        pixels = self.env.render(mode="rgb_array")
        # Some envs (eg CartPole-v1) won't render without being reset first.
        if pixels is None:
            self.env.reset()
            pixels = self.env.render(mode="rgb_array")
        self.observation_space = spaces.Box(shape=pixels.shape, low=0, high=255, dtype=pixels.dtype)
        self._env = env

    def observation(self, observation):
        obs = self.env.render(mode="rgb_array")
        return obs


class ImageTransformWrapper(gym.ObservationWrapper):
    def __init__(self, env, key, resolution=None, greyscale=False):
        """Transform a uint8 image into the format expected by a model."""
        super().__init__(env)
        self._env = env
        noop = transforms.Lambda(lambda x: x)
        self.transform = transforms.Compose(
            [
                # input is numpy (h w c)
                transforms.ToPILImage(),
                transforms.Grayscale() if greyscale else noop,
                transforms.Resize((resolution, resolution)) if resolution else noop,
                # Converts to tensor and from [0,255] to [0,1]
                # Output is (c h w)
                transforms.ToTensor(),
                # Convert that range to [-0.5, 0.5]
                transforms.Lambda(lambda x: x - 0.5),
            ]
        )

        assert isinstance(env.observation_space, spaces.Dict)
        space = env.observation_space[key]

        assert isinstance(space, spaces.Box)
        assert np.all(space.high == 255)
        assert np.all(space.low == 0)
        assert space.dtype == np.uint8, space.dtype
        assert len(space.shape) == 3
        if space.shape[2] in (1, 3):
            self.channels = space.shape[2] if not greyscale else 1
        else:
            assert False
        self.res = resolution if resolution else space.shape[1]
        new_space = spaces.Box(shape=(self.channels, self.res, self.res), low=-0.5, high=0.5, dtype=np.float32)
        self.observation_space = copy.copy(env.observation_space)
        self.observation_space[key] = new_space
        self.key = key

    def observation(self, obs):
        image = obs[self.key]
        image = self.transform(image).numpy()
        assert image.shape == (self.channels, self.res, self.res), image.shape
        obs[self.key] = image
        return obs


class ClipActionWrapper(gym.ActionWrapper):
    """Clip actions in Box envs that are out of bounds."""

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Dict)
        super().__init__(env)

    def action(self, action):
        for k, space in self.action_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                action[k] = np.clip(action[k], space.low, space.high)
        return action


class DiscreteActionToIntWrapper(gym.ActionWrapper):
    """Mujoco (maybe all gym envs?) only take ints for discrete actions.
    My code generates (arrays of) floats (this is what torch.distributions.Bernoulli.sample() gives).

    So for now I'll leave them in floats in my code, and convert them in this wrapper."""

    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = int(action.item())
        return action


# Fix sampling from the gym.spaces.Discrete to return a one-hot action vector.
class OneHotDiscrete(gym.spaces.Discrete):
    def sample(self):
        sample = self.np_random.randint(self.n)
        return F.one_hot(torch.tensor(sample), num_classes=self.n).numpy()


class OneHotMultiDiscrete(gym.spaces.MultiDiscrete):
    def __init__(self, nvec: list[int], dtype=np.int64, seed=None):
        assert np.issubdtype(dtype, (int, np.integer))
        self.max_categories = max(nvec)
        super().__init__(nvec, dtype, seed)

    def sample(self):
        sample = super().sample()
        # sample = self.np_random.randint(self.n)
        return F.one_hot(torch.tensor(sample), num_classes=self.max_categories).numpy()


class OneHotActionWrapper(gym.ActionWrapper):
    """Convert all discrete spaces to a OneHotMultiDiscrete space.
    My code will generate one-hot actions with shape (num_actions, num_categories),
    and this wrapper will convert to integer actions of shape necessary for each space type.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        self._env = env

        self.action_space = copy.deepcopy(env.action_space)
        for k, space in self.action_space.items():
            if isinstance(space, gym.spaces.Discrete):
                # self.action_space[k] = OneHotDiscrete(space.n)
                self.action_space[k] = OneHotMultiDiscrete([space.n])
            if isinstance(space, gym.spaces.MultiDiscrete):
                self.action_space[k] = OneHotMultiDiscrete(space.nvec)
            if isinstance(space, gym.spaces.MultiBinary):
                self.action_space[k] = OneHotMultiDiscrete([2] * space.n)

    def action(self, action_dict):
        out = copy.copy(action_dict)
        for k, space in self._env.action_space.items():
            action = action_dict[k]
            if isinstance(action, torch.Tensor):
                action = action.numpy()

            if isinstance(space, gym.spaces.Discrete):
                assert action.shape == (1, space.n)
                out[k] = np.argmax(action[0]).item()
            elif isinstance(space, gym.spaces.MultiDiscrete):
                # TODO: comment this assert out for perf
                assert action.shape == (len(space.nvec), max(space.nvec))
                out[k] = np.argmax(action, axis=-1)
            elif isinstance(space, gym.spaces.MultiBinary):
                assert action.shape == (space.n, 2)
                out[k] = np.argmax(action, axis=-1)
        return out


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ElapsedTimeWrapper(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

        assert max_episode_steps is not None
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.observation_space = copy.copy(self.env.observation_space)
        self.observation_space["elapsed_time"] = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def observation(self, observation):
        assert self._elapsed_steps <= self._max_episode_steps
        observation["elapsed_time"] = np.array(((self._elapsed_steps / self._max_episode_steps) - 0.5) * 2)
        return observation

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        return self.observation(observation), reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        observation = self.env.reset(**kwargs)
        return self.observation(observation)


class NormalizeActions(gym.ActionWrapper):
    """Adapts envs with arbitrary action space bounds to take actions with a standard (-1, 1) range.

    Does not adapt non-Box spaces or dimensions that have non-finite bounds."""

    def __init__(self, env):
        super().__init__(env)
        self._env = env
        self.bounds = {}
        assert isinstance(env.action_space, gym.spaces.Dict)
        # TODO: don't modify the base action space.
        self.action_space = self._env.action_space
        for k, space in env.action_space.items():
            if isinstance(space, gym.spaces.Box):
                mask = np.logical_and(np.isfinite(space.low), np.isfinite(space.high))
                if any(mask == 0):
                    print(f"Warning: action space '{k}' has non-finite bounds. Action rescaling will not be applied.")
                low = np.where(mask, space.low, -1)
                high = np.where(mask, space.high, 1)
                self.bounds[k] = (mask, low, high)

                new_low = np.where(mask, -np.ones_like(low), low)
                new_high = np.where(mask, np.ones_like(low), high)
                self.action_space[k] = gym.spaces.Box(new_low, new_high, dtype=np.float32)

    def action(self, action):
        for k, v in action.items():
            if isinstance(self.action_space[k], gym.spaces.Box):
                mask, low, high = self.bounds[k]
                new = (v + 1) / 2 * (high - low) + low
                action[k] = np.where(mask, new, v)
        return action


class ScaleRewards(gym.RewardWrapper):
    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale
