# mypy: ignore-errors
# We ignore all types in this file because typing in `gym==0.25.*` is a mess.
# Will fix when we update gym to `>=0.26` or wait for the `1.0` release.

import copy
from collections import defaultdict
from collections import deque
from typing import Any
from typing import Callable
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import gym
import numpy as np
import torch
from gym import Wrapper
from gym import spaces
from gym.spaces import Box
from loguru import logger
from numpy.typing import DTypeLike
from numpy.typing import NDArray
from torch.nn import functional as F
from torchvision import transforms
from tree import map_structure

# All wrappers should operate only on ndarrays, no torch Tensors!
ArrayType = np.typing.NDArray
DictActionType = dict[str, ArrayType]
DictObservationType = dict[str, ArrayType]

GenericObservationType = TypeVar("GenericObservationType")
GenericActionType = TypeVar("GenericActionType")


class DictObsActionWrapper(Wrapper[DictObservationType, DictActionType]):
    """Give dictionary observations, and take dictionary actions."""

    def __init__(
        self,
        env: gym.Env[Union[DictObservationType, NDArray], Union[DictObservationType, NDArray]],
        obs_key: str = "wrapped",
        action_key: str = "wrapped",
    ) -> None:
        super().__init__(env)

        # Build observation space
        self.wrapped_obs = False
        if not isinstance(self.env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({obs_key: self.env.observation_space})
            self.wrapped_obs = True

        self.wrapped_action = False
        self.action_space: gym.spaces.Space
        if not isinstance(self.action_space, gym.spaces.Dict):
            self.action_space = gym.spaces.Dict({action_key: self.action_space})
            self.wrapped_action = True

        self._env = env
        self.obs_key = obs_key
        self.action_key = action_key

    def reset(self, *args, **kwargs) -> DictObservationType:  # type: ignore[no-untyped-def]
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)

    def step(self, action: DictActionType) -> tuple[DictObservationType, float, bool, dict]:
        unwrapped_action = self.action(action)
        observation, reward, done, info = self.env.step(unwrapped_action)
        return self.observation(observation), reward, done, info

    def action(self, action: DictActionType) -> Union[NDArray, DictActionType]:
        if self.wrapped_action:
            return action[self.action_key]
        else:
            return action

    def observation(self, observation: Union[NDArray, DictObservationType]) -> DictObservationType:
        if self.wrapped_obs:
            assert isinstance(observation, np.ndarray)
            return {self.obs_key: observation}
        else:
            return observation


class PixelObsWrapper(gym.ObservationWrapper):
    """Render state-based envs to pixels and use that as the observation."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        pixels: NDArray = self.env.render(mode="rgb_array")
        # Some envs (eg CartPole-v1) won't render without being reset first.
        if pixels is None:
            self.env.reset()
            pixels = self.env.render(mode="rgb_array")
        self.observation_space = spaces.Box(shape=pixels.shape, low=0, high=255, dtype=pixels.dtype)
        self._env = env

    def observation(self, observation: NDArray) -> NDArray:
        obs: NDArray = self.env.render(mode="rgb_array")
        return obs


class ImageTransformWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, key: str, resolution: Optional[int] = None, greyscale: bool = False) -> None:
        """Transform a uint8 image into the format expected by a model.

        We actually leave images as uint8s until training, for the 4x lower data size."""
        super().__init__(env)
        self._env = env
        noop = transforms.Lambda(lambda x: x)
        self.transform = transforms.Compose(
            [
                # input is numpy (h w c)
                transforms.ToPILImage(),
                transforms.Grayscale() if greyscale else noop,
                transforms.Resize((resolution, resolution)) if resolution else noop,
                # transforms.ToTensor(), # this converts hwc to chw, and from 0-255 to 0-1
                transforms.PILToTensor(),  # this converts hwc to chw, without scaling
                # Convert that range to [-0.5, 0.5]
                # transforms.Lambda(lambda x: x - 0.5),
            ]
        )

        observation_space = env.observation_space
        assert isinstance(observation_space, spaces.Dict)
        space = observation_space[key]

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
        # new_space = spaces.Box(shape=(self.channels, self.res, self.res), low=-0.5, high=0.5, dtype=np.float32)
        new_space = spaces.Box(shape=(self.channels, self.res, self.res), low=0, high=255, dtype=np.uint8)
        self.observation_space = copy.copy(observation_space)
        self.observation_space.spaces[key] = new_space
        self.key = key

    def observation(self, obs: Dict[str, NDArray]) -> Dict[str, NDArray]:
        image = obs[self.key]
        image = self.transform(image).numpy()
        assert image.shape == (self.channels, self.res, self.res), image.shape
        obs[self.key] = image
        return obs


class ClipActionWrapper(gym.ActionWrapper):
    """Clip actions in Box envs that are out of bounds."""

    def __init__(self, env: gym.Env) -> None:
        assert isinstance(env.action_space, gym.spaces.Dict)
        super().__init__(env)

    def action(self, action: DictActionType) -> DictActionType:
        for k, space in self.action_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                action[k] = np.clip(action[k], space.low, space.high)
        return action

    def reverse_action(self, action: DictActionType) -> DictActionType:
        raise NotImplementedError


class DiscreteActionToIntWrapper(gym.ActionWrapper):
    """Mujoco (maybe all gym envs?) only take ints for discrete actions.
    My code generates (arrays of) floats (this is what torch.distributions.Bernoulli.sample() gives).

    So for now I'll leave them in floats in my code, and convert them in this wrapper."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def action(self, action: np.floating) -> Union[np.floating, int]:
        if isinstance(self.action_space, gym.spaces.Discrete):
            return int(action.item())
        else:
            return action


class OneHotDiscrete(gym.spaces.Discrete):
    def sample(self, mask: Optional[NDArray] = None) -> NDArray:
        assert mask is None
        sample = self.np_random.randint(self.n)
        return F.one_hot(torch.from_numpy(sample), num_classes=self.n).numpy()


class OneHotMultiDiscrete(gym.spaces.MultiDiscrete):
    def __init__(self, nvec: List[int], dtype: DTypeLike = np.float32) -> None:
        # the dtype is default float32 since that's what torch.OneHotCategorical.sample() returns,
        # for some reason (presumably to make backprop work).
        # This is kinda weird tho because then it makes nvec a float32 too.. this superclass really expects an int
        # But we need self.dtype to be the type that we pass around during rollouts,
        # since this is where it gets the dtype to use for storing the actions.
        # One option is to make self.dtype be an integer type, use integers all thru rollouts/inference,
        # and floats in training. Confusing though.
        # Another is to always use float, and just patch this class to make it handle that properly.
        # That's the approach I've gone with for now.
        self.max_categories = max(nvec)
        super().__init__(nvec=nvec, dtype=dtype)

    def sample(self, mask: Optional[NDArray] = None) -> NDArray:
        assert mask is None
        sample = (self.np_random.random(self.nvec.shape) * self.nvec).astype(np.int32)
        return (
            F.one_hot(torch.from_numpy(sample).long(), num_classes=self.max_categories).to(dtype=torch.float32).numpy()
        )


class OneHotActionWrapper(gym.ActionWrapper):
    """Convert all discrete spaces to a OneHotMultiDiscrete space.
    My code will generate one-hot actions with shape (num_actions, num_categories),
    and this wrapper will convert to integer actions of shape necessary for each space type.

    NOTE: This probably won't work quite right for a true MultiDiscrete space with diff number of classes
    for each dimension. Since we'll use max(num_classes) as the one-hot vector length for all.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        self._env = env

        self.action_space = copy.deepcopy(env.action_space)
        for k, space in self.action_space.spaces.items():
            if isinstance(space, gym.spaces.Discrete):
                # self.action_space[k] = OneHotDiscrete(space.n)
                self.action_space.spaces[k] = OneHotMultiDiscrete([space.n])
            if isinstance(space, gym.spaces.MultiDiscrete):
                self.action_space.spaces[k] = OneHotMultiDiscrete(space.nvec)
                assert len(set(space.nvec)) == 1, "not sure we support heterogenous MultiDiscrete spaces properly."
            if isinstance(space, gym.spaces.MultiBinary):
                self.action_space.spaces[k] = OneHotMultiDiscrete([2] * space.n)

    def action(self, action_dict: DictActionType) -> dict[str, Any]:
        out: dict[str, Any] = copy.copy(action_dict)
        for k, space in self._env.action_space.spaces.items():
            action = action_dict[k]
            assert isinstance(action, np.ndarray)

            if isinstance(space, gym.spaces.Discrete):
                assert action.shape == (1, space.n)
                out[k] = np.argmax(action[0]).item()
            elif isinstance(space, gym.spaces.MultiDiscrete):
                # assert action.shape == (len(space.nvec), max(space.nvec))
                out[k] = np.argmax(action, axis=-1)
            elif isinstance(space, gym.spaces.MultiBinary):
                assert action.shape == (space.n, 2)
                out[k] = np.argmax(action, axis=-1)
        return out

    def reverse_action(self, action: Dict[str, NDArray]):
        raise NotImplementedError


class TimeLimit(gym.Wrapper[GenericObservationType, GenericActionType]):
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None) -> None:
        super().__init__(env)
        self._max_episode_steps: Optional[int] = max_episode_steps
        self._elapsed_steps: Optional[int] = None

    def step(self, action: GenericActionType) -> tuple[GenericObservationType, float, bool, dict]:
        assert self._max_episode_steps is not None, "max_episode_steps must be set"
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, *args, **kwargs) -> GenericObservationType:
        self._elapsed_steps = 0
        return self.env.reset(*args, **kwargs)


class ElapsedTimeWrapper(gym.Wrapper[DictObservationType, GenericActionType]):
    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None) -> None:
        super().__init__(env)
        self._max_episode_steps: Optional[int] = max_episode_steps
        self._elapsed_steps: Optional[int] = None

        assert self._max_episode_steps is not None
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self.observation_space = copy.copy(self.env.observation_space)
        self.observation_space["elapsed_time"] = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def observation(self, observation: DictObservationType) -> DictObservationType:
        assert self._elapsed_steps <= self._max_episode_steps
        observation["elapsed_time"] = np.array(((self._elapsed_steps / self._max_episode_steps) - 0.5) * 2)
        return observation

    def step(self, action: GenericActionType) -> tuple[DictObservationType, float, bool, dict]:
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        return self.observation(observation), reward, done, info

    def reset(self, *args, **kwargs) -> DictObservationType:
        self._elapsed_steps = 0
        observation = self.env.reset(*args, **kwargs)
        return self.observation(observation)


class NormalizeActions(gym.ActionWrapper):
    """Adapts envs with arbitrary action space bounds to take actions with a standard (-1, 1) range.

    Does not adapt non-Box spaces or dimensions that have non-finite bounds."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._env = env
        self.bounds = {}
        assert isinstance(env.action_space, gym.spaces.Dict)
        self.action_space = copy.deepcopy(env.action_space)
        for k, space in env.action_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                mask = np.logical_and(np.isfinite(space.low), np.isfinite(space.high))
                if any(mask == 0):
                    logger.warning(f"Action space '{k}' has non-finite bounds. Action rescaling will not be applied.")
                low = np.where(mask, space.low, -1)
                high = np.where(mask, space.high, 1)
                self.bounds[k] = (mask, low, high)

                new_low = np.where(mask, -np.ones_like(low), low)
                new_high = np.where(mask, np.ones_like(low), high)
                self.action_space[k] = gym.spaces.Box(new_low, new_high, dtype=np.float32)

    def action(self, action: DictActionType) -> DictActionType:
        for k, v in action.items():
            if isinstance(self.action_space[k], gym.spaces.Box):
                mask, low, high = self.bounds[k]
                new = (v + 1) / 2 * (high - low) + low
                action[k] = np.where(mask, new, v)
        return action

    def reverse_action(self, action: DictActionType) -> DictActionType:
        raise NotImplementedError


class ScaleRewards(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale: float = 1.0, func: Callable[[float], float] = lambda x: x) -> None:
        super().__init__(env)
        self.scale = scale
        self.func = func

    def reward(self, reward: float) -> float:
        return self.func(reward) * self.scale


class ActionRepeat(gym.Wrapper[GenericObservationType, GenericActionType]):
    def __init__(self, env: gym.Env, amount: int) -> None:
        super().__init__(env)
        self._amount = amount

    def step(self, action: GenericActionType) -> tuple[GenericObservationType, float, bool, dict]:
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        # Note: this will just return the info of the most recent step
        return obs, total_reward, done, info


class DictFrameStack(gym.ObservationWrapper):
    """Frame-stack all observations in a Dict space.

    Shapes will be converted from (c, *) to (c * num_stack, *). So for images, this will only work with (c,h,w).
    For the first steps after a reset, where there aren't `num_stack` prior steps,
    the observation at t=0 will be repeated back in time to fill.
    """

    def __init__(self, env: gym.Env, num_stack: int) -> None:
        super().__init__(env)
        self._env = env
        self.num_stack = num_stack

        assert isinstance(env.observation_space, gym.spaces.Dict)
        new_observation_space: dict[str, gym.Space] = {}
        for k, space in env.observation_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                low = np.repeat(space.low, num_stack, axis=0)
                high = np.repeat(space.high, num_stack, axis=0)
                new_observation_space[k] = Box(low=low, high=high, dtype=space.dtype)
            else:
                assert False, (k, space)

        self.observation_space = gym.spaces.Dict(new_observation_space)
        self.history: dict[str, Deque] = defaultdict(lambda: deque(maxlen=self.num_stack))

    def add_to_history(self, obs: dict[str, NDArray]) -> None:
        for k, v in obs.items():
            self.history[k].append(v)

    def reset(self, **kwargs):
        self.history: dict[str, Deque] = defaultdict(lambda: deque(maxlen=self.num_stack))
        obs = self.env.reset(**kwargs)
        # We'll initialize the buffer with copies of the first frame
        [self.add_to_history(obs) for _ in range(self.num_stack - 1)]
        return self.observation(obs)

    def observation(self, obs: Dict[str, NDArray]) -> Dict[str, NDArray]:
        self.add_to_history(obs)
        out: dict[str, NDArray] = {}
        for k, v in obs.items():
            assert len(self.history[k]) == self.num_stack
            # images should be chw, and scalars/vectors should be stacked on dim 0 also
            out[k] = np.concatenate(self.history[k], axis=0)
        return out


class Torchify(gym.Wrapper):
    """Allow the client to use Torch tensors instead of the usual numpy arrays."""

    def to_torch(self, x):
        return map_structure(lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x, x)

    def to_numpy(self, x):
        return map_structure(lambda x: x.numpy() if isinstance(x, torch.Tensor) else x, x)

    def step(self, action):
        action = self.to_numpy(action)
        obs, reward, done, info = self.env.step(action)
        return self.to_torch(obs), float(reward), bool(done), self.to_torch(info)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self.to_torch(obs)
        return obs


class RecordEpisodeStatistics(gym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``

    Frame skipping is currently **NOT** taken into account.

    I use this instead of gym.RecordEpisodeStatistics because that has a nested info key that I didn't
    want to deal with supporting.
    And I use this wrapper instead of building it into the worker, because we often want to record the reward
    before any reward scaling is applied.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_return: float = 0
        self.episode_length: int = 0

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_return = 0
        self.episode_length = 0
        return observations

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_return += reward
        self.episode_length += 1
        info["cumulative_episode_return"] = self.episode_return
        info["cumulative_episode_length"] = self.episode_length
        return obs, reward, done, info
