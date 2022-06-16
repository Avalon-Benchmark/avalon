import copy
import signal
from collections import OrderedDict
from collections import defaultdict
from typing import Dict

import attr
import gym
import numpy as np
import numpy.typing as npt
import torch
import torch.distributions
import torch.nn.functional as F
from gym import Wrapper
from gym import spaces
from gym.spaces import flatdim
from psutil import NoSuchProcess
from psutil import Process

from agent.godot_gym import AvalonGodotEnvWrapper
from agent.godot_gym import GodotEnvParams
from agent.godot_gym import GodotObsTransformWrapper
from agent.godot_gym import ScaleAndSquashAction
from agent.godot_gym import TrainingProtocolChoice
from agent.godot_gym import VRActionType
from agent.torchbeast.core.environment import Environment
from agent.torchbeast.core.vtrace import VTraceFromLogitsReturns
from agent.torchbeast.core.vtrace import from_importance_weights
from common.log_utils import logger
from contrib.serialization import Serializable
from datagen.world_creation.constants import AvalonTask


def get_dict_action_space(num_actions: int):
    # return spaces.Dict({"discrete": spaces.Discrete(num_actions)})
    return VRActionType.to_gym_space()


class DictifyAtari(gym.Wrapper):
    def __init__(self, env: gym.Env):
        """Make action space a dictionary"""
        gym.Wrapper.__init__(self, env)
        # self.observation_space = spaces.Dict({"rgb": self.env.observation_space})
        self.action_space = spaces.Dict({"discrete": self.env.action_space})

    def step(self, action: np.ndarray):
        """Repeat action, sum reward, and max over last observations."""
        action_dict = spaces.utils.unflatten(self.action_space, action)
        obs, reward, done, info = self.env.step(action_dict["discrete"])

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def wrap_atari(env):
    return DictifyAtari(env)


def vtrace_from_dist(
    behavior_policy_dist: torch.distributions.Distribution,
    target_policy_dist: torch.distributions.Distribution,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
):
    """V-trace for policies with defined distribution."""

    target_action_log_probs = target_policy_dist.log_prob(actions)
    behavior_action_log_probs = behavior_policy_dist.log_prob(actions)
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


def compute_policy_gradient_loss_from_dist(dist: torch.distributions.Distribution, actions, advantages):
    log_prob = -dist.log_prob(actions).view_as(advantages)
    return torch.sum(log_prob * advantages.detach())


def flatten_tensor(space, x):
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``gym.spaces``.
    """
    if isinstance(space, spaces.Box):
        return x
    elif isinstance(space, spaces.Discrete):
        onehot = F.one_hot(x, num_classes=space.n).float()
        return onehot
    elif isinstance(space, spaces.Tuple):
        return torch.cat([flatten_tensor(s, x_part) for x_part, s in zip(x, space.spaces)], dim=-1)
    elif isinstance(space, spaces.Dict):
        return torch.cat([flatten_tensor(s, x[key]) for key, s in space.spaces.items()], dim=-1)
    elif isinstance(space, spaces.MultiBinary):
        return x
    elif isinstance(space, spaces.MultiDiscrete):
        return x
    else:
        raise NotImplementedError


def unflatten_tensor(space: spaces.Space, x: torch.Tensor):
    """Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``gym.spaces``.
    """
    batch_dims = x.shape[:-1]
    if isinstance(space, spaces.Box):
        return x.reshape(batch_dims + space.shape)
    elif isinstance(space, spaces.Discrete):
        return torch.argmax(x, dim=-1)
    elif isinstance(space, spaces.Tuple):
        dims = [flatdim(s) for s in space.spaces]
        list_flattened = torch.split(x, dims, dim=-1)
        list_unflattened = [unflatten_tensor(s, flattened) for flattened, s in zip(list_flattened, space.spaces)]
        return tuple(list_unflattened)
    elif isinstance(space, spaces.Dict):
        dims = [flatdim(s) for s in space.spaces.values()]
        list_flattened = torch.split(x, dims, dim=-1)
        list_unflattened = [
            (key, unflatten_tensor(s, flattened)) for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
        return OrderedDict(list_unflattened)
    elif isinstance(space, spaces.MultiBinary):
        return x.reshape(batch_dims + space.shape)
    elif isinstance(space, spaces.MultiDiscrete):
        return x.reshape(batch_dims + space.shape)
    else:
        raise NotImplementedError


def _transform_observation_rgbd(x: npt.NDArray) -> npt.NDArray:
    x = x[::-1]
    return np.transpose(x, axes=(2, 0, 1))


# Proprioception will go to this many rows of the image
PROPRIOCEPTION_ROW_COUNT = 1
IS_PROPRIOCEPTION_USED = True


class GodotToPyTorch(GodotObsTransformWrapper):
    """
    Image shape to channels x weight x height, add extra channels for proprioception
    """

    def __init__(self, env):
        super().__init__(env)
        self.transforms["rgbd"] = _transform_observation_rgbd
        new_shape = self.observation_space.spaces["rgbd"].shape
        proprioception_space = copy.deepcopy(self.observation_space)
        proprioception_space.spaces.pop("rgb", None)
        proprioception_space.spaces.pop("rgbd", None)
        proprioception_space.spaces.pop("depth", None)
        self.proprioception_dim = 0
        if len(proprioception_space.spaces) > 0:
            self.proprioception_dim = flatdim(proprioception_space)
            assert self.proprioception_dim <= PROPRIOCEPTION_ROW_COUNT * new_shape[2], "Too much proprioception data!"

        if IS_PROPRIOCEPTION_USED:
            new_shape = (new_shape[0] + 1, new_shape[1], new_shape[2])

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=np.uint8,
        )

    def observation(self, observation):
        normalized_observation = super().observation(observation)
        rgbd_data = normalized_observation["rgbd"]
        prop_data_flat = np.concatenate(
            [v for k, v in normalized_observation.items() if k not in {"rgbd", "rgb", "depth"}]
        )
        prop_data_flat = np.clip(prop_data_flat, a_min=-1, a_max=1) * 127.5 + 127.5
        prop_data_flat = prop_data_flat.astype(np.uint8)
        prop_data_reshaped = np.zeros(self.observation_space.shape[1:], dtype=np.uint8) + 127
        row_count = self.proprioception_dim // self.observation_space.shape[1] + 1
        prop_data_idx = 0
        for row in range(PROPRIOCEPTION_ROW_COUNT):
            row_len = min(prop_data_flat.shape[0] - prop_data_idx, self.observation_space.shape[2])
            prop_data_reshaped[row + 1, :row_len] = prop_data_flat[prop_data_idx:row_len]
            prop_data_idx += row_len

        return np.concatenate([rgbd_data, np.expand_dims(prop_data_reshaped, axis=0)], axis=0)


def obs_to_frame(observation: torch.Tensor, include_depth: bool = True) -> torch.Tensor:
    if IS_PROPRIOCEPTION_USED:
        assert (
            observation.shape[-3] == 5
        ), f"Got observation shape {observation.shape} but expected channels dim to be 5"
        return observation[..., :-1, :, :]
    else:
        return observation


def obs_to_proprioception(observation: torch.Tensor) -> torch.Tensor:
    if IS_PROPRIOCEPTION_USED:
        assert (
            observation.shape[-3] == 5
        ), f"Got observation shape {observation.shape} but expected channels dim to be 5"
        batch_shape = observation.shape[:-3]
        return observation[..., 4, :PROPRIOCEPTION_ROW_COUNT, :].view(batch_shape + (-1,))
    else:
        # slicing to return a tensor that has a zero dim
        return observation[..., 0, 0, :0]


class FlattenAction(gym.ActionWrapper):
    def action(self, action: np.ndarray):
        return spaces.utils.unflatten(self.action_space, action.flatten())

    def reverse_action(self, action):
        return spaces.utils.flatten(self.action_space, action)


@attr.s(auto_attribs=True, collect_by_mro=True)
class WrappedGodotEnvParams(Serializable):
    env_params: GodotEnvParams
    task_difficulty_update: float
    meta_difficulty_update: float


def godot_config_from_flags(
    flags, random_int: int, is_fixed_generator: bool = False, num_fixed_worlds_per_task: int = 6
) -> WrappedGodotEnvParams:
    num_gpus = torch.cuda.device_count()
    return WrappedGodotEnvParams(
        env_params=GodotEnvParams(
            random_int=random_int,
            gpu_id=(random_int % num_gpus),
            energy_cost_coefficient=flags.energy_cost_coefficient,
            max_frames=flags.max_frames,
            fixed_world_max_difficulty=flags.fixed_world_max_difficulty,
            is_fixed_generator=is_fixed_generator,
            num_fixed_worlds_per_task=num_fixed_worlds_per_task,
            training_protocol=TrainingProtocolChoice[flags.training_protocol.upper()],
            is_task_curriculum_used=not flags.is_task_curriculum_disabled,
            is_meta_curriculum_used=not flags.is_meta_curriculum_disabled,
        ),
        task_difficulty_update=flags.task_difficulty_update,
        meta_difficulty_update=flags.meta_difficulty_update,
    )


def create_godot_env(params: WrappedGodotEnvParams):
    logger.info(f"Creating godot env with params {params}")
    env = AvalonGodotEnvWrapper(params.env_params)
    # env = WarpFrame(env, grayscale=False, dict_space_key="rgb")
    env = ScaleAndSquashAction(env)
    env = FlattenAction(env)
    env = GodotToPyTorch(env)
    if not params.env_params.is_fixed_generator:
        env = CurriculumWrapper(
            env,
            task_difficulty_update=params.task_difficulty_update,
            meta_difficulty_update=params.meta_difficulty_update,
        )
    return env


def wrap_evaluation_godot_env(env) -> Environment:
    env = ScaleAndSquashAction(env)
    env = FlattenAction(env)
    env = GodotToPyTorch(env)
    return Environment(env)


def force_cudnn_initialization():
    s = 32
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


# TODO: continue stopping children and looking for children until there are no new ones
def _destroy_process_tree(parent_pid: int, final_signal: int = signal.SIGKILL):
    """
    First stop everything, then kill everything
    Inspired by this: http://stackoverflow.com/a/3211182/1380514
    """
    for sig in (signal.SIGSTOP, final_signal):
        try:
            parent = Process(parent_pid)
        except NoSuchProcess:
            return
        children = parent.children(recursive=True)
        try:
            parent.send_signal(sig)
        except NoSuchProcess:
            pass
        for process in children:
            try:
                process.send_signal(sig)
            except NoSuchProcess:
                pass


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
