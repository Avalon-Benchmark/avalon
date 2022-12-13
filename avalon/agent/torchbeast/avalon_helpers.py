import copy
import json
import os.path
import pickle
import signal
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List

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

from avalon.agent.godot.godot_gym import LEVEL_OUTPUT_PATH
from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import GodotObsTransformWrapper
from avalon.agent.godot.godot_gym import ScaleAndSquashAction
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.agent.godot.godot_gym import VRAction
from avalon.agent.godot.godot_gym import task_groups_from_training_protocol
from avalon.agent.torchbeast.core.environment import Environment
from avalon.agent.torchbeast.core.vtrace import VTraceFromLogitsReturns
from avalon.agent.torchbeast.core.vtrace import from_importance_weights
from avalon.common.log_utils import logger
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.datagen.world_creation.constants import AvalonTask
from avalon.datagen.world_creation.constants import int_to_avalon_task
from avalon.datagen.world_creation.world_generator import get_world_params_for_task_groups


def get_dict_action_space(num_actions: int):
    # return spaces.Dict({"discrete": spaces.Discrete(num_actions)})
    return VRAction.to_gym_space()


class DictifyAtari(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
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

    def __init__(self, env) -> None:
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

    def reverse_action(self, action: np.ndarray):
        return spaces.utils.flatten(self.action_space, action)


@attr.s(auto_attribs=True, collect_by_mro=True)
class WrappedGodotEnvironmentParams:
    env_params: GodotEnvironmentParams
    is_resume_allowed: bool = True


TORCHBEAST_ENV_LOGS_PATH = Path(f"{FILESYSTEM_ROOT}/torchbeast/logs")


def godot_config_from_flags(flags, env_index: int = 0) -> WrappedGodotEnvironmentParams:
    num_gpus = torch.cuda.device_count()
    # TODO make less bad?
    if flags.mode == "test":
        random_int = 2000
        goal_progress_path = TORCHBEAST_ENV_LOGS_PATH
    else:
        random_int = env_index + flags.env_seed
        goal_progress_path = None
    return WrappedGodotEnvironmentParams(
        env_params=GodotEnvironmentParams(
            gpu_id=(env_index % num_gpus),
            env_index=env_index,
            seed=random_int,
            env_count=flags.num_servers,
            energy_cost_coefficient=flags.energy_cost_coefficient,
            fixed_world_max_difficulty=flags.fixed_world_max_difficulty,
            mode=flags.mode,
            training_protocol=TrainingProtocolChoice[flags.training_protocol.upper()],
            is_task_curriculum_used=not flags.is_task_curriculum_disabled,
            is_meta_curriculum_used=flags.is_meta_curriculum_enabled,
            is_debugging_godot=flags.mode == "test",
            head_pitch_coefficient=flags.head_pitch_coefficient,
            head_roll_coefficient=flags.head_roll_coefficient,
            goal_progress_path=goal_progress_path,
            energy_cost_aggregator=flags.energy_cost_aggregator,
            is_video_logged=flags.save_test_videos,
            is_action_logged=flags.save_test_actions,
            task_difficulty_update=flags.task_difficulty_update,
            meta_difficulty_update=flags.meta_difficulty_update,
            fixed_worlds_load_from_path=Path(flags.fixed_world_path) if flags.fixed_world_path else None,
        ),
        is_resume_allowed=True,
    )


def get_num_test_worlds_from_flags(flags) -> int:
    params = godot_config_from_flags(flags).env_params
    if params.fixed_worlds_load_from_path:
        return len(list(params.fixed_worlds_load_from_path.glob("*")))
    task_groups = task_groups_from_training_protocol(params.training_protocol, params.is_meta_curriculum_used)
    difficulties = tuple(
        np.linspace(
            params.fixed_world_min_difficulty, params.fixed_world_max_difficulty, params.num_fixed_worlds_per_task
        )
    )
    world_params = get_world_params_for_task_groups(
        task_groups,
        difficulties,
        Path(LEVEL_OUTPUT_PATH),
        params.seed,
    )
    return len(world_params)


def get_goal_progress_files() -> List[Path]:
    return list(TORCHBEAST_ENV_LOGS_PATH.rglob("goal_progress.json"))


def get_avalon_test_scores(prefix: str = "test/") -> Dict[str, float]:
    goal_progress_files = get_goal_progress_files()

    results_by_task = defaultdict(lambda: defaultdict(list))
    for filename in goal_progress_files:
        with open(filename, "r") as f:
            run_data = json.load(f)
        final_data = run_data[-1]["log"]
        task = final_data["task"].lower()
        results_by_task[task]["episode_length"].append(len(run_data))
        results_by_task[task]["episode_reward"].append(sum(x["reward"] for x in run_data))
        results_by_task[task]["final_success"].append(final_data["success"])
    log_results: Dict[str, float] = {}
    for task, results in results_by_task.items():
        for result_key, result_values in results.items():
            log_results[f"{prefix}{task}/{result_key}"] = sum(result_values) / len(result_values)
            log_results[f"{prefix}{task}/num_episodes"] = len(result_values)
    return log_results


def create_godot_env(params: WrappedGodotEnvironmentParams):
    logger.info(f"Creating godot env with params {params}")
    env = AvalonEnv(params.env_params)
    # env = WarpFrame(env, grayscale=False, dict_space_key="rgb")
    env = ScaleAndSquashAction(env)
    env = FlattenAction(env)
    env = GodotToPyTorch(env)
    if not params.env_params.is_fixed_generator:
        env = CurriculumWrapper(
            env,
            task_difficulty_update=params.env_params.task_difficulty_update,
            meta_difficulty_update=params.env_params.meta_difficulty_update,
            is_resume_allowed=params.is_resume_allowed,
        )
    return env


def wrap_evaluation_godot_env(env) -> Environment:
    env = ScaleAndSquashAction(env)
    env = FlattenAction(env)
    env = GodotToPyTorch(env)
    return Environment(env)


def force_cudnn_initialization() -> None:
    s = 32
    dev = torch.device("cuda")
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


# TODO: continue stopping children and looking for children until there are no new ones
def _destroy_process_tree(parent_pid: int, final_signal: int = signal.SIGKILL) -> None:
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
    # TODO: deduplicate this
    def __init__(self, env, task_difficulty_update: float, meta_difficulty_update: float, is_resume_allowed: bool):
        super().__init__(env)
        self.difficulties = defaultdict(float)
        self.task_difficulty_update = task_difficulty_update
        self.meta_difficulty_update = meta_difficulty_update
        self.meta_difficulty = 0.0
        if is_resume_allowed:
            self.load_curriculum_state_from_file_if_exists()

    def step(self, action: Dict[str, torch.Tensor]):
        observation, reward, done, info = self.env.step(action)
        if done:
            task = AvalonTask[int_to_avalon_task[int(info["task"])]]
            update_step = self.task_difficulty_update  # * np.random.uniform()
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
            self.env.set_task_difficulty(task, self.difficulties[task])
            self.env.set_meta_difficulty(self.meta_difficulty)
            self.save_curriculum_state_to_file()
        return observation, reward, done, info

    def save_curriculum_state_to_file(self) -> None:
        with open(self.env.curriculum_save_path, "wb") as f:
            pickle.dump({"difficulties": self.difficulties, "meta_difficulty": self.meta_difficulty}, f)

    def load_curriculum_state_from_file_if_exists(self) -> None:
        filepath = self.env.curriculum_save_path
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.difficulties = data["difficulties"]
            self.meta_difficulty = data["meta_difficulty"]
            logger.info(f"Loaded curriculum from {filepath}")
