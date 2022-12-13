"""This is an alternative implementation of the rollout worker.
Used to verify the output of the primary rollout worker.
Intended to be simple enough to be able to readily verify correctness,
but still support all the features we want to test.

The main simplification is just in not handling partial batches.
It's also lacks the complexity from performance tuning.
"""
from collections import defaultdict
from multiprocessing.context import BaseContext
from typing import Optional
from typing import Type

import attr
import gym
import torch
from torch import Tensor

from avalon.agent.common.envs import build_env
from avalon.agent.common.params import Params
from avalon.agent.common.storage import TrajectoryStorage
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import StepData
from avalon.agent.common.util import postprocess_uint8_to_float


class DummyStorage(TrajectoryStorage):
    def __init__(self, params: Params, step_data_type: Type[StepData], num_workers: int) -> None:
        super().__init__(params, step_data_type, num_workers)
        self.storage = None

    def to_packed(self):
        assert self.storage is not None
        return self.step_data_type.batch_sequence_type(**self.storage)

    def reset(self) -> None:
        self.storage = None


class SimpleRolloutManager:
    def __init__(
        self,
        params: Params,
        num_workers: int,
        is_multiprocessing: bool,
        storage: TrajectoryStorage,
        obs_space: gym.spaces.Dict,
        model: Algorithm,
        rollout_device: torch.device,
        multiprocessing_context: BaseContext,
    ) -> None:
        # TODO: handle the context / fork/spawn method somehow
        vector_env_cls = gym.vector.AsyncVectorEnv if is_multiprocessing else gym.vector.SyncVectorEnv
        self.num_workers = num_workers
        self.params = params
        self.model = model
        self.rollout_device = rollout_device
        self.storage = storage

        params = attr.evolve(params, env_params=attr.evolve(params.env_params, env_count=num_workers))
        # the torchify wrapper doesn't work, they still get coerced to numpy in the VecEnv
        self.envs = vector_env_cls(
            [
                lambda i=i: build_env(
                    env_params=attr.evolve(params.env_params, env_index=params.env_params.env_index + i),
                    torchify=False,
                )
                for i in range(num_workers)
            ]
        )

        # We need to carry these between rollout batches
        obs_numpy = self.envs.reset()
        self.obs = {k: torch.from_numpy(v) * 0 for k, v in obs_numpy.items()}
        self.dones = torch.zeros((self.num_workers,), dtype=torch.bool, device=self.rollout_device)

    def run_rollout(
        self,
        num_steps: Optional[int] = None,
        num_episodes: Optional[int] = None,
        exploration_mode: str = "explore",
    ) -> None:
        storage = defaultdict(list)
        storage["observation"] = defaultdict(list)
        storage["action"] = defaultdict(list)

        assert num_episodes is None
        self.model.eval()
        to_run = torch.ones((self.num_workers,), dtype=torch.bool)
        for i in range(num_steps):
            with torch.no_grad():
                torch_obs = {k: v.to(device=self.rollout_device, non_blocking=True) for k, v in self.obs.items()}
                torch_dones = self.dones.to(device=self.rollout_device, non_blocking=True)
                torch_obs = postprocess_uint8_to_float(torch_obs, center=self.params.center_observations)
                actions, to_store = self.model.rollout_step(
                    torch_obs, torch_dones, to_run, exploration_mode=exploration_mode
                )

            assert self.params.obs_first is True
            for k, v in self.obs.items():
                storage["observation"][k].append(v)
            for k, v in actions.items():
                storage["action"][k].append(v)
            actions_numpy = {k: v.numpy() for k, v in actions.items()}
            obs_numpy, rewards, dones, infos = self.envs.step(actions_numpy)
            self.obs = {k: torch.from_numpy(v) for k, v in obs_numpy.items()}
            rewards = torch.from_numpy(rewards).to(dtype=torch.float32)
            dones = torch.from_numpy(dones).to(dtype=torch.bool)
            self.dones = dones
            is_terminal = dones.clone()

            if self.params.time_limit_bootstrapping:
                # infos is a tuple of info dicts
                for worker_id, done in enumerate(dones):
                    if "TimeLimit.truncated" in infos:
                        # Note: i think i'm playing a little loose with how the infos work here.
                        # They also have a _field_name bool parameter that probably says something like
                        # if that worker_id gave an info?
                        is_terminal[worker_id] = done and not infos["TimeLimit.truncated"][worker_id]
                        if done and not is_terminal[worker_id]:
                            # terminal_observation field name changed to final_observation
                            # in a gym update; seems in the version we're currently using the vector env
                            # got the new name while regular envs still have the old name.
                            terminal_observation = infos["final_observation"][worker_id]
                            additional_reward = self.time_limit_bootstrapping(terminal_observation)
                            rewards[worker_id] = rewards[worker_id] + additional_reward

            storage["reward"].append(rewards)
            storage["done"].append(dones)
            storage["is_terminal"].append(is_terminal)
            for k, v in to_store.items():
                storage[k].append(v)

        # we include one extra observation
        for k, v in self.obs.items():
            storage["observation"][k].append(v)

        # Pack everything into a single tensor per (nested) key
        for k, v in storage.items():
            if k in ("observation", "action"):
                storage[k] = {k: torch.stack(v, dim=1) for k, v in storage[k].items()}
            else:
                storage[k] = torch.stack(v, dim=1)

        self.storage.storage = storage

    def time_limit_bootstrapping(self, terminal_obs: dict[str, Tensor]) -> float:
        # NOTE: this won't work (as written) for recurrent algorithms!!
        # value bootstrapping for envs with artificial time limits imposed by a gym TimeLimit wrapper
        # see https://arxiv.org/pdf/1712.00378.pdf for context
        # this does get wrapper obs transforms applied
        terminal_obs_torch = {
            k: torch.from_numpy(v).to(self.rollout_device).unsqueeze(0) for k, v in terminal_obs.items()
        }
        terminal_obs_torch = postprocess_uint8_to_float(terminal_obs_torch, center=self.params.center_observations)
        with torch.no_grad():
            terminal_value, _ = self.model(terminal_obs_torch)
        return self.params.discount * terminal_value[0].detach().cpu().item()  # type: ignore

    def shutdown(self) -> None:
        try:
            self.envs.close()
        except BrokenPipeError:
            pass
