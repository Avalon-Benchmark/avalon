from abc import ABC
from abc import abstractmethod

# ObsType = Dict[str, Tensor]
from typing import Any
from typing import Dict
from typing import Generic
from typing import List
from typing import Tuple
from typing import TypeVar

import attr
import gym
import torch
from torch import Tensor

from avalon.agent.common.params import Params
from avalon.agent.common.util import pack_1d_list
from avalon.agent.common.util import pack_2d_list
from avalon.agent.common.util import seed_and_run_deterministically_if_enabled

Info = Dict[str, Any]
Observation = Dict[str, Tensor]  # represents a single timestep. atoms can have any shape
ObservationBatch = Dict[str, Tensor]  # shape (batch_size, ...)
ObservationSequenceBatch = Dict[str, Tensor]  # shape (batch_size, num_timesteps, ...)
ObservationSequence = Dict[str, Tensor]  # shape (num_timesteps, ...)

# ActionType = Dict[str, Tensor]
Action = Dict[str, Tensor]  # represents a single timestep. atoms can have any shape
ActionBatch = Dict[str, Tensor]  # shape (batch_size, ...)
ActionSequenceBatch = Dict[str, Tensor]  # shape (batch_size, num_timesteps, ...)
ActionSequence = Dict[str, Tensor]  # shape (num_timesteps, ...)

# The state is like {"stoch": tensor, "deter": tensor}
LatentBatch = Dict[str, Tensor]
StateActionBatch = Tuple[LatentBatch, ActionBatch]


@attr.s(auto_attribs=True, frozen=True)
class SequenceData:
    # All atoms should have shape (timesteps, ...)
    observation: ObservationSequence
    action: ActionSequence
    reward: Tensor
    done: Tensor
    is_terminal: Tensor


@attr.s(auto_attribs=True, frozen=True)
class BatchSequenceData:
    # All atoms should have shape (batch_size, num_timesteps, ...)
    observation: ObservationSequenceBatch
    action: ActionSequenceBatch
    reward: Tensor
    done: Tensor
    is_terminal: Tensor


@attr.s(auto_attribs=True, frozen=True)
class BatchData:
    # All atoms should have shape (batch_size, ...)
    observation: ObservationBatch
    action: ActionBatch
    reward: Tensor
    done: Tensor
    is_terminal: Tensor


@attr.s(auto_attribs=True, frozen=True)
class StepData:
    # if obs_first=True, for a given timestep, the order is (observation, action, reward/done/info)
    # if obs_first=False, for a given timestep, the order is (action, reward/done/info, observation)
    # if we extend this with extra inference fields, they'll go with the obs, since it's computed from the obs.
    # So eg for PPO, obs_first=True has ordering (observation/value/policy, action, reward/done/info)
    batch_type = BatchData
    sequence_type = SequenceData
    batch_sequence_type = BatchSequenceData

    observation: Dict[str, Tensor]
    action: Dict[str, Tensor]
    reward: float
    done: bool
    is_terminal: bool
    info: Dict[str, Any]

    def pack_sequence_batch(self, batch: List[List["StepData"]]) -> "BatchSequenceData":
        return pack_2d_list(batch, out_cls=self.batch_sequence_type)  # type: ignore

    def pack_sequence(self, sequence: List["StepData"]) -> "SequenceData":
        return pack_1d_list(sequence, out_cls=self.sequence_type)  # type: ignore


ParamsType = TypeVar("ParamsType", bound=Params)


class Algorithm(torch.nn.Module, ABC, Generic[ParamsType]):
    step_data_type: StepData = StepData

    def __init__(self, params: ParamsType, obs_space: gym.spaces.Dict, action_space: gym.spaces.Dict) -> None:
        seed_and_run_deterministically_if_enabled()

        super().__init__()
        self.params = params
        self.obs_space = obs_space
        self.action_space = action_space

    @abstractmethod
    def rollout_step(
        self,
        next_obs: Observation,
        dones: Tensor,
        indices_to_run: Tensor,
        exploration_mode: str,
    ) -> Tuple[ActionBatch, dict]:
        raise NotImplementedError

    def reset_state(self) -> None:
        pass

    @abstractmethod
    def train_step(self, batch_data: BatchSequenceData, step: int) -> int:
        """Returns the new step count (sometimes multiple gradient steps are taken here)."""
        raise NotImplementedError
