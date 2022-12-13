import os
from pathlib import Path
from typing import Tuple

import gym
from torch import Tensor

from avalon.agent.common.params import Params
from avalon.agent.common.trainer import OnPolicyTrainer
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.types import ObservationBatch
from avalon.agent.godot.godot_eval import test
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.contest.agents.contest_algorithm import load_algorithm
from avalon.contest.contest_params import ContestAlgorithmParams


class ContestAlgorithmWrapper(Algorithm):
    """Wraps a user-provided algorithm into our internal Algorithm class."""

    def __init__(self, params: Params, observation_space: gym.spaces.Dict, action_space: gym.spaces.Dict) -> None:
        super().__init__(params, observation_space, action_space)
        self.contest_algorithm = load_algorithm()

    def rollout_step(
        self,
        next_obs: ObservationBatch,
        dones: Tensor,  # shape (batch_size, )
        indices_to_run: list[bool],  # shape (batch_size, )
        exploration_mode: str,
    ) -> Tuple[ActionBatch, dict]:
        step_actions = self.contest_algorithm.rollout_step(next_obs, dones)
        return step_actions, {}

    def train_step(self, batch_data: BatchSequenceData, step: int) -> int:
        raise NotImplementedError


if __name__ == "__main__":
    fixed_worlds_path = os.getenv("FIXED_WORLDS_PATH", "/tmp/avalon_worlds/minival/")

    params = ContestAlgorithmParams(
        project="contest_agent",
        is_testing=True,
        is_training=False,
        env_params=GodotEnvironmentParams(fixed_worlds_load_from_path=Path(fixed_worlds_path)),
    )
    trainer = OnPolicyTrainer(params)

    assert trainer.params.observation_space is not None
    assert trainer.params.action_space is not None

    algorithm = ContestAlgorithmWrapper(params, trainer.params.observation_space, trainer.params.action_space)

    try:
        test(trainer.params, algorithm)
    finally:
        trainer.shutdown()
