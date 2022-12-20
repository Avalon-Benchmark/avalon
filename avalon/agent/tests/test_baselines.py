from pathlib import Path
from typing import Tuple

import pytest
import torch

from avalon.agent.common.envs import build_env
from avalon.agent.common.params import EnvironmentParams
from avalon.agent.common.params import Params
from avalon.agent.common.util import hash_model
from avalon.agent.tests.fixtures import _make_agent
from avalon.agent.tests.fixtures import agent_and_env_
from avalon.agent.tests.fixtures import complete_agent_params
from avalon.agent.tests.fixtures import device_
from avalon.agent.tests.generate_test_data import run_deterministic_train_step
from avalon.agent.tests.utils import _load
from avalon.agent.tests.utils import assert_tensors_equal
from avalon.agent.tests.utils import run_deterministic_forward_pass
from avalon.contrib.testing_utils import slow_integration_test
from avalon.contrib.testing_utils import use
from avalon.datagen.world_creation.tests.fixtures import seed_


@slow_integration_test
@use(seed_, agent_and_env_, device_)
def test_rollout_step_deterministic(
    seed: int, agent_and_env: Tuple[Tuple[str, str], Tuple[Params, EnvironmentParams]], device: torch.device
) -> None:
    _names, (agent_params, env_params) = agent_and_env

    env = build_env(env_params)
    agent_params = complete_agent_params(agent_params, env_params, seed)
    agent = _make_agent(agent_params)
    agent = agent.to(device)

    observations = [env.reset()]
    for i in range(100):
        observation, *rest = env.step(env.action_space.sample())
        observations.append(observation)

    with torch.no_grad():
        for observation in observations:
            first_action = run_deterministic_forward_pass(agent, observation, seed)
            for i in range(100):
                action = run_deterministic_forward_pass(agent, observation, seed)
                assert_tensors_equal(action, first_action)
    env.close()
    del env
    del agent


@use(seed_, agent_and_env_)
def test_forward_pass_matches_reference(
    seed: int, agent_and_env: Tuple[Tuple[str, str], Tuple[Params, EnvironmentParams]]
) -> None:
    test_data_path = Path(__file__).parent / "data"
    (agent_name, env_name), (agent_params, env_params) = agent_and_env
    for observation_kind in ("initial", "random"):
        observation = _load(test_data_path / f"{env_name}_{observation_kind}_observation.pkl")
        reference_action = _load(test_data_path / f"{agent_name}_{env_name}_{observation_kind}_forward_pass.pkl")
        agent_params = complete_agent_params(agent_params, env_params, seed=seed)
        agent = _make_agent(agent_params)
        actual_action = run_deterministic_forward_pass(agent, observation)
        assert_tensors_equal(actual_action, reference_action)
        del agent


@pytest.mark.skip(
    reason="This failed in a flaky way (when no related code was changed), so am disabling until maksis can think of a nice way to make deterministic"
)
@slow_integration_test
@use(seed_, agent_and_env_)
def test_model_matches_reference_after_single_train_step(
    seed: int, agent_and_env: Tuple[Tuple[str, str], Tuple[Params, EnvironmentParams]]
) -> None:
    """
    Note: the (dreamer, avalon) combo is not inherently deterministic, because the AsyncRolloutManager (which generates
    rollouts) and DataLoader (which loads them, cuts a fragment and passes it along for training) are asynchronous,
    and the behavior is dependent on function duration. While the config used here should only fetch a single episode,
    and I haven't seen it fail - be warned it is ultimately flaky underneath. If it starts failing in CI, feel free
    to disable it (or find an elegant solution to sync the two processes).
    """
    test_data_path = Path(__file__).parent / "data"
    (agent_name, env_name), (agent_params, env_params) = agent_and_env
    trainer = run_deterministic_train_step(agent_params, env_params, seed)
    reference_hash = _load(test_data_path / f"{agent_name}_{env_name}_trained_model_hash.pkl")
    assert hash_model(trainer.algorithm) == reference_hash
