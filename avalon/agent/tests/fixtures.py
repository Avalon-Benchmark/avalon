from typing import Any
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import cast

import attr
import torch

from avalon.agent.common.envs import build_env
from avalon.agent.common.get_algorithm_cls import get_algorithm_cls
from avalon.agent.common.params import EnvironmentParams
from avalon.agent.common.params import Params
from avalon.agent.common.test_envs import TestEnvironmentParams
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import ParamsType
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.ppo.params import PPOParams
from avalon.agent.train_dreamer_test_envs import DreamerTestEnvParams
from avalon.agent.train_ppo_test_envs import TestPPOParams
from avalon.contrib.testing_utils import RequestFixture
from avalon.contrib.testing_utils import fixture
from avalon.contrib.utils import make_deterministic
from avalon.datagen.world_creation.tests.params import CANONICAL_SEED

_test_environment_params = {
    "test_case1": TestEnvironmentParams(task="case1", long_episode_length=1000),
    "test_case1_continuous_action": TestEnvironmentParams(task="case1_continuous_action", long_episode_length=1000),
    "test_case2": TestEnvironmentParams(task="case2", long_episode_length=1000),
    "test_case6": TestEnvironmentParams(task="case6", long_episode_length=1000),
    "test_hybrid1": TestEnvironmentParams(task="hybrid1", long_episode_length=1000),
}


def complete_agent_params(
    agent_params: ParamsType, env_params: EnvironmentParams, seed: int, **agent_kwargs: Any
) -> ParamsType:
    make_deterministic(seed)
    dummy_env = build_env(env_params)
    agent_params = attr.evolve(
        agent_params,
        env_params=env_params,
        observation_space=dummy_env.observation_space,
        action_space=dummy_env.action_space,
        **agent_kwargs,
    )
    dummy_env.close()
    del dummy_env
    return agent_params


def _make_agent(finalized_agent_params: Params) -> Algorithm:
    assert finalized_agent_params.observation_space is not None
    assert finalized_agent_params.action_space is not None
    algorithm_cls = get_algorithm_cls(finalized_agent_params)
    return algorithm_cls(
        finalized_agent_params, finalized_agent_params.observation_space, finalized_agent_params.action_space
    )


def get_valid_agent_env_combos(seed: int) -> Mapping[Tuple[str, str], Tuple[Params, EnvironmentParams]]:
    combos: Dict[Tuple[str, str], Tuple[Params, EnvironmentParams]] = {}
    # Test envs
    for env_name, env_params in list(_test_environment_params.items()):
        combos[("ppo", env_name)] = (TestPPOParams(num_workers=1, multiprocessing=False, batch_size=50), env_params)
        combos[("dreamer", env_name)] = (
            DreamerTestEnvParams(
                num_workers=1,
                multiprocessing=False,
                prefill_eps_per_dataloader=1,
            ),
            attr.evolve(env_params),
        )

    # Common benchmark envs
    # These theoretically work for most tests, but did exhibit some non-determinism for the full training step. We
    # don't care for them that much at the moment, but may be useful in the future
    # combos[("dreamer", "atari")] = (
    #     DreamerAtariParams(num_workers=1, multiprocessing=False),
    #     DmcEnvironmentParams(suite="atari", task="atlantis", action_repeat=4, time_limit=27000),
    # )
    # combos[("dreamer", "dmc")] = (
    #     DreamerDMCParams(num_workers=1, multiprocessing=False, prefill_eps_per_dataloader=1),
    #     DmcEnvironmentParams(
    #         suite="dmc", task="walker_walk", action_repeat=2, time_limit=None, include_proprio=False, include_rgb=True
    #     ),
    # )

    # Avalon
    combos[("ppo", "avalon")] = (
        PPOParams(num_workers=1, multiprocessing=False),
        GodotEnvironmentParams(env_index=seed),
    )
    combos[("dreamer", "avalon")] = (
        DreamerParams(num_workers=1, multiprocessing=False, prefill_eps_per_dataloader=1),
        GodotEnvironmentParams(env_index=seed),
    )

    return combos


@fixture
def device_() -> torch.device:
    return torch.device("cuda")


@fixture(params=get_valid_agent_env_combos(CANONICAL_SEED).items())
def agent_and_env_(request: RequestFixture) -> Tuple[Tuple[str, str], Tuple[Params, EnvironmentParams]]:
    return cast(Tuple[Tuple[str, str], Tuple[Params, EnvironmentParams]], request.param)
