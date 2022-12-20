import argparse
import os
import time
import warnings
from functools import partial
from pathlib import Path

import attr
import torch
from loguru import logger
from pytest import MonkeyPatch
from requests.exceptions import RequestsDependencyWarning

from avalon.agent.common.envs import build_env
from avalon.agent.common.params import EnvironmentParams
from avalon.agent.common.params import Params
from avalon.agent.common.trainer import OffPolicyTrainer
from avalon.agent.common.trainer import OnPolicyTrainer
from avalon.agent.common.trainer import Trainer
from avalon.agent.common.util import hash_model
from avalon.agent.common.util import setup_new_process
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.ppo.params import PPOParams
from avalon.agent.tests.fixtures import _make_agent
from avalon.agent.tests.fixtures import complete_agent_params
from avalon.agent.tests.fixtures import get_valid_agent_env_combos
from avalon.agent.tests.utils import _dump
from avalon.agent.tests.utils import _load
from avalon.agent.tests.utils import run_deterministic_forward_pass
from avalon.common.log_utils import configure_remote_logger
from avalon.contrib.utils import make_deterministic
from avalon.datagen.world_creation.tests.params import CANONICAL_SEED


def generate_test_input_data(seed: int, test_data_path: Path) -> None:
    for (agent_name, env_name), (agent_params, env_params) in get_valid_agent_env_combos(seed).items():
        logger.info(f"Generating input data for {(agent_name, env_name)}")
        make_deterministic(seed)
        env = build_env(env_params)
        env.action_space.seed(seed)
        initial_observation = env.reset()
        _dump(initial_observation, test_data_path / f"{env_name}_initial_observation.pkl")
        random_observation, *rest = env.step(env.action_space.sample())
        _dump(random_observation, test_data_path / f"{env_name}_random_observation.pkl")
        env.close()
        del env


def run_deterministic_train_step(
    agent_params: Params, env_params: EnvironmentParams, seed: int = CANONICAL_SEED
) -> Trainer:
    with MonkeyPatch.context() as mp:
        # TODO: d67559db-84db-4d5f-8392-d551ed6a75c5
        # Deterministic backward passes on GPU need a deterministic scatter_add implementation that's only on Torch 1.13
        # see: https://github.com/pytorch/pytorch/commit/5b58140d1a471b144baf66cc61a45a86746f0215
        # Note that inference (or any other model in a different process) will still run on GPU - we only care about the
        # learner since that's where `scatter_add` is used.
        cpu_device = torch.device("cpu")

        def forced_cpu(_device: str) -> torch.device:
            return cpu_device

        mp.setattr(torch, "device", forced_cpu)
        mp.setattr(torch.cuda, "is_available", lambda: False)

        # We can't monkeypatch the functions directly to seed, since some of them are in different processes
        mp.setenv("AVALON_MODEL_SEED", str(seed))

        if isinstance(agent_params, DreamerParams):
            # For OffPolicyTrainer, we want to sleep as little as possible so that we load the first episode ASAP,
            # otherwise we risk non-determinism if two eps are there when we load in data (in which case their order is
            # non-deterministic since we can't trivially patch uuids to be non-random (as they use os.urandom)
            sleep_100ms = partial(time.sleep, 0.1)

            def quick_sleep(_secs: int) -> None:
                sleep_100ms()

            mp.setattr(time, "sleep", quick_sleep)

            agent_params = complete_agent_params(agent_params, env_params, seed=seed)
            agent_params = attr.evolve(agent_params, min_fragment_len=30, max_fragment_len=30, batch_size=10)
            make_deterministic(seed)
            dreamer_trainer = OffPolicyTrainer(agent_params)
            dreamer_trainer.start = time.time()
            try:
                dreamer_trainer.start_train_rollouts()
                dreamer_trainer.train_step()
            finally:
                dreamer_trainer.shutdown()
            return dreamer_trainer
        elif isinstance(agent_params, PPOParams):
            agent_params = complete_agent_params(agent_params, env_params, seed=seed)
            agent_params = attr.evolve(agent_params, batch_size=8, num_steps=32)
            ppo_trainer = OnPolicyTrainer(agent_params)
            ppo_trainer.start = time.time()
            try:
                ppo_trainer.train_step()
            finally:
                ppo_trainer.shutdown()
            return ppo_trainer
        else:
            raise NotImplementedError(type(agent_params))


def generate_test_output_data(seed: int, test_data_path: Path) -> None:
    for (agent_name, env_name), (agent_params, env_params) in get_valid_agent_env_combos(seed).items():
        logger.info(f"Generating output data for {(agent_name, env_name)}")
        for observation_kind in ("initial", "random"):
            observation = _load(test_data_path / f"{env_name}_{observation_kind}_observation.pkl")
            agent = _make_agent(complete_agent_params(agent_params, env_params, seed=seed))
            output = run_deterministic_forward_pass(agent, observation, seed)
            _dump(output, test_data_path / f"{agent_name}_{env_name}_{observation_kind}_forward_pass.pkl")
            del agent

        trainer = run_deterministic_train_step(agent_params, env_params, seed)
        model_hash = hash_model(trainer.algorithm)
        logger.info(f"{model_hash=}")
        _dump(model_hash, test_data_path / f"{agent_name}_{env_name}_trained_model_hash.pkl")


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="store_const", const=True)
    parser.add_argument("--output", action="store_const", const=True)
    parser.add_argument("--seed", action="store", type=int, default=CANONICAL_SEED)
    args = parser.parse_args()

    os.environ["WANDB_MODE"] = "disabled"
    output_path = Path(__file__).parent / "data"
    output_path.mkdir(exist_ok=True, parents=True)
    assert args.input or args.output, "Must specify which test data to generate"
    if args.input:
        generate_test_input_data(args.seed, test_data_path=output_path)
    if args.output:
        generate_test_output_data(args.seed, test_data_path=output_path)
