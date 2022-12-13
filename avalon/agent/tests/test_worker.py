import warnings

import attr
import torch
import tree

from avalon.agent.common.params import Params
from avalon.agent.common.params import ProcgenEnvironmentParams
from avalon.agent.common.storage import EpisodeStorage
from avalon.agent.common.storage import FragmentStorage
from avalon.agent.common.storage import TrajectoryStorage
from avalon.agent.common.test_envs import TestEnvironmentParams
from avalon.agent.common.trainer import Trainer
from avalon.agent.common.util import hash_tensor
from avalon.agent.common.worker import RolloutManager
from avalon.agent.ppo.params import PPOParams
from avalon.agent.ppo.ppo_types import PPOSequenceData
from avalon.agent.tests.simple_worker import DummyStorage
from avalon.agent.tests.simple_worker import SimpleRolloutManager
from avalon.agent.train_ppo_procgen import ProcgenPPOParams
from avalon.agent.train_ppo_test_envs import TestPPOParams
from avalon.contrib.testing_utils import integration_test
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.contrib.utils import make_deterministic


class DummyTrainer(Trainer):
    def __init__(self, params, rollout_manager_cls, storage_cls) -> None:
        self.rollout_manager_cls = rollout_manager_cls
        self.storage_cls = storage_cls
        super().__init__(params)

    def create_rollout_manager(self) -> RolloutManager:
        rollout_manager = self.rollout_manager_cls(
            params=self.params,
            num_workers=self.params.num_workers,
            is_multiprocessing=self.params.multiprocessing,
            storage=self.train_storage,
            obs_space=self.params.observation_space,
            # storage_mode=StorageMode.EPISODE,
            model=self.algorithm,
            rollout_device=torch.device("cuda"),
            multiprocessing_context=self.multiprocessing_context,
        )
        self.to_cleanup.append(rollout_manager)
        return rollout_manager

    def create_train_storage(self):
        return self.storage_cls(self.params, self.step_data_type, num_workers=self.params.num_workers)

    def create_dataloader(self):
        return iter([])


def compare(x, y) -> None:
    assert hash_tensor(x) == hash_tensor(y)


def rollout(params, rollout_manager_cls, storage_cls, num_steps, extract_data_fn):
    make_deterministic(0)
    trainer = DummyTrainer(params, rollout_manager_cls, storage_cls)
    try:
        while True:
            make_deterministic(0)
            trainer.train_rollout_manager.run_rollout(num_steps=num_steps)
            # fragment = trainer.train_storage.to_packed()
            yield extract_data_fn(trainer.train_storage)
    finally:
        trainer.shutdown()


def compare_on_policy_rollouts(params: Params, num_steps: int) -> None:
    def get_data(storage: TrajectoryStorage):
        return storage.to_packed()

    simple_rollout = rollout(params, SimpleRolloutManager, DummyStorage, num_steps=num_steps, extract_data_fn=get_data)
    normal_rollout = rollout(params, RolloutManager, FragmentStorage, num_steps=num_steps, extract_data_fn=get_data)
    for i in range(3):
        a = next(simple_rollout)
        b = next(normal_rollout)
        assert a.value.sum() != 0  # just a little sanity check that we're getting data

        tree.map_structure(lambda x, y: compare(x, y), attr.asdict(a), attr.asdict(b))


@integration_test
def test_on_policy_rollouts(num_steps: int = 100) -> None:
    """Compares our rollout worker, with the most basic settings, to an alternative implementation.
    Checks that the output of both are exactly identical.

    A limitation of this test is that there's no way to guarantee that partial batches are actually used.
    And this is a common source of bugs.
    Probably would to add a way to simulate this (eg add a delay when resetting),
    or make sure to test on Avalon which we know to have slow resets.
    """
    params = ProcgenPPOParams(
        deterministic=True,
        num_workers=3,
        num_steps=num_steps,
        multiprocessing=True,
        allow_partial_batches=True,
        time_limit_bootstrapping=False,
        wandb_mode="disabled",
    )
    compare_on_policy_rollouts(params, num_steps)


@integration_test
def test_time_limit_bootstrapping(num_steps: int = 20) -> None:
    """Check that time limit bootstrapping works.
    This env config is specifically designed for testing time limit bootstrapping,
    although since we're using an untrained model that doesn't matter so much. Any env would be fine probably.
    """
    params = TestPPOParams(
        env_params=TestEnvironmentParams(long_episode_length=500000, time_limit=5, task="case1"),
        deterministic=True,
        num_workers=3,
        num_steps=num_steps,
        multiprocessing=True,
        allow_partial_batches=True,
        time_limit_bootstrapping=True,
        wandb_mode="disabled",
    )
    compare_on_policy_rollouts(params, num_steps=num_steps)


def compare_off_policy_rollouts(params: Params, num_steps: int) -> None:
    def get_data_simple(storage: TrajectoryStorage):
        # The SimpleWorker uses a fragment storage; we need to convert this to episodes
        fragment = storage.to_packed()

        def extract_episode(fragment, worker_id, start_i, end_i):
            # start_i and end_i are inclusive - eg start=0 and end=1 will return 2 timesteps.
            episode = tree.map_structure(lambda x: x[worker_id, start_i : end_i + 1], fragment)
            # this keeps the dtype PPOBatchSequenceData thru the map process;
            # need to downgrade it to just a sequencedata type
            episode = PPOSequenceData(**attr.asdict(episode))
            return episode

        episodes = []
        episode_start = [0] * params.num_workers
        for t in range(params.num_steps):
            for worker_id in range(params.num_workers):
                if fragment.done[worker_id, t]:
                    ep = extract_episode(fragment, worker_id, episode_start[worker_id], t)
                    episodes.append(ep)
                    episode_start[worker_id] = t + 1

        # eps won't be in same order in both impls; need a unique key for each.
        # we'll hope the observations are unique!
        return {hash_tensor(x.observation["rgb"]): x for x in episodes}

    def get_data_normal(storage: TrajectoryStorage):
        eps = [x[0].pack_sequence(x) for x in storage.recent_eps]
        return {hash_tensor(x.observation["rgb"]): x for x in eps}

    simple_rollout = rollout(
        params, SimpleRolloutManager, DummyStorage, num_steps=num_steps, extract_data_fn=get_data_simple
    )
    normal_rollout = rollout(
        params, RolloutManager, EpisodeStorage, num_steps=num_steps, extract_data_fn=get_data_normal
    )

    # This test only works for one "step", since the fragment->episode logic will break beyond that.
    # But steps don't have much meaning in off-policy anyways.
    simple_eps = next(simple_rollout)
    normal_eps = next(normal_rollout)

    assert len(set(simple_eps.keys())) > 0  # generally it would be a bug/misconfiguration if we don't get any eps
    assert set(simple_eps.keys()) == set(normal_eps.keys())
    for k in simple_eps.keys():
        tree.map_structure(lambda x, y: compare(x, y), simple_eps[k], normal_eps[k])

    # this should be redundant
    tree.map_structure(lambda x, y: compare(x, y), simple_eps, normal_eps)


@attr.s(auto_attribs=True, frozen=True)
class OffPolicyPPOParams(PPOParams):
    """PPO isn't supposed to be used off policy, so we have to hack this a bit to get it running."""

    log_rollout_metrics_every: int = 10000  # this ensures that we keep in memory all the eps
    data_dir: str = f"{FILESYSTEM_ROOT}/data/rollouts/"
    min_fragment_len: int = 0
    max_fragment_len: int = 0
    discard_short_eps: bool = False
    replay_buffer_size_timesteps_per_manager: int = 10_000_000


@integration_test
def test_off_policy_rollouts(num_steps: int = 500) -> None:
    # We use PPO in this test just because it has nice simple stateless inference.
    # Since we're not training, it's ok that it's not off-policy.
    params = OffPolicyPPOParams(
        env_params=ProcgenEnvironmentParams(),
        deterministic=True,
        num_workers=4,
        multiprocessing=True,
        allow_partial_batches=True,
        log_rollout_metrics_every=10000,  # this ensures that we keep in memory all the eps
        num_steps=num_steps,
        time_limit_bootstrapping=False,
        wandb_mode="disabled",
    )
    compare_off_policy_rollouts(params, num_steps)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    test_on_policy_rollouts()
