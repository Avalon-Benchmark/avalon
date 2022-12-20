import queue
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Iterator

import attr
import torch
from loguru import logger
from torch.utils.data import DataLoader

from avalon.agent.common import wandb_lib
from avalon.agent.common.dataloader import ReplayDataset
from avalon.agent.common.dataloader import worker_init_fn
from avalon.agent.common.params import DmcEnvironmentParams
from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.storage import EpisodeStorage

# from avalon.agent.common.storage import StorageMode
from avalon.agent.common.trainer import Trainer
from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.util import pack_1d_list
from avalon.agent.common.util import setup_new_process
from avalon.agent.common.worker import RolloutManager
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.dreamer.params import OffPolicyParams
from avalon.common.log_utils import configure_remote_logger


class DreamerTrainer(Trainer[OffPolicyParams]):
    """This `Trainer` is designed to exactly replicate the training setup in danijar's dreamerv2 repo.

    Instead of running the data-collection and training in separate, unsynchronized processes, as we do normally,
    this runs them in an alternating manner in a single process.
    This results in a fixed constant ratio between training and env steps.
    """

    def __init__(self, params: OffPolicyParams) -> None:
        self.wandb_queue = queue.Queue()  # type: ignore[var-annotated]
        self.train_rollout_dir = str(Path(params.data_dir) / "train" / str(uuid.uuid4()))
        super().__init__(params)

        # Prefill so we have enough steps to form a first batch
        rollout_steps = params.prefill_steps // params.num_workers
        self.train_rollout_manager.run_rollout(
            num_steps=rollout_steps,
            num_episodes=1,
            exploration_mode="explore",
        )
        self.env_step += self.params.num_workers * rollout_steps

        # Pretrain (train without doing fresh rollouts). Lets us train a bit before using the model for rollouts.
        if not self.start:
            self.start = time.time()
        for _ in range(params.pretrain_steps):
            super().train_step()

    def create_rollout_manager(self):
        rollout_manager = RolloutManager(
            params=self.params,
            num_workers=self.params.num_workers,
            is_multiprocessing=self.params.multiprocessing,
            storage=self.train_storage,
            obs_space=self.params.observation_space,
            model=self.algorithm,
            rollout_device=torch.device(f"cuda:{self.params.inference_gpus[0]}"),
            multiprocessing_context=self.multiprocessing_context,
        )
        self.to_cleanup.append(rollout_manager)
        return rollout_manager

    def create_train_storage(self) -> EpisodeStorage:
        return EpisodeStorage(self.params, self.train_rollout_dir, self.wandb_queue)

    def create_dataloader(self) -> Iterator[BatchSequenceData]:
        train_dataset = ReplayDataset(self.params, self.train_rollout_dir, update_interval=4000)
        return iter(
            DataLoader(
                train_dataset,
                batch_size=self.params.batch_size,
                shuffle=False,
                num_workers=self.params.num_dataloader_workers,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=2,
                worker_init_fn=worker_init_fn,
                collate_fn=partial(pack_1d_list, out_cls=BatchSequenceData),
            )
        )

    def train_step(self) -> None:
        rollout_steps = self.params.train_every
        self.train_rollout_manager.run_rollout(
            num_steps=rollout_steps,
            exploration_mode="explore",
        )
        self.env_step += self.params.num_workers * rollout_steps

        old_i = self.i
        super().train_step()
        assert self.i == old_i + 1, "Off-policy algorithms must increment i by only 1"

        wandb_lib.log_from_queue(self.wandb_queue, prefix=f"rollout/")

    @property
    def frames_per_batch(self):
        assert self.params.min_fragment_len == self.params.max_fragment_len
        return self.params.batch_size * self.params.min_fragment_len


@attr.s(auto_attribs=True, frozen=True)
class DreamerDMCParams(DreamerParams):
    name: str = "torch"
    total_env_steps: int = 250_000
    pcont: bool = False
    prioritize_ends: bool = False
    worker_managers: int = 1
    num_workers: int = 1
    env_params: DmcEnvironmentParams = DmcEnvironmentParams(
        suite="dmc", task="walker_walk", action_repeat=2, time_limit=None
    )
    free_nats: float = 1
    policy_entropy_scale: float = 1e-4
    model_lr: float = 3e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    deter_size: int = 200
    rssm_hidden_size: int = 200
    train_gpu: int = 0
    inference_gpus: tuple[int, ...] = (0,)
    log_freq_hist: int = 100
    log_freq_scalar: int = 20
    log_freq_media: int = 500
    checkpoint_every: int = 2500
    num_dataloader_workers: int = 1
    log_rollout_metrics_every: int = 1
    prefill_steps: int = 1000
    pretrain_steps: int = 100
    freeze_actor_steps: int = 0
    prefill_eps_per_dataloader: int = 1


def run(params: DreamerDMCParams) -> None:
    # TODO: do this!!
    # assert False, "check param counts match before running this again!"

    trainer = DreamerTrainer(params)
    try:
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    try:
        default_params = DreamerDMCParams()
        default_params = parse_args(default_params)
        # default_params = attr.evolve(
        #     default_params,
        #     name=f"{default_params.name}_{default_params.env_params.suite}_{default_params.env_params.task}",
        # )
        run(default_params)
    except Exception as e:
        logger.exception(e)
        raise
