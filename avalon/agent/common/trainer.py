import signal
import time
import uuid
from abc import ABC
from abc import abstractmethod
from ctypes import c_int
from functools import partial
from pathlib import Path
from typing import Generic
from typing import Iterator
from typing import List
from typing import Optional
from typing import Protocol

import attr
import torch
import wandb
from loguru import logger
from torch.utils.data import DataLoader
from tree import map_structure

from avalon.agent.common import wandb_lib
from avalon.agent.common.action_model import visualize_actions
from avalon.agent.common.dataloader import ReplayDataset
from avalon.agent.common.dataloader import worker_init_fn
from avalon.agent.common.envs import build_env
from avalon.agent.common.get_algorithm_cls import get_algorithm_cls
from avalon.agent.common.storage import FragmentStorage
from avalon.agent.common.storage import ModelStorage
from avalon.agent.common.storage import TrajectoryStorage
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.types import ParamsType
from avalon.agent.common.types import StepData
from avalon.agent.common.util import get_checkpoint_file
from avalon.agent.common.util import pack_1d_list
from avalon.agent.common.util import postprocess_uint8_to_float
from avalon.agent.common.worker import AsyncRolloutManager
from avalon.agent.common.worker import RolloutManager
from avalon.agent.dreamer.params import OffPolicyParams
from avalon.agent.godot.godot_eval import log_rollout_stats_packed
from avalon.agent.ppo.params import OnPolicyParams
from avalon.agent.ppo.params import PPOParams
from avalon.agent.ppo.ppo_types import PPOStepData
from avalon.contrib.utils import make_deterministic

# Just a temporary reminder to not use <1.11
assert int(torch.__version__.split(".")[1]) >= 11


class Cleanable(Protocol):
    def shutdown(self) -> None:
        pass


class Trainer(ABC, Generic[ParamsType]):
    def __init__(self, params: ParamsType) -> None:
        # TODO: fix this properly
        import openturns

        openturns.Tensor
        # Patch the issue where openturns resets the signal handler
        signal.signal(signal.SIGINT, signal.default_int_handler)

        # Use an explict context to fix this bug: https://github.com/pytorch/pytorch/issues/3492
        # Spawn is less fiddly, but fork launches faster
        self.multiprocessing_context = torch.multiprocessing.get_context(params.multiprocessing_mode)
        # Fixing the bug where torch gets really slow on small fast ops on CPU
        # https://github.com/pytorch/pytorch/issues/80777
        torch.set_num_threads(1)

        # allow tf32 (also need to call this in any subprocesses that may need it, eg inference)
        # leaving this off for now since it gave issues in PPO
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True

        self.params = params
        self.to_cleanup: List[Cleanable] = []

        self.start: Optional[float] = None
        self.i = 0
        self.env_step = 0

        if isinstance(params, PPOParams):
            self.step_data_type = PPOStepData
        else:
            self.step_data_type = StepData

        self.get_spaces()
        self.train_storage = self.create_train_storage()

        if params.deterministic:
            make_deterministic(seed=0)

        self.algorithm = self.create_model()
        self.train_rollout_manager = self.create_rollout_manager()
        self.train_dataloader = self.create_dataloader()
        self.wandb_run = self.wandb_init()
        # wandb.watch(self.algorithm, log=None, log_freq=self.params.log_freq_hist, log_graph=True)

    def get_spaces(self) -> None:
        dummy_env = build_env(self.params.env_params)
        self.params = attr.evolve(
            self.params, observation_space=dummy_env.observation_space, action_space=dummy_env.action_space
        )
        dummy_env.close()

    def create_train_storage(self) -> Optional[TrajectoryStorage]:
        return None

    def wandb_init(self):
        # This needs to happen after all other processes launch
        run_name = (
            f"{self.params.env_params.suite}:{self.params.env_params.task}"
            if not self.params.name
            else self.params.name
        )
        run = wandb.init(
            name=run_name,
            project=self.params.project,
            config=attr.asdict(self.params),
            tags=self.params.tags,
            group=self.params.group,
            mode=self.params.wandb_mode,
        )
        wandb_lib.SCALAR_FREQ = self.params.log_freq_scalar
        wandb_lib.HIST_FREQ = self.params.log_freq_hist
        wandb_lib.MEDIA_FREQ = self.params.log_freq_media
        return run

    @abstractmethod
    def create_rollout_manager(self):
        raise NotImplementedError

    def create_model(self) -> Algorithm:
        algorithm_cls = get_algorithm_cls(self.params)
        assert self.params.observation_space is not None
        assert self.params.action_space is not None
        algorithm = algorithm_cls(self.params, self.params.observation_space, self.params.action_space)

        if self.params.resume_from:
            checkpoint_path = get_checkpoint_file(self.params.resume_from)
            algorithm.load_state_dict(torch.load(checkpoint_path, map_location=self.params.train_device))
            logger.info("RESUMED MODEL FROM CHECKPOINT")

        algorithm = algorithm.to(self.params.train_device)

        logger.info(f"model has {sum(p.numel() for p in algorithm.parameters() if p.requires_grad)} params")
        return algorithm

    @abstractmethod
    def create_dataloader(self) -> Iterator:
        raise NotImplementedError

    def train(self) -> None:
        if not self.start:
            self.start = time.time()
        while True:
            self.train_step()

            wandb_lib.log_scalar("env_step", self.env_step, self.i)
            if self.env_step >= self.params.total_env_steps:
                break

        self.checkpoint(filename="final.pt")

    @property
    def frames_per_batch(self):
        return self.params.batch_size

    def train_step(self) -> None:
        rollouts: BatchSequenceData = next(self.train_dataloader)
        start = time.time()
        start_i = self.i
        rollouts = map_structure(lambda x: x.to(self.params.train_device, non_blocking=True), rollouts)
        rollouts = attr.evolve(
            rollouts,
            observation=postprocess_uint8_to_float(rollouts.observation, center=self.params.center_observations),
        )

        self.i = self.algorithm.train_step(rollouts, self.i)

        wandb_lib.log_scalar(
            "timings/train_fps", self.frames_per_batch * (self.i - start_i) / (time.time() - start), self.i
        )
        wandb_lib.log_scalar("true_env_step", self.env_step * self.params.env_params.action_repeat, self.i)

        if self.i % self.params.checkpoint_every == 0:
            self.checkpoint()

        wandb_lib.log_iteration_time(self.frames_per_batch, self.i)
        wandb_lib.log_scalar("timings/cumulative_env_fps", self.env_step / (time.time() - self.start), self.i)
        wandb_lib.log_scalar(
            "timings/cumulative_train_fps", self.i * self.frames_per_batch / (time.time() - self.start), self.i
        )

        # Visualize the observations
        if self.i % wandb_lib.MEDIA_FREQ == 0:
            # the post-transforms have already been applied.
            for k, v in self.params.observation_space.spaces.items():
                if len(v.shape) == 3:
                    obs_video = rollouts.observation[k][:8]
                    wandb_lib.log_video(f"videos/obs/{k}", obs_video + 0.5, self.i, freq=1)

        # Visualize the actions
        if self.i % wandb_lib.MEDIA_FREQ == 0:
            visualize_actions(self.params.action_space, rollouts.action, prefix="rollout_actions", freq=1)

        # Evaluation
        # val.update(algorithm, i, env_step)
        # self.i += 1

    def checkpoint(self, filename: Optional[str] = None) -> None:
        if not filename:
            filename = f"model_{self.i}.pt"
        model_filename = Path(wandb.run.dir) / filename  # type: ignore
        torch.save(self.algorithm.state_dict(), model_filename)
        wandb.save(str(model_filename), policy="now")  # type: ignore
        wandb_lib.log_scalar("last_checkpoint", self.i, self.i, freq=1)

    def test(self):
        if not self.params.is_train_only:
            raise NotImplementedError

    def shutdown(self, finish_wandb_quietly=False) -> None:
        # The main thread won't join until we close all the processes we have open.
        # TODO: we don't automatically cleanup the off-policy dataloader(s)
        for item in self.to_cleanup:
            logger.info(f"shutting down {item}")
            item.shutdown()
        self.to_cleanup = []
        if finish_wandb_quietly:
            logger.info("finishing wandb")
            wandb.finish(quiet=True)
        logger.info("trainer shutdown is finished")


class OffPolicyTrainer(Trainer[OffPolicyParams]):
    def __init__(self, params: OffPolicyParams) -> None:
        self.train_rollout_dir = str(Path(params.data_dir) / "train" / str(uuid.uuid4()))

        # Necessary to fix some bug: https://github.com/wandb/client/issues/1994
        # Also necessary to get wandb to shut down properly?
        wandb.require(experiment="service")
        wandb.setup()

        self.train_rollout_managers: list[AsyncRolloutManager] = []
        super().__init__(params)
        ModelStorage.clean()

    def log_rollout_stats(self) -> None:
        for manager in self.train_rollout_managers:
            wandb_lib.log_from_queue(manager.wandb_queue, prefix=f"rollout_manager_{manager.rollout_manager_id}/")

    def start_train_rollouts(self) -> None:
        [manager.start() for manager in self.train_rollout_managers]

    def shutdown_train_rollouts(self) -> None:
        [manager.shutdown() for manager in self.train_rollout_managers]
        self.train_rollout_managers = []

    def create_rollout_manager(self) -> None:
        self.env_step_counters = []
        for i in range(self.params.worker_managers):
            # we don't need a lock here because there's only one writer and precise read/write ordering doesn't matter
            env_step_counter = self.multiprocessing_context.Value(c_int, lock=False)
            train_game_player = AsyncRolloutManager(
                params=self.params,
                obs_space=self.params.observation_space,
                action_space=self.params.action_space,
                rollout_manager_id=i,
                env_step_counter=env_step_counter,
                multiprocessing_context=self.multiprocessing_context,
                train_rollout_dir=self.train_rollout_dir,
            )
            self.env_step_counters.append(env_step_counter)
            self.to_cleanup.append(train_game_player)
            self.train_rollout_managers.append(train_game_player)

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
        old_i = self.i
        super().train_step()
        assert self.i == old_i + 1, "Off-policy algorithms must increment i by only 1"
        self.env_step = sum([x.value for x in self.env_step_counters])
        if self.i % self.params.rollout_model_update_freq == 0:
            logger.debug("saving model")
            ModelStorage.push_model(self.algorithm)
        self.log_rollout_stats()

    @property
    def frames_per_batch(self):
        assert self.params.min_fragment_len == self.params.max_fragment_len
        return self.params.batch_size * self.params.min_fragment_len


class OnPolicyTrainer(Trainer[OnPolicyParams]):
    def __init__(
        self, params: OnPolicyParams, rollout_manager_cls=RolloutManager, storage_cls=FragmentStorage
    ) -> None:
        self.rollout_manager_cls = rollout_manager_cls
        self.storage_cls = storage_cls
        super().__init__(params)

    def create_rollout_manager(self):
        rollout_manager = self.rollout_manager_cls(
            params=self.params,
            num_workers=self.params.num_workers,
            is_multiprocessing=self.params.multiprocessing,
            storage=self.train_storage,
            obs_space=self.params.observation_space,
            model=self.algorithm,
            rollout_device=self.params.inference_device,
            multiprocessing_context=self.multiprocessing_context,
        )
        self.to_cleanup.append(rollout_manager)
        return rollout_manager

    def create_train_storage(self) -> TrajectoryStorage:
        return self.storage_cls(self.params, step_data_type=self.step_data_type, num_workers=self.params.num_workers)

    def rollout_step(self) -> None:
        # Run rollouts
        start = time.time()
        self.train_rollout_manager.run_rollout(
            num_steps=self.params.num_steps,
            exploration_mode="explore",
        )
        rollout_fps = (self.params.num_workers * self.params.num_steps) / (time.time() - start)
        wandb_lib.log_scalar("timings/rollout_fps", rollout_fps, self.i)
        self.env_step += self.params.num_workers * self.params.num_steps

        if self.params.env_params.suite == "godot":
            log_rollout_stats_packed(self.train_storage.storage, self.i)
        else:
            # Either use our default worker, or wrap in our (not gym's!) RecordEpisodeStatistics for this to work.
            try:
                if "info" in self.train_storage.storage.keys():
                    rollouts = self.train_storage.storage
                    cumulative_rewards = []
                    episode_lengths = []
                    for worker_id in range(self.params.num_workers):
                        for t in range(self.params.num_steps):
                            if rollouts["done"][worker_id, t]:
                                cumulative_rewards.append(rollouts["info"]["cumulative_episode_return"][worker_id, t])
                                episode_lengths.append(rollouts["info"]["cumulative_episode_length"][worker_id, t])
                    if len(cumulative_rewards) > 0:
                        wandb_lib.log_histogram(f"rollout/cumulative_reward", cumulative_rewards, self.i)
                        wandb_lib.log_histogram(f"rollout/episode_length", episode_lengths, self.i)
                    wandb_lib.log_scalar(f"rollout/num_finished_episodes", len(episode_lengths), self.i)
            except (AttributeError, TypeError):
                # Storage doesn't support this
                pass

    def create_dataloader(self) -> Iterator[BatchSequenceData]:
        def dataloader():
            while True:
                self.train_storage.reset()
                self.rollout_step()
                rollouts: BatchSequenceData = self.train_storage.to_packed()
                # go ahead and send to cuda now, will make the next step faster
                rollouts = map_structure(lambda x: x.to(self.params.train_device, non_blocking=True), rollouts)
                yield rollouts

        return iter(dataloader())
