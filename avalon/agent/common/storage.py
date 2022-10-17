import collections
import datetime
import pathlib
import pickle
import random
import shutil
import uuid
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import torch
from loguru import logger

from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.types import SequenceData
from avalon.agent.common.types import StepData
from avalon.agent.dreamer.params import OffPolicyParams
from avalon.agent.dreamer.params import Params

try:
    import pyarrow
except ImportError:
    pass


class ModelStorage:
    # TODO: move this somewhere better + parameterize
    # TODO: make this have a unique path so we can run multiple runs on the same machine
    model_path = Path("/tmp/model.pt")

    @classmethod
    def clean(cls) -> None:
        cls.model_path.unlink(missing_ok=True)

    @classmethod
    def push_model(cls, model: torch.nn.Module) -> None:
        """Save the model to storage."""
        torch.save(model.state_dict(), str(cls.model_path))

    @classmethod
    def load_model(cls, model: torch.nn.Module, device: torch.device) -> None:
        """Load the parameters in storage into the provided model."""
        if cls.model_path.exists():
            try:
                model.load_state_dict(torch.load(str(cls.model_path), map_location=device))
                logger.info("loaded model")
            except Exception as e:
                # Most commonly we fail here if the file is being written while we're reading.
                # RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory
                # EOFError: Ran out of input
                logger.warning(f"exception loading model in worker: {e}")
        else:
            logger.info("no model found")


class StorageMode(Enum):
    FRAGMENT = "FRAGMENT"
    EPISODE = "EPISODE"


class TrajectoryStorage(ABC):
    @abstractmethod
    def add_timestep_samples(self, samples: Dict[str, StepData]) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def shutdown(self):
        pass


class InMemoryStorage(TrajectoryStorage):
    """
    We store everything here in lists. So the actual ndarray atoms are a single timestep of a single obs.
    This is for efficient single additions when the size isn't known in advance.
    If we always add entire episodes, it may be more efficient to collapse into large ndarrays.
    Could have two instances of this class for both cases if needed, if the implementation is sufficiently abstracted.
    """

    def __init__(self, params: Params):
        self.storage: Dict[str, List[StepData]] = defaultdict(list)
        self.params = params

    def add_timestep_samples(self, samples: Dict[str, StepData]) -> None:
        """Add a set of samples representing a single timestep, keyed by a batch/episode identifer."""
        for key, v in samples.items():
            self.storage[key].append(v)

    def to_packed(self) -> BatchSequenceData:
        # Grab a random instance just to get access to its `pack_sequence_batch` method
        instance = self.storage[list(self.storage.keys())[0]][0]
        # Note: this operation can be a bit slow/variable. Lots of mallocing.
        return instance.pack_sequence_batch(list(self.storage.values()))

    def reset(self) -> None:
        self.storage = defaultdict(list)


class DiskStorage(TrajectoryStorage):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        params: OffPolicyParams,
        rollout_dir: str,
        # wandb: Optional[Run],
        wandb_queue=None,  # not typeable, but a Optional[Queue] :(
        discard_short_eps: bool = True,  # should be true in training, false in eval
    ):
        self.ongoing: Dict[str, List[StepData]] = defaultdict(list)
        self.params = params
        # self.wandb = wandb
        self.wandb_queue = wandb_queue

        self.data_dir = Path(rollout_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.timesteps_in_buffer = 0
        self.total_timesteps = 0
        self.episode_counter = 0
        self.recent_eps: collections.deque[List[StepData]] = collections.deque(maxlen=params.log_rollout_metrics_every)

        self.log_freq = params.log_rollout_metrics_every
        self.discard_short_eps = discard_short_eps

    def log_stats(self) -> None:
        # Compute training rollout statistics
        if not self.wandb_queue:
            return
        if not self.log_freq:
            return
        if self.episode_counter % self.log_freq != 0:
            return

        if self.params.env_params.suite == "godot":
            # Can't use log_rollout_stats because these are full episodes, not fragments,
            # so the analysis is a bit different. We can make stronger statements here.
            successes: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
            keys = ["success", "difficulty"]
            for episode in self.recent_eps:
                last_step = episode[-1]
                task = last_step.info["task"].lower()
                for field in keys:
                    successes[task][f"final_{field}"].append(last_step.info[field])
                successes[task][f"episode_length"].append(len(episode))
            # Data is a dict (task) of dicts (keys) of lists
            for task, x in successes.items():
                for field, y in x.items():
                    self.wandb_queue.put(("scalar", f"{task}/{field}", np.mean(y)))
                self.wandb_queue.put(("scalar", f"{task}/num_episodes", len(y)))
        else:
            rewards = []
            lengths = []
            for episode in self.recent_eps:
                total_reward = sum([step.reward for step in episode])
                length = len(episode)
                rewards.append(total_reward)
                lengths.append(length)
            # TODO: add histograms here
            # Note: action repeats have not been factored in here
            self.wandb_queue.put(("scalar", f"return_mean", np.mean(rewards)))
            self.wandb_queue.put(("scalar", f"length_mean", np.mean(lengths)))

        self.wandb_queue.put(("scalar", f"buffer_size_timesteps", self.timesteps_in_buffer))
        self.wandb_queue.put(("scalar", f"total_timesteps", self.total_timesteps))
        self.wandb_queue.put(("scalar", f"true_buffer_size", measure_buffer_size(str(self.data_dir))))
        self.wandb_queue.put(("scalar", f"total_episodes", self.episode_counter))

    def add_timestep_samples(self, samples: Dict[str, StepData]) -> None:
        """Add a set of samples representing a single timestep, keyed by a batch/episode identifer."""
        for key, v in samples.items():
            self.ongoing[key].append(v)
            if v.done:
                ep = self.ongoing[key]
                ep_len = len(ep)
                del self.ongoing[key]
                if len(ep) < self.params.min_fragment_len and self.discard_short_eps:
                    logger.debug(f"not adding short ep of len {ep_len}")
                    continue
                self.episode_counter += 1
                self.recent_eps.append(ep)
                packed_ep = v.pack_sequence(ep)
                self.save_episode(packed_ep)
                self.timesteps_in_buffer += ep_len
                self.total_timesteps += ep_len

                # enforce timestep limit
                # TODO: kinda weird that we will be deleting other processes' files
                # TODO: if one of these classes crashes, then the allowed buffer size will grow
                # probably should just count all the filenames instead - was just trying to avoid that.
                # because nobody is keeping an absolute track of the size of the buffer.
                if self.timesteps_in_buffer > self.params.replay_buffer_size_timesteps_per_manager * 1.1:
                    files = [p.name for p in pathlib.Path(self.data_dir).iterdir() if p.is_file()]
                    random.shuffle(files)
                    while self.timesteps_in_buffer > self.params.replay_buffer_size_timesteps_per_manager:
                        file = files.pop()
                        try:
                            (self.data_dir / file).unlink()
                            self.timesteps_in_buffer -= parse_filename(str(file))["timesteps"]
                        except FileNotFoundError:
                            # This happens if two processes are cleaning up simultaneously.
                            continue

                self.log_stats()

    def save_episode(self, episode: SequenceData) -> None:
        """Write an episode to disk."""
        # if self.params.observation_compression:
        #     for k in ("rgb", "rgbd"):
        #         if k in episode.observation:
        #             episode.observation[k] = pack(episode.observation[k])
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        identifier = str(uuid.uuid4().hex)
        length = len(episode.done)
        filename = self.data_dir / f"{timestamp}-{identifier}-{length}.npz"
        with filename.open("wb") as f2:
            pickle.dump(episode, f2)

    def shutdown(self):
        shutil.rmtree(str(self.data_dir), ignore_errors=True)


class LambdaStorage(TrajectoryStorage):
    """
    Ephemeral pass-through storage for when we're looking to extract some metrics out of a large number of fragments/eps
    that would be too costly to store on-disk or in memory.

    When storage_mode == StorageMode.FRAGMENT, calls all hooks on the fragment and discards them.
    When storage_mode == StorageMode.EPISODE, stores trajectories until episode is complete, then calls hooks and then
    discards.
    """

    def __init__(
        self, params: Params, lambdas: Tuple[Callable[[List[StepData]], None], ...], storage_mode: StorageMode
    ):
        self.params = params
        self.lambdas = lambdas
        self.storage_mode = storage_mode
        self.storage: Dict[str, List[StepData]] = defaultdict(list)

    def add_timestep_samples(self, samples: Dict[str, StepData]) -> None:
        for key, step_data in samples.items():
            self.storage[key].append(step_data)

        purge_keys = []
        for key, trajectory in self.storage.items():
            if self.storage_mode == StorageMode.FRAGMENT or trajectory[-1].done:
                for hook in self.lambdas:
                    hook(trajectory)
                purge_keys.append(key)

        for key in purge_keys:
            del self.storage[key]


# Currently not using this because the speedup was minimal, but leaving it here because there's a high chance
# we'll decide to come back to it. I've already gone back and forth between these a couple times so far.
# class FixedSizeStorage(TrajectoryStorage):
#     """A storage optimized for storing a known shape of data.
#
#     This one is currently intended to be a drop-in replacement for InMemoryStorage,
#     but optimized for PPO. It's a bit hard-coded to that case right now, in fact.
#
#     In practice the perf gain here isn't terribly significant, but pack_batch was inconsistent there.
#
#     But, we could extend this to being in shared memory and having each worker put samples directly in here,
#     which might have some further gain vs sending the data over pipes."""
#
#     def __init__(self, params: Params):
#         self.params = params
#         self.reset()
#
#     def add_timestep_samples(self, samples: Dict[str, StepData]) -> None:
#         """Add a set of samples representing a single timestep, keyed by a batch/episode identifer."""
#         for worker_id_str, worker_obs in samples.items():
#             worker_id = int(worker_id_str)
#             for k, v in worker_obs.items():
#                 if k == "info":
#                     self.infos[worker_id].append(v)
#                     continue
#                 self.storage[k][worker_id, self.step[worker_id]] = v
#             self.step[worker_id] += 1
#
#     def to_packed(self) -> StepData:
#         return self.storage
#
#     def reset(self) -> None:
#         # TODO: don't need to create all these shapes each step - just overwrite the old data.
#         self.storage = {
#             f"obs__{k}": np.zeros((self.params.num_workers, self.params.num_steps, *v.shape), dtype=v.dtype)
#             for k, v in self.params.observation_space.items()
#         }
#         for k, v in self.params.action_space.items():
#             if isinstance(v, OneHotMultiDiscrete):
#                 self.storage[f"action__{k}"] = np.zeros(
#                     (self.params.num_workers, self.params.num_steps, *v.shape, v.max_categories), dtype=v.dtype
#                 )
#             elif isinstance(v, Box):
#                 self.storage[f"action__{k}"] = np.zeros(
#                     (self.params.num_workers, self.params.num_steps, *v.shape), dtype=v.dtype
#                 )
#             else:
#                 assert False
#
#         self.storage["rewards"] = np.zeros((self.params.num_workers, self.params.num_steps), dtype=np.float32)
#         self.storage["dones"] = np.zeros((self.params.num_workers, self.params.num_steps), dtype=np.bool)
#         self.storage["is_terminal"] = np.zeros((self.params.num_workers, self.params.num_steps), dtype=np.bool)
#         self.storage["values"] = np.zeros((self.params.num_workers, self.params.num_steps), dtype=np.float32)
#         self.storage["policy_probs"] = np.zeros((self.params.num_workers, self.params.num_steps), dtype=np.float32)
#         # different workers can be on different steps
#         self.step = [0] * self.params.num_workers
#         self.infos = defaultdict(list)


def pack(data: Any):  # type: ignore
    """Compress any python type."""
    import lz4.frame

    # Observed compression ratio on synthetic godot numpy data: 17MB -> 2-3MB
    data = pyarrow.serialize(data).to_buffer().to_pybytes()
    data = lz4.frame.compress(data)
    return data


def unpack(data):  # type: ignore
    """Uncompress something compressed with `pack`."""
    import lz4.frame

    data = lz4.frame.decompress(data)
    data = pyarrow.deserialize(data)
    return data


def parse_filename(filename: str) -> Dict:
    """Split a filename into its components."""
    # 20220610T123952-edd6a7dfbff544939976663265fb4147-40.npz
    timestamp, hash, timesteps = filename.split(".")[0].split("-")
    return {"timestamp": timestamp, "hash": hash, "timesteps": int(timesteps)}


def measure_buffer_size(folder: str) -> int:
    """Calculate the total number of frames in the given folder."""
    folder_path = Path(folder)
    files = [p.name for p in pathlib.Path(folder_path).iterdir() if p.is_file()]
    return sum([parse_filename(filename)["timesteps"] for filename in files])
