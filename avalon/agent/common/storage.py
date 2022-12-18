"""
The challenge here is how to make classes that can quickly ingress data in a format convenient
to the rollout worker, store the data efficiently,
and equally quickly egress the data in a format useful to the various algorithms.
Ideally we would have a minimal set of storage classes that can provide good performance across all uses.
"""
import collections
import datetime
import pathlib
import pickle
import random
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numba
import numpy as np
import torch
from loguru import logger
from torch import Tensor
from tree import map_structure

from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.types import SequenceData
from avalon.agent.common.types import StepData
from avalon.agent.common.util import create_action_storage
from avalon.agent.common.util import create_observation_storage
from avalon.agent.dreamer.params import OffPolicyParams
from avalon.agent.dreamer.params import Params
from avalon.datagen.world_creation.constants import int_to_avalon_task

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


class TrajectoryStorage:
    def __init__(self, params: Params, step_data_type: Type[StepData], num_workers: int) -> None:
        self.params = params
        self.num_workers = num_workers
        self.step_data_type = step_data_type

    def partial_batch_update(
        self,
        key: str,
        value: Union[dict, Tensor],
        mask: Tensor,
        timesteps: Tensor,
        is_extra_observation: bool = False,
    ) -> None:
        """This ingress method is optimized for the current rollout worker.
        The worker always passes a full batch of values, but not all are valid, nor are they all on the same timestep.
        So we pass extra fields giving this necessary info to construct a valid update to storage.
        - `key`: the storage key, eg `observation` or `done`
        - `value`: a batch of values (one for each rollout worker) (or a dict of those if the key has nested values)
        - `mask`: a boolean array indicating which values should be written to storage
        - `timesteps`: an integer array, indicating which timestep into the episode or fragment the value corresponds to.
        - `is_extra_observation`: on-policy algs require n+1 observations, for boostrapping; this flag indicates if this
            is that extra final observation. That obs should be ignored on other storage types.
        """
        raise NotImplementedError

    def mark_step_finished(self, done_idxs) -> None:
        """Indicate that a given rollout step is over; ie we have received a full set of keys.
        The expectation is that between calling this method, each key will be updated exactly once."""
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class EpisodeStorage(TrajectoryStorage):
    def __init__(  # type: ignore[no-untyped-def]
        self,
        params: OffPolicyParams,
        step_data_type: Type[StepData],
        num_workers: int,
        rollout_dir: Optional[str] = None,
        wandb_queue=None,  # not typeable, but a Optional[Queue] :(
        episode_callback: Optional[Callable[[list[StepData]], None]] = None,
        discard_short_eps: bool = True,  # should be true in training, false in eval
        in_memory_buffer_size: Optional[int] = None,  # if none, set to params.log_rollout_metrics_every
    ) -> None:
        """Episode-based storage.
        - if `rollout_dir` is None, won't save episodes to disk
        - if `wandb_queue` is None, won't log stats
        - if a `episode_callback` is given, will run this hook on every completed episode
        """
        super().__init__(params, step_data_type, num_workers)
        self.wandb_queue = wandb_queue
        self.episode_callback = episode_callback
        self.discard_short_eps = discard_short_eps

        self.store_to_disk = rollout_dir is not None
        if self.store_to_disk:
            self.data_dir = Path(rollout_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(self.data_dir)

        if in_memory_buffer_size is None:
            in_memory_buffer_size = params.log_rollout_metrics_every
        self.recent_eps: collections.deque[List[StepData]] = collections.deque(maxlen=in_memory_buffer_size)

        # {worker_id: {key: value, ...}, ...}
        self.current_step_data = defaultdict(lambda: defaultdict(dict))
        self.ongoing_episodes: list[List[StepData]] = [[] for _ in range(num_workers)]

        self.timesteps_in_buffer = 0
        self.total_timesteps = 0
        self.episode_counter = 0
        self.last_logged_episode = 0

    def partial_batch_update(
        self,
        key: str,
        value: Union[dict, Tensor],
        mask: Tensor,
        timesteps: Tensor,
        is_extra_observation: bool = False,
    ) -> None:
        if is_extra_observation:
            return

        # This code is a little weird - where we might expect "scalars", are actually single-element tensors,
        # because slicing one item from a 1-d tensor gives a single-item tensor (unlike in numpy).
        # But it is helpful for one reason, which is that the dtype is preserved -
        # a fp32 stays a fp32 vs if it were converted to a python float and then back to tensor,
        # it would end up as a fp64 tensor by default.

        # copying/cloning the tensor is necessary, since otherwise we're just saving a reference
        # to a specific slice of a tensor that is (potentially) reused in the worker.

        # Annoyingly there's not a great way to use a `map_structure` command here since the target atoms
        # can be immutable, or nonexisting. Thus the duplication.
        if isinstance(value, Tensor):
            for i, v in enumerate(value):
                if mask[i]:
                    self.current_step_data[i][key] = v.clone()
        elif isinstance(value, dict):
            for k2, v2 in value.items():
                for i, v in enumerate(v2):
                    if mask[i]:
                        self.current_step_data[i][key][k2] = v.clone()

    def mark_step_finished(self, done_idxs) -> None:
        """The worker uses this to indicate that it's finished with a rollout step."""
        for worker_id in [worker_id for worker_id, done in enumerate(done_idxs) if done]:
            v = self.current_step_data[worker_id]
            del self.current_step_data[worker_id]
            # StepData takes an info, don't need to filter it out here
            v = self.step_data_type(**v)
            self.ongoing_episodes[worker_id].append(v)
            if v.done:
                ep = self.ongoing_episodes[worker_id]
                self.ongoing_episodes[worker_id] = []
                ep_len = len(ep)
                if self.discard_short_eps and len(ep) < self.params.min_fragment_len:
                    logger.debug(f"not adding short ep of len {ep_len}")
                    continue
                self.episode_counter += 1
                # We do these before packing the ep because the infos are lost in the current packing code.
                self.recent_eps.append(ep)
                if self.episode_callback:
                    self.episode_callback(ep)
                packed_ep = v.pack_sequence(ep)

                self.timesteps_in_buffer += ep_len
                self.total_timesteps += ep_len
                self.save_episode(packed_ep)
                self.log_stats()

    def log_stats(self) -> None:
        # Compute training rollout statistics
        if not self.wandb_queue:
            return
        if not self.params.log_rollout_metrics_every:
            return
        if (self.episode_counter - self.last_logged_episode) < self.params.log_rollout_metrics_every:
            return
        self.last_logged_episode = self.episode_counter

        if self.params.env_params.suite == "godot":
            # Can't use log_rollout_stats because these are full episodes, not fragments,
            # so the analysis is a bit different. We can make stronger statements here.
            successes: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
            keys = ["success", "difficulty"]
            for episode in self.recent_eps:
                last_step = episode[-1]
                task = int_to_avalon_task[int(last_step.info["task"])].lower()
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
            # Note: action repeats have not been factored in here
            self.wandb_queue.put(("scalar", f"return_mean", np.mean(rewards)))
            self.wandb_queue.put(("scalar", f"length_mean", np.mean(lengths)))

        self.wandb_queue.put(("scalar", f"buffer_size_timesteps", self.timesteps_in_buffer))
        self.wandb_queue.put(("scalar", f"total_timesteps", self.total_timesteps))
        if self.store_to_disk:
            self.wandb_queue.put(("scalar", f"true_buffer_size", measure_buffer_size(str(self.data_dir))))
        self.wandb_queue.put(("scalar", f"total_episodes", self.episode_counter))

    def save_episode(self, episode: SequenceData) -> None:
        """Write an episode to disk."""
        if self.store_to_disk:
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

            self.enforce_buffer_size()

    def enforce_buffer_size(self) -> None:
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

    def shutdown(self) -> None:
        if self.store_to_disk:
            shutil.rmtree(str(self.data_dir), ignore_errors=True)


@numba.njit(parallel=False, cache=True)
def numba_assign(target, source, mask, timesteps) -> None:
    """There's not a good way (that I could find) to make this operation fast in torch/numpy natively.
    The operation, in numpy notation, is `target[mask][timesteps] = source[mask]`
    So this numba function provides a pretty significant speedup.
    """
    n = len(target)
    for worker_id in numba.prange(n):
        if mask[worker_id]:
            t = timesteps[worker_id]
            # This actual memory copy is maybe slower in numba than numpy for large arrays?
            target[worker_id, t] = source[worker_id]


class FragmentStorage(TrajectoryStorage):
    """In-memory storage for on-policy / PPO-like algorithms.
    These receive an array of fixed shape (num_envs, num_timesteps) each iteration for training.
    The storage is typically reset between iterations.

    The one exception is that the observation has shape (num_envs, num_timesteps+1), for value boostrapping.
    """

    def __init__(self, params: Params, step_data_type: Type[StepData], num_workers: int) -> None:
        super().__init__(params, step_data_type, num_workers)
        self.storage = self.build_storage()

    def mark_step_finished(self, *args, **kwargs) -> None:
        pass

    def partial_batch_update(
        self, key: str, value: Union[dict, Tensor], mask: Tensor, timesteps: Tensor, is_extra_observation: bool = False
    ) -> None:
        # This codepath is pretty carefully optimized; modify only if you know what you're doing!
        def assign(target, source) -> None:
            args = (target.numpy(), source.numpy(), mask.numpy(), timesteps.numpy())
            numba_assign(*args)

        map_structure(assign, self.storage[key], value)

    def to_packed(self) -> BatchSequenceData:
        return self.step_data_type.batch_sequence_type(**{k: v for k, v in self.storage.items() if k != "info"})

    def reset(self, fast_reset: bool = False, retain_final_obs: bool = True) -> None:
        # This relies on us replacing, not modifying the old observation when resetting.
        # Could do a deepcopy but unnecessary as-is.
        old_observation = self.storage["observation"]

        # A "fast reset" is actually just not resetting anything, lol. Just let the new overwrite the old.
        # Does potetially make debugging more confusing.
        if not fast_reset:
            self.storage = self.build_storage()

        if retain_final_obs:
            # We init obs[t=0] as the final obs from the last batch (that +1 extra obs)
            # This should typically always be used.
            for k, v in old_observation.items():
                self.storage["observation"][k][:, 0] = v[:, -1]

    def build_storage(self):
        storage = {
            # Note the extra timestep here
            "observation": create_observation_storage(
                self.params.observation_space, batch_shape=(self.num_workers, self.params.num_steps + 1)
            ),
            "action": create_action_storage(
                self.params.action_space, batch_shape=(self.num_workers, self.params.num_steps)
            ),
            "reward": torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.float32),
            "value": torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.float32),
            "policy_prob": torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.float32),
            "policy_entropy": torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.float32),
            "done": torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.bool),
            "is_terminal": torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.bool),
            "info": {
                k: torch.zeros((self.num_workers, self.params.num_steps), dtype=torch.float32)
                for k in self.params.env_params.info_fields
            },
        }
        return storage


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
