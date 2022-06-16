import random
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import numpy.typing as npt

from agent.dataloader import save_episodes

TransitionType = Dict[str, npt.NDArray]


def pack_list_of_dicts(data: List[Dict[str, npt.NDArray]], ignore_keys=("info",)) -> Dict[str, npt.NDArray]:
    """The 1-d version of pack_batch."""
    out = {}
    for key in data[0].keys():
        if key in ignore_keys:
            continue
        out[key] = np.stack([x[key] for x in data], axis=0)
        # out[key] = torch.stack([x[key] for x in data], dim=0)
    return out


def pack_batch(data: List[List[TransitionType]]) -> TransitionType:
    """Pack a list of lists of dicts into a dict of ndarrays.

    (batch_size, num_steps, dict[str, (*atom_shape)]) -> dict[str, (batch_size, num_steps, *atom_shape)].
    """
    batch_size = len(data)
    num_steps = len(data[0])
    out = {}
    for key in data[0][0].keys():
        if key == "info":
            # Don't use these for training.
            continue
        out[key] = np.stack([np.stack([transition[key] for transition in trajectory]) for trajectory in data])
        assert out[key].shape[:2] == (batch_size, num_steps)
    return out


class TrajectoryStorage:
    """
    We store everything here in lists. So the actual ndarray atoms are a single timestep of a single obs.
    This is for efficient single additions when the size isn't known in advance.
    If we always add entire episodes, it may be more efficient to collapse into large ndarrays.
    Could have two instances of this class for both cases if needed, if the implementation is sufficiently abstracted.
    """

    def __init__(self, args, mode="train"):
        # The mode flag here is only relevant if we're writing to disk
        # Storage is like: {"episode_0": [{"step": 0, "obs_rgb": ...}, {"step": 1, "obs_rgb": ...}]}
        self.ongoing: Dict[str, List[TransitionType]] = defaultdict(list)
        self.storage: Dict[str, List[TransitionType]] = defaultdict(list)
        # For O(1) key sampling. Must manually maintain that set(self.storage.keys()) == set(self.storage_keys)
        self.storage_keys = []
        # Do we store ongoing episodes separately from the completed ones?
        # Usually want this to be True if doing off-policy learning on many epsisodes,
        # but false eg for PPO where we're only storing fragments anyways.
        self.separate_ongoing = args.separate_ongoing
        self.args = args
        self.mode = mode

        assert not ((not self.args.separate_ongoing) and self.args.write_episodes_to_disk), "this combo doesn't work"

    def add_timestep_samples(self, samples: Dict[str, TransitionType]):
        """Add a set of samples representing a single timestep, keyed by a batch/episode identifer."""
        for key, v in samples.items():
            if self.separate_ongoing:
                self.ongoing[key].append(v)
                if v["dones"]:
                    self.storage[key] = self.ongoing[key]
                    del self.ongoing[key]
                    self.storage_keys.append(key)

                    if self.args.write_episodes_to_disk:
                        ep = pack_list_of_dicts(self.storage[key])
                        save_episodes(Path(self.args.data_dir) / self.mode, [ep])
            else:
                if key not in self.storage:
                    self.storage_keys.append(key)
                self.storage[key].append(v)

    def sample_fragment(self, min_fragment_len: int, max_fragment_len: int, balance=True) -> List[TransitionType]:
        """Sample a valid fragment from the episode (or whole episode if fragment_length=None).

        With balance=False (danijar's default, although I think the paper refers to balance=True?),
        just uniformly sample from valid fragments in the episode.
        This means the chance of seeing a terminal transition is ~1/ep_len.
        balance=True increases this likelihood by increasing the likelihood of sampling the final fragment.
        """
        while True:
            # TODO: finish up the short-episode sampler here
            # episode = self.storage[random.choice(self.storage_keys)]
            # total = len(episode)
            # length = min(total, max_fragment_len)
            # # Randomize length to avoid all chunks ending at the same time in case the
            # # episodes are all of the same length.
            # # TODO: why would this matter?
            # length -= np.random.randint(max_fragment_len)
            # length = max(min_fragment_len, length)
            # upper = total - length + 1
            # if self._prioritize_ends:
            #     upper += min_fragment_len
            # index = min(np.random.randint(upper), total - length)
            # TODO: is_first??
            assert min_fragment_len == max_fragment_len
            fragment_length = min_fragment_len
            episode = self.storage[random.choice(self.storage_keys)]
            total = len(episode)
            available = total - fragment_length
            if available < 1:
                # TODO: figure out a better way of dealing with this.
                print(f"Skipped short episode of length {total}.")
                continue
            if balance:
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available))
            episode = episode[index : index + fragment_length]
            return episode

    def sample_fragment_batch(
        self, batch_size: int, min_fragment_len: int, max_fragment_len: int, balance: bool = True
    ) -> List[List[TransitionType]]:
        """Use if this buffer stores entire episodes."""
        return [self.sample_fragment(min_fragment_len, max_fragment_len, balance) for _ in range(batch_size)]

    def to_packed(self) -> TransitionType:
        return pack_batch(list(self.storage.values()))

    def reset(self) -> None:
        self.storage = defaultdict(list)
