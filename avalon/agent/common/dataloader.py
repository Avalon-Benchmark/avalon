import pathlib
import pickle
import random
import time
from typing import Iterator

import attr
import numpy as np
from loguru import logger
from torch.utils.data import IterableDataset
from tree import map_structure

from avalon.agent.common.types import SequenceData
from avalon.agent.dreamer.params import OffPolicyParams

# TODO: consider redoing this with memory-mapped np arrays as the storage.
# Can load slices of them from disk (automatically cached by kernel as needed) without reading the whole file
# or having to keep everything in memory.
# Would be a bit tricky to implement though since we have a bunch of arrays per episode.
# I guess not so many that we wouldn't just have a file for each, although some files would be p small.


def partition_file(filename: str, i: int, num_workers: int) -> bool:
    """Given a filename of the form blah-UUID4-blah, like 20220608T113534-bfc1430dab5245dbac6d0a34f906244f-49.npz,
    decide if it is assigned to this worker."""
    # TODO: this janky method doesn't seem to make quiiiite an even split. But so far it's good enough.
    x = ord(filename.split("-")[1][0])
    return x % num_workers == i


WORKER_ID = 0


def worker_init_fn(i: int) -> None:
    """We want an incrementing id for each dataloader; I can't find a cleaner way than this."""
    global WORKER_ID
    WORKER_ID = i


class ReplayDataset(IterableDataset):
    """A dataset that loads episodes from disk and returns random fragments from those episodes.

    Episodes are cached in memory for higher throughput.
    The full dataset is split among the `params.num_dataloader_workers` workers to save memory.
    """

    def __init__(self, params: OffPolicyParams, storage_dir: str, update_interval: int = 100) -> None:
        self.params = params
        self.cache: dict[str, SequenceData] = {}
        # For O(1) key sampling. Must manually maintain that set(self.storage.keys()) == set(self.storage_keys)
        self.cache_keys: list[str] = []
        self.storage_dir = pathlib.Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.time_since_last_update = None
        self.update_interval = update_interval

        # Clear any old files in this folder, without deleting it (as other processes may already be accessing it).
        for f in self.storage_dir.iterdir():
            if f.is_file():
                f.unlink()

    def load_into_cache(self) -> None:
        """Update the cache to match the state of the disk folder."""
        global WORKER_ID
        disk_filenames = [f.name for f in sorted(pathlib.Path(self.storage_dir).iterdir()) if f.is_file()]
        # Shuffle so we're not loading a biased part of the dataset first
        random.shuffle(disk_filenames)
        loaded = 0
        for filename in disk_filenames:
            # only load our slice of the dataset
            if not partition_file(filename, WORKER_ID, self.params.num_dataloader_workers):
                continue
            if filename not in self.cache:
                try:
                    with open(self.storage_dir / filename, "rb") as f:
                        episode: SequenceData = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Could not load episode: {e}")
                    continue
                self.cache[filename] = episode
                self.cache_keys.append(filename)
                loaded += 1
                # # When we open an existing dataset, let's not load it all in upfront, which would block training..
                # if loaded > 500:
                #     logger.info("loaded 500 eps, continuing")
                #     break
        disk_filenames_set = set(disk_filenames)
        for filename in self.cache_keys:
            if filename not in disk_filenames_set:
                del self.cache[filename]
                self.cache_keys.remove(filename)

    def get_trajectory(self):
        """Randomly sample a single trajectory fragment from the dataset.

        Samples from episodes randomly. May not be ideal if episodes are of widely varying length;
        may be preferrable to weight samples by length.
        """
        global WORKER_ID
        if self.time_since_last_update is None or self.time_since_last_update > self.update_interval:
            # TODO: alter this somehow so that all dataloaders aren't updating their caches at the same time.
            self.load_into_cache()
            self.time_since_last_update = 0
        else:
            self.time_since_last_update += 1

        while len(self.cache) < self.params.prefill_eps_per_dataloader:
            logger.debug("waiting for samples")
            time.sleep(5)
            self.load_into_cache()

        while True:
            # Can relax this once we add the is_first handling and handle this properly in algos.
            # This codepath is otherwise ready for variable length episodes.
            assert self.params.min_fragment_len == self.params.max_fragment_len

            episode = self.cache[random.choice(self.cache_keys)]

            total = len(episode.done)
            assert total >= self.params.min_fragment_len
            length = total
            length = min(length, self.params.max_fragment_len)
            # Randomize length to avoid all chunks ending at the same time in case the
            # episodes are all of the same length.
            # TODO: why?
            # This will only do something if we have frames to spare
            # (and tbh could be improved such that with 55 frames available and a min of 50, we apply randomness more often than 10% of the time).
            length -= np.random.randint(self.params.min_fragment_len)
            length = max(self.params.min_fragment_len, length)
            upper = total - length + 1
            if self.params.prioritize_ends:
                upper += self.params.min_fragment_len
            index = min(np.random.randint(upper), total - length)
            assert index >= 0  # should be impossible, just double checking

            # sequence['is_first'] = np.zeros(len(sequence['action']), np.bool)
            # sequence['is_first'][0] = True

            # TODO: this won't work if we use a subclass of SequenceData.
            # Should make a func like attr.map_structure or something
            data = map_structure(lambda x: x[index : index + length], attr.asdict(episode))
            assert self.params.min_fragment_len <= len(data["done"]) <= self.params.max_fragment_len
            return SequenceData(**data)

    def __iter__(self) -> Iterator:
        while True:
            yield self.get_trajectory()
