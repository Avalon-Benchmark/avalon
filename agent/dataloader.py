import datetime
import io
import pathlib
import random
import uuid

import numpy as np
from torch.utils.data import IterableDataset


class ReplayDataset(IterableDataset):
    def __init__(self, args, storage_dir, update_interval=100):
        self.args = args
        self.cache = {}
        # For O(1) key sampling. Must manually maintain that set(self.storage.keys()) == set(self.storage_keys)
        self.cache_keys = []
        self.storage_dir = pathlib.Path(storage_dir).expanduser()
        self.time_since_last_update = None
        self.update_interval = update_interval

    def load_into_cache(self):
        # TODO: design a scheme where each dataloader only loads/uses a slice of the whole dataset.
        # So we're not duplicating the whole dataset in each dataloader's memory.
        # Ie hash the filename to an integer hash, take the last 5 digits, and split into n segments or something.
        # Also allow setting a memory limit so we can load huge datasets dynamically.
        for filename in self.storage_dir.glob("*.npz"):
            if filename not in self.cache:
                try:
                    with filename.open("rb") as f:
                        episode = np.load(f)
                        # TODO: why?
                        episode = {k: episode[k] for k in episode.keys()}
                except Exception as e:
                    print(f"Could not load episode: {e}")
                    continue
                self.cache[filename] = episode
                self.cache_keys.append(filename)

    def load_episode(self):
        """
        Samples episodes randomly. May not be ideal if episodes are of widely varying length;
        may be preferrable to weight samples by length.

        Returns episodes of varying lengths, if a frame is sampled too close to episode end.
        May cause difficulties with batched training.

        - rescan: how often to update the list of episodes from disk
        """
        if self.time_since_last_update is None or self.time_since_last_update > self.update_interval:
            self.load_into_cache()
            self.time_since_last_update = 0
        else:
            self.time_since_last_update += 1

        assert len(self.cache) > 0

        while True:
            assert self.args.min_fragment_len == self.args.max_fragment_len
            fragment_length = self.args.min_fragment_len
            episode = self.cache[random.choice(self.cache_keys)]
            total = len(episode[list(episode.keys())[0]])
            available = total - fragment_length
            if available < 1:
                print(f"Skipped short episode of length {total}.")
                continue
            if self.args.balanced_sampler:
                index = min(random.randint(0, total), available)
            else:
                index = int(random.randint(0, available))
            # TODO: Fix this
            # episode = episode[index : index + fragment_length]
            sliced_episode = {k: v[index : index + fragment_length] for k, v in episode.items()}
            assert all([len(v) == fragment_length for v in sliced_episode.values()])
            return sliced_episode

    def __iter__(self):
        # Sending the data to the GPU here might save some time.
        # note: rescan interval logic doesn't quite make sense. should be train_steps * batch size.
        # but then that doesn't play friendly with rescan interval. as is we rescan every couple batches i think.
        while True:
            yield self.load_episode()


def save_episodes(directory, episodes):
    # TODO: prepend a mode
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    for episode in episodes:
        identifier = str(uuid.uuid4().hex)
        length = len(episode[list(episode.keys())[0]])
        filename = directory / f"{timestamp}-{identifier}-{length}.npz"
        with io.BytesIO() as f1:
            np.savez_compressed(f1, **episode)
            f1.seek(0)
            with filename.open("wb") as f2:
                f2.write(f1.read())
