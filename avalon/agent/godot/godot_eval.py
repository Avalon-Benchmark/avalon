import os
import shutil
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from threading import Thread
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import attrs
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.pyplot import bar
from numpy.typing import NDArray

from avalon.agent.common import wandb_lib
from avalon.agent.common.params import Params
from avalon.agent.common.storage import EpisodeStorage
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import StepData
from avalon.agent.common.worker import RolloutManager
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.ppo.params import PPOParams
from avalon.agent.ppo.ppo_types import PPOStepData
from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import TEMP_BUCKET_NAME
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.utils import TEMP_DIR
from avalon.contrib.utils import create_temp_file_path
from avalon.datagen.world_creation.constants import int_to_avalon_task

BIG_SEPARATOR = "-" * 80
RESULT_TAG = "DATALOADER:0 TEST RESULTS"


def log_rollout_stats_packed(packed_rollouts: Dict[str, NDArray], i: int) -> None:
    """Log stats, when rollouts is a dict of packed BatchSequence tensors."""
    successes: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    keys = ["success", "difficulty"]
    if packed_rollouts["done"].sum() == 0:
        return
    for worker, timestep in np.argwhere(packed_rollouts["done"]).T:
        info = packed_rollouts["info"]
        task = int_to_avalon_task[int(info["task"][worker, timestep])].lower()
        for field in keys:
            successes[task][f"final_{field}"].append(info[field][worker, timestep])
    # Data is a dict (task) of dicts (keys) of lists
    for task, x in successes.items():
        for field, y in x.items():
            wandb_lib.log_histogram(f"train/{task}/{field}", y, i, hist_freq=10)
        wandb_lib.log_scalar(f"train/{task}/num_episodes", len(y), i)


Episode = List[StepData]
DifficultyBin = Tuple[float, float]  # [from, to)
EpisodesByDifficulty = DefaultDict[DifficultyBin, List[Episode]]
EpisodesByTaskByDifficulty = DefaultDict[str, EpisodesByDifficulty]


def get_difficulty_bin_name(difficulty_bin: DifficultyBin) -> str:
    return f"{difficulty_bin[0]:.2f}_to_{difficulty_bin[1]:.2f}"


def get_episode_difficulty_bin(episode: Episode, difficulty_bin_size: float) -> DifficultyBin:
    difficulty_bins = np.arange(difficulty_bin_size, 1 + difficulty_bin_size, difficulty_bin_size)  # interval ends
    difficulty = episode[-1].info["difficulty"]
    difficulty_bin_idx = np.digitize(difficulty, difficulty_bins, right=True)
    difficulty_bin_end = difficulty_bins[difficulty_bin_idx]
    difficulty_bin_start = difficulty_bin_end - difficulty_bin_size
    difficulty_bin = difficulty_bin_start, difficulty_bin_end
    return difficulty_bin


def log_video_by_difficulty(
    episode: Episode, difficulty_bin: DifficultyBin, prefix: str = "test", infix: str = ""
) -> None:
    if infix != "":
        infix += "/"
    difficulty_bin_name = get_difficulty_bin_name(difficulty_bin)
    video = torch.stack([step.observation["rgbd"] for step in episode])
    wandb_lib.log_video(f"{prefix}/videos/{infix}{difficulty_bin_name}", video, step=None, normalize=True, freq=1)


def log_success_by_difficulty(
    successes_by_difficulty: DefaultDict[DifficultyBin, List[int]], prefix: str = "test", suffix: Optional[str] = None
) -> None:
    if suffix is not None:
        suffix = f"/{suffix}"
    data = []
    for difficulty_bin, successes in successes_by_difficulty.items():
        difficulty_bin_name = get_difficulty_bin_name(difficulty_bin)
        success_rate = np.mean(successes)
        data.append((difficulty_bin_name, success_rate))
    data = sorted(data, key=lambda item: item[0])
    difficulty_bins, success_rates = zip(*data)
    fig = plt.figure()
    plt.ylabel("success rate")
    plt.xlabel("difficulty bin")
    plt.ylim((0, 1))
    bar(difficulty_bins, success_rates)
    wandb.log({f"{prefix}/success_by_difficulty{suffix}": wandb.Image(fig)})


def load_worlds_from_s3(data_key: str, target_path: Path, bucket_name: str = TEMP_BUCKET_NAME) -> int:
    """Download pre-generated worlds from a S3 tarball to a local directory."""
    logger.debug("started loading fixed worlds from s3 file")
    shutil.rmtree(str(target_path), ignore_errors=True)
    target_path.mkdir(parents=True)
    s3_client = SimpleS3Client(bucket_name)
    logger.info(f"Downloading data from {data_key} to {target_path}")
    with create_temp_file_path() as temp_file_path:
        s3_client.download_to_file(data_key, temp_file_path)
        with tarfile.open(temp_file_path, "r:gz") as f:
            f.extractall(target_path)
    # TODO: figure out a better way to count episodes?
    episode_count = len(os.listdir(str(target_path)))
    logger.debug("finished loading fixed worlds from s3 file")
    return episode_count


def test(params: Params, model: Algorithm, log: bool = True, log_extra: Optional[Dict[str, float]] = None):
    """Run evaluation for Godot."""
    params = attrs.evolve(params, env_params=attrs.evolve(params.env_params, mode="test"))
    # We have to pull env_params out to make mypy happy - type narrowing doesn't seem to work on instance attributes
    env_params = params.env_params
    assert isinstance(env_params, GodotEnvironmentParams)
    assert env_params.env_index == 0

    start_time = time.monotonic()

    if env_params.fixed_worlds_s3_key:
        # Load worlds from S3, if we have that enabled
        # e.g. "a5101fb5fca577a35a0749ba45ae28006823136f/test_worlds.tar.gz"
        # the worlds typically have to be put in a specific folder because they contain absolute paths...
        fixed_worlds_path = env_params.fixed_worlds_load_from_path
        if not fixed_worlds_path:
            fixed_worlds_path = Path(TEMP_DIR) / "eval_worlds"
        num_worlds = load_worlds_from_s3(env_params.fixed_worlds_s3_key, fixed_worlds_path)
        env_params = attrs.evolve(env_params, fixed_worlds_load_from_path=fixed_worlds_path)
        params = attrs.evolve(params, env_params=env_params)
    elif env_params.fixed_worlds_load_from_path:
        # Got a path but no s3 key, assume the files are already locally available
        num_worlds = len(os.listdir(str(env_params.fixed_worlds_load_from_path)))
    else:
        num_worlds = env_params.test_episodes_per_task * env_params.num_tasks

    model.reset_state()
    model.eval()

    # Set up hooks for extracting episode info we care about
    difficulty_bin_size = env_params.eval_difficulty_bin_size
    seen_worlds: set[int] = set()
    success_by_task_and_difficulty_bin: DefaultDict[str, DefaultDict[DifficultyBin, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    world_scores = {}
    video_logging_threads = []

    def collect_episode_stats(episode: Episode) -> None:
        world_index = episode[-1].info["world_index"]
        if world_index in seen_worlds:
            # We may run the same world twice, don't count them twice!
            logger.debug(f"got dupe world {world_index}")
            return
        else:
            seen_worlds.add(world_index)
        task = int_to_avalon_task[int(episode[-1].info["task"])].lower()
        success = episode[-1].info["success"]
        difficulty_bin = get_episode_difficulty_bin(episode, difficulty_bin_size)
        success_by_task_and_difficulty_bin[task][difficulty_bin].append(success)
        is_first_episode_in_group = len(success_by_task_and_difficulty_bin[task][difficulty_bin]) == 1
        if log and is_first_episode_in_group:
            # Note: wandb.log is not thread safe, this might cause issues. But it's fast :)
            thread = Thread(
                target=log_video_by_difficulty, args=(episode, difficulty_bin), kwargs={"infix": task}, daemon=True
            )
            thread.start()
            video_logging_threads.append(thread)
            # log_video_by_difficulty(episode, difficulty_bin, infix=task)

        # Stuff for collecting scores
        world_scores[world_index] = episode[-1].info["score"]

        # difficulty = episode[-1].info["difficulty"]
        # success = episode[-1].info["success"]
        # episode_length = len(episode)
        # logger.info(f"ep with difficulty {difficulty} had success {success} in {episode_length} steps")

    num_workers = min(params.eval_workers, num_worlds)

    if isinstance(params, PPOParams):
        step_data_type = PPOStepData
    else:
        step_data_type = StepData

    test_storage = EpisodeStorage(
        params,
        step_data_type,
        episode_callback=collect_episode_stats,
        num_workers=num_workers,
        discard_short_eps=False,
        in_memory_buffer_size=0,
    )

    # Maybe i should make a "free-running", "n_steps", and "n_episodes" worker.
    # And use the n_episodes version for eval.
    multiprocessing_context = torch.multiprocessing.get_context("spawn")
    assert params.observation_space is not None
    player = RolloutManager(
        params=params,
        num_workers=num_workers,
        is_multiprocessing=True,
        storage=test_storage,
        obs_space=params.observation_space,
        model=model,
        rollout_device=torch.device("cuda:0"),
        multiprocessing_context=multiprocessing_context,
    )
    test_storage.reset()

    try:
        logger.info(f"running {num_worlds} evaluation episodes")
        # Will potentially run some worlds multiple times
        player.run_rollout(
            num_episodes=int(np.ceil(num_worlds / num_workers)), exploration_mode=params.eval_exploration_mode
        )
        logger.debug("finished rollout, shutting down workers")
    finally:
        player.shutdown()

    [thread.join() for thread in video_logging_threads]
    end_time = time.monotonic()
    test_log: dict[str, float] = {"test_time": end_time - start_time}
    if log_extra is not None:
        test_log.update(log_extra)
    total_episodes_logged = 0
    all_successes: list[float] = []
    for task, success_by_difficulty_bin in success_by_task_and_difficulty_bin.items():
        if log:
            log_success_by_difficulty(success_by_difficulty_bin, suffix=task)
        task_successes: list[int] = sum(list(success_by_difficulty_bin.values()), [])
        all_successes.extend(task_successes)
        test_log[f"{task}_success_rate"] = float(np.mean(task_successes))
        for difficulty_bin, successes in success_by_difficulty_bin.items():
            total_episodes_logged += len(successes)
    test_log["overall_success_rate"] = float(np.mean(all_successes))
    logger.info(success_by_task_and_difficulty_bin)
    assert (
        total_episodes_logged >= num_worlds
    ), f"Expected to log at least {num_worlds}, but only logged {total_episodes_logged}"

    if log:
        wandb.log({f"test/{k}": v for k, v in test_log.items()})
        logger.info(BIG_SEPARATOR)
        logger.info(RESULT_TAG)
        logger.info(test_log)
        logger.info(BIG_SEPARATOR)

    # if env_params.fixed_worlds_load_from_path:
    #     # Special logging for the fixed evaluation worlds
    #     project = params.resume_from_project if params.resume_from_project else params.project
    #     # TODO: make this get the current wandb run_id or some other identifier if we're not loading a run
    #     run_id = params.resume_from_run
    #     filename = params.resume_from_filename
    #     fixed_world_key = env_params.fixed_worlds_s3_key if env_params.fixed_worlds_s3_key else "test"
    #     result_key = f"avalon_eval__{project}_{run_id}_{filename}__{fixed_world_key}__final"
    #     record_data = {
    #         "wandb_run": f"{project}/{run_id}/{filename}",
    #         "baseline": "PPO",
    #         "data_key": fixed_world_key,
    #         "all_results": world_scores,
    #     }
    #     logger.debug(record_data)
    #
    #     logger.info(f"Saving result to '{result_key}'")
    #     s3_client = SimpleS3Client()
    #     s3_client.save(result_key, json.dumps(record_data).encode())

    return test_log
