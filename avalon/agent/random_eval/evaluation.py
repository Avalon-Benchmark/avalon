# %%
import json
import multiprocessing as mp
import os
import tarfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Sequence

import sentry_sdk
import torch

from avalon.agent.godot.godot_gym import AvalonEnv
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.common.error_utils import setup_sentry
from avalon.common.log_utils import configure_parent_logging
from avalon.common.log_utils import configure_remote_logger
from avalon.common.log_utils import logger
from avalon.contrib.s3_utils import SimpleS3Client
from avalon.contrib.utils import FILESYSTEM_ROOT
from avalon.datagen.godot_env.actions import VRAction
from avalon.datagen.godot_env.godot_env import GodotEnv

EVAL_TEMP_PATH = "/tmp/avalon_eval"


class RandomActionDistribution:
    def __init__(self, real_log_stds: Sequence[float], discrete_logits: Sequence[float], seed: int) -> None:
        torch.random.manual_seed(seed)
        real_scales = torch.tensor(real_log_stds).exp()
        real_means = torch.zeros_like(real_scales)
        self.real_dist = torch.distributions.Normal(loc=real_means, scale=real_scales)
        self.discrete_dist = torch.distributions.Bernoulli(logits=torch.tensor(discrete_logits))

    def sample(self):
        return {"real": self.real_dist.sample(), "discrete": self.discrete_dist.sample()}


@torch.no_grad()
def do_eval_rollouts(policy: RandomActionDistribution, env: GodotEnv) -> Dict[str, float]:
    # DO ROLLOUTS
    observation = env.reset()
    frame_count = 1
    scores_by_world_index: Dict[str, float] = {}
    while True:
        action = VRAction.from_input(policy.sample())
        observation, goal_progress = env.act(action)
        frame_count += 1
        world_index = goal_progress.log["world_index"]

        if world_index in scores_by_world_index:
            break
        if goal_progress.is_done:
            info = goal_progress.log
            logger.info(
                f"Episode {world_index}, {info['task']} ({info['difficulty']}) score: {info['score']:.3f} in {frame_count} frames"
            )
            scores_by_world_index[world_index] = goal_progress.log["score"]

            env.reset()
            frame_count = 1
    return scores_by_world_index


def run_policy_on_world_ids(
    eval_world_dir: str, process_index: int = 0, num_processes: int = 1, seed: int = 0
) -> Dict[str, float]:
    policy = RandomActionDistribution([0] * 18, [0] * 3, seed=seed)
    params = GodotEnvironmentParams(
        mode="test", fixed_worlds_load_from_path=Path(eval_world_dir), env_index=process_index, env_count=num_processes
    )
    env = AvalonEnv(params)
    result = do_eval_rollouts(policy, env)
    return result


def get_random_result_key(seed: int, data_key: str):
    return "__".join(["random_policy", str(seed), data_key]).replace("/", "_")


def run_evaluation(worlds_path: Path, data_key: str, seed: int) -> None:

    s3_client = SimpleS3Client()

    setup_sentry(percent_of_traces_to_capture=1.0)

    if data_key is None:
        assert worlds_path.is_dir()
    else:
        logger.info(f"Downloading data from {data_key} to {worlds_path}")
        os.makedirs(worlds_path, exist_ok=True)
        os.makedirs(EVAL_TEMP_PATH, exist_ok=True)
        s3_client.download_to_file(data_key, Path(EVAL_TEMP_PATH) / data_key)
        with tarfile.open(Path(EVAL_TEMP_PATH) / data_key, "r:gz") as f:
            f.extractall(worlds_path)

    num_processes = 10
    all_results = {}

    def on_done(result: Dict[str, float]) -> None:
        logger.info(f"Finished generating {result}")
        all_results.update(result)

    def on_error(error: BaseException) -> None:
        logger.error(f"Evaluation failed! {error}")
        traceback.print_exc()
        sentry_sdk.capture_exception(error)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_processes, initializer=configure_parent_logging) as worker_pool:
        requests = []
        for process_index in range(num_processes):
            request = worker_pool.apply_async(
                run_policy_on_world_ids,
                kwds={
                    "eval_world_dir": str(worlds_path),
                    "process_index": process_index,
                    "num_processes": num_processes,
                    "seed": seed,
                },
                callback=on_done,
                error_callback=on_error,
            )
            requests.append(request)
        worker_pool.close()
        worker_pool.join()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record_data = {
        "baseline": "random",
        "data_key": data_key,
        "eval_world_dir": str(worlds_path),
        "start_time": start_time,
        "end_time": end_time,
        "all_results": all_results,
    }

    result_key = get_random_result_key(seed, data_key)
    logger.info(f"Saving result to '{result_key}'")
    s3_client.save(result_key, json.dumps(record_data))  # type: ignore


if __name__ == "__main__":
    configure_remote_logger()
    seed = 0
    worlds_path = Path(f"{FILESYSTEM_ROOT}/avalon/worlds/viewable_worlds")
    data_key = "avalon_worlds__0824_full.tar.gz"
    run_evaluation(worlds_path=worlds_path, data_key=data_key, seed=seed)
