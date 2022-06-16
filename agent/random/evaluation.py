# %%
import json
import multiprocessing as mp
import os
import tarfile
import traceback
from datetime import datetime
from pathlib import Path
from random import shuffle
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import sentry_sdk
import torch

from agent.evaluation import EVAL_TEMP_PATH
from agent.evaluation import EvaluationGodotEnv
from agent.evaluation import get_world_folders
from agent.evaluation import world_id_from_world_folder_name
from common.log_utils import logger
from contrib.s3_utils import SimpleS3Client
from datagen.godot_env import VRActionType


class RandomActionDistribution:
    def __init__(self, real_log_stds: Sequence[float], discrete_logits: Sequence[float], seed: int):
        torch.random.manual_seed(seed)
        real_scales = torch.tensor(real_log_stds).exp()
        real_means = torch.zeros_like(real_scales)
        self.real_dist = torch.distributions.Normal(loc=real_means, scale=real_scales)
        self.discrete_dist = torch.distributions.Bernoulli(logits=torch.tensor(discrete_logits))

    def sample(self):
        return {"real": self.real_dist.sample(), "discrete": self.discrete_dist.sample()}


@torch.no_grad()
def do_eval_rollouts(policy: RandomActionDistribution, env: EvaluationGodotEnv):
    # DO ROLLOUTS
    observation = env.reset()
    frame_count = 1
    scores_by_world_index: Dict[str, float] = {}
    while True:
        action = VRActionType.from_input(policy.sample())
        observation, goal_progress = env.act(action)
        frame_count += 1
        world_index = goal_progress.log["world_index"]

        if world_index in scores_by_world_index:
            break
        if goal_progress.is_done:
            info = goal_progress.log
            print(
                f"Episode {world_index}, {info['task']} ({info['difficulty']}) score: {info['score']:.3f} in {frame_count} frames"
            )
            scores_by_world_index[world_index] = goal_progress.log["score"]

            env.reset()
            frame_count = 1
    return scores_by_world_index


def run_policy_on_world_ids(eval_world_dir: str, world_ids: Tuple[int, ...], seed: int):
    policy = RandomActionDistribution([0] * 18, [0] * 3, seed=seed)
    env = EvaluationGodotEnv(eval_world_dir=eval_world_dir)
    env.set_eval_world_ids(world_ids)
    result = do_eval_rollouts(policy, env)
    return result


def get_random_result_key(seed: int, data_key: str):
    return "__".join(["random_policy", str(seed), data_key]).replace("/", "_")


def run_evaluation(worlds_path: Path, data_key: Optional[str], seed: int):

    s3_client = SimpleS3Client()

    sentry_sdk.init(
        dsn="https://198a62315b2c4c2a99cb8a5493224e2f@o568344.ingest.sentry.io/6453090",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
    )

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
    world_id_list = [world_id_from_world_folder_name(x) for x in get_world_folders(str(worlds_path))]
    shuffle(world_id_list)
    num_worlds_per_process = len(world_id_list) // num_processes + min(1, len(world_id_list) % num_processes)

    process_world_ids = [
        tuple(world_id_list[i : i + num_worlds_per_process])
        for i in range(0, len(world_id_list), num_worlds_per_process)
    ]
    logger.info(f"Splitting workload into {[len(x) for x in process_world_ids]}")

    all_results = {}
    all_errors = []

    def on_done(result):
        logger.info(f"Finished generating {result}")
        all_results.update(result)

    def on_error(error: BaseException):
        logger.error(f"Evaluation failed! {error}")
        traceback.print_exc()
        sentry_sdk.capture_exception(error)

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=num_processes) as worker_pool:
        requests = []
        for world_ids in process_world_ids:
            request = worker_pool.apply_async(
                run_policy_on_world_ids,
                kwds={"eval_world_dir": str(worlds_path), "world_ids": world_ids, "seed": seed},
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
    s3_client.save(result_key, json.dumps(record_data))


if __name__ == "__main__":
    seed = 0
    worlds_path = Path("/tmp/avalon_worlds/b21aeff3-59c5-49a8-9c19-f3feceea1b8a")
    data_key = "avalon_worlds__b21aeff3-59c5-49a8-9c19-f3feceea1b8a.tar.gz"
    run_evaluation(worlds_path=worlds_path, data_key=data_key, seed=seed)
