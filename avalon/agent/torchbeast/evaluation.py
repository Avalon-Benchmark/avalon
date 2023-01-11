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
from typing import Tuple

import nest
import sentry_sdk
import torch

from avalon.agent.evaluation import EVAL_TEMP_PATH
from avalon.agent.evaluation import EvaluationGodotEnv
from avalon.agent.evaluation import get_wandb_result_key
from avalon.agent.evaluation import get_world_folders
from avalon.agent.evaluation import world_id_from_world_folder_name
from avalon.agent.torchbeast.avalon_helpers import force_cudnn_initialization
from avalon.agent.torchbeast.avalon_helpers import wrap_evaluation_godot_env
from avalon.agent.torchbeast.core.environment import Environment
from avalon.agent.torchbeast.polybeast_learner import Net
from avalon.common.error_utils import setup_sentry
from avalon.common.log_utils import configure_parent_logging
from avalon.common.log_utils import logger
from avalon.common.wandb_utils import load_checkpoint_from_wandb_run
from avalon.contrib.s3_utils import SimpleS3Client


@torch.no_grad()
def do_eval_rollouts(model: Net, env: Environment):
    force_cudnn_initialization()
    # DO ROLLOUTS
    observation = env.initial()
    core_state = model.initial_state(batch_size=1)
    core_state = nest.map(lambda t: t.to("cuda:0"), core_state)
    scores_by_world_index: Dict[int, float] = {}
    frame_count = 1
    while True:
        (flat_action, policy_logits, baseline), core_state = model(observation, core_state)
        observation = env.step(flat_action)
        frame_count += 1
        world_index = observation["info"]["world_index"]

        if world_index in scores_by_world_index:
            break
        if observation["done"].item():
            info = observation["info"]
            logger.info(
                f"Episode {world_index}, {info['task']} ({info['difficulty']}) score: {info['score']:.3f} in {frame_count} frames"
            )
            scores_by_world_index[world_index] = observation["info"]["score"]
            frame_count = 1
    return scores_by_world_index


def run_checkpoint_on_world_ids(
    checkpoint_path: str, eval_world_dir: str, world_ids: Tuple[int, ...], gpu_id: int = 0
):
    device = torch.device(f"cuda:{gpu_id}")
    env = wrap_evaluation_godot_env(EvaluationGodotEnv(eval_world_dir=eval_world_dir, gpu_id=gpu_id))
    env.gym_env.set_eval_world_ids(world_ids)
    model = Net(use_lstm=True).to(device=device)
    checkpoint_states = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    result = do_eval_rollouts(model, env)
    return result


def run_evaluation(worlds_path: Path, data_key: Optional[str], wandb_run: str, checkpoint_filename: str) -> None:

    s3_client = SimpleS3Client()
    result_key = get_wandb_result_key(wandb_run, checkpoint_filename, data_key)

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

    logger.info(f"Downloading checkpoint {checkpoint_filename} from wandb run {wandb_run}")
    checkpoint_path = load_checkpoint_from_wandb_run(wandb_run, checkpoint_filename)

    try:
        previous_result = json.loads(s3_client.load(result_key))
        all_results = previous_result["all_results"]
        logger.info(f"Skipping {len(all_results)} already completed worlds")
    except KeyError:
        all_results = {}
    all_errors = []

    num_processes = 5
    world_id_list = [world_id_from_world_folder_name(x) for x in get_world_folders(str(worlds_path))]
    world_id_list = [x for x in world_id_list if str(x) not in all_results]
    shuffle(world_id_list)
    num_worlds_per_process = len(world_id_list) // num_processes + min(1, len(world_id_list) % num_processes)

    process_world_ids = [
        tuple(world_id_list[i : i + num_worlds_per_process])
        for i in range(0, len(world_id_list), num_worlds_per_process)
    ]
    logger.info(f"Splitting workload into {[len(x) for x in process_world_ids]}")

    def on_done(result) -> None:
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
        for world_ids in process_world_ids:
            request = worker_pool.apply_async(
                run_checkpoint_on_world_ids,
                kwds={"checkpoint_path": checkpoint_path, "eval_world_dir": str(worlds_path), "world_ids": world_ids},
                callback=on_done,
                error_callback=on_error,
            )
            requests.append(request)
        worker_pool.close()
        worker_pool.join()
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record_data = {
        "wandb_run": wandb_run,
        "baseline": "IMPALA",
        "checkpoint_filename": checkpoint_filename,
        "data_key": data_key,
        "eval_world_dir": str(worlds_path),
        "start_time": start_time,
        "end_time": end_time,
        "all_results": all_results,
    }

    logger.info(f"Saving result to '{result_key}'")
    s3_client.save(result_key, json.dumps(record_data))


if __name__ == "__main__":

    wandb_run = "sourceress/abe__torchbeast/38z8lhh9"
    checkpoint_filename = "model_step_50003200.tar"
    worlds_path = Path("/tmp/avalon_worlds/b21aeff3-59c5-49a8-9c19-f3feceea1b8a")
    data_key = "avalon_worlds__b21aeff3-59c5-49a8-9c19-f3feceea1b8a.tar.gz"
    run_evaluation(
        worlds_path=worlds_path, data_key=data_key, wandb_run=wandb_run, checkpoint_filename=checkpoint_filename
    )
