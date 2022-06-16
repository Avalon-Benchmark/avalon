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

import sentry_sdk
import torch

from agent.evaluation import EVAL_TEMP_PATH
from agent.evaluation import EvaluationGodotEnv
from agent.evaluation import get_wandb_result_key
from agent.evaluation import get_world_folders
from agent.evaluation import load_checkpoint_from_wandb_run
from agent.evaluation import world_id_from_world_folder_name
from agent.ppo.envs import wrap_godot_eval_env
from agent.ppo.main import GodotPPOParams
from agent.ppo.ppo import PPO
from agent.torchbeast.avalon_helpers import force_cudnn_initialization
from common.log_utils import logger
from contrib.s3_utils import SimpleS3Client


@torch.no_grad()
def do_eval_rollouts(model: PPO, env: EvaluationGodotEnv, device="cuda:0"):
    force_cudnn_initialization()
    # DO ROLLOUTS
    observation = env.reset()
    scores_by_world_index: Dict[int, float] = {}
    frame_count = 1
    while True:
        obs_tensor = {k: torch.tensor(v, device=device).unsqueeze(0) for k, v in observation.items()}
        step_values, dist = model(obs_tensor)
        step_actions = dist.sample()
        step_actions = {k: v.detach().cpu()[0] for k, v in step_actions.items()}
        observation, reward, is_done, info = env.step(step_actions)
        frame_count += 1
        world_index = info["world_index"]
        if world_index in scores_by_world_index:
            break
        if is_done:
            print(
                f"Episode {world_index}, {info['task']} ({info['difficulty']}) score: {info['score']:.3f} in {frame_count} frames"
            )
            scores_by_world_index[world_index] = info["score"]

            observation = env.reset()
            frame_count = 1
    return scores_by_world_index


def run_checkpoint_on_world_ids(
    checkpoint_path: str, eval_world_dir: str, world_ids: Tuple[int, ...], device: str = "cuda:0"
):
    env = wrap_godot_eval_env(EvaluationGodotEnv(eval_world_dir=eval_world_dir))
    env.set_eval_world_ids(world_ids)
    args = GodotPPOParams().parse_args("")
    model = PPO(args, env.observation_space, env.action_space).to(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    result = do_eval_rollouts(model, env, device)
    return result


def run_evaluation(worlds_path: Path, data_key: Optional[str], wandb_run: str, checkpoint_filename: str):

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

    logger.info(f"Downloading checkpoint {checkpoint_filename} from wandb run {wandb_run}")
    checkpoint_path = load_checkpoint_from_wandb_run(wandb_run, checkpoint_filename)

    num_processes = 5
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
        "baseline": "PPO",
        "checkpoint_filename": checkpoint_filename,
        "data_key": data_key,
        "eval_world_dir": str(worlds_path),
        "start_time": start_time,
        "end_time": end_time,
        "all_results": all_results,
    }

    result_key = get_wandb_result_key(wandb_run, checkpoint_filename, data_key)
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
