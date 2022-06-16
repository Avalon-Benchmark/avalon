import gzip
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Set

import numpy as np
import torch
import wandb

from agent.godot_gym import LEVEL_OUTPUT_PATH
from agent.ppo import wandb_lib
from agent.ppo.envs import build_env
from agent.ppo.params import PPOParams
from agent.ppo.ppo import PPO
from agent.storage import TrajectoryStorage
from agent.worker import GamePlayer
from common.log_utils import logger
from common.utils import TMP_DATA_DIR
from common.visual_utils import get_path_to_video_png
from datagen.godot_env import GODOT_ERROR_LOG_PATH

SUCCESS_RESULT_KEY = "overall_success_rate"

BIG_SEPARATOR = "-" * 80
RESULT_TAG = "DATALOADER:0 TEST RESULTS"


class GodotPPOParams(PPOParams):
    pass


def main(args: PPOParams):
    # The default method of fork is "not thread safe" and specifically deadlocks on mujoco env rendering.
    # Forkserver is safer but is slow to launch processes - like a few seconds each.
    ctx = mp.get_context(args.mp_method)

    device = torch.device(args.device)

    if not args.name:
        args.name = f"{args.env_name}"
    wandb.init(name=args.name, project=args.project, config=args.as_dict(), tags=[args.tag], monitor_gym=True)
    os.makedirs(GODOT_ERROR_LOG_PATH, exist_ok=True)
    wandb.save(f"{GODOT_ERROR_LOG_PATH}/*", base_path=GODOT_ERROR_LOG_PATH)
    wandb_lib.SCALAR_FREQ = args.log_freq_scalar
    wandb_lib.HIST_FREQ = args.log_freq_hist
    wandb_lib.MEDIA_FREQ = args.log_freq_media
    WATCH_FREQ = 50

    shutil.rmtree(LEVEL_OUTPUT_PATH, ignore_errors=True)

    dummy_env = build_env(args, mode="dummy")
    action_space = dummy_env.action_space
    observation_space = dummy_env.observation_space

    train_storage = TrajectoryStorage(args)
    env_fn = partial(build_env, args=args, mode="train")
    train_game_player = GamePlayer(
        args,
        env_fn,
        args.num_workers,
        args.multiprocessing,
        train_storage,
        ctx,
        observation_space,
        mode="fragment",
    )

    if not args.is_train_only:
        val_storage = TrajectoryStorage(args)
        env_fn = partial(build_env, args=args, mode="val")
        # TODO: for some reason this crashes if set to multiprocessing with 4 workers.
        # But not singleprocessing w 4 workers or multiprocessing w 1 worker.
        val_game_player = GamePlayer(
            args,
            env_fn,
            args.valtest_num_workers,
            args.valtest_multiprocessing,
            val_storage,
            ctx,
            observation_space,
            mode="episode",
        )

    model = PPO(args, observation_space, action_space)
    model = model.to(args.device)

    # The freq here is computed based on the number of forward() calls, which in this model is different than i
    wandb_lib.watch(model, freq=WATCH_FREQ * args.ppo_epochs * args.num_batches)

    # Main loop
    i = 0
    env_step = 0
    last_eval = None
    num_tasks = dummy_env.num_tasks
    while True:
        # Run rollouts
        start = time.time()
        train_game_player.run_rollout(args.num_steps, model, device, action_space)
        rollout_fps = (args.num_workers * args.num_steps) / (time.time() - start)
        wandb_lib.log_scalar("timings/rollout_fps", rollout_fps, i)
        env_step += args.num_workers * args.num_steps

        # Compute training rollout statistics
        if "godot" in args.env_name:
            successes = defaultdict(lambda: defaultdict(list))
            keys = ["success", "difficulty"]
            for fragment in train_storage.storage.values():
                for timestep in fragment:
                    if timestep["dones"]:
                        task = timestep["info"]["task"].lower()
                        for field in keys:
                            successes[task][f"final_{field}"].append(timestep["info"][field])
            # Data is a dict (task) of dicts (keys) of lists
            for task, x in successes.items():
                for field, y in x.items():
                    wandb_lib.log_histogram(f"train/{task}/{field}", y, i, hist_freq=10)
                wandb_lib.log_scalar(f"train/{task}/num_episodes", len(y), i)

        rollouts = train_storage.to_packed()
        final_observations = train_game_player.next_obs
        i = model.train_batch(rollouts, final_observations, i)

        wandb_lib.log_scalar("env_step", env_step, i)
        # Record timings in terms of env steps/sec
        wandb_lib.log_iteration_time(args.num_workers * args.num_steps / (args.num_batches * args.ppo_epochs), i)

        # NOTE: these are over the past 100 episodes
        wandb_lib.log_histogram("rollout/train/ep_len", train_game_player.episode_length, i)
        wandb_lib.log_histogram("rollout/train/ep_rew", train_game_player.episode_rewards, i)

        if env_step >= args.total_env_steps:
            break

        train_storage.reset()

        # Evaluation
        if not args.is_train_only:
            if last_eval is None or (env_step - last_eval) > args.val_freq:
                model_filename = Path(wandb.run.dir) / f"model_{env_step}.pt"
                torch.save(model.state_dict(), model_filename)
                wandb.save(str(model_filename), policy="now")

                val_storage.reset()
                while len(val_storage.storage) < args.val_episodes_per_task * num_tasks:
                    num_steps = 1000
                    val_game_player.run_rollout(num_steps, model, device, action_space)

                successes = defaultdict(lambda: defaultdict(list))
                val_log: Dict[str, Any] = {}
                keys = ["success"]
                seen_videos: Set[int] = set()
                for episode in val_storage.storage.values():
                    video_id = episode[-1]["info"]["video_id"]
                    if video_id in seen_videos:
                        # We overgenerate videos, don't count them twice!
                        continue
                    else:
                        seen_videos.add(video_id)

                    task = episode[-1]["info"]["task"].lower()
                    difficulty = episode[-1]["info"]["difficulty"]
                    for field in keys:
                        successes[task][f"final_{field}"].append(episode[-1]["info"][field])
                    successes[task]["episode_length"].append(len(episode))
                    # ep reward can be nonzero even if success is zero due to early-termination boostrapping
                    ep_reward = sum([step["rewards"] for step in episode])
                    successes[task]["episode_reward"].append(ep_reward)

                    # final_frame is white for success and black for failure
                    final_frame = (
                        torch.tensor(episode[0]["obs__rgbd"][:3]) * 0.0 + episode[-1]["info"]["success"] - 0.5
                    )
                    video_tensor = (
                        torch.stack([torch.tensor(step["obs__rgbd"][:3]) for step in episode] + [final_frame] * 5)
                        + 0.5
                    )

                    video_png_path = get_path_to_video_png(video_tensor, normalize=False)
                    val_log[f"val/{task}/video_{video_id}_diff_{difficulty:3g}"] = wandb.Image(video_png_path)

                # Data is a dict (task) of dicts (keys) of lists
                for task, x in successes.items():
                    for field, y in x.items():
                        # TODO: Make this hist if there's a decent number of samples
                        val_log[f"val/{task}/{field}"] = np.mean(y)
                    val_log[f"val/{task}/num_episodes"] = len(y)

                wandb.log(val_log, step=i)

                last_eval = env_step

    wandb.save(f"{GODOT_ERROR_LOG_PATH}/*", base_path=GODOT_ERROR_LOG_PATH, policy="now")
    train_game_player.shutdown()

    if not args.is_train_only:
        test_storage = TrajectoryStorage(args)
        env_fn = partial(build_env, args=args, mode="test")
        test_game_player = GamePlayer(
            args,
            env_fn,
            args.valtest_num_workers,
            args.valtest_multiprocessing,
            test_storage,
            ctx,
            observation_space,
            mode="episode",
        )
        test_storage.reset()
        while len(test_storage.storage) < args.test_episodes_per_task * num_tasks:
            num_steps = 1000
            test_game_player.run_rollout(num_steps, model, device, action_space)

        task_success = defaultdict(list)
        seen_videos: Set[int] = set()
        for episode in test_storage.storage.values():
            video_id = episode[-1]["info"]["video_id"]
            if video_id in seen_videos:
                # We overgenerate videos, don't count them twice!
                continue
            else:
                seen_videos.add(video_id)

            task = episode[-1]["info"]["task"].lower()
            success = episode[-1]["info"]["success"]
            task_success[task].append(success)

        test_log = {}
        all_values = []
        for task, values in task_success.items():
            test_log[f"{task}_success_rate"] = np.mean(values)
            all_values.extend(values)
        test_log[SUCCESS_RESULT_KEY] = np.mean(all_values)
        wandb.log({f"test/{k}": v for k, v in test_log.items()})

        logger.info(BIG_SEPARATOR)
        logger.info(RESULT_TAG)
        logger.info(test_log)
        logger.info(BIG_SEPARATOR)

        val_game_player.shutdown()
        test_game_player.shutdown()

    wandb.finish()

    return test_log


if __name__ == "__main__":
    # Necessary to fix some bug: https://github.com/wandb/client/issues/1994
    wandb.require("service")
    args = GodotPPOParams().parse_args()
    main(args)
