# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import os
import random
import tarfile
import threading
import time
import timeit
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Set

import libtorchbeast
import nest
import numpy as np
import torch
import wandb
from gym import spaces
from torch import nn
from torch.nn import functional as F

from avalon.agent.godot.godot_gym import CURRICULUM_BASE_PATH
from avalon.agent.ppo.observation_model import ImpalaConvNet
from avalon.agent.torchbeast.avalon_helpers import IS_PROPRIOCEPTION_USED
from avalon.agent.torchbeast.avalon_helpers import PROPRIOCEPTION_ROW_COUNT
from avalon.agent.torchbeast.avalon_helpers import TORCHBEAST_ENV_LOGS_PATH
from avalon.agent.torchbeast.avalon_helpers import compute_policy_gradient_loss_from_dist
from avalon.agent.torchbeast.avalon_helpers import flatten_tensor
from avalon.agent.torchbeast.avalon_helpers import force_cudnn_initialization
from avalon.agent.torchbeast.avalon_helpers import get_avalon_test_scores
from avalon.agent.torchbeast.avalon_helpers import get_dict_action_space
from avalon.agent.torchbeast.avalon_helpers import get_goal_progress_files
from avalon.agent.torchbeast.avalon_helpers import get_num_test_worlds_from_flags
from avalon.agent.torchbeast.avalon_helpers import obs_to_frame
from avalon.agent.torchbeast.avalon_helpers import obs_to_proprioception
from avalon.agent.torchbeast.avalon_helpers import unflatten_tensor
from avalon.agent.torchbeast.avalon_helpers import vtrace_from_dist
from avalon.agent.torchbeast.core.environment import Environment
from avalon.agent.torchbeast.model import DictActionHead
from avalon.common.error_utils import setup_sentry
from avalon.common.log_utils import logger
from avalon.common.visual_utils import encode_video
from avalon.common.wandb_utils import get_latest_checkpoint_filename
from avalon.datagen.godot_env.interactive_godot_process import GODOT_ERROR_LOG_PATH

os.environ["OMP_NUM_THREADS"] = "1"

# exactly this separator is used by pytorch lightning...
BIG_SEPARATOR = "-" * 80
RESULT_TAG = "DATALOADER:0 TEST RESULTS"

# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--pipes_basename", default="unix:/tmp/polybeast",
                    help="Basename for the pipes for inter-process communication. "
                    "Has to be of the type unix:/some/path.")
parser.add_argument("--mode", default="train",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--project", default="abe__torchbeast",
                    help="Wandb project.")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="~/torchbeast",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=8, type=int, metavar="N",
                    help="Number of actors.")
parser.add_argument("--total_steps", default=50_000_000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=32, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=100, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_learner_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--num_inference_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", action="store_true",
                    help="Disable CUDA.")
parser.add_argument("--num_actions", default=6, type=int, metavar="A",
                    help="Number of actions.")
parser.add_argument("--no_lstm", action="store_true",
                    help="DO NOT use LSTM in agent model.")
parser.add_argument("--max_learner_queue_size", default=None, type=int, metavar="N",
                    help="Optional maximum learner queue size. Defaults to batch_size.")

# Loss settings.
parser.add_argument("--entropy_cost", default=5e-3, type=float,
                    help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=2, type=float,
                    help="Baseline cost/multiplier.")
parser.add_argument("--discounting", default=0.998, type=float,
                    help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=1.5e-4, type=float,
                    metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.95, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.005, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=200.0, type=float,
                    help="Global gradient norm clip.")

# Misc settings.
parser.add_argument("--write_profiler_trace", action="store_true",
                    help="Collect and write a profiler trace "
                    "for chrome://tracing/.")
parser.add_argument("--is_validating_during_training", action="store_true",
                    help="Run validation during training")
parser.add_argument("--logging_rate", default=15, type=float,
                    help="Logging rate in seconds.")
parser.add_argument("--checkpoint_rate", default=15, type=float,
                    help="Validation/checkpointing rate in minutes.")
parser.add_argument("--test_episodes_per_task", default=51, type=int,
                    help="Number of validation episodes.")
parser.add_argument("--save_test_videos", action="store_true",
                    help="Save videos of test rollouts")
parser.add_argument("--save_test_actions", action="store_true",
                    help="Save actions from test rollouts")
# yapf: enable


def _set_seed(seed) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages**2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


class Net(nn.Module):
    def __init__(
        self,
        num_actions=6,
        use_lstm=False,
        img_dim: int = 96,
        encoder_out_dim: int = 256,
        is_sampling_mode: bool = False,
    ) -> None:
        super(Net, self).__init__()
        # TODO: get real action space

        action_space = get_dict_action_space(num_actions)
        self._action_space = action_space
        self.action_head = DictActionHead(action_space)
        self.num_actions = self.action_head.num_inputs
        self.last_action_dim = spaces.flatdim(action_space)
        self.use_lstm = use_lstm
        self.is_sampling_mode = is_sampling_mode

        self.img_encoder = ImpalaConvNet(img_dim=img_dim)

        if IS_PROPRIOCEPTION_USED:
            self.prop_fc = nn.Linear(img_dim * PROPRIOCEPTION_ROW_COUNT, encoder_out_dim, bias=False)
            self.prop_fc.weight.data.fill_(0.0)
        else:
            self.prop_fc = None

        # FC output size + last reward.
        core_output_size = encoder_out_dim + 1

        if use_lstm:
            self.core = nn.LSTM(core_output_size, encoder_out_dim, num_layers=1)
            core_output_size = encoder_out_dim

        self.policy = nn.Linear(core_output_size, self.num_actions)
        self.baseline = nn.Linear(core_output_size, 1)

    def initial_state(self, batch_size=1):
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))

    def forward(self, inputs, core_state):
        x = obs_to_frame(inputs["frame"])
        T, B, *_ = x.shape
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        x = x.float() / 255.0
        x = self.img_encoder(x)

        if self.prop_fc is not None:
            x_prop = obs_to_proprioception(inputs["frame"])
            x_prop = torch.flatten(x_prop, 0, 1)  # Merge time and batch.
            x_prop = x_prop.float() / 255.0 - 0.5
            x = x + self.prop_fc(x_prop)

        clipped_reward = torch.clamp(inputs["reward"], -1, 1).view(T * B, 1)
        core_input = torch.cat([x, clipped_reward], dim=-1)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                # Reset core state to zero whenever an episode ended.
                # Make `done` broadcastable with (num_layers, B, hidden_size)
                # states:
                nd = nd.view(1, -1, 1)
                core_state = nest.map(nd.mul, core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)

        if self.is_sampling_mode:
            action = self.action_head(policy_logits).sample_mode()
        else:
            action = self.action_head(policy_logits).sample()

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        flat_action = self.flatten(action).view(T, B, self.last_action_dim)

        return (flat_action, policy_logits, baseline), core_state

    def flatten(self, action):
        return flatten_tensor(self._action_space, action)

    def unflatten(self, flat_action):
        return unflatten_tensor(self._action_space, flat_action)


def inference(flags, inference_batcher, model, lock=threading.Lock()) -> None:  # noqa: B008
    with torch.no_grad():
        for batch in inference_batcher:
            batched_env_outputs, agent_state = batch.get_inputs()
            frame, reward, done, *_ = batched_env_outputs
            frame = frame.to(flags.actor_device, non_blocking=True)
            reward = reward.to(flags.actor_device, non_blocking=True)
            done = done.to(flags.actor_device, non_blocking=True)
            agent_state = nest.map(lambda t: t.to(flags.actor_device, non_blocking=True), agent_state)
            with lock:
                outputs = model(dict(frame=frame, reward=reward, done=done), agent_state)
            outputs = nest.map(lambda t: t.cpu(), outputs)
            batch.set_outputs(outputs)


EnvOutput = collections.namedtuple("EnvOutput", "frame rewards done episode_step episode_return")
AgentOutput = collections.namedtuple("AgentOutput", "action policy_logits baseline")
Batch = collections.namedtuple("Batch", "env agent")


def learn(
    flags,
    learner_queue,
    model,
    actor_model,
    optimizer,
    scheduler,
    stats,
    lock=threading.Lock(),
) -> None:
    model.train()
    for tensors in learner_queue:
        if flags.mode != "train":
            continue

        tensors = nest.map(lambda t: t.to(flags.learner_device), tensors)

        batch, initial_agent_state = tensors
        env_outputs, actor_outputs = batch
        frame, reward, done, *_ = env_outputs

        lock.acquire()  # Only one thread learning at a time.
        learner_outputs, unused_state = model(dict(frame=frame, reward=reward, done=done), initial_agent_state)

        # Take final value function slice for bootstrapping.
        learner_outputs = AgentOutput._make(learner_outputs)
        bootstrap_value = learner_outputs.baseline[-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = nest.map(lambda t: t[1:], batch)
        learner_outputs = nest.map(lambda t: t[:-1], learner_outputs)

        # Turn into namedtuples again.
        env_outputs, actor_outputs = batch
        env_outputs = EnvOutput._make(env_outputs)
        actor_outputs = AgentOutput._make(actor_outputs)
        learner_outputs = AgentOutput._make(learner_outputs)

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(env_outputs.rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = env_outputs.rewards

        discounts = (~env_outputs.done).float() * flags.discounting

        behavior_policy_dist = model.action_head(actor_outputs.policy_logits)
        target_policy_dist = model.action_head(learner_outputs.policy_logits)
        action_dict = model.unflatten(actor_outputs.action)
        vtrace_returns = vtrace_from_dist(
            behavior_policy_dist=behavior_policy_dist,
            target_policy_dist=target_policy_dist,
            actions=action_dict,
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs.baseline,
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss_from_dist(
            target_policy_dist,
            action_dict,
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(vtrace_returns.vs - learner_outputs.baseline)
        entropy_loss = flags.entropy_cost * compute_entropy_loss(learner_outputs.policy_logits)

        total_loss = pg_loss + baseline_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())

        episode_returns = env_outputs.episode_return[env_outputs.done]
        stats["step"] = stats.get("step", 0) + flags.unroll_length * flags.batch_size
        stats["episode_returns"] = tuple(episode_returns.cpu().numpy())
        stats["mean_episode_return"] = torch.mean(episode_returns).item()
        stats["mean_episode_step"] = torch.mean(env_outputs.episode_step.float()).item()
        stats["total_loss"] = total_loss.item()
        stats["pg_loss"] = pg_loss.item()
        stats["baseline_loss"] = baseline_loss.item()
        stats["entropy_loss"] = entropy_loss.item()

        stats["learner_queue_size"] = learner_queue.size()

        if not len(episode_returns):
            # Hide the mean-of-empty-tuple NaN as it scares people.
            stats["mean_episode_return"] = None

        lock.release()


@torch.no_grad()
def run_test(env: Environment, model: Net, num_episodes: int, log_prefix="", is_video_logged=True, seed: int = 0):
    force_cudnn_initialization()
    _set_seed(seed)

    # DO ROLLOUTS
    observation = env.initial()
    core_state = model.initial_state(batch_size=1)
    core_state = nest.map(lambda t: t.to("cuda:0"), core_state)
    episodes = []

    def get_obs_to_log(observation):
        return {k: v for k, v in observation.items() if k != "frame" or is_video_logged}

    episode = [get_obs_to_log(observation)]
    while len(episodes) < num_episodes:
        (flat_action, policy_logits, baseline), core_state = model(observation, core_state)
        observation = env.step(flat_action)
        episode.append(get_obs_to_log(observation))
        if observation["done"].item():
            episodes.append(episode)
            episode = [get_obs_to_log(observation)]

    # GET ROLLOUT STATS
    results_by_task = defaultdict(lambda: defaultdict(list))
    log_data: Dict[str, Any] = {}
    keys = ["success"]
    seen_episodes: Set[int] = set()
    for episode in episodes:
        episode_id = episode[-1]["info"]["episode_id"]
        if episode_id in seen_episodes:
            # We overgenerate episodes, don't count them twice!
            continue
        else:
            seen_episodes.add(episode_id)

        task = episode[-1]["info"]["task"].lower()
        difficulty = episode[-1]["info"]["difficulty"]
        for field in keys:
            results_by_task[task][f"final_{field}"].append(episode[-1]["info"][field])
        results_by_task[task]["episode_length"].append(episode[-1]["episode_step"].item())
        results_by_task[task]["episode_reward"].append(episode[-1]["episode_return"].item())
        if is_video_logged:
            # final_frame is white for success and black for failure
            final_frame = (
                torch.zeros_like(obs_to_frame(episode[0]["frame"])[..., :3, :, :]) + episode[-1]["info"]["success"]
            )
            video_tensor = torch.cat(
                [obs_to_frame(step["frame"])[..., :3, :, :] / 255.0 for step in episode[:-1]] + [final_frame], dim=0
            ).squeeze(1)
            video_png_path = encode_video(video_tensor, normalize=False)
            log_data[f"{log_prefix}{task}/video_{episode_id}_diff_{difficulty:3g}"] = wandb.Image(video_png_path)

    # AGGREGATE ROLLOUT STATS
    for task, x in results_by_task.items():
        for field, y in x.items():
            log_data[f"{log_prefix}{task}/{field}"] = np.mean(y)
        log_data[f"{log_prefix}{task}/num_episodes"] = len(y)

    return log_data


def train(flags):
    # The queue the learner threads will get their data from.
    # Setting `minimum_batch_size == maximum_batch_size`
    # makes the batch size static.
    learner_queue = libtorchbeast.BatchingQueue(
        batch_dim=1,
        minimum_batch_size=flags.batch_size,
        maximum_batch_size=flags.batch_size,
        check_inputs=True,
        maximum_queue_size=flags.max_learner_queue_size,
    )

    # The "batcher", a queue for the inference call. Will yield
    # "batch" objects with `get_inputs` and `set_outputs` methods.
    # The batch size of the tensors will be dynamic.
    inference_batcher = libtorchbeast.DynamicBatcher(
        batch_dim=1,
        minimum_batch_size=1,
        maximum_batch_size=512,
        timeout_ms=500,
        check_outputs=True,
    )

    addresses = []
    connections_per_server = 1
    pipe_id = 0
    while len(addresses) < flags.num_actors:
        for _ in range(connections_per_server):
            addresses.append(f"{flags.pipes_basename}.{pipe_id}")
            if len(addresses) == flags.num_actors:
                break
        pipe_id += 1

    is_sampling_mode = flags.mode == "test"
    model = Net(num_actions=flags.num_actions, use_lstm=flags.use_lstm, is_sampling_mode=is_sampling_mode)
    model = model.to(device=flags.learner_device)

    actor_model = Net(num_actions=flags.num_actions, use_lstm=flags.use_lstm, is_sampling_mode=is_sampling_mode)
    actor_model.to(device=flags.actor_device)

    # The ActorPool that will run `flags.num_actors` many loops.
    actors = libtorchbeast.ActorPool(
        unroll_length=flags.unroll_length,
        learner_queue=learner_queue,
        inference_batcher=inference_batcher,
        env_server_addresses=addresses,
        initial_agent_state=actor_model.initial_state(),
    )

    def run():
        try:
            actors.run()
        except Exception as e:
            logger.error("Exception in actorpool thread!")
            traceback.print_exc()
            raise e

    actorpool_thread = threading.Thread(target=run, name="actorpool-thread", daemon=True)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * flags.unroll_length * flags.batch_size, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stats = {}

    # Load state from a checkpoint, if it exists.
    run_path = "/".join([wandb.run.entity, wandb.run.project, flags.xpid])
    load_checkpoint_filename = flags.load_checkpoint_filename or get_latest_checkpoint_filename(
        run_path, prefix="model_step_", suffix=".tar"
    )
    if load_checkpoint_filename is not None:
        checkpoint = wandb.restore(load_checkpoint_filename, run_path=run_path, replace=True)
        checkpoint_states = torch.load(checkpoint.name, map_location=flags.learner_device)

        model.load_state_dict(checkpoint_states["model_state_dict"])
        optimizer.load_state_dict(checkpoint_states["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_states["scheduler_state_dict"])
        stats = checkpoint_states["stats"]
        logger.info(f"Resuming {flags.xpid} from checkpoint {load_checkpoint_filename}, current stats:\n{stats}")
    else:
        assert flags.mode == "train", f"No checkpoint for run {run_path}, cannot test!"

    # Initialize actor model like learner model.
    actor_model.load_state_dict(model.state_dict())

    learner_threads = [
        threading.Thread(
            target=learn,
            name="learner-thread-%i" % i,
            args=(
                flags,
                learner_queue,
                model,
                actor_model,
                optimizer,
                scheduler,
                stats,
            ),
        )
        for i in range(flags.num_learner_threads)
    ]
    inference_threads = [
        threading.Thread(
            target=inference,
            name="inference-thread-%i" % i,
            args=(flags, inference_batcher, actor_model),
        )
        for i in range(flags.num_inference_threads)
    ]
    if flags.mode == "test":
        num_test_worlds = get_num_test_worlds_from_flags(flags)
    else:
        num_test_worlds = 0

    actorpool_thread.start()
    for t in learner_threads + inference_threads:
        t.start()

    def checkpoint(step) -> None:
        if flags.disable_checkpoint:
            return
        checkpointpath = Path(wandb.run.dir) / f"model_step_{step}.tar"
        logger.info(f"Saving checkpoint to {checkpointpath}")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "stats": stats,
                "flags": vars(flags),
            },
            checkpointpath,
        )
        wandb.save(str(checkpointpath), policy="now")
        wandb.save(str(CURRICULUM_BASE_PATH / "*"), policy="now", base_path=str(CURRICULUM_BASE_PATH.parent))

    test_data = {}
    try:
        last_validation_time = timeit.default_timer()
        last_sample_time = timeit.default_timer()
        while True:
            start_time = timeit.default_timer()
            start_step = stats.get("step", 0)
            if flags.mode == "train" and start_step >= flags.total_steps:
                break
            time.sleep(flags.logging_rate)
            end_step = stats.get("step", 0)

            if flags.mode == "train":
                log_data = stats.copy()
                if (
                    flags.checkpoint_rate > 0
                    and timeit.default_timer() - last_validation_time > 60 * flags.checkpoint_rate
                ):
                    # match up checkpoints with validation, run every checkpoint_rate
                    checkpoint(end_step)
                    last_validation_time = timeit.default_timer()
                logger.info(
                    f"Step {end_step} @ {(end_step - start_step) / (timeit.default_timer() - start_time):.1f} SPS. "
                    f"Inference batcher size: {inference_batcher.size()}. Learner queue size: {learner_queue.size()}."
                    f" Other stats: {stats}"
                )
                wandb.log(log_data)

                if end_step > start_step:
                    last_sample_time = timeit.default_timer()
                elif timeit.default_timer() - last_sample_time > 600:
                    logger.warning("No samples for 10 minutes! Exiting training!")
                    break
            else:
                num_goal_progress_files = len(get_goal_progress_files())
                logger.info(f"{num_goal_progress_files} of {num_test_worlds} test worlds completed")
                # aggressively trying to get these uploaded...
                error_logs = list(Path(GODOT_ERROR_LOG_PATH).glob("*.tar"))
                if len(error_logs) > 0:
                    logger.error(f"Godot errors detected! Uploading and quitting: {error_logs}")
                    for error_log in error_logs:
                        wandb.save(str(error_log), base_path=GODOT_ERROR_LOG_PATH, policy="now")
                    break
                if num_goal_progress_files >= num_test_worlds:
                    break
    except KeyboardInterrupt:
        pass  # Close properly.
    else:
        step = stats["step"]
        if flags.mode == "train":
            logger.info(f"Learning finished after {step} steps. Checkpointing.")
            test_data = stats
            checkpoint(step)
        else:
            logger.info(f"Testing finished! Collecting scores...")
            test_data = get_avalon_test_scores()
            test_data["step"] = step
            wandb.log(test_data)

            test_data["config/total_steps"] = flags.total_steps
            logger.info(BIG_SEPARATOR)
            logger.info(RESULT_TAG)
            logger.info(test_data)
            logger.info(BIG_SEPARATOR)
            if flags.save_test_videos or flags.save_test_actions:
                for log_dir in TORCHBEAST_ENV_LOGS_PATH.glob("*"):
                    if not log_dir.is_dir():
                        continue
                    tar_path = log_dir.with_suffix(".tar.gz")
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(log_dir, arcname=TORCHBEAST_ENV_LOGS_PATH)
                    wandb.save(str(tar_path), base_path=str(TORCHBEAST_ENV_LOGS_PATH), policy="now")

    # Done with learning. Stop all the ongoing work.
    logger.info("Closing queues and joining threads")
    inference_batcher.close()
    learner_queue.close()
    actorpool_thread.join()
    for t in learner_threads + inference_threads:
        t.join()
    logger.info("Learning cleaned up.")

    return test_data


def main(flags):
    flags.use_lstm = not flags.no_lstm
    if not flags.pipes_basename.startswith("unix:"):
        raise Exception("--pipes_basename has to be of the form unix:/some/path.")

    setup_sentry(percent_of_traces_to_capture=1.0)

    os.makedirs(GODOT_ERROR_LOG_PATH, exist_ok=True)
    wandb.save(
        f"{GODOT_ERROR_LOG_PATH}/*", base_path=GODOT_ERROR_LOG_PATH
    )  # Should watch for anything here and upload immediately?

    if not flags.disable_cuda and torch.cuda.is_available():
        logger.info("Using CUDA.")
        flags.learner_device = torch.device("cuda:0")
        actor_device_index = "1" if torch.cuda.device_count() > 1 else "0"
        flags.actor_device = torch.device(f"cuda:{actor_device_index}")
    else:
        logger.info("Not using CUDA.")
        flags.learner_device = torch.device("cpu")
        flags.actor_device = torch.device("cpu")

    if flags.max_learner_queue_size is None:
        flags.max_learner_queue_size = flags.batch_size

    if flags.write_profiler_trace:
        logger.info("Running with profiler.")
        with torch.autograd.profiler.profile() as prof:
            result = train(flags)
        filename = "chrome-%s.trace" % time.strftime("%Y%m%d-%H%M%S")
        logger.info("Writing profiler trace to '%s.gz'", filename)
        prof.export_chrome_trace(filename)
        os.system("gzip %s" % filename)
    else:
        result = train(flags)

    return result


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
