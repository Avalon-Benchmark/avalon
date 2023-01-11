import multiprocessing.synchronize
import signal
import time
import traceback
from multiprocessing.context import BaseContext
from typing import Any
from typing import Optional
from typing import Tuple

import attr
import gym
import torch
from loguru import logger
from torch import Tensor
from tree import map_structure

from avalon.agent.common.envs import build_env
from avalon.agent.common.get_algorithm_cls import get_algorithm_cls
from avalon.agent.common.params import EnvironmentParams
from avalon.agent.common.params import Params
from avalon.agent.common.storage import EpisodeStorage
from avalon.agent.common.storage import ModelStorage
from avalon.agent.common.storage import TrajectoryStorage
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import StepData
from avalon.agent.common.util import create_action_storage
from avalon.agent.common.util import create_observation_storage
from avalon.agent.common.util import get_checkpoint_file
from avalon.agent.common.util import masked_copy_structure
from avalon.agent.common.util import postprocess_uint8_to_float
from avalon.agent.common.util import seed_and_run_deterministically_if_enabled
from avalon.agent.common.util import setup_new_process
from avalon.agent.dreamer.params import OffPolicyParams
from avalon.common.error_utils import capture_exception
from avalon.common.error_utils import setup_sentry
from avalon.common.log_utils import configure_parent_logging


class AsyncRolloutManager:
    def __init__(  # type: ignore
        self,
        params: OffPolicyParams,
        obs_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        rollout_manager_id: int,
        env_step_counter,  # multiprocessing.Value; typing for this is broken
        multiprocessing_context: BaseContext,
        train_rollout_dir: str,
    ) -> None:
        seed_and_run_deterministically_if_enabled()

        self.params = params
        self.obs_space = obs_space
        self.action_space = action_space
        self.rollout_manager_id = rollout_manager_id
        self.multiprocessing_context = multiprocessing_context
        self.env_step_counter = env_step_counter
        self.shutdown_event = multiprocessing_context.Event()
        self.train_rollout_dir = train_rollout_dir
        self.wandb_queue = multiprocessing_context.Queue()

    def start(self) -> None:
        ps = self.multiprocessing_context.Process(
            target=self.safe_entrypoint, args=(self.shutdown_event, self.wandb_queue), daemon=False
        )
        ps.start()

    def shutdown(self) -> None:
        logger.info("set shutdown event")
        self.shutdown_event.set()

    def safe_entrypoint(self, shutdown_event: multiprocessing.synchronize.Event, wandb_queue) -> None:  # type: ignore[no-untyped-def]
        # We don't wrap this in a loop because it really shouldn't fail flakily.
        # This is just for debugging a fatal crash.
        try:
            configure_parent_logging()
            self.entrypoint(shutdown_event, wandb_queue)
        except Exception as e:
            logger.warning("exception in AsyncGamePlayer process")
            logger.warning(e)
            logger.warning(traceback.format_exc())
            # This basically just gets eaten
            raise
        finally:
            logger.info("closing queue")
            self.wandb_queue.close()
            logger.info("queue closed")

    def entrypoint(self, shutdown_event: multiprocessing.synchronize.Event, wandb_queue) -> None:  # type: ignore[no-untyped-def]
        setup_new_process()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        seed_and_run_deterministically_if_enabled()

        params = attr.evolve(
            self.params,
            env_params=attr.evolve(
                self.params.env_params,
                env_index=self.params.env_params.env_index + self.rollout_manager_id * self.params.num_workers,
            ),
        )
        # Spread among the inference gpus evenly
        rollout_device = torch.device(
            f"cuda:{self.params.inference_gpus[self.rollout_manager_id % len(self.params.inference_gpus)]}"
        )
        storage = EpisodeStorage(
            self.params,
            rollout_dir=self.train_rollout_dir,
            wandb_queue=wandb_queue,
            step_data_type=StepData,
            num_workers=params.num_workers,
        )
        algorithm_cls = get_algorithm_cls(self.params)
        model = algorithm_cls(params, self.obs_space, self.action_space)

        if self.params.resume_from:
            checkpoint_path = get_checkpoint_file(self.params.resume_from)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.params.train_device))
            logger.info("RESUMED MODEL FROM CHECKPOINT")

        # You cannot have initialized cuda in the parent process before forking for this to work.
        model = model.to(rollout_device)
        while True:
            if shutdown_event.is_set():
                break
            player = RolloutManager(
                params=self.params,
                num_workers=self.params.num_workers,
                is_multiprocessing=self.params.multiprocessing,
                storage=storage,
                obs_space=self.obs_space,
                model=model,
                rollout_device=rollout_device,
                multiprocessing_context=self.multiprocessing_context,
            )
            while True:
                if shutdown_event.is_set():
                    # TODO: we should really pass this through to the Manager - so it stops immediately,
                    # versus on the next iteration (which could be a bit).
                    logger.info("shutting down player")
                    player.shutdown()
                    logger.info("shutting down storage")
                    storage.shutdown()
                    logger.info("done with shutdown")
                    break
                try:
                    start = time.time()
                    ModelStorage.load_model(model, rollout_device)
                    # TODO: convert to a non-fixed step count
                    player.run_rollout(
                        num_steps=self.params.num_steps,
                        exploration_mode="explore",
                    )
                    rollout_fps = (self.params.num_workers * self.params.num_steps) / (time.time() - start)
                    logger.info(
                        f"player {self.rollout_manager_id}: env steps: {self.env_step_counter.value}, fps:{rollout_fps}"
                    )
                    self.env_step_counter.value += self.params.num_workers * self.params.num_steps
                    wandb_queue.put(("scalar", "env_step", self.env_step_counter.value))
                except Exception as e:
                    if self.params.debug:
                        raise
                    logger.warning("caught exception in async worker, will try to reset.")
                    logger.warning(e)
                    logger.warning(traceback.format_exc())
                    # capture_exception(e)
                    try:
                        player.shutdown()
                        storage.shutdown()
                    except Exception as e2:
                        logger.warning(e2)
                        logger.warning(traceback.format_exc())
                    break


class RolloutManager:
    """A manager class for running multiple game-playing processes."""

    def __init__(
        self,
        params: Params,
        num_workers: int,
        is_multiprocessing: bool,
        storage: TrajectoryStorage,
        obs_space: gym.spaces.Dict,
        model: Algorithm,
        rollout_device: torch.device,
        multiprocessing_context: BaseContext,
    ) -> None:
        self.multiprocessing = is_multiprocessing
        self.num_workers = num_workers
        self.params = params
        self.model = model
        self.rollout_device = rollout_device
        self.multiprocessing_context = multiprocessing_context

        params = attr.evolve(params, env_params=attr.evolve(params.env_params, env_count=num_workers))

        self.step_storage = self.setup_step_storage()

        # Start game-playing processes
        self.workers = [
            EnvironmentContainer(i, params, is_multiprocessing, multiprocessing_context, self.step_storage)
            for i in range(num_workers)
        ]

        self.storage = storage
        self.observation_space = obs_space

    def setup_step_storage(self):
        # Note the use of Tensor.share_memory_() for the components that are shared with the workers

        # the ordering here is (reward, done, obs (next_obs)), and then from next_obs we compute (action, value, etc)
        # so depending on which substep you're on, the ordering of the two groups will differ.
        # but key is that the observation is the one following the reward/done.
        return {
            "terminal_obs": create_observation_storage(
                self.params.observation_space, batch_shape=(self.num_workers,), use_shared_memory=True
            ),
            "obs_for_inference": create_observation_storage(
                self.params.observation_space, batch_shape=(self.num_workers,), use_shared_memory=True
            ),
            "obs_to_store": create_observation_storage(
                self.params.observation_space, batch_shape=(self.num_workers,), use_shared_memory=True
            ),
            "action": create_action_storage(
                self.params.action_space, batch_shape=(self.num_workers,), use_shared_memory=True
            ),
            "reward": torch.zeros((self.num_workers,), dtype=torch.float32).share_memory_(),
            "done": torch.zeros((self.num_workers,), dtype=torch.bool).share_memory_(),
            "is_terminal": torch.zeros((self.num_workers,), dtype=torch.bool).share_memory_(),
            "ready_for_new_step": torch.ones((self.num_workers,), dtype=torch.bool).share_memory_(),
            "close": torch.zeros((self.num_workers,), dtype=torch.bool).share_memory_(),
            "info": {
                k: torch.zeros((self.num_workers,), dtype=torch.float32).share_memory_()
                for k in self.params.env_params.info_fields
            },
        }

    def run_rollout(
        self,
        num_steps: Optional[int] = None,
        num_episodes: Optional[int] = None,
        exploration_mode: str = "explore",
    ) -> None:
        """Set num_steps and/or num_episodes to run only that many steps/episodes,
        otherwise if left as None there will be no limit. Meets both criteria if both are set.
        """
        self.model.eval()
        fragment_timestep = torch.zeros((self.num_workers,), dtype=torch.int32)

        # Note that we don't reset step_storage between rollouts.
        # This is intentional - "obs_for_inference" is used for generating the first action,
        # and dones need to be carried over too for algorithms that use them in inference.
        # And the rest could be zeroed but isn't for performance reasons (possibly should be for aid in debugging tho).

        for i, worker in enumerate(self.workers):
            worker.remaining_steps = num_steps if num_steps is not None else 0
            worker.remaining_episodes = num_episodes if num_episodes is not None else 0
            worker.ready_for_new_step = True
            worker.completed_rollout = False

        while True:
            if self.num_active_workers == 0:
                break

            # run the model
            inference_mask = self.ready_for_new_step
            inference_timesteps = fragment_timestep
            step_actions, algorithm_extra_info = self.run_inference(exploration_mode)
            masked_copy_structure(self.step_storage["action"], step_actions, inference_mask)
            self.request_step()

            # send the inference data to storage. doing it here hides the time behind the step time.
            self.storage.partial_batch_update(
                "action", step_actions, mask=inference_mask, timesteps=inference_timesteps
            )
            for k, v in algorithm_extra_info.items():
                self.storage.partial_batch_update(k, v, mask=inference_mask, timesteps=inference_timesteps)

            step_time = time.time()
            threshold_time = None
            got_data = torch.zeros((self.num_workers,), dtype=torch.bool)
            while True:
                num_active_workers = self.num_active_workers
                ready_for_new_step = self.ready_for_new_step
                if num_active_workers == 0:
                    break
                if self.params.allow_partial_batches:
                    needed_percent = max(0.5, min(0.6, 1 - (1 / num_active_workers)))
                else:
                    needed_percent = 1
                if ready_for_new_step.sum() >= num_active_workers:
                    break
                if ready_for_new_step.sum() / num_active_workers + 1e-5 >= needed_percent:
                    if threshold_time is None:
                        threshold_time = time.time()
                    if (threshold_time - step_time) * 1.25 < (time.time() - step_time):
                        break

                # Receive step data from workers
                for i, worker in enumerate(self.workers):
                    if worker.completed_rollout or worker.ready_for_new_step or not worker.is_result_ready():
                        continue

                    # once we run this, the worker's data is ready to access
                    self.get_worker_data(worker, i)
                    got_data[i] = True

            # TODO: move this to when we're waiting on the stepping to hide the time
            for k in ["done", "reward", "is_terminal", "info"]:
                self.storage.partial_batch_update(k, self.step_storage[k], mask=got_data, timesteps=fragment_timestep)
            self.storage.partial_batch_update(
                "observation", self.step_storage["obs_to_store"], mask=got_data, timesteps=fragment_timestep
            )
            self.storage.mark_step_finished(got_data)
            fragment_timestep[got_data] += 1

        # We store the final observation too, so that training can use it for value bootstrapping
        # In the case of a reset, this will be from the next episode, and GAE should stop bootstrapping
        # in that case. It would be improper to give the terminal obs, since that is never otherwise provided
        # in on-policy training. And the normal `obs_to_store` is still from the previous step.
        # This will be ignored (from the flag is_extra_observation) in storage methods that don't use it.
        self.storage.partial_batch_update(
            "observation",
            self.step_storage["obs_for_inference"],
            mask=torch.ones((self.num_workers,), dtype=torch.bool),
            timesteps=torch.ones((self.num_workers,), dtype=torch.int32) * -1,
            is_extra_observation=True,
        )

    def request_step(self) -> None:
        for i, worker in enumerate(self.workers):
            if worker.ready_for_new_step and not worker.completed_rollout:
                worker.send_step()

    def run_inference(self, exploration_mode: str) -> Tuple[ActionBatch, dict]:
        with torch.no_grad():
            torch_obs = {
                k: v.to(device=self.rollout_device, non_blocking=True)
                for k, v in self.step_storage["obs_for_inference"].items()
            }
            torch_dones = self.step_storage["done"].to(device=self.rollout_device, non_blocking=True)
            torch_obs = postprocess_uint8_to_float(torch_obs, center=self.params.center_observations)
            # we pass ready_for_new_step to allow the model to not update state for non-ready workers.
            # But we expect to get the num_workers as the returned batch size (some are ignored).
            step_actions, to_store = self.model.rollout_step(
                torch_obs, torch_dones, self.ready_for_new_step, exploration_mode=exploration_mode
            )
        return step_actions, to_store

    def get_worker_data(
        self,
        worker: "EnvironmentContainer",
        i: int,
    ) -> None:
        done = worker.get_step_data()
        truncated = done and not self.step_storage["is_terminal"][i]
        if self.params.time_limit_bootstrapping and truncated:
            terminal_observation = map_structure(lambda x: x[i], self.step_storage["terminal_obs"])
            additional_reward = self.time_limit_bootstrapping(terminal_observation)
            self.step_storage["reward"][i] = self.step_storage["reward"][i] + additional_reward

    @property
    def ready_for_new_step(self) -> Tensor:
        return torch.tensor([worker.ready_for_new_step for worker in self.workers], dtype=torch.bool)

    @property
    def num_active_workers(self) -> int:
        return sum([worker.completed_rollout is False for worker in self.workers])

    def time_limit_bootstrapping(self, terminal_obs: dict[str, Tensor]) -> float:
        # NOTE: this won't work (as written) for recurrent algorithms!!
        # value bootstrapping for envs with artificial time limits imposed by a gym TimeLimit wrapper
        # see https://arxiv.org/pdf/1712.00378.pdf for context
        # this does get wrapper obs transforms applied
        terminal_obs_torch = {k: v.to(self.rollout_device).unsqueeze(0) for k, v in terminal_obs.items()}
        terminal_obs_torch = postprocess_uint8_to_float(terminal_obs_torch, center=self.params.center_observations)
        with torch.no_grad():
            terminal_value, _ = self.model(terminal_obs_torch)
        return self.params.discount * terminal_value[0].detach().cpu().item()  # type: ignore

    def shutdown(self) -> None:
        [worker.close() for worker in self.workers]


class EnvironmentContainer:
    """Abstracts away the details of interacting with a worker."""

    def __init__(
        self,
        i: int,
        params: Params,
        is_multiprocessing: bool,
        multiprocessing_context: BaseContext,
        step_storage: dict[str, Any],
    ) -> None:
        self.params = params
        self.i = i
        self.is_multiprocessing = is_multiprocessing
        env_params = attr.evolve(params.env_params, env_index=params.env_params.env_index + i)
        if is_multiprocessing:
            parent_conn, child_conn = multiprocessing_context.Pipe()
            worker = EnvironmentContainerProcess(i, child_conn, env_params, step_storage, params.obs_first)
            # Can't have daemon=True because the godot env spawns subprocesses
            self.ps = multiprocessing_context.Process(target=worker.run, args=(), daemon=False)
            # self.ps = threading.Thread(target=worker.run, args=())
            self.ps.start()
            self.parent_conn = parent_conn
        else:
            self.worker = EnvironmentContainerProcess(i, None, env_params, step_storage, params.obs_first)

        self.ready_for_new_step: bool = True
        self.completed_rollout: bool = False
        # maybe it would be more intuitive to count up instead of down here?
        self.remaining_steps: int = 0
        self.remaining_episodes: int = 0
        self.step_storage = step_storage

        assert params.observation_space is not None

    def get_step_data(self):
        # The side-effect of this function returning is that we are guaranteed the obs/reward/etc memory
        # arrays have been written by the worker.
        done = self.step_storage["done"][self.i]

        self.ready_for_new_step = True
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if done:
            self.remaining_episodes = max(0, self.remaining_episodes - 1)
        if self.remaining_steps == 0 and self.remaining_episodes == 0:
            self.ready_for_new_step = False
            self.completed_rollout = True

        return done

    def send_step(self) -> None:
        if self.is_multiprocessing:
            self.step_storage["ready_for_new_step"][self.i] = False
        else:
            self.worker.step()
        self.ready_for_new_step = False

    def is_result_ready(self) -> bool:
        if self.is_multiprocessing:
            return self.step_storage["ready_for_new_step"][self.i].item() is True
        else:
            return True

    def close(self) -> None:
        if self.is_multiprocessing:
            try:
                self.step_storage["close"][self.i] = True
            except BrokenPipeError:
                logger.warning("tried to close EnvironmentContainerProcess but pipe was already broken")
            except EOFError:
                logger.warning("tried to close EnvironmentContainerProcess but got EOF error.")
        else:
            self.worker.shutdown()


class EnvironmentContainerProcess:
    """A worker for running an environment, intended to be run on a separate
    process."""

    def __init__(  # type: ignore
        self,
        index: int,
        pipe,  # multiprocessing.Pipe (impossible to type afaik)
        env_params: EnvironmentParams,
        step_storage: dict[str, Any],
        obs_first: bool,
    ) -> None:
        self.worker_index = index
        self.pipe = pipe
        self.episode_steps: int = 0
        self.episode_rewards: float = 0

        self.env_params = env_params
        self.env: Optional[gym.Env] = None

        self.step_storage = step_storage
        self.obs_first = obs_first

    def lazy_init_env(self) -> None:
        if self.env is None:
            setup_sentry()
            self.env = build_env(env_params=self.env_params)
            self.next_obs = self.env.reset()
            self.next_obs = {k: v * 0 for k, v in self.next_obs.items()}

    def shutdown(self) -> None:
        if self.env:
            try:
                self.env.close()
            except TimeoutError as e:
                capture_exception(e)
                logger.warning(f"env did not close cleanly due to a TimeoutError: {e}")
                logger.warning(traceback.format_exc())
            except Exception as e:
                capture_exception(e)
                logger.warning(f"worker failed to shutdown with error {e}")
                logger.warning(traceback.format_exc())

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        configure_parent_logging()
        setup_new_process()
        try:
            while True:
                if self.step_storage["ready_for_new_step"][self.worker_index] == 0:
                    self.step()
                elif self.step_storage["close"][self.worker_index]:
                    break
                else:
                    time.sleep(2e-4)  # 200us
        except KeyboardInterrupt:
            pass
        except Exception as e:
            capture_exception(e)
            raise
        finally:
            self.shutdown()

    def step(self):
        """Perform a single step of the environment."""
        self.lazy_init_env()
        assert self.env is not None
        action = map_structure(lambda x: x[self.worker_index], self.step_storage["action"])
        next_obs, reward, done, info = self.env.step(action)  # type: ignore

        self.episode_rewards += reward

        if done:
            terminal_obs = next_obs
            # We need to store this for value bootstrapping, even if it's not used for inference or training
            for k, v in next_obs.items():
                self.step_storage["terminal_obs"][k][self.worker_index] = v

            next_obs = self.env.reset()
            self.episode_steps = 0
            self.episode_rewards = 0

        self.episode_steps += self.env_params.action_repeat

        is_terminal = done and not info.get("TimeLimit.truncated", False)
        assert isinstance(reward, (int, float))

        if self.obs_first:
            # Since `next_obs` comes after the reward/done, we need to use the obs from last step
            obs_to_store = self.next_obs
            # if the ep was done, next_obs is from the reset() call,
            # which will be the right thing to store on the next step.
            # Note: in this case, we never store (or use at all) the terminal observation. This is correct.
            self.next_obs = next_obs
        else:
            if done:
                # We need to handle this case differently. next_obs is actually from the next ep (from reset());
                # it wouldn't be good to store that as part of this one.
                # Instead we store the "terminal observation" which is the one received along with the done signal.
                # We handle both the time_limit and no_time_limit case the same.
                # NOTE: Interestingly, this means the agent never sees the first observation of an episode
                # (ie the one that comes from reset()) in training (but it does in inference).
                obs_to_store = terminal_obs  # type: ignore
            else:
                obs_to_store = next_obs

        # inference always uses next_obs; after a done, this is the obs received from reset().
        for k, v in next_obs.items():
            self.step_storage["obs_for_inference"][k][self.worker_index] = v
        # but we store the observation we selected above
        for k, v in obs_to_store.items():
            self.step_storage["obs_to_store"][k][self.worker_index] = v

        self.step_storage["reward"][self.worker_index] = reward
        self.step_storage["done"][self.worker_index] = done
        self.step_storage["is_terminal"][self.worker_index] = is_terminal

        self.step_storage["ready_for_new_step"][self.worker_index] = True

        for k in self.env_params.info_fields:
            self.step_storage["info"][k][self.worker_index] = info.get(k, 0)

        assert "terminal_observation" not in info  # we need to keep the infos small to keep this fast.
        return done, info
