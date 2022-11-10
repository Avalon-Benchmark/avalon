import multiprocessing.synchronize
import signal
import time
import traceback
import uuid
import warnings
from multiprocessing.context import BaseContext
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import attr
import gym
import numpy as np
import sentry_sdk
import torch
from loguru import logger
from torch import Tensor
from tree import map_structure

from avalon.agent.common.envs import build_env
from avalon.agent.common.get_algorithm_cls import get_algorithm_cls
from avalon.agent.common.params import EnvironmentParams
from avalon.agent.common.params import Params
from avalon.agent.common.storage import DiskStorage
from avalon.agent.common.storage import ModelStorage
from avalon.agent.common.storage import StorageMode
from avalon.agent.common.storage import TrajectoryStorage
from avalon.agent.common.types import Action
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import AlgorithmInferenceExtraInfoBatch
from avalon.agent.common.types import Info
from avalon.agent.common.types import Observation
from avalon.agent.common.types import StepData
from avalon.agent.common.util import get_checkpoint_file
from avalon.agent.common.util import postprocess_uint8_to_float
from avalon.agent.common.util import seed_and_run_deterministically_if_enabled
from avalon.agent.dreamer.params import OffPolicyParams
from avalon.common.error_utils import capture_exception

numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}


class AsyncRolloutManager:
    def __init__(  # type: ignore
        self,
        params: OffPolicyParams,
        obs_space: gym.spaces.Dict,
        action_space: gym.spaces.Dict,
        rollout_manager_id: int,
        env_step_counter,  # multiprocessing.Value, typing for this is broken
        multiprocessing_context: BaseContext,
        train_rollout_dir: str,
    ):
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
        self.shutdown_event.set()

    def safe_entrypoint(self, shutdown_event: multiprocessing.synchronize.Event, wandb_queue) -> None:  # type: ignore[no-untyped-def]
        # We don't wrap this in a loop because it really shouldn't fail flakily.
        # This is just for debugging a fatal crash.
        try:
            self.entrypoint(shutdown_event, wandb_queue)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.warning("exception in AsyncGamePlayer process")
            logger.warning(e)
            logger.warning(traceback.format_exc())
            # This basically just gets eaten
            raise

    def entrypoint(self, shutdown_event: multiprocessing.synchronize.Event, wandb_queue) -> None:  # type: ignore[no-untyped-def]
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.set_num_threads(1)
        signal.signal(signal.SIGINT, signal.default_int_handler)
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
        storage = DiskStorage(self.params, rollout_dir=self.train_rollout_dir, wandb_queue=wandb_queue)
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
                storage_mode=StorageMode.EPISODE,
                model=model,
                rollout_device=rollout_device,
                multiprocessing_context=self.multiprocessing_context,
            )
            while True:
                if shutdown_event.is_set():
                    player.shutdown()
                    storage.shutdown()
                    break
                try:
                    start = time.time()
                    ModelStorage.load_model(model, rollout_device)
                    # TODO: convert to a non-fixed step count
                    player.run_rollout(
                        num_steps=self.params.rollout_steps,
                        exploration_mode="explore",
                    )
                    rollout_fps = (self.params.num_workers * self.params.rollout_steps) / (time.time() - start)
                    logger.info(
                        f"player {self.rollout_manager_id}: env steps: {self.env_step_counter.value}, fps:{rollout_fps}"
                    )
                    self.env_step_counter.value += self.params.num_workers * self.params.rollout_steps
                    wandb_queue.put(("scalar", "env_step", self.env_step_counter.value))
                except KeyboardInterrupt:
                    player.shutdown()
                    storage.shutdown()
                    raise
                except Exception as e:
                    if self.params.debug:
                        raise
                    logger.warning("caught exception in async worker, will try to reset.")
                    logger.warning(e)
                    logger.warning(traceback.format_exc())
                    # capture_exception(e)
                    try:
                        player.shutdown()
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
        storage_mode: StorageMode,
        model: Algorithm,
        rollout_device: torch.device,
        multiprocessing_context: BaseContext,
    ):
        self.multiprocessing = is_multiprocessing
        self.num_workers = num_workers
        self.storage_mode = storage_mode
        self.params = params
        self.model = model
        self.rollout_device = rollout_device
        self.multiprocessing_context = multiprocessing_context

        params = attr.evolve(params, env_params=attr.evolve(params.env_params, env_count=num_workers))

        # Start game-playing processes
        self.workers = [
            EnvironmentContainer(i, storage_mode, params, is_multiprocessing, multiprocessing_context)
            for i in range(num_workers)
        ]

        self.storage = storage

        # the observation returned in each env after taking the previous action. used for inference.
        self.next_obs = {
            k: torch.zeros(size=(num_workers, *v.shape), dtype=numpy_to_torch_dtype_dict[v.dtype.type])
            for k, v in obs_space.spaces.items()
        }
        self.dones = [False for _ in range(self.num_workers)]
        self.observation_space = obs_space

    def run_rollout(
        self,
        num_steps: Optional[int] = None,
        num_episodes: Optional[int] = None,
        exploration_mode: str = "explore",
    ):
        """Set num_steps and/or num_episodes to run only that many steps/episodes,
        otherwise if left as None there will be no limit. Meets both criteria if both are set.
        """
        # TODO: make an off-policy optimized version of this method.
        self.model.eval()

        for worker in self.workers:
            worker.remaining_steps = num_steps if num_steps is not None else 0
            worker.remaining_episodes = num_episodes if num_episodes is not None else 0
            worker.ready_for_new_step = True
            worker.completed_rollout = False

        while True:
            if self.num_active_workers == 0:
                break
            step_data_batch: Dict[str, StepData] = {}

            # run the model
            step_actions, to_store = self.run_inference(exploration_mode)

            self.send_action_and_request_step(step_actions)

            step_time = time.time()
            threshold_time = None
            while True:
                if self.num_active_workers == 0:
                    break
                needed_percent = max(0.5, min(0.6, 1 - (1 / self.num_active_workers)))
                if sum(self.ready_for_new_step) >= self.num_active_workers:
                    break
                if sum(self.ready_for_new_step) / self.num_active_workers + 1e-5 >= needed_percent:
                    if threshold_time is None:
                        threshold_time = time.time()
                    if (threshold_time - step_time) * 1.25 < (time.time() - step_time):
                        break

                # Receive step data from workers
                for i, worker in enumerate(self.workers):
                    if worker.completed_rollout or worker.ready_for_new_step or not worker.is_result_ready():
                        continue

                    step_data, episode_id = self.worker_update(worker, step_actions, to_store, i)
                    step_data_batch[episode_id] = step_data

            assert self.num_active_workers == 0 or sum(self.ready_for_new_step) / self.num_active_workers >= 0.49
            self.storage.add_timestep_samples(step_data_batch)

    def send_action_and_request_step(self, step_actions: ActionBatch) -> None:
        for i, worker in enumerate(self.workers):
            if worker.ready_for_new_step and not worker.completed_rollout:
                worker.send_step(map_structure(lambda x: x[i], step_actions))
                worker.ready_for_new_step = False

    def run_inference(self, exploration_mode: str) -> Tuple[ActionBatch, AlgorithmInferenceExtraInfoBatch]:
        with torch.no_grad():
            torch_obs = {k: v.to(device=self.rollout_device) for k, v in self.next_obs.items()}
            torch_dones = torch.tensor(self.dones, dtype=torch.bool, device=self.rollout_device)
            torch_obs = postprocess_uint8_to_float(torch_obs)
            # we pass ready_for_new_step to allow the model to not update state for non-ready workers.
            # But we expect to get the num_workers as the returned batch size (some are ignored).
            step_actions, to_store = self.model.rollout_step(
                torch_obs, torch_dones, self.ready_for_new_step, exploration_mode=exploration_mode
            )
        return step_actions, to_store

    def worker_update(
        self,
        worker: "EnvironmentContainer",
        step_actions: ActionBatch,
        to_store: AlgorithmInferenceExtraInfoBatch,
        i: int,
    ) -> Tuple[StepData, str]:
        step_obs, done, step_data, episode_id = worker.get_step_data(
            actions=map_structure(lambda x: x[i], step_actions),
        )
        step_data = self.model.build_algorithm_step_data(step_data, extra_info=map_structure(lambda x: x[i], to_store))

        if done and self.params.time_limit_bootstrapping and step_data.info.get("TimeLimit.truncated", False) is True:
            step_data = attr.evolve(step_data, reward=step_data.reward + self.time_limit_bootstrapping(step_data.info))

        # Store the new observation+done for inference
        for k, v in step_obs.items():
            self.next_obs[k][i] = v
        self.dones[i] = done
        return step_data, episode_id

    @property
    def ready_for_new_step(self) -> list[bool]:
        return [worker.ready_for_new_step for worker in self.workers]

    @property
    def num_active_workers(self) -> int:
        return sum([worker.completed_rollout is False for worker in self.workers])

    def time_limit_bootstrapping(self, info: Info) -> float:
        # value bootstrapping for envs with artificial time limits imposed by a gym TimeLimit wrapper
        # see https://arxiv.org/pdf/1712.00378.pdf for context
        # this does get wrapper obs transforms applied
        terminal_obs = info["terminal_observation"]
        terminal_obs_torch = {
            k: torch.from_numpy(v).to(self.rollout_device).unsqueeze(0) for k, v in terminal_obs.items()
        }
        terminal_obs_torch = postprocess_uint8_to_float(terminal_obs_torch)
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
        storage_mode: StorageMode,
        params: Params,
        is_multiprocessing: bool,
        multiprocessing_context: BaseContext,
    ):
        self.params = params
        self.i = i
        self.is_multiprocessing = is_multiprocessing
        if is_multiprocessing:
            parent_conn, child_conn = multiprocessing_context.Pipe()
            env_params = attr.evolve(params.env_params, env_index=params.env_params.env_index + i)
            worker = EnvironmentContainerProcess(i, child_conn, storage_mode, env_params)
            # Can't have daemon=True because the godot env spawns subprocesses
            self.ps = multiprocessing_context.Process(target=worker.run, args=(), daemon=False)
            self.ps.start()
            self.parent_conn = parent_conn
        else:
            self.current_action: Optional[Action] = None
            self.worker = EnvironmentContainerProcess(i, None, storage_mode, params.env_params)

        self.ready_for_new_step: bool = True
        self.waiting_on_result: bool = False  # just for asserting correct order of operations
        self.completed_rollout: bool = False
        # maybe it would be more intuitive to count up instead of down here?
        self.remaining_steps: int = 0
        self.remaining_episodes: int = 0

        assert params.observation_space is not None
        self.next_obs: Dict[str, Tensor] = {
            k: torch.zeros(size=(*v.shape,), dtype=numpy_to_torch_dtype_dict[v.dtype.type])
            for k, v in params.observation_space.spaces.items()
        }

    def get_step_data(self, actions: Action) -> Tuple[Observation, bool, StepData, str]:
        result = self.get_result()
        received_obs, reward, done, info, episode_id = result
        received_obs = {k: torch.from_numpy(v) for k, v in received_obs.items()}

        # is_terminal indicates if a true environment termination happened (not a time limit)
        is_terminal = done and not info.get("TimeLimit.truncated", False)
        assert isinstance(reward, (int, float))

        stored_obs: Dict[str, Tensor] = {}
        for k, v in received_obs.items():
            if self.params.obs_first:
                # Since `received_obs` comes after the reward/done, we need to use the obs from last step
                stored_obs[k] = self.next_obs[k]
                self.next_obs[k] = v
            else:
                if done:
                    # We need to handle this case differently. received_obs is actually from the next ep;
                    # it wouldn't be good to store that as part of this one.
                    # Instead we store the "terminal observation" which is the one received along with the done signal.
                    # We handle both the time_limit and no_time_limit case the same.
                    stored_obs[k] = torch.from_numpy(info["terminal_observation"][k])
                else:
                    # Not done
                    stored_obs[k] = v

        self.ready_for_new_step = True
        self.remaining_steps = max(0, self.remaining_steps - 1)
        if done:
            self.remaining_episodes = max(0, self.remaining_episodes - 1)
        if self.remaining_steps == 0 and self.remaining_episodes == 0:
            self.ready_for_new_step = False
            self.completed_rollout = True

        step_data = StepData(
            observation=stored_obs,
            reward=reward,
            done=done,
            is_terminal=is_terminal,
            info=info,
            action=actions,
        )
        # We return the unmodified `received_obs` here to be used for the next inference pass.
        # At a reset, this will be the new observation from the next episode, which is indeed what we want for inference.
        return received_obs, done, step_data, episode_id

    def send_step(self, action: Action):
        assert self.waiting_on_result is False
        action = {k: v.numpy() for k, v in action.items()}
        if self.is_multiprocessing:
            self.parent_conn.send(("step", action))
            self.waiting_on_result = True
        else:
            assert self.current_action is None
            self.waiting_on_result = True
            self.current_action = action

    def is_result_ready(self):
        if self.is_multiprocessing:
            return self.parent_conn.poll()
        else:
            return True

    def get_result(self):
        assert self.waiting_on_result is True
        if self.is_multiprocessing:
            result = self.parent_conn.recv()
        else:
            result = self.worker.step(self.current_action)
            self.current_action = None
        self.waiting_on_result = False
        return result

    def close(self):
        if self.is_multiprocessing:
            try:
                # need to receive the result to unblock the worker to receive the close signal
                if self.is_result_ready():
                    self.get_result()
                self.parent_conn.send(("close", None))
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
        mode: StorageMode,
        env_params: EnvironmentParams,
    ):
        self.mode = mode
        self.index = index
        self.pipe = pipe
        self.episode_steps: int = 0
        self.episode_rewards: float = 0

        self.index = index
        self.env_params = env_params
        self.env: Optional[gym.Env] = None

        self.episode_id = self.new_episode_id()

    def new_episode_id(self) -> str:
        if self.mode == StorageMode.FRAGMENT:
            return str(self.index)
        elif self.mode == StorageMode.EPISODE:
            return str(uuid.uuid4())
        else:
            assert False

    def lazy_init_env(self) -> None:
        if self.env is None:
            sentry_sdk.init(  # type: ignore[abstract]
                dsn="https://198a62315b2c4c2a99cb8a5493224e2f@o568344.ingest.sentry.io/6453090",
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for performance monitoring.
                # We recommend adjusting this value in production.
                # traces_sample_rate=1.0,
            )
            self.env = build_env(env_params=self.env_params)
            self.env.reset()

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

    def run(self) -> None:
        """The worker entrypoint, will wait for commands from the main
        process and execute them."""
        signal.signal(signal.SIGINT, signal.default_int_handler)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            while True:
                cmd, action = self.pipe.recv()
                if cmd == "step":
                    result = self.step(action)
                    self.pipe.send(result)
                elif cmd == "close":
                    break
                else:
                    raise RuntimeError("Got unrecognized cmd %s" % cmd)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            capture_exception(e)
            raise
        finally:
            self.shutdown()

    def step(self, action: Dict[str, Any]) -> Tuple[Observation, float, bool, Info, str]:
        """Perform a single step of the environment."""
        self.lazy_init_env()
        assert self.env is not None
        next_obs, reward, done, info = self.env.step(action)  # type: ignore
        self.episode_rewards += reward
        episode_id = self.episode_id

        if done:
            info["final_episode_length"] = self.episode_steps
            info["final_episode_rewards"] = self.episode_rewards
            info["terminal_observation"] = next_obs
            next_obs = self.env.reset()

            self.episode_steps = 0
            self.episode_rewards = 0
            self.episode_id = self.new_episode_id()

        self.episode_steps += self.env_params.action_repeat

        return next_obs, reward, done, info, episode_id
