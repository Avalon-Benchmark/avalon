import time
import uuid
from collections import defaultdict
from collections import deque

import gym
import numpy as np
import sentry_sdk
import torch
from sentry_sdk import capture_exception
from tree import map_structure

from agent.storage import TrajectoryStorage


class GamePlayer:
    """A manager class for running multiple game-playing processes."""

    def __init__(
        self,
        args,
        env_fn,
        num_workers: int,
        multiprocessing: bool,
        storage: TrajectoryStorage,
        ctx,
        obs_space,
        mode,
    ):
        # we keep a running snapshot of recent statistics with a len-100 dequeue
        self.episode_length: deque = deque(maxlen=args.stats_buffer_size)
        self.episode_rewards: deque = deque(maxlen=args.stats_buffer_size)
        self.multiprocessing = multiprocessing
        self.num_workers = num_workers
        assert mode in ("fragment", "episode")
        self.mode = mode
        self.args = args

        # Start game-playing processes
        self.processes = []
        for i in range(num_workers):
            if self.multiprocessing:
                parent_conn, child_conn = ctx.Pipe()
                worker = SubprocWorker(i, child_conn, env_fn, mode, args)
                # Can't have daemon=True because the godot env spawns subprocesses
                ps = ctx.Process(target=worker.run, args=(), daemon=False)
                ps.start()
                self.processes.append((ps, parent_conn))
            else:
                worker = SubprocWorker(i, None, env_fn, mode, args)
                self.processes.append((worker, None))

        self.storage = storage

        # This is obs_{t+1}
        self.next_obs = {
            k: np.zeros(shape=(num_workers, *v.shape), dtype=np.float32) for k, v in obs_space.spaces.items()
        }
        # This is done_t
        self.dones = [False for _ in range(self.num_workers)]

        self.observation_space = obs_space

    def shutdown(self):
        if self.multiprocessing:
            for i, (p, pipe) in enumerate(self.processes):
                pipe.send(("close", None, None))

    def run_rollout(
        self,
        num_steps: int,
        model: torch.nn.Module,
        rollout_device: torch.device,
        action_space: gym.Space,
    ):
        """Rollouts at index i in the buffer are:
        observation, action + value computed from that obs | step | reward, done
        """
        model.eval()

        remaining_steps = [num_steps] * self.num_workers
        ready_for_new_step = [True] * self.num_workers

        while True:
            if sum(remaining_steps) == 0:
                break
            step_data = defaultdict(dict)

            # run the model
            torch_obs = {k: torch.tensor(v, device=rollout_device) for k, v in self.next_obs.items()}
            torch_dones = torch.tensor(self.dones, dtype=torch.bool, device=rollout_device)

            # step_actions should be a dict with values of shape (sum(ready_for_new_step), action_dim)
            # TODO: suprisingly, it's meaningfully slower to only run the ready workers than to run them all, for PPO at least.
            # I think slicing the obs is slowing things down? Try to optimize that.
            with torch.no_grad():
                step_actions, to_store = model.rollout_step(torch_obs, torch_dones, ready_for_new_step)

            # make a mapping from worker_id to the id of the corresponding entry in step_actions and to_store -
            # it's not 1-1 because we only run the indices that are ready for a new step
            action_mapping = (torch.cumsum(torch.tensor(ready_for_new_step), dim=0) - 1).tolist()

            if self.multiprocessing:
                # Send the selected actions to workers and request a step
                for i, (p, pipe) in enumerate(self.processes):
                    if ready_for_new_step[i] and remaining_steps[i] > 0:
                        # slice out just this worker's action
                        action = map_structure(lambda x: x[action_mapping[i]], step_actions)
                        pipe.send(("step", None, action))
                        ready_for_new_step[i] = False
            else:
                ready_for_new_step = [False for _ in ready_for_new_step]

            step_time = time.time()
            threshold_time = None
            active_workers = len([x for x in remaining_steps if x != 0])
            while True:
                if active_workers == 0:
                    break
                needed_percent = max(0.5, min(0.6, 1 - (1 / active_workers)))
                if sum(ready_for_new_step) >= active_workers:
                    break
                if sum(ready_for_new_step) / active_workers + 1e-5 >= needed_percent:
                    if threshold_time is None:
                        threshold_time = time.time()
                    if (threshold_time - step_time) * 1.25 < (time.time() - step_time):
                        break

                # Receive step data from workers
                for i, (p, pipe) in enumerate(self.processes):
                    if self.multiprocessing:
                        if remaining_steps[i] == 0 or ready_for_new_step[i] or not pipe.poll():
                            continue
                        data = pipe.recv()
                        (step_obs, reward, done, info, episode_id) = data
                    else:
                        action = {k: v[action_mapping[i]] for k, v in step_actions.items()}
                        (step_obs, reward, done, info, episode_id) = p.step(action, None)
                    try:
                        self.episode_length.append(info["final_episode_length"])
                        self.episode_rewards.append(info["final_episode_rewards"])
                    except KeyError:
                        pass

                    if self.args.time_limit_bootstrapping:
                        # value bootstrapping for envs with artificial time limits imposed by a gym TimeLimit wrapper
                        # see https://arxiv.org/pdf/1712.00378.pdf for context
                        if done and info.get("TimeLimit.truncated", False) is True:
                            # this does get wrapper obs transforms applied
                            terminal_obs = info["terminal_observation"]
                            terminal_obs_torch = {
                                k: torch.tensor(v).to(rollout_device).float().unsqueeze(0)
                                for k, v in terminal_obs.items()
                            }
                            with torch.no_grad():
                                terminal_value, _ = model(terminal_obs_torch)
                            reward += self.args.discount * terminal_value[0].detach().cpu().item()

                    # TODO: i'm not 100% positive that the episode_id stuff will be correct around an episode reset.
                    # is_terminal indicates if a true environment termination happened (not a time limit)
                    is_terminal = done and not info.get("TimeLimit.truncated", False)
                    # record all the data for this step
                    assert isinstance(reward, (int, float))
                    step_data[episode_id]["rewards"] = reward
                    step_data[episode_id]["dones"] = done
                    step_data[episode_id]["is_terminal"] = is_terminal
                    for k, v in to_store.items():
                        # This is where we store stuff like values, policy_probs
                        step_data[episode_id][k] = v[action_mapping[i]]
                    step_data[episode_id]["info"] = info
                    for k, v in action_space.items():
                        step_data[episode_id][f"action__{k}"] = step_actions[k][action_mapping[i]]

                    # record the new observation+done for policy evaluation for the next step
                    for k, v in step_obs.items():
                        if self.args.obs_first:
                            step_data[episode_id][f"obs__{k}"] = self.next_obs[k][i].copy()
                            self.next_obs[k][i] = v
                        else:
                            self.next_obs[k][i] = v
                            # TODO: After an epsisode reset, this observation really belongs with the new episode, no?
                            # But since trajectories start with an action in this mode, i guess it just goes with this ep.
                            step_data[episode_id][f"obs__{k}"] = self.next_obs[k][i].copy()
                    self.dones[i] = done

                    ready_for_new_step[i] = True
                    remaining_steps[i] -= 1
                    if remaining_steps[i] == 0:
                        ready_for_new_step[i] = False
                        active_workers -= 1

            self.storage.add_timestep_samples(step_data)


class SubprocWorker:
    """A worker for running an environment, intended to be run on a separate
    process."""

    def __init__(self, index, pipe, env_fn, mode, args):
        self.mode = mode
        self.index = index
        self.pipe = pipe
        self.episode_steps = 0
        self.episode_rewards = 0
        self.previous_lives = 0

        self.env_fn = env_fn
        self.index = index
        self.args = args
        self.env = None

        self.episode_id = self.new_episode_id()

    def new_episode_id(self):
        if self.mode == "fragment":
            return str(self.index)
        elif self.mode == "episode":
            return str(uuid.uuid4())
        else:
            assert False

    def lazy_init_env(self):
        if self.env is None:
            sentry_sdk.init(
                dsn="https://198a62315b2c4c2a99cb8a5493224e2f@o568344.ingest.sentry.io/6453090",
                # Set traces_sample_rate to 1.0 to capture 100%
                # of transactions for performance monitoring.
                # We recommend adjusting this value in production.
                traces_sample_rate=1.0,
            )
            self.env = self.env_fn(seed=self.index)
            self.env.reset()

    def run(self):
        """The worker entrypoint, will wait for commands from the main
        process and execute them."""
        try:
            while True:
                cmd, t, action = self.pipe.recv()
                if cmd == "step":
                    self.pipe.send(self.step(action, t))
                elif cmd == "close":
                    self.pipe.send(None)
                    break
                else:
                    raise RuntimeError("Got unrecognized cmd %s" % cmd)
        except KeyboardInterrupt:
            print("worker: got KeyboardInterrupt")
        except Exception as e:
            capture_exception(e)
            raise
        finally:
            try:
                self.env.close()
            except Exception as e:
                capture_exception(e)
                raise

    def step(self, action, t):
        """Perform a single step of the environment."""
        # Uncomment this for random actions
        # action = self.env.action_space.sample()
        self.lazy_init_env()
        next_obs, reward, done, info = self.env.step(action)
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

        self.episode_steps += self.args.action_repeat

        return next_obs, reward, done, info, episode_id
