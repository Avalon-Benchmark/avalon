# mypy: ignore-errors
# disabling type checking in this test code
"""Simple custom state-based environments to quickly validate things."""
import attr
import gym
import numpy as np

from avalon.agent.common.params import EnvironmentParams


def overwrite_params(params, new_params) -> None:
    for k, v in new_params.items():
        print(f"WARNING: want arg {k} but got {getattr(params, k)} with {v}")
        # setattr(params, k, v)


@attr.s(auto_attribs=True, frozen=True)
class TestEnvironmentParams(EnvironmentParams):
    suite: str = "test"
    task: str = "case1"

    # TODO: should consider ending the episode with the proper TimeLimit flag to make value calculations
    # act properly as if it's supposed to be infinite.
    long_episode_length: int = 1_000_000


def get_env(task, params: TestEnvironmentParams):
    if task == "dummy":
        return DummyEnv()
    elif task == "case1":
        # One long episode, returning reward 1 each step.
        # Use with discount = .75.
        # Expect value estimate to be 4. Value EV=1
        # No episode ends (or just relatively long episodes), reward always 1.
        # Actions should stay random - they are ignored.
        return Test1(params.long_episode_length, discrete=True)
    elif task == "case1_continuous_action":
        return Test1(params.long_episode_length, discrete=False)
    elif task == "case2":
        # Reward is the value of the last obs (-1 or 1)
        # Looking for: action should stay random
        # Looking for: value should have an EV of about .5 (since we can only know one of the discounted rewards)
        # Looking for: reward EV to be 1.
        return Test2(params.long_episode_length, reward_mode="obs")
    elif task == "case6":
        # Reward is 1 only if the action matches the last obs (-1 or 1)
        # So in a perfect run, the avg step reward should equal 1
        # Looking for: all metrics should show perfect predictions and action selections
        return Test2(params.long_episode_length, reward_mode="obs_and_action")
        # return Test2(1000000, reward_mode="obs")
    elif task == "hybrid1":
        # Penalize bad actions. Correct actions are some function of the last observation.
        # Perfect policy has reward of 0, bad policy will have negative reward.
        # Look for: 0 overall return, other metrics showing good predictions.
        return TestHybridAction(params.long_episode_length)
        # return Test2(1000000, reward_mode="obs")
    else:
        assert False


class DummyEnv:
    def __init__(self) -> None:
        self._random = np.random.RandomState(seed=0)
        self._step = None

    @property
    def observation_space(self):
        low = np.zeros([3, 84, 84], dtype=np.float32)
        high = 255 * np.ones([3, 84, 84], dtype=np.float32)
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        action_dim = 5
        low = -np.ones([action_dim], dtype=np.float32)
        high = np.ones([action_dim], dtype=np.float32)
        return gym.spaces.Box(low, high)

    def reset(self):
        self._step = 0
        obs = np.zeros((3, 84, 84), dtype=np.float32)
        return obs

    def step(self, action):
        obs = np.zeros((3, 84, 84), dtype=np.float32)
        # obs = self.observation_space.sample()
        reward = self._random.uniform(0, 1).astype(np.float32)
        self._step += 1
        done = self._step >= 1000
        info = {}
        return obs, reward, done, info

    def close(self) -> None:
        pass


class Test1:
    """Agent gets reward of 1 every step for num_steps. Total reward is num_steps.

    Ways to use:
    - set n = 1.
    - set n = inf. Set discount to .75. First step value est should converge to 4.

    """

    __test__ = False

    def __init__(self, num_steps, discrete=True) -> None:
        self._step = None
        self.num_steps = num_steps
        self.discrete = discrete

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,))

    @property
    def action_space(self):
        if self.discrete:
            return gym.spaces.Discrete(2)
        else:
            return gym.spaces.Box(-1, 1, shape=(1,))

    def reset(self):
        self._step = 0
        obs = np.zeros((1,), dtype=np.float32)
        return obs

    def step(self, action):
        obs = np.zeros((1,), dtype=np.float32)
        reward = float(1)
        self._step += 1
        done = self._step >= self.num_steps
        info = {}
        return obs, reward, done, info

    def close(self) -> None:
        pass


class Test2:
    """Random obs in the set (-1, 1). Binary action. Multiple reward modes."""

    __test__ = False

    def __init__(self, num_steps, reward_mode="obs_and_action") -> None:
        self._step = None
        self.num_steps = num_steps
        self.last_obs = np.zeros((1,))
        self.reward_mode = reward_mode

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Discrete(2)

    def reset(self):
        self._step = 0
        obs = np.zeros((1,), dtype=np.float32)
        return obs

    def step(self, action):
        assert isinstance(action, int)
        if self.reward_mode == "obs_and_action":
            # Obs is a random bool.
            # action should match obs. binary reward based on this.
            reward = self.last_obs.item() * (action * 2 - 1)
        elif self.reward_mode == "obs":
            reward = self.last_obs.item()
        else:
            assert False
        self._step += 1
        done = self._step >= self.num_steps
        info = {}
        # high is non-inclusive.
        obs = np.random.randint(0, 2, size=(1,)).astype(np.float32) * 2 - 1
        # obs = np.random.uniform(-1, 1, size=(1,))
        self.last_obs = obs
        return obs.astype(dtype=np.float32), reward, done, info

    def close(self) -> None:
        pass


class TestHybridAction:
    __test__ = False

    def __init__(self, num_steps) -> None:
        self._step = None
        self.num_steps = num_steps
        self.last_obs = np.zeros((1,))

    @property
    def observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(1,))

    @property
    def action_space(self):
        return gym.spaces.Dict({"discrete": gym.spaces.Discrete(2), "continuous": gym.spaces.Box(0, 1, shape=(1,))})

    def reset(self):
        self._step = 0
        obs = np.zeros((1,)).astype(np.float32)
        return obs

    def step(self, action):
        # TODO: continuous actions come in as an array, discrete come in as integers.
        # Standardize on always being an array, probably?
        continuous = action["continuous"].item()
        discrete = action["discrete"]

        # At every step we generate a random obs in (-1, 1).
        # The reward is the distance between last_obs and continuous * sign(discrete)
        # Continuous is nonnegative, so it will control the scale. Discrete will control the sign.
        # Both have to have a nontrivial solution to get the optimal policy.
        # Reward will be 0 if we are perfect.
        reward = -1 * (self.last_obs.item() - (continuous * (discrete * 2 - 1))) ** 2

        self._step += 1
        done = self._step >= self.num_steps
        info = {}
        obs = np.random.uniform(-1, 1, size=(1,)).astype(np.float32)
        self.last_obs = obs
        return obs, reward, done, info

    def close(self) -> None:
        pass
