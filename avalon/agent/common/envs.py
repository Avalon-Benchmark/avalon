import warnings
from typing import Any
from typing import List
from typing import Tuple
from typing import cast

import gym
import numpy as np
from gym.wrappers import TimeLimit
from loguru import logger

from avalon.agent.common import wrappers
from avalon.agent.common.params import DmcEnvironmentParams
from avalon.agent.common.params import EnvironmentParams
from avalon.agent.common.params import ProcgenEnvironmentParams
from avalon.agent.common.test_envs import TestEnvironmentParams
from avalon.agent.common.wrappers import ElapsedTimeWrapper
from avalon.common.type_utils import assert_not_none


def build_env(env_params: EnvironmentParams, torchify: bool = True) -> gym.Env:
    # TODO: I should add a "checker" wrapper that checks that observations and actions match that specced in the Space.
    seed = env_params.env_index
    env: gym.Env
    if env_params.suite == "dmc":
        # For rendering, you'll need libglew-(dev/2.0/2.1), and MUJOCO_GL=egl
        assert isinstance(env_params, DmcEnvironmentParams)
        assert env_params.task is not None
        assert "-" not in env_params.task, "use underscores in task names"
        # This camera config comes from the dreamerv2 repo
        camera = dict(
            quadruped_walk=2,
            quadruped_run=2,
            quadruped_escape=2,
            quadruped_fetch=2,
            locom_rodent_maze_forage=1,
            locom_rodent_two_touch=1,
        ).get(env_params.task, 0)
        env = DeepMindControl(
            env_params.task,
            size=(64, 64),
            include_state=env_params.include_proprio,
            include_rgb=env_params.include_rgb,
            camera=camera,
            seed=seed,
        )
        env = wrappers.RecordEpisodeStatistics(env)  # must happen before reward scaling
        env = wrappers.ActionRepeat(env, env_params.action_repeat)
        # rescales actions from standard ranges to the envs desired range.
        if env_params.time_limit:
            env = wrappers.TimeLimit(env, max_episode_steps=env_params.time_limit // env_params.action_repeat)
        env = wrappers.DictObsActionWrapper(env)
        if env_params.include_rgb:
            env = wrappers.ImageTransformWrapper(env, key="rgb", greyscale=False, resolution=None)
    elif env_params.suite == "godot":
        from avalon.agent.godot.godot_gym import AvalonEnv
        from avalon.agent.godot.godot_gym import GodotEnvironmentParams
        from avalon.agent.godot.godot_gym import GodotObsTransformWrapper
        from avalon.agent.godot.godot_gym import ScaleAndSquashAction

        assert isinstance(env_params, GodotEnvironmentParams)
        assert env_params.time_limit is None, "godot has its own time limit"
        assert env_params.action_repeat == 1

        # Note: This will seed itself properly using env_index
        env = AvalonEnv(env_params)
        # We don't use the TimeLimit wrapper because the time limit is dynamic,
        # so we trust that the godot env gives the proper TimeLimit.truncated signal
        # (which it should) for the timelimit boostrapping to work properly if enabled.
        env = GodotObsTransformWrapper(env, greyscale=env_params.greyscale)
        if env_params.mode == "train":
            from avalon.agent.godot.godot_gym import CurriculumWrapper

            env = CurriculumWrapper(
                env,  # type: ignore[arg-type]
                task_difficulty_update=env_params.task_difficulty_update,
                meta_difficulty_update=env_params.meta_difficulty_update,
            )
        env = ScaleAndSquashAction(env, scale=1)
        env = wrappers.OneHotActionWrapper(env)
        # env = RewardSoftClipWrapper(env, scale=5)
    elif env_params.suite == "test":
        # Note: haven't implemented proper seeding in these test envs.
        assert env_params.action_repeat == 1
        from avalon.agent.common.test_envs import get_env

        assert type(env_params) == TestEnvironmentParams
        env = get_env(env_params.task, env_params)
        env = wrappers.DictObsActionWrapper(env)
        env = wrappers.OneHotActionWrapper(env)
        env = wrappers.RecordEpisodeStatistics(env)  # must happen before reward scaling
        if env_params.time_limit:
            env = wrappers.TimeLimit(env, max_episode_steps=env_params.time_limit // env_params.action_repeat)
    elif env_params.suite == "gym":
        assert env_params.action_repeat == 1
        # Annoyingly, gym envs apply their own time limit already.
        logger.info("time limit arg ignored in gym envs")
        env = gym.make(assert_not_none(env_params.task))
        env.seed(seed)
        # Hacky. Relies on the TimeWrapper being the outermost wrapper. Not sure the better way.
        assert isinstance(env, (ElapsedTimeWrapper, TimeLimit))
        max_steps = env._max_episode_steps
        logger.info(f"env has a time limit of {max_steps} steps")
        if env_params.pixel_obs_wrapper:
            env = wrappers.PixelObsWrapper(env)
            env = wrappers.DictObsActionWrapper(env, obs_key="rgb")
        else:
            env = wrappers.DictObsActionWrapper(env, obs_key="state")  # type: ignore
        if env_params.pixel_obs_wrapper:
            env = wrappers.ImageTransformWrapper(env, key="rgb", greyscale=True, resolution=64)
        env = wrappers.OneHotActionWrapper(env)
        if env_params.elapsed_time_obs:
            env = wrappers.ElapsedTimeWrapper(env, max_steps)
        env = wrappers.RecordEpisodeStatistics(env)  # must happen before reward scaling
    elif env_params.suite == "atari":
        assert env_params.task is not None
        assert env_params.action_repeat == 4
        assert env_params.time_limit == 27000
        assert env_params.elapsed_time_obs is False
        # These are the settings from dreamerv2
        env = Atari(env_params.task, action_repeat=env_params.action_repeat, size=(64, 64), grayscale=True)
        env.seed(seed)
        if env_params.time_limit:
            # danijar applies the time limit in agent-steps, not action-repeated env steps
            env = wrappers.TimeLimit(env, max_episode_steps=env_params.time_limit)
        env = wrappers.DictObsActionWrapper(env, obs_key="rgb")
        env = wrappers.OneHotActionWrapper(env)
        env = wrappers.RecordEpisodeStatistics(env)  # must happen before reward scaling
        # Note the tanh here!
        env = wrappers.ScaleRewards(env, func=np.tanh)
        # Just converts from hwc to chw
        env = wrappers.ImageTransformWrapper(env, key="rgb")
    elif env_params.suite == "procgen":
        warnings.filterwarnings("ignore", message=".*Future gym versions will require.*")
        # Need this import to register the procgen envs?
        import procgen  # isort: skip

        assert isinstance(env_params, ProcgenEnvironmentParams)
        assert env_params.task is not None
        assert env_params.time_limit is None, "procgen has its own time limits (altho they don't set the info flags)"

        env = gym.make(
            f"procgen-{env_params.task}-v0",
            start_level=env_params.start_level,
            num_levels=env_params.num_levels,
            distribution_mode=env_params.distribution_mode,
            rand_seed=env_params.env_index,
        )
        env = wrappers.DictObsActionWrapper(env, obs_key="rgb")
        env = wrappers.OneHotActionWrapper(env)
        env = wrappers.RecordEpisodeStatistics(env)  # must happen before reward scaling
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))  # type: ignore[no-any-return]
        # Just converts from hwc to chw
        env = wrappers.ImageTransformWrapper(env, key="rgb")
    else:
        assert False
    env = wrappers.NormalizeActions(env)
    env = wrappers.ScaleRewards(env, env_params.reward_scale)
    env = wrappers.ClipActionWrapper(env)
    if env_params.frame_stack != 1:
        env = wrappers.DictFrameStack(env, num_stack=env_params.frame_stack)
    if torchify:
        # My worker requires this, but it won't work with eg the builtin gym vecenv.
        env = wrappers.Torchify(env)
    return env


DMC_TASKS = [
    "acrobot_swingup",
    "cartpole_balance",
    "cartpole_balance_sparse",
    "cartpole_swingup",
    "cartpole_swingup_sparse",
    "cheetah_run",
    "cup_catch",
    "finger_spin",
    "finger_turn_easy",
    "finger_turn_hard",
    "hopper_hop",
    "hopper_stand",
    "pendulum_swingup",
    "quadruped_run",
    "quadruped_walk",
    "reacher_easy",
    "reacher_hard",
    "walker_walk",
    "walker_stand",
    "walker_run",
]

PROCGEN_ENVS = [
    "coinrun",
    "starpilot",
    "caveflyer",
    "dodgeball",
    "fruitbot",
    "chaser",
    "miner",
    "jumper",
    "leaper",
    "maze",
    "bigfish",
    "heist",
    "climber",
    "plunder",
    "ninja",
    "bossfight",
]


class DeepMindControl(gym.Env):
    def __init__(
        self,
        name: str,
        size: Tuple[int, int] = (64, 64),
        camera: Any = None,
        include_state: bool = False,
        include_rgb: bool = True,
        seed: int = 0,
    ) -> None:
        from dm_control import suite

        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        self._env = suite.load(domain, task, task_kwargs={"random": seed})
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.include_state = include_state
        self.include_rgb = include_rgb
        # TODO: fix!
        # We just ignore all scalar spaces, because this is how danijar did it (presumably accidentally).
        self.scalar_spaces: List[str] = []

    @property
    def observation_space(self) -> gym.spaces.Dict:  # type: ignore[override]
        spaces: dict[str, gym.Space] = {}
        if self.include_state:
            for key, value in self._env.observation_spec().items():
                logger.warning("gym spaces do not give observation ranges. no rescaling will be applied.")
                if value.shape == ():
                    self.scalar_spaces.append(key)
                    continue
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        if self.include_rgb:
            spaces["rgb"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Box:  # type: ignore[override]
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):  # type: ignore
        time_step = self._env.step(action)
        obs = {}
        if self.include_state:
            state = {k: v for k, v in dict(time_step.observation).items() if k not in self.scalar_spaces}
            state = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in state.items()}
            obs |= state
        if self.include_rgb:
            obs["rgb"] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):  # type: ignore
        time_step = self._env.reset()
        obs = {}
        if self.include_state:
            state = {k: v for k, v in dict(time_step.observation).items() if k not in self.scalar_spaces}
            state = {k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in state.items()}
            obs |= state
        if self.include_rgb:
            obs["rgb"] = self.render()
        return obs

    def render(self, *args, **kwargs):  # type: ignore
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)


ATARI_TASKS = [
    # "adventure",
    # "air_raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank_heist",
    "battle_zone",
    "beam_rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    # "carnival",
    "centipede",
    "chopper_command",
    "crazy_climber",
    # "defender",
    "demon_attack",
    "double_dunk",
    # "elevator_action",
    "enduro",
    "fishing_derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice_hockey",
    "jamesbond",
    # "journey_escape",
    # "kaboom",
    "kangaroo",
    "krull",
    "kung_fu_master",
    "montezuma_revenge",
    "ms_pacman",
    "name_this_game",
    "phoenix",
    "pitfall",
    "pong",
    # "pooyan",
    "private_eye",
    "qbert",
    "riverraid",
    "road_runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space_invaders",
    "star_gunner",
    "tennis",
    "time_pilot",
    "tutankham",
    "up_n_down",
    "venture",
    "video_pinball",
    "wizard_of_wor",
    "yars_revenge",
    "zaxxon",
]


class Atari(gym.Env):
    def __init__(
        self,
        name: str,
        action_repeat: int = 4,
        size: Tuple[int, int] = (84, 84),
        grayscale: bool = True,
        noops: int = 30,
        life_done: bool = False,
        sticky: bool = True,
        all_actions: bool = False,
    ) -> None:
        assert size[0] == size[1]
        import gym.envs.atari
        import gym.wrappers

        if name == "james_bond":
            name = "jamesbond"
        # this source is in https://github.com/mgbellemare/Arcade-Learning-Environment/blob/master/src/gym/envs/atari/environment.py
        env = gym.envs.atari.AtariEnv(
            game=name,
            obs_type="image",
            frameskip=1,
            repeat_action_probability=0.25 if sticky else 0.0,
            full_action_space=all_actions,
        )
        # Avoid unnecessary rendering in inner env.
        env._get_obs = lambda: None  # type: ignore
        # Tell wrapper that the inner env has no action repeat.
        env.spec = gym.envs.registration.EnvSpec("NoFrameskip-v0")
        self._env = gym.wrappers.AtariPreprocessing(env, noops, action_repeat, size[0], life_done, grayscale)
        self._size = size
        self._grayscale = grayscale

    @property
    def observation_space(self) -> gym.spaces.Box:  # type: ignore[override]
        shape = self._size + (1 if self._grayscale else 3,)
        return gym.spaces.Box(0, 255, shape, np.uint8)

    @property
    def action_space(self) -> gym.spaces.Box:  # type: ignore[override]
        return cast(gym.spaces.Box, self._env.action_space)

    def step(self, action: int):  # type: ignore
        image, reward, done, info = self._env.step(action)
        if self._grayscale:
            image = image[..., None]
        # info["is_terminal"] = done
        # info["is_first"] = False
        # info["is_last"] = done
        # image = rearrange(image, "h w c -> c h w")
        return image, reward, done, info

    def reset(self):  # type: ignore
        image = self._env.reset()
        if self._grayscale:
            image = image[..., None]
        # image = rearrange(image, "h w c -> c h w")
        return image

    def close(self):  # type: ignore
        return self._env.close()

    def render(self, mode: str = "human"):
        raise NotImplementedError
