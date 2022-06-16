import gym
import numpy as np

from agent.godot_gym import AvalonGodotEnvWrapper
from agent.godot_gym import GodotEnvParams
from agent.godot_gym import GodotObsTransformWrapper
from agent.godot_gym import ScaleAndSquashAction
from agent.wrappers import ClipActionWrapper
from agent.wrappers import CurriculumWrapper
from agent.wrappers import DictObsActionWrapper
from agent.wrappers import ElapsedTimeWrapper
from agent.wrappers import ImageTransformWrapper
from agent.wrappers import NormalizeActions
from agent.wrappers import OneHotActionWrapper
from agent.wrappers import PixelObsWrapper
from agent.wrappers import ScaleRewards
from agent.wrappers import TimeLimit
from common.errors import SwitchError
from datagen.godot_env import GodotEnv


def build_env(args, seed=0, mode="train"):
    suite, task = args.env_name.split("_", 1)
    if suite == "dmc":
        import quarantine.zack.dreamer_danijar.wrappers as wrappers

        # this has dict obs space with images at ["rgb"]
        env = DeepMindControl(task)
        env = wrappers.ActionRepeat(env, args.action_repeat)
        # rescales actions from standard ranges to the envs desired range.
        env = TimeLimit(env, max_episode_steps=args.time_limit / args.action_repeat)
        env = DictObsActionWrapper(env)
        env = ImageTransformWrapper(env, key="rgb", greyscale=False, resolution=None)
    elif suite == "godot":
        assert args.action_repeat == 1

        # The seed offset is to keep the different types of envs from overwriting each other's config files.
        if mode == "train":
            seed_offset = 0
            num_fixed_worlds_per_task = 0
            is_fixed_generator = False
        elif mode == "val":
            seed_offset = 1000
            num_fixed_worlds_per_task = args.val_episodes_per_task
            is_fixed_generator = True
        elif mode == "test":
            seed_offset = 2000
            num_fixed_worlds_per_task = args.test_episodes_per_task
            is_fixed_generator = True
        elif mode == "dummy":
            seed_offset = 3000
            num_fixed_worlds_per_task = 0
            is_fixed_generator = False
        else:
            raise SwitchError(f"Unknown mode {mode}")

        config = GodotEnvParams(
            random_int=seed + seed_offset,
            max_frames=args.time_limit,
            num_fixed_worlds_per_task=num_fixed_worlds_per_task,
            is_fixed_generator=is_fixed_generator,
            energy_cost_coefficient=args.energy_cost_coefficient,
            fixed_world_max_difficulty=args.fixed_world_max_difficulty,
            gpu_id=0,
        )
        env = AvalonGodotEnvWrapper(config)
        # We don't use the TimeLimit wrapper because the time limit is dynamic,
        # so we trust that the godot env gives the proper TimeLimit.truncated signal
        # (which it should) for the timelimit boostrapping to work properly if enabled.
        env = GodotObsTransformWrapper(env)
        if mode == "train":
            env = CurriculumWrapper(
                env,
                task_difficulty_update=args.task_difficulty_update,
                meta_difficulty_update=args.meta_difficulty_update,
            )
        env = ScaleAndSquashAction(env, scale=1)
        env = OneHotActionWrapper(env)
    elif suite == "gym":
        assert args.action_repeat == 1
        # Annoyingly, gym envs apply their own time limit already.
        print("time limit arg ignored in gym envs")
        env = gym.make(task)
        # Hacky. Relies on the TimeWrapper being the outermost wrapper. Not sure the better way.
        max_steps = env._max_episode_steps
        print(f"env has a time limit of {max_steps} steps")
        # env = DiscreteActionToIntWrapper(env)
        if args.pixel_obs_wrapper:
            env = PixelObsWrapper(env)
            env = DictObsActionWrapper(env, obs_key="rgb")
        else:
            env = DictObsActionWrapper(env, obs_key="state")
        if args.pixel_obs_wrapper:
            env = ImageTransformWrapper(env, key="rgb", greyscale=True, resolution=64)
        env = ClipActionWrapper(env)
        env = OneHotActionWrapper(env)
        if args.elapsed_time_obs:
            env = ElapsedTimeWrapper(env, max_steps)
    else:
        assert False
    env = NormalizeActions(env)
    env = ScaleRewards(env, args.reward_scale)
    return env


def wrap_godot_eval_env(env: GodotEnv):
    env = GodotObsTransformWrapper(env)
    env = ScaleAndSquashAction(env, scale=1)
    env = OneHotActionWrapper(env)
    env = NormalizeActions(env)
    env = ScaleRewards(env, 1)
    return env


class DeepMindControl:
    def __init__(self, name: str, size=(64, 64), camera=None, include_state=False, include_rgb=True):
        domain, task = name.split("_", 1)
        if domain == "cup":  # Only domain with multiple words.
            domain = "ball_in_cup"
        if isinstance(domain, str):
            from dm_control import suite

            self._env = suite.load(domain, task)
        else:
            assert task is None
            self._env = domain()
        self._size = size
        if camera is None:
            camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera
        self.include_state = include_state
        self.include_rgb = include_rgb

    @property
    def observation_space(self):
        spaces = {}
        if self.include_state:
            for key, value in self._env.observation_spec().items():
                print("warning: gym spaces do not give observation ranges. no rescaling will be applied.")
                spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, dtype=np.float32)
        if self.include_rgb:
            spaces["rgb"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        time_step = self._env.step(action)
        obs = {}
        if self.include_state:
            obs |= dict(time_step.observation)
        if self.include_rgb:
            obs["rgb"] = self.render()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {"discount": np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = {}
        if self.include_state:
            obs |= dict(time_step.observation)
        if self.include_rgb:
            obs["rgb"] = self.render()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.physics.render(*self._size, camera_id=self._camera)
