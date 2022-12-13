from enum import Enum
from typing import Optional

import attr
import gym.spaces
import torch


class ClippedNormalMode(Enum):
    NO_CLIPPING = "NO_CLIPPING"
    SAMPLE_DIST = "SAMPLE_DIST"
    TRUNCATED_NORMAL = "TRUNCATED_NORMAL"


@attr.s(auto_attribs=True, frozen=True)
class EnvironmentParams:
    suite: str = "godot"
    task: Optional[str] = None
    env_index: int = 0  # each environment will get a unique worker id in range [0, num_workers * num_worker_groups)
    env_count: int = 1  # this will get set automatically to num_workers
    action_repeat: int = 1
    time_limit: Optional[int] = None
    reward_scale: float = 1  # scale reward magnitude by this amount
    pixel_obs_wrapper: bool = False
    elapsed_time_obs: bool = False  # include an observation of the elapsed time
    mode: str = "train"
    frame_stack: int = 1
    info_fields: list[str] = [
        "cumulative_episode_return",
        "cumulative_episode_length",
    ]  # only non-nested scalar float-castable supported


@attr.s(auto_attribs=True, frozen=True)
class DmcEnvironmentParams(EnvironmentParams):
    suite: str = "dmc"
    include_proprio: bool = True
    include_rgb: bool = False


@attr.s(auto_attribs=True, frozen=True)
class ProcgenEnvironmentParams(EnvironmentParams):
    suite: str = "procgen"
    task: str = "starpilot"
    num_levels: int = 0
    start_level: int = 0
    distribution_mode: str = "easy"


@attr.s(auto_attribs=True, frozen=True)
class Params:
    # wandb
    project: str
    name: Optional[str] = None  # wandb run name
    tags: tuple[str, ...] = ()
    wandb_mode: str = "online"  # online, offline, disabled
    log_freq_hist: int = 50
    log_freq_scalar: int = 1
    log_freq_media: int = 250
    group: Optional[str] = None
    suggestion_uuid: str = ""  # for use by BONES and friends

    # environment
    env_params: EnvironmentParams = EnvironmentParams()

    # worker
    multiprocessing: bool = True
    num_workers: int = 8  # number of environments per worker group
    num_steps: int = 256  # number of steps per rollout
    time_limit_bootstrapping: bool = False
    # if obs_first=True, for a given timestep, the order is (observation, action, reward/done/info)
    # if obs_first=False, for a given timestep, the order is (action, reward/done/info, observation)
    obs_first: bool = True
    center_observations: bool = True
    allow_partial_batches: bool = True  # should we wait for all envs to be ready, or allow a partial batch?

    train_gpu: int = 0  # currently this does nothing (but will happen automatically)
    inference_gpus: tuple[int, ...] = (0,)  # only applied if there are async worker managers
    multiprocessing_mode: str = "fork"  # `spawn` or `fork`. `fork` is faster, `spawn` more compatable

    # training
    is_training: bool = True
    batch_size: int = 100
    # a string specifying the possible places to resume from, either local file or wandb run
    # valid formats include:
    #     wandb://{project}/{run}/{file_name}
    #     file://{absolute_file_path}
    # for example:
    #     wandb://untitled-ai/sf3189ytcfg/checkpoint.pt
    #     file:///home/user/runs/good_run/checkpoint.pt
    resume_from: Optional[str] = None
    checkpoint_every: int = 10_000
    total_env_steps: int = 1_000_000
    discount = 0.98

    # Policy/action head parameters
    normal_std_from_model: bool = True
    policy_normal_init_std: float = 1
    policy_normal_min_std: float = 0.01
    clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.NO_CLIPPING  # is this really the best default now?
    center_and_clamp_discrete_logits: bool = False

    # val/test
    is_testing: bool = False
    eval_exploration_mode: str = "eval"  # "eval" to use the mode of the distribution, "explore" to sample
    eval_workers: int = 16

    # spaces
    observation_space: Optional[gym.spaces.Dict] = None
    action_space: Optional[gym.spaces.Dict] = None

    debug: bool = False  # Disables some crash-recovery that obscures crashes when debugging.
    deterministic: bool = False

    @property
    def train_device(self) -> torch.device:
        return torch.device(f"cuda:{self.train_gpu}")
