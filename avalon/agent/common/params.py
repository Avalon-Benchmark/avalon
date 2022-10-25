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


@attr.s(auto_attribs=True, frozen=True)
class DmcEnvironmentParams(EnvironmentParams):
    suite: str = "dmc"
    include_proprio: bool = True
    include_rgb: bool = False


@attr.s(auto_attribs=True, frozen=True)
class Params:
    # wandb
    project: str
    name: Optional[str] = None  # wandb run name
    tags: tuple[str, ...] = ()
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
    time_limit_bootstrapping: bool = False
    obs_first: bool = True

    train_gpu: int = 0  # currently this does nothing (but will happen automatically)
    inference_gpus: tuple[int, ...] = (0,)  # only applied if there are async worker managers

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
    normal_std_from_model: bool = True
    policy_normal_init_std: float = 1
    policy_normal_min_std: float = 0.01
    clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.NO_CLIPPING  # is this really the best default now?

    # val/test
    is_testing: bool = False
    eval_exploration_mode: str = "eval"  # "eval" to use the mode of the distribution, "explore" to sample
    eval_workers: int = 16

    # spaces
    observation_space: Optional[gym.spaces.Dict] = None
    action_space: Optional[gym.spaces.Dict] = None

    debug: bool = False  # Disables some crash-recovery that obscures crashes when debugging.

    @property
    def train_device(self) -> torch.device:
        return torch.device(f"cuda:{self.train_gpu}")
