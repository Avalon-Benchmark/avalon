from typing import Optional

from tap import Tap

from agent.observation_model import ObservationModelParams


class PPOParams(Tap):
    name: Optional[str] = None
    env_name: str = "godot_default"
    model: str = "godot"
    discount: float = 0.98
    lam: float = 0.9
    clip_range: float = 0.1
    clip_range_vf: Optional[float] = None
    value_loss_coef: float = 1
    entropy_coef: float = 3e-4
    num_workers: int = 8
    num_steps: int = 150
    num_batches: int = 1
    ppo_epochs: int = 2
    lr: float = 4e-4
    clip_grad_norm: float = 0.5
    device: str = "cuda"
    project: str = "avalon__ppo"
    total_env_steps: int = 30_000_000
    tag: str = "untagged"
    pixel_obs_wrapper: bool = False
    mp_method: str = "fork"
    multiprocessing: bool = True
    log_freq_hist: int = 50
    log_freq_scalar: int = 1
    log_freq_media: int = 250
    val: bool = False
    val_freq: int = 200_000
    val_episodes_per_task: int = 6
    test_episodes_per_task: int = 51
    time_limit: int = 120
    task_difficulty_update: float = 3e-4
    meta_difficulty_update: float = 3e-5
    is_val_rollout_saved: bool = False
    is_test_rollout_saved: bool = True
    is_train_only: bool = False
    energy_cost_coefficient: float = 1e-4
    num_task_groups: int = 4
    is_reward_dense: bool = False
    fixed_world_max_difficulty: float = 0.5

    elapsed_time_obs: bool = False
    reward_scale: float = 1.0
    stats_buffer_size: int = 10
    normal_std_from_model: bool = True
    clipped_normal_mode: str = "none"
    separate_ongoing: bool = False  # train on incomplete episodes
    write_episodes_to_disk: bool = False
    valtest_multiprocessing: bool = False
    valtest_num_workers: int = 1

    model_params: ObservationModelParams = ObservationModelParams()

    # don't routinely change these!
    action_repeat: int = 1
    obs_first: bool = True
    time_limit_bootstrapping: bool = True

    def __init__(self):
        super().__init__(self, explicit_bool=True)

    @property
    def batch_size(self) -> int:
        return int(self.num_steps * self.num_workers / self.num_batches)

    def process_args(self):
        assert self.num_steps * self.num_workers % self.num_batches == 0, "batch size does not divide cleanly"

        # These should only be changed with great care.
        assert self.action_repeat == 1
        assert self.obs_first is True
        assert self.time_limit_bootstrapping is True
