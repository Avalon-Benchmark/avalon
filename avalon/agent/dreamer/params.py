import attr

from avalon.agent.common.params import ClippedNormalMode
from avalon.agent.common.params import Params
from avalon.contrib.utils import FILESYSTEM_ROOT


@attr.s(auto_attribs=True, frozen=True)
class OffPolicyParams(Params):
    worker_managers: int = 1
    # ratio between these two affects sample efficiency
    num_steps: int = 500  # env steps (per worker) per loop. this is how often we reload a new model and log stats.
    train_steps: int = 100  # train steps per training iteration
    min_fragment_len: int = 50  # min length of an episode fragment to train on
    max_fragment_len: int = 50  # max length of an episode fragment to train on

    data_dir: str = f"{FILESYSTEM_ROOT}/data/rollouts/"
    num_dataloader_workers: int = 1
    prioritize_ends: bool = True  # in dreamerv2, this is true by default and false in dmc_vision and dmc_proprio
    separate_ongoing: bool = True  # don't train on incomplete episodes
    prefill_steps: int = 5000  # only used in the danijar-replica trainer
    pretrain_steps: int = 0
    replay_buffer_size_timesteps: int = 1_000_000
    observation_compression: bool = False

    rollout_model_update_freq: int = 100  # how many training steps between pushing a new model

    log_rollout_metrics_every: int = 20  # how many rollout episodes between logging stats
    prefill_eps_per_dataloader: int = 5  # how many episodes to load before a dataloader starts serving batches
    multiprocessing_mode: str = "spawn"  # currently, spawn is necessary for the off policy to work. could be fixed.

    @property
    def replay_buffer_size_timesteps_per_manager(self) -> int:
        return int(self.replay_buffer_size_timesteps / self.worker_managers)


@attr.s(auto_attribs=True, frozen=True)
class DreamerParams(OffPolicyParams):
    """Note: these are all the "default" params in the dreamerv2 config.
    PLEASE DON'T CHANGE THESE!
    I want to maintain this config for replicating danijar's repo.
    """

    # overrides
    project: str = "avalon_dreamer"
    clip_grad_norm: float = 100.0
    reward_loss_scale: float = 1
    discount: float = 0.99
    obs_first: bool = False
    normal_std_from_model: bool = True
    clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.TRUNCATED_NORMAL
    policy_normal_init_std: float = (
        0  # Seems strange to start with 0 (well, .1) std but afaict this is how dreamerv2 does it
    )
    policy_normal_min_std: float = 0.1
    replay_buffer_size_timesteps: int = 2_000_000
    min_fragment_len: int = 50
    max_fragment_len: int = 50
    prioritize_ends: bool = True
    batch_size: int = 16

    # new dreamer stuff
    horizon: int = 15  # how long to roll out imagination
    stoch_size: int = 32  # the size of the stochastic part of the latent. only the output shape of some MLPs
    # the size of the deterministic part of the latent. also the hidden and state size of the GRU
    # this defaulted to 200 in dreamerv1, and 1024 in dreamerv2 (but 200 for dmc, 600 atari)
    deter_size: int = 1024
    # hidden layer size for MLPs in RSSM model
    # note that this is 1024 default in dreamerv2 (but 200 in dmc, 600 in atari), and 400 in dreamerv1
    rssm_hidden_size: int = 1024
    encoder_mlp_hidden_size: int = 400
    # 400 in all dreamerv2 configs
    decoder_mlp_hidden_size: int = 400

    model_lr: float = 1e-4
    value_lr: float = 2e-4
    actor_lr: float = 8e-5
    adam_eps: float = 1e-5
    adam_weight_decay: float = 1e-6

    free_nats: float = 0.0
    kl_loss_scale: float = 1.0
    kl_balance: float = 0.8  # trains posterior faster than prior
    disclam: float = 0.95  # lambda for GAE
    # 1e-4 is the value used in dreamer for walker walk. Much larger in atari.
    policy_entropy_scale: float = 2e-3
    pcont: bool = True  # predict done signals
    pcont_loss_scale: float = 1.0  # weight of pcont loss
    actor_grad: str = "dynamics"  # which type of learning to use for the actor, "dynamics" to backprop through model
    freeze_actor_steps: int = 0  # don't train the actor for the first n steps to let the world model stabilize a bit
    value_target_network_update_freq: int = 100

    # TODO: figure out why "eval" doesn't work well, when that's what danijar always used.
    eval_exploration_mode: str = "explore"  # "eval" to use the mode of the distribution, "explore" to sample
    eval_workers: int = 16
    train_every: int = 5  # only used in the dreamerv2-style trainer. how many env steps per training step.

    def __attrs_post_init__(self) -> None:
        # These should only be changed with great care.
        assert self.obs_first is False
        assert self.time_limit_bootstrapping is False
