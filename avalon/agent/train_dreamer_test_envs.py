import attr

from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.test_envs import TestEnvironmentParams
from avalon.agent.common.trainer import OffPolicyTrainer
from avalon.agent.common.util import setup_new_process
from avalon.agent.dreamer.params import DreamerParams
from avalon.common.log_utils import configure_remote_logger

FRAGMENT_LENGTH = 30


@attr.s(auto_attribs=True, frozen=True)
class DreamerTestEnvParams(DreamerParams):
    total_env_steps: int = 20000
    env_params: TestEnvironmentParams = TestEnvironmentParams(long_episode_length=500)
    rollout_model_update_freq: int = 25
    log_rollout_metrics_every: int = 1
    train_steps: int = 1000  # how often to load checkpoints
    value_target_network_update_freq: int = 25

    batch_size: int = 10
    max_fragment_len: int = FRAGMENT_LENGTH
    min_fragment_len: int = FRAGMENT_LENGTH
    pcont: bool = False
    prioritize_ends: bool = False
    actor_grad: str = "reinforce"
    rssm_hidden_size: int = 32
    deter_size: int = 32
    stoch_size: int = 16
    encoder_mlp_hidden_size: int = 32
    decoder_mlp_hidden_size: int = 32

    worker_managers: int = 1
    num_workers: int = 4
    train_gpu: int = 0
    inference_gpus: tuple[int] = (0,)

    log_freq_hist: int = 10
    log_freq_scalar: int = 1
    log_freq_media: int = 25


def run(params: DreamerParams) -> None:
    trainer = OffPolicyTrainer(params)
    try:
        trainer.start_train_rollouts()
        trainer.train()
    finally:
        trainer.shutdown(finish_wandb_quietly=True)
        del trainer.train_dataloader


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    default_params = DreamerTestEnvParams()
    default_params = parse_args(default_params)
    run(default_params)
