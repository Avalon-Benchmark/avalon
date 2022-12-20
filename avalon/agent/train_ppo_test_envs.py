import attr
from loguru import logger

from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.test_envs import TestEnvironmentParams
from avalon.agent.common.trainer import OnPolicyTrainer
from avalon.agent.common.util import setup_new_process
from avalon.agent.ppo.params import PPOParams
from avalon.common.log_utils import configure_remote_logger


@attr.s(auto_attribs=True, frozen=True)
class TestPPOParams(PPOParams):
    total_env_steps: int = 100_000
    num_steps: int = 50
    num_workers: int = 4
    batch_size: int = 50 * 4  # must equal num_steps * num_workers
    discount: float = 0.75
    env_params: TestEnvironmentParams = TestEnvironmentParams(long_episode_length=500000)
    log_freq_hist: int = 10
    log_freq_scalar: int = 1
    log_freq_media: int = 25
    entropy_coef: float = 1e-3  # with a small entropy penalty, random actions should stay random unless useful

    __test__ = False


def run(params: TestPPOParams) -> None:
    assert params.num_batches == 1
    assert params.batch_size == params.num_steps * params.num_workers

    trainer = OnPolicyTrainer(params)
    try:
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    try:
        default_params = TestPPOParams()
        default_params = parse_args(default_params)
        run(default_params)
    except Exception as e:
        logger.exception(e)
        raise
