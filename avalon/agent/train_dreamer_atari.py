import attr
from loguru import logger

from avalon.agent.common.params import DmcEnvironmentParams
from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.util import setup_new_process
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.train_dreamer_dmc import DreamerTrainer
from avalon.common.log_utils import configure_remote_logger

"""
- note: the time limit appears to be applied in agent env steps, not (action-repeated) env_steps.
- the exact model params aren't entirely clear for the atari continuous latents.
  - maksis ran: hidden 600, deter 600, stoch 32
    - does this line up with the paper curves?
  - and the dreamerv1 config default was: hidden 400 (?), deter 200, stoch 30
  - hidden -> just for mlps. deter is the hidden,state size of GRU.
"""


@attr.s(auto_attribs=True, frozen=True)
class DreamerAtariParams(DreamerParams):
    name: str = "torch"
    total_env_steps: int = 50_000_000
    pcont: bool = True
    prioritize_ends: bool = True
    worker_managers: int = 1
    num_workers: int = 1
    env_params: DmcEnvironmentParams = DmcEnvironmentParams(
        suite="atari", task="atlantis", action_repeat=4, time_limit=27000
    )
    actor_grad: str = "reinforce"
    free_nats: float = 0
    policy_entropy_scale: float = 1e-3
    model_lr: float = 2e-4
    actor_lr: float = 4e-5
    value_lr: float = 1e-4
    rssm_hidden_size: int = 600
    deter_size: int = 600
    discount: float = 0.999
    kl_loss_scale: float = 0.1
    pcont_loss_scale: float = 5
    prefill_steps: int = 50000
    pretrain_steps: int = 1
    freeze_actor_steps: int = 0
    train_every: int = 16
    train_gpu: int = 0
    inference_gpus: tuple[int, ...] = (0,)
    log_freq_hist: int = 2000
    log_freq_scalar: int = 25
    log_freq_media: int = 1000
    checkpoint_every: int = 10_000
    num_dataloader_workers: int = 1
    log_rollout_metrics_every: int = 1
    prefill_eps_per_dataloader: int = 1


def run(params: DreamerAtariParams) -> None:
    # params = attr.evolve(params, name=f"{params.name}_{params.env_params.suite}_{params.env_params.task}")
    trainer = DreamerTrainer(params)
    try:
        trainer.train()
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    try:
        default_params = DreamerAtariParams()
        default_params = parse_args(default_params)
        run(default_params)
    except Exception as e:
        logger.exception(e)
        raise
