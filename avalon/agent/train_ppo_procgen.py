from typing import Optional

import attr
import torch
from loguru import logger

from avalon.agent.common.models import ActivationFunction
from avalon.agent.common.params import ProcgenEnvironmentParams
from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.trainer import OnPolicyTrainer
from avalon.agent.common.util import setup_new_process
from avalon.agent.ppo.params import ObservationModelParams
from avalon.agent.ppo.params import PPOParams
from avalon.common.log_utils import configure_remote_logger

NUM_WORKERS = 64
NUM_STEPS = 256


@attr.s(auto_attribs=True, frozen=True)
class ProcgenPPOParams(PPOParams):
    project: str = "zack_zack_ppo"
    total_env_steps: int = 25_000_000
    num_steps: int = NUM_STEPS
    num_workers: int = NUM_WORKERS
    batch_size: int = int(NUM_STEPS * NUM_WORKERS / 8)
    ppo_epochs: int = 3
    discount: float = 0.999
    lam: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = 0.2
    baselines_style_vf_loss: bool = True
    lr: float = 5e-4
    clip_grad_norm: float = 0.5
    time_limit_bootstrapping: bool = False
    env_params: ProcgenEnvironmentParams = ProcgenEnvironmentParams()
    log_freq_hist: int = 500
    log_freq_scalar: int = 50
    log_freq_media: int = 500
    checkpoint_every: int = 2500
    model_params: ObservationModelParams = ObservationModelParams(
        mlp_activation_fn=ActivationFunction.ACTIVATION_TANH,  # not used
        num_mlp_layers=1,  # no MLP, just adapting shape from 256 to output dim
        encoder_output_dim=256,
        num_cnn_base_channels=16,
    )

    eval_exploration_mode: str = "eval"  # TODO: try explore here also
    center_observations: bool = False
    center_and_clamp_discrete_logits: bool = False


def run(params: ProcgenPPOParams) -> None:
    trainer = OnPolicyTrainer(params)
    try:
        if params.is_training:
            trainer.train()
            trainer.shutdown()
            if trainer.train_storage:
                trainer.train_storage.reset()
            torch.cuda.empty_cache()  # just for seeing what's going on
        if params.is_testing:
            pass
    finally:
        trainer.shutdown()


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    try:
        default_params = ProcgenPPOParams()
        default_params = parse_args(default_params)
        run(default_params)
    except Exception as e:
        logger.exception(e)
        raise
