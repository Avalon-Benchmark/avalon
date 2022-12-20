import time

import attr
from loguru import logger

from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.trainer import OffPolicyTrainer
from avalon.agent.common.util import setup_new_process
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.godot.godot_eval import test
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.common.error_utils import capture_exception
from avalon.common.log_utils import configure_remote_logger
from avalon.datagen.godot_env.interactive_godot_process import GODOT_ERROR_LOG_PATH

FRAGMENT_LENGTH = 30


@attr.s(auto_attribs=True, frozen=True)
class DreamerGodotParams(DreamerParams):
    total_env_steps: int = 50_000_000
    replay_buffer_size_timesteps: int = 1_000_000
    pcont: bool = False
    prioritize_ends: bool = False
    worker_managers: int = 1
    num_workers: int = 16
    discount: float = 0.99
    model_lr: float = 1e-4
    value_lr: float = 1e-4
    actor_lr: float = 1e-5
    clip_grad_norm: float = 100
    env_params: GodotEnvironmentParams = GodotEnvironmentParams(
        task_difficulty_update=6e-4,
        energy_cost_coefficient=1e-8,
        training_protocol=TrainingProtocolChoice.MULTI_TASK_BASIC,
        fixed_world_max_difficulty=0.5,
        # Make the "eating" always be part of the final fragment, so the end is at least theoretically predictable.
        # Will make episodes where the reward is gotten in < 2 frames be skipped, though.
        num_frames_alive_after_food_is_gone=FRAGMENT_LENGTH - 2,
        gpu_id=1,
        test_episodes_per_task=51,
    )
    batch_size: int = 116
    log_freq_hist: int = 2000
    log_freq_scalar: int = 25
    log_freq_media: int = 500
    checkpoint_every: int = 20000
    free_nats: float = 0
    kl_loss_scale: float = 30
    policy_entropy_scale: float = 2e-3
    pcont_loss_scale: float = 10
    reward_loss_scale: float = 10
    kl_balance: float = 0.8
    disclam: float = 0.95
    freeze_actor_steps: int = 500
    min_fragment_len: int = FRAGMENT_LENGTH
    max_fragment_len: int = FRAGMENT_LENGTH
    train_gpu: int = 0
    inference_gpus: tuple[int, ...] = (1,)
    actor_grad: str = "reinforce"
    log_rollout_metrics_every: int = 100
    is_training: bool = True
    is_testing: bool = False
    num_dataloader_workers: int = 1
    rssm_hidden_size: int = 600
    deter_size: int = 800
    stoch_size: int = 32
    center_and_clamp_discrete_logits = True


def run(params: DreamerParams):
    trainer = OffPolicyTrainer(params)
    try:
        if params.is_training:
            logger.info(f"starting training at {time.time()}")
            trainer.start_train_rollouts()
            trainer.train()
            trainer.shutdown_train_rollouts()
            logger.info(f"finished training at {time.time()}")
        trainer.shutdown()
        del trainer.train_dataloader
        if params.is_testing:
            logger.info(f"starting eval at {time.time()}")
            test(trainer.params, trainer.algorithm, log=True)
            logger.info(f"finished eval at {time.time()}")
    except Exception as e:
        if not params.debug:
            capture_exception(e)
        raise
    finally:
        logger.info("shutting down")
        if not params.debug:
            try:
                trainer.wandb_run.save(f"{GODOT_ERROR_LOG_PATH}/*")
            except Exception as e:
                logger.warning(f"wandb godot tarball save failed: {e}")
        trainer.shutdown(finish_wandb_quietly=True)


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    try:
        default_params = DreamerGodotParams()
        default_params = parse_args(default_params)
        run(default_params)
    except Exception as e:
        logger.exception(e)
        raise
