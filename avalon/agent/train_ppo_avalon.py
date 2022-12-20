import attr
import torch
from loguru import logger

from avalon.agent.common.params import ClippedNormalMode
from avalon.agent.common.parse_args import parse_args
from avalon.agent.common.trainer import OnPolicyTrainer
from avalon.agent.common.util import setup_new_process
from avalon.agent.godot.godot_eval import test
from avalon.agent.godot.godot_gym import GodotEnvironmentParams
from avalon.agent.godot.godot_gym import TrainingProtocolChoice
from avalon.agent.ppo.params import PPOParams
from avalon.common.log_utils import configure_remote_logger


@attr.s(auto_attribs=True, frozen=True)
class AvalonPPOParams(PPOParams):
    total_env_steps: int = 50_000_000
    num_steps: int = 200
    num_workers: int = 16
    batch_size: int = 200 * 16  # must equal num_steps * num_workers
    ppo_epochs: int = 2
    discount: float = 0.99
    lam: float = 0.83
    value_loss_coef: float = 1
    entropy_coef: float = 1.5e-4
    clip_range: float = 0.03
    lr: float = 2.5e-4
    clip_grad_norm: float = 0.5
    env_params: GodotEnvironmentParams = GodotEnvironmentParams(
        # if 100% success, difficulty in N steps will be N / (avg_ep_len * num_workers * num_tasks) * diff_update
        # eg 10M steps / (200 steps * 16 workers * 16 tasks) * 3e-4 = ~.06 max achievable difficulty
        task_difficulty_update=3e-4,
        meta_difficulty_update=3e-5,
        is_meta_curriculum_used=False,
        energy_cost_coefficient=1e-8,
        training_protocol=TrainingProtocolChoice.MULTI_TASK_BASIC,
        test_episodes_per_task=101,
        fixed_world_max_difficulty=0.5,
    )
    log_freq_hist: int = 500
    log_freq_scalar: int = 50
    log_freq_media: int = 500
    checkpoint_every: int = 2500

    eval_exploration_mode: str = "eval"  # TODO: try explore here also

    # TODO: should try converting this to the TruncatedNormal one
    policy_normal_init_std: float = 1
    policy_normal_min_std: float = 0.01
    clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.NO_CLIPPING
    center_and_clamp_discrete_logits = True


def run(params: AvalonPPOParams) -> None:
    assert params.num_batches == 1
    assert params.batch_size == params.num_steps * params.num_workers

    trainer = OnPolicyTrainer(params)
    try:
        if params.is_training:
            trainer.train()
            trainer.shutdown()
            if trainer.train_storage:
                trainer.train_storage.reset()
            torch.cuda.empty_cache()  # just for seeing what's going on
        if params.is_testing:
            test(trainer.params, trainer.algorithm, log=True)

    finally:
        trainer.shutdown()


if __name__ == "__main__":
    configure_remote_logger(level="INFO")
    setup_new_process()
    try:
        default_params = AvalonPPOParams()
        default_params = parse_args(default_params)
        run(default_params)
    except Exception as e:
        logger.exception(e)
        raise
