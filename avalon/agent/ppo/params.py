from typing import Optional

import attr
import torch

from avalon.agent.common.models import ActivationFunction
from avalon.agent.common.params import Params


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ObservationModelParams:
    encoder_output_dim: int = 256
    num_mlp_layers: int = 2
    num_cnn_base_channels: int = 16
    mlp_activation_fn: ActivationFunction = ActivationFunction.ACTIVATION_RELU


@attr.s(auto_attribs=True, frozen=True)
class OnPolicyParams(Params):
    num_steps: int = 256

    @property
    def inference_device(self):
        return torch.device(f"cuda:{self.inference_gpus[0]}")


@attr.s(auto_attribs=True, frozen=True)
class PPOParams(OnPolicyParams):
    # Default overrides
    project: str = "avalon_ppo"
    # don't routinely change these!
    obs_first: bool = True
    time_limit_bootstrapping: bool = True
    num_steps: int = 256
    num_workers: int = 8
    batch_size: int = 256 * 8

    # PPO-specific stuff
    model: str = "godot"
    discount: float = 0.99
    lam: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    value_loss_coef: float = 0.5
    baselines_style_vf_loss: bool = False
    entropy_mode: str = "regularized"  # can be "max" or "regularized", same meaning as in `garage` VPG
    entropy_coef: float = 0.0  # entropy coef for use in either entropy mode. set to 0 to disable both.

    ppo_epochs: int = 10
    lr: float = 3e-4
    clip_grad_norm: float = 0.5

    normal_std_from_model: bool = True
    # policy_normal_init_std: float = 1
    # policy_normal_min_std: float = 0.01
    # clipped_normal_mode: ClippedNormalMode = ClippedNormalMode.TRUNCATED_NORMAL

    model_params: ObservationModelParams = ObservationModelParams()

    def __attrs_post_init__(self) -> None:
        assert len(self.inference_gpus) == 1, "we can only use one inference gpu"

        # These should only be changed with great care.
        assert self.env_params.action_repeat == 1
        assert self.obs_first is True

    @property
    def num_batches(self) -> int:
        assert (
            self.num_steps * self.num_workers % self.batch_size == 0
        ), f"batch size {self.batch_size} does not divide cleanly into steps ({self.num_steps}) * workers ({self.num_workers})"
        return int(self.num_steps * self.num_workers / self.batch_size)
