from __future__ import annotations

from typing import Dict
from typing import Tuple

import torch
from gym import Space
from gym import spaces
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from avalon.agent.common.action_model import DictActionDist
from avalon.agent.common.action_model import DictActionHead
from avalon.agent.common.models import MLP
from avalon.agent.common.types import ObservationBatch
from avalon.agent.ppo.model import PPOModel
from avalon.agent.ppo.model import mlp_init
from avalon.agent.ppo.params import PPOParams
from avalon.common.type_utils import assert_not_none


class ObservationModel(PPOModel):
    """This model is designed to match the sb3 model closely.

    Not exactly sure how close it ended up being, but that was the intent.
    """

    def __init__(self, params: PPOParams, obs_space: Space, action_space: Space) -> None:
        assert isinstance(obs_space, spaces.Dict), "We must have a dict obs space for this ObservationModel"
        assert isinstance(action_space, spaces.Dict), "We must have a dict action space for this ObservationModel"
        self.obs_space: spaces.Dict = obs_space
        self.action_space = action_space

        super().__init__()

        self.params = params
        # logger.info(f"ObservationModel with params {self.params}")

        # Concatenate image inputs, then concatenate the rest of the inputs
        _image_inputs = ("rgbd", "rgb")
        self.image_keys = sorted([x for x in obs_space.spaces.keys() if x in _image_inputs])
        encoder_output_dim = self.params.model_params.encoder_output_dim
        self.image_encoder = None
        if len(self.image_keys) > 0:
            assert len(self.image_keys) == 1, "concatenatenation of multiple images is not yet supported"
            image_key = self.image_keys[0]
            img_size = assert_not_none(obs_space[image_key].shape)[1]
            img_channels = assert_not_none(obs_space[image_key].shape)[0]
            self.image_encoder = ImpalaConvNet(
                input_channels=img_channels,
                img_dim=img_size,
                out_dim=encoder_output_dim,
                num_base_channels=self.params.model_params.num_cnn_base_channels,
            )

        # Collect remaining vectors
        self.vector_keys = sorted([x for x in obs_space.spaces.keys() if x not in _image_inputs])
        self.vector_encoder_fc = None
        if len(self.vector_keys) > 0:
            vector_dim = 0
            for key in self.vector_keys:
                space = obs_space[key]
                assert (
                    len(assert_not_none(space.shape)) == 1
                ), f"Expected non image observations to have one dim, got shape {space.shape} for {key}"
                vector_dim += assert_not_none(space.shape)[0]

            self.vector_encoder_fc = torch.nn.Linear(vector_dim, encoder_output_dim, bias=False)

            # Init to zero, keeps consistency across runs with/without proprioception
            self.vector_encoder_fc.weight.data.fill_(0.0)

        self.action_head = DictActionHead(action_space, params)
        self.policy_net = MLP(
            input_dim=encoder_output_dim,
            output_dim=self.action_head.num_inputs,
            num_layers=self.params.model_params.num_mlp_layers,
            hidden_dim=encoder_output_dim,
            activation_function=self.params.model_params.mlp_activation_fn,
        )
        # Note: this init matches baselines:ppo2 for the case of num_mlp_layers==1, but not otherwise.
        if self.params.model_params.num_mlp_layers == 1:
            mlp_init(self.policy_net.net, gain=0.01, bias=0)
        elif self.params.model_params.num_mlp_layers > 1:
            # I believe this is in ppo2. Haven't tried it yet though, been just using the default.
            # for layer in self.policy_net.net[:-1]:
            #     mlp_init(layer, gain=math.sqrt(2))
            mlp_init(self.policy_net.net[-1], gain=0.01, bias=0)  # type: ignore

        self.value_net = MLP(
            input_dim=encoder_output_dim,
            output_dim=1,
            num_layers=self.params.model_params.num_mlp_layers,
            hidden_dim=encoder_output_dim,
            activation_function=self.params.model_params.mlp_activation_fn,
        )
        # Note: this init matches baselines:ppo2 for the case of num_mlp_layers==1, but not otherwise.
        if self.params.model_params.num_mlp_layers == 1:
            mlp_init(self.value_net.net, gain=1, bias=0)

    def forward(self, obs: ObservationBatch) -> Tuple[Tensor, DictActionDist]:
        encodings = []
        if self.image_encoder:
            image_obs = self._get_image_observation(obs)
            img_encoding = self.image_encoder(image_obs)
            encodings.append(img_encoding)

        if self.vector_encoder_fc:
            vector_obs = self._get_vector_observation(obs)
            encodings.append(self.vector_encoder_fc(vector_obs))

        encoding = torch.stack(encodings, dim=0).sum(dim=0)
        batch_size = encoding.shape[0]
        value = self.value_net(encoding).reshape((batch_size,))
        action_logits = self.policy_net(encoding)
        action_dist = self.action_head(action_logits)
        return value, action_dist

    def _get_image_observation(self, obs: Dict[str, Tensor]) -> Tensor:
        return obs[self.image_keys[0]]

    def _get_vector_observation(self, obs: Dict[str, Tensor]) -> Tensor:
        """Convert the dict obs to a single vector input."""
        vector_parts = [obs[key] for key in self.vector_keys]
        vector_obs = torch.cat(vector_parts, dim=-1)
        return vector_obs


IMPALA_RES_OUT_LOOKUP = {64: 8, 84: 11, 96: 12}


class ResnetBlock(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


def impala_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ResnetBlock(out_channels),
        ResnetBlock(out_channels),
    )


class ImpalaConvNet(nn.Module):
    """This has been matched carefully to the one in dreamerv2, and also openai/baselines."""

    def __init__(self, out_dim, img_dim, input_channels, num_base_channels: int = 16) -> None:
        super().__init__()
        blocks = []
        in_channels = input_channels
        for out_channels in [num_base_channels, num_base_channels * 2, num_base_channels * 2]:
            blocks.append(impala_block(in_channels, out_channels))
            in_channels = out_channels

        self.conv = nn.Sequential(*blocks)

        if img_dim not in IMPALA_RES_OUT_LOOKUP:
            raise NotImplementedError()
        res_out_dim = IMPALA_RES_OUT_LOOKUP[img_dim]
        # Note: (intentionally) leaving this with default init
        self.fc = nn.Linear(res_out_dim * res_out_dim * num_base_channels * 2, out_dim)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return x
