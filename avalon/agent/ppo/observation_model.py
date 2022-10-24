from __future__ import annotations

from typing import Dict
from typing import Tuple

import torch
from gym import Space
from gym import spaces
from torch import Tensor
from torch import nn
from torch.nn import Module
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

    This is used in non-godot PPO.
    """

    # vector_encoder_fc: Optional[torch.nn.Linear]
    # action_encoder_fc: Optional[torch.nn.Linear]
    # reward_encoder_fc: Optional[torch.nn.Linear]
    # policy_net: Optional[MLP]
    # value_net: MLP

    def __init__(self, params: PPOParams, obs_space: Space, action_space: Space):
        assert isinstance(obs_space, spaces.Dict), "We must have a dict obs space for this ObservationModel"
        assert isinstance(action_space, spaces.Dict), "We must have a dict action space for this ObservationModel"
        self.obs_space: spaces.Dict = obs_space
        self.action_space = action_space

        super().__init__()

        self.params = params
        # logger.info(f"ObservationModel with params {self.params}")

        # Concatenate image inputs, then concatenate the rest of the inputs
        _image_inputs = ("rgbd",)
        self.image_keys = sorted([x for x in obs_space.spaces.keys() if x in _image_inputs])
        encoder_output_dim = self.params.model_params.encoder_output_dim
        self.image_encoder = None
        if len(self.image_keys) > 0:
            img_size = assert_not_none(obs_space["rgbd"].shape)[1]
            img_channels = assert_not_none(obs_space["rgbd"].shape)[0]
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
        )
        if isinstance(self.policy_net.net, torch.nn.Linear):
            mlp_init(self.policy_net.net, gain=0.01)
        else:
            mlp_init(self.policy_net.net[-1], gain=0.01)  # type: ignore

        self.value_net = MLP(
            input_dim=encoder_output_dim,
            output_dim=1,
            num_layers=self.params.model_params.num_mlp_layers,
            hidden_dim=encoder_output_dim,
        )

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
        return obs["rgbd"]

    def _get_vector_observation(self, obs: Dict[str, Tensor]) -> Tensor:
        """Convert the dict obs to a single vector input."""
        vector_parts = [obs[key] for key in self.vector_keys]
        vector_obs = torch.cat(vector_parts, dim=-1)
        return vector_obs


IMPALA_RES_OUT_LOOKUP = {64: 8, 84: 11, 96: 12}


class ImpalaConvNet(nn.Module):
    """This is used in godot PPO."""

    def __init__(self, out_dim: int = 256, img_dim: int = 96, input_channels: int = 4, num_base_channels: int = 16):
        super().__init__()

        feat_convs: list[Module] = []
        resnet1: list[Module] = []
        resnet2: list[Module] = []

        for num_ch in [num_base_channels, num_base_channels * 2, num_base_channels * 2]:
            sub_feats_convs: list[Module] = []
            sub_feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            sub_feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            feat_convs.append(nn.Sequential(*sub_feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block: list[Module] = []
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                resnet_block.append(nn.ReLU())
                resnet_block.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=num_ch,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )
                if i == 0:
                    resnet1.append(nn.Sequential(*resnet_block))
                else:
                    resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(feat_convs)
        self.resnet1 = nn.ModuleList(resnet1)
        self.resnet2 = nn.ModuleList(resnet2)

        if img_dim not in IMPALA_RES_OUT_LOOKUP:
            raise NotImplementedError()
        res_out_dim = IMPALA_RES_OUT_LOOKUP[img_dim]
        self.fc = nn.Linear(res_out_dim * res_out_dim * num_base_channels * 2, out_dim)

    def forward(self, x: torch.Tensor):
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            x = x + self.resnet1[i](x)
            x = x + self.resnet2[i](x)

        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return x
