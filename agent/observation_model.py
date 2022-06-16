from typing import Dict
from typing import List
from typing import Optional

import attr
import numpy as np
import torch
from gym import spaces
from loguru import logger
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from agent.action_model import DictActionHead
from contrib.serialization import Serializable


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 200,
        output_dim: int = 200,
        num_layers: int = 2,
    ):
        super().__init__()
        self.act_fn = torch.nn.ReLU

        assert num_layers >= 1, "This MLP requires >= 1 layer"
        if num_layers == 1:
            self.net = torch.nn.Linear(input_dim, output_dim)
            return

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(self.act_fn())

            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        result = self.net(x)
        assert isinstance(result, Tensor)
        return result


@attr.s(auto_attribs=True, hash=True, collect_by_mro=True)
class ObservationModelParams(Serializable):
    encoder_output_dim: int = 256
    num_mlp_layers: int = 2


class ObservationModel(torch.nn.Module):
    """This model is designed to match the sb3 model closely.

    Not exactly sure how close it ended up being, but that was the intent.
    """

    vector_encoder_fc: Optional[torch.nn.Linear]
    action_encoder_fc: Optional[torch.nn.Linear]
    reward_encoder_fc: Optional[torch.nn.Linear]
    policy_net: Optional[MLP]
    value_net: Optional[MLP]

    def __init__(self, args, obs_space, action_space, params: ObservationModelParams):
        assert isinstance(obs_space, spaces.Dict), "We must have a dict space for this ObservationModel"
        self.obs_space = obs_space
        self.action_space = action_space

        super().__init__()

        self.params = params
        logger.info(f"ObservationModel with params {self.params}")

        # Concatenate image inputs, then concatenate the rest of the inputs
        _image_inputs = ("rgbd",)
        self.image_keys = sorted([x for x in obs_space.spaces.keys() if x in _image_inputs])
        encoder_output_dim = self.params.encoder_output_dim
        self.image_encoder = None
        if len(self.image_keys) > 0:
            img_size = obs_space["rgbd"].shape[1]
            img_channels = 4
            self.image_encoder = ImpalaConvNet(
                input_channels=img_channels, img_dim=img_size, out_dim=encoder_output_dim
            )

        # Collect remaining vectors
        self.vector_keys = sorted([x for x in obs_space.spaces.keys() if x not in _image_inputs])
        self.vector_encoder_fc = None
        if len(self.vector_keys) > 0:
            vector_dim = 0
            for key in self.vector_keys:
                space = obs_space[key]
                assert (
                    len(space.shape) == 1
                ), f"Expected non image observations to have one dim, got shape {space.shape} for {key}"
                vector_dim += space.shape[0]

            self.vector_encoder_fc = torch.nn.Linear(vector_dim, encoder_output_dim, bias=False)

            # Init to zero, keeps consistency across runs with/without proprioception
            self.vector_encoder_fc.weight.data.fill_(0.0)

        self.action_head = DictActionHead(action_space, args)
        self.policy_net = MLP(
            input_dim=encoder_output_dim,
            output_dim=self.action_head.num_inputs,
            num_layers=self.params.num_mlp_layers,
            hidden_dim=encoder_output_dim,
        )
        if isinstance(self.policy_net.net, torch.nn.Linear):
            mlp_init(self.policy_net.net, gain=0.01)
        else:
            mlp_init(self.policy_net.net[-1], gain=0.01)

        self.value_net = MLP(
            input_dim=encoder_output_dim,
            output_dim=1,
            num_layers=self.params.num_mlp_layers,
            hidden_dim=encoder_output_dim,
        )

    def forward(self, obs):
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
    def __init__(self, out_dim: int = 256, img_dim: int = 96, input_channels: int = 4):
        super().__init__()

        self.feat_convs = []
        self.resnet1 = []
        self.resnet2 = []

        for num_ch in [16, 32, 32]:
            feats_convs = []
            feats_convs.append(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=num_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            feats_convs.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.feat_convs.append(nn.Sequential(*feats_convs))

            input_channels = num_ch

            for i in range(2):
                resnet_block = []
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
                    self.resnet1.append(nn.Sequential(*resnet_block))
                else:
                    self.resnet2.append(nn.Sequential(*resnet_block))

        self.feat_convs = nn.ModuleList(self.feat_convs)
        self.resnet1 = nn.ModuleList(self.resnet1)
        self.resnet2 = nn.ModuleList(self.resnet2)

        if img_dim not in IMPALA_RES_OUT_LOOKUP:
            raise NotImplementedError()
        res_out_dim = IMPALA_RES_OUT_LOOKUP[img_dim]
        self.fc = nn.Linear(res_out_dim * res_out_dim * 32, out_dim)

    def forward(self, x: torch.Tensor):
        res_input = None
        for i, fconv in enumerate(self.feat_convs):
            x = fconv(x)
            res_input = x
            x = self.resnet1[i](x)
            x += res_input
            res_input = x
            x = self.resnet2[i](x)
            x += res_input

        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc(x))
        return x


def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def mlp_init(module, gain=np.sqrt(2), bias=0.0):
    """Helper to initialize a layer weight and bias."""
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, bias)
    return module
