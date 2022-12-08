# mypy: ignore-errors
# TODO: type this file
import math
from typing import Optional
from typing import Tuple

import gym
import torch
from einops import rearrange
from loguru import logger
from torch import Tensor
from torch import nn
from torch.distributions import Bernoulli
from torch.distributions import Distribution
from torch.distributions import Independent
from torch.distributions import Normal
from torch.nn import functional as F
from tree import map_structure

from avalon.agent.common.action_model import DictActionHead
from avalon.agent.common.models import ACTIVATION_MODULE_LOOKUP
from avalon.agent.common.models import MLP
from avalon.agent.common.models import ActivationFunction
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import LatentBatch
from avalon.agent.common.types import ObservationBatch
from avalon.agent.dreamer.tools import static_scan


def init_weights(m: torch.nn.Module, gain: float = 1.0) -> None:
    # It appears that tf uses glorot init with gain 1 by default for all weights, and zero for all biases
    # at least for these layers
    if isinstance(m, nn.Linear):
        # This is the default used in a tensorflow keras Dense layer, with gain=1
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        m.bias.data.fill_(0.0)


class GRUCell(nn.Module):
    # This is the custom cell Danijar wrote in his dreamerv2 repo.
    # Default initializer values are the values used for all configs.
    def __init__(
        self, input_dim: int, state_size: int, norm: bool = True, act=torch.tanh, update_bias: float = -1
    ) -> None:
        """
        - input_dim: the size of the input vector
        - state_size: the size of the reset, cand, update, state, and output vectors
        """
        super().__init__()
        self._state_size = state_size
        self._act = act
        self._update_bias = update_bias
        self._layer = torch.nn.Linear(
            in_features=input_dim + state_size, out_features=3 * state_size, bias=norm is not None
        )
        if norm:
            self._norm = torch.nn.LayerNorm(normalized_shape=3 * state_size)

        self.apply(init_weights)

    def forward(self, inputs, state):
        # inputs should be hidden_size, state should be deter_size
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.chunk(parts, 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output


class RSSM(nn.Module):
    def __init__(self, actdim, embed_size=1024, stoch=30, deter=200, hidden=200, act=F.elu) -> None:
        super().__init__()
        self._activation = act
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self.min_std = 0.1  # this is the value used in the dreamerv2 configs

        # These two layers are common/shared for the prior and posterior.
        self.img1 = nn.Linear(actdim + stoch, self._hidden_size)
        self._cell = GRUCell(input_dim=self._hidden_size, state_size=self._deter_size, norm=True)

        # Prior model
        self.img2 = nn.Linear(self._deter_size, self._hidden_size)
        self.img3 = nn.Linear(self._hidden_size, 2 * self._stoch_size)

        # Posterior model
        self._embed_size = embed_size
        self.obs1 = nn.Linear(self._deter_size + self._embed_size, self._hidden_size)
        self.obs2 = nn.Linear(self._hidden_size, 2 * self._stoch_size)

        self.apply(init_weights)

    def initial(self, batch_size: int, device) -> LatentBatch:
        """This is the initial latent state."""
        return dict(
            mean=torch.zeros([batch_size, self._stoch_size], device=device),
            std=torch.zeros([batch_size, self._stoch_size], device=device),
            stoch=torch.zeros([batch_size, self._stoch_size], device=device),
            deter=torch.zeros([batch_size, self._deter_size], device=device),
        )

    def observe(self, embed: Tensor, action: ActionBatch, state=None) -> Tuple[LatentBatch, LatentBatch]:
        """Generates state estimations given a sequence of observations and actions.
        Only used in training?"""

        if state is None:
            state = self.initial(batch_size=embed.shape[0], device=embed.device)
        # these are moving the time axis to the front, for the static_scan
        embed = rearrange(embed, "b t ... -> t b ...")
        action = {k: rearrange(v, "b t ... -> t b ...") for k, v in action.items()}
        # here the nest structure is a tuple of tensors, and a tuple of dicts of tensors
        # (state, state) is just to give the proper structure for the output
        post, prior = static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs), (action, embed), (state, state)
        )
        # Moving the batch axis back in front
        post = {k: v.permute(1, 0, 2) for k, v in post.items()}
        prior = {k: v.permute(1, 0, 2) for k, v in prior.items()}
        return post, prior

    def imagine(self, action: ActionBatch, state: LatentBatch) -> LatentBatch:
        # TODO: only used for the imagination viz. figure out how to share with train() imagine.
        # or probably just move this into the viz logic
        # Takes a list of actions and an initial state and rolls it out in imagination.
        assert isinstance(state, dict), state
        prior = static_scan(self.img_step, action, state)
        prior = {k: v.permute(1, 0, 2) for k, v in prior.items()}
        return prior

    def get_feat(self, state: LatentBatch) -> Tensor:
        """Combine stoch and deter state into a single vector."""
        feat = torch.cat([state["stoch"], state["deter"]], -1)
        return feat

    def get_dist(self, state: LatentBatch) -> torch.distributions.Distribution:
        dist = Independent(Normal(state["mean"], state["std"]), 1)
        return dist

    def obs_step(
        self, prev_state: LatentBatch, prev_action: ActionBatch, embed: Tensor
    ) -> Tuple[LatentBatch, LatentBatch]:
        """Observation is the process of taking s_{t-1}, a_{t-1}, and o_{t} and generating s_{t}.
        In other words, estimating the current (unobserved) state given the current observation.

        This is what the posterior model does."""
        prior = self.img_step(prev_state, prev_action)

        # Compute the posterior
        # embed is the observation embedding
        x = torch.cat([prior["deter"], embed], -1)
        x = self._activation(self.obs1(x))
        x = self.obs2(x)
        mean, std = torch.chunk(x, 2, -1)

        std = 2 * torch.sigmoid(std / 2) + self.min_std
        stoch_dist = self.get_dist({"mean": mean, "std": std})

        # TODO: this should be the mode if we're doing eval. Does eval ever use this codepath?
        stoch = stoch_dist.rsample()
        post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}
        return post, prior

    def img_step(self, prev_state: LatentBatch, prev_action: ActionBatch) -> LatentBatch:
        """Imagination is the process of taking s_{t-1} and a_{t-1} and generating s_{t}.
        In other words, imagining what the next state will be given the previous state and action.

        This is what the prior model does."""
        # Compute the RNN (this is the deterministic part)
        # Flatten actions out into a vector per batch element. Batch must always only have 1 axis here.
        prev_action = torch.cat([x.reshape([x.shape[0], -1]) for x in prev_action.values()], dim=-1)
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self._activation(self.img1(x))
        deter = self._cell(x, prev_state["deter"])

        # Compute the prior
        y = self._activation(self.img2(deter))
        y = self.img3(y)
        mean, std = torch.chunk(y, 2, -1)
        std = 2 * torch.sigmoid(std / 2) + self.min_std
        stoch = self.get_dist({"mean": mean, "std": std}).rsample()
        prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter}
        return prior

    def kl_loss(
        self, post: LatentBatch, prior: LatentBatch, balance: float = 0.8, free: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        kld = torch.distributions.kl_divergence
        sg = lambda x: map_structure(torch.detach, x)
        if math.isclose(balance, 0.5):
            value = kld(self.get_dist(post), self.get_dist(prior))
            # Shape of value is (batch, timesteps)
            loss = torch.clamp(value, min=free).mean()
        else:
            value_lhs = value = kld(self.get_dist(post), self.get_dist(sg(prior)))
            value_rhs = kld(self.get_dist(sg(post)), self.get_dist(prior))
            loss_lhs = torch.clamp(value_lhs.mean(), min=free)
            loss_rhs = torch.clamp(value_rhs.mean(), min=free)
            loss = (1 - balance) * loss_lhs + balance * loss_rhs
        return loss, value


def is_image_space(x: gym.spaces.Space) -> bool:
    return isinstance(x, gym.spaces.Box) and len(x.shape) == 3


def is_vector_space(x: gym.spaces.Space) -> bool:
    return isinstance(x, gym.spaces.Box) and len(x.shape) == 1


class HybridEncoder(nn.Module):
    """Takes a dict obs space composed of image and vector Box spaces, and encodes it all to a single latent vector."""

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        mlp_hidden_dim=400,
        activation_function: ActivationFunction = ActivationFunction.ACTIVATION_ELU,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, gym.spaces.Dict)
        self.obs_space = obs_space
        # Out dim is dynamic - just the native output size of the MLP + conv, concatted together.
        self.out_dim = 0
        activation_module = ACTIVATION_MODULE_LOOKUP[activation_function.value]

        self.image_keys = [k for k, v in obs_space.spaces.items() if is_image_space(v)]
        self.vector_keys = [k for k, v in obs_space.spaces.items() if is_vector_space(v)]
        # TODO: handle scalar spaces! Either here or coerce them to vectors in the wrapper.
        scalar_spaces = [k for k, v in obs_space.spaces.items() if v.shape == ()]
        if len(scalar_spaces) > 0:
            logger.warning("reminder: ignoring scalar spaces!")
        assert len(obs_space) == len(self.image_keys) + len(self.vector_keys) + len(scalar_spaces)

        if self.image_keys:
            # We expect images to be (c, h, w)
            img_size = obs_space.spaces[self.image_keys[0]].shape[1]
            self.img_channels = 0
            for key in self.image_keys:
                space = obs_space.spaces[key]
                assert space.shape[1] == img_size and space.shape[2] == img_size
                self.img_channels += space.shape[0]

            self.image_encoder = ConvEncoder(self.img_channels, img_size)
            self.cnn_dim = self.image_encoder.output_dim
            self.out_dim += self.cnn_dim
            assert 100 < self.cnn_dim < 2000, "just a good heuristic"

        if self.vector_keys:
            self.vector_dim = 0
            for key in self.vector_keys:
                space = obs_space.spaces[key]
                self.vector_dim += space.shape[0]

            self.vector_encoder_fc = MLP(
                input_dim=self.vector_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=mlp_hidden_dim,
                num_layers=4,
                activation_function=activation_function,
            )
            self.vector_activation = activation_module()
            self.out_dim += mlp_hidden_dim

        self.apply(init_weights)

    def forward(self, obs: ObservationBatch) -> Tensor:
        encodings = []
        if self.image_keys:
            image_parts = [obs[key] for key in self.image_keys]
            # Ensure that we applied the post-transform properly everywhere
            assert all([x.dtype == torch.float32 for x in image_parts])
            image_obs = torch.cat(image_parts, dim=-3)
            img_encoding = self.image_encoder(image_obs)
            assert img_encoding.shape[-1] == self.cnn_dim
            encodings.append(img_encoding)

        if self.vector_keys:
            vector_parts = [obs[key] for key in self.vector_keys]
            vector_obs = torch.cat(vector_parts, dim=-1)
            assert vector_obs.shape[-1] == self.vector_dim
            x = self.vector_encoder_fc(vector_obs)
            x = self.vector_activation(x)
            encodings.append(x)

        encoding = torch.cat(encodings, dim=-1)
        assert encoding.shape[-1] == self.out_dim
        return encoding


class HybridDecoder(nn.Module):
    """The inverse of the HybridEncoder"""

    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        latent_dim: int,
        skip_keys: Tuple[str] = (),
        mlp_hidden_dim=400,
        activation_function: ActivationFunction = ActivationFunction.ACTIVATION_ELU,
    ) -> None:
        super().__init__()
        assert isinstance(obs_space, gym.spaces.Dict)
        self.obs_space = obs_space

        self.image_keys = [k for k, v in obs_space.spaces.items() if is_image_space(v) if k not in skip_keys]
        self.vector_keys = [k for k, v in obs_space.spaces.items() if is_vector_space(v) if k not in skip_keys]
        # TODO: handle scalar spaces! Either here or coerce them to vectors in the wrapper.
        scalar_spaces = [k for k, v in obs_space.spaces.items() if v.shape == ()]
        skipped_keys = [k for k in obs_space.spaces.keys() if k in skip_keys]
        assert len(obs_space) == len(self.image_keys) + len(self.vector_keys) + len(scalar_spaces) + len(skipped_keys)

        if self.image_keys:
            # We expect images to be (c, h, w)
            img_size = obs_space.spaces[self.image_keys[0]].shape[1]
            self.img_channels = 0
            for key in self.image_keys:
                space = obs_space.spaces[key]
                assert space.shape[1] == img_size and space.shape[2] == img_size
                self.img_channels += space.shape[0]
            # The decoder already has a linear to adapt the input size
            self.image_decoder = ConvDecoder(input_dim=latent_dim, out_channels=self.img_channels, res=img_size)

        if self.vector_keys:
            self.vector_dim = 0
            for key in self.vector_keys:
                space = obs_space.spaces[key]
                self.vector_dim += space.shape[0]
            self.vector_decoder_fc = MLP(
                input_dim=latent_dim,
                hidden_dim=mlp_hidden_dim,
                output_dim=self.vector_dim,
                num_layers=5,  # danijar has 4 hidden and 1 output layer (per space, which we have merged)
                activation_function=activation_function,
            )

        self.apply(init_weights)

    def forward(self, latent: Tensor) -> ObservationBatch:
        # We're only going to handle latents of shape (b, t, latent_dim) here
        assert len(latent.shape) == 3
        batch_size, timesteps = latent.shape[:2]
        out = {}
        if self.image_keys:
            decoded_img = self.image_decoder(latent)
            # assert decoded_img.shape == (batch_size, timesteps, self.img_channels, 64, 64)
            start_channel = 0
            for key in self.image_keys:
                channels = self.obs_space.spaces[key].shape[0]
                out[key] = decoded_img[:, :, start_channel : start_channel + channels]
                start_channel += channels

        if self.vector_keys:
            decoded_vec = self.vector_decoder_fc(latent)
            assert decoded_vec.shape == (batch_size, timesteps, self.vector_dim)
            start_dim = 0
            for key in self.vector_keys:
                dims = self.obs_space.spaces[key].shape[0]
                out[key] = decoded_vec[:, :, start_dim : start_dim + dims]
                start_dim += dims

        return out


def compute_conv_output_res(
    input_res: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    return math.floor(((input_res + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)


def compute_conv_transpose_output_res(
    input_res: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, dilation: int = 1
) -> int:
    return (input_res - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1


class ConvEncoder(nn.Module):
    def __init__(self, input_channels: int = 3, input_res: int = 96) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_res = input_res

        # for 64x64, we will output a shape of 2x2x384 (or whatever ordering we use here..)
        # for 96x96, we will output the same, requiring 1 extra layer. 2x2x384
        kernels = {
            64: [4, 4, 4, 4],
            96: [4, 4, 4, 4, 2],
        }[input_res]
        depth = {
            64: 48,
            96: 24,
        }[input_res]
        layers = []
        in_channels = input_channels
        current_res = input_res
        for i, kernel in enumerate(kernels):
            out_channels = 2**i * depth
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=2, padding=0
            )
            in_channels = out_channels
            current_res = compute_conv_output_res(current_res, kernel, stride=2, padding=0)
            layers.append(conv)
            layers.append(nn.ELU())

        self.conv = nn.Sequential(*layers)

        self.output_dim = in_channels * (current_res * current_res)
        self.apply(init_weights)

    def forward(self, obs: Tensor) -> Tensor:
        # This can get called with shape (b, c, h, w) or (b, t, c, h, w)
        x = obs
        # Convert (b, t, c, h, w) to (b, c, h, w) if starting with the former
        x = torch.reshape(x, (-1,) + x.shape[-3:])
        assert x.shape[1:] == (self.input_channels, self.input_res, self.input_res)
        x = self.conv(x)
        # Reshape back to the original b/t shape, flatten out the rest
        shape = obs.shape[:-3] + (self.output_dim,)
        x = torch.reshape(x, shape)
        return x


class ConvDecoder(nn.Module):
    def __init__(self, input_dim: int, out_channels: int, depth: int = 48, res: int = 96) -> None:
        super().__init__()
        self._depth = depth
        self.res = res
        self._shape = (out_channels, self.res, self.res)
        self.out_channels = out_channels

        self.embedding_size = 32 * self._depth
        self.fc1 = nn.Linear(input_dim, self.embedding_size)

        layers = []
        kernels = {
            64: [5, 5, 6, 6],
            96: [4, 4, 4, 5, 4],
        }[res]
        current_res = 1
        current_in_channels = self.embedding_size
        for i, kernel in enumerate(kernels):
            current_out_channels = 2 ** (len(kernels) - i - 2) * depth
            if i == len(kernels) - 1:
                conv = nn.ConvTranspose2d(current_in_channels, out_channels, kernel_size=kernel, stride=2)
                layers.append(conv)
            else:
                conv = nn.ConvTranspose2d(current_in_channels, current_out_channels, kernel_size=kernel, stride=2)
                layers.append(conv)
                layers.append(nn.ELU())
            current_in_channels = current_out_channels
            current_res = compute_conv_transpose_output_res(current_res, kernel, stride=2)

        assert current_res == res, (current_res, res)
        self.conv = nn.Sequential(*layers)

        self.apply(init_weights)

    def forward(self, features: Tensor) -> Tensor:
        # c is stoch + deter
        b, t, c = features.shape
        x = self.fc1(features)  # this is not followed by a nonlinearity
        x = rearrange(x, "b t c -> (b t) c 1 1", b=b, t=t, c=self.embedding_size)
        x = self.conv(x)
        dist_mean = rearrange(x, "(b t) c h w-> b t c h w", b=b, t=t, h=self.res, w=self.res, c=self.out_channels)
        # shape is [50, 50, self.res, self.res, 3]
        return dist_mean


class DenseDecoder(nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, ...],
        layers: int,
        in_dim: int,
        units: int,
        dist: str,
        action_head: Optional[DictActionHead] = None,
        activation_function: ActivationFunction = ActivationFunction.ACTIVATION_ELU,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.dist = dist

        # Dreamerv2 has an additional Linear layer in the `DistLayer`.
        # So they actually have `layers` + 1 Linear layers.
        self.mlp = MLP(
            in_dim,
            hidden_dim=units,
            output_dim=math.prod(output_shape),
            num_layers=layers + 1,
            activation_function=activation_function,
        )
        if self.dist == "action_head":
            assert action_head
            self.action_head = action_head

        self.apply(init_weights)

    def forward(self, x: Tensor, raw_feats=False) -> Distribution:
        batch_shape = x.shape[:-1]
        x = self.mlp(x)
        x = x.view(batch_shape + self.output_shape)
        if raw_feats:
            return x
        if self.dist == "normal":
            # The std 1 here is kinda weird. Why not let it predict its own std?
            # Oh, this is essentially just a MSE loss with this configuration.
            return Independent(Normal(x, 1), len(self.output_shape))
        elif self.dist == "binary":
            # x is just the batch shape, so this will properly construct a dist with proper batch shape and scalar event shape
            return Bernoulli(logits=x)
        elif self.dist == "action_head":
            dist = self.action_head(x)
            return dist
        raise NotImplementedError(self.dist)
