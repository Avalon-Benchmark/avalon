from typing import Optional
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from avalon.common.errors import SwitchError


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


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=512) -> None:
        """Initializer.
        num_channels: the number of channels in the input images (eg 3
            for RGB images, or 12 for a stack of 4 RGB images).
        num_outputs: the dimension of the output distribution.
        dist: the output distribution (eg Discrete or Normal).
        hidden_size: the size of the final actor+critic linear layers

        TODO: this needs updated to the new action types.
        """
        super().__init__()

        # TODO: make this more general to allow non-scalar spaces
        assert len(observation_space["wrapped"].shape) == 3
        num_channels = observation_space["wrapped"].shape[0]
        self.num_outputs = len(action_space.spaces)

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu")
        )

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            Flatten(),
            # init_(nn.Linear(32 * 7 * 7, hidden_size)),
            # nn.ReLU(),
            mlp_init(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.Tanh(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = mlp_init(nn.Linear(hidden_size, 1), gain=1.0)
        self.action_head = DictActionHead(action_space)
        self.actor_linear = mlp_init(nn.Linear(hidden_size, self.action_head.num_inputs), gain=0.01)

    def forward(self, obs):
        """x should have shape (batch_size, num_channels, 84, 84)."""
        x = obs["wrapped"]
        batch_size = x.shape[0]
        x = self.main(x)
        value = self.critic_linear(x).reshape((batch_size,))
        action_logits = self.actor_linear(x)
        assert value.shape == (batch_size,)
        assert action_logits.shape == (batch_size, self.action_head.num_inputs)
        return value, self.action_head(action_logits)


class MLPBase(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64) -> None:
        super().__init__()

        # TODO: make this more general to allow non-scalar spaces
        assert len(observation_space["wrapped"].shape) == 1
        num_inputs = observation_space["wrapped"].shape[0]

        self.action_head = DictActionHead(action_space)

        self.actor = nn.Sequential(
            mlp_init(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            mlp_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            mlp_init(nn.Linear(hidden_size, self.action_head.num_inputs), gain=0.01),
        )

        self.critic = nn.Sequential(
            mlp_init(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            mlp_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            mlp_init(nn.Linear(hidden_size, 1), gain=1.0),
        )

    def forward(self, obs):
        x = obs["wrapped"]
        batch_size = x.shape[0]
        value = self.critic(x).reshape((batch_size,))
        action_logits = self.actor(x)
        assert value.shape == (batch_size,)
        assert action_logits.shape == (batch_size, self.action_head.num_inputs)
        return value, self.action_head(action_logits)


# TODO: this needs fixed to take a 2-d policy logit
class DiscreteHead(nn.Module):
    """A module that builds a Categorical distribution from logits."""

    def __init__(self, num_outputs, **kwargs) -> None:
        super().__init__()
        self.num_inputs = num_outputs
        self.num_outputs = 1

    def forward(self, x):
        # x should have shape (batch_size, num_actions, num_categories)
        assert x.shape[-1] == self.num_inputs
        probs = F.softmax(x, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        # Categorical already has only one output, no need for independent!
        # dist = torch.distributions.Independent(dist, reinterpreted_batch_ndims=num_batch_dims)
        return dist


class BernoulliHead(nn.Module):
    def __init__(self, n_outputs, bias: Optional[Tuple[float, ...]] = None, **kwargs) -> None:
        super().__init__()
        self.num_inputs = n_outputs
        self.num_outputs = n_outputs
        if bias is None:
            self.bias = tuple(0 for _ in range(n_outputs))
        else:
            assert len(bias) == n_outputs
            self.bias = bias

    def forward(self, x):
        # x should have shape (batch_size, 1)
        assert x.shape[-1] == self.num_inputs
        x = torch.clamp(x, -6, 6)
        bias = torch.tensor(self.bias).view([1] * (len(x.shape) - 1) + [self.num_inputs]).to(x.device)
        dist = torch.distributions.Bernoulli(logits=x + bias)
        dist = torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        return dist


class NormalHead(nn.Module):
    """A module that builds a Diagonal Gaussian distribution from means.

    Standard deviations are learned parameters in this module.
    """

    def __init__(self, action_space: gym.spaces.Box, **kwargs) -> None:
        super().__init__()
        assert isinstance(action_space, gym.spaces.Box)
        assert len(action_space.shape) == 1
        self.num_outputs = action_space.shape[0]
        self.num_inputs = self.num_outputs * 2
        # initial variance is e^0 = 1
        # self.log_std = nn.Parameter(torch.zeros(self.num_outputs))
        self.register_buffer("center", torch.tensor((action_space.high + action_space.low) / 2))
        # TODO: should we put the scale into the initial value of log_std instead?
        # Would effect the effective LR of that parameter.
        self.register_buffer("scale", torch.tensor((action_space.high - action_space.low) / 2))

    def forward(self, x):
        # x should have shape (batch_size, action_dim)
        assert x.shape[-1] == self.num_inputs
        means, log_stds = torch.chunk(x, 2, dim=-1)
        log_stds = torch.clamp(log_stds, -6, 6)
        dist = torch.distributions.Normal(loc=means, scale=log_stds.exp())
        dist = torch.distributions.Independent(dist, reinterpreted_batch_ndims=1)
        return dist


class DictActionHead(torch.nn.Module):
    def __init__(self, action_space) -> None:
        super().__init__()
        self.action_space = action_space

        # Build action heads
        self.num_inputs = 0
        action_heads = {}
        for k, space in self.action_space.spaces.items():
            # TODO: major perf gains from not forcing boxes to have scalar shape
            if isinstance(space, gym.spaces.Box):
                action_heads[k] = NormalHead(space)
            elif isinstance(space, gym.spaces.Discrete):
                action_heads[k] = DiscreteHead(space.n)
            elif isinstance(space, gym.spaces.MultiBinary):
                action_heads[k] = BernoulliHead(space.n)
            else:
                assert False

            self.num_inputs += action_heads[k].num_inputs
        self.action_heads = torch.nn.ModuleDict(action_heads)

    def forward(self, action_logits):
        batch_size = action_logits.shape[:-1]
        assert action_logits.shape[-1] == self.num_inputs

        dists = {}
        # NOTE: this depends on the fact that dicts are ordered in recent CPython.
        i = 0
        for k, head in self.action_heads.items():
            # we want the action logit to have shape (batch_size, num_inputs)
            logits = action_logits[..., i : i + head.num_inputs]
            assert logits.shape[-1] == head.num_inputs
            dists[k] = head(logits)
            i += head.num_outputs
        return DictActionDist(dists, self.action_heads)


class DictActionDist(torch.distributions.Distribution):
    def __init__(self, dists, action_heads) -> None:
        super().__init__(validate_args=False)
        self.dists = dists
        self.action_heads = action_heads

    def sample(self, device="cpu"):
        """Returns an action as a dict of batched tensors."""
        # rsample is not used - gradients propagate thru log_prob, not sample
        out = {}
        for k, v in self.dists.items():
            # should have shape (batch_size, num_outputs)
            sample = v.sample().to(device=device)
            # assert len(sample.shape) == 2
            # assert sample.dtype == torch.float32
            # assert sample.shape[-1] == self.action_heads[k].num_outputs
            out[k] = sample
        return out

    def log_prob(self, actions):
        """Compute the log prob of the given action under the given dist (batchwise).

        Log prob is a scalar (per batch element.)"""
        # actions is a dict of tensors of shape (batch_size, num_outputs)
        log_probs = []
        for k, dist in self.dists.items():
            # TODO: this used to call .float() . . . why?
            log_prob = dist.log_prob(actions[k])
            log_probs.append(log_prob)

        return torch.stack(log_probs, dim=-1).sum(dim=-1)

    def entropy(self):
        entropies = []
        for k, dist in self.dists.items():
            entropy = dist.entropy()
            # The entropy is over the entire action space, so it reduces to a single scalar per batch element
            assert len(entropy.shape) == 1
            entropies.append(entropy)
        return torch.stack(entropies, dim=1).sum(dim=1)

    def sample_mode(self, device="cpu"):
        out = {}
        for k, v in self.dists.items():
            if isinstance(v, torch.distributions.Independent):
                v = v.base_dist
            if isinstance(v, torch.distributions.Normal):
                sample = v.mean
            elif isinstance(v, torch.distributions.Categorical):
                _value, sample = torch.max(v.logits, dim=-1)
            elif isinstance(v, torch.distributions.Bernoulli):
                sample = torch.where(
                    v.logits > 0,
                    torch.ones_like(v.logits, dtype=torch.int),
                    torch.zeros_like(v.logits, dtype=torch.int),
                )
            else:
                raise SwitchError(f"Unknown distribution for mode, {v.__class__}")
            out[k] = sample.to(device=device)
        return out
