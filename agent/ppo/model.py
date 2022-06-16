import gym
import torch.nn as nn

from agent.action_model import DictActionHead
from agent.observation_model import init
from agent.observation_model import mlp_init


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CNNBase(nn.Module):
    def __init__(self, args, observation_space, action_space, hidden_size=512):
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
        self.action_head = DictActionHead(action_space, args)
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
    def __init__(self, args, observation_space, action_space, hidden_size=64):
        super().__init__()

        assert isinstance(observation_space, gym.spaces.Dict)
        assert len(observation_space) == 1
        self.key = list(observation_space.keys())[0]
        assert len(observation_space[self.key].shape) == 1
        num_inputs = observation_space[self.key].shape[0]

        self.action_head = DictActionHead(action_space, args)

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
        x = obs[self.key]
        batch_size = x.shape[0]
        value = self.critic(x).reshape((batch_size,))
        action_logits = self.actor(x)
        assert value.shape == (batch_size,)
        assert action_logits.shape == (batch_size, self.action_head.num_inputs)
        return value, self.action_head(action_logits)
