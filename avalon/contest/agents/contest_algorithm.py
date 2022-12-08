import torch
from torch import Tensor

from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import ObservationBatch


class RandomAlgorithm:
    """
    Instruction to participant: replace this class with your own algorithm.
    """

    def __init__(self) -> None:
        torch.manual_seed(0)
        self.real_dist = torch.distributions.Normal(loc=torch.tensor([0.0] * 18), scale=torch.tensor([1.0] * 18))
        self.discrete_dist = torch.distributions.Bernoulli(probs=torch.tensor([0.5] * 3))

    def rollout_step(
        self,
        next_obs: ObservationBatch,
        dones: Tensor,  # shape (batch_size, )
    ) -> ActionBatch:
        batch_size = next_obs["rgbd"].shape[0]
        real_actions = self.real_dist.sample((batch_size,))

        discrete_actions = torch.zeros((batch_size, 3, 2))
        discrete_sample = self.discrete_dist.sample((batch_size,))
        discrete_actions[:, :, 1] = discrete_sample
        discrete_actions[:, :, 0] = 1 - discrete_sample

        step_actions: ActionBatch = {"real": real_actions, "discrete": discrete_actions}

        return step_actions


def load_algorithm():
    return RandomAlgorithm()
