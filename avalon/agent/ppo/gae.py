"""
Code for computing Generalized Advantage Estimation.
"""
import torch
from torch import Tensor


def gae(rewards: Tensor, values: Tensor, dones: Tensor, gamma: float, lam: float, last_values: Tensor) -> Tensor:
    """Compute generalized advantage estimate.
    Expects ordering from rollouts with obs_first=True, ie (obs/value, action, reward/done).
    rewards: a list of rewards at each step.
    values: the value estimate of the state at each step.
    dones: an array of the same shape as rewards, with a 1 if the
        episode ended at that step and a 0 otherwise.
    gamma: the discount factor.
    lam: the GAE lambda parameter. if lam=0, we perform 1-step value boostrapping.
        if lam=1, we use discounted emperical returns (with value backup at the end of the fragment)
    last_values: the values computed from the terminal observation (O_T+1)
    """
    # Invert dones to have 0 if the episode ended and 1 otherwise
    not_dones = (dones * -1) + 1

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = torch.zeros(size=(N,), device=rewards.device, dtype=torch.float32)
    advantages = torch.zeros(size=(N, T), device=rewards.device, dtype=torch.float32)
    for t in reversed(range(T)):
        if t == T - 1:
            next_values = last_values
        else:
            next_values = values[:, t + 1]

        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * next_values * not_dones[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this step's delta
        gae_step = delta + gamma * lam * not_dones[:, t] * gae_step
        assert gae_step.shape == (N,)
        # And store it
        advantages[:, t] = gae_step
    return advantages
