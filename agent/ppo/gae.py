"""
Code for computing Generalized Advantage Estimation.
"""
import numpy as np
import torch


def gae(rewards, values, dones, gamma, lam, last_values):
    """Compute generalized advantage estimate.
    rewards: a list of rewards at each step.
    values: the value estimate of the state at each step.
    dones: an array of the same shape as rewards, with a 1 if the
        episode ended at that step and a 0 otherwise.
    gamma: the discount factor.
    lam: the GAE lambda parameter.
    last_values: the values computed from the terminal observation (O_T+1)
    """
    is_torch = isinstance(rewards, torch.Tensor)
    # Invert dones to have 0 if the episode ended and 1 otherwise
    not_dones = (dones * -1) + 1

    N = rewards.shape[0]
    T = rewards.shape[1]
    if is_torch:
        gae_step = torch.zeros(size=(N,), device=rewards.device, dtype=torch.float32)
        advantages = torch.zeros(size=(N, T), device=rewards.device, dtype=torch.float32)
    else:
        gae_step = np.zeros((N,))
        advantages = np.zeros((N, T))
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
