import itertools
from typing import Any
from typing import Optional
from typing import Tuple

import gym
import numpy as np
import torch
import wandb
from einops import rearrange
from einops import repeat
from matplotlib import pyplot as plt
from nptyping import Float32
from nptyping import NDArray
from nptyping import Shape
from skimage.metrics import structural_similarity as ssim_metric
from torch import Tensor
from torch import nn
from torch.distributions import Bernoulli
from torch.distributions import Independent
from torch.distributions import Normal
from tree import flatten
from tree import map_structure

from avalon.agent.common import wandb_lib
from avalon.agent.common.action_model import DictActionDist
from avalon.agent.common.action_model import DictActionHead
from avalon.agent.common.action_model import StraightThroughOneHotCategorical
from avalon.agent.common.action_model import visualize_action_dists
from avalon.agent.common.types import ActionBatch
from avalon.agent.common.types import Algorithm
from avalon.agent.common.types import BatchSequenceData
from avalon.agent.common.types import LatentBatch
from avalon.agent.common.types import ObservationBatch
from avalon.agent.common.types import StateActionBatch
from avalon.agent.common.util import explained_variance
from avalon.agent.common.util import get_avalon_model_seed
from avalon.agent.dreamer.models import RSSM
from avalon.agent.dreamer.models import DenseDecoder
from avalon.agent.dreamer.models import HybridDecoder
from avalon.agent.dreamer.models import HybridEncoder
from avalon.agent.dreamer.params import DreamerParams
from avalon.agent.dreamer.tools import lambda_return
from avalon.agent.dreamer.tools import pack_list_of_dicts


class Dreamer(Algorithm[DreamerParams]):
    def __init__(self, params: DreamerParams, obs_space: gym.spaces.Dict, action_space: gym.spaces.Dict) -> None:
        super().__init__(params, obs_space, action_space)

        self._encode = HybridEncoder(obs_space, mlp_hidden_dim=self.params.encoder_mlp_hidden_size)
        feature_size = self.params.deter_size + self.params.stoch_size
        self._decode = HybridDecoder(obs_space, feature_size)
        self._reward = DenseDecoder((), 4, feature_size, self.params.decoder_mlp_hidden_size, dist="normal")
        self._value_current = DenseDecoder((), 4, feature_size, self.params.decoder_mlp_hidden_size, dist="normal")
        self._value_lagged = DenseDecoder((), 4, feature_size, self.params.decoder_mlp_hidden_size, dist="normal")
        self._value_lagged.load_state_dict(self._value_current.state_dict())

        action_head = DictActionHead(action_space, params)
        self._actor = DenseDecoder(
            (action_head.num_inputs,),
            4,
            feature_size,
            self.params.decoder_mlp_hidden_size,
            dist="action_head",
            action_head=action_head,
        )
        self._dynamics = RSSM(
            actdim=action_head.num_outputs,
            embed_size=self._encode.out_dim,
            stoch=self.params.stoch_size,
            deter=self.params.deter_size,
            hidden=self.params.rssm_hidden_size,
        )
        if self.params.pcont:
            # Will use a Bernoulli output dist
            self._pcont = DenseDecoder((), 4, feature_size, self.params.decoder_mlp_hidden_size, dist="binary")

        model_modules = [self._encode, self._dynamics, self._decode, self._reward]
        if self.params.pcont:
            model_modules.append(self._pcont)
        self._model_params = list(itertools.chain(*[list(module.parameters()) for module in model_modules]))
        self._model_opt = torch.optim.Adam(
            self._model_params,
            self.params.model_lr,
            eps=self.params.adam_eps,
            weight_decay=self.params.adam_weight_decay,
        )
        self._value_opt = torch.optim.Adam(
            self._value_current.parameters(),
            self.params.value_lr,
            eps=self.params.adam_eps,
            weight_decay=self.params.adam_weight_decay,
        )
        self._actor_opt = torch.optim.Adam(
            self._actor.parameters(),
            self.params.actor_lr,
            eps=self.params.adam_eps,
            weight_decay=self.params.adam_weight_decay,
        )

        # self.first_step = False
        # if not self.first_step:
        #     logger.info(f"encoder has {sum(p.numel() for p in self._encode.parameters() if p.requires_grad)} params")
        #     logger.info(f"decoder has {sum(p.numel() for p in self._decode.parameters() if p.requires_grad)} params")
        #     logger.info(f"reward has {sum(p.numel() for p in self._reward.parameters() if p.requires_grad)} params")
        #     logger.info(f"dynamics has {sum(p.numel() for p in self._dynamics.parameters() if p.requires_grad)} params")
        #     logger.info(f"value has {sum(p.numel() for p in self._value_current.parameters() if p.requires_grad)} params")
        #     logger.info(f"actor has {sum(p.numel() for p in self._actor.parameters() if p.requires_grad)} params")
        #     # logger.info([(k, p.shape) for k, p in self._actor.named_parameters() if p.requires_grad])
        #     self.first_step = True

        # This is a tuple of (latent, action)
        self.last_rollout_state: Optional[StateActionBatch] = None

    def rollout_step(
        self,
        next_obs: ObservationBatch,
        dones: Tensor,
        indices_to_run: Tensor,
        exploration_mode: str,
    ) -> Tuple[ActionBatch, dict]:
        # Only used in the GamePlayer
        # dones should be "did this env give a done signal after the last step".
        # In other words, obs should follow done. o_t, done_{t-1}

        # This computes the policy given an observation
        # The inputs should all be tensors/structures of tensors on GPU

        if self.last_rollout_state is None:
            device = next_obs[list(next_obs.keys())[0]].device
            batch_size = len(dones)
            prev_latent = self._dynamics.initial(batch_size, device)
            # Using random actions as an initial action probably isn't ideal, but zero isn't a valid 1-hot action..
            # so that seemed worse.
            if (seed := get_avalon_model_seed()) is not None:
                self.action_space.seed(seed)
            prev_action = self.action_space.sample()
            prev_action = map_structure(lambda x: torch.tensor(x, device=device), prev_action)
            prev_action = map_structure(lambda x: repeat(x, "... -> b ...", b=batch_size), prev_action)
            self.last_rollout_state = (prev_latent, prev_action)

        # next_obs = {k: v[indices_to_run] for k, v in next_obs.items()}
        # Check the batch sizes match - that we're not carrying over an old rollout state.
        assert dones.shape[0] == flatten(self.last_rollout_state)[0].shape[0]

        # We want to set done to false for anything that claims to be done but isn't running this step.
        # This will result in no masking for those states.
        # indices_to_run_torch = torch.tensor(indices_to_run, device=dones.device, dtype=torch.bool)
        indices_to_run_torch = indices_to_run.to(device=dones.device)
        dones = dones & indices_to_run_torch
        dones = dones.to(dtype=torch.float32)

        # Mask the state to 0 for any envs that have finished?
        mask = 1 - dones

        # we need this because the action can have a variable number of dims
        def multiply_vector_along_tensor_batch_dim(x: Tensor, vector: Tensor) -> Tensor:
            assert len(vector.shape) == 1
            extra_dims = (1,) * (x.dim() - 1)
            return x * vector.view(-1, *extra_dims)

        self.last_rollout_state = map_structure(
            lambda x: multiply_vector_along_tensor_batch_dim(x, mask) if x is not None else x, self.last_rollout_state
        )
        assert self.last_rollout_state is not None

        # sliced_state = map_structure(lambda x: x[indices_to_run], self.last_rollout_state)
        action, new_state = self.policy(next_obs, self.last_rollout_state, mode=exploration_mode)
        # action, new_state = self.policy(next_obs, sliced_state, mode="explore")

        # Not sure if map_structure works nicely with inplace operations, so we'll do it manually.
        for k1, v1 in enumerate(self.last_rollout_state):
            for k2, v2 in v1.items():
                v2[indices_to_run] = new_state[k1][k2][indices_to_run]

        action = map_structure(lambda x: x.cpu(), action)
        return action, {}

    def reset_state(self) -> None:
        self.last_rollout_state = None

    def policy(
        self, obs: ObservationBatch, prev_state: StateActionBatch, mode: str = "explore"
    ) -> Tuple[ActionBatch, StateActionBatch]:
        """Encode the observation, pass it into the observation model along with the previous state/action
        to generate a new state estimate, and use that to generate a policy.

        state is a tuple of (latent, action)"""

        # Obs is/can be numpy array
        assert prev_state is not None
        prev_latent, prev_action = prev_state

        embed = self._encode(obs)
        latent, _ = self._dynamics.obs_step(prev_latent, prev_action, embed)
        feat = self._dynamics.get_feat(latent)
        if mode == "explore":
            action = self._actor(feat).rsample()
        elif mode == "eval":
            action = self._actor(feat).mode()
        else:
            assert False
        state = (latent, action)
        return action, state

    def _imagine_ahead(self, start: LatentBatch, dones: Tensor) -> dict[str, Any]:
        # In the sequence, at a given index, it's an (action -> state) pair. action comes first.
        # Thus the dummy "0" action at the front.

        start = {k: rearrange(v, "b t n -> (b t) n") for k, v in start.items()}
        start["feat"] = self._dynamics.get_feat(start)
        start["action"] = {k: torch.zeros_like(v) for k, v in self._actor(start["feat"]).rsample().items()}  # type: ignore
        seq = {k: [v] for k, v in start.items()}
        for _ in range(self.params.horizon):
            action = self._actor(seq["feat"][-1].detach()).rsample()
            state = self._dynamics.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self._dynamics.get_feat(state)
            for key, value in {**state, "action": action, "feat": feat}.items():
                seq[key].append(value)
        # These now have shape (imag_steps, batch_size * fragment_steps)
        seq_packed = {k: torch.stack(v, 0) for k, v in seq.items() if k != "action"}
        if self.params.pcont:
            disc = self.params.discount * self._pcont(seq_packed["feat"]).probs
            # Override discount prediction for the first step with the true
            # discount factor from the replay buffer.
            dones = rearrange(dones, "b t -> (b t)")
            true_first = 1.0 - dones.float()
            true_first *= self.params.discount
            disc = torch.cat([true_first[None], disc[1:]], 0)
        else:
            disc = self.params.discount * torch.ones(seq_packed["feat"].shape[:-1], device=seq_packed["feat"].device)
        seq_packed["discount"] = disc
        # Shift discount factors because they imply whether the following state
        # will be valid, not whether the current state is valid.
        # TODO: I don't like how the same discount factor is used for value discounting and for this weight.
        # Seems like they have different purposes and should be treated separately.
        seq_packed["weight"] = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)
        seq_packed["action"] = pack_list_of_dicts(seq["action"])  # type: ignore
        return seq_packed

    def train_step(self, batch_data: BatchSequenceData, step: int) -> int:
        self.train()
        # Shape of batch_data elements should be (batch_size, timesteps, ...)
        # Images should be (c, h, w), with range [-.5, .5]

        next_obs = batch_data.observation
        actions = batch_data.action
        rewards = batch_data.reward
        is_terminal = batch_data.is_terminal

        batch_size, timesteps = is_terminal.shape

        # Train the encoder and RSSM model. The actor and value models are not used anywhere here.
        embed = self._encode(next_obs)
        # This is where the loop over all timesteps happens
        post, prior = self._dynamics.observe(embed, actions)
        feat = self._dynamics.get_feat(post)
        assert len(feat.shape) == 3
        obs_pred = self._decode(feat)
        # Reinterpret all but the batch dim (no time dim here)
        # Note: log_likelihood of a Normal with constant std can be reinterpreted as a scaled MSE.
        # is a Normal with std 1 appropriate for vectors too? I guess why not, esp since MSE would seem appropriate.
        obs_dists = {k: Independent(Normal(mean, 1), len(mean.shape) - 2) for k, mean in obs_pred.items()}
        obs_likelihoods = {k: v.log_prob(next_obs[k]) for k, v in obs_dists.items()}
        assert all([v.shape == (batch_size, timesteps) for v in obs_likelihoods.values()])
        autoencoder_mask = 1 - is_terminal.int()
        assert autoencoder_mask.shape == (batch_size, timesteps)

        # Note: these logs will have some terms that are masked out in the loss.
        # No great way to log those, i guess we would reshape and slice them out if we were being proper.
        for k, pred in obs_pred.items():
            target = next_obs[k]
            wandb_lib.log_histogram(f"train/observations/{k}_pred", pred, step)
            wandb_lib.log_histogram(f"train/observations/{k}_target", target, step)
            if step % self.params.log_freq_hist == 0:
                # Log EVs of scalar observations
                if len(target.shape) == 3 and target.shape[-1] == 1:
                    ev = explained_variance(pred, target)
                    wandb_lib.log_scalar(f"train/observations/{k}_EV", ev, step, freq=1)

        reward_pred = self._reward(feat)
        # Note we mask out autoencoding of the terminal timestep.
        # This is because that observation actually comes from the start of the next episode (or is masked to grey,
        # as we actually do). We don't want to try to predict this, if we need to know if the episode ended,
        # we have a pcont signal for that.
        # For done but non-terminal (time limit), we should still be given the true next frame, so predicting that is fine.
        likelihoods: dict[str, Tensor] = {}
        # likelihoods["obs"] = sum([(x * autoencoder_mask).mean() for x in obs_likelihoods.values()])
        likelihoods |= {k: (v * autoencoder_mask).mean() for k, v in obs_likelihoods.items()}
        likelihoods["reward"] = reward_pred.log_prob(rewards).mean()
        likelihoods["reward"] *= self.params.reward_loss_scale
        if self.params.pcont:
            # this is "probability of continue" - an estimator of the done signal
            pcont_pred = self._pcont(feat)
            assert type(pcont_pred) == Bernoulli
            # Label smoothing option
            # pcont_target = 1 - 0.9 * is_terminal.float()
            pcont_target = 1 - is_terminal.float()
            loss = -torch.binary_cross_entropy_with_logits(pcont_pred.logits, pcont_target)
            likelihoods["pcont"] = loss.mean()
            likelihoods["pcont"] *= self.params.pcont_loss_scale

            wandb_lib.log_histogram("train/model/pcont/pred", pcont_pred.probs, step)
            wandb_lib.log_histogram("train/model/pcont/target", pcont_target, step)
            wandb_lib.log_scalar("train/model/pcont/loss_mean", -1 * likelihoods["pcont"], step)

        reward_ev = explained_variance(reward_pred.mean, rewards.float())
        wandb_lib.log_scalar("train/reward/ev", reward_ev, step)
        wandb_lib.log_histogram("train/reward/target", rewards.float(), step)
        wandb_lib.log_histogram("train/reward/pred", reward_pred.mean, step)

        prior_dist = self._dynamics.get_dist(prior)
        post_dist = self._dynamics.get_dist(post)

        wandb_lib.log_histogram("train/kl/prior_mean", prior_dist.mean, step)
        wandb_lib.log_histogram("train/kl/prior_std", prior_dist.stddev, step)
        wandb_lib.log_histogram("train/kl/post_mean", prior_dist.mean, step)
        wandb_lib.log_histogram("train/kl/post_std", prior_dist.stddev, step)
        wandb_lib.log_histogram("train/kl/stoch", post["stoch"], step)
        wandb_lib.log_histogram("train/kl/deter", post["deter"], step)
        wandb_lib.log_histogram("train/kl/embed", embed, step)

        # Dreamerv1 approach
        # FWIW, the dreamerv1 approach works differently than the dreamerv2 approach with balance=.5,
        # which doesn't make much sense. Same with kl_loss_scale=2 to account for smaller grads.
        # Emperically, the v1 approach did better at generating high reward EVs in dmc_cartpole_balance
        # div = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
        # div_clipped = torch.maximum(div, torch.tensor(self._c.free_nats))
        # kl_loss = div_clipped
        # wandb_lib.log_scalar("train/model/div", div, step)

        # Dreamerv2 approach.
        kl_loss, kl_value = self._dynamics.kl_loss(
            post, prior, balance=self.params.kl_balance, free=self.params.free_nats
        )
        assert len(kl_loss.shape) == 0
        wandb_lib.log_scalar("train/model/div", kl_value.mean(), step)

        model_loss = self.params.kl_loss_scale * kl_loss - sum(likelihoods.values())
        wandb_lib.log_scalar("train/model/kl_loss_mean", self.params.kl_loss_scale * kl_loss, step)

        self._model_opt.zero_grad(set_to_none=True)
        model_loss.backward()
        model_norm = nn.utils.clip_grad_norm_(self._model_params, self.params.clip_grad_norm)
        self._model_opt.step()

        wandb_lib.log_histogram("train/model/prior_ent", prior_dist.entropy(), step)
        wandb_lib.log_histogram("train/model/post_ent", post_dist.entropy(), step)
        for name, logprob in likelihoods.items():
            wandb_lib.log_scalar(f"train/model/{name}_loss_mean", -logprob.mean(), step)
        wandb_lib.log_scalar("train/model/loss_mean", model_loss.mean(), step)
        wandb_lib.log_scalar("train/model/grad_norm", model_norm.mean(), step)

        ## IMAGINE #############################

        if (self.params.actor_lr > 0 or self.params.value_lr > 0) and step > self.params.freeze_actor_steps:
            # Train the actor model
            seq = self._imagine_ahead({k: v.detach() for k, v in post.items()}, is_terminal)
            # These rewards have not seen any new actions.
            # So they are the rewards received *in* this timestep, ie before another action is taken.
            reward = self._reward(seq["feat"]).mean
            # NOTE: using the "target" value model here. This value computation is used as a target for training the value network too.
            # Wait, how does dynamics loss work if we're not backproping thru the value model?
            # Well, i guess gradients can still pass thru the lagged model even if it's not being trained?
            slow_value = self._value_lagged(seq["feat"]).mean
            disc = seq["discount"]
            weight = seq["weight"].detach()
            # Skipping last time step because it is used for bootstrapping.
            # Value target i corresponds to the state in seq["feat"] i
            value_target = lambda_return(
                reward[:-1], slow_value[:-1], disc[:-1], bootstrap=slow_value[-1], lambda_=self.params.disclam, axis=0
            )
            assert len(value_target) == len(reward) - 1

        ## ACTOR #############################

        # Actions:      0   [a1]  [a2]   a3
        #                  ^  |  ^  |  ^  |
        #                 /   v /   v /   v
        # States:     [z0]->[z1]-> z2 -> z3
        # Targets:     t0   [t1]  [t2]
        # Baselines:  [v0]  [v1]   v2    v3
        # Entropies:        [e1]  [e2]
        # Weights:    [ 1]  [w1]   w2    w3
        # Loss:              l1    l2

        # Reward:      r0   [r1]  [r2]   r3
        # Two states are lost at the end of the trajectory, one for the boostrap
        # value prediction and one because the corresponding action does not lead
        # anywhere anymore. One target is lost at the start of the trajectory
        # because the initial state comes from the replay buffer.

        # Explaining the above:
        # A state is a (s, a) pair. So z1 is the state of the world right before we take action a1.
        # Let's think just about training the policy that predicted a1. a1 was predicted from z0.
        # The reward resulting from taking action a1 is r1. the value of state z1 (containing r1 and future rewards)
        # is v1.
        # The baseline is the value of the 'state' that existed before we decided on action a1. This doesn't exist
        # as a discrete state in this model, but instead we have state z0, which is the same logical state -
        # z0 is the state where we decided on the action before a1, and thus with a perfect model would know the resulting
        # state after stepping the env also.
        # So the baseline is v0.

        # Reinforce works by saying: if things worked out well in the rollout from this state, when action_x was taken,
        # update the policy to make it more likely to take that action.
        # "worked out well" specifically means "worked better than expected" - and "expected" is the value estimate
        # before seeing the action.
        # So we want the value at this state, the policy computed from this state,
        # and the action actually taken after observing that state.

        if self.params.actor_lr > 0 and step > self.params.freeze_actor_steps:
            policy = self._actor(seq["feat"][:-2].detach())
            assert type(policy) == DictActionDist
            if self.params.actor_grad == "dynamics":
                objective = value_target[1:]
            elif self.params.actor_grad == "reinforce":
                # Why do we recompute this here? We compute the same thing above. I guess different gradient flow somehow? but grads don't even go thru this.
                baseline = self._value_lagged(seq["feat"][:-2]).mode
                advantage = (value_target[1:] - baseline).detach()
                _action = {k: v[1:-1].detach() for k, v in seq["action"].items()}
                objective = policy.log_prob(_action) * advantage
            elif self.params.actor_grad == "hybrid":
                # Dynamics works well for continuous, REINFORCE works well for discrete.
                # We can compute REINFORCE just on the discrete actions. Can't split up the dynamics as easily tho
                # without recomputing the imagination rollout with carefully placed detach()es.
                # So here we just compute dynamics loss normally, and add the discrete-only REINFORCE loss.
                # TODO: this is hacky - the categorical should always be wrapped in an independent so this works..
                discrete_policy = DictActionDist(
                    {
                        k: v
                        for k, v in policy.dists.items()
                        if isinstance(v.base_dist, StraightThroughOneHotCategorical)  # type: ignore
                    }
                )
                assert len(discrete_policy.dists) > 0

                baseline = self._value_lagged(seq["feat"][:-2]).mode
                advantage = (value_target[1:] - baseline).detach()
                discrete_action = {
                    k: v[1:-1].detach() for k, v in seq["action"].items() if k in discrete_policy.dists.keys()
                }
                discrete_reinforce_objective = discrete_policy.log_prob(discrete_action) * advantage
                dynamics_objective = value_target[1:]
                assert discrete_reinforce_objective.shape == dynamics_objective.shape
                # Could add mixing weights here
                objective = discrete_reinforce_objective + dynamics_objective
            else:
                assert False

            # Entropy shape should be (batch * fragment_len, imag_len)
            actor_entropy = policy.entropy()
            actor_entropy_loss = actor_entropy * self.params.policy_entropy_scale
            assert objective.shape == actor_entropy_loss.shape
            objective = objective + actor_entropy_loss

            actor_loss = -1 * objective * weight[:-2]
            self._actor_opt.zero_grad(set_to_none=True)
            actor_loss.mean().backward()
            actor_norm = nn.utils.clip_grad_norm_(self._actor.parameters(), self.params.clip_grad_norm)
            self._actor_opt.step()

            wandb_lib.log_histogram("train/actor/loss", actor_loss, step)  # note this includes the entropy penalty
            wandb_lib.log_scalar("train/actor/grad_norm", actor_norm.mean(), step)
            wandb_lib.log_scalar("train/actor/entropy_loss_mean", -1 * actor_entropy_loss.mean(), step)
            wandb_lib.log_histogram("train/actor/entropy", actor_entropy, step)
            wandb_lib.log_histogram("train/actor/steps_imagined", seq["weight"].sum(dim=0), step)
            wandb_lib.log_histogram("train/reward/imagined", reward[:-1], step)

            # Visualize actions
            visualize_action_dists(self.action_space, policy, prefix="imagination_policy")

        ## VALUE #############################
        if self.params.value_lr > 0 and step > self.params.freeze_actor_steps:
            # Train the value model
            # I'm curious why they train the value model in imagination but not in the actual rollouts.
            # Oh, I guess it has to be on-policy and that's only the case in imagination.
            value_pred = self._value_current(seq["feat"].detach()[:-1], raw_feats=True)
            # isn't the log prob of a Normal with std 1 the same as MSE?
            # Not using an Independent is appropriate here since the event is a scalar and the batch is (t, b)
            value_pred = Normal(value_pred, 1)
            value_loss = -1 * (weight[:-1] * value_pred.log_prob(value_target.detach())).mean()

            self._value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_norm = nn.utils.clip_grad_norm_(self._value_current.parameters(), self.params.clip_grad_norm)
            self._value_opt.step()

            # Update the "target" network.
            # TODO: an EMA might be more appropriate.
            if step % self.params.value_target_network_update_freq == 0:
                self._value_lagged.load_state_dict(self._value_current.state_dict())

            value_ev = explained_variance(value_pred.mean * weight[:-1], value_target.float() * weight[:-1])
            wandb_lib.log_scalar("train/value/ev", value_ev, step)
            # TODO: these histograms of things that are soft-masked don't really making sense.
            # Without the mask, we're showing hists containing junk. But if we apply the mask,
            # the hist ends up with wrong values. Ideally we'd make a "weighted hist".
            # In the meantime, just consider they'll be containing junk unless the imag rollouts rarely contain ep ends.
            wandb_lib.log_histogram("train/value/pred_slow", slow_value, step)
            wandb_lib.log_histogram("train/value/pred", value_pred.mean, step)
            wandb_lib.log_histogram("train/value/target", value_target, step)

            wandb_lib.log_scalar("train/value/grad_norm", value_norm.mean(), step)
            wandb_lib.log_scalar("train/value/loss_mean", value_loss.mean(), step)
            wandb_lib.log_histogram("train/value/weight", weight, step)

        # if "rgbd" in next_obs and step % self.params.log_freq_media == 0:
        for k, v in next_obs.items():
            if len(v.shape) == 5 and step % wandb_lib.MEDIA_FREQ == 0:  # b, t, c, h, w
                batch_size = 6
                # Do an imagination rollout. Can we use the imagine_ahead logic instead of replicating here?
                truth = next_obs[k][:batch_size, :, :3] + 0.5
                recon = obs_pred[k][:batch_size, :, :3]
                # we observe the first 5 frames to estimate the state
                sliced_actions = {k: v[:batch_size, :5] for k, v in actions.items()}
                init, _ = self._dynamics.observe(embed[:batch_size, :5], sliced_actions)
                init = {k: v[:, -1] for k, v in init.items()}
                # Then do an imagination rollout from there
                actual_actions = {k: rearrange(v[:batch_size, 5:], "b t ... -> t b ...") for k, v in actions.items()}
                prior = self._dynamics.imagine(actual_actions, init)
                openl = self._decode(self._dynamics.get_feat(prior))[k][:, :, :3]
                # First 5 frames are recon, next are imagination
                model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
                error = (model - truth + 1) / 2
                comparison = torch.cat([truth, model, error], 3)
                wandb_lib.log_video("video/openl", comparison, step, freq=1, num_images_per_row=batch_size)

                # similarity metrics
                truth_np = truth.detach().cpu().numpy()
                recon_np = recon.detach().cpu().numpy()
                openl_np = openl.detach().cpu().numpy()
                recon_ssim = calculate_framewise_ssim(truth_np + 0.5, recon_np + 0.5)
                openl_ssim = calculate_framewise_ssim(truth_np[:, 5:] + 0.5, openl_np + 0.5)
                recon_ssim_plot = make_ssim_plot(recon_ssim, "Reconstructed observation SSIM by frame across batches")
                openl_ssim_plot = make_ssim_plot(openl_ssim, "Imagination rollout SSIM by frame across batches")
                wandb.log({f"images/recon_ssim": wandb.Image(recon_ssim_plot)})
                wandb.log({f"images/openl_ssim": wandb.Image(openl_ssim_plot)})
        return step + 1


NPObservationBatch = NDArray[Shape["Batch, Frame, Channel, Height, Width"], Float32]
NPFramewiseMetric = NDArray[Shape["Batch, Frame"], Float32]


def calculate_framewise_ssim(ground_truth: NPObservationBatch, predictions: NPObservationBatch) -> NPFramewiseMetric:
    ssim = np.zeros(ground_truth.shape[:2])
    batch_size, fragment_length = ssim.shape
    for batch in range(batch_size):
        for frame in range(fragment_length):
            ssim[batch, frame] = ssim_metric(ground_truth[batch, frame], predictions[batch, frame], channel_axis=0)
    return ssim


def make_ssim_plot(ssim: NPFramewiseMetric, title: str):
    fig, ax = plt.subplots(1)
    plt.plot(ssim.T)
    plt.ylabel("SSIM")
    plt.xlabel("Distance in frames")
    plt.title(title)
    return fig
