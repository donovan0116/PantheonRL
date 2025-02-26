import sys

from gym import spaces
from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch as th
import numpy as np
from stable_baselines3.common.logger import JSONOutputFormat
from torch.nn import functional as F

from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy

from .common.my_on_policy_algorithm import MyOnPolicyAlgorithm
from .policies.MyActorCriticPolicy import MyActorCriticPolicy
from .policies.MyIdeaPolicy import MyIdeaPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")

class MyPPO(MyOnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "MlppppppPolicy": MyIdeaPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(self, args):
        super().__init__(
            args["policy"],
            args["env"],
            learning_rate=args["learning_rate"],
            n_steps=args["n_steps"],
            gamma=args["gamma"],
            gae_lambda=args["gae_lambda"],
            ent_coef=args["ent_coef"],
            vf_coef=args["vf_coef"],
            max_grad_norm=args["max_grad_norm"],
            use_sde=args["use_sde"],
            sde_sample_freq=args["sde_sample_freq"],
            tensorboard_log=args["tensorboard_log"],
            policy_kwargs=args["policy_kwargs"],
            verbose=args["verbose"],
            device=args["device"],
            seed=args["seed"],
            _init_setup_model=False,
            tom_model = args["ToM_model"],
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if args["normalize_advantage"]:
            assert (
                args["batch_size"] > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        self.batch_size = args["batch_size"]
        self.n_epochs = args["n_epochs"]
        self.clip_range = args["clip_range"]
        self.clip_range_vf = args["clip_range_vf"]
        self.normalize_advantage = args["normalize_advantage"]
        self.target_kl = args["target_kl"]

        if args["_init_setup_model"]:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, entropy_losses_comm = [], []
        pg_losses, value_losses = [], []
        pg_losses_comm, value_losses_comm = [], []
        clip_fractions, clip_fractions_comm = [], []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                actions_comm = rollout_data.actions_comm
                if isinstance(self.action_space, spaces.Discrete):
                    actions_comm = rollout_data.actions_comm.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, values_comm, log_prob_comm, entropy_comm = self.policy.evaluate_actions(rollout_data.observations, actions, actions_comm)

                values = values.flatten()
                values_comm = values_comm.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages_comm = rollout_data.advantages_comm
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                if self.normalize_advantage and len(advantages_comm) > 1:
                    advantages_comm = (advantages_comm - advantages_comm.mean()) / (advantages_comm.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                ratio_comm = th.exp(log_prob_comm - rollout_data.old_log_prob_comm)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                policy_loss_comm_1 = advantages_comm * ratio_comm
                policy_loss_comm_2 = advantages_comm * th.clamp(ratio_comm, 1 - clip_range, 1 + clip_range)
                policy_loss_comm = -th.min(policy_loss_comm_1, policy_loss_comm_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Logging
                pg_losses_comm.append(policy_loss_comm.item())
                clip_fraction_comm = th.mean((th.abs(ratio_comm - 1) > clip_range).float()).item()
                clip_fractions_comm.append(clip_fraction_comm)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                    values_pred_comm = values_comm
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                    values_pred_comm = rollout_data.old_values_comm + th.clamp(
                        values_comm - rollout_data.old_values_comm, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                value_loss_comm = F.mse_loss(rollout_data.returns_comm, values_pred_comm)
                value_losses_comm.append(value_loss_comm.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                    entropy_loss_comm = -th.mean(-log_prob_comm)
                else:
                    entropy_loss = -th.mean(entropy)
                    entropy_loss_comm = -th.mean(entropy_comm)

                entropy_losses.append(entropy_loss.item())
                entropy_losses_comm.append(entropy_loss_comm.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                loss_comm = policy_loss_comm + self.ent_coef * entropy_loss_comm + self.vf_coef * value_loss_comm

                # 尝试将两个loss使用同一个优化器进行优化
                loss += loss_comm

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/entropy_loss_comm", np.mean(entropy_losses_comm))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/policy_gradient_loss_comm", np.mean(pg_losses_comm))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/value_loss_comm", np.mean(value_losses_comm))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/clip_fraction_comm", np.mean(clip_fractions_comm))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/loss_comm", loss_comm.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )