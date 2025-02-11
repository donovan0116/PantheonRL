from gym import spaces
from typing import Any, Dict, Optional, Type, TypeVar, Union

import torch as th
import numpy as np
from torch.nn import functional as F

from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from .policies.MyActorCriticPolicy import MyActorCriticPolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

SelfBaseModel = TypeVar("SelfBaseModel", bound="BaseModel")

class MyPPO(OnPolicyAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "MlppppppPolicy": MyActorCriticPolicy,
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

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            obs = rollout_data.observations
            actions = rollout_data.actions
            rewards = rollout_data.rewards
            actions_comm = rollout_data.action_comm
            rewards_comm = rollout_data.reward_comm

            # 计算策略的值
            values, log_prob, entropy, values_comm, comm_log_prob = self.policy.evaluate_actions(
                obs, actions, actions_comm
            )

            # 计算主策略损失
            advantages = rollout_data.advantages
            policy_loss = -(advantages * log_prob).mean()
            value_loss = F.mse_loss(values, rollout_data.returns)

            # 计算通信策略损失
            comm_advantage = rewards_comm - values_comm
            comm_policy_loss = -(comm_advantage * comm_log_prob).mean()
            comm_value_loss = F.mse_loss(values_comm, rewards_comm)

            # 计算总损失
            loss = policy_loss + value_loss + comm_policy_loss + comm_value_loss

            # 反向传播
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

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