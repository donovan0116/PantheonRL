import numpy as np
import torch as th
from gym import spaces
from rlcard.games.doudizhu.utils import action
from scipy.constants import value
from scipy.stats import entropy
from torch import nn
import warnings
import collections
from functools import partial

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)


class MyIdeaPolicy(BasePolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            main_net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
            comm_net_arch: Union[List[int], Dict[str, List[int]], List[Dict[str, List[int]]], None] = None,
            activation_fn=nn.Tanh,
            ortho_init=True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            share_features_extractor: bool = True,
            normalize_images: bool = True,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(observation_space, action_space)

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        """
        默认net_arch
        net_arch = dict(pi=[64, 64], vf=[64, 64])
        """

        if main_net_arch is None:
            if features_extractor_class == NatureCNN:
                main_net_arch = []
            else:
                main_net_arch = dict(pi=[64, 32, 64], vf=[64, 64])

        if comm_net_arch is None:
            if features_extractor_class == NatureCNN:
                comm_net_arch = []
            else:
                comm_net_arch = dict(pi=[64, 32, 64], vf=[64, 64])

        # **主网络**（main network）：负责决策环境中的 action
        self.main_mlp = MlpExtractor(observation_space.shape[0], main_net_arch, activation_fn)
        self.action_dist = make_proba_distribution(action_space)
        self.main_action_net = nn.Linear(self.main_mlp.latent_dim_pi, self.action_space.n)
        self.main_value_net = nn.Linear(self.main_mlp.latent_dim_vf, 1)
        # **通信网络**（comm network）：决定是否通信
        self.comm_mlp = MlpExtractor(observation_space.shape[0], comm_net_arch, activation_fn)
        self.comm_action_net = nn.Linear(self.comm_mlp.latent_dim_pi, 1)
        self.comm_value_net = nn.Linear(self.comm_mlp.latent_dim_vf, 1)

        # 共享优化器
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs):

        if obs.dtype != th.float32:
            # print(f"Warning: Input tensor dtype is {obs.dtype}, converting to torch.float32")
            obs = obs.to(dtype=th.float32)  # 转换为 float32

        main_latent_pi, main_latent_vf = self.main_mlp(obs)
        action_mean = self.main_action_net(main_latent_pi)
        action_distribution = self.action_dist.proba_distribution(action_mean)
        actions = action_distribution.get_actions()
        log_prob = action_distribution.log_prob(actions)
        value = self.main_value_net(main_latent_vf)

        comm_latent_pi, comm_latent_vf = self.comm_mlp(obs)
        # action_comm = th.sigmoid(self.comm_action_net(comm_latent_pi))  # 通信决策
        action_comm_mean = self.comm_action_net(comm_latent_pi)
        action_comm_distribution = self.action_dist.proba_distribution(action_comm_mean)
        action_comm = action_comm_distribution.get_actions()
        log_prob_comm = action_comm_distribution.log_prob(action_comm)
        value_comm = self.comm_value_net(comm_latent_vf)

        # return action_distribution, value, action_comm, value_comm
        return (th.concat((actions, action_comm), dim=0),
                th.concat((value, value_comm), dim=0),
                th.concat((log_prob, log_prob_comm), dim=0)
                )

    def evaluate_actions(self, obs, actions, actions_comm):

        main_latent_pi, main_latent_vf = self.main_mlp(obs)
        action_mean = self.main_action_net(main_latent_pi)
        action_distribution = self.action_dist.proba_distribution(action_mean)
        log_prob = action_distribution.log_prob(actions)
        # shadow name
        entropy_ = action_distribution.entropy()
        value_ = self.main_value_net(main_latent_vf)

        comm_latent_pi, comm_latent_vf = self.comm_mlp(obs)
        action_comm_mean = self.comm_action_net(comm_latent_pi)
        action_comm_distribution = self.action_dist.proba_distribution(action_comm_mean)
        log_prob_comm = action_comm_distribution.log_prob(actions_comm)
        entropy_comm = action_comm_distribution.entropy()
        value_comm = self.comm_value_net(comm_latent_vf)


        return value_, log_prob, entropy_, value_comm, log_prob_comm, entropy_comm

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        action_distribution, _, _, _ = self.forward(obs)
        if deterministic:
            return action_distribution.mode()
        return action_distribution.sample()

    def predict_values(self, obs: th.Tensor):
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # 这里本来是提取features，但是本环境中不需要提取observation的特征，直接提取原observation
        # features = super().extract_features(obs, self.vf_features_extractor)
        _, main_latent_vf = self.main_mlp(obs.to(th.float32))
        _, comm_latent_vf = self.comm_mlp(obs.to(th.float32))
        return self.main_value_net(main_latent_vf), self.comm_value_net(comm_latent_vf)