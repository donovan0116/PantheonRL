import torch as th
from torch import nn
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.torch_layers import MlpExtractor


class MyIdeaPolicy(BasePolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            activation_fn=nn.Tanh,
            ortho_init=True,
    ):
        super().__init__(observation_space, action_space)

        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        # **主网络**（main network）：负责决策环境中的 action
        self.main_mlp = MlpExtractor(observation_space.shape[0], net_arch, activation_fn)
        self.action_dist = make_proba_distribution(action_space)
        self.action_net = nn.Linear(self.main_mlp.latent_dim_pi, action_space.shape[0])
        self.value_net = nn.Linear(self.main_mlp.latent_dim_vf, 1)

        # **通信网络**（comm network）：决定是否通信
        self.comm_mlp = MlpExtractor(observation_space.shape[0], net_arch, activation_fn)
        self.comm_action_net = nn.Linear(self.comm_mlp.latent_dim_pi, 1)  # 二分类任务
        self.comm_value_net = nn.Linear(self.comm_mlp.latent_dim_vf, 1)

        # 共享优化器
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def forward(self, obs):
        """
        前向计算两个网络的动作
        """
        # 处理主策略
        main_latent_pi, main_latent_vf = self.main_mlp(obs)
        action_mean = self.action_net(main_latent_pi)
        action_distribution = self.action_dist.proba_distribution(action_mean)
        value = self.value_net(main_latent_vf)

        # 处理通信策略
        comm_latent_pi, comm_latent_vf = self.comm_mlp(obs)
        action_comm = th.sigmoid(self.comm_action_net(comm_latent_pi))  # 通信决策
        value_comm = self.comm_value_net(comm_latent_vf)

        return action_distribution, value, action_comm, value_comm

    def evaluate_actions(self, obs, actions, actions_comm):
        """
        计算 log_prob, entropy, values
        """
        action_distribution, value, action_comm, value_comm = self.forward(obs)

        log_prob = action_distribution.log_prob(actions)
        entropy = action_distribution.entropy()
        comm_log_prob = th.log(action_comm + 1e-8) * actions_comm + th.log(1 - action_comm + 1e-8) * (1 - actions_comm)

        return value, log_prob, entropy, value_comm, comm_log_prob
