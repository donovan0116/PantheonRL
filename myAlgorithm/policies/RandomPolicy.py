import numpy as np
from stable_baselines3.common.policies import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, observation_space, action_space, lr_schedule):
        super(RandomPolicy, self).__init__(observation_space, action_space, features_extractor=None)

    def forward(self, obs):
        return np.array([self.action_space.sample() for _ in range(len(obs))])

    def _predict(self, observation, deterministic=False):
        return np.array([self.action_space.sample() for _ in range(len(observation))])


import numpy as np
from pantheonrl.common.agents import OnPolicyAgent
from stable_baselines3.common.base_class import BaseAlgorithm


class RandomAgent(OnPolicyAgent):
    def __init__(self, action_space):
        # 这里要调用父类 OnPolicyAgent，并传入一个 "伪" 的 policy
        class DummyPolicy(BaseAlgorithm):
            def __init__(self, action_space):
                self.action_space = action_space

            def predict(self, obs, deterministic=False):
                return np.array([self.action_space.sample() for _ in range(len(obs))]), None

        dummy_policy = DummyPolicy(action_space)
        super().__init__(dummy_policy)  # 这里正确调用了父类初始化

    def predict(self, obs, deterministic=False):
        return np.array([self.action_space.sample() for _ in range(len(obs))]), None

