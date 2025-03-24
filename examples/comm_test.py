import gym
import numpy as np
from typing import List, Dict, Any, Tuple

import redis
from ray.rllib.core.learner.learner import torch

from myAlgorithm.ImplicitRewardPolicy.ToMNet import *

import yaml
from stable_baselines3 import PPO

from myAlgorithm.my_ppo import MyPPO
from myAlgorithm.common.comm_agent_wrapper import SimpleCommunicativePartner
from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
from gym import spaces
from myAlgorithm.common.myEnv import InteractiveOvercookedEnv

with open('../myAlgorithm/config/my_ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

args = config

# 使用示例
def main():
    base_env = gym.make(args['env']['id'], layout_name=args['env']['layout'])
    env = InteractiveOvercookedEnv(base_env)
    args['env'] = env

    r = redis.Redis(host='127.0.0.1', port=6379, db=0)
    r.flushdb()

    # 初始化智能体
    # ego_agent = PPO('MlpPolicy', env, verbose=1)
    partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=0))
    partner = SimpleCommunicativePartner(partner)
    env.add_agent(partner, 'partner')

    # hyper parameters
    input_size = env.observation_space.shape[0] + 1
    lamb = 0.5
    data_num = 4000
    seq_len = 2
    args['seq_len'] = seq_len
    batch_size = 32
    model = ToMNet(input_size=input_size, hidden_size=[64, 256, input_size], output_size=input_size * seq_len)

    fake_dataset = make_fake_dataset(env, 320, 2)
    args['fake_dataset'] = fake_dataset
    train_step1(model, fake_dataset, 32, 10)
    train_step2(model, fake_dataset, 32, 10)
    # torch.save(model.state_dict(), args['ToM_model_path'])
    print("training finish")
    args['ToM_model'] = model



    # # 训练过程
    # state = env.reset()
    # for _ in range(1000):
    #     action, _ = ego_agent.predict(state)
    #     state, reward, done, _ = env.step(action)
    #     partner_state = state[1]
    #     partner_action = state[2]
    #     state = state[0]
    #
    #     # 模拟人类响应（实际应由真实人类交互）
    #     if env.communication_state['is_communicating']:
    #         env.human_response(np.random.randint(0, 3))
    #
    #     if done:
    #         break
    ego = MyPPO(args)
    env.add_agent(ego, 'ego')
    r.flushdb()
    ego.learn(total_timesteps=10000)
    ego.save("ppo_model")

if __name__ == "__main__":
    main()