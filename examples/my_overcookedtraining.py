"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from myAlgorithm.MyPPO import MyPPO
from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
from myAlgorithm.ImplicitRewardPolicy.ToMNet import *

layout = 'simple'
assert layout in LAYOUT_LIST

with open('../myAlgorithm/config/my_ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

args = config

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make(args['env']['id'], layout_name=args['env']['layout'])
args['env'] = env

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress

partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=0))
env.add_partner_agent(partner)

# hyper parameters
input_size = env.observation_space.shape[0] + 1
lamb = 0.5
data_num = 4000
seq_len = 10
batch_size = 32
model = ToMNet(input_size=input_size, hidden_size=[64, 256, input_size], output_size=input_size * seq_len)
# if args['ToM_model_train']:
#     # train ToMNet
#     fake_dataset = make_fake_dataset(env, 4000, 10)
#     train_step1(model, fake_dataset, 32, 100)
#     train_step2(model, fake_dataset, 32, 100)
#     torch.save(model.state_dict(), args['ToM_model_path'])
#     print("training finish")
# else:
#     checkpoint = torch.load(args['ToM_model_path'])
#     model.load_state_dict(checkpoint['model_state_dict'])

# train ToMNet
fake_dataset = make_fake_dataset(env, 4000, 10)
train_step1(model, fake_dataset, 32, 100)
train_step2(model, fake_dataset, 32, 100)
# torch.save(model.state_dict(), args['ToM_model_path'])
print("training finish")
args['ToM_model'] = model

# Finally, you can construct an ego agent and train it in the environment
ego = MyPPO(args)
ego.learn(total_timesteps=1000000)
