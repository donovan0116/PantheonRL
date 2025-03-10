"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""

import gym
import numpy as np
import redis
import torch
import yaml
from stable_baselines3 import PPO

from myAlgorithm.MyPPO import MyPPO
from myAlgorithm.common.comm_agent_wrapper import SimpleCommunicativePartner
from myAlgorithm.common.myEnv import InteractiveOvercookedEnv
from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
from myAlgorithm.ImplicitRewardPolicy.ToMNet import *

layout = 'simple'
assert layout in LAYOUT_LIST

with open('../myAlgorithm/config/my_ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

args = config

base_env = gym.make(args['env']['id'], layout_name=args['env']['layout'])
env = InteractiveOvercookedEnv(base_env)
args['env'] = env

r = redis.Redis(host='127.0.0.1', port=6379, db=0)
r.flushdb()

partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=0))
partner = SimpleCommunicativePartner(partner)
env.add_agent(partner, 'partner')

# hyper parameters
input_size = env.observation_space.shape[0] + 1
seq_len = 2
args['seq_len'] = seq_len
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
fake_dataset = make_fake_dataset(env, 320, 2)
args['fake_dataset'] = fake_dataset
train_step1(model, fake_dataset, 32, 100)
train_step2(model, fake_dataset, 32, 100)
# torch.save(model.state_dict(), args['ToM_model_path'])
print("training finish")
args['ToM_model'] = model

ego = MyPPO(args)
env.add_agent(ego, 'ego')
r.flushdb()
ego.learn(total_timesteps=1000000)
ego.save('./ego_model/')
