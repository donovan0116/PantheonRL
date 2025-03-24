"""
This is a simple example training script for PantheonRL.

To run this script, remember to first install overcooked
via the instructions in the README.md
"""
import pickle
import cloudpickle

import gym
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST

layout = 'simple'
assert layout in LAYOUT_LIST

# Since pantheonrl's MultiAgentEnv is a subclass of the gym Env, you can
# register an environment and construct it using gym.make.
env = gym.make('OvercookedMultiEnv-v0', layout_name=layout)

# Before training your ego agent, you first need to add your partner agents
# to the environment. You can create adaptive partner agents using
# OnPolicyAgent (for PPO/A2C) or OffPolicyAgent (for DQN/SAC). If you set
# verbose to true for these agents, you can also see their learning progress
partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=1))
env.add_partner_agent(partner)
with open("./env.pkl", "wb") as f:
    cloudpickle.dump(env, f)

with open("./env.pkl", "rb") as f:
    my_env = cloudpickle.load(f)

# Finally, you can construct an ego agent and train it in the environment
ego = PPO('MlpPolicy', env, verbose=1)
ego.learn(total_timesteps=10000)

