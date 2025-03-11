import dill
import gym
import yaml

from myAlgorithm.common.myEnv import InteractiveOvercookedEnv

with open('../../myAlgorithm/config/my_ppo_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

args = config

base_env = gym.make(args['env']['id'], layout_name=args['env']['layout'])
env = InteractiveOvercookedEnv(base_env)
try:
    dill.dumps(env)  # 使用更严格的序列化库测试
except Exception as e:
    print(f"序列化失败: {e}")
