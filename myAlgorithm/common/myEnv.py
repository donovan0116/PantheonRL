import json

import gym
import numpy as np
import redis
import torch
import yaml
from stable_baselines3 import PPO

from pantheonrl.common.agents import OnPolicyAgent
from overcookedgym.overcooked_utils import LAYOUT_LIST
from myAlgorithm.ImplicitRewardPolicy.ToMNet import *

class InteractiveOvercookedEnv(gym.Wrapper):
    def __init__(self, env):
        """
        初始化交互式Overcooked环境

        :param env: 原始环境实例
        """
        super().__init__(env)

        # 通信代理
        self.agents = {
            'ego': None,
            'partner': None
        }

        # 通信状态
        self.communication_state = {
            'is_communicating': False,
            'communicator': None,  # 'ego' 或 'partner'
            'communication_type': None,
            'communication_options': [],
            'current_choice': None
        }

        # 通信选项定义
        self.communication_options = [
            {
                'id': 'speed_priority',
                'description': '优先速度执行',
                'strategy_modification': {
                    'priority': 'speed',
                    'weight': 0.7
                }
            },
            {
                'id': 'collaboration_priority',
                'description': '优先协作执行',
                'strategy_modification': {
                    'priority': 'collaboration',
                    'weight': 0.7
                }
            },
            {
                'id': 'balanced_strategy',
                'description': '平衡策略执行',
                'strategy_modification': {
                    'priority': 'balanced',
                    'weight': 0.5
                }
            }
        ]
        # self.redis_handler = redis.Redis(host='127.0.0.1', port=6379, db=0)
        self.comm_space_address = './comm_space.yaml'

    def add_agent(self, agent, role='ego'):
        """
        添加智能体

        :param agent: 智能体实例
        :param role: 智能体角色 ('ego' 或 'partner')
        """
        if role not in ['ego', 'partner']:
            raise ValueError("角色必须是 'ego' 或 'partner'")

        self.agents[role] = agent

        # 如果底层环境支持，也添加到底层环境
        if hasattr(self.env, 'add_partner_agent') and role == 'partner':
            self.env.add_partner_agent(agent)

    def reset(self, **kwargs):
        """
        重置环境

        :param kwargs: 额外的重置参数
        :return: 重置后的初始状态
        """
        # 重置通信状态
        self.communication_state = {
            'is_communicating': False,
            'communicator': None,
            'communication_type': None,
            'communication_options': [],
            'current_choice': None
        }

        # 重置底层环境
        initial_state = self.env.reset(**kwargs)

        # 重置智能体
        for role, agent in self.agents.items():
            if agent is not None and hasattr(agent, 'reset'):
                agent.reset()

        return [initial_state, np.copy(initial_state), -1]

    def _should_communicate(self) -> bool:
        """
        决定是否需要发起通信

        :return: 是否发起通信
        """
        # 通信触发策略：随机触发 + 性能相关触发
        communication_triggers = [
            np.random.random() < 0.1,  # 10%随机触发
            # 可以添加更多触发条件，如性能阈值
        ]

        return any(communication_triggers)

    def _initiate_communication(self, communicator: str = None):
        """
        发起通信过程，返回通信状态及选项的字典
        """
        if communicator is None:
            # 随机选择通信发起者
            communicator = np.random.choice(list(self.agents.keys()))

        self.communication_state.update({
            'is_communicating': True,
            'communicator': communicator,
            'communication_options': self.communication_options
        })

        print(f"{communicator.upper()}智能体请求通信，通信选项：")
        for idx, option in enumerate(self.communication_options):
            print(f"{idx + 1}. {option['description']}")

        # 创建要返回的字典对象
        return_dict = {
            'communicator': communicator.upper(),
            'message': f"{communicator.upper()}智能体请求通信",
            'options': []
        }

        # 添加通信选项到返回字典
        for idx, option in enumerate(self.communication_options):
            return_dict['options'].append({
                'id': idx + 1,
                'description': option['description']
            })

        return return_dict

    def handle_communication_response(self, choice_index: int):
        """
        处理通信响应

        :param choice_index: 选择的选项索引
        """
        if not self.communication_state['is_communicating']:
            raise ValueError("当前非通信状态")

        if 0 <= choice_index < len(self.communication_options):
            choice = self.communication_options[choice_index]

            # 更新通信状态
            self.communication_state.update({
                'current_choice': choice,
                'is_communicating': False
            })

            # 执行策略修改
            self._apply_strategy_modification(choice)
        else:
            raise ValueError("无效的选项")

    def _apply_strategy_modification(self, choice):
        """
        应用策略修改

        :param choice: 选择的通信选项
        """
        modification = choice['strategy_modification']
        communicator = self.communication_state['communicator']

        # 根据通信发起者和策略修改执行相应操作
        print(f"{communicator.upper()}智能体应用策略：{choice['description']}")

        # 具体的策略修改逻辑（需要根据实际算法实现）
        agent = self.agents[communicator]
        if agent is not None:
            # 示例：修改agent的某些超参数
            if hasattr(agent, 'modify_strategy'):
                agent.modify_strategy(modification)

    def step(self, action, action_comm=0.0):
        """
        执行环境步骤

        :param action: ego的动作
        :param action_comm: ego的通信决策
        :return: 下一个状态、奖励、是否结束、额外信息
        """

        # 执行底层环境步骤
        # Q：为什么加这个条件判断呢？
        # A：因为make_fake_dataset的时候需要调用这里，但是那个时候还不涉及到action_comm， 因此传入的action是float
        # 在step的时候，action_comm存在，因此需要处理
        if isinstance(action, list):
            action_comm = action[1]
            action = action[0]
        # if action_comm:
        #     communication_options = self._initiate_communication('ego')
        #     self.redis_handler.set('communication_request', '1')
        #     self.redis_handler.set('communication_options', json.dumps(communication_options))
        #
        # next_state, reward, done, info = self.env.step(action)
        # partner_state = next_state[1]
        # partner_action = next_state[2]
        #
        # # partner模拟通信响应（实际应由人类或智能体决策）
        # if self.communication_state['is_communicating']:
        #     communication_choice = yaml.safe_load(self.redis_handler.get('communication_choice'))
        #     if communication_choice:
        #         self.handle_communication_response(communication_choice)
        #         self.redis_handler.delete('communication_choice')
        if action_comm:
            communication_options = self._initiate_communication('ego')
            with open(self.comm_space_address, 'r') as f:
                config = yaml.safe_load(f)

            config['communication_request'] = '1'
            config['communication_options'] = communication_options
        next_state, reward, done, info = self.env.step(action)
        partner_state = next_state[1]
        partner_action = next_state[2]

        if self.communication_state['is_communicating']:
            with open(self.comm_space_address, 'r') as f:
                config = yaml.safe_load(f)
            communication_choice = config['communication_choice']
            if communication_choice:
                self.handle_communication_response(communication_choice)
                with open(self.comm_space_address, 'r') as f:
                    config = yaml.safe_load(f)
                config['communication_choice'] = None
                with open(self.comm_space_address, 'w') as f:
                    yaml.dump(config, f)

        return next_state, reward, done, info

    def render(self, **kwargs):
        """
        渲染环境

        :param kwargs: 渲染参数
        :return: 渲染结果
        """
        return self.env.render(**kwargs)


# 使用示例
def main():
    # 创建基础环境
    with open('../../myAlgorithm/config/my_ppo_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    args = config
    base_env = gym.make(args['env']['id'], layout_name=args['env']['layout'])

    # 包装交互式环境
    env = InteractiveOvercookedEnv(base_env)

    # 创建智能体
    ego_agent = PPO('MlpPolicy', env, verbose=1)
    partner_agent = OnPolicyAgent(PPO('MlpPolicy', env, verbose=1))

    # 添加智能体
    env.add_agent(ego_agent, 'ego')
    env.add_agent(partner_agent, 'partner')

    # 训练过程
    state = env.reset()
    for _ in range(1000):
        # Ego agent选择动作
        action, _ = ego_agent.predict(state)
        state, reward, done, _ = env.step(action)

        if done:
            break


if __name__ == "__main__":
    main()