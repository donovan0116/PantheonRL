import json

import numpy as np
import os

import yaml
import gym
from stable_baselines3 import PPO

from myAlgorithm.common.myEnv import InteractiveOvercookedEnv
from overcookedgym.overcooked_utils import LAYOUT_LIST
from pantheonrl.common.agents import OnPolicyAgent
import redis


class SimpleCommunicativePartner:
    """
    简化版通信智能体，专注于：
    1. 从文件读取通信策略
    2. 接收环境通信信号
    3. 返回通信选择
    该包装类负责包装partner智能体，将底层的perdict实现，并在外层包装一个通信处理策略。
    该类has-a redis句柄，每次predict的时候，检查一个量查看是否环境那边发起了通信，若发起了，则获取通信选项，并调用包装类中的handle方法，
    在handle方法中选择通信内容并反馈给redis句柄
    """

    def __init__(
            self,
            model,
            strategy_file_path="communication_strategy.txt"
    ):
        """
        初始化简化版通信智能体

        Args:
            model: 基础PPO模型
            strategy_file_path: 通信策略文件路径
        """
        self.model = model
        self.strategy_file_path = strategy_file_path

        # 默认通信策略
        self.communication_strategy = {
            'preference': 'balanced',  # 'speed', 'collaboration', 'balanced'
        }

        # 尝试从文件加载策略
        self._load_strategy()
        self.redis_handler = redis.Redis(host='127.0.0.1', port=6379, db=0)
        # self.comm_space_address = None

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        获取模型的动作预测
        """
        return self.model.predict(observation, state, episode_start, deterministic)

    def handle_communication(self, comm_options):
        """
        处理来自环境的通信请求

        Args:
            comm_options: 通信选项字典，包含 'options' 列表

        Returns:
            int: 选择的通信选项索引
        """
        # 重新加载最新的通信策略
        self._load_strategy()

        # 根据偏好选择通信选项
        preferred_strategy = self.communication_strategy['preference']

        # 将选项描述映射到策略类型
        strategy_mapping = {
            '优先速度执行': 'speed',
            '优先协作执行': 'cooperation',
            '平衡策略执行': 'balanced'
        }

        # 对选项进行评分
        option_scores = []
        for option in comm_options['options']:
            score = 0
            option_strategy = strategy_mapping.get(option['description'], '')

            if option_strategy == preferred_strategy:
                score = 2  # 最匹配的选项
            elif option_strategy in preferred_strategy or preferred_strategy in option_strategy:
                score = 1  # 部分匹配的选项
            option_scores.append(score)

        # 选择得分最高的选项，如果有多个相同得分则随机选择
        max_score = max(option_scores)
        best_indices = [i for i, score in enumerate(option_scores) if score == max_score]
        selected_index = np.random.choice(best_indices)

        # 记录选择的选项
        selected_option = comm_options['options'][selected_index]
        self._log_selection(selected_option)

        # 返回选择的选项ID（从1开始的索引）
        return selected_option['id'] - 1

    def get_action(self, ob):
        if self.redis_handler.get('communication_request'):
            comm_options = yaml.safe_load(self.redis_handler.get('communication_options'))
            selected_index = self.handle_communication(comm_options)
            self.redis_handler.set('communication_choice', json.dumps(selected_index))
            self.redis_handler.delete('communication_request')
            self.redis_handler.delete('communication_options')
        # if self.comm_space_address:
        #     with open(self.comm_space_address, 'r') as f:
        #         config_ = yaml.safe_load(f)
        #     if config_['communication_request']:
        #         comm_options = yaml.safe_load(config_['communication_options'])
        #         selected_index = self.handle_communication(comm_options)
        #         config_['communication_choice'] = selected_index
        #         with open(self.comm_space_address, 'w') as f:
        #             yaml.dump(config_, f)
        #             config_['communication_request'] = False
        #             config_['communication_options'] = None


        if hasattr(self.model, 'get_action'):
            return self.model.get_action(ob)

    def update(self, reward, done):

        if hasattr(self.model, 'update'):
            return self.model.update(reward, done)

    def _load_strategy(self):
        """
        从文件加载通信策略
        """
        try:
            if os.path.exists(self.strategy_file_path):
                with open(self.strategy_file_path, 'r') as f:
                    strategy = f.read().strip().lower()
                    if strategy in ['speed', 'collaboration', 'balanced']:
                        self.communication_strategy['preference'] = strategy
        except Exception as e:
            print(f"加载通信策略时出错: {e}")

    def _log_selection(self, selected_option):
        """
        记录所选择的通信选项
        """
        try:
            log_file = "communication_log.txt"
            with open(log_file, 'a') as f:
                f.write(f"选择了通信选项: {selected_option['description']}\n")
        except Exception as e:
            print(f"记录通信选择时出错: {e}")

    def reset(self):
        """
        当环境重置时重置智能体状态
        """
        pass  # 简化版没有需要重置的状态

