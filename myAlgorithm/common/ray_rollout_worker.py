import os.path
import pickle
import random
import time
import cloudpickle

import numpy as np
import ray
import redis
import torch
import torch as th
from flask import session
from fontTools.ttx import process
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

from gym import spaces

from myAlgorithm.ImplicitRewardPolicy.ImplicitReward import compute_reward_comm
from myAlgorithm.ImplicitRewardPolicy.ToMNet import make_fake_dataset, insert_dataset, ToMNet
from myAlgorithm.common.comm_agent_wrapper import SimpleCommunicativePartner
from pantheonrl.common.agents import OnPolicyAgent


@ray.remote(num_gpus=0.2)
class RolloutWorker:
    def __init__(
            self,
            worker_id: int,
            args,
            env_maker,
            policy,
            observation_space,
            action_space,
            device,
            gamma,
            use_sde,
            sde_sample_freq,
            dataset_seq_len,
            redis_config = None,
            # 新增
            tom_model_config = None,
            tom_model_weights = None
    ):
        """初始化Rollout Worker"""
        self.worker_id = worker_id
        self.policy = policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gamma = gamma
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.dataset_seq_len = dataset_seq_len
        # 初始化env
        self.env, self.env_for_fake_dataset = env_maker(args)
        # 初始化Tom模型
        self.tom_model = None
        if tom_model_config is not None:
            # 根据配置创建模型
            input_size = tom_model_config.get("input_size")
            hidden_size = tom_model_config.get("hidden_size")
            output_size = tom_model_config.get("output_size")

            if all([input_size, hidden_size, output_size]):
                self.tom_model = ToMNet(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size
                )

                # 如果有权重，加载它们
                if tom_model_weights is not None:
                    self.tom_model.load_state_dict(tom_model_weights)

        # 初始化Redis客户端
        if redis_config is None:
            redis_config = {
                "host": "localhost",
                "port": 6379,
                "db": 1,
                "password": None
            }

        self.redis_client = redis.Redis(
            host=redis_config["host"],
            port=redis_config["port"],
            db=redis_config["db"],
            password=redis_config["password"]
        )

        # 本地状态初始化
        self.dataset_item = []

        self.all_dataset_items = None
        self.hidden_old = None

    def collect_steps(
            self,
            process_id,
            # args,
            # env_maker,
            policy,
            last_obs,
            last_episode_starts,
            pre_computed_reward_comm,
            session_id,
            start_step,
            n_steps_to_collect,
            # 文件对象
            tom_model_weights,
    ):
        self.policy = policy
        """创建随机种子"""
        seed = process_id + 1000
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        """创建环境"""
        # env, env_for_fake_dataset = env_maker(args)
        if self.all_dataset_items is None:
            self.all_dataset_items = make_fake_dataset(self.env_for_fake_dataset, 320, self.dataset_seq_len)
        if self.tom_model is not None:
            self.tom_model.to(self.device)
            self.tom_model.load_state_dict(tom_model_weights)
            self.hidden_old, _ = self.tom_model(self.all_dataset_items[0])
        """收集指定数量的步骤"""
        self.policy.set_training_mode(False)
        current_step = 0

        # 初始状态
        _last_obs = self.env.reset()
        _last_episode_starts = last_episode_starts
        comm_rewards = []

        # 用于追踪当前worker的状态
        worker_state = {
            "worker_id": self.worker_id,
            "session_id": session_id,
            "steps_collected": 0,
            "dataset_items": [],
        }

        step_data_keys = []

        while current_step < n_steps_to_collect:
            if self.use_sde and self.sde_sample_freq > 0 and current_step % self.sde_sample_freq == 0:
                # 采样新的噪声矩阵
                self.policy.reset_noise(self.env.num_envs)

            with th.no_grad():
                # 转换为pytorch tensor
                obs_tensor = obs_as_tensor(_last_obs[0], self.device)
                actions, values, log_probs = self.policy(obs_tensor.unsqueeze(0))

            actions = actions.cpu().numpy()

            # 收到决策action和通信action，将其分别裁剪并将决策输入env
            action = np.array([actions[0]])
            action_comm = np.array([actions[1]])
            clipped_actions = action

            value, value_comm = th.split(values, 1, dim=0)
            log_prob, log_prob_comm = th.split(log_probs, 1, dim=0)

            # 裁剪动作以避免越界错误
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # 环境步骤
            new_obs, rewards, dones, infos = self.env.step([clipped_actions[0], action_comm[0]])
            partner_new_obs = new_obs[1]
            partner_action = new_obs[2]

            # 收集队友的state和action
            dataset_item_tensor = th.concat([
                th.FloatTensor(partner_new_obs),
                th.FloatTensor([partner_action])
            ])

            # 将数据添加到本地列表
            self.dataset_item.append(dataset_item_tensor)

            # 达到指定长度时，存入Redis
            if len(self.dataset_item) == self.dataset_seq_len:
                dataset_key = f"{session_id}:worker:{self.worker_id}:dataset:{current_step}"
                self.redis_client.set(
                    dataset_key,
                    pickle.dumps(self.dataset_item)
                )
                worker_state["dataset_items"].append(dataset_key)

                self.all_dataset_items = insert_dataset(self.all_dataset_items, self.dataset_item)
                self.dataset_item = []

            # reward_comm, hidden_old = compute_reward_comm(
            #     self.all_dataset_items,
            #     self.hidden_old,
            #     self.tom_model
            # )
            # self.hidden_old = hidden_old
            # 使用预计算的通信奖励
            reward_comm = pre_computed_reward_comm
            comm_rewards.append(reward_comm)
            if dones[0]:
                ep_rew_comm = sum(comm_rewards)
                infos[0]['episode']['r_c'] = round(ep_rew_comm, 6)
                comm_rewards = []
            # 处理超时情况
            for idx, done in enumerate(dones):
                if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # 将步骤数据存入Redis
            step_data = {
                "last_obs": _last_obs[0],
                "action": action,
                "rewards": rewards,
                "last_episode_starts": _last_episode_starts,
                "value": value.detach(),
                "log_prob": log_prob.detach(),
                "action_comm": np.array(action_comm[0]),
                "reward_comm": np.array(reward_comm),
                "log_prob_comm": log_prob_comm.detach(),
                "value_comm": value_comm.detach(),
                "step": start_step + current_step,
                "worker_id": self.worker_id,
                "infos": infos
            }

            step_key = f"{session_id}:worker:{self.worker_id}:step:{current_step}"
            self.redis_client.set(step_key, pickle.dumps(step_data))
            step_data_keys.append(step_key)

            # 更新状态
            _last_obs = new_obs
            _last_episode_starts = dones
            current_step += 1

        # 计算最后一个时间步的价值
        with th.no_grad():
            final_value, final_value_comm = self.policy.predict_values(obs_as_tensor(new_obs[0], self.device))

        # 返回worker状态和最终值
        final_state = {
            "last_obs": new_obs[0],
            "last_episode_starts": dones,
            "final_value": final_value.detach(),
            "final_value_comm": final_value_comm.detach(),
        }

        final_key = f"{session_id}:worker:{self.worker_id}:final"
        self.redis_client.set(final_key, pickle.dumps(final_state))

        # 更新worker状态
        worker_state["steps_collected"] = current_step
        worker_state["step_keys"] = step_data_keys
        worker_state["final_key"] = final_key

        return worker_state