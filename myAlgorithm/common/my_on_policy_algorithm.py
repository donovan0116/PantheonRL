import json
import os
import pickle
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym

import numpy as np
import torch
import torch as th
import yaml
from anyio import value
from gym import spaces
from stable_baselines3 import PPO

from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
import multiprocessing
import multiprocessing as mp
from torch.multiprocessing import Queue
from tensorflow.python.ops.numpy_ops import ndarray

import redis

from overcookedgym.overcooked_utils import LAYOUT_LIST
from pantheonrl.common.agents import OnPolicyAgent
# 引我自己的buffer
from .my_buffers import RolloutBuffer, DictRolloutBuffer

from .communicateUtils.comm_interact import comm
from ..ImplicitRewardPolicy.ToMNet import make_fake_dataset, insert_dataset, ToMNet, train_step1, train_step2
from ..ImplicitRewardPolicy.ImplicitReward import compute_reward_comm

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class MyOnPolicyAlgorithm(BaseAlgorithm):

    def __init__(
            self,
            policy: Union[str, Type[ActorCriticPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule],
            n_steps: int,
            gamma: float,
            gae_lambda: float,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            tensorboard_log: Optional[str] = None,
            monitor_wrapper: bool = True,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
            supported_action_spaces: Optional[Tuple[spaces.Space, ...]] = None,
            dataset_data_num: int = 3200,
            dataset_seq_len: int = 10,
            tom_model=None,
            fake_dataset_ = None
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.dataset_data_num = dataset_data_num
        self.dataset_seq_len = dataset_seq_len
        self.dataset = fake_dataset_
        self.dataset_item = []
        self.tom_model = tom_model
        self.hidden_old, _ = tom_model(self.dataset[0])
        self.comm_rewards = []
        if tom_model is None:
            self.tom_model = ToMNet(
                input_size=env.observation_space.shape[0] + 1,
                hidden_size=[64, 256, env.observation_space.shape[0] + 1],
                output_size=env.observation_space.shape[0] + 1 * 10)

        if _init_setup_model:
            self._setup_model()
        self.n_workers = 1

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, spaces.Dict) else RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def collect_rollouts(
            self,
            env,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
            n_workers: int = 1,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :param n_workers: Number of workers for distribute sampler
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs[0], self.device)
                actions, values, log_probs = self.policy(obs_tensor.unsqueeze(0))
            actions = actions.cpu().numpy()

            # 收到决策action和通信action，将其分别裁剪并将决策输入env
            action = np.array([actions[0]])
            action_comm = np.array([actions[1]])
            clipped_actions = action

            value, value_comm = th.split(values, 1, dim=0)

            log_prob, log_prob_comm = th.split(log_probs, 1, dim=0)

            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step([[clipped_actions[0], action_comm[0]]])
            partner_new_obs = new_obs.copy()
            partner_new_obs[0] = new_obs[0][1]
            partner_action = new_obs.copy()
            partner_action[0] = new_obs[0][2]
            new_obs[0] = new_obs[0][0]
            # 已经生成队友的state和action，收集seq次组成一个tensor将其纳入dataset中
            self.dataset_item.append(
                torch.concat(
                    [
                        torch.FloatTensor(partner_new_obs[0]),
                        torch.FloatTensor([partner_action[0]])
                    ]
                )
            )
            if len(self.dataset_item) == self.dataset_seq_len:
                self.dataset = insert_dataset(self.dataset, self.dataset_item)
                self.dataset_item = []
            # 为了测试全流程，暂时设定reward_comm和reward相等
            reward_comm, hidden_old = compute_reward_comm(self.dataset, self.hidden_old, self.tom_model)
            self.hidden_old = hidden_old
            # print(f"reward_comm: {reward_comm.item()}")
            # reward_comm = rewards
            self.comm_rewards.append(reward_comm)
            if dones:
                ep_rew_comm = sum(self.comm_rewards)
                infos[0]['episode']['r_c'] = round(ep_rew_comm, 6)
                self.comm_rewards = []

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
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

            rollout_buffer.add(self._last_obs, action, rewards, self._last_episode_starts, value, log_prob, action_comm,
                               reward_comm, log_prob_comm, value_comm)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            value, value_comm = self.policy.predict_values(obs_as_tensor(new_obs[0], self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=value, dones=dones)
        rollout_buffer.compute_returns_and_advantage_comm(last_values=value_comm, dones=dones)

        callback.on_rollout_end()

        return True

    def collect_rollouts_multiprocess(
            self,
            env,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
            n_workers: int = 4,
            redis_config: Dict[str, Any] = None,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``
        with multiple processes.

        :param env: The training environment
        :param callback: Callback that will be called at each step
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :param n_workers: Number of worker processes
        :param redis_config: Configuration for Redis connection
        :return: True if function returned with at least `n_rollout_steps` collected,
            False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # 默认Redis配置
        if redis_config is None:
            redis_config = {
                'host': 'localhost',
                'port': 6379,
                'db': 1,
                'decode_responses': False  # 对于二进制数据设为False
            }

        # 用于JSON序列化的Redis客户端
        redis_json_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            decode_responses=True  # 对于JSON设为True
        )

        # 用于二进制数据的Redis客户端
        redis_binary_client = redis.Redis(
            host=redis_config['host'],
            port=redis_config['port'],
            db=redis_config['db'],
            decode_responses=False
        )

        # 清除之前的数据
        redis_json_client.flushdb()

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        # 准备回调
        callback.on_rollout_start()

        # 重置缓冲区
        rollout_buffer.reset()

        # 计算每个worker应该收集的步数
        steps_per_worker = (n_rollout_steps + n_workers - 1) // n_workers

        # 提前计算一次reward_comm，所有worker都使用这个值
        reward_comm, hidden_old = compute_reward_comm(self.dataset, self.hidden_old, self.tom_model)
        self.hidden_old = hidden_old

        # 为每个worker创建一个环境副本
        # 注意：这里假设环境可以被复制，如果不能，需要修改策略
        worker_envs = [env for _ in range(n_workers)]

        # 创建进程间同步对象
        process_barrier = mp.Barrier(n_workers)
        main_process_ready_event = mp.Event()
        worker_ready_queue = Queue()

        # 创建并启动worker进程
        processes = []
        for i in range(n_workers):
            p = mp.Process(
                target=worker_process,
                args=(
                    i,
                    worker_envs[i],
                    self.policy,
                    self.device,
                    self.action_space,
                    steps_per_worker,
                    redis_config,
                    np.random.randint(0, 1000000),  # global_seed
                    self.use_sde,
                    self.sde_sample_freq,
                    self.gamma,
                    self.dataset_seq_len,
                    reward_comm.item() if isinstance(reward_comm, th.Tensor) else reward_comm,
                    process_barrier,
                    worker_ready_queue,
                    main_process_ready_event,
                )
            )
            p.daemon = True
            p.start()
            processes.append(p)

        # 等待所有worker准备就绪
        worker_ids = []
        for _ in range(n_workers):
            worker_id = worker_ready_queue.get()
            worker_ids.append(worker_id)

        # 通知所有worker可以开始
        main_process_ready_event.set()

        # 等待所有worker完成
        all_completed = False
        while not all_completed:
            time.sleep(0.1)  # 避免忙等
            completed_count = 0
            for worker_id in worker_ids:
                if redis_json_client.get(f"{worker_id}:completed") == "1":
                    completed_count += 1

            if completed_count == n_workers:
                all_completed = True

        # 收集所有rollout数据并填充buffer
        rollout_data_keys = []
        dataset_keys = []

        # 查找所有相关的键
        for worker_id in worker_ids:
            # 获取所有rollout键
            worker_rollout_keys = redis_json_client.keys(f"{worker_id}:rollout:*")
            rollout_data_keys.extend(worker_rollout_keys)

            # 获取所有dataset序列键
            worker_dataset_keys = redis_json_client.keys(f"{worker_id}:dataset_seq:*")
            dataset_keys.extend(worker_dataset_keys)

        # 排序键以确保按正确顺序处理
        rollout_data_keys.sort(key=lambda x: int(x.split(":")[-1]))

        # 处理dataset更新
        for dataset_key in dataset_keys:
            dataset_seq_data = json.loads(redis_json_client.get(dataset_key))
            # 将JSON数据转换回tensor
            tensor_seq = []
            for item in dataset_seq_data:
                partner_obs = torch.FloatTensor(item['partner_obs'])
                partner_action = torch.FloatTensor([item['partner_action']])
                tensor_seq.append(torch.concat([partner_obs, partner_action]))

            # 更新主进程的dataset
            self.dataset = insert_dataset(self.dataset, tensor_seq)

        # 填充rollout buffer
        for rollout_key in rollout_data_keys:
            data = json.loads(redis_json_client.get(rollout_key))

            # 转换回适当的数据类型
            last_obs = np.array(data['last_obs']) if isinstance(data['last_obs'][0], list) else data['last_obs']
            action = np.array(data['action'])
            rewards = np.array(data['rewards'])
            last_episode_starts = np.array(data['last_episode_starts'])
            value = th.tensor(data['value'])
            log_prob = th.tensor(data['log_prob'])
            action_comm = np.array(data['action_comm'])
            reward_comm = data['reward_comm']
            log_prob_comm = th.tensor(data['log_prob_comm'])
            value_comm = th.tensor(data['value_comm'])

            # 添加到rollout buffer
            rollout_buffer.add(
                last_obs, action, rewards, last_episode_starts,
                value, log_prob, action_comm, reward_comm,
                log_prob_comm, value_comm
            )

            # 更新主进程的状态
            self._last_obs = np.array(data['new_obs']) if isinstance(data['new_obs'][0], list) else data['new_obs']
            self._last_episode_starts = np.array(data['dones'])

            # 更新回调
            # 给access to local variables
            locals_dict = {
                'self': self,
                'env': env,
                'callback': callback,
                'rollout_buffer': rollout_buffer,
                'n_rollout_steps': n_rollout_steps,
                'infos': [{}],  # 这里可能需要从worker获取更详细的infos
                'dones': np.array(data['dones']),
                'rewards': rewards,
                'new_obs': self._last_obs,
                'n_steps': int(rollout_key.split(":")[-1]),
            }
            callback.update_locals(locals_dict)
            if callback.on_step() is False:
                # 清理进程
                for p in processes:
                    p.terminate()
                return False

        # 获取最后一个timestep的值
        last_worker = worker_ids[-1]
        final_data = json.loads(redis_json_client.get(f"{last_worker}:final"))
        final_value = th.tensor(final_data['final_value'])
        final_value_comm = th.tensor(final_data['final_value_comm'])
        final_dones = np.array(final_data['final_dones'])

        # 计算returns和advantages
        rollout_buffer.compute_returns_and_advantage(last_values=final_value, dones=final_dones)
        rollout_buffer.compute_returns_and_advantage_comm(last_values=final_value_comm, dones=final_dones)

        callback.on_rollout_end()

        # 清理进程
        for p in processes:
            p.join()

        # 清理Redis数据
        redis_json_client.flushdb()

        return True


    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
            self: SelfOnPolicyAlgorithm,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 1,
            tb_log_name: str = "OnPolicyAlgorithm",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            # continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
            #                                                       n_rollout_steps=self.n_steps,
            #                                                       n_workers=self.n_workers)
            continue_training = self.collect_rollouts_multiprocess(self.env, callback, self.rollout_buffer,
                                                                 n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            if self.num_timesteps % 32 == 0:
                train_step1(self.tom_model, self.dataset, 32, 100)
                train_step2(self.tom_model, self.dataset, 32, 100)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean",
                                       safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_rew_comm_mean",
                                       safe_mean([ep_info["r_c"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean",
                                       safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

def worker_process(
        process_id: int,
        env,
        policy,
        device,
        action_space,
        steps_per_worker: int,
        redis_config: Dict[str, Any],
        global_seed: int,
        use_sde: bool,
        sde_sample_freq: int,
        gamma: float,
        dataset_seq_len: int,
        pre_computed_reward_comm: float,
        process_barrier: mp.Barrier,
        worker_ready_queue: Queue,
        main_process_ready_event: mp.Event,
):
    """
    Worker process to collect rollouts

    :param process_id: ID of this worker process
    :param env: The training environment (copied for each worker)
    :param policy: The policy to use for rollout collection
    :param device: Device to use for tensor operations
    :param action_space: Action space of the environment
    :param steps_per_worker: Number of steps to collect per worker
    :param redis_config: Configuration for Redis connection
    :param global_seed: Seed for random number generators
    :param use_sde: Whether to use state-dependent exploration
    :param sde_sample_freq: Frequency of noise matrix resampling
    :param gamma: Discount factor
    :param dataset_seq_len: Length of dataset sequence
    :param pre_computed_reward_comm: Pre-computed communication reward
    :param process_barrier: Barrier for process synchronization
    :param worker_ready_queue: Queue to signal worker is ready
    :param main_process_ready_event: Event to wait for main process
    """
    # Set worker seed for reproducibility
    np.random.seed(global_seed + process_id)
    th.manual_seed(global_seed + process_id)

    # 连接到Redis
    r = redis.Redis(**redis_config)

    # 初始化本地状态
    worker_last_obs = env.reset()
    worker_last_episode_starts = np.ones((env.num_envs,), dtype=bool)
    dataset_item = []
    comm_rewards = []
    n_steps = 0

    # 生成唯一的worker ID
    worker_id = f"worker_{process_id}_{uuid.uuid4().hex[:8]}"

    # Switch to eval mode (this affects batch norm / dropout)
    policy.set_training_mode(False)

    # Sample new weights for the state dependent exploration
    if use_sde:
        policy.reset_noise(env.num_envs)

    # 通知主进程worker已准备好
    worker_ready_queue.put(worker_id)

    # 等待主进程准备完成
    main_process_ready_event.wait()

    # 同步所有worker开始工作
    process_barrier.wait()

    # 开始收集rollout
    while n_steps < steps_per_worker:
        if use_sde and sde_sample_freq > 0 and n_steps % sde_sample_freq == 0:
            # Sample a new noise matrix
            policy.reset_noise(env.num_envs)

        with th.no_grad():
            # Convert to pytorch tensor
            obs_tensor = obs_as_tensor(worker_last_obs[0], device)
            actions, values, log_probs = policy(obs_tensor.unsqueeze(0))

        actions = actions.cpu().numpy()

        # 收到决策action和通信action，将其分别裁剪并将决策输入env
        action = np.array([actions[0]])
        action_comm = np.array([actions[1]])
        clipped_actions = action

        value, value_comm = th.split(values, 1, dim=0)
        log_prob, log_prob_comm = th.split(log_probs, 1, dim=0)

        # Clip the actions to avoid out of bound error
        if isinstance(action_space, spaces.Box):
            clipped_actions = np.clip(actions, action_space.low, action_space.high)

        new_obs, rewards, dones, infos = env.step([[clipped_actions[0], action_comm[0]]])
        partner_new_obs = new_obs.copy()
        partner_new_obs[0] = new_obs[0][1]
        partner_action = new_obs.copy()
        partner_action[0] = new_obs[0][2]
        new_obs[0] = new_obs[0][0]

        # 收集partner的state和action，但不直接进行tensor concat，而是存入Redis
        partner_data = {
            'partner_obs': partner_new_obs[0].tolist(),
            'partner_action': partner_action[0].tolist()
        }

        dataset_item.append(partner_data)

        if len(dataset_item) == dataset_seq_len:
            # 将序列存入Redis, 主进程会负责更新dataset
            dataset_key = f"{worker_id}:dataset_seq:{n_steps}"
            r.set(dataset_key, json.dumps(dataset_item))
            dataset_item = []

        # 使用预计算的reward_comm
        reward_comm = pre_computed_reward_comm
        comm_rewards.append(reward_comm)

        if dones:
            ep_rew_comm = sum(comm_rewards)
            infos[0]['episode']['r_c'] = round(ep_rew_comm, 6)
            comm_rewards = []

        n_steps += 1

        if isinstance(action_space, spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstraping with value function
        for idx, done in enumerate(dones):
            if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = policy.predict_values(terminal_obs)[0]
                rewards[idx] += gamma * terminal_value

        # 将rollout数据保存到Redis
        rollout_data = {
            'last_obs': worker_last_obs.tolist() if isinstance(worker_last_obs, np.ndarray) else [
                o.tolist() if isinstance(o, np.ndarray) else o for o in worker_last_obs],
            'action': action.tolist(),
            'rewards': rewards.tolist() if isinstance(rewards, np.ndarray) else rewards,
            'last_episode_starts': worker_last_episode_starts.tolist(),
            'value': value.cpu().numpy().tolist(),
            'log_prob': log_prob.cpu().numpy().tolist(),
            'action_comm': action_comm.tolist(),
            'reward_comm': reward_comm if not isinstance(reward_comm, (
                np.ndarray, th.Tensor)) else reward_comm.tolist() if isinstance(reward_comm,
                                                                                np.ndarray) else reward_comm.cpu().numpy().tolist(),
            'log_prob_comm': log_prob_comm.cpu().numpy().tolist(),
            'value_comm': value_comm.cpu().numpy().tolist(),
            'new_obs': new_obs.tolist() if isinstance(new_obs, np.ndarray) else [
                o.tolist() if isinstance(o, np.ndarray) else o for o in new_obs],
            'dones': dones.tolist() if isinstance(dones, np.ndarray) else dones,
        }

        # 存储到Redis
        rollout_key = f"{worker_id}:rollout:{n_steps}"
        r.set(rollout_key, json.dumps(rollout_data))

        worker_last_obs = new_obs
        worker_last_episode_starts = dones

    # 将最后一个timestep的值计算并存入Redis
    with th.no_grad():
        final_value, final_value_comm = policy.predict_values(obs_as_tensor(new_obs[0], device))

    final_data = {
        'final_value': final_value.cpu().numpy().tolist(),
        'final_value_comm': final_value_comm.cpu().numpy().tolist(),
        'final_dones': dones.tolist() if isinstance(dones, np.ndarray) else dones,
    }

    r.set(f"{worker_id}:final", json.dumps(final_data))

    # 通知完成
    r.set(f"{worker_id}:completed", "1")

    # 等待所有worker完成
    process_barrier.wait()
