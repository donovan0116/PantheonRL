import json
import os
import pickle
import cloudpickle
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
from click.core import batch
from gym import spaces
from stable_baselines3 import PPO

from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import (
    VecEnv, DummyVecEnv, is_vecenv_wrapped, VecTransposeImage)
import multiprocessing
import multiprocessing as mp
from torch.multiprocessing import Queue
from tensorflow.python.ops.numpy_ops import ndarray
from stable_baselines3.common.preprocessing import check_for_nested_spaces, is_image_space, is_image_space_channels_first

import redis
import ray

from overcookedgym.overcooked_utils import LAYOUT_LIST
from pantheonrl.common.agents import OnPolicyAgent
from .comm_agent_wrapper import SimpleCommunicativePartner
from .myEnv import InteractiveOvercookedEnv
# 引我自己的buffer
from .my_buffers import RolloutBuffer, DictRolloutBuffer

from .communicateUtils.comm_interact import comm
from .ray_rollout_worker import RolloutWorker
from ..ImplicitRewardPolicy.ToMNet import make_fake_dataset, insert_dataset, ToMNet, train_step1, train_step2, \
    pre_process_dataset
from ..ImplicitRewardPolicy.ImplicitReward import compute_reward_comm

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


def env_factory(args):
    args['create_from_env_factory'] = True
    env = gym.make(args['env']['id'], layout_name=args['env']['layout'])
    env = InteractiveOvercookedEnv(env)
    args['env'] = env
    # # 加载baseline
    # weight_path = os.path.expanduser('~/PycharmProjects/myPRL/PantheonRL/examples/models/partner_model.zip')
    # pretrained_partner = PPO.load(weight_path, env=env)
    # for param in pretrained_partner.policy.parameters():
    #     param.requires_grad = False
    # partner = OnPolicyAgent(pretrained_partner)

    partner = OnPolicyAgent(PPO('MlpPolicy', env, verbose=0))

    partner = SimpleCommunicativePartner(partner)
    env.add_agent(partner, 'partner')
    from ..my_ppo import MyPPO
    ego = MyPPO(args)
    env.add_agent(ego, 'ego')
    return my_wrap_env(env, 1), env


def my_wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
    if not isinstance(env, VecEnv):
        if not is_wrapped(env, Monitor) and monitor_wrapper:
            if verbose >= 1:
                print("Wrapping the env with a `Monitor` wrapper")
            env = Monitor(env)
        if verbose >= 1:
            print("Wrapping the env in a DummyVecEnv.")
        env = DummyVecEnv([lambda: env])

    # Make sure that dict-spaces are not nested (not supported)
    check_for_nested_spaces(env.observation_space)

    if not is_vecenv_wrapped(env, VecTransposeImage):
        wrap_with_vectranspose = False
        if isinstance(env.observation_space, spaces.Dict):
            # If even one of the keys is a image-space in need of transpose, apply transpose
            # If the image spaces are not consistent (for instance one is channel first,
            # the other channel last), VecTransposeImage will throw an error
            for space in env.observation_space.spaces.values():
                wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)
                )
        else:
            wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                env.observation_space
            )

        if wrap_with_vectranspose:
            if verbose >= 1:
                print("Wrapping the env in a VecTransposeImage.")
            env = VecTransposeImage(env)

    return env


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
            fake_dataset_ = None,
            create_from_env_factory: bool = False,
            n_workers: int = 4,
            batch_size: int = 32,
            redis_config: Optional[Dict[str, Any]] = None,
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
        if not create_from_env_factory:
            self.hidden_old, _ = tom_model(self.dataset[0:batch_size])
        self.comm_rewards = []
        self.n_workers = n_workers

        # Redis配置
        if redis_config is None:
            self.redis_config = {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "password": None
            }
        else:
            self.redis_config = redis_config

        # 初始化Redis客户端
        self.redis_client = redis.Redis(
            host=self.redis_config["host"],
            port=self.redis_config["port"],
            db=self.redis_config["db"],
            password=self.redis_config["password"]
        )

        if tom_model is None:
            self.tom_model = ToMNet(
                input_size=env.observation_space.shape[0] + 1,
                hidden_size=[64, 256, env.observation_space.shape[0] + 1],
                output_size=env.observation_space.shape[0] + 1 * 10)

        # 初始化hidden_old
        if self.dataset is not None and not create_from_env_factory:
            self.hidden_old, _ = self.tom_model(self.dataset[0:batch_size])

        if _init_setup_model:
            self._setup_model()

        # 初始化env_maker的参数
        with open('../myAlgorithm/config/my_ppo_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        self.args = config
        # self.env_args = {
        #     "env_id": self.args['env']['id'],
        #     "env_layout_name": self.args['env']['layout'],
        # }

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


    def dis_collect_rollouts(
            self,
            env,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
            n_workers: int = None,
            tom_model_weight = None
    ) -> bool:
        """
        使用多进程和Ray框架收集experiences，并将其填入RolloutBuffer.

        :param env: 训练环境
        :param callback: 在每一步调用的回调函数
        :param rollout_buffer: 用于填充的rollout缓冲区
        :param n_rollout_steps: 每个环境要收集的经验数量
        :param n_workers: 用于分布式采样的worker数量
        :return: 如果函数至少收集了n_rollout_steps经验则返回True，
                 如果回调函数提前终止rollout则返回False.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        if n_workers is None:
            n_workers = self.n_workers

        self.policy.set_training_mode(False)

        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        session_id = str(uuid.uuid4())

        # 使用cuda预先计算通信奖励
        with th.cuda.device(0):
            if self.tom_model is not None:
                tom_model_cuda = self.tom_model.to('cuda')
            else:
                tom_model_cuda = None

            if self.dataset is not None:
                dataset_cuda = self.dataset.to('cuda')
            else:
                dataset_cuda = None

            if self.hidden_old is not None:
                hidden_old_cuda = self.hidden_old.to('cuda')
            else:
                hidden_old_cuda = None

            if tom_model_cuda is not None and dataset_cuda is not None:
                reward_comm, hidden_new = compute_reward_comm(
                    dataset_cuda, hidden_old_cuda, tom_model_cuda, batch_size=self.batch_size)

                self.hidden_old = hidden_new.to(self.device)
            else:
                reward_comm = th.tensor(0.0)

        steps_per_worker = n_rollout_steps // n_workers
        remaining_steps = n_rollout_steps % n_workers

        worker_futures = []

        for i in range(n_workers):
            worker_steps = steps_per_worker
            if i == n_workers - 1:
                worker_steps += remaining_steps

            future = self.workers[i].collect_steps.remote(
                process_id=i,
                # args=self.args,
                # env_maker=env_factory,
                policy=self.policy,
                last_obs=self._last_obs,
                last_episode_starts=self._last_episode_starts,
                pre_computed_reward_comm=reward_comm,
                session_id=session_id,
                start_step=i * steps_per_worker,
                n_steps_to_collect=worker_steps,
                tom_model_weights=tom_model_weight
            )
            worker_futures.append(future)

        worker_results = ray.get(worker_futures)

        all_step_data = []
        all_dataset_items = []

        # 收集每个worker的所有步骤数据
        for result in worker_results:
            # 获取步骤数据
            for step_key in result["step_keys"]:
                step_data_bytes = self.redis_client.get(step_key)
                if step_data_bytes:
                    step_data = pickle.loads(step_data_bytes)
                    all_step_data.append(step_data)
                    # 删除已处理的数据
                    self.redis_client.delete(step_key)

            # 获取数据集项
            for dataset_key in result["dataset_items"]:
                dataset_bytes = self.redis_client.get(dataset_key)
                if dataset_bytes:
                    dataset_items = pickle.loads(dataset_bytes)
                    dataset_items = torch.stack(dataset_items)
                    all_dataset_items.append(dataset_items)
                    # 删除已处理的数据
                    self.redis_client.delete(dataset_key)

            # 获取最终状态（仅使用最后一个worker的结果）
            if result["worker_id"] == n_workers - 1:
                final_state_bytes = self.redis_client.get(result["final_key"])
                if final_state_bytes:
                    final_state = pickle.loads(final_state_bytes)
                    self._last_obs = final_state["last_obs"]
                    self._last_episode_starts = final_state["last_episode_starts"]
                    final_value = final_state["final_value"]
                    final_value_comm = final_state["final_value_comm"]
                    self.redis_client.delete(result["final_key"])

        # 按照步骤顺序排序数据
        all_step_data.sort(key=lambda x: x["step"])

        # 更新数据集
        if len(all_dataset_items) > 0:
            self.dataset = insert_dataset(self.dataset, all_dataset_items)

        # 将数据添加到rollout buffer
        for step_data in all_step_data:
            rollout_buffer.add(
                step_data["last_obs"],
                step_data["action"],
                step_data["rewards"],
                step_data["last_episode_starts"],
                step_data["value"],
                step_data["log_prob"],
                step_data["action_comm"],
                step_data["reward_comm"],
                step_data["log_prob_comm"],
                step_data["value_comm"]
            )

            # 更新info buffer
            self._update_info_buffer(step_data["infos"])

            # 更新时间步计数
            self.num_timesteps += env.num_envs

            # 给回调函数访问局部变量的权限
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

        # 计算returns和advantage
        rollout_buffer.compute_returns_and_advantage(last_values=final_value, dones=self._last_episode_starts)
        rollout_buffer.compute_returns_and_advantage_comm(last_values=final_value_comm, dones=self._last_episode_starts)

        callback.on_rollout_end()

        return True

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
            partner_new_obs = new_obs[1]
            partner_action = new_obs[2]
            # 已经生成队友的state和action，收集seq次组成一个tensor将其纳入dataset中
            self.dataset_item.append(
                torch.concat(
                    [
                        torch.FloatTensor(partner_new_obs),
                        torch.FloatTensor([partner_action])
                    ]
                )
            )
            if len(self.dataset_item) == self.dataset_seq_len:
                self.dataset = insert_dataset(self.dataset, self.dataset_item)
                self.dataset_item = []

            reward_comm, hidden_old = compute_reward_comm(
                self.dataset, self.hidden_old, self.tom_model, batch_size=self.batch_size)

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
        tom_model_weight = self.tom_model.state_dict()

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                                  n_rollout_steps=self.n_steps,
                                                                  n_workers=self.n_workers)
            # continue_training = self.dis_collect_rollouts(self.env, callback, self.rollout_buffer,
            #                                               n_rollout_steps=self.n_steps,
            #                                               n_workers=self.n_workers,
            #                                               tom_model_weight=tom_model_weight)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self.dataset = pre_process_dataset(self.dataset, self.batch_size)

            if self.num_timesteps % 32 == 0:
                train_step1(self.tom_model, self.dataset, self.batch_size, 100)
                train_step2(self.tom_model, self.dataset, self.batch_size, 100)
                tom_model_weight = self.tom_model.state_dict()

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
