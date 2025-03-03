import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
import torch as th
from gym import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv
from tensorflow.python.ops.numpy_ops import ndarray

# 引我自己的buffer
from .my_buffers import RolloutBuffer, DictRolloutBuffer

from .communicateUtils.comm_interact import comm
from ..ImplicitRewardPolicy.ToMNet import make_fake_dataset, insert_dataset, ToMNet
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
            tom_model=None
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
        self.dataset = make_fake_dataset(env, self.dataset_data_num, self.dataset_seq_len)
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
            env: VecEnv,
            callback: BaseCallback,
            rollout_buffer: RolloutBuffer,
            n_rollout_steps: int,
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
            if action_comm[0]:
                # 当收到通信action，则进行通信
                # 通信过程包括：1.发送action到其他agent，2.等待其他agent的回复，3.将回复的action输入self并处理
                comm()

            value, value_comm = th.split(values, 1, dim=0)

            log_prob, log_prob_comm = th.split(log_probs, 1, dim=0)

            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
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
            reward_comm = compute_reward_comm(self.dataset, self.hidden_old, self.tom_model)
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

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

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
