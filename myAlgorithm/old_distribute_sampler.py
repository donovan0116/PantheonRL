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