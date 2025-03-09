# 实现内部奖励的函数
import numpy as np
from scipy.stats import entropy

from .ToMNet import ToMNet
import torch
import torch.nn.functional as F


# the function now is in discrete version
def compute_reward_comm(dataset, hidden_old, tom_net, num_state_bins=100):
    batch_size, seq_len, feature_dim = dataset.shape
    state_dim = feature_dim - 1  # 状态维度（62）

    state = dataset[..., :-1]  # 形状 (batch, seq_len, 62)
    action = dataset[..., -1].long()  # 形状 (batch, seq_len)

    state_bins = torch.linspace(state.min(), state.max(), num_state_bins)
    state_discrete = torch.bucketize(state, state_bins)  # 形状 (batch, seq_len, 62)

    joint_hist = torch.zeros((num_state_bins, action.max().item() + 1))  # (离散状态数量, 动作类别数量)

    for i in range(batch_size):
        for j in range(seq_len):
            s_idx = state_discrete[i, j].sum().item() % num_state_bins  # 直接用状态的和 modulo 离散化
            a_idx = action[i, j].item()
            joint_hist[s_idx, a_idx] += 1

    if joint_hist.sum() > 0:
        joint_prob = joint_hist / joint_hist.sum()
        joint_prob_flat = joint_prob.view(-1)
        nonzero_probs = joint_prob_flat[joint_prob_flat > 0]
        entropy = -torch.sum(nonzero_probs * torch.log2(nonzero_probs))
    else:
        entropy = torch.tensor(0.0)
    # entropy = torch.tensor(0.0)

    # ========== 计算 KL 散度 ==========
    hidden, _ = tom_net(dataset[:, 0, :])  # 只使用当前时间步
    eps = 1e-10

    log_hidden = torch.log(hidden + eps)
    log_hidden_old = torch.log(hidden_old + eps)

    KL_div = F.kl_div(log_hidden_old, log_hidden.exp(), reduction='batchmean')
    assert KL_div != float('nan')

    return np.array(-KL_div.item() - entropy.item()), hidden
