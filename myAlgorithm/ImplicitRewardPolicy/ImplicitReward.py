# 实现内部奖励的函数
import numpy as np
from scipy.stats import entropy

from .ToMNet import ToMNet
import torch
import torch.nn.functional as F


# the function now is in discrete version
def compute_reward_comm(dataset, hidden_old, tom_net, num_state_bins=100, epsilon=1e-10):
    batch_size, seq_len, feature_dim = dataset.shape
    state_dim = feature_dim - 1

    state = dataset[..., :-1]
    action = dataset[..., -1].long()

    state_bins = torch.linspace(state.min(), state.max(), num_state_bins)
    state_discrete = torch.bucketize(state, state_bins)

    joint_hist = torch.zeros((num_state_bins, action.max().item() + 1))

    for i in range(batch_size):
        for j in range(seq_len):
            s_idx = state_discrete[i, j].sum().item() % num_state_bins
            a_idx = action[i, j].item()
            joint_hist[s_idx, a_idx] += 1

    if joint_hist.sum() > 0:
        joint_prob = joint_hist / joint_hist.sum()
        joint_prob_flat = joint_prob.view(-1)
        nonzero_probs = joint_prob_flat[joint_prob_flat > 0]
        inner_entropy = -torch.sum(nonzero_probs * torch.log2(nonzero_probs))
    else:
        inner_entropy = torch.tensor(0.0)

    inner_entropy = torch.log1p(inner_entropy)

    hidden, _ = tom_net(dataset[:, 0, :])
    log_hidden = torch.log(hidden + epsilon)
    log_hidden_old = torch.log(hidden_old + epsilon)

    KL_div = F.kl_div(log_hidden_old, log_hidden.exp(), reduction='batchmean')

    KL_div_min, KL_div_max = 0.1, 1.0
    KL_div_norm = (KL_div - KL_div_min) / (KL_div_max - KL_div_min)
    KL_div_norm = torch.clamp(KL_div_norm, 0, 1)

    lambda_entropy = 1  # 调整熵的权重
    reward_internal = -lambda_entropy * inner_entropy - KL_div_norm

    return reward_internal.item(), hidden
