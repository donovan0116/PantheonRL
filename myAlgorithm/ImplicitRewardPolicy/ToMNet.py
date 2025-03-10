import gym
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from click.core import batch
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from pantheonrl.common.agents import OnPolicyAgent

MAX_DATASET_NUM = 3200


class ToMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lamb=0.5):
        super(ToMNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_size = hidden_size
        self.lamb = lamb
        if isinstance(hidden_size, list):
            self.lstm = nn.LSTM(input_size, hidden_size[0], batch_first=True)
            layer = []
            for i in range(len(hidden_size) - 1):
                layer.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
                layer.append(nn.Tanh())
            layer.append(nn.Linear(hidden_size[-1], output_size))
            self.decoder = nn.Sequential(*layer)
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            # used for training
            self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        _, (compress_plan, _) = self.lstm(x)
        compress_plan = torch.softmax(compress_plan, dim=-1)
        recovery = self.decoder(compress_plan)
        return compress_plan, recovery.squeeze(0)


# in the first step of training, we only train the ToMNet as an encoder
def train_step1(model_, dataset, batch_size=32, epoch=10, recon_loss_fn=None, optimizer=None):
    # dataset: (data_num, batch_size, seq_len, input_size)
    # reconstruction loss for encoder-decoder
    if recon_loss_fn is None:
        recon_loss_fn = nn.MSELoss()

    if optimizer is None:
        optimizer = optim.Adam(model_.parameters(), lr=0.001)

    torch.autograd.set_detect_anomaly(True)

    for i in range(epoch):
        for batch_idx, data in enumerate(batch_generator(dataset, batch_size)):
            _, data_prime = model_(data)
            loss = recon_loss_fn(data.flatten(1, 2), data_prime)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Training 1] epoch:{}, step: {}, loss: {}".format(i, batch_idx, loss.item()))


# in the second step of training, we train the ToMNet in representation space
def train_step2(model_, dataset, batch_size=32, epoch=10, optimizer=None):
    if optimizer is None:
        optimizer = optim.Adam(model_.parameters(), lr=0.001)
    for i in range(epoch):
        for batch_idx, data in enumerate(batch_generator(dataset, batch_size)):
            # compress_plan的格式为(num_layers * num_directions, batch_size, hidden_size)
            compress_plan, _ = model_(data)
            compress_plan_prime = compress_plan + torch.randn_like(compress_plan)
            mat = metrix_c(compress_plan, compress_plan_prime)

            diag_elements = torch.diagonal(mat, dim1=-2, dim2=-1)
            loss1 = torch.sum((1 - diag_elements) ** 2, dim=-1)

            mask = ~torch.eye(mat.size(-1), dtype=torch.bool)
            off_diag_elements = mat[..., mask].view(mat.shape[0], mat.shape[1], -1)
            loss2 = torch.sum(off_diag_elements ** 2, dim=-1)

            loss = loss1 + model_.lamb * loss2
            loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("[Training 2] epoch:{}, step: {}, loss: {}".format(i, batch_idx, loss.item()))


# compute the cosine metrix C
def metrix_c(z, z_p):
    # check if z and z_p are the same size
    assert z.shape == z_p.shape
    z_expanded = z.unsqueeze(-1)
    z_p_expanded = z_p.unsqueeze(-2)
    dot_product = torch.matmul(z_expanded, z_p_expanded)
    norm_z = z.norm(dim=-1, keepdim=True).unsqueeze(-2)
    norm_z_p = z_p.norm(dim=-1, keepdim=True).unsqueeze(-2)
    result = dot_product / (norm_z * norm_z_p + 1e-8)
    return result


def make_fake_dataset(env, data_num, seq_len):
    """
    params:
    env: env that used to sample
    data_num: the num of data, one data with the structure of :
    Tensor[[s_1, a_1], [s_2, a_2], ...]
    """
    result = []
    result_in_one_seq = []
    state = env.reset()
    act = env.action_space.sample()
    for i in range(data_num):
        state, reward, done, _ = env.step(act)
        if isinstance(state, list):
            result_in_one_seq.append(np.append(state[0], float(act)))
        else:
            result_in_one_seq.append(np.append(state, float(act)))
        if len(result_in_one_seq) == seq_len:
            result.append(torch.FloatTensor(result_in_one_seq))
            result_in_one_seq = []
        act = env.action_space.sample()
        if done:
            state = env.reset()
    for i in range(len(result)):
        # 数据集中每一项转换成tensor
        with torch.no_grad():
            result[i] = torch.tensor(result[i]).float().unsqueeze(0)
    result = torch.concat(result).detach()
    return result

def insert_dataset(dataset, dataset_item: list):
    # dataset_item中是一个seq长度的s-a pair，以tensor格式
    # 需要将这个list插入到dataset中
    # list的长度是seq_len，每一项的长度是state_dim+action_dim
    # 首先将其转换为tensor(10, 63)
    # 然后将其插入到dataset中
    dataset_item = torch.stack(dataset_item)
    batch_size, seq_len, input_size = dataset.shape
    batch_size += 1
    dataset = torch.concat((dataset, dataset_item.unsqueeze(0)), dim=0)
    return dataset

def batch_generator(dataset, batch_size):
    # dataset的格式为(batch_size, seq_len, input_size)
    # 分别代表批量大小、序列长度、输入维度
    # 这里设定batch_size为32，序列长度为10，输入维度为63
    while len(dataset) % batch_size != 0:
        dataset = dataset[0:]

    if len(dataset) > MAX_DATASET_NUM:
        dataset = dataset[0:MAX_DATASET_NUM]

    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        yield batch_data


if __name__ == '__main__':
    # Test training process
    pass
