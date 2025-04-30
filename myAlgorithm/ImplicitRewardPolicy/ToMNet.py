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
    def __init__(self, input_size, hidden_size, output_size, lamb=0.5, lstm_num_layers=1, lstm_num_directions=1):
        super(ToMNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_num_directions = lstm_num_directions
        self.lamb = lamb
        self.mu_layer = nn.Linear(self.hidden_size[0], self.hidden_size[0])
        self.log_var_layer = nn.Linear(self.hidden_size[0], self.hidden_size[0])
        if isinstance(self.hidden_size, list):
            self.lstm = nn.LSTM(input_size, self.hidden_size[0], batch_first=True)
            layer = []
            for i in range(len(self.hidden_size) - 1):
                layer.append(nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]))
                layer.append(nn.Tanh())
            layer.append(nn.Linear(self.hidden_size[-1], self.output_size))
            self.decoder = nn.Sequential(*layer)
        else:
            self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
            # used for training
            self.decoder = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        if len(x.size()) != 3:
            print("stop")
        batch_size, seq_len, input_size = x.size()
        num_layers = self.lstm_num_layers
        num_directions = self.lstm_num_directions
        hidden_size = self.hidden_size[0]
        h_0 = torch.randn(num_layers * num_directions, batch_size, hidden_size).to(self.device)
        c_0 = torch.randn(num_layers * num_directions, batch_size, hidden_size).to(self.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        h_last = h_n[-1] if not self.lstm_num_directions == 2 else torch.cat((h_n[-2], h_n[-1]), dim=1)
        mu = self.mu_layer(h_last)
        log_var = self.log_var_layer(h_last)
        std = torch.exp(0.5 * log_var)

        # reparameterization trick
        eps = torch.randn_like(std)
        z = mu + std * eps
        z = z.unsqueeze(0)
        z = F.softmax(z, dim=-1)

        recovery = self.decoder(z)
        return z, recovery.squeeze(0)


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
        if i % 10 == 0:
            print("[Training 1] epoch:{}, loss: {}".format(i, loss.item()))
        if loss.item() < 0.0015:
            # 防止过拟合
            return


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
        if i % 10 == 0:
            print("[Training 2] epoch:{}, loss: {}".format(i, loss.item()))


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


def make_fake_dataset(env, data_num, seq_len, device='cuda'):
    result = []
    result_in_one_seq = []
    state = env.reset()
    act = env.action_space.sample()

    # Pre-allocate a list to store sequence tensors
    sequences = []

    for i in range(data_num):
        state, reward, done, *_ = env.step(act)

        # Handle different state formats
        if isinstance(state, list):
            result_in_one_seq.append(np.append(state[0], float(act)))
        else:
            result_in_one_seq.append(np.append(state, float(act)))

        if len(result_in_one_seq) == seq_len:
            with torch.no_grad():
                seq_tensor = torch.tensor(result_in_one_seq,
                                          dtype=torch.float32,
                                          device=device)
            sequences.append(seq_tensor.unsqueeze(0))
            result_in_one_seq = []
        act = env.action_space.sample()
        if done:
            state = env.reset()
    if sequences:
        with torch.no_grad():
            result = torch.cat(sequences, dim=0).detach()
    else:
        result = torch.empty(0, seq_len, len(result_in_one_seq[0]) if result_in_one_seq else 0,
                             dtype=torch.float32, device=device)

    return result

def insert_dataset(dataset, dataset_item: list):
    # dataset_item中是一个seq长度的s-a pair，以tensor格式
    # 需要将这个list插入到dataset中
    # list的长度是seq_len，每一项的长度是state_dim+action_dim
    # 首先将其转换为tensor(10, 63)
    # 然后将其插入到dataset中
    # 尾插法，后端的数据是新的
    dataset_item = torch.stack(dataset_item).to('cuda')
    if dataset_item.dim() == 2:
        dataset_item = dataset_item.unsqueeze(0)
    dataset = torch.concat((dataset, dataset_item), dim=0)
    return dataset

def batch_generator(dataset, batch_size):
    # dataset的格式为(batch_size, seq_len, input_size)
    # 分别代表批量大小、序列长度、输入维度
    # 这里设定batch_size为32，序列长度为10，输入维度为63
    # while len(dataset) % batch_size != 0:
    #     dataset = dataset[0:]
    #
    # if len(dataset) > MAX_DATASET_NUM:
    #     dataset = dataset[0:MAX_DATASET_NUM]

    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        yield batch_data

def pre_process_dataset(dataset, batch_size):
    # dataset由尾插法构成
    # 首先处理成batch_size整数倍，逐步弹出头部数据
    while len(dataset) % batch_size != 0:
        dataset = dataset[1:]

    if len(dataset) > MAX_DATASET_NUM:
        dataset = dataset[-MAX_DATASET_NUM:]

    return dataset


if __name__ == '__main__':
    # Test training process
    pass
