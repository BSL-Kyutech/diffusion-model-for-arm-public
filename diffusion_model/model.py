import torch
import torch.nn as nn
from simulator import definition as armdef
import numpy as np
import pandas as pd
import math

class MiddleLayer(nn.Module):
    def __init__(self, target_d, d):
        super().__init__()
        self.encoder = FC(d)
        self.decoder = FC(d)
        self.fc = nn.Linear(target_d, d)

    def forward(self, x, target):
        x = self.encoder(x)
        x = x + self.fc(target)
        x = self.decoder(x)
        return x

class Model(nn.Module):
    def __init__(self, steps):
        super().__init__()
        self.d = 1024

        self.m = nn.ZeroPad1d(
            (0, self.d - armdef.arm.spring_joint_count*2))

        self.pe = PositionalEncoding(steps, self.d)
        self.middles = nn.ModuleList([])

    def forward(self, x: torch.Tensor, step: torch.Tensor, pos: torch.Tensor):
        for middle in self.middles:
            x = self.m(x)
            x = self.pe(x, step)
            x = middle(x, pos)
            x = x[:, :armdef.arm.spring_joint_count*2]
        return x

    def denoise(self, xt: torch.Tensor, steps: int, pos):
        for i in reversed(range(1, steps)):
            step = torch.FloatTensor([i]).cuda()
            z = torch.randn_like(xt)
            step = torch.Tensor([i]).long()
            xt_ = xt.view(1, -1)
            if i == 1:
                xt = (
                    1/torch.sqrt(alpha[i]))*(xt - (torch.sqrt(beta[i]))*self(xt_, step, pos))
            else:
                xt = (1/torch.sqrt(alpha[i]))*(xt-(beta[i]/torch.sqrt(1-alpha_[i]))*self(
                    xt_, step, pos))+torch.sqrt((1-alpha_[i-1])/(1-alpha_[i])*beta[i])*z
            xt = xt.view(-1)
        return xt

class ModelForXY(Model):
    def __init__(self, steps):
        super().__init__(steps)
        self.middles.append(MiddleLayer(2, self.d))

start_beta = 1e-4
end_beta = 0.02
steps = 25
n = 1024

beta = torch.FloatTensor(steps)
alpha = torch.FloatTensor(steps)
alpha_ = torch.FloatTensor(steps)


def pre_calc_beta_and_alpha():
    for i in range(1, steps):
        beta[i] = end_beta*((i-1)/(steps-1))+start_beta * \
            ((steps-1-(i-1))/(steps-1))
        alpha[i] = 1-beta[i]
        alpha_[i] = alpha[i]
        if i-1 >= 1:
            alpha_[i] *= alpha_[i-1]

pre_calc_beta_and_alpha()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        features = ([str(i) for i in range(armdef.arm.spring_joint_count*2)])
        self.pos = df[['x', 'y']].values
        self.x = df[features].values
        self.theta = df['theta'].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x[idx])
        pos = torch.FloatTensor(self.pos[idx])
        theta = torch.FloatTensor([self.theta[idx]])
        return x, pos, theta


class PositionalEncoding(torch.nn.Module):
    def __init__(self, steps, d):
        super().__init__()
        pos = torch.arange(steps).unsqueeze(1)
        div = torch.pow(10000, torch.arange(0, d, 2)/d)
        self.pe = torch.zeros(steps, d)
        self.pe[:, 0::2] = torch.sin(pos/div)
        self.pe[:, 1::2] = torch.cos(pos/div)
        self.d = d

    def forward(self, x, step):
        step = step.expand(self.d, -1).T
        pe_ = torch.gather(self.pe, 0, step.cpu()).to(x.device)
        x = x*math.sqrt(self.d)+pe_
        return x


class FC(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(d, d)
        self.bn = nn.BatchNorm1d(d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def extract(t, x_shape):
    batch_size = t.shape[0]
    out = alpha_.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,)*(len(x_shape) - 1))).to(t.device)


def gen_xt(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
    at_ = extract(t, x0.shape)
    x = torch.sqrt(at_)*x0+torch.sqrt(1-at_)*noise
    t = t.view(x.shape[0], 1)
    return x


def normalize(x: torch.Tensor):
    x -= 15
    x /= 15
    return x


def denormalize(x: torch.Tensor):
    x *= 15
    x += 15
    return x
