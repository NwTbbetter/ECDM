import torch.nn as nn
from torch.nn import functional as F
import math
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=torch.int,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level if use_affine_level is not None else False
        #print(self.use_affine_level)
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

class CNN(nn.Module):
    def __init__(self, in_channel=150, out_channel=150, hidden=256, with_noise_level_emb=True):
        super(CNN, self).__init__()
        self.noise_level_mlp = PositionalEncoding(hidden)

        self.conv0 = nn.Conv2d(in_channel, hidden, kernel_size=1)
        self.conv1_1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=1)
        self.conv3_1 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1)
        self.nin_a = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.nin_b = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.nin_c = nn.Conv2d(hidden, out_channel, kernel_size=1)

        if with_noise_level_emb:
            self.noise_func1 = FeatureWiseAffine(hidden, hidden)
            self.noise_func2 = FeatureWiseAffine(hidden, hidden * 2)
            self.noise_func3 = FeatureWiseAffine(hidden, hidden)
    def forward(self, data, time=None):
        # data = data.permute(0, 3, 1, 2).contiguous()
        t = self.noise_level_mlp(time) if time != None else None
        # t = time

        out = F.relu(self.conv0(data))
        if time is not None:
            out =self.noise_func1(out, t)
        temp = out
        out = F.relu(self.conv1_1(out))
        # if time is not None:
        #     out =self.noise_func1(out, t)
        out = F.relu(self.conv1_2(out))
        # if time is not None:
        #     out =self.noise_func1(out, t)
        out = out + temp

        out = F.relu(self.conv2(out))
        if time is not None:
            out =self.noise_func2(out, t)
        temp = out
        out = F.relu(self.conv3_1(out))
        # if time is not None:
        #     out =self.noise_func2(out, t)
        out = F.relu(self.conv3_2(out))
        # if time is not None:
        #     out =self.noise_func2(out, t)
        out = out + temp
        
        out = self.conv4(out)
        # if time is not None:
        #     out =self.noise_func3(out, t)
        temp = out
        out = F.relu(self.nin_a(out))
        # if time is not None:
        #     out =self.noise_func3(out, t)
        out = F.relu(self.nin_b(out))
        if time is not None:
            out =self.noise_func3(out, t)
        out = out + temp
        
        out = self.nin_c(out)
        return out