import torch.nn as nn
from torch.nn import functional as F
import math
import torch
from einops import rearrange
import numbers

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class LayerNorm_Without_Shape(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_Without_Shape, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return self.body(x)

# self.him1 = HIM(dim = 31, num_heads = 2, bias = False, embed_dim = 64, LayerNorm_type = 'WithBias')
class HIM(nn.Module):
    def __init__(self, dim, num_heads, bias, embed_dim, LayerNorm_type, qk_scale=None):
        super(HIM, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # self.pos_embed1 = nn.Parameter(torch.randn(1, 6192, 64), requires_grad=True)
        # self.pos_embed2 = nn.Parameter(torch.randn(1, 1548, 64), requires_grad=True)

        self.norm1 = LayerNorm_Without_Shape(dim, LayerNorm_type)
        self.norm2 = LayerNorm_Without_Shape(embed_dim*2, LayerNorm_type)

        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(embed_dim*2, 2*dim, bias=bias)
        
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, prior):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        _x = self.norm1(x)
        prior = self.norm2(prior)
        
        q = self.q(_x)
        kv = self.kv(prior)
        k,v = kv.chunk(2, dim=-1)   

        q = rearrange(q, 'b n (head c) -> b head n c', head=self.num_heads)
        k = rearrange(k, 'b n (head c) -> b head n c', head=self.num_heads)
        v = rearrange(v, 'b n (head c) -> b head n c', head=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head n c -> b n (head c)', head=self.num_heads)
        out = self.proj(out)

        # sum
        x = x + out
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W).contiguous()

        return x

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
    def __init__(self, in_channel=31, out_channel=31, hidden=64, with_noise_level_emb=True):
        super(CNN, self).__init__()
        self.noise_level_mlp = PositionalEncoding(hidden)
        
        self.conv = nn.Conv2d(in_channel, 32, kernel_size=1)
        self.him1 = HIM(dim = 32, num_heads = 2, bias = False, embed_dim = 64, LayerNorm_type = 'WithBias')
        self.conv1 = nn.Conv2d(32, hidden, kernel_size=1)
        self.conv1_1 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)


        self.him2 = HIM(dim = hidden, num_heads = 2, bias = False, embed_dim = 64, LayerNorm_type = 'WithBias')
        self.conv2 = nn.Conv2d(hidden, hidden * 2, kernel_size=1)
        self.conv3_1 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3, padding=1)


        self.him3 = HIM(dim = hidden * 2, num_heads = 2, bias = False, embed_dim = 64, LayerNorm_type = 'WithBias')
        self.conv4 = nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1)
        self.nin_a = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.nin_b = nn.Conv2d(hidden, hidden, kernel_size=1)


        self.nin_c = nn.Conv2d(hidden, out_channel, kernel_size=1)

        if with_noise_level_emb:
            self.noise_func1 = FeatureWiseAffine(hidden, hidden)
            self.noise_func2 = FeatureWiseAffine(hidden, hidden * 2)
            self.noise_func3 = FeatureWiseAffine(hidden, hidden)
    def forward(self, data, prior, time=None):
        # data = data.permute(0, 3, 1, 2).contiguous()
        t = self.noise_level_mlp(time) if time != None else None
        # t = time

        # out = F.relu(self.conv0(data))
        out = self.conv(data)
        prior1 = prior
        prior2 = prior # 32,16,128

        out = self.him1(out, prior)
        out = F.relu(self.conv1(out))
        # if time is not None:
        #     out =self.noise_func1(out, t)
        temp = out
        out = F.relu(self.conv1_1(out))
        # if time is not None:
        #     out =self.noise_func1(out, t)
        out = F.relu(self.conv1_2(out))
        # if time is not None:
        #     out =self.noise_func1(out, t)
        out = out + temp


        out = self.him2(out, prior1)
        out = F.relu(self.conv2(out))
        # if time is not None:
        #     out =self.noise_func2(out, t)
        temp = out
        out = F.relu(self.conv3_1(out))
        # if time is not None:
        #     out =self.noise_func2(out, t)
        out = F.relu(self.conv3_2(out))
        # if time is not None:
        #     out =self.noise_func2(out, t)
        out = out + temp
        

        out = self.him3(out, prior2)
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