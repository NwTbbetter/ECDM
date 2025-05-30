import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import copy

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.increase_ratio = 8
        self.Lambda1 = 1
        self.Lambda2 = 1
        self.n_epoch = 10000

    def checkpoint(net, epoch, name):
        save_model_path = os.path.join(opt.save_model_path, opt.log_name, systime)
        os.makedirs(save_model_path, exist_ok=True)
        model_name = 'epoch_{}_{:03d}.pth'.format(name, epoch)
        save_model_path = os.path.join(save_model_path, model_name)
        torch.save(net.state_dict(), save_model_path)
        print('Checkpoint saved to {}'.format(save_model_path))


    def get_generator(self,loss):
        # global operation_seed_counter
        # operation_seed_counter += 1
        loss['operation_seed_counter'] += 1
        g_cuda_generator = torch.Generator(device="cuda")
        g_cuda_generator.manual_seed(loss['operation_seed_counter'])
        return g_cuda_generator




    def space_to_depth(self, x, block_size):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
        return unfolded_x.view(n, c * block_size**2, h // block_size,
                            w // block_size)


    def generate_mask_pair(self, img, loss):
        # prepare masks (N x C x H/2 x W/2)
        n, c, h, w = img.shape
        mask1 = torch.zeros(size=(n * (h // 2) * (w // 2) * 4, ),
                            dtype=torch.bool,
                            device=img.device)
        mask2 = torch.zeros(size=(n * (h // 2) * (w // 2) * 4, ),
                            dtype=torch.bool,
                            device=img.device)
        # prepare random mask pairs
        idx_pair = torch.tensor(
            [[0, 1], [0, 2], [1, 3], [2, 3], [1, 0], [2, 0], [3, 1], [3, 2]],
            dtype=torch.int64,
            device=img.device)
        rd_idx = torch.zeros(size=(n * (h // 2) * (w // 2), ),
                            dtype=torch.int64,
                            device=img.device)
        torch.randint(low=0,
                    high=8,
                    size=(n * (h // 2) * (w // 2), ),
                    generator=self.get_generator(loss),
                    out=rd_idx)
        rd_pair_idx = idx_pair[rd_idx]
        rd_pair_idx += torch.arange(start=0,
                                    end=n * (h // 2) * (w // 2) * 4,
                                    step=4,
                                    dtype=torch.int64,
                                    device=img.device).reshape(-1, 1)
        # get masks
        mask1[rd_pair_idx[:, 0]] = 1
        mask2[rd_pair_idx[:, 1]] = 1
        return mask1, mask2


    def generate_subimages(self, img, mask):
        n, c, h, w = img.shape
        subimage = torch.zeros(n,
                            c,
                            h // 2,
                            w // 2,
                            dtype=img.dtype,
                            layout=img.layout,
                            device=img.device)
        # per channel
        for i in range(c):
            img_per_channel = self.space_to_depth(img[:, i:i + 1, :, :], block_size=2)
            img_per_channel = img_per_channel.permute(0, 2, 3, 1).reshape(-1)
            subimage[:, i:i + 1, :, :] = img_per_channel[mask].reshape(
                
                n, h // 2, w // 2, 1).permute(0, 3, 1, 2)
        return subimage
