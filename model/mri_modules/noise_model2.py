import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import copy
from .utils import *
from .loss import Loss


class N2N(nn.Module):
    '''
    Noise model as in Noise2Noise
    '''

    def __init__(
            self,
            le,
            denoise_fn,
            first_denoisor

    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.le = le
        self.first_denoisor = first_denoisor
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.loss = Loss()

    @torch.no_grad()
    def denoise(self, x_in):
        y_p = self.first_denoisor(x_in['condition'])
        latent = self.le(y_p)
        return self.denoise_fn(x_in['condition'], latent)

    def p_losses(self, x_in, setting, noise=None):
        debug_results = dict()
        with torch.no_grad():
            y_p = self.first_denoisor(x_in['X'])

        latent = self.le(y_p)
        setting['number'] += 1
        noisy_sim1 = x_in['X']
        # noisy_sim1 = x_in['condition']
        mask1, mask2 = self.loss.generate_mask_pair(noisy_sim1, setting)

        noisy_sub1 = self.loss.generate_subimages(noisy_sim1, mask1)
        noisy_sub2 = self.loss.generate_subimages(noisy_sim1, mask2)

        with torch.no_grad():
            noisy_denoised1 = self.denoise_fn(noisy_sim1, latent)

        noisy_sub1_denoised = self.loss.generate_subimages(noisy_denoised1, mask1)
        noisy_sub2_denoised = self.loss.generate_subimages(noisy_denoised1, mask2)

        noisy_output1 = self.denoise_fn(noisy_sub1, latent)
        noisy_target1 = noisy_sub2

        # Lambda = setting['current_epoch'] / self.loss.n_epoch * self.loss.increase_ratio
        # print(setting['current_step'])
        Lambda = setting['current_step'] / self.loss.n_epoch * self.loss.increase_ratio
        # print(Lambda)
        # print("epoch", setting['current_step'])
        # print(self.loss.n_epoch)

        diff1 = noisy_output1 - noisy_target1
        exp_diff1 = noisy_sub1_denoised - noisy_sub2_denoised

        # diff2 = noisy_sub1 - noisy_sub2

        loss1 = torch.mean(diff1 ** 2) #+ torch.mean(diff2 **2)
        setting['epoch_loss1'] += loss1.item()

        loss2 = Lambda * (torch.mean((diff1 - exp_diff1) ** 2))
        setting['epoch_loss2'] += loss2.item()

        loss_all = self.loss.Lambda1 * loss1 + self.loss.Lambda2 * loss2

        setting['epoch_all_loss'] += loss_all.item()
        print('iteration:{:06d}, Loss1={:.6f}, Lambda={}, Lambda1={},Loss2={:.6f},Loss_Full={:.6f}'
              .format(setting['current_step'], setting['epoch_loss1'] / setting['number'], Lambda, self.loss.Lambda1,
                      setting['epoch_loss2'] / setting['number'], setting['epoch_all_loss'] / setting['number']))

        return dict(total_loss=loss_all)
        # x_start = x_in['X'] # our
        # [b, c, w, h] = x_start.shape

        # x_recon = self.denoise_fn(x_in['condition'])

        # loss1 = self.mse_loss(x_recon, x_in['X'])

        # return dict(total_loss=loss1)

    def forward(self, x,setting, *args, **kwargs):
        return self.p_losses(x, setting, *args, **kwargs)
