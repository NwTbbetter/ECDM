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
from .CNN_d import CNN_d

class N2N(nn.Module):
    '''
    Noise model as in Noise2Noise
    '''

    def __init__(
            self,
            le,
            denoise_fn,
            net_d,
            first_denoisor

    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        # self.denoise_fn.load_state_dict(torch.load('/home/bxy/DDM_experiments/dw_DDM_latent2/p_hardi/denoise_gen.pth'))
        self.le = le
        # self.le.load_state_dict(torch.load('/home/bxy/DDM_experiments/dw_DDM_latent2/p_hardi/le_gen.pth'))
        # self.le.eval()
        self.first_denoisor = first_denoisor
        self.net_d = net_d


        self.loss = Loss()
        # self.net_d = CNN_d
        self.set_new_noise_schedule('cuda')

    @torch.no_grad()
    def denoise(self, x_in):
        with torch.no_grad():
            y_p = self.first_denoisor(x_in['condition'])
            prior_z = self.le(y_p)
            prior = self.p_sample_loop_wo_variance(prior_z)
            # prior_c = self.net_le_dm(x_in['condition'])
            # prior_noisy = torch.randn_like(prior_c)
            # prior = self.p_sample_loop_wo_variance(prior_noisy, x_in = prior_c)
            return self.denoise_fn(x_in['condition'], prior)

    def p_sample_loop_wo_variance(self, x_noisy, x_in=None, ema_model=False):
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_wo_variance(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def p_sample_wo_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean

    def warmup_beta(self,linear_start, linear_end, n_timestep, warmup_frac):
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = np.linspace(
            linear_start, linear_end, warmup_time, dtype=np.float64)
        return betas

    def set_new_noise_schedule(self, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        # β1, β2, ..., βΤ (T)
        betas = self.make_beta_schedule(
            schedule='linear',
            n_timestep=8,
            linear_start=0.1,
            linear_end=0.99)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        # α1, α2, ..., αΤ (T)
        alphas = 1. - betas
        # α1, α1α2, ..., α1α2...αΤ (T)
        alphas_cumprod = np.cumprod(alphas, axis=0)
        # 1, α1, α1α2, ...., α1α2...αΤ-1 (T)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # 1, √α1, √α1α2, ...., √α1α2...αΤ (T+1)
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def make_beta_schedule(self, schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        linear_start = float(linear_start)
        linear_end = float(linear_end)
        if schedule == 'quad':
            betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                                n_timestep, dtype=np.float64) ** 2
        elif schedule == 'linear':
            betas = np.linspace(linear_start, linear_end,
                                n_timestep, dtype=np.float64)
        elif schedule == 'warmup10':
            betas = self.warmup_beta(linear_start, linear_end,
                                     n_timestep, 0.1)
        elif schedule == 'warmup50':
            betas = self.warmup_beta(linear_start, linear_end,
                                     n_timestep, 0.5)
        elif schedule == 'const':
            betas = linear_end * np.ones(n_timestep, dtype=np.float64)
        elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / np.linspace(n_timestep,
                                     1, n_timestep, dtype=np.float64)
        elif schedule == "cosine":
            timesteps = (
                    torch.arange(n_timestep + 1, dtype=torch.float64) /
                    n_timestep + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(schedule)
        return betas

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        # if condition_x is None:
        #     raise RuntimeError('Must have LQ/LR condition')

        if ema_model:
            print("TODO")
        else:
            # x_recon = self.predict_start_from_noise(x, t=t, noise=self.net_d(x, condition_x, torch.full(x.shape, t,
            #                                                                                             device=self.betas.device,
            #                                                                                             dtype=torch.long)))
            x_recon = self.predict_start_from_noise(x, t = t, noise=self.net_d(x, torch.full((x.shape[0],), t, device=x.device)))


        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise



    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, setting, noise=None):
        debug_results = dict()
        with torch.no_grad():            
            y_p = self.first_denoisor(x_in['condition'])
        prior_z = self.le(y_p)
        # prior_z1 = self.le(x_in['condition'])

        # prior_d = self.net_le_dm(x_in['X'])

        # diffusion-forward
        t = 8
        # [b, 4c']
        noise = default(noise, lambda: torch.randn_like(prior_z))
        # noise = prior_z - prior_z1
        # sample xt/x_noisy (from x0/x_start)
        prior_noisy = self.q_sample(
            x_start=prior_z, sqrt_alpha_cumprod=self.alphas_cumprod[t - 1],
            noise=noise)
        # diffusion-reverse
        prior = self.p_sample_loop_wo_variance(prior_noisy)#, x_in = prior_d)

        setting['number'] += 1
        noisy_sim1 = x_in['X']
        # noisy_sim1 = x_in['condition']
        mask1, mask2 = self.loss.generate_mask_pair(noisy_sim1, setting)

        noisy_sub1 = self.loss.generate_subimages(noisy_sim1, mask1)
        noisy_sub2 = self.loss.generate_subimages(noisy_sim1, mask2)

        with torch.no_grad():
            noisy_denoised1 = self.denoise_fn(noisy_sim1, prior)

        noisy_sub1_denoised = self.loss.generate_subimages(noisy_denoised1, mask1)
        noisy_sub2_denoised = self.loss.generate_subimages(noisy_denoised1, mask2)

        noisy_output1 = self.denoise_fn(noisy_sub1, prior)
        noisy_target1 = noisy_sub2

        # Lambda = setting['current_epoch'] / self.loss.n_epoch * self.loss.increase_ratio
        # print(setting['current_step'])
        Lambda = setting['current_step'] / self.loss.n_epoch * self.loss.increase_ratio
        # print(Lambda)
        # print("epoch", setting['current_step'])
        # print(self.loss.n_epoch)

        diff1 = noisy_output1 - noisy_target1
        exp_diff1 = noisy_sub1_denoised - noisy_sub2_denoised
        
        loss1 = torch.mean(diff1 ** 2)
        setting['epoch_loss1'] += loss1.item()

        loss2 = Lambda * (torch.mean((diff1 - exp_diff1) ** 2))
        setting['epoch_loss2'] += loss2.item()

        loss_all = self.loss.Lambda1 * loss1 + self.loss.Lambda2 * loss2 #+ torch.mean((prior - prior_z) ** 2)

        setting['epoch_all_loss'] += loss_all.item()
        print('iteration:{:06d}, Loss1={:.6f}, Lambda={}, Loss2={:.6f},Loss_Full={:.6f}'
              .format(setting['current_step'], loss1, Lambda,
                      loss2 / Lambda, loss_all))

        return dict(total_loss=loss_all)
        # x_start = x_in['X'] # our
        # [b, c, w, h] = x_start.shape

        # x_recon = self.denoise_fn(x_in['condition'])

        # loss1 = self.mse_loss(x_recon, x_in['X'])

        # return dict(total_loss=loss1)

    def forward(self, x, setting, *args, **kwargs):
        return self.p_losses(x, setting, *args, **kwargs)
