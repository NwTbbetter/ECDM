import logging
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import os
from model.mri_modules import CNN_latent
from model.mri_modules import noise_model3
from torch.optim import lr_scheduler
from .base_model import BaseModel
from model.mri_modules import latent_encoder_arch
logger = logging.getLogger('base')
import numpy as np
torch.set_printoptions(precision=10)
from functools import partial

class DDM2Stage3(BaseModel):
    def __init__(self, opt):
        super(DDM2Stage3, self).__init__(opt)
        self.opt = opt

        self.denoise_fn = CNN_latent.CNN(in_channel=opt['model']['in_channel'],\
                                out_channel=opt['model']['out_channel'],\
                                hidden=opt['model']['hidden'], with_noise_level_emb = False)
        self.le = latent_encoder_arch.latent_encoder_gelu()
        from model.mri_modules import CNN
        self.first_denoisor = CNN.CNN(in_channel=opt['model']['in_channel'],\
                                out_channel=opt['model']['out_channel'],\
                                hidden=opt['model']['hidden'], with_noise_level_emb = False)

        # from model.mri_modules import CNN_d
        # self.net_d = CNN_d.CNN_d()

        from model.mri_modules import dit
        self.net_d = dit.DiT(depth=6, hidden_size=128, patch_size=4, num_heads=4)

        self.netG = noise_model3.N2N(
            self.le,
            self.denoise_fn,
            self.net_d,
            self.first_denoisor
        )

        self.netG = self.set_device(self.netG)

        self.optG = torch.optim.Adam(
            self.netG.parameters(), lr=opt['train']["optimizer"]["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, opt['train']['n_iter'],
                                                                    eta_min=opt['train']["optimizer"]["lr"] * 0.01)

        self.log_dict = OrderedDict()
        self.load_network()
        self.counter = 0

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self, setting):

        self.optG.zero_grad()

        outputs = self.netG(self.data, setting)

        l_pix = outputs['total_loss']
        l_pix.backward()

        self.optG.step()
        self.scheduler.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        # with torch.no_grad(): # TTT
        if isinstance(self.netG, nn.DataParallel):
            self.denoised = self.netG.module.denoise(
                self.data)
        else:
            self.denoised = self.netG.denoise(
                self.data)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['denoised'] = self.denoised.detach().float().cpu()
            out_dict['X'] = self.data['X'].detach().float().cpu()

        return out_dict

    def print_network(self):
        pass

    def save_network(self, epoch, iter_step, save_last_only=False):

        if not save_last_only:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        else:
            le_path = os.path.join(
                self.opt['path']['checkpoint'], 'le_gen.pth')
            denoise_fn_path = os.path.join(
                self.opt['path']['checkpoint'], 'denoised_gen.pth')
            diffusion_path = os.path.join(
                self.opt['path']['checkpoint'], 'diffusion_gen.pth')
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_opt.pth')
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        de_state = network.denoise_fn.state_dict()
        for key, param in de_state.items():
            de_state[key] = param.cpu()
        torch.save(de_state, denoise_fn_path)

        le_state = network.le.state_dict()
        for key, param in le_state.items():
            le_state[key] = param.cpu()
        torch.save(le_state, le_path)

        diffusion_state = network.net_d.state_dict()
        for key, param in diffusion_state.items():
            diffusion_state[key] = param.cpu()
        torch.save(diffusion_state, diffusion_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(denoise_fn_path))

    def load_network(self):
        stage1_load_path = self.opt['train']['resume_state']
        stage2_load_path = self.opt['train']['latent_state']
        network = self.netG
        if stage1_load_path is not None:     
            first_denoisor_path = '{}_stage1_gen.pth'.format(stage1_load_path)

            network.first_denoisor.load_state_dict(torch.load(
                first_denoisor_path), strict=True)        

        if stage2_load_path is not None:
            le_path = '{}le_gen.pth'.format(stage2_load_path)
            denoise_fn_path = '{}denoised_gen.pth'.format(stage2_load_path)
            diffusion_path = '{}diffusion_gen.pth'.format(stage2_load_path)

            network.le.load_state_dict(torch.load(
                le_path), strict=True)
            network.denoise_fn.load_state_dict(torch.load(
                denoise_fn_path), strict=True)
            
            if os.path.exists(diffusion_path):
                network.net_d.load_state_dict(torch.load(diffusion_path), strict=True)
