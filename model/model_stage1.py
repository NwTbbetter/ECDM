import logging
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
import os
from model.mri_modules import CNN
from model.mri_modules import noise_model
from torch.optim import lr_scheduler
from .base_model import BaseModel
logger = logging.getLogger('base')

torch.set_printoptions(precision=10)


class DDM2Stage1(BaseModel):
    def __init__(self, opt):
        super(DDM2Stage1, self).__init__(opt)
        self.opt = opt

        # basic uent
        self.denoisor = CNN.CNN(in_channel=opt['model']['in_channel'],\
                                out_channel=opt['model']['out_channel'],\
                                hidden=opt['model']['hidden'], with_noise_level_emb = False)

        self.netG = noise_model.N2N(
            self.denoisor
        )

        self.netG = self.set_device(self.netG)


        self.optG = torch.optim.Adam(
            self.netG.parameters(), lr=opt['train']["optimizer"]["lr"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, opt['train']['n_iter'],\
                                                                     eta_min=opt['train']["optimizer"]["lr"]*0.01)
        
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
        #with torch.no_grad(): # TTT
        if isinstance(self.netG, nn.DataParallel):
            self.denoised = self.netG.module.denoise(
                self.data)
        else:
            self.denoised = self.netG.denoise(
                self.data)
        self.netG.train()

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

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
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_stage1_gen.pth')
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_stage1_opt.pth')
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.first_denoisor.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['train']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_stage1_gen.pth'.format(load_path)
            opt_path = '{}_stage1_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.first_denoisor.load_state_dict(torch.load(
                gen_path), strict=True)

            # network = torch.load(load_path)

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
