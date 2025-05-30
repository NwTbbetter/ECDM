import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
# from tensorboardX import SummaryWriter
import os
import numpy as np
import nibabel as nib
# from CNN import CNN
from time import time


print(torch.__version__, torch.version.cuda)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/hardi_150.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-o', '--val_out', type=str, default='stage3.nii.gz')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args, stage=3)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False    

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info('[Phase 1] Training noise model!')

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    trainer = Model.create_noise_model_diffusion(opt)
    logger.info('Initial Model Finished')
    # Train
    current_step = trainer.begin_step
    current_epoch = trainer.begin_epoch
    n_iter = opt['train']['n_iter']

    loss = dict(operation_seed_counter=0, n_iter=n_iter)
    loss_all_history = []    
    if opt['phase'] == 'train':
        # t = time()
        while current_step < n_iter:
            current_epoch += 1
            loss['epoch_loss1'] = 0
            loss['epoch_loss2'] = 0
            loss['epoch_all_loss'] = 0
            loss['number'] = 0
            for _, train_data in enumerate(train_loader):
                current_step += 1
                loss['current_epoch'] = current_epoch
                loss['current_step'] = current_step
                if current_step > n_iter:
                    break
                trainer.feed_data(train_data)
                trainer.optimize_parameters(loss)
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = trainer.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                    logger.info(message)
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)
                
                    for _, val_data in enumerate(val_loader):
                        idx += 1
                        trainer.feed_data(val_data)
                        trainer.test(continous=False)
                
                        visuals = trainer.get_current_visuals()
                        v_index = opt['datasets']['val']['val_volume_idx']
                        visuals['denoised'] = visuals['denoised'][:, v_index, :, :]
                        visuals['X'] = visuals['X'][:, v_index, :, :]
                        denoised_img = Metrics.tensor2img(visuals['denoised'])  # uint8
                        input_img = Metrics.tensor2img(visuals['X'])  # uint8
                
                        Metrics.save_img(
                            denoised_img[:, :], '{}/{}_{}_denoised.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            input_img[:, :], '{}/{}_{}_input.png'.format(result_path, current_step, idx))

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    trainer.save_network(current_epoch, current_step, save_last_only=True)
        # print('Training time: ', time() - t)
    t = time()
    if opt['phase'] == 'val':
        prediction_all = []
        prediction = []
        for _,  val_data in enumerate(val_loader):
            trainer.feed_data(val_data)
            trainer.test(continous=False)
        
            visuals = trainer.get_current_visuals(need_LR=False)
            # print(visuals['denoised'].shape)
            denoised_img = Metrics.tensor2img(visuals['denoised'], out_type=np.float32)
            # print(visuals['denoised'].shape)
            denoised_img = np.expand_dims(denoised_img, axis=2)
            if len(prediction) == 0:
                prediction = denoised_img
            else:
                prediction = np.concatenate((prediction,denoised_img), axis=-2)
        
        print('Training time: ', time() - t)
        prediction = prediction.astype(np.float32) * val_set.mmax
        print(prediction.shape)
        
        img_prediction = nib.Nifti1Image(np.float32(prediction), val_set.ori_affine)
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        nib.save(img_prediction, result_path + '/'+ args.val_out)
        print('save img successful')

        # from psnr_slice import compute_metrics
        # snr,cnr,snr_min = compute_metrics(result_path + '/'+ args.val_out)

