import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from dipy.io.image import load_nifti, save_nifti
import nibabel as nib

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

snr = 'D_61_700_2000_snr40' #保存命名

dmri_data_path = '../data/new/p_gt.nii.gz' #干净数据路径
noisy_data_path = '../data/new/p_noisy_20.nii.gz' #噪声数据路径
bvals_path = '../data/new/p61_bval' #bval路径
bvecs_path = '../data/new/p61_bvec' #bvec路径
mask_path = '../data/new/p_b0_mask.nii.gz' #mask文件路径

dmri_data = nib.load(dmri_data_path).get_fdata()
noisy_data = nib.load(noisy_data_path).get_fdata()

# from dipy.segment.mask import median_otsu
# b0_mask, mask = median_otsu(noisy_data, vol_idx=[0])
# bvals = np.loadtxt(bvals_path)
mask = nib.load(mask_path).get_fdata()
noisy_data = noisy_data[...,:31]
dmri_data = dmri_data[...,:31]
# mask = mask[...,:31]

# noisy_data = noisy_data.astype(np.float32) / np.max(noisy_data, axis=(0,1,2), keepdims=True).astype(np.float32)
# dmri_data = dmri_data.astype(np.float32) / np.max(dmri_data, axis=(0,1,2), keepdims=True).astype(np.float32)

sigma = estimate_sigma(noisy_data, N=32)
t = time()
denoised_arr = nlmeans(noisy_data, sigma=sigma, mask=mask, patch_radius=1,
              block_radius=2, rician=False)
print("Time taken for NLM ", -t + time())

metrics_prediction = []
metrics_dmri_data = []

x_size = mask.shape[0]
y_size = mask.shape[1]
z_size = mask.shape[2]
for xx in range(0, x_size, 1):
    for yy in range(0, y_size, 1):
        for zz in range(0, z_size, 1):
            if mask[xx, yy, zz] > 0:
                metrics_prediction.append(denoised_arr[xx, yy, zz, :])
                metrics_dmri_data.append(dmri_data[xx, yy, zz, :])
            else:
                denoised_arr[xx, yy, zz, :] = 0
                dmri_data[xx, yy, zz, :] = 0

save_nifti('./denoisy_nlm_'+str(snr)+'.nii.gz', denoised_arr, np.eye(4))
# rediual = np.sqrt((denoised_arr - dmri_data) ** 2)
# save_nifti('./denoisy_nlm_'+str(snr)+'_rediual.nii.gz', rediual, np.eye(4))

metrics_dmri_data = np.array(metrics_dmri_data).reshape(-1)
metrics_prediction = np.array(metrics_prediction).reshape(-1)

max = int(metrics_dmri_data.max())
psnr_value = psnr(metrics_dmri_data, metrics_prediction, data_range=max)
ssim_value = ssim(metrics_dmri_data, metrics_prediction, data_range=max)
rmse_value = np.sqrt(mse(metrics_dmri_data, metrics_prediction))

print('psnr is {}, ssim is {}, rmse is {}'.format(psnr_value, ssim_value, rmse_value))