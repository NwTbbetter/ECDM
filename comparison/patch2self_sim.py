import numpy as np
import os
import os.path as osp
from time import time
import matplotlib.pyplot as plt
from dipy.io.image import save_nifti
import nibabel as nib
from dipy.denoise.patch2self import patch2self
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

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
bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)

sel_b = bvals != 0
# b0_size = 10
# noisy_data = noisy_data[..., sel_b]
# bvals = bvals[sel_b]
noisy_data = noisy_data[..., :31]
bvals = bvals[:31]
mask = nib.load(mask_path).get_fdata()
dmri_data = dmri_data[...,:31]

# noisy_data = noisy_data.astype(np.float32) / np.max(noisy_data, axis=(0,1,2), keepdims=True).astype(np.float32)
# dmri_data = dmri_data.astype(np.float32) / np.max(dmri_data, axis=(0,1,2), keepdims=True).astype(np.float32)

t = time()
denoised_arr = patch2self(noisy_data, bvals, model='ols', shift_intensity=True,
                              clip_negative_vals=False, b0_threshold=50)
print("Time taken for Patch2self ", -t + time())
print(denoised_arr.shape)

metrics_prediction = []
metrics_dmri_data = []

x_size = mask.shape[0]
y_size = mask.shape[1]
z_size = mask.shape[2]


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

save_nifti('./dw15_denoisy_patch2self_'+str(snr)+'.nii.gz', denoised_arr, np.eye(4))
# rediual = np.sqrt((denoised_arr - dmri_data) ** 2)
# save_nifti('./denoisy_patch2self_'+str(snr)+'_rediual.nii.gz', rediual, np.eye(4))

metrics_dmri_data = np.array(metrics_dmri_data).reshape(-1)
metrics_prediction = np.array(metrics_prediction).reshape(-1)

max = int(metrics_dmri_data.max())
psnr_value = psnr(metrics_dmri_data, metrics_prediction, data_range=max)
ssim_value = ssim(metrics_dmri_data, metrics_prediction, data_range=max)
rmse_value = np.sqrt(mse(metrics_dmri_data, metrics_prediction))

print('psnr is {}, ssim is {}, rmse is {}'.format(psnr_value, ssim_value, rmse_value))

# log_path = os.path.join("./patch2self_"+str(snr)+".csv")
# with open(log_path, "a") as f:
#      f.writelines("psnr is {}, ssim is {}, rmse is {}\n".format(psnr_value, ssim_value, rmse_value))


