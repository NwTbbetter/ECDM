import numpy as np
import nibabel as nib
from time import time
import os
from dipy.io.image import save_nifti
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
# load main pca function using Marcenko-Pastur distribution
from dipy.denoise.localpca import mppca
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs



snr = 'D_1000_snr20_eddy'


dmri_data_path = '../data/new/p_gt.nii.gz' #干净数据路径
noisy_data_path = '../data/new/p_noisy_15.nii.gz' #噪声数据路径
bvals_path = '../data/new/p61_bval' #bval路径
bvecs_path = '../data/new/p61_bvec' #bvec路径
mask_path = '../data/new/p_b0_mask.nii.gz' #mask文件路径

# dmri_data_path = '../data/dw.nii.gz' #干净数据路径
# noisy_data_path = '../data/dw_15.nii.gz' #噪声数据路径
# bvals_path = '../data/dw_bval' #bval路径
# bvecs_path = '../data/dw_bvec' #bvec路径
# mask_path = '../data/dw_b0_mask.nii.gz' #mask文件路径

dmri_data = nib.load(dmri_data_path).get_fdata()
noisy_data = nib.load(noisy_data_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

bvals, bvecs = read_bvals_bvecs(bvals_path, bvecs_path)
sel_b = bvals != 0
noisy_data = noisy_data[..., sel_b]
gtab = gradient_table(bvals[sel_b], bvecs[sel_b])
bvals = gtab.bvals
bvecs = gtab.bvecs

noisy_data = noisy_data[..., :31]
dmri_data = dmri_data[..., :31]
print(noisy_data.shape)

# noisy_data = noisy_data.astype(np.float32) / np.max(noisy_data, axis=(0,1,2), keepdims=True).astype(np.float32)
# dmri_data = dmri_data.astype(np.float32) / np.max(dmri_data, axis=(0,1,2), keepdims=True).astype(np.float32)

t = time()

denoised_arr = mppca(noisy_data, patch_radius=2)

print("Time taken for local MP-PCA ", -t + time())

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

# denoised_arr = np.concatenate((noisy_data[:,:,:,:sel_b], denoised_arr), axis=-1)
save_nifti('./dw15_denoisy_mppca_'+str(snr)+'.nii.gz', denoised_arr, np.eye(4))
# rediual = np.sqrt((denoised_arr - dmri_data) ** 2)
# save_nifti('./denoisy_mppca_'+str(snr)+'_rediual.nii.gz', rediual, np.eye(4))

metrics_dmri_data = np.array(metrics_dmri_data)
metrics_prediction = np.array(metrics_prediction)
# print(metrics_prediction.shape)
# print(metrics_dmri_data.shape)

# metrics_dmri_data = metrics_dmri_data[:,:,27,30]
# metrics_prediction = metrics_prediction[:,:,27,30]
metrics_dmri_data = np.array(metrics_dmri_data).reshape(-1)
metrics_prediction = np.array(metrics_prediction).reshape(-1)


max = int(metrics_dmri_data.max())
psnr_value = psnr(metrics_dmri_data, metrics_prediction, data_range=max)
ssim_value = ssim(metrics_dmri_data, metrics_prediction, data_range=max)
rmse_value = np.sqrt(mse(metrics_dmri_data, metrics_prediction))

print('psnr is {}, ssim is {}, rmse is {}'.format(psnr_value, ssim_value, rmse_value))

# log_path = os.path.join("./mppca_"+str(snr)+".csv")
# with open(log_path, "a") as f:
#      f.writelines("psnr is {}, ssim is {}, rmse is {}\n".format(psnr_value, ssim_value, rmse_value))