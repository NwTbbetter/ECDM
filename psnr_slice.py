import nibabel as nib
import numpy as np

from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_metrics(prediction_path):
    gt_data = nib.load("../data/dw.nii.gz").get_fdata()
    print(gt_data.shape)
    gt_data = gt_data[...,:31]

    prediction = nib.load(prediction_path).get_fdata()
    print(prediction.shape)

    mask = nib.load('../data/dw_b0_mask.nii.gz').get_fdata()

    x_size = mask.shape[0]
    y_size = mask.shape[1]
    z_size = mask.shape[2]
    metrics_prediction = []
    metrics_dmri_data = []

    for xx in range(0, x_size, 1):
        for yy in range(0, y_size, 1):
                for zz in range(0, z_size, 1):
                    if mask[xx, yy, zz] > 0:
                        metrics_prediction.append(prediction[xx, yy, zz, :])
                        metrics_dmri_data.append(gt_data[xx, yy, zz, :])
                    else:
                        prediction[xx, yy, zz, :] = 0
                        gt_data[xx, yy, zz, :] = 0
    metrics_dmri_data = np.array(metrics_dmri_data).reshape(-1)
    metrics_prediction = np.array(metrics_prediction).reshape(-1)

    max = int(metrics_dmri_data.max())
    psnr_value = psnr(metrics_dmri_data, metrics_prediction, data_range=max)
    ssim_value = ssim(metrics_dmri_data, metrics_prediction, data_range=max)
    rmse_value = np.sqrt(mse(metrics_dmri_data, metrics_prediction))
    print("psnr:", psnr_value,"ssim:", ssim_value,"rmse:", rmse_value)
    return psnr_value, ssim_value, rmse_value

if __name__ == '__main__':
    compute_metrics('./stage3_best_20.nii.gz')