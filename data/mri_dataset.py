from curses import raw
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torch
from dipy.io.image import save_nifti, load_nifti
from matplotlib import pyplot as plt
import nibabel as nib


class MRIDataset(Dataset):
    def __init__(self, dataroot, valid_mask, phase='train', image_size=128, in_channel=1, val_volume_idx=50, val_slice_idx=40,
                 padding=1):
        self.padding = padding // 2
        self.phase = phase
        self.in_channel = in_channel

        # read data
        # raw_data, _ = load_nifti(dataroot) # width, height, slices, gradients
        raw_img = nib.load(dataroot)
        self.ori_affine = raw_img.affine
        raw_data = raw_img.get_fdata().astype(np.float32)
        print('Loaded data of size:', raw_data.shape)
        # normalize data
        
        # parse mask
        assert type(valid_mask) is (list or tuple) and len(valid_mask) == 2

        self.mmax = np.max(raw_data, axis=(0,1,2), keepdims=True)[...,valid_mask[0]:valid_mask[1]]
        self.raw_data = raw_data[...,valid_mask[0]:valid_mask[1]]/self.mmax

        if val_slice_idx == 'all':
            self.val_slice_idx = range(0, self.raw_data.shape[-2])
        elif type(val_slice_idx) is int:
            self.val_slice_idx = [val_slice_idx]
        elif type(val_slice_idx) is list:
            self.val_slice_idx = val_slice_idx

    def __len__(self):       
        return self.raw_data.shape[2] if self.phase == 'train' else len(self.val_slice_idx)

    def __getitem__(self, index):
        raw_input = self.raw_data

        if self.phase == 'val':
            s_index = self.val_slice_idx[index]
            raw_input = raw_input[:, :, s_index: s_index + 1 , :]
        else:
            raw_input = raw_input[:, :, index:index + 1 , :]

        w, h, c, d = raw_input.shape
        raw_input = np.reshape(raw_input, (w, h, -1))
        raw_input = torch.from_numpy(raw_input)
        raw_input = torch.permute(raw_input, (2, 0, 1))

        ret = dict(X=raw_input, condition=raw_input)

        return ret


if __name__ == "__main__":

    # hardi
    valid_mask = np.zeros(160,)
    valid_mask[10:] += 1
    valid_mask = valid_mask.astype(np.bool8)
    dataset = MRIDataset('.../HARDI150.nii.gz', valid_mask,
                         phase='train', val_volume_idx=40, padding=3)
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(trainloader):
        if i < 95 != 0:
            continue
        if i > 108:
            break
        img = data['X']
        condition = data['condition']
        img = img.numpy()
        condition = condition.numpy()

        vis = np.hstack((img[0].transpose(1,2,0), condition[0,[0]].transpose(1,2,0), condition[0,[1]].transpose(1,2,0)))
        # plt.imshow(img[0].transpose(1,2,0), cmap='gray')
        # plt.show()
        # plt.imshow(condition[0,[0]].transpose(1,2,0), cmap='gray')
        # plt.show()
        # plt.imshow(condition[0,[1]].transpose(1,2,0), cmap='gray')
        # plt.show()

        plt.imshow(vis, cmap='gray')
        plt.show()
        #break
