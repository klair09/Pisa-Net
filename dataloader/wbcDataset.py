import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json


class WBCDataset(Dataset):
    def __init__(self, img_ids, data_dir, hsi_ext='.npy', rgb_ext='.npy', mask_ext='.npy'):
        super(WBCDataset, self).__init__()
        self.filenames = img_ids   # ['M-roi104', 'M-roi105', 'M-roi106', ...]
        self.data_dir = data_dir
        self.hsi_extension = hsi_ext
        self.rgb_extension = rgb_ext
        self.mask_extension = mask_ext
        self.length = len(self.filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        cell = fname.split('-')[0]
        data_dir = os.path.join(self.data_dir, cell)
        hsi_dir = os.path.join(data_dir, 'hsi')
        rgb_dir = os.path.join(data_dir, 'rgb')
        mask_dir = os.path.join(data_dir, 'mask')

        hsi_image = np.load(os.path.join(hsi_dir, fname + self.hsi_extension))   # [256, 320, 51]
        rgb_image = np.load(os.path.join(rgb_dir, fname + self.rgb_extension))   # [256, 320, 3]
        mask = np.load(os.path.join(mask_dir, fname + self.mask_extension))   # [256, 320]

        hsi_image = hsi_image[:, :, 10:26]
        hsi_image = hsi_image / np.max(hsi_image)
        rgb_image = rgb_image / np.max(rgb_image)

        # mask二值化，细胞核1，背景和其他类0
        mask[mask != 2] = 0
        mask[mask == 2] = 1

        hsi_image = cv2.resize(hsi_image, None, fx=0.4, fy=0.5, interpolation=cv2.INTER_LINEAR)  # (128, 128, 16)
        rgb_image = cv2.resize(rgb_image, None, fx=0.4, fy=0.5, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=0.4, fy=0.5, interpolation=cv2.INTER_LINEAR)

        hsi_image = hsi_image.transpose((2, 0, 1)).copy().astype(np.float32)
        rgb_image = rgb_image.transpose((2, 0, 1)).copy().astype(np.float32)

        return hsi_image, rgb_image, mask.astype(np.float32), fname


if __name__ == '__main__':
    dataset_root = r'E:\Assignment\Datasets\WBC'

    with open('D:\Projects\paper_code\dataloader\json_wbc\WBC_train_val_test.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    print(len(fcc_data['test']), fcc_data['test'])

    test_dataset = WBCDataset(img_ids=fcc_data['test'],  data_dir=dataset_root)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, drop_last=False)

    for i, (hsi, rgb, label, filenames) in enumerate(test_loader, start=1):
        print(i, "hsi: {}  rgb: {}  label: {}".format(hsi.shape, rgb.shape, label.shape))

