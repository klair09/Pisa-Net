import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json


class HsiRGBDataset(Dataset):
    def __init__(self, img_ids, hsi_dir, rgb_dir, mask_dir, hsi_ext='.npy', rgb_ext='.jpg', mask_ext='.png', three_band=False):
        super(HsiRGBDataset, self).__init__()
        self.filenames = img_ids   # ['2021-77690-3-10x-roi3-digi_3', '2021-77690-3-10x-roi3-digi_4', ...]
        self.hsi_dir = hsi_dir
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.hsi_extension = hsi_ext
        self.rgb_extension = rgb_ext
        self.mask_extension = mask_ext
        self.three_band=three_band
        self.length = len(self.filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        rgb_image = cv2.imread(os.path.join(self.rgb_dir, fname + self.rgb_extension), cv2.IMREAD_COLOR)[:512, :576, :]
        fname = fname.split("digi")[0] + 'mono' + fname.split("digi")[1]   # hsi/mask be like '77690-roi10-mono_1'  RGB be like '77690-roi10-digi_1'
        hsi_image = np.load(os.path.join(self.hsi_dir, fname + self.hsi_extension))[:512, :576, :]
        mask = cv2.imread(os.path.join(self.mask_dir, fname + self.mask_extension), cv2.IMREAD_GRAYSCALE)[:512, :576]

        hsi_image = hsi_image[:, :, 16:32]   # select bands
        hsi_image = hsi_image / np.max(hsi_image)
        rgb_image = rgb_image / np.max(rgb_image)
        threshold = 128
        _, mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)

        hsi_image = cv2.resize(hsi_image, None, fx=0.22, fy=0.25, interpolation=cv2.INTER_LINEAR)  # (128, 128, 16)
        rgb_image = cv2.resize(rgb_image, None, fx=0.22, fy=0.25, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=0.22, fy=0.25, interpolation=cv2.INTER_LINEAR)

        hsi_image = hsi_image.transpose((2, 0, 1)).copy().astype(np.float32)  # (C, H, W)
        rgb_image = rgb_image.transpose((2, 0, 1)).copy().astype(np.float32)

        return hsi_image, rgb_image, mask.astype(np.float32), fname


if __name__ == '__main__':
    dataset = {
        'img_test': r'E:\Assignment\Datasets\PLGC\test\hyper\images',
        'mask_test': r'E:\Assignment\Datasets\PLGC\test\hyper\masks',
        'rgb_total': r'E:\Assignment\Datasets\PLGC\color_stitch'
    }

    with open('plgc_txt/PLGC_demo.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    print(len(fcc_data['test']), fcc_data['test'])
    test_ids = [p.split('.')[0] for p in fcc_data["test"]]

    test_dataset = HsiRGBDataset(img_ids=test_ids, hsi_dir=dataset['img_train'], rgb_dir=dataset['rgb_total'], mask_dir=dataset['mask_train'])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    for i, (hsi, rgb, label, fname) in enumerate(test_loader, start=1):
        print(i, "hsi: {}  rgb: {}  label: {}".format(hsi.shape, rgb.shape, label.shape))
