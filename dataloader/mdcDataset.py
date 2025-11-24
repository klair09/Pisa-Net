import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MDCDataset(Dataset):
    def __init__(self, img_ids, hsi_dir, rgb_dir, mask_dir):
        super(MDCDataset, self).__init__()
        self.filenames = img_ids   # ['040579-20x-roi2_1', '034247_2-20x-roi5_2', ...]
        self.hsi_dir = hsi_dir
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.length = len(self.filenames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        hsi_image = np.load(os.path.join(self.hsi_dir, fname+'.npy'))   # [256, 320, 60]
        rgb_image = np.load(os.path.join(self.rgb_dir, fname+'.npy'))    # [256, 320, 3]
        mask = np.load(os.path.join(self.mask_dir, fname+'.npy'))    # [256, 320]

        hsi_image = hsi_image[:, :, 12:28]
        hsi_image = hsi_image / np.max(hsi_image)
        rgb_image = rgb_image / np.max(rgb_image)
        threshold = 128
        _, mask = cv2.threshold(mask, threshold, 1, cv2.THRESH_BINARY)   # {0,255} -> {0,1}

        hsi_image = cv2.resize(hsi_image, None, fx=0.4, fy=0.5, interpolation=cv2.INTER_LINEAR)  # (128, 128, 16)
        rgb_image = cv2.resize(rgb_image, None, fx=0.4, fy=0.5, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        hsi_image = hsi_image.transpose((2, 0, 1)).copy().astype(np.float32)
        rgb_image = rgb_image.transpose((2, 0, 1)).copy().astype(np.float32)

        return hsi_image, rgb_image, mask.astype(np.float32), fname


if __name__ == '__main__':
    hsi_path = r'E:\Assignment\Datasets\MDC\preprocess_hyper_downfour_input\npy_files'
    rgb_path = r'E:\Assignment\Datasets\MDC\RGB\down_four_npy_files'
    mask_path = r'E:\Assignment\Datasets\MDC\mask-new\mask_down_four\npy_files'

    import json
    with open('json_mdc/MDC_train_val_test.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    print(len(fcc_data['test']), fcc_data['test'])
    test_ids = [p.split('.')[0] for p in fcc_data["test"]]

    test_dataset = MDCDataset(img_ids=test_ids,  hsi_dir=hsi_path, rgb_dir=rgb_path, mask_dir=mask_path)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

    for i, (hsi, rgb, label, fname) in enumerate(test_loader, start=1):
        print(i, "hsi: {}  rgb: {}  label: {}".format(hsi.shape, rgb.shape, label.shape))
