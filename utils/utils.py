import glob
import logging
import random
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os
import torch.nn.functional as F
import yaml
from torch.autograd import Function
from tqdm import tqdm
from .matrics import binary_metric_per_image


def set_seed(SEED=0):
    # PyTorch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(SEED)

    # Numpy
    np.random.seed(SEED)

    # Python's Random
    random.seed(SEED)


def cal_seg_metric(outputs, label, class_mean_dice,    # outputs: Tensor, label: Tensor, class_mean_dice: AverageMeter
                   is_train=True, dice_per_class=False):
    outputs = outputs.cpu().detach().numpy()
    targets = label.cpu().detach().numpy()
    for o, l in zip(outputs, targets):
        _, _, metric3 = binary_metric_per_image(o, l, is_train, dice_per_class)
        class_mean_dice.update(metric3.dice, label.shape[0])


def save_overlay_slices(patient_id, slice_idx, images, predict, labels, output_dir, folder):
    save_path = f'{output_dir}/{folder}'   # 'BraTS/trainer_visualize/train_epoch_iter/m1'
    os.makedirs(os.path.join(save_path, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)

    images = images.squeeze(1).cpu().numpy()  # B×D×H×W
    predict = predict.argmax(dim=1).detach().cpu().numpy()  # B×D×H×W
    labels = labels.cpu().numpy()  # B×D×H×W

    for i in range(images.shape[0]):
        image, pred, mask = images[0], predict[0], labels[0]
        for d in range(image.shape[0]):
            color_mask, color_gt = visualize_one_sample(image[d], pred[d], mask[d])
            cv2.imwrite(f'{save_path}/pred/{patient_id[i]}_{str(slice_idx)}_{d}.png', color_mask)
            cv2.imwrite(f'{save_path}/mask/{patient_id[i]}_{str(slice_idx)}_{d}.png', color_gt)


def visualize_one_sample(image, pred, label):  # numpy
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    p_et = pred == 3  # Enhanced tumor
    p_tc = np.logical_or(pred == 1, pred == 3)  # Tumor core
    p_wt = np.logical_or(p_tc, pred == 2)  # Whole tumor
    color_mask = image.copy()
    color_mask[p_wt] = [0, 255, 0]  # Green
    color_mask[p_tc] = [0, 0, 255]  # Red
    color_mask[p_et] = [10, 220, 240]  # Yellow

    t_et = label == 3  # Enhanced tumor
    t_tc = np.logical_or(label == 1, label == 3)  # Tumor core
    t_wt = np.logical_or(t_tc, label == 2)  # Whole tumor
    color_gt = image.copy()
    color_gt[t_wt] = [0, 255, 0]
    color_gt[t_tc] = [0, 0, 255]
    color_gt[t_et] = [10, 220, 240]

    return color_mask, color_gt


def save_metrics_to_excel(dice, hd, precision, recall, iou, acc, specificity, f1, save_path):
    import pandas as pd
    metrics_data = {
        'Dice': dice,
        'HD': hd,
        'Precision': precision,
        'Recall': recall,
        'IoU': iou,
        'Accuracy': acc,
        'Specificity': specificity,
        'F1': f1
    }
    with pd.ExcelWriter(os.path.join(save_path, 'metrics2.xlsx')) as writer:
        for metric, values in metrics_data.items():
            df = pd.DataFrame({metric: values})
            df.to_excel(writer, sheet_name=metric, index=False)
    print("\n\nExcel file for metrics created successfully!")


def cal_test_metrics(outs, labels, is_train=False,   # outs: numpy.array, labels: numpy.array
                     dice_per_class=False, save_metrics=False, save_path=None,
                     low_quantile=None):
    print(outs.shape, labels.shape)  # (218, 2, 256, 288)   (218, 256, 288)

    dice1, dice2 = [], []
    dice, hd, precision, recall, iou, acc, specificity, asd, f1 = [], [], [], [], [], [], [], [], []
    for o, l in zip(outs, labels):
        metric1, metric2, metric3 = binary_metric_per_image(o, l, is_train, dice_per_class)
        if dice_per_class:
            dice1.append(metric1.dice)
            dice2.append(metric2.dice)
        dice.append(metric3.dice)
        if not is_train:
            hd.append(metric3.hd)
            precision.append(metric3.precision)
            recall.append(metric3.recall)
            iou.append(metric3.iou)
            acc.append(metric3.accuracy)
            specificity.append(metric3.specificity)
            f1.append(metric3.f1)

    if dice_per_class:
        dice1 = np.array(dice1)
        dice2 = np.array(dice2)
        tqdm.write('Dice (background-only) = {:.2f} ± {:.2f}'.format(dice1.mean() * 100, dice1.std() * 100))
        tqdm.write('Dice (foreground-only) = {:.2f} ± {:.2f}'.format(dice2.mean() * 100, dice2.std() * 100))

    dice = np.array(dice)
    tqdm.write('Dice (class-mean) = {:.2f}±{:.2f}'.format(dice.mean() * 100, dice.std() * 100))

    if not is_train:
        iou = np.array(iou)
        tqdm.write('IoU = {:.2f}±{:.2f}'.format(iou.mean() * 100, iou.std() * 100))
        acc = np.array(acc)
        tqdm.write('Accuracy = {:.2f}±{:.2f}'.format(acc.mean() * 100, acc.std() * 100))
        precision = np.array(precision)
        tqdm.write('Precision = {:.2f}±{:.2f}'.format(precision.mean() * 100, precision.std() * 100))
        recall = np.array(recall)
        tqdm.write('Recall = {:.2f}±{:.2f}'.format(recall.mean() * 100, recall.std() * 100))
        specificity = np.array(specificity)
        tqdm.write('Specificity = {:.2f}±{:.2f}'.format(specificity.mean() * 100, specificity.std() * 100))
        f1 = np.array(f1)
        tqdm.write('F1 = {:.2f}±{:.2f}'.format(f1.mean() * 100, f1.std() * 100))
        hd = np.array(hd)

        if low_quantile is not None:
            lower_quantile = np.quantile(hd, low_quantile)
            upper_quantile = np.quantile(hd, 1-lower_quantile)
            hd = hd[(hd >= lower_quantile) & (hd <= upper_quantile)]
            print(hd.shape)

        tqdm.write('HD score = {:.2f}±{:.2f}'.format(hd.mean(), hd.std()))

    if save_metrics and save_path is not None:
        save_metrics_to_excel(dice, hd, precision, recall, iou, acc, specificity, f1, save_path)


def save_predict_map(logits, mask, filenames, folder, GT='color'):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    predict = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)   像素值[0,1]
    predict = np.clip(predict * 255, 0, 255)  # 像素值[0,255]
    mask = (mask * 255).cpu().numpy()   # (B, H, W)   像素值0、255
    predict = np.uint8(predict)
    mask = np.uint8(mask)

    if GT == 'boundary':   # ground truth 用红色轮廓线进行表示
        for i in range(predict.shape[0]):
            pred = predict[i, :, :]
            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            m = mask[i, :, :]
            contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                cv2.drawContours(pred, [c], 0, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(folder, filenames[i] + '.jpg'), pred)

    elif GT == 'color':   # true positive、false positive、false negative 用三种色块进行表示
        for i in range(predict.shape[0]):
            pred = predict[i, :, :]
            m = mask[i, :, :]

            tp = (pred != 0) & (m != 0)
            fp = (pred != 0) & (m == 0)
            fn = (pred == 0) & (m != 0)

            color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            '''PisaNet'''
            color_mask[tp] = [248, 203, 227]
            color_mask[fp] = [97, 217, 254]
            color_mask[fn] = [80, 208, 146]
            cv2.imwrite(os.path.join(folder, filenames[i] + '.jpg'), color_mask)


def save_segmentation(logits, mask, n_class, filenames, folder, save_mask=False):  # for GTA dataset (3-classes)
    predict = logits.argmax(dim=1).cpu().numpy()  # (B, H, W)   类别标签：0，1，2
    # predict = np.clip(predict * 255, 0, 255)
    mask = mask.cpu().numpy()   # (B, H, W)
    predict = np.uint8(predict)
    mask = np.uint8(mask)

    def visualize(x, type):
        for i in range(x.shape[0]):
            x_ = x[i, :, :]
            color_mask = np.zeros((x_.shape[0], x_.shape[1], 3), dtype=np.uint8)
            colors = [[255, 255, 136], [0, 128, 255], [102, 255, 255], [248, 203, 227], [97, 217, 254]]
            for j in range(n_class):
                color_mask[x_ == j] = colors[j]
            cv2.imwrite(f'{folder}/{type}/{filenames[i]}.png', color_mask)

    pred_path = os.path.join(folder, 'pred')
    if not os.path.exists(pred_path):
        os.makedirs(pred_path, exist_ok=True)
    visualize(predict, 'pred')

    if save_mask:
        mask_path = os.path.join(folder, 'mask')
        if not os.path.exists(mask_path):
            os.makedirs(mask_path, exist_ok=True)
        visualize(mask, 'mask')

