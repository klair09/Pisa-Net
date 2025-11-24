import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from medpy_binary import __surface_distances, jc, dc


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def average(self):
        return self.avg


class LOG():
    def __init__(self):
        self.loss = AverageMeter()
        self.metric = AverageMeter()
        self.hsi_loss = AverageMeter()
        self.rgb_loss = AverageMeter()
        self.fuse_loss = AverageMeter()

    def update_log(self, n, loss, hsi_loss, rgb_loss, fuse_loss, logit, label):
        self.loss.update(loss.item(), n)
        self.hsi_loss.update(hsi_loss.item(), n)
        self.rgb_loss.update(rgb_loss.item(), n)
        self.fuse_loss.update(fuse_loss.item(), n)

        logit = logit.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        for o, l in zip(logit, label):
            _, _, m = binary_metric_per_image(o, l)
            self.metric.update(m.dice, n)


def cal_seg_metric(outputs, label, class_mean_dice,    # outputs: Tensor, label: Tensor, class_mean_dice: AverageMeter
                   is_train=True, dice_per_class=False):
    outputs = outputs.cpu().detach().numpy()
    targets = label.cpu().detach().numpy()
    for o, l in zip(outputs, targets):
        _, _, metric3 = binary_metric_per_image(o, l, is_train, dice_per_class)
        class_mean_dice.update(metric3.dice, label.shape[0])


def intersect_and_union(pred_label, label, num_classes, ignore_index=None):
    if ignore_index is not None:
        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(intersect, bins=np.arange(num_classes + 1))
    area_pred, _ = np.histogram(pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))
    area_union = area_pred + area_label - area_intersect
    return area_intersect, area_union, area_pred, area_label


def hd(result, reference, voxelspacing=None, connectivity=1):
    if np.sum(result) == 0:   # {foreground pixel}=∅, hd=inf
        return -1

    hd1 = __surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


class SegMetric():
    def __init__(self, n_class, index=(1,)):
        self.n_class = n_class
        self.index = index
        self.dice = None
        self.hd = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.iou = None
        self.specificity = None
        self.f1 = None

    def fit(self, pred, target, EPS=1e-6, calculate_all=False):
        try:
            assert len(pred.shape) == len(target.shape) == 2
        except:
            print(pred.shape, target.shape)

        self.dice = dc(pred, target)

        if calculate_all:
            result = numpy.atleast_1d(pred.astype(numpy.bool_))
            reference = numpy.atleast_1d(target.astype(numpy.bool_))
            tp = numpy.count_nonzero(result & reference)
            tn = numpy.count_nonzero(~result & ~reference)
            fp = numpy.count_nonzero(result & ~reference)
            fn = numpy.count_nonzero(~result & reference)

            self.hd = hd(pred, target)
            self.iou = jc(pred, target)
            self.accuracy = (tp + tn) / float(tp + fp + tn + fn + EPS)
            self.precision = tp / float(tp + fp + EPS)
            self.recall = tp / float(tp + fn + EPS)
            self.specificity = tn / float(tn + fp + EPS)
            self.f1 = 2 * self.precision * self.recall / float(self.precision + self.recall + EPS)


def binary_metric_per_image(logits, target, is_train=True, dice_per_class=False):   # numpy
    predict = np.argmax(logits, axis=0)
    metric1, metric2, metric3 = None, None, None

    metric3 = SegMetric(n_class=2, index=(0, 1))  # background & foreground
    if is_train:
        metric3.fit(predict, target)   # calculate dice only
    else:
        metric3.fit(predict, target, calculate_all=True)   # calculate all metrics

    if dice_per_class:
        metric1 = SegMetric(n_class=2, index=(0,))  # background
        metric1.fit(predict, target)

        metric2 = SegMetric(n_class=2, index=(1,))  # foreground
        metric2.fit(predict, target)

    return metric1, metric2, metric3


class ConfusionMatrix:
    def __init__(self, test=None, reference=None):
        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.reference = reference
        self.test = test

    def compute(self):
        assert self.test.shape == self.reference.shape, "Shape mismatch: {} and {}".format(self.test.shape, self.reference.shape)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())

    def get_matrix(self):
        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break
        return self.tp, self.fp, self.tn, self.fn


def dice_per_image(test=None, reference=None, eps=1e-7):   # test: [2,H,W]   reference: [2,H,W]
    """2TP / (2TP + FP + FN)"""

    test_dice = []
    for i in range(test.shape[0]):   # each class in one-hot
        confusion_matrix = ConfusionMatrix(test[i], reference[i])
        tp, fp, tn, fn = confusion_matrix.get_matrix()
        dc = float((2 * tp) / (2 * tp + fp + fn + eps))
        test_dice.append(dc)
    return np.array(test_dice)


def tpr_fpr_per_image(test=None, reference=None, eps=1e-7):   # test: [2,H,W]   reference: [2,H,W]
    tpr, fpr = [], []
    for i in range(test.shape[0]):  # each class in one-hot
        confusion_matrix = ConfusionMatrix(test[i], reference[i])
        tp, fp, tn, fn = confusion_matrix.get_matrix()
        TPR = float(tp / (tp + fn + eps))
        FPR = float(fp / (fp + tn + eps))
        tpr.append(np.round(TPR, 4))
        fpr.append(np.round(FPR, 4))
    return np.array(tpr), np.array(fpr)


def onehot_encoding(label, n_classes=2):   # label: numpy
    label = torch.tensor(label)
    label = label.long()
    B, H, W = label.shape
    one_hot = torch.zeros(B, n_classes, H, W, device=label.device)
    one_hot.scatter_(1, label.unsqueeze(1), 1)
    return one_hot.cpu().detach().numpy()
