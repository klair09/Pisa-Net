import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_intra_loss(features, subset_idx, mode='vector'):
    if mode == 'vector':
        # [N, S, h, w] -> [a, S*h*w]
        subset = features[subset_idx].reshape(len(subset_idx), -1)
    elif mode == 'map':
        # [B, N, H, W] -> [B, a, H*W]
        subset = features[:, subset_idx].reshape(features.shape[0], len(subset_idx), -1)
        subset = subset.mean(dim=0)  # [a, H*W]
    else:
        raise ValueError("mode must be 'vector' or 'map'")
    a = subset.shape[0]
    loss = 0.0
    count = 0
    for i in range(a):
        for j in range(i + 1, a):
            cos_sim = F.cosine_similarity(subset[i].unsqueeze(0), subset[j].unsqueeze(0), dim=-1)
            loss += 1.0 - cos_sim
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=features.device)


def compute_inter_loss(features, subset_idx, mode='vector'):
    all_idx = list(range(features.shape[0])) if mode == 'vector' else list(range(features.shape[1]))
    complement_idx = list(set(all_idx) - set(subset_idx))

    if mode == 'vector':
        A = features[subset_idx].reshape(len(subset_idx), -1)
        B = features[complement_idx].reshape(len(complement_idx), -1)
    elif mode == 'map':
        A = features[:, subset_idx].reshape(features.shape[0], len(subset_idx), -1).mean(dim=0)
        B = features[:, complement_idx].reshape(features.shape[0], len(complement_idx), -1).mean(dim=0)
    else:
        raise ValueError("mode must be 'vector' or 'map'")

    loss = 0.0
    count = 0
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            cos_sim = F.cosine_similarity(A[i].unsqueeze(0), B[j], dim=-1)
            loss += 1.0 / (1.0 - cos_sim + 1e-6)
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0, device=features.device)


class CELoss(nn.Module):
    def __init__(self, num_classes=2, ignore_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        return F.cross_entropy(prediction, target.long(), ignore_index=self.ignore_index)


class LovaszSoftmax(nn.Module):
    def __init__(self):
        super(LovaszSoftmax, self).__init__()

    def _lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper

        https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)

        # Jaccard索引（IoU score） = intersection / union
        # Jaccard损失 = 1 - Jaccard索引
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, input, target, ignore_index=None):
        B, N_CLASS, H, W = input.shape
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, N_CLASS)  # [B * H * W, N_CLASS]
        target = target.view(-1)  # [B * H * W, ]
        # target = target // 255

        if ignore_index is None:
            pass
        else:
            mask = (target != ignore_index)
            input = input[mask.nonzero().squeeze()]
            target = target[mask]

        if input.numel() == 0:  # 处理边缘情况，当没有有效预测时，梯度应该是0
            return input * 0

        N_CLASS = input.shape[1]
        assert N_CLASS > 1
        loss = []
        for c in range(N_CLASS):
            foreground = (target == c).float()
            if foreground.sum() == 0:
                continue
            predict = input[:, c]
            loss_c = (torch.autograd.Variable(foreground) - predict).abs()  # 第c个类别的pixel errors
            loss_c_sorted, loss_idx = torch.sort(loss_c, 0, descending=True)
            foreground_sorted = foreground[loss_idx]
            grad = self._lovasz_grad(foreground_sorted)  # 根据降序排列的前景标记计算Lovasz梯度（定义式中的Jaccard系数）
            loss.append(torch.dot(loss_c_sorted, torch.autograd.Variable(grad)))  # 点积：损失和梯度之间的加权和

        loss = torch.stack(loss)
        return loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, n_classes=2, include_background=True, squared_pred=True, weight=[1.3,0.8,0.9], redunction='mean'):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.weight = weight
        self.reduction = redunction

    def _dice_loss(self, score, target, smooth=1e-5):    # score: B*H*W, target: B*H*W
        target = target.float()
        dim = (1, 2)
        intersect = torch.sum(score * target, dim=dim)
        if self.squared_pred:
            y_sum = torch.sum(target * target, dim=dim)
            z_sum = torch.sum(score * score, dim=dim)
        else:
            y_sum = torch.sum(target, dim=dim)
            z_sum = torch.sum(score, dim=dim)
        loss = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return loss.mean()

    def forward(self, inputs, target):
        target = F.one_hot(target.long(), num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        assert inputs.size() == target.size(), (
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size()))

        if self.weight is None:
            self.weight = [1] * self.n_classes

        loss = 0.0
        for i in range(0 if self.include_background else 1, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * self.weight[i]

        den = self.n_classes if self.include_background else (self.n_classes - 1)
        return (loss / den) if self.reduction == 'mean' else loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, include_background=True, squared_pred=True, one_hot=True, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.squared_pred = squared_pred
        self.reduction = reduction
        self.one_hot = one_hot

    def forward(self, predict, target):    # input:[B, C, H, W]   target:[B, H, W]
        if self.one_hot:
            target = F.one_hot(target.long(), num_classes=predict.shape[1]).permute(0, 3, 1, 2).float()   # [B, C, H, W]

        if not self.include_background:
            predict = predict[:, 1:]
            target = target[:, 1:]

        # predict = predict.contiguous().view(predict.shape[0], -1)
        # target = target.contiguous().view(target.shape[0], -1)
        predict = rearrange(predict, 'b c h w -> (b c) (h w)')
        target = rearrange(target, 'b c h w -> (b c) (h w)')

        if self.squared_pred:
            den = torch.sum(target * target, dim=1) + torch.sum(predict * predict, dim=1)
        else:
            den = torch.sum(predict, dim=1) + torch.sum(target, dim=1)
        num = torch.sum(predict * target, dim=1)
        loss = 1 - (2 * num + self.smooth) / (den + self.smooth)
        return loss.mean() if self.reduction == 'mean' else loss










