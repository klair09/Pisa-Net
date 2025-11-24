from torch.nn import init
import cv2
import os
import numpy as np


### initalize the module
def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def _get_feature_maps(cur_img, layer):
    for j in range(1, layer):
        cur_img = cv2.resize(cur_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cur_img = cv2.normalize(cur_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feature_map_colored = cv2.applyColorMap(cur_img, cv2.COLORMAP_JET)
    return feature_map_colored

def save_feature_maps(x, fname, froot, layer):
    for i in range(x.shape[0]):  # x:[b,c,h,w]
        rgb_img = x[i].detach().cpu().numpy()
        save_path = os.path.join(froot, f'{fname[i]}_layer{layer}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for idx in range(rgb_img.shape[0]):
            cur_img = rgb_img[idx]  # (h,w)
            feature_map_colored = _get_feature_maps(cur_img, layer)
            for j in range(layer // 2, 3):
                cur_img = cv2.resize(cur_img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_path, fname[i] + f'_l{layer}_c{idx}.png'), feature_map_colored)
    return layer + 1