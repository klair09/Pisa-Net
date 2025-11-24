import torch
import torch.nn as nn
from .utils import init_weights
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.GroupNorm(num_groups=1, num_channels=out_size),
                                     nn.LeakyReLU(negative_slope=0.1, inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs, fname=None, layer=None):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

            if fname and layer:
                folder = r'D:\Projects\paper_code\dataloader\HSIs_to_show\feature_maps_unet2d\\' + str(layer)
                if not os.path.exists(folder):
                    os.makedirs(folder, exist_ok=True)

                for k in range(x.shape[0]):  # x:[b,c,h,w]
                    mp = x[k].detach().cpu().numpy()
                    mp = np.sum(mp, axis=0)

                    mp = np.clip((mp / 128) * 255, 0, 255).astype(np.uint8)

                    ax = plt.subplot(1, 1, 1)
                    im = ax.imshow(mp, cmap='coolwarm')
                    ax.axis('off')
                    divider = plt.matplotlib.colorbar.make_axes_gridspec(ax)
                    cbar = plt.colorbar(im, cax=divider[0], orientation='vertical')
                    cbar.formatter = FormatStrFormatter('%.1f')
                    cbar.update_ticks()
                    plt.savefig(os.path.join(folder, fname[k] + str(i) + '.png'))

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                 nn.UpsamplingBilinear2d(scale_factor=2),
                 nn.Conv2d(in_size, out_size, 1))
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)

        for feature in low_feature:
            outputs0 = outputs0 + feature

        return outputs0

