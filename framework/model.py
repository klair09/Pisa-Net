import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers import unetConv2, unetUp
from utils import init_weights


class FFU(nn.Module):
    def __init__(self, in_channel=1024, out_channel=512,  num_layers=3,
                 kernel_size=3, stride=1, padding=1, threshold=0.1, reduction=4):
        super(FFU, self).__init__()
        self.num_layers = num_layers
        self.layer_in = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channel)
        )
        self.s_lambda_in = nn.Parameter(torch.Tensor([threshold]))

        for i in range(self.num_layers):
            down = nn.Sequential(
                nn.Conv2d(out_channel, out_channel // reduction, kernel_size, stride, padding, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channel // reduction)
            )
            setattr(self, f'down_{i}', down)

            up = nn.Sequential(
                nn.Conv2d(out_channel // reduction, out_channel, kernel_size, stride, padding, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channel)
            )
            setattr(self, f'up_{i}', up)

            s_lambda = nn.Parameter(torch.Tensor([threshold]))
            setattr(self, f's_lambda_{i}', s_lambda)

    def forward(self, input):
        x_in = self.layer_in(input)
        x = torch.mul(torch.sign(x_in), F.relu(torch.abs(x_in) - self.s_lambda_in))

        for i in range(self.num_layers):
            x_down = getattr(self, f'down_{i}')(x)
            x_up = getattr(self, f'up_{i}')(x_down)
            x = x - x_up + x_in
            x = torch.mul(torch.sign(x), F.relu(torch.abs(x) - getattr(self, f's_lambda_{i}')))

        return x


class MLP(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=32, num_layers=1):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(hidden_channel),
            nn.Conv2d(hidden_channel, in_channel, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channel)
        )
        self.out_layer = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x):   # x:[b,n,c]
        for i in range(self.num_layers):
            x = self.layers(x)
        x = self.out_layer(x)
        return x


class FourierTransform(nn.Module):
    def __init__(self, in_channel):
        super(FourierTransform, self).__init__()
        self.amplitude_conv = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.phase_conv = nn.Conv2d(in_channel, in_channel, 1, 1)

    def forward(self, x, freq_mask):  # x: [b,c,h,w]
        x_fft = torch.fft.fft(x, dim=1)
        x_fft = torch.fft.fftshift(x_fft, dim=1)

        x_fft = x_fft * freq_mask

        x_amplitude = self.amplitude_conv(x_fft.real)
        x_phase = self.phase_conv(x_fft.imag)
        x_fft = torch.complex(x_amplitude, x_phase)

        x_ifft = torch.fft.ifftshift(x_fft, dim=1)
        x_ifft = torch.fft.ifft(x_ifft, dim=1)
        x = torch.real(x_ifft)
        x = F.relu(x, inplace=True)

        return x


class SpectralLearner(nn.Module):
    def __init__(self, input_channels, output_channels, window_size=(4, 4)):
        super(SpectralLearner, self).__init__()
        self.window_size = window_size
        self.ft_low = FourierTransform(input_channels)
        self.ft_mid = FourierTransform(input_channels)
        self.ft_high = FourierTransform(input_channels)
        self.mlp = MLP(input_channels * 3, output_channels, num_layers=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: [b,c,h,w]
        n_band = 16
        low_thresh, high_thresh = 0.33, 0.66
        freqs = torch.linspace(0, 1, steps=n_band, device=x.device)
        low_mask = (freqs.abs() <= low_thresh).float().view(1, n_band, 1, 1)
        mid_mask = ((freqs.abs() > low_thresh) & (freqs.abs() < high_thresh)).float().view(1, n_band, 1, 1)
        high_mask = (freqs.abs() >= high_thresh).float().view(1, n_band, 1, 1)

        x_low_freq = self.ft_low(x, low_mask)
        x_mid_freq = self.ft_mid(x, mid_mask)
        x_high_freq = self.ft_high(x, high_mask)

        x_ft = torch.cat((x_low_freq, x_mid_freq, x_high_freq), dim=1)  # [b,3c,h,w]
        x_flat = self.mlp(x_ft)

        window_h, window_w = self.window_size
        x_windows = x_flat.unfold(2, window_h, window_h).unfold(3, window_w, window_w)  # [b,c,h//window_h,w//window_w,window_h,window_w]
        x_vectors = x_windows.mean(dim=(-1, -2))  # [b,c,h//window_h,w//window_w]
        x_vectors = rearrange(x_vectors, 'b c nh nw -> b (nh nw) c')

        att = torch.bmm(x_vectors, x_vectors.transpose(1, 2))  # [b,N,N]
        att = torch.softmax(att, dim=1)
        att = torch.bmm(x_vectors.transpose(1, 2), att.permute(0, 2, 1))  # [b,c,N]

        _, _, h, w = x.shape
        h, w = h // window_h, w // window_w
        x_att = rearrange(att, 'b c (nh nw) -> b c nh nw', nh=h, nw=w)
        x_att_expanded = x_att.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, window_h, window_w)  # [b,2,nh,nw,window_h,window_w]
        x_att = rearrange(x_att_expanded, 'b c nh nw wh ww -> b c (nh wh) (nw ww)')  # [b,c,h,w]

        x = x_att * self.gamma + x_flat

        return x


class RFFNet(nn.Module):
    def __init__(self, rgb_in=3, hsi_in=16, n_classes=2, feature_scale=1, is_deconv=True, is_batchnorm=True,
                 ffu_layers=2, ffu_threshold=0.1, n_concepts=7, window_size=(8, 8)):
        super(RFFNet, self).__init__()
        self.window_size = window_size
        filters = [64, 128, 256, 512]
        filters = [int(x / feature_scale) for x in filters]
        print(f"Filter numbers of each layer in RFF-Net: {filters}")
        # rgb encode
        self.rgb_maxpool = nn.MaxPool2d(kernel_size=2)
        self.rgb_conv1 = unetConv2(rgb_in, filters[0], is_batchnorm)
        self.rgb_conv2 = unetConv2(filters[0], filters[1], is_batchnorm)
        self.rgb_conv3 = unetConv2(filters[1], filters[2], is_batchnorm)
        self.rgb_center = unetConv2(filters[2], filters[3], is_batchnorm)
        # # hsi encode
        self.hsi_maxpool = nn.MaxPool2d(kernel_size=2)
        self.hsi_conv1 = unetConv2(hsi_in, filters[0], is_batchnorm)
        self.hsi_conv2 = unetConv2(filters[0], filters[1], is_batchnorm)
        self.hsi_conv3 = unetConv2(filters[1], filters[2], is_batchnorm)
        self.hsi_center = unetConv2(filters[2], filters[3], is_batchnorm)
        # feature filtering units
        for i in range(len(filters)):
            ffu = FFU(2 * filters[i], filters[i], num_layers=ffu_layers, threshold=ffu_threshold)
            setattr(self, f'ffu{i + 1}', ffu)
        # decode
        self.up_concat3 = unetUp(filters[3], filters[2], is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], is_deconv)
        self.final2d = nn.Conv2d(filters[0], n_classes, 1)
        self._init_weight()
        # spectral learning
        self.spec_lerner1 = SpectralLearner(hsi_in, hsi_in, window_size=(4, 4))
        self.spec_lerner2 = SpectralLearner(hsi_in, hsi_in, window_size=(1, 8))
        self.spec_lerner3 = SpectralLearner(hsi_in, hsi_in, window_size=(8, 1))
        self.spec_lerner4 = SpectralLearner(hsi_in, n_classes)
        # decision fusion
        self.sim_mlp = MLP(in_channel=n_concepts, out_channel=n_classes)
        self.final_mlp = MLP(in_channel=n_classes * 2, out_channel=n_classes)
        self.active = torch.nn.Sigmoid()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, rgb, hsi, prototypes):  # rgb:[b,3,h,w]; hsi:[b,16,h,w]; prototypes:[n,16,h,w]
        # rgb encode
        rgb_conv1 = self.rgb_conv1(rgb)  # [b,64,h,w]
        rgb_maxpool1 = self.rgb_maxpool(rgb_conv1)  # [b,64,h//2,w//2]
        rgb_conv2 = self.rgb_conv2(rgb_maxpool1)  # [b,128,h//2,w//2]
        rgb_maxpool2 = self.rgb_maxpool(rgb_conv2)  # [b,128,h//4,w//4]
        rgb_conv3 = self.rgb_conv3(rgb_maxpool2)  # [b,256,h//4,w//4]
        rgb_maxpool3 = self.rgb_maxpool(rgb_conv3)  # [b,256,h//8,w//8]
        rgb_center = self.rgb_center(rgb_maxpool3)  # [b,512,h//8,w//8]

        # hsi encode
        hsi_conv1 = self.hsi_conv1(hsi)
        hsi_maxpool1 = self.hsi_maxpool(hsi_conv1)
        hsi_conv2 = self.hsi_conv2(hsi_maxpool1)
        hsi_maxpool2 = self.hsi_maxpool(hsi_conv2)
        hsi_conv3 = self.hsi_conv3(hsi_maxpool2)
        hsi_maxpool3 = self.hsi_maxpool(hsi_conv3)
        hsi_center = self.hsi_center(hsi_maxpool3)

        # feature selection
        f_selected1 = self.ffu1(torch.cat((rgb_conv1, hsi_conv1), dim=1))  # [b,128,h,w] -> [b,64,h,w]
        f_selected2 = self.ffu2(torch.cat((rgb_conv2, hsi_conv2), dim=1))  # [b,256,h//2,w//2] -> [b,128,h//2,w//2]
        f_selected3 = self.ffu3(torch.cat((rgb_conv3, hsi_conv3), dim=1))  # [b,512,h//4,w//4] -> [b,256,h//4,w//4]
        f_selected_center = self.ffu4(torch.cat((rgb_center, hsi_center), dim=1))  # [b,1024,h//8,w//8] -> [b,512,h//8,w//8]

        # decode
        up3 = self.up_concat3(f_selected_center, f_selected3)  # [b,256,h//4,w//4]
        up2 = self.up_concat2(up3, f_selected2)  # [b,128,h//2,w//2]
        up1 = self.up_concat1(up2, f_selected1)  # [b,64,h,w]
        final = self.final2d(up1)  # [b,2,h,w]

        # spectral learning
        spec1 = self.spec_lerner1(hsi)
        spec2 = self.spec_lerner2(spec1)
        spec3 = self.spec_lerner3(spec2)  # [b,16,h,w]
        prot1 = self.spec_lerner1(prototypes)
        prot2 = self.spec_lerner2(prot1)
        prot3 = self.spec_lerner3(prot2)  # [N,16,a,a]

        # window-based similarity
        window_h, window_w = self.window_size
        spec = spec3.unfold(2, window_h, window_h).unfold(3, window_w, window_w)  # [b,c,h//window_h,w//window_w,window_h,window_w]
        prot = prot3.unfold(2, window_h, window_h).unfold(3, window_w, window_w).squeeze(-1).squeeze(-1)  # [N,16,1,1,window_h,window_w]
        spec = rearrange(spec, 'b c nh nw wh ww -> b nh nw (c wh ww)')  # [b,nh,nw,c*window_h*window_w]
        prot = rearrange(prot, 'n c 1 1 wh ww -> n (c wh ww)')  # [N,c*window_h*window_w]
        similarity = torch.matmul(spec, prot.T)  # [b,nh,nw,N] where N is the number of prototypes
        similarity = rearrange(similarity, 'b nh nw N -> b N nh nw')
        y_pred = self.sim_mlp(similarity)  # [b,2,nh,nw]
        y_pred_expanded = y_pred.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, window_h, window_w)  # [b,2,nh,nw,window_h,window_w]
        y_pred = rearrange(y_pred_expanded, 'b c nh nw wh ww -> b c (nh wh) (nw ww)')  # [b,2,h,w]

        # final output
        final = self.final_mlp(torch.cat((final, y_pred), dim=1))
        final = self.active(final)
        return final


if __name__ == '__main__':
    model = RFFNet(rgb_in=3, hsi_in=16, n_classes=2, n_concepts=7, window_size=(8, 8)).cuda()
    print(model)

    x = torch.rand(1, 3, 128, 128).cuda()
    y = torch.rand(1, 16, 128, 128).cuda()
    z = torch.rand(7, 16, 8, 8).cuda()
    out = model(x, y, z)
    print(out.shape)
