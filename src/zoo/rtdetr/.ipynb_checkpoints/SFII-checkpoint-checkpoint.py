import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .CBAM import CBAM

""" CVPR2024 https://openaccess.thecvf.com/content/CVPR2024/html/Cong_A_Semi-supervised_Nighttime_Dehazing_Baseline_with_Spatial-Frequency_Aware_and_Realistic_CVPR_2024_paper.html
现有基于深度学习的研究已经广泛探索了白天图像去雾问题，然而很少有研究考虑夜间雾霾场景的特点。夜间和白天雾霾有两个区别。
首先，夜间场景中可能存在多个光照强度较低的活动有色光源，可能造成雾霾辉光和噪声，具有局部耦合和频率不一致的特性。
其次，由于模拟数据和真实数据之间的域差异，将在模拟数据上训练的去雾模型应用于真实数据时可能会出现不真实的亮度。
为了解决以上两个问题，我们提出了一种用于真实世界夜间去雾的半监督模型。
首先，将空间注意和频谱滤波作为空间频域信息交互模块来解决第一个问题。
其次，设计了一种基于伪标签的再训练策略和基于局部窗口的亮度损失的半监督训练过程，以抑制雾霾和辉光，同时实现真实的亮度。
在公共基准上的实验验证了所提出方法的有效性及其优于最先进方法的优越性。
"""


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock_Conv, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.trans_layer = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans_layer(out)
        out = self.conv2(out)
        return out + x


class SpaBlock(nn.Module):
    def __init__(self, nc):
        super(SpaBlock, self).__init__()
        self.block = ResBlock_Conv(in_channel=nc, out_channel=nc)

    def forward(self, x):
        yy = self.block(x)
        return yy


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Frequency_Spectrum_Dynamic_Aggregation(nn.Module):
    def __init__(self, nc):
        super(Frequency_Spectrum_Dynamic_Aggregation, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, 64, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            CBAM(64),
            nn.Conv2d(64, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, 64, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            CBAM(64),
            nn.Conv2d(64, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x = torch.fft.rfft2(x, norm='ortho')
        ori_mag = torch.abs(x)
        ori_pha = torch.angle(x)
        mag = self.processmag(ori_mag)
        mag = ori_mag + mag
        pha = self.processpha(ori_pha)
        pha = ori_pha + pha
        with torch.no_grad():
            real = mag * torch.cos(pha)
            imag = mag * torch.sin(pha)
            x_out = torch.complex(real, imag)
        # real = mag * torch.cos(pha)
        # imag = mag * torch.sin(pha)
        # x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='ortho')
        return x_out


class BidomainNonlinearMapping(nn.Module):

    def __init__(self, in_nc):
        super(BidomainNonlinearMapping, self).__init__()
        # self.spatial_process = SpaBlock(4)
        from .hybrid_encoder import EdgeCspRepLayer, CSPRepLayer
        self.spatial_process = CSPRepLayer(4, 4, expansion=4, num_blocks=1, useglu=False)
        self.srmCsp = CSPRepLayer(3, 3, expansion=4, num_blocks=1, useglu=False)

        self.bottlenecks = CSPRepLayer(7, 4, expansion=4, num_blocks=1, useglu=False)
        
        self.frequency_process = Frequency_Spectrum_Dynamic_Aggregation(in_nc)
        self.cat = nn.Conv2d(2 * in_nc+1, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        
        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.cuda.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)
        self.weight_hori = self.weight_const_hori
        self.gamma = nn.Parameter(torch.zeros(1))
        self.inputNorm = nn.BatchNorm2d(num_features= 4, eps=1e-5)

        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.cuda.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)
        self.weight_vertical = self.weight_const_vertical

        # ch_in = 64
        # conv_def = [
        #     [3, ch_in // 2, 3, 2, "conv1_1"],
        #     [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
        #     [ch_in // 2, ch_in, 3, 1, "conv1_3"],
        # ]
        # act='relu'
        # from ...nn.backbone.presnet import ConvNormLayer, OrderedDict
        # self.conv1 = nn.Sequential(OrderedDict([
        #     (name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def
        # ]))
        # self.edgeConv = nn.Sequential(OrderedDict([
        #     (name, ConvNormLayer(cin, cout, k, s, act=act)) for cin, cout, k, s, name in conv_def
        # ]))
        # state = torch.hub.load_state_dict_from_url('https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth', map_location='cpu')
        # conv1_state_dict = {k: v for k, v in state.items() if k.startswith('conv1')}
        #         # 处理预训练模型的state_dict键名
        # new_state_dict = {}
        # for key in state.keys():
        #     if key.startswith('conv1.'):
        #         new_key = key[len('conv1.'):]  # 移除前缀
        #         new_state_dict[new_key] = state[key]

        # # 加载处理后的state_dict
        # self.conv1.load_state_dict(new_state_dict)
        # self.edgeConv.load_state_dict(new_state_dict)
        # print(f'加载了PResNet的conv1的state_dict')
    
    def forward(self, x):
        x_edge = self.get_edge(x)
        x0 = torch.concat([x, x_edge], 1)
        x1 = self.spatial_process(x0)

        # x_freq = torch.fft.rfft2(x, norm='backward')
        x_freq_spatial = self.frequency_process(x)
        # x_freq_spatial = self.srm(x)
        x_freq_spatial = self.srmCsp(x_freq_spatial)
        # x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')

        xcat = torch.concat([x1, x_freq_spatial], 1)
        xcat = self.bottlenecks(xcat)
        # x_out = torch.concat([x, xcat], 1)
        x_out = self.sigmoid(xcat) * x0 * self.gamma + x0*(1-self.gamma)
        # print(xout.shape)
        return x_out

    def get_edge(self, im):
        x3 = Variable(im[:, 2].unsqueeze(1))
        weight_hori = Variable(self.weight_hori)
        weight_vertical = Variable(self.weight_vertical)

        try:
            x_hori = F.conv2d(x3, weight_hori, padding=1)
            # x_hori = self.conv2d_hori(x3)
        except:
            print('horizon error')
        try:
            x_vertical = F.conv2d(x3, weight_vertical, padding=1)
            # x_vertical = self.conv2d_vertical(x3)
        except:
            print('vertical error')

        # get edge image
        edge_detect = (torch.add(x_hori.pow(2), x_vertical.pow(2))).pow(0.5)
        return edge_detect


if __name__ == '__main__':
    in_nc = 3  # 输入通道数
    block = BidomainNonlinearMapping(in_nc)
    input = torch.rand(4, in_nc, 64, 64)
    output = block(input)
    print(input.size())
    print(output.size())

