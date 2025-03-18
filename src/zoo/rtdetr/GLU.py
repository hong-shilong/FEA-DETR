# https://arxiv.org/pdf/2311.17132
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from thop import profile
import torch.nn.functional as F
from torchvision.utils import save_image

from .CBAM import CBAM
from .trans_3D_4D import to_3d, to_4d
from torch.autograd import Variable
from torch.cuda.amp import autocast


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        # B, N, C = x.shape
        x = to_4d(x, H, W)
        # print(x.shape)
        # print(torch.max(x))
        x = self.dwconv(x)
        # print(torch.max(x))
        # print(x.shape)
        x = to_3d(x)
        # x = x.flatten(2).transpose(1, 2)
        # print(x.shape)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2).to(torch.device("cuda"))
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features).to(torch.device("cuda"))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x0 = x
        B, C, H, W = x.shape
        # print(torch.min(x))

        x = to_3d(x)
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.dwconv(x, H, W)
        # print(torch.min(x))
        x = self.act(x) * v
        # print(torch.max(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = to_4d(x, H, W)  + x0
        return x


class GluCSP(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.inputNorm = nn.BatchNorm2d(num_features=in_features + 1, eps=1e-5)
        # self.attention = GlobalContextBlock(4,1)
        from .hybrid_encoder import RepVggBlock

        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(in_features + 1, hidden_features, act='silu'),
            RepVggBlock(hidden_features, hidden_features, act='silu'),
            RepVggBlock(hidden_features, hidden_features, act='silu'),
        ])
        self.mlp0 = ConvolutionalGLU(hidden_features, hidden_features, 1, act_layer, drop)
        self.attenNorm = nn.BatchNorm2d(num_features=4, eps=1e-5)

    def forward(self, x1):
        x2 = self.get_edge(x1)
        x = torch.concat([x1, x2], dim=1)
        x = self.inputNorm(x)
        # print(x.shape)
        x = self.bottlenecks(x)
        x = self.attenNorm(x)
        x = self.mlp0(x)

        x = x2 * x + x2

        # print(x.shape)
        return x


class edgeCspLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        kernel_const_hori = np.array([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype='float32')
        kernel_const_hori = torch.cuda.FloatTensor(kernel_const_hori).unsqueeze(0)
        self.weight_const_hori = nn.Parameter(data=kernel_const_hori, requires_grad=False)
        self.weight_hori = self.weight_const_hori
        self.gamma = nn.Parameter(torch.zeros(1))

        kernel_const_vertical = np.array([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype='float32')
        kernel_const_vertical = torch.cuda.FloatTensor(kernel_const_vertical).unsqueeze(0)
        self.weight_const_vertical = nn.Parameter(data=kernel_const_vertical, requires_grad=False)
        self.weight_vertical = self.weight_const_vertical

        from .hybrid_encoder import EdgeCspRepLayer, CSPRepLayer
        # self.bottlenecks = EdgeCspRepLayer(4, 3, expansion=4, useglu=False, num_blocks=2)
        self.bottlenecks = CSPRepLayer(4, 3, expansion=4, useglu=False)

        # self.sigmoid = nn.Sigmoid()
        # self.norm = nn.LayerNorm()
        # self.inputNorm = nn.BatchNorm2d(num_features=in_features + 1, eps=1e-5)
        # self.attention = GlobalContextBlock(4,1)
        # self.attention = CBAM(4, 1)
        # self.mlp0 = ConvolutionalGLU(hidden_features, hidden_features, 1, act_layer, drop)
        # self.attenNorm = nn.BatchNorm2d(num_features=hidden_features, eps=1e-5)

    def forward(self, x1):
        x2 = self.get_edge(x1)
        x2 = torch.concat([x1, x2], dim=1)
        # x = self.inputNorm(x)
        # print(x.shape)
        # x = self.attention(x) + x
        with autocast(enabled=False):
            x = self.bottlenecks(x2) + x1
        # x = torch.concat([x1, x], dim=1)
        # x = self.sigmoid(x) * x1
        # x = torch.concat([x1, x], dim=1)
        # x = self.attenNorm(x)
        # x = self.mlp0(x)

        # x = x2 * x + x2

        # print(x.shape)
        return x

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


class SelectGlu(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.inputNorm = nn.BatchNorm2d(num_features=in_features, eps=1e-5)
        self.fc1 = nn.Linear(in_features, hidden_features * 2).to(torch.device("cuda"))
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features).to(torch.device("cuda"))
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print(x.shape)
        B, C, H, W = x.shape
        # print(torch.min(x))
        x0 = self.inputNorm(x)

        x = to_3d(x0)

        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.dwconv(x, H, W)
        # print(torch.min(x))
        x = self.act(x) * v
        # print(torch.max(x))

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = to_4d(x, H, W)
        # x = x + x0
        return x


class Attention(nn.Module):
    """Top-K Selective Attention (TTSA)
   Tips:
       Mainly borrows from DRSFormer (https://github.com/cschenxiang/DRSformer)
   """

    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape  # C=30，即通道数

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature  # b 1 C C

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        # print(111, mask1.scatter_(-1, index, 1.))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)  # [1 6 30 30]
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


if __name__ == '__main__':
    # x = torch.randn(4, 3, 640, 640)
    import os

    print(os.getcwd())
    img = torch.load('./src/nn/backbone/img.pt').to(torch.device("cuda"))[3:4, :, :, :]
    save_image(img, './img/ori_.png')

    B, C, H, W = img.shape
    gpu1 = torch.device("cuda")
    attention = TwoGlu(C, 16).to(gpu1)
    x = attention(img)
    save_image(x, './img/DOAM_.png')
    print('output shape:' + str(x.shape))
    flops, params = profile(attention, inputs=(img,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params))
    # FLOPs = 3.599028224G
    # Params = 6443.0
