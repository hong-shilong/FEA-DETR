import torch
import torch.nn as nn
import torch.nn.functional as F

class SRM(nn.Module):
    def __init__(self):
        super(SRM, self).__init__()
        # 初始化滤波器权重
        self.f1 = nn.Parameter(torch.tensor([[0, 0, 0, 0, 0], 
                                             [0, -1, -2, -1, 0], 
                                             [0, 2, 4, 2, 0],
                                             [0, -1, -2, -1, 0],
                                             [0, 0, 0, 0, 0]]) / 4.0, requires_grad=False)

        self.f2 = nn.Parameter(torch.tensor([[-1, 2, -2, 2, -1],
                                             [2, -6, 8, -6, 2],
                                             [-2, 8, -12, 8, -2],
                                             [2, -6, 8, -6, 2],
                                             [-1, 2, -2, 2, -1]]) / 12.0, requires_grad=False)

        self.f3 = nn.Parameter(torch.tensor([[0, 0, 0, 0, 0],
                                             [0, 1, 0, 1, 0],
                                             [0, -2, 4, -2, 0],
                                             [0, 1, 0, 1, 0],
                                             [0, 0, 0, 0, 0]]) / 2.0, requires_grad=False)

    def forward(self, x):
        # 将滤波器扩展到所有输入通道
        f1 = self.f1.repeat(x.size(1), 1, 1, 1)
        f2 = self.f2.repeat(x.size(1), 1, 1, 1)
        f3 = self.f3.repeat(x.size(1), 1, 1, 1)
        
        # 应用滤波器
        xh_f1 = F.conv2d(x, f1, padding=2, groups=x.size(1))
        xh_f2 = F.conv2d(x, f2, padding=2, groups=x.size(1))
        xh_f3 = F.conv2d(x, f3, padding=2, groups=x.size(1))
        
        # 根据需要，您可以在这里添加合并这三个特征图的逻辑
        # 示例：简单地求和这三个特征图
        xh = xh_f1 + xh_f2 + xh_f3
        return xh
